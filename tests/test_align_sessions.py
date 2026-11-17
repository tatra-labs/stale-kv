import random

import pytest

from stale_kv.align import Segment, align, check_plan, common_prefix, summarize
from stale_kv.measure import whole
from stale_kv.ngram import prompt_lookup
from stale_kv.sessions import Session, parse_messages, placeholder


def _rand(n, seed, lo=0, hi=50_000):
    r = random.Random(seed)
    return [r.randrange(lo, hi) for _ in range(n)]


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix([], [1]) == 0
    assert common_prefix([1, 2], [1, 2]) == 2


def test_align_masking_edit_shape():
    """The SWE-agent edit: prefix | observation -> placeholder | retained middle | new tail."""
    pre, obs, ph, mid, tail = _rand(50, 1), _rand(40, 2), _rand(6, 3), _rand(120, 4), _rand(30, 5)
    old = pre + obs + mid
    new = pre + ph + mid + tail
    segs = align(old, new, min_run=16)
    check_plan(segs, old, new)
    assert segs == [
        Segment("keep", 0, 50, 0),
        Segment("new", 50, 6),
        Segment("map", 56, 120, 90),
        Segment("new", 176, 30),
    ]
    assert segs[2].delta == -34
    assert summarize(segs, len(old)) == {
        "prefix": 50, "reused": 120, "staged": 36, "dropped_old": 40, "n_map": 1, "n_new": 2,
    }


def test_align_drops_reply_header_tokens_that_the_next_prompt_omits():
    """The reply was generated after an empty think block that later prompts do not render."""
    head, think, reply, tail = _rand(30, 6), _rand(4, 7), _rand(60, 8), _rand(20, 9)
    old = head + think + reply
    new = head + reply + tail
    segs = align(old, new)
    check_plan(segs, old, new)
    assert [s.kind for s in segs] == ["keep", "map", "new"]
    assert segs[1].delta == -4


def test_align_short_runs_are_not_mapped():
    old = _rand(40, 10)
    new = old[:10] + _rand(5, 11) + old[20:28] + _rand(5, 12)  # an 8-token run, below min_run=16
    segs = align(old, new, min_run=16)
    check_plan(segs, old, new)
    assert [s.kind for s in segs] == ["keep", "new"]
    segs8 = align(old, new, min_run=8)
    check_plan(segs8, old, new)
    assert "map" in [s.kind for s in segs8]


def test_align_keeps_runs_in_order():
    a, b = _rand(30, 13), _rand(30, 14)
    old = a + b
    new = b + a  # swapped: only one of the two can be mapped in order
    segs = align(old, new, min_run=8)
    check_plan(segs, old, new)
    assert sum(s.length for s in segs if s.kind == "map") == 30


@pytest.mark.parametrize("seed", range(20))
def test_align_random_edits_always_tile(seed):
    r = random.Random(seed)
    old = _rand(r.randrange(50, 400), seed, hi=r.choice([5, 50, 50_000]))
    new = list(old)
    for _ in range(r.randrange(1, 5)):
        i = r.randrange(len(new) + 1)
        j = min(len(new), i + r.randrange(0, 40))
        new[i:j] = _rand(r.randrange(0, 30), seed + 100, hi=50)
    segs = align(old, new, min_run=r.choice([1, 4, 16]))
    check_plan(segs, old, new)


def test_check_plan_rejects_a_bad_plan():
    with pytest.raises(AssertionError):
        check_plan([Segment("map", 0, 3, 0)], [1, 2, 3], [1, 2, 4])


def _session(n):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"reply {i}"},
                 {"role": "user", "content": [{"type": "text", "text": f"OBSERVATION:\nout {i}\nline2"}]}]
    return parse_messages("s", msgs)


def test_parse_messages_and_masking_schedule():
    s = _session(8)
    assert s.n_requests == 8 and s.observations[0] == "OBSERVATION:\nout 0\nline2"
    # Request 1 sees only the head; request t sees t-1 replies and observations.
    assert len(s.messages(1, 5)) == 2 and len(s.messages(8, 5)) == 2 + 14
    m = s.messages(8, 5)  # observations 1..7 seen, keep the last 5 -> 1 and 2 masked
    obs = [x["content"] for x in m[3::2]]
    assert obs[:2] == [placeholder("x\ny\nz")] * 2 and obs[2].startswith("OBSERVATION")
    assert placeholder("a\nb") == "Old environment output: (2 lines omitted)"
    assert [t for t in range(1, 9) if s.is_miss(t, 5)] == [7, 8]
    assert not s.is_miss(8, None)
    # Exactly one more observation is masked at each miss request.
    for t in (7, 8):
        prev = sum(x["content"].startswith("Old") for x in s.messages(t - 1, 5))
        cur = sum(x["content"].startswith("Old") for x in s.messages(t, 5))
        assert cur == prev + 1


def test_parse_messages_rejects_malformed():
    assert parse_messages("x", [{"role": "user", "content": "a"}] * 4) is None
    bad = [{"role": "system", "content": ""}, {"role": "user", "content": ""},
           {"role": "assistant", "content": ""}, {"role": "assistant", "content": ""}]
    assert parse_messages("x", bad) is None
    assert isinstance(_session(1), Session)


def test_prompt_lookup_copies_continuation():
    ctx = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3]
    assert prompt_lookup(ctx, 3) == [4, 5, 9]
    assert prompt_lookup([7, 8], 4) == []  # nothing to match
    assert prompt_lookup(ctx, 10, stop={5}) == [4, 5]


def test_prompt_lookup_never_matches_the_suffix_itself():
    assert prompt_lookup([1, 2, 3], 2) == []


def test_prompt_lookup_prefers_latest_occurrence():
    ctx = [1, 2, 7, 0, 1, 2, 8, 0, 1, 2]
    assert prompt_lookup(ctx, 1) == [8]


def test_whole_window():
    stop = {0}
    assert whole([5] * 32, 32, 32, stop)
    assert whole([5] * 40, 35, 32, stop)  # the first 32 are all in
    assert not whole([5] * 32, 31, 32, stop)
    assert whole([5, 5, 0], 3, 32, stop)  # short reply, fully drafted and accepted
    assert not whole([5, 5, 5], 3, 32, stop)  # short draft that did not finish the reply
    assert not whole([], 0, 32, stop)


class _CharTok:
    """A stand-in tokenizer: one token per character, a ChatML-like template with an empty think block."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        text = "".join(f"<{m['role']}>{m['content']}|" for m in msgs)
        return text + ("<assistant>" + "<think>\n\n</think>\n\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, tok):
        return 0


def test_miss_request_tokens_and_reply_rendering_counterfactual():
    from stale_kv.sessions import Renderer, miss_request

    s = _session(8)
    r = Renderer(_CharTok())
    req = miss_request(s, 7, 5, r)
    prev = r.prompt(s.messages(6, 5))
    assert req.old == prev + [ord(c) for c in "reply 5"] + [0]
    assert req.new == r.prompt(s.messages(7, 5))
    alt = miss_request(s, 7, 5, r, reply_as_rendered=True)
    think = [ord(c) for c in "<think>\n\n</think>\n\n"]
    assert alt.old == prev[: -len(think)] + [ord(c) for c in "reply 5"] + [0]
    assert alt.new == req.new
    with pytest.raises(AssertionError):
        miss_request(s, 6, 5, r)  # request 6 masks nothing new

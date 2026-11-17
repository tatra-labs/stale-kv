"""KV engine and RoPE tests on a tiny random-init Qwen3 in float64 (no weights, no GPU needed)."""

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from stale_kv.align import Segment, align
from stale_kv.kv import Engine, accepted, rerotate

V = 97


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        rope_theta=10000.0, tie_word_embeddings=True,
    )
    m = Qwen3ForCausalLM(cfg).to(torch.float64).eval()
    m.config._attn_implementation = "eager"
    return m


@pytest.fixture(scope="module")
def eng(model):
    return Engine(model, max_len=256, use_graphs=False)


def _tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (n,), generator=g).tolist()


@torch.no_grad()
def _hf_cache(model, toks, start=0):
    cache = DynamicCache(config=model.config)
    pos = torch.arange(start, start + len(toks))[None]
    out = model(input_ids=torch.tensor([toks]), position_ids=pos, past_key_values=cache, use_cache=True)
    return cache, out.logits[0]


@pytest.mark.parametrize("delta", [-37, -1, 5, 300])
def test_rerotation_equals_recompute_for_a_pure_shift(model, delta):
    """Shifting a whole sequence changes no hidden state under RoPE, so every layer's keys at p+delta
    are the keys at p rotated by delta, and the values are identical. Hugging Face computes RoPE
    angles in float32 even for a float64 model, which puts ~1e-5 of error into the recomputed side
    itself at positions near 700, hence the tolerance."""
    toks = _tokens(24, 1)
    a, _ = _hf_cache(model, toks, 400)
    b, _ = _hf_cache(model, toks, 400 + delta)
    freqs = model.model.rotary_emb.inv_freq.float()
    for la, lb in zip(a.layers, b.layers):
        assert (rerotate(la.keys, delta, freqs) - lb.keys).abs().max() < 1e-4
        assert (la.values - lb.values).abs().max() < 1e-5


def test_rerotate_zero_is_identity_and_composes():
    k = torch.randn(1, 2, 5, 16, dtype=torch.float64)
    f = 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16))
    assert rerotate(k, 0, f) is k
    assert torch.allclose(rerotate(rerotate(k, 7, f), -19, f), rerotate(k, -12, f), atol=1e-6)


def test_engine_matches_hf_logits_and_kv(model, eng):
    toks = _tokens(40, 2)
    cache, logits = _hf_cache(model, toks)
    buf = eng.buffer()
    rows = eng.forward(buf, toks, all_logits=True)
    assert buf.len == 40
    assert torch.allclose(rows, logits.float(), atol=1e-5)
    # HF's eager attention takes its softmax in float32, so deeper layers agree to ~5e-7, not 1e-12.
    for li, layer in enumerate(cache.layers):
        assert torch.allclose(buf.k[li, :, :, :40], layer.keys, atol=1e-5)
        assert torch.allclose(buf.v[li, :, :, :40], layer.values, atol=1e-5)


def test_chunked_prefill_equals_one_shot(eng):
    toks = _tokens(50, 3)
    a = eng.prefill(eng.buffer(), toks, chunk=None)
    b = eng.prefill(eng.buffer(), toks, chunk=7)
    assert torch.allclose(a.k[:, :, :, :50], b.k[:, :, :, :50], atol=1e-10)
    assert torch.allclose(a.v[:, :, :, :50], b.v[:, :, :, :50], atol=1e-10)


def test_decode_matches_hf_greedy_and_leaves_cache_intact(model, eng):
    ctx = _tokens(30, 4)
    buf = eng.prefill(eng.buffer(), ctx[:-1])
    before = buf.k[:, :, :, :29].clone()
    drafts, gaps = eng.decode_greedy(buf, ctx[-1], 12, stop=set())
    assert buf.len == 29 and len(drafts) == 12 and len(gaps) == 12 and min(gaps) >= 0
    assert torch.equal(buf.k[:, :, :, :29], before)
    seq = list(ctx)
    for _ in range(12):  # reference: full recompute, argmax of the last row
        _, logits = _hf_cache(model, seq)
        seq.append(int(logits[-1].argmax()))
    assert drafts == seq[30:]


def test_verify_accepts_its_own_greedy_drafts(eng):
    ctx = _tokens(30, 5)
    buf = eng.prefill(eng.buffer(), ctx[:-1])
    drafts, _ = eng.decode_greedy(buf, ctx[-1], 12, stop=set())
    rows = eng.score_chunk(buf, ctx[-1], drafts)
    assert buf.len == 29 and rows.shape[0] == 13
    assert accepted(drafts, rows, "fixed") == 12
    assert accepted(drafts, rows, "anymax") == 12
    bad = list(drafts)
    bad[4] = (bad[4] + 1) % V
    assert accepted(bad, rows, "fixed") == 4


def test_decode_stops_after_stop_token(eng):
    ctx = _tokens(20, 6)
    buf = eng.prefill(eng.buffer(), ctx[:-1])
    drafts, _ = eng.decode_greedy(buf, ctx[-1], 12, stop=set())
    cut, gaps = eng.decode_greedy(buf, ctx[-1], 12, stop={drafts[2]})
    i = drafts.index(drafts[2])
    assert cut == drafts[: i + 1] and len(gaps) == i + 1


def test_accepted_anymax_accepts_ties():
    rows = torch.tensor([[1.0, 3.0, 3.0], [0.0, 0.0, 5.0]])
    assert accepted([2, 2], rows, "fixed") == 0  # argmax picks id 1, the lowest tied id
    assert accepted([2, 2], rows, "anymax") == 2
    assert accepted([], rows) == 0


def test_provisional_equals_exact_when_nothing_is_stale(eng):
    """An edit that only replaces the tail leaves a pure prefix: provisional KV == exact KV."""
    old = _tokens(40, 7)
    new = old[:30] + _tokens(6, 8)
    segs = align(old, new, min_run=4)
    assert [s.kind for s in segs] == ["keep", "new"]
    prov = eng.make_provisional(eng.prefill(eng.buffer(), old), segs, new, chunk=4)
    exact = eng.prefill(eng.buffer(), new)
    assert prov.len == exact.len == 36
    assert torch.allclose(prov.k[:, :, :, :36], exact.k[:, :, :, :36], atol=1e-10)
    assert torch.allclose(prov.v[:, :, :, :36], exact.v[:, :, :, :36], atol=1e-10)


@pytest.mark.parametrize("mid_old,mid_new", [(8, 0), (8, 3), (2, 9)])
def test_mapped_run_is_rerotated_stale_kv(eng, mid_old, mid_new):
    """A run that moves after an edited span carries the old KV, re-rotated by its delta (which is
    positive when the replacement is longer than what it replaced), while the replacement span is
    computed fresh over the provisional cache."""
    a, b = _tokens(10, 9), _tokens(20, 10)
    old = a + _tokens(mid_old, 11) + b
    new = a + [(t + 1) % V for t in _tokens(mid_new, 12)] + b
    segs = align(old, new, min_run=4)
    m = [s for s in segs if s.kind == "map"]
    assert len(m) == 1 and m[0].delta == mid_new - mid_old and m[0].length == 20
    stale = eng.prefill(eng.buffer(), old)
    sk, sv = stale.k.clone(), stale.v.clone()
    prov = eng.make_provisional(stale, segs, new)
    s0, n0 = 10 + mid_old, 10 + mid_new
    assert torch.allclose(prov.k[:, :, :, n0 : n0 + 20], rerotate(sk[:, :, :, s0 : s0 + 20], n0 - s0, eng.freqs))
    assert torch.equal(prov.v[:, :, :, n0 : n0 + 20], sv[:, :, :, s0 : s0 + 20])
    exact = eng.prefill(eng.buffer(), new)
    # The prefix is exact; the moved run is stale (it once attended to the old middle).
    assert torch.allclose(prov.v[:, :, :, :10], exact.v[:, :, :, :10], atol=1e-10)
    assert not torch.allclose(prov.v[-1, :, :, n0 : n0 + 20], exact.v[-1, :, :, n0 : n0 + 20], atol=1e-6)


def test_rotation_matters(eng):
    """Without re-rotation the moved keys sit at the wrong positions."""
    a, mid, b = _tokens(10, 13), _tokens(30, 14), _tokens(20, 15)
    old, new = a + mid + b, a + b
    segs = align(old, new, min_run=4)
    rot = eng.make_provisional(eng.prefill(eng.buffer(), old), segs, new, rotate=True)
    norot = eng.make_provisional(eng.prefill(eng.buffer(), old), segs, new, rotate=False)
    assert not torch.allclose(rot.k[:, :, :, 10:30], norot.k[:, :, :, 10:30])
    assert torch.equal(rot.v[:, :, :, 10:30], norot.v[:, :, :, 10:30])


def test_buffer_overflow_is_an_error(eng):
    with pytest.raises(ValueError):
        eng.forward(eng.buffer(), _tokens(300, 16))

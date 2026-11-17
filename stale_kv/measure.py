"""The offline acceptance study for one cache-miss request (the paper's Section 3.2, at laptop scale).

For a request whose context was edited, draft the reply five ways and verify every draft against
the exact KV in one forward pass:

* ``exact``  - draft on the exact KV (the ceiling; misses only come from numerics: single-token
  drafting vs one verification chunk);
* ``stale``  - draft on the provisional KV built from the invalidated state (Callo);
* ``norot``  - as ``stale`` but without re-rotating the moved keys (ablation);
* ``nomid``  - draft on the valid prefix and the staged spans only, dropping the reused middle
  (the paper's "target model with only the valid prefix" baseline, in our reading);
* ``ngram``  - prompt-lookup n-grams over the new prompt (model-free baseline).

It also replays the exact reply on the provisional KV (teacher forcing) to measure per-position
next-token agreement and total variation against the exact distributions.

Two buffers are enough: ``exact`` holds the exact KV throughout; ``work`` is reused for the
invalidated state, each provisional variant and the prefix-only cache.
"""

from __future__ import annotations

import time

import torch

from .align import align, summarize
from .kv import Engine, KVBuffer, accepted
from .ngram import prompt_lookup
from .sessions import MissRequest


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def whole(drafts: list[int], a: int, k: int, stop: set[int]) -> bool:
    """Is the first ``k``-token window fully accepted? A window that ends early counts only if it
    ends on a stop token (the whole reply was drafted and accepted)."""
    d = drafts[:k]
    return bool(d) and min(a, k) == len(d) and (len(d) == k or d[-1] in stop)


@torch.no_grad()
def measure(
    eng: Engine,
    exact: KVBuffer,
    work: KVBuffer,
    req: MissRequest,
    stop: set[int],
    k_long: int = 111,
    k_short: int = 32,
    min_run: int = 16,
    chunk: int = 112,
    ngram_strip: list[int] | None = None,
) -> dict:
    new_ctx, last = req.new[:-1], req.new[-1]
    segs = align(req.old, new_ctx, min_run=min_run)
    rec: dict = {"key": req.key, "t": req.t, "keep_last": req.keep_last, "len_old": len(req.old),
                 "len_new": len(req.new), **summarize(segs, len(req.old))}
    times = {}

    # Exact KV: the reference reply, the ceiling, and the n-gram baseline.
    _sync(); t0 = time.perf_counter()
    eng.prefill(exact.reset(), new_ctx)
    _sync(); times["exact_prefill"] = time.perf_counter() - t0
    g, g_gaps = eng.decode_greedy(exact, last, k_long, stop)
    rows_g = eng.score_chunk(exact, last, g)
    rec["exact"] = {"drafts": g, "gaps": [round(x, 4) for x in g_gaps],
                    "a_fixed": accepted(g, rows_g, "fixed"), "a_anymax": accepted(g, rows_g, "anymax")}
    # Prompt lookup: Qwen3's generation prompt ends in an empty think block that history replies do
    # not carry, so it is stripped first; the lookup then continues from the reply header, which
    # matches the start of earlier replies (what the n-gram baseline sees under other templates).
    ng_ctx = req.new
    if ngram_strip and req.new[-len(ngram_strip):] == ngram_strip:
        ng_ctx = req.new[: -len(ngram_strip)]
    ng = prompt_lookup(ng_ctx, k_short, stop=stop)
    rec["ngram"] = {"drafts": ng, "a_fixed": accepted(ng, eng.score_chunk(exact, last, ng), "fixed") if ng else 0}
    n_tf = min(len(g), k_short)
    p_exact = torch.softmax(rows_g[:n_tf], dim=-1)
    del rows_g

    # Provisional KV without re-rotation (ablation).
    eng.prefill(work.reset(), req.old)
    eng.make_provisional(work, segs, new_ctx, rotate=False, chunk=chunk)
    d_norot, _ = eng.decode_greedy(work, last, k_short, stop)

    # Provisional KV as Callo builds it.
    _sync(); t0 = time.perf_counter()
    eng.prefill(work.reset(), req.old)
    _sync(); times["stale_prefill"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    eng.make_provisional(work, segs, new_ctx, rotate=True, chunk=chunk)
    _sync(); times["staging"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d_stale, s_gaps = eng.decode_greedy(work, last, k_long, stop)
    _sync(); times["draft"] = time.perf_counter() - t0
    times["draft_tokens"] = k_long
    # Teacher forcing: feed the exact reply on the provisional KV.
    q = torch.softmax(eng.score_chunk(work, last, g[:n_tf])[:n_tf], dim=-1)
    tv = 0.5 * (p_exact - q).abs().sum(dim=-1)
    agree = p_exact.argmax(-1) == q.argmax(-1)
    rec["teacher_forced"] = {"n": n_tf, "agree": int(agree.sum()), "tv": [round(float(x), 6) for x in tv]}
    del q, p_exact

    # Valid prefix + staged spans only (reused middle dropped).
    nomid_tokens = [tok for s in segs if s.kind != "map" for tok in new_ctx[s.new_start : s.new_start + s.length]]
    eng.prefill(work.reset(), nomid_tokens)
    d_nomid, _ = eng.decode_greedy(work, last, k_short, stop)

    # Verify every draft against the exact KV.
    _sync(); t0 = time.perf_counter()
    rows = eng.score_chunk(exact, last, d_stale)
    _sync(); times["verify"] = time.perf_counter() - t0
    times["verify_tokens"] = len(d_stale) + 1
    rec["stale"] = {"drafts": d_stale, "gaps": [round(x, 4) for x in s_gaps],
                    "a_fixed": accepted(d_stale, rows, "fixed"), "a_anymax": accepted(d_stale, rows, "anymax")}
    for name, d in (("norot", d_norot), ("nomid", d_nomid)):
        rec[name] = {"drafts": d, "a_fixed": accepted(d, eng.score_chunk(exact, last, d), "fixed")}
    rec["times"] = {k: round(v, 5) if isinstance(v, float) else v for k, v in times.items()}
    return rec

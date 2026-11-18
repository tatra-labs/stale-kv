"""Time this repo's engine on the local GPU and try to fit a step model to it.

    python experiments/bench_engine.py --model Qwen/Qwen3-0.6B

Measures single-token drafting steps (CUDA graph replay) against context length and eager chunks
against chunk size and prior context, then fits
    decode step:  t = a0 + a_c * ctx / 1000
    chunk:        t = b0 + b_tok * c + 1e-6 * (b_x * c * x + b_c * c^2 / 2)
in milliseconds. The engine runs one request at a time, so nothing here measures batching.

On the laptop behind the README this fit is not usable: each eager forward costs ~90 ms of Python and
kernel-launch overhead whatever its size, graph-replayed decode steps took 27-50 ms with other jobs on
the GPU, and the fitted slopes come out negative. The simulator therefore uses the paper's H200 fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from stale_kv.kv import Engine  # noqa: E402


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    eng = Engine(m, max_len=8192 + 2048 + 128)
    buf = eng.buffer()
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 100_000, 10_000).tolist()

    dec = []
    for ctx in (256, 1024, 2048, 4096, 6144, 8192):
        eng.prefill(buf.reset(), toks[:ctx])
        eng.decode_greedy(buf, 1, 4, set())  # capture
        ms = timed(lambda: eng.decode_greedy(buf, 1, 32, set()), 5) / 32
        dec.append((ctx, ms))
        print(f"decode ctx={ctx}: {ms:.2f} ms/token", flush=True)

    chunks = []
    for x in (0, 2048, 6144):
        for c in (1, 16, 64, 112, 256, 512, 1024, 2048):
            def run():
                buf.len = x
                eng.forward(buf, toks[x : x + c])
            eng.prefill(buf.reset(), toks[:x])
            ms = timed(run, 5)
            chunks.append((c, x, ms))
            print(f"chunk c={c} x={x}: {ms:.2f} ms", flush=True)

    d = np.array(dec)
    a_c, a0 = np.polyfit(d[:, 0] / 1000, d[:, 1], 1)
    C = np.array(chunks)
    A = np.stack([np.ones(len(C)), C[:, 0], 1e-6 * C[:, 0] * C[:, 1], 1e-6 * C[:, 0] ** 2 / 2], 1)
    (b0, b_tok, b_x, b_c), *_ = np.linalg.lstsq(A, C[:, 2], rcond=None)
    fit = {"model": args.model, "gpu": torch.cuda.get_device_name(),
           "decode": {"a0": a0, "a_c": a_c}, "chunk": {"b0": b0, "b_tok": b_tok, "b_x": b_x, "b_c": b_c},
           "decode_points": dec, "chunk_points": chunks}
    out = ROOT / "results" / "sim" / f"bench-{args.model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fit, indent=1))
    print(json.dumps({k: fit[k] for k in ("decode", "chunk")}, indent=1))


if __name__ == "__main__":
    main()

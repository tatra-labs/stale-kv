"""Offline acceptance study: draft on stale KV vs exact KV for sampled cache-miss requests.

Usage:
    python experiments/run_acceptance.py --model Qwen/Qwen3-0.6B --keep-last 5 --n 240

Writes one JSON line per request to results/acceptance/<model>-L<N>.jsonl and resumes if the
file already exists. Requests are sampled in one seeded order that depends only on the tokenizer,
so every model sees the same requests.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from stale_kv.kv import Engine  # noqa: E402
from stale_kv.measure import measure  # noqa: E402
from stale_kv.sessions import Renderer, load_parquet, miss_request  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ticks-00001-of-00008.parquet"


def sample_requests(sessions, renderer, keep_last, n, max_ctx, per_session, seed, reply_as_rendered=False):
    rng = random.Random(seed)
    order = list(range(len(sessions)))
    rng.shuffle(order)
    out = []
    for si in order:
        s = sessions[si]
        ts = [t for t in range(keep_last + 2, s.n_requests + 1)]
        rng.shuffle(ts)
        taken = 0
        for t in ts:
            req = miss_request(s, t, keep_last, renderer, reply_as_rendered)
            if len(req.new) > max_ctx or len(req.old) > max_ctx:
                continue
            out.append(req)
            taken += 1
            if taken >= per_session or len(out) >= n:
                break
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--keep-last", type=int, default=5)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--per-session", type=int, default=3)
    ap.add_argument("--k-long", type=int, default=111)
    ap.add_argument("--k-short", type=int, default=32)
    ap.add_argument("--min-run", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bf16-logits", action="store_true", help="round logits to the model dtype (ablation)")
    ap.add_argument("--reply-as-rendered", action="store_true",
                    help="invalidated state holds the reply without the think block (paper App. B.3 check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=1500, help="trajectories to load from the shard")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = Renderer(tok)
    sessions = load_parquet(str(DATA), limit=args.sessions)
    reqs = sample_requests(sessions, renderer, args.keep_last, args.n, args.max_ctx, args.per_session, args.seed,
                           args.reply_as_rendered)
    print(f"{len(sessions)} sessions, {len(reqs)} miss requests sampled", flush=True)

    name = args.model.split("/")[-1]
    suffix = ("" if args.dtype == "bfloat16" else f"-{args.dtype}") + ("-bf16logits" if args.bf16_logits else "")
    suffix += "-asrendered" if args.reply_as_rendered else ""
    out = Path(args.out) if args.out else ROOT / "results" / "acceptance" / f"{name}-L{args.keep_last}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text().splitlines() if line.strip()}

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()
    eng = Engine(model, max_len=args.max_ctx + args.k_long + 8, fp32_logits=not args.bf16_logits)
    think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    exact, work = eng.buffer(), eng.buffer()
    stop = {renderer.im_end, tok.convert_tokens_to_ids("<|endoftext|>")}
    t_start = time.time()
    with out.open("a") as f:
        for i, req in enumerate(reqs):
            if req.key in done:
                continue
            rec = measure(eng, exact, work, req, stop, k_long=args.k_long, k_short=args.k_short,
                          min_run=args.min_run, ngram_strip=think)
            rec["model"] = name
            rec["dtype"] = args.dtype
            rec["fp32_logits"] = not args.bf16_logits
            f.write(json.dumps(rec) + "\n")
            f.flush()
            s = rec["stale"]
            print(
                f"[{i + 1}/{len(reqs)}] {req.key} new={rec['len_new']} reused={rec['reused']} "
                f"staged={rec['staged']} stale a={s['a_fixed']}/{len(s['drafts'])} "
                f"exact a={rec['exact']['a_fixed']}/{len(rec['exact']['drafts'])} "
                f"ngram={rec['ngram']['a_fixed']} nomid={rec['nomid']['a_fixed']} norot={rec['norot']['a_fixed']} "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB {time.time() - t_start:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()

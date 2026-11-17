"""Modelled serving latency: plain prefill-decode disaggregation vs Callo, replaying SWE-agent sessions.

    python experiments/run_sim.py --keep-last 5 --records Qwen3-0.6B-L5

1. Builds (and caches) the token-level workload: for every request of the first --sessions
   trajectories, its prompt length, prefix-cache hit, reusable stale tokens, staged tokens, reply
   length and whether it is a cache miss. These come from rendering and aligning real prompts
   (no model is run).
2. Finds the plain system's capacity on the paper's H200 step model, then compares mean latency at
   12%, 52% and 100% of it (the paper's light / moderate / full loads), with draft acceptance replayed
   from the named acceptance records (and, as an upper bound, from the exact-KV drafting records).

Every number this prints is modelled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from stale_kv.align import align, summarize  # noqa: E402
from stale_kv.analysis import fit_table, is_calibration  # noqa: E402
from stale_kv.sim import PAPER_H200, AcceptanceModel, Config, Request, simulate  # noqa: E402

DATA = ROOT / "data" / "raw" / "ticks-00001-of-00008.parquet"


def build_workload(keep_last: int, n_sessions: int, offset: int) -> list[list[dict]]:
    from transformers import AutoTokenizer

    from stale_kv.sessions import Renderer, load_parquet

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    rend = Renderer(tok)
    sessions = load_parquet(str(DATA), limit=offset + n_sessions)[offset:]
    out = []
    for si, s in enumerate(sessions):
        reqs, prev = [], None
        for t in range(1, s.n_requests + 1):
            new = rend.prompt(s.messages(t, keep_last))
            reply = rend.reply(s.replies[t - 1])
            if prev is None:
                H = R = st = 0
            else:
                sm = summarize(align(prev, new[:-1]), len(prev))
                H, R, st = sm["prefix"], sm["reused"], sm["staged"]
            reqs.append({"t": t, "L": len(new), "H": H, "R": R, "s": st, "out": len(reply),
                         "miss": s.is_miss(t, keep_last)})
            prev = new + reply
        out.append(reqs)
        print(f"workload L{keep_last}: session {si + 1}/{len(sessions)}", end="\r", flush=True)
    print()
    return out


def load_records(name):
    p = ROOT / "results" / "acceptance" / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def to_requests(workload):
    return [[Request(0, q["t"], q["L"], q["H"], q["R"], q["s"], q["out"], q["miss"]) for q in sess] for sess in workload]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-last", type=int, default=5)
    ap.add_argument("--records", nargs="+", default=["Qwen3-0.6B-L5"])
    ap.add_argument("--sessions", type=int, default=100, help="trajectories in the workload")
    ap.add_argument("--offset", type=int, default=1500, help="skip the trajectories the acceptance runs sample from")
    ap.add_argument("--n", type=int, default=300, help="sessions replayed per simulation")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cache = ROOT / "results" / "sim" / f"workload-L{args.keep_last}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        workload = json.loads(cache.read_text())
    else:
        workload = build_workload(args.keep_last, args.sessions, args.offset)
        cache.write_text(json.dumps(workload))
    sessions = to_requests(workload)
    flat = [q for s in workload for q in s]
    miss = [q for q in flat if q["miss"]]
    wl = {
        "sessions": len(workload), "requests": len(flat), "miss_rate": len(miss) / len(flat),
        "prefill_per_miss_p50": float(np.median([q["L"] - q["H"] for q in miss])),
        "prefill_per_miss_p90": float(np.percentile([q["L"] - q["H"] for q in miss], 90)),
        "reusable_p50": float(np.median([q["R"] for q in miss])),
        "reusable_p90": float(np.percentile([q["R"] for q in miss], 90)),
        "staged_p50": float(np.median([q["s"] for q in miss])),
        "reply_p50": float(np.median([q["out"] for q in flat])),
        "prompt_p50": float(np.median([q["L"] for q in flat])),
    }
    print("workload", json.dumps(wl))

    base = Config(PAPER_H200, callo=False)
    # Capacity: the highest session rate at which plain serving's mean latency stays under 3x its
    # light-load value (our stand-in for the paper's window test).
    low = np.mean(simulate(sessions, 0.002, base, None, n_sessions=args.n, seed=0)["lat_all"])
    lo, hi = 0.002, 0.5
    for _ in range(14):
        mid = (lo * hi) ** 0.5
        lat = np.mean(simulate(sessions, mid, base, None, n_sessions=args.n, seed=0)["lat_all"])
        lo, hi = (mid, hi) if lat <= 3 * low else (lo, mid)
    capacity = lo
    print(f"light-load mean latency {low:.2f}s, capacity {capacity:.4f} sessions/s (modelled)")

    results = {"workload": wl, "capacity_sessions_per_s": capacity, "light_load_latency_s": float(low),
               "step_model": PAPER_H200.name, "rows": []}
    sources = []
    for name in args.records:
        recs = load_records(name)
        table = fit_table([r for r in recs if is_calibration(r)])
        evals = [r for r in recs if not is_calibration(r)]
        for method in ("stale", "exact"):
            rows = [{"reused": r["reused"], **r[method]} for r in evals]
            sources.append((f"{name}:{method}", AcceptanceModel(rows, table)))
    for frac, label in ((0.12, "light"), (0.52, "moderate"), (1.0, "full")):
        rate = frac * capacity
        b = [simulate(sessions, rate, base, None, n_sessions=args.n, seed=sd) for sd in range(args.seeds)]
        b_all = np.mean([np.mean(x["lat_all"]) for x in b])
        b_miss = np.mean([np.mean(x["lat_miss"]) for x in b])
        for src, acc in sources:
            c = [simulate(sessions, rate, Config(PAPER_H200, callo=True), acc, n_sessions=args.n, seed=sd)
                 for sd in range(args.seeds)]
            c_all = np.mean([np.mean(x["lat_all"]) for x in c])
            c_miss = np.mean([np.mean(x["lat_miss"]) for x in c])
            drafts = sum(x["drafts"] for x in c)
            row = {"load": label, "rate": rate, "acceptance": src,
                   "speedup_miss": b_miss / c_miss, "speedup_all": b_all / c_all,
                   "base_lat_all": b_all, "callo_lat_all": c_all, "base_lat_miss": b_miss, "callo_lat_miss": c_miss,
                   "admitted_share_of_misses": sum(x["admitted"] for x in c) / max(sum(len(x["lat_miss"]) for x in c), 1),
                   "draft_acceptance": sum(x["accepted"] for x in c) / max(drafts, 1),
                   "mean_drafts_per_verify": drafts / max(sum(x["admitted"] for x in c), 1),
                   "parked": sum(x["parked"] for x in c)}
            results["rows"].append(row)
            print(f"{label:8s} {src:28s} miss x{row['speedup_miss']:.3f} all x{row['speedup_all']:.3f} "
                  f"(base {b_all:.2f}s / {b_miss:.2f}s) admitted {row['admitted_share_of_misses']:.2f} "
                  f"accept {row['draft_acceptance']:.2f}", flush=True)
    out = ROOT / "results" / "sim" / f"sim-L{args.keep_last}.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

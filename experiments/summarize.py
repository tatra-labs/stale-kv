"""Tables and figures from results/acceptance/*.jsonl.

Writes results/summary.json and docs/figures/*.png, and prints the README tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stale_kv.analysis import BIN_NAMES, early_stop, fit_table, gap_table, is_calibration  # noqa: E402
from stale_kv.measure import whole  # noqa: E402

RUNS = ["Qwen3-0.6B-L5", "Qwen3-1.7B-L5", "Qwen3-0.6B-L10", "Qwen3-1.7B-L10",
        "Qwen3-0.6B-L5-asrendered", "Qwen3-0.6B-L5-bf16logits"]
ABLATIONS = {"Qwen3-0.6B-L5-asrendered": "Qwen3-0.6B-L5", "Qwen3-0.6B-L5-bf16logits": "Qwen3-0.6B-L5"}
PAPER_TABLE6 = {
    "Llama-3.3-70B": [0.63, 0.69, 0.89, 0.97, 1.00, 1.00, 1.00],
    "Mistral-Small-3.2": [0.70, 0.92, 0.99, 1.00, 1.00, 1.00, 1.00],
    "Gemma-4-31B": [0.57, 0.78, 0.86, 0.95, 1.00, 1.00, 1.00],
}
STOP = {151645, 151643}  # <|im_end|>, <|endoftext|> in the Qwen3 tokenizer


def load(name):
    p = ROOT / "results" / "acceptance" / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def run_length(m: dict, cap: int) -> int:
    """Accepted drafts, counting a fully accepted reply that ended on a stop token as reaching cap."""
    d, a = m["drafts"], m["a_fixed"]
    if d and a == len(d) and (d[-1] in STOP or len(d) >= cap):
        return cap
    return a


def stats(recs):
    out = {"n": len(recs)}
    if not recs:
        return out
    arr = lambda f: np.array([f(r) for r in recs], dtype=float)  # noqa: E731
    out["len_new_p50"] = float(np.median(arr(lambda r: r["len_new"])))
    out["len_new_p90"] = float(np.percentile(arr(lambda r: r["len_new"]), 90))
    out["reused_p50"] = float(np.median(arr(lambda r: r["reused"])))
    out["staged_p50"] = float(np.median(arr(lambda r: r["staged"])))
    out["prefix_p50"] = float(np.median(arr(lambda r: r["prefix"])))
    for m in ("stale", "exact", "ngram", "nomid", "norot"):
        a = arr(lambda r: r[m]["a_fixed"])
        out[f"{m}_whole32"] = float(np.mean([whole(r[m]["drafts"], r[m]["a_fixed"], 32, STOP) for r in recs]))
        out[f"{m}_acc32"] = float(np.mean(np.minimum(a, 32)))
        if m in ("stale", "exact"):
            out[f"{m}_whole32_anymax"] = float(np.mean([whole(r[m]["drafts"], r[m]["a_anymax"], 32, STOP) for r in recs]))
            out[f"{m}_acc111"] = float(np.mean(a))
            out[f"{m}_whole111"] = float(np.mean([whole(r[m]["drafts"], r[m]["a_fixed"], 111, STOP) for r in recs]))
            out[f"{m}_drafted111"] = float(np.mean(arr(lambda r: len(r[m]["drafts"]))))
    tf_n = arr(lambda r: r["teacher_forced"]["n"]).sum()
    out["tf_agree"] = float(arr(lambda r: r["teacher_forced"]["agree"]).sum() / tf_n)
    tv = np.concatenate([r["teacher_forced"]["tv"] for r in recs])
    out["tf_tv_le_0.01"] = float(np.mean(tv <= 0.01))
    out["tf_tv_median"] = float(np.median(tv))
    # Near ties (gap < 0.5 on the draft path, up to the first rejection).
    pos_gaps, first_miss_gaps = [], []
    for r in recs:
        s = r["stale"]
        a = s["a_fixed"]
        pos_gaps += s["gaps"][: a + 1]
        if a < len(s["drafts"]):
            first_miss_gaps.append(s["gaps"][a])
    pos_gaps, first_miss_gaps = np.array(pos_gaps), np.array(first_miss_gaps)
    out["near_tie_share_of_positions"] = float(np.mean(pos_gaps < 0.5))
    out["near_tie_share_of_first_mismatches"] = float(np.mean(first_miss_gaps < 0.5)) if len(first_miss_gaps) else None
    hit, tot = gap_table(recs)
    out["agree_gap_ge1"] = float(hit[3:].sum() / max(tot[3:].sum(), 1))
    out["agree_gap_lt05"] = float(hit[:2].sum() / max(tot[:2].sum(), 1))
    return out


def early_stop_eval(recs):
    cal = [r for r in recs if is_calibration(r)]
    ev = [r for r in recs if not is_calibration(r)]
    table = fit_table(cal)
    _, tot = gap_table(cal)
    kept_drafts = made = acc_all = acc_kept = 0
    for r in ev:
        s = r["stale"]
        k = early_stop(s["gaps"], table)
        made += len(s["drafts"])
        kept_drafts += k
        acc_all += s["a_fixed"]
        acc_kept += min(s["a_fixed"], k)
    return {
        "n_calibration": len(cal), "n_eval": len(ev), "table": [round(float(x), 3) for x in table],
        "table_support": tot.astype(int).tolist(),
        "drafts_kept_share": kept_drafts / max(made, 1), "accepted_retained_share": acc_kept / max(acc_all, 1),
        "wasted_drafts_without_stop": (made - acc_all) / max(made, 1),
        "wasted_drafts_with_stop": (kept_drafts - acc_kept) / max(kept_drafts, 1),
    }


def figures(all_recs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    runs = [n for n in RUNS[:4] if all_recs.get(n)]
    methods = [("exact", "exact KV (ceiling)", "#4c78a8"), ("stale", "stale KV (Callo)", "#f58518"),
               ("norot", "stale, no re-rotation", "#b279a2"), ("nomid", "valid prefix only", "#54a24b"),
               ("ngram", "prompt-lookup n-gram", "#9d9da1")]

    # 1. Whole-window acceptance, 32-token window.
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.16
    for j, (m, label, c) in enumerate(methods):
        vals = [100 * np.mean([whole(r[m]["drafts"], r[m]["a_fixed"], 32, STOP) for r in all_recs[n]]) for n in runs]
        xs = np.arange(len(runs)) + (j - 2) * w
        ax.bar(xs, vals, w, label=label, color=c)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(runs)), [n.replace("Qwen3-", "") for n in runs])
    ax.set_ylabel("miss requests whose whole\n32-token window is accepted (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "whole_window.png", dpi=160)
    plt.close(fig)

    # 2. Accepted-length survival, 111-token cap.
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, runs):
        for m, label, c in methods[:2] + methods[3:4]:
            cap = 32 if m == "nomid" else 111
            a = np.array([run_length(r[m], cap) for r in all_recs[n]])
            xs = np.arange(0, cap + 1)
            ax.plot(xs, [100 * np.mean(a >= x) for x in xs], label=label, color=c)
        ax.set_title(n.replace("Qwen3-", ""), fontsize=9)
        ax.set_xlabel("first k drafts all accepted")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("miss requests (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "accepted_length.png", dpi=160)
    plt.close(fig)

    # 3. Agreement by logit gap vs the paper's Table 6.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(BIN_NAMES))
    for name, vals in PAPER_TABLE6.items():
        ax.plot(xs, vals, "--", color="#bbbbbb", lw=1)
    ax.plot([], [], "--", color="#bbbbbb", label="paper (Llama-70B, Mistral-24B, Gemma-31B)")
    for n, c in zip(runs, ["#f58518", "#e45756", "#4c78a8", "#72b7b2"]):
        hit, tot = gap_table(all_recs[n])
        ok = tot >= 10
        ax.plot(xs[ok], (hit / np.maximum(tot, 1))[ok], "o-", color=c, label=n.replace("Qwen3-", ""))
    ax.set_xticks(xs, BIN_NAMES)
    ax.set_xlabel("top-1 minus top-2 logit gap of the draft (bins with >= 10 drafts)")
    ax.set_ylabel("P(draft accepted)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "gap_table.png", dpi=160)
    plt.close(fig)

    # 4. Edit size vs acceptance (Spearman per run; the plot shows the 0.6B L5 pair).
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    pair = [("Qwen3-0.6B-L5", "default", "#f58518"), ("Qwen3-0.6B-L5-asrendered", "reply as later prompts render it", "#4c78a8")]
    keys = {r["key"] for r in all_recs.get("Qwen3-0.6B-L5-asrendered", [])}
    for n, label, c in pair:
        recs = [r for r in all_recs.get(n, []) if r["key"] in keys]
        if not recs:
            continue
        drop = np.array([r["dropped_old"] for r in recs])
        acc = np.array([min(r["stale"]["a_fixed"], 32) for r in recs])
        edges = np.quantile(drop, [0, 0.25, 0.5, 0.75, 1.0])
        idx = np.clip(np.searchsorted(edges, drop, side="right") - 1, 0, 3)
        ax.plot(range(4), [acc[idx == q].mean() for q in range(4)], "o-", color=c, label=f"0.6B L5, {label}")
    ax.set_xticks(range(4), ["smallest", "2nd", "3rd", "largest"])
    ax.set_xlabel("quartile of invalidated tokens dropped by the edit (same 100 requests)")
    ax.set_ylabel("stale drafts accepted\n(of the first 32)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "staleness.png", dpi=160)
    plt.close(fig)

    # 5. Modelled serving speedups, if the simulator has run.
    sims = {L: ROOT / "results" / "sim" / f"sim-L{L}.json" for L in (5, 10)}
    sims = {L: json.loads(p.read_text()) for L, p in sims.items() if p.exists()}
    if sims:
        fig, axes = plt.subplots(1, len(sims), figsize=(4.6 * len(sims), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (L, sim) in zip(axes, sims.items()):
            srcs = list(dict.fromkeys(r["acceptance"] for r in sim["rows"]))
            loads = ["light", "moderate", "full"]
            w = 0.8 / len(srcs)
            for j, src in enumerate(srcs):
                vals = [next(r["speedup_miss"] for r in sim["rows"] if r["load"] == ld and r["acceptance"] == src)
                        for ld in loads]
                model, method = src.split(":")
                label = f"{model.split('-')[1]} {'stale KV' if method == 'stale' else 'exact KV (upper bound)'}"
                ax.bar(np.arange(3) + (j - (len(srcs) - 1) / 2) * w, vals, w, label=label)
            ax.axhline(1.0, color="#555555", lw=0.8)
            ax.set_xticks(range(3), loads)
            ax.set_title(f"SWE-agent L{L} (modelled)", fontsize=9)
            ax.set_ylim(0.95, max(1.1, ax.get_ylim()[1] + 0.01))
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("miss-request latency, plain / Callo")
        axes[-1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "sim_speedup.png", dpi=160)
        plt.close(fig)


def main():
    all_recs = {n: load(n) for n in RUNS}
    summary = {}
    for n, recs in all_recs.items():
        if not recs:
            continue
        summary[n] = stats(recs)
        summary[n]["early_stop"] = early_stop_eval(recs)
        hit, tot = gap_table(recs)
        summary[n]["gap_table"] = {b: (round(float(h / t), 3) if t else None, int(t)) for b, h, t in zip(BIN_NAMES, hit, tot)}
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["n", "len_new_p50", "reused_p50", "staged_p50", "stale_whole32", "stale_whole32_anymax", "exact_whole32",
            "exact_whole32_anymax", "ngram_whole32", "nomid_whole32", "norot_whole32", "stale_acc32", "exact_acc32",
            "norot_acc32", "nomid_acc32", "ngram_acc32",
            "stale_acc111", "exact_acc111", "tf_agree", "tf_tv_le_0.01", "agree_gap_ge1", "agree_gap_lt05",
            "near_tie_share_of_positions", "near_tie_share_of_first_mismatches"]
    names = list(summary)
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        cells = []
        for n in names:
            v = summary[n].get(k)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"| {k} | " + " | ".join(cells) + " |")
    for n in names:
        print(n, "gap table", summary[n]["gap_table"])
        print(n, "early stop", summary[n]["early_stop"])
    # Does acceptance fall with the size of the span the edit removed? Spearman per run.
    from scipy.stats import spearmanr

    for n, recs in all_recs.items():
        if len(recs) >= 20:
            rho, pv = spearmanr([r["dropped_old"] for r in recs], [min(r["stale"]["a_fixed"], 32) for r in recs])
            summary[n]["spearman_dropped_vs_accepted32"] = {"rho": float(rho), "p": float(pv)}
            print(n, f"dropped vs accepted@32: Spearman rho {rho:.3f}, p {pv:.3f}")
    # Ablations, compared with their base run on the requests both contain.
    for abl, base in ABLATIONS.items():
        if not all_recs.get(abl):
            continue
        by_key = {r["key"]: r for r in all_recs[base]}
        pairs = [(by_key[r["key"]], r) for r in all_recs[abl] if r["key"] in by_key]
        out = {"n": len(pairs)}
        for tag, side in (("base", 0), ("ablation", 1)):
            recs = [p[side] for p in pairs]
            for m in ("stale", "exact"):
                out[f"{tag}_{m}_whole32"] = float(np.mean([whole(r[m]["drafts"], r[m]["a_fixed"], 32, STOP) for r in recs]))
                out[f"{tag}_{m}_acc32"] = float(np.mean([min(r[m]["a_fixed"], 32) for r in recs]))
        summary[abl]["vs_base"] = out
        print(abl, "vs", base, json.dumps({k: round(v, 3) for k, v in out.items()}))
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    figures(all_recs)


if __name__ == "__main__":
    main()

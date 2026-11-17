"""Prompt-lookup n-gram drafting, the model-free baseline.

Match the longest suffix (up to ``max_n`` tokens) of the context against an earlier occurrence and
propose the tokens that followed it. Repeat on the extended context until ``k`` tokens are drafted
or nothing matches. The most recent occurrence wins, which suits agent transcripts where the latest
file view or command is the likeliest to be quoted again.
"""

from __future__ import annotations


def prompt_lookup(context: list[int], k: int, max_n: int = 3, stop: set[int] | None = None) -> list[int]:
    stop = stop or set()
    ctx = list(context)
    last: dict[tuple[int, ...], int] = {}  # n-gram -> end index (exclusive) of its latest occurrence

    def index_upto(end: int, start: int) -> None:
        for e in range(max(start, 1), end + 1):
            for n in range(1, max_n + 1):
                if e - n >= 0:
                    last[tuple(ctx[e - n : e])] = e

    # Index every n-gram that ends strictly before the current end, so the suffix cannot match itself.
    index_upto(len(ctx) - 1, 1)
    out: list[int] = []
    while len(out) < k:
        cont: list[int] = []
        for n in range(min(max_n, len(ctx)), 0, -1):
            e = last.get(tuple(ctx[-n:]))
            if e is not None and e < len(ctx):
                cont = ctx[e : e + (k - len(out))]
                break
        if not cont:
            break
        hit_stop = False
        for tok in cont:
            before = len(ctx)
            ctx.append(tok)
            out.append(tok)
            index_upto(before, before)
            if tok in stop:
                hit_stop = True
                break
        if hit_stop:
            break
    return out

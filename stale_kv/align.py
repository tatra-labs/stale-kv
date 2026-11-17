"""In-order alignment of the invalidated token sequence with the new prompt.

Callo "finds runs of identical tokens and keeps them in order", so that every position of the new
prompt either maps to a position of the previous request or is new. This module produces that plan:

* ``keep`` - the common prefix. Same tokens at the same positions, so this KV is exact (it is what an
  ordinary prefix cache serves).
* ``map``  - a run of the old sequence that reappears later in the new prompt, shifted by
  ``delta = new_start - old_start``. Its KV is *stale*: those tokens once attended to context that
  the edit removed. Keys must be re-rotated by ``delta``; values are copied unchanged.
* ``new``  - tokens with no counterpart (the placeholder, the new tool output, the generation
  prompt). A decode instance computes these itself, as chunked prefill over the provisional cache.

Runs are found greedily from left to right with a hashed ``min_run``-gram index, then extended
both ways, so a run is never shorter than ``min_run`` tokens and runs never cross.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    kind: str  # "keep" | "map" | "new"
    new_start: int
    length: int
    old_start: int = -1

    @property
    def delta(self) -> int:
        return self.new_start - self.old_start if self.kind != "new" else 0


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    return p


def align(old: list[int], new: list[int], min_run: int = 16) -> list[Segment]:
    """Plan for building ``new``'s KV out of ``old``'s. Segments tile ``new`` exactly, in order."""
    if min_run < 1:
        raise ValueError("min_run must be >= 1")
    segs: list[Segment] = []
    p = common_prefix(old, new)
    if p:
        segs.append(Segment("keep", 0, p, 0))

    index: dict[tuple[int, ...], list[int]] = {}
    for i in range(p, len(old) - min_run + 1):
        index.setdefault(tuple(old[i : i + min_run]), []).append(i)

    cursor = p  # first old position still available (runs keep order)
    pending = p  # start of the not-yet-emitted run of new tokens
    j = p
    while j < len(new):
        i = -1
        if j + min_run <= len(new):
            cands = index.get(tuple(new[j : j + min_run]))
            if cands:
                k = bisect.bisect_left(cands, cursor)
                if k < len(cands):
                    i = cands[k]
        if i < 0:
            j += 1
            continue
        back = 0  # extend the run backwards into pending new tokens
        while j - back - 1 >= pending and i - back - 1 >= cursor and new[j - back - 1] == old[i - back - 1]:
            back += 1
        length = min_run  # and forwards as far as the tokens agree
        while j + length < len(new) and i + length < len(old) and new[j + length] == old[i + length]:
            length += 1
        if j - back > pending:
            segs.append(Segment("new", pending, j - back - pending))
        segs.append(Segment("map", j - back, length + back, i - back))
        cursor = i + length
        j += length
        pending = j
    if pending < len(new):
        segs.append(Segment("new", pending, len(new) - pending))
    return segs


def summarize(segs: list[Segment], old_len: int) -> dict:
    keep = sum(s.length for s in segs if s.kind == "keep")
    mapped = sum(s.length for s in segs if s.kind == "map")
    new = sum(s.length for s in segs if s.kind == "new")
    return {
        "prefix": keep,
        "reused": mapped,
        "staged": new,
        "dropped_old": old_len - keep - mapped,
        "n_map": sum(1 for s in segs if s.kind == "map"),
        "n_new": sum(1 for s in segs if s.kind == "new"),
    }


def check_plan(segs: list[Segment], old: list[int], new: list[int]) -> None:
    """Raise if the plan does not tile ``new`` or maps a position to a different token."""
    pos = 0
    last_old = -1
    for s in segs:
        if s.new_start != pos or s.length <= 0:
            raise AssertionError(f"segment {s} does not continue at {pos}")
        if s.kind in ("keep", "map"):
            if s.old_start < last_old:
                raise AssertionError("runs out of order")
            if old[s.old_start : s.old_start + s.length] != new[s.new_start : s.new_start + s.length]:
                raise AssertionError(f"segment {s} maps different tokens")
            last_old = s.old_start + s.length
        pos += s.length
    if pos != len(new):
        raise AssertionError("plan does not cover the new prompt")

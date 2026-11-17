"""Shared analysis helpers: the logit-gap agreement table and the early-stop rule (paper Table 6)."""

from __future__ import annotations

import hashlib

import numpy as np

BINS = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
BIN_NAMES = ["<.25", ".25-.5", ".5-1", "1-2", "2-4", "4-8", ">=8"]
THETA = 0.3


def gap_bin(g: float) -> int:
    return int(np.searchsorted(BINS, g, side="right") - 1)


def gap_table(recs: list[dict], method: str = "stale") -> tuple[np.ndarray, np.ndarray]:
    """Counts behind P(draft accepted | its logit-gap bin, every earlier draft accepted).

    Uses each request's drafts up to and including the first rejected one.
    """
    hit = np.zeros(len(BIN_NAMES))
    tot = np.zeros(len(BIN_NAMES))
    for r in recs:
        s = r[method]
        a = s["a_fixed"]
        for i, g in enumerate(s["gaps"][: a + 1]):
            b = gap_bin(g)
            tot[b] += 1
            hit[b] += i < a
    return hit, tot


def fit_table(recs: list[dict]) -> list[float]:
    hit, tot = gap_table(recs)
    return [float(h / t) if t else 1.0 for h, t in zip(hit, tot)]


def early_stop(gaps: list[float], table: list[float], theta: float = THETA) -> int:
    """Drafts made before the estimated cumulative acceptance drops below ``theta``."""
    p = 1.0
    for i, g in enumerate(gaps):
        p *= table[gap_bin(g)]
        if p < theta:
            return i
    return len(gaps)


def is_calibration(rec: dict) -> bool:
    """One session in five (by hash of its trajectory id) is held out to fit the early-stop table."""
    traj = rec["key"].rsplit("#", 1)[0]
    return int(hashlib.sha1(traj.encode()).hexdigest(), 16) % 5 == 0

import random

import numpy as np
import pytest

from stale_kv.analysis import early_stop, fit_table
from stale_kv.sim import PAPER_H200, AcceptanceModel, Config, Request, StepModel, simulate


def _sessions(n=6, T=10, miss_from=7, prompt=6000, reply=120):
    out = []
    for _ in range(n):
        reqs = []
        for t in range(1, T + 1):
            miss = t >= miss_from
            L = prompt + 300 * t
            H = 0 if t == 1 else (L - 2500 if miss else L - 400)
            reqs.append(Request(0, t, L, H, 2000 if miss else 120, 80 if miss else 400, reply, miss))
        out.append(reqs)
    return out


def _records(a, n=40, length=111):
    return [{"reused": 2000, "drafts": [1] * length, "a_fixed": a, "gaps": [5.0] * length} for _ in range(n)]


def test_step_model_matches_paper_equation():
    s = PAPER_H200
    assert StepModel.G(128) == 0 and StepModel.G(129) == 1 and StepModel.G(600) == 3
    assert s.decode(16, 16 * 20_000, 0, 0, 16) == pytest.approx(22.45 + 0.111 * 16 + 0.108 * 320)
    assert s.prefill(8192, 0) == pytest.approx(22.45 + 30.8 * 32 + 1e-6 * 13.7 * 8192**2 / 2)
    assert s.transfer(1000) == pytest.approx(1000 * 1000 * 327_680 / 69.2e9)


def test_every_request_completes_and_plain_ignores_acceptance():
    sess = _sessions()
    base = simulate(sess, 0.05, Config(PAPER_H200), None, n_sessions=8, seed=0)
    assert len(base["lat_all"]) == 8 * 10 and len(base["lat_miss"]) == 8 * 4
    again = simulate(sess, 0.05, Config(PAPER_H200), AcceptanceModel(_records(50), [1.0] * 7),
                     n_sessions=8, seed=0)
    assert again["lat_all"] == base["lat_all"]


def test_callo_with_perfect_drafts_is_faster_on_misses_and_equal_on_hits():
    sess = _sessions()
    base = simulate(sess, 0.01, Config(PAPER_H200), None, n_sessions=6, seed=1)
    acc = AcceptanceModel(_records(111), [1.0] * 7)
    callo = simulate(sess, 0.01, Config(PAPER_H200, callo=True), acc, n_sessions=6, seed=1)
    assert callo["admitted"] > 0 and callo["accepted"] == callo["drafts"] > 0
    assert np.mean(callo["lat_miss"]) < np.mean(base["lat_miss"])
    hit_b = [d[2] for d in base["done"] if not d[3]]
    assert len(hit_b) == len(base["done"])


def test_callo_with_useless_drafts_costs_little():
    sess = _sessions()
    base = simulate(sess, 0.01, Config(PAPER_H200), None, n_sessions=6, seed=2)
    acc = AcceptanceModel(_records(0), [1.0] * 7)
    callo = simulate(sess, 0.01, Config(PAPER_H200, callo=True), acc, n_sessions=6, seed=2)
    assert callo["accepted"] == 0
    # A rejected window still commits the correction token, so misses are never much slower.
    assert np.mean(callo["lat_miss"]) < 1.05 * np.mean(base["lat_miss"])


def test_acceptance_model_extends_censored_records_with_kaplan_meier():
    recs = [{"reused": 100, "drafts": [1] * 10, "a_fixed": 10, "gaps": [3.0] * 10}]  # censored at 10
    recs += [{"reused": 100, "drafts": [1] * 111, "a_fixed": 20, "gaps": [3.0] * 111} for _ in range(9)]
    acc = AcceptanceModel(recs, [1.0] * 7)
    assert acc.hazard[20] == pytest.approx(1.0)  # every record still at risk at 20 is rejected there
    rng = random.Random(0)
    for _ in range(50):
        a, gaps = acc.draw(100, rng)
        assert a == 20
        assert len(gaps) >= 10


def test_early_stop_rule():
    table = [0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0]
    assert early_stop([5.0] * 20, table) == 20
    assert early_stop([0.1, 0.1], table, theta=0.3) == 1  # 0.5, then 0.25 < 0.3
    assert early_stop([0.6, 0.6, 0.6, 0.6], table, theta=0.7) == 3  # 0.9^4 = 0.66
    recs = [{"stale": r} for r in _records(5, n=3, length=10)]  # gap 5.0 falls in the 4-8 bin
    assert fit_table(recs)[5] == pytest.approx(5 / 6)
    assert fit_table([])[0] == 1.0

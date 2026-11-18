"""A discrete-event model of one prefill (P) and one decode (D) instance, with and without Callo.

Everything this module reports is *modelled*. Step times come from a fitted cost model (the paper's
Eq. 2 for Llama-3.3-70B FP8 on H200); draft acceptance is replayed from measured records. Simplifications against the paper's simulator: one P and one D
instance, FIFO prefill with whole 8,192-token chunks, a fixed per-transfer bandwidth (no link
sharing), closed-loop sessions with a fixed tool time, and yielding that parks the most recently
admitted speculator first.

Request lifecycle, both systems: the prompt's uncached suffix is prefilled on P, its KV is
transferred, and D decodes the reply in a batch of at most ``batch`` requests.

Callo adds, for an admitted cache miss: take a free D slot on arrival, stage the spans that have no
stale counterpart (``staged`` tokens, at most ``chunk`` per step), then draft one token per D step on
the provisional KV until the exact KV arrives, an early-stop rule fires, or the draft cap is hit.
When the exact KV arrives the drafts are verified in one (k+1)-row step, and the accepted prefix plus
one token are committed.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np

from .analysis import early_stop


@dataclass
class StepModel:
    """Engine step times in ms (Callo paper, App. F.1, Eq. 2)."""

    a0: float
    a_r: float
    a_c: float  # per 1k context tokens per scheduled request
    a_p: float  # per locally prefilled (staged) token
    a_v: float  # per verification token
    a_t: float  # per 256-row group above 128 scheduled tokens
    b0: float
    b_t: float
    b_x: float  # ms per million (chunk x prior-context) token pairs
    b_c: float  # ms per million (chunk^2 / 2) token pairs
    kv_bytes_per_token: float
    bandwidth: float  # bytes / s for one transfer
    name: str = ""

    @staticmethod
    def G(n: int) -> int:
        return 0 if n <= 128 else math.ceil(n / 256)

    def decode(self, r: int, ctx_sum: float, n_p: int, n_v: int, n: int) -> float:
        return self.a0 + self.a_r * r + self.a_c * ctx_sum / 1000 + self.a_p * n_p + self.a_v * n_v + self.a_t * self.G(n)

    def prefill(self, c: int, x: int) -> float:
        return self.b0 + self.b_t * self.G(c) + 1e-6 * (self.b_x * c * x + self.b_c * c * c / 2)

    def transfer(self, tokens: int) -> float:
        return 1000 * tokens * self.kv_bytes_per_token / self.bandwidth


# The paper's fit (Llama-3.3-70B FP8, H200 NVL pair over NVLink; App. F.2).
PAPER_H200 = StepModel(a0=22.45, a_r=0.111, a_c=0.108, a_p=0.148, a_v=0.092, a_t=12.3,
                       b0=22.45, b_t=30.8, b_x=5.16, b_c=13.7,
                       kv_bytes_per_token=327_680, bandwidth=69.2e9, name="paper H200 / Llama-3.3-70B")


@dataclass
class Request:
    sid: int
    t: int
    L: int  # prompt tokens
    H: int  # prefix-cache hit (tokens reused exactly)
    R: int  # reusable invalidated tokens (stale runs)
    s: int  # tokens to stage locally for a provisional KV
    out: int  # reply tokens to decode
    miss: bool  # this request's context edit invalidated a reused span


class AcceptanceModel:
    """Replays measured drafting records.

    Each record has the drafts' logit gaps and the index of the first rejected draft (or none, if
    every draft made was accepted). A request borrows a record from the five measured requests closest
    in reusable length, as the paper's simulator does. A record with no rejection is extended past its
    end with Kaplan-Meier rejection probabilities pooled over all records.
    """

    def __init__(self, records: list[dict], gap_table: list[float], cap: int = 111):
        self.cap = cap
        self.table = gap_table
        rows = []
        for r in records:
            d = r["drafts"]
            a = r["a_fixed"]
            rows.append((r["reused"], a, a < len(d), r["gaps"]))
        rows.sort(key=lambda x: x[0])
        self.rows = rows
        self.R = np.array([x[0] for x in rows])
        times = np.array([x[1] for x in rows])
        events = np.array([x[2] for x in rows])
        self.hazard = np.zeros(cap + 1)
        for i in range(cap + 1):
            at_risk = (times >= i).sum()
            ev = ((times == i) & events).sum()
            self.hazard[i] = ev / at_risk if at_risk else 1.0

    def draw(self, R: int, rng: random.Random) -> tuple[int, list[float]]:
        """(accepted-run length, gaps) for a request with ``R`` reusable tokens."""
        j = int(np.searchsorted(self.R, R))
        lo, hi = max(0, j - 3), min(len(self.rows), j + 2)
        _, a, rejected, gaps = self.rows[rng.randrange(lo, max(hi, lo + 1))]
        if rejected:
            return a, gaps
        k = a
        while k < self.cap and rng.random() >= self.hazard[k]:
            k += 1
        return k, gaps + [gaps[-1] if gaps else 0.0] * (self.cap - len(gaps))

    def early_stop(self, gaps: list[float], theta: float) -> int:
        g = gaps[: self.cap]
        i = early_stop(g, self.table, theta)
        return i if i < len(g) else self.cap


@dataclass
class Config:
    step: StepModel
    callo: bool = False
    batch: int = 16
    chunk: int = 112  # staging tokens per request per step (C_s = 128 - batch limit)
    draft_cap: int = 111
    min_reusable: int = 256
    stage_budget: int = 4096  # K: staged tokens x current batch size (incl. this request) <= K
    theta: float = 0.3
    p_chunk: int = 8192
    tool_time: float = 2.0  # s between a reply and the next request of the session


@dataclass
class _Live:
    req: Request
    arrival: float
    state: str = "wait"  # wait | stage | draft | parked | verify | decode
    stage_left: int = 0
    drafts: int = 0
    stop_at: int = 0
    accept: int = 0
    exact: bool = False
    remaining: int = 0
    ctx: int = 0
    speculated: bool = False
    p_left: int = 0
    p_done: int = 0


@dataclass(order=True)
class _Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    obj: object = field(compare=False, default=None)


def simulate(sessions: list[list[Request]], rate: float, cfg: Config, acc: AcceptanceModel | None,
             n_sessions: int = 300, seed: int = 0) -> dict:
    """Replay ``n_sessions`` sessions (sampled with replacement) arriving as a Poisson process at
    ``rate`` sessions/s. Returns per-request latencies (s) and counters."""
    rng = random.Random(seed)
    srng = random.Random(seed + 1)  # session choice and arrivals: identical across configs
    arng = random.Random(seed + 2)  # acceptance draws
    ev: list[_Event] = []
    seq = 0

    def push(t, kind, obj=None):
        nonlocal seq
        seq += 1
        heapq.heappush(ev, _Event(t, seq, kind, obj))

    t = 0.0
    for i in range(n_sessions):
        t += srng.expovariate(rate)
        push(t, "session", (i, srng.randrange(len(sessions))))

    p_queue: list[_Live] = []
    p_busy = False
    d_running: list[_Live] = []
    d_ready: list[_Live] = []
    d_busy = False
    lat_all, lat_miss, done = [], [], []
    stats = {"admitted": 0, "drafts": 0, "accepted": 0, "committed_by_verify": 0, "parked": 0}
    sess_reqs: dict[int, list[Request]] = {}

    def arrive(now, live: _Live):
        nonlocal p_busy
        r = live.req
        live.p_left = r.L - r.H
        live.p_done = r.H
        live.ctx = r.L
        p_queue.append(live)
        if (cfg.callo and acc is not None and r.miss and r.R >= cfg.min_reusable
                and r.s * (len(d_running) + 1) <= cfg.stage_budget and len(d_running) < cfg.batch and not d_ready):
            live.state = "stage"
            live.stage_left = r.s
            live.speculated = True
            live.accept, gaps = acc.draw(r.R, arng)
            live.stop_at = min(cfg.draft_cap, acc.early_stop(gaps, cfg.theta), r.out - 1)
            d_running.append(live)
            stats["admitted"] += 1
        kick(now)

    def kick(now):
        nonlocal p_busy, d_busy
        if not p_busy and p_queue:
            live = p_queue[0]
            c = min(cfg.p_chunk, live.p_left)
            push(now + cfg.step.prefill(c, live.p_done) / 1000, "p_step", (live, c))
            p_busy = True
        if not d_busy:
            # Yield: ready requests need slots that speculators are holding.
            while d_ready and len(d_running) >= cfg.batch:
                spec = [x for x in d_running if x.state in ("stage", "draft") and not x.exact]
                if not spec:
                    break
                x = spec[-1]
                d_running.remove(x)
                x.state = "parked"
                stats["parked"] += 1
            while d_ready and len(d_running) < cfg.batch:
                x = d_ready.pop(0)
                x.state = "verify" if x.drafts > 0 else "decode"
                if x.state == "decode":
                    x.remaining = x.req.out
                d_running.append(x)
            if d_running and any(x.state in ("decode", "verify", "stage") or
                                 (x.state == "draft" and x.drafts < x.stop_at) for x in d_running):
                r = n_p = n_v = n = 0
                ctx = 0.0
                plan = []
                for x in d_running:
                    if x.state == "decode":
                        r += 1; n += 1; ctx += x.ctx; plan.append((x, "decode", 1))
                    elif x.state == "verify":
                        k = x.drafts + 1
                        r += 1; n += k; n_v += k; ctx += x.ctx; plan.append((x, "verify", k))
                    elif x.state == "stage":
                        c = min(cfg.chunk, x.stage_left)
                        r += 1; n += c; n_p += c; ctx += x.ctx; plan.append((x, "stage", c))
                    elif x.state == "draft" and x.drafts < x.stop_at:
                        r += 1; n += 1; ctx += x.ctx; plan.append((x, "draft", 1))
                dt = cfg.step.decode(r, ctx, n_p, n_v, n) / 1000
                push(now + dt, "d_step", plan)
                d_busy = True

    def finish(now, x: _Live):
        lat = now - x.arrival
        lat_all.append(lat)
        if x.req.miss:
            lat_miss.append(lat)
        done.append((x.req.sid, x.req.t, lat, x.speculated))
        d_running.remove(x)
        sid = x.req.sid
        if x.req.t < len(sess_reqs[sid]):
            push(now + cfg.tool_time, "request", sess_reqs[sid][x.req.t])

    while ev:
        e = heapq.heappop(ev)
        now = e.time
        if e.kind == "session":
            i, si = e.obj
            reqs = [Request(i, q.t, q.L, q.H, q.R, q.s, q.out, q.miss) for q in sessions[si]]
            sess_reqs[i] = reqs
            push(now, "request", reqs[0])
        elif e.kind == "request":
            arrive(now, _Live(e.obj, now))
        elif e.kind == "p_step":
            live, c = e.obj
            live.p_left -= c
            live.p_done += c
            p_busy = False
            if live.p_left <= 0:
                p_queue.pop(0)
                push(now + cfg.step.transfer(live.req.L - live.req.H) / 1000, "kv", live)
            kick(now)
        elif e.kind == "kv":
            x = e.obj
            x.exact = True
            if x.state in ("stage", "draft"):
                x.state = "verify" if x.drafts > 0 else "decode"
                if x.state == "decode":
                    x.remaining = x.req.out
            else:  # waiting (never admitted) or parked
                d_ready.append(x)
            kick(now)
        elif e.kind == "d_step":
            d_busy = False
            for x, kind, k in e.obj:
                if kind == "decode":
                    x.remaining -= 1
                    x.ctx += 1
                    if x.remaining <= 0:
                        finish(now, x)
                elif kind == "verify":
                    committed = min(x.drafts, x.accept) + 1
                    stats["drafts"] += x.drafts
                    stats["accepted"] += min(x.drafts, x.accept)
                    stats["committed_by_verify"] += committed
                    x.ctx += committed
                    x.remaining = x.req.out - committed
                    x.state = "decode"
                    if x.remaining <= 0:
                        finish(now, x)
                elif kind == "stage":
                    x.stage_left -= k
                    if x.stage_left <= 0 and x.state == "stage":
                        x.state = "draft"
                elif kind == "draft":
                    if x.state == "draft":
                        x.drafts += 1
                    elif x.state == "verify":
                        pass  # exact KV arrived during this step; the draft row was wasted
            kick(now)

    return {"lat_all": lat_all, "lat_miss": lat_miss, "done": done, **stats}

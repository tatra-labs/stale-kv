"""Replay recorded SWE-agent trajectories as a sequence of model requests under observation masking.

SWE-agent's ``LastNObservations`` history processor keeps the last N tool outputs and replaces each
older one with a one-line placeholder. With the default polling of 1, every request after the
(N+1)-th masks one more observation than the request before it. That edit lands in the middle of
the prompt, so the previous request's KV stops being a prefix of the new prompt: a cache miss.

A request here is the prompt the model would see at step t. Its *invalidated state* is the previous
request's prompt followed by the reply that request produced, which is exactly the KV a decode
instance holds when the new request arrives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def placeholder(content: str) -> str:
    """The text SWE-agent's LastNObservations puts in place of an old tool output."""
    return f"Old environment output: ({len(content.splitlines())} lines omitted)"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


@dataclass
class Session:
    """A trajectory split into its fixed head, the assistant replies and the tool observations."""

    traj_id: str
    head: list[dict]  # system prompt + task statement (never masked)
    replies: list[str]  # replies[i] is the model's (i+1)-th reply
    observations: list[str]  # observations[i] answers replies[i]
    meta: dict = field(default_factory=dict)

    @property
    def n_requests(self) -> int:
        return len(self.replies)

    def messages(self, t: int, keep_last: int | None) -> list[dict]:
        """Chat messages of request t (1-based): the prompt that produced reply t.

        Request t has seen replies 1..t-1 and observations 1..t-1. Observation i is masked when
        i <= (t - 1) - keep_last, so request t masks one more observation than request t-1 once
        t >= keep_last + 2. ``keep_last=None`` disables masking (append-only, every request a hit).
        """
        msgs = [dict(m) for m in self.head]
        n_seen = t - 1
        for i in range(n_seen):
            msgs.append({"role": "assistant", "content": self.replies[i]})
            obs = self.observations[i]
            if keep_last is not None and (i + 1) <= n_seen - keep_last:
                obs = placeholder(obs)
            msgs.append({"role": "user", "content": obs})
        return msgs

    def is_miss(self, t: int, keep_last: int | None) -> bool:
        """True when request t edits the previous request's context (it masks a new observation)."""
        return keep_last is not None and t >= keep_last + 2


def parse_messages(traj_id: str, messages: list[dict], meta: dict | None = None) -> Session | None:
    """Split a SWE-agent message list into head / replies / observations.

    Expects system, user (task), then alternating assistant/user. Returns None for anything else,
    so malformed trajectories are skipped rather than silently mis-replayed.
    """
    msgs = [{"role": m["role"], "content": _text(m["content"])} for m in messages]
    if len(msgs) < 4 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        return None
    head, rest = msgs[:2], msgs[2:]
    replies, observations = [], []
    for i in range(0, len(rest) - 1, 2):
        a, o = rest[i], rest[i + 1]
        if a["role"] != "assistant" or o["role"] != "user":
            return None
        replies.append(a["content"])
        observations.append(o["content"])
    if not replies:
        return None
    return Session(traj_id=traj_id, head=head, replies=replies, observations=observations, meta=meta or {})


def load_parquet(path: str, limit: int | None = None) -> list[Session]:
    """Load SWE-smith ``ticks`` trajectories (SWE-agent 1.x, backtick actions) from a parquet shard."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["messages", "traj_id", "instance_id", "model", "resolved"])
    sessions = []
    for row in table.to_pylist():
        s = parse_messages(
            row["traj_id"],
            json.loads(row["messages"]),
            {"instance_id": row["instance_id"], "model": row["model"], "resolved": row["resolved"]},
        )
        if s is not None:
            sessions.append(s)
        if limit is not None and len(sessions) >= limit:
            break
    return sessions


class Renderer:
    """Turns messages into token ids with the model's own chat template (non-thinking mode)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prompt(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tok.encode(text, add_special_tokens=False)

    def reply(self, content: str) -> list[int]:
        """Tokens a decode instance holds after generating ``content``: the text then <|im_end|>."""
        return self.tok.encode(content, add_special_tokens=False) + [self.im_end]


@dataclass
class MissRequest:
    """One cache-miss request: the invalidated state and the new prompt, as token ids."""

    traj_id: str
    t: int
    keep_last: int
    old: list[int]  # previous prompt + its reply, as the decode instance generated it
    new: list[int]  # the new prompt, ending in the generation prompt

    @property
    def key(self) -> str:
        return f"{self.traj_id}#{self.t}"


def miss_request(session: Session, t: int, keep_last: int, renderer: Renderer,
                 reply_as_rendered: bool = False) -> MissRequest:
    """``reply_as_rendered`` is a counterfactual (the paper's App. B.3 check): the invalidated state
    holds the previous reply as later prompts render it, without the empty think block that Qwen3's
    generation prompt put in front of it, so only the masking edit makes the reused KV stale."""
    assert session.is_miss(t, keep_last), "request t does not edit its predecessor's context"
    prev = renderer.prompt(session.messages(t - 1, keep_last))
    if reply_as_rendered:
        think = renderer.tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        if prev[-len(think):] == think:
            prev = prev[: -len(think)]
    old = prev + renderer.reply(session.replies[t - 2])
    new = renderer.prompt(session.messages(t, keep_last))
    return MissRequest(session.traj_id, t, keep_last, old, new)

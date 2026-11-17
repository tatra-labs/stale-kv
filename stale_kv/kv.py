"""A small KV engine over a Hugging Face Qwen3: prefill, provisional KV, greedy drafting, verification.

The model's own modules (embeddings, norms, projections, rotary embedding, MLP) do the math; this
module only owns the KV layout. Each cache is a preallocated ``KVBuffer`` of shape
``[layers, 1, kv_heads, max_len, head_dim]`` holding post-RoPE keys and values, plus a length.
Preallocation lets greedy drafting run as one CUDA graph per step with no host round trips, which
matters on a laptop where launching ~500 small kernels per token through Python costs ~150 ms.

Chunks (prefill, staging, verification) run eagerly and attend over ``[0, len + T)``. Single-token
drafting attends over the whole buffer with a mask, so its numerics differ slightly from a chunk's:
the same situation as a serving engine whose decode and prefill kernels differ, and the reason
drafting on exact KV is not accepted 100% of the time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .align import Segment


def rerotate(keys: torch.Tensor, delta: int, freqs: torch.Tensor) -> torch.Tensor:
    """Move post-RoPE keys by ``delta`` positions: R(p + delta) k = R(delta) R(p) k.

    Rotate-half layout, as in the model. Computed in float32 and cast back, so a bf16 cache picks up
    one extra rounding.
    """
    if delta == 0:
        return keys
    ang = freqs.to(keys.device, torch.float32) * float(delta)
    cos = torch.cat([ang.cos(), ang.cos()])
    sin = torch.cat([ang.sin(), ang.sin()])
    k = keys.float()
    half = k.shape[-1] // 2
    rot = torch.cat([-k[..., half:], k[..., :half]], dim=-1)
    return (k * cos + rot * sin).to(keys.dtype)


def accepted(drafts: list[int], rows: torch.Tensor, rule: str = "fixed") -> int:
    """Number of drafts accepted in order. ``fixed``: draft == row argmax (lowest id wins ties).
    ``anymax``: the draft's logit equals the row maximum (accepts any exact tie)."""
    if not drafts:
        return 0
    d = torch.tensor(drafts, device=rows.device)
    r = rows[: len(drafts)]
    if rule == "fixed":
        ok = r.argmax(dim=-1) == d
    elif rule == "anymax":
        ok = r.gather(1, d[:, None])[:, 0] == r.max(dim=-1).values
    else:
        raise ValueError(rule)
    bad = (~ok).nonzero()
    return int(bad[0, 0]) if len(bad) else len(drafts)


def _gqa_attention(q, k, v, mask, scale):
    """Grouped-query attention without copying K/V: the query heads that share a KV head are folded
    into the query-length axis (HF's ``repeat_kv`` pairs query head h with KV head h // n_rep).
    This keeps SDPA on its memory-efficient masked kernel; ``enable_gqa=True`` with a mask falls back
    to the math kernel, which materialises a heads x T x len score matrix."""
    _, hq, t, d = q.shape
    hk = k.shape[1]
    rep = hq // hk
    qg = q.reshape(1, hk, rep * t, d)
    mg = mask.repeat(1, 1, rep, 1) if rep > 1 else mask
    o = F.scaled_dot_product_attention(qg, k, v, attn_mask=mg, scale=scale)
    return o.reshape(1, hq, t, d)


class KVBuffer:
    def __init__(self, n_layers, kv_heads, head_dim, max_len, dtype, device):
        shape = (n_layers, 1, kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.len = 0
        self.max_len = max_len

    def reset(self) -> "KVBuffer":
        self.len = 0
        return self


class Engine:
    def __init__(self, hf_model, max_len: int, use_graphs: bool | None = None, fp32_logits: bool = True):
        self.hf = hf_model
        self.fp32_logits = fp32_logits
        self.m = hf_model.model
        self.cfg = hf_model.config
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype
        rotary = self.m.rotary_emb
        if getattr(rotary, "rope_type", "default") != "default":
            raise NotImplementedError("re-rotation by a constant delta assumes unscaled RoPE")
        self.freqs = rotary.inv_freq.float()
        self.n_layers = self.cfg.num_hidden_layers
        self.kv_heads = self.cfg.num_key_value_heads
        self.head_dim = getattr(self.cfg, "head_dim", None) or self.cfg.hidden_size // self.cfg.num_attention_heads
        self.max_len = max_len
        self.use_graphs = (self.device.type == "cuda") if use_graphs is None else use_graphs
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._pool = None
        # Static decode state (one set, shared by every buffer's graph).
        self._ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._step = torch.zeros(1, dtype=torch.long, device=self.device)
        self._out_tok = torch.zeros(max_len, dtype=torch.long, device=self.device)
        self._out_gap = torch.zeros(max_len, dtype=torch.float32, device=self.device)
        self._arange = torch.arange(max_len, device=self.device)

    def buffer(self) -> KVBuffer:
        return KVBuffer(self.n_layers, self.kv_heads, self.head_dim, self.max_len, self.dtype, self.device)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """(T, D) final hidden states -> (T, V) float32 logits.

        With ``fp32_logits`` (default) the bf16 LM head writes float32 output instead of rounding the
        logits to bf16. In bf16, logits near 20 are spaced 0.125 apart, so exact ties and one-step
        flips between two numerically different forward passes are common, and they would swamp
        the effect being measured. ``fp32_logits=False`` reproduces a bf16-logit engine.
        """
        w = self.hf.lm_head.weight
        if self.fp32_logits and h.is_cuda and h.dtype in (torch.bfloat16, torch.float16):
            return torch.mm(h, w.t(), out_dtype=torch.float32)
        return F.linear(h, w).float()

    # ---- eager chunk forward -------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, buf: KVBuffer, tokens, all_logits: bool = False) -> torch.Tensor:
        """Run ``tokens`` at positions ``buf.len ...``, append their KV, return float32 logits."""
        T = len(tokens)
        n = buf.len
        if n + T > buf.max_len:
            raise ValueError(f"buffer overflow: {n} + {T} > {buf.max_len}")
        ids = torch.as_tensor(list(tokens), dtype=torch.long, device=self.device)[None]
        pos = torch.arange(n, n + T, device=self.device)[None]
        h = self.m.embed_tokens(ids)
        cos, sin = self.m.rotary_emb(h, pos)
        kpos = torch.arange(n + T, device=self.device)
        mask = (kpos[None, :] <= pos[0][:, None])[None, None]  # (1, 1, T, n+T)
        for li, layer in enumerate(self.m.layers):
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).view(1, T, -1, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(1, T, -1, self.head_dim)).transpose(1, 2)
            v = attn.v_proj(x).view(1, T, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            buf.k[li, :, :, n : n + T] = k
            buf.v[li, :, :, n : n + T] = v
            o = _gqa_attention(q, buf.k[li, :, :, : n + T], buf.v[li, :, :, : n + T], mask, attn.scaling)
            h = res + attn.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        buf.len = n + T
        return self._logits(self.m.norm(h if all_logits else h[:, -1:])[0])

    @torch.no_grad()
    def prefill(self, buf: KVBuffer, tokens, chunk: int | None = 2048) -> KVBuffer:
        step = chunk or max(len(tokens), 1)
        for s in range(0, len(tokens), step):
            self.forward(buf, tokens[s : s + step])
        return buf

    # ---- provisional KV --------------------------------------------------------------------------
    @torch.no_grad()
    def make_provisional(self, buf: KVBuffer, segs: list[Segment], new_tokens: list[int],
                         rotate: bool = True, chunk: int = 112) -> KVBuffer:
        """Turn ``buf`` (holding the invalidated sequence's KV) into the provisional KV of
        ``new_tokens``, in place.

        First every ``keep``/``map`` run is moved to its new position (keys re-rotated by the run's
        delta unless ``rotate=False``; values copied unchanged). Then the ``new`` spans are computed
        left to right as chunked prefill over what has been assembled - Callo's staging, with the
        paper's C_s = 112 tokens per chunk. Causal attention means a new span only reads positions
        before it, all of which are final by then.
        """
        runs = [s for s in segs if s.kind == "map"]  # "keep" runs are already in place
        for li in range(self.n_layers):  # one layer at a time keeps the copies small
            moved = []
            for s in runs:
                a, b = s.old_start, s.old_start + s.length
                k = buf.k[li, :, :, a:b].clone()
                moved.append((s, rerotate(k, s.delta, self.freqs) if rotate else k, buf.v[li, :, :, a:b].clone()))
            for s, k, v in moved:
                buf.k[li, :, :, s.new_start : s.new_start + s.length] = k
                buf.v[li, :, :, s.new_start : s.new_start + s.length] = v
        for s in segs:
            if s.kind != "new":
                continue
            buf.len = s.new_start
            self.prefill(buf, new_tokens[s.new_start : s.new_start + s.length], chunk=chunk)
        buf.len = len(new_tokens)
        return buf

    # ---- greedy drafting -------------------------------------------------------------------------
    def _decode_step(self, buf: KVBuffer) -> None:
        """One greedy step on static tensors: reads _ids/_pos/_step, writes the token and gap."""
        h = self.m.embed_tokens(self._ids[None])  # (1, 1, D)
        cos, sin = self.m.rotary_emb(h, self._pos[None])
        mask = (self._arange <= self._pos)[None, None, None]  # (1, 1, 1, max_len)
        for li, layer in enumerate(self.m.layers):
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).view(1, 1, -1, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(1, 1, -1, self.head_dim)).transpose(1, 2)
            v = attn.v_proj(x).view(1, 1, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            buf.k[li].index_copy_(2, self._pos, k)
            buf.v[li].index_copy_(2, self._pos, v)
            o = _gqa_attention(q, buf.k[li], buf.v[li], mask, attn.scaling)
            h = res + attn.o_proj(o.transpose(1, 2).reshape(1, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self._logits(self.m.norm(h)[0])[0]
        top = torch.topk(logits, 2).values
        nxt = torch.argmax(logits).view(1)  # first maximum: ties go to the lowest token id
        self._out_tok.index_copy_(0, self._step, nxt)
        self._out_gap.index_copy_(0, self._step, (top[0] - top[1]).view(1))
        self._ids.copy_(nxt)
        self._pos.add_(1)
        self._step.add_(1)

    def _graph(self, buf: KVBuffer):
        key = id(buf)
        if key not in self._graphs:
            saved = (self._ids.clone(), self._pos.clone(), self._step.clone())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):  # warm up kernels before capture, writing only the scratch end
                    self._pos.fill_(self.max_len - 1); self._step.fill_(0)
                    self._decode_step(buf)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
                self._decode_step(buf)
            self._pool = g.pool()
            self._graphs[key] = g
            self._ids.copy_(saved[0]); self._pos.copy_(saved[1]); self._step.copy_(saved[2])
        return self._graphs[key]

    @torch.no_grad()
    def decode_greedy(self, buf: KVBuffer, first: int, k: int, stop: set[int]) -> tuple[list[int], list[float]]:
        """Greedily draft up to ``k`` tokens after feeding ``first`` at position ``buf.len``.

        Returns the tokens (cut after the first stop token) and each step's top-1 minus top-2 logit
        gap. Draft KV goes into the buffer's scratch tail; ``buf.len`` is unchanged afterwards.
        """
        n = buf.len
        if n + k > buf.max_len:
            raise ValueError("no room to draft")
        self._ids.fill_(first)
        self._pos.fill_(n)
        self._step.fill_(0)
        if self.use_graphs:
            g = self._graph(buf)
            self._ids.fill_(first); self._pos.fill_(n); self._step.fill_(0)
            for _ in range(k):
                g.replay()
        else:
            for _ in range(k):
                self._decode_step(buf)
        toks = self._out_tok[:k].tolist()
        gaps = self._out_gap[:k].tolist()
        for i, t in enumerate(toks):
            if t in stop:
                toks, gaps = toks[: i + 1], gaps[: i + 1]
                break
        return toks, gaps

    @torch.no_grad()
    def score_chunk(self, buf: KVBuffer, last: int, drafts: list[int]) -> torch.Tensor:
        """One forward pass over ``[last] + drafts`` on ``buf``; row i scores draft i (row
        ``len(drafts)`` is the bonus row). This is Callo's verification step. ``buf.len`` is restored."""
        n = buf.len
        rows = self.forward(buf, [last] + list(drafts), all_logits=True)
        buf.len = n
        return rows

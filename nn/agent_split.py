"""Two agents from one trained model, with no retraining (2026-11-13).

Agent A reads a document once and keeps the reading layer's memory of it. Agent B never sees the document: it runs its
own text (a question) and its answer through the layers, and its only view of the document is the reading layer's read
of A's memory. Built on the cached decoding (`nn/incremental.py`): A's part is the prefill, B's part is the steps.

What separates this from the cached decoding of one row:
  * B's local (sliding-window) layers read B's own tokens only. In one row they would also read A's last tokens
    (the window is 128 at the 30M-300M sizes), which is raw text crossing between the agents.
  * the reading layer is unchanged: for B's query it is one softmax over A's memory and B's own keys.

So the only thing that crosses is the reading layer's memory: per document token and loop pass one key and one value
(`memory_numbers`), from which A also builds what its read needs (LSA's buckets, E37's concepts). Two ways to move it:
  * hand over the memory (B holds a copy and reads it);
  * remote read: B sends each query, A runs the read over its memory and returns the read vector and the log of its
    softmax mass, B merges that with the read of its own keys (two softmaxes join exactly by log-sum-exp). The traffic
    is then independent of the document's length (`remote_read_numbers`).
Both give the same outputs; this module computes them in one process and counts the traffic.

`bits` re-states A's memory with fewer bits per number before B reads it (a uniform grid per dimension over the
document, the index rebuilt from the coarse values). It is a measurement of how much of the 16 bits B's answers use,
not a proposed wire format.

Supported reads: full attention, LSA's read, E37's concept read (what `decode_kind` returns as dense / lsa / concept).
The note reads keep pages that straddle the two agents and are refused here.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F

from nn.incremental import IncrementalDecoder
from nn.perceiver_ar_lm import PerceiverARLM


def coarse(x: torch.Tensor, bits: int) -> torch.Tensor:
    """x [B, n, g, d] on a uniform grid of 2^bits levels per (row, head, dimension), spanning ± the largest magnitude
    of that dimension over the n entries; each value moves to the middle of its cell (1 bit = sign × half the range)."""
    if bits <= 0 or bits >= 16 or x.shape[1] == 0:
        return x
    xf = x.float()
    scale = xf.abs().amax(1, keepdim=True).clamp_min(1e-8)
    half = 2 ** (bits - 1)
    cell = torch.floor(xf / scale * half).clamp(-half, half - 1)
    return ((cell + 0.5) / half * scale).to(x.dtype)


class SplitDecoder(IncrementalDecoder):
    """`read` = agent A (the document's memory); `listen` + `answer` = agent B (question in, answer out)."""

    def __init__(self, model: PerceiverARLM):
        super().__init__(model)
        if self.kind not in ("dense", "lsa", "concept"):
            raise NotImplementedError(f"two-agent split: the {self.kind} read keeps pages that straddle the agents")
        if tuple(self.cfg.ngram_orders or ()):
            raise NotImplementedError("two-agent split: n-gram input tables read A's last tokens as text")
        self.cut = 0            # B's local layers read keys from this position on (0 = one agent, no split)
        self._local = False
        self._snap: dict = {}
        self._built: Optional[tuple] = None  # (memory state, the read's index built from it)

    # ------------------------------------------------------------------ agent A
    @torch.no_grad()
    def read(self, buf: torch.Tensor, doc: torch.Tensor, L0: int) -> None:
        """A reads its document: buf / doc [1, Lb] with the document in [0, L0) and room after it for B's tokens
        (eos, document id −1). Keeps the reading layer's memory of the document as read."""
        self.prefill(buf, doc, L0)
        g = self.gi
        self._snap = {"K": {k: v.clone() for k, v in self.K.items() if k[0] == g},
                      "V": {k: v.clone() for k, v in self.V.items() if k[0] == g},
                      "X": {k: v.clone() for k, v in self.X.items()},
                      "buf": self.buf.clone(), "doc": self.doc.clone()}
        self._built = None

    def memory_numbers(self) -> int:
        """Numbers in A's memory of its document: one key and one value per token and loop pass."""
        k, v = self.K[(self.gi, 0)], self.V[(self.gi, 0)]
        return int(self.L0 * self.passes * (k.shape[2] * k.shape[3] + v.shape[2] * v.shape[3]))

    def remote_read_numbers(self) -> tuple[int, int]:
        """Per token B processes, the numbers B sends (its queries, one per head and pass) and A returns (the read
        vector and the log of the softmax mass, per head and pass)."""
        att = self.m.layers[self.gi].attn
        dv = int(self.V[(self.gi, 0)].shape[-1])
        return self.passes * att.h * att.dh, self.passes * att.h * (dv + 1)

    # ------------------------------------------------------------------ agent B
    @torch.no_grad()
    def listen(self, *, split: bool = True, bits: int = 0, blank: bool = False, tail: int = 0) -> None:
        """Back to the moment A finished reading. `split`: B's local layers do not read A's tokens. `bits`: A's memory
        on a coarse grid (see `coarse`). `blank`: A's memory is hidden from the reading layer too (B on its own text,
        at the same positions) — the floor. `tail`: with the split, B's local layers still read A's last `tail`
        tokens (how much of A's raw ending B's first tokens lean on; the local window = no split)."""
        s, L0 = self._snap, self.L0
        for key in s["K"]:
            k, v = s["K"][key].clone(), s["V"][key].clone()
            if bits:
                k[:, :L0], v[:, :L0] = coarse(k[:, :L0], bits), coarse(v[:, :L0], bits)
            self.K[key], self.V[key] = k, v
        for key in s["X"]:
            self.X[key] = s["X"][key].clone()
        self.buf, self.doc = s["buf"].clone(), s["doc"].clone()
        if blank:
            self.doc[:, :L0] = -2  # another document: no read of B's reaches it, and positions restart at B's text
        self.t, self._ds_cols = L0, {}
        self.cut = max(1, L0 - int(tail)) if split else 0
        if blank:
            from nn.exact_read import doc_starts

            self._ds0 = doc_starts(self.doc)
        self._index((int(bits), bool(blank)))

    def _index(self, state: tuple) -> None:
        """What the read derives from A's keys (LSA's buckets, E37's concepts), built once per state of the memory
        and handed out as a fresh copy: B's steps write into it (document starts of its positions; a rebuild when its
        own tokens close a block)."""
        name = {"lsa": "lsa_groups", "concept": "cc"}.get(self.kind)
        if name is None:
            return
        if self._built is None or self._built[0] != state:
            self._build_lsa() if self.kind == "lsa" else self._build_concepts()
            self._built = (state, copy.deepcopy(getattr(self, name)))
        else:
            setattr(self, name, copy.deepcopy(self._built[1]))

    @torch.no_grad()
    def answer(self, question: torch.Tensor, new: int, *, eos: int) -> list[int]:
        """B reads its own text token by token (`question` [n]) and then writes up to `new` tokens, greedy."""
        cfg, W = self.cfg, self.m.lm_head.weight
        h = None
        for tok in question:
            h = self.step(tok.view(1))
        out: list[int] = []
        for i in range(int(new)):
            logits = F.linear(h, W).float()
            if cfg.logit_softcap:
                logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
            nxt = logits.argmax(-1)
            out.append(int(nxt))
            if int(nxt) == eos or i + 1 == int(new):
                break
            h = self.step(nxt)
        return out

    # ------------------------------------------------------------------ the split itself
    def _attn(self, i, r, att, h, ids, cos, sin, pos, t):
        self._local = att.pattern == "swa"
        return super()._attn(i, r, att, h, ids, cos, sin, pos, t)

    def _window_read(self, q, K, V, t, w):
        if self._local and self.cut and t >= self.cut:
            w = min(int(w), t - self.cut + 1)  # a local layer of B's: its own tokens only
        return super()._window_read(q, K, V, t, w)


def row_for(ids: torch.Tensor, room: int, *, eos: int, pad: int = 256) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One fixed-length row as the eval harness lays it out: `ids` [n] first, then eos with document id −1, with at
    least `room` free slots, padded to a multiple of `pad`. Returns buf [1, Lb], doc [1, Lb], n."""
    n = int(ids.shape[0])
    Lb = -(-(n + int(room)) // pad) * pad
    buf = torch.full((1, Lb), eos, dtype=ids.dtype, device=ids.device)
    buf[0, :n] = ids
    doc = torch.full((1, Lb), -1, dtype=torch.long, device=ids.device)
    doc[0, :n] = 0
    return buf, doc, n


__all__ = ["SplitDecoder", "coarse", "row_for"]

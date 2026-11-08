"""E33p instrumentation, part A (plan §8): training dynamics measured inside one forward, identical in every arm.

`FlowRecorder` is armed for ONE forward (the first micro-batch of a log step, `training/flow_callback.py`). While
armed, the blocks, the reading layer and the model hand it tensors; it keeps on-GPU scalars only and turns them into
floats once, in `finish` (one host sync). Nothing here changes the forward's numerics: every statistic is computed
from detached tensors under `no_grad`.

Keys (W&B, plan §8 A):
  flow/L{i}[/r{r}]/x_rms, attn_rel, ffn_rel     residual RMS into each block, ‖Δh‖/‖h‖ of its attention and FFN
  read/r{r}/{bucket}/mass_{source}, entropy     reading-layer attention on sampled queries, by position bucket;
                                                sources: window, opened, notes, sinks, beyond (dense raw keys
                                                outside the window)
  select/r{r}/opened, cap_full, pages, skipped, overlap      the per-query choice (opening arms)
  writer/assign_entropy, empty_notes, members_max_mean, kappa_ratio/r{r}
  lens/L{i}[/r{r}]                              next-token loss read from each block's output (logit lens)
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F


def _first_row(cidx) -> SimpleNamespace:
    """The batch row 0 of an E37 ConceptIndex, enough for `summary_logit` on sampled rows."""
    return SimpleNamespace(mu_k=cidx.mu_k[:1], n=cidx.n[:1], var=cidx.var[:1])

BUCKETS = ((0, 512), (512, 2048), (2048, 8192), (8192, 16384))


class FlowRecorder:
    def __init__(self, n_sample: int = 64, lens_tokens: int = 1024, buckets=BUCKETS):
        self.n_sample, self.lens_tokens, self.buckets = int(n_sample), int(lens_tokens), tuple(buckets)
        self.armed = False
        self.round: Optional[int] = None
        self.stats: dict[str, torch.Tensor] = {}
        self.resid: list[tuple[str, torch.Tensor]] = []
        self.ids: Optional[torch.Tensor] = None
        self._prev_chosen: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ lifecycle
    def begin(self, input_ids: torch.Tensor) -> None:
        self.stats, self.resid, self.round, self._prev_chosen = {}, [], None, None
        self.ids = input_ids.detach()

    def _tag(self, name: str) -> str:
        return f"{name}/r{self.round}" if self.round is not None else name

    def _put(self, key: str, value) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(float(value))
        self.stats[key] = value.detach().float().reshape(())

    def _sample_rows(self, S: int, device) -> tuple[torch.Tensor, list[tuple[str, slice]]]:
        """Up to n_sample query positions per position bucket (evenly spaced), and each bucket's slice of them."""
        rows, parts, n = [], [], 0
        for lo, hi in self.buckets:
            hi = min(hi, S)
            if hi - lo < 1:
                continue
            k = min(self.n_sample, hi - lo)
            r = torch.linspace(lo, hi - 1, k, device=device).round().long().unique()
            rows.append(r)
            parts.append((f"{lo}-{hi}", slice(n, n + r.numel())))
            n += r.numel()
        return torch.cat(rows), parts

    # ------------------------------------------------------------------ blocks and residuals
    @torch.no_grad()
    def sublayer(self, layer_idx: int, kind: str, x: torch.Tensor, delta: torch.Tensor) -> None:
        base = self._tag(f"flow/L{layer_idx}")
        xr = x.detach().float().pow(2).mean().sqrt()
        if kind == "attn":
            self._put(f"{base}/x_rms", xr)
        self._put(f"{base}/{kind}_rel", delta.detach().float().pow(2).mean().sqrt() / xr.clamp_min(1e-12))

    @torch.no_grad()
    def residual(self, layer_idx: int, x: torch.Tensor) -> None:
        T = min(self.lens_tokens, x.shape[1] - 1)
        if T > 0:
            self.resid.append((self._tag(f"lens/L{layer_idx}"), x[0, -T - 1:-1].detach().clone()))

    # ------------------------------------------------------------------ the reading layer
    def _capture(self, **kw) -> None:
        """Hook for evaluation/flow_probe.py (ProbeRecorder): the reading layer's probabilities and sets, per call."""

    def _mass(self, logits: torch.Tensor, allowed: torch.Tensor, sources: dict[str, torch.Tensor],
              parts) -> torch.Tensor:
        """logits [h, Q, KV] fp32, allowed [Q, KV] bool, sources name → [Q, KV] bool; per bucket mass and entropy.
        Returns the probabilities [h, Q, KV]."""
        p = torch.softmax(logits.masked_fill(~allowed[None], float("-inf")), dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        ent = -(p * (p + 1e-12).log()).sum(-1).mean(0)                                   # [Q]
        mass = {k: (p * m[None].to(p.dtype)).sum(-1).mean(0) for k, m in sources.items()}  # [Q]
        for name, sl in parts:
            base = self._tag("read") + f"/{name}"
            self._put(f"{base}/entropy", ent[sl].mean())
            for k, v in mass.items():
                self._put(f"{base}/mass_{k}", v[sl].mean())
        return p

    @torch.no_grad()
    def reading_open(self, attn, *, q, k, v, k_bar, v_bar, ctx, key_valid, opened, gist, index) -> None:
        """Notebook arms (window, e33, notes_open, pages_open): mass on window / opened / notes, and the choice."""
        from nn.exact_read import counts, dense_open

        B, S, h, dh = q.shape
        rows, parts = self._sample_rows(S, q.device)
        doc = ctx.doc[0]
        rw = int(ctx.raw_window or 0)
        j = torch.arange(S, device=q.device)
        causal = (j[None] <= rows[:, None]) & (doc[None] == doc[rows][:, None]) & (doc[rows] >= 0)[:, None]
        window = causal & ((rows[:, None] - j[None]) < rw) if rw > 0 else causal
        op = (dense_open(opened.T[:1], opened.index.codes[:, :1], cap=opened.cap, raw_window=rw, q_idx=rows,
                         doc=ctx.doc[:1])[0] & causal) if opened is not None else torch.zeros_like(causal)
        notes_ok = ctx.override not in ("none", "raw") and gist
        nv = ((ctx.slot_doc[0][None] == doc[rows][:, None]) & (ctx.slot_close[0][None] >= 0)
              & (ctx.slot_close[0][None] <= rows[:, None]))
        if not notes_ok:
            nv = torch.zeros_like(nv)
        K = torch.cat([k[0, :, 0], k_bar[0, :, 0].to(k.dtype)], 0).float()                 # [S+C, dh]
        lg = torch.einsum("qhd,kd->hqk", q[0, rows].float(), K) * dh ** -0.5
        allowed = torch.cat([window | op, nv], -1)
        z = torch.zeros_like(nv)
        pr = self._mass(lg, allowed, {"window": torch.cat([window, z], -1), "opened": torch.cat([op & ~window, z], -1),
                                      "notes": torch.cat([torch.zeros_like(window), nv], -1)}, parts)
        self._capture(kind="notebook", rows=rows, p=pr, S=S, opened=op & ~window, ctx=ctx, index=index)
        if index is not None:
            self._writer(index, k, k_bar)
        if opened is not None:
            self._selection(opened, rows, doc_start=opened.index.doc_start[0, rows], rw=rw, index=index, q=q,
                            k_bar=k_bar, nv=nv)

    @torch.no_grad()
    def reading_dense(self, attn, *, q, k, v, doc_ids, key_valid) -> None:
        """Dense ceiling: mass inside the 512-token window vs beyond it."""
        B, S, h, dh = q.shape
        rows, parts = self._sample_rows(S, q.device)
        doc = doc_ids[0] if doc_ids is not None else torch.zeros(S, dtype=torch.long, device=q.device)
        j = torch.arange(S, device=q.device)
        causal = (j[None] <= rows[:, None]) & (doc[None] == doc[rows][:, None])
        window = causal & ((rows[:, None] - j[None]) < 512)
        lg = torch.einsum("qhd,kd->hqk", q[0, rows].float(), k[0, :, 0].float()) * dh ** -0.5
        pr = self._mass(lg, causal, {"window": window, "beyond": causal & ~window}, parts)
        self._capture(kind="dense", rows=rows, p=pr, S=S, opened=causal & ~window, ctx=None, index=None)

    @torch.no_grad()
    def reading_lsa(self, attn, *, q, k, v, doc, key_valid, opened, sink, window) -> None:
        from nn.exact_read import dense_open

        B, S, h, dh = q.shape
        rows, parts = self._sample_rows(S, q.device)
        d = doc[0]
        j = torch.arange(S, device=q.device)
        causal = (j[None] <= rows[:, None]) & (d[None] == d[rows][:, None]) & (d[rows] >= 0)[:, None]
        win = causal & ((rows[:, None] - j[None]) < window)
        sk = causal & sink[0][None] & ~win
        op = (dense_open(opened.T[:1], opened.index.codes[:, :1], cap=opened.cap, raw_window=window, q_idx=rows,
                         doc=doc[:1])[0] & causal & ~win & ~sk) if opened is not None else torch.zeros_like(win)
        lg = torch.einsum("qhd,kd->hqk", q[0, rows].float(), k[0, :, 0].float()) * dh ** -0.5
        pr = self._mass(lg, win | sk | op, {"window": win, "sinks": sk, "opened": op}, parts)
        self._capture(kind="lsa", rows=rows, p=pr, S=S, opened=op, ctx=None, index=None)
        if opened is not None:
            self._selection(opened, rows, doc_start=opened.index.doc_start[0, rows], rw=window, index=None, q=None,
                            k_bar=None, nv=None)

    @torch.no_grad()
    def reading_concept(self, attn, *, q, k, v, doc, cidx, opened, sink) -> None:
        """E37 concept read: mass on window / sinks / opened members / concept summaries, the choice, and the
        concepts' sizes (nn/concept_memory.py)."""
        from nn.concept_memory import summary_logit
        from nn.exact_read import dense_open

        B, S, h, dh = q.shape
        rows, parts = self._sample_rows(S, q.device)
        d = doc[0]
        blk = cidx.block if cidx is not None else 512
        j = torch.arange(S, device=q.device)
        causal = (j[None] <= rows[:, None]) & (d[None] == d[rows][:, None]) & (d[rows] >= 0)[:, None]
        win = causal & (j[None] >= ((rows // blk - 1) * blk)[:, None])
        sk = causal & sink[0][None] & ~win
        op = (dense_open(opened.T[:1], opened.index.codes[:, :1], cap=opened.cap, raw_window=blk, q_idx=rows,
                         doc=doc[:1])[0] & causal & ~win & ~sk) if opened is not None else torch.zeros_like(win)
        lg = torch.einsum("qhd,kd->hqk", q[0, rows].float(), k[0, :, 0].float()) * dh ** -0.5
        allowed, sources = win | sk | op, {"window": win, "sinks": sk, "opened": op}
        if cidx is not None:
            c_from = (cidx.block_of + 2) * blk
            vis = (rows[:, None] >= c_from[None]) & (cidx.c_doc[0][None] == d[rows][:, None]) & (d[rows] >= 0)[:, None]
            if opened is not None:
                vis = vis & ~(opened.T[0, rows].to(torch.int32) < int(opened.cap))
            sl = summary_logit(q[:1, rows], _first_row(cidx))[0].permute(1, 0, 2)            # [h, Q, C]
            lg = torch.cat([lg, sl], dim=-1)
            pad = torch.zeros(rows.numel(), cidx.n_concepts, dtype=torch.bool, device=q.device)
            allowed = torch.cat([allowed, vis], dim=-1)
            sources = {k_: torch.cat([m, pad], dim=-1) for k_, m in sources.items()}
            sources["summaries"] = torch.cat([torch.zeros_like(win), vis], dim=-1)
            n = cidx.n[0]
            self._put("concept/empty", (n == 0).float().mean())
            self._put("concept/max_members", n.view(-1, cidx.per_block).amax(-1).mean())
        self._mass(lg, allowed, sources, parts)
        if opened is not None:
            self._selection(opened, rows, doc_start=opened.index.doc_start[0, rows], rw=blk, index=None, q=None,
                            k_bar=None, nv=None)

    # ------------------------------------------------------------------ choice and writer
    def _selection(self, opened, rows, *, doc_start, rw, index, q, k_bar, nv) -> None:
        from nn.exact_read import counts

        gi = opened.index
        G = gi.n_groups
        Tq = opened.T[0, rows].to(torch.int32)                                             # [Q, G]
        chosen = Tq < opened.cap
        gid = torch.arange(G, device=Tq.device)[None].expand_as(Tq)
        cnt = counts(gi, gid[None], doc_start[None, :, None].expand(1, -1, G),
                     (rows - rw)[None, :, None].expand(1, -1, G))[0]                       # row 0 only
        used = (cnt * chosen).sum(-1)
        base = self._tag("select")
        self._put(f"{base}/cap_full", (used >= opened.cap).float().mean())
        self._put(f"{base}/groups", chosen.sum(-1).float().mean())
        if index is not None and index.mode == "notes":
            pages = chosen.view(Tq.shape[0], -1, index.per_page).any(-1)
            self._put(f"{base}/pages", pages.sum(-1).float().mean())
        if self._prev_chosen is not None and self._prev_chosen.shape == chosen.shape:
            inter = (chosen & self._prev_chosen).sum(-1).float()
            union = (chosen | self._prev_chosen).sum(-1).float().clamp_min(1)
            self._put(f"{base}/overlap", (inter / union).mean())
        self._prev_chosen = chosen
        if q is not None and nv is not None and index is not None and index.mode == "notes":
            dh = q.shape[-1]
            lg = torch.einsum("qhd,cd->qhc", q[0, rows].float(), k_bar[0, :, 0].float()) * dh ** -0.5
            s = torch.nan_to_num(torch.softmax(lg.masked_fill(~nv[:, None], float("-inf")), -1), nan=0.0).sum(1)
            elig = nv & (gi.g_minpos[0][None] <= (rows - rw)[:, None])
            floor = torch.where(chosen, s, torch.full_like(s, float("inf"))).amin(-1, keepdim=True)
            self._put(f"{base}/skipped", (nv & ~elig & (s > floor)).sum(-1).float().mean())

    @torch.no_grad()
    def writer_assign(self, A: torch.Tensor, pick: torch.Tensor) -> None:
        """Every notebook arm, once per forward: A [B, n_w, K, W] (heads averaged), pick [B, n_w, W] bool (row 0)."""
        A, on = A[0].float(), pick[0].bool()
        K = A.shape[1]
        p = A.permute(0, 2, 1)[on]                                                         # [T, K]
        if p.numel():
            ent = -(p * (p + 1e-12).log()).sum(-1) / math.log(max(K, 2))
            self._put("writer/assign_entropy", ent.mean())
        best = A.argmax(1)                                                                 # [n_w, W]
        members = (F.one_hot(best, K) * on[..., None]).sum(1).float()                      # [n_w, K]
        mem = members[on.any(-1)]
        if mem.numel():
            self._put("writer/empty_notes", (mem == 0).float().mean())
            self._put("writer/members_max_mean", mem.max() / mem.mean().clamp_min(1e-6))

    def _writer(self, index, k, k_bar) -> None:
        """Tied arms, every round: ‖κ_n‖ against the assignment-weighted mean ‖k_t‖ of its tokens."""
        A = index.assign[0]                                                                # [n_w, K, W]
        kn = k[0, index.page_tok.reshape(-1), 0].float().norm(dim=-1).view(A.shape[0], -1)   # [n_w, W]
        mean_norm = (A * kn[:, None]).sum(-1) / A.sum(-1).clamp_min(1e-6)                  # [n_w, K]
        kap = k_bar[0, :, 0].float().norm(dim=-1).view(A.shape[0], -1)
        has = A.sum(-1) > 1e-3
        ratio = (kap / mean_norm.clamp_min(1e-6)) * has
        self._put(self._tag("writer/kappa_ratio"), ratio.sum() / has.sum().clamp_min(1))

    # ------------------------------------------------------------------ finish (one host sync)
    @torch.no_grad()
    def finish(self, model) -> dict[str, float]:
        cfg = model.config
        if self.resid and self.ids is not None:
            T = self.resid[0][1].shape[0]
            tgt = self.ids[0, -T:]
            W = model.lm_head.weight
            sc = float(cfg.logit_softcap or 0.0)
            for name, x in self.resid:
                lg = F.linear(model.final_norm(x.to(W.dtype)), W).float()
                if sc > 0:
                    lg = sc * torch.tanh(lg / sc)
                self._put(name, F.cross_entropy(lg, tgt))
        keys = list(self.stats)
        out = {}
        if keys:
            dev = next(iter(self.stats.values())).device
            vals = torch.stack([self.stats[k_].to(dev) for k_ in keys]).tolist()
            out = {k_: float(v) for k_, v in zip(keys, vals)}
        self.stats, self.resid, self.ids, self._prev_chosen = {}, [], None, None
        return out

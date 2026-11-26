"""E39 — living concepts: dynamic spans and document-wide concepts in the reading layer.

Spec: docs/experiments_specs/ahead/E39_living_concepts.md · plan: E39_plan.md.

The reading layer's memory (`PerceiverARLM`, `par_global_read = "living"`), grouped the way text is written:

1. **Spans** (once per forward, from the first loop pass's keys, no gradient). A span is a run of 1 … `span_max`
   adjacent tokens. A new span starts where a token's unit key stops resembling the key before it
   (cos(k̂_t, k̂_{t−1}) < θ), where a `stride`-token stride starts, and after `span_max` tokens. A stride keeps at
   most span_cap − stride / span_max such cuts, its lowest-cosine ones, so it holds at most `span_cap` spans. Only
   tokens that can be read from memory form spans: not a sink, not padding, and of the document that still runs at
   the first token of stride s + 2 (a reader sees its own stride and the one before as raw tokens, the rest as
   memory, so a document that ends earlier has no such reader).
2. **Concepts** (once per forward, `levels` 2). Strides are taken in reading order. Each span is compared with its
   document's concepts as they stood after the stride before (cosine of the span's centre, the normalised sum of
   its unit keys, and the concept's, the normalised sum of its members' unit keys). It joins the best one when the
   cosine is at least λ, otherwise it founds a concept; founders of one stride that resemble each other are merged
   by three parallel rounds of the leader rule. A new document starts again at slot 0. With `c_max` slots in use a
   founder joins its nearest concept whatever the cosine.
3. **Statistics** (every pass, with gradient to the members' keys and values). Per span: n, Σk, Σ|k|², Σv. Per
   concept one **version per stride**: the sums over its spans of that stride and all earlier strides of the
   document. A reader in stride τ reads version τ − 2, so it never sees a statistic holding a token it must not see.
4. **The read** (every pass). The choice, without gradient: score every concept of the reader's version by the
   estimated attention weight of an average member, best query head, q·μ_k/√d + |q|² σ² / (2d) (the summary logit
   without log n, so size does not win); keep the best `keep`; rank their visible spans by the same score on the
   span's own statistics and open them best first until `top_k` tokens are held (`search = "flat"`: every visible
   span competes). Then one softmax over the document's sinks ‖ the reader's stride and the one before ‖ the opened
   tokens ‖ one summary per concept, with key μ_k, value μ_v and logit bias log(n − n_opened) + |q|² σ² / (2d) (+
   the length scale): a concept opened whole drops out, one opened in part keeps a summary for the rest. Every past
   token of the document is counted once: in the window, opened, or inside one summary.

θ and λ are held by slow controllers in training (`servo_step`: mean span `span_target` tokens, a share
`found_target` of the spans founding) and are frozen at test time.

Two implementations of the same read: `attend_living` (flex: the concept read's raw-token call and a summary call
over the versions, merged by their log-sum-exp; training and evaluation up to the lengths whose tables fit) and
`living_read_stream` (per chunk of queries: the choice, then one fp32 softmax over gathered keys; no [B, S, ·]
table; differentiable). The second is the reference path of the tests and the `sdpa` backend, the long-length
path, and the cached decoding's one-query read (nn/incremental.py).

Shapes at 150M, 16k row: S = 16,384; stride 512 → n_st = 30 memory strides; span slots G = 30 · 128 = 3,840;
concept slots C = 1,024; k, v [B, S, 1, 64]; versions μ_k [B, 30, 1,024, 64].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nn.exact_read import GroupIndex, OpenTable, choose, doc_starts, groups_from_spans

NEG = float("-inf")
BLOCK = 128   # flex attention's block size


# ------------------------------------------------------------------------------------------------ grouping
@torch.no_grad()
def cut_spans(khat: torch.Tensor, member: torch.Tensor, theta, *, stride: int, span_max: int,
              span_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Span slot of every token. khat [B, L, d] unit keys (L a multiple of `stride`), member [B, L] bool (within a
    stride the members are one document's tokens) → labels [B, L / stride, stride] long, the token's span within
    its stride in position order, 0 … span_cap − 1 (−1 = not a member), and the span starts [B, L] bool.

    One shifted dot product for the row; a stride with more natural cuts than span_cap − stride / span_max keeps
    its lowest-cosine ones; a run longer than span_max is split every span_max. So a span has 1 … span_max tokens,
    never crosses a stride, and a stride holds at most span_cap spans."""
    B, L, _ = khat.shape
    n_st = L // stride
    t = torch.arange(L, device=khat.device)
    cos = F.pad((khat[:, 1:] * khat[:, :-1]).sum(-1), (1, 0), value=-2.0)
    inner = member & F.pad(member[:, :-1], (1, 0), value=False) & (t % stride != 0)[None]   # may continue a span
    natural = inner & (cos < theta)
    room = span_cap - stride // span_max
    if room < stride:
        c = torch.where(natural, cos, torch.full_like(cos, 3.0)).view(B, n_st, stride)
        low = c.topk(max(room, 0), dim=-1, largest=False).indices
        keep = torch.zeros(B, n_st, stride, dtype=torch.bool, device=khat.device).scatter_(-1, low, True)
        natural = natural & keep.view(B, L)
    start = member & (~inner | natural)
    run0 = torch.cummax(torch.where(start, t[None].expand(B, L), torch.zeros(B, L, dtype=t.dtype, device=t.device)),
                        dim=1).values
    start = start | (member & ~start & ((t[None] - run0) % span_max == 0))
    slot = start.view(B, n_st, stride).long().cumsum(-1) - 1
    labels = torch.where(member.view(B, n_st, stride), slot, torch.full_like(slot, -1))
    return labels, start


@torch.no_grad()
def form_concepts(u_sum: torch.Tensor, valid: torch.Tensor, sdoc: torch.Tensor, lam, *, c_max: int,
                  rounds: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The leader rule, stride by stride, batched over rows. u_sum [B, n_st, K, d]: Σ k̂ over each span's tokens;
    valid [B, n_st, K] bool; sdoc [B, n_st] long, the document of the stride's spans (−1 = none).

    Returns cof [B, n_st, K] long (each span's concept slot, −1 for an empty span), n_c [B, n_st] long (slots in
    use after each stride, counted from the start of the stride's document) and the number of founders (a scalar
    tensor: concepts created over the whole batch)."""
    B, n_st, K, d = u_sum.shape
    dev = u_sum.device
    csum = torch.zeros(B, c_max, d, device=dev)
    n_cur = torch.zeros(B, dtype=torch.long, device=dev)
    cur_doc = torch.full((B,), -2, dtype=torch.long, device=dev)
    cof = torch.full((B, n_st, K), -1, dtype=torch.long, device=dev)
    n_c = torch.zeros(B, n_st, dtype=torch.long, device=dev)
    founders = torch.zeros((), dtype=torch.long, device=dev)
    ar = torch.arange(K, device=dev)
    lower = (ar[None, :] < ar[:, None])[None]                                           # [1, K, K]: j earlier than i
    slots = torch.arange(c_max, device=dev)[None, None, :]
    for s in range(n_st):
        ok = valid[:, s]
        d_s = sdoc[:, s]
        new_doc = (d_s >= 0) & (d_s != cur_doc)                                         # its concepts start at slot 0
        csum = torch.where(new_doc[:, None, None], torch.zeros_like(csum), csum)
        n_cur = torch.where(new_doc, torch.zeros_like(n_cur), n_cur)
        cur_doc = torch.where(d_s >= 0, d_s, cur_doc)
        u = F.normalize(u_sum[:, s], dim=-1)
        sim = torch.bmm(u, F.normalize(csum, dim=-1).transpose(1, 2))                   # [B, K, c_max]
        best, arg = sim.masked_fill(slots >= n_cur[:, None, None], -2.0).max(-1)
        join = ok & (best >= lam)
        fnd = ok & ~join
        near = (torch.bmm(u, u.transpose(1, 2)) >= lam) & fnd[:, :, None] & fnd[:, None, :]
        unres = fnd.clone()
        lead_of = ar[None].expand(B, K).clone()
        for _ in range(rounds):
            lead = unres & ~(near & lower & unres[:, None, :]).any(-1)                  # no earlier unresolved near
            fol = near & lead[:, None, :]
            has = fol.any(-1) & unres & ~lead
            lead_of = torch.where(has, fol.float().argmax(-1), lead_of)                 # its earliest new leader
            unres = unres & ~lead & ~has
        is_lead = fnd & (lead_of == ar[None])                                           # the unresolved rest lead too
        new_id = is_lead.long().cumsum(-1) - 1
        room = c_max - n_cur
        took = is_lead & (new_id < room[:, None])
        cid = (n_cur[:, None] + new_id).gather(1, lead_of)
        full = ~took.gather(1, lead_of)                                                 # its leader got no slot
        # the slots are full: the nearest concept of the strides before (slot 0 when the document has none yet)
        cid = torch.where(full, torch.where(n_cur[:, None] > 0, arg, torch.zeros_like(arg)), cid)
        assign = torch.where(join, arg, torch.where(fnd, cid, torch.full_like(cid, -1)))
        n_new = took.sum(-1)
        n_cur = n_cur + n_new
        founders = founders + n_new.sum()
        csum.scatter_add_(1, assign.clamp(min=0)[..., None].expand(B, K, d), u_sum[:, s] * ok[..., None])
        cof[:, s] = assign
        n_c[:, s] = n_cur
    return cof, n_c, founders


@dataclass
class LivingIndex:
    """One forward's grouping, built from the first pass's keys and used by every pass.
    G = n_st · span_cap span slots (slot g = stride · span_cap + its place in the stride), C concept slots."""

    groups: GroupIndex               # span slots as groups, for the exact reads (nn.exact_read)
    labels: torch.Tensor             # [B, n_st, stride] long — each token's span within its stride, −1 = none
    tok_slot: torch.Tensor           # [B, n_st · stride] long — its span slot, G = none (a dump index)
    span_n: torch.Tensor             # [B, G] fp32 — member tokens (0 = empty slot)
    span_first: torch.Tensor         # [B, G] long — first position (spans are contiguous; 0 for an empty slot)
    span_doc: torch.Tensor           # [B, G] long — document (−1 = empty slot)
    sdoc: torch.Tensor               # [B, n_st] long — the document whose tokens stride s holds (−1 = none)
    tokens: torch.Tensor             # scalar — member tokens of the batch
    spans: torch.Tensor              # scalar — spans of the batch
    # levels 2
    cof: Optional[torch.Tensor] = None       # [B, G] long — concept slot of each span, −1 = empty slot
    n_c: Optional[torch.Tensor] = None       # [B, n_st] long — concept slots in use after each stride
    seg: Optional[torch.Tensor] = None       # [B, n_st, n_st] fp32 — 1 where stride t ≤ s holds the document of s
    ver_n: Optional[torch.Tensor] = None     # [B, n_st, C] fp32 — tokens of each concept in each version
    founders: Optional[torch.Tensor] = None  # scalar — concepts founded in the batch (levels 2)
    stride: int = 512
    span_cap: int = 128
    span_max: int = 16
    c_max: int = 0
    levels: int = 2                          # 1: every span is its own concept (C = G, slot = span slot)

    @property
    def n_strides(self) -> int:
        return int(self.labels.shape[1])


def concept_slots(S: int) -> int:
    """C_max: S / 16 concept slots per row, rounded up to whole flex blocks (1,024 at 16k, 4,096 at 64k)."""
    return max(BLOCK, -(-(-(-S // 16)) // BLOCK) * BLOCK)


@torch.no_grad()
def build_living(k: torch.Tensor, doc: torch.Tensor, sink: torch.Tensor, *, stride: int, span_max: int,
                 span_cap: int, theta, lam, levels: int = 2, c_max: int = 0) -> Optional[LivingIndex]:
    """Spans and concepts of every stride some query sees as memory (stride ≤ the last query's stride − 2).
    k [B, S, 1, d] the reading layer's first-pass keys (after their norm, no position code), doc [B, S] (−1 =
    padding), sink [B, S] bool. None when the row has no such stride. `levels` 1 (arm `living_spans`): no leader
    rule, every span is its own concept (concept slot = span slot, C = G), so the same read searches the spans flat
    and keeps one summary per span for what it did not open."""
    B, S = doc.shape
    n_st = (S - 1) // stride - 1
    if n_st <= 0:
        return None
    dev = k.device
    d = k.shape[-1]
    L, K = n_st * stride, span_cap
    G = n_st * K
    khat = F.normalize(k[:, :L, 0].detach().float(), dim=-1)
    ahead = doc[:, (torch.arange(n_st, device=dev) + 2) * stride]                       # [B, n_st]
    member = ((doc[:, :L].view(B, n_st, stride) == ahead[..., None]) & (ahead[..., None] >= 0)
              & ~sink[:, :L].view(B, n_st, stride))
    labels, _ = cut_spans(khat, member.view(B, L), theta, stride=stride, span_max=span_max, span_cap=span_cap)
    ds = doc_starts(doc)
    st_ar = torch.arange(n_st, device=dev)
    g_close = ((st_ar + 2) * stride).repeat_interleave(K)[None].expand(B, -1).contiguous()
    gi = groups_from_spans(labels, [s * stride for s in range(n_st)], S=S, per_span=K, doc_start=ds,
                           g_close=g_close, g_doc=torch.zeros_like(g_close), any_doc=True, sink=sink)
    tok_slot = torch.where(labels >= 0, labels + (st_ar * K)[None, :, None], torch.full_like(labels, G)).view(B, L)
    span_n = torch.zeros(B, G + 1, device=dev).scatter_add_(1, tok_slot, torch.ones(B, L, device=dev))[:, :G]
    has = span_n > 0
    sdoc = torch.where(member.any(-1), ahead, torch.full_like(ahead, -1))
    span_doc = torch.where(has, sdoc.repeat_interleave(K, dim=1), torch.full_like(g_close, -1))
    lidx = LivingIndex(groups=gi, labels=labels, tok_slot=tok_slot, span_n=span_n,
                       span_first=torch.where(has, gi.g_minpos, torch.zeros_like(gi.g_minpos)), span_doc=span_doc,
                       sdoc=sdoc, tokens=member.sum(), spans=has.sum(), stride=stride, span_cap=K,
                       span_max=span_max, levels=int(levels))
    if levels < 2:
        # every span its own concept: slot g holds span g; version s lists the slots of strides ≤ s
        C = -(-G // BLOCK) * BLOCK
        cof = torch.where(has, torch.arange(G, device=dev)[None].expand(B, G), torch.full_like(g_close, -1))
        n_c = ((st_ar + 1) * K)[None].expand(B, n_st).contiguous()
        founders = None
    else:
        C = int(c_max) or concept_slots(S)
        u_sum = torch.zeros(B, G + 1, d, device=dev).scatter_add_(1, tok_slot[..., None].expand(B, L, d), khat)
        cof, n_c, founders = form_concepts(u_sum[:, :G].view(B, n_st, K, d), has.view(B, n_st, K), sdoc, lam,
                                           c_max=C)
    seg = ((sdoc[:, :, None] == sdoc[:, None, :]) & (sdoc[:, :, None] >= 0)
           & (st_ar[None, :] <= st_ar[:, None])[None]).float()
    cof = cof.reshape(B, n_st, K)
    inc_n = torch.zeros(B, n_st, C + 1, device=dev).scatter_add_(
        2, torch.where(cof >= 0, cof, torch.full_like(cof, C)), span_n.view(B, n_st, K))[:, :, :C]
    lidx.cof, lidx.n_c, lidx.seg, lidx.founders, lidx.c_max = cof.view(B, G), n_c, seg, founders, C
    lidx.ver_n = torch.einsum("bst,btc->bsc", seg, inc_n)
    return lidx


@dataclass
class LivingStats:
    """One pass's statistics. The versions carry gradient to the member tokens' keys and values; the spans'
    statistics only rank what opens (no gradient)."""

    span_mu_k: torch.Tensor          # [B, G, d]
    span_var: torch.Tensor           # [B, G] mean per-dimension key variance
    ver_mu_k: torch.Tensor           # [B, n_st, C, d]
    ver_var: torch.Tensor            # [B, n_st, C]
    ver_mu_v: torch.Tensor           # [B, n_st, C, d]


def living_stats(k: torch.Tensor, v: torch.Tensor, lidx: LivingIndex) -> LivingStats:
    """k, v [B, S, 1, d] of this pass → span statistics and the concepts' versions (sums over the spans of each
    stride and the earlier strides of its document)."""
    B = k.shape[0]
    d = k.shape[-1]
    n_st, K, C = lidx.n_strides, lidx.span_cap, lidx.c_max
    G = n_st * K
    L = n_st * lidx.stride
    kk, vv = k[:, :L, 0].float(), v[:, :L, 0].float()
    ix = lidx.tok_slot[..., None].expand(B, L, d)
    sk = kk.new_zeros(B, G + 1, d).scatter_add(1, ix, kk)[:, :G]
    sv = vv.new_zeros(B, G + 1, d).scatter_add(1, ix, vv)[:, :G]
    sk2 = kk.new_zeros(B, G + 1).scatter_add(1, lidx.tok_slot, (kk ** 2).sum(-1))[:, :G]
    with torch.no_grad():
        inv = 1.0 / lidx.span_n.clamp_min(1.0)
        span_mu = sk * inv[..., None]
        span_var = (sk2 * inv - (span_mu ** 2).sum(-1)).clamp_min(0.0) / d
    cx = torch.where(lidx.cof >= 0, lidx.cof, torch.full_like(lidx.cof, C)).view(B, n_st, K)
    cxd = cx[..., None].expand(B, n_st, K, d)
    inc_k = kk.new_zeros(B, n_st, C + 1, d).scatter_add(2, cxd, sk.view(B, n_st, K, d))[:, :, :C]
    inc_v = vv.new_zeros(B, n_st, C + 1, d).scatter_add(2, cxd, sv.view(B, n_st, K, d))[:, :, :C]
    inc_k2 = kk.new_zeros(B, n_st, C + 1).scatter_add(2, cx, sk2.view(B, n_st, K))[:, :, :C]
    inv = (1.0 / lidx.ver_n.clamp_min(1.0))
    mu_k = torch.einsum("bst,btcd->bscd", lidx.seg, inc_k) * inv[..., None]
    mu_v = torch.einsum("bst,btcd->bscd", lidx.seg, inc_v) * inv[..., None]
    var = (torch.einsum("bst,btc->bsc", lidx.seg, inc_k2) * inv - (mu_k ** 2).sum(-1)).clamp_min(0.0) / d
    return LivingStats(span_mu_k=span_mu, span_var=span_var, ver_mu_k=mu_k, ver_var=var, ver_mu_v=mu_v)


def match_score(qc: torch.Tensor, mu: torch.Tensor, var: Optional[torch.Tensor]) -> torch.Tensor:
    """The estimated attention weight of an average member, best query head: q·μ/√d + |q|² σ² / (2d).
    qc [B, ch, h, d] fp32, mu [B, N, d], var [B, N] or None → [B, ch, N]."""
    d = qc.shape[-1]
    s = torch.einsum("bqhd,bnd->bqhn", qc, mu) / math.sqrt(d)
    if var is not None:
        s = s + ((qc ** 2).sum(-1, keepdim=True) / (2 * d)) * var[:, None, None, :]
    return s.amax(2)


def _log_n(n: torch.Tensor) -> torch.Tensor:
    return torch.where(n > 0, n.clamp_min(1.0).log(), torch.zeros_like(n))


@torch.no_grad()
def _choose_chunk(qc, qd, qp, ver: int, lidx: LivingIndex, st: LivingStats, *, top_k: int, width: int, keep: int,
                  variance: bool, search: str, score: str = "match"):
    """The choice of the queries `qp` (all in stride ver + 2): ids [B, ch, m] long, off / take [B, ch, m] int32
    (places used before each chosen span; tokens it opens), n_open [B, ch, C] int16 (tokens opened per concept).
    `score` "match": the estimated weight of an average member (size does not win); "mass": + log n, the estimated
    attention mass of the whole group (the concept read's rule; a test switch)."""
    B, ch = qd.shape
    K, C = lidx.span_cap, lidx.c_max
    Gv = (ver + 1) * K                                                                  # span slots two strides back
    cand = (lidx.span_doc[:, None, :Gv] == qd[..., None]) & (qd >= 0)[..., None]
    cof = lidx.cof[:, :Gv].clamp(min=0)
    mass = score == "mass"
    if search != "flat" and lidx.levels >= 2:
        sc = match_score(qc, st.ver_mu_k[:, ver].detach(), st.ver_var[:, ver].detach() if variance else None)
        if mass:
            sc = sc + _log_n(lidx.ver_n[:, ver])[:, None, :]
        okc = (lidx.ver_n[:, ver] > 0)[:, None, :] & (lidx.sdoc[:, ver][:, None] == qd)[..., None]
        top = sc.masked_fill(~okc, NEG).topk(min(int(keep), C), dim=-1)
        keepc = torch.zeros(B, ch, C + 1, dtype=torch.bool, device=qc.device).scatter_(
            2, torch.where(top.values > NEG, top.indices, torch.full_like(top.indices, C)), True)
        cand = cand & keepc.gather(2, cof[:, None, :].expand(B, ch, Gv))
    ss = match_score(qc, st.span_mu_k[:, :Gv], st.span_var[:, :Gv] if variance else None)
    if mass:
        ss = ss + _log_n(lidx.span_n[:, :Gv])[:, None, :]
    ids, off, cnt = choose(ss, cand, lidx.groups, qp, m=width, cap=top_k, raw_window=lidx.stride, return_counts=True)
    take = torch.minimum(cnt, top_k - off).clamp(min=0)
    n_open = torch.zeros(B, ch, C, dtype=torch.int16, device=qc.device).scatter_add_(
        2, cof.gather(1, ids.reshape(B, -1)).view(ids.shape), take.to(torch.int16))
    return ids, off, take, n_open


def _chunk(budget: int, per_query: int, cap: int) -> int:
    return max(1, min(int(cap), budget // max(int(per_query), 1)))


@torch.no_grad()
def living_open_table(q: torch.Tensor, lidx: LivingIndex, st: LivingStats, doc: torch.Tensor, *, top_k: int,
                      width: int, keep: int, variance: bool = True, search: str = "two_level",
                      score: str = "match", chunk: int = 0) -> tuple[OpenTable, torch.Tensor]:
    """Every query's choice as the table T [B, S, G] (nn.exact_read.OpenTable) and n_open [B, S, C] int16, the
    tokens it opens per concept. q [B, S, h, d]."""
    B, S, h, d = q.shape
    K, C, stride = lidx.span_cap, lidx.c_max, lidx.stride
    tab = OpenTable(lidx.groups, B, S, cap=top_k)
    tab.span_blocks = 2                                   # a span of ≤ 16 adjacent tokens touches at most two blocks
    n_open = torch.zeros(B, S, C, dtype=torch.int16, device=q.device)
    for tau in range(2, -(-S // stride)):
        ver = tau - 2
        hi = min(S, (tau + 1) * stride)
        step = chunk or _chunk(1 << 25, B * h * max((ver + 1) * K, C), stride)
        for s0 in range(tau * stride, hi, step):
            s1 = min(hi, s0 + step)
            qp = torch.arange(s0, s1, device=q.device)
            ids, off, _, no = _choose_chunk(q[:, s0:s1].float(), doc[:, s0:s1], qp, ver, lidx, st, top_k=top_k,
                                            width=width, keep=keep, variance=variance, search=search, score=score)
            tab.add(s0, ids, off)
            n_open[:, s0:s1] = no
    return tab, n_open


# ------------------------------------------------------------------------------------------------ the read (flex)
def version_block_mask(mask_mod: Callable, *, B: int, S: int, n_st: int, C: int, stride: int, n_c: torch.Tensor):
    """Block mask of the summary call (queries S, keys n_st · C: version v's concepts at [v·C, (v + 1)·C)), built
    from the structure instead of evaluating every (query, key) pair: a 128-query block lies in one stride τ and
    lists the ⌈n_c[v] / 128⌉ key blocks of its version v = τ − 2 that hold a concept. Every listed block is
    partial, so `mask_mod` decides each entry. n_c [B, n_st] long."""
    from torch.nn.attention.flex_attention import BlockMask

    dev = n_c.device
    nqb = -(-S // BLOCK)
    per = C // BLOCK
    nkb = n_st * per
    ver = (torch.arange(nqb, device=dev) * BLOCK) // stride - 2
    v = ver.clamp(0, n_st - 1)
    num = torch.where((ver >= 0)[None], -(-n_c[:, v] // BLOCK), torch.zeros_like(n_c[:, v])).to(torch.int32)
    idx = ((v * per)[:, None] + torch.arange(per, device=dev)[None]).to(torch.int32)    # [nqb, per]
    idx = F.pad(idx, (0, nkb - per)) if per < nkb else idx[:, :nkb]
    return BlockMask.from_kv_blocks(num[:, None], idx[None, None].expand(B, 1, nqb, nkb).contiguous(), None, None,
                                    BLOCK_SIZE=BLOCK, mask_mod=mask_mod, seq_lengths=(S, n_st * C))


def attend_living(q, k, v, *, lidx: Optional[LivingIndex], st: Optional[LivingStats], opened: Optional[OpenTable],
                  n_open: Optional[torch.Tensor], doc, sink, ds, n_sinks: int, key_valid, stride: int,
                  summaries: bool = True, variance: bool = True, sum_len: int = 0, share: Optional[list] = None):
    """The flex read: sinks ‖ the query's stride and the one before ‖ opened tokens ‖ one summary per concept of
    the query's version, one softmax (two calls merged by their log-sum-exp). q [B,S,h,d], k / v [B,S,1,d], doc
    [B,S] (−1 = pad), sink [B,S] bool, ds [B,S] document starts → [B,S,h,d]. `share` (a list) receives the
    summaries' share of the weight [B,h,S], for the training statistics."""
    from torch.nn.attention.flex_attention import AuxRequest

    from nn.concept_memory import flex_raw_call, merge_calls, summary_log_scale
    from nn.perceiver_ar_lm import _get_flex

    B, S, h, d = q.shape
    g = k.shape[2]
    doc32 = doc.to(torch.int32)
    o1, l1 = flex_raw_call(q, k, k, v, two=False, doc32=doc32, sk=sink.to(torch.int32), ds=ds, n_sinks=n_sinks,
                           key_valid=key_valid, opened=opened, blk=stride)
    if not summaries or lidx is None or st is None:
        return o1.transpose(1, 2)
    n_st, C = lidx.n_strides, lidx.c_max
    if stride % BLOCK or C % BLOCK:
        raise ValueError("the flex living read needs living_stride and the concept slots in whole 128-blocks")
    Ks = st.ver_mu_k.reshape(B, n_st * C, 1, d).to(k.dtype)
    Vs = st.ver_mu_v.reshape(B, n_st * C, 1, d).to(v.dtype)
    # captured by score_mod / mask_mod: no gradient (see attend_concepts on atomic adds into shared entries)
    nver = lidx.ver_n.reshape(B, n_st * C).contiguous()
    vhalf = ((0.5 * st.ver_var) if variance else torch.zeros_like(st.ver_var)).detach().reshape(B, n_st * C).contiguous()
    qn2 = ((q.detach().float() ** 2).sum(-1).transpose(1, 2) / d).contiguous()         # [B, h, S]
    vdoc = lidx.sdoc.to(torch.int32).contiguous()
    nop = n_open if n_open is not None else torch.zeros(B, S, C, dtype=torch.int16, device=q.device)

    def spred(b, hh, qi, j):
        ver = torch.div(j, C, rounding_mode="floor")
        rem = nver[b, j] - nop[b, qi, j - ver * C]
        return ((ver == torch.div(qi, stride, rounding_mode="floor") - 2) & (vdoc[b, ver] == doc32[b, qi])
                & (doc32[b, qi] >= 0) & (rem > 0))

    def smod(score, b, hh, qi, j):
        rem = nver[b, j] - nop[b, qi, j - torch.div(j, C, rounding_mode="floor") * C]
        return score + torch.log(rem.clamp_min(1.0)) + qn2[b, hh, qi] * vhalf[b, j]

    bm = version_block_mask(spred, B=B, S=S, n_st=n_st, C=C, stride=stride, n_c=lidx.n_c)
    o2, aux2 = _get_flex()(q.transpose(1, 2).contiguous(), Ks.transpose(1, 2).contiguous(),
                           Vs.transpose(1, 2).contiguous(), block_mask=bm, score_mod=smod, enable_gqa=(g != h),
                           return_aux=AuxRequest(lse=True))
    logrho = summary_log_scale(ds, S, stride, sum_len)
    if share is not None:
        out, w2 = merge_calls(o1, l1, o2, aux2.lse, logrho, q.dtype, return_share=True)
        share.append(w2)
        return out.transpose(1, 2)
    return merge_calls(o1, l1, o2, aux2.lse, logrho, q.dtype).transpose(1, 2)


# ------------------------------------------------------------------------------------------------ the read (stream)
def _take(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x [B, S, …] at pos [B, ch, P] → [B, ch, P, …]."""
    B, ch, P = pos.shape
    flat = pos.reshape(B, ch * P)
    if x.dim() == 2:
        return x.gather(1, flat).view(B, ch, P)
    return x.gather(1, flat[..., None].expand(B, ch * P, x.shape[-1])).view(B, ch, P, x.shape[-1])


def living_read_stream(q, k, v, *, lidx: Optional[LivingIndex], st: Optional[LivingStats], doc, ds, n_sinks: int,
                       key_valid, top_k: int, width: int, keep: int, stride: int, summaries: bool = True,
                       variance: bool = True, search: str = "two_level", score: str = "match", sum_len: int = 0,
                       q_offset: int = 0, chunk: int = 0, trace: Optional[Callable] = None) -> torch.Tensor:
    """The same read without a [B, S, ·] table: per chunk of queries of one stride, the choice, then one fp32
    softmax over the window ‖ the document's sinks outside it ‖ the opened tokens (gathered) ‖ the summaries of the
    version. Differentiable (the choice itself carries no gradient). `lidx` None: sinks and window only.

    q [B, Q, h, d] holds the queries [q_offset, q_offset + Q) of the row that k / v [B, S, 1, d], doc and ds
    [B, S] cover. `trace(s0, s1, parts)` receives each chunk's weight per part, {"window" | "sinks" | "opened" |
    "summaries": [B, ch, h]} (probes). Returns [B, Q, h, d]."""
    from nn.concept_memory import summary_log_scale

    B, Q, h, d = q.shape
    S = int(k.shape[1])
    q0 = int(q_offset)
    dev = q.device
    sc = d ** -0.5
    kf, vf = k[:, :, 0], v[:, :, 0]
    use_mem = lidx is not None and st is not None
    C = lidx.c_max if use_mem else 0
    logrho = summary_log_scale(ds, S, stride, sum_len) if (use_mem and summaries) else None
    out = []
    s0 = q0
    while s0 < q0 + Q:
        tau = s0 // stride
        hi = min(q0 + Q, (tau + 1) * stride)
        ver = tau - 2
        per = (2 * stride + n_sinks + (top_k + C if (use_mem and ver >= 0) else 0)) * h
        s1 = min(hi, s0 + (chunk or _chunk(1 << 25, B * per, stride)))
        ch = s1 - s0
        qp = torch.arange(s0, s1, device=dev)
        qc = q[:, s0 - q0:s1 - q0].float()
        qd = doc[:, s0:s1]
        okq = (qd >= 0)[..., None]
        names, logits, vals = [], [], []
        # the window: the query's stride and the one before
        j0 = max(0, (tau - 1) * stride)
        jw = torch.arange(j0, s1, device=dev)
        ok = (jw[None, :] <= qp[:, None])[None] & (doc[:, j0:s1][:, None, :] == qd[..., None]) & okq
        if key_valid is not None:
            ok = ok & (key_valid.bool()[:, j0:s1][:, None, :] | (jw[None, :] == qp[:, None])[None])
        lw = torch.einsum("bqhd,bkd->bqhk", qc, kf[:, j0:s1].float()) * sc
        names.append("window")
        logits.append(lw.masked_fill(~ok[:, :, None, :], NEG))
        vals.append(("shared", vf[:, j0:s1].float()))
        # the document's sinks outside the window
        if n_sinks and j0 > 0:
            sp = ds[:, s0:s1][..., None] + torch.arange(n_sinks, device=dev)            # [B, ch, n]
            spc = sp.clamp(max=S - 1)
            ok = (sp < j0) & (_take(doc, spc) == qd[..., None]) & okq
            if key_valid is not None:
                ok = ok & _take(key_valid.bool(), spc)
            ls = torch.einsum("bqhd,bqpd->bqhp", qc, _take(kf, spc).float()) * sc
            names.append("sinks")
            logits.append(ls.masked_fill(~ok[:, :, None, :], NEG))
            vals.append(("own", _take(vf, spc).float()))
        if use_mem and ver >= 0:
            ids, off, take, n_open = _choose_chunk(qc.detach(), qd, qp, ver, lidx, st, top_k=top_k, width=width,
                                                   keep=keep, variance=variance, search=search, score=score)
            # the opened tokens: each chosen span's first `take` members, packed into top_k places
            F_ = lidx.span_max
            m = ids.shape[-1]
            r = torch.arange(F_, device=dev)
            first = lidx.span_first.gather(1, ids.reshape(B, -1)).view(B, ch, m)
            okf = (r < take[..., None]).reshape(B, ch, m * F_)
            posf = (first[..., None] + r).reshape(B, ch, m * F_)
            P = min(int(top_k), m * F_)
            dest = torch.where(okf, okf.long().cumsum(-1) - 1, torch.full_like(posf, P))
            pos = torch.zeros(B, ch, P + 1, dtype=torch.long, device=dev).scatter_(2, dest, posf)[..., :P]
            ok = (torch.arange(P, device=dev) < okf.sum(-1, keepdim=True)) & okq
            if key_valid is not None:
                ok = ok & _take(key_valid.bool(), pos)
            lo = torch.einsum("bqhd,bqpd->bqhp", qc, _take(kf, pos).float()) * sc
            names.append("opened")
            logits.append(lo.masked_fill(~ok[:, :, None, :], NEG))
            vals.append(("own", _take(vf, pos).float()))
            if summaries:
                rem = lidx.ver_n[:, ver][:, None, :] - n_open                           # [B, ch, C]
                ok = (rem > 0) & (lidx.sdoc[:, ver][:, None] == qd)[..., None] & okq
                bias = torch.log(rem.clamp_min(1.0))[:, :, None, :]
                if variance:
                    bias = bias + ((qc.detach() ** 2).sum(-1, keepdim=True) / (2 * d)) * st.ver_var[:, ver].detach()[:, None, None, :]
                if logrho is not None:
                    bias = bias + logrho[:, s0:s1][:, :, None, None]
                lsum = torch.einsum("bqhd,bcd->bqhc", qc, st.ver_mu_k[:, ver]) * sc + bias
                names.append("summaries")
                logits.append(lsum.masked_fill(~ok[:, :, None, :], NEG))
                vals.append(("shared", st.ver_mu_v[:, ver]))
        w = torch.nan_to_num(torch.softmax(torch.cat(logits, dim=-1), dim=-1), nan=0.0)
        o, a, parts = 0.0, 0, {}
        for name, lg, (kind, val) in zip(names, logits, vals):
            n = lg.shape[-1]
            wi = w[..., a:a + n]
            a += n
            parts[name] = wi.sum(-1)
            o = o + (torch.einsum("bqhk,bkd->bqhd", wi, val) if kind == "shared"
                     else torch.einsum("bqhp,bqpd->bqhd", wi, val))
        if trace is not None:
            trace(s0, s1, parts)
        out.append(o.to(q.dtype))
        s0 = s1
    return torch.cat(out, dim=1)


# ------------------------------------------------------------------------------------------------ controllers
def servo_step(value: torch.Tensor, observed: torch.Tensor, target: float, step: float) -> torch.Tensor:
    """One step of a threshold's controller: against the relative error of the observed rate (clipped to ±1), kept
    in [−1, 1]. Both thresholds cut more when they are higher (θ: a cut where the cosine is below it; λ: a span
    founds where its best cosine is below it), so a rate above its target lowers the threshold."""
    err = ((observed - target) / target).clamp(-1.0, 1.0)
    return (value - step * err).clamp(-1.0, 1.0)


STAT_KEYS = ("tokens", "spans", "spans_len1", "spans_full", "strides", "strides_at_cap", "rows", "concepts",
             "single_span", "largest_tokens", "row_tokens", "founders", "summary_weight", "weight_n", "forwards")


@torch.no_grad()
def living_numbers(lidx: LivingIndex) -> torch.Tensor:
    """The grouping's own numbers of one forward as sums (readout R4), in STAT_KEYS order up to `founders`; ratios
    are formed when they are logged (`stats_report`). No host sync."""
    B = lidx.span_n.shape[0]
    n_st, K = lidx.n_strides, lidx.span_cap
    n = lidx.span_n
    has = n > 0
    per_stride = has.view(B, n_st, K).sum(-1)
    used = per_stride > 0
    z = torch.zeros((), device=n.device)
    out = [lidx.tokens.float(), has.sum().float(), (n == 1).sum().float(), (n >= lidx.span_max).sum().float(),
           used.sum().float(), (per_stride >= K).sum().float()]
    if lidx.cof is not None:
        C = lidx.c_max
        last = lidx.ver_n[:, -1]                                                         # tokens per concept at the end
        rows = last.sum(-1) > 0
        spans_c = torch.zeros(B, C + 1, device=n.device).scatter_add_(
            1, torch.where((lidx.cof >= 0) & (lidx.span_doc == lidx.sdoc[:, -1:]), lidx.cof, torch.full_like(lidx.cof, C)),
            has.float())[:, :C]
        fnd = lidx.founders if lidx.founders is not None else has.sum()                # levels 1: every span
        out += [rows.sum().float(), (last > 0).sum().float(), (spans_c == 1).sum().float(), last.amax(-1).sum(),
                last.sum(), fnd.float()]
    else:
        out += [z, z, z, z, z, z]
    return torch.stack(out)


def stats_report(acc: torch.Tensor, theta: float, lam: float) -> dict:
    """The accumulated sums as the logged ratios (keys `living/…`)."""
    v = dict(zip(STAT_KEYS, [float(x) for x in acc.tolist()]))
    if v["forwards"] <= 0:
        return {}

    def div(a, b):
        return v[a] / v[b] if v[b] > 0 else 0.0

    out = {"living/theta": theta, "living/lam": lam, "living/span_len": div("tokens", "spans"),
           "living/share_len1": div("spans_len1", "spans"), "living/share_len_max": div("spans_full", "spans"),
           "living/spans_per_stride": div("spans", "strides"), "living/strides_at_cap": div("strides_at_cap", "strides"),
           "living/member_tokens_per_forward": div("tokens", "forwards")}
    if v["rows"] > 0:
        out.update({"living/concepts_per_doc": div("concepts", "rows"), "living/found_rate": div("founders", "spans"),
                    "living/single_span_share": div("single_span", "concepts"),
                    "living/largest_share": div("largest_tokens", "row_tokens")})
    if v["weight_n"] > 0:
        out["living/summary_weight"] = div("summary_weight", "weight_n")
    return out

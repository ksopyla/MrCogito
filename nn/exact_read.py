"""E33p — the exact read: each query opens the raw keys of the groups it chose (notes, pages, LSA buckets).

Spec: docs/experiments_specs/ahead/E33p_note_indexed_exact_read.md · plan: E33p_plan.md (§0, §3).

A *group* is a set of earlier raw tokens: a note's member tokens (the page tokens whose largest writer assignment is
that note), a page's picked tokens, or an LSA bucket. Groups live in *spans* (a page, an LSA block of keys); a token
belongs to at most one group per span. Spans may overlap (pages, stride < window): the spans are split into `layers`
whose members never overlap, and each layer gives every token one code (−1 = not a member there).

Per query i, without gradient: rank the candidate groups, take the best m, fill `cap` places best group first, each
group's members oldest first, counting only members in [doc_start(i), i − raw_window] (the rest are in i's window or
in another document; sinks are never members). The result is one table T[b, i, g] = the places used before group g
(`cap` when g is not chosen). Raw token j, a member of g at rank ord(j) (members of g in [doc_start(j), j)), is
opened for i iff i − j ≥ raw_window and T[i, g] + ord(j) < cap. A token in two chosen groups opens once (an OR),
counted in both (≤ cap distinct tokens per query). The attention kernel reads T through one 2-D gather per layer plus
1-D lookups along the key axis; `dense_open` is the same rule as a dense mask (SDPA, tests, probes).

Shapes (s30, 16k row): S = 16,384; notes G = 2,720 (85 pages × 32), pages G = 85, LSA buckets G = 2,048
(32 closed 512-key blocks × 64); T [B, S, G] int16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

ORD_BITS = 9            # code = gid · 512 + ord (ord < 512: members of one group within a ≤ 512-token span)
ORD_MOD = 1 << ORD_BITS
SINK = -2               # code of an LSA sink token (layer 0)
_BIG = 1 << 40


@dataclass
class GroupIndex:
    """Groups of raw keys for one forward (or one loop round, for LSA). G = n_s · per_span groups, gid = s · per_span + l."""

    span_start: torch.Tensor   # [n_s] long — first position of each span
    span_len: int              # L
    per_span: int              # groups per span
    prefix: torch.Tensor       # [B, n_s, L + 1, per_span] int32 — members of group (s, l) at span offsets < o
    codes: torch.Tensor        # [n_layers, B, S] int32 — gid · 512 + ord, −1 not a member, −2 sink (layer 0)
    doc_start: torch.Tensor    # [B, S] long — first position of each token's document
    g_close: torch.Tensor      # [B, G] long — the group is visible to queries i ≥ g_close (−1 = never)
    g_doc: torch.Tensor        # [B, G] long — its document (−1 = none); `any_doc` groups are visible across documents
    g_minpos: torch.Tensor     # [B, G] long — oldest member (≥ 2⁴⁰ when it has none)
    g_kb: torch.Tensor         # [B, G, 2] long — first and last 128-key block holding a member (lo > hi when none)
    any_doc: bool = False

    @property
    def n_groups(self) -> int:
        return int(self.g_close.shape[1])


def span_layers(starts: list[int], length: int) -> list[int]:
    """Assign spans to layers so that spans of one layer never overlap (greedy, in start order). Static (Python)."""
    ends: list[int] = []
    out = []
    for s in starts:
        for li, e in enumerate(ends):
            if e <= s:
                ends[li] = s + length
                out.append(li)
                break
        else:
            ends.append(s + length)
            out.append(len(ends) - 1)
    return out


def doc_starts(doc: torch.Tensor) -> torch.Tensor:
    """[B, S] long: the first position of each token's document (contiguous documents; −1 ids count as documents)."""
    B, S = doc.shape
    idx = torch.arange(S, device=doc.device)[None].expand(B, S)
    new = torch.ones_like(doc, dtype=torch.bool)
    new[:, 1:] = doc[:, 1:] != doc[:, :-1]
    return torch.cummax(torch.where(new, idx, torch.zeros_like(idx)), dim=1).values


def groups_from_spans(member: torch.Tensor, span_start: list[int], *, S: int, per_span: int, doc_start: torch.Tensor,
                      g_close: torch.Tensor, g_doc: torch.Tensor, any_doc: bool = False,
                      sink: Optional[torch.Tensor] = None, block: int = 128) -> GroupIndex:
    """Build a GroupIndex from each span's member labels.

    member [B, n_s, L] long: the group (0 … per_span − 1) of the token at offset o of span s, −1 = not a member
    (unpicked, out of range, sink); positions span_start[s] + o must be < S where member ≥ 0.
    g_close / g_doc [B, G]: visibility of each group (set by the caller). sink [B, S] bool: LSA sinks (code −2)."""
    B, n_s, L = member.shape
    dev = member.device
    st = torch.tensor(span_start, device=dev, dtype=torch.long)
    pos = st[:, None] + torch.arange(L, device=dev)[None]                               # [n_s, L]
    valid = (member >= 0) & (pos < S)[None]
    ml = torch.where(valid, member, torch.zeros_like(member))
    onehot = (ml[..., None] == torch.arange(per_span, device=dev)) & valid[..., None]   # [B, n_s, L, K]
    prefix = F.pad(onehot.to(torch.int32).cumsum(2), (0, 0, 1, 0))                      # [B, n_s, L + 1, K]
    # ord: members of the token's group in [doc_start(token), token)
    pos_c = pos.clamp(max=S - 1)
    before = prefix[:, :, :L].gather(-1, ml[..., None]).squeeze(-1)                      # [B, n_s, L]
    ds = doc_start[:, pos_c.reshape(-1)].view(B, n_s, L)
    o_ds = (ds - st[None, :, None]).clamp(0, L)
    at_ds = prefix.gather(2, o_ds[..., None].expand(B, n_s, L, per_span)).gather(-1, ml[..., None]).squeeze(-1)
    ordv = (before - at_ds).clamp(min=0)
    gid = torch.arange(n_s, device=dev)[None, :, None] * per_span + ml
    code = torch.where(valid, gid.to(torch.int32) * ORD_MOD + ordv.to(torch.int32), torch.full_like(ordv, -1,
                                                                                                    dtype=torch.int32))
    layers = span_layers(list(span_start), L)
    n_layers = max(layers) + 1 if layers else 1
    codes = torch.full((n_layers, B, S + 1), -1, dtype=torch.int32, device=dev)
    for li in range(n_layers):
        sel_l = [s for s, x in enumerate(layers) if x == li]  # static: no host sync
        if not sel_l:
            continue
        sel = torch.tensor(sel_l, device=dev)
        p = pos[sel].reshape(-1)
        c = code[:, sel].reshape(B, -1)
        v = valid[:, sel].reshape(B, -1)
        dest = torch.where(v, p[None].expand(B, -1), torch.full_like(c, S, dtype=torch.long))
        codes[li].scatter_(1, dest, c)
    codes = codes[..., :S].contiguous()
    if sink is not None:
        codes[0] = torch.where(sink, torch.full_like(codes[0], SINK), codes[0])
    big = torch.full_like(pos, _BIG)[None, :, :, None]
    p4 = pos[None, :, :, None].expand(B, n_s, L, per_span)
    minpos = torch.where(onehot, p4, big).amin(2).reshape(B, n_s * per_span)
    maxpos = torch.where(onehot, p4, -torch.ones_like(p4)).amax(2).reshape(B, n_s * per_span)
    kb = torch.stack([torch.where(minpos < _BIG, minpos // block, torch.full_like(minpos, 1)),
                      torch.where(maxpos >= 0, maxpos // block, torch.zeros_like(maxpos))], dim=-1)
    return GroupIndex(span_start=st, span_len=L, per_span=per_span, prefix=prefix, codes=codes, doc_start=doc_start,
                      g_close=g_close, g_doc=g_doc, g_minpos=minpos, g_kb=kb, any_doc=any_doc)


def counts(index: GroupIndex, gid: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Members of each group `gid` [B, …] at positions in [lo, hi] (inclusive; broadcast to gid's shape), int32."""
    B = gid.shape[0]
    L, K = index.span_len, index.per_span
    s = torch.div(gid, K, rounding_mode="floor")
    l = gid - s * K
    st = index.span_start[s]
    o_hi = (hi + 1 - st).clamp(0, L)
    o_lo = torch.minimum((lo - st).clamp(0, L), o_hi)
    flat = index.prefix[:B].reshape(B, -1)  # the first B rows (probes pass one row)

    def at(o):
        i = ((s * (L + 1) + o) * K + l).reshape(B, -1)
        return flat.gather(1, i).view(gid.shape)

    return (at(o_hi) - at(o_lo)).clamp(min=0)


@torch.no_grad()
def choose(score: torch.Tensor, cand: torch.Tensor, index: GroupIndex, q_pos: torch.Tensor, *, m: int, cap: int,
           raw_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One query chunk: top-m candidates, then the places each takes. score / cand [B, ch, G], q_pos [ch] long.

    Returns ids [B, ch, m'] long (m' = min(m, G), best first) and off [B, ch, m'] int32: places used before each
    chosen group, `cap` where the group is not a real choice or the cap is already full."""
    B, ch, G = score.shape
    mm = min(int(m), G)
    top = score.masked_fill(~cand, float("-inf")).topk(mm, dim=-1)
    ids = top.indices
    valid = top.values > float("-inf")
    lo = index.doc_start[:, q_pos][..., None].expand(B, ch, mm)
    hi = (q_pos - int(raw_window))[None, :, None].expand(B, ch, mm)
    cnt = torch.where(valid, counts(index, ids, lo, hi), torch.zeros_like(ids, dtype=torch.int32))
    off = cnt.cumsum(-1) - cnt
    off = torch.where(valid & (off < cap), off, torch.full_like(off, cap))
    return ids, off.to(torch.int32)


class OpenTable:
    """The per-query table T [B, S, G] int16 and the key blocks each 128-query block may open, built chunk by chunk."""

    def __init__(self, index: GroupIndex, B: int, S: int, *, cap: int, block: int = 128):
        dev = index.codes.device
        self.index, self.S, self.cap, self.block = index, S, int(cap), block
        G = index.n_groups
        self.T = torch.full((B, S, max(G, 1)), int(cap), dtype=torch.int16, device=dev)
        self.nqb, self.nkb = -(-S // block), -(-S // block)
        self.need = torch.zeros(B, self.nqb * self.nkb + 1, dtype=torch.bool, device=dev)
        self.span_blocks = -(-index.span_len // block) + 1

    def add(self, s0: int, ids: torch.Tensor, off: torch.Tensor) -> None:
        B, ch, m = ids.shape
        if m == 0:
            return
        self.T[:, s0:s0 + ch].scatter_(2, ids, off.to(torch.int16))
        q_pos = torch.arange(s0, s0 + ch, device=ids.device)
        qb = (q_pos // self.block)[None, :, None]
        kb = self.index.g_kb.gather(1, ids.reshape(B, -1, 1).expand(B, ch * m, 2)).view(B, ch, m, 2)
        live = off < self.cap
        dummy = self.nqb * self.nkb
        for d in range(self.span_blocks):
            b_ = kb[..., 0] + d
            ok = live & (b_ <= kb[..., 1]) & (b_ < self.nkb)
            idx = torch.where(ok, qb * self.nkb + b_, torch.full_like(b_, dummy))
            self.need.scatter_(1, idx.reshape(B, -1), True)

    def extra_blocks(self) -> torch.Tensor:
        """[B, nqb, nkb] long: raw key blocks with an opened token for some query of the block (−1 = none)."""
        B = self.need.shape[0]
        need = self.need[:, :-1].view(B, self.nqb, self.nkb)
        ids = torch.arange(self.nkb, device=need.device)[None, None].expand_as(need)
        return torch.where(need, ids, torch.full_like(ids, -1))


class ChoiceList:
    """E33t stage C (plan §10.1–2): each query's choice kept as a list instead of the table T [B, S, G], whose size
    grows with S² (4.3 GB at 128k, 275 GB at 1M): the chosen groups `ids` [B, S, m] int32, best first, and the places
    used before each, `off` [B, S, m] int16 (`cap` = not a real choice, or the cap already full) — 0.8 GB at 1M with
    m = 128. Same `add` as OpenTable, so `note_open_table` / `lsa_open_table` fill either. Evaluation only.

    `opened` gives each query's opened raw positions, the same tokens T opens when every token is in at most one
    group per forward (one code layer: the catalogue's notes on non-overlapping pages, LSA's closed blocks; more
    layers are refused): chosen group g with off_g < cap opens its members in [doc_start(i), i − raw_window], oldest
    first, at most cap − off_g of them, i.e. members[g, r0_g + r] for r < min(cnt_g, cap − off_g), with r0_g its
    members before doc_start(i) and cnt_g its members in the range (the count `choose` used).

    `row0` (cached decoding, nn/incremental.py): the list holds the queries [row0, row0 + S) only (positions stay
    absolute in `add` / `opened`); `members` reuses a members table built once for this index."""

    def __init__(self, index: GroupIndex, B: int, S: int, *, cap: int, m: int, row0: int = 0,
                 members: Optional[torch.Tensor] = None):
        if int(index.codes.shape[0]) != 1:
            raise NotImplementedError("the choice list needs one code layer (spans that do not overlap)")
        dev = index.codes.device
        self.index, self.S, self.cap = index, S, int(cap)
        self.row0 = int(row0)
        self.m = max(1, int(m))
        self.ids = torch.zeros(B, S, self.m, dtype=torch.int32, device=dev)
        self.off = torch.full((B, S, self.m), self.cap, dtype=torch.int16, device=dev)
        self._members: Optional[torch.Tensor] = members

    def add(self, s0: int, ids: torch.Tensor, off: torch.Tensor) -> None:
        ch, mm = ids.shape[1], ids.shape[2]
        if mm == 0:
            return
        r = s0 - self.row0
        self.ids[:, r:r + ch, :mm] = ids.to(torch.int32)
        self.off[:, r:r + ch, :mm] = off.to(torch.int16)

    def members(self) -> torch.Tensor:
        """[B, G, F] long: each group's member positions, oldest first (−1 = none); F = the largest group (≥ 1)."""
        if self._members is None:
            gi = self.index
            B, S = gi.codes.shape[1], gi.codes.shape[2]
            K, L, G = gi.per_span, gi.span_len, gi.n_groups
            c = gi.codes[0].long()                                                       # [B, S]
            is_m = c >= 0
            gid = torch.where(is_m, torch.div(c, ORD_MOD, rounding_mode="floor"), torch.zeros_like(c))
            s = torch.div(gid, K, rounding_mode="floor")
            l = gid - s * K
            j = torch.arange(S, device=c.device)[None].expand(B, S)
            o = (j - gi.span_start[s]).clamp(0, L)
            rank = gi.prefix.reshape(B, -1).gather(1, (s * (L + 1) + o) * K + l)          # members of g before j
            Fm = max(1, int(gi.prefix[:, :, L].max())) if gi.prefix.numel() else 1
            mem = torch.full((B, G * Fm + 1), -1, dtype=torch.long, device=c.device)
            dest = torch.where(is_m & (rank < Fm), gid * Fm + rank, torch.full_like(gid, G * Fm))
            mem.scatter_(1, dest, j)
            self._members = mem[:, :G * Fm].view(B, G, Fm)
        return self._members

    def opened(self, s0: int, s1: int, raw_window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Opened raw positions of queries [s0, s1): pos [B, ch, m·F] long (0 where not opened) and ok [B, ch, m·F]
        bool. At most `cap` entries of a query are ok, each a distinct earlier token of its document."""
        gi = self.index
        mem = self.members()
        B, G, Fm = mem.shape
        ids = self.ids[:, s0 - self.row0:s1 - self.row0].long()
        off = self.off[:, s0 - self.row0:s1 - self.row0].to(torch.int32)
        ch, m = ids.shape[1], ids.shape[2]
        qp = torch.arange(s0, s1, device=ids.device)
        live = off < self.cap
        ds = gi.doc_start[:, s0:s1][..., None].expand(B, ch, m)
        hi = (qp - int(raw_window))[None, :, None].expand(B, ch, m)
        cnt = torch.where(live, counts(gi, ids, ds, hi), torch.zeros_like(off))
        n = torch.minimum(cnt, self.cap - off).clamp(min=0)
        r0 = counts(gi, ids, torch.zeros_like(ds), ds - 1)                                # members before the document
        r = torch.arange(Fm, device=ids.device)
        take = r0[..., None] + r                                                          # [B, ch, m, F]
        ok = (r < n[..., None]) & (take < Fm)
        flat = ids[..., None] * Fm + take.clamp(max=Fm - 1)
        pos = mem.reshape(B, -1).gather(1, flat.reshape(B, -1)).view(B, ch, m, Fm)
        ok = ok & (pos >= 0)
        pos = torch.where(ok, pos, torch.zeros_like(pos))
        return pos.reshape(B, ch, m * Fm), ok.reshape(B, ch, m * Fm)


# Flex attention addresses the buffers a mask_mod captures with 32-bit offsets: a buffer of 2³¹ elements or more is
# read out of bounds (illegal memory access; E33p notes at 128k: T is 131,072 × ~21,800 ≈ 2.9e9, 2026-11-07).
_FLEX_MAX_NUMEL = 2**31 - 1


def _split_queries(T: torch.Tensor) -> tuple[list[torch.Tensor], int]:
    """T [B, S, G] as query chunks of fewer than 2³¹ elements each (fresh tensors), and the chunk length. One chunk
    (T itself, no copy) up to that size, i.e. at every length the models train on; beyond it, one extra copy of T."""
    B, S, G = (int(x) for x in T.shape)
    if B * S * G <= _FLEX_MAX_NUMEL:
        return [T], S
    qc = max(1, _FLEX_MAX_NUMEL // (B * G))
    return [T[:, i:i + qc].clone() for i in range(0, S, qc)], qc


def open_term(T: torch.Tensor, codes: torch.Tensor, *, cap: int, raw_window: int) -> Callable:
    """mask_mod piece (b, q, j) → bool for raw key j (already clamped to [0, S)): j opened for query q. It does not
    test documents: every caller ANDs it with its own causal same-document test (LSA buckets may mix documents)."""
    n_layers = int(codes.shape[0])
    G = int(T.shape[-1])
    cap_i, rw = int(cap), int(raw_window)
    parts, qc = _split_queries(T)

    def t_at(b, q, g):
        if len(parts) == 1:
            return parts[0][b, q, g]
        out = None
        for ci, P in enumerate(parts):  # static: one lookup per chunk, the query's own chunk wins
            v = P[b, torch.clamp(q - ci * qc, 0, int(P.shape[1]) - 1), g]
            out = v if out is None else torch.where(q >= ci * qc, v, out)
        return out

    def opened(b, q, j):
        hit = None
        for li in range(n_layers):
            c = codes[li, b, j]
            g = torch.div(c, ORD_MOD, rounding_mode="floor")
            o = c - g * ORD_MOD
            h = (c >= 0) & (t_at(b, q, torch.clamp(g, 0, G - 1)).to(torch.int32) + o < cap_i)
            hit = h if hit is None else hit | h
        return hit & (q - j >= rw)

    return opened


def dense_open(T: torch.Tensor, codes: torch.Tensor, *, cap: int, raw_window: int,
               q_idx: Optional[torch.Tensor] = None, doc: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B, Q, S] bool — the same rule as `open_term` for the query rows `q_idx` (default: all); with `doc` [B, S],
    also the same-document test, i.e. exactly the opened tokens the read uses."""
    B, S = codes.shape[1], codes.shape[2]
    G = T.shape[-1]
    if q_idx is None:
        q_idx = torch.arange(S, device=T.device)
    Tq = T[:, q_idx].to(torch.int32)                                                    # [B, Q, G]
    Q = Tq.shape[1]
    hit = torch.zeros(B, Q, S, dtype=torch.bool, device=T.device)
    for li in range(codes.shape[0]):
        c = codes[li].to(torch.int64)
        g = torch.div(c, ORD_MOD, rounding_mode="floor")
        o = c - g * ORD_MOD
        tg = Tq.gather(2, g.clamp(0, G - 1)[:, None, :].expand(B, Q, S))
        hit |= (c >= 0)[:, None, :] & (tg + o[:, None, :] < cap)
    j = torch.arange(S, device=T.device)
    hit = hit & ((q_idx[:, None] - j[None, :]) >= raw_window)[None]
    if doc is not None:
        hit = hit & (doc[:, q_idx][:, :, None] == doc[:, None, :]) & (doc[:, q_idx] >= 0)[:, :, None]
    return hit


# --------------------------------------------------------------------------------------- E33p notes and pages
@dataclass
class NoteOpen:
    """Per-forward state of the note-indexed read (built once after the writer, `PerceiverARLM._run_layers`)."""

    assign: torch.Tensor          # [B, n_w, K, W] fp32 (grad) — A[t, n], last latent round, heads averaged
    page_tok: torch.Tensor        # [n_w, W] long — row index of each page slot (clamped)
    groups: GroupIndex            # what opens: the notes, or the pages
    mode: str                     # "none" | "notes" | "pages"
    k: int                        # groups chosen per query
    cap: int
    per_page: int                 # notes per page (K)
    # E33t, the catalogue read (defaults = E33p)
    read: str = "gist"            # "index": the notes only choose what opens
    score: str = "sum_softmax"    # "max_cos": cosine, best query head
    levels: int = 1               # 2: pages first, then their notes
    keep_pages: int = 16
    new_per_pass: bool = False    # a pass may not open notes the query opened earlier
    labels: Optional[torch.Tensor] = None   # [B, n_w, W] filed note per page slot (balanced filing), −1 = none
    pick: Optional[torch.Tensor] = None     # [B, n_w, W] valid page slots


def balance_assign(A: torch.Tensor, valid: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """E33t balanced filing, soft part: Sinkhorn iterations on each page's assignment so every note receives about the
    same mass (SwAV's equipartition). A [B, n_w, K, W] (a distribution over the K notes per token), valid [B, n_w, W]
    bool → Â [B, n_w, K, W]: per page, each note's mass → n_valid / K, each valid token's mass over notes → 1 (exact
    after the last step); invalid tokens 0. Differentiable."""
    m = valid[:, :, None, :].to(A.dtype)
    a = A * m
    K = A.shape[2]
    target = m.sum(-1, keepdim=True) / K                                                 # [B, n_w, 1, 1]
    for _ in range(int(iters)):
        a = a * (target / a.sum(-1, keepdim=True).clamp_min(eps))
        a = a / a.sum(2, keepdim=True).clamp_min(eps) * m
    return a


@torch.no_grad()
def file_capped(A_hat: torch.Tensor, valid: torch.Tensor, cap: int = 16) -> torch.Tensor:
    """E33t balanced filing, hard part: each valid page slot is filed under one note, at most `cap` per note.
    Tokens propose their notes best first (by Â); a note accepts its proposers best first until full; a rejected token
    proposes its next note in the next round. With K · cap ≥ W every token is filed within K rounds (a token rejected
    by all K notes would need all K full). A_hat [B, n_w, K, W], valid [B, n_w, W] → labels [B, n_w, W] long, −1 =
    not valid."""
    B, n_w, K, W = A_hat.shape
    a = A_hat.detach().float().permute(0, 1, 3, 2).reshape(-1, W, K)                    # [P, W, K]
    v = valid.reshape(-1, W).bool()
    P = a.shape[0]
    dev = a.device
    labels = torch.full((P, W), -1, dtype=torch.long, device=dev)
    fill = torch.zeros(P, K + 1, dtype=torch.long, device=dev)
    order = a.argsort(dim=-1, descending=True)                                           # [P, W, K]
    idx = torch.arange(W, device=dev)[None].expand(P, W)
    for r in range(K):
        todo = v & (labels < 0)
        if not bool(todo.any()):
            break
        prop = order[..., r]                                                             # [P, W]
        sc = a.gather(-1, prop[..., None]).squeeze(-1).clamp(0.0, 1.0)
        key = torch.where(todo, prop, torch.full_like(prop, K))                          # K = not proposing
        srt = torch.argsort(key.float() * 2.0 - sc, dim=-1)                              # note asc, score desc
        sk = key.gather(-1, srt)
        new = torch.ones_like(sk, dtype=torch.bool)
        new[:, 1:] = sk[:, 1:] != sk[:, :-1]
        start = torch.cummax(torch.where(new, idx, torch.zeros_like(idx)), dim=-1).values
        rank = torch.empty_like(idx).scatter_(-1, srt, idx - start)                      # rank among its note's proposers
        room = (cap - fill).gather(-1, key)
        accept = todo & (rank < room)
        labels = torch.where(accept, prop, labels)
        fill.scatter_add_(-1, torch.where(accept, prop, torch.full_like(prop, K)), accept.long())
    return labels.view(B, n_w, W)


def page_keys(k: torch.Tensor, page_tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """E33t two-level search: π_p = mean of the page's valid tokens' keys. k [B, S, 1, dh], page_tok [n_w, W],
    valid [B, n_w, W] → [B, n_w, dh] fp32 (no gradient needed: the choice is made without one)."""
    B = k.shape[0]
    n_w, W = page_tok.shape
    kp = k[:, page_tok.reshape(-1), 0].detach().float().view(B, n_w, W, -1)
    m = valid.to(kp.dtype)[..., None]
    return (kp * m).sum(2) / m.sum(2).clamp_min(1.0)


def tied_note_keys(assign: torch.Tensor, k: torch.Tensor, page_tok: torch.Tensor) -> torch.Tensor:
    """κ_n = Σ_t A[t, n] k_t / Σ_t A[t, n] over the note's page. assign [B, n_w, K, W], k [B, S, g, dh] → [B, n_w·K, g, dh]
    (fp32, outside autocast; a note with no mass gets κ = 0)."""
    B, n_w, K, W = assign.shape
    g, dh = k.shape[2], k.shape[3]
    with torch.autocast(device_type=k.device.type, enabled=False):
        kp = k[:, page_tok.reshape(-1)].float().view(B, n_w, W, g, dh)
        a = assign.float()
        num = torch.einsum("bnkw,bnwgd->bnkgd", a, kp)
        den = a.sum(-1)[..., None, None].clamp_min(1e-6)
        return (num / den).reshape(B, n_w * K, g, dh)


def note_groups(assign: torch.Tensor, pick: torch.Tensor, page_starts: list[int], *, S: int, doc_start: torch.Tensor,
                slot_close: torch.Tensor, slot_doc: torch.Tensor, mode: str,
                labels: Optional[torch.Tensor] = None) -> GroupIndex:
    """Notes (member = argmax_k A on picked tokens, or the given filing `labels` [B, n_w, W]) or pages (member =
    picked) as groups. slot_* [B, n_w·K]."""
    B, n_w, K, W = assign.shape
    if mode == "pages":
        member = torch.where(pick, torch.zeros_like(pick, dtype=torch.long), torch.full_like(pick, -1, dtype=torch.long))
        g_close, g_doc, per = slot_close[:, ::K], slot_doc[:, ::K], 1
    else:
        best = labels if labels is not None else assign.detach().argmax(dim=2)          # [B, n_w, W]
        member = torch.where(pick, best, torch.full_like(best, -1))
        g_close, g_doc, per = slot_close, slot_doc, K
    return groups_from_spans(member, page_starts, S=S, per_span=per, doc_start=doc_start,
                             g_close=g_close.contiguous(), g_doc=g_doc.contiguous())


@torch.no_grad()
def note_open_table(q: torch.Tensor, kappa: torch.Tensor, no: NoteOpen, *, doc: torch.Tensor, slot_close: torch.Tensor,
                    slot_doc: torch.Tensor, raw_window: int, chunk: int = 0, exclude: Optional[torch.Tensor] = None,
                    page_key: Optional[torch.Tensor] = None, impl: str = "table", q_offset: int = 0,
                    members: Optional[torch.Tensor] = None):
    """E33p choice for every query: s_i(n) = Σ_h softmax_n(q_{i,h}·κ_n / √dh) over the notes visible to i (closed rule),
    eligible = visible ∧ a member older than i's window; notes: top-k notes; pages: the page score is the sum of its
    notes' scores, top-k eligible pages. q [B, S, h, dh], κ [B, C, 1, dh].

    E33t: `no.score = "max_cos"` scores s_i(n) = max_h cos(q_{i,h}, κ_n); `exclude` [B, S, G] bool removes the groups a
    query opened in earlier passes; `page_key` [B, n_w, dh] with `no.levels = 2` keeps, per query, the
    `no.keep_pages` best visible pages holding an eligible note (same score rule on the page keys) and lets only their
    notes compete. `impl = "list"` (E33t stage C, evaluation): the choice as a ChoiceList instead of the table.

    `q_offset` (cached decoding, nn/incremental.py; list only): q holds the queries [q_offset, q_offset + Sq) of a row
    of doc.shape[1] tokens; `members` passes the index's members table along (built once per index)."""
    B, Sq, h, dh = q.shape
    q0 = int(q_offset)
    if (q0 or Sq != doc.shape[1]) and (impl != "list" or exclude is not None):
        raise ValueError("a query range is for the list read without pass exclusion")
    S = q0 + Sq
    C = kappa.shape[1]
    K = no.per_page
    gi = no.groups
    dev = q.device
    G = gi.n_groups
    if chunk <= 0:
        chunk = max(128, min(1024, (1 << 25) // max(B * h * C, 1) // 128 * 128))
    kb = kappa[:, :, 0].float()
    sc = dh ** -0.5
    cos = no.score == "max_cos"
    if cos:
        kb = F.normalize(kb, dim=-1)
    two = no.levels == 2 and page_key is not None and no.mode == "notes"
    pk = F.normalize(page_key.float(), dim=-1) if two else None
    tab = (ChoiceList(gi, B, Sq, cap=no.cap, m=min(int(no.k), G), row0=q0, members=members) if impl == "list"
           else OpenTable(gi, B, S, cap=no.cap))
    n_vis_ok = (slot_doc >= 0) & (slot_close >= 0)
    for s0 in range(q0, S, chunk):
        s1 = min(S, s0 + chunk)
        qp = torch.arange(s0, s1, device=dev)
        dq = doc[:, s0:s1]
        vis = (n_vis_ok[:, None, :] & (slot_doc[:, None, :] == dq[..., None]) & (dq[..., None] >= 0)
               & (slot_close[:, None, :] <= qp[None, :, None]))                          # [B, ch, C]
        qc = q[:, s0 - q0:s1 - q0].float()
        if cos:
            qc = F.normalize(qc, dim=-1)
            s_note = torch.einsum("bshd,bcd->bshc", qc, kb).amax(2)                       # [B, ch, C]
        else:
            lg = torch.einsum("bshd,bcd->bshc", qc, kb) * sc
            p = torch.softmax(lg.masked_fill(~vis[:, :, None], float("-inf")), dim=-1)
            s_note = torch.nan_to_num(p, nan=0.0).sum(2)                                  # [B, ch, C]
        if no.mode == "pages":
            score = s_note.view(B, s1 - s0, C // K, K).sum(-1)
            gvis = vis.view(B, s1 - s0, C // K, K).any(-1)
        else:
            score, gvis = s_note, vis
        cand = gvis & (gi.g_minpos[:, None, :] <= (qp - raw_window)[None, :, None])
        if exclude is not None:
            cand = cand & ~exclude[:, s0:s1]
        if two:
            n_w = C // K
            ps = torch.einsum("bshd,bpd->bshp", qc, pk).amax(2)                            # [B, ch, n_w]
            pok = cand.view(B, s1 - s0, n_w, K).any(-1)
            kp = min(int(no.keep_pages), n_w)
            top = ps.masked_fill(~pok, float("-inf")).topk(kp, dim=-1)
            keep = torch.zeros_like(pok).scatter_(-1, top.indices, top.values > float("-inf"))
            cand = cand & keep.repeat_interleave(K, dim=-1)
        ids, off = choose(score, cand, gi, qp, m=no.k, cap=no.cap, raw_window=raw_window)
        tab.add(s0, ids, off)
    return tab


def _masked_rank_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over rows with ≥ 3 masked entries of the Spearman correlation of a and b on the masked entries. [Q, G]."""
    big = torch.finfo(torch.float32).max
    def ranks(x):
        x = x.float().masked_fill(~mask, -big)
        return x.argsort(-1).argsort(-1).float()
    ra, rb = ranks(a), ranks(b)
    n = mask.sum(-1, keepdim=True).clamp_min(1).float()
    m = mask.float()
    ma, mb = (ra * m).sum(-1, keepdim=True) / n, (rb * m).sum(-1, keepdim=True) / n
    da, db = (ra - ma) * m, (rb - mb) * m
    c = (da * db).sum(-1) / ((da.pow(2).sum(-1) * db.pow(2).sum(-1)).sqrt().clamp_min(1e-9))
    ok = mask.sum(-1) >= 3
    return c[ok].mean() if bool(ok.any()) else torch.zeros((), device=a.device)


def index_loss(q: torch.Tensor, k: torch.Tensor, sink_k: Optional[torch.Tensor], kappa: torch.Tensor,
               opened: "OpenTable", *, doc: torch.Tensor, raw_window: int, n_queries: int, n_neg: int,
               log_tau: torch.Tensor, seen: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> tuple[torch.Tensor, dict]:
    """E33t index loss for one pass of the catalogue read: the notes' scores learn to predict where the read's
    attention lands. For `n_queries` query positions per row (evenly spaced from the window on): the read's softmax
    over sinks ‖ its window ‖ its opened tokens is recomputed (fp32, detached: the target); target mass of a chosen
    note = the probability on its opened tokens; of a negative (one of `n_neg` sampled choosable, unchosen notes, not in
    `seen`) = the probability its eligible members would get with the same normaliser; both averaged over heads and
    renormalised over the set. Prediction = softmax over the set of τ · max_h cos(q_{i,h}, κ_n), q detached, κ with
    gradient, τ = exp(log_tau). Loss = KL(target ‖ prediction), mean over queries with ≥ 2 notes in the set.
    q [B, S, h, dh], k [B, S, 1, dh], sink_k [n, dh] or None, κ [B, G, 1, dh], doc [B, S] (−1 = pad),
    seen [B, S, G] bool. Returns (loss, stats) with stats `kl`, `corr` (Spearman of score and target mass on the
    chosen notes) and `set` (notes per query)."""
    B, S, h, dh = q.shape
    gi = opened.index
    G = gi.n_groups
    cap = opened.cap
    dev = q.device
    rw = int(raw_window)
    lo = min(rw, S - 1)
    rows = torch.linspace(lo, S - 1, max(1, min(int(n_queries), S - lo)), device=dev).round().long().unique()
    Q = rows.numel()
    scale = dh ** -0.5
    losses, corrs, sizes = [], [], []
    tau = log_tau.exp()
    j = torch.arange(S, device=dev)
    with torch.autocast(device_type=dev.type, enabled=False):
        for b in range(B):
            d = doc[b]
            ok_q = d[rows] >= 0
            qb = q[b, rows].detach().float()                                             # [Q, h, dh]
            lg = torch.einsum("qhd,sd->hqs", qb, k[b, :, 0].detach().float()) * scale     # [h, Q, S]
            causal = (j[None] <= rows[:, None]) & (d[None] == d[rows][:, None]) & ok_q[:, None]
            win = causal & ((rows[:, None] - j[None]) < rw)
            op = dense_open(opened.T[b:b + 1], gi.codes[:, b:b + 1], cap=cap, raw_window=rw, q_idx=rows,
                            doc=doc[b:b + 1])[0] & causal & ~win
            allowed = win | op
            parts = [lg.masked_fill(~allowed[None], float("-inf"))]
            if sink_k is not None and sink_k.shape[0] > 0:
                parts.append(torch.einsum("qhd,nd->hqn", qb, sink_k.detach().float()) * scale)
            lse = torch.logsumexp(torch.cat(parts, -1), dim=-1)                           # [h, Q]
            pw = torch.exp(lg - lse[..., None]).mean(0)                                   # [Q, S] would-be, head mean
            elig = causal & ~win
            m_open = torch.zeros(Q, G + 1, device=dev)
            m_elig = torch.zeros(Q, G + 1, device=dev)
            for li in range(gi.codes.shape[0]):
                c = gi.codes[li, b]
                gid = torch.where(c >= 0, torch.div(c, ORD_MOD, rounding_mode="floor"), torch.full_like(c, G)).long()
                gq = gid[None].expand(Q, S)
                m_open.scatter_add_(1, torch.where(op, gq, torch.full_like(gq, G)), pw * op)
                m_elig.scatter_add_(1, torch.where(elig, gq, torch.full_like(gq, G)), pw * elig)
            m_open, m_elig = m_open[:, :G], m_elig[:, :G]
            chosen = (opened.T[b, rows].to(torch.int32) < cap) & (m_open > 0)
            neg_ok = (m_elig > 0) & ~chosen
            if seen is not None:
                neg_ok = neg_ok & ~seen[b, rows]
            neg = torch.zeros_like(chosen)
            if n_neg > 0 and bool(neg_ok.any()):
                r = torch.rand(Q, G, device=dev, generator=generator).masked_fill(~neg_ok, -1.0)
                top = r.topk(min(int(n_neg), G), dim=-1)
                neg.scatter_(1, top.indices, top.values >= 0)
            in_set = chosen | neg
            t = torch.where(chosen, m_open, torch.where(neg, m_elig, torch.zeros_like(m_open)))
            t = t / t.sum(-1, keepdim=True).clamp_min(1e-12)
            qn = F.normalize(qb, dim=-1)
            kn = F.normalize(kappa[b, :, 0].float(), dim=-1)                              # grad → κ
            s = torch.einsum("qhd,gd->qhg", qn, kn).amax(1)                              # [Q, G]
            logp = torch.log_softmax((tau * s).masked_fill(~in_set, float("-inf")), dim=-1)
            use = ok_q & (in_set.sum(-1) >= 2)
            if bool(use.any()):
                kl = (t * (torch.log(t.clamp_min(1e-12)) - logp)).masked_fill(~in_set, 0.0).sum(-1)
                losses.append(kl[use].mean())
                corrs.append(_masked_rank_corr(s.detach(), m_open, chosen & use[:, None]))
                sizes.append(in_set[use].sum(-1).float().mean())
    if not losses:
        z = (kappa.sum() + log_tau) * 0.0
        return z, {"kl": z.detach(), "corr": z.detach(), "set": z.detach()}
    loss = torch.stack(losses).mean()
    return loss, {"kl": loss.detach(), "corr": torch.stack(corrs).mean(), "set": torch.stack(sizes).mean()}

"""E37 — concept memory for the reading layer: encoder → reasoner → decoder read.

Spec: docs/experiments_specs/ahead/E37_concept_memory.md.

The three blocks, inside one causal language model (`PerceiverARLM`, `par_global_read = "concept"`):

1. **Encoder (memory).** The layers below the reading layer turn tokens into states x, keys k and values v. Every
   closed block of `block` keys is grouped into `per_block` **concepts** by content: a balanced soft k-means on the
   unit keys (Sinkhorn-balanced assignment, a few rounds, no gradient), each key filed under its best concept. A
   concept keeps the **links to its member tokens** (their positions, for exact reads) and **sufficient statistics**
   of them: count n, mean key μ_k, mean value μ_v, isotropic key variance σ², mean state z. Statistics add up, so
   concepts of concepts are exact sums (the route to long pages and 10M tokens).
2. **Reasoner.** A small looped transformer over the concept states z (`ConceptReasoner`): a concept attends to the
   concepts of its own and earlier blocks of the same document, so it can connect ideas across the document. Its
   output is added to each concept's value (zero at the start: the read begins as the plain summary read).
3. **Decoder read.** Each query reads one softmax over: its document's first `sinks` tokens; the raw tokens of its
   own block and the block before (a block-aligned window); the member tokens of the concepts with the largest
   estimated attention mass (exact; best query head, best concept first until `top_k` tokens); and **one summary
   entry for every other concept** of older blocks. A summary's logit is the second-order estimate of the
   log-sum-exp of its members' logits (the exponential kernel's moment-generating function):
       log n + q·μ_k/√d + |q|² σ² / (2d),
   and its value is μ_v (+ the reasoner's update). The same estimate picks what to open, so the largest
   contributions are read exactly and the rest coarsely (a fast-multipole split: near field exact, far field by
   moments). Every past token of the document is represented exactly once: inside the window, opened, or inside one
   summary. With pure concepts (identical member keys) the summary read equals full attention.

Related work (2026-11-07 scout): SPLA (arXiv 2601.22379) estimates positional blocks' mass from mean and diagonal
covariance and reads unread blocks as residual linear attention; RetroInfer (2505.02922) estimates unread clusters as
centroid score × size, training-free; LSA (2610.42147) clusters keys by Ward merges and picks by cosine, no summaries
in the paper. Here: balanced content clusters, natively trained, a softmax summary per cluster with gradient to the
members, choice by estimated mass, and a reasoner over the clusters.

Shapes at the 100M scale, 16k row: S = 16,384; blocks of 512 → 32 blocks, 64 concepts each, C = 2,048 summaries;
k, v [B, S, 1, 64]; z [B, C, 896].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.exact_read import GroupIndex, OpenTable, _split_queries, choose, doc_starts, groups_from_spans


# ------------------------------------------------------------------------------------------------ encoder
@torch.no_grad()
def farthest_points(keys: torch.Tensor, valid: torch.Tensor, n: int) -> torch.Tensor:
    """n unit keys per block by farthest-point sampling on cosine similarity. keys [N, L, d] unit, valid [N, L] →
    [N, n, d]. Blocks with fewer valid keys than n repeat their last pick."""
    N, L, d = keys.shape
    big = 2.0
    first = valid.float().argmax(1)                                                    # first valid key
    idx = [first]
    maxsim = torch.einsum("nld,nd->nl", keys, keys[torch.arange(N), first]).masked_fill(~valid, big)
    for _ in range(n - 1):
        nxt = maxsim.argmin(1)
        idx.append(nxt)
        maxsim = torch.maximum(maxsim, torch.einsum("nld,nd->nl", keys, keys[torch.arange(N), nxt]))
    pick = torch.stack(idx, 1)                                                         # [N, n]
    return keys.gather(1, pick[..., None].expand(N, n, d))


@torch.no_grad()
def sinkhorn_labels(keys: torch.Tensor, valid: torch.Tensor, n_concepts: int, *, iters: int = 3,
                    tau: float = 0.05, rounds: int = 4) -> torch.Tensor:
    """Balanced soft k-means on unit keys, per block. keys [N, L, d] fp32 unit rows, valid [N, L] bool →
    labels [N, L] long (the concept of each key, −1 where not valid).

    Centres start at n_concepts keys picked by farthest-point sampling (the first valid key, then each time the key
    least similar to those picked; deterministic and content-based, so interleaved topics separate). Each round:
    logits = keys · centres / tau, a log-domain Sinkhorn plan that gives every key mass 1 and every concept an equal
    share of the block's valid keys, then centres = normalised plan-weighted key sums. Labels = each key's best
    concept under the last plan (roughly balanced: the plan is, its argmax nearly so)."""
    N, L, d = keys.shape
    C = int(n_concepts)
    vf = valid.float()
    cent = farthest_points(keys, valid, C)
    seed = cent
    neg = torch.finfo(torch.float32).min / 4
    n_valid = vf.sum(1, keepdim=True).clamp_min(1.0)                                  # [N, 1]
    log_col = torch.log(n_valid / C)                                                   # each concept's share
    logp = None
    for _ in range(max(1, iters)):
        logits = torch.einsum("nld,ncd->nlc", keys, cent) / tau
        logp = logits.masked_fill(~valid[..., None], neg)
        for _ in range(rounds):
            logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)                  # each key: mass 1
            logp = logp.masked_fill(~valid[..., None], neg)
            logp = logp - torch.logsumexp(logp, dim=1, keepdim=True) + log_col[..., None]  # each concept: L/C
        logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)
        p = logp.exp() * vf[..., None]
        cent = F.normalize(torch.einsum("nlc,nld->ncd", p, keys) + 1e-12 * seed, dim=-1)
    labels = logp.argmax(-1)
    return torch.where(valid, labels, torch.full_like(labels, -1))


@dataclass
class ConceptIndex:
    """One reading-layer call's concepts. C = n_blocks · per_block, concept c = block · per_block + slot."""

    groups: GroupIndex           # member links (exact reads), nn.exact_read
    centroid: torch.Tensor       # [B, C, d] unit mean key (the choice rule; no gradient)
    n: torch.Tensor              # [B, C] members (0 = empty)
    mu_k: torch.Tensor           # [B, C, d] mean key (with gradient to the members' keys)
    var: torch.Tensor            # [B, C] mean per-dimension key variance of the members
    mu_v: torch.Tensor           # [B, C, d] mean value
    z: Optional[torch.Tensor]    # [B, C, H] mean state of the members (the reasoner's input), or None
    block_of: torch.Tensor       # [C] long, the block of each concept
    c_doc: torch.Tensor          # [B, C] long, the document of the concept's block (−1: mixed, padding or empty)
    block: int
    per_block: int

    @property
    def n_concepts(self) -> int:
        return int(self.n.shape[1])


def build_concepts(k: torch.Tensor, v: torch.Tensor, x: Optional[torch.Tensor], doc: torch.Tensor, sink: torch.Tensor,
                   *, block: int, per_block: int, iters: int = 3, tau: float = 0.05) -> Optional[ConceptIndex]:
    """Concepts of every block some query sees as old (block index ≤ last query's block − 2).

    k, v [B, S, 1, d] (the reading layer's keys after their norm, no position code), x [B, S, H] or None, doc [B, S]
    (−1 = padding), sink [B, S] bool. The grouping carries no gradient; the statistics do (through the members' k, v
    and x)."""
    B, S = doc.shape
    d = k.shape[-1]
    nb = (S - 1) // block - 1  # blocks with at least one query two blocks later
    if nb <= 0:
        return None
    L = nb * block
    kk = k[:, :L, 0].float()                                                           # [B, L, d]
    vv = v[:, :L, 0].float()
    valid = ((doc >= 0) & ~sink)[:, :L]
    unit = F.normalize(kk.detach(), dim=-1).reshape(B * nb, block, d)
    labels = sinkhorn_labels(unit, valid.reshape(B * nb, block), per_block, iters=iters, tau=tau)
    labels = labels.view(B, nb, block)
    onehot = (labels[..., None] == torch.arange(per_block, device=k.device)).float()  # [B, nb, block, K]
    n = onehot.sum(2)                                                                  # [B, nb, K]
    inv = 1.0 / n.clamp_min(1.0)
    kb = kk.reshape(B, nb, block, d)
    mu_k = torch.einsum("bnlk,bnld->bnkd", onehot, kb) * inv[..., None]
    sq = torch.einsum("bnlk,bnl->bnk", onehot, (kb ** 2).sum(-1)) * inv
    var = (sq - (mu_k ** 2).sum(-1)).clamp_min(0.0) / d
    mu_v = torch.einsum("bnlk,bnld->bnkd", onehot, vv.reshape(B, nb, block, d)) * inv[..., None]
    z = None
    if x is not None:
        z = torch.einsum("bnlk,bnlh->bnkh", onehot, x[:, :L].float().reshape(B, nb, block, -1)) * inv[..., None]
        z = z.reshape(B, nb * per_block, -1).to(x.dtype)
    # the document of each block, when the whole block is one document (summaries never mix documents)
    db = doc[:, :L].reshape(B, nb, block)
    one_doc = (db == db[..., :1]).all(-1) & (db[..., 0] >= 0)
    blk_doc = torch.where(one_doc, db[..., 0], torch.full_like(db[..., 0], -1))
    c_doc = blk_doc.repeat_interleave(per_block, dim=1)
    n_flat = n.reshape(B, nb * per_block)
    c_doc = torch.where(n_flat > 0, c_doc, torch.full_like(c_doc, -1))
    block_of = torch.arange(nb, device=k.device).repeat_interleave(per_block)
    ds = doc_starts(doc)
    # member links for the exact reads; a concept may be opened by queries two blocks later (block-aligned window)
    member = labels
    g_close = ((block_of + 2) * block)[None].expand(B, -1).contiguous()
    gi = groups_from_spans(member, [b * block for b in range(nb)], S=S, per_span=per_block, doc_start=ds,
                           g_close=g_close, g_doc=torch.zeros_like(g_close), any_doc=True, sink=sink)
    mu_k = mu_k.reshape(B, nb * per_block, d)
    return ConceptIndex(groups=gi, centroid=F.normalize(mu_k.detach(), dim=-1), n=n_flat,
                        mu_k=mu_k, var=var.reshape(B, nb * per_block), mu_v=mu_v.reshape(B, nb * per_block, d),
                        z=z, block_of=block_of, c_doc=c_doc, block=block, per_block=per_block)


def summary_logit(q: torch.Tensor, cidx: ConceptIndex, *, variance: bool = True) -> torch.Tensor:
    """Each concept's estimated log attention mass for each query and head, the logit of its summary entry:
    log n + q·μ_k/√d + |q|² σ² / (2d). q [B, s, h, d] → [B, s, h, C] fp32 (−inf for empty concepts)."""
    d = q.shape[-1]
    qf = q.float()
    lg = torch.einsum("bshd,bcd->bshc", qf, cidx.mu_k.float()) / math.sqrt(d)
    lg = lg + torch.where(cidx.n > 0, cidx.n.clamp_min(1.0).log(), torch.full_like(cidx.n, -math.inf))[:, None, None]
    if variance:
        lg = lg + ((qf ** 2).sum(-1, keepdim=True) / (2 * d)) * cidx.var[:, None, None]
    return lg


@torch.no_grad()
def concept_open_table(q: torch.Tensor, cidx: ConceptIndex, *, top_k: int, width: int, variance: bool = True,
                       chunk: int = 0) -> OpenTable:
    """Each query opens the concepts with the largest estimated attention mass (its summary logit, best query head),
    concepts of blocks two or more before its own, best first, until `top_k` member tokens (nn.exact_read.choose:
    members of the query's document only). The opened concepts leave the summary read. q [B, S, h, d]."""
    B, S, h, d = q.shape
    gi = cidx.groups
    C = cidx.n_concepts
    if chunk <= 0:
        chunk = max(128, min(1024, (1 << 25) // max(B * h * C, 1) // 128 * 128))
    tab = OpenTable(gi, B, S, cap=top_k)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qp = torch.arange(s0, s1, device=q.device)
        score = summary_logit(q[:, s0:s1], cidx, variance=variance).amax(2)              # [B, ch, C]
        visible = (gi.g_close[:, None, :] <= qp[None, :, None]) & torch.isfinite(score)
        ids, off = choose(torch.nan_to_num(score, neginf=-1e30), visible, gi, qp, m=width, cap=top_k,
                          raw_window=cidx.block)
        tab.add(s0, ids, off)
    return tab


# ------------------------------------------------------------------------------------------------ reasoner
class ConceptReasoner(nn.Module):
    """A few tied pre-norm transformer layers over the concept states, run `loops` times with the input added back
    each loop (input injection). A concept attends to concepts of its own and earlier blocks in its document.
    z [B, C, H] → value update [B, C, d] (zero at initialisation)."""

    def __init__(self, hidden: int, d_value: int, *, dim: int, layers: int, heads: int, loops: int):
        super().__init__()
        self.loops = max(1, int(loops))
        self.inp = nn.Linear(hidden, dim, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.RMSNorm(dim), qkv=nn.Linear(dim, 3 * dim, bias=False), o=nn.Linear(dim, dim, bias=False),
                n2=nn.RMSNorm(dim), up=nn.Linear(dim, 4 * dim, bias=False), down=nn.Linear(4 * dim, dim, bias=False))))
        self.heads = int(heads)
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, d_value, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, z: torch.Tensor, block_of: torch.Tensor, c_doc: torch.Tensor,
                loops: Optional[int] = None) -> torch.Tensor:
        B, C, _ = z.shape
        u = self.inp(z)
        ok = (c_doc >= 0)
        mask = ((block_of[None, :, None] >= block_of[None, None, :]) & (c_doc[:, :, None] == c_doc[:, None, :])
                & ok[:, :, None] & ok[:, None, :])
        mask = mask | torch.eye(C, dtype=torch.bool, device=z.device)[None]            # no empty rows
        h = torch.zeros_like(u)
        for _ in range(int(loops or self.loops)):
            h = h + u                                                                  # input injection
            for blk in self.blocks:
                a = blk["qkv"](blk["n1"](h)).view(B, C, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
                o = F.scaled_dot_product_attention(a[0], a[1], a[2], attn_mask=mask[:, None])
                h = h + blk["o"](o.transpose(1, 2).reshape(B, C, -1))
                h = h + blk["down"](F.silu(blk["up"](blk["n2"](h))))
        return self.out(self.norm(h))


# ------------------------------------------------------------------------------------------------ decoder read
def _chosen_lookup(opened: Optional[OpenTable]):
    """(b, q, c) → bool, c chosen by q (opened at least partly). None when nothing is opened."""
    if opened is None:
        return None
    parts, qc = _split_queries(opened.T)
    cap = int(opened.cap)
    G = int(opened.T.shape[-1])

    def chosen(b, q, c):
        c = torch.clamp(c, 0, G - 1)
        if len(parts) == 1:
            t = parts[0][b, q, c]
        else:
            t = None
            for ci, P in enumerate(parts):
                vv = P[b, torch.clamp(q - ci * qc, 0, int(P.shape[1]) - 1), c]
                t = vv if t is None else torch.where(q >= ci * qc, vv, t)
        return t.to(torch.int32) < cap

    return chosen


def attend_concepts(q, k, v, *, cidx: Optional[ConceptIndex], dv: Optional[torch.Tensor], doc, key_valid, backend,
                    opened: Optional[OpenTable], sink: torch.Tensor, ds: torch.Tensor, n_sinks: int,
                    summaries: bool = True, variance: bool = True):
    """The decoder's read: sinks ‖ block-aligned window ‖ opened members ‖ concept summaries, one softmax.
    q [B,S,h,d], k/v [B,S,1,d], doc [B,S] (−1 = pad), sink [B,S] bool, ds [B,S] document starts. Returns [B,S,h,d]."""
    from nn.exact_read import dense_open, open_term
    from nn.perceiver_ar_lm import _band_block_mask, _get_flex

    B, S, h, d = q.shape
    g = k.shape[2]
    blk = cidx.block if cidx is not None else 512
    use_sum = summaries and cidx is not None
    C = cidx.n_concepts if use_sum else 0
    if use_sum:
        vs = cidx.mu_v if dv is None else cidx.mu_v + dv.to(cidx.mu_v.dtype)
        K = torch.cat([k, cidx.mu_k[:, :, None].to(k.dtype)], dim=1)
        V = torch.cat([v, vs[:, :, None].to(v.dtype)], dim=1)
        logn = torch.where(cidx.n > 0, cidx.n.clamp_min(1.0).log(), torch.zeros_like(cidx.n)).contiguous()
        vhalf = (0.5 * cidx.var).contiguous() if variance else torch.zeros_like(cidx.var)
        qn2 = (q.float() ** 2).sum(-1).transpose(1, 2).contiguous() / d                # [B, h, S]
        c_from = ((cidx.block_of + 2) * blk)[None].expand(B, -1).to(torch.int32).contiguous()
        c_doc = cidx.c_doc.to(torch.int32).contiguous()
    else:
        K, V = k, v
    qt, kt, vt = (t.transpose(1, 2) for t in (q, K, V))
    doc32 = doc.to(torch.int32)
    sk = sink.to(torch.int32)
    if backend == "flex":
        term = open_term(opened.T, opened.index.codes, cap=opened.cap, raw_window=blk) if opened is not None else None
        chosen = _chosen_lookup(opened) if use_sum else None

        def pred(b, hh, qi, j):
            jr = torch.clamp(j, max=S - 1)
            same = (doc32[b, jr] == doc32[b, qi]) & (doc32[b, qi] >= 0)
            keep = (j >= (qi // blk - 1) * blk) | (sk[b, jr] > 0)
            if term is not None:
                keep = keep | term(b, qi, jr)
            ok = (jr <= qi) & same & keep & (j < S)
            if key_valid is not None:
                ok = ok & (key_valid[b, jr] | (jr == qi))
            if not use_sum:
                return ok
            c = torch.clamp(j - S, 0, C - 1)
            vis = (j >= S) & (qi >= c_from[b, c]) & (c_doc[b, c] == doc32[b, qi]) & (doc32[b, qi] >= 0)
            if chosen is not None:
                vis = vis & ~chosen(b, qi, c)
            return ok | vis

        score_mod = None
        if use_sum:
            def score_mod(score, b, hh, qi, j):
                c = torch.clamp(j - S, 0, C - 1)
                return torch.where(j >= S, score + logn[b, c] + qn2[b, hh, qi] * vhalf[b, c], score)

        nqb = -(-S // 128)
        first_q = torch.arange(nqb, device=q.device) * 128
        extra = [torch.stack([ds[:, first_q] // 128,
                              torch.clamp(ds[:, first_q] + max(n_sinks - 1, 0), max=S - 1) // 128], dim=-1)]
        if opened is not None:
            extra.append(opened.extra_blocks())
        if use_sum:
            first, last = S // 128, (S + C - 1) // 128
            ids = torch.arange(first, last + 1, device=q.device)
            # a slot KV block is needed by a query block once its earliest concept is old enough
            cf = F.pad(c_from[0], (S - first * 128, (last + 1) * 128 - S - C), value=2 ** 30).view(-1, 128).amin(-1)
            q_hi = torch.clamp(first_q + 127, max=S - 1)
            need = cf[None, :] <= q_hi[:, None]
            extra.append(torch.where(need, ids[None], torch.full_like(ids[None], -1))[None].expand(B, -1, -1))
        bm = _band_block_mask(pred, B=B, Q_LEN=S, KV_LEN=S + C, window=2 * blk, causal=True, device=q.device,
                              extra_blocks=torch.cat(extra, dim=-1))
        out = _get_flex()(qt.contiguous(), kt.contiguous(), vt.contiguous(), block_mask=bm, enable_gqa=(g != h),
                          score_mod=score_mod)
        return out.transpose(1, 2)
    # reference path (CPU / MPS tests): the same rule as a dense mask and an additive bias
    if g != h:
        kt = kt.repeat_interleave(h // g, dim=1)
        vt = vt.repeat_interleave(h // g, dim=1)
    qi = torch.arange(S, device=q.device)[:, None]
    j = torch.arange(S, device=q.device)[None, :]
    same = (doc[:, :, None] == doc[:, None, :]) & (doc[:, :, None] >= 0)
    keep = (j >= (qi // blk - 1) * blk)[None] | sink[:, None, :]
    if opened is not None:
        keep = keep | dense_open(opened.T, opened.index.codes, cap=opened.cap, raw_window=blk)
    mask = (j <= qi)[None] & same & keep
    if key_valid is not None:
        mask = mask & (key_valid.bool()[:, None, :] | torch.eye(S, dtype=torch.bool, device=q.device)[None])
    bias = torch.zeros(B, h, S, S + C, device=q.device, dtype=torch.float32)
    if use_sum:
        cf = c_from.long()
        vis = ((qi >= cf[:, None, :]) & (cidx.c_doc[:, None, :] == doc[:, :, None]) & (doc[:, :, None] >= 0))
        if opened is not None:
            vis = vis & ~(opened.T[:, :, :C].to(torch.int32) < int(opened.cap))
        mask = torch.cat([mask, vis], dim=-1)
        bias[..., S:] = logn[:, None, None, :] + qn2[..., None] * vhalf[:, None, None, :]
    bias = bias.masked_fill(~mask[:, None], float("-inf"))
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias.to(qt.dtype))
    return out.transpose(1, 2)

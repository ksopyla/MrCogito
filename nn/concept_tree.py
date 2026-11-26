"""E41 — the concept tree: several levels of summaries over the concept memory, a top-down read, a reasoner.

Spec: docs/experiments_specs/ahead/E41_tree_concept_model.md · plan: E41_plan.md.

Built on the concept read (nn/concept_memory.py, `par_global_read = "concept"`); selected with
`par_global_read = "tree"`. Three parts:

1. **The tree (writer).** Level 1 is the concept read's concepts (`ConceptIndex`: `per_block` content groups per
   closed `block`-key block, links to member tokens, the statistics n, μ_k, σ², μ_v, mean state z). Level 2 groups a
   block's concepts into `probes` nodes by content (the same balanced k-means, on the concepts' unit mean keys);
   level ℓ + 1 groups the `fan · probes` nodes of `fan` neighbouring level-ℓ containers into `probes` nodes. A
   container of level ℓ ≥ 2 spans `fan^(ℓ−2)` blocks. Statistics add, so a node's n, μ_k, σ², μ_v and z are exact
   sums over its member tokens; no weights. A container that holds more than one document is never built (its
   children stay roots). An entry may be read by queries at positions ≥ `frm` (its container's last block + 2
   blocks, the concept read's rule).

2. **The read.** Per query, top-down: the readable nodes of a level are the roots (their parent cannot be read
   yet) and the children of followed nodes; the best `follow` of them (by estimated attention mass) are followed,
   every other one is read as one summary entry with the concept read's logit log n + q·μ_k/√d + |q|² σ² / (2d).
   At level 1 the concept read's choice opens member tokens up to `top_k`. Every past token of the document is
   represented once: in the window, opened, or inside exactly one summary. With every node followed the read is
   the concept read, exactly (tests T2). Two scores for the choice: `descent` (a node's own estimate) and
   `lookahead` (the log-sum-exp of its concepts' estimates: the flat read's routing; costs the concept read's
   choice, so it is for rows up to about 128k).

3. **The reasoner** (`TreeReasoner`): a small looped transformer over the nodes of level ≥ 2. A node attends to
   nodes of its document that became readable no later than itself. It writes, zero at initialisation: a key and a
   value correction for every node (level-1 concepts through their parent and their own state), a plan vector per
   block (added to the token stream of the blocks that may read it), and the next-concept prediction.

Training signals carried by this module (each off by default):
- **closed-book rows** (`row_reads`): a row's read is drawn from {as normal, bounded tree read, cut at a level};
- **next concept** (`NextConcept`): the reasoner's pooled state after block b predicts the code histograms of block
  b + 1's level-2 nodes and concepts; codes are the nearest entries of a slowly updated codebook over a fixed random
  projection of the node states (BEST-RQ targets with NCM's EMA codebook);
- **route** (`route_loss`): a node's own estimate regresses the log-sum-exp of its concepts' estimates.

Shapes at 150M, 64k row: S = 65,536, 126 closed blocks; level 1: 8,064 concepts; level 2: 1,008; level 3 (4k
containers): 120; level 4 (32k): 8; q [B, S, 14, 64], k / v [B, S, 1, 64], z [B, E, 896].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.concept_memory import ConceptIndex, concept_labels, summary_log_scale
from nn.exact_read import OpenTable, choose

BIG = 1 << 30
ALL = -1  # a "follow" budget: every readable node of the level is followed


# ------------------------------------------------------------------------------------------------ the tree
@dataclass
class TreeIndex:
    """Every summary entry of one reading-layer call: level 1 (the concepts) first, then levels 2 … L.
    Entry e of level ℓ is `off[ℓ − 1] + i`; E = off[L]."""

    cidx: ConceptIndex
    off: list            # [L + 1] first entry of each level, then E
    n: torch.Tensor      # [B, E] member tokens (0 = empty)
    key: torch.Tensor    # [B, E, d] mean key (+ the reasoner's correction), with gradient
    var: torch.Tensor    # [B, E] mean per-dimension key variance
    val: torch.Tensor    # [B, E, d] mean value (+ the reasoner's correction)
    z: Optional[torch.Tensor]  # [B, E, H] mean state of the members, or None
    frm: torch.Tensor    # [B, E] long: first query position that may read the entry (BIG = never)
    pfrm: torch.Tensor   # [B, E] long: first position its parent may be read (BIG = none: a root for ever)
    par: torch.Tensor    # [B, E] long: the parent's entry (0 where none)
    doc: torch.Tensor    # [B, E] long: the entry's document (−1 = none: never a summary)
    probes: int
    fan: int

    @property
    def levels(self) -> int:
        return len(self.off) - 1

    @property
    def n_entries(self) -> int:
        return int(self.off[-1])

    @property
    def n_concepts(self) -> int:
        return int(self.off[1])

    def span(self, level: int) -> tuple:
        return int(self.off[level - 1]), int(self.off[level])


def tree_levels(n_blocks: int, fan: int, max_levels: int = 12) -> int:
    """Levels of a tree over `n_blocks` closed blocks: 2 (concepts, block nodes) + one per complete container size."""
    lv, n = 2, n_blocks // fan
    while n >= 1 and lv < max_levels:
        lv, n = lv + 1, n // fan
    return lv


def build_tree(cidx: ConceptIndex, *, probes: int = 8, fan: int = 8, iters: int = 3, tau: float = 0.05,
               max_levels: int = 12) -> TreeIndex:
    """The levels above `cidx`'s concepts. The grouping carries no gradient; the statistics do."""
    B, C1 = cidx.n.shape
    d = cidx.mu_k.shape[-1]
    K, P, blk = cidx.per_block, int(probes), cidx.block
    nb = C1 // K
    dev = cidx.n.device
    n1 = cidx.n.float()
    mu1, v1 = cidx.mu_k.float(), cidx.mu_v.float()
    var1 = cidx.var.float()
    frm1 = ((cidx.block_of + 2) * blk)[None].expand(B, -1)
    frm1 = torch.where(n1 > 0, frm1, torch.full_like(frm1, BIG))
    ns, keys, vars_, vals, frms, docs = [n1], [mu1], [var1], [v1], [frm1], [cidx.c_doc]
    zs = [cidx.z.float()] if cidx.z is not None else None
    pars, pfrms = [], []                                        # of each level's entries, filled one level later
    # children of the level being built: [B, n_cont, G, …]
    cn, cmu, cv = n1.view(B, nb, K), mu1.view(B, nb, K, d), v1.view(B, nb, K, d)
    csq = (var1 * d + (mu1 ** 2).sum(-1)).view(B, nb, K)        # mean squared key norm of each child's members
    cz = zs[0].view(B, nb, K, -1) if zs is not None else None
    cdoc = cidx.c_doc.view(B, nb, K)
    cont_doc = cdoc.amax(-1)                                    # [B, nb]: the block's document (−1: mixed / empty)
    n_cont, cb, off, base = nb, 1, [0, C1], C1
    while True:
        G = cn.shape[2]
        ok = cont_doc >= 0                                      # the container holds one document
        valid = (cn > 0) & (cdoc >= 0) & ok[..., None]
        unit = F.normalize(cmu.detach(), dim=-1).reshape(B * n_cont, G, d)
        lab = concept_labels(unit, valid.reshape(B * n_cont, G), P, iters=iters, tau=tau).view(B, n_cont, G)
        w = (lab[..., None] == torch.arange(P, device=dev)).float() * cn[..., None]     # [B, n_cont, G, P]
        pn = w.sum(2)                                                                   # [B, n_cont, P]
        inv = 1.0 / pn.clamp_min(1.0)
        pmu = torch.einsum("bcgp,bcgd->bcpd", w, cmu) * inv[..., None]
        psq = torch.einsum("bcgp,bcg->bcp", w, csq) * inv
        pvar = (psq - (pmu ** 2).sum(-1)).clamp_min(0.0) / d
        pv = torch.einsum("bcgp,bcgd->bcpd", w, cv) * inv[..., None]
        pz = torch.einsum("bcgp,bcgh->bcph", w, cz) * inv[..., None] if cz is not None else None
        first = torch.arange(n_cont, device=dev) * cb
        pfrom = ((first + cb + 1) * blk)[None, :, None].expand(B, n_cont, P)
        live = (pn > 0) & ok[..., None]
        pfrom = torch.where(live, pfrom, torch.full_like(pfrom, BIG))
        pdoc = torch.where(live, cont_doc[..., None].expand(B, n_cont, P), torch.full_like(pfrom, -1))
        # the children's parent links
        has = lab >= 0
        pidx = torch.arange(n_cont, device=dev)[None, :, None] * P + lab.clamp_min(0)   # [B, n_cont, G]
        child_pfrm = torch.where(has, pfrom.reshape(B, n_cont * P).gather(1, pidx.reshape(B, -1)).view(B, n_cont, G),
                                 torch.full_like(pidx, BIG))
        n_child = ns[-1].shape[1]
        par = torch.zeros(B, n_child, dtype=torch.long, device=dev)
        pf = torch.full((B, n_child), BIG, dtype=torch.long, device=dev)
        used = n_cont * G                                        # children inside complete containers
        par[:, :used] = torch.where(has, base + pidx, torch.zeros_like(pidx)).reshape(B, used)
        pf[:, :used] = child_pfrm.reshape(B, used)
        pars.append(par)
        pfrms.append(pf)
        ns.append(pn.reshape(B, -1))
        keys.append(pmu.reshape(B, -1, d))
        vars_.append(pvar.reshape(B, -1))
        vals.append(pv.reshape(B, -1, d))
        frms.append(pfrom.reshape(B, -1))
        docs.append(pdoc.reshape(B, -1))
        if zs is not None:
            zs.append(pz.reshape(B, n_cont * P, -1))
        base += n_cont * P
        off.append(base)
        n_next = n_cont // fan
        if n_next < 1 or len(off) - 1 >= max_levels:
            break
        m = n_next * fan                                         # containers that fill complete parents

        def regroup(t, tail=()):
            return t[:, :m].reshape(B, n_next, fan * P, *tail)

        cn, cmu, cv, csq = regroup(pn), regroup(pmu, (d,)), regroup(pv, (d,)), regroup(psq)
        cz = regroup(pz, (pz.shape[-1],)) if pz is not None else None
        cdoc = regroup(pdoc)
        cd = cont_doc[:, :m].view(B, n_next, fan)
        same = (cd == cd[..., :1]).all(-1) & (cd[..., 0] >= 0)
        cont_doc = torch.where(same, cd[..., 0], torch.full_like(cd[..., 0], -1))
        n_cont, cb = n_next, cb * fan
    top = ns[-1].shape[1]                                        # the last level has no parents
    pars.append(torch.zeros(B, top, dtype=torch.long, device=dev))
    pfrms.append(torch.full((B, top), BIG, dtype=torch.long, device=dev))
    return TreeIndex(cidx=cidx, off=off, n=torch.cat(ns, 1), key=torch.cat(keys, 1), var=torch.cat(vars_, 1),
                     val=torch.cat(vals, 1), z=torch.cat(zs, 1) if zs is not None else None,
                     frm=torch.cat(frms, 1), pfrm=torch.cat(pfrms, 1), par=torch.cat(pars, 1),
                     doc=torch.cat(docs, 1), probes=P, fan=int(fan))


def node_logit(q: torch.Tensor, tree: TreeIndex, lo: int, hi: int, *, variance: bool = True) -> torch.Tensor:
    """Entries lo … hi − 1: each one's estimated log attention mass for each query and head,
    log n + q·key/√d + |q|² σ² / (2d). q [B, s, h, d] → [B, s, h, hi − lo] fp32 (−inf for empty entries)."""
    d = q.shape[-1]
    qf = q.float()
    n = tree.n[:, lo:hi]
    lg = torch.einsum("bshd,bcd->bshc", qf, tree.key[:, lo:hi].float()) / math.sqrt(d)
    lg = lg + torch.where(n > 0, n.clamp_min(1.0).log(), torch.full_like(n, -math.inf))[:, None, None]
    if variance:
        lg = lg + ((qf ** 2).sum(-1, keepdim=True) / (2 * d)) * tree.var[:, lo:hi][:, None, None]
    return lg


def _lse_to_parents(score: torch.Tensor, par: torch.Tensor, pfrm: torch.Tensor, lo_p: int, n_p: int) -> torch.Tensor:
    """Log-sum-exp of the children's scores at their parents. score [B, ch, n_c] (−inf = nothing), par / pfrm
    [B, n_c] (the children's parent entry and its first readable position) → [B, ch, n_p] (−inf: no child)."""
    B, ch, _ = score.shape
    fin = torch.isfinite(score) & (pfrm < BIG)[:, None]
    m = torch.where(fin, score, torch.full_like(score, -1e30)).amax(-1, keepdim=True)
    e = torch.where(fin, torch.exp(score - m), torch.zeros_like(score))
    idx = (par - lo_p).clamp(0, n_p - 1)[:, None].expand(B, ch, -1)
    acc = torch.zeros(B, ch, n_p, dtype=score.dtype, device=score.device).scatter_add_(2, idx, e)
    return torch.where(acc > 0, acc.clamp_min(1e-30).log() + m, torch.full_like(acc, -math.inf))


# ------------------------------------------------------------------------------------------------ a row's read
@dataclass
class RowReads:
    """What each row's queries may read. follow [B, L + 1] long, indexed by level (entries 0 and 1 unused): nodes
    followed per query at that level (ALL = every readable one, 0 = none: the level's nodes are read as summaries
    and nothing below them). cap [B] long: tokens a query may open."""

    follow: torch.Tensor
    cap: torch.Tensor

    def flat(self, top_k: int) -> bool:
        return bool((self.follow[:, 2:] == ALL).all()) and bool((self.cap >= top_k).all())


def fixed_reads(B: int, levels: int, *, follow: int, top_k: int, device) -> RowReads:
    f = torch.full((B, levels + 1), int(follow) if follow > 0 else ALL, dtype=torch.long, device=device)
    return RowReads(follow=f, cap=torch.full((B,), int(top_k), dtype=torch.long, device=device))


def _row_uniform(ids: torch.Tensor, salt: int) -> torch.Tensor:
    """One number in [0, 1) per row, a fixed function of the row's tokens (the same in every loop pass and in a
    gradient-checkpoint recomputation). ids [B, S] → [B] fp64."""
    S = ids.shape[1]
    w = torch.arange(1, min(S, 257) + 1, device=ids.device, dtype=torch.float64)
    h = (ids[:, :w.numel()].double() * w).sum(-1) + ids[:, -1].double() * 7919.0 + float(salt) * 104729.0
    x = torch.sin(h * 12.9898) * 43758.5453
    return x - torch.floor(x)


def row_reads(ids: torch.Tensor, levels: int, *, mix: tuple, follow: int, top_k: int, small: int = 64) -> RowReads:
    """The closed-book rows (training): each row draws its read. `mix` = (as normal, bounded, cut) shares.
    As normal: every node followed, `top_k` tokens. Bounded: `follow` nodes per level, `top_k` tokens. Cut: a level
    m ∈ 1 … L drawn evenly; nothing below level m is read (follow 0 at levels ≤ m, every node above) and no token
    is opened, except that half of the m = 1 rows open `small` tokens."""
    B = ids.shape[0]
    dev = ids.device
    u, u2, u3 = _row_uniform(ids, 1), _row_uniform(ids, 2), _row_uniform(ids, 3)
    p_norm, p_bound = float(mix[0]), float(mix[1])
    f = torch.full((B, levels + 1), ALL, dtype=torch.long, device=dev)
    cap = torch.full((B,), int(top_k), dtype=torch.long, device=dev)
    bounded = (u >= p_norm) & (u < p_norm + p_bound)
    cut = u >= p_norm + p_bound
    f[bounded, 2:] = int(follow)
    m = (u2 * levels).long().clamp(max=levels - 1) + 1                                   # 1 … L
    lv = torch.arange(levels + 1, device=dev)[None]
    f = torch.where(cut[:, None] & (lv <= m[:, None]) & (lv >= 2), torch.zeros_like(f), f)
    cap = torch.where(cut, torch.where((m == 1) & (u3 < 0.5), torch.full_like(cap, int(small)),
                                       torch.zeros_like(cap)), cap)
    return RowReads(follow=f, cap=cap)


# ------------------------------------------------------------------------------------------------ the choice
@torch.no_grad()
def tree_choice(q: torch.Tensor, tree: TreeIndex, doc: torch.Tensor, reads: RowReads, *, top_k: int, width: int,
                variance: bool = True, route: str = "lookahead", open_tokens: bool = True, chunk: int = 0):
    """Each query's top-down choice. q [B, S, h, d], doc [B, S] (−1 = pad).

    Returns (opened, X): the open table of the member tokens (None when nothing may be opened) and X [B, S, E − C1]
    bool, the followed nodes of levels ≥ 2 (None when every row follows everything: the concept read)."""
    B, S, h, d = q.shape
    cidx = tree.cidx
    gi = cidx.groups
    C1, E, L = tree.n_concepts, tree.n_entries, tree.levels
    flat = bool((reads.follow[:, 2:] == ALL).all())
    if chunk <= 0:
        chunk = max(128, min(1024, (1 << 25) // max(B * h * C1, 1) // 128 * 128))
    tab = OpenTable(gi, B, S, cap=top_k) if open_tokens and bool((reads.cap > 0).any()) else None
    X = None if flat else torch.zeros(B, S, E - C1, dtype=torch.bool, device=q.device)
    if tab is None and X is None:
        return None, None
    look = route == "lookahead"
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qp = torch.arange(s0, s1, device=q.device)
        qd = doc[:, s0:s1]
        qc = q[:, s0:s1]
        sc1 = node_logit(qc, tree, 0, C1, variance=variance).amax(2)                    # [B, ch, C1]
        par_ok = None                                                                    # level 1: parent followed
        if X is not None:
            up, prev, plo = {}, sc1, 0
            for lv in range(2, L + 1):
                lo, hi = tree.span(lv)
                if look:
                    plo_, phi_ = tree.span(lv - 1)
                    up[lv] = _lse_to_parents(prev, tree.par[:, plo_:phi_], tree.pfrm[:, plo_:phi_], lo, hi - lo)
                    prev = up[lv]
                else:
                    up[lv] = node_logit(qc, tree, lo, hi, variance=variance).amax(2)
            Xc = X[:, s0:s1]
            for lv in range(L, 1, -1):
                lo, hi = tree.span(lv)
                avail = ((tree.frm[:, None, lo:hi] <= qp[None, :, None]) & (tree.doc[:, None, lo:hi] == qd[..., None])
                         & (qd[..., None] >= 0))
                root = tree.pfrm[:, None, lo:hi] > qp[None, :, None]
                if lv < L:
                    pe = (tree.par[:, lo:hi] - C1).clamp_min(0)[:, None].expand(B, s1 - s0, hi - lo)
                    cand = avail & (root | Xc.gather(2, pe))
                else:
                    cand = avail
                kf = reads.follow[:, lv]                                                 # [B]
                if bool((kf == ALL).all()):
                    sel = cand
                else:
                    sc = torch.where(cand & torch.isfinite(up[lv]), up[lv], torch.full_like(up[lv], -math.inf))
                    kmax = int(min(max(int(kf.max()), 1), hi - lo))
                    top = sc.topk(kmax, dim=-1).values                                   # [B, ch, kmax]
                    pick = (kf.clamp(1, kmax) - 1)[:, None, None].expand(B, s1 - s0, 1)
                    thr = top.gather(2, pick)
                    sel = cand & (sc >= thr) & torch.isfinite(sc) & (kf > 0)[:, None, None]
                    sel = torch.where((kf == ALL)[:, None, None], cand, sel)
                Xc[:, :, lo - C1:hi - C1] = sel
            pe1 = (tree.par[:, :C1] - C1).clamp_min(0)[:, None].expand(B, s1 - s0, C1)
            par_ok = (tree.pfrm[:, None, :C1] > qp[None, :, None]) | Xc.gather(2, pe1)
        if tab is not None:
            visible = (gi.g_close[:, None, :] <= qp[None, :, None]) & torch.isfinite(sc1)
            if par_ok is not None:
                visible = visible & par_ok
            visible = visible & (reads.cap > 0)[:, None, None]
            ids, off = choose(torch.nan_to_num(sc1, neginf=-1e30), visible, gi, qp, m=width, cap=top_k,
                              raw_window=cidx.block)
            small = reads.cap < top_k
            if bool(small.any()):  # a row's smaller budget: groups that start past it stay closed
                off = torch.where(off.long() < reads.cap[:, None, None], off, torch.full_like(off, top_k))
            tab.add(s0, ids, off)
    return tab, X


# ------------------------------------------------------------------------------------------------ the read
def _visible(tree: TreeIndex, X: Optional[torch.Tensor], opened: Optional[OpenTable], doc: torch.Tensor,
             q_idx: torch.Tensor) -> torch.Tensor:
    """[B, Q, E] bool: the summary entries each query `q_idx` [Q] reads (the rule, as a dense mask)."""
    B = doc.shape[0]
    C1, E = tree.n_concepts, tree.n_entries
    Q = q_idx.numel()
    qi = q_idx[None, :, None]
    qd = doc[:, q_idx][..., None]
    vis = (qi >= tree.frm[:, None, :]) & (tree.doc[:, None, :] == qd) & (qd >= 0)
    gone = torch.zeros(B, Q, E, dtype=torch.bool, device=doc.device)                    # followed / opened
    if opened is not None:
        gone[:, :, :C1] = opened.T[:, q_idx, :C1].to(torch.int32) < int(opened.cap)
    if X is None:
        gone[:, :, C1:] = True                                                           # every node followed
        return vis & ~gone
    gone[:, :, C1:] = X[:, q_idx]
    pe = (tree.par - C1).clamp_min(0)[:, None, :].expand(B, Q, E)
    par_ok = (qi < tree.pfrm[:, None, :]) | X[:, q_idx].gather(2, pe)
    return vis & par_ok & ~gone


def attend_tree(q, k, v, *, tree: TreeIndex, X: Optional[torch.Tensor], opened: Optional[OpenTable], doc, sink, ds,
                n_sinks: int, key_valid, backend: str, variance: bool = True, summaries: bool = True,
                dv_tok: Optional[torch.Tensor] = None, sum_len: int = 0):
    """One softmax over sinks ‖ block-aligned window ‖ opened members ‖ the readable summary entries of every level.
    q [B,S,h,d], k / v [B,S,1,d] → [B,S,h,d]. With `dv_tok` [B,S,1,d] (the reasoner's update spread to the member
    tokens) older tokens are read with it and the query's own block from a raw copy, as in the concept read."""
    from nn.exact_read import dense_open

    B, S, h, d = q.shape
    g = k.shape[2]
    blk = tree.cidx.block
    C1, E = tree.n_concepts, tree.n_entries
    two = dv_tok is not None
    v1 = v + dv_tok.to(v.dtype) if two else v
    lo_e, hi_e = (0, E) if X is not None else (0, C1)                                   # flat: level 1 only
    n_e = hi_e - lo_e if summaries else 0
    logrho = summary_log_scale(ds, S, blk, sum_len) if n_e else None
    if backend == "flex":
        return _attend_tree_flex(q, k, v, v1, tree=tree, X=X, opened=opened, doc=doc, sink=sink, ds=ds,
                                 n_sinks=n_sinks, key_valid=key_valid, variance=variance, n_e=n_e, two=two,
                                 logrho=logrho)
    # reference path (CPU / MPS tests, short rows): a dense mask and an additive bias
    Ks, Vs = [k, k] if two else [k], [v1, v] if two else [v1]
    if n_e:
        Ks.append(tree.key[:, :hi_e, None].to(k.dtype))
        Vs.append(tree.val[:, :hi_e, None].to(v.dtype))
    K, V = torch.cat(Ks, dim=1), torch.cat(Vs, dim=1)
    KV = K.shape[1]
    qt, kt, vt = (t.transpose(1, 2) for t in (q, K, V))
    if g != h:
        kt = kt.repeat_interleave(h // g, dim=1)
        vt = vt.repeat_interleave(h // g, dim=1)
    qi = torch.arange(S, device=q.device)[:, None]
    j = torch.arange(S, device=q.device)[None, :]
    own_lo = (qi // blk) * blk
    same = (doc[:, :, None] == doc[:, None, :]) & (doc[:, :, None] >= 0)
    keep = (j >= own_lo - blk)[None] | sink[:, None, :]
    if opened is not None:
        keep = keep | dense_open(opened.T, opened.index.codes, cap=opened.cap, raw_window=blk)
    if two:
        keep = keep & (j < own_lo)[None]
    eye = torch.eye(S, dtype=torch.bool, device=q.device)[None]
    mask = (j <= qi)[None] & same & keep
    if key_valid is not None:
        mask = mask & (key_valid.bool()[:, None, :] | eye)
    parts = [mask]
    if two:
        m2 = (j <= qi)[None] & same & (j >= own_lo)[None]
        if key_valid is not None:
            m2 = m2 & (key_valid.bool()[:, None, :] | eye)
        parts.append(m2)
    bias = torch.zeros(B, h, S, KV, device=q.device, dtype=torch.float32)
    if n_e:
        parts.append(_visible(tree, X, opened, doc, torch.arange(S, device=q.device))[:, :, :hi_e])
        n = tree.n[:, :hi_e]
        logn = torch.where(n > 0, n.clamp_min(1.0).log(), torch.zeros_like(n)).detach()
        vhalf = (0.5 * tree.var[:, :hi_e] if variance else torch.zeros_like(n)).detach()
        qn2 = (q.detach().float() ** 2).sum(-1).transpose(1, 2) / d                      # [B, h, S]
        bias[..., KV - hi_e:] = logn[:, None, None, :] + qn2[..., None] * vhalf[:, None, None, :]
        if logrho is not None:
            bias[..., KV - hi_e:] += logrho[:, None, :, None]
    mask = torch.cat(parts, dim=-1)
    empty = ~mask.any(-1)
    bias = bias.masked_fill(~(mask | empty[..., None])[:, None], float("-inf"))
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias.to(qt.dtype))
    return out.masked_fill(empty[:, None, :, None], 0.0).transpose(1, 2)


def _attend_tree_flex(q, k, v, v1, *, tree: TreeIndex, X, opened, doc, sink, ds, n_sinks, key_valid, variance, n_e,
                      two, logrho):
    """The flex path: the raw-token call of the concept read, and one call over the summary entries of every level
    with the visibility rule in mask_mod and the bias in score_mod; merged by their log-sum-exp (exact)."""
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask

    from nn.concept_memory import _chosen_lookup, flex_raw_call, merge_calls
    from nn.perceiver_ar_lm import _CHUNKED_MASK_PAIRS, _chunked_block_mask, _get_flex

    B, S, h, d = q.shape
    g = k.shape[2]
    blk = tree.cidx.block
    C1 = tree.n_concepts
    doc32 = doc.to(torch.int32)
    Kr, Vr = (torch.cat([k, k], dim=1), torch.cat([v1, v], dim=1)) if two else (k, v1)
    o1, l1 = flex_raw_call(q, k, Kr, Vr, two=two, doc32=doc32, sk=sink.to(torch.int32), ds=ds, n_sinks=n_sinks,
                           key_valid=key_valid, opened=opened, blk=blk)
    if not n_e:
        return o1.transpose(1, 2)
    Ks = tree.key[:, :n_e, None].to(k.dtype)
    Vs = tree.val[:, :n_e, None].to(v.dtype)
    n = tree.n[:, :n_e]
    # captured by score_mod / mask_mod: no gradient (see attend_concepts on atomic adds into shared entries)
    logn = torch.where(n > 0, n.clamp_min(1.0).log(), torch.zeros_like(n)).detach().contiguous()
    vhalf = ((0.5 * tree.var[:, :n_e]) if variance else torch.zeros_like(n)).detach().contiguous()
    qn2 = ((q.detach().float() ** 2).sum(-1).transpose(1, 2) / d).contiguous()            # [B, h, S]
    frm = tree.frm[:, :n_e].to(torch.int32).contiguous()
    edoc = tree.doc[:, :n_e].to(torch.int32).contiguous()
    chosen = _chosen_lookup(opened)
    if X is None:  # every node followed: level 1 only, the concept read's rule

        def spred(b, hh, qi, e):
            vis = (qi >= frm[b, e]) & (edoc[b, e] == doc32[b, qi]) & (doc32[b, qi] >= 0)
            if chosen is not None:
                vis = vis & ~chosen(b, qi, e)
            return vis
    else:
        n_up = int(X.shape[-1])
        pfrm = tree.pfrm[:, :n_e].to(torch.int32).contiguous()
        pe = (tree.par[:, :n_e] - C1).clamp(0, n_up - 1).to(torch.int32).contiguous()

        def spred(b, hh, qi, e):
            vis = (qi >= frm[b, e]) & (edoc[b, e] == doc32[b, qi]) & (doc32[b, qi] >= 0)
            par_ok = (qi < pfrm[b, e]) | X[b, qi, pe[b, e]]
            gone = X[b, qi, torch.clamp(e - C1, 0, n_up - 1)] & (e >= C1)
            if chosen is not None:
                gone = gone | (chosen(b, qi, torch.clamp(e, max=C1 - 1)) & (e < C1))
            return vis & par_ok & ~gone

    def smod(score, b, hh, qi, e):
        return score + logn[b, e] + qn2[b, hh, qi] * vhalf[b, e]

    if S * n_e > _CHUNKED_MASK_PAIRS:
        bm = _chunked_block_mask(spred, B=B, Q_LEN=S, KV_LEN=n_e, device=q.device)
    else:
        bm = create_block_mask(spred, B=B, H=None, Q_LEN=S, KV_LEN=n_e, device=q.device,
                               _compile=torch.cuda.is_available())
    o2, aux2 = _get_flex()(q.transpose(1, 2).contiguous(), Ks.transpose(1, 2).contiguous(),
                           Vs.transpose(1, 2).contiguous(), block_mask=bm, score_mod=smod, enable_gqa=(g != h),
                           return_aux=AuxRequest(lse=True))
    return merge_calls(o1, l1, o2, aux2.lse, logrho, q.dtype).transpose(1, 2)


def token_update(tree: TreeIndex, dv1: torch.Tensor, S: int) -> torch.Tensor:
    """Level-1 value corrections spread to the member tokens (the concept read's rule): [B, C1, d] → [B, S, 1, d]."""
    from nn.concept_memory import token_update as _tu

    return _tu(tree.cidx, dv1, S)

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
# For scripts that ask where a read went (None: off; nothing in training or the tests sets it). A dict
# {"q0": first query position watched, "pos": LongTensor [T] of source positions, "log": []}: every call of the
# top-down read appends (the watched queries' positions [n], whether each source position was opened for each of
# them [B, n, T], how many tokens each of them opened [B, n]). With a "mass" list in it, every call also appends
# the watched queries' attention by part (`_watch_mass`). scripts/e41_needle_why.py.
WATCH: Optional[dict] = None


@torch.no_grad()
def _watch_mass(logits: list, opens_at: list, n_fixed: int, qp: torch.Tensor) -> None:
    """Where the watched queries' attention went in one call of the top-down read: per head, the share on the
    window (part 0), the sinks, the opened tokens, the summaries, and on each watched source position among the
    opened tokens. `opens_at`: (index in `logits`, positions [B, ch, P]) of the opened-token parts; parts before
    `n_fixed` are the window and the sinks, the others summaries. Appends (query positions [n], {part: [B, n, h],
    "tgt": [B, n, h, T]}) to WATCH["mass"]."""
    sel = qp >= WATCH["q0"]
    if not bool(sel.any()):
        return
    w = torch.nan_to_num(torch.softmax(torch.cat([lg[:, sel] for lg in logits], -1).float(), -1), nan=0.0)
    tgt = WATCH["pos"].to(w.device)
    pos_of = dict(opens_at)
    rec = {k: torch.zeros(w.shape[:3]) for k in ("win", "sink", "open", "sum")}
    rec["tgt"] = torch.zeros(*w.shape[:3], tgt.numel())
    a = 0
    for i, lg in enumerate(logits):
        n_ = lg.shape[-1]
        wi = w[..., a:a + n_]
        a += n_
        part = "win" if i == 0 else "sink" if i < n_fixed else "open" if i in pos_of else "sum"
        rec[part] += wi.sum(-1).cpu()
        if part == "open":
            hit = (pos_of[i][:, sel][..., None] == tgt).to(wi.dtype)                     # [B, n, P, T]
            rec["tgt"] += torch.einsum("bnhp,bnpt->bnht", wi, hit).cpu()
    WATCH["mass"].append((qp[sel].cpu(), rec))
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
    val_h: Optional[torch.Tensor] = None  # [B, E, h, d] a value per query head, added to `val` (wide summaries)

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
    descent: Optional[torch.Tensor] = None   # [B] bool: rows whose choice scores a node by its own estimate

    def flat(self, top_k: int) -> bool:
        return bool((self.follow[:, 2:] == ALL).all()) and bool((self.cap >= top_k).all())


def fixed_reads(B: int, levels: int, *, follow: int, top_k: int, device, cut: int = 0) -> RowReads:
    """Every row the same read: `follow` nodes per level (≤ 0: all). `cut` m ≥ 1 (an evaluation switch): nothing
    below level m is read and no token is opened (m = 1: every concept a summary, the closed book)."""
    f = torch.full((B, levels + 1), int(follow) if follow > 0 else ALL, dtype=torch.long, device=device)
    cap = torch.full((B,), int(top_k), dtype=torch.long, device=device)
    if cut >= 1:
        f[:, 2:min(int(cut), levels) + 1] = 0
        cap = torch.zeros_like(cap)
    return RowReads(follow=f, cap=cap)


def _row_uniform(ids: torch.Tensor, salt: int, n: int) -> torch.Tensor:
    """One number in [0, 1) per row, a fixed function of the row's first `n` tokens (the same in every loop pass
    and in a gradient-checkpoint recomputation). ids [B, S] → [B] fp64."""
    w = torch.arange(1, min(ids.shape[1], max(int(n), 1)) + 1, device=ids.device, dtype=torch.float64)
    h = (ids[:, :w.numel()].double() * w).sum(-1) + float(salt) * 104729.0
    x = torch.sin(h * 12.9898) * 43758.5453
    return x - torch.floor(x)


def row_reads(ids: torch.Tensor, levels: int, *, mix: tuple, follow: int, top_k: int, small: int = 64,
              block: int = 512, step: Optional[int] = None, top_down: bool = False) -> RowReads:
    """The closed-book rows (training): each row draws its read. `mix` = (as normal, bounded, cut) shares.
    As normal: every node followed, `top_k` tokens. Bounded: a budget drawn evenly from `follow` / 4, / 2, × 1, × 4
    nodes per level, `top_k` tokens; half of the bounded rows choose by `descent` (a node's own estimate), half by
    `lookahead`. Cut: a level m ∈ 1 … L drawn evenly; nothing below level m is read (follow 0 at levels ≤ m, every
    node above) and no token is opened, except that half of the m = 1 rows open `small` tokens.

    The draw reads the row's first min(256, 2 · block) tokens only: no query among them has a memory to read
    (its window covers them), so the draw tells no query anything about its own future.

    `step` (an int): the draw is a function of that number alone, the same for every row (and, with the forward's
    count as the number, on every card). With several cards every step waits for the slowest card, and a bounded
    row is 1.3 × a flat row while a cut row is several times faster: drawn per row, most steps hold a slow row
    (run 1: 0.75 × the recall run's speed); drawn per step, cut steps are fast on every card.

    `top_down` (rows of any length, `tree_read_sparse`): no row follows every node. An as-normal row follows
    4 × `follow` per level, a cut row keeps its drawn bound above the cut, and every row chooses by `descent`."""
    B = ids.shape[0]
    dev = ids.device
    nh = min(256, 2 * int(block))
    if step is not None:
        seed = torch.full((B, 1), float(int(step) % 1000003), dtype=torch.float64, device=dev)
        u, u2, u3 = _row_uniform(seed, 1, 1), _row_uniform(seed, 2, 1), _row_uniform(seed, 3, 1)
    else:
        u, u2, u3 = _row_uniform(ids, 1, nh), _row_uniform(ids, 2, nh), _row_uniform(ids, 3, nh)
    p_norm, p_bound = float(mix[0]), float(mix[1])
    f = torch.full((B, levels + 1), ALL, dtype=torch.long, device=dev)
    cap = torch.full((B,), int(top_k), dtype=torch.long, device=dev)
    bounded = (u >= p_norm) & (u < p_norm + p_bound)
    cut = u >= p_norm + p_bound
    sizes = torch.tensor([max(follow // 4, 1), max(follow // 2, 1), int(follow), 4 * int(follow)], device=dev)
    fb = sizes[(u2 * 4).long().clamp(max=3)]                                              # [B]
    f[:, 2:] = torch.where(bounded[:, None], fb[:, None].expand(B, levels - 1), f[:, 2:])
    if top_down:
        f[:, 2:] = torch.where((bounded | cut)[:, None], fb[:, None], sizes[3].expand(B)[:, None]).expand(B, levels - 1)
    m = (u2 * levels).long().clamp(max=levels - 1) + 1                                   # 1 … L
    lv = torch.arange(levels + 1, device=dev)[None]
    f = torch.where(cut[:, None] & (lv <= m[:, None]) & (lv >= 2), torch.zeros_like(f), f)
    cap = torch.where(cut, torch.where((m == 1) & (u3 < 0.5), torch.full_like(cap, int(small)),
                                       torch.zeros_like(cap)), cap)
    return RowReads(follow=f, cap=cap, descent=torch.ones_like(bounded) if top_down else bounded & (u3 < 0.5))


# ------------------------------------------------------------------------------------------------ the choice
@torch.no_grad()
def tree_choice(q: torch.Tensor, tree: TreeIndex, doc: torch.Tensor, reads: RowReads, *, top_k: int, width: int,
                variance: bool = True, route: str = "lookahead", open_tokens: bool = True, chunk: int = 0,
                scan: bool = False):
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
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qp = torch.arange(s0, s1, device=q.device)
        ids, off, Xc = choose_chunk(q[:, s0:s1], doc[:, s0:s1], qp, tree, reads, top_k=top_k, width=width,
                                    variance=variance, route=route, open_tokens=tab is not None,
                                    need_x=X is not None, scan=scan)
        if X is not None:
            X[:, s0:s1] = Xc
        if tab is not None:
            tab.add(s0, ids, off)
    return tab, X


@torch.no_grad()
def choose_chunk(qc: torch.Tensor, qd: torch.Tensor, qp: torch.Tensor, tree: TreeIndex, reads: RowReads, *,
                 top_k: int, width: int, variance: bool = True, route: str = "lookahead", open_tokens: bool = True,
                 need_x: bool = True, scan: bool = False, rerank: int = 0, kt: Optional[torch.Tensor] = None):
    """The top-down choice of one chunk of queries (also one query: the cached decoder). qc [B, ch, h, d], qd
    [B, ch] their documents, qp [ch] their positions. Returns (ids, off) of `nn.exact_read.choose` (None, None when
    nothing may be opened) and Xc [B, ch, E − C1] bool, the followed nodes (None unless `need_x`).

    `scan` (2026-11-27): the tokens are opened from the best concepts of the whole row, as the concept read chooses
    them, whatever was followed; the followed nodes then bound the summaries only. A node of thousands of tokens
    averages one fact away, so walking down by a node's own estimate loses single facts on long rows; a concept
    (about 8 tokens) does not."""
    B, ch = qd.shape
    cidx = tree.cidx
    gi = cidx.groups
    C1, L = tree.n_concepts, tree.levels
    # rows that choose by a node's own estimate (`descent`); the others by the summed concept mass (`lookahead`)
    own = reads.descent if reads.descent is not None else torch.full((B,), route != "lookahead", dtype=torch.bool,
                                                                     device=qc.device)
    look, desc = bool((~own).any()), bool(own.any())
    sc1 = node_logit(qc, tree, 0, C1, variance=variance).amax(2)                        # [B, ch, C1]
    par_ok, Xc = None, None                                                              # level 1: parent followed
    if need_x:
        up, prev = {}, sc1
        for lv in range(2, L + 1):
            lo, hi = tree.span(lv)
            if look:
                plo_, phi_ = tree.span(lv - 1)
                prev = _lse_to_parents(prev, tree.par[:, plo_:phi_], tree.pfrm[:, plo_:phi_], lo, hi - lo)
                up[lv] = prev
            if desc:
                mine = node_logit(qc, tree, lo, hi, variance=variance).amax(2)
                up[lv] = torch.where(own[:, None, None], mine, up[lv]) if look else mine
        Xc = torch.zeros(B, ch, tree.n_entries - C1, dtype=torch.bool, device=qc.device)
        for lv in range(L, 1, -1):
            lo, hi = tree.span(lv)
            avail = ((tree.frm[:, None, lo:hi] <= qp[None, :, None]) & (tree.doc[:, None, lo:hi] == qd[..., None])
                     & (qd[..., None] >= 0))
            root = tree.pfrm[:, None, lo:hi] > qp[None, :, None]
            if lv < L:
                pe = (tree.par[:, lo:hi] - C1).clamp_min(0)[:, None].expand(B, ch, hi - lo)
                cand = avail & (root | Xc.gather(2, pe))
            else:
                cand = avail
            kf = reads.follow[:, lv]                                                     # [B]
            if bool((kf == ALL).all()):
                sel = cand
            else:
                sc = torch.where(cand & torch.isfinite(up[lv]), up[lv], torch.full_like(up[lv], -math.inf))
                kmax = int(min(max(int(kf.max()), 1), hi - lo))
                top = sc.topk(kmax, dim=-1).values                                       # [B, ch, kmax]
                pick = (kf.clamp(1, kmax) - 1)[:, None, None].expand(B, ch, 1)
                thr = top.gather(2, pick)
                sel = cand & (sc >= thr) & torch.isfinite(sc) & (kf > 0)[:, None, None]
                sel = torch.where((kf == ALL)[:, None, None], cand, sel)
            Xc[:, :, lo - C1:hi - C1] = sel
        pe1 = (tree.par[:, :C1] - C1).clamp_min(0)[:, None].expand(B, ch, C1)
        par_ok = (tree.pfrm[:, None, :C1] > qp[None, :, None]) | Xc.gather(2, pe1)
    if not open_tokens:
        return None, None, Xc
    visible = (gi.g_close[:, None, :] <= qp[None, :, None]) & torch.isfinite(sc1)
    if par_ok is not None and not scan:
        visible = visible & par_ok
    visible = visible & (reads.cap > 0)[:, None, None]
    score = torch.nan_to_num(sc1, neginf=-1e30)
    if rerank > 0 and scan:
        # `tree_read_sparse`'s re-rank: the scan's best concepts in the order of their best member token's own score
        if kt is None:
            raise ValueError("the re-rank needs the token keys")
        top = score.masked_fill(~visible, NEG).topk(min(max(int(width), int(rerank)), C1), dim=-1)
        best = member_best(qc, kt, member_table(gi), top.indices, top.values > NEG, gi.doc_start[:, qp],
                           (qp - cidx.block)[None])
        score = torch.full_like(score, -1e30).scatter_(2, top.indices, torch.nan_to_num(best, neginf=-1e30))
        visible = visible & (score > -1e29)
    ids, off = choose(score, visible, gi, qp, m=width, cap=top_k, raw_window=cidx.block)
    if bool((reads.cap < top_k).any()):  # a row's smaller budget: groups that start past it stay closed
        off = torch.where(off.long() < reads.cap[:, None, None], off, torch.full_like(off, top_k))
    return ids, off, Xc


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
    if n_e and tree.val_h is not None:  # wide summaries: each head reads its own slice
        vt = torch.cat([vt[:, :, :KV - hi_e], vt[:, :, KV - hi_e:] + tree.val_h[:, :hi_e].transpose(1, 2).to(vt.dtype)],
                       dim=2)
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
    gs = g                                                                               # key/value heads of the call
    if tree.val_h is not None:  # wide summaries: a value per query head (the keys repeat)
        Vs = Vs + tree.val_h[:, :n_e].to(v.dtype)
        Ks = Ks.expand(B, n_e, h, d)
        gs = h
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
                           Vs.transpose(1, 2).contiguous(), block_mask=bm, score_mod=smod, enable_gqa=(gs != h),
                           return_aux=AuxRequest(lse=True))
    return merge_calls(o1, l1, o2, aux2.lse, logrho, q.dtype).transpose(1, 2)


NEG = float("-inf")


def _take(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x [B, S, …] at pos [B, ch, P] → [B, ch, P, …]."""
    B, ch, P = pos.shape
    idx = pos.reshape(B, ch * P)
    if x.dim() == 2:
        return x.gather(1, idx).view(B, ch, P)
    return x.gather(1, idx[..., None].expand(B, ch * P, x.shape[-1])).view(B, ch, P, x.shape[-1])


def tree_read_stream(q, k, v, *, tree: TreeIndex, reads: RowReads, doc, ds, n_sinks: int, key_valid, top_k: int,
                     width: int, variance: bool = True, route: str = "lookahead", summaries: bool = True,
                     open_tokens: bool = True, dv_tok: Optional[torch.Tensor] = None, sum_len: int = 0,
                     q_offset: int = 0, chunk: int = 0, scan: bool = False) -> torch.Tensor:
    """The same read without a [B, S, ·] table (long rows: T is 4 GB at 128k and 260 GB at 1M): per chunk of
    queries of one block, the top-down choice, then one fp32 softmax over the window ‖ the document's sinks outside
    it ‖ the opened tokens (gathered) ‖ the summary entries the chunk may read. Differentiable (the choice carries
    no gradient). q [B, Q, h, d] holds the queries [q_offset, q_offset + Q) of the row that k / v [B, S, 1, d], doc
    and ds [B, S] cover. Returns [B, Q, h, d]."""
    from nn.exact_read import ChoiceList

    B, Q, h, d = q.shape
    S = int(k.shape[1])
    q0 = int(q_offset)
    dev = q.device
    sc = d ** -0.5
    blk = tree.cidx.block
    C1, E = tree.n_concepts, tree.n_entries
    kf, vf = k[:, :, 0], v[:, :, 0]
    vu = vf if dv_tok is None else vf + dv_tok[:, :, 0].to(vf.dtype)                      # older tokens: with the update
    gi = tree.cidx.groups
    flat = reads.flat(top_k)
    hi_e = E if not flat else C1
    logrho = summary_log_scale(ds, S, blk, sum_len) if summaries else None
    members = None
    per_block = C1 // max((tree.span(2)[1] - tree.span(2)[0]) // tree.probes, 1)
    out = []
    s0 = q0
    while s0 < q0 + Q:
        tau = s0 // blk
        n1 = min(C1, max(tau - 1, 0) * per_block)                                        # concepts the block may read
        per = (2 * blk + n_sinks + top_k + n1 + (hi_e - C1)) * h
        cs = chunk or max(8, min(blk, (1 << 25) // max(B * per, 1)))
        s1 = min(q0 + Q, (tau + 1) * blk, s0 + cs)
        ch = s1 - s0
        qp = torch.arange(s0, s1, device=dev)
        qc = q[:, s0 - q0:s1 - q0].float()
        qd = doc[:, s0:s1]
        okq = (qd >= 0)[..., None]
        own_lo = tau * blk
        logits, vals = [], []
        # the window: the query's block and the one before (its own block read raw)
        j0 = max(0, (tau - 1) * blk)
        jw = torch.arange(j0, s1, device=dev)
        ok = (jw[None, :] <= qp[:, None])[None] & (doc[:, j0:s1][:, None, :] == qd[..., None]) & okq
        if key_valid is not None:
            ok = ok & (key_valid.bool()[:, j0:s1][:, None, :] | (jw[None, :] == qp[:, None])[None])
        lw = torch.einsum("bqhd,bkd->bqhk", qc, kf[:, j0:s1].float()) * sc
        logits.append(lw.masked_fill(~ok[:, :, None, :], NEG))
        vw = torch.where((jw < own_lo)[None, :, None], vu[:, j0:s1], vf[:, j0:s1])
        vals.append(("shared", vw.float()))
        # the document's sinks outside the window
        if n_sinks and j0 > 0:
            sp = ds[:, s0:s1][..., None] + torch.arange(n_sinks, device=dev)             # [B, ch, n]
            spc = sp.clamp(max=S - 1)
            ok = (sp < j0) & (_take(doc, spc) == qd[..., None]) & okq
            if key_valid is not None:
                ok = ok & _take(key_valid.bool(), spc)
            ls = torch.einsum("bqhd,bqpd->bqhp", qc, _take(kf, spc).float()) * sc
            logits.append(ls.masked_fill(~ok[:, :, None, :], NEG))
            vals.append(("own", _take(vu, spc).float()))
        if tau >= 2:
            ids, off, Xc = choose_chunk(qc.detach(), qd, qp, tree, reads, top_k=top_k, width=width,
                                        variance=variance, route=route, open_tokens=open_tokens,
                                        need_x=not flat, scan=scan)
            chosen = torch.zeros(B, ch, C1, dtype=torch.bool, device=dev)
            if ids is not None:
                cl = ChoiceList(gi, B, ch, cap=top_k, m=ids.shape[-1], row0=s0, members=members)
                cl.add(s0, ids, off)
                pos, ok = cl.opened(s0, s1, blk)
                members = cl.members()
                # keep the opened places only (at most top_k per query), packed to the front
                P = int(min(top_k, pos.shape[-1]))
                dest = torch.where(ok, ok.long().cumsum(-1) - 1, torch.full_like(pos, P)).clamp(max=P)
                pk = torch.zeros(B, ch, P + 1, dtype=torch.long, device=dev).scatter_(2, dest, pos)[..., :P]
                ok = (torch.arange(P, device=dev) < ok.sum(-1, keepdim=True)) & okq
                if key_valid is not None:
                    ok = ok & _take(key_valid.bool(), pk)
                lo = torch.einsum("bqhd,bqpd->bqhp", qc, _take(kf, pk).float()) * sc
                logits.append(lo.masked_fill(~ok[:, :, None, :], NEG))
                vals.append(("own", _take(vu, pk).float()))
                chosen.scatter_(2, ids, off < top_k)
            if summaries:
                # entries the chunk may read: the concepts of blocks ≤ tau − 2, then every higher level
                sel = torch.cat([torch.arange(n1, device=dev), torch.arange(C1, hi_e, device=dev)])
                frm, edoc = tree.frm[:, sel], tree.doc[:, sel]
                vis = (qp[None, :, None] >= frm[:, None, :]) & (edoc[:, None, :] == qd[..., None]) & okq
                gone = torch.zeros(B, ch, sel.numel(), dtype=torch.bool, device=dev)
                gone[:, :, :n1] = chosen[:, :, :n1]
                if Xc is not None:
                    gone[:, :, n1:] = Xc
                    pe = (tree.par[:, sel] - C1).clamp_min(0)[:, None, :].expand(B, ch, sel.numel())
                    vis = vis & ((qp[None, :, None] < tree.pfrm[:, sel][:, None, :]) | Xc.gather(2, pe))
                vis = vis & ~gone
                n = tree.n[:, sel]
                bias = torch.where(n > 0, n.clamp_min(1.0).log(), torch.zeros_like(n)).detach()[:, None, None, :]
                if variance:
                    bias = bias + ((qc.detach() ** 2).sum(-1, keepdim=True) / (2 * d)) \
                        * tree.var[:, sel].detach()[:, None, None, :]
                if logrho is not None:
                    bias = bias + logrho[:, s0:s1][:, :, None, None]
                lsum = torch.einsum("bqhd,bcd->bqhc", qc, tree.key[:, sel].float()) * sc + bias
                logits.append(lsum.masked_fill(~vis[:, :, None, :], NEG))
                if tree.val_h is not None:
                    vals.append(("heads", tree.val[:, sel, None].float() + tree.val_h[:, sel].float()))
                else:
                    vals.append(("shared", tree.val[:, sel].float()))
        w = torch.nan_to_num(torch.softmax(torch.cat(logits, dim=-1), dim=-1), nan=0.0)
        o, a = 0.0, 0
        for lg, (kind, val) in zip(logits, vals):
            n_ = lg.shape[-1]
            wi = w[..., a:a + n_]
            a += n_
            if kind == "shared":
                o = o + torch.einsum("bqhk,bkd->bqhd", wi, val)
            elif kind == "heads":
                o = o + torch.einsum("bqhk,bkhd->bqhd", wi, val)
            else:
                o = o + torch.einsum("bqhp,bqpd->bqhd", wi, val)
        out.append(o.to(q.dtype))
        s0 = s1
    return torch.cat(out, dim=1)


# ------------------------------------------------------------------------------------------------ the top-down read
def _pick(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """x [B, E, …] at ids [B, …] → [B, …, …]."""
    B = x.shape[0]
    idx = ids.reshape(B, -1, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
    return x.gather(1, idx).view(*ids.shape, *x.shape[2:])


def member_table(gi) -> torch.Tensor:
    """[B, G, F] long: each concept's member positions, oldest first (−1 = none); built once per index."""
    mem = getattr(gi, "_member_table", None)
    if mem is None:
        from nn.exact_read import ChoiceList

        mem = ChoiceList(gi, int(gi.codes.shape[1]), 1, cap=1, m=1).members()
        object.__setattr__(gi, "_member_table", mem)
    return mem


@torch.no_grad()
def member_best(q: torch.Tensor, kt: torch.Tensor, mem: torch.Tensor, gids: torch.Tensor, valid: torch.Tensor,
                lo: torch.Tensor, hi: torch.Tensor, key_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The exact score of each of a query's candidate concepts: the best of its member tokens' own scores (q·k, the
    best head) among the members the query may open, positions in [lo, hi]. q [B, ch, h, d], kt [B, S, d] the token
    keys, mem [B, G, F] (`member_table`), gids / valid [B, ch, m], lo / hi [B or 1, ch]. Returns [B, ch, m] fp32
    (−inf: no member to open)."""
    B, ch, m = gids.shape
    Fm = mem.shape[2]
    pos = _pick(mem, gids)                                                               # [B, ch, m, F]
    ok = valid[..., None] & (pos >= lo[..., None, None]) & (pos <= hi[..., None, None])
    pc = pos.clamp_min(0).view(B, ch, m * Fm)
    if key_valid is not None:
        ok = ok & _take(key_valid.bool(), pc).view(B, ch, m, Fm)
    s = torch.einsum("bqhd,bqpd->bqhp", q, _take(kt, pc).to(q.dtype)).float().amax(2).view(B, ch, m, Fm)
    return s.masked_fill(~ok, NEG).amax(-1)


@torch.no_grad()
def sparse_index(tree: TreeIndex) -> dict:
    """What the top-down read needs besides the tree. `kids[ℓ]` = (order [B, n_ℓ] long, start [B, n_{ℓ+1}] long,
    count [B, n_{ℓ+1}] long): the entries of level ℓ sorted by parent, and where each node of level ℓ + 1 finds its
    children in that order (a node has 8 children on average and up to 64: packed, not padded). `roots[ℓ]` = (ids
    [B, R] long, ok [B, R] bool): the level's nodes that never get a parent (the top level, a document's last
    containers, the children of a container that holds two documents), or None when the level has none."""
    B = tree.n.shape[0]
    dev = tree.n.device
    kids, roots = {}, {}
    for lv in range(1, tree.levels + 1):
        lo, hi = tree.span(lv)
        frm, pfrm = tree.frm[:, lo:hi], tree.pfrm[:, lo:hi]
        fr = (frm < BIG) & (pfrm >= BIG)
        R = int(fr.sum(1).max()) if hi > lo else 0
        roots[lv] = None
        if R:
            idx = fr.to(torch.int8).argsort(dim=1, descending=True, stable=True)[:, :R]
            roots[lv] = (idx + lo, fr.gather(1, idx))
        if lv == tree.levels:
            break
        plo, phi = tree.span(lv + 1)
        n_par, n_c = phi - plo, hi - lo
        has = (frm < BIG) & (pfrm < BIG)
        key = torch.where(has, tree.par[:, lo:hi] - plo, torch.full_like(frm, n_par))
        order = key.argsort(dim=1, stable=True)
        cnt = torch.zeros(B, n_par + 1, dtype=torch.long, device=dev).scatter_add_(1, key, torch.ones_like(key))
        start = cnt.cumsum(1) - cnt
        kids[lv] = (order + lo, start[:, :n_par], cnt[:, :n_par])
    return {"kids": kids, "roots": roots}


def reads_top_down(reads: RowReads, route: str) -> bool:
    """A read the top-down path can make: a bound at every level, every row choosing by a node's own estimate."""
    own = bool(reads.descent.all()) if reads.descent is not None else route != "lookahead"
    return own and bool((reads.follow[:, 2:] != ALL).all())


def tree_read_sparse(q, k, v, *, tree: TreeIndex, reads: RowReads, doc, ds, n_sinks: int, key_valid, top_k: int,
                     width: int, variance: bool = True, summaries: bool = True, open_tokens: bool = True,
                     dv_tok: Optional[torch.Tensor] = None, sum_len: int = 0, q_offset: int = 0, chunk: int = 0,
                     index: Optional[dict] = None, reuse: Optional[torch.Tensor] = None,
                     scan: bool = False, span: int = 0, rerank: int = 0) -> torch.Tensor:
    """The bounded read made top-down, for rows of any length (2026-11-27). The table and streamed reads score every
    concept of the row for every query before they bound what is read, so their cost per query grows with the row
    (the reason no row was longer than 65,536 tokens). Here a query scores, level by level from the top, only the
    nodes it may reach: the children of the nodes it followed one level up, and the level's roots (nodes whose
    parent is not closed yet, or never will be). Its cost is `follow` × children per level whatever the row's length.

    The same read as `tree_read_stream` for rows that choose by a node's own estimate (`descent`) with a bound at
    every level (`reads_top_down`): the same followed nodes, opened tokens and readable summaries, one fp32 softmax
    over the window ‖ sinks ‖ opened tokens ‖ summaries. Differentiable. With gradients on, a chunk of queries
    is computed without a graph and recomputed for its own backward (`_SparseChunk`), and what it owes the row's
    tensors (keys, values, node keys and values) is added to one buffer per tensor that reaches them once
    (`_SparseAnchor`): neither the memory nor the time of a training step grows faster than the row.
    q [B, Q, h, d] holds the queries [q_offset, q_offset + Q). Returns [B, Q, h, d].

    `scan`: the tokens are opened from the best concepts of the whole row (`choose_chunk`'s `scan`): every closed
    concept is scored, head by head with a running maximum (one matrix product each: 8 numbers per token of the
    row, no table), and the walk down bounds the summaries only. The cost per query then grows with the row, with
    a small constant.

    `span` n > 0 (experimental, this read only): every opened token brings the n tokens after it (inside the
    query's document and before its window), the best concepts' runs first, each position once, within the same
    cap. A fact is a run of tokens: the released model finds where a planted number starts at 256k (the first
    token right in 6 of 8) and loses the digits after it, because each later digit must win the choice again by a
    weak cue (the digit before it), while the place itself keeps matching the question.

    `rerank` n > 0 (with `scan`): the scan's best n concepts (by the summary's estimate) are put in the order of
    their best member token's own score for the query (`member_best`), and the best `width` of those open tokens.
    A summary stands for about 8 tokens: the one token a query wants can score at the top of the row while its
    summary ranks in the hundreds (planted numbers at 128k-256k, 2026-11-27), and a place found by one pass of the
    reading layer is lost by the next. The cost is n × 8 more keys gathered per query, whatever the row's length.

    `reuse` [B, Q, h, d]: this call's output, computed before from the same inputs (a gradient checkpoint's
    recomputation of the layer): the chunks hand it on instead of computing it again; their backward is unchanged."""
    from nn.exact_read import ChoiceList, counts

    B, Q, h, d = q.shape
    S = int(k.shape[1])
    q0 = int(q_offset)
    dev = q.device
    sc = d ** -0.5
    blk = tree.cidx.block
    L, P, fan = tree.levels, tree.probes, tree.fan
    if bool((reads.follow[:, 2:] == ALL).any()):
        raise ValueError("the top-down read needs a bound on the nodes followed at every level")
    kt, vt = k[:, :, 0], v[:, :, 0]
    vu = vt if dv_tok is None else vt + dv_tok[:, :, 0].to(vt.dtype)
    gi = tree.cidx.groups
    ix = index if index is not None else sparse_index(tree)
    logrho = summary_log_scale(ds, S, blk, sum_len) if summaries else None
    wide = tree.val_h is not None
    fol = reads.follow
    kmax = [0, 0] + [int(fol[:, lv].max()) for lv in range(2, L + 1)]
    logn = torch.where(tree.n > 0, tree.n.clamp_min(1.0).log(), torch.zeros_like(tree.n)).detach().float()
    var = tree.var.detach().float()
    # dot products and values in the queries' own precision (bf16 under autocast, as the flex kernel); logits and
    # the softmax in fp32
    wd = q.dtype
    # the row's tensors, by slot: the graph's own (`orig`) and their data (`bigs`, what a chunk reads)
    KT, VT, VU, KEY, VAL, VALH = 0, 1, (1 if dv_tok is None else 2), 3, 4, 5
    orig = [kt.to(wd), vt.to(wd), None if dv_tok is None else vu.to(wd), tree.key.to(wd), tree.val.to(wd),
            tree.val_h.to(wd) if wide else None]
    bigs = [None if t is None else t.detach() for t in orig]
    needs = [t is not None and t.requires_grad for t in orig]
    opens = open_tokens and bool((reads.cap > 0).any())
    members = [None]
    rr = int(rerank) if (scan and opens) else 0
    if rr > 0:
        members[0] = member_table(gi)
    # a query's own candidates: about 8 children per followed node (the chunk size's estimate, with a margin)
    n_own = sum(max(kmax[lv + 1], 1) * 12 + 8 for lv in range(1, L))
    n_sh = sum((ix["roots"][lv][0].shape[1] if ix["roots"][lv] is not None else 0) + (2 * fan + 1) * P
               for lv in range(1, L + 1))
    per = (2 * blk + n_sh) * h + (n_sinks + top_k + n_own) * (h + 2 * d) + (n_own * h * d if wide else 0)
    C1 = tree.n_concepts
    if scan:
        per += 2 * C1                                                                    # a query's row of the scan
    if rr > 0:                                                                           # the candidates' member keys
        per += min(max(rr, int(width)), C1) * int(members[0].shape[2]) * (d + h + 2)
    grad = torch.is_grad_enabled() and (q.requires_grad or any(needs))
    acc: dict = {}                                                                       # slot → the gradient so far
    scan_keys: list = []                                                                 # the scan's key matrix, once

    def accumulate(leaves: list, grads) -> None:
        for (slot, where, _), g in zip(leaves, grads):
            if g is None:
                continue
            a = acc.get(slot)
            if a is None:
                a = acc[slot] = torch.zeros(bigs[slot].shape, dtype=torch.float32, device=dev)
            if isinstance(where, slice):
                a[:, where] += g.float()
            else:
                idx = where.reshape(B, -1)
                gg = g.reshape(B, idx.shape[1], *a.shape[2:]).float()
                a.scatter_add_(1, idx.reshape(B, -1, *([1] * (a.dim() - 2))).expand_as(gg), gg)

    @torch.no_grad()
    def scan_concepts(qw, qq, qp):
        """The best `width` concepts of the whole row for each query, by estimated mass (the largest over the
        heads): ids [B, ch, m] best first, and which of them are real choices."""
        # a concept's score for a head is q·key/√d + |q|² var / 2d + log n: the first two are one product of the
        # query with [key/√d, var] (the row's concepts, built once), the last does not depend on the head
        # In fp32 whatever the model's precision: the choice is among tens of thousands of concepts, and bf16 steps
        # of 0.125-0.25 at these scores would reorder the ones near the cut (the table read scores in fp32 too).
        if not scan_keys:
            k1 = bigs[KEY][:, :C1].float() * sc
            scan_keys.append(torch.cat([k1, var[:, :C1, None]], -1).transpose(1, 2) if variance
                             else k1.transpose(1, 2))                                    # [B, d (+1), C1]
        qf = qw.float()
        qa = torch.cat([qf, qq], -1) if variance else qf                                 # [B, ch, h, d (+1)]
        best = None
        with torch.autocast(device_type=qa.device.type, enabled=False):
            for hh in range(h):
                sh = torch.bmm(qa[:, :, hh], scan_keys[0])                               # [B, ch, C1]
                best = sh if best is None else torch.maximum(best, sh)
        may = ((gi.g_close[:, None, :] <= qp[None, :, None]) & (tree.n[:, None, :C1] > 0)
               & (reads.cap > 0)[:, None, None])
        score = (best.float() + logn[:, None, :C1]).masked_fill(~may, NEG)
        top = score.topk(min(max(int(width), rr), C1), dim=-1)
        if WATCH is not None and "rank" in WATCH:
            sel = qp >= WATCH["q0"]
            if bool(sel.any()):
                # the watched positions' concepts (layer 0's codes: concept id x 512 + order; negative = none) and
                # how many concepts of the row score above each, for each watched query
                code = gi.codes[0][:, WATCH["pos"].to(dev)]                              # [B, T]
                cid = torch.div(code, 512, rounding_mode="floor").clamp(0, C1 - 1).long()
                ss = score[:, sel]                                                       # [B, n, C1]
                mine = ss.gather(2, cid[:, None, :].expand(B, ss.shape[1], cid.shape[1]))
                rank = (ss[:, :, None, :] > mine[..., None]).sum(-1)
                rank = torch.where((code >= 0)[:, None, :] & torch.isfinite(mine), rank, torch.full_like(rank, -1))
                WATCH["rank"].append((qp[sel].cpu(), rank.cpu()))
        if WATCH is not None and "tok_rank" in WATCH:
            sel = qp >= WATCH["q0"]
            if bool(sel.any()):
                # the watched positions themselves: how many of the tokens the query may open (its document, before
                # its window) score above each by their own key (the best head): the choice an exact read would make
                tgt = WATCH["pos"].to(dev)
                st = torch.einsum("bnhd,bsd->bnhs", qf[:, sel], bigs[KT].float()).amax(2)  # [B, n, S]
                at = torch.arange(S, device=dev)[None, None, :]
                qs = qp[sel]
                st = st.masked_fill(~((at <= (qs - blk)[None, :, None])
                                      & (at >= gi.doc_start[:, qs][..., None])), NEG)
                mine = st[:, :, tgt]                                                     # [B, n, T]
                rk = (st[:, :, None, :] > mine[..., None]).sum(-1)
                rk = torch.where(mine > NEG, rk, torch.full_like(rk, -1))
                WATCH["tok_rank"].append((qs.cpu(), rk.cpu()))
        if rr > 0:
            # the candidates in the order of their best member token's own score; the best `width` are the choice
            gids = top.indices
            best = member_best(qw, bigs[KT], members[0], gids, top.values > NEG, gi.doc_start[:, qp],
                               (qp - blk)[None], key_valid)
            top = best.topk(min(int(width), gids.shape[2]), dim=-1)
            return gids.gather(2, top.indices), top.values > NEG
        return top.indices, top.values > NEG

    @torch.no_grad()
    def open_from(gids, valid, qp, okq, s0, s1):
        """The tokens the chosen concepts open, best first until the row's cap: positions pk [B, ch, P], which of
        them are real okp, and the places used before each concept, off [B, ch, m] (`top_k`: not opened)."""
        ch, mm = gids.shape[1], gids.shape[2]
        lo_ = gi.doc_start[:, qp][..., None].expand(B, ch, mm)
        hi_ = (qp - blk)[None, :, None].expand(B, ch, mm)
        cnt = torch.where(valid, counts(gi, gids, lo_, hi_), torch.zeros_like(gids, dtype=torch.int32))
        off = cnt.cumsum(-1) - cnt
        off = torch.where(valid & (off < top_k), off, torch.full_like(off, top_k))
        off = torch.where(off.long() < reads.cap[:, None, None], off, torch.full_like(off, top_k))
        cl = ChoiceList(gi, B, ch, cap=top_k, m=mm, row0=s0, members=members[0])
        cl.add(s0, gids, off.to(torch.int32))
        pos, okp = cl.opened(s0, s1, blk)
        members[0] = cl.members()
        Pn = int(min(top_k, pos.shape[-1]))
        dest = torch.where(okp, okp.long().cumsum(-1) - 1, torch.full_like(pos, Pn)).clamp(max=Pn)
        pk = torch.zeros(B, ch, Pn + 1, dtype=torch.long, device=dev).scatter_(2, dest, pos)[..., :Pn]
        okp = (torch.arange(Pn, device=dev) < okp.sum(-1, keepdim=True)) & okq
        if key_valid is not None:
            okp = okp & _take(key_valid.bool(), pk)
        if span > 0 and Pn > 0:
            r1 = span + 1
            n2 = Pn * r1
            cand = (pk[..., None] + torch.arange(r1, device=dev)).reshape(B, ch, n2)   # a token, then its run
            ok2 = okp[..., None].expand(B, ch, Pn, r1).reshape(B, ch, n2)
            ok2 = ok2 & (cand <= (qp - blk)[None, :, None])                            # before the query's window
            cc = cand.clamp(max=S - 1)
            ok2 = ok2 & (_take(doc, cc) == doc[:, qp][..., None])
            if key_valid is not None:
                ok2 = ok2 & _take(key_valid.bool(), cc)
            # each position once (the first, best, mention wins), then the cap by that order
            pri = torch.arange(n2, device=dev)
            key = torch.where(ok2, cc * n2 + pri, torch.full_like(cc, S * n2))
            srt = key.sort(dim=-1).values
            ps, pr = torch.div(srt, n2, rounding_mode="floor"), srt % n2
            first = torch.ones_like(ps, dtype=torch.bool)
            first[..., 1:] = ps[..., 1:] != ps[..., :-1]
            good = first & (ps < S)
            top = torch.where(good, pr, torch.full_like(pr, n2)).topk(min(top_k, n2), dim=-1, largest=False)
            pk = ps.gather(2, top.indices).clamp(max=S - 1)
            okp = top.values < n2
        if WATCH is not None:
            sel = qp >= WATCH["q0"]
            if bool(sel.any()):
                tgt = WATCH["pos"].to(dev)
                hit = ((pk[:, sel][..., None] == tgt) & okp[:, sel][..., None]).any(2)
                WATCH["log"].append((qp[sel].cpu(), hit.cpu(), okp[:, sel].sum(-1).cpu()))
        return pk, okp, off

    def one(s0: int, s1: int, qc: torch.Tensor, leaves: Optional[list] = None) -> torch.Tensor:
        def big(slot: int, where) -> torch.Tensor:
            """Rows of one of the row's tensors: a slice of positions, or the rows at `where` [B, …]. When the
            chunk is recomputed for its gradient they are leaves, and their gradient goes back by `accumulate`."""
            t = bigs[slot][:, where] if isinstance(where, slice) else _pick(bigs[slot], where)
            if leaves is not None and needs[slot]:
                t = t.detach().requires_grad_()
                leaves.append((slot, where, t))
            return t

        ch = s1 - s0
        t0 = s0 // blk
        # several whole blocks (each read against its own window and its own last nodes), or part of one block
        multi = t0 >= 2 and s0 % blk == 0 and ch % blk == 0
        nbk, blen = (ch // blk, blk) if multi else (1, ch)
        tb = torch.arange(t0, t0 + nbk, device=dev)                                      # [nbk] the blocks
        qp = torch.arange(s0, s1, device=dev)
        qpb = qp.view(nbk, blen)
        qw = qc.to(wd)
        qc = qc.float()
        qd = doc[:, s0:s1]
        okq = (qd >= 0)[..., None]
        logits, vals = [], []
        # the window: the query's block and the one before (its own block read raw)
        if multi:
            a0_ = (t0 - 1) * blk
            pair = lambda x: torch.cat([x[:, :-1], x[:, 1:]], dim=2)                     # [B, nbk + 1, blk, …] → pairs
            jw = (tb[:, None] - 1) * blk + torch.arange(2 * blk, device=dev)[None]       # [nbk, 2 blk] key positions
            dw = pair(doc[:, a0_:s1].view(B, nbk + 1, blk))
            ok = ((jw[:, None, :] <= qpb[:, :, None])[None] & (dw[:, :, None, :] == qd.view(B, nbk, blk)[..., None])
                  & okq.view(B, nbk, blk, 1))
            if key_valid is not None:
                kvw = pair(key_valid.bool()[:, a0_:s1].view(B, nbk + 1, blk))
                ok = ok & (kvw[:, :, None, :] | (jw[:, None, :] == qpb[:, :, None])[None])
            lw = torch.einsum("bnqhd,bnkd->bnqhk", qw.view(B, nbk, blk, h, d),
                              pair(big(KT, slice(a0_, s1)).view(B, nbk + 1, blk, d))).float() * sc
            logits.append(lw.masked_fill(~ok[:, :, :, None, :], NEG).reshape(B, ch, h, 2 * blk))
            if VU == VT:
                vals.append(("block", pair(big(VT, slice(a0_, s1)).view(B, nbk + 1, blk, d))))
            else:  # the block before: with the update; the query's own block: raw
                vals.append(("block", torch.cat([big(VU, slice(a0_, s1 - blk)).view(B, nbk, blk, d),
                                                 big(VT, slice(a0_ + blk, s1)).view(B, nbk, blk, d)], dim=2)))
        else:
            own_lo = t0 * blk
            j0 = max(0, (t0 - 1) * blk)
            jw = torch.arange(j0, s1, device=dev)
            ok = (jw[None, :] <= qp[:, None])[None] & (doc[:, j0:s1][:, None, :] == qd[..., None]) & okq
            if key_valid is not None:
                ok = ok & (key_valid.bool()[:, j0:s1][:, None, :] | (jw[None, :] == qp[:, None])[None])
            lw = torch.einsum("bqhd,bkd->bqhk", qw, big(KT, slice(j0, s1))).float() * sc
            logits.append(lw.masked_fill(~ok[:, :, None, :], NEG))
            if VU == VT:
                vals.append(("shared", big(VT, slice(j0, s1))))
            else:
                vals.append(("shared", torch.where((jw < own_lo)[None, :, None], big(VU, slice(j0, s1)),
                                                   big(VT, slice(j0, s1)))))
        # the document's sinks outside the window
        j0q = ((qp // blk - 1) * blk).clamp_min(0)                                       # [ch] each window's start
        if n_sinks and t0 + nbk - 2 > 0:
            sp = ds[:, s0:s1][..., None] + torch.arange(n_sinks, device=dev)
            spc = sp.clamp(max=S - 1)
            ok = (sp < j0q[None, :, None]) & (_take(doc, spc) == qd[..., None]) & okq
            if key_valid is not None:
                ok = ok & _take(key_valid.bool(), spc)
            ls = torch.einsum("bqhd,bqpd->bqhp", qw, big(KT, spc)).float() * sc
            logits.append(ls.masked_fill(~ok[:, :, None, :], NEG))
            vals.append(("own", big(VU, spc)))
        if t0 < 2:
            return _mix(logits, vals)
        n_fixed, opens_at = len(logits), []                                              # for the recorder only
        qq = (qc.detach() ** 2).sum(-1, keepdim=True) / (2 * d) if variance else None     # [B, ch, h, 1]
        rho = logrho[:, s0:s1][:, :, None, None] if logrho is not None else 0.0
        cat = lambda *ts: torch.cat([t for t in ts if t is not None], -1)
        f_ids, f_ok = None, None                                                         # followed one level up
        for lv in range(L, 0, -1):
            lo, hi = tree.span(lv)
            # roots every query of the row shares: nodes that never get a parent
            rid, nr, r_root, r_cand, r_lg = None, 0, None, None, None
            if ix["roots"][lv] is not None:
                rid, rok = ix["roots"][lv]
                nr = rid.shape[1]
                r_root = rok[:, None, :] & (_pick(tree.frm, rid)[:, None, :] <= qp[None, :, None])
                r_cand = r_root & (_pick(tree.doc, rid)[:, None, :] == qd[..., None]) & okq
                r_lg = torch.einsum("bqhd,bnd->bqhn", qw, big(KEY, rid)).float() * sc \
                    + _pick(logn, rid)[:, None, None, :]
                if variance:
                    r_lg = r_lg + qq * _pick(var, rid)[:, None, None, :]
            # roots the queries of a block share: its last nodes, whose parent is not closed yet
            tid, nt, t_cand, t_lg = None, 0, None, None
            if 2 <= lv < L:
                u = torch.div(tb - 1, fan ** (lv - 2), rounding_mode="floor")            # [nbk]
                c0 = ((torch.div(u, fan, rounding_mode="floor") - 1) * fan).clamp_min(0)
                c1 = (u + 1).clamp(max=(hi - lo) // P)
                nt = (2 * fan + 1) * P
                rel = c0[:, None] * P + torch.arange(nt, device=dev)[None]               # [nbk, nt] places in the level
                tv = rel < (c1 * P)[:, None]
                tid = (lo + rel.clamp(max=max(hi - lo - 1, 0)))[None].expand(B, nbk, nt)
                pf = _pick(tree.pfrm, tid)
                t_root = (tv[None, :, None, :] & (_pick(tree.frm, tid)[:, :, None, :] <= qpb[None, :, :, None])
                          & (pf < BIG)[:, :, None, :] & (pf[:, :, None, :] > qpb[None, :, :, None]))
                t_cand = (t_root & (_pick(tree.doc, tid)[:, :, None, :] == qd.view(B, nbk, blen)[..., None])
                          & okq.view(B, nbk, blen, 1)).reshape(B, ch, nt)
                t_lg = torch.einsum("bnqhd,bnkd->bnqhk", qw.view(B, nbk, blen, h, d), big(KEY, tid)).float() * sc \
                    + _pick(logn, tid)[:, :, None, None, :]
                if variance:
                    t_lg = t_lg + qq.view(B, nbk, blen, h, 1) * _pick(var, tid)[:, :, None, None, :]
                t_lg = t_lg.reshape(B, ch, h, nt)
            # each query's own candidates: the children of the nodes it followed one level up
            oid, o_cand, o_lg = None, None, None
            if lv < L and f_ids is not None:
                order, start, count = ix["kids"][lv]
                pl = (f_ids - tree.span(lv + 1)[0]).clamp_min(0)
                c_f = torch.where(f_ok, _pick(count, pl), torch.zeros_like(pl))          # [B, ch, kk] children each
                cum = c_f.cumsum(-1)
                tot = cum[..., -1:]
                W = int(tot.max())
            if lv < L and f_ids is not None and W > 0:
                j = torch.arange(W, device=dev).expand(B, ch, W).contiguous()            # the packed places
                slot = torch.searchsorted(cum, j, right=True).clamp(max=f_ids.shape[-1] - 1)
                at = _pick(start, pl).gather(2, slot) + j - (cum - c_f).gather(2, slot)
                o_cand = j < tot
                oid = torch.where(o_cand, _pick(order, at.clamp(0, order.shape[1] - 1)), torch.zeros_like(j))
                o_lg = torch.einsum("bqhd,bqnd->bqhn", qw, big(KEY, oid)).float() * sc \
                    + _pick(logn, oid)[:, :, None, :]
                if variance:
                    o_lg = o_lg + qq * _pick(var, oid)[:, :, None, :]
            if rid is None and tid is None and oid is None:
                f_ids, f_ok = None, None
                if lv == 1 and scan and opens:  # nothing to summarise, but the scan still opens tokens
                    pk, okp, _ = open_from(*scan_concepts(qw, qq, qp), qp, okq, s0, s1)
                    lo_t = torch.einsum("bqhd,bqpd->bqhp", qw, big(KT, pk)).float() * sc
                    opens_at.append((len(logits), pk))
                    logits.append(lo_t.masked_fill(~okp[:, :, None, :], NEG))
                    vals.append(("own", big(VU, pk)))
                continue
            with torch.no_grad():
                amax = lambda x: None if x is None else x.detach().amax(2)
                cand = cat(r_cand, t_cand, o_cand)
                score = cat(amax(r_lg), amax(t_lg), amax(o_lg))                          # [B, ch, N]
                ids_all = cat(None if rid is None else rid[:, None, :].expand(B, ch, nr),
                              None if tid is None else tid[:, :, None, :].expand(B, nbk, blen, nt).reshape(B, ch, nt),
                              oid)
                N = cand.shape[-1]
                taken = torch.zeros(B, ch, N, dtype=torch.bool, device=dev)              # followed / opened
                opened = None
                if lv >= 2:
                    km = min(max(kmax[lv], 1), N)
                    top = score.masked_fill(~cand, NEG).topk(km, dim=-1)
                    sel = torch.isfinite(top.values) & (torch.arange(km, device=dev)[None, None, :]
                                                         < fol[:, lv][:, None, None])
                    taken.scatter_(2, top.indices, sel)
                    f_ids, f_ok = ids_all.gather(2, top.indices), sel
                elif opens and scan:
                    # the tokens: the best concepts of the whole row; those among this query's summaries are read
                    # as tokens instead
                    gids, valid = scan_concepts(qw, qq, qp)
                    pk, okp, off = open_from(gids, valid, qp, okq, s0, s1)
                    chosen = torch.zeros(B, ch, C1, dtype=torch.bool, device=dev).scatter_(2, gids, off < top_k)
                    taken = chosen.gather(2, ids_all)
                    opened = (pk, okp)
                elif opens:
                    # the tokens: the best concepts a query may open, until the row's cap
                    may = cat(None if rid is None else r_root & (_pick(gi.g_close, rid)[:, None, :] <= qp[None, :, None]),
                              None if oid is None else o_cand & (_pick(gi.g_close, oid) <= qp[None, :, None]))
                    may = may & (reads.cap > 0)[:, None, None]
                    top = score.masked_fill(~may, NEG).topk(min(int(width), N), dim=-1)
                    pk, okp, off = open_from(ids_all.gather(2, top.indices), top.values > NEG, qp, okq, s0, s1)
                    taken.scatter_(2, top.indices, off < top_k)
                    opened = (pk, okp)
                vis = cand & ~taken
            if opened is not None:
                pk, okp = opened
                lo_t = torch.einsum("bqhd,bqpd->bqhp", qw, big(KT, pk)).float() * sc
                opens_at.append((len(logits), pk))
                logits.append(lo_t.masked_fill(~okp[:, :, None, :], NEG))
                vals.append(("own", big(VU, pk)))
            if summaries:
                if rid is not None:
                    logits.append((r_lg + rho).masked_fill(~vis[:, :, None, :nr], NEG))
                    vals.append(("heads", big(VAL, rid)[:, :, None] + big(VALH, rid)) if wide
                                else ("shared", big(VAL, rid)))
                if tid is not None:
                    logits.append((t_lg + rho).masked_fill(~vis[:, :, None, nr:nr + nt], NEG))
                    vals.append(("blockheads", big(VAL, tid)[:, :, :, None] + big(VALH, tid)) if wide
                                else ("block", big(VAL, tid)))
                if oid is not None:
                    logits.append((o_lg + rho).masked_fill(~vis[:, :, None, nr + nt:], NEG))
                    vals.append(("ownheads", big(VAL, oid)[:, :, :, None] + big(VALH, oid)) if wide
                                else ("own", big(VAL, oid)))
        if WATCH is not None and "mass" in WATCH:
            _watch_mass(logits, opens_at, n_fixed, qp)
        return _mix(logits, vals)

    out = []
    s0 = q0
    run = _ChunkRun(one, accumulate, wd)
    anchor = _SparseAnchor.apply(acc, *orig) if grad else None
    budget = max(8, (1 << 28) // max(B * per, 1))                                        # queries per chunk
    while s0 < q0 + Q:
        tau = s0 // blk
        cs = chunk or budget
        nb = min(cs, q0 + Q - s0) // blk
        if tau >= 2 and s0 % blk == 0 and nb >= 1:                                       # whole blocks
            s1 = s0 + nb * blk
        else:
            s1 = min(q0 + Q, (tau + 1) * blk, s0 + cs)
        qc = q[:, s0 - q0:s1 - q0]
        if grad:
            out.append(_SparseChunk.apply(run, s0, s1, qc, anchor,
                                          None if reuse is None else reuse[:, s0 - q0:s1 - q0]))
        else:
            out.append(one(s0, s1, qc).to(wd))
        s0 = s1
    return torch.cat(out, dim=1)


class _ChunkRun:
    """One read's chunk function, the side channel of its gradients, and its output's precision."""

    def __init__(self, fn, accumulate, dtype):
        self.fn, self.accumulate, self.dtype = fn, accumulate, dtype


class _SparseAnchor(torch.autograd.Function):
    """The row's tensors of one top-down read. Its output is an input of every chunk, so its backward runs after
    all of theirs and hands each tensor the gradient the chunks accumulated for it, once."""

    @staticmethod
    def forward(ctx, acc, *tensors):
        ctx.acc = acc
        ctx.dtypes = [None if t is None else t.dtype for t in tensors]
        ref = next(t for t in tensors if t is not None)
        return ref.new_zeros(())

    @staticmethod
    def backward(ctx, _):
        out = []
        for slot, dt in enumerate(ctx.dtypes):
            g = ctx.acc.pop(slot, None)
            out.append(None if g is None or dt is None else g.to(dt))
        return (None, *out)


class _SparseChunk(torch.autograd.Function):
    """One chunk of queries of a top-down read: computed without a graph; for its backward, computed again with one
    (only this chunk's graph is alive at a time), the gradient returned for its queries and accumulated for the
    row's tensors."""

    @staticmethod
    def forward(ctx, run, s0, s1, qc, anchor, done):
        ctx.run, ctx.span = run, (s0, s1)
        ctx.save_for_backward(qc)
        if done is not None:  # computed by the forward pass this call recomputes
            return done.to(run.dtype).clone()
        return run.fn(s0, s1, qc).to(run.dtype)

    @staticmethod
    def backward(ctx, g):
        (qc,) = ctx.saved_tensors
        run, (s0, s1) = ctx.run, ctx.span
        with torch.enable_grad():
            q_ = qc.detach().requires_grad_()
            leaves: list = []
            o = run.fn(s0, s1, q_, leaves).to(run.dtype)
            grads = torch.autograd.grad(o, [q_] + [t for _, _, t in leaves], g, allow_unused=True)
        run.accumulate(leaves, grads[1:])
        return None, None, None, grads[0], g.new_zeros(()), None


def _mix(logits: list, vals: list) -> torch.Tensor:
    """One softmax over the parts' logits [B, ch, h, n_i]; the weighted values. A part's values are shared by the
    chunk ([B, n, d]; `heads`: [B, n, h, d]), by each block of its queries (`block`: [B, nbk, n, d]; `blockheads`:
    [B, nbk, n, h, d]) or each query's own ([B, ch, n, d]; `ownheads`: [B, ch, n, h, d])."""
    w = torch.nan_to_num(torch.softmax(torch.cat(logits, dim=-1), dim=-1), nan=0.0)
    o, a = 0.0, 0
    for lg, (kind, val) in zip(logits, vals):
        n_ = lg.shape[-1]
        wi = w[..., a:a + n_].to(val.dtype)
        a += n_
        if kind == "shared":
            o = o + torch.einsum("bqhk,bkd->bqhd", wi, val)
        elif kind == "heads":
            o = o + torch.einsum("bqhk,bkhd->bqhd", wi, val)
        elif kind == "ownheads":
            o = o + torch.einsum("bqhp,bqphd->bqhd", wi, val)
        elif kind in ("block", "blockheads"):  # val [B, nbk, n, (h,) d]: one set of values per block of queries
            Bq, ch_, h_, _ = wi.shape
            wb = wi.reshape(Bq, val.shape[1], ch_ // val.shape[1], h_, n_)
            ob = torch.einsum("bnqhk,bnkd->bnqhd" if kind == "block" else "bnqhk,bnkhd->bnqhd", wb, val)
            o = o + ob.reshape(Bq, ch_, h_, -1)
        else:
            o = o + torch.einsum("bqhp,bqpd->bqhd", wi, val)
    return o


def token_update(tree: TreeIndex, dv1: torch.Tensor, S: int) -> torch.Tensor:
    """Level-1 value corrections spread to the member tokens (the concept read's rule): [B, C1, d] → [B, S, 1, d]."""
    from nn.concept_memory import token_update as _tu

    return _tu(tree.cidx, dv1, S)


# ------------------------------------------------------------------------------------------------ the reasoner
class TreeReasoner(nn.Module):
    """A small looped transformer over the nodes of level ≥ 2 (ConceptReasoner's block, input injection each loop).
    A node attends to nodes of its document that became readable no later than itself, so what it writes is safe
    for every query that may read it. From the reasoned states, all zero at initialisation:

    - a key and a value correction for every node of level ≥ 2, and for every concept (its own state through a
      small MLP, plus its parent's reasoned state): the summary a query reads is the average plus the correction;
    - a plan vector per block (the pooled state of the block's level-2 nodes), for the token stream;
    - the next-concept prediction: after block b, the code histograms of block b + 1's level-2 nodes and concepts.

    Codes (the targets): the nearest entry of a codebook over a fixed random projection of the node states; the
    codebooks follow the assigned vectors by a moving average, updated once per training forward from the previous
    forward's sums (so a forward and its gradient-checkpoint recomputation see the same codebook), summed over the
    cards. z [B, E, H] → dk, dv [B, E, d], plan [B, nb, H], aux [2] (coarse, fine next-concept cross-entropies)."""

    def __init__(self, hidden: int, d_kv: int, *, dim: int = 512, layers: int = 2, heads: int = 8, loops: int = 2,
                 plan: bool = True, next_concept: bool = True, codes_coarse: int = 1024, codes_fine: int = 4096,
                 code_dim: int = 64, code_decay: float = 0.99, max_levels: int = 16, keys: bool = True,
                 aux_detach: bool = False, plan_from: str = "state", ahead: int = 1):
        super().__init__()
        # keys False: no key corrections (the summaries' keys stay the exact means: the choice and the logits of
        # the concept read); aux_detach: the next-concept prediction reads the reasoned states without gradient
        # into them (it trains its own layers only), so the extra loss cannot move what the read uses
        self.keys, self.aux_detach = bool(keys), bool(aux_detach)
        # plan_from "pred": the plan is read from the predictor's layer (what it expects next), not from the
        # reasoned state; `ahead`: the block the prediction is about (2 = the block whose tokens read this plan)
        self.plan_from, self.ahead = str(plan_from), max(1, int(ahead))
        self.loops, self.heads, self.dim = max(1, int(loops)), int(heads), int(dim)
        self.inp = nn.Linear(hidden, dim, bias=False)
        self.lvl = nn.Embedding(max_levels + 1, dim)
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.RMSNorm(dim), qkv=nn.Linear(dim, 3 * dim, bias=False), o=nn.Linear(dim, dim, bias=False),
                n2=nn.RMSNorm(dim), up=nn.Linear(dim, 4 * dim, bias=False), down=nn.Linear(4 * dim, dim, bias=False))))
        self.norm = nn.RMSNorm(dim)
        self.out_k = nn.Linear(dim, d_kv, bias=False)
        self.out_v = nn.Linear(dim, d_kv, bias=False)
        # level 1: the concept's own state and its parent's reasoned state
        self.inp1 = nn.Linear(hidden, dim, bias=False)
        self.n1 = nn.RMSNorm(dim)
        self.up1 = nn.Linear(dim, 2 * dim, bias=False)
        self.down1 = nn.Linear(2 * dim, dim, bias=False)
        self.norm1 = nn.RMSNorm(dim)
        self.out_k1 = nn.Linear(dim, d_kv, bias=False)
        self.out_v1 = nn.Linear(dim, d_kv, bias=False)
        self.plan = nn.Linear(dim, hidden, bias=False) if plan else None
        self.next_concept = bool(next_concept)
        if self.next_concept:
            self.pred = nn.Linear(dim, dim, bias=False)
            self.head_c = nn.Linear(dim, int(codes_coarse), bias=False)
            self.head_f = nn.Linear(dim, int(codes_fine), bias=False)
            g = torch.Generator().manual_seed(4141)
            self.register_buffer("code_proj", torch.randn(hidden, int(code_dim), generator=g) / math.sqrt(hidden))
            for name, K in (("c", int(codes_coarse)), ("f", int(codes_fine))):
                self.register_buffer(f"book_{name}", F.normalize(torch.randn(K, int(code_dim), generator=g), dim=-1))
                self.register_buffer(f"count_{name}", torch.ones(K))
                # the last forward's sums, applied at the start of the next one: per code the assigned vectors and
                # their count, then a spare vector and its count (re-seeds a code nothing is assigned to)
                self.register_buffer(f"pend_{name}", torch.zeros(K, 2 * int(code_dim) + 2), persistent=False)
            self.code_decay = float(code_decay)
        self.zero_outputs()

    def zero_outputs(self) -> None:
        for lin in (self.out_k, self.out_v, self.out_k1, self.out_v1, self.plan):
            if lin is not None:
                nn.init.zeros_(lin.weight)

    # -- codebooks -------------------------------------------------------------------------------------
    @torch.no_grad()
    def begin_forward(self, training: bool) -> None:
        """Apply the previous training forward's codebook sums (summed over the cards: every card calls this in
        every training forward, so they hold the same codebooks)."""
        if not (self.next_concept and training):
            return
        for name in ("c", "f"):
            pend = getattr(self, f"pend_{name}")
            tot = pend.float().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(tot, op=torch.distributed.ReduceOp.SUM)
            pend.zero_()
            book, count = getattr(self, f"book_{name}"), getattr(self, f"count_{name}")
            cd = book.shape[1]
            vec, cnt, spare, scnt = tot[:, :cd], tot[:, cd], tot[:, cd + 1:2 * cd + 1], tot[:, 2 * cd + 1]
            if float(cnt.sum()) <= 0:
                continue
            a = self.code_decay
            cf = count.float() * a + (1 - a) * cnt
            bf = book.float()
            new = F.normalize(a * bf + (1 - a) * F.normalize(vec, dim=-1), dim=-1)
            bf = torch.where((cnt > 0)[:, None], new, bf)
            dead = (cf < 0.05 * float(cf.mean())) & (scnt > 0)                            # nothing assigned for long
            bf = torch.where(dead[:, None], F.normalize(spare, dim=-1), bf)
            cf = torch.where(dead, cf.mean().expand_as(cf), cf)
            book.copy_(bf.to(book.dtype))
            count.copy_(cf.to(count.dtype))

    @torch.no_grad()
    def codes(self, z: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
        """Each node's code. z [B, N, H], valid [B, N] → [B, N] long (−1 where not valid); in training the sums for
        the next codebook update are kept (overwritten by every call: the last one counts)."""
        book = getattr(self, f"book_{name}").float()                                     # fp32 whatever the model's type
        e = F.normalize(z.detach().float() @ self.code_proj.float(), dim=-1)             # [B, N, cd]
        code = (e @ book.t()).argmax(-1)
        if self.training:
            K, cd = book.shape
            ef = e[valid]
            cf = code[valid]
            pend = torch.zeros(K, 2 * cd + 2, device=z.device, dtype=torch.float32)
            pend[:, :cd].index_add_(0, cf, ef)
            pend[:, cd].index_add_(0, cf, torch.ones_like(cf, dtype=torch.float32))
            slot = torch.arange(ef.shape[0], device=z.device) % K                        # a spare vector per code
            pend[:, cd + 1:2 * cd + 1].index_add_(0, slot, ef)
            pend[:, 2 * cd + 1].index_add_(0, slot, torch.ones_like(slot, dtype=torch.float32))
            getattr(self, f"pend_{name}").copy_(pend.to(getattr(self, f"pend_{name}").dtype))
        return torch.where(valid, code, torch.full_like(code, -1))

    # -- forward -----------------------------------------------------------------------------------------
    def forward(self, tree: TreeIndex):
        z = tree.z
        B, E, H = z.shape
        C1, P = tree.n_concepts, tree.probes
        dev = z.device
        zu = z[:, C1:].to(self.inp.weight.dtype)
        N = zu.shape[1]
        level = torch.cat([torch.full((tree.off[lv] - tree.off[lv - 1],), lv, dtype=torch.long, device=dev)
                           for lv in range(2, tree.levels + 1)])
        frm, doc = tree.frm[:, C1:], tree.doc[:, C1:]
        ok = frm < BIG
        u = self.inp(zu) + self.lvl(level.clamp(max=self.lvl.num_embeddings - 1))[None]
        mask = ((frm[:, :, None] >= frm[:, None, :]) & (doc[:, :, None] == doc[:, None, :])
                & ok[:, :, None] & ok[:, None, :])
        mask = mask | torch.eye(N, dtype=torch.bool, device=dev)[None]                   # no empty rows
        h = torch.zeros_like(u)
        for _ in range(self.loops):
            h = h + u                                                                    # input injection
            for blk in self.blocks:
                a = blk["qkv"](blk["n1"](h)).view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
                o = F.scaled_dot_product_attention(a[0], a[1], a[2], attn_mask=mask[:, None])
                h = h + blk["o"](o.transpose(1, 2).reshape(B, N, -1))
                h = h + blk["down"](F.silu(blk["up"](blk["n2"](h))))
        hn = self.norm(h) * ok[..., None].to(h.dtype)
        dk_up, dv_up = self.out_k(hn), self.out_v(hn)
        # level 1
        z1 = z[:, :C1].to(self.inp1.weight.dtype)
        has = (tree.pfrm[:, :C1] < BIG)
        pe = (tree.par[:, :C1] - C1).clamp_min(0)[..., None].expand(B, C1, self.dim)
        h1 = self.inp1(z1) + h.gather(1, pe) * has[..., None].to(h.dtype)
        h1 = h1 + self.down1(F.silu(self.up1(self.n1(h1))))
        h1n = self.norm1(h1) * (tree.n[:, :C1] > 0)[..., None].to(h1.dtype)
        dk = torch.cat([self.out_k1(h1n), dk_up], dim=1)
        dv = torch.cat([self.out_v1(h1n), dv_up], dim=1)
        if not self.keys:
            dk = dk * 0.0
        # the blocks' pooled states (their level-2 nodes): the plan and the predictor
        lo2, hi2 = tree.span(2)
        nb = (hi2 - lo2) // P
        ok2 = ok[:, :nb * P].view(B, nb, P)
        pooled = (hn[:, :nb * P].view(B, nb, P, -1) * ok2[..., None].to(hn.dtype)).sum(2)
        pooled = pooled / ok2.sum(-1, keepdim=True).clamp_min(1).to(hn.dtype)
        pr = None
        if self.next_concept:
            pr = F.silu(self.pred(pooled.detach() if self.aux_detach else pooled))
        if self.plan is None:
            plan = None
        else:
            plan = self.plan(pr if (self.plan_from == "pred" and pr is not None) else pooled)
        aux = torch.zeros(2, device=dev, dtype=torch.float32)
        if self.next_concept and self.training:  # the prediction is a training signal only
            code_c = self.codes(zu[:, :nb * P], ok[:, :nb * P], "c").view(B, nb, P)      # level-2 nodes
            code_f = self.codes(z1, tree.n[:, :C1] > 0, "f").view(B, nb, -1)             # concepts
            ah = self.ahead
            bdoc = doc[:, :nb * P].view(B, nb, P).amax(-1)                               # a block's document (−1 none)
            pair = ((bdoc[:, :-ah] >= 0) & (bdoc[:, :-ah] == bdoc[:, ah:])) if nb > ah else None
            if pair is not None and bool(pair.any()):
                losses = []
                for head, code in ((self.head_c, code_c), (self.head_f, code_f)):
                    K = head.out_features
                    tgt = torch.zeros(B, nb, K + 1, device=dev)
                    tgt.scatter_add_(2, torch.where(code >= 0, code, torch.full_like(code, K)),
                                     torch.ones_like(code, dtype=torch.float32))
                    tgt = tgt[:, ah:, :K]                                                # block b + ahead's histogram
                    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1.0)
                    logp = F.log_softmax(head(pr[:, :-ah]).float(), dim=-1)
                    ce = -(tgt * logp).sum(-1)                                           # [B, nb − 1]
                    losses.append((ce * pair).sum() / pair.sum().clamp_min(1))
                aux = torch.stack(losses)
            else:  # every parameter in the graph on every card (DDP)
                aux = aux + 0.0 * (self.head_c.weight.sum() + self.head_f.weight.sum() + pr.sum()).float()
        return dk, dv, plan, aux


@torch.no_grad()
def read_stats(tree: TreeIndex, X: Optional[torch.Tensor], opened: Optional[OpenTable], doc: torch.Tensor,
               queries: int = 64) -> dict:
    """Probe scalars of one read, over a sample of queries: summary entries read per query (all levels, and per
    level), nodes followed, concepts opened, and the share of readable tokens each query's summaries stand for."""
    B, S = doc.shape
    blk = tree.cidx.block
    qi = torch.linspace(min(2 * blk, S - 1), S - 1, int(min(queries, S)), device=doc.device).long()
    vis = _visible(tree, X, opened, doc, qi).float()                                     # [B, Q, E]
    live = (doc[:, qi] >= 0).float()
    den = live.sum().clamp_min(1.0)
    out = {"tree/levels": float(tree.levels), "tree/entries": float((vis.sum(-1) * live).sum() / den),
           "tree/summarised_tokens": float(((vis * tree.n[:, None]).sum(-1) * live).sum() / den)}
    for lv in range(1, tree.levels + 1):
        lo, hi = tree.span(lv)
        out[f"tree/entries_l{lv}"] = float((vis[:, :, lo:hi].sum(-1) * live).sum() / den)
    if X is not None:
        out["tree/followed"] = float((X[:, qi].float().sum(-1) * live).sum() / den)
    if opened is not None:
        used = (opened.T[:, qi].to(torch.int32) < int(opened.cap)).sum(-1).float()
        out["tree/opened_concepts"] = float((used * live).sum() / den)
    return out


def plan_tokens(plan: torch.Tensor, tree: TreeIndex, doc: torch.Tensor, lag: int = 2) -> torch.Tensor:
    """Each token gets the plan written after the newest block it may read (its block − `lag`), when that block is in
    its document. plan [B, nb, H], doc [B, S] → [B, S, H] (zero elsewhere)."""
    B, S = doc.shape
    nb = plan.shape[1]
    blk, P = tree.cidx.block, tree.probes
    lo2, _ = tree.span(2)
    bdoc = tree.doc[:, lo2:lo2 + nb * P].view(B, nb, P).amax(-1)                          # [B, nb]
    tb = torch.arange(S, device=doc.device) // blk - lag                                 # [S]
    idx = tb.clamp(0, nb - 1)[None].expand(B, S)
    ok = (tb >= 0)[None] & (tb < nb)[None] & (bdoc.gather(1, idx) == doc) & (doc >= 0)
    out = plan.gather(1, idx[..., None].expand(B, S, plan.shape[-1]))
    return out * ok[..., None].to(out.dtype)


def plan_at(plan: torch.Tensor, tree: TreeIndex, t: int, doc_t: torch.Tensor, lag: int = 2) -> torch.Tensor:
    """`plan_tokens` for the one position t (the cached decoder): plan [B, nb, H], doc_t [B] → [B, H]."""
    B, nb, H = plan.shape
    b = t // tree.cidx.block - lag
    if b < 0 or b >= nb:
        return torch.zeros(B, H, dtype=plan.dtype, device=plan.device)
    P = tree.probes
    lo2, _ = tree.span(2)
    bdoc = tree.doc[:, lo2 + b * P: lo2 + (b + 1) * P].amax(-1)                           # [B]
    ok = (bdoc == doc_t) & (doc_t >= 0)
    return plan[:, b] * ok[:, None].to(plan.dtype)


def route_loss(q: torch.Tensor, tree: TreeIndex, dk: torch.Tensor, doc: torch.Tensor, *, queries: int = 256,
               variance: bool = True) -> torch.Tensor:
    """Parent predicts children: for a sample of queries, a node's own estimated log-mass (through its key
    correction only) regresses the log-sum-exp of its concepts' estimates. q [B, S, h, d], dk [B, E, d]; `tree.key`
    already holds the corrections."""
    B, S, h, d = q.shape
    C1, E, L = tree.n_concepts, tree.n_entries, tree.levels
    blk = tree.cidx.block
    n_q = int(min(queries, max(S - 2 * blk, 1)))
    qi = torch.linspace(2 * blk, S - 1, n_q, device=q.device).long()                     # readers with a memory
    qs = q[:, qi].detach()                                                               # [B, Q, h, d]
    with torch.no_grad():
        t = node_logit(qs, tree, 0, C1, variance=variance).reshape(B, n_q * h, C1)
        tgt = []
        for lv in range(2, L + 1):
            lo, hi = tree.span(lv)
            plo, phi = tree.span(lv - 1)
            t = _lse_to_parents(t, tree.par[:, plo:phi], tree.pfrm[:, plo:phi], lo, hi - lo)
            tgt.append(t)
        tgt = torch.cat(tgt, dim=-1).view(B, n_q, h, E - C1)
        ok = ((tree.frm[:, None, C1:] <= qi[None, :, None]) & (tree.doc[:, None, C1:] == doc[:, qi][..., None])
              & (doc[:, qi][..., None] >= 0))[:, :, None] & torch.isfinite(tgt)
    key = tree.key[:, C1:].detach().float() - dk[:, C1:].detach().float() + dk[:, C1:].float()
    n = tree.n[:, C1:]
    est = torch.einsum("bqhd,bed->bqhe", qs.float(), key) / math.sqrt(d)
    est = est + torch.where(n > 0, n.clamp_min(1.0).log(), torch.zeros_like(n))[:, None, None]
    if variance:
        est = est + ((qs.float() ** 2).sum(-1, keepdim=True) / (2 * d)) * tree.var[:, None, None, C1:].detach()
    err = F.smooth_l1_loss(est, torch.where(ok, tgt, est.detach()), reduction="none")
    return (err * ok).sum() / ok.sum().clamp_min(1) + 0.0 * dk.sum().float()

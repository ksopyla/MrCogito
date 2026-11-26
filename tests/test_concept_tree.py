"""E41 — the concept tree (nn/concept_tree.py): levels of summaries, the top-down read, the reasoner.

What must hold:
  T1  every node's statistics (n, key, spread, value, state) equal brute-force sums over its member tokens, and its
      first readable position is its container's last block + 2 blocks;
  T2  with every node followed the read is the concept read, exactly;
  T3  every past token is counted once for any budgets: the visible summaries' members + the opened tokens;
  T4  no read looks ahead: changing later tokens leaves earlier outputs unchanged (bounded and cut reads, both routes);
  T5  with pure nodes (identical member keys and values) a read cut at level 2 equals full causal attention;
  T6  gradients: a cut read reaches the far tokens' keys and values through the upper levels' summaries;
  T7  the bounded choice follows the nodes with the largest summed concept mass (lookahead), `follow` per query;
  T8  the rows' draw: fixed per row, the shares asked for, a cut row reads nothing below its level.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import nn.concept_tree as ct
from nn.concept_memory import attend_concepts, build_concepts, concept_open_table
from nn.concept_tree import ALL, BIG, RowReads, attend_tree, build_tree, fixed_reads, row_reads, tree_choice
from nn.exact_read import dense_open, doc_starts

BLK, PER, P, FAN = 16, 4, 2, 2


def _sink(doc, n=2):
    ds = doc_starts(doc)
    pos = torch.arange(doc.shape[1])[None]
    return (pos - ds < n) & (doc >= 0), ds


def _inputs(B=2, S=288, d=8, h=3, H=12, seed=0, docs=False):
    torch.manual_seed(seed)
    q, k, v = torch.randn(B, S, h, d), torch.randn(B, S, 1, d), torch.randn(B, S, 1, d)
    x = torch.randn(B, S, H)
    doc = torch.zeros(B, S, dtype=torch.long)
    if docs:
        doc[0, 100:] = 1
    sink, ds = _sink(doc)
    return q, k, v, x, doc, sink, ds


def _tree(k, v, x, doc, sink):
    ci = build_concepts(k, v, x, doc, sink, block=BLK, per_block=PER)
    return build_tree(ci, probes=P, fan=FAN)


def _members(tree, b):
    """For row b: [E, S] bool, the member tokens of every entry (through the parent links)."""
    ci = tree.cidx
    C1, E = tree.n_concepts, tree.n_entries
    nb = C1 // PER
    S = nb * BLK
    mem = torch.zeros(E, S, dtype=torch.bool)
    lab = ci.labels[b]                                                                   # [nb, BLK]
    for blk in range(nb):
        for o in range(BLK):
            if lab[blk, o] >= 0:
                mem[blk * PER + int(lab[blk, o]), blk * BLK + o] = True
    for lv in range(1, tree.levels):
        lo, hi = tree.span(lv)
        for e in range(lo, hi):
            if tree.pfrm[b, e] < BIG:
                mem[int(tree.par[b, e])] |= mem[e]
    return mem


# ------------------------------------------------------------------ T1 statistics
@pytest.mark.parametrize("docs", [False, True])
def test_nodes_are_exact_sums_of_their_members(docs):
    q, k, v, x, doc, sink, ds = _inputs(seed=1, docs=docs)
    tree = _tree(k, v, x, doc, sink)
    nb = tree.n_concepts // PER
    assert tree.levels == ct.tree_levels(nb, FAN) == 6
    assert [tree.span(lv)[1] - tree.span(lv)[0] for lv in range(1, 7)] == [64, 32, 16, 8, 4, 2]
    for b in range(k.shape[0]):
        mem = _members(tree, b)
        for lv in range(2, tree.levels + 1):
            lo, hi = tree.span(lv)
            cb = FAN ** (lv - 2)
            for e in range(lo, hi):
                m = mem[e].nonzero().squeeze(-1)
                assert tree.n[b, e] == len(m), (lv, e)
                cont = (e - lo) // P
                if len(m) == 0 or tree.doc[b, e] < 0:
                    assert tree.frm[b, e] == BIG
                    continue
                assert tree.frm[b, e] == (cont * cb + cb + 1) * BLK
                assert (m // BLK >= cont * cb).all() and (m // BLK < (cont + 1) * cb).all()
                assert (doc[b, m] == tree.doc[b, e]).all()
                torch.testing.assert_close(tree.key[b, e], k[b, m, 0].mean(0), atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(tree.val[b, e], v[b, m, 0].mean(0), atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(tree.var[b, e], k[b, m, 0].var(0, unbiased=False).mean(), atol=1e-5,
                                           rtol=1e-3)
                torch.testing.assert_close(tree.z[b, e], x[b, m].mean(0), atol=1e-5, rtol=1e-4)
        # a container with two documents is never built: its children stay roots
        if docs and b == 0:
            lo, hi = tree.span(2)
            mixed = 100 // BLK                                                           # the block holding the cut
            assert (tree.frm[b, lo + mixed * P: lo + (mixed + 1) * P] == BIG).all()
            lo3, _ = tree.span(3)
            assert (tree.frm[b, lo3 + (mixed // FAN) * P: lo3 + (mixed // FAN + 1) * P] == BIG).all()


# ------------------------------------------------------------------ T2 flat = the concept read
@pytest.mark.parametrize("docs", [False, True])
def test_every_node_followed_is_the_concept_read(docs):
    q, k, v, x, doc, sink, ds = _inputs(seed=2, docs=docs)
    B, S = doc.shape
    ci = build_concepts(k, v, None, doc, sink, block=BLK, per_block=PER)
    opened = concept_open_table(q, ci, top_k=24, width=12)
    ref = attend_concepts(q, k, v, cidx=ci, dv=None, doc=doc, key_valid=None, backend="sdpa", opened=opened,
                          sink=sink, ds=ds, n_sinks=2)
    tree = build_tree(ci, probes=P, fan=FAN)
    reads = fixed_reads(B, tree.levels, follow=0, top_k=24, device=doc.device)
    assert reads.flat(24)
    tab, X = tree_choice(q, tree, doc, reads, top_k=24, width=12)
    assert X is None and torch.equal(tab.T, opened.T)
    out = attend_tree(q, k, v, tree=tree, X=X, opened=tab, doc=doc, sink=sink, ds=ds, n_sinks=2, key_valid=None,
                      backend="sdpa")
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    # the same with the followed table spelled out (every readable node followed)
    f = reads.follow.clone()
    f[:, 2:] = 10_000
    tab2, X2 = tree_choice(q, tree, doc, RowReads(follow=f, cap=reads.cap), top_k=24, width=12)
    out2 = attend_tree(q, k, v, tree=tree, X=X2, opened=tab2, doc=doc, sink=sink, ds=ds, n_sinks=2, key_valid=None,
                       backend="sdpa")
    # a row of one document: the same choice and the same read. A row that packs two documents differs on purpose:
    # the concept read ranks other documents' concepts too (they open nothing but take places among `width`);
    # the tree's choice only ranks nodes of the query's document.
    rows = slice(1, 2) if docs else slice(0, B)
    assert X2 is not None and torch.equal(tab2.T[rows], opened.T[rows])
    torch.testing.assert_close(out2[rows], ref[rows], atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------ T3 counted once
@pytest.mark.parametrize("route", ["lookahead", "descent"])
@pytest.mark.parametrize("follow,cap", [(1, 0), (2, 0), (3, 4096), (ALL, 0), (0, 0)])
def test_every_past_token_is_counted_once(route, follow, cap):
    q, k, v, x, doc, sink, ds = _inputs(seed=3)
    B, S = doc.shape
    tree = _tree(k, v, None, doc, sink)
    L = tree.levels
    f = torch.full((B, L + 1), follow, dtype=torch.long)
    if follow == 0:                                                                      # a cut at level 3
        f[:, 4:] = ALL
    reads = RowReads(follow=f, cap=torch.full((B,), cap, dtype=torch.long))
    top_k = 4096
    tab, X = tree_choice(q, tree, doc, reads, top_k=top_k, width=64, route=route)
    qs = torch.arange(S)
    vis = ct._visible(tree, X, tab, doc, qs)                                             # [B, S, E]
    summ = (vis.float() * tree.n[:, None, :]).sum(-1)
    op = torch.zeros(B, S)
    if tab is not None:
        op = dense_open(tab.T, tab.index.codes, cap=tab.cap, raw_window=BLK, doc=doc).sum(-1).float()
    far = ((qs // BLK - 1) * BLK).clamp_min(0)                                           # tokens before the window
    want = torch.stack([(~sink[b, :]).cumsum(0)[(far - 1).clamp_min(0)] * (far > 0) for b in range(B)]).float()
    torch.testing.assert_close(summ + op, want)
    if follow == 0:                                                                      # nothing below level 3 read
        lo3, _ = tree.span(3)
        # an entry below the cut is read only while its parent cannot be read yet (a root for that query)
        assert not (vis[:, :, :lo3] & (qs[None, :, None] >= tree.pfrm[:, None, :lo3])).any()
        lo2, _ = tree.span(2)
        assert not vis[:, :, :lo2].any()                                                 # a concept's parent closes with it


# ------------------------------------------------------------------ T4 no look-ahead
@pytest.mark.parametrize("route", ["lookahead", "descent"])
def test_reads_never_look_ahead(route):
    q, k, v, x, doc, sink, ds = _inputs(seed=4, docs=True)
    B, S = doc.shape
    t = 200

    def run(q, k, v):
        tree = _tree(k, v, None, doc, sink)
        f = torch.full((B, tree.levels + 1), 2, dtype=torch.long)
        f[1, 2:4] = 0                                                                    # row 1: cut at level 3
        reads = RowReads(follow=f, cap=torch.tensor([24, 0]))
        tab, X = tree_choice(q, tree, doc, reads, top_k=24, width=12, route=route)
        return attend_tree(q, k, v, tree=tree, X=X, opened=tab, doc=doc, sink=sink, ds=ds, n_sinks=2,
                           key_valid=None, backend="sdpa")

    a = run(q, k, v)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    torch.manual_seed(99)
    q2[:, t:], k2[:, t:], v2[:, t:] = torch.randn_like(q[:, t:]), torch.randn_like(k[:, t:]), torch.randn_like(v[:, t:])
    b = run(q2, k2, v2)
    torch.testing.assert_close(a[:, :t], b[:, :t], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(a[:, t:], b[:, t:])


# ------------------------------------------------------------------ T5 pure nodes
def test_pure_nodes_cut_at_level_two_is_full_attention():
    torch.manual_seed(5)
    B, S, d, h = 1, 192, 8, 3
    doc = torch.zeros(B, S, dtype=torch.long)
    sink, ds = _sink(doc)
    blk = torch.arange(S) // BLK
    half = (torch.arange(S) % 2)                                                         # two keys per block
    pk, pv = F.normalize(torch.randn(S // BLK, 2, d), dim=-1) * 3, torch.randn(S // BLK, 2, d)
    k, v = pk[blk, half][None, :, None].clone(), pv[blk, half][None, :, None].clone()
    q = torch.randn(B, S, h, d)
    tree = _tree(k, v, None, doc, sink)
    lo, hi = tree.span(2)
    assert float(tree.var[:, lo:hi][tree.n[:, lo:hi] > 0].max()) < 1e-6                  # level-2 nodes are pure
    f = torch.full((B, tree.levels + 1), ALL, dtype=torch.long)
    f[:, 2] = 0
    tab, X = tree_choice(q, tree, doc, RowReads(follow=f, cap=torch.zeros(B, dtype=torch.long)), top_k=8, width=4)
    assert tab is None
    out = attend_tree(q, k, v, tree=tree, X=X, opened=None, doc=doc, sink=sink, ds=ds, n_sinks=2, key_valid=None,
                      backend="sdpa")
    qt = q.transpose(1, 2)
    kt, vt = k.transpose(1, 2).expand(B, h, S, d), v.transpose(1, 2).expand(B, h, S, d)
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    ref = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=causal[None, None]).transpose(1, 2)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-4)


# ------------------------------------------------------------------ T6 gradients
def test_a_cut_read_reaches_far_keys_and_values():
    q, k, v, x, doc, sink, ds = _inputs(seed=6)
    B, S = doc.shape
    k, v = k.requires_grad_(), v.requires_grad_()
    tree = _tree(k, v, None, doc, sink)
    f = torch.full((B, tree.levels + 1), ALL, dtype=torch.long)
    f[:, 2:4] = 0                                                                        # cut at level 3
    tab, X = tree_choice(q, tree, doc, RowReads(follow=f, cap=torch.zeros(B, dtype=torch.long)), top_k=8, width=4)
    out = attend_tree(q, k, v, tree=tree, X=X, opened=None, doc=doc, sink=sink, ds=ds, n_sinks=2, key_valid=None,
                      backend="sdpa")
    out[:, -BLK:].pow(2).sum().backward()                                                # the last block's queries
    far = slice(2, S - 3 * BLK)                                                          # outside their window
    assert float(k.grad[:, far].abs().sum()) > 0 and float(v.grad[:, far].abs().sum()) > 0
    assert torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


# ------------------------------------------------------------------ T7 the bounded choice
def test_lookahead_follows_the_largest_summed_mass():
    q, k, v, x, doc, sink, ds = _inputs(seed=7)
    B, S = doc.shape
    tree = _tree(k, v, None, doc, sink)
    L, C1 = tree.levels, tree.n_concepts
    reads = fixed_reads(B, L, follow=2, top_k=24, device=doc.device)
    tab, X = tree_choice(q, tree, doc, reads, top_k=24, width=12, route="lookahead")
    qi = S - 1
    sc1 = ct.node_logit(q[:, qi:qi + 1], tree, 0, C1).amax(2)[:, 0]                      # [B, C1]
    for b in range(B):
        mem = _members(tree, b)
        lo, hi = tree.span(L)
        cmem = torch.stack([(mem[e][None] & mem[:C1]).any(-1) for e in range(lo, hi)])   # [n_top, C1]
        mass = torch.stack([torch.logsumexp(sc1[b][m], 0) if m.any() else torch.tensor(-math.inf) for m in cmem])
        got = X[b, qi, lo - C1:hi - C1]
        assert int(got.sum()) == min(2, int(torch.isfinite(mass).sum()))
        assert set(got.nonzero().squeeze(-1).tolist()) == set(mass.topk(int(got.sum())).indices.tolist())
        for lv in range(2, L):                                                           # at most `follow` per level
            lo, hi = tree.span(lv)
            assert int(X[b, qi, lo - C1:hi - C1].sum()) <= 2


# ------------------------------------------------------------------ T8 the rows' draw
def test_row_reads_are_fixed_per_row_and_follow_the_mix():
    torch.manual_seed(8)
    ids = torch.randint(0, 5000, (4000, 64))
    L = 5
    r1 = row_reads(ids, L, mix=(0.4, 0.2, 0.4), follow=16, top_k=512)
    r2 = row_reads(ids, L, mix=(0.4, 0.2, 0.4), follow=16, top_k=512)
    assert torch.equal(r1.follow, r2.follow) and torch.equal(r1.cap, r2.cap)
    normal = (r1.follow[:, 2:] == ALL).all(-1) & (r1.cap == 512)
    bounded = (r1.follow[:, 2:] == 16).all(-1) & (r1.cap == 512)
    cut = ~normal & ~bounded
    assert abs(float(normal.float().mean()) - 0.4) < 0.04 and abs(float(bounded.float().mean()) - 0.2) < 0.04
    assert (r1.cap[cut] <= 64).all()
    f = r1.follow[cut]
    for row in f[:200]:
        zero = (row[2:] == 0).nonzero().squeeze(-1)
        if len(zero):                                                                    # zeros are the lowest levels
            assert zero.tolist() == list(range(len(zero)))
        assert (row[2 + len(zero):] == ALL).all()
    assert (r1.cap[cut & (r1.follow[:, 2] == 0)] == 0).all()                             # a cut above level 1: closed

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


# ------------------------------------------------------------------ T9 the streamed read
@pytest.mark.parametrize("docs", [False, True])
@pytest.mark.parametrize("follow,cap,route", [(ALL, 24, "lookahead"), (2, 24, "lookahead"), (1, 24, "descent"),
                                              (2, 0, "lookahead"), (0, 0, "lookahead")])
def test_the_streamed_read_equals_the_table_read(docs, follow, cap, route):
    """Long rows read chunk by chunk without the [B, S, ·] tables: the same output, also for part of the queries,
    with token updates, a cut, and gradients."""
    from nn.concept_tree import tree_read_stream

    q, k, v, x, doc, sink, ds = _inputs(seed=11, docs=docs)
    B, S = doc.shape
    k, v = k.requires_grad_(), v.requires_grad_()
    tree = _tree(k, v, None, doc, sink)
    f = torch.full((B, tree.levels + 1), follow, dtype=torch.long)
    if follow == 0:
        f[:, 4:] = ALL                                                                   # a cut at level 3
    reads = RowReads(follow=f, cap=torch.full((B,), cap, dtype=torch.long))
    dv_tok = 0.3 * torch.randn(B, S, 1, k.shape[-1])
    tab, X = tree_choice(q, tree, doc, reads, top_k=24, width=12, route=route)
    ref = attend_tree(q, k, v, tree=tree, X=X, opened=tab, doc=doc, sink=sink, ds=ds, n_sinks=2, key_valid=None,
                      backend="sdpa", dv_tok=dv_tok)
    kw = dict(tree=tree, reads=reads, doc=doc, ds=ds, n_sinks=2, key_valid=None, top_k=24, width=12, route=route,
              dv_tok=dv_tok)
    out = tree_read_stream(q, k, v, chunk=7, **kw)
    if docs and follow != ALL and cap > 0:
        rows = slice(1, 2)  # a packed row: the table read's flat choice ranks other documents' concepts (see T2)
    else:
        rows = slice(0, B)
    if not (docs and follow == ALL):                                                     # T2's note again
        torch.testing.assert_close(out[rows], ref[rows], atol=2e-5, rtol=1e-4)
    part = tree_read_stream(q[:, 100:150], k, v, q_offset=100, **kw)
    torch.testing.assert_close(part, out[:, 100:150], atol=2e-5, rtol=1e-4)
    g_ref = torch.autograd.grad(ref[rows].pow(2).sum(), [k, v], retain_graph=True)
    g_out = torch.autograd.grad(out[rows].pow(2).sum(), [k, v])
    if not (docs and follow == ALL):
        for a, b in zip(g_out, g_ref):
            torch.testing.assert_close(a, b, atol=5e-4, rtol=5e-3)


# ------------------------------------------------------------------ T8 the rows' draw
def test_row_reads_are_fixed_per_row_and_follow_the_mix():
    torch.manual_seed(8)
    ids = torch.randint(0, 5000, (4000, 64))
    L = 5
    r1 = row_reads(ids, L, mix=(0.4, 0.2, 0.4), follow=16, top_k=512)
    r2 = row_reads(ids, L, mix=(0.4, 0.2, 0.4), follow=16, top_k=512)
    assert torch.equal(r1.follow, r2.follow) and torch.equal(r1.cap, r2.cap)
    normal = (r1.follow[:, 2:] == ALL).all(-1) & (r1.cap == 512)
    bounded = (r1.follow[:, 2:] > 0).all(-1) & (r1.cap == 512)
    assert set(r1.follow[bounded, 2].tolist()) == {4, 8, 16, 64}                         # the budget is drawn
    assert (r1.follow[bounded, 2:] == r1.follow[bounded, 2:3]).all()                     # the same at every level
    assert 0.3 < float(r1.descent[bounded].float().mean()) < 0.7 and not r1.descent[~bounded].any()
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


# =================================================================== the model
#   M1  a tree model with every new part at its starting value is the concept-memory model (same weights), in
#       evaluation and in training with all rows read as normal;
#   M2  no look-ahead through the whole model: bounded reads, cut rows in training mode, the reasoner with nonzero
#       outputs, the plan, token updates;
#   M3  the reasoner learns from zero (its outputs get gradient, its inputs none yet), the auxiliary losses are
#       finite and reach the predictor; gradient checkpointing gives the same loss and gradients;
#   M4  the codebooks move only between forwards, and a short training run lowers the next-concept loss;
#   M5  short rows (no closed block) run, and every reasoner parameter stays in the graph.
from nn.perceiver_ar_lm import PerceiverARLM  # noqa: E402
from tests.test_latent_memory import cfg  # noqa: E402

CONCEPT = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
               par_global_read="concept", concept_block=BLK, concept_per_block=PER, concept_sinks=2,
               concept_top_k=8, concept_iters=3)
TREE = dict(concept_tree=True, tree_probes=P, tree_fan=FAN)
REASON = dict(tree_reasoner_layers=1, tree_reasoner_dim=16, tree_reasoner_heads=2, tree_plan=True)
AUX = dict(tree_next_concept=0.1, tree_route_loss=0.1, tree_codes_coarse=8, tree_codes_fine=16, tree_code_dim=8,
           tree_route_queries=16)


def _model(seed=0, warm=False, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(cfg(**CONCEPT, **kw)).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    rs = m.layers[m.config.global_layer_index].attn.tree_reasoner
    if rs is not None and warm:                                                          # nonzero corrections and plan
        for lin in (rs.out_k, rs.out_v, rs.out_k1, rs.out_v1, rs.plan):
            if lin is not None:
                lin.weight.data.normal_(0, 0.2)
    return m


def _ids(S=192, B=2, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 60, (B, S), generator=g)


@pytest.mark.parametrize("kw", [TREE, {**TREE, **REASON}, {**TREE, **REASON, **AUX, "tree_token_update": True}],
                         ids=["read", "reasoner", "all"])
def test_a_fresh_tree_model_is_the_concept_model(kw):
    base = _model()
    m = _model(**kw)
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("tree_reasoner" in k for k in missing)
    ids = _ids()
    with torch.no_grad():
        torch.testing.assert_close(m(ids).logits, base(ids).logits, atol=1e-5, rtol=1e-4)
    m.train(), base.train()
    lm = m(ids, labels=ids).loss
    lb = base(ids, labels=ids).loss
    if "tree_next_concept" in kw:                                                        # the extra losses are added
        st = m.pop_train_stats()
        torch.testing.assert_close(torch.tensor(st["tree/next_word_loss"]), lb.detach().cpu(), atol=1e-5, rtol=1e-5)
        assert float(lm) > float(lb)
    else:
        torch.testing.assert_close(lm, lb, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kw", [
    {**TREE, "tree_follow": 2},
    {**TREE, "tree_follow": 1, "tree_route": "descent"},
    {**TREE, **REASON, "tree_follow": 2, "tree_token_update": True},
    {**TREE, **REASON, **AUX, "tree_row_mix": "0.2,0.3,0.5", "tree_train_follow": 2},
], ids=["bounded", "descent", "reasoner_bounded", "training_rows"])
def test_the_tree_model_never_reads_ahead(kw):
    m = _model(warm=True, **kw)
    if "tree_row_mix" in kw:
        m.train()                                                                        # the rows' draws are active
    ids = _ids(S=224, B=3)
    with torch.no_grad():
        base = m(ids).logits
        for t in range(40, 224, 23):
            ids2 = ids.clone()
            ids2[:, t:] = (ids2[:, t:] + 7) % 57 + 3
            d = (base - m(ids2).logits)[:, :t].abs().amax(-1)
            assert (d < 1e-4).all(), (t, d.nonzero()[:5].tolist())


def test_a_draw_per_step_is_the_same_for_every_row_and_changes_with_the_step():
    ids = _ids(S=224, B=6)
    seen = set()
    for step in range(1, 200):
        r = row_reads(ids, 5, mix=(0.4, 0.2, 0.4), follow=16, top_k=512, block=BLK, step=step)
        assert (r.follow == r.follow[:1]).all() and (r.cap == r.cap[:1]).all()
        seen.add((tuple(r.follow[0].tolist()), int(r.cap[0])))
    assert len(seen) >= 8                                                                # normal, 4 budgets, cuts
    normal = sum(1 for f, c in seen if all(x == ALL for x in f[2:]) and c == 512)
    assert normal == 1
    m = _model(**{**TREE, **REASON, **AUX, "tree_row_mix": "0.4,0.2,0.4", "tree_row_draw": "step"})
    att = m.layers[m.config.global_layer_index].attn
    m.train()
    m(ids, labels=ids).loss.backward()
    m(ids, labels=ids).loss.backward()
    assert att._tree_step == 2
    m.eval()
    m(ids)
    assert att._tree_step == 2                                                           # evaluation does not count


def test_training_rows_draw_depends_on_the_row_only():
    """The draw must not change between a forward and its recomputation, nor between loop passes."""
    ids = _ids(S=224, B=6)
    a = row_reads(ids, 5, mix=(0.2, 0.3, 0.5), follow=2, top_k=8, block=BLK)
    ids2 = ids.clone()
    ids2[:, 2 * BLK:] = 3                                                                # only the first two blocks count
    b = row_reads(ids2, 5, mix=(0.2, 0.3, 0.5), follow=2, top_k=8, block=BLK)
    assert torch.equal(a.follow, b.follow) and torch.equal(a.cap, b.cap)
    assert len({tuple(r.tolist()) for r in a.follow}) > 1                                # rows differ


def test_the_reasoner_learns_from_zero_and_checkpointing_agrees():
    kw = {**TREE, **REASON, **AUX, "tree_row_mix": "0.4,0.2,0.4", "tree_train_follow": 2}
    import copy

    m = _model(**kw)
    att = m.layers[m.config.global_layer_index].attn
    rs = att.tree_reasoner
    ids = _ids(S=224, B=3)
    m.train()
    m2 = copy.deepcopy(m)                                                                # the same state, recomputed
    out = m(ids, labels=ids)
    out.loss.backward()
    st = m.pop_train_stats()
    assert all(math.isfinite(v) for v in st.values()) and st["tree/next_concept_fine"] > 0
    for lin in (rs.out_k, rs.out_v, rs.out_k1, rs.out_v1, rs.plan, rs.head_c, rs.head_f):
        assert lin.weight.grad is not None and float(lin.weight.grad.abs().sum()) > 0, lin
    assert float(rs.inp.weight.grad.abs().sum()) > 0                                     # through the predictor
    assert att.wk.weight.grad is not None and float(att.wk.weight.grad.abs().sum()) > 0
    g_ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    loss_ref = float(out.loss)
    m2.gradient_checkpointing_enable()
    out2 = m2(ids, labels=ids)
    out2.loss.backward()
    assert abs(float(out2.loss) - loss_ref) < 1e-5
    worst = 0.0
    for n, p in m2.named_parameters():
        if n in g_ref:
            assert p.grad is not None, n
            worst = max(worst, float((p.grad - g_ref[n]).abs().max() / g_ref[n].abs().max().clamp_min(1e-8)))
    assert worst < 2e-3, worst


def test_codebooks_move_between_forwards_and_the_prediction_improves():
    kw = {**TREE, **REASON, **AUX}
    m = _model(**kw)
    rs = m.layers[m.config.global_layer_index].attn.tree_reasoner
    ids = _ids(S=224, B=4)
    m.train()
    book0 = rs.book_f.clone()
    m(ids, labels=ids)
    assert torch.equal(rs.book_f, book0) and float(rs.pend_f.abs().sum()) > 0            # kept for the next forward
    m(ids, labels=ids)
    assert not torch.equal(rs.book_f, book0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for step in range(30):
        opt.zero_grad()
        m(ids, labels=ids).loss.backward()
        opt.step()
        st = m.pop_train_stats()
        nc = st["tree/next_concept_coarse"] + st["tree/next_concept_fine"]
        first = nc if first is None else first
        last = nc
    assert last < first - 0.05, (first, last)
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(ids).logits).all()
    book1 = rs.book_f.clone()
    m(ids)                                                                               # evaluation moves nothing
    assert torch.equal(rs.book_f, book1)


def test_memory_modes_switch_the_tree_read_and_put_it_back():
    from evaluation.long_context_use import memory_mode

    m = _model(warm=True, **{**TREE, **REASON})
    att = m.layers[m.config.global_layer_index].attn
    ids = _ids(S=224, B=2)
    with torch.no_grad():
        on = m(ids).logits
        outs = {}
        for mode in ("tree1", "descent1", "cut1", "cut3", "sumonly"):
            with memory_mode(m, mode):
                outs[mode] = m(ids).logits
            assert (att.tree["follow"], att.tree["route"], att.tree["cut"]) == (0, "lookahead", 0)
            assert not torch.allclose(outs[mode][:, 64:], on[:, 64:], atol=1e-4), mode
        torch.testing.assert_close(outs["cut1"], outs["sumonly"], atol=1e-5, rtol=1e-4)   # the closed book
        torch.testing.assert_close(m(ids).logits, on)
    with pytest.raises(ValueError, match="tree read"):
        with memory_mode(_model(), "tree4"):
            pass


@pytest.mark.parametrize("kw", [{}, {"tree_follow": 2}, {"tree_follow": 1, "tree_route": "descent"}],
                         ids=["flat", "bounded", "descent"])
def test_the_list_switch_reads_the_same(kw):
    m = _model(warm=True, **{**TREE, **REASON, "tree_token_update": True, **kw})
    ids = _ids(S=224, B=2)
    with torch.no_grad():
        want = m(ids).logits
        with m.open_impl("list"):
            got = m(ids).logits
    torch.testing.assert_close(got, want, atol=2e-4, rtol=1e-3)


def test_a_model_cast_to_another_float_type_still_reads_and_decodes():
    """Generation casts the model (bf16 on a GPU); buffers and fp32 intermediates must not clash. Here: fp64."""
    from nn.incremental import IncrementalDecoder

    m = _model(warm=True, **{**TREE, **REASON, **AUX, "tree_follow": 2, "tree_token_update": True}).double()
    ids = _ids(S=224, B=2)
    with torch.no_grad():
        out = m(ids).logits
        assert torch.isfinite(out).all()
        dec = IncrementalDecoder(m)
        buf = torch.zeros(2, 232, dtype=torch.long)
        buf[:, :224] = ids
        doc = torch.full((2, 232), -1, dtype=torch.long)
        doc[:, :224] = 0
        dec.prefill(buf, doc, 224)
        h = dec.step(torch.tensor([5, 9]))
        assert torch.isfinite(h).all()
    m.train()                                                                            # and it still trains
    loss = m(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)


def test_short_rows_run_and_keep_the_reasoner_in_the_graph():
    m = _model(**{**TREE, **REASON, **AUX})
    rs = m.layers[m.config.global_layer_index].attn.tree_reasoner
    ids = _ids(S=40, B=2)                                                                # no closed block
    m.train()
    out = m(ids, labels=ids)
    out.loss.backward()
    assert math.isfinite(float(out.loss))
    assert all(p.grad is not None for p in rs.parameters())

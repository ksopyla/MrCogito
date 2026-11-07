"""E37 — concept memory read (nn/concept_memory.py): encoder → reasoner → decoder read.

What must hold:
  T1  the balanced clustering labels every valid key, never a sink or pad, and stays near balanced;
  T2  concept statistics equal brute-force sums over the labelled members;
  T3  with pure concepts (identical member keys and values) the summary read equals full causal attention;
  T4  opening every member equals full causal attention for any keys (opened concepts leave the summaries);
  T5  the model never reads ahead (later tokens do not move earlier logits), with and without the reasoner;
  T6  gradients: the summaries reach the members' keys and values; the reasoner's update starts at zero and learns.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import nn.concept_memory as cm
from nn.concept_memory import attend_concepts, build_concepts, concept_open_table, sinkhorn_labels
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg

BLK, PER = 16, 4     # tiny block: 16 keys, 4 concepts of ~4
CONCEPT = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
               par_global_read="concept", concept_block=BLK, concept_per_block=PER, concept_sinks=2,
               concept_top_k=8, concept_iters=3)


def dense_causal(q, k, v, doc):
    """Full causal attention within each document. q [B,S,h,d], k/v [B,S,1,d]."""
    B, S, h, d = q.shape
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2).expand(B, h, S, d), v.transpose(1, 2).expand(B, h, S, d)
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None]
    mask = (j <= i)[None] & (doc[:, :, None] == doc[:, None, :]) & (doc[:, :, None] >= 0)
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask[:, None]).transpose(1, 2)


def _sink(doc, n):
    from nn.exact_read import doc_starts

    ds = doc_starts(doc)
    pos = torch.arange(doc.shape[1])[None]
    return (pos - ds < n) & (doc >= 0), ds


# ------------------------------------------------------------------ T1 clustering
def test_clustering_labels_valid_keys_and_stays_balanced():
    torch.manual_seed(0)
    N, L, d, C = 6, 64, 16, 8
    keys = F.normalize(torch.randn(N, L, d), dim=-1)
    valid = torch.ones(N, L, dtype=torch.bool)
    valid[0, :5] = False
    lab = sinkhorn_labels(keys, valid, C)
    assert lab.shape == (N, L)
    assert (lab[~valid] == -1).all() and (lab[valid] >= 0).all() and (lab[valid] < C).all()
    sizes = torch.stack([(lab == c).sum(1) for c in range(C)], 1)[1:]                   # rows with all valid
    assert sizes.max() <= 3 * L // C, sizes                                              # near balanced
    # well-separated groups are recovered exactly
    proto = F.normalize(torch.randn(C, d), dim=-1)
    grp = torch.arange(L) % C
    keys2 = proto[grp][None].expand(2, L, d).contiguous()
    lab2 = sinkhorn_labels(keys2, torch.ones(2, L, dtype=torch.bool), C)
    for c in range(C):
        assert lab2[0, grp == c].unique().numel() == 1


# ------------------------------------------------------------------ T2 statistics
def test_statistics_equal_brute_force_sums():
    torch.manual_seed(1)
    B, S, d, H = 2, 80, 8, 12
    k, v, x = torch.randn(B, S, 1, d), torch.randn(B, S, 1, d), torch.randn(B, S, H)
    doc = torch.zeros(B, S, dtype=torch.long)
    sink, _ = _sink(doc, 2)
    ci = build_concepts(k, v, x, doc, sink, block=BLK, per_block=PER)
    nb = (S - 1) // BLK - 1
    assert ci.n_concepts == nb * PER
    lab = sinkhorn_labels(F.normalize(k[:, : nb * BLK, 0], dim=-1).reshape(B * nb, BLK, d),
                          (~sink[:, : nb * BLK]).reshape(B * nb, BLK), PER).view(B, nb * BLK)
    for b in range(B):
        for c in range(ci.n_concepts):
            blk, slot = divmod(c, PER)
            pos = torch.arange(blk * BLK, (blk + 1) * BLK)
            m = pos[lab[b, blk * BLK:(blk + 1) * BLK] == slot]
            assert ci.n[b, c] == len(m)
            if len(m):
                torch.testing.assert_close(ci.mu_k[b, c], k[b, m, 0].mean(0))
                torch.testing.assert_close(ci.mu_v[b, c], v[b, m, 0].mean(0))
                torch.testing.assert_close(ci.var[b, c], k[b, m, 0].var(0, unbiased=False).mean(), atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(ci.z[b, c], x[b, m].mean(0))


# ------------------------------------------------------------------ T3 / T4 exactness
def _pure_inputs(S=96, d=8, h=3, seed=2, docs=False):
    """Keys and values identical within each (block, group); groups are the clustering's own labels."""
    torch.manual_seed(seed)
    B = 1
    doc = torch.zeros(B, S, dtype=torch.long)
    if docs:
        doc[:, 40:] = 1
    sink, ds = _sink(doc, 2)
    proto_k = torch.randn(S // BLK, PER, d)
    proto_v = torch.randn(S // BLK, PER, d)
    grp = torch.arange(S) % PER
    blk = torch.arange(S) // BLK
    k = (proto_k[blk, grp] + 0.0)[None, :, None]
    v = proto_v[blk, grp][None, :, None]
    q = torch.randn(B, S, h, d)
    return q, k, v, doc, sink, ds


def test_pure_concepts_summary_read_equals_full_attention():
    q, k, v, doc, sink, ds = _pure_inputs()
    ci = build_concepts(k, v, None, doc, sink, block=BLK, per_block=PER)
    assert float(ci.var.max()) < 1e-6                                                   # concepts are pure
    out = attend_concepts(q, k, v, cidx=ci, dv=None, doc=doc, key_valid=None, backend="sdpa", opened=None,
                          sink=sink, ds=ds, n_sinks=2)
    torch.testing.assert_close(out, dense_causal(q, k, v, doc), atol=1e-5, rtol=1e-4)


def test_opening_everything_equals_full_attention_for_any_keys():
    torch.manual_seed(3)
    S, d, h = 96, 8, 3
    doc = torch.zeros(1, S, dtype=torch.long)
    doc[:, 50:] = 1
    sink, ds = _sink(doc, 2)
    q, k, v = torch.randn(1, S, h, d), torch.randn(1, S, 1, d), torch.randn(1, S, 1, d)
    ci = build_concepts(k, v, None, doc, sink, block=BLK, per_block=PER)
    op = concept_open_table(q, ci, top_k=S, width=ci.n_concepts)
    out = attend_concepts(q, k, v, cidx=ci, dv=None, doc=doc, key_valid=None, backend="sdpa", opened=op,
                          sink=sink, ds=ds, n_sinks=2)
    torch.testing.assert_close(out, dense_causal(q, k, v, doc), atol=1e-5, rtol=1e-4)


def test_summary_mass_estimate_tracks_the_true_log_sum_exp():
    """With concepts of nearby keys the second-order estimate is closer to the true log-sum-exp than the first."""
    torch.manual_seed(4)
    d = 8
    q = torch.randn(1, 64, 2, d)
    base = torch.randn(1, 96, 1, d)
    k = base + 0.0
    k[:, :, 0] = F.normalize(torch.randn(1, 96, d), dim=-1) * 0.0 + base[:, :, 0]
    doc = torch.zeros(1, 96, dtype=torch.long)
    sink, _ = _sink(doc, 0)
    ci = build_concepts(k, torch.randn_like(k), None, doc, sink, block=BLK, per_block=PER)
    est2 = cm.summary_logit(q, ci, variance=True)
    est1 = cm.summary_logit(q, ci, variance=False)
    nb = ci.n_concepts // PER
    lab = sinkhorn_labels(F.normalize(k[:, : nb * BLK, 0], dim=-1).reshape(nb, BLK, d),
                          torch.ones(nb, BLK, dtype=torch.bool), PER).view(nb * BLK)
    true = torch.full_like(est2, float("-inf"))
    lg = torch.einsum("bshd,btd->bsht", q, k[:, : nb * BLK, 0]) / d ** 0.5
    for c in range(ci.n_concepts):
        blk, slot = divmod(c, PER)
        m = torch.arange(blk * BLK, (blk + 1) * BLK)[lab[blk * BLK:(blk + 1) * BLK] == slot]
        if len(m):
            true[..., c] = torch.logsumexp(lg[..., m], dim=-1)
    ok = torch.isfinite(true)
    assert (est2[ok] - true[ok]).abs().mean() < (est1[ok] - true[ok]).abs().mean()


# ------------------------------------------------------------------ T5 no future leak
def _model(**kw):
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(**CONCEPT, **kw)).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    if m.layers[m.config.global_layer_index].attn.concept_reasoner is not None:
        m.layers[m.config.global_layer_index].attn.concept_reasoner.out.weight.data.normal_(0, 0.2)
    return m


@pytest.mark.parametrize("kw", [{}, {"concept_reasoner_layers": 1, "concept_reasoner_dim": 16,
                                     "concept_reasoner_heads": 2}], ids=["no_reasoner", "reasoner"])
def test_no_future_leak(kw):
    m = _model(**kw)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(3, 60, (1, 96), generator=g)
    with torch.no_grad():
        base = m(ids).logits
        for t in range(40, 96, 9):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 57 + 3
            d = (base - m(ids2).logits)[0, :t].abs().amax(-1)
            assert (d < 1e-5).all(), (t, d.nonzero().flatten()[:5].tolist())


# ------------------------------------------------------------------ T6 gradients
def test_gradients_reach_members_and_the_reasoner_learns():
    m = _model(concept_reasoner_layers=1, concept_reasoner_dim=16, concept_reasoner_heads=2)
    att = m.layers[m.config.global_layer_index].attn
    att.concept_reasoner.out.weight.data.zero_()
    g = torch.Generator().manual_seed(6)
    ids = torch.randint(3, 60, (1, 96), generator=g)
    m.train()
    out = m(ids, labels=ids)
    out.loss.backward()
    assert att.wk.weight.grad is not None and att.wk.weight.grad.abs().sum() > 0
    assert att.concept_reasoner.out.weight.grad.abs().sum() > 0                          # learns from zero
    assert att.concept_reasoner.inp.weight.grad.abs().sum() == 0                         # nothing flows yet (out = 0)


def test_reasoner_update_reaches_opened_members_and_summaries():
    """dv = 0 leaves the read unchanged (the two copies agree); a nonzero dv changes the read of old queries through
    both the opened members and the summaries, and never the queries whose memory is empty."""
    torch.manual_seed(8)
    S, d, h = 96, 8, 3
    doc = torch.zeros(1, S, dtype=torch.long)
    sink, ds = _sink(doc, 2)
    q, k, v = torch.randn(1, S, h, d), torch.randn(1, S, 1, d), torch.randn(1, S, 1, d)
    ci = build_concepts(k, v, None, doc, sink, block=BLK, per_block=PER)
    op = concept_open_table(q, ci, top_k=8, width=ci.n_concepts)
    kw = dict(cidx=ci, doc=doc, key_valid=None, backend="sdpa", opened=op, sink=sink, ds=ds, n_sinks=2)
    base = attend_concepts(q, k, v, dv=None, **kw)
    zero = torch.zeros(1, ci.n_concepts, d)
    same = attend_concepts(q, k, v, dv=zero, dv_tok=cm.token_update(ci, zero, S), **kw)
    torch.testing.assert_close(same, base, atol=1e-6, rtol=1e-5)
    dv = torch.randn(1, ci.n_concepts, d)
    tok = cm.token_update(ci, dv, S)
    moved = attend_concepts(q, k, v, dv=dv, dv_tok=tok, **kw)
    # the first block reads only itself (raw copy); from block 1 on, the previous block carries its update
    assert (moved[0, :BLK] - base[0, :BLK]).abs().max() < 1e-6
    assert (moved[0, 3 * BLK:] - base[0, 3 * BLK:]).abs().max() > 1e-3
    # members get their own concept's update; tokens outside concept blocks and sinks get none
    nb = ci.labels.shape[1]
    for t in (5, 20, 40):
        lab = ci.labels[0].reshape(-1)[t]
        want = dv[0, (t // BLK) * PER + lab] if lab >= 0 else torch.zeros(d)
        torch.testing.assert_close(tok[0, t, 0], want)
    assert tok[0, nb * BLK:].abs().max() == 0 and tok[0, :2].abs().max() == 0


def test_eval_switch_reads_window_and_sinks_only():
    m = _model()
    att = m.layers[m.config.global_layer_index].attn
    ids = torch.randint(3, 60, (1, 96), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        on = m(ids).logits
        att.concept_off = True
        off = m(ids).logits
        att.concept_off = False
    assert (on[0, :2 * BLK] - off[0, :2 * BLK]).abs().max() < 1e-5                    # nothing old yet
    assert (on[0, 3 * BLK:] - off[0, 3 * BLK:]).abs().max() > 1e-4                     # concepts are read later

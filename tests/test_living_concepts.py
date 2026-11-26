"""E39 — living concepts (nn/living_concepts.py): spans, document-wide concepts, the two-level read. CPU, tiny sizes;
the flex path is tested against the streamed read on a GPU (tests/test_living_concepts_cuda.py).

What must hold:
  L1  the batched span cut and leader rule equal the pre-flight reference (scripts/e39_preflight.py) row by row,
      also with the concept slots full;
  L2  the bounds: a span has 1 … span_max tokens, a stride at most span_cap spans, no span crosses a stride or a
      document; members are exactly the tokens a later reader sees as memory (no sink, no padding, the document
      still runs two strides on); concepts never pass C_max, and with the slots full a founder joins a concept;
  L3  span statistics and the concepts' versions equal brute-force sums, with a document change inside the row;
  L4  every past token of the document is counted once for every reader (window, sinks, opened, remaining counts),
      and the open table opens exactly the counted tokens;
  L5  identical keys and values inside each concept → the read equals full causal attention; opening everything →
      the same for any keys;
  L6  spans only, scored by mass, opening whole spans: the living read equals the concept read with the same labels;
  L7  the model never reads ahead; padding readers read zero; gradients reach the members' keys and values through
      the summaries;
  L8  the thresholds: frozen in evaluation, moving toward their targets in training (once per forward, also with
      gradient checkpointing), and their counters summed over the cards in every training forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nn.concept_memory import attend_concepts, build_concepts, concept_open_table
from nn.exact_read import ORD_MOD, doc_starts
from nn.living_concepts import (build_living, living_open_table, living_read_stream, living_stats, servo_step)
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_concept_memory import dense_causal
from tests.test_latent_memory import cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import e39_preflight as P  # noqa: E402

STRIDE, SMAX, SCAP, SINKS = 16, 4, 8, 2   # tiny geometry: 16-token strides, spans of ≤ 4, ≤ 8 spans per stride
LIVING = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
              par_global_read="living", living_stride=STRIDE, living_span_max=SMAX, living_span_cap=SCAP,
              living_keep=3, living_width=8, concept_sinks=SINKS, concept_top_k=8, living_theta=0.3,
              living_lam=0.5)
GEO = P.Geo(stride=STRIDE, span_max=SMAX, span_cap=SCAP, keep=3, cap=8, width=8, sinks=SINKS)


def _sink(doc):
    ds = doc_starts(doc)
    pos = torch.arange(doc.shape[1])[None]
    return (pos - ds < SINKS) & (doc >= 0), ds


def _run_keys(B, S, d, pool=5, noise=0.5, seed=0, max_run=9):
    """Keys in runs of 1 … max_run tokens around a few shared directions (so spans rejoin earlier concepts)."""
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(pool, d, generator=g), dim=-1)
    out = torch.empty(B, S, d)
    for b in range(B):
        t = 0
        while t < S:
            r = int(torch.randint(1, max_run + 1, (1,), generator=g))
            p = int(torch.randint(0, pool, (1,), generator=g))
            out[b, t:t + r] = base[p]
            t += r
    return F.normalize(out + noise * torch.randn(B, S, d, generator=g) / d ** 0.5, dim=-1)


def _two_docs(S=160, cut=70, pad=10):
    doc = torch.zeros(2, S, dtype=torch.long)
    doc[:, cut:] = 1
    doc[:, S - pad:] = -1
    doc[1] = 0                       # the second row: one document, no padding
    return doc


# ------------------------------------------------------------------ L1 the reference
@pytest.mark.parametrize("c_max", [0, 3], ids=["room", "full"])
def test_batched_grouping_equals_the_reference(c_max):
    B, n_st, d, theta, lam = 3, 6, 8, 0.45, 0.5
    S = (n_st + 2) * STRIDE                                   # n_st memory strides
    L = n_st * STRIDE
    k = _run_keys(B, S, d)
    doc = torch.zeros(B, S, dtype=torch.long)
    sink, _ = _sink(doc)
    lidx = build_living(k[:, :, None], doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=theta,
                        lam=lam, levels=2, c_max=c_max)
    assert lidx.n_strides == n_st
    G = n_st * SCAP
    for b in range(B):
        member = ~sink[b, :L]
        sid, n_spans, _ = P.cut_spans(k[b, :L], member, doc[b, :L], theta, GEO)
        cof, nC, after, *_ = P.leader_concepts(k[b, :L], sid, n_spans, lam, GEO, c_max=c_max or None)
        slot = lidx.tok_slot[b]
        assert ((slot < G) == member).all()
        uniq = torch.unique(slot[member])                     # span slots in position order
        assert uniq.numel() == n_spans
        assert torch.equal(torch.searchsorted(uniq, slot[member]), sid[member])
        assert torch.equal(lidx.cof[b, uniq], cof)
        assert torch.equal(lidx.n_c[b], after.to(lidx.n_c.dtype))
        assert int(lidx.n_c[b, -1]) == nC
    if c_max:
        assert int(lidx.n_c.max()) == c_max                   # the slots filled up, later founders joined


# ------------------------------------------------------------------ L2 bounds and membership
def test_bounds_and_membership():
    S, d = 160, 8
    doc = _two_docs(S)
    sink, _ = _sink(doc)
    k = _run_keys(2, S, d, seed=1, max_run=2)                # many natural cuts: the per-stride cap acts
    lidx = build_living(k[:, :, None], doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.6,
                        lam=0.5, levels=2)
    n_st = (S - 1) // STRIDE - 1
    L = n_st * STRIDE
    G = n_st * SCAP
    pos = torch.arange(L)
    for b in range(2):
        ahead = doc[b, (pos // STRIDE + 2) * STRIDE]
        want = (doc[b, :L] == ahead) & (ahead >= 0) & ~sink[b, :L]
        slot = lidx.tok_slot[b]
        assert torch.equal(slot < G, want)
        for g in torch.unique(slot[slot < G]).tolist():
            p = pos[slot == g]
            assert 1 <= p.numel() <= SMAX
            assert torch.equal(p, torch.arange(int(p[0]), int(p[0]) + p.numel()))      # contiguous
            assert (p // STRIDE).unique().numel() == 1                                  # one stride
            assert doc[b, p].unique().numel() == 1                                      # one document
            assert int(lidx.span_first[b, g]) == int(p[0]) and float(lidx.span_n[b, g]) == p.numel()
        per = (lidx.span_n[b].view(n_st, SCAP) > 0).sum(-1)
        assert (per <= SCAP).all()
        assert ((lidx.cof[b] >= 0) == (lidx.span_n[b] > 0)).all()
    # stride 3 of row 0 ([48, 64)) holds document 0 only, which ends before stride 5 starts: no members
    assert (lidx.tok_slot[0, 48:64] == G).all()
    # the per-stride cap on natural cuts acted somewhere (runs of 1-2 tokens give more dips than room for cuts):
    # such a stride still holds at most span_cap spans
    kh = F.normalize(k, dim=-1)
    cos = F.pad((kh[:, 1:] * kh[:, :-1]).sum(-1), (1, 0), value=-2.0)[:, :L]
    inner = (lidx.tok_slot < G) & F.pad(lidx.tok_slot[:, :-1] < G, (1, 0), value=False) & (pos % STRIDE != 0)
    dips = (inner & (cos < 0.6)).view(2, n_st, STRIDE).sum(-1)
    assert int(dips.max()) > SCAP - STRIDE // SMAX
    # concept slots: a new document starts again at 0, and a small C_max is never passed
    small = build_living(k[:, :, None], doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.6,
                         lam=0.99, levels=2, c_max=3)
    assert int(small.n_c.max()) <= 3 and int(small.cof.max()) < 3
    assert ((small.cof >= 0) == (small.span_n > 0)).all()                               # every span placed


# ------------------------------------------------------------------ L3 statistics
def test_statistics_and_versions_equal_brute_force_sums():
    torch.manual_seed(2)
    S, d = 160, 8
    doc = _two_docs(S)
    sink, _ = _sink(doc)
    k, v = torch.randn(2, S, 1, d), torch.randn(2, S, 1, d)
    for levels in (1, 2):
        lidx = build_living(k, doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.0, lam=0.3,
                            levels=levels)
        st = living_stats(k, v, lidx)
        n_st, C = lidx.n_strides, lidx.c_max
        L = n_st * STRIDE
        G = n_st * SCAP
        for b in range(2):
            slot = lidx.tok_slot[b]
            for g in torch.unique(slot[slot < G]).tolist():
                m = (slot == g).nonzero().flatten()
                torch.testing.assert_close(st.span_mu_k[b, g], k[b, m, 0].mean(0))
                torch.testing.assert_close(st.span_var[b, g], k[b, m, 0].var(0, unbiased=False).mean(),
                                           atol=1e-5, rtol=1e-4)
            sstride = torch.arange(G) // SCAP
            for s in range(n_st):
                dd = int(lidx.sdoc[b, s])
                for c in range(C):
                    gs = ((lidx.cof[b] == c) & (sstride <= s) & (lidx.span_doc[b] == dd) & (dd >= 0)).nonzero()
                    m = torch.isin(slot, gs.flatten()) & (slot < G)
                    n = int(m.sum())
                    assert float(lidx.ver_n[b, s, c]) == n
                    if n:
                        idx = m.nonzero().flatten()
                        torch.testing.assert_close(st.ver_mu_k[b, s, c], k[b, idx, 0].mean(0))
                        torch.testing.assert_close(st.ver_mu_v[b, s, c], v[b, idx, 0].mean(0))
                        torch.testing.assert_close(st.ver_var[b, s, c], k[b, idx, 0].var(0, unbiased=False).mean(),
                                                   atol=1e-5, rtol=1e-4)
        assert L == n_st * STRIDE


# ------------------------------------------------------------------ L4 counted once
@pytest.mark.parametrize("search", ["two_level", "flat"])
def test_every_past_token_is_counted_once(search):
    torch.manual_seed(3)
    S, d, h = 160, 8, 3
    doc = _two_docs(S)
    sink, ds = _sink(doc)
    k, v, q = torch.randn(2, S, 1, d), torch.randn(2, S, 1, d), torch.randn(2, S, h, d)
    lidx = build_living(k, doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.0, lam=0.3)
    st = living_stats(k, v, lidx)
    tab, n_open = living_open_table(q, lidx, st, doc, top_k=7, width=8, keep=2, search=search)
    gi = lidx.groups
    codes = gi.codes[0].long()
    g_of = torch.div(codes, ORD_MOD, rounding_mode="floor")
    for b in range(2):
        for t in range(2 * STRIDE, S):
            dt = int(doc[b, t])
            if dt < 0:
                assert int(n_open[b, t].sum()) == 0
                continue
            tau = t // STRIDE
            past = (doc[b, :t + 1] == dt)
            j = torch.arange(t + 1)
            window = past & (j >= (tau - 1) * STRIDE)
            sinks = past & sink[b, :t + 1] & ~window
            ver = tau - 2
            mem_n = lidx.ver_n[b, ver] if int(lidx.sdoc[b, ver]) == dt else torch.zeros(lidx.c_max)
            rem = mem_n - n_open[b, t].float()
            assert (rem >= 0).all()
            assert int(n_open[b, t].sum()) <= 7
            assert int(window.sum() + sinks.sum() + mem_n.sum()) == int(past.sum())
            # the open table opens exactly the counted tokens, all of them older than the window
            ok = codes[b, :t + 1] >= 0
            row = tab.T[b, t].long()
            hit = ok & (row.gather(0, g_of[b, :t + 1].clamp(min=0)) + (codes[b, :t + 1] - g_of[b, :t + 1] * ORD_MOD)
                        < 7)
            assert int(hit.sum()) == int(n_open[b, t].sum())
            assert not (hit & window).any() and (doc[b, :t + 1][hit] == dt).all()


# ------------------------------------------------------------------ L5 exactness
def _pure(S=128, d=8, h=3, seed=4):
    """Runs of 4 tokens aligned to the strides, each run one of 5 orthogonal directions, keys and values identical
    inside a direction: the spans are the runs and the concepts are pure."""
    g = torch.Generator().manual_seed(seed)
    proto_k = torch.linalg.qr(torch.randn(d, d, generator=g))[0][:5] * 2.0
    proto_v = torch.randn(5, d, generator=g)
    which = torch.randint(0, 5, (S // 4,), generator=g).repeat_interleave(4)
    k = proto_k[which][None, :, None]
    v = proto_v[which][None, :, None]
    q = torch.randn(1, S, h, d, generator=g)
    doc = torch.zeros(1, S, dtype=torch.long)
    return q, k, v, doc


def test_pure_concepts_read_equals_full_attention():
    q, k, v, doc = _pure()
    sink, ds = _sink(doc)
    lidx = build_living(k, doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.5, lam=0.5)
    st = living_stats(k, v, lidx)
    assert float(st.ver_var.max()) < 1e-5 and int(lidx.n_c[0, -1]) <= 5                 # pure, ≤ 5 concepts
    out = living_read_stream(q, k, v, lidx=lidx, st=st, doc=doc, ds=ds, n_sinks=SINKS, key_valid=None, top_k=8,
                             width=8, keep=2, stride=STRIDE)
    torch.testing.assert_close(out, dense_causal(q, k, v, doc), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("levels", [1, 2])
def test_opening_everything_equals_full_attention_for_any_keys(levels):
    torch.manual_seed(5)
    S, d, h = 160, 8, 3
    doc = _two_docs(S)
    sink, ds = _sink(doc)
    q, k, v = torch.randn(2, S, h, d), torch.randn(2, S, 1, d), torch.randn(2, S, 1, d)
    lidx = build_living(k, doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.0, lam=0.3, levels=levels)
    st = living_stats(k, v, lidx)
    out = living_read_stream(q, k, v, lidx=lidx, st=st, doc=doc, ds=ds, n_sinks=SINKS, key_valid=None, top_k=S,
                             width=lidx.groups.n_groups, keep=lidx.c_max, stride=STRIDE)
    want = dense_causal(q, k, v, doc)
    real = doc >= 0
    torch.testing.assert_close(out[real], want[real], atol=1e-5, rtol=1e-4)
    assert (out[~real] == 0).all()                                                     # padding readers read zero


# ------------------------------------------------------------------ L6 spans only = the concept read
def test_spans_only_by_mass_equals_the_concept_read_with_the_same_labels():
    """One document per row: the concept read's choice also ranks other documents' concepts (they open nothing
    there, nn/exact_read.choose counts the query's document only), the living read ranks the reader's own only."""
    torch.manual_seed(6)
    S, d, h = 160, 8, 3
    doc = torch.zeros(2, S, dtype=torch.long)
    sink, ds = _sink(doc)
    q, k, v = torch.randn(2, S, h, d), torch.randn(2, S, 1, d), torch.randn(2, S, 1, d)
    lidx = build_living(k, doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.0, lam=0.3, levels=1)
    st = living_stats(k, v, lidx)
    # two whole spans opened (a budget no span pair can fill), the rest as one summary each
    got = living_read_stream(q, k, v, lidx=lidx, st=st, doc=doc, ds=ds, n_sinks=SINKS, key_valid=None, top_k=100,
                             width=2, keep=1, stride=STRIDE, search="flat", score="mass")
    cidx = build_concepts(k, v, None, doc, sink, block=STRIDE, per_block=SCAP, labels=lidx.labels)
    op = concept_open_table(q, cidx, top_k=100, width=2)
    want = attend_concepts(q, k, v, cidx=cidx, dv=None, doc=doc, key_valid=None, backend="sdpa", opened=op,
                           sink=sink, ds=ds, n_sinks=SINKS, block=STRIDE)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
    # and the model's own spans-only switch takes the concept read with those labels
    m = _model(living_levels=1, living_score="mass")
    assert m(torch.randint(3, 60, (1, 96))).logits.isfinite().all()


# ------------------------------------------------------------------ L7 the model
def _model(**kw):
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(**{**LIVING, **kw})).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


@pytest.mark.parametrize("kw", [{}, {"living_levels": 1}, {"living_search": "flat"}],
                         ids=["living", "spans", "flat"])
def test_no_future_leak(kw):
    m = _model(**kw)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(3, 60, (1, 112), generator=g)
    with torch.no_grad():
        base = m(ids).logits
        for t in range(40, 112, 9):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 57 + 3
            d = (base - m(ids2).logits)[0, :t].abs().amax(-1)
            assert (d < 1e-5).all(), (t, d.nonzero().flatten()[:5].tolist())


def test_gradients_reach_members_through_the_summaries():
    torch.manual_seed(7)
    S, d, h = 128, 8, 3
    doc = torch.zeros(1, S, dtype=torch.long)
    sink, ds = _sink(doc)
    q = torch.randn(1, S, h, d)
    k = torch.randn(1, S, 1, d, requires_grad=True)
    v = torch.randn(1, S, 1, d, requires_grad=True)
    lidx = build_living(k.detach(), doc, sink, stride=STRIDE, span_max=SMAX, span_cap=SCAP, theta=0.0, lam=0.3)
    st = living_stats(k, v, lidx)
    out = living_read_stream(q, k, v, lidx=lidx, st=st, doc=doc, ds=ds, n_sinks=SINKS, key_valid=None, top_k=1,
                             width=1, keep=1, stride=STRIDE)
    out[:, 6 * STRIDE:].pow(2).sum().backward()             # readers whose window holds no stride before 4
    gk, gv = k.grad[0, :, 0].abs().sum(-1), v.grad[0, :, 0].abs().sum(-1)
    mem = lidx.tok_slot[0, :3 * STRIDE] < lidx.groups.n_groups                       # members of strides 0-2
    assert (gk[:3 * STRIDE][mem] > 0).all() and (gv[:3 * STRIDE][mem] > 0).all()
    assert torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


# ------------------------------------------------------------------ L8 thresholds
def test_servo_step_direction_and_bounds():
    v = torch.tensor(0.5)
    assert float(servo_step(v, torch.tensor(0.25), 0.125, 0.01)) == pytest.approx(0.49)   # too many cuts: lower θ
    assert float(servo_step(v, torch.tensor(0.0), 0.125, 0.01)) == pytest.approx(0.51)    # too few: raise it
    assert float(servo_step(torch.tensor(0.999), torch.tensor(0.0), 0.125, 0.01)) == 1.0  # kept in [−1, 1]


@pytest.mark.parametrize("start,up", [(-0.9, True), (0.9, False)])
def test_cut_threshold_moves_toward_its_target_in_training_only(start, up):
    m = _model(living_theta=start, living_span_target=3.0, living_servo_step=0.01)
    att = m.layers[m.config.global_layer_index].attn
    ids = torch.randint(3, 60, (2, 320), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        m(ids)
    assert float(att.living_theta) == pytest.approx(start)                             # frozen in evaluation
    m.train()
    for _ in range(3):
        m(ids)
    moved = float(att.living_theta) - start
    assert (moved > 0) == up and 0 < abs(moved) <= 0.03 + 1e-6                          # at most a step per forward
    stats = m.pop_train_stats()
    assert stats["living/theta"] == pytest.approx(float(att.living_theta)) and stats["living/span_len"] > 0
    assert m.pop_train_stats() == {}                                                    # the sums were reset


def test_join_threshold_moves_toward_its_target():
    m = _model(living_lam=0.99, living_servo_step=0.01)
    att = m.layers[m.config.global_layer_index].attn
    m.train()
    ids = torch.randint(3, 60, (2, 320), generator=torch.Generator().manual_seed(9))
    m(ids)
    assert float(att.living_lam) < 0.99                                                 # nearly every span founded


def test_counters_are_summed_once_per_forward_even_without_memory(monkeypatch):
    """Every training forward calls the same collective on every card (a row too short for memory sends zeros),
    and gradient checkpointing's recomputation neither rebuilds the grouping nor moves the thresholds again."""
    calls = []

    def fake_all_reduce(t, op=None):
        calls.append(t.clone())
        t.mul_(2)                                         # a second card with the same counters

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    m = _model(living_servo_step=0.01)
    att = m.layers[m.config.global_layer_index].attn
    m.train()
    m(torch.randint(3, 60, (1, 24)))                     # no stride two back: no memory, zeros sent
    assert len(calls) == 1 and float(calls[0].abs().sum()) == 0
    m.gradient_checkpointing_enable()
    before = float(att.living_theta)
    out = m(torch.randint(3, 60, (2, 320)), labels=torch.randint(3, 60, (2, 320)))
    out.loss.backward()
    assert len(calls) == 2                               # the recomputation in the backward sent nothing
    assert 0 < abs(float(att.living_theta) - before) <= 0.01 + 1e-6

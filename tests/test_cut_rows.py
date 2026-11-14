"""E37c cut rows (nn/perceiver_ar_lm.py: `cut_ids`, `par_cut_*`): two agents inside one training row.

What must hold, for full attention, LSA's read and E37's concept read:
  T1  no cut = the plain forward (an all-False `cut_ids`, and the switches on in eval mode);
  T2  one cut after A's text: the positions after it equal agent B of the two-agent decoder (`nn/agent_split.py`,
      split on), and the positions before it equal the plain forward (A never sees the cut);
  T3  the local layers stop at a cut: with the reading layer's output switched off, the side after a cut equals its
      text alone;
  T4  several cuts chain: every side equals the single-cut row that ends at it;
  T5  training draws cuts with the configured gaps, a gradient step runs, and eval draws none;
  T6  the draw: gaps inside [min, max], the share of cut rows, document starts and padding in the side ids;
  T7  models the cut cannot serve are refused (a notebook, n-gram input tables).
"""
from __future__ import annotations

import pytest
import torch

from nn.agent_split import SplitDecoder, row_for
from nn.perceiver_ar_lm import PerceiverARLM, cut_sides, draw_cuts
from tests.test_agent_split import ARMS, EOS, LA, LB, NEW, b_states, make, texts
from tests.test_latent_memory import cfg


def row(seed=1):
    """A's text, B's text and a fixed continuation as one row [1, LA + LB + NEW]."""
    a, q = texts(seed)
    g = torch.Generator().manual_seed(9)
    return torch.cat([a, q, torch.randint(3, 80, (NEW,), generator=g)])[None], a, q


def cut_at(S, *pos):
    c = torch.zeros(1, S, dtype=torch.bool)
    for p in pos:
        c[0, p] = True
    return c


@pytest.mark.parametrize("arm", list(ARMS))
def test_no_cut_is_the_plain_forward(arm):
    full, _, _ = row()
    doc = torch.zeros_like(full)
    m = make(arm)
    want = m.hidden_states(full, doc_ids=doc)
    assert torch.equal(m.hidden_states(full, doc_ids=doc, cut_ids=cut_at(full.shape[1])), want)
    on = make(arm, par_cut_gap_min=4, par_cut_gap_max=32)  # same seed: same weights; eval mode draws nothing
    assert torch.equal(on.hidden_states(full, doc_ids=doc), want)
    assert torch.equal(on(full, doc_ids=doc).logits, m(full, doc_ids=doc).logits)


@pytest.mark.parametrize("arm", list(ARMS))
def test_cut_row_is_the_two_agent_decoder(arm):
    m = make(arm)
    full, a, q = row()
    doc = torch.zeros_like(full)
    got = m.hidden_states(full, doc_ids=doc, cut_ids=cut_at(full.shape[1], LA))[0]
    plain = m.hidden_states(full, doc_ids=doc)[0]
    assert torch.allclose(got[:LA], plain[:LA], atol=1e-5), "A's side must not see the cut"
    assert not torch.allclose(got[LA:], plain[LA:], atol=1e-3), "B's side must change"
    dec = SplitDecoder(m)
    dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
    dec.listen(split=True)
    want = b_states(dec, q)
    assert torch.allclose(got[LA:], want, atol=2e-4, rtol=2e-4), float((got[LA:] - want).abs().max())


@pytest.mark.parametrize("arm", list(ARMS))
def test_local_layers_stop_at_the_cut(arm):
    """With the reading layer's output switched off, nothing of A can reach B in a cut row: B's side equals B's text
    alone (the local layers' positions are relative). Without the cut the local layers read A's last tokens."""
    m = make(arm)
    m.layers[m.config.global_layer_index].attn.wo.weight.data.zero_()
    full, _, _ = row()
    doc = torch.zeros_like(full)
    alone = m.hidden_states(full[:, LA:], doc_ids=doc[:, LA:])[0]
    got = m.hidden_states(full, doc_ids=doc, cut_ids=cut_at(full.shape[1], LA))[0, LA:]
    assert torch.allclose(got, alone, atol=2e-4, rtol=2e-4), float((got - alone).abs().max())
    plain = m.hidden_states(full, doc_ids=doc)[0, LA:]
    assert not torch.allclose(plain, alone, atol=1e-3)


@pytest.mark.parametrize("arm", list(ARMS))
def test_cuts_chain(arm):
    m = make(arm)
    full, _, _ = row()
    S = full.shape[1]
    doc = torch.zeros_like(full)
    c1, c2 = 70, LA
    both = m.hidden_states(full, doc_ids=doc, cut_ids=cut_at(S, c1, c2))[0]
    first = m.hidden_states(full, doc_ids=doc, cut_ids=cut_at(S, c1))[0]
    assert torch.allclose(both[:c2], first[:c2], atol=1e-5)          # causal: a later cut changes nothing before it
    assert not torch.allclose(both[c2:], first[c2:], atol=1e-3)


@pytest.mark.parametrize("arm", list(ARMS))
def test_training_draws_cuts(arm):
    torch.manual_seed(0)
    m = make(arm, par_cut_gap_min=8, par_cut_gap_max=48)
    full, _, _ = row()
    ids = torch.cat([full, row(2)[0]])
    doc = torch.zeros_like(ids)
    m.train()
    out = m(ids, labels=ids.clone(), doc_ids=doc)
    cuts = m._last_cuts
    assert cuts is not None and cuts.shape == ids.shape and bool(cuts.any(1).all())
    for r in cuts:
        at = torch.cat([torch.zeros(1, dtype=torch.long), r.nonzero()[:, 0]])
        gaps = at[1:] - at[:-1]
        assert int(gaps.min()) >= 8 and int(gaps.max()) <= 48
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    m(ids, labels=ids.clone(), doc_ids=doc)
    assert m._last_cuts is None
    given = cut_at(ids.shape[1], 40).expand(2, -1)
    m.train()
    m(ids, labels=ids.clone(), doc_ids=doc, cut_ids=given)  # given cuts win over the draw
    assert torch.equal(m._last_cuts, given)


def test_draw_and_sides():
    g = torch.Generator().manual_seed(0)
    cuts = draw_cuts(64, 4096, 32, 1024, 1.0, generator=g)
    assert cuts.shape == (64, 4096) and not bool(cuts[:, :32].any())
    gaps = []
    for r in cuts:
        at = torch.cat([torch.zeros(1, dtype=torch.long), r.nonzero()[:, 0]])
        gaps.append(at[1:] - at[:-1])
    gaps = torch.cat(gaps).float()
    assert int(gaps.min()) >= 32 and int(gaps.max()) <= 1024
    # log-uniform: about half of the gaps below the geometric middle (181)
    assert 0.4 < float((gaps < 181).float().mean()) < 0.6
    half = draw_cuts(400, 256, 8, 64, 0.5, generator=g).any(1).float().mean()
    assert 0.4 < float(half) < 0.6
    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 1, -1, -1]])
    c = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    assert cut_sides(doc, c).tolist() == [[0, 0, 1, 2, 2, 3, 3, -1, -1]]
    assert cut_sides(doc, torch.zeros_like(c)).tolist() == [[0, 0, 0, 1, 1, 1, 1, -1, -1]]


def test_refusals():
    with pytest.raises(ValueError, match="cut rows"):
        PerceiverARLM(cfg(**dict(ARMS["dense"], ngram_orders=(2,), par_cut_gap_min=4, par_cut_gap_max=32)))
    with pytest.raises(ValueError, match="par_cut"):
        cfg(**dict(ARMS["dense"], ngram_orders=(), par_cut_gap_min=64, par_cut_gap_max=32))
    from tests.test_latent_memory import lm_cfg  # a notebook model: pages straddle a cut

    m = PerceiverARLM(lm_cfg()).eval()
    ids = torch.randint(3, 80, (1, 64))
    with pytest.raises(ValueError, match="cut rows"):
        m.hidden_states(ids, doc_ids=torch.zeros_like(ids), cut_ids=cut_at(64, 30))

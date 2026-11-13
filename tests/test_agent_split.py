"""Two agents from one model (nn/agent_split.py): A reads the document, B answers from A's memory.

What must hold, for full attention, LSA's read and E37's concept read:
  T1  with the split off, A reading and then B stepping through its text writes the tokens of one row decoded the
      usual way, and B's hidden states equal that row's;
  T2  with the split on, nothing of A's reaches B except the reading layer's memory: scrambling every local layer's
      cache of A's tokens leaves B's hidden states unchanged (and with the split off it moves them);
  T3  A's memory hidden too = B decoding its own text alone;
  T4  B's answer depends on which document A read;
  T5  the coarse grid: 16 bits is the identity, the error falls as the bits rise, 1 bit keeps the signs;
  T6  the traffic counts, and the reads the split cannot serve are refused.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nn.agent_split import SplitDecoder, coarse, row_for
from nn.incremental import IncrementalDecoder
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_concept_memory import CONCEPT
from tests.test_latent_memory import cfg
from tests.test_lsa_read import DENSE, LSA

ARMS = {"dense": DENSE, "lsa": LSA, "concept": CONCEPT}
EOS = 0
LA, LB, NEW = 150, 14, 10   # the document, B's text, the answer: with blocks of 16-32 the answer crosses a block end


def make(arm, seed=3, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(cfg(**dict(ARMS[arm], ngram_orders=(), pre_layers=2, stack_layers=3, **kw))).eval()
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def texts(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 80, (LA,), generator=g), torch.randint(3, 80, (LB,), generator=g)


def b_states(dec, q, new=NEW):
    """B steps through its text and a fixed continuation; the hidden state after every step [n, d]."""
    g = torch.Generator().manual_seed(9)
    toks = torch.cat([q, torch.randint(3, 80, (new,), generator=g)])
    return torch.stack([dec.step(t.view(1))[0] for t in toks])


@pytest.mark.parametrize("arm", list(ARMS))
def test_relay_equals_one_row(arm):
    m = make(arm)
    a, q = texts()
    buf, doc, L0 = row_for(torch.cat([a, q]), NEW, eos=EOS, pad=16)
    one = IncrementalDecoder(m).greedy(buf.clone(), doc.clone(), L0, NEW, eos=EOS)[0, L0:].tolist()
    dec = SplitDecoder(m)
    dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
    dec.listen(split=False)
    assert dec.answer(q, NEW, eos=EOS) == one
    # hidden states of B's positions against the plain forward over the whole row
    g = torch.Generator().manual_seed(9)
    cont = torch.randint(3, 80, (NEW,), generator=g)
    full = torch.cat([a, q, cont])[None]
    want = m.hidden_states(full, doc_ids=torch.zeros_like(full))[0, LA:]
    dec.listen(split=False)
    got = b_states(dec, q)
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4), float((got - want).abs().max())


@pytest.mark.parametrize("arm", list(ARMS))
def test_split_reads_only_the_memory(arm):
    m = make(arm)
    a, q = texts()
    dec = SplitDecoder(m)
    dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
    dec.listen(split=True)
    clean = b_states(dec, q)
    dec.listen(split=False)
    relay = b_states(dec, q)
    assert not torch.allclose(clean, relay, atol=1e-3)  # the split changes what B computes: A's last tokens are gone
    for key in dec.K:  # every local layer's cache of A's tokens: noise
        if key[0] != dec.gi:
            dec.K[key][:, :LA] = torch.randn_like(dec.K[key][:, :LA])
            dec.V[key][:, :LA] = torch.randn_like(dec.V[key][:, :LA])
    dec.listen(split=True)
    assert torch.equal(b_states(dec, q), clean)
    dec.listen(split=False)
    assert not torch.allclose(b_states(dec, q), relay, atol=1e-3)


@pytest.mark.parametrize("arm", list(ARMS))
def test_blank_memory_is_b_alone(arm):
    m = make(arm)
    a, q = texts()
    dec = SplitDecoder(m)
    dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
    dec.listen(blank=True)
    got = b_states(dec, q)
    g = torch.Generator().manual_seed(9)
    alone = torch.cat([q, torch.randint(3, 80, (NEW,), generator=g)])[None]
    want = m.hidden_states(alone, doc_ids=torch.zeros_like(alone))[0]
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4), float((got - want).abs().max())


@pytest.mark.parametrize("arm", list(ARMS))
def test_answer_depends_on_the_document(arm):
    m = make(arm)
    _, q = texts()
    out = []
    for seed in (1, 2):
        a, _ = texts(seed)
        dec = SplitDecoder(m)
        dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
        dec.listen(split=True)
        out.append(b_states(dec, q))
    assert not torch.allclose(out[0], out[1], atol=1e-3)


def test_coarse_grid():
    x = torch.randn(1, 40, 1, 8)
    assert torch.equal(coarse(x, 16), x) and torch.equal(coarse(x, 0), x)
    err = [float((coarse(x, b) - x).abs().mean()) for b in (1, 2, 4, 8)]
    assert err == sorted(err, reverse=True) and err[-1] < 0.02
    one = coarse(x, 1)
    assert torch.equal(torch.sign(one), torch.sign(x))
    assert one.abs().unique(dim=1).shape[1] == 1  # one magnitude per dimension
    assert len(coarse(x, 3)[0, :, 0, 0].unique()) <= 8


def test_traffic_counts_and_refusals():
    m = make("concept")
    a, q = texts()
    dec = SplitDecoder(m)
    dec.read(*row_for(a, LB + NEW, eos=EOS, pad=16))
    att = m.layers[dec.gi].attn
    assert dec.memory_numbers() == LA * dec.passes * 2 * att.dh  # one key head, one value head
    up, down = dec.remote_read_numbers()
    assert up == dec.passes * att.h * att.dh and down == dec.passes * att.h * (att.dh + 1)
    seen = []
    dec.trace = lambda r, t, keys, summ: seen.append((r, t, int(keys[:, :LA].sum()), summ))
    dec.listen(split=True)
    dec.answer(q, 3, eos=-1)
    assert len(seen) == (LB + 2) * dec.passes and all(t >= LA for _, t, _, _ in seen)
    with pytest.raises(NotImplementedError):
        SplitDecoder(PerceiverARLM(cfg(**dict(DENSE, ngram_orders=(2,)))).eval())
    # coarse memory still answers (the index is rebuilt from the coarse keys)
    dec.trace = None
    dec.listen(split=True, bits=4)
    assert len(dec.answer(q, 3, eos=-1)) == 3
    assert F.mse_loss(dec.K[(dec.gi, 0)][:, :LA].float(), dec._snap["K"][(dec.gi, 0)][:, :LA].float()) > 0

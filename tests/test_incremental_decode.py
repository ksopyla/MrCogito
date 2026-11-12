"""Cached greedy decoding (nn/incremental.py): the row is read once, then each step runs only the new position.

The cached decoding must write the tokens of the uncached harness loop
(`evaluation/lm_eval_perceiver_ar.PerceiverARLMEval._model_generate`, one forward over the fixed-length row per token),
and each step's hidden state must equal that forward's at the same position: full attention, LSA's read (answers that
cross a block end, so the buckets are rebuilt) and the catalogue read (E33t, reuse arm), with a left-padded row.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nn.incremental import IncrementalDecoder, decode_kind
from tests.test_catalogue_read import CAT
from tests.test_catalogue_read import make as make_cat
from tests.test_lsa_read import DENSE, LSA
from tests.test_lsa_read import make as make_lsa
from tests.test_note_open import NOTES, row
from tests.test_note_open import make as make_notes

CAT_REUSE = dict(CAT, lm_open_new_per_pass=False)
DEEP = dict(pre_layers=2, stack_layers=3)


class _Tok:
    eos_token_id = 0

    @staticmethod
    def decode(toks):
        return " ".join(str(t) for t in toks)


def _model(arm, deep=False):
    extra = DEEP if deep else {}
    if arm == "dense":
        return make_lsa(**DENSE, **extra)
    if arm == "lsa":
        return make_lsa(**LSA, **extra)
    if arm == "catalogue":
        return make_cat(seed=2, **CAT_REUSE, **extra)
    raise ValueError(arm)


def _prompt(L0=150, B=2, pad=9):
    ids = row(L0, B=B)
    ids[ids == 0] = 5  # no eos in the context
    attn = torch.ones_like(ids)
    attn[1, :pad] = 0  # the harness left-pads shorter contexts
    ids[1, :pad] = 0
    return ids, attn


def _generate(m, ids, attn, new, monkeypatch, cached, pad=16):
    pytest.importorskip("lm_eval")
    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval

    monkeypatch.setenv("PERCEIVER_AR_GEN_CACHE", "1" if cached else "0")
    fake = SimpleNamespace(model=m, tokenizer=_Tok(), GEN_PAD=pad)
    return PerceiverARLMEval._model_generate(fake, ids.clone(), ids.shape[1] + new, stop=None, attention_mask=attn)


@pytest.mark.parametrize("arm", ["dense", "lsa", "catalogue"])
def test_kinds(arm):
    assert decode_kind(_model(arm)) == arm


@pytest.mark.parametrize("deep", [False, True], ids=["small", "deep"])
@pytest.mark.parametrize("arm", ["dense", "lsa", "catalogue"])
@pytest.mark.parametrize("pad", [16, 256])
def test_cached_decoding_writes_the_uncached_tokens(arm, deep, pad, monkeypatch):
    m = _model(arm, deep)
    ids, attn = _prompt()
    new = 12
    calls = []
    real = IncrementalDecoder.greedy

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(IncrementalDecoder, "greedy", spy)
    want = _generate(m, ids, attn, new, monkeypatch, cached=False, pad=pad)
    assert not calls
    got = _generate(m, ids, attn, new, monkeypatch, cached=True, pad=pad)
    assert calls, "the cached path was not taken"
    assert torch.equal(got, want)


@pytest.mark.parametrize("arm", ["dense", "lsa", "catalogue"])
def test_each_step_equals_the_full_forward(arm):
    """Hidden states of the cached steps vs the uncached forward over the same row at the same position, fp32. The
    LSA prompt ends 10 tokens before a block end (block 32), so the steps cross it and the buckets are rebuilt."""
    m = _model(arm)
    L0, new, Lb = 150, 12, 176
    ids, attn = _prompt(L0)
    buf = torch.zeros(2, Lb, dtype=torch.long)
    buf[:, :L0] = ids
    doc = torch.full((2, Lb), -1, dtype=torch.long)
    doc[:, :L0] = torch.where(attn.bool(), 0, -1)
    torch.manual_seed(7)
    answer = torch.randint(3, 80, (2, new))  # any tokens: the step must follow what was written
    dec = IncrementalDecoder(m)
    with torch.no_grad():
        h = dec.prefill(buf.clone(), doc.clone(), L0)
        for i in range(new):
            want = m.hidden_states(buf, doc_ids=doc)[:, L0 - 1 + i]
            torch.testing.assert_close(h, want, rtol=1e-4, atol=1e-4)
            buf[:, L0 + i] = answer[:, i]
            doc[:, L0 + i] = 0
            h = dec.step(answer[:, i])


def test_catalogue_refuses_answers_past_the_exact_horizon(monkeypatch):
    """The prefill's notes are exact while the answer is shorter than raw_window − lm_window (16 here); a longer
    request falls back to the uncached loop, which still decodes."""
    m = _model("catalogue")
    assert IncrementalDecoder(m).max_new_tokens() == int(m.config.message_raw_window) - int(m.config.lm_window)
    ids, attn = _prompt()
    with pytest.raises(NotImplementedError):
        IncrementalDecoder(m).greedy(torch.zeros(2, 200, dtype=torch.long), torch.zeros(2, 200, dtype=torch.long),
                                     150, 40, eos=0)
    got = _generate(m, ids, attn, 20, monkeypatch, cached=True)
    want = _generate(m, ids, attn, 20, monkeypatch, cached=False)
    assert torch.equal(got, want)


def test_unsupported_reads_fall_back(monkeypatch):
    """E33p's notes read (gist) is not cached yet: the decoder refuses it and the harness keeps the uncached loop."""
    m = make_notes(**NOTES)
    with pytest.raises(NotImplementedError):
        IncrementalDecoder(m)
    ids, attn = _prompt()
    got = _generate(m, ids, attn, 6, monkeypatch, cached=True)
    want = _generate(m, ids, attn, 6, monkeypatch, cached=False)
    assert torch.equal(got, want)

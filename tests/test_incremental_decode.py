"""Cached greedy decoding (nn/incremental.py): the row is read once, then each step runs only the new position.

The cached decoding must write the tokens of the uncached harness loop
(`evaluation/lm_eval_perceiver_ar.PerceiverARLMEval._model_generate`, one forward over the fixed-length row per token),
and each step's hidden state must equal that forward's at the same position: full attention (with and without
RoPE on the reading layer), LSA's read (answers that
cross a block end, so the buckets are rebuilt), the catalogue read (E33t, reuse arm), E33p's notes read (pages rewritten
as the answer grows) and E37's concept read (with the reasoner and the length scale), with a left-padded row.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nn.incremental import IncrementalDecoder, decode_kind
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_catalogue_read import CAT
from tests.test_catalogue_read import make as make_cat
from tests.test_concept_memory import CONCEPT
from tests.test_latent_memory import cfg as lm_base_cfg
from tests.test_lsa_read import DENSE, LSA
from tests.test_lsa_read import make as make_lsa
from tests.test_note_open import NOTES, PAGES, row
from tests.test_note_open import make as make_notes

CAT_REUSE = dict(CAT, lm_open_new_per_pass=False)
DEEP = dict(pre_layers=2, stack_layers=3)
REASONER = dict(concept_reasoner_layers=1, concept_reasoner_dim=16, concept_reasoner_heads=2)
ARMS = ["dense", "dense_rope", "lsa", "catalogue", "notes", "concept", "concept_reasoner", "concept_scaled"]


class _Tok:
    eos_token_id = 0

    @staticmethod
    def decode(toks):
        return " ".join(str(t) for t in toks)


def _model(arm, deep=False):
    extra = DEEP if deep else {}
    if arm == "dense":
        return make_lsa(**DENSE, **extra)
    if arm == "dense_rope":  # E37's fair full-attention control: RoPE on the reading layer too
        return make_lsa(**dict(DENSE, global_nope=False), **extra)
    if arm == "lsa":
        return make_lsa(**LSA, **extra)
    if arm == "catalogue":
        return make_cat(seed=2, **CAT_REUSE, **extra)
    if arm == "notes":  # E33p notes read: pages of 16 every 12 tokens, raw window 32; the answer rewrites its pages
        return make_notes(seed=4, **NOTES, **extra)
    if arm.startswith("concept"):  # E37, blocks of 16: a 150-token prompt and 12 answer tokens cross 160
        kw = dict(CONCEPT, **extra)
        if arm == "concept_reasoner":
            kw.update(REASONER)
        if arm == "concept_scaled":  # the length scale acts past 48 tokens
            kw.update(concept_sum_len=48)
        torch.manual_seed(3)
        m = PerceiverARLM(lm_base_cfg(**kw)).eval()
        for layer in m.layers:
            layer.attn.wo.weight.data.normal_(0, 0.2)
            layer.mlp.down.weight.data.normal_(0, 0.2)
        rs = m.layers[m.config.global_layer_index].attn.concept_reasoner
        if rs is not None:
            rs.out.weight.data.normal_(0, 0.2)
        return m
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


@pytest.mark.parametrize("arm", ARMS)
def test_kinds(arm):
    assert decode_kind(_model(arm)) == arm.split("_")[0]


@pytest.mark.parametrize("deep", [False, True], ids=["small", "deep"])
@pytest.mark.parametrize("arm", ARMS)
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


@pytest.mark.parametrize("arm", ARMS)
def test_each_step_equals_the_full_forward(arm):
    """Hidden states of the cached steps vs the uncached forward over the same row at the same position, fp32. The
    LSA prompt ends 10 tokens before a block end (block 32), so the steps cross it and the buckets are rebuilt; the
    concept prompt crosses a block end of 16 (the concepts and the reasoner's updates are rebuilt)."""
    m = _model(arm)
    L0, new, Lb = 150, 12, 208  # long enough that the block holding the answer has concepts (E37: (Lb − 1) // 16 − 1 blocks)
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


@pytest.mark.parametrize("arm", ["catalogue", "notes"])
def test_note_reads_refuse_answers_past_the_exact_horizon(arm, monkeypatch):
    """A page holding answer tokens has no member old enough to open while the answer is shorter than raw_window −
    lm_window (16 for both test models); a longer request falls back to the uncached loop, which still decodes."""
    m = _model(arm)
    assert IncrementalDecoder(m).max_new_tokens() == int(m.config.message_raw_window) - int(m.config.lm_window)
    ids, attn = _prompt()
    with pytest.raises(NotImplementedError):
        IncrementalDecoder(m).greedy(torch.zeros(2, 200, dtype=torch.long), torch.zeros(2, 200, dtype=torch.long),
                                     150, 40, eos=0)
    got = _generate(m, ids, attn, 20, monkeypatch, cached=True)
    want = _generate(m, ids, attn, 20, monkeypatch, cached=False)
    assert torch.equal(got, want)


@pytest.mark.parametrize("which", ["pages", "catalogue_softmax"])
def test_unsupported_reads_fall_back(which, monkeypatch):
    """Not cached: E33p's page-level opening, and a catalogue scored by a softmax over the visible notes (the pages
    holding answer tokens would count). The decoder refuses them and the harness keeps the uncached loop."""
    m = make_notes(**PAGES) if which == "pages" else make_cat(seed=2, **dict(CAT_REUSE, lm_open_score="sum_softmax"))
    with pytest.raises(NotImplementedError):
        IncrementalDecoder(m)
    ids, attn = _prompt()
    got = _generate(m, ids, attn, 6, monkeypatch, cached=True)
    want = _generate(m, ids, attn, 6, monkeypatch, cached=False)
    assert torch.equal(got, want)


@pytest.mark.parametrize("arm", ARMS)
def test_read_once_answer_many(arm):
    """A document read once, then several questions from its snapshot: each answer is the one a fresh prefill over
    document + question decodes, and asking again from the same snapshot gives the same answer (a question leaves
    nothing behind). Question + answer cross a block end for LSA (32) and the concept read (16) and stay under the
    note reads' exact horizon (16)."""
    m = _model(arm)
    L0, new, Lb = 150, 6, 208
    ids, attn = _prompt(L0)
    torch.manual_seed(11)
    questions = [torch.randint(3, 80, (2, 5)), torch.randint(3, 80, (2, 9))]
    buf = torch.zeros(2, Lb, dtype=torch.long)
    buf[:, :L0] = ids
    doc = torch.full((2, Lb), -1, dtype=torch.long)
    doc[:, :L0] = torch.where(attn.bool(), 0, -1)

    want = []
    for q in questions:  # the reference: the question is part of the prompt the prefill reads
        n = q.shape[1]
        b, d = buf.clone(), doc.clone()
        b[:, L0: L0 + n] = q
        d[:, L0: L0 + n] = 0
        out = IncrementalDecoder(m).greedy(b, d, L0 + n, new, eos=0)
        want.append(out[:, L0 + n:])

    dec = IncrementalDecoder(m)
    dec.prefill(buf.clone(), doc.clone(), L0)
    snap = dec.snapshot()
    got = [dec.ask(snap, q, new, eos=0) for q in questions]
    again = dec.ask(snap, questions[0], new, eos=0)
    for g, w in zip(got, want):
        assert torch.equal(g, w)
    assert torch.equal(again, got[0])
    assert not torch.equal(questions[0][:, :5], questions[1][:, :5])  # the two questions differ


def test_ask_needs_room_and_respects_the_exact_horizon():
    m = _model("dense")
    ids, attn = _prompt(150)
    buf = torch.zeros(2, 160, dtype=torch.long)
    buf[:, :150] = ids
    doc = torch.full((2, 160), -1, dtype=torch.long)
    doc[:, :150] = torch.where(attn.bool(), 0, -1)
    dec = IncrementalDecoder(m)
    dec.prefill(buf, doc, 150)
    snap = dec.snapshot()
    with pytest.raises(ValueError):
        dec.ask(snap, torch.randint(3, 80, (2, 8)), 6, eos=0)  # 14 slots needed, 10 free
    assert dec.ask(snap, torch.randint(3, 80, (2, 4)), 6, eos=0).shape[1] <= 6
    notes = _model("notes")
    d2 = IncrementalDecoder(notes)
    with pytest.raises(NotImplementedError):
        d2.ask({}, torch.randint(3, 80, (2, 12)), 6, eos=0)  # 18 > the horizon of 16


@pytest.mark.parametrize("arm", ["dense", "lsa", "concept", "catalogue"])
def test_document_session_answers_as_a_fresh_prompt(arm):
    """evaluation/doc_session.DocumentSession: one read, two questions, each answer the tokens a fresh cached
    decoding of document + question writes (the row is one document, no padding)."""
    from evaluation.doc_session import DocumentSession

    m = _model(arm)
    ids = _prompt(150)[0][:1]
    torch.manual_seed(5)
    qs = [torch.randint(3, 80, (4,)), torch.randint(3, 80, (7,))]
    s = DocumentSession(m, _Tok(), room=16, pad=16)
    r = s.read(ids[0])
    assert r["tokens"] == 150 and r["row"] == 176
    for q in qs:
        text, sec, toks = s.ask(q, 6)
        n = len(q)
        buf = torch.zeros(1, 176, dtype=torch.long)
        buf[0, :150], buf[0, 150:150 + n] = ids[0], q
        doc = torch.full((1, 176), -1, dtype=torch.long)
        doc[0, :150 + n] = 0
        want = IncrementalDecoder(m).greedy(buf, doc, 150 + n, 6, eos=0)[0, 150 + n:].tolist()
        while want and want[-1] == 0:
            want.pop()
        assert toks == want and text == _Tok.decode(want) and sec >= 0

"""Two eval-time fixes found on E33p stage 1 (2026-11-07).

1. The opening table T [B, S, G] is looked up in query chunks once it reaches 2³¹ elements (flex attention indexes
   captured buffers with 32-bit offsets; the notes arm crashed at 128k). The chunked lookup must answer exactly as
   the single table.
2. RULER's greedy decoding keeps one row length for the whole answer (flex attention recompiled every step and fell
   back to its dense-math path, out of memory at 16k). Where a position's output does not depend on the row's length
   (full attention, LSA's read) it must write the same tokens as growing the row.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import nn.exact_read as er
from tests.test_note_open import NOTES, PAGES, RW, _forward_state, make, packed_docs, row


def _table(S=160):
    from nn.exact_read import note_open_table

    m, ids, ctx = _forward_state(kw=NOTES, S=S, doc=packed_docs())
    no = ctx.index
    torch.manual_seed(4)
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    kap = torch.randn(1, ctx.n_slots, 1, m.config.head_dim)
    tab = note_open_table(q, kap, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    return tab, no


def test_chunked_table_lookup_equals_the_single_table(monkeypatch):
    tab, no = _table()
    S = tab.T.shape[1]
    b = torch.zeros(S, S, dtype=torch.long)
    qi = torch.arange(S)[:, None].expand(S, S)
    kv = torch.arange(S)[None].expand(S, S)
    whole = er.open_term(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)(b, qi, kv)
    G = tab.T.shape[-1]
    for chunk in (1, 7, 64, S - 1):  # chunk lengths, including 1 query per chunk and a ragged last chunk
        monkeypatch.setattr(er, "_FLEX_MAX_NUMEL", chunk * G)
        parts, qc = er._split_queries(tab.T)
        assert qc == chunk and len(parts) == -(-S // chunk)
        assert all(p.numel() <= chunk * G for p in parts)
        got = er.open_term(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)(b, qi, kv)
        assert torch.equal(got, whole)
    assert whole.any() and not whole.all()  # the test sees both opened and closed entries


def test_small_tables_are_not_copied():
    tab, _ = _table()
    parts, qc = er._split_queries(tab.T)
    assert len(parts) == 1 and parts[0] is tab.T and qc == tab.T.shape[1]


# ------------------------------------------------------------------ fixed-length greedy decoding
class _Tok:
    eos_token_id = 0

    @staticmethod
    def decode(toks):
        return " ".join(str(t) for t in toks)


def _grow_the_row(model, ids, attn, new, stop_after=None):
    """The decoding before the fix: the row grows by one token per step."""
    doc = torch.where(attn.bool(), torch.zeros_like(attn), torch.full_like(attn, -1))
    W = model.lm_head.weight
    cfg = model.config
    with torch.no_grad():
        for _ in range(new):
            h = model.hidden_states(ids, doc_ids=doc)[:, -1]
            logits = F.linear(h, W).float()
            if cfg.logit_softcap:
                logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
            ids = torch.cat([ids, logits.argmax(-1)[:, None]], 1)
            doc = torch.cat([doc, torch.zeros_like(doc[:, :1])], 1)
    return ids


def _decode(m, pad, new=12):
    pytest.importorskip("lm_eval")
    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval

    ids = row(150, B=2)
    ids[ids == 0] = 5  # no eos in the context
    attn = torch.ones_like(ids)
    attn[1, :9] = 0  # the harness left-pads shorter contexts
    ids[1, :9] = 0
    fake = SimpleNamespace(model=m, tokenizer=_Tok(), GEN_PAD=pad)
    got = PerceiverARLMEval._model_generate(fake, ids.clone(), ids.shape[1] + new, stop=None, attention_mask=attn)
    return ids, attn, got


@pytest.mark.parametrize("arm", ["dense", "lsa"])
@pytest.mark.parametrize("pad", [16, 256])
def test_fixed_length_decoding_equals_growing_the_row(arm, pad):
    """Models whose output at a position does not depend on the row's length (full attention, LSA's read): the
    fixed-length decoding writes exactly the tokens the growing row wrote."""
    from tests.test_lsa_read import DENSE, LSA
    from tests.test_lsa_read import make as make_lsa

    m = make_lsa(**(DENSE if arm == "dense" else LSA))
    ids, attn, got = _decode(m, pad)
    want = _grow_the_row(m, ids.clone(), attn, got.shape[1] - ids.shape[1])
    for b in range(2):  # equal up to the first eos per row (after it the old loop kept writing, the new one writes eos)
        g, w = got[b, ids.shape[1]:].tolist(), want[b, ids.shape[1]:].tolist()
        n = w.index(0) + 1 if 0 in w else len(w)
        assert g[:n] == w[:n]


@pytest.mark.parametrize("kw", [NOTES, PAGES], ids=["notes", "pages"])
def test_fixed_length_decoding_runs_on_note_models(kw):
    """Note models: the output at position t moves by ~3 % with the row's length (not with later tokens' content;
    the last page is placed at [S − W, S), so the page grid's tail depends on S; training rows of 4k / 16k sit on
    the grid). The fixed-length decoding puts every answer position inside a longer row, as in training; here only
    its mechanics are checked: one shape for every step, the context kept, eos after the first eos."""
    m = make(**kw)
    ids, attn, got = _decode(m, 16)
    assert got.shape[0] == 2 and ids.shape[1] < got.shape[1] <= ids.shape[1] + 12
    assert torch.equal(got[:, :ids.shape[1]], ids)
    for b in range(2):
        out = got[b, ids.shape[1]:].tolist()
        if 0 in out:
            assert all(t == 0 for t in out[out.index(0):])

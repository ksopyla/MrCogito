"""`evaluation/long_context_use.memory_mode` on a notebook (catalogue) model, 2026-11-21.

'off' must change the catalogue's predictions (notes off, so nothing opens) and restore the model afterwards;
'nosum' is refused, since the catalogue's notes only choose what opens; a model with no switchable read refuses
every mode but 'on'.
"""
from __future__ import annotations

import pytest
import torch

from evaluation.long_context_use import memory_mode
from tests.test_catalogue_read import CAT, make
from tests.test_note_open import row


def _loss(m, ids):
    with torch.no_grad():
        _, per, _ = m(input_ids=ids, labels=ids, doc_ids=torch.zeros_like(ids), return_per_token_loss=True)
    return per.float()


def test_catalogue_off_changes_predictions_and_restores():
    m = make(**CAT)
    ids = row(160)
    on = _loss(m, ids)
    with memory_mode(m, "off"):
        off = _loss(m, ids)
    again = _loss(m, ids)
    assert not torch.allclose(on, off), "memory off changed nothing on the catalogue"
    assert torch.allclose(on, again), "memory_mode did not restore the catalogue"
    with memory_mode(m, "on"):
        assert torch.allclose(_loss(m, ids), on)


def test_catalogue_nosum_refused():
    m = make(**CAT)
    with pytest.raises(ValueError, match="nosum"):
        with memory_mode(m, "nosum"):
            pass


def test_no_switchable_read_refused():
    from tests.test_latent_memory import lm_cfg
    from nn.perceiver_ar_lm import PerceiverARLM

    cfg = lm_cfg()
    cfg.message_write = "block_mean"  # no notebook writer
    m = PerceiverARLM(cfg).eval()
    assert m.memory_writer is None
    with pytest.raises(ValueError, match="no concept, LSA or notebook read"):
        with memory_mode(m, "off"):
            pass


# ------------------------------------------------------------------ E40 "summaries only" on the concept read
def _concept_model(**kw):
    from tests.test_concept_memory import _model

    return _model(**kw)


def test_concept_sumonly_opens_nothing_and_restores():
    """'sumonly': nothing is opened, every old concept is read as its summary. Before the first old block it equals
    the read as trained (nothing to open or summarise yet); later it differs from 'on', 'nosum' and 'off'; it does not
    depend on the open budget (nothing is opened); the switch is restored afterwards."""
    from tests.test_concept_memory import BLK

    m = _concept_model()
    att = m.layers[m.config.global_layer_index].attn
    ids = torch.randint(3, 60, (1, 96), generator=torch.Generator().manual_seed(7))
    on = _loss(m, ids)
    out = {}
    for mode in ("sumonly", "nosum", "off"):
        with memory_mode(m, mode):
            assert att.concept["open"] == (mode != "sumonly")
            out[mode] = _loss(m, ids)
    assert att.concept["open"] and att.concept["summaries"] and not att.concept_off
    torch.testing.assert_close(_loss(m, ids), on)
    early = slice(0, 2 * BLK - 1)                       # per[t] predicts t + 1: no query sees an old block yet
    late = slice(3 * BLK, None)
    torch.testing.assert_close(out["sumonly"][:, early], on[:, early])
    for mode in ("nosum", "off"):
        assert (out["sumonly"][:, late] - out[mode][:, late]).abs().max() > 1e-4, mode
    assert (out["sumonly"][:, late] - on[:, late]).abs().max() > 1e-4
    # nothing is opened, so the open budget does not matter in this mode (it does for the read as trained)
    k0 = att.concept["top_k"]
    try:
        att.concept["top_k"] = 2
        with memory_mode(m, "sumonly"):
            small = _loss(m, ids)
        on_small = _loss(m, ids)
    finally:
        att.concept["top_k"] = k0
    torch.testing.assert_close(small, out["sumonly"])
    assert (on_small - on).abs().max() > 1e-6


def test_sumonly_refused_without_concepts_and_unknown_modes_refused():
    m = make(**CAT)
    with pytest.raises(ValueError, match="sumonly"):
        with memory_mode(m, "sumonly"):
            pass
    with pytest.raises(ValueError, match="unknown memory mode"):
        with memory_mode(_concept_model(), "nonsense"):
            pass

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

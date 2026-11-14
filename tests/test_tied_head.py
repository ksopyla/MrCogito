"""The tied output head (nn.perceiver_ar_lm.TiedFactorizedHead): one token table for input and output."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nn.perceiver_ar_lm import PerceiverARLM, TiedFactorizedHead, analytic_param_count
from tests.test_latent_memory import cfg


def _pair():
    torch.manual_seed(0)
    plain = PerceiverARLM(cfg(logit_softcap=30.0, z_loss=1e-4))
    torch.manual_seed(0)
    tied = PerceiverARLM(cfg(logit_softcap=30.0, z_loss=1e-4, lm_head_tied=True))
    return plain, tied


def test_tied_head_counts_the_table_once():
    plain, tied = _pair()
    c = tied.config
    V, H, e = c.vocab_size, c.hidden_size, c.token_embedding_dim
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert isinstance(tied.lm_head, TiedFactorizedHead)
    assert n(plain) - n(tied) == V * H - H * e
    assert "lm_head.weight" not in tied.state_dict() and "lm_head.proj.weight" in tied.state_dict()
    assert sum(1 for k in tied.state_dict() if k.endswith("tok.weight")) == 1
    assert analytic_param_count(plain.config).dense - analytic_param_count(tied.config).dense == V * H - H * e


def test_tied_head_logits_agree_on_every_path_and_train_the_table():
    _, m = _pair()
    ids = torch.randint(3, 60, (2, 40))
    out = m(input_ids=ids, return_logits=True, labels=ids)
    # the effective matrix (cached decoding, evaluation) gives the loss path's logits
    hid = m.hidden_states(ids)
    torch.testing.assert_close(F.linear(hid, m.lm_head.weight), m.lm_head(hid), rtol=1e-5, atol=1e-5)
    out.loss.backward()
    assert m.embed.tok.weight.grad is not None and m.lm_head.proj.weight.grad is not None
    assert torch.isfinite(out.loss)


def test_tied_head_survives_save_and_load(tmp_path):
    _, m = _pair()
    m.eval()
    ids = torch.randint(3, 60, (1, 30))
    with torch.no_grad():
        a = m(input_ids=ids, return_logits=True).logits
    m.save_pretrained(tmp_path)
    m2 = PerceiverARLM.from_pretrained(tmp_path).eval()
    assert isinstance(m2.lm_head, TiedFactorizedHead) and m2.lm_head._tok is m2.embed.tok
    with torch.no_grad():
        b = m2(input_ids=ids, return_logits=True).logits
    torch.testing.assert_close(a, b)

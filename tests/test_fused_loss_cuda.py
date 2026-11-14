"""Liger's fused linear + cross-entropy (the --fast / --liger path) against the chunked loss, on a GPU.

The two must be the same loss: same value and gradients in fp32 (logit soft-cap 30, z-loss 1e-4, ignored targets in
the row), and with ce_norm (the SFT step token mean) the CE / z-loss weighting of the chunked path (the fused path
then falls back to it). Skipped without CUDA or liger-kernel."""
from __future__ import annotations

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARLM, _liger_flce
from tests.test_latent_memory import cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or _liger_flce() is None,
                                reason="needs CUDA and liger-kernel")


def _pair():
    torch.manual_seed(0)
    a = PerceiverARLM(cfg(logit_softcap=30.0, z_loss=1e-4, use_liger=False)).cuda().float()
    b = PerceiverARLM(cfg(logit_softcap=30.0, z_loss=1e-4, use_liger=True)).cuda().float()
    b.load_state_dict(a.state_dict())
    return a, b


def _batch():
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 80, (2, 48), generator=g).cuda()
    labels = ids.clone()
    labels[0, 5:9] = -100
    labels[1, 30:] = -100
    return ids, labels


@pytest.mark.parametrize("with_norm", [False, True], ids=["mean", "ce_norm"])
def test_fused_loss_equals_chunked(with_norm):
    a, b = _pair()
    ids, labels = _batch()
    kw = {"ce_norm": torch.tensor(57.0, device="cuda")} if with_norm else {}
    la = a(input_ids=ids, labels=labels, **kw).loss
    lb = b(input_ids=ids, labels=labels, **kw).loss
    la.backward()
    lb.backward()
    torch.testing.assert_close(lb, la, rtol=1e-5, atol=1e-6)
    ga, gb = dict(a.named_parameters()), dict(b.named_parameters())
    for name in ("lm_head.weight", "final_norm.weight"):
        torch.testing.assert_close(gb[name].grad, ga[name].grad, rtol=1e-4, atol=1e-6)
    worst = max(float((gb[k].grad - ga[k].grad).abs().max()) for k in ga if ga[k].grad is not None)
    assert worst < 1e-5

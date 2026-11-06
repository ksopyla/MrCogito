"""E33p stage 0 on a GPU: the flex kernel equals the reference (SDPA) read for the new arms, forward and backward.

Skipped without CUDA. Plan: docs/experiments_specs/ahead/E33p_plan.md §7 (CUDA-only).
"""
from __future__ import annotations

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg, lm_cfg
from tests.test_lsa_read import LSA
from tests.test_note_open import NOTES, PAGES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flex attention kernels)")

CASES = {
    "notes_open": (lm_cfg, dict(NOTES, lm_open_k=8, lm_open_cap=64)),
    "pages_open": (lm_cfg, dict(PAGES, lm_open_cap=64)),
    "lsa": (cfg, dict(LSA, lsa_top_k=32)),
}


def _pair(name):
    build, kw = CASES[name]
    out = []
    for backend in ("flex", "sdpa"):
        torch.manual_seed(0)
        m = PerceiverARLM(build(**dict(kw, attn_backend=backend))).cuda().train()
        for layer in m.layers:
            layer.attn.wo.weight.data.normal_(0, 0.2)
            layer.mlp.down.weight.data.normal_(0, 0.2)
        out.append(m)
    out[1].load_state_dict(out[0].state_dict())
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("packed", [False, True])
def test_flex_equals_sdpa_forward_and_backward(name, packed):
    flex, ref = _pair(name)
    S = 640
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 80, (2, S), generator=g).cuda()
    doc = None
    if packed:
        doc = torch.zeros(2, S, dtype=torch.long, device="cuda")
        doc[1, 300:] = 1
    lf = flex(ids, labels=ids, doc_ids=doc)
    lr = ref(ids, labels=ids, doc_ids=doc)
    assert torch.allclose(lf.loss, lr.loss, atol=2e-4, rtol=1e-4), (float(lf.loss), float(lr.loss))
    lf.loss.backward()
    lr.loss.backward()
    worst = 0.0
    for (n, p), (_, q) in zip(flex.named_parameters(), ref.named_parameters()):
        if p.grad is None:
            assert q.grad is None or q.grad.abs().max() == 0, n
            continue
        d = (p.grad - q.grad).abs().max() / q.grad.abs().max().clamp_min(1e-8)
        worst = max(worst, float(d))
        assert d < 2e-2, (n, float(d))
    print(name, packed, "worst relative grad diff", worst)


@pytest.mark.parametrize("name", sorted(CASES))
def test_bf16_autocast_trains(name):
    flex, _ = _pair(name)
    opt = torch.optim.AdamW(flex.parameters(), lr=1e-3)
    ids = torch.randint(3, 80, (2, 1024), device="cuda")
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = flex(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

"""E37 stage 0 on a GPU: the flex kernel equals the reference (SDPA) concept read, forward and backward.

Skipped without CUDA. Spec: docs/experiments_specs/ahead/E37_concept_memory.md (Plan, stage 0).
"""
from __future__ import annotations

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flex attention kernels)")

# flex kernels need head dims of at least 16; blocks of 128 so the slot columns cross several KV blocks
BASE = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
            par_global_read="concept", concept_block=128, concept_per_block=16, concept_sinks=4, concept_top_k=64,
            head_dim=16, hidden_size=64, num_attention_heads=4)
CASES = {
    "e37a": BASE,
    "e37b": dict(BASE, concept_reasoner_layers=1, concept_reasoner_dim=32, concept_reasoner_heads=2),
    "e37a_no_summaries": dict(BASE, concept_summaries=False),
}


def _pair(name):
    out = []
    for backend in ("flex", "sdpa"):
        torch.manual_seed(0)
        m = PerceiverARLM(cfg(**dict(CASES[name], attn_backend=backend))).cuda().train()
        for layer in m.layers:
            layer.attn.wo.weight.data.normal_(0, 0.2)
            layer.mlp.down.weight.data.normal_(0, 0.2)
            if layer.attn.concept_reasoner is not None:
                layer.attn.concept_reasoner.out.weight.data.normal_(0, 0.2)
        out.append(m)
    out[1].load_state_dict(out[0].state_dict())
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("packed", [False, True])
def test_flex_equals_sdpa_forward_and_backward(name, packed):
    flex, ref = _pair(name)
    S = 1024
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 80, (2, S), generator=g).cuda()
    doc = None
    if packed:
        doc = torch.zeros(2, S, dtype=torch.long, device="cuda")
        doc[1, 600:] = 1
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


@pytest.mark.parametrize("name", ["e37a", "e37b"])
def test_bf16_autocast_trains(name):
    flex, _ = _pair(name)
    opt = torch.optim.AdamW(flex.parameters(), lr=1e-3)
    ids = torch.randint(3, 80, (2, 2048), device="cuda")
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = flex(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

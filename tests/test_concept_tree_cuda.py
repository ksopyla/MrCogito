"""E41 on a GPU: the flex kernel equals the reference (SDPA) tree read, forward and backward; bf16 trains;
gradient checkpointing agrees; the 150M shapes run and a fresh tree model is the concept model.

Skipped without CUDA. Spec: docs/experiments_specs/ahead/E41_tree_concept_model.md.
"""
from __future__ import annotations

import copy

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flex attention kernels)")

# flex kernels need head dims of at least 16; blocks of 128 so the summary columns cross several KV blocks
BASE = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
            par_global_read="concept", concept_block=128, concept_per_block=16, concept_sinks=4, concept_top_k=64,
            head_dim=16, hidden_size=64, num_attention_heads=4, concept_tree=True, tree_probes=4, tree_fan=2)
REASON = dict(tree_reasoner_layers=1, tree_reasoner_dim=32, tree_reasoner_heads=2, tree_plan=True)
AUX = dict(tree_next_concept=0.1, tree_route_loss=0.1, tree_codes_coarse=32, tree_codes_fine=64, tree_code_dim=16,
           tree_route_queries=64)
CASES = {
    "flat": BASE,
    "bounded": dict(BASE, tree_follow=2),
    "descent": dict(BASE, tree_follow=1, tree_route="descent"),
    "reasoner_bounded": dict(BASE, **REASON, tree_follow=2, tree_token_update=True),
    "training_rows": dict(BASE, **REASON, **AUX, tree_row_mix="0.3,0.3,0.4", tree_train_follow=2),
    "sum_len": dict(BASE, **REASON, tree_follow=2, concept_sum_len=384),
    "wide_bounded": dict(BASE, tree_wide=True, tree_follow=2),
    "wide_training_rows": dict(BASE, tree_wide=True, tree_row_mix="0.3,0.3,0.4", tree_train_follow=2),
    "wide_flat": dict(BASE, tree_wide=True),
    "top_down_rows": dict(BASE, tree_sparse=True, tree_row_mix="0.3,0.4,0.3", tree_train_follow=2),
    "top_down_wide": dict(BASE, tree_sparse=True, tree_wide=True, tree_row_mix="0.0,0.7,0.3", tree_train_follow=4),
    "top_down_scan": dict(BASE, tree_sparse=True, tree_scan=True, tree_row_mix="0.0,0.7,0.3", tree_train_follow=4),
    "scan_bounded": dict(BASE, tree_scan=True, tree_follow=2, tree_route="descent"),
}


def _pair(name):
    out = []
    for backend in ("flex", "sdpa"):
        torch.manual_seed(0)
        m = PerceiverARLM(cfg(**dict(CASES[name], attn_backend=backend))).cuda().train()
        for layer in m.layers:
            layer.attn.wo.weight.data.normal_(0, 0.2)
            layer.mlp.down.weight.data.normal_(0, 0.2)
            rs = layer.attn.tree_reasoner
            if rs is not None:
                for lin in (rs.out_k, rs.out_v, rs.out_k1, rs.out_v1, rs.plan):
                    lin.weight.data.normal_(0, 0.2)
            if layer.attn.tree_writer is not None:
                layer.attn.tree_writer.weight.data.normal_(0, 0.3)
        out.append(m)
    out[1].load_state_dict(out[0].state_dict())
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("packed", [False, True])
def test_flex_equals_sdpa_forward_and_backward(name, packed):
    flex, ref = _pair(name)
    S = 2048                                                     # 14 closed blocks: levels 1–5
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 80, (3, S), generator=g).cuda()
    doc = None
    if packed:
        doc = torch.zeros(3, S, dtype=torch.long, device="cuda")
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
        # top-down rows read through the same code on both backends: what differs is the local layers' rounding,
        # which a choice on a near tie can turn into another read for a query (seen: 6 % on one scalar gate)
        assert d < (0.1 if name.startswith("top_down") else 2e-2), (n, float(d))
    print(name, packed, "worst relative grad diff", worst)


@pytest.mark.parametrize("name", ["bounded", "training_rows", "top_down_rows", "top_down_wide", "top_down_scan"])
def test_bf16_autocast_trains(name):
    flex, _ = _pair(name)
    opt = torch.optim.AdamW(flex.parameters(), lr=1e-3)
    ids = torch.randint(3, 80, (2, 4096), device="cuda")
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = flex(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in flex.parameters() if p.grad is not None)
        opt.step()
        opt.zero_grad()


def test_gradient_checkpointing_agrees_on_flex():
    flex, _ = _pair("training_rows")
    twin = copy.deepcopy(flex)
    twin.gradient_checkpointing_enable()
    ids = torch.randint(3, 80, (2, 2048), device="cuda")
    a = flex(ids, labels=ids)
    b = twin(ids, labels=ids)
    assert torch.allclose(a.loss, b.loss, atol=1e-5, rtol=1e-5)
    a.loss.backward()
    b.loss.backward()
    for (n, p), (_, q) in zip(flex.named_parameters(), twin.named_parameters()):
        if p.grad is not None:
            d = (p.grad - q.grad).abs().max() / p.grad.abs().max().clamp_min(1e-8)
            assert d < 2e-2, (n, float(d))


def test_fresh_tree_model_is_the_concept_model_on_flex():
    torch.manual_seed(0)
    plain = {k: v for k, v in BASE.items() if not k.startswith(("concept_tree", "tree_"))}
    base = PerceiverARLM(cfg(**dict(plain, attn_backend="flex"))).cuda().eval()
    tree = PerceiverARLM(cfg(**dict(BASE, **REASON, **AUX, attn_backend="flex"))).cuda().eval()
    missing, unexpected = tree.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("tree_reasoner" in k for k in missing)
    ids = torch.randint(3, 80, (2, 2048), device="cuda")
    with torch.no_grad():
        assert torch.allclose(tree(ids).logits, base(ids).logits, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("scan", [False, True])
def test_top_down_read_equals_the_flex_read_on_gpu(scan):
    """The same bounded read by descent through the flex kernel and through the top-down path (fp32 and bf16), also
    with the tokens opened from the best concepts of the whole row."""
    kw = dict(BASE, tree_follow=2, tree_route="descent", attn_backend="flex", tree_scan=scan)
    torch.manual_seed(0)
    dense = PerceiverARLM(cfg(**kw)).cuda().eval()
    for layer in dense.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    sparse = PerceiverARLM(cfg(**dict(kw, tree_sparse=True))).cuda().eval()
    sparse.load_state_dict(dense.state_dict())
    ids = torch.randint(3, 80, (2, 4096), device="cuda")
    with torch.no_grad():
        a, b = dense(ids).logits, sparse(ids).logits
        assert torch.allclose(a, b, atol=2e-4, rtol=1e-3), float((a - b).abs().max())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = dense(ids).logits.float(), sparse(ids).logits.float()
        assert float((a - b).abs().max()) < 0.2 and float((a - b).abs().mean()) < 0.01


def test_150m_shapes_run_at_16k():
    """The real head and block sizes on a 16k row (30 closed blocks, levels 1–3): a bounded read, a training row
    with every signal, backward; and the read statistics."""
    kw = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
              par_global_read="concept", concept_block=512, concept_per_block=64, concept_sinks=4, concept_top_k=512,
              head_dim=64, hidden_size=128, num_attention_heads=2, intermediate_size=256, block=128, pre_window=128,
              concept_tree=True, tree_probes=8, tree_fan=8, tree_reasoner_layers=1, tree_reasoner_dim=64,
              tree_reasoner_heads=2, tree_plan=True, tree_next_concept=0.1, tree_route_loss=0.1,
              tree_row_mix="0.0,0.5,0.5", tree_train_follow=16, attn_backend="flex")
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(**kw)).cuda().train()
    att = m.layers[m.config.global_layer_index].attn
    ids = torch.randint(3, 90, (2, 16384), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    m.eval()
    att.tree["follow"] = 16
    att.tree_stats = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.isfinite(m(ids).logits.float()).all()
    st = att.tree_stats
    assert st["tree/levels"] == 3 and st["tree/entries"] < 600, st
    print({k: round(v, 1) for k, v in st.items()})

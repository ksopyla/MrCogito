"""E39 on a GPU: the flex living read equals the streamed reference read (the sdpa backend), forward and backward,
at tiny sizes and with the 150M shapes at 4k and 16k; cached decoding (the streamed read, one query a step) equals the
flex forward across a stride end at 4k and 16k; two cards hold the same thresholds after training steps.

Skipped without CUDA. Spec: docs/experiments_specs/ahead/E39_living_concepts.md (unit tests).
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flex attention kernels)")

# flex kernels need head dims of at least 16 and strides of whole 128-blocks
BASE = dict(message_boundary_token_id=-1, message_loop_rounds=2, global_nope=True, num_kv_heads=1,
            par_global_read="living", living_stride=128, living_span_max=16, living_span_cap=16, living_keep=4,
            living_width=16, concept_sinks=4, concept_top_k=64, living_theta=0.3, living_lam=0.5,
            head_dim=16, hidden_size=64, num_attention_heads=4)
CASES = {
    "living": BASE,
    "living_spans": dict(BASE, living_levels=1),
    "living_flat": dict(BASE, living_search="flat"),
    "living_no_summaries": dict(BASE, concept_summaries=False),
    "living_sum_len": dict(BASE, concept_sum_len=384),
}


def _pair(name, **kw):
    out = []
    for backend in ("flex", "sdpa"):
        torch.manual_seed(0)
        m = PerceiverARLM(cfg(**dict(CASES[name], attn_backend=backend, **kw))).cuda().train()
        for layer in m.layers:
            layer.attn.wo.weight.data.normal_(0, 0.2)
            layer.mlp.down.weight.data.normal_(0, 0.2)
        out.append(m)
    out[1].load_state_dict(out[0].state_dict())
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("packed", [False, True])
def test_flex_equals_stream_forward_and_backward(name, packed):
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
            assert q.grad is None, n
            continue
        scale = float(q.grad.abs().max()) + 1e-6
        worst = max(worst, float((p.grad - q.grad).abs().max()) / scale)
    assert worst < 2e-3, worst
    ga = flex.layers[flex.config.global_layer_index].attn
    gb = ref.layers[ref.config.global_layer_index].attn
    assert float(ga.living_theta) == float(gb.living_theta) and float(ga.living_lam) == float(gb.living_lam)


@pytest.mark.parametrize("name", ["living", "living_spans"])
def test_bf16_autocast_trains(name):
    m, _ = _pair(name)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(4)
    losses = []
    for _ in range(4):
        ids = torch.randint(3, 80, (2, 1024), generator=g).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses))), losses
    assert m.pop_train_stats()["living/span_len"] > 0


@pytest.mark.parametrize("S", [4096, 16384])
def test_150m_shapes_flex_equals_stream(S):
    """The reading layer's shapes at 150M (d 64, 14 query heads, one key head, strides of 512, spans of ≤ 16, 128
    per stride, 512 opened tokens, 32 concepts kept), random keys: the two reads agree, outputs and gradients."""
    from nn.exact_read import doc_starts
    from nn.living_concepts import attend_living, build_living, living_open_table, living_read_stream, living_stats

    torch.manual_seed(5)
    B, h, d = 1, 14, 64
    dev = "cuda"
    q = torch.randn(B, S, h, d, device=dev, requires_grad=True)
    base = torch.randn(B, S // 8, 1, d, device=dev).repeat_interleave(8, 1)       # runs of 8 similar keys
    k = (base + 0.6 * torch.randn(B, S, 1, d, device=dev)).requires_grad_(True)
    v = torch.randn(B, S, 1, d, device=dev, requires_grad=True)
    doc = torch.zeros(B, S, dtype=torch.long, device=dev)
    doc[:, S // 3:] = 1
    ds = doc_starts(doc)
    pos = torch.arange(S, device=dev)[None]
    sink = (pos - ds < 4) & (doc >= 0)
    lidx = build_living(k.detach(), doc, sink, stride=512, span_max=16, span_cap=128, theta=0.5, lam=0.65)
    assert 4 <= float(lidx.tokens) / float(lidx.spans) <= 12
    st = living_stats(k, v, lidx)
    tab, n_open = living_open_table(q, lidx, st, doc, top_k=512, width=256, keep=32)
    of = attend_living(q, k, v, lidx=lidx, st=st, opened=tab, n_open=n_open, doc=doc, sink=sink, ds=ds, n_sinks=4,
                       key_valid=None, stride=512, sum_len=0)
    st2 = living_stats(k, v, lidx)
    orf = living_read_stream(q, k, v, lidx=lidx, st=st2, doc=doc, ds=ds, n_sinks=4, key_valid=None, top_k=512,
                             width=256, keep=32, stride=512)
    torch.testing.assert_close(of.float(), orf.float(), atol=2e-4, rtol=1e-3)
    w = torch.randn_like(of)
    gf = torch.autograd.grad((of * w).sum(), (q, k, v))
    gr = torch.autograd.grad((orf * w).sum(), (q, k, v))
    for a, b in zip(gf, gr):
        assert float((a - b).abs().max()) / (float(b.abs().max()) + 1e-6) < 2e-3


@pytest.mark.parametrize("L0", [4090, 16378])
def test_cached_decoding_equals_the_flex_forward(L0):
    """The answer crosses a stride end (4096 / 16384), so the stride two before joins the memory mid-answer."""
    from nn.incremental import IncrementalDecoder

    torch.manual_seed(6)
    kw = dict(BASE, living_stride=512, living_span_cap=128, concept_top_k=512, living_keep=32, living_width=256,
              attn_backend="flex")
    m = PerceiverARLM(cfg(**kw)).cuda().eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    new = 12
    Lb = L0 + new + 128
    g = torch.Generator().manual_seed(7)
    buf = torch.zeros(1, Lb, dtype=torch.long)
    buf[:, :L0] = torch.randint(3, 80, (1, L0), generator=g)
    doc = torch.full((1, Lb), -1, dtype=torch.long)
    doc[:, :L0] = 0
    answer = torch.randint(3, 80, (1, new), generator=g)
    buf, doc, answer = buf.cuda(), doc.cuda(), answer.cuda()
    dec = IncrementalDecoder(m)
    with torch.no_grad():
        h = dec.prefill(buf.clone(), doc.clone(), L0)
        for i in range(new):
            want = m.hidden_states(buf, doc_ids=doc)[:, L0 - 1 + i]
            torch.testing.assert_close(h, want, rtol=1e-3, atol=1e-3)
            buf[:, L0 + i] = answer[:, i]
            doc[:, L0 + i] = 0
            h = dec.step(answer[:, i])


def _rank(rank, world, path, out):
    torch.distributed.init_process_group("nccl", init_method=f"file://{path}", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(**dict(BASE, attn_backend="flex", living_servo_step=0.01))).cuda().train()
    g = torch.Generator().manual_seed(100 + rank)                    # different rows on each card
    for _ in range(3):
        ids = torch.randint(3, 80, (2, 1024), generator=g).cuda()
        m(ids, labels=ids).loss.backward()
    att = m.layers[m.config.global_layer_index].attn
    th = torch.stack([att.living_theta, att.living_lam]).float()
    allv = [torch.zeros_like(th) for _ in range(world)]
    torch.distributed.all_gather(allv, th)
    if rank == 0:
        torch.save(torch.stack(allv).cpu(), out)
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_two_cards_hold_the_same_thresholds():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "th.pt")
        mp.spawn(_rank, args=(2, os.path.join(tmp, "store"), out), nprocs=2, join=True)
        th = torch.load(out, weights_only=True)
    assert torch.equal(th[0], th[1]), th
    assert float(th[0, 0]) != 0.3                                    # the cut threshold moved

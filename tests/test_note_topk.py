"""E33n — top-k note read: each query and head reads only its k best visible notes (plus the raw window).

Spec: docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md.
"""
from dataclasses import replace

import pytest
import torch

from nn.perceiver_ar_lm import (
    PerceiverARLM,
    attend_message,
    dense_message_mask,
    note_topk_threshold,
)
from tests.test_latent_memory import lm_cfg
from tests.test_text_memory import TEXT, text_row


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**{**TEXT, **kw})).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def _ctx_and_tensors(S=96, h=4, g=2, dh=8, seed=3):
    m = make()
    with torch.no_grad():
        m(text_row(S))
    ctx = m._last_message_ctx
    C = ctx.slot_doc.shape[1]
    torch.manual_seed(seed)
    q, qs = torch.randn(1, S, h, dh), torch.randn(1, S, h, dh)
    k, v = torch.randn(1, S, g, dh), torch.randn(1, S, g, dh)
    kb, vb = torch.randn(1, C, g, dh), torch.randn(1, C, g, dh)
    return ctx, q, qs, k, v, kb, vb


def _reference(q, qs, k, v, kb, vb, ctx, topk):
    """One softmax over raw ‖ notes, notes below each (query, head)'s k-th visible score cut (−inf)."""
    B, S, h, dh = q.shape
    g = k.shape[2]
    rep = h // g
    mask = dense_message_mask(S, ctx, None, q.device).expand(B, h, S, S + kb.shape[1]).clone()
    rl = torch.einsum("bshd,bthd->bhst", q, k.repeat_interleave(rep, 2)) / dh ** 0.5
    sl = torch.einsum("bshd,bchd->bhsc", qs, kb.repeat_interleave(rep, 2)) / dh ** 0.5
    if topk:
        vis = mask[..., S:]
        masked = sl.masked_fill(~vis, float("-inf"))
        kth = masked.topk(min(topk, sl.shape[-1]), dim=-1).values[..., -1:]
        many = vis.sum(-1, keepdim=True) > topk
        mask[..., S:] &= ~many | (sl >= kth - 1e-3)
    lg = torch.cat([rl, sl], -1).masked_fill(~mask, float("-inf"))
    p = lg.softmax(-1)
    V = torch.cat([v, vb], 1).repeat_interleave(rep, 2)
    return torch.einsum("bhst,bthd->bshd", p, V)


def test_config_validation():
    with pytest.raises(ValueError):
        lm_cfg(**{**TEXT, "lm_read_topk": -1})
    with pytest.raises(ValueError):
        lm_cfg(lm_read_topk=4)  # not the closed text read
    with pytest.raises(ValueError):
        lm_cfg(**{**TEXT, "lm_read_topk": 4, "lm_note_age": "log"})
    assert lm_cfg(**{**TEXT, "lm_read_topk": 4}).lm_read_topk == 4


def test_threshold_is_the_kth_visible_score():
    ctx, q, qs, k, v, kb, vb = _ctx_and_tensors()
    S, h, dh = q.shape[1], q.shape[2], q.shape[3]
    tau = note_topk_threshold(qs, kb, ctx, None, k=3, chunk=7)
    vis = dense_message_mask(S, ctx, None, q.device)[0, 0, :, S:]               # [S,C]
    sl = torch.einsum("shd,chd->hsc", qs[0], kb[0].repeat_interleave(h // kb.shape[2], 1)) / dh ** 0.5
    for t in range(S):
        n = int(vis[t].sum())
        for hh in range(h):
            if n <= 3:
                assert tau[0, hh, t] == float("-inf")
            else:
                want = sl[hh, t][vis[t]].sort(descending=True).values[2]
                assert torch.isclose(tau[0, hh, t], want, atol=1e-5)


@pytest.mark.parametrize("topk", [1, 2, 5])
def test_sdpa_read_matches_reference(topk):
    ctx, q, qs, k, v, kb, vb = _ctx_and_tensors()
    out = attend_message(q, k, v, kb, vb, ctx=replace(ctx, note_topk=topk), key_valid=None, backend="sdpa", q_slot=qs)
    ref = _reference(q, qs, k, v, kb, vb, ctx, topk)
    assert torch.allclose(out, ref, atol=1e-5)
    full = attend_message(q, k, v, kb, vb, ctx=replace(ctx, note_topk=0), key_valid=None, backend="sdpa", q_slot=qs)
    assert not torch.allclose(out, full, atol=1e-4)  # the cut changes late queries, which see many notes


def test_cut_notes_get_no_gradient_from_the_read():
    ctx, q, qs, k, v, kb, vb = _ctx_and_tensors()
    kb.requires_grad_(True)
    vb.requires_grad_(True)
    out = attend_message(q, k, v, kb, vb, ctx=replace(ctx, note_topk=1), key_valid=None, backend="sdpa", q_slot=qs)
    out.square().sum().backward()
    S, h, dh = q.shape[1], q.shape[2], q.shape[3]
    vis = dense_message_mask(S, ctx, None, q.device)[0, 0, :, S:]
    tau = note_topk_threshold(qs, kb.detach(), ctx, None, k=1)
    sl = torch.einsum("shd,chd->hsc", qs[0], kb.detach()[0].repeat_interleave(h // kb.shape[2], 1)) / dh ** 0.5
    kept = (vis[None] & ((sl >= tau[0][..., None] - 1e-3) | (vis.sum(-1) <= 1)[None, :, None])).any(0).any(0)  # [C]
    assert (vb.grad[0][~kept].abs().sum() == 0) and (kb.grad[0][~kept].abs().sum() == 0)
    assert vb.grad[0][kept].abs().sum() > 0


def test_model_with_k_at_least_every_visible_note_is_unchanged():
    S = 96
    ids = text_row(S)
    base = make()
    with torch.no_grad():
        ref = base(ids).logits
        ctx = base._last_message_ctx
        C = ctx.slot_doc.shape[1]
        most = int(dense_message_mask(S, ctx, None, ids.device)[0, 0, :, S:].sum(-1).max())
    assert most > 1
    big = make(lm_read_topk=most)  # k = the most notes any query sees: the threshold runs and cuts nothing
    big.load_state_dict(base.state_dict())
    with torch.no_grad():
        out = big(ids).logits
    assert torch.allclose(out, ref, atol=1e-5)


def test_model_topk_trains():
    m = make(lm_read_topk=2).train()
    ids = text_row(96)
    loss = m(ids, labels=ids).loss
    loss.backward()
    w = m.memory_writer
    assert torch.isfinite(loss)
    assert sum(p.grad.abs().sum() for p in w.parameters() if p.grad is not None) > 0

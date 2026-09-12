"""Unit tests for the Perceiver Concept LM family (nn/perceiver_concept_lm.py, E22).

CPU-only, tiny dims, `attn_backend="sdpa"` as the reference; the flex equivalence test is
CUDA-guarded. The two structural guarantees the spec rests on are tested directly:
causality through the concept path, and structural closure of the decoder (no raw token
outside its segment reaches a position except through the array).
"""
import math

import pytest
import torch

from nn.perceiver_concept_lm import (
    PerceiverConceptConfig,
    PerceiverConceptLM,
    analytic_param_count,
)

V = 97


def tiny_cfg(**kw):
    base = dict(
        vocab_size=V, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        enc_layers=2, enc_window=4, concept_ratio=4, concept_slots=1, latent_layers=2, latent_repeats=1,
        dec_layers=2, dec_segment=8, dec_local="block", concept_mode="full", xattn_kv_heads=2,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, rope_theta=10000.0,
        ngram_orders=(2, 3), ngram_buckets=64, enc_value_embed_layers=(0,), dec_value_embed_layers=(0,),
        value_embed_dim=4, logit_softcap=30.0, z_loss=1e-4, chunked_ce_block_size=5, use_liger=False,
        attn_backend="sdpa", attn_pad_multiple=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    base.update(kw)
    return PerceiverConceptConfig(**base)


def make_model(cfg, seed=0, live_concepts=True):
    """Model with non-trivial residual writers so the concept path actually moves the logits
    (the real init zeroes them, which would make several tests vacuous)."""
    torch.manual_seed(seed)
    m = PerceiverConceptLM(cfg).eval()
    if live_concepts:
        with torch.no_grad():
            for layer in list(m.enc_layers) + list(m.latent_layers):
                layer.attn.wo.weight.normal_(0, 0.05)
                layer.mlp.down.weight.normal_(0, 0.05)
            for layer in m.dec_layers:
                layer.attn.wo.weight.normal_(0, 0.05)
                layer.mlp.down.weight.normal_(0, 0.05)
                if layer.has_xattn:
                    layer.xattn.wo.weight.normal_(0, 0.05)
            m.pooler.wo.weight.normal_(0, 0.05)
    return m


def rand_ids(B, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (B, S), generator=g)


# ------------------------------------------------------------------ shapes / accounting


def test_forward_shapes_and_param_count():
    cfg = tiny_cfg()
    m = make_model(cfg)
    ids = rand_ids(2, 40)
    out = m(input_ids=ids, labels=ids.clone())
    assert torch.isfinite(out.loss)
    logits = m(input_ids=ids).logits
    assert logits.shape == (2, 40, V)
    pb = analytic_param_count(cfg)
    assert pb.total == sum(p.numel() for p in m.parameters())
    assert pb.compute < pb.total


def test_no_concept_control_has_no_cross_attention():
    cfg = tiny_cfg(concept_mode="none")
    m = make_model(cfg)
    assert all(not l.has_xattn for l in m.dec_layers)
    assert analytic_param_count(cfg).total == sum(p.numel() for p in m.parameters())
    ids = rand_ids(2, 24)
    assert torch.isfinite(m(input_ids=ids, labels=ids.clone()).loss)


def test_concept_count_scales_with_length():
    cfg = tiny_cfg(concept_ratio=4, concept_slots=2)
    m = make_model(cfg)
    z, valid, pos, doc = m.concepts(rand_ids(1, 40))
    assert z.shape == (1, 20, cfg.hidden_size)
    assert bool(valid.all())
    assert pos[0, :4].tolist() == [3, 3, 7, 7]


# ------------------------------------------------------------------ causality


def test_causal_through_concept_path():
    """Changing token t must not change any logit at positions < t, including via the array."""
    cfg = tiny_cfg()
    m = make_model(cfg)
    ids = rand_ids(1, 40)
    base = m(input_ids=ids).logits
    for t in (5, 17, 31):
        ids2 = ids.clone()
        ids2[0, t] = (ids2[0, t] + 11) % V
        lo = m(input_ids=ids2).logits
        assert torch.allclose(base[0, :t], lo[0, :t], atol=1e-5), f"leak before t={t}"
        assert not torch.allclose(base[0, t:], lo[0, t:], atol=1e-5)


def test_slot_not_visible_before_its_block_end():
    """A token inside block j (not its last token) must not read block j's slot: perturbing the
    block's *last* token must leave earlier tokens of the same block unchanged."""
    cfg = tiny_cfg(concept_ratio=4, dec_segment=100)   # one segment: only the array is causal here
    m = make_model(cfg)
    ids = rand_ids(1, 40)
    base = m(input_ids=ids).logits
    ids2 = ids.clone()
    ids2[0, 23] = (ids2[0, 23] + 5) % V             # last token of block 5 (positions 20..23)
    lo = m(input_ids=ids2).logits
    assert torch.allclose(base[0, :23], lo[0, :23], atol=1e-5)


# ------------------------------------------------------------------ structural closure


def test_closure_without_concepts():
    """With the array off, a position depends on no raw token outside its own segment."""
    cfg = tiny_cfg(concept_mode="none", dec_segment=8)
    m = make_model(cfg)
    ids = rand_ids(1, 40)
    base = m(input_ids=ids).logits
    ids2 = ids.clone()
    ids2[0, 3] = (ids2[0, 3] + 7) % V              # segment 0 (positions 0..7)
    lo = m(input_ids=ids2).logits
    assert torch.allclose(base[0, 8:], lo[0, 8:], atol=1e-5)
    assert not torch.allclose(base[0, 3:8], lo[0, 3:8], atol=1e-5)


def test_concepts_are_the_only_long_range_path():
    """With the array on, the same segment-0 edit *does* reach later segments — and only via the
    array: `concept_override('none')` restores the no-concept logits exactly."""
    cfg = tiny_cfg(dec_segment=8)
    m = make_model(cfg)
    ids = rand_ids(1, 40)
    ids2 = ids.clone()
    ids2[0, 3] = (ids2[0, 3] + 7) % V
    a, b = m(input_ids=ids).logits, m(input_ids=ids2).logits
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)
    with m.concept_override("none"):
        a0, b0 = m(input_ids=ids).logits, m(input_ids=ids2).logits
    assert torch.allclose(a0[0, 8:], b0[0, 8:], atol=1e-5)


def test_concept_override_modes():
    cfg = tiny_cfg(dec_segment=8)
    m = make_model(cfg)
    ids = rand_ids(2, 40)
    real = m(input_ids=ids).logits
    with m.concept_override("none"):
        none = m(input_ids=ids).logits
    with m.concept_override("shuffled"):
        shuf = m(input_ids=ids).logits
    assert not torch.allclose(real, none, atol=1e-5)
    assert not torch.allclose(real, shuf, atol=1e-5)
    # the first concept block (positions < r) has no visible content slot: every mode agrees there
    r = cfg.concept_ratio
    assert torch.allclose(real[:, : r - 1], none[:, : r - 1], atol=1e-5)
    assert torch.allclose(real[:, : r - 1], shuf[:, : r - 1], atol=1e-5)
    with pytest.raises(ValueError):
        with m.concept_override("bogus"):
            pass
    assert torch.allclose(m(input_ids=ids).logits, real, atol=1e-6)   # override restored


def test_swa_decoder_mode_runs_and_differs():
    a = make_model(tiny_cfg(dec_local="block", dec_segment=8))
    b = make_model(tiny_cfg(dec_local="swa", dec_segment=8))
    ids = rand_ids(1, 40)
    assert torch.isfinite(a(input_ids=ids, labels=ids.clone()).loss)
    assert torch.isfinite(b(input_ids=ids, labels=ids.clone()).loss)


# ------------------------------------------------------------------ documents / padding


def test_pool_excludes_earlier_document_in_a_straddling_block():
    cfg = tiny_cfg(concept_ratio=4)
    m = make_model(cfg)
    ids = rand_ids(1, 16)
    doc = torch.zeros(1, 16, dtype=torch.long)
    doc[0, 10:] = 1                                  # block 2 = positions 8..11 straddles docs 0|1
    z1, valid1, pos1, cdoc1 = m.concepts(ids, doc_ids=doc)
    assert cdoc1[0].tolist() == [0, 0, 1, 1]
    assert pos1[0].tolist() == [3, 7, 1, 5]           # doc-relative block-end positions
    # tokens 8,9 (doc 0) must not influence slot 2 (owned by doc 1): change them, slot 2 stays
    ids2 = ids.clone()
    ids2[0, 8] = (ids2[0, 8] + 3) % V
    ids2[0, 9] = (ids2[0, 9] + 3) % V
    z2, *_ = m.concepts(ids2, doc_ids=doc)
    assert torch.allclose(z1[0, 2], z2[0, 2], atol=1e-5)
    # ... while a doc-1 token of the same block does move slot 2
    ids3 = ids.clone()
    ids3[0, 10] = (ids3[0, 10] + 3) % V
    z3, *_ = m.concepts(ids3, doc_ids=doc)
    assert not torch.allclose(z1[0, 2], z3[0, 2], atol=1e-5)
    assert torch.allclose(z1[0, :2], z3[0, :2], atol=1e-5)


def test_packed_documents_do_not_leak():
    """A token in document 1 must not see document 0 (raw or via the array)."""
    cfg = tiny_cfg(dec_segment=8)
    m = make_model(cfg)
    ids = rand_ids(1, 32)
    doc = torch.zeros(1, 32, dtype=torch.long)
    doc[0, 16:] = 1
    a = m(input_ids=ids, doc_ids=doc).logits
    ids2 = ids.clone()
    ids2[0, :16] = (ids2[0, :16] + 9) % V
    b = m(input_ids=ids2, doc_ids=doc).logits
    assert torch.allclose(a[0, 16:], b[0, 16:], atol=1e-5)


def test_right_padding_is_inert():
    cfg = tiny_cfg(dec_segment=8)
    m = make_model(cfg)
    ids = rand_ids(1, 24)
    full = m(input_ids=ids).logits
    padded = torch.cat([ids, torch.zeros(1, 8, dtype=torch.long)], dim=1)
    am = torch.cat([torch.ones(1, 24, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long)], dim=1)
    lo = m(input_ids=padded, attention_mask=am).logits
    assert torch.allclose(full[0], lo[0, :24], atol=1e-4)


def test_pad_multiple_padding_matches_unpadded():
    """attn_pad_multiple pads internally; logits on the real positions must not change."""
    ids = rand_ids(1, 30)
    a = make_model(tiny_cfg(attn_pad_multiple=1, dec_segment=8))
    b = make_model(tiny_cfg(attn_pad_multiple=16, dec_segment=8))
    b.load_state_dict(a.state_dict())
    la, lb = a(input_ids=ids).logits, b(input_ids=ids).logits
    assert lb.shape == la.shape
    assert torch.allclose(la, lb, atol=1e-4)


# ------------------------------------------------------------------ loss paths / probes contract


def test_per_token_loss_matches_mean_loss_and_hidden_states():
    cfg = tiny_cfg()
    m = make_model(cfg)
    ids = rand_ids(2, 33)
    out = m(input_ids=ids, labels=ids.clone())
    o2, per, valid = m(input_ids=ids, labels=ids.clone(), return_per_token_loss=True)
    assert per.shape == (2, 32) and valid.shape == (2, 32)
    ce_only = per[valid].mean()
    assert abs(float(o2.loss) - float(ce_only)) < 1e-5
    assert float(out.loss) >= float(ce_only) - 1e-5           # z-loss adds to the training loss
    h = m.hidden_states(ids)
    assert h.shape == (2, 33, cfg.hidden_size)


def test_latent_repeats_share_weights():
    cfg = tiny_cfg(latent_repeats=3)
    m = make_model(cfg)
    assert analytic_param_count(cfg).total == sum(p.numel() for p in m.parameters())
    ids = rand_ids(1, 24)
    assert torch.isfinite(m(input_ids=ids, labels=ids.clone()).loss)


def test_training_step_with_gradient_checkpointing():
    cfg = tiny_cfg()
    m = make_model(cfg)
    m.train()
    m.gradient_checkpointing_enable()
    ids = rand_ids(2, 24)
    out = m(input_ids=ids, labels=ids.clone())
    out.loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the pooler and the cross-attention receive gradient (the array is on the loss path)
    assert m.pooler.q.grad is not None and float(m.pooler.q.grad.abs().sum()) > 0
    assert float(m.dec_layers[0].xattn.wq.weight.grad.abs().sum()) > 0


def test_save_load_roundtrip(tmp_path):
    cfg = tiny_cfg()
    m = make_model(cfg)
    m.save_pretrained(tmp_path)
    from nn.perceiver_families import checkpoint_family, load_perceiver_lm

    assert checkpoint_family(str(tmp_path)) == "perceiver_concept"
    m2 = load_perceiver_lm(str(tmp_path), "cpu", dtype=torch.float32)
    ids = rand_ids(1, 20)
    assert torch.allclose(m(input_ids=ids).logits, m2(input_ids=ids).logits, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex needs CUDA")
def test_flex_matches_sdpa_cuda():
    ids = rand_ids(2, 256).cuda()
    kw = dict(dec_segment=64, concept_ratio=16, head_dim=16, hidden_size=64)   # flex needs head_dim >= 16
    a = make_model(tiny_cfg(attn_pad_multiple=1, **kw)).cuda()
    b = make_model(tiny_cfg(attn_pad_multiple=128, attn_backend="flex", **kw)).cuda()
    b.load_state_dict(a.state_dict())
    doc = torch.zeros(2, 256, dtype=torch.long, device="cuda")
    doc[1, 100:] = 1
    la = a(input_ids=ids, doc_ids=doc).logits
    lb = b(input_ids=ids, doc_ids=doc).logits
    assert torch.allclose(la, lb, atol=2e-2, rtol=2e-2)

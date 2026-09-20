"""E30 sliding-window Perceiver banks: geometry, exclusive write, not a mean."""
import math

import pytest
import torch

from evaluation.bapo_models import ArchSpec, build_model, n_params
from nn.perceiver_ar_lm import (
    PerceiverARConfig,
    PerceiverARLM,
    SlidingWindowPerceiverCompressor,
    analytic_param_count,
    swp_geometry,
)

V = 97
M = 90


def cfg(**kw):
    base = dict(
        vocab_size=V, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        par_mode="perceiver", pre_layers=1, pre_window=4, global_layers=1, stack_layers=1, block=6,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, rope_theta=10000.0, nope_every=0,
        ngram_orders=(2,), ngram_buckets=32, value_embed_layers=(), value_embed_dim=4,
        logit_softcap=0.0, z_loss=0.0, chunked_ce_block_size=8, use_liger=False,
        attn_backend="sdpa", attn_pad_multiple=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        zero_init_residuals=False,
    )
    base.update(kw)
    return PerceiverARConfig(**base)


def make_swp(**kw):
    kw.setdefault("message_boundary_token_id", M)
    kw.setdefault("message_write", "sw_perceiver")
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(**kw)).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def test_default_write_is_block_mean_and_byte_identical():
    a = PerceiverARLM(cfg())
    b = PerceiverARLM(cfg(message_write="block_mean"))
    x = torch.randint(3, 80, (2, 12))
    with torch.no_grad():
        assert torch.equal(a(x).logits, b(x).logits)
    assert a.config.message_write == "block_mean"
    assert all(l.attn.compressor is None for l in a.layers)


def test_swp_geometry_tiny_slides_and_bridge_uses_k32():
    c = cfg(message_boundary_token_id=M, message_write="sw_perceiver")
    tiny = swp_geometry(c, 128)
    assert tiny.sliding and tiny.n_windows >= 2
    assert tiny.bank_size == 8
    assert tiny.window == 64
    assert tiny.stride == 48
    assert tiny.n_slots == tiny.n_windows * tiny.bank_size
    assert tiny.starts == tuple(sorted(set(tiny.starts)))
    bridge = swp_geometry(c, 512)
    assert bridge.bank_size == 32
    assert bridge.window == 256
    assert bridge.stride == 192
    assert bridge.n_windows >= 3
    assert 256 in bridge.starts  # flush-right covers the tail
    assert bridge.compression == pytest.approx(512 / bridge.n_slots)


def test_mixed_window_pool_valid_keeps_sender_tokens():
    """A window that straddles QUERY still writes sender-side slots from sender tokens."""
    m = make_swp(swp_bank_size=4, swp_coverage=4, swp_auto_fit=False)
    S = 32
    ids = torch.randint(3, 80, (1, S))
    qpos = 10
    ids[0, qpos] = M
    labels = torch.full((1, S), -100)
    labels[0, qpos:] = ids[0, qpos:]
    loss = m(ids, labels=labels).loss
    assert torch.isfinite(loss)
    ctx = m._last_message_ctx
    assert ctx is not None
    assert (ctx.slot_side == 0).any()
    pv = ctx.pool_valid
    assert pv is not None
    # QUERY itself is side ≥1; sender pool must include some tokens before it.
    assert bool(pv[0, :qpos].any())


def test_swp_forbids_inplace_and_identity():
    with pytest.raises(ValueError, match="inplace"):
        cfg(message_boundary_token_id=M, message_write="sw_perceiver", message_slots_inplace=True)
    with pytest.raises(ValueError, match="identity"):
        cfg(message_boundary_token_id=M, message_write="sw_perceiver", message_identity_slots=True)
    with pytest.raises(ValueError, match="prefix_ae"):
        cfg(message_boundary_token_id=M, message_write="sw_perceiver", message_prefix_ae=True)


def test_e30_factory_ignores_e21_identity_inplace_flags():
    spec = ArchSpec(
        name="e30",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=M,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_prefix_ae=True,
        zero_init_residuals=False,
    )
    model = build_model(
        "e30",
        vocab_size=V,
        seq_len=32,
        answer_start=24,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        spec=spec,
        seed=0,
    )
    assert model.config.message_write == "sw_perceiver"
    assert model.config.message_slots_inplace is False
    assert model.config.message_identity_slots is False
    assert model.config.message_prefix_ae is False


def test_compressor_shapes_and_not_a_mean():
    c = cfg(message_boundary_token_id=M, message_write="sw_perceiver")
    comp = SlidingWindowPerceiverCompressor(c)
    B, S = 2, 48
    h = torch.randn(B, S, c.hidden_size)
    k_raw = torch.randn(B, S, c.num_kv_heads, c.head_dim)
    v = torch.randn(B, S, c.num_kv_heads, c.head_dim)
    k_norm = torch.nn.RMSNorm(c.head_dim)
    k_bar, v_bar = comp(h, k_raw, v, k_norm)
    geo = comp.geometry(S)
    assert k_bar.shape == (B, geo.n_slots, c.num_kv_heads, c.head_dim)
    assert v_bar.shape == k_bar.shape
    # Distinct queries → slots in one window are not all equal (not a uniform mean).
    k0 = k_bar[0, : geo.bank_size].reshape(geo.bank_size, -1)
    dist = torch.cdist(k0, k0)
    assert float(dist.max()) > 1e-5
    assert comp._last_entropy == comp._last_entropy
    assert comp._last_entropy < math.log(geo.window) + 1e-5


def test_exclusive_none_changes_logits_and_params_match_analytic():
    m = make_swp()
    gi = m.config.global_layer_index
    assert isinstance(m.layers[gi].attn.compressor, SlidingWindowPerceiverCompressor)
    assert sum(p.numel() for p in m.parameters()) == analytic_param_count(m.config).total
    ids = torch.randint(3, 80, (2, 24))
    ids[:, 10] = M
    with torch.no_grad():
        real = m(ids).logits
        with m.message_override("none"):
            none = m(ids).logits
    # Receiver tokens (after QUERY) must move when slots are dropped.
    assert not torch.allclose(real[:, 10:], none[:, 10:], atol=1e-5)


def test_loss_finite_and_participation_without_query():
    m = make_swp()
    ids = torch.randint(3, 80, (2, 16))
    loss = m(ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    q = m.layers[m.config.global_layer_index].attn.compressor.q
    assert q.grad is not None


def test_bapo_factory_e30_under_10m_and_finite_loss():
    spec = ArchSpec(
        name="e30",
        hidden=128,
        head_dim=32,
        local_window=16,
        message_boundary_token_id=M,
        zero_init_residuals=False,
    )
    model = build_model(
        "e30",
        vocab_size=V,
        seq_len=128,
        answer_start=96,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        spec=spec,
        seed=0,
    )
    assert n_params(model) < 10_000_000
    assert model.config.message_write == "sw_perceiver"
    ids = torch.randint(3, 80, (2, 32))
    ids[:, 12] = M
    labels = ids.clone()
    labels[:, :12] = -100
    loss = model(ids, labels=labels).loss
    assert torch.isfinite(loss)
    geo = swp_geometry(model.config, 128)
    assert geo.sliding


def test_ce_still_falling_gate():
    from verification.bapo_capability_probe import _ce_still_falling

    assert not _ce_still_falling([])
    assert not _ce_still_falling([{"ce_nats": 1.4}] * 6)
    falling = [{"ce_nats": 1.5}] * 4 + [{"ce_nats": 1.2}, {"ce_nats": 1.1}]
    assert _ce_still_falling(falling)
    assert not _ce_still_falling([{"ce_nats": 1.5}, {"ce_nats": 1.45}])

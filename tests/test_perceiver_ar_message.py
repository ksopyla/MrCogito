"""E21 message boundary on the Perceiver AR family (`nn/perceiver_ar_lm.py`): severance of every
local channel, prefix visible to receivers only through KVCompressor slots, probe overrides,
arm-U identity at r=1, and the receiver-only round trip through `prefix_kv(as_message=True)`.
CPU / sdpa, tiny config."""
import pytest
import torch

from unittest.mock import patch

from nn.perceiver_ar_lm import (
    KVCompressor,
    MESSAGE_QUERY_NBHD_DEFAULT,
    MessageCtx,
    PerceiverARConfig,
    PerceiverARLM,
    analytic_param_count,
    attend_inplace,
    attend_message,
    build_message_anchors,
    dense_bool_mask,
    dense_inplace_mask,
    dense_message_mask,
    exclusive_visible,
    make_message_mask_pred,
    mix_inplace_kv,
)

V = 97
M = 90          # boundary token id (a vocabulary id the random inputs never use)


def cfg(**kw):
    base = dict(
        vocab_size=V, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        par_mode="perceiver", pre_layers=1, pre_window=4, global_layers=1, stack_layers=3, block=6,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, rope_theta=10000.0, nope_every=2,
        ngram_orders=(2, 3), ngram_buckets=64, value_embed_layers=(0, 1, 3), value_embed_dim=4,
        logit_softcap=30.0, z_loss=0.0, chunked_ce_block_size=5, use_liger=False,
        attn_backend="sdpa", attn_pad_multiple=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    base.update(kw)
    return PerceiverARConfig(**base)


def make_model(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(cfg(**kw)).eval()
    for layer in m.layers:  # give every path signal (wo/down are zero-init)
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def rand_ids(B, S, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 80, (B, S), generator=g)


def with_boundary(ids, P):
    ids = ids.clone()
    ids[:, P] = M
    return ids


# ------------------------------------------------------------------ config / off-path identity


def test_off_by_default_is_byte_identical_and_validated():
    a = make_model(seed=0)
    b = make_model(seed=0, message_boundary_token_id=-1)
    assert all(l.attn.compressor is None for l in a.layers)
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
    x = rand_ids(2, 12)
    with torch.no_grad():
        assert torch.equal(a(x).logits, b(x).logits)
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, message_compress_ratio=0)
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=V + 5)
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, par_mode="dense")
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, attn_backend="flash")
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, message_extra_slot_attends=-1)
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, message_global_anchors="full_prefix")
    with pytest.raises(ValueError):
        cfg(message_boundary_token_id=M, message_anchor_window=-1)


def test_enabled_without_boundary_token_is_inert_and_counts_params():
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=4)
    off = make_model(seed=0)
    gi = on.config.global_layer_index
    assert on.layers[gi].attn.compressor is not None and all(
        l.attn.compressor is None for i, l in enumerate(on.layers) if i != gi)
    assert sum(p.numel() for p in on.parameters()) == analytic_param_count(on.config).total
    # an E18 checkpoint loads into the E21 model with only the compressor missing (warm start)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and all("compressor" in k for k in missing) and len(missing) == 2
    assert float(on.layers[gi].attn.compressor.u.abs().sum()) == 0.0
    assert float(on.layers[gi].attn.compressor.delta.weight.abs().sum()) == 0.0
    x = rand_ids(2, 12)
    with torch.no_grad():
        assert torch.allclose(on(x).logits, off(x).logits, atol=1e-6)
    # DDP guard: the compressor participates in the graph even without a boundary
    loss = on(x, labels=x).loss
    loss.backward()
    assert on.layers[gi].attn.compressor.u.grad is not None


# ------------------------------------------------------------------ masks


def _ctx(side, doc, slot_doc, slot_side, override="real"):
    B, S = side.shape
    return MessageCtx(side=side, doc=doc, local_doc_ids=doc, slot_doc=slot_doc, slot_side=slot_side,
                      slot_pos=torch.zeros_like(slot_doc), override=override)


def test_message_mask_no_raw_key_crosses_the_boundary_and_slots_are_side_gated():
    S, r = 8, 2
    P = 5
    side = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])
    doc = torch.zeros(1, S, dtype=torch.long)
    # blocks: [0,1] [2,3] [4,5] [6,7] -> slot 2 straddles P (side 0/1) -> invalid
    slot_doc = torch.tensor([[0, 0, -1, 0]])
    slot_side = torch.tensor([[0, 0, 0, 1]])
    for override in ("real", "none", "swapped", "raw"):
        ctx = _ctx(side, doc, slot_doc, slot_side, override)
        dense = dense_message_mask(S, ctx, None, "cpu")[0, 0]
        pred = make_message_mask_pred(S, ctx, None)
        nb = slot_doc.shape[1]
        for q in range(S):
            for kv in range(S + nb):
                got = bool(pred(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))
                assert got == bool(dense[q, kv]), (override, q, kv)
        raw, slot = dense[:, :S], dense[:, S:]
        for q in range(S):
            for j in range(S):
                crosses = side[0, j] != side[0, q]
                if override == "raw":
                    assert raw[q, j] == (j <= q)
                else:
                    assert raw[q, j] == ((j <= q) and not crosses), (override, q, j)
        if override in ("none", "raw"):
            assert not slot.any()
        else:
            # senders see no slot; receivers see the two complete sender blocks, not the straddling
            # one and not the receiver-side block
            assert not slot[:P].any()
            assert torch.equal(slot[P:], torch.tensor([[True, True, False, False]] * (S - P)))
        assert dense.any(dim=1).all()   # no empty query row (sdpa would NaN)


def test_message_pred_equals_dense_mask_on_packed_multi_side_rows_and_captures_int32_only():
    """The tag-based flex predicate (two int32 buffers) is the dense reference mask exactly, on
    packed rows with several documents, up to three sides per document, padding and invalid
    (straddling) slots; and it captures no int64 tensor (four int64 tiles blew the sm86 shared
    memory budget at head_dim 128)."""
    torch.manual_seed(7)
    B, S, r = 2, 48, 4
    doc = torch.tensor([[0] * 20 + [1] * 28, [2] * 16 + [3] * 24 + [-1] * 8])
    side = torch.zeros(B, S, dtype=torch.long)
    side[0, 9:14] = 1; side[0, 14:20] = 2            # doc 0: three sides
    side[0, 35:] = 1                                  # doc 1: two sides
    side[1, 6:16] = 1                                 # doc 2: two sides
    side[1, 30:40] = 1                                # doc 3: two sides, then pad
    key_valid = doc >= 0
    nb = S // r
    docp, sidep = doc.view(B, nb, r), side.view(B, nb, r)
    homog = (docp == docp[..., :1]).all(-1) & (sidep == sidep[..., :1]).all(-1) & (docp[..., 0] >= 0)
    slot_doc = torch.where(homog, docp[..., 0], torch.full_like(docp[..., 0], -1))
    slot_side = torch.where(homog, sidep[..., 0], torch.zeros_like(sidep[..., 0]))
    assert (slot_doc < 0).sum() >= 3
    qi = torch.arange(S)[:, None]
    kj = torch.arange(S + nb)[None, :]
    for kv_ok in (None, key_valid):
        for override in ("real", "none", "swapped", "raw"):
            ctx = _ctx(side, doc, slot_doc, slot_side, override)
            assert ctx.n_sides == 0 and ctx.tag_stride() == 6      # lazily derived: 3 sides
            pred = make_message_mask_pred(S, ctx, kv_ok)
            for cell in pred.__closure__:
                v = cell.cell_contents
                if torch.is_tensor(v):
                    assert v.dtype in (torch.int32, torch.bool), v.dtype
            got = torch.stack([pred(torch.tensor(b), torch.tensor(0), qi, kj) for b in range(B)])
            ref = dense_message_mask(S, ctx, kv_ok, "cpu")[:, 0]
            rows = key_valid if kv_ok is not None else torch.ones_like(key_valid)
            assert torch.equal(got[rows], ref[rows]), (override, kv_ok is not None)
    # sanity on the geometry itself: doc 0's side-2 tokens see the two complete side-0 blocks of
    # their own document (blocks 2 and 3 straddle a side change), none of doc 1's
    ctx = _ctx(side, doc, slot_doc, slot_side, "real")
    slot = dense_message_mask(S, ctx, None, "cpu")[0, 0, :, S:]
    expect = (slot_doc[0] == 0) & (slot_side[0] < 2)
    assert torch.equal(slot[19], expect) and expect.sum() == 2


def test_kv_compressor_is_mean_pool_at_init_and_identity_at_ratio_one():
    c = cfg(message_boundary_token_id=M, message_compress_ratio=4)
    comp = KVCompressor(c)
    B, S, g, dh, d = 2, 10, c.num_kv_heads, c.head_dim, c.hidden_size
    h = torch.randn(B, S, d)
    k_raw, v = torch.randn(B, S, g, dh), torch.randn(B, S, g, dh)
    k_norm = torch.nn.RMSNorm(dh)
    k_bar, v_bar = comp(h, k_raw, v, k_norm)
    assert k_bar.shape == (B, 3, g, dh) and v_bar.shape == (B, 3, g, dh)   # ceil(10/4) = 3 blocks
    assert torch.allclose(k_bar[:, 0], k_norm(k_raw[:, :4].mean(1)), atol=1e-6)
    assert torch.allclose(v_bar[:, 1], v[:, 4:8].mean(1), atol=1e-6)
    assert torch.allclose(v_bar[:, 2], v[:, 8:10].mean(1), atol=1e-6)      # partial block: mean of what exists
    one = KVCompressor(cfg(message_boundary_token_id=M, message_compress_ratio=1))
    k1, v1 = one(h, k_raw, v, k_norm)
    assert torch.allclose(k1, k_norm(k_raw), atol=1e-6) and torch.allclose(v1, v, atol=1e-6)
    # learned weights move it away from the mean (non-trivial gradient path)
    comp.u.data.normal_()
    k2, _ = comp(h, k_raw, v, k_norm)
    assert not torch.allclose(k2[:, 0], k_norm(k_raw[:, :4].mean(1)), atol=1e-4)


def test_identity_slots_r1_equals_token_kv_even_after_scrambling_u_delta():
    """`--message_identity_slots` bypasses u/delta: r=1 is k_norm(k_raw), v at init and after scramble."""
    k_norm = torch.nn.RMSNorm(cfg().head_dim)
    B, S, g, dh, d = 2, 10, cfg().num_kv_heads, cfg().head_dim, cfg().hidden_size
    h = torch.randn(B, S, d)
    k_raw, v = torch.randn(B, S, g, dh), torch.randn(B, S, g, dh)
    ident = KVCompressor(cfg(message_boundary_token_id=M, message_compress_ratio=1,
                             message_identity_slots=True))
    k1, v1 = ident(h, k_raw, v, k_norm)
    assert torch.allclose(k1, k_norm(k_raw), atol=1e-6) and torch.allclose(v1, v, atol=1e-6)
    ident.u.data.normal_(0, 5.0)
    ident.delta.weight.data.normal_(0, 5.0)
    k2, v2 = ident(h, k_raw, v, k_norm)
    assert torch.allclose(k2, k_norm(k_raw), atol=1e-6) and torch.allclose(v2, v, atol=1e-6)
    off = KVCompressor(cfg(message_boundary_token_id=M, message_compress_ratio=1))
    assert off.identity_slots is False
    off.delta.weight.data.normal_(0, 5.0)
    k_off, _ = off(h, k_raw, v, k_norm)
    assert not torch.allclose(k_off, k_norm(k_raw), atol=1e-4)


def test_identity_slots_r16_is_frozen_mean_pool_not_last_token():
    """`--message_identity_slots` at r=16 is k_norm(mean of 16) / mean(v), not last-token
    copy and not a no-op. Scrambling u/delta must not move the slots."""
    torch.manual_seed(0)
    c = cfg(message_boundary_token_id=M, message_compress_ratio=16, message_identity_slots=True)
    comp = KVCompressor(c)
    B, S, g, dh, d = 2, 40, c.num_kv_heads, c.head_dim, c.hidden_size  # 2 full blocks + rem 8
    h = torch.randn(B, S, d)
    k_raw, v = torch.randn(B, S, g, dh), torch.randn(B, S, g, dh)
    k_norm = torch.nn.RMSNorm(dh)
    k_bar, v_bar = comp(h, k_raw, v, k_norm)
    assert k_bar.shape == (B, 3, g, dh) and v_bar.shape == (B, 3, g, dh)
    assert torch.allclose(k_bar[:, 0], k_norm(k_raw[:, :16].mean(1)), atol=1e-6)
    assert torch.allclose(v_bar[:, 0], v[:, :16].mean(1), atol=1e-6)
    assert not torch.allclose(v_bar[:, 0], v[:, 15], atol=1e-4)
    assert not torch.allclose(k_bar[:, 0], k_norm(k_raw[:, 15]), atol=1e-4)
    assert torch.allclose(v_bar[:, 1], v[:, 16:32].mean(1), atol=1e-6)
    assert torch.allclose(v_bar[:, 2], v[:, 32:40].mean(1), atol=1e-6)
    k_saved, v_saved = k_bar.clone(), v_bar.clone()
    comp.u.data.normal_(0, 5.0)
    comp.delta.weight.data.normal_(0, 5.0)
    k2, v2 = comp(h, k_raw, v, k_norm)
    assert torch.allclose(k2, k_saved, atol=1e-6) and torch.allclose(v2, v_saved, atol=1e-6)
    learned = KVCompressor(cfg(message_boundary_token_id=M, message_compress_ratio=16))
    learned.u.data.normal_(0, 5.0)
    k_l, _ = learned(h, k_raw, v, k_norm)
    assert not torch.allclose(k_l[:, 0], k_norm(k_raw[:, :16].mean(1)), atol=1e-4)


# ------------------------------------------------------------------ severance / channel isolation


def test_receiver_depends_on_the_prefix_only_through_the_slots():
    """Perturbing a prefix token must change receiver logits under `real` (through the slots) and
    must change nothing under `none`; sender logits before the perturbed token never change."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S), P)
    y = x.clone()
    y[0, 2] = (y[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, b = model(x).logits, model(y).logits
        assert not torch.allclose(a[0, P:], b[0, P:], atol=1e-6)          # message carries content
        assert torch.allclose(a[0, :2], b[0, :2], atol=1e-6)               # causality on the sender side
        with model.message_override("none"):
            a0, b0 = model(x).logits, model(y).logits
            assert torch.allclose(a0[0, P:], b0[0, P:], atol=1e-6)          # nothing leaks locally
            assert not torch.allclose(a0[0, P:], a[0, P:], atol=1e-6)       # the slots did something
        with model.message_override("raw"):
            ar = model(x).logits
            assert not torch.allclose(ar[0, P:], a[0, P:], atol=1e-6)       # raw ≠ compressed (r=2)
    # gradient view of the same statement: receiver CE w.r.t. the prefix embeddings
    for mode, expect_grad in (("none", False), ("real", True)):
        emb = {}
        h = model.embed.register_forward_hook(lambda mod, i, o: emb.setdefault("x0", o))
        with model.message_override(mode):
            logits = model(x).logits
        h.remove()
        loss = torch.nn.functional.cross_entropy(logits[0, P:-1], x[0, P + 1:])
        g = torch.autograd.grad(loss, emb["x0"])[0]
        prefix_grad = float(g[0, :P].abs().sum())
        assert (prefix_grad > 0) == expect_grad, (mode, prefix_grad)


def test_ratio_one_matches_raw_override_and_the_swap_uses_the_other_row():
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1)
    S, P = 14, 8
    x = with_boundary(rand_ids(2, S, seed=3), P)
    with torch.no_grad():
        real = model(x).logits
        with model.message_override("raw"):
            raw = model(x).logits
        assert torch.allclose(real, raw, atol=1e-5)                         # arm U == uncompressed prefix
        with model.message_override("swapped"):
            sw = model(x).logits
        # row 0 now reads row 1's prefix: equal to a forward of [prefix_1 | suffix_0]
        mixed = x.clone()
        mixed[0, :P] = x[1, :P]
        ref = model(mixed).logits
        assert torch.allclose(sw[0, P:], ref[0, P:], atol=1e-5)
        assert torch.allclose(sw[0, :P], real[0, :P], atol=1e-6)            # sender side untouched


def test_inplace_default_off_matches_concat_and_r1_matches_raw():
    """In-place slots default off (concat path). At r=1, in-place real ≈ raw override (same geometry)."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    concat = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1)
    concat_on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                           message_slots_inplace=False)
    inplace = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                         message_slots_inplace=True)
    with torch.no_grad():
        assert torch.allclose(concat(x).logits, concat_on(x).logits, atol=1e-6)
        with concat.message_override("raw"):
            raw = concat(x).logits
        ip = inplace(x).logits
        assert torch.allclose(ip, raw, atol=1e-4)
        # concat extra-KV at r=2 is not the in-place path
        c2 = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2)
        i2 = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                        message_slots_inplace=True)
        assert not torch.allclose(c2(x).logits[:, 8:], i2(x).logits[:, 8:], atol=1e-4)


def test_inplace_hides_uncompressed_remainder_from_receivers():
    """Remainder-off incomplete last sender block stays invisible to receivers under in-place."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                       message_slots_inplace=True)
    S, P = 16, 10                              # sender len 10: 3 complete blocks + remainder at 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y_rem = x.clone()
    y_rem[0, P - 1] = (y_rem[0, P - 1] + 7) % 80 + 3
    y_blk = x.clone()
    y_blk[0, 2] = (y_blk[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, rem, blk = model(x).logits, model(y_rem).logits, model(y_blk).logits
        assert torch.allclose(a[0, P:], rem[0, P:], atol=1e-5)
        assert not torch.allclose(a[0, P:], blk[0, P:], atol=1e-5)


def test_inplace_raw_kv_default_off_is_inert_and_skips_compressor_when_on():
    """`--message_inplace_raw_kv` default off (concat/inplace compressor unchanged).
    When on with inplace, token K/V is used: scrambling the compressor does not move
    receiver logits. Exclusive remainder hiding still holds. Concat ignores the flag."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    concat = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2)
    concat_flag = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                             message_inplace_raw_kv=True)
    ip = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                    message_slots_inplace=True)
    ip_off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                        message_slots_inplace=True, message_inplace_raw_kv=False)
    ip_raw = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                        message_slots_inplace=True, message_inplace_raw_kv=True)
    with torch.no_grad():
        assert torch.allclose(concat(x).logits, concat_flag(x).logits, atol=1e-6)
        assert torch.allclose(ip(x).logits, ip_off(x).logits, atol=1e-6)
        assert not torch.allclose(ip(x).logits[:, 8:], ip_raw(x).logits[:, 8:], atol=1e-4)
        gi = ip_raw.config.global_layer_index
        before = ip_raw(x).logits
        ip_raw.layers[gi].attn.compressor.u.data.normal_(0, 5.0)
        ip_raw.layers[gi].attn.compressor.delta.weight.data.normal_(0, 5.0)
        after = ip_raw(x).logits
        assert torch.allclose(before, after, atol=1e-5)


def test_inplace_raw_kv_r1_matches_raw_override():
    """r=1 + inplace raw token KV: same length-S geometry and token values as override=raw."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    raw_kv = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                        message_slots_inplace=True, message_inplace_raw_kv=True)
    raw_m = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1)
    with torch.no_grad():
        with raw_m.message_override("raw"):
            raw = raw_m(x).logits
        assert torch.allclose(raw_kv(x).logits, raw, atol=1e-4)


def test_inplace_raw_kv_hides_uncompressed_remainder_from_receivers():
    """Raw token KV under inplace still hides the incomplete last sender block."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                       message_slots_inplace=True, message_inplace_raw_kv=True)
    S, P = 16, 10
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y_rem = x.clone()
    y_rem[0, P - 1] = (y_rem[0, P - 1] + 7) % 80 + 3
    y_blk = x.clone()
    y_blk[0, 2] = (y_blk[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, rem, blk = model(x).logits, model(y_rem).logits, model(y_blk).logits
        assert torch.allclose(a[0, P:], rem[0, P:], atol=1e-5)
        assert not torch.allclose(a[0, P:], blk[0, P:], atol=1e-5)


def test_inplace_identity_slots_uses_scatter_not_raw_kv_and_ignores_u():
    """Inplace + identity_slots still goes through compressor/scatter (raw_kv off).
    Scrambling u/delta must not move logits. r=1 matches inplace raw token KV at init."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    ident = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    raw_kv = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                        message_slots_inplace=True, message_inplace_raw_kv=True)
    learned = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                         message_slots_inplace=True)
    assert ident.config.message_inplace_raw_kv is False
    assert ident.config.message_identity_slots is True
    with torch.no_grad():
        assert torch.allclose(ident(x).logits, raw_kv(x).logits, atol=1e-4)
        gi = ident.config.global_layer_index
        before = ident(x).logits
        ident.layers[gi].attn.compressor.u.data.normal_(0, 5.0)
        ident.layers[gi].attn.compressor.delta.weight.data.normal_(0, 5.0)
        after = ident(x).logits
        assert torch.allclose(before, after, atol=1e-5)
        learned.layers[gi].attn.compressor.delta.weight.data.normal_(0, 5.0)
        assert not torch.allclose(learned(x).logits[:, 8:], ident(x).logits[:, 8:], atol=1e-4)


def test_inplace_identity_r16_is_mean_pool_not_r1_noop():
    """Inplace identity at r=16 is not a no-op copy of every token (r=1 identity).
    Scrambling u/delta must not move receiver logits."""
    x = with_boundary(rand_ids(2, 32, seed=3), 16)  # sender len 16 = one r=16 block
    r16 = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=16,
                     message_slots_inplace=True, message_identity_slots=True)
    r1 = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                    message_slots_inplace=True, message_identity_slots=True)
    assert r16.config.message_inplace_raw_kv is False
    with torch.no_grad():
        assert not torch.allclose(r16(x).logits[:, 16:], r1(x).logits[:, 16:], atol=1e-4)
        gi = r16.config.global_layer_index
        before = r16(x).logits
        r16.layers[gi].attn.compressor.u.data.normal_(0, 5.0)
        r16.layers[gi].attn.compressor.delta.weight.data.normal_(0, 5.0)
        after = r16(x).logits
        assert torch.allclose(before, after, atol=1e-5)


def test_r1_identity_covers_every_sender_token_on_exclusive_global():
    """r=1 inplace identity is not a type-cue coverage hole: every sender position is a
    replace slot, so the exclusive global read sees identity K/V of the whole prefix."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    assert ctx.ratio == 1
    k = torch.zeros(1, S, 2, 8)
    v = torch.zeros(1, S, 2, 8)
    k_bar = torch.zeros(1, S, 2, 8)
    v_bar = torch.zeros(1, S, 2, 8)
    _, _, replace = mix_inplace_kv(k, v, k_bar, v_bar, ctx)
    assert bool(replace[0, :P].all())
    assert not bool(replace[0, P])
    mask = dense_inplace_mask(S, ctx, None, replace, "cpu")
    assert bool(mask[0, 0, P, :P].all())
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P
    assert concat[P, S:S + P].tolist() == [True] * P


def test_keep_local_swa_default_off_severs_window_across_query():
    """Default E21: QUERY is a SWA document start. Local window cannot see P-1."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    assert model.config.message_keep_local_swa is False
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    assert ctx.local_doc_ids[0, P - 1].item() != ctx.local_doc_ids[0, P].item()
    swa = dense_bool_mask(S, "swa", 4, None, ctx.local_doc_ids, "cpu", batch=1)
    assert not bool(swa[0, 0, P, P - 1])
    y = x.clone()
    y[0, P - 1] = (y[0, P - 1] + 7) % 80 + 3
    with torch.no_grad(), model.message_override("none"):
        a, b = model(x).logits, model(y).logits
    assert torch.allclose(a[0, P:], b[0, P:], atol=1e-5)


def test_keep_local_swa_crosses_boundary_but_global_stays_exclusive():
    """`--message_keep_local_swa`: SWA at QUERY can attend P-1; exclusive global mask unchanged."""
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                      message_slots_inplace=True, message_identity_slots=True)
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True,
                     message_keep_local_swa=True)
    assert on.config.message_keep_local_swa is True
    assert off.config.message_keep_local_swa is False
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    pos = on._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx_off = off._message_context(x, None, None, pos)
        ctx_on = on._message_context(x, None, None, pos)
    assert torch.equal(ctx_on.local_doc_ids, ctx_on.doc)
    assert torch.equal(ctx_on.side, ctx_off.side)
    assert torch.equal(ctx_on.slot_doc, ctx_off.slot_doc)
    swa_on = dense_bool_mask(S, "swa", 4, None, ctx_on.local_doc_ids, "cpu", batch=1)
    swa_off = dense_bool_mask(S, "swa", 4, None, ctx_off.local_doc_ids, "cpu", batch=1)
    assert bool(swa_on[0, 0, P, P - 1])
    assert not bool(swa_off[0, 0, P, P - 1])
    k = torch.zeros(1, S, 2, 8)
    v = torch.zeros(1, S, 2, 8)
    k_bar = torch.zeros(1, S, 2, 8)
    v_bar = torch.zeros(1, S, 2, 8)
    _, _, replace_off = mix_inplace_kv(k, v, k_bar, v_bar, ctx_off)
    _, _, replace_on = mix_inplace_kv(k, v, k_bar, v_bar, ctx_on)
    assert torch.equal(replace_off, replace_on)
    mask_off = dense_inplace_mask(S, ctx_off, None, replace_off, "cpu")
    mask_on = dense_inplace_mask(S, ctx_on, None, replace_on, "cpu")
    assert torch.equal(mask_off, mask_on)
    concat_off = dense_message_mask(S, ctx_off, None, "cpu")
    concat_on = dense_message_mask(S, ctx_on, None, "cpu")
    assert torch.equal(concat_off, concat_on)
    assert concat_on[0, 0, P, :P].tolist() == [False] * P
    y = x.clone()
    y[0, P - 1] = (y[0, P - 1] + 7) % 80 + 3
    with torch.no_grad(), on.message_override("none"):
        a, b = on(x).logits, on(y).logits
    assert not torch.allclose(a[0, P:], b[0, P:], atol=1e-5)


def test_identity_kv_are_post_pre_swa_not_frozen_embeddings():
    """r=1 inplace identity slots are projected from the residual AFTER the SWA pre-layer,
    not from frozen/unmixed token embeddings. Knob A is already true for pre-SWA."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    assert model.config.pre_layers >= 1
    assert model.config.message_extra_slot_attends == 0
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = {}
    gi = model.config.global_layer_index
    assert model.layers[gi].attn.pattern == "full"

    def hook(mod, args):
        captured["attn_x"] = args[0].detach()

    model.layers[gi].attn.register_forward_pre_hook(hook)
    with torch.no_grad():
        x0 = model.embed(x)
        _ = model(x)
    assert captured["attn_x"].shape == x0.shape
    assert not torch.allclose(captured["attn_x"], x0, atol=1e-4)


def test_extra_slot_attends_default_off_is_byte_identical_and_param_matched():
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True)
    expl = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                      message_slots_inplace=True, message_identity_slots=True,
                      message_extra_slot_attends=0)
    one = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True,
                     message_extra_slot_attends=1)
    assert off.config.message_extra_slot_attends == 0
    assert expl.config.message_extra_slot_attends == 0
    assert one.config.message_extra_slot_attends == 1
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in one.parameters())
    with torch.no_grad():
        assert torch.allclose(off(x).logits, expl(x).logits, atol=1e-6)
        assert not torch.allclose(off(x).logits[:, 8:], one(x).logits[:, 8:], atol=1e-4)


def test_extra_slot_attends_reuses_frozen_exclusive_kv_inplace():
    """extra=1: two exclusive attends, identical K/V objects, no concat raw-prefix leak."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = []
    real = attend_inplace

    def wrapped(q, k, v, **kw):
        captured.append((k, v))
        return real(q, k, v, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_inplace", wrapped):
        _ = model(x)
    assert len(captured) == 2
    assert captured[0][0] is captured[1][0]
    assert captured[0][1] is captured[1][1]
    assert torch.equal(captured[0][0], captured[1][0])
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    assert ctx.extra_slot_attends == 1
    k = torch.zeros(1, S, 2, 8)
    v = torch.zeros(1, S, 2, 8)
    _, _, replace = mix_inplace_kv(k, v, k, v, ctx)
    mask = dense_inplace_mask(S, ctx, None, replace, "cpu")
    assert bool(mask[0, 0, P, :P].all())
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P
    assert concat[P, S:S + P].tolist() == [True] * P


def test_extra_slot_attends_reuses_frozen_exclusive_kv_concat():
    """Concat extra=1: two attends, frozen slot K/V, receivers still cannot see raw prefix."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                       message_extra_slot_attends=1)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = []
    real = attend_message

    def wrapped(q, k, v, k_bar, v_bar, **kw):
        captured.append((k, v, k_bar, v_bar))
        return real(q, k, v, k_bar, v_bar, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_message", wrapped):
        _ = model(x)
    assert len(captured) == 2
    assert captured[0][2] is captured[1][2]
    assert captured[0][3] is captured[1][3]
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P


def test_extra_slot_attends_hides_uncompressed_remainder():
    """Extra exclusive hop still hides the incomplete last sender block (not raw prefix)."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1)
    S, P = 16, 10
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y_rem = x.clone()
    y_rem[0, P - 1] = (y_rem[0, P - 1] + 7) % 80 + 3
    y_blk = x.clone()
    y_blk[0, 2] = (y_blk[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, rem, blk = model(x).logits, model(y_rem).logits, model(y_blk).logits
        assert torch.allclose(a[0, P:], rem[0, P:], atol=1e-5)
        assert not torch.allclose(a[0, P:], blk[0, P:], atol=1e-5)


def test_extra_slot_attends_updates_q_at_query_over_frozen_kv():
    """extra=1 is two exclusive attends: Q at QUERY changes; K/V stay the same object."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1)
    assert model.config.message_update_slot_kv is False
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = []
    real = attend_inplace

    def wrapped(q, k, v, **kw):
        captured.append((q.detach().clone(), k, v))
        return real(q, k, v, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_inplace", wrapped):
        _ = model(x)
    assert len(captured) == 2
    assert captured[0][1] is captured[1][1]
    assert captured[0][2] is captured[1][2]
    assert not torch.allclose(captured[0][0][:, P], captured[1][0][:, P], atol=1e-4)


def test_extra_hop_query_q_can_contain_sender_slot_content():
    """Hop-2 Q at QUERY is sensitive to a sender identity slot — type can enter Q."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y = x.clone()
    y[0, 2] = (y[0, 2] + 7) % 80 + 3
    qs = []
    real = attend_inplace

    def wrapped(q, k, v, **kw):
        qs.append(q.detach().clone())
        return real(q, k, v, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_inplace", wrapped):
        _ = model(x)
        _ = model(y)
    # qs: [hop1_x, hop2_x, hop1_y, hop2_y]
    assert len(qs) == 4
    assert not torch.allclose(qs[1][:, P], qs[3][:, P], atol=1e-4)


def test_update_slot_kv_default_off_matches_prior_exclusive_e21():
    """Default extra=0 matches prior exclusive E21 even if update_slot_kv is set."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    prior = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    extra0_on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                           message_slots_inplace=True, message_identity_slots=True,
                           message_update_slot_kv=True)
    extra1_frozen = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                               message_slots_inplace=True, message_identity_slots=True,
                               message_extra_slot_attends=1)
    extra1_on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                           message_slots_inplace=True, message_identity_slots=True,
                           message_extra_slot_attends=1, message_update_slot_kv=True)
    extra1_off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                            message_slots_inplace=True, message_identity_slots=True,
                            message_extra_slot_attends=1, message_update_slot_kv=False)
    assert prior.config.message_update_slot_kv is False
    assert extra0_on.config.message_extra_slot_attends == 0
    assert extra0_on.config.message_update_slot_kv is True
    assert extra1_on.config.message_update_slot_kv is True
    assert sum(p.numel() for p in prior.parameters()) == sum(p.numel() for p in extra1_on.parameters())
    with torch.no_grad():
        assert torch.allclose(prior(x).logits, extra0_on(x).logits, atol=1e-6)
        assert torch.allclose(extra1_frozen(x).logits, extra1_off(x).logits, atol=1e-6)
        assert not torch.allclose(extra1_frozen(x).logits[:, 8:], extra1_on(x).logits[:, 8:], atol=1e-4)


def test_update_slot_kv_rewrites_exclusive_kv_between_hops_inplace():
    """update_slot_kv: two exclusive attends, K/V tensors change, not full prefix."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1, message_update_slot_kv=True)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = []
    real = attend_inplace

    def wrapped(q, k, v, **kw):
        captured.append((k, v))
        return real(q, k, v, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_inplace", wrapped):
        _ = model(x)
    assert len(captured) == 2
    assert captured[0][0] is not captured[1][0]
    assert captured[0][1] is not captured[1][1]
    assert not torch.equal(captured[0][0], captured[1][0])
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    assert ctx.extra_slot_attends == 1
    assert ctx.update_slot_kv is True
    k = torch.zeros(1, S, 2, 8)
    v = torch.zeros(1, S, 2, 8)
    _, _, replace = mix_inplace_kv(k, v, k, v, ctx)
    mask = dense_inplace_mask(S, ctx, None, replace, "cpu")
    assert bool(mask[0, 0, P, :P].all())
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P
    assert concat[P, S:S + P].tolist() == [True] * P


def test_update_slot_kv_rewrites_concat_slot_kv_not_raw_prefix():
    """Concat update_slot_kv: slot K/V change between hops; receivers still miss raw prefix."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                       message_extra_slot_attends=1, message_update_slot_kv=True)
    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    captured = []
    real = attend_message

    def wrapped(q, k, v, k_bar, v_bar, **kw):
        captured.append((k, v, k_bar, v_bar))
        return real(q, k, v, k_bar, v_bar, **kw)

    import nn.perceiver_ar_lm as pal
    with torch.no_grad(), patch.object(pal, "attend_message", wrapped):
        _ = model(x)
    assert len(captured) == 2
    assert captured[0][0] is captured[1][0]  # raw keys stay the first-hop snapshot
    assert captured[0][2] is not captured[1][2]
    assert not torch.equal(captured[0][2], captured[1][2])
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P


def test_update_slot_kv_hides_uncompressed_remainder():
    """Rewriting slot K/V between hops still hides the incomplete last sender block."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_extra_slot_attends=1, message_update_slot_kv=True)
    S, P = 16, 10
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y_rem = x.clone()
    y_rem[0, P - 1] = (y_rem[0, P - 1] + 7) % 80 + 3
    y_blk = x.clone()
    y_blk[0, 2] = (y_blk[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, rem, blk = model(x).logits, model(y_rem).logits, model(y_blk).logits
        assert torch.allclose(a[0, P:], rem[0, P:], atol=1e-5)
        assert not torch.allclose(a[0, P:], blk[0, P:], atol=1e-5)


def test_default_global_layers_is_one_exclusive_block():
    """Default global_layers=1 is one exclusive global Attention+FFN, extra hops 0."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                       message_slots_inplace=True, message_identity_slots=True)
    assert model.config.global_layers == 1
    assert model.config.message_extra_slot_attends == 0
    assert model.config.stack_layers == 3
    pats = [l.attn.pattern for l in model.layers]
    assert pats.count("full") == 1
    gi = model.config.global_layer_index
    assert pats[gi] == "full"
    assert model.layers[gi].attn.compressor is not None
    assert sum(l.attn.compressor is not None for l in model.layers) == 1
    assert analytic_param_count(model.config).total == sum(p.numel() for p in model.parameters())


def test_global_layers_two_are_two_exclusive_blocks_not_extra_hops():
    """global_layers=2: two sequential exclusive attend+FFN blocks, not extra hops / not extra SWA."""
    one = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True,
                     global_layers=1, stack_layers=2)
    two = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True,
                     global_layers=2, stack_layers=2)
    hop = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True,
                     global_layers=1, stack_layers=2, message_extra_slot_attends=1)
    extra_stack = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                             message_slots_inplace=True, message_identity_slots=True,
                             global_layers=1, stack_layers=3)
    assert one.config.global_layers == 1
    assert two.config.global_layers == 2
    assert two.config.message_extra_slot_attends == 0
    assert two.config.message_update_slot_kv is False
    assert hop.config.message_extra_slot_attends == 1
    assert [l.attn.pattern for l in one.layers] == ["swa", "full", "swa", "swa"]
    assert [l.attn.pattern for l in two.layers] == ["swa", "full", "full", "swa", "swa"]
    assert [l.attn.pattern for l in hop.layers] == ["swa", "full", "swa", "swa"]
    assert [l.attn.pattern for l in extra_stack.layers].count("full") == 1
    full = [i for i, l in enumerate(two.layers) if l.attn.pattern == "full"]
    assert full == [1, 2]
    for i in full:
        assert two.layers[i].attn.compressor is not None
        assert two.layers[i].mlp is not None
    assert sum(l.attn.compressor is not None for l in two.layers) == 2
    assert sum(l.attn.compressor is not None for l in one.layers) == 1
    # extra hops reuse the same Attention; a second global Block adds params
    assert sum(p.numel() for p in hop.parameters()) == sum(p.numel() for p in one.parameters())
    assert sum(p.numel() for p in two.parameters()) > sum(p.numel() for p in one.parameters())
    assert analytic_param_count(two.config).total == sum(p.numel() for p in two.parameters())

    events = []
    real = attend_inplace

    def wrapped_attend(q, k, v, **kw):
        events.append(("attend", id(k)))
        return real(q, k, v, **kw)

    def mlp_hook(_mod, _inp, _out):
        events.append(("mlp1",))

    S, P = 16, 9
    x = with_boundary(rand_ids(1, S, seed=4), P)
    import nn.perceiver_ar_lm as pal
    handle = two.layers[1].mlp.register_forward_hook(mlp_hook)
    try:
        with torch.no_grad(), patch.object(pal, "attend_inplace", wrapped_attend):
            _ = two(x)
    finally:
        handle.remove()
    attends = [e for e in events if e[0] == "attend"]
    assert len(attends) == 2
    assert attends[0][1] != attends[1][1]  # each Block projects its own exclusive K/V
    attend_ix = [i for i, e in enumerate(events) if e[0] == "attend"]
    mlp_ix = [i for i, e in enumerate(events) if e[0] == "mlp1"]
    assert len(mlp_ix) == 1
    assert attend_ix[0] < mlp_ix[0] < attend_ix[1]  # FFN between the two exclusive reads

    pos = two._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = two._message_context(x, None, None, pos)
    assert ctx.extra_slot_attends == 0
    k = torch.zeros(1, S, 2, 8)
    _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
    mask = dense_inplace_mask(S, ctx, None, replace, "cpu")
    assert bool(mask[0, 0, P, :P].all())  # r=1 identity: exclusive slots cover sender
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert concat[P, :P].tolist() == [False] * P  # still not raw prefix


def test_global_layers_two_hides_uncompressed_remainder():
    """Second exclusive global Block still hides the incomplete last sender block."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                       message_slots_inplace=True, message_identity_slots=True,
                       global_layers=2, stack_layers=2)
    assert model.config.global_layers == 2
    assert model.config.message_extra_slot_attends == 0
    S, P = 16, 10
    x = with_boundary(rand_ids(1, S, seed=4), P)
    y_rem = x.clone()
    y_rem[0, P - 1] = (y_rem[0, P - 1] + 7) % 80 + 3
    y_blk = x.clone()
    y_blk[0, 2] = (y_blk[0, 2] + 7) % 80 + 3
    with torch.no_grad():
        a, rem, blk = model(x).logits, model(y_rem).logits, model(y_blk).logits
        assert torch.allclose(a[0, P:], rem[0, P:], atol=1e-5)
        assert not torch.allclose(a[0, P:], blk[0, P:], atol=1e-5)


MARK = 91  # type-mark id; random inputs use 3..80, boundary is M=90


def test_global_anchors_default_off_matches_prior_e21():
    """Default `none` is byte-identical to omitting the field; no extra params."""
    x = with_boundary(rand_ids(2, 14, seed=3), 8)
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                     message_slots_inplace=True, message_identity_slots=True)
    expl = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                      message_slots_inplace=True, message_identity_slots=True,
                      message_global_anchors="none")
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=1,
                    message_slots_inplace=True, message_identity_slots=True,
                    message_global_anchors="type_marks", message_anchor_token_ids=(MARK,))
    assert off.config.message_global_anchors == "none"
    assert expl.config.message_global_anchors == "none"
    assert on.config.message_global_anchors == "type_marks"
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in on.parameters())
    with torch.no_grad():
        assert torch.allclose(off(x).logits, expl(x).logits, atol=1e-6)
        # no type-mark tokens in `x` → type_marks is a no-op vs prior E21
        assert torch.allclose(off(x).logits, on(x).logits, atol=1e-6)


def test_type_mark_anchors_join_exclusive_kv_not_full_prefix_inplace():
    """r=4 remainder: type-mark in the incomplete block is visible; other remainder is not.

    Exclusive = slots ∪ anchors, not the full sender prefix.
    """
    r, S, P = 4, 18, 10  # sender 0..9: two full blocks + remainder 8,9
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=r,
                     message_slots_inplace=True, message_identity_slots=True)
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=r,
                    message_slots_inplace=True, message_identity_slots=True,
                    message_global_anchors="type_marks", message_anchor_token_ids=(MARK,))
    x = with_boundary(rand_ids(1, S, seed=11), P)
    x[0, P - 1] = MARK  # remainder type mark
    pos = on._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx_off = off._message_context(x, None, None, pos)
        ctx_on = on._message_context(x, None, None, pos)
    assert ctx_off.anchor_mode == "none"
    assert not bool(ctx_off.anchor[0].any())
    assert ctx_on.anchor_mode == "type_marks"
    assert ctx_on.anchor[0].tolist()[P - 1] is True
    assert int(ctx_on.anchor[0].sum()) == 1
    assert int(ctx_on.anchor[0].sum()) < S  # sparse vs seq
    k = torch.zeros(1, S, 2, 8)
    v = torch.zeros_like(k)
    _, _, replace = mix_inplace_kv(k, v, k, v, ctx_on)
    vis = exclusive_visible(replace, ctx_on)
    assert bool(replace[0, 0]) and not bool(replace[0, P - 1])  # mark is non-slot remainder
    assert bool(vis[0, P - 1]) and not bool(vis[0, P - 2])  # other remainder stays hidden
    assert not bool(vis[0, :P].all())  # not full prefix
    mask_on = dense_inplace_mask(S, ctx_on, None, vis, "cpu")
    mask_off = dense_inplace_mask(S, ctx_off, None, exclusive_visible(replace, ctx_off), "cpu")
    assert bool(mask_on[0, 0, P, P - 1]) and not bool(mask_off[0, 0, P, P - 1])
    assert not bool(mask_on[0, 0, P, P - 2])
    concat_on = dense_message_mask(S, ctx_on, None, "cpu")[0, 0]
    concat_off = dense_message_mask(S, ctx_off, None, "cpu")[0, 0]
    assert concat_off[P, :P].tolist() == [False] * P
    assert bool(concat_on[P, P - 1]) and not bool(concat_on[P, P - 2])
    y_drop = x.clone()
    y_drop[0, P - 1] = (int(x[0, 0]) % 70) + 3
    with torch.no_grad():
        a_on, drop_on = on(x).logits, on(y_drop).logits
        a_off, drop_off = off(x).logits, off(y_drop).logits
        assert not torch.allclose(a_on[0, P:], drop_on[0, P:], atol=1e-5)
        assert torch.allclose(a_off[0, P:], drop_off[0, P:], atol=1e-5)


def test_query_nbhd_anchors_are_sparse_and_not_full_prefix():
    """query_nbhd window=2 on a long remainder: only the last 2 sender tokens leak."""
    r, S, P = 8, 20, 14  # sender 0..13; complete [0:8]; remainder 8..13 (6 tokens)
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=r,
                       message_slots_inplace=True, message_identity_slots=True,
                       message_global_anchors="query_nbhd", message_anchor_window=2)
    assert MESSAGE_QUERY_NBHD_DEFAULT == 4
    x = with_boundary(rand_ids(1, S, seed=6), P)
    pos = model._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx = model._message_context(x, None, None, pos)
    anc = ctx.anchor[0]
    assert anc[P - 1] and anc[P - 2]
    assert not bool(anc[P - 3])  # remainder beyond window stays a non-anchor
    assert int(anc.sum()) == 2
    assert int(anc.sum()) < S
    k = torch.zeros(1, S, 2, 8)
    _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
    vis = exclusive_visible(replace, ctx)
    assert bool(vis[0, P - 1]) and bool(vis[0, P - 2])
    assert not bool(vis[0, 9])  # remainder token outside nbhd and not a slot
    assert not bool(vis[0, :P].all())
    mask = dense_inplace_mask(S, ctx, None, vis, "cpu")
    assert bool(mask[0, 0, P, P - 1]) and not bool(mask[0, 0, P, 9])
    concat = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    assert bool(concat[P, P - 1]) and not bool(concat[P, 9])
    assert concat[P, :P].float().sum() < P  # not full raw prefix


def test_build_message_anchors_union_query_nbhd_plus_type():
    S, P = 16, 10
    ids = with_boundary(rand_ids(1, S, seed=2), P)
    ids[0, 2] = MARK
    side = torch.zeros(1, S, dtype=torch.long)
    side[:, P:] = 1
    doc = torch.zeros(1, S, dtype=torch.long)
    none = build_message_anchors(ids, side, doc, mode="none", token_ids=(MARK,), window=2)
    marks = build_message_anchors(ids, side, doc, mode="type_marks", token_ids=(MARK,), window=2)
    nbhd = build_message_anchors(ids, side, doc, mode="query_nbhd", token_ids=(MARK,), window=2)
    both = build_message_anchors(ids, side, doc, mode="query_nbhd+type", token_ids=(MARK,), window=2)
    assert not bool(none.any())
    assert marks[0].tolist()[2] and int(marks.sum()) == 1
    assert nbhd[0, P - 1] and nbhd[0, P - 2] and not bool(nbhd[0, 2])
    assert bool(both[0, 2]) and bool(both[0, P - 1])
    assert int(both.sum()) == 3
    assert int(both.sum()) < S


def test_receiver_only_round_trip_via_prefix_kv_as_message():
    """S6: a second process holding only `prefix_kv(prefix, as_message=True)` reproduces the
    receiver's logits."""
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3)
    S, P = 20, 12                              # 12 = 4 complete blocks of 3
    x = with_boundary(rand_ids(1, S, seed=5), P)
    with torch.no_grad():
        full = model(x).logits[0, P:]
        k_bar, v_bar, slot_pos = model.prefix_kv(x[:, :P], as_message=True)
        assert k_bar.shape == (1, 4, 2, 8) and torch.equal(slot_pos[0], torch.tensor([2, 5, 8, 11]))
        recv = model(x[:, P:], message_kv=(k_bar, v_bar, slot_pos), position_offset=P).logits[0]
    assert torch.allclose(full, recv, atol=1e-4), (full - recv).abs().max()
    # a partial trailing block is not part of the message (the receiver could not read it either)
    with torch.no_grad():
        k7, _, pos7 = model.prefix_kv(x[:, :7], as_message=True)
    assert k7.shape[1] == 2 and pos7.tolist() == [[2, 5]]


def test_remainder_pooling_produces_last_incomplete_sender_slot():
    """QUERY that is not r-aligned drops the straddling block unless message_pool_remainder."""
    r, S, P = 4, 18, 10
    # sender 0..9 (2 full blocks + remainder 8,9); QUERY at 10; block [8,11] straddles
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=r)
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=r,
                    message_pool_remainder=True)
    assert off.config.message_pool_remainder is False
    x = with_boundary(rand_ids(1, S, seed=11), P)
    pos = on._positions(S, 1, None, x.device)
    with torch.no_grad():
        ctx_off = off._message_context(x, None, None, pos)
        ctx_on = on._message_context(x, None, None, pos)
    assert ctx_off.slot_doc[0, 2].item() == -1
    assert ctx_off.pool_valid is None
    assert ctx_on.slot_doc[0, 2].item() == 0
    assert ctx_on.slot_side[0, 2].item() == 0
    assert ctx_on.pool_valid[0, 8:12].tolist() == [True, True, False, False]
    mask_on = dense_message_mask(S, ctx_on, None, "cpu")[0, 0]
    mask_off = dense_message_mask(S, ctx_off, None, "cpu")[0, 0]
    assert bool(mask_on[P, S + 2]) and not bool(mask_off[P, S + 2])
    gi = on.config.global_layer_index
    layer = on.layers[gi]
    with torch.no_grad():
        h = layer.attn_norm(on._run_layers(x, None, None, None, capture_input_of=gi))
        k_raw, v = layer.attn.kv_raw(h, x)
        _, v_c = layer.attn.compressor(h, k_raw, v, layer.attn.k_norm, ctx_on.pool_valid)
    assert torch.allclose(v_c[:, 2], v[:, 8:10].mean(1), atol=1e-5)
    with torch.no_grad():
        assert torch.isfinite(on(x, labels=x).loss)


def test_prefix_kv_as_message_keeps_remainder_block_when_flag_on():
    off = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3)
    on = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=3,
                    message_pool_remainder=True)
    x = rand_ids(1, 7, seed=5)
    with torch.no_grad():
        k_off, _, pos_off = off.prefix_kv(x, as_message=True)
        k_on, _, pos_on = on.prefix_kv(x, as_message=True)
    assert k_off.shape[1] == 2 and pos_off.tolist() == [[2, 5]]
    assert k_on.shape[1] == 3 and pos_on.tolist() == [[2, 5, 6]]


def test_packed_rows_each_document_has_its_own_boundary():
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2)
    S = 16
    x = rand_ids(1, S, seed=9)
    doc = torch.tensor([[0] * 8 + [1] * 8])
    x[0, 5] = M          # doc 0 boundary
    x[0, 12] = M         # doc 1 boundary
    with torch.no_grad():
        ctx = model._message_context(x, doc, None, model._positions(S, 1, doc, x.device))
    assert ctx.side.tolist() == [[0] * 5 + [1] * 3 + [0] * 4 + [1] * 4]
    assert ctx.local_doc_ids.tolist() == [[0] * 5 + [1] * 3 + [2] * 4 + [3] * 4]
    # blocks of 2: [0,1][2,3] sender-0 ; [4,5] straddles ; [6,7] receiver-0 ; [8,9][10,11] sender-1 ; [12,13] receiver-1 ...
    assert ctx.slot_doc.tolist() == [[0, 0, -1, 0, 1, 1, 1, 1]]
    assert ctx.slot_side.tolist() == [[0, 0, 0, 1, 0, 0, 1, 1]]
    mask = dense_message_mask(S, ctx, None, "cpu")[0, 0]
    # doc-1 receiver (t=13) sees doc-1 sender slots (4,5) only; raw keys only 12..13
    assert mask[13, S:].tolist() == [False, False, False, False, True, True, False, False]
    assert mask[13, :S].tolist() == [False] * 12 + [True, True] + [False] * 2
    # doc-0 receiver (t=6) sees slots 0,1 only and raw keys 5..6
    assert mask[6, S:].tolist() == [True, True, False, False, False, False, False, False]
    assert mask[6, :S].tolist() == [False] * 5 + [True, True] + [False] * 9
    with torch.no_grad():
        out = model(x, doc_ids=doc, labels=x)
    assert torch.isfinite(out.loss)
    rem = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=2,
                     message_pool_remainder=True)
    with torch.no_grad():
        ctx_r = rem._message_context(x, doc, None, rem._positions(S, 1, doc, x.device))
    # straddling [4,5] becomes a sender-0 remainder slot (token 4 only)
    assert ctx_r.slot_doc.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    assert ctx_r.slot_side.tolist() == [[0, 0, 0, 1, 0, 0, 1, 1]]
    mask_r = dense_message_mask(S, ctx_r, None, "cpu")[0, 0]
    assert mask_r[6, S:].tolist() == [True, True, True, False, False, False, False, False]


def test_boundary_forward_trains_and_padding_is_safe():
    model = make_model(seed=0, message_boundary_token_id=M, message_compress_ratio=4)
    S, P = 18, 10
    x = with_boundary(rand_ids(2, S, seed=11), P)
    am = torch.ones_like(x)
    am[1, 15:] = 0                       # right padding on row 1
    labels = x.masked_fill(am == 0, -100)
    out = model(x, attention_mask=am, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    gi = model.config.global_layer_index
    assert model.layers[gi].attn.compressor.u.grad is not None
    assert torch.isfinite(model.layers[gi].attn.compressor.u.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex needs CUDA")
def test_message_flex_matches_sdpa_cuda():
    """Whole-model parity of the flex block-mask path (raw keys ‖ slots, KV_LEN = S + n_slots) with
    the dense sdpa mask, in every probe mode, plus a finite backward through the compressor."""
    import copy

    kw = dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, num_kv_heads=2, head_dim=64,
              block=128, pre_window=64, attn_pad_multiple=128, message_boundary_token_id=M, message_compress_ratio=8)
    sdpa = make_model(seed=0, **kw).cuda().to(torch.bfloat16)
    gi = sdpa.config.global_layer_index
    with torch.no_grad():   # a non-trivial message (delta is zero-init); drawn once, shared below
        sdpa.layers[gi].attn.compressor.delta.weight.normal_(0, 0.05)
    torch.manual_seed(0)
    flex = PerceiverARLM(cfg(attn_backend="flex", **kw)).cuda().to(torch.bfloat16)
    flex.load_state_dict(copy.deepcopy(sdpa.state_dict()))
    flex.eval()
    S, P = 1024, 400
    x = with_boundary(rand_ids(2, S, seed=3), P).cuda()
    for mode in ("real", "none", "swapped", "raw"):
        with torch.no_grad(), sdpa.message_override(mode), flex.message_override(mode):
            a = sdpa(input_ids=x).logits.float()
            b = flex(input_ids=x).logits.float()
        scale = a.abs().max().item()
        assert (a - b).abs().max().item() < 0.05 * scale, (mode, (a - b).abs().max().item(), scale)
    out = flex.train()(input_ids=x, labels=x.clone())
    out.loss.backward()
    g = flex.layers[gi].attn.compressor.u.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0

"""E21 message boundary on the Perceiver AR family (`nn/perceiver_ar_lm.py`): severance of every
local channel, prefix visible to receivers only through KVCompressor slots, probe overrides,
arm-U identity at r=1, and the receiver-only round trip through `prefix_kv(as_message=True)`.
CPU / sdpa, tiny config."""
import pytest
import torch

from nn.perceiver_ar_lm import (
    KVCompressor,
    MessageCtx,
    PerceiverARConfig,
    PerceiverARLM,
    analytic_param_count,
    dense_message_mask,
    make_message_mask_pred,
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

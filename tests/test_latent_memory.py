"""E31 sliding-window latent memory: geometry, competition, per-head picks, causality."""
import pytest
import torch

from evaluation.bapo_models import ArchSpec, build_model, n_params
from nn.latent_memory import LatentMemoryWriter, lm_geometry, window_pick
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM, analytic_param_count

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


def lm_cfg(**kw):
    base = dict(
        message_boundary_token_id=M, message_write="latent_memory", lm_context="page_bidir",
        lm_window=16, lm_stride=12, lm_latents=4, lm_latent_dim=32, lm_heads=4, lm_writer_dim=32,
        lm_enc_layers=1, lm_rounds=2, lm_reader_tokens=2,
    )
    base.update(kw)
    return cfg(**base)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**kw)).eval()
    for layer in m.layers:  # warm residuals so the channel is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def row(S=32, qpos=10, seed=1, extra_q=None):
    torch.manual_seed(seed)
    ids = torch.randint(3, 80, (1, S))
    ids[0, qpos] = M
    if extra_q is not None:
        ids[0, extra_q] = M
    return ids


def future_leaks(m, ids, first_receiver):
    """Positions t' < t whose logits move when token t (a receiver token) is perturbed."""
    bad = []
    with torch.no_grad():
        base = m(ids).logits
        for t in range(first_receiver, ids.shape[1]):
            if int(ids[0, t]) == M:
                continue
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 80 + 3
            d = (base - m(ids2).logits)[0, :t].abs().amax(-1)
            moved = (d > 1e-5).nonzero().flatten().tolist()
            if moved:
                bad.append((t, moved))
    return bad


def test_default_config_builds_no_writer():
    m = PerceiverARLM(cfg())
    assert m.memory_writer is None
    m2 = PerceiverARLM(cfg(message_boundary_token_id=M))
    assert m2.memory_writer is None


def test_geometry_fixed_no_auto_shrink():
    c = lm_cfg(lm_window=256, lm_stride=192, lm_latents=32, lm_reader_tokens=5)
    g = lm_geometry(c, 1024)
    assert g.starts == (0, 192, 384, 576, 768)
    assert g.n_slots == 5 * 32 * 5
    short = lm_geometry(c, 128)  # shorter than the window: one window, K not shrunk
    assert short.n_windows == 1 and short.window == 128 and short.latents == 32


def test_window_pick_is_per_window_and_single_side():
    S, q = 32, 10
    side = torch.zeros(1, S, dtype=torch.long)
    side[0, q:] = 1
    doc = torch.zeros(1, S, dtype=torch.long)
    tok, pick, sdoc, sside, has = window_pick(side, doc, None, (0, 12, 16), 16)
    # window 0 straddles QUERY: only sender tokens 0..9
    assert pick[0, 0].nonzero().flatten().tolist() == list(range(10))
    # window 1 (12..27) is receiver-only: its own side, never mixed into window 0
    assert bool(pick[0, 1].all()) and int(sside[0, 1]) == 1 and int(sside[0, 0]) == 0


def test_writer_shapes_and_slot_tags():
    m = make()
    ids = row()
    with torch.no_grad():
        m(ids)
    ctx = m._last_message_ctx
    g = lm_geometry(m.config, ids.shape[1])
    k, v = ctx.slots
    assert k.shape == (1, g.n_slots, m.config.num_kv_heads, m.config.head_dim)
    assert v.shape == k.shape
    assert ctx.slot_doc.shape == (1, g.n_slots) and ctx.slot_pos.shape == (1, g.n_slots)
    # slot positions sit inside their window
    per = g.latents * g.reader_tokens
    for wi, st in enumerate(g.starts):
        p = ctx.slot_pos[0, wi * per:(wi + 1) * per]
        assert int(p.min()) >= st and int(p.max()) < st + g.window


def test_competition_shares_sum_to_one_over_latents():
    w = LatentMemoryWriter(lm_cfg(lm_null_latent=True))
    z = torch.randn(3, w.K + 1, w.D)
    x = torch.randn(3, 16, w.dw)
    mask = torch.ones(3, 16, dtype=torch.bool)
    # recompute the pre-renormalisation shares the module uses
    q = w.qn(w.wq(w.z_norm(z)).view(3, w.K + 1, w.hl, w.dhl))
    k = w.kn(w.wk(w.x_norm(x)).view(3, 16, w.hl, w.dhl))
    logits = torch.einsum("bkhd,bwhd->bhkw", q, k) / w.dhl ** 0.5 + w.prior[:, :, :16].permute(1, 0, 2)[None]
    shares = torch.softmax(logits, dim=2)
    assert torch.allclose(shares.sum(2), torch.ones(3, w.hl, 16), atol=1e-5)
    wts, _ = w._latent_read(z, x, mask, 16)
    assert torch.allclose(wts.sum(-1), torch.ones_like(wts.sum(-1)), atol=1e-3)


def test_heads_are_not_averaged():
    """Each head keeps its own distribution over tokens (E30 averaged them)."""
    w = LatentMemoryWriter(lm_cfg(lm_competition=False))
    torch.manual_seed(0)
    z = torch.randn(2, w.K, w.D)
    x = torch.randn(2, 16, w.dw)
    wts, _ = w._latent_read(z, x, torch.ones(2, 16, dtype=torch.bool), 16)
    assert wts.shape == (2, w.hl, w.K, 16)
    assert not torch.allclose(wts[:, 0], wts[:, 1])


def test_null_latent_is_not_a_slot():
    a = make(lm_null_latent=False)
    b = make(lm_null_latent=True)
    ids = row()
    with torch.no_grad():
        a(ids), b(ids)
    assert a._last_message_ctx.slots[0].shape == b._last_message_ctx.slots[0].shape


@pytest.mark.parametrize("context", ["page_bidir", "bixt"])
@pytest.mark.parametrize("competition", [True, False])
def test_no_future_leak_single_and_double_query(context, competition):
    m = make(lm_context=context, lm_competition=competition, lm_rounds=3 if context == "bixt" else 2)
    assert future_leaks(m, row(), 11) == []
    assert future_leaks(m, row(extra_q=20), 11) == []


def test_sw_perceiver_leak_fixed():
    """E30: a window straddling QUERY no longer pools receiver tokens of a later window."""
    torch.manual_seed(0)
    m = PerceiverARLM(cfg(message_boundary_token_id=M, message_write="sw_perceiver",
                          swp_bank_size=4, swp_coverage=4, swp_auto_fit=False)).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    assert future_leaks(m, row(), 11) == []


def test_memory_is_load_bearing_and_trains():
    m = make()
    ids = row()
    with torch.no_grad():
        real = m(ids).logits
        with m.message_override("none"):
            cut = m(ids).logits
        with m.message_override("swapped"):
            m(torch.cat([ids, row(seed=2)]))
    assert not torch.allclose(real[0, 11:], cut[0, 11:])
    m.train()
    labels = torch.full_like(ids, -100)
    labels[0, 11:] = ids[0, 11:]
    loss = m(ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert m.memory_writer.q.grad is not None and float(m.memory_writer.q.grad.abs().sum()) > 0
    assert m.memory_writer.to_k.weight.grad.abs().sum() > 0


def test_participation_without_query():
    m = make()
    m.train()
    ids = torch.randint(3, 80, (1, 32))
    loss = m(ids, labels=ids).loss
    loss.backward()
    assert m.memory_writer.q.grad is not None


def test_analytic_params_match():
    m = PerceiverARLM(lm_cfg())
    bd = analytic_param_count(m.config)
    assert bd.total == sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("arch", ["e31_page", "e31_bixt", "e30_ctx"])
def test_factory_builds_and_trains(arch):
    spec = ArchSpec(name="t", hidden=64, head_dim=16, message_boundary_token_id=M,
                    lm_window=16, lm_stride=12, lm_latents=4, lm_latent_dim=32, lm_heads=4,
                    lm_writer_dim=32, lm_enc_layers=1, lm_reader_tokens=2,
                    token_embedding_dim=16, ngram_orders=())
    m = build_model(arch, vocab_size=V, seq_len=32, answer_start=11, pad_id=0, bos_id=1, eos_id=2, spec=spec)
    assert n_params(m) > 0
    ids = row()
    labels = torch.full_like(ids, -100)
    labels[0, 11:] = ids[0, 11:]
    loss = m(ids, labels=labels).loss
    assert torch.isfinite(loss)
    if arch == "e30_ctx":
        assert m.config.pre_window == 64 and m.config.message_write == "sw_perceiver"
    else:
        assert m.config.message_write == "latent_memory"
        assert m.config.lm_context == ("bixt" if arch == "e31_bixt" else "page_bidir")


def _token_identity_ratio(model, ids):
    """Within-window spread of token states after the writer's context step, relative to
    their mean: ≈ 0 means every token collapsed to the window average (identity lost)."""
    w = model.memory_writer
    captured = {}
    orig = w._latent_read

    def spy(z, x, mask, W):
        captured.setdefault("x", x.detach())
        return orig(z, x, mask, W)

    w._latent_read = spy
    with torch.no_grad():
        model(ids)
    w._latent_read = orig
    x = captured["x"]
    spread = (x - x.mean(1, keepdim=True)).norm(dim=-1).mean()
    return float(spread / x.mean(1).norm(dim=-1).mean().clamp(min=1e-6))


@pytest.mark.parametrize("context", ["page_bidir", "bixt"])
def test_token_identity_survives_the_context_step(context):
    """Regression: at init the tiny in_proj output was wiped by the first residual update."""
    torch.manual_seed(0)
    m = PerceiverARLM(lm_cfg(lm_context=context, lm_enc_layers=2)).eval()  # post_init scales
    assert _token_identity_ratio(m, row()) > 0.5


@pytest.mark.parametrize("context", ["page_bidir", "bixt"])
def test_slot_values_carry_row_content_at_init(context):
    """Regression: values were the latent's identity (≈ constant across rows), so the reader
    had nothing row-specific to learn from. Values now start as the heads' read-out."""
    torch.manual_seed(0)
    m = PerceiverARLM(lm_cfg(lm_context=context)).eval()
    ids = torch.randint(3, 80, (8, 32))
    ids[:, 10] = M
    with torch.no_grad():
        m(ids)
    v = m._last_message_ctx.slots[1].float().reshape(8, -1, m.config.num_kv_heads * m.config.head_dim)
    ratio = v.std(0).mean() / v.std(1).mean()
    assert float(ratio) > 0.1


# -- long-context length ladder ---------------------------------------------------------

def _sender_moved(m, ids, pos_from, pos_to):
    """Did perturbing token 1 move logits in [pos_from, pos_to)?"""
    with torch.no_grad():
        base = m(ids).logits
        ids2 = ids.clone()
        ids2[0, 1] = (ids2[0, 1] + 7) % 80 + 3
        d = (base - m(ids2).logits)[0].abs().amax(-1)
    return bool((d[pos_from:pos_to] > 1e-5).any()), bool((d[pos_to + 1:] > 1e-5).any())


def test_raw_window_leaves_memory_as_the_only_long_path():
    """With message_raw_window the global read's raw keys are local: a far sender token no
    longer reaches later senders, but the receiver still reads it through the memory."""
    ids = row(S=64, qpos=50)
    full = make(seed=0)
    win = make(seed=0, message_raw_window=8)
    # reach of the local stack: pre window 4 + global window 8 + stack block 6 < 30
    far_full, recv_full = _sender_moved(full, ids, 30, 50)
    far_win, recv_win = _sender_moved(win, ids, 30, 50)
    assert far_full and not far_win
    assert recv_full and recv_win


def test_raw_window_zero_is_unchanged():
    ids = row(S=40, qpos=20)
    a, b = make(seed=3), make(seed=3, message_raw_window=0)
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)


def test_checkpoint_and_ladder_roundtrip(tmp_path):
    """probe --save_ckpt → length_ladder at a longer length (tiny, CPU, seconds)."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ck = tmp_path / "ck"
    probe = [
        sys.executable, str(root / "verification/bapo_capability_probe.py"), "--scale", "tiny",
        "--recipe", "recall_single", "--arch", "e31_page", "--hidden", "64", "--head_dim", "16",
        "--steps", "2", "--k1_mult", "1", "--batch", "2", "--eval_every", "2", "--eval_rows", "2",
        "--lm_window", "32", "--lm_stride", "24", "--lm_latents", "4", "--lm_latent_dim", "32",
        "--lm_heads", "4", "--lm_writer_dim", "32", "--lm_enc_layers", "1", "--lm_reader_tokens", "2",
        "--token_embedding_dim", "16", "--ngram_orders", "none", "--message_raw_window", "16",
        "--no-skip_uncalibrated", "--amp", "off", "--save_ckpt", str(ck), "--out", str(tmp_path / "run"),
    ]
    r = subprocess.run(probe, cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 2), r.stdout[-2000:] + r.stderr[-2000:]
    assert (ck / "e31_page.pt").exists()
    state = torch.load(ck / "e31_page.pt", weights_only=False)
    assert state["spec"]["message_raw_window"] == 16 and state["data"]["seq_len"] == 128
    lad = [
        sys.executable, str(root / "verification/length_ladder.py"), "--ckpt", str(ck),
        "--lengths", "128", "512", "--rows", "4", "--amp", "off", "--backend", "sdpa",
    ]
    r = subprocess.run(lad, cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((ck / "ladder.json").read_text())
    cell = rep["results"]["e31_page"]["512"]
    assert cell["rows"] == 4 and cell["trained_seq_len"] == 128
    assert cell["memory_slots"] > rep["results"]["e31_page"]["128"]["memory_slots"]


def test_chunked_block_mask_matches_the_one_shot_mask():
    """Long global reads build the flex mask in query chunks; the result must be identical."""
    from torch.nn.attention.flex_attention import create_block_mask

    from nn.perceiver_ar_lm import _chunked_block_mask

    S, nb = 300, 100
    torch.manual_seed(0)
    side = torch.zeros(2, S, dtype=torch.long)
    side[:, 250:] = 1
    tag = side.to(torch.int32)

    def pred(b, h, q, kv):
        raw = (kv < S) & (kv <= q) & (q - kv < 64) & (tag[b, q] == tag[b, torch.clamp(kv, max=S - 1)])
        slot = (kv >= S) & (tag[b, q] >= 1)
        return raw | slot

    one = create_block_mask(pred, B=2, H=None, Q_LEN=S, KV_LEN=S + nb, device="cpu")
    chunked = _chunked_block_mask(pred, B=2, Q_LEN=S, KV_LEN=S + nb, device="cpu", max_pairs=128 * (S + nb))
    assert torch.equal(one.to_dense(), chunked.to_dense())

"""E33d — a page distance on the note read: one signed slope per head of the global read, starting
at 0, score −= s_h · f((pos(reader) − pos(note)) / unit), added outside the key.

Spec: docs/experiments_specs/ahead/E33d_note_page_distance.md (+ _plan.md, section 7).
"""
import math

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARConfig, attend_message, dense_message_mask, note_age_bias
from tests.test_latent_memory import lm_cfg
from tests.test_text_memory import TEXT, make, text_row

LOOP = dict(message_loop_rounds=2, message_loop_exit_aux=0.3, message_loop_exit_targets="answer")
# On a GPU host the flex path is compiled (as in training) and must get CUDA tensors; on a laptop it runs eager on CPU.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-4 if DEV == "cuda" else 1e-5
# compiled flex on CUDA needs >= 16 dims per head; the laptop keeps the shared tiny shape
WIDE = dict(hidden_size=64, head_dim=16) if DEV == "cuda" else {}
HD = WIDE.get("head_dim", 8)


def _slopes(m):
    return [(i, layer.attn.note_age_slope) for i, layer in enumerate(m.layers)
            if layer.attn.note_age_slope is not None]


# ------------------------------------------------------------------ config and defaults
def test_defaults_build_no_slope_and_validation():
    assert not _slopes(make(**TEXT))
    with pytest.raises(ValueError):
        lm_cfg(**TEXT, lm_note_age="bogus")
    with pytest.raises(ValueError):  # the exam read's slot positions are not the notes' read positions
        lm_cfg(lm_note_age="log")
    with pytest.raises(ValueError):
        lm_cfg(**TEXT, lm_note_age="log", lm_note_age_unit=-1)
    # an old checkpoint's config.json (no E33d keys) loads with the age term off
    d = lm_cfg(**TEXT).to_dict()
    d.pop("lm_note_age"), d.pop("lm_note_age_unit")
    c = PerceiverARConfig(**d)
    assert c.lm_note_age == "none" and c.lm_note_age_unit == 0


def test_one_slope_per_head_on_the_global_read_only():
    m = make(**TEXT, lm_note_age="log")
    s = _slopes(m)
    assert [i for i, _ in s] == [m.config.global_layer_index]
    assert s[0][1].shape == (m.config.num_attention_heads,) and torch.all(s[0][1] == 0)
    assert s[0][1] is not None and m.layers[s[0][0]].attn.note_age_unit == float(m.config.lm_stride)


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_zero_slopes_are_exactly_e33(backend):
    a = make(seed=3, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend)
    b = make(seed=3, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend, lm_note_age="log")
    res = b.load_state_dict(a.state_dict(), strict=False)
    gi = a.config.global_layer_index
    assert res.missing_keys == [f"layers.{gi}.attn.note_age_slope"] and not res.unexpected_keys
    a, b, ids = a.to(DEV), b.to(DEV), text_row(S=48).to(DEV)
    with torch.no_grad():
        assert torch.allclose(a(ids).logits, b(ids).logits, atol=TOL)


# ------------------------------------------------------------------ the bias itself
def test_bias_values_log_and_linear():
    pos = torch.tensor([[0.0, 100.0, 400.0]])
    spos = torch.tensor([[0.0, 300.0]])
    slope = torch.tensor([1.0, -2.0])
    b = note_age_bias(pos, spos, slope, 192, "log")
    assert b.shape == (1, 2, 3, 2)
    assert math.isclose(float(b[0, 0, 2, 0]), -math.log1p(400 / 192), rel_tol=1e-6)
    assert math.isclose(float(b[0, 0, 2, 1]), -math.log1p(100 / 192), rel_tol=1e-6)
    assert math.isclose(float(b[0, 1, 2, 0]), 2 * math.log1p(400 / 192), rel_tol=1e-6)
    assert float(b[0, 0, 1, 1]) == 0.0  # a note ahead of the reader (masked anyway) gets age 0
    lin = note_age_bias(pos, spos, slope, 192, "linear")
    assert math.isclose(float(lin[0, 0, 2, 0]), -400 / 192, rel_tol=1e-6)


def _read_setup(S=40, seed=0):
    torch.manual_seed(seed)
    m = make(**TEXT, **WIDE, message_raw_window=8).to(DEV)
    ids = text_row(S=S).to(DEV)
    with torch.no_grad():
        m(ids)
    ctx = m._last_message_ctx
    B, h, g, dh, nb = 1, 4, 2, HD, ctx.n_slots
    r = lambda *shape: torch.randn(*shape).to(DEV)  # noqa: E731
    t = dict(q=r(B, S, h, dh), q_un=r(B, S, h, dh), k=r(B, S, g, dh), v=r(B, S, g, dh),
             kb=r(B, nb, g, dh), vb=r(B, nb, g, dh))
    pos = torch.arange(S, device=DEV)[None]
    return ctx, t, pos, (B, S, h, g, dh, nb)


def test_sdpa_read_with_bias_matches_reference_and_flex():
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup()
    slope = torch.tensor([0.0, 1.5, -1.0, 3.0], device=DEV)
    unit = 12.0
    nbias = (slope, pos, ctx.slot_pos, unit, "log")
    out = attend_message(t["q"], t["k"], t["v"], t["kb"], t["vb"], ctx=ctx, key_valid=None, backend="sdpa",
                         q_slot=t["q_un"], note_bias=nbias)
    rep = h // g
    kr, vr = t["k"].repeat_interleave(rep, 2), t["v"].repeat_interleave(rep, 2)
    kbr, vbr = t["kb"].repeat_interleave(rep, 2), t["vb"].repeat_interleave(rep, 2)
    raw = torch.einsum("bqhd,bkhd->bhqk", t["q"], kr) / dh ** 0.5
    notes = torch.einsum("bqhd,bkhd->bhqk", t["q_un"], kbr) / dh ** 0.5
    notes = notes + note_age_bias(pos, ctx.slot_pos, slope, unit, "log")
    mask = dense_message_mask(S, ctx, None, DEV)
    w = torch.softmax(torch.cat([raw, notes], -1).masked_fill(~mask, float("-inf")), -1)
    ref = torch.einsum("bhqk,bkhd->bqhd", w, torch.cat([vr, vbr], 1))
    assert torch.allclose(out, ref, atol=TOL)
    out_f = attend_message(t["q"], t["k"], t["v"], t["kb"], t["vb"], ctx=ctx, key_valid=None, backend="flex",
                           q_slot=t["q_un"], note_bias=nbias)
    assert torch.allclose(out_f, out, atol=1e-4), float((out_f - out).abs().max())


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_positive_slope_prefers_newer_notes_negative_older(backend):
    """Every note has the same key, so content cannot tell them apart; value channel 0 counts note
    mass and channel 1 carries the note's position. The read's mean note position must rise with the
    slope (newer notes) and fall below the content-only read with a negative slope (older notes)."""
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup(S=48)
    kb = t["kb"][:, :1].expand(B, nb, g, dh).contiguous()
    vb = torch.zeros(B, nb, g, dh, device=DEV)
    vb[..., 0] = 1.0
    vb[..., 1] = ctx.slot_pos.float()[..., None]
    v = torch.zeros_like(t["v"])

    def mean_note_pos(sl):
        nbias = (torch.full((h,), sl, device=DEV), pos, ctx.slot_pos, 12.0, "log")
        o = attend_message(t["q"], t["k"], v, kb, vb, ctx=ctx, key_valid=None, backend=backend,
                           q_slot=t["q_un"], note_bias=nbias)
        return (o[0, -1, :, 1] / o[0, -1, :, 0]).mean()  # last reader: sees the most closed pages

    newer, flat, older = mean_note_pos(3.0), mean_note_pos(0.0), mean_note_pos(-3.0)
    assert newer > flat + 1.0 and older < flat - 1.0, (float(newer), float(flat), float(older))


# ------------------------------------------------------------------ training
@pytest.mark.parametrize("backend", ["sdpa", pytest.param("flex", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex attention has no backward pass on CPU"))])
def test_slopes_receive_gradient_through_the_loop(backend):
    dev = "cuda" if backend == "flex" else "cpu"
    m = make(seed=5, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend, lm_note_age="log")
    m = m.to(dev).train()
    ids = text_row(S=48).to(dev)
    m(ids, labels=ids).loss.backward()
    (_, s), = _slopes(m)
    assert s.grad is not None and float(s.grad.abs().sum()) > 0




def test_text_checks_arch_carries_the_age_term():
    from evaluation.text_checks import ARCHES, count_params, model_args

    for arch in ("e33d", "e33d_plain"):
        assert ARCHES[arch].args["lm_note_age"] == "log" and ARCHES[arch].has_notebook
    base = model_args("e31c_loop_plain", "screen", 1152)
    new = model_args("e33d_plain", "screen", 1152)
    assert {k for k in new if new[k] != base.get(k)} == {"lm_note_age"}
    assert count_params(new) - count_params(base) == new["num_attention_heads"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention has no backward pass on CPU")
def test_flex_slope_gradient_matches_sdpa():
    """The flex path keeps a per-query copy of the slopes (fast atomic adds); its gradient must equal the
    SDPA path's, with non-zero slopes of both signs."""
    grads = {}
    for backend in ("sdpa", "flex"):
        m = make(seed=6, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend, lm_note_age="log")
        (gi, s), = _slopes(m)
        with torch.no_grad():
            s.copy_(torch.tensor([0.5, -0.3, 0.0, 1.0]))
        m = m.to("cuda").train()
        ids = text_row(S=48).to("cuda")
        m(ids, labels=ids).loss.backward()
        grads[backend] = m.layers[gi].attn.note_age_slope.grad.detach().cpu()
    assert torch.allclose(grads["flex"], grads["sdpa"], rtol=2e-3, atol=1e-5), grads

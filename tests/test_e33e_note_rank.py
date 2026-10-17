"""E33e — the notes' order among matching notes: score −= s_h · log(1 + soft count of newer notes that match the
reader about as well as its best note). Same per-head strength as E33d (starts at 0); a lone fact pays nothing.

Spec: docs/experiments_specs/ahead/E33e_note_rank.md.
"""
import math

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARConfig, attend_message, dense_message_mask, note_rank_logcount
from tests.test_e33d_note_age import DEV, HD, LOOP, TOL, WIDE, _read_setup, _slopes
from tests.test_latent_memory import lm_cfg
from tests.test_text_memory import TEXT, make, text_row

RANK = dict(lm_note_age="rank")


def _brute(q_slot, kb, ctx, scale, rep, margin, sharp):
    """Reference: for every reader t, head h, note c — the summed gate of visible notes on strictly newer pages."""
    B, S, h, dh = q_slot.shape
    C, g = kb.shape[1], kb.shape[2]
    kbh = kb.repeat_interleave(h // g, dim=2)
    out = torch.zeros(B, h, S, C // rep)
    for b in range(B):
        for t in range(S):
            vis = [c for c in range(C) if ctx.slot_doc[b, c] >= 0 and ctx.slot_doc[b, c] == ctx.doc[b, t]
                   and 0 <= ctx.slot_close[b, c] <= t]
            for hh in range(h):
                if not vis:
                    continue
                l = {c: float(q_slot[b, t, hh] @ kbh[b, c, hh]) * scale for c in vis}
                m = max(l.values())
                for w in range(C // rep):
                    close_w = int(ctx.slot_close[b, w * rep])
                    cnt = sum(1 / (1 + math.exp(-sharp * (l[c] - m + margin))) for c in vis
                              if int(ctx.slot_close[b, c]) > close_w)
                    out[b, hh, t, w] = math.log1p(cnt)
    return out


def test_config_accepts_rank_and_old_configs_load():
    c = lm_cfg(**TEXT, **RANK)
    assert c.lm_note_age == "rank" and c.lm_note_rank_margin == 1.0 and c.lm_note_rank_sharp == 2.0
    with pytest.raises(ValueError):
        lm_cfg(**TEXT, **RANK, lm_note_rank_sharp=0.0)
    d = lm_cfg(**TEXT, lm_note_age="log").to_dict()
    d.pop("lm_note_rank_margin"), d.pop("lm_note_rank_sharp")
    c2 = PerceiverARConfig(**d)  # an E33d checkpoint's config.json
    assert c2.lm_note_age == "log" and c2.lm_note_rank_margin == 1.0


def test_logcount_matches_the_brute_force_reference():
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup(S=40)
    m = make(**TEXT, **WIDE, message_raw_window=8)
    rep = m.config.lm_latents * m.config.lm_reader_tokens
    q, kb = t["q_un"].cpu(), t["kb"].cpu()
    ctx_cpu = type(ctx)(**{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(ctx).items()})
    for margin, sharp in ((1.0, 2.0), (3.0, 0.5)):
        got = note_rank_logcount(q, kb, ctx_cpu, scale=dh ** -0.5, rep=rep, margin=margin, sharp=sharp, chunk=7)
        ref = _brute(q, kb, ctx_cpu, dh ** -0.5, rep, margin, sharp)
        assert torch.allclose(got, ref, atol=1e-5), float((got - ref).abs().max())


def _one_page_matches(S=48, match_pages=(0,), strong=8.0):
    """Notes keyed so that only the pages in `match_pages` match the readers' query (score `strong`),
    every other note scores 0. Returns the log-count for the last reader, head 0, per page."""
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup(S=S)
    m = make(**TEXT, **WIDE, message_raw_window=8)
    rep = m.config.lm_latents * m.config.lm_reader_tokens
    q = torch.zeros(B, S, h, dh, device=DEV)
    q[..., 0] = 1.0
    kb = torch.zeros(B, nb, g, dh, device=DEV)
    for w in match_pages:
        kb[:, w * rep:(w + 1) * rep, :, 0] = strong
    lc = note_rank_logcount(q, kb, ctx, scale=1.0, rep=rep, margin=1.0, sharp=2.0)
    n_vis_pages = int((ctx.slot_close[0, ::rep] <= S - 1).sum())
    return lc[0, 0, -1], rep, n_vis_pages


def test_a_lone_fact_pays_nothing_however_many_pages_follow():
    lc, rep, n = _one_page_matches(match_pages=(0,))
    assert n >= 3
    # the matching page has no newer match: count ≈ rep · (n−1) · σ(−2·7) ≈ 0, where a page-distance read
    # would charge this oldest page the most
    assert float(lc[0]) < 1e-3


def test_among_matching_pages_the_newest_pays_nothing_and_older_pay_more():
    lc, rep, n = _one_page_matches(match_pages=(0, 1))
    assert n >= 3
    # page 0 is charged for page 1's notes, each matching at the best score: gate σ(sharp · margin) = σ(2)
    full = rep / (1 + math.exp(-2.0))
    assert float(lc[1]) < 1e-3 and math.isclose(float(lc[0]), math.log1p(full), rel_tol=1e-3)


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_zero_strength_is_exactly_e33(backend):
    a = make(seed=3, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend)
    b = make(seed=3, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend, **RANK)
    res = b.load_state_dict(a.state_dict(), strict=False)
    gi = a.config.global_layer_index
    assert res.missing_keys == [f"layers.{gi}.attn.note_age_slope"] and not res.unexpected_keys
    a, b, ids = a.to(DEV), b.to(DEV), text_row(S=48).to(DEV)
    with torch.no_grad():
        assert torch.allclose(a(ids).logits, b(ids).logits, atol=TOL)


def test_sdpa_and_flex_reads_agree_with_the_rank_bias():
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup()
    m = make(**TEXT, **WIDE, message_raw_window=8)
    rep = m.config.lm_latents * m.config.lm_reader_tokens
    lc = note_rank_logcount(t["q_un"], t["kb"], ctx, scale=dh ** -0.5, rep=rep,
                            margin=1.0, sharp=2.0)
    slope = torch.tensor([0.0, 1.5, -1.0, 3.0], device=DEV)
    nbias = (slope, lc, None, rep, "rank")
    out = attend_message(t["q"], t["k"], t["v"], t["kb"], t["vb"], ctx=ctx, key_valid=None, backend="sdpa",
                         q_slot=t["q_un"], note_bias=nbias)
    r = h // g
    kr, vr = t["k"].repeat_interleave(r, 2), t["v"].repeat_interleave(r, 2)
    kbr, vbr = t["kb"].repeat_interleave(r, 2), t["vb"].repeat_interleave(r, 2)
    raw = torch.einsum("bqhd,bkhd->bhqk", t["q"], kr) / dh ** 0.5
    notes = torch.einsum("bqhd,bkhd->bhqk", t["q_un"], kbr) / dh ** 0.5
    notes = notes - slope[None, :, None, None] * lc.repeat_interleave(rep, dim=-1)
    mask = dense_message_mask(S, ctx, None, DEV)
    w = torch.softmax(torch.cat([raw, notes], -1).masked_fill(~mask, float("-inf")), -1)
    ref = torch.einsum("bhqk,bkhd->bqhd", w, torch.cat([vr, vbr], 1))
    assert torch.allclose(out, ref, atol=TOL)
    out_f = attend_message(t["q"], t["k"], t["v"], t["kb"], t["vb"], ctx=ctx, key_valid=None, backend="flex",
                           q_slot=t["q_un"], note_bias=nbias)
    assert torch.allclose(out_f, out, atol=1e-4), float((out_f - out).abs().max())


@pytest.mark.parametrize("backend", ["sdpa", pytest.param("flex", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex attention has no backward pass on CPU"))])
def test_strength_receives_gradient_through_the_loop(backend):
    dev = "cuda" if backend == "flex" else "cpu"
    m = make(seed=5, **TEXT, **LOOP, **WIDE, message_raw_window=8, attn_backend=backend, **RANK)
    (gi, s), = _slopes(m)
    with torch.no_grad():
        s.fill_(0.5)  # at 0 the gradient can vanish on a tiny random row; any value works for the plumbing check
    m = m.to(dev).train()
    ids = text_row(S=48).to(dev)
    m(ids, labels=ids).loss.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0


def test_notebook_off_skips_the_count():
    m = make(seed=2, **TEXT, **LOOP, **WIDE, message_raw_window=8, **RANK).to(DEV)
    (gi, s), = _slopes(m)
    with torch.no_grad():
        s.fill_(2.0)
    ids = text_row(S=48).to(DEV)
    with torch.no_grad(), m.message_override("none"):
        assert torch.isfinite(m(ids).logits).all()


def test_text_checks_arch_carries_the_rank_term():
    from evaluation.text_checks import ARCHES, count_params, model_args

    for arch in ("e33e", "e33e_plain"):
        assert ARCHES[arch].args["lm_note_age"] == "rank" and ARCHES[arch].has_notebook
    d = model_args("e33d_plain", "screen", 1152)
    e = model_args("e33e_plain", "screen", 1152)
    assert {k for k in e if e[k] != d.get(k)} == {"lm_note_age"}
    assert count_params(e) == count_params(d)


def test_exam_time_probe_puts_the_rank_read_on_an_e33_model():
    """evaluation/text_checks_eval.py --note_order rank: strength 0 leaves an E33 model's logits unchanged;
    a positive strength changes them (the read now counts newer matches)."""
    from evaluation.text_checks_eval import apply_note_order

    m = make(seed=4, **TEXT, **LOOP, **WIDE, message_raw_window=8).to(DEV).eval()
    ids = text_row(S=48).to(DEV)
    with torch.no_grad():
        base = m(ids).logits
        done = apply_note_order(m, "rank", 0.0, 1.0, 2.0)
        assert list(done) == [f"layers.{m.config.global_layer_index}.attn"]
        assert torch.allclose(m(ids).logits, base, atol=TOL)
        apply_note_order(m, "rank", 3.0, 1.0, 2.0)
        assert not torch.allclose(m(ids).logits, base, atol=1e-3)
        m._loop_rounds_override = 1  # rounds probe: one round decodes through the coda
        assert torch.isfinite(m(ids).logits).all()

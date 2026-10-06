"""E31c — the E31 notebook on plain text: closed-window read + reader-relative slot keys.

Spec: docs/experiments_specs/ahead/E31c_text_memory_read.md (+ _plan.md, tests 1–8).
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from evaluation.bapo_models import ArchSpec, build_model, n_params
from nn.perceiver_ar_lm import (
    PerceiverARConfig,
    PerceiverARLM,
    _band_block_mask,
    _message_extra_blocks,
    attend_message,
    dense_message_mask,
    make_message_mask_pred,
)
from tests.test_latent_memory import _covers, lm_cfg, row

TEXT = dict(message_boundary_token_id=-1, lm_read="closed", lm_slot_pos="reader", lm_addr="none",
            lm_reader_tokens=1)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**kw)).eval()
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def text_row(S=48, seed=1):
    torch.manual_seed(seed)
    return torch.randint(3, 80, (1, S))  # no QUERY token anywhere


def first_close(m, S):
    ctx = m._last_message_ctx
    c = ctx.slot_close[0][ctx.slot_doc[0] >= 0]
    return int(c.min())


# ------------------------------------------------------------------ 1. defaults unchanged
def test_config_defaults_and_validation():
    c = lm_cfg()
    assert c.lm_read == "exclusive" and c.message_enabled
    t = lm_cfg(**TEXT)
    assert t.message_enabled and t.message_boundary_token_id == -1
    with pytest.raises(ValueError):
        lm_cfg(lm_read="bogus")
    with pytest.raises(ValueError):  # the closed read is a latent-memory read
        lm_cfg(message_write="block_mean", lm_read="closed")
    with pytest.raises(ValueError):  # exclusive latent memory still needs a question boundary
        lm_cfg(message_boundary_token_id=-1)
    # a config without the new keys (an old checkpoint's config.json) loads as E31
    d = lm_cfg().to_dict()
    d.pop("lm_read")
    assert PerceiverARConfig(**d).lm_read == "exclusive"


def test_explicit_exclusive_is_the_default_model():
    ids = row()
    a, b = make(seed=4), make(seed=4, lm_read="exclusive")
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)
    assert a._last_message_ctx.read_rule == "exclusive" and not a._last_message_ctx.slot_nope


# ------------------------------------------------------------------ 2. closed = exclusive on exam answers
def _spec(**kw):
    return ArchSpec(name="shared", hidden=64, head_dim=16, n_kv_heads=1, pre_layers=1, global_layers=1,
                    stack_layers=2, local_window=16, token_embedding_dim=32, ngram_orders=(),
                    lm_latent_dim=64, lm_writer_dim=32, zero_init_residuals=False, message_raw_window=256, **kw)


def _exam(recipe, scale, **over):
    from data.bapo_ladder import config_for, resolve_recipe
    from verification.bapo_capability_probe import _boundary_token_id, make_batch

    rc = resolve_recipe(recipe)
    c = config_for(scale, rc.task, **{**rc.overrides, **over})
    ids, labels = make_batch(c, np.random.default_rng(0), 3, "cpu")
    return c, ids, labels, _boundary_token_id(c)


def _build(arch, c, q_id, **spec_kw):
    return build_model(arch, vocab_size=c.vocab.vocab_size, seq_len=c.seq_len, answer_start=c.answer_start,
                       pad_id=c.vocab.control("eos"), bos_id=c.vocab.control("bos"), eos_id=c.vocab.control("eos"),
                       spec=_spec(message_boundary_token_id=q_id, **spec_kw), seed=0).eval()


EXAMS = [
    ("recall_single", "tiny", {}),
    ("select_1decoy", "tiny", {}),
    ("chain_ordered", "bridge_1k", {"hops": 4}),
    ("fact_markov_single", "bridge", {}),
]


@pytest.mark.parametrize("recipe,scale,over", EXAMS)
@pytest.mark.parametrize("arch", ["e31_li_m1", "e33a_loop"])
def test_closed_read_gives_exam_answers_exactly_the_e31_notes(arch, recipe, scale, over):
    """On exam rows every book window closes before QUERY and the answer region is shorter than a
    window, so the answers read the same notes; book tokens' extra reads never reach them."""
    c, ids, labels, q_id = _exam(recipe, scale, **over)
    excl = _build(arch, c, q_id)
    closed = _build(arch, c, q_id, lm_read="closed")
    assert closed.config.lm_read == "closed" and excl.config.lm_read == "exclusive"
    closed.load_state_dict(excl.state_dict())
    with torch.no_grad():
        a, b = excl(ids).logits, closed(ids).logits
    ans = labels != -100
    assert bool(ans.any())
    assert torch.allclose(a[ans], b[ans], atol=1e-5, rtol=0), float((a[ans] - b[ans]).abs().max())
    if c.seq_len > 256:  # book tokens now read the closed book windows
        assert not torch.allclose(a[~ans], b[~ans], atol=1e-5)


# ------------------------------------------------------------------ 3. text causality
@pytest.mark.parametrize("slot_pos", ["reader", "read"])
def test_text_mode_is_causal_with_packed_documents(slot_pos):
    m = make(**{**TEXT, "lm_slot_pos": slot_pos, "message_raw_window": 8})
    ids = text_row(S=48)
    doc = torch.zeros(1, 48, dtype=torch.long)
    doc[0, 30:] = 1  # two packed documents; window 24..39 straddles them
    with torch.no_grad():
        base = m(ids, doc_ids=doc).logits
        assert m._last_message_ctx is not None and m._last_message_ctx.read_rule == "closed"
        for t in range(1, 48):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 77 + 3
            moved = (base - m(ids2, doc_ids=doc).logits)[0, :t].abs().amax(-1)
            assert float(moved.max()) < 1e-5, (t, (moved > 1e-5).nonzero().flatten().tolist())


def test_documents_never_read_each_others_notes():
    m = make(**TEXT, message_raw_window=8)
    ids = text_row(S=48)
    doc = torch.zeros(1, 48, dtype=torch.long)
    doc[0, 30:] = 1
    with torch.no_grad():
        base = m(ids, doc_ids=doc).logits
        ids2 = ids.clone()
        ids2[0, :30] = (ids2[0, :30] + 5) % 77 + 3  # rewrite document 0 entirely
        assert torch.allclose(base[0, 30:], m(ids2, doc_ids=doc).logits[0, 30:], atol=1e-5)


# ------------------------------------------------------------------ 4. the notebook is live in text
def test_notebook_is_read_by_every_token_after_its_window_closes():
    m = make(**TEXT, message_raw_window=8)
    ids = text_row(S=48)
    with torch.no_grad():
        real = m(ids).logits
        fc = first_close(m, 48)
        with m.message_override("none"):
            none = m(ids).logits
    d = (real - none)[0].abs().amax(-1)
    assert float(d[:fc].max()) < 1e-6  # nothing closed yet: no note to read
    assert bool((d[fc:] > 1e-4).all())  # from the first closed window on, every token reads notes


def test_text_loss_trains_the_writer():
    m = make(**TEXT, message_raw_window=8).train()
    ids = text_row(S=48)
    out = m(ids, labels=ids.clone())
    assert torch.isfinite(out.loss)
    out.loss.backward()
    g = [p.grad for p in m.memory_writer.parameters() if p.grad is not None]
    assert g and sum(float(x.abs().sum()) for x in g) > 0


def test_old_behaviour_without_query_was_no_notebook():
    """Exclusive E31 on a row with no QUERY skips the notebook (why text needs the closed read)."""
    m = make()
    with torch.no_grad():
        m(text_row(S=48))
    assert m._last_message_ctx is None


# ------------------------------------------------------------------ 5. mask semantics
def test_closed_mask_rule_dense_and_flex_predicate():
    m = make(**TEXT, message_raw_window=8)
    ids = text_row(S=60)
    doc = torch.zeros(1, 60, dtype=torch.long)
    doc[0, 40:] = 1
    with torch.no_grad():
        m(ids, doc_ids=doc)
    ctx = m._last_message_ctx
    S, nb = 60, ctx.n_slots
    dense = dense_message_mask(S, ctx, None, "cpu")[0, 0]  # [S, S + nb]
    q = torch.arange(S)[:, None]
    want = (ctx.slot_doc[0][None] == doc[0][:, None]) & (ctx.slot_doc[0][None] >= 0) & (ctx.slot_close[0][None] <= q)
    assert torch.equal(dense[:, S:], want)
    assert bool(want.any()) and not bool(want.all())
    pred = make_message_mask_pred(S, ctx, None)
    qq = torch.arange(S)[:, None].expand(S, S + nb)
    kv = torch.arange(S + nb)[None].expand(S, S + nb)
    grid = pred(torch.zeros_like(qq), None, qq, kv)
    assert torch.equal(grid, dense)


@pytest.mark.parametrize("S", [200, 300])
def test_band_block_lists_cover_the_closed_read(S):
    m = make(**TEXT, message_raw_window=64)
    ids = text_row(S=S)
    with torch.no_grad():
        m(ids)
    ctx = m._last_message_ctx
    pred = make_message_mask_pred(S, ctx, None)
    extra = _message_extra_blocks(ctx, S)
    assert extra is not None and bool((extra[0, 0] < 0).any())  # the first query block skips unclosed slots
    bm = _band_block_mask(pred, B=1, Q_LEN=S, KV_LEN=S + ctx.n_slots, window=64, causal=True, device="cpu",
                          extra_blocks=extra)
    assert _covers(bm, pred, 1, S, S + ctx.n_slots)[0]


# ------------------------------------------------------------------ 6. reader slot keys are exact NoPE
def test_reader_read_is_one_softmax_rope_raw_nope_slots():
    torch.manual_seed(0)
    m = make(**TEXT, message_raw_window=8)
    ids = text_row(S=40)
    with torch.no_grad():
        m(ids)
    ctx = m._last_message_ctx
    B, S, h, g, dh, nb = 1, 40, 4, 2, 8, ctx.n_slots
    q, q_un = torch.randn(B, S, h, dh), torch.randn(B, S, h, dh)
    k, v = torch.randn(B, S, g, dh), torch.randn(B, S, g, dh)
    kb, vb = torch.randn(B, nb, g, dh), torch.randn(B, nb, g, dh)
    out = attend_message(q, k, v, kb, vb, ctx=ctx, key_valid=None, backend="sdpa", q_slot=q_un)
    rep = h // g
    kr, vr = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
    kbr, vbr = kb.repeat_interleave(rep, 2), vb.repeat_interleave(rep, 2)
    logits = torch.cat([torch.einsum("bqhd,bkhd->bhqk", q, kr), torch.einsum("bqhd,bkhd->bhqk", q_un, kbr)], -1)
    logits = logits / dh ** 0.5
    mask = dense_message_mask(S, ctx, None, "cpu")
    w = torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)
    ref = torch.einsum("bhqk,bkhd->bqhd", w, torch.cat([vr, vbr], 1))
    assert torch.allclose(out, ref, atol=1e-5)


def test_reader_keys_are_unrotated_and_flex_matches_sdpa():
    a = make(seed=2, **TEXT, message_raw_window=8)
    b = make(seed=2, **TEXT, message_raw_window=8, attn_backend="flex")
    b.load_state_dict(a.state_dict())
    ids = text_row(S=48)
    with torch.no_grad():
        la = a(ids).logits
        lb = b(ids).logits
    assert a._last_message_ctx.slot_nope
    assert torch.allclose(la, lb, atol=2e-4), float((la - lb).abs().max())


def test_reader_slot_keys_ignore_position():
    """Shifting every position (position_offset) leaves the slot logits alone: NoPE slots and
    relative RoPE on raw keys make the whole model shift-invariant."""
    m = make(**TEXT, message_raw_window=8)
    ids = text_row(S=48)
    with torch.no_grad():
        a = m(ids).logits
        b = m(ids, position_offset=5000).logits
    assert torch.allclose(a, b, atol=1e-4)


# ------------------------------------------------------------------ 7. trainer factory
def test_text_trainer_factory_builds_the_e31c_model():
    from types import SimpleNamespace

    from training.concept_pretraining_factories import _build_perceiver_ar_model

    tok = type("T", (), {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "__len__": lambda self: 97})()
    from training.concept_pretraining_args import ModelArguments

    ma = ModelArguments(
        model_family="perceiver_ar", hidden_size=64, intermediate_size=128, token_embedding_dim=16,
        par_mode="perceiver", par_pre_layers=1, par_pre_window=16, par_global_layers=1, num_hidden_layers=2,
        par_block=16, head_dim=16, num_kv_heads=1, par_ngram_orders="", par_value_embed_layers="",
        attn_backend="sdpa", attn_pad_multiple=1, use_liger=False,
        message_write="latent_memory", lm_read="closed", lm_slot_pos="reader", lm_addr="none",
        lm_reader_tokens=1, lm_window=32, lm_stride=24, lm_latents=4, lm_latent_dim=32, lm_heads=4,
        lm_writer_dim=32, lm_enc_layers=1, message_raw_window=16, message_override="none",
    )
    da = SimpleNamespace(max_seq_length=96, tokenizer_name="test")
    model, config, _ = _build_perceiver_ar_model(tok, ma, da)
    assert config.lm_read == "closed" and config.message_write == "latent_memory"
    assert config.message_raw_window == 16 and model.memory_writer is not None
    assert model._message_override == "none" and config.message_override == "none"
    # the control's saved config keeps it without a notebook when reloaded
    assert PerceiverARLM(PerceiverARConfig(**config.to_dict()))._message_override == "none"
    ids = torch.randint(3, 90, (2, 96))
    out = model(ids, labels=ids.clone())
    assert torch.isfinite(out.loss) and model._last_message_ctx.read_rule == "closed"


# ------------------------------------------------------------------ 8. suite variants
def test_suite_variants_build_with_e31_params():
    kw = dict(vocab_size=17, seq_len=512, answer_start=480, pad_id=12, bos_id=11, eos_id=12)
    base = build_model("e31_li_m1", spec=_spec(message_boundary_token_id=10), seed=0, **kw)
    m1 = build_model("e31c_m1", spec=_spec(message_boundary_token_id=10), seed=0, **kw)
    loop = build_model("e31c_loop", spec=_spec(message_boundary_token_id=10), seed=0, **kw)
    assert m1.config.lm_read == "closed" and m1.config.lm_slot_pos == "reader"
    assert m1.config.lm_reader_tokens == 1 and m1.config.lm_addr == "none"
    assert n_params(m1) == n_params(base)
    assert loop.config.message_loop_rounds == 4 and loop.config.message_loop_exit_targets == "answer"
    assert loop.config.lm_read == "closed"
    from evaluation.capability_suite import ARCH_FLAGS

    assert ARCH_FLAGS["e31c_m1"] == ARCH_FLAGS["e31_li_m1"]


# ------------------------------------------------------------------ text checks smoke fixes (2026-10-06)
def test_reader_read_on_mps_matches_cpu():
    """MPS SDPA returns the QK width when QK ≠ V width; attend_message pads V so it stays exact."""
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    m = make(seed=3, **TEXT, message_raw_window=8)
    ids = text_row(S=48)
    with torch.no_grad():
        cpu = m(ids).logits
        mps = m.to("mps")(ids.to("mps")).logits.cpu()
    # MPS float32 matmuls round more coarsely: the notebook-free model shows the same ~7e-3 gap
    assert cpu.shape == mps.shape
    assert torch.allclose(cpu, mps, atol=2e-2), float((cpu - mps).abs().max())


def test_text_loop_eval_loss_is_plain_next_token_loss():
    """E33a loop on text: the exit aux trains, but the eval loss is the plain CE every arm shares."""
    m = make(**TEXT, message_raw_window=8, message_loop_rounds=4, message_loop_exit_aux=0.3,
             message_loop_exit_targets="answer")
    ids = text_row(S=48)
    with torch.no_grad():
        ev = m(ids, labels=ids.clone()).loss
        m.config.message_loop_exit_aux = 0.0
        plain = m(ids, labels=ids.clone()).loss
        m.config.message_loop_exit_aux = 0.3
        tr = m.train()(ids, labels=ids.clone()).loss
    assert torch.allclose(ev, plain)
    assert float(tr) > float(plain) + 1e-3

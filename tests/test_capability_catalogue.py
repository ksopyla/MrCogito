"""E33t catalogue read in the capability checks: the named variant `e33t_catalogue_n4` is the text arm
`catalogue_reuse_n4` on the capability platform (spec docs/experiments_specs/ahead/E33t_catalogue_read.md,
"Capability checks"). Older arches must build exactly as before."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from evaluation.bapo_models import ARCH_VARIANTS, ArchSpec, _note_open_fields, build_model, n_params

V = "e33t_catalogue_n4"
# trainer flag (scripts/run_e33n.py) → PerceiverARConfig field, where the names differ
_RENAME = {"par_global_nope": "global_nope"}


def _spec(**kw):
    return ArchSpec(name="shared", hidden=64, head_dim=16, n_kv_heads=1, pre_layers=1, global_layers=1,
                    stack_layers=2, local_window=16, token_embedding_dim=32, ngram_orders=(),
                    lm_latent_dim=64, lm_writer_dim=32, zero_init_residuals=False, message_raw_window=256, **kw)


def _exam(recipe="recall_single", scale="bridge_1k", rows=2, **over):
    from data.bapo_ladder import config_for, resolve_recipe
    from verification.bapo_capability_probe import _boundary_token_id, make_batch

    rc = resolve_recipe(recipe)
    c = config_for(scale, rc.task, **{**rc.overrides, **over})
    ids, labels = make_batch(c, np.random.default_rng(0), rows, "cpu")
    return c, ids, labels, _boundary_token_id(c)


def _build(arch, c, q_id, **spec_kw):
    return build_model(arch, vocab_size=c.vocab.vocab_size, seq_len=c.seq_len, answer_start=c.answer_start,
                       pad_id=c.vocab.control("eos"), bos_id=c.vocab.control("bos"), eos_id=c.vocab.control("eos"),
                       spec=_spec(message_boundary_token_id=q_id, **spec_kw), seed=0)


def test_variant_carries_the_text_arm_settings():
    """Every catalogue / writer / loop setting of the text stage-B winner reaches the model config unchanged; only
    platform knobs (width, depth, raw window, embeddings) come from the capability platform."""
    from scripts.run_e33n import ARMS

    c, _, _, q_id = _exam()
    cfg = _build(V, c, q_id).config
    arm = ARMS["catalogue_reuse_n4"]
    platform = {"par_mode", "message_write", "message_boundary_token_id", "message_raw_window", "lm_context",
                "lm_latent_dim", "lm_heads", "lm_writer_dim", "lm_enc_layers", "lm_rounds"}
    checked = 0
    for k, v in arm.items():
        if k in platform:
            continue
        assert getattr(cfg, _RENAME.get(k, k)) == v, k
        checked += 1
    assert checked >= 18
    assert cfg.global_nope is True  # the trainer sets par_global_nope for every E33n/E33t arm
    # the writer's own sizes are the E31 platform's, which equal the text arm's
    for k in ("lm_latent_dim", "lm_heads", "lm_writer_dim", "lm_enc_layers"):
        assert getattr(ArchSpec(name="x"), k) == arm[k], k


def test_variant_reads_across_query_like_text():
    """The opened tokens are raw book tokens; with a QUERY boundary the answer side could never read them, so the
    variant drops the boundary (closed read, message machinery on) while every other arch keeps it."""
    c, _, _, q_id = _exam()
    assert q_id >= 0
    cat = _build(V, c, q_id)
    assert cat.config.message_boundary_token_id == -1 and cat.config.message_enabled
    assert cat.config.lm_read == "closed" and cat.config.message_raw_window == 256
    for arch in ("e31_li_m1", "e31c_loop", "e33a_loop"):
        assert _build(arch, c, q_id).config.message_boundary_token_id == q_id, arch


@pytest.mark.parametrize("arch", ["e31_li_m1", "e31c_m1", "e31c_loop", "e33a_loop", "e31_page", "dense"])
def test_older_arches_pass_no_new_config(arch):
    from evaluation.bapo_models import resolve_arch

    base, spec = resolve_arch(arch, _spec(message_boundary_token_id=10))
    assert _note_open_fields(spec, base) == {}
    c, _, _, q_id = _exam()
    cfg = _build(arch, c, q_id).config
    assert cfg.lm_open == "none" and cfg.lm_note_read == "gist" and not cfg.global_nope


def test_platform_size_fits_the_capability_cap():
    """At the 30M capability platform (width 960, 1 + 1 + 2 layers, 128-d token embedding) the variant stays
    under the 40M cap the core package runs with, about the champion's size."""
    c, _, _, q_id = _exam()
    spec = ArchSpec(name="shared", hidden=960, head_dim=64, n_kv_heads=1, pre_layers=1, global_layers=1,
                    stack_layers=2, token_embedding_dim=128, ngram_orders=(), message_boundary_token_id=q_id,
                    zero_init_residuals=False, message_raw_window=256)
    kw = dict(vocab_size=c.vocab.vocab_size, seq_len=c.seq_len, answer_start=c.answer_start,
              pad_id=c.vocab.control("eos"), bos_id=c.vocab.control("bos"), eos_id=c.vocab.control("eos"), seed=0)
    cat, champ = build_model(V, spec=spec, **kw), build_model("e31_li_m1", spec=spec, **kw)
    n, nc = n_params(cat), n_params(champ)
    assert n < 40_000_000 and abs(n - nc) / nc < 0.05, (n, nc)


def test_trains_on_exam_rows_with_index_loss_and_reads_opened_tokens():
    """One training step on 1k lookup rows: finite loss including the index loss, gradients into the writer (through
    the note keys) and the empty slots; the notebook switch (override none) changes the answers' logits, so the
    answers do read what the catalogue opens."""
    c, ids, labels, q_id = _exam(rows=2)
    m = _build(V, c, q_id).train()
    out = m(ids, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    att = m.layers[m.config.global_layer_index].attn
    assert float(att.sink_k.grad.abs().sum()) > 0
    assert sum(float(p.grad.abs().sum()) for p in m.memory_writer.parameters() if p.grad is not None) > 0
    m.eval()
    with torch.no_grad():
        on = m(ids).logits
        m._message_override = "none"
        off = m(ids).logits
        m._message_override = "real"
    ans = labels != -100
    assert not torch.allclose(on[ans], off[ans], atol=1e-6)


@pytest.mark.parametrize("recipe,scale,over", [("far_copy", "tiny", {}), ("recall_single", "tiny_wide", {}),
                                               ("recall_single", "bridge", {"evidence_align": "right"}),
                                               ("chain_parallel", "bridge_1k", {"hops": 2, "chain_overhang": 1,
                                                                               "key_len": 8})])
def test_runs_at_every_core_length(recipe, scale, over):
    """Short books (128 / 256 tokens) close no page before the window: the read is window + empty slots and the
    index loss has nothing to score, which must not fail."""
    c, ids, labels, q_id = _exam(recipe, scale, **over)
    m = _build(V, c, q_id).train()
    out = m(ids, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_spec_round_trips_through_a_saved_checkpoint():
    """length_ladder.py rebuilds the model from the saved spec dict (ArchSpec(**spec))."""
    c, ids, _, q_id = _exam()
    a = _build(V, c, q_id).eval()
    spec = ArchSpec(**asdict(replace(_spec(message_boundary_token_id=q_id), name="shared")))
    b = build_model(V, vocab_size=c.vocab.vocab_size, seq_len=c.seq_len, answer_start=c.answer_start,
                    pad_id=c.vocab.control("eos"), bos_id=c.vocab.control("bos"), eos_id=c.vocab.control("eos"),
                    spec=spec, seed=0).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)
    assert ARCH_VARIANTS[V][0] == "e31_page"

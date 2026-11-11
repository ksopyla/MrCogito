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


# ------------------------------------------------------------------ the core package as a study phase
def _core():
    from scripts.study_plans.e30_vs_e31 import jobs

    return jobs(f"core_{V}")


def test_core_phase_covers_every_active_core_task_on_seeds_0_to_2():
    """Every active C0–C5 task, seeds 0–2, each named so the board maps it onto its task and this variant (C5: the
    final curriculum stage; C1.lookup-16k: the 16k stage of the len_lookup chain)."""
    from analysis.capability_board import evidence
    from evaluation.capability_tasks import PACKAGES, TASKS

    js = _core()
    led = {"kind": "study", "name": "t", "host": "h", "collected": "2026-11-11",
           "jobs": [{"name": j["name"], "status": "done", "results": {V: {"p0": 0.9, "cand": 0.9}}, "ladders": {}}
                    for j in js]}
    ev, _ = evidence([led])
    got = {(e["task"], e["seed"]) for e in ev if e["model"] == V and e["label"] == "same"}
    want = {(t.id, s) for t in TASKS if t.level in PACKAGES["core"]["levels"] and t.status == "active"
            for s in (0, 1, 2)}
    assert want <= got, sorted(want - got)
    assert all("--stop_on" in j["args"] for j in js)  # the v4 stop rule everywhere
    names = {j["name"] for j in js}
    assert all(j["init"] in names for j in js if j.get("init"))  # every curriculum parent is in the phase


def test_core_phase_keeps_the_written_recipes():
    from evaluation.capability_tasks import TRAIN_1K_X4, TRAIN_1K_X16
    from scripts.study_plans.e30_vs_e31 import _last

    by = {j["name"]: j for j in _core()}
    lk2 = by[f"v4_C1_lookup-2k_{V}_s0"]["args"]  # the suite v3 budget of L3.lookup-2k at 30m, written out
    assert (_last(lk2, "--lr", 0.0), _last(lk2, "--steps", 0), _last(lk2, "--k1_mult", 0)) == (1e-4, 1200, 6)
    assert _last(lk2, "--grad_accum", 0) == 4 and _last(lk2, "--batch", 0) == 32  # micro 8: memory only
    h1, h2 = by[f"v4_C5_pchain2-1k_h1_{V}_s1"], by[f"v4_C5_pchain2-1k_h2_{V}_s1"]
    assert h2["init"] == h1["name"] and h1["ladder"] is None and h2["ladder"][-1] == 131072
    assert _last(h1["args"], "--steps", 0) == int(TRAIN_1K_X4[TRAIN_1K_X4.index("--steps") + 1])
    assert _last(h2["args"], "--steps", 0) == int(TRAIN_1K_X16[TRAIN_1K_X16.index("--steps") + 1])
    assert _last(h2["args"], "--replay_hops", 0) == 1 and _last(h2["args"], "--hops", 0) == 2
    b16 = by[f"len_lookup_{V}_s2_b16k"]["args"]
    assert "--attn_backend" not in b16 and _last(b16, "--grad_accum", 0) == 8  # sdpa (the champion's), micro 1


def test_heat_schedule_rests_every_process_in_the_same_window():
    from verification.heat_pause import rest_left, schedule

    s = (1000.0, 600.0, 120.0)  # work 10 min, rest 2 min
    assert rest_left(999, s) == 0 and rest_left(1000 + 300, s) == 0
    assert rest_left(1000 + 600, s) == 120 and rest_left(1000 + 660, s) == 60
    assert rest_left(1000 + 720, s) == 0 and rest_left(1000 + 720 + 610, s) == 110
    assert schedule({}) is None and schedule({"MC_HEAT_EPOCH": "1", "MC_HEAT_BURST_MIN": "0",
                                              "MC_HEAT_COOL_MIN": "5"}) is None


def test_queue_scripts_gate_on_busy_cards_and_heat(tmp_path, monkeypatch):
    import sys

    from scripts import run_study_queue

    monkeypatch.setattr(sys, "argv", ["x", "--plan", "e30_vs_e31", "--phase", f"core_{V}", "--out", str(tmp_path),
                                      "--host", "polonez", "--gpus", "1", "2", "--only", "C0_copy-128",
                                      "--mode", "scripts", "--heat", "330,20", "--max_busy", "3"])
    assert run_study_queue.main() == 0
    s = (tmp_path / "launch" / f"polonez_core_{V}_gpu2.sh").read_text()
    assert "MC_HEAT_BURST_MIN=330 MC_HEAT_COOL_MIN=20" in s and "polonez_core_" in s
    assert "busy_others" in s and "-ge 3" in s and "sleep 90" in s
    first_job = s.index("JOB v4_C0_copy-128")
    assert s.rindex("heat_pause.py", 0, first_job) > s.rindex("busy_others) -ge 3", 0, first_job) > 0


def test_dense_open_in_query_chunks_equals_one_shot(monkeypatch):
    """dense_open builds the sdpa path's opened-token mask in query chunks (the one-shot gather index needed 16 GB at
    16k): the mask is identical whatever the chunk size."""
    import nn.exact_read as er

    torch.manual_seed(0)
    B, S, G, cap = 2, 300, 40, 16
    T = torch.randint(0, cap + 3, (B, S, G), dtype=torch.int16)
    codes = torch.randint(-1, G * er.ORD_MOD, (2, B, S))
    doc = torch.zeros(B, S, dtype=torch.long)
    doc[:, 200:] = 1
    q_idx = torch.tensor([0, 5, 150, 299])
    one = [er.dense_open(T, codes, cap=cap, raw_window=7), er.dense_open(T, codes, cap=cap, raw_window=7,
                                                                          q_idx=q_idx, doc=doc)]
    monkeypatch.setattr(er, "_DENSE_OPEN_ELEMS", B * S * 7)  # 7 query rows per chunk
    many = [er.dense_open(T, codes, cap=cap, raw_window=7), er.dense_open(T, codes, cap=cap, raw_window=7,
                                                                           q_idx=q_idx, doc=doc)]
    assert all(torch.equal(a, b) for a, b in zip(one, many)) and bool(one[0].any())

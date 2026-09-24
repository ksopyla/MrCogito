"""Capability suite: frozen cells, size presets, runner plan, scorecard rules."""
import json

import pytest

from data.bapo_ladder import SCALES, config_for, resolve_recipe, rung_card
from evaluation.capability_suite import (
    CELLS,
    CELL_BY_ID,
    LEVELS,
    REFERENCES,
    SIZES,
    TIERS,
    budget_for,
    cells_for,
    lr_for,
)


def _card(cell):
    rec = resolve_recipe(cell.recipe)
    over = dict(rec.overrides)
    over.update(dict(cell.overrides))
    return rung_card(SCALES[cell.scale], rec.task, **over)


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.id)
def test_every_cell_builds_with_its_recorded_prize(cell):
    card = _card(cell)
    assert card["prize_bits"] == pytest.approx(cell.prize_bits)
    assert cell.level in LEVELS
    assert cell.seq_len == int(dict(cell.overrides).get("seq_len", SCALES[cell.scale].seq_len))


def test_cell_ids_are_unique_and_every_level_has_a_gating_cell_up_to_l5():
    assert len({c.id for c in CELLS}) == len(CELLS)
    for lv in range(0, 6):
        assert any(c.level == lv and not c.stretch for c in CELLS), lv


def test_policy_covers_every_size_and_cell():
    for s in SIZES:
        for c in CELLS:
            lr, warm, _ = lr_for(s, c)
            assert 0 < lr <= 1e-3
            b = budget_for(c)
            assert b.steps * b.k1_mult >= 3200 and b.batch >= 1


def test_references_point_at_real_cells_and_sizes():
    for r in REFERENCES:
        assert r.cell in CELL_BY_ID and r.size in SIZES
        assert 0 <= r.bits <= CELL_BY_ID[r.cell].prize_bits + 1e-6


def test_tiers_grow_in_difficulty():
    screen, standard, full = (set(c.id for c in cells_for(t)) for t in ("screen", "standard", "full"))
    assert screen < standard < full
    assert all(not CELL_BY_ID[c].stretch for c in standard)
    assert TIERS["full"].seeds >= TIERS["standard"].seeds >= TIERS["screen"].seeds


def test_size_presets_match_the_ledger_param_counts():
    """5.11 / 8.97 M dense on the probe factory (30M / 50M checked the same way, slower)."""
    from evaluation.bapo_models import ArchSpec, build_model, n_params

    cfg = config_for(SCALES["bridge_1k"], "recall")
    v = cfg.vocab
    for name in ("5m", "10m"):
        s = SIZES[name]
        spec = ArchSpec(name="t", hidden=s.hidden, head_dim=s.head_dim, n_kv_heads=s.kv_heads,
                        pre_layers=s.pre_layers, global_layers=s.global_layers,
                        stack_layers=s.stack_layers, local_window=16)
        m = build_model("dense", vocab_size=v.vocab_size, seq_len=1024, answer_start=cfg.answer_start,
                        pad_id=v.control("eos"), bos_id=v.control("bos"), eos_id=v.control("eos"), spec=spec)
        assert n_params(m) / 1e6 == pytest.approx(s.dense_params_m, abs=0.02)
        assert n_params(m) < s.max_params


# --- runner -------------------------------------------------------------------------------

def test_plan_builds_probe_commands(tmp_path):
    from scripts.run_capability_suite import assign_gpus, plan

    jobs = plan(["e30"], ["5m", "30m"], "screen", out=str(tmp_path))
    n_cells = len(cells_for("screen"))
    assert len(jobs) == 2 * n_cells * TIERS["screen"].seeds
    assert len({j.out_dir for j in jobs}) == len(jobs)
    j = next(j for j in jobs if j.cell == "L1.copy-512" and j.size == "30m")
    cmd = " ".join(j.cmd)
    assert "--scale bridge --recipe far_copy --evidence_align right" in cmd
    assert "--arch dense e30" in cmd and "--hidden 960" in cmd and "--warm_residuals" in cmd
    assert "--swp_n_heads 8" in cmd and "--no-skip_uncalibrated" in cmd
    by = assign_gpus(jobs, ["0", "1", "2"])
    assert sum(len(v) for v in by.values()) == len(jobs)


def test_plan_rejects_unregistered_arch(tmp_path):
    from scripts.run_capability_suite import plan

    with pytest.raises(SystemExit):
        plan(["not_an_arch"], ["5m"], "screen", out=str(tmp_path))


def test_lr_pair_and_chain_overrides(tmp_path):
    from scripts.run_capability_suite import plan

    jobs = plan(["e30"], ["50m"], "standard", out=str(tmp_path), cell_ids=("L4.chain-2k",), seeds=1, lr_pair=True)
    assert len(jobs) == 2 and {j.lr for j in jobs} == {5e-5, 2.5e-5}
    assert "--hops 4" in " ".join(jobs[0].cmd) and "--seq_len 2048" in " ".join(jobs[0].cmd)


# --- scorecard ------------------------------------------------------------------------------

def _write(tmp, size, cell, seed, arch_results, lr=1e-4):
    d = tmp / size / cell / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"size": size, "cell": cell, "level": CELL_BY_ID[cell].level,
                                            "seed": seed, "lr": lr}))
    prize = CELL_BY_ID[cell].prize_bits
    res = {}
    for arch, (acc, bits, tps) in arch_results.items():
        res[arch] = {"info": {"recovered_bits": bits, "prize_bits": prize, "information_flow": bits / prize},
                     "final": {"acc": acc, "acc_se": 0.01, "step": 100},
                     "examples_to_criterion": {"examples": 800 if acc >= 0.75 else None,
                                               "step": 100 if acc >= 0.75 else None,
                                               "confirmed": acc >= 0.75},
                     "throughput": {"tokens_per_sec": tps}, "params": 1}
    (d / f"{cell}.json").write_text(json.dumps({"results": res}))


def test_scorecard_levels_frontier_and_verdict(tmp_path):
    from analysis.capability_scorecard import aggregate, load_runs, scale_verdict

    strong = {"dense": (0.99, 60, 100.0), "cand": (0.9, 55, 80.0)}
    for size in ("10m", "30m"):
        for c in cells_for("screen"):
            _write(tmp_path, size, c.id, 0, strong)
        # long reach: dense 0 (trainability), candidate passes
        _write(tmp_path, size, "L3.lookup-1k", 0, {"dense": (0.25, 0, 100.0), "cand": (0.8, 50 if size == "30m" else 45, 80.0)})
    best = aggregate(load_runs(tmp_path))
    v = scale_verdict(best, "cand", ["10m", "30m"])
    assert v["frontier_by_size"]["30m"] == 3
    assert v["verdict"] == "scale up", v["reasons"]
    assert v["trend"]["regresses"] == 0


def test_scorecard_uncalibrated_and_regression(tmp_path):
    from analysis.capability_scorecard import aggregate, level_status, load_runs, scale_verdict

    # dense also fails → level uncalibrated, not a fail of the candidate
    _write(tmp_path, "30m", "L0.copy-128", 0, {"dense": (0.3, 0, 100.0), "cand": (0.3, 0, 100.0)})
    _write(tmp_path, "30m", "L0.lookup-128", 0, {"dense": (0.99, 31, 100.0), "cand": (0.9, 30, 100.0)})
    # regression with size on a hard cell
    _write(tmp_path, "10m", "L4.chain-1k", 0, {"dense": (0.99, 63, 100.0), "cand": (0.6, 40, 100.0)})
    _write(tmp_path, "30m", "L4.chain-1k", 0, {"dense": (0.99, 63, 100.0), "cand": (0.4, 20, 100.0)})
    best = aggregate(load_runs(tmp_path))
    assert level_status(best, "cand", "30m")[0] == "uncalibrated"
    v = scale_verdict(best, "cand", ["10m", "30m"])
    assert v["verdict"] != "scale up"
    assert v["trend"]["regresses"] == 1


def test_scorecard_replays_the_e30_ledger_as_not_ready(tmp_path):
    """The recorded 30M E30 results must not earn 'scale up' (lookalike and lookup @1k miss)."""
    from analysis.capability_scorecard import aggregate, load_runs, scale_verdict

    acc = lambda bits, prize: 0.25 + 0.75 * bits / prize  # rough bits → accuracy for the replay
    for c in cells_for("standard"):
        refs = {r.arch: r.bits for r in REFERENCES if r.cell == c.id and r.size == "30m"}
        if "e30" not in refs:
            continue
        p = c.prize_bits
        _write(tmp_path, "30m", c.id, 0, {"dense": (acc(refs.get("dense", 0), p), refs.get("dense", 0), 100.0),
                                          "e30x": (acc(refs["e30"], p), refs["e30"], 90.0)})
    best = aggregate(load_runs(tmp_path))
    v = scale_verdict(best, "e30x", ["30m"])
    assert v["verdict"] != "scale up"
    assert v["frontier_by_size"]["30m"] < 2

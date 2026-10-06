"""Text capability checks — definitions, runner and scorecard (evaluation/text_checks.py,
scripts/run_text_checks.py, analysis/text_checks_scorecard.py)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.text_checks_scorecard import analyse, passes, shortcut  # noqa: E402
from evaluation.text_checks import ARCHES, ENV_OF, TIERS, fit_params, grad_accum, in_band, load_recipe  # noqa: E402


@pytest.mark.parametrize("tier", ["screen", "main"])
def test_every_model_lands_in_its_parameter_band(tier):
    for arch in ARCHES:
        _, n = fit_params(arch, tier)
        assert in_band(n, tier), (arch, tier, n)


@pytest.mark.parametrize("tier", ["screen", "main"])
@pytest.mark.parametrize("gpus", [1, 3, 4])
def test_global_batch_is_the_same_on_polonez_and_odra(tier, gpus):
    a = grad_accum(tier, gpus)
    assert a * gpus * TIERS[tier].per_device_batch == TIERS[tier].global_rows


def test_every_setting_reaches_the_launcher():
    launcher = (ROOT / "scripts" / "train_concept_pretraining_multigpu.sh").read_text()
    for arg, env in ENV_OF.items():
        assert re.search(rf'^{env}="\$\{{{env}:-', launcher, re.M), (arg, env)
    for env in ("GPU_IDS", "TRAIN_OUTPUT_DIR"):
        assert env in launcher + (ROOT / "scripts" / "remote_paths.sh").read_text()


def test_recipe_cards_exist_and_have_every_tier():
    for arch in ARCHES:
        card = load_recipe(arch)
        assert set(card["lr"]) >= {"screen", "main", "smoke"}
        assert card["length_method"] == "none"


def _fake_data(tmp: Path) -> Path:
    d = tmp / "data"
    d.mkdir()
    (d / "text_checks_meta.json").write_text(json.dumps({"version": "test", "mix": {"mean_row_tokens": 2048.0}}))
    return d


@pytest.mark.parametrize("phase", ["tune", "train"])
def test_plan_writes_valid_scripts(tmp_path, phase):
    data = _fake_data(tmp_path)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "scripts/run_text_checks.py", "plan", "--phase", phase, "--tier", "screen",
                        "--arches", "dense", "e31c", "--host", "odra", "--gpus", "0", "1", "2", "--data", str(data),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    scripts = list(out.rglob("*.sh"))
    assert scripts
    for s in scripts:
        assert subprocess.run(["bash", "-n", str(s)]).returncode == 0, s
    card = json.loads(next(out.glob("jobs/t*_dense*/job.json")).read_text())
    assert card["params_in_band"] and card["n_gpus"] in (1, 3)
    if phase == "train":
        assert card["steps"] == pytest.approx(TIERS["screen"].tokens / (96 * 2048), rel=1e-3)
        sh = (out / "jobs" / "train_dense" / "job.sh").read_text()
        assert "export GRADIENT_ACCUMULATION_STEPS=4" in sh and "export GPU_IDS=0,1,2" in sh


def _cell(task, L, exact, removed=0.0, floor=0.1, split="id"):
    return {"split": split, "task": task, "level": "", "length": L, "n": 400, "exact": exact,
            "exact_se": (exact * (1 - exact) / 400) ** 0.5, "pick": None, "removed_exact": removed, "floor": floor}


def _model(arch, cells, story=2.0):
    return {"arch": arch, "tier": "screen", "seq_len": 4096, "steps": 100, "tokens_per_step": 1e6,
            "cells": {(c["split"], c["task"], c["length"]): c for c in cells}, "off": {}, "story_loss": story,
            "curve": {50: {"lookup": _cell("lookup", 4096, 0.8)}, 100: {}}}


def test_scorecard_rules():
    assert passes(_cell("lookup", 4096, 0.8)) and not passes(_cell("lookup", 4096, 0.7))
    assert shortcut(_cell("keyed", 4096, 0.9, removed=0.5, floor=0.1))
    dense = _model("dense", [_cell(t, 4096, 0.9) for t in ("quote", "lookup", "keyed")] + [_cell("latest", 4096, 0.3)])
    cand = _model("e31c", [_cell("quote", 4096, 0.9), _cell("lookup", 4096, 0.9), _cell("lookup", 65536, 0.8),
                           _cell("keyed", 4096, 0.5), _cell("latest", 4096, 0.9)], story=2.06)
    res = analyse({"dense": dense, "e31c": cand})["screen"]
    assert res["calibrated"]["latest"] is False            # dense misses: uncalibrated, gates nothing
    m = res["models"]["e31c"]
    assert m["frontier"] == "T2"                            # keyed fails, T3 blocks
    assert m["reach"]["lookup"] == 65536
    assert m["tokens_to_pass"]["lookup"] == 50 * 1e6
    assert m["language_ok"] is False and m["language_gap"] == pytest.approx(0.03)

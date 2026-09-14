"""Live-log parser for the exclusive-slot scaling campaign."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_plot_mod():
    spec = importlib.util.spec_from_file_location(
        "plot_scale_hard",
        REPO / "verification" / "plot_scale_hard.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step_re_captures_lr_and_wall():
    mod = _load_plot_mod()
    line = (
        "  [D] step  3000  examples   96000  train 0.6772  "
        "eval CE 0.6983  acc 0.605  lr 3.00e-04  (1.94 s/step)"
    )
    m = mod.STEP_RE.search(line)
    assert m is not None
    assert m.group(1) == "D"
    assert int(m.group(2)) == 3000
    assert int(m.group(3)) == 96000
    assert float(m.group(6)) == pytest.approx(0.605)
    assert float(m.group(7)) == pytest.approx(3e-4)
    assert float(m.group(8)) == pytest.approx(1.94)


def test_parse_log_bundle_d9_params_and_lr(tmp_path: Path):
    mod = _load_plot_mod()
    log = tmp_path / "cell_seq512_r8_D9.log"
    log.write_text(
        "device=cpu\n"
        "task=far_copy seq=512 alphabet=4 min_gap=32 dec_segment=32 slots=64\n"
        "--- arm D ---\n"
        "  [D] step  3000  examples   96000  train 0.6772  "
        "eval CE 0.6983  acc 0.605  lr 3.00e-04  (1.94 s/step)\n"
    )
    bundle = mod.parse_log_bundle(log)
    assert bundle is not None
    assert bundle["results"]["D"]["params"] == 4_974_227
    assert bundle["summary"]["D"]["lr"] == pytest.approx(3e-4)
    assert bundle["summary"]["D"]["acc"] == pytest.approx(0.605)
    assert bundle["in_progress"] is True
    wall = bundle["results"]["D"]["trace"][0]["wall_s"]
    assert wall == pytest.approx(3000 * 1.94)


def test_difficulty_key_tags_true_reach_min_gap():
    mod = _load_plot_mod()
    bundle = {
        "run_name": "reach_seq512_A",
        "config": {"task": "far_copy", "seq_len": 512, "ratio": 8, "min_gap": 128},
    }
    assert mod.difficulty_key(bundle) == "seq512 r=8 gap128"
    padded = {
        "run_name": "cell_seq512_r8_A",
        "config": {"task": "far_copy", "seq_len": 512, "ratio": 8, "min_gap": 32},
    }
    assert mod.difficulty_key(padded) == "seq512 r=8"


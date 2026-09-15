"""Durable JSON writes and exclusive-slot law inventory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_write_result_json_falls_back_when_primary_is_dead(tmp_path: Path, monkeypatch):
    probe = _load("symbolic_channel_probe", "verification/symbolic_channel_probe.py")
    durable = tmp_path / "cache"
    monkeypatch.setattr(probe, "DURABLE_RESULT_DIRS", (durable,))
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("blocked")
    primary = blocker / "reach_seq1024_A.json"

    written = probe.write_result_json(primary, {"acc": 0.959, "arm": "A"})
    assert written == [str(durable / "reach_seq1024_A.json")]
    payload = json.loads((durable / "reach_seq1024_A.json").read_text())
    assert payload["acc"] == pytest.approx(0.959)


def test_write_result_json_raises_when_every_dest_fails(tmp_path: Path, monkeypatch):
    probe = _load("symbolic_channel_probe", "verification/symbolic_channel_probe.py")
    blocker = tmp_path / "nope"
    blocker.write_text("blocked")
    monkeypatch.setattr(probe, "DURABLE_RESULT_DIRS", (blocker / "also_dead",))
    with pytest.raises(OSError, match="failed to write result JSON"):
        probe.write_result_json(blocker / "out.json", {"x": 1})


def test_law_inventory_has_closed_cells_and_seq1024_a():
    inv = json.loads((REPO / "docs/4_Research_Notes/exclusive_slot_law_inventory.json").read_text())
    cells = {(c["run"], c["arm"]): c for c in inv["cells"]}
    a = cells[("reach_seq1024_A", "A")]
    assert a["hit_95"] is True
    assert a["acc"] == pytest.approx(0.959)
    assert a["examples"] == 96000
    assert a["min_gap"] == 256
    assert a["params_m"] == pytest.approx(5.11)
    assert cells[("cell_seq256_r32_A", "A")]["hit_95"] is False
    assert cells[("cell_chain_h2_D9", "D")]["hit_95"] is False
    assert cells[("reach_seq512_C", "C")]["acc"] == pytest.approx(0.2537, abs=1e-3)
    d9 = cells[("reach_seq1024_D9", "D")]
    assert d9["hit_95"] is True
    assert d9["acc"] == pytest.approx(0.9707, abs=1e-3)
    assert d9["examples"] == 136000
    assert d9["min_gap"] == 256
    assert d9["params_m"] == pytest.approx(4.97)
    c1024 = cells[("reach_seq1024_C", "C")]
    assert c1024["hit_95"] is False
    assert c1024["acc"] == pytest.approx(0.2605, abs=1e-3)
    assert c1024["examples"] == 38400
    assert not inv.get("in_flight")


def test_plot_exclusive_slot_law_writes_pngs(tmp_path: Path, monkeypatch):
    plot = _load("plot_exclusive_slot_law", "verification/plot_exclusive_slot_law.py")
    monkeypatch.setattr(plot, "OUTS", [tmp_path])
    assert plot.main() == 0
    assert (tmp_path / "exclusive_slot_law_steps_sizes_acc.png").is_file()
    assert (tmp_path / "exclusive_slot_working_law.png").is_file()

"""Capability checks v4 task definitions stay consistent with the engine, the suite and the rules."""
from data.bapo_ladder import resolve_recipe
from evaluation.capability_suite import CELL_BY_ID, CELLS
from evaluation.capability_tasks import (BATTERY_V1, FLAWED, LEVEL_BY_ID, PACKAGES, SUITE_V3, TASK_BY_ID, TASKS,
                                         legacy_battery, legacy_suite)


def test_every_task_is_buildable_and_well_formed():
    assert len(TASK_BY_ID) == len(TASKS)
    for t in TASKS:
        resolve_recipe(t.recipe)  # raises on an unknown recipe
        assert t.status in ("active", "calibrating", "flawed")
        if t.status == "flawed":
            assert t.level is None and t.flaw, t.id
        else:
            assert t.level in LEVEL_BY_ID and t.id.startswith(t.level + "."), t.id
            assert not t.flaw
        assert t.measures and t.name


def test_curricula_are_written_out_and_start_from_random_init():
    for t in TASKS:
        if t.curriculum:
            assert "random init" in t.curriculum, t.id


def test_every_suite_v3_cell_has_a_v4_home_with_the_same_exam():
    assert set(SUITE_V3) == {c.id for c in CELLS}
    for cell_id, task_id in SUITE_V3.items():
        cell, task = CELL_BY_ID[cell_id], TASK_BY_ID[task_id]
        assert cell.seq_len == task.train_len, cell_id
        assert cell.recipe == task.recipe, cell_id
    assert legacy_suite("L6.shuffled-1k") == ("X.shuffled2-1k", "flawed")
    assert legacy_suite("L3.lookup-2k") == ("C1.lookup-2k", "same")


def test_legacy_battery_labels_are_honest():
    assert set(BATTERY_V1.values()) <= set(TASK_BY_ID)
    assert legacy_battery("e31_li_m1", "lookup_16k") == ("C1.lookup-16k", "same")
    assert legacy_battery("e31_li_m1", "lookup_2k") == ("C1.lookup-2k", "settings-differ")
    assert legacy_battery("e31_li_m1", "chain_2k") == ("C4.chain4-2k", "curriculum-differs")
    assert legacy_battery("e33a_loopft", "pchain3_1k") == ("X.pchain3-1k-v1", "flawed")
    assert legacy_battery("e31_li", "match3_1k") == ("X.match3-1k", "flawed")
    assert legacy_battery("e31_li_m1", "lookup_8k") == (None, "unmapped")  # a curriculum stage, not a task


def test_flawed_tasks_never_sit_in_a_package_level():
    levels = {lv for p in PACKAGES.values() for lv in p["levels"]}
    assert all(TASK_BY_ID[t].level not in levels for t in FLAWED)


def test_multi_candidate_exams_score_the_picked_candidate_and_state_their_floor():
    for t in TASKS:
        assert t.score in ("first", "candidate"), t.id
        if t.floor is not None:  # a first-letter floor sits above chance; a picked-candidate floor is 1 / #candidates
            assert (t.chance <= t.floor if t.score == "first" else 0 < t.floor) and t.floor < 1, t.id
    for h in (2, 3, 4):
        t = TASK_BY_ID[f"C5.pchain{h}-1k"]
        assert t.score == "candidate" and t.floor is not None and t.floor < 0.15
        assert "--chain_overhang" in t.args and "random init" in t.curriculum
    assert TASK_BY_ID["C1.keyed4-1k"].score == "candidate"

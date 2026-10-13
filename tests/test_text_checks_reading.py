"""Reading rules of the text capability checks (spec §10.5, 2026-10-13): far / near facts, effective floors,
"learned", and the scorer's far/near fields. Pure functions: no model, no data."""
import pytest

from evaluation.text_checks import depth_counts, effective_floor, far_near, headline, learned


def test_depth_counts_follow_the_builder():
    # the builder assigns bin i % 3 (early, middle, late)
    for n in (1, 2, 3, 200, 400, 401):
        bins = [("early", "middle", "late")[i % 3] for i in range(n)]
        assert depth_counts(n) == {d: bins.count(d) for d in ("early", "middle", "late")}


def test_far_near_from_old_ledger_cell():
    # dense plain, lookup at 4k (2026-10-10): 42/134 early, 58/133 middle, 57/133 late
    cell = {"task": "lookup", "n": 400, "exact": 157 / 400,
            "by_depth": {"early": 42 / 134, "middle": 58 / 133, "late": 57 / 133}}
    fn = far_near(cell)
    assert fn["far_n"] == 267 and fn["near_n"] == 133
    assert fn["far"] == pytest.approx(100 / 267)
    assert fn["near"] == pytest.approx(57 / 133)


def test_far_near_refuses_inconsistent_bins():
    cell = {"task": "lookup", "n": 400, "exact": 0.4, "by_depth": {"early": 0.3, "middle": 0.5, "late": 0.4}}
    assert far_near(cell) is None  # 0.3 × 134 is not a whole number of items


def test_far_near_absent_for_spread_tasks():
    assert far_near({"task": "latest", "n": 400, "exact": 0.3, "by_depth": {"spread": 0.3}}) is None
    h = headline("latest", {"task": "latest", "n": 400, "exact": 0.3, "by_depth": {"spread": 0.3}})
    assert h["kind"] == "all" and h["score"] == 0.3


def test_scorer_fields_win_over_derivation():
    cell = {"task": "quote", "n": 9, "exact": 0.5, "far": 0.25, "far_n": 6, "near": 1.0, "near_n": 3,
            "by_depth": {"early": 0.0, "middle": 0.0, "late": 0.0}}
    assert far_near(cell)["far"] == 0.25


def test_effective_floor_and_learned():
    assert effective_floor("latest", 1 / 11) == 0.20
    assert effective_floor("lookup", 1 / 12) == pytest.approx(1 / 12)
    # latest at 1k from runs that learned nothing (19-26 %) is not "learned" against the 1-in-5 reader
    assert not learned(0.26, 400, "latest", 1 / 11)
    assert learned(0.32, 400, "latest", 1 / 11)
    # lookup: 10 points above guessing is the bar at 400 items
    assert not learned(0.17, 400, "lookup", 1 / 12)
    assert learned(0.20, 400, "lookup", 1 / 12)
    # few items: three standard errors dominate
    assert not learned(0.30, 20, "lookup", 1 / 12)
    assert not learned(None, 400, "lookup", 1 / 12)


def test_summarize_reports_far_and_near():
    from evaluation.text_checks_eval import summarize

    rows = []
    for i in range(30):
        d = ("early", "middle", "late")[i % 3]
        rows.append({"split": "id", "task": "lookup", "level": "T2", "length": 4096, "depth": d, "floor": 1 / 12,
                     "exact": d == "late", "pick": None, "pick_ambiguous": False, "pick_prob": None, "answer_nll": 1.0})
    c = summarize(rows)[0]
    assert c["far"] == 0.0 and c["far_n"] == 20 and c["near"] == 1.0 and c["near_n"] == 10
    assert c["by_depth_n"] == {"early": 10, "late": 10, "middle": 10}
    assert far_near(c) == {"far": 0.0, "far_n": 20, "near": 1.0, "near_n": 10}

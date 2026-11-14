"""evaluation/tiny_ml_board.py reproduces the Tiny-ML Leaderboard's score (values read from the live page's own
getScore / getEfficiencyScore on 2026-11-14), and the runner's leaderboard tier asks for the board's metrics."""
from __future__ import annotations

import json
import math

from evaluation.run_lm_eval_suite import LEADERBOARD_METRIC, _tasks_for, summarize
from evaluation.tiny_ml_board import from_lm_eval, load_board, parse_params, score


def test_reproduces_the_board():
    rows = {r["name"]: r for r in score(load_board())}
    live = {"Slayer139 1.02": (81.70, 83.36), "Boris-2": (80.19, 81.71), "JugnuLM-110M-R2+": (79.17, 81.53),
            "Tiny Chick 10M V3": (77.76, 87.74), "Glint-2": (67.15, 81.73)}
    for name, (overall, eff) in live.items():
        assert round(rows[name]["overall"], 2) == overall and round(rows[name]["efficiency"], 2) == eff, name


def test_params_parse_like_javascript():
    assert parse_params("143.72M") == 143.72e6 and parse_params("1.06M") == 1.06e6 and parse_params("9.5K") == 9500
    assert math.isnan(parse_params("~100M")) and parse_params(141_510_000) == 141.51e6


def test_candidate_and_lm_eval_json(tmp_path):
    res = {"results": {"blimp": {"acc,none": 0.80}, "arc_easy": {"acc,none": 0.55, "acc_norm,none": 0.5},
                       "wikitext": {"byte_perplexity,none": 1.9, "word_perplexity,none": 30.0}}}
    (tmp_path / "r.json").write_text(json.dumps(res))
    c = from_lm_eval(str(tmp_path / "r.json"))
    assert abs(c["blimp"] - 80) < 1e-9 and abs(c["arc"] - 55) < 1e-9 and c["wiki"] == 1.9
    me = next(r for r in score(load_board(), {**c, "name": "x", "params": 141.5e6}) if r.get("candidate"))
    assert 1 <= me["rank"] <= 101 and me["efficiency"] > me["overall"] > 0
    assert _tasks_for("leaderboard", None) == ["blimp", "arc_easy", "wikitext"]
    row = summarize(res["results"], _tasks_for("leaderboard", None), LEADERBOARD_METRIC)
    assert row["arc_easy/acc"] == 0.55 and row["wikitext/byte_perplexity"] == 1.9
    assert "wikitext" not in [k.split("/")[0] for k, v in row.items() if k == "avg_acc"]  # no perplexity in the mean

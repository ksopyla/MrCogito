#!/usr/bin/env python
"""Text capability board: the one dashboard of the text capability checks.

Reads the committed ledgers (`docs/2_Experiments_Registry/results/capability/text/*.json`, written by
`scripts/pull_text_checks_results.sh`) and the hand-written notes (`analysis/text_board_notes.py`: status,
round verdicts, plain model names), and writes one self-contained page:
  * status brief: where we are and what is next;
  * results, one block per round: the shared budget (parameters, data, epochs, compute), each model's cost and
    language score, the task table against the guessing rate, and the round's observations;
  * length charts and learning curves for a chosen round;
  * calibration and pipeline checks (kept apart from results): tiny pipeline runs, step-size tuning, data
    builds, superseded rounds, the original round plan;
  * the task guide and the sources.
Process: docs/engineering_specs/text_capability_checks.md (skill `text-checks`).

    uv run python analysis/text_board.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.text_board_notes import LADDER, NAMES, ROUNDS, SCOREBOARD, STATUS, SUPERSEDED  # noqa: E402
from analysis.text_checks_scorecard import LEVEL, analyse  # noqa: E402
from evaluation.text_checks import FRONTIER_TASKS, GATING_TASKS, PASS, TEXT_CHECKS_VERSION, TIERS  # noqa: E402

LEDGER_DIR = ROOT / "docs" / "2_Experiments_Registry" / "results" / "capability" / "text"
OUT = ROOT / "docs" / "3_Evaluations_and_Baselines" / "text_capability_board.html"
TASKS = list(GATING_TASKS) + list(FRONTIER_TASKS)
ORDER = ["dense", "local", "e31c", "e31c_loop", "dense_plain", "local_plain", "e31c_plain", "e31c_loop_plain"]
# Task guide, as of data v2 (2026-10-10: every guessing rate below 10 %; the v1 values, used by the 30M round of
# 9-10 Oct, are in brackets; the v0 differences are in the spec)
GUIDE = {
    "language": ("T0 Language", "Can it write the language?", "Next-token loss on held-out stories it never saw "
                 "(nats per token, lower is better). Not pass/fail: a memory model must not buy recall with worse "
                 "language (within 2 % of dense)."),
    "quote": ("T1 Quote", "Can it repeat a sentence it read earlier?",
              "Twelve people [v1: four] each have a sign with 4-6 common words; the question asks for one "
              "person's sign, word for word. Guessing one of the signs: 8 % [25 %]."),
    "lookup": ("T2 Lookup", "Can it find one person's fact?",
               "Twelve people's homes [four] (invented place names) are stated; the question asks one. "
               "Guessing: 8 % [25 %]."),
    "keyed": ("T3 Keyed", "Can it pick the right fact among many similar ones?",
              "Sixteen people's homes in the same sentence shapes, a quarter of the names one syllable apart. "
              "Guessing: 6 % (unchanged)."),
    "latest": ("T4 Latest", "Can it track a changing fact?",
               "Five people each move ten times [four], interleaved, and someone else moves last: only the asked "
               "person's own last move is right. Guessing among their eleven places: 9 % [20 %]."),
    "compose": ("T5 Compose", "Can it combine two facts?",
                "Twelve people [eight] each have a teacher and a home; the question asks where the teacher of the "
                "teacher lives. Every chain has the same shape. Guessing: 8 % [12 %]."),
    "count": ("T6 Count", "Can it count events?", "How many times did a person visit a place, when everyone, "
              "look-alike names included, visits it too. Answers zero to ten [six]: guessing 9 % [14 %]. "
              "Frontier: reported, not gating."),
    "deduce": ("T7 Deduce", "Can it chain rules?", "Made-up category rules, one chain per person (Pim is a wump. "
               "Every wump is a tove. Every tove is shiny.), twelve chains ending in twelve properties; the "
               "question asks what Pim is like. Guessing: 8 % [v1 asked yes/no: 50 %]. Frontier: reported, "
               "not gating."),
}


def load_ledgers(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            led = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if led.get("kind") == "text_checks":
            led["_file"] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            led["_id"] = f"{led['run']}.{led['host']}"
            out.append(led)
    return out


def _mix_tokens(data: dict | None) -> float | None:
    mix = (data or {}).get("mix") or {}
    try:
        return mix["story_rows"] * mix["mean_story_row_tokens"] + mix["world_rows"] * mix["mean_world_row_tokens"]
    except (KeyError, TypeError):
        return None


def _round(led: dict) -> dict:
    """One ledger → one round: budget, per-model cost and scores (scorecard rules applied within the round)."""
    ms = {}
    for key, m in led.get("models", {}).items():
        m = dict(m)
        for k in ("cells", "off"):
            m[k] = {tuple(int(x) if i == 2 else x for i, x in enumerate(s.split("|"))): c for s, c in m[k].items()}
        m["curve"] = {int(k): v for k, v in m["curve"].items()}
        ms[key] = m
    res = analyse(ms) if ms else {}
    trains = {j["arch"]: j for j in led.get("jobs", []) if j.get("job", "").startswith("train_")}
    evals = {j["arch"]: j for j in led.get("jobs", []) if j.get("kind") == "eval"}
    mix = _mix_tokens(led.get("data"))
    models, calibrated, tier = [], {}, None
    for tier, block in res.items():
        calibrated = block["calibrated"]
        for key, m in block["models"].items():
            j = trains.get(m["arch"], {})
            name, what = NAMES.get(m["arch"], (m["arch"], ""))
            wall_h = (j.get("active_seconds") or 0) / 3600
            models.append({
                "key": f"{m['arch']}@{led['_id']}", "arch": m["arch"], "name": name, "what": what, "tier": tier,
                "params": m.get("params") or j.get("params"), "lr": j.get("lr"), "status": m.get("status"),
                "wall_h": wall_h or None, "gpu_h": (wall_h * (j.get("n_gpus") or 1)) or None, "n_gpus": j.get("n_gpus"),
                "tokens_per_second": j.get("tokens_per_second"), "trained": j.get("finished"),
                "examined": evals.get(m["arch"], {}).get("finished"),
                "story_loss": m.get("story_loss"), "language_gap": m.get("language_gap"), "frontier": m.get("frontier"),
                "seq_len": m["seq_len"], "tokens_per_step": m["tokens_per_step"],
                "cells": {f"{s}|{t}|{L}": {k: c.get(k) for k in ("exact", "exact_se", "pick", "pick_prob", "removed_exact", "floor", "n",
                                                                  "answer_nll", "by_depth")}
                          for (s, t, L), c in m["cells"].items()},
                "off": {f"{s}|{t}|{L}": c.get("exact") for (s, t, L), c in m["off"].items()},
                "curve": {str(step): {t: c.get("exact") for t, c in cells.items()} for step, cells in m["curve"].items()},
                "eval_loss": j.get("eval_loss") or [],
            })
    models.sort(key=lambda m: ORDER.index(m["arch"]) if m["arch"] in ORDER else 9)
    any_train = next(iter(trains.values()), {})
    tokens = any_train.get("tokens")
    t = TIERS.get(tier or any_train.get("tier") or "", None)
    budget = {
        "tier": tier or any_train.get("tier"), "target_params": t.target_params if t else None,
        "data_version": (led.get("data") or {}).get("version"), "tokens": tokens, "mix_tokens": mix,
        "epochs": (tokens / mix) if tokens and mix else None, "seq_len": t.seq_len if t else None,
        "rows_per_step": any_train.get("global_rows"), "tokens_per_step":
            (any_train.get("global_rows") or 0) * (any_train.get("mean_row_tokens") or 0) or None,
        "steps": any_train.get("steps"), "cap_gpu_h": any_train.get("cap_gpu_hours"), "n_gpus": any_train.get("n_gpus"),
        "lengths": (next(iter(evals.values()), {}).get("lengths") or (list(t.eval_lengths) if t else [])),
    }
    return {"id": led["_id"], "notes": ROUNDS.get(led["_id"], {}), "budget": budget, "models": models,
            "calibrated": calibrated, "collected": led.get("collected")}


FAMILIES = {"dense": ("dense_plain",), "e31c": ("e31c_plain",), "e33": ("e31c_loop_plain",)}


def scoreboard(ledgers: list[dict]) -> dict:
    """The main table: per setup (mixed-task / single-task training, plain input, 30M, 1.5B tokens) × task × model
    family, every seed's exact score at 4k and at the longest exam length, its state, and the leader per task."""
    fam_of = {a: f for f, archs in FAMILIES.items() for a in archs}
    out = {s["key"]: {} for s in SCOREBOARD["setups"]}
    floors = {}
    for led in ledgers:
        rid = led["_id"]
        setup = next((s["key"] for s in SCOREBOARD["setups"]
                      if rid in s.get("runs", ()) or (s.get("prefix") and rid.startswith(s["prefix"]))), None)
        if not setup:
            continue
        only = rid.split("_")[1] if setup == "single" else None  # pt_<task>_30m_<host>
        states = {j["arch"]: j["state"] for j in led.get("jobs", []) if j.get("kind") == "train"}
        evals = {j["arch"]: j["state"] for j in led.get("jobs", []) if j.get("kind") == "eval"}
        for arch, st in states.items():
            fam = fam_of.get(arch)
            if not fam:
                continue
            ms = [m for m in led.get("models", {}).values() if m["arch"] == arch]
            cells = ms[0]["cells"] if ms else {}
            seed = next((j.get("seed") for j in led["jobs"] if j.get("arch") == arch and j.get("kind") == "train"), 0)
            for task in TASKS:
                if only and task != only:
                    continue
                by_len = {int(k.split("|")[2]): c for k, c in cells.items() if k.startswith(f"id|{task}|")}
                for c in by_len.values():
                    floors[task] = c.get("floor")
                done = st in ("done", "over_budget") and evals.get(arch) == "done" and by_len
                entry = {"run": rid, "seed": seed or 0, "state": "done" if done else ("running" if st in ("running", "done") else "queued")}
                if done:
                    lmax = max(by_len)
                    entry.update(at4k=(by_len.get(4096) or {}).get("exact"), atmax=by_len[lmax].get("exact"), lmax=lmax,
                                 at1k=(by_len.get(1024) or {}).get("exact"))
                out[setup].setdefault(task, {}).setdefault(fam, []).append(entry)
    # queued single-task work that has no run folder yet
    for task in SCOREBOARD.get("single_queue", []):
        for fam in FAMILIES:
            out["single"].setdefault(task, {}).setdefault(fam, []) or out["single"][task][fam].append({"state": "queued"})
    # leader per (setup, task): best seed's best length, if clearly above guessing and ahead of the others
    leaders = {}
    for setup, tasks in out.items():
        for task, fams in tasks.items():
            best = {f: max((max(e.get("at4k") or 0, e.get("atmax") or 0) for e in es if e.get("state") == "done"), default=None)
                    for f, es in fams.items()}
            scored = sorted(((v, f) for f, v in best.items() if v is not None), reverse=True)
            fl = floors.get(task) or 0
            unfinished = any(all(e.get("state") != "done" for e in es) for es in fams.values())
            if unfinished:
                lead = "pending"
            elif not scored or scored[0][0] < fl + 0.10:
                lead = None
            elif len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
                lead = "tie"
            else:
                lead = scored[0][1]
            leaders[f"{setup}|{task}"] = lead
    return {"setups": SCOREBOARD["setups"], "cells": out, "floors": floors, "leaders": leaders,
            "families": {"dense": "Dense", "e31c": "E31c notebook", "e33": "E33 loop"}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger_dir", default=str(LEDGER_DIR))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    ledgers = load_ledgers(Path(a.ledger_dir))
    rounds = [_round(led) for led in ledgers if led["_id"] in ROUNDS]
    rounds.sort(key=lambda r: r.get("collected") or "", reverse=True)
    # charts: first a combined view of the main comparison (mixed training, plain input), then every run
    mixed_ids = SCOREBOARD["setups"][0]["runs"]
    mixed = [r for i in mixed_ids for r in rounds if r["id"] == i]
    chart_rounds = list(rounds)
    if mixed:
        models = []
        for r in mixed:
            for m in r["models"]:
                m = dict(m)
                if "_s1_" in r["id"]:
                    m["name"], m["color"] = m["name"] + " · seed 1", "--s2"
                models.append(m)
        combo = {**mixed[-1], "id": "combined_mixed_plain", "models": models,
                 "notes": {"title": "Main comparison: mixed training, plain input — dense (2 seeds), E31c, E33 loop"}}
        chart_rounds = [combo] + chart_rounds
    jobs = [{**j, "run": led["_id"]} for led in ledgers for j in led.get("jobs", [])]
    # calibration side: tiny pipeline runs, step-size tuning, data builds, superseded rounds
    pipeline = [{"run": led["_id"], "arch": j["arch"], "name": NAMES.get(j["arch"], (j["arch"],))[0], "params": j.get("params"),
                 "tokens": j.get("tokens"), "seconds": j.get("active_seconds"), "state": j["state"], "finished": j.get("finished"),
                 "story_loss": (led.get("models", {}).get(f"{j['arch']}@{j.get('tier')}") or {}).get("story_loss")}
                for led in ledgers for j in led.get("jobs", []) if j.get("tier") == "smoke" and j.get("kind") == "train"]
    tuning = [{"arch": j["arch"], "name": NAMES.get(j["arch"], (j["arch"],))[0], "tier": j["tier"], "lr": j["lr"],
               "loss": (j.get("eval_loss") or [[None, None]])[-1][1], "tokens": j.get("tokens"), "run": j["run"],
               "data": next((led.get("data") or {}).get("version") for led in ledgers if led["_id"] == j["run"])}
              for j in jobs if j.get("job", "").startswith("tune_")]
    data_builds = [{"run": led["_id"], "version": (led.get("data") or {}).get("version"), "mix_tokens": _mix_tokens(led.get("data")),
                    "eval_items": (led.get("data") or {}).get("eval_items"), "collected": led.get("collected")}
                   for led in ledgers if led.get("data")]
    superseded = [{"run": k, "note": v} for k, v in SUPERSEDED.items()]
    data = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": TEXT_CHECKS_VERSION, "pass": PASS, "status": STATUS, "ladder": LADDER, "scoreboard": scoreboard(ledgers),
        "tasks": TASKS, "gating": list(GATING_TASKS), "levels": LEVEL, "guide": GUIDE,
        "rounds": rounds, "chart_rounds": chart_rounds, "pipeline": pipeline, "tuning": tuning, "data_builds": data_builds, "superseded": superseded,
        "sources": [{"file": led["_file"], "run": led["_id"], "collected": led["collected"],
                     "data": (led.get("data") or {}).get("version"), "jobs": len(led.get("jobs", [])),
                     "done": sum(j["state"] in ("done", "over_budget") for j in led.get("jobs", []))} for led in ledgers],
    }
    template = (Path(__file__).parent / "text_board_template.html").read_text()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=str)))
    print(f"wrote {a.out}: {len(ledgers)} ledgers, {len(rounds)} result rounds "
          f"({sum(len(r['models']) for r in rounds)} models), {len(pipeline)} pipeline runs, {len(tuning)} tuning runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

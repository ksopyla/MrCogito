#!/usr/bin/env python
"""Text capability board: the one dashboard of the text capability checks.

Reads the committed ledgers (`docs/2_Experiments_Registry/results/capability/text/*.json`, written by
`scripts/pull_text_checks_results.sh`) and the hand-written notes (`analysis/text_board_notes.py`: status, claims,
protocol log, run standings, round verdicts, plain model names), and writes one self-contained page — the mission
control of the text checks:
  * status brief: what runs, what is next, what blocks;
  * what we know: every claim with its state (holds / single run / open / retracted) and what would settle it;
  * the scoreboard of the runs in use, read with the current rules (spec §10.5);
  * all runs with their standing (in use / calibration / history …) and why, filterable by standing, exams and date;
  * the protocol log: every change and what it did to earlier results;
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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.text_board_notes import (ARCHS, CHANGES, CLAIMS, DECISIONS, GOAL, LADDER, RECIPES, SIGNALS, NAMES, PROTOCOL_LOG,  # noqa: E402
                                       ROUNDS, SCOREBOARD, STATUS, SUPERSEDED)
from analysis.text_checks_scorecard import LEVEL, analyse, learned_cell  # noqa: E402
from evaluation.text_checks import (FRONTIER_TASKS, GATING_TASKS, PASS, TEXT_CHECKS_VERSION, TIERS,  # noqa: E402
                                    effective_floor, far_near, headline)

LEDGER_DIR = ROOT / "docs" / "2_Experiments_Registry" / "results" / "capability" / "text"
OUT = ROOT / "docs" / "3_Evaluations_and_Baselines" / "text_capability_board.html"
TASKS = list(GATING_TASKS) + list(FRONTIER_TASKS)
ORDER = ["dense", "local", "e31c", "e31c_loop", "e33d", "dense_plain", "local_plain", "e31c_plain", "e31c_loop_plain",
         "e33d_plain"]
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

# One short illustrative example per task (the shape of an exam item, not its exact template).
EXAMPLES = {
    "quote": ("… Ana's sign said: the old red door is open. …", "What did Ana's sign say?", "the old red door is open."),
    "lookup": ("… Ana lives in Brindle. … Bo lives in Corvo. …", "Where does Ana live?", "Brindle."),
    "keyed": ("… Ana lives in Brindle. … Ane lives in Corvo. …", "Where does Ane live?", "Corvo."),
    "latest": ("… Ana moved to Brindle. … Ana moved to Corvo. … Bo moved to Dell.", "Where is Ana now?", "Corvo."),
    "compose": ("… Ana's teacher is Bo. … Bo's teacher is Cid. … Cid lives in Dell. …",
                "Where does the teacher of Ana's teacher live?", "Dell."),
    "count": ("… Ana visited the mill. … Ane visited the mill. … Ana visited the mill. …",
              "How many times did Ana visit the mill?", "two."),
    "deduce": ("… Pim is a wump. … Every wump is a tove. … Every tove is shiny. …", "What is Pim like?", "shiny."),
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
                "cells": {f"{s}|{t}|{L}": {**{k: c.get(k) for k in ("exact", "exact_se", "pick", "pick_prob", "removed_exact", "floor",
                                                                     "n", "answer_nll", "by_depth")},
                                           **(far_near(c) or {}), "eff_floor": effective_floor(t, c.get("floor")),
                                           "learned": learned_cell(c)}
                          for (s, t, L), c in m["cells"].items()},
                "off": {f"{s}|{t}|{L}": c.get("exact") for (s, t, L), c in m["off"].items()},
                "off_far": {f"{s}|{t}|{L}": (far_near(c) or {}).get("far") for (s, t, L), c in m["off"].items()},
                "curve": {str(step): {t: c.get("exact") for t, c in cells.items()} for step, cells in m["curve"].items()},
                "curve_far": {str(step): {t: (headline(t, c) or {}).get("score") for t, c in cells.items()}
                              for step, cells in m["curve"].items()},
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
    lad = next((r for r in LADDER if r.get("id") == led["_id"]), {})
    notes = ROUNDS.get(led["_id"]) or {"title": lad.get("run", led["_id"]), "dates": lad.get("when", ""),
                                       "verdict": lad.get("result", "")}
    return {"id": led["_id"], "notes": notes, "budget": budget, "models": models,
            "calibrated": calibrated, "collected": led.get("collected")}


FAMILIES = {"dense": ("dense_plain",), "e31c": ("e31c_plain",), "e33": ("e31c_loop_plain",), "e33d": ("e33d_plain",)}


def _h(cells: dict, split: str, task: str, L: int) -> dict | None:
    c = cells.get(f"{split}|{task}|{L}")
    h = headline(task, c)
    if h:
        h["learned"] = learned_cell(c)
        h["se"] = (h["score"] * (1 - h["score"]) / h["n"]) ** 0.5 if h.get("n") else 0.0
    return h


def scoreboard(ledgers: list[dict]) -> dict:
    """The main table: per setup (mixed-task / single-task training, plain input, 30M, 1.5B tokens) × task × model
    family, every seed's headline score (far facts for quote / lookup / keyed, spec §10.5) at 4k and at the longest
    exam length, the rephrased-fact score at 4k, its state, and the leader per task at 4k."""
    fam_of = {a: f for f, archs in FAMILIES.items() for a in archs}
    out = {s["key"]: {} for s in SCOREBOARD["setups"]}
    floors = {}
    blocked = {r["id"] for r in LADDER if r.get("standing") == "blocked" and r.get("id")}  # e.g. a server is off
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
                job = next((j for j in led["jobs"] if j.get("arch") == arch and j.get("kind") == "train"), {})
                wall = (job.get("active_seconds") or 0) / 3600
                entry = {"run": rid, "seed": seed or 0, "story": ms[0].get("story_loss") if ms else None,
                         "wall_h": wall or None, "gpu_h": (wall * (job.get("n_gpus") or 1)) or None,
                         "state": "done" if done else ("blocked" if rid in blocked else
                                                       "running" if st in ("running", "done") else "queued")}
                if done:
                    lmax = max(by_len)
                    h4, hmax, h1 = _h(cells, "id", task, 4096), _h(cells, "id", task, lmax), _h(cells, "id", task, 1024)
                    para = _h(cells, "paraphrase", task, 4096)
                    crowd = _h(cells, "crowded", task, 4096)
                    entry.update(at4k=(h4 or {}).get("score"), se4k=(h4 or {}).get("se"), learned4k=(h4 or {}).get("learned"),
                                 near4k=(h4 or {}).get("near"), all4k=(h4 or {}).get("all"), kind=(h4 or {}).get("kind"),
                                 atmax=(hmax or {}).get("score"), learnedmax=(hmax or {}).get("learned"), lmax=lmax,
                                 at1k=(h1 or {}).get("score"), para4k=(para or {}).get("score"),
                                 learned_para=(para or {}).get("learned"),
                                 crowd4k=(crowd or {}).get("score"), learned_crowd=(crowd or {}).get("learned"),
                                 learned1k=(h1 or {}).get("learned"))
                out[setup].setdefault(task, {}).setdefault(fam, []).append(entry)
    # queued work that has no run folder yet: single-task runs by task; whole families of a setup (every task)
    for task in SCOREBOARD.get("single_queue", []) if "single" in out else []:
        for fam in FAMILIES:
            out["single"].setdefault(task, {}).setdefault(fam, []) or out["single"][task][fam].append({"state": "queued"})
    for key, fams in SCOREBOARD.get("queued", {}).items():
        for task in TASKS:
            for fam in fams:
                out[key].setdefault(task, {}).setdefault(fam, []) or out[key][task][fam].append({"state": "queued"})
    # leader per (setup, task) at the 4k training length, where every model is inside its training range: the best
    # seed's headline score, if learned (clearly above the effective floor) and ahead of the next family by more than
    # 5 points and 2 combined standard errors. Flags: one seed only; dense never learned the task (uncalibrated).
    # Past the training length dense has no length method, so "reads at 16k" lists the families that learned there.
    leaders, readers = {}, {}
    for setup, tasks in out.items():
        for task, fams in tasks.items():
            done = {f: [e for e in es if e.get("state") == "done"] for f, es in fams.items()}
            best = {f: max(es, key=lambda e: e.get("at4k") or 0) for f, es in done.items() if es}
            scored = sorted(((e.get("at4k") or 0, f) for f, e in best.items()), reverse=True)
            unfinished = any(not es for es in done.values())
            info = {"lead": None, "single_seed": False,
                    "uncalibrated": not any(e.get("learned4k") for e in done.get("dense", []))}
            if unfinished:
                info["lead"] = "pending"
            elif scored and best[scored[0][1]].get("learned4k"):
                top = best[scored[0][1]]
                info["lead"] = scored[0][1]
                if len(scored) > 1:
                    nxt = best[scored[1][1]]
                    gap = scored[0][0] - scored[1][0]
                    margin = max(0.05, 2 * ((top.get("se4k") or 0) ** 2 + (nxt.get("se4k") or 0) ** 2) ** 0.5)
                    if gap <= margin:
                        info["lead"] = "tie"
                if info["lead"] != "tie":
                    es = done[info["lead"]]
                    info["single_seed"] = len(es) < 2
                    info["seeds_learned"] = f"{sum(bool(e.get('learned4k')) for e in es)} of {len(es)}"
            leaders[f"{setup}|{task}"] = info
            readers[f"{setup}|{task}"] = [f for f, es in done.items()
                                          if any(e.get("learnedmax") and (e.get("lmax") or 0) > 4096 for e in es)]
    setups = [{**st, "families": st.get("families") or ["dense", "e31c", "e33"]} for st in SCOREBOARD["setups"]]
    return {"setups": setups, "cells": out, "floors": floors,
            "eff_floors": {t: effective_floor(t, f) for t, f in floors.items()}, "leaders": leaders, "readers": readers,
            "families": {"dense": "Dense", "e31c": "E31c notebook", "e33": "E33 loop", "e33d": "E33d page distance"}}


def recipe_board(rounds: list[dict]) -> list[dict]:
    """The recipe leaderboard: per recipe (RECIPES in the notes) and per measure, the runs' scores (headline: far facts
    for quote / lookup / keyed, every item otherwise) on the current exams, plus held-out shapes / extra hops, the
    notebook-off control and story loss. Runs without a ledger yet are counted as pending."""
    by_id = {r["id"]: r for r in rounds}
    cols = [(f"id|{t}|{L}", t, L) for L in (4096, 16384) for t in TASKS]
    cols += [("paraphrase|compose|4096", "compose", 4096), ("harder|compose|4096", "compose", 4096),
             ("paraphrase|deduce|4096", "deduce", 4096), ("harder|deduce|4096", "deduce", 4096)]
    out = []
    for rc in RECIPES:
        runs, pending = [], []
        for rid in rc["runs"]:
            r = by_id.get(rid)
            if not r or not r["models"]:
                pending.append(rid)
                continue
            runs.append((rid, r["models"][0]))
        cells = {}
        for key, task, L in cols:
            vals = []
            for _, m in runs:
                h = headline(task, m["cells"].get(key))
                if h:
                    vals.append(h["score"])
            if vals:
                vals.sort()
                cells[key] = {"mid": vals[len(vals) // 2], "lo": vals[0], "hi": vals[-1], "n": len(vals),
                              "floor": (runs[0][1]["cells"].get(key) or {}).get("eff_floor")}
        off = [m["off"].get("id|compose|4096") for _, m in runs if m["off"].get("id|compose|4096") is not None]
        story = [m["story_loss"] for _, m in runs if m.get("story_loss")]
        out.append({**rc, "cells": cells, "n_runs": len(runs), "pending": pending,
                    "off_compose": (sorted(off)[len(off) // 2] if off else None),
                    "story": (sorted(story)[len(story) // 2] if story else None)})
    return out


def goal_progress(rounds: list[dict]) -> dict:
    """The author's goal (GOAL in the notes): per task and length, the best notebook model and the best dense model on
    the current exams (data v2), every run that is not history. Exact answers over all items (these tasks have no far
    / near split)."""
    out = {}
    for task in GOAL["tasks"]:
        for L in GOAL["lengths"]:
            best = {"notebook": None, "dense": None}
            for r in rounds:
                if not str(r["budget"].get("data_version") or "").endswith("v2") or r.get("standing") == "history":
                    continue
                for m in r["models"]:
                    c = m["cells"].get(f"id|{task}|{L}") or {}
                    if c.get("exact") is None:
                        continue
                    side = "dense" if m["arch"].startswith("dense") else "notebook"
                    if best[side] is None or c["exact"] > best[side]["score"]:
                        best[side] = {"score": c["exact"], "name": m["name"].split(",")[0], "run": r["notes"]["title"],
                                      "floor": c.get("eff_floor") or c.get("floor")}
            out[f"{task}|{L}"] = best
    return {**GOAL, "best": out}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger_dir", default=str(LEDGER_DIR))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    ledgers = load_ledgers(Path(a.ledger_dir))
    rounds = [r for r in (_round(led) for led in ledgers if led["_id"] in ROUNDS or led.get("models")) if r["models"]]
    # each run's standing (in use / calibration / history …), why, start date and exams version, from the run list
    standing = {r["id"]: r for r in LADDER if r.get("id")}
    for r in rounds:
        st = standing.get(r["id"], {})
        r.update(standing=st.get("standing", "history"), why=st.get("why", ""), start=st.get("start", ""),
                 data=st.get("data") or (r["budget"].get("data_version") or "").replace("text-world-", ""))
    rounds.sort(key=lambda r: r.get("collected") or "", reverse=True)
    # charts: first a combined view of the main comparison (mixed training, plain input), then every run
    mixed_ids = SCOREBOARD["setups"][0]["runs"]
    mixed = [r for i in mixed_ids for r in rounds if r["id"] == i]
    chart_rounds = list(rounds)
    if mixed:
        models = []
        for r in mixed:
            seed = re.search(r"_s(\d+)_", r["id"])
            for m in r["models"]:
                m = dict(m)
                if seed:  # several runs of one model: one colour per architecture, one dash per seed
                    m["name"] = m["name"] + f" · run {int(seed[1]) + 1}"
                    m["dash"] = ("", "6 3", "2 3")[int(seed[1]) % 3]
                models.append(m)
        combo = {**mixed[-1], "id": "combined_mixed_plain", "models": models,
                 "notes": {"title": "Main comparison — " + SCOREBOARD["setups"][0]["title"]}}
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
    unnamed = sorted({m["arch"] for r in rounds for m in r["models"] if m["arch"] not in NAMES})
    if unnamed:  # the board would print raw arch ids (capability_boards.md, "Names")
        print(f"warning: no plain name in NAMES for {unnamed}", file=sys.stderr)
    states = {"holds", "single run", "open", "retracted"}  # the board's claim groups
    bad = [c["claim"][:60] for c in CLAIMS if c.get("state") not in states]
    if bad:
        raise SystemExit(f"claims with a state the board does not show ({sorted(states)}): {bad}")
    data = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": TEXT_CHECKS_VERSION, "pass": PASS, "status": STATUS, "ladder": LADDER, "scoreboard": scoreboard(ledgers),
        "claims": CLAIMS, "protocol_log": PROTOCOL_LOG, "decisions": DECISIONS, "archs": ARCHS, "changes": CHANGES,
        "goal": goal_progress(rounds), "recipes": recipe_board(rounds), "signals": SIGNALS,
        "examples": EXAMPLES,
        "exam_runs": {v: sum(1 for r in LADDER if r.get("data") == v and r.get("id")) for v in
                      sorted({r.get("data") for r in LADDER if r.get("data")})},
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

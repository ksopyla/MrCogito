#!/usr/bin/env python
"""Text capability board: the one dashboard of the text capability checks (draft `text-v0`).

Reads the committed ledgers (`docs/2_Experiments_Registry/results/capability/text/*.json`, written by
`scripts/pull_text_checks_results.sh`) and the round plan (`scripts/study_plans/text_r1.py`), and writes one
self-contained page, the text counterpart of the DNA capability board:
  * the round plan: every experiment with its state (planned / running / done / failed), gate and cost;
  * per model: frontier level, language loss vs dense, tokens to pass, reach, flags;
  * the task table (T1–T7 × models, exact answer at the training length and the longest length read);
  * length charts, learning curves (score vs training tokens), development-loss curves, tuning curves;
  * runs and cost (state, active GPU time, throughput), the task guide, and the sources.
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

from analysis.text_checks_scorecard import LEVEL, analyse  # noqa: E402
from evaluation.text_checks import FRONTIER_TASKS, GATING_TASKS, PASS, TEXT_CHECKS_VERSION, TIERS  # noqa: E402

LEDGER_DIR = ROOT / "docs" / "2_Experiments_Registry" / "results" / "capability" / "text"
OUT = ROOT / "docs" / "3_Evaluations_and_Baselines" / "text_capability_board.html"
TASKS = list(GATING_TASKS) + list(FRONTIER_TASKS)
GUIDE = {
    "language": ("T0 Language", "Can it write the language?", "Next-token loss on held-out stories it never saw. "
                 "Not pass/fail: a memory model must not buy recall with worse language (within 2 % of dense)."),
    "quote": ("T1 Quote", "Can it copy a sentence it read earlier?",
              "Four cast members have a sign with 4–6 random common words; the question asks for one sign word for word. "
              "Tests carrying an exact span across the document."),
    "lookup": ("T2 Lookup", "Can it find one fact?",
               "Only the asked person's home is stated (an invented place name); everyone else gets an unrelated job fact. "
               "The simplest long-range retrieval — findable without matching the name."),
    "keyed": ("T3 Keyed", "Can it pick the right fact among many similar ones?",
              "Sixteen people's homes are stated in the same sentence shapes, a quarter of the names are look-alikes "
              "(one syllable apart). The model must match the name in the question."),
    "latest": ("T4 Latest", "Can it track a changing fact?",
               "The asked person moves four times; other people move after the last move, so the most recently "
               "mentioned place is wrong. Needs order, not just retrieval."),
    "compose": ("T5 Compose", "Can it combine two facts?",
                "Everyone has a sister and a home; the question asks where the sister (of the sister) lives. "
                "Every chain has the same shape, so only following the links works."),
    "count": ("T6 Count", "Can it aggregate?", "How many times did a person visit a place, with look-alike names also "
              "visiting. Answers zero to six, balanced. Frontier: reported, not gating."),
    "deduce": ("T7 Deduce", "Can it chain rules?", "Made-up category rules (Every wump is a tove. Every tove is shiny.) "
               "with an opposite distractor chain; yes/no balanced. Frontier: reported, not gating."),
}


def load_ledgers(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            led = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if led.get("kind") == "text_checks":
            led["_file"] = str(p.relative_to(ROOT))
            out.append(led)
    return out


def _models(ledgers: list[dict]) -> dict:
    """Merge every ledger's models and re-run the scorecard rules on the union (seeds and runs together)."""
    ms = {}
    for led in ledgers:
        for key, m in led.get("models", {}).items():
            m = dict(m)
            for k in ("cells", "off"):
                m[k] = {tuple(int(x) if i == 2 else x for i, x in enumerate(s.split("|"))): c for s, c in m[k].items()}
            m["curve"] = {int(k): v for k, v in m["curve"].items()}
            m["run"] = f"{led['run']}.{led['host']}"
            ms[key] = m
    res = analyse(ms) if ms else {}
    out, cal = {}, {}
    for tier, block in res.items():
        cal[tier] = block["calibrated"]
        for key, m in block["models"].items():
            out[key] = {
                "key": key, "arch": m["arch"], "tier": tier, "run": m["run"], "params": m.get("params"),
                "status": m.get("status"), "story_loss": m.get("story_loss"), "language_gap": m.get("language_gap"),
                "language_ok": m.get("language_ok"), "frontier": m.get("frontier"), "reach": m.get("reach"),
                "tokens_to_pass": m.get("tokens_to_pass"), "notebook_gain": m.get("notebook_gain"),
                "shortcuts": m.get("shortcuts"), "seq_len": m["seq_len"], "tokens_per_step": m["tokens_per_step"],
                "cells": {f"{s}|{t}|{L}": {k: c.get(k) for k in ("exact", "exact_se", "pick", "removed_exact", "floor", "n",
                                                                  "by_depth")}
                          for (s, t, L), c in m["cells"].items()},
                "off": {f"{s}|{t}|{L}": c.get("exact") for (s, t, L), c in m["off"].items()},
                "curve": {str(step): {t: c.get("exact") for t, c in cells.items()} for step, cells in m["curve"].items()},
            }
    return out, cal


def _experiments(ledgers: list[dict]) -> list[dict]:
    from scripts.study_plans.text_r1 import EXPERIMENTS

    jobs = {(led["run"], j["job"]): {**j, "host": led["host"]} for led in ledgers for j in led.get("jobs", [])}
    out = []
    for x in EXPERIMENTS:
        states = []
        for run, names in x["runs"].items():
            for n in names:
                states.append(jobs.get((run, n), {}).get("state", "planned"))
        if states and all(s in ("done", "over_budget") for s in states):
            state = "done"
        elif any(s == "failed" for s in states):
            state = "failed"
        elif any(s == "running" for s in states):
            state = "running"
        elif any(s in ("done", "over_budget") for s in states):
            state = "partial"
        elif any(s == "pending" for s in states):
            state = "queued"
        else:
            state = "planned"
        out.append({**x, "state": state, "n_jobs": len(states),
                    "n_done": sum(s in ("done", "over_budget") for s in states)})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger_dir", default=str(LEDGER_DIR))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    ledgers = load_ledgers(Path(a.ledger_dir))
    models, cal = _models(ledgers)
    jobs = [{**j, "run": led["run"], "host": led["host"]} for led in ledgers for j in led.get("jobs", [])]
    tuning = [{"arch": j["arch"], "tier": j["tier"], "lr": j["lr"], "loss": (j.get("eval_loss") or [[None, None]])[-1][1],
               "state": j["state"], "run": j["run"]} for j in jobs if j.get("job", "").startswith("tune_")]
    dev_curves = [{"key": f"{j['arch']}@{j['tier']}" + (f"#{j['seed']}" if j.get("seed") else ""), "run": j["run"],
                   "tokens_per_step": j["global_rows"] * j["mean_row_tokens"], "eval_loss": j.get("eval_loss") or []}
                  for j in jobs if j.get("job", "").startswith("train_") and j.get("eval_loss")]
    order = ["dense", "local", "e31c", "e31c_loop"]
    keys = sorted(models, key=lambda k: (["smoke", "screen", "main"].index(models[k]["tier"])
                                         if models[k]["tier"] in ("smoke", "screen", "main") else 9,
                                         order.index(models[k]["arch"]) if models[k]["arch"] in order else 9, k))
    data = {
        "generated": datetime.datetime.now().isoformat(timespec="minutes"), "version": TEXT_CHECKS_VERSION, "pass": PASS,
        "tasks": TASKS, "gating": list(GATING_TASKS), "levels": LEVEL, "guide": GUIDE,
        "tiers": {k: {"params": t.target_params, "tokens": t.tokens, "seq_len": t.seq_len, "cap": t.cap_gpu_hours,
                      "lengths": list(t.eval_lengths)} for k, t in TIERS.items()},
        "experiments": _experiments(ledgers), "models": [models[k] for k in keys], "calibrated": cal,
        "tuning": tuning, "dev_curves": dev_curves,
        "jobs": [{k: j.get(k) for k in ("run", "host", "job", "kind", "arch", "tier", "state", "params", "lr", "steps",
                                         "tokens", "active_seconds", "tokens_per_second", "n_gpus", "cap_gpu_hours",
                                         "live_loss", "seed")} for j in jobs],
        "sources": [{"file": led["_file"], "run": led["run"], "host": led["host"], "collected": led["collected"],
                     "data": (led.get("data") or {}).get("version"), "jobs": len(led.get("jobs", [])),
                     "done": sum(j["state"] in ("done", "over_budget") for j in led.get("jobs", []))} for led in ledgers],
    }
    template = (Path(__file__).parent / "text_board_template.html").read_text()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=str)))
    print(f"wrote {a.out}: {len(ledgers)} ledgers, {len(models)} models, "
          f"{sum(x['state'] == 'done' for x in data['experiments'])}/{len(data['experiments'])} experiments done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

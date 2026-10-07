#!/usr/bin/env python
"""Capability board: the one dashboard of the capability checks (v4 structure).

Reads the committed ledger (`docs/2_Experiments_Registry/results/capability/`, written by
`scripts/pull_capability_results.sh`) and the task definitions (`evaluation/capability_tasks.py`), and
writes one self-contained page plus the re-run list:
  * every past result is mapped onto its v4 task with a match label (same, settings-differ,
    curriculum-differs, not-from-scratch, flawed) — nothing is re-scored or re-run here;
  * the task table: levels C0–C7, one row per task, first-letter score at the training length and at
    128k per model, with the row status (active, partial, missing, calibrating, flawed);
  * the no-harm check against the champion, on `same` evidence only;
  * the length charts (first letter vs input length), grouped by level;
  * the task guide (what every level and task measures) and the re-run list
    (`docs/3_Evaluations_and_Baselines/capability_reruns.md`).
Process: docs/engineering_specs/capability_checks.md (skill `capability-checks`).

    uv run python analysis/capability_board.py
    uv run python analysis/capability_board.py --champion e31_li_m1 --rerun_models e31_li_m1 dense e33a_loop
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.capability_ledger import LEDGER_DIR, load_ledgers, study_jobs, suite_rows  # noqa: E402
from evaluation.capability_suite import CELL_BY_ID, lr_for  # noqa: E402
from evaluation.capability_tasks import (LEVELS, SEEDS, TASK_BY_ID, TASKS, VERSION,  # noqa: E402
                                         legacy_battery, legacy_suite)

OUT = ROOT / "docs" / "3_Evaluations_and_Baselines" / "capability_board.html"
RERUNS = ROOT / "docs" / "3_Evaluations_and_Baselines" / "capability_reruns.md"
LABELS = ("same", "settings-differ", "curriculum-differs", "not-from-scratch", "flawed")  # best first
# curriculum stages of a battery exam are drawn next to the task they belong to (not tasks themselves)
STAGE_OF = {"lookup_8k": "C1.lookup-16k", "chain_8k": "C4.chain4-2k", "recall8_8k": "C3.recall8-1k",
            "recall16_8k": "C3.recall16-1k"}
PASS, HARM_POINTS = 0.75, 0.05
SPEC = ROOT / "docs" / "engineering_specs" / "capability_checks.md"
GITHUB = "https://github.com/ksopyla/MrCogito/blob/dev/"  # where the page's references point (the page is published off-repo)
MULTI_CANDIDATE = ("lookalike", "chain", "shuffled", "unique", "story")

# Standard battery slots, in reading order: (slot id, exam, training stage label)
STANDARD = [
    ("lookup_2k", "Lookup", "trained at 2k"), ("lookup_8k", "Lookup", "+ 8k stage"),
    ("lookup_16k", "Lookup", "+ 16k stage"), ("chain_2k", "In-order 4-hop chain", "trained at 2k"),
    ("chain_8k", "In-order 4-hop chain", "+ 8k stage"), ("recall8_1k", "Recall among 8", "trained at 1k"),
    ("recall8_8k", "Recall among 8", "+ 8k stage"), ("recall16_1k", "Recall among 16", "trained at 1k"),
    ("recall16_8k", "Recall among 16", "+ 8k stage"), ("decoy8_1k", "Fact among 8 decoys", "trained at 1k"),
    ("unique_1k", "Unique item", "trained at 1k"), ("match3_1k", "Triple match", "trained at 1k"),
    ("chain8_1k", "In-order 8-hop chain", "trained at 1k"),
]
EXTRA_NAMES = {"shuf2": "Shuffled 2-hop", "shuf3": "Shuffled 3-hop", "pchain2": "Parallel 2-hop chain",
               "pchain3": "Parallel 3-hop chain", "pchain4": "Parallel 4-hop chain", "count": "Count",
               "majority": "Majority"}
STAGE = {"": None, "b8k": "8k", "b16k": "16k"}

_PATTERNS = [  # (regex, how to build (variant, exam, stage))
    (re.compile(r"^len_(lookup|chain)_(.+)_s\d+(?:_(b8k|b16k))?$"), lambda m: (m[2], m[1], m[3] or "")),
    (re.compile(r"^hard_([a-z0-9]+)_(.+)_s\d+(?:_(b8k))?$"), lambda m: (m[2], m[1], m[3] or "")),
    (re.compile(r"^dense_([a-z0-9]+)_s\d+$"), lambda m: ("dense", m[1], "")),
    (re.compile(r"^([a-z0-9]+)_(lookup|chain)_([a-z0-9]+)_s\d+(?:_(b8k|b16k))?$"),
     lambda m: (f"{m[1]}_{m[3]}", m[2], m[4] or "")),
    (re.compile(r"^([a-z0-9]+)_hard_([a-z0-9]+)_([a-z0-9]+)_s\d+(?:_(b8k))?$"),
     lambda m: (f"{m[1]}_{m[3]}", m[2], m[4] or "")),
    # a variant's own reasoning exams, e.g. E33a's parallel-chain curriculum `e33a_pchain3_loop_s1`
    (re.compile(r"^([a-z0-9]+)_(pchain\d)_([a-z0-9]+)_s\d+$"), lambda m: (f"{m[1]}_{m[3]}", m[2], "")),
]


def parse_job(name: str):
    for rx, build in _PATTERNS:
        m = rx.match(name)
        if m:
            variant, exam, stage = build(m)
            base = "2k" if exam in ("lookup", "chain") else "1k"
            return variant, f"{exam}_{STAGE[stage] or base}"
    return None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def battery(ledgers: list[dict]) -> dict:
    """slot → variant → length → {first, mean, seeds, first_seeds}"""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"first": [], "mean": []})))
    for led in ledgers:
        for j in study_jobs(led):
            if j["status"] != "done":
                continue
            parsed = parse_job(j["job"])
            if not parsed:
                continue
            variant, slot = parsed
            for L, v in (j["ladders"].get("ladder") or {}).items():
                cell = raw[slot][variant][int(L)]
                cell["first"].append(v.get("first_acc"))
                cell["mean"].append(v.get("acc"))
    out = {}
    for slot, by_v in raw.items():
        out[slot] = {}
        for variant, by_len in by_v.items():
            # first letter only where every seed has it, so a line never mixes seed sets
            out[slot][variant] = {
                str(L): {"first": _med(c["first"]) if None not in c["first"] else None, "mean": _med(c["mean"]),
                         "seeds": sum(x is not None for x in c["mean"]),
                         "first_seeds": sum(x is not None for x in c["first"])}
                for L, c in sorted(by_len.items())}
    return out


def _seed(job: dict) -> int | None:
    if job.get("seed") is not None:
        return int(job["seed"])
    m = re.search(r"_s(\d+)(?:_|$)", job.get("job", ""))
    return int(m[1]) if m else None


def _score(task: str, first, cand):
    """The task's own score: first answer letter, or the greedy picked candidate (score="candidate"); a run
    without the greedy answer (before probe --answer_exact) has no score on a candidate-scored task."""
    return cand if TASK_BY_ID[task].score == "candidate" else first


def evidence(ledgers: list[dict]) -> tuple[list[dict], str | None]:
    """One record per (finished job, arch) that maps onto a v4 task, with its match label."""
    versions = sorted({led["suite_version"] for led in ledgers if led.get("suite_version")})
    current = versions[-1] if versions else None
    out = []
    for led in ledgers:
        if led["kind"] == "suite":
            if led.get("suite_version") != current:
                continue
            for r in suite_rows(led):
                task, label = legacy_suite(r["cell"])
                if task is None or r["size"] != "30m":
                    continue
                via = f"suite {r['cell']}"
                default_lr = lr_for(r["size"], CELL_BY_ID[r["cell"]])[0]
                if label == "same" and r.get("lr") is not None and abs(r["lr"] - default_lr) > 1e-12:
                    label, via = "settings-differ", f"{via} at step {r['lr']:g} (suite default {default_lr:g})"
                out.append({"task": task, "model": r["arch"], "label": label, "seed": r["seed"],
                            "score": _score(task, r.get("p0"), r.get("cand")), "mean": r["acc"],
                            "ladder": {int(L): v for L, v in (r.get("ladder") or {}).items()},
                            "source": r["source"], "collected": r.get("collected") or "", "via": via})
        else:
            for j in study_jobs(led):
                if j["status"] != "done":
                    continue
                # v4 runs built from the task definition itself (`task_job`): `{conf|v4}_{C1_edge-1k}_{arch}_s{seed}`
                # the variant comes from the name: a variant may be an arch plus flags (e33a_loop = e31_li_m1 + loop)
                m = re.match(r"^(?:confp?|v4|recA)_(C\d)_(.+?-\d+k)(?:_h(\d))?_(.+)_s(\d+)$", j["job"])
                hops = re.search(r"(?:pchain|path)(\d)-", m[2]) if m else None
                final = m and (m[3] is None or (hops and int(m[3]) == int(hops[1])))  # curriculum: last stage only
                if m and final and f"{m[1]}.{m[2]}" in TASK_BY_ID:
                    task = f"{m[1]}.{m[2]}"
                    lad = {int(L): v.get("first_acc") for L, v in (j["ladders"].get("ladder") or {}).items()}
                    out.append({"task": task, "model": m[4], "label": "same", "seed": int(m[5]),
                                "score": _score(task, j.get("p0"), j.get("cand")), "mean": j.get("acc"),
                                "ladder": lad, "source": j["source"], "collected": "", "via": "v4 task recipe"})
                    continue
                if m:  # an earlier curriculum stage of a v4 run: not a result of the task itself
                    continue
                parsed = parse_job(j["job"])
                if not parsed:
                    continue
                variant, slot = parsed
                task, label = legacy_battery(variant, slot)
                if task is None:
                    continue
                lad = {int(L): v.get("first_acc") for L, v in (j["ladders"].get("ladder") or {}).items()}
                out.append({"task": task, "model": variant, "label": label, "seed": _seed(j),
                            "score": _score(task, j.get("p0"), j.get("cand")),
                            "mean": j.get("acc"), "ladder": lad, "source": j["source"], "collected": "",
                            "via": f"battery {slot}"})
    return out, current


# calibration jobs (`cal_*` / `cal2_*`, dense from scratch): name fragment → v4 task, and what the variant changes
CAL_TASKS = (("lookup1k", "C1.lookup-1k"), ("edge", "C1.edge-1k"), ("keyed4", "C1.keyed4-1k"), ("recall8", "C3.recall8-1k"),
             ("pchain2", "C5.pchain2-1k"), ("pchain3", "C5.pchain3-1k"), ("count", "C6.count-1k"))
CAL_KNOBS = (("path", "written path answer (recipe B)"), ("dense8", "8-layer dense (learnability check)"), ("hc", "hop count in question"), ("mix", "half the rows at the previous hop count"),
             ("direct", "task recipe, no curriculum"), ("_to_", "curriculum from the previous stage"), ("noover", "no overhang (4 facts)"),
             ("k16v16", "16-letter keys and values"), ("k16", "16-letter nodes"), ("k8", "8-letter nodes"),
             ("lr3e-4", "step 3e-4"), ("x16", "16x budget"), ("x4", "4x budget"), ("k4", "4-letter nodes"),
             ("nc2", "2 chains"), ("256_to_1k", "256-token stage then 1k"), ("x1", "suite budget"), ("_512_", "512-token book"),
             ("_256_", "256-token book"))


def calibration(ledgers: list[dict]) -> list[dict]:
    """Dense-from-scratch calibration runs: one row per job, scored on the task's own score."""
    rows = []
    for led in ledgers:
        if led["kind"] != "study":
            continue
        for j in study_jobs(led):
            name = j["job"]
            if not re.match(r"^(cal\d?|confp?)_", name) or j["status"] not in ("done", "failed"):
                continue
            stem = name.split("_dense")[0]
            m = re.match(r"^confp?_(C\d)_(.+?)(?:_h(\d))?$", stem)  # confirmation runs built from the task definition
            if not m:
                m2 = re.match(r"^cal\d_(C\d)_(.+?)_h(\d)$", stem)  # calibration runs built from a task definition
                if m2:
                    m = m2
            task = (f"{m[1]}.{m[2]}" if m else
                    next((t for frag, t in CAL_TASKS if frag in stem.split("_to_")[-1]), None))
            if task is None:
                continue
            t = TASK_BY_ID[task]
            knobs = [txt for frag, txt in CAL_KNOBS if frag in name + "_"]
            if "_to_" in name and "x16" not in name or (re.match(r"^cal[23]_", name) and "lookup1k" not in name):
                knobs = [k for k in knobs if k != "4x budget"] + ["4x budget"]
            if m:
                knobs = (["written recipe (confirmation)"] if name.startswith("conf") else ["written recipe"]) \
                    + ([f"curriculum stage {m[3]} hop{'s' if int(m[3]) > 1 else ''}"] if m[3] else [])
            rows.append({"job": name, "task": task,
                         "round": ("confirm" if name.startswith("conf") else int(name[3]) if name[3].isdigit() else 1),
                         "model": j["arch"], "variant": ", ".join(dict.fromkeys(knobs)) or "task recipe",
                         "score": _score(task, j.get("p0"), j.get("cand")), "first": j.get("p0"),
                         "first_greedy": j.get("first_greedy"), "teacher": j.get("acc"),
                         "floor": t.floor, "chance": t.chance, "score_kind": t.score, "source": j["source"]})
    rows.sort(key=lambda r: (str(r["round"]), r["task"], r["job"]))
    return rows


def latest_log(n: int = 6) -> list[dict]:
    """The newest rows of the spec's calibration log (date, round, finding, decision), newest first, as plain text."""
    rows, inside = [], False
    for line in SPEC.read_text().splitlines():
        if line.startswith("### Calibration log"):
            inside = True
        elif inside and line.startswith("#"):
            break
        elif inside and line.startswith("| 20"):
            cells = [re.sub(r"\*\*|`", "", c).strip() for c in line.strip().strip("|").split(" | ")]
            if len(cells) >= 4:
                rows.append(dict(zip(("date", "round", "finding", "decision"), cells[:4])))
    return rows[::-1][:n]


def _shown_path(p: Path, ledger_dir: Path) -> str:
    """Repo-relative path; a ledger read from elsewhere (a preview with uncommitted pulls) is marked so."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return f"{LEDGER_DIR.relative_to(ROOT)}/{p.resolve().relative_to(ledger_dir.resolve())} (not committed)"


def _one_per_seed(rs: list[dict]) -> list[dict]:
    """A seed re-run (e.g. again with saved weights for the ladder) replaces the older run of that seed:
    prefer the run with a ladder, then the latest collected. Runs without a seed all count."""
    keep, by_seed = [r for r in rs if r["seed"] is None], defaultdict(list)
    for r in rs:
        if r["seed"] is not None and r["seed"] in SEEDS:  # extra diagnostic seeds never enter the medians
            by_seed[r["seed"]].append(r)
    for runs in by_seed.values():
        keep.append(max(runs, key=lambda r: (bool(r["ladder"]), r.get("collected") or "", r["source"])))
    return keep


def table(ev: list[dict]) -> dict:
    """task → model → the best-labelled evidence: medians over seeds at the training length and per length."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in ev:
        by[e["task"]][e["model"]][e["label"]].append(e)
    out = {}
    for task, models in by.items():
        out[task] = {}
        for model, labels in models.items():
            label = next(l for l in LABELS if l in labels)
            rs = _one_per_seed(labels[label])
            lens = sorted({L for r in rs for L in r["ladder"]})
            ladder = {}
            for L in lens:
                vals = [r["ladder"].get(L) for r in rs]
                if vals and None not in vals:          # first letter only when every seed has it
                    ladder[str(L)] = _med(vals)
            out[task][model] = {
                "label": label, "score": _med([r["score"] for r in rs]), "mean": _med([r["mean"] for r in rs]),
                "seeds": sorted({r["seed"] for r in rs if r["seed"] is not None}), "ladder": ladder,
                "via": sorted({r["via"] for r in rs}), "sources": sorted({r["source"] for r in rs}),
                "other": sorted(l for l in labels if l != label),
            }
    return out


def no_harm(tbl: dict, champion: str, models: list[str]) -> dict:
    """Lost = more than HARM_POINTS below the champion where it passes; `same` evidence only."""
    out = {}
    for m in models:
        if m in (champion, "dense"):
            continue
        kept, lost, missing = [], [], []
        for task, by_m in tbl.items():
            c, v = by_m.get(champion), by_m.get(m)
            if not c or c["label"] != "same":
                continue
            points = [("train", c["score"], v["score"] if v and v["label"] == "same" else None)]
            points += [(L, cv, v["ladder"].get(L) if v and v["label"] == "same" else None)
                       for L, cv in c["ladder"].items()]
            for where, cv, vv in points:
                if cv is None or cv < PASS:
                    continue
                item = {"where": f"{task}" + ("" if where == "train" else f" @ {int(where) // 1024}k"), "champion": cv}
                if vv is None:
                    missing.append(item)
                elif vv < cv - HARM_POINTS:
                    lost.append({**item, "variant": vv})
                else:
                    kept.append({**item, "variant": vv})
        out[m] = {"kept": len(kept), "lost": lost, "missing": len(missing)}
    return out


def reference(task_id: str, tbl: dict) -> str | None:
    """The task's reference (rule 5, 2026-10-07): dense when it passes on the full seed set; otherwise the best
    model that passes from random init (`same` evidence, seeds 0–2, median ≥ PASS); None if no model does."""
    ok = {m: c["score"] for m, c in tbl.get(task_id, {}).items()
          if c["label"] == "same" and c["score"] is not None and c["score"] >= PASS
          and set(SEEDS) <= set(c["seeds"])}
    if "dense" in ok:
        return "dense"
    return max(ok, key=ok.get) if ok else None


def row_status(task, tbl: dict, champion: str) -> dict:
    if task.status == "flawed":
        return {"status": "flawed", "note": task.flaw}
    if task.status == "calibrating":
        return {"status": "calibrating", "note": "recipe not fixed yet"}
    c = tbl.get(task.id, {}).get(champion)
    if not c or c["label"] != "same":
        return {"status": "missing", "note": "champion not run on the written recipe"}
    notes = []
    if len(c["seeds"]) < len(SEEDS):
        notes.append(f"{len(c['seeds'])} of {len(SEEDS)} seeds")
    if task.ladder and not c["ladder"]:
        notes.append("no length ladder")
    return {"status": "partial" if notes else "active", "note": "; ".join(notes)}


def reruns(tbl: dict, models: list[str]) -> list[dict]:
    """What must run (from scratch, seeds 0–2) before the v4 table is complete. Flawed tasks: nothing."""
    items = []
    for t in TASKS:
        if t.status == "calibrating":
            have = sorted({f"{m} ({tbl[t.id][m]['label']})" for m in tbl.get(t.id, {}) if m in models})
            items.append({"task": t.id, "tasks": [t.id], "kind": "calibrate", "models": ["dense", models[0]],
                          "what": "find the from-scratch recipe: step-size pair, v3 budget vs 2×"
                                  + (", the written curriculum vs none" if t.curriculum else "")
                                  + "; then run every model on seeds 0, 1, 2" + (" and ladder to 128k" if t.ladder else ""),
                          "legacy": ", ".join(have) or "none"})
    # active tasks: group into concrete jobs — one row per (model, missing seeds), listing the tasks
    train, ladder = defaultdict(list), defaultdict(list)
    for t in TASKS:
        if t.status != "active":
            continue
        for m in models:
            e = tbl.get(t.id, {}).get(m)
            have = set(e["seeds"]) if e and e["label"] == "same" else set()
            need = tuple(s for s in SEEDS if s not in have)
            if need:
                train[(m, need)].append(t.id)
            if t.ladder and m != "dense" and not (e and e["label"] == "same" and e["ladder"]):
                ladder[m].append(t.id)
    for (m, need), tasks in train.items():
        items.append({"kind": "train", "models": [m], "tasks": tasks, "task": tasks[0],
                      "what": f"train from scratch, seed{'s' if len(need) > 1 else ''} {', '.join(map(str, need))} "
                              f"— {len(tasks)} task{'s' if len(tasks) > 1 else ''}",
                      "legacy": "no matching result" if len(need) == len(SEEDS) else
                                f"seeds {', '.join(str(s) for s in SEEDS if s not in need)} exist"})
    for m, tasks in ladder.items():
        items.append({"kind": "ladder", "models": [m], "tasks": tasks, "task": tasks[0],
                      "what": f"save the weights of the seeds 0–2 runs and read them up to 128k — {len(tasks)} tasks "
                              "(the suite runs kept no weights, so this rides on the retrain)",
                      "legacy": "training-length score only"})
    return items


def reruns_md(items: list[dict], models: list[str]) -> str:
    kinds = {"calibrate": "Calibrate first (no frozen from-scratch recipe)",
             "train": "Train from scratch (missing seeds)", "ladder": "Length ladder missing"}
    lines = [f"# Capability checks — re-run list ({datetime.date.today().isoformat()})", "",
             f"Generated by `analysis/capability_board.py` from the committed ledger and `evaluation/capability_tasks.py` "
             f"({VERSION}). Models: {', '.join(models)}. Every run is from scratch, seeds 0–2, first-letter "
             "scoring; flawed tasks are never re-run. Process: `docs/engineering_specs/capability_checks.md`.", ""]
    for k, title in kinds.items():
        sel = [i for i in items if i["kind"] == k]
        if not sel:
            continue
        lines += [f"## {title} — {len(sel)}", "", "| task | model(s) | what to run | what exists now |", "|---|---|---|---|"]
        lines += [f"| {', '.join(f'`{x}`' for x in i.get('tasks', [i['task']]))} | {', '.join(i['models'])} | {i['what']} | {i['legacy']} |"
                  for i in sel]
        lines.append("")
    flawed = [t for t in TASKS if t.status == "flawed"]
    lines += ["## Not re-run: flawed tasks", ""] + [f"- `{t.id}` — {t.flaw}" for t in flawed] + [""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger_dir", default=str(LEDGER_DIR))
    p.add_argument("--champion", default="e31_li_m1")
    p.add_argument("--rerun_models", nargs="*", default=None,
                   help="models the re-run list covers (default: champion, dense, registered battery variants)")
    p.add_argument("--show", nargs="*", default=None, help="models switched on when the page opens")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--reruns_out", default=str(RERUNS))
    args = p.parse_args()
    ledgers = load_ledgers(Path(args.ledger_dir))
    if not ledgers:
        raise SystemExit(f"no ledger files under {args.ledger_dir}")
    ev, suite_version = evidence(ledgers)
    tbl = table(ev)
    bat = battery([l for l in ledgers if l["kind"] == "study"])
    registered = []
    try:
        from scripts.study_plans.e30_vs_e31 import BATTERY_VARIANTS
        registered = [f"{k['tag']}_{k['arm']}" for k in BATTERY_VARIANTS.values()]
    except ImportError:
        pass
    models = sorted({e["model"] for e in ev} | {v for slot in bat.values() for v in slot})
    order = [args.champion, *registered, "e31_li", "e30_li", "dense"]
    models = [m for m in order if m in models] + [m for m in models if m not in order]
    visible = args.show or [m for m in [args.champion, *registered, "dense"] if m in models]  # older builds: one click away
    rerun_models = args.rerun_models or [args.champion, "dense", *registered]
    items = reruns(tbl, rerun_models)
    Path(args.reruns_out).write_text(reruns_md(items, rerun_models))

    slots = []
    for s in sorted(bat, key=lambda x: (STANDARD.index(next(t for t in STANDARD if t[0] == x)) if x in [t[0] for t in STANDARD] else 99, x)):
        task, label = legacy_battery("_", s)
        task = task or STAGE_OF.get(s)
        if task is None:  # a study job that is no v4 task (e.g. an ad-hoc probe): not on the board
            continue
        exam, length = s.rsplit("_", 1)
        std = next((t for t in STANDARD if t[0] == s), None)
        slots.append({"id": s, "task": task, "level": TASK_BY_ID[task].level if task and TASK_BY_ID[task].level else "X",
                      "exam": std[1] if std else EXTRA_NAMES.get(exam, exam),
                      "stage": std[2] if std else f"trained at {length}",
                      "label": "curriculum stage" if s in STAGE_OF else label})
    lv_order = [lv.id for lv in LEVELS] + ["X"]
    std_ix = {t[0]: i for i, t in enumerate(STANDARD)}
    slots.sort(key=lambda sl: (lv_order.index(sl["level"]), std_ix.get(sl["id"], 99), sl["task"] or "", sl["id"]))
    tasks = [{"id": t.id, "level": t.level or "X", "name": t.name, "measures": t.measures, "recipe": t.recipe,
              "args": " ".join(t.args), "train_len": t.train_len, "prize": t.prize_bits, "chance": t.chance,
              "score_kind": t.score, "floor": t.floor, "reference": reference(t.id, tbl),
              "curriculum": t.curriculum, "ladder": t.ladder, "task_status": t.status, "flaw": t.flaw,
              **row_status(t, tbl, args.champion)} for t in TASKS]
    data = {
        "generated": datetime.date.today().isoformat(), "version": VERSION, "champion": args.champion,
        "models": models, "visible": visible, "pass": PASS, "harm_points": HARM_POINTS, "seeds": list(SEEDS),
        "levels": [{"id": lv.id, "name": lv.name, "question": lv.question, "measures": lv.measures} for lv in LEVELS],
        "tasks": tasks, "table": tbl, "no_harm": no_harm(tbl, args.champion, models),
        "slots": slots, "battery": {s["id"]: bat[s["id"]] for s in slots},
        "reruns": items, "rerun_models": rerun_models, "suite_version": suite_version,
        "calibration": calibration(ledgers), "latest": latest_log(), "github": GITHUB,
        "sources": [{"file": _shown_path(Path(l["_file"]), Path(args.ledger_dir)), "kind": l["kind"], "name": l["name"],
                     "host": l["host"], "collected": l["collected"], "archive": l["archive_path"],
                     "done": sum(j["status"] == "done" for j in l["jobs"]), "jobs": len(l["jobs"]),
                     "suite_version": l.get("suite_version")} for l in ledgers],
    }
    template = (Path(__file__).parent / "capability_board_template.html").read_text()
    Path(args.out).write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    from collections import Counter
    print(f"wrote {args.out}: {len(models)} models, {len(tasks)} tasks, {len(slots)} length charts")
    print("  rows:", dict(Counter(t["status"] for t in tasks)))
    print(f"  re-run list: {len(items)} items -> {args.reruns_out}", dict(Counter(i["kind"] for i in items)))
    for v, r in data["no_harm"].items():
        if r["kept"] or r["lost"]:
            print(f"  {v}: kept {r['kept']}, lost {len(r['lost'])}, not run {r['missing']} (vs {args.champion}, same evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

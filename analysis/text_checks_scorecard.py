#!/usr/bin/env python
"""Scorecard for the text capability checks (draft `text-v0`): one page comparing every model in
one or more run folders (scripts/run_text_checks.py output).

Spec §10–§13: pass = exact answer ≥ 75 % with the evidence-removed control at its floor; a task
gates only where the dense model passes it at the training length (else `uncalibrated`);
frontier = highest of T1–T5 passing at the training length; reach = longest passing length;
tokens to pass from the learning-curve checkpoints; language no-harm = held-out story loss within
2 % of dense; notebook contribution = score with the notebook minus without.

  uv run python analysis/text_checks_scorecard.py --in_dir Cache/text_checks/r1_screen [more dirs] \
      [--out_dir <dir>]   # writes scorecard.md + scorecard.json (default: the first in_dir)
      [--ledger docs/2_Experiments_Registry/results/capability/text/<run>.<host>.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.text_checks import FRONTIER_TASKS, GATING_TASKS, PASS, SHORTCUT_MARGIN, TEXT_CHECKS_VERSION  # noqa: E402

LEVEL = {"quote": "T1", "lookup": "T2", "keyed": "T3", "latest": "T4", "compose": "T5", "count": "T6", "deduce": "T7"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def shortcut(c: dict) -> bool:
    if c.get("removed_exact") is None or c.get("floor") is None:
        return False
    se = c.get("exact_se") or 0.0
    return c["removed_exact"] > c["floor"] + max(SHORTCUT_MARGIN, 2 * se)


def passes(c: dict | None) -> bool:
    return bool(c) and c.get("exact") is not None and c["exact"] >= PASS and not shortcut(c)


def collect(dirs: list[Path]) -> dict:
    models = {}
    for d in dirs:
        for jj in sorted((d / "jobs").glob("train_*/job.json")):
            card = json.loads(jj.read_text())
            res = _load(jj.parent / "result.json") or {}
            ev = d / "jobs" / f"eval_{card['arch']}"
            key = f"{card['arch']}@{card['tier']}" + (f"#{card['seed']}" if card.get("seed") else "")
            m = {"arch": card["arch"], "tier": card["tier"], "role": card.get("role"), "params": card["params"],
                 "params_in_band": card.get("params_in_band"), "seq_len": card["model_args"].get("max_seq_length") or 0,
                 "steps": card["steps"], "tokens_per_step": card["global_rows"] * card["mean_row_tokens"],
                 "status": res.get("status", "not run"), "active_hours": (res.get("active_seconds") or 0) / 3600,
                 "n_gpus": card["n_gpus"], "lr": card["lr"], "run": str(d),
                 "tokens_per_second": (sum(res["tokens_per_second"]) / len(res["tokens_per_second"])
                                       if res.get("tokens_per_second") else None),
                 "cells": {}, "off": {}, "curve": {}, "story_loss": None}
            from evaluation.text_checks import TIERS

            m["seq_len"] = TIERS[card["tier"]].seq_len
            for name in ("final_id.json", "final_extra.json"):
                r = _load(ev / name)
                if r:
                    m["story_loss"] = r.get("story_loss") if r.get("story_loss") is not None else m["story_loss"]
                    for c in r["cells"]:
                        m["cells"][(c["split"], c["task"], c["length"])] = c
            r = _load(ev / "final_id_notebook_off.json")
            if r:
                for c in r["cells"]:
                    m["off"][(c["split"], c["task"], c["length"])] = c
            for cp in sorted(ev.glob("curve_checkpoint-*.json")):
                step = int(cp.stem.split("-")[-1])
                r = json.loads(cp.read_text())
                m["curve"][step] = {c["task"]: c for c in r["cells"] if c["split"] == "id"}
            if m["cells"]:
                m["curve"][m["steps"]] = {t: c for (s, t, L), c in m["cells"].items() if s == "id" and L == m["seq_len"]}
            models[key] = m
    return models


def analyse(models: dict) -> dict:
    by_tier = defaultdict(dict)
    for k, m in models.items():
        by_tier[m["tier"]][k] = m
    out = {}
    for tier, ms in by_tier.items():
        dense = next((m for m in ms.values() if m["arch"] == "dense"), None)
        tasks = list(GATING_TASKS) + list(FRONTIER_TASKS)
        calibrated = {t: passes(dense["cells"].get(("id", t, dense["seq_len"]))) if dense else False for t in tasks}
        for k, m in ms.items():
            L0 = m["seq_len"]
            lengths = sorted({L for (s, t, L) in m["cells"] if s == "id"})
            frontier = None
            for t in GATING_TASKS:
                if not calibrated[t]:
                    continue  # uncalibrated levels neither block nor count
                if passes(m["cells"].get(("id", t, L0))):
                    frontier = LEVEL[t]
                else:
                    break
            reach = {t: max([L for L in lengths if passes(m["cells"].get(("id", t, L)))], default=None) for t in tasks}
            retention = {}
            for t in tasks:
                base = (m["cells"].get(("id", t, L0)) or {}).get("exact")
                if base:
                    retention[t] = {L: round((m["cells"].get(("id", t, L)) or {}).get("exact", 0) / base, 3)
                                    for L in lengths if L > L0}
            ttp = {}
            for t in tasks:
                for step in sorted(m["curve"]):
                    if passes(m["curve"][step].get(t)):
                        ttp[t] = step * m["tokens_per_step"]
                        break
            nb = {}
            for (s, t, L), c in m["off"].items():
                on = m["cells"].get((s, t, L))
                if on and on.get("exact") is not None and c.get("exact") is not None:
                    nb[f"{t}@{L}"] = round(on["exact"] - c["exact"], 3)
            lang = None
            if dense and dense.get("story_loss") and m.get("story_loss"):
                lang = (m["story_loss"] - dense["story_loss"]) / dense["story_loss"]
            m.update(frontier=frontier, reach=reach, retention=retention, tokens_to_pass=ttp, notebook_gain=nb,
                     language_gap=lang, language_ok=(lang is None or lang <= 0.02),
                     shortcuts=sorted(f"{t}@{L}" for (s, t, L), c in m["cells"].items() if s == "id" and shortcut(c)))
        out[tier] = {"calibrated": calibrated, "models": ms}
    return out


def _f(v, pct=True):
    if v is None:
        return "–"
    return f"{100 * v:.0f}" if pct else f"{v:.3g}"


def render(res: dict) -> str:
    lines = [f"# Text capability checks — scorecard ({TEXT_CHECKS_VERSION})", "",
             "Exact-answer accuracy in %, pass ≥ 75 % with the evidence-removed control at its floor. "
             "`(f)` = guessing floor. Draft protocol: budgets and tasks are still being calibrated.", ""]
    for tier, block in res.items():
        ms = block["models"]
        cal = block["calibrated"]
        lines += [f"## Tier `{tier}`", "",
                  "Calibrated (dense passes at the training length): "
                  + ", ".join(f"{LEVEL[t]} {t} {'yes' if v else 'no'}" for t, v in cal.items()), ""]
        lines += ["| model | params | status | story loss (vs dense) | frontier | tokens to pass T2 | tokens/s | active h |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, m in ms.items():
            gap = "" if m.get("language_gap") is None else f" ({100 * m['language_gap']:+.1f} %{'' if m['language_ok'] else ' ✗'})"
            sl = "–" if m.get("story_loss") is None else f"{m['story_loss']:.3f}{gap}"
            t2 = m["tokens_to_pass"].get("lookup")
            lines.append(f"| {k} | {m['params'] / 1e6:.1f}M{'' if m['params_in_band'] else ' ✗band'} | {m['status']} | {sl} | "
                         f"{m['frontier'] or '–'} | {'–' if t2 is None else f'{t2 / 1e9:.2g}B'} | "
                         f"{'–' if not m['tokens_per_second'] else f'{m[chr(116) + 'okens_per_second']:.0f}'} | {m['active_hours']:.1f} |")
        lines.append("")
        lengths = sorted({L for m in ms.values() for (s, t, L) in m["cells"] if s == "id"})
        for t in list(GATING_TASKS) + list(FRONTIER_TASKS):
            floor = next((m["cells"][("id", t, L)]["floor"] for m in ms.values() for L in lengths if ("id", t, L) in m["cells"]), None)
            lines += [f"### {LEVEL[t]} {t}  (floor {_f(floor)} %{'' if cal.get(t) else ', uncalibrated'})", "",
                      "| model | " + " | ".join(f"{L // 1024}k" if L >= 1024 else str(L) for L in lengths) + " | reach |",
                      "|---|" + "---|" * (len(lengths) + 1)]
            for k, m in ms.items():
                row = []
                for L in lengths:
                    c = m["cells"].get(("id", t, L))
                    if not c:
                        row.append("–")
                        continue
                    mark = "**" if passes(c) else ""
                    flag = " ⚠" if shortcut(c) else ""
                    row.append(f"{mark}{_f(c['exact'])}{mark}{flag}")
                r = m["reach"].get(t)
                lines.append(f"| {k} | " + " | ".join(row) + f" | {'–' if r is None else r} |")
            lines.append("")
        extra = [(k, s, t, L, c) for k, m in ms.items() for (s, t, L), c in sorted(m["cells"].items()) if s != "id"]
        if extra:
            lines += ["### Harder and paraphrase splits (reported, not gating)", "", "| model | split | task | length | exact | floor |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {k} | {s} | {t} | {L} | {_f(c['exact'])} | {_f(c['floor'])} |" for k, s, t, L, c in extra]
            lines.append("")
        nb = [(k, m["notebook_gain"]) for k, m in ms.items() if m["notebook_gain"]]
        if nb:
            lines += ["### Notebook contribution (exact with notebook − without, points)", ""]
            lines += [f"- {k}: " + ", ".join(f"{x} {100 * v:+.0f}" for x, v in g.items()) for k, g in nb]
            lines.append("")
        sc = [(k, m["shortcuts"]) for k, m in ms.items() if m["shortcuts"]]
        if sc:
            lines += ["### ⚠ Shortcut flags (answered without the evidence: not evidence)", ""]
            lines += [f"- {k}: {', '.join(v)}" for k, v in sc]
            lines.append("")
    return "\n".join(lines)


def _jsonable(res):
    def conv(m):
        m = dict(m)
        for key in ("cells", "off"):
            m[key] = {f"{s}|{t}|{L}": c for (s, t, L), c in m[key].items()}
        m["curve"] = {str(k): v for k, v in m["curve"].items()}
        return m

    return {tier: {"calibrated": b["calibrated"], "models": {k: conv(m) for k, m in b["models"].items()}}
            for tier, b in res.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", nargs="+", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ledger", default=None,
                   help="also write the compact result here, e.g. "
                        "docs/2_Experiments_Registry/results/capability/text/<run>.<host>.json (commit it)")
    a = p.parse_args()
    dirs = [Path(d) for d in a.in_dir]
    res = analyse(collect(dirs))
    out = Path(a.out_dir or dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    md = render(res)
    (out / "scorecard.md").write_text(md)
    (out / "scorecard.json").write_text(json.dumps(_jsonable(res), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else str(x)))
    if a.ledger:
        Path(a.ledger).parent.mkdir(parents=True, exist_ok=True)
        Path(a.ledger).write_text((out / "scorecard.json").read_text())
    print(md)
    print(f"\n→ {out / 'scorecard.md'}" + (f" · ledger {a.ledger}" if a.ledger else ""))


if __name__ == "__main__":
    main()

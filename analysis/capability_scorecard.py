#!/usr/bin/env python
"""Score a capability-suite run: per-cell results, level passes, comparison with past
architectures, a size-scaling trend and a scale-up verdict.

Reads every `job.json` written by `scripts/run_capability_suite.py` and the probe's rung
JSON next to it. Writes `scorecard.md`, `scorecard.html`, `scorecard.json` and
`cells.csv` into `--out_dir` (default: the input folder).

  uv run python analysis/capability_scorecard.py --in_dir Cache/capability/e30_standard

Rules (spec: docs/engineering_specs/capability_suite.md):
  * a cell **passes** when the median answer-token accuracy over seeds is ≥ 75 %;
  * a level passes when every gating (non-stretch) cell that was run passes;
  * the **frontier** is the highest level L such that levels 0..L all pass;
  * **scale up** needs, at ≥ 30M: frontier ≥ 2, at least one long-reach / reasoning cell
    (L3/L4) that passes or ties/beats the best past architecture, no regression with size,
    and training speed ≥ 0.5 × the dense model.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.capability_suite import (  # noqa: E402
    CELL_BY_ID,
    CEILING_FRACTION,
    LEVELS,
    PASS_ACC,
    SCALE_RULE,
    SIZE_ORDER,
    SUITE_VERSION,
    references_for,
)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load_runs(in_dir: Path) -> list[dict]:
    """One row per (job, arch): the final metrics of that arch in that job."""
    rows = []
    for jp in sorted(in_dir.rglob("job.json")):
        job = json.loads(jp.read_text())
        rungs = [p for p in jp.parent.glob("*.json") if p.name not in ("job.json", "summary.json")]
        if not rungs:
            continue
        bundle = json.loads(rungs[0].read_text())
        for arch, r in bundle.get("results", {}).items():
            info, final = r.get("info", {}), r.get("final", {})
            etc = r.get("examples_to_criterion") or {}
            rows.append({
                "size": job["size"], "cell": job["cell"], "level": job["level"], "seed": job["seed"],
                "lr": job["lr"], "arch": arch,
                "bits": info.get("recovered_bits"), "prize_bits": info.get("prize_bits"),
                "flow": info.get("information_flow"), "acc": final.get("acc"),
                "acc_se": final.get("acc_se"), "params": r.get("params"),
                "examples_to_75": etc.get("examples") if etc.get("confirmed", etc.get("step") is not None) else None,
                "tokens_per_sec": (r.get("throughput") or {}).get("tokens_per_sec"),
                "per_position_acc": final.get("per_position_acc"),
                "steps": final.get("step"),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    """(size, cell, arch) → medians over seeds, at the best step size when several were run."""
    by_lr: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_lr[(r["size"], r["cell"], r["arch"], r["lr"])].append(r)
    best: dict[tuple, dict] = {}
    for (size, cell, arch, lr), rs in by_lr.items():
        agg = {
            "size": size, "cell": cell, "arch": arch, "lr": lr, "seeds": len(rs),
            "level": rs[0]["level"],
            "bits": _median([r["bits"] for r in rs]), "prize_bits": rs[0]["prize_bits"],
            "acc": _median([r["acc"] for r in rs]), "acc_se": _median([r["acc_se"] for r in rs]),
            "flow": _median([r["flow"] for r in rs]), "params": rs[0]["params"],
            "examples_to_75": _median([r["examples_to_75"] for r in rs]),
            "reached_75": sum(r["examples_to_75"] is not None for r in rs),
            "tokens_per_sec": _median([r["tokens_per_sec"] for r in rs]),
        }
        key = (size, cell, arch)
        if key not in best or (agg["bits"] or 0) > (best[key]["bits"] or 0):
            best[key] = agg
    for a in best.values():  # every pass flag first: the dense comparison below reads them
        a["pass"] = a["acc"] is not None and a["acc"] >= PASS_ACC
    for (size, cell, arch), a in best.items():
        dense = best.get((size, cell, "dense"))
        a["dense_bits"] = dense["bits"] if dense else None
        a["dense_pass"] = bool(dense and dense["pass"]) if dense else None
        a["vs_dense"] = (a["bits"] / dense["bits"]) if dense and dense["bits"] else None
        a["matches_ceiling"] = bool(a["dense_pass"] and a["vs_dense"] is not None and a["vs_dense"] >= CEILING_FRACTION)
        refs = {k: v for k, v in references_for(cell, size).items() if k != arch}
        past = {k: v.bits for k, v in refs.items() if k != "dense"}
        a["past_best_arch"] = max(past, key=past.get) if past else None
        a["past_best_bits"] = past[a["past_best_arch"]] if past else None
        a["past_dense_bits"] = refs["dense"].bits if "dense" in refs else None
        a["vs_past_best"] = (a["bits"] - a["past_best_bits"]) if past and a["bits"] is not None else None
    return best


def level_status(best: dict, arch: str, size: str) -> dict:
    """Level number → pass | fail | not-run, over gating (non-stretch) cells."""
    out = {}
    for lv in LEVELS:
        gating = [c for c in CELL_BY_ID.values() if c.level == lv and not c.stretch]
        run = [best[(size, c.id, arch)] for c in gating if (size, c.id, arch) in best]
        if not run:
            stretch = [best[(size, c.id, arch)] for c in CELL_BY_ID.values()
                       if c.level == lv and c.stretch and (size, c.id, arch) in best]
            out[lv] = f"stretch {sum(r['pass'] for r in stretch)}/{len(stretch)}" if stretch else "not-run"
        elif all(r["pass"] for r in run):
            out[lv] = "pass"
        elif all(r["pass"] or r["dense_pass"] is False for r in run):
            # every miss is on a cell the dense model also missed on this run: the exam was not
            # learnable at this budget, so it says nothing against the architecture
            out[lv] = "uncalibrated"
        else:
            out[lv] = "fail"
    return out


def frontier(levels: dict) -> int:
    f = -1
    for lv in sorted(levels):
        if levels[lv] == "pass":
            f = lv
        else:
            break
    return f


def scale_verdict(best: dict, arch: str, sizes: list[str]) -> dict:
    rule = SCALE_RULE
    sizes = [s for s in SIZE_ORDER if s in sizes]
    reasons: list[str] = []
    per_size = {s: level_status(best, arch, s) for s in sizes}
    fronts = {s: frontier(per_size[s]) for s in sizes}
    big = [s for s in sizes if SIZE_ORDER.index(s) >= SIZE_ORDER.index(rule.min_size_for_verdict)]
    top = big[-1] if big else (sizes[-1] if sizes else None)

    # trend across sizes, per cell
    trend = {"improves": 0, "flat": 0, "regresses": 0, "cells": {}}
    cells = sorted({c for (s, c, a) in best if a == arch})
    for c in cells:
        seq = [(s, best[(s, c, arch)]["bits"]) for s in sizes if (s, c, arch) in best]
        if len(seq) < 2:
            continue
        diffs = [b2 - b1 for (_, b1), (_, b2) in zip(seq, seq[1:]) if b1 is not None and b2 is not None]
        if any(d < -rule.trend_tolerance_bits for d in diffs):
            kind = "regresses"
        elif sum(diffs) > rule.trend_tolerance_bits:
            kind = "improves"
        else:
            kind = "flat"
        trend[kind] += 1
        trend["cells"][c] = {"kind": kind, "bits_by_size": dict(seq)}

    hard_wins, speed = 0, None
    if top:
        for c in CELL_BY_ID.values():
            a = best.get((top, c.id, arch))
            if a and c.level in (3, 4) and not c.stretch:
                if a["pass"] or (a["vs_past_best"] is not None and a["vs_past_best"] >= -rule.trend_tolerance_bits):
                    hard_wins += 1
        ratios = []
        for (s, c, a2), v in best.items():
            if s == top and a2 == arch and v["tokens_per_sec"]:
                d = best.get((s, c, "dense"))
                if d and d["tokens_per_sec"]:
                    ratios.append(v["tokens_per_sec"] / d["tokens_per_sec"])
        speed = _median(ratios)

    ok_size = bool(big)
    ok_front = top is not None and fronts.get(top, -1) >= rule.min_frontier_level
    ok_hard = hard_wins >= rule.beats_past_on_hard
    ok_trend = trend["regresses"] == 0
    ok_speed = speed is None or speed >= rule.min_speed_vs_dense
    if not ok_size:
        reasons.append(f"no run at ≥ {rule.min_size_for_verdict}; the verdict needs one")
    if not ok_front:
        f = fronts.get(top, -1)
        reasons.append(
            f"{'no level passes' if f < 0 else f'frontier is L{f}'} at {top}; "
            f"need L{rule.min_frontier_level} (every L0–L2 cell passing)"
        )
    if not ok_hard:
        reasons.append("no long-reach / reasoning cell (L3/L4) passes or ties the best past architecture")
    if not ok_trend:
        reasons.append(f"{trend['regresses']} cell(s) get worse with size")
    if not ok_speed:
        reasons.append(f"trains at {speed:.2f}× the dense model's speed (need ≥ {rule.min_speed_vs_dense})")
    if ok_size and ok_front and ok_hard and ok_trend and ok_speed:
        verdict = "scale up"
    elif fronts and max(fronts.values()) >= 1 and (hard_wins or trend["improves"]):
        verdict = "promising — fix before scaling"
    else:
        verdict = "not ready"
    return {"verdict": verdict, "reasons": reasons, "frontier_by_size": fronts, "levels_by_size": per_size,
            "trend": trend, "hard_wins_at_top": hard_wins, "speed_vs_dense_at_top": speed, "top_size": top}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def _f(x, nd=1, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def render_md(best: dict, verdicts: dict, arches: list[str], sizes: list[str]) -> str:
    lines = [f"# Capability scorecard (suite {SUITE_VERSION})", ""]
    for arch in arches:
        v = verdicts[arch]
        lines += [f"## {arch}: **{v['verdict']}**", ""]
        for r in v["reasons"]:
            lines.append(f"- {r}")
        lines += ["", "| level | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
        for lv, L in LEVELS.items():
            row = [v["levels_by_size"][s].get(lv, "not-run") for s in sizes]
            lines.append(f"| L{lv} {L.name} | " + " | ".join(row) + " |")
        lines += ["", "| size | cell | bits / prize | acc (±SE) | pass | dense bits | past best | examples to 75% | lr | seeds |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for s in sizes:
            for c in CELL_BY_ID:
                a = best.get((s, c, arch))
                if not a:
                    continue
                past = f"{a['past_best_arch']} {_f(a['past_best_bits'])}" if a["past_best_arch"] else "—"
                lines.append(
                    f"| {s} | {c}{' (stretch)' if CELL_BY_ID[c].stretch else ''} | {_f(a['bits'])} / {_f(a['prize_bits'], 0)} | "
                    f"{_f(100 * a['acc'] if a['acc'] is not None else None, 0)}% (±{_f(100 * a['acc_se'] if a['acc_se'] else None, 1)}) | "
                    f"{'yes' if a['pass'] else 'no'} | {_f(a['dense_bits'])}{'' if a['dense_pass'] is not False else ' (dense < 75%)'} | {past} | "
                    f"{a['examples_to_75'] if a['examples_to_75'] is not None else '—'} | {a['lr']:g} | {a['seeds']} |")
        lines.append("")
    return "\n".join(lines)


_COL = {"pass": "#1a7f37", "fail": "#c62828", "uncalibrated": "#b26b00", "not-run": "#9a9892"}


def render_html(best: dict, verdicts: dict, arches: list[str], sizes: list[str]) -> str:
    e = html.escape
    parts = [f"""<meta charset="utf-8"><title>Capability scorecard</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#f7f6f2;color:#111}}
h1{{font-size:24px}}h2{{font-size:19px;margin-top:32px}}table{{border-collapse:collapse;margin:8px 0 18px}}
td,th{{padding:6px 9px;border-bottom:1px solid #dcdad3;text-align:left;vertical-align:middle}}th{{font-size:12px;color:#666;text-transform:uppercase}}
.pill{{display:inline-block;padding:2px 9px;border-radius:99px;color:#fff;font-weight:600;font-size:12px}}
.bar{{height:10px;background:#e7e5df;border-radius:5px;position:relative;width:180px}}.bar i{{position:absolute;left:0;top:0;bottom:0;border-radius:5px}}
.tick{{position:absolute;top:-3px;bottom:-3px;width:2px;background:#111}}.note{{color:#555;font-size:12.5px}}</style>
<h1>Capability scorecard <span class="note">suite {e(SUITE_VERSION)}</span></h1>
<p class="note">Pass = median answer accuracy ≥ 75 %. Bars: recovered bits as a share of the prize; black tick = best past
architecture at the same size (from the ledger); grey tick = dense model in this run.</p>"""]
    for arch in arches:
        v = verdicts[arch]
        col = {"scale up": "#1a7f37", "promising — fix before scaling": "#b26b00"}.get(v["verdict"], "#c62828")
        parts.append(f'<h2>{e(arch)} <span class="pill" style="background:{col}">{e(v["verdict"])}</span></h2>')
        if v["reasons"]:
            parts.append("<ul>" + "".join(f"<li>{e(r)}</li>" for r in v["reasons"]) + "</ul>")
        parts.append("<table><tr><th>level</th>" + "".join(f"<th>{e(s)}</th>" for s in sizes) + "</tr>")
        for lv, L in LEVELS.items():
            parts.append(f"<tr><td>L{lv} · {e(L.name)}</td>" + "".join(
                f'<td><span class="pill" style="background:{_COL.get(v["levels_by_size"][s].get(lv, "not-run"), "#4a3aa7")}">'
                f'{e(v["levels_by_size"][s].get(lv, "not-run"))}</span></td>' for s in sizes) + "</tr>")
        parts.append("</table><table><tr><th>size</th><th>cell</th><th>share of prize</th><th>bits</th><th>acc</th>"
                     "<th>dense</th><th>past best</th><th>examples to 75%</th></tr>")
        for s in sizes:
            for c, cell in CELL_BY_ID.items():
                a = best.get((s, c, arch))
                if not a:
                    continue
                prize = a["prize_bits"] or cell.prize_bits
                share = max(0.0, min(1.0, (a["bits"] or 0) / prize))
                fill = "#1a7f37" if a["pass"] else "#2a78d6"
                ticks = ""
                if a["past_best_bits"] is not None:
                    ticks += f'<span class="tick" style="left:{180 * min(1, a["past_best_bits"] / prize):.0f}px"></span>'
                if a["dense_bits"] is not None:
                    ticks += f'<span class="tick" style="background:#999;left:{180 * min(1, a["dense_bits"] / prize):.0f}px"></span>'
                parts.append(
                    f"<tr><td>{e(s)}</td><td>{e(c)}{' <span class=note>(stretch)</span>' if cell.stretch else ''}"
                    f"<div class=note>{e(cell.what)}</div></td>"
                    f'<td><div class="bar"><i style="width:{180 * share:.0f}px;background:{fill}"></i>{ticks}</div></td>'
                    f"<td>{_f(a['bits'])} / {_f(prize, 0)}</td><td>{_f(100 * (a['acc'] or 0), 0)}%</td>"
                    f"<td>{_f(a['dense_bits'])}</td><td>{e(str(a['past_best_arch'] or '—'))} {_f(a['past_best_bits'])}</td>"
                    f"<td>{a['examples_to_75'] if a['examples_to_75'] is not None else '—'}</td></tr>")
        parts.append("</table>")
        tr = v["trend"]
        parts.append(f"<p class=note>Size trend over cells run at ≥ 2 sizes: {tr['improves']} improve, {tr['flat']} flat, "
                     f"{tr['regresses']} regress. Speed vs dense at {e(str(v['top_size']))}: "
                     f"{_f(v['speed_vs_dense_at_top'], 2)}×.</p>")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", nargs="+", required=True,
                   help="one or more suite output folders (e.g. the Odra and Polonez halves of one run)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--arch", nargs="*", default=None, help="architectures to score (default: every non-dense arch found)")
    args = p.parse_args()
    in_dirs = [Path(d) for d in args.in_dir]
    out_dir = Path(args.out_dir or in_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for d in in_dirs for r in load_runs(d)]
    if not rows:
        raise SystemExit(f"no job.json + rung JSON under {', '.join(map(str, in_dirs))}")
    best = aggregate(rows)
    arches = args.arch or sorted({a for (_, _, a) in best if a != "dense"}) or ["dense"]
    sizes = [s for s in SIZE_ORDER if any(k[0] == s for k in best)]
    verdicts = {a: scale_verdict(best, a, sizes) for a in arches}
    (out_dir / "scorecard.md").write_text(render_md(best, verdicts, arches, sizes))
    (out_dir / "scorecard.html").write_text(render_html(best, verdicts, arches, sizes))
    (out_dir / "scorecard.json").write_text(json.dumps({
        "suite_version": SUITE_VERSION, "verdicts": verdicts,
        "cells": [v for v in best.values()],
    }, indent=2, default=str))
    with (out_dir / "cells.csv").open("w", newline="") as fh:
        keys = ["size", "cell", "level", "arch", "lr", "seeds", "bits", "prize_bits", "acc", "acc_se", "pass",
                "dense_bits", "vs_dense", "past_best_arch", "past_best_bits", "vs_past_best", "examples_to_75",
                "tokens_per_sec", "params"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for v in sorted(best.values(), key=lambda v: (SIZE_ORDER.index(v["size"]), v["cell"], v["arch"])):
            w.writerow(v)
    for a in arches:
        v = verdicts[a]
        print(f"{a}: {v['verdict']}  frontier {v['frontier_by_size']}")
        for r in v["reasons"]:
            print(f"   - {r}")
    print(f"wrote {out_dir}/scorecard.md · scorecard.html · scorecard.json · cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

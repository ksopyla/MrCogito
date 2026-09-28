#!/usr/bin/env python
"""Collect a study's per-job results into compact tables (markdown + JSON).

For each `<root>/<job>/`:
- the probe rung JSON gives final accuracy ± SE, bits, examples to 75 %, whether the budget
  was extended, steps and s/step, plus a per-letter profile (first 8 vs last 8 answer letters);
- `ladder.json`, if present, gives accuracy per length;
- the `DONE` / `FAILED` markers give the status.

    uv run python analysis/study_table.py --root Cache/study/e30_vs_e31 [--prefix ratio_] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _k(L: int) -> str:
    return f"{L // 1024}k" if L >= 1024 else str(L)


def job_row(d: Path) -> dict | None:
    rungs = [p for p in d.glob("*.json") if p.name not in ("ladder.json", "summary.json")]
    status = "done" if (d / "DONE").exists() else ("FAILED" if (d / "FAILED").exists() else "running")
    row: dict = {"job": d.name, "status": status}
    if rungs:
        r = json.loads(rungs[0].read_text())
        for arch, res in (r.get("results") or {}).items():
            fin, info = res.get("final") or {}, res.get("info") or {}
            ppa = fin.get("per_position_acc") or []
            etc = res.get("examples_to_criterion") or {}
            row.update({
                "arch": arch, "acc": fin.get("acc"), "se": fin.get("acc_se"),
                "bits": info.get("recovered_bits"), "prize": info.get("prize_bits"),
                "best": res.get("best_acc"), "ext": res.get("k1_extended"),
                "step": fin.get("step"), "s_step": (res.get("throughput") or {}).get("sec_per_step"),
                "ex75": etc.get("examples") if etc.get("confirmed") else None,
                "p0": ppa[0] if ppa else None,
                "first8": sum(ppa[:8]) / 8 if len(ppa) >= 16 else None,
                "last8": sum(ppa[-8:]) / 8 if len(ppa) >= 16 else None,
                "params_m": (res.get("params") or 0) / 1e6,
            })
            break
    elif status == "running":
        log = d / "probe.log"
        if log.exists():
            steps = [ln for ln in log.read_text().splitlines() if " step " in ln and "acc" in ln]
            if steps:
                row["last"] = steps[-1].strip()[:110]
    lad = d / "ladder.json"
    if lad.exists():
        res = json.loads(lad.read_text()).get("results") or {}
        for arch, by_len in res.items():
            row["ladder"] = {int(L): v.get("acc") for L, v in by_len.items() if isinstance(v, dict) and "acc" in v}
            row["ladder_first"] = {int(L): v.get("first_acc") for L, v in by_len.items()
                                   if isinstance(v, dict) and "first_acc" in v}
            break
    return row


def fmt(v, pct=True):
    if v is None:
        return "·"
    return f"{100 * v:.0f}" if pct else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="Cache/study/e30_vs_e31")
    p.add_argument("--prefix", nargs="*", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--first", action="store_true",
                   help="ladder cells show first-letter accuracy (honest on multi-candidate exams; '·' if not measured)")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir() and x.name != "launch"):
        if args.prefix and not any(d.name.startswith(s) for s in args.prefix):
            continue
        r = job_row(d)
        if r:
            rows.append(r)
    lens = sorted({L for r in rows for L in (r.get("ladder") or {})})
    head = ["job", "st", "acc", "±se", "p0", "bits", "ext", "ex75(k)", "1st8", "last8", "s/step"] + [_k(L) for L in lens]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        lad = (r.get("ladder_first") or {}) if args.first else (r.get("ladder") or {})
        cells = [r["job"], r["status"][:4], fmt(r.get("acc")), fmt(r.get("se")), fmt(r.get("p0")),
                 fmt(r.get("bits"), False), "y" if r.get("ext") else "",
                 f"{r['ex75'] / 1000:.0f}" if r.get("ex75") else "·",
                 fmt(r.get("first8")), fmt(r.get("last8")), fmt(r.get("s_step"), False)]
        cells += [fmt(lad.get(L)) for L in lens]
        print("| " + " | ".join(cells) + " |")
        if r.get("last"):
            print(f"|   ↳ {r['last']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

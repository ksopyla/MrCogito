#!/usr/bin/env python
"""Capability board: the one visual summary of every architecture's capability checks.

Reads the committed capability ledger (`docs/2_Experiments_Registry/results/capability/`, written by
`scripts/pull_capability_results.sh`) and writes one self-contained HTML page:
  * the no-harm check of each variant against the champion (default `e31_li_m1`): a capability is
    lost when the variant is more than 5 points below the champion where the champion passes (≥ 75 %);
  * the length battery (the E31b protocol): first-letter accuracy against input length, one small
    chart per exam and training stage, one line per variant;
  * the capability suite: the honest score per cell (first letter on multi-candidate cells, the
    mean over answer letters otherwise) per architecture and size, current suite version only.
Process: docs/engineering_specs/capability_checks.md (skill `capability-checks`).

    uv run python analysis/capability_board.py                       # all variants with a battery
    uv run python analysis/capability_board.py --variants e31_li_m1 e33a_loop e31_li --champion e31_li_m1

Variant ids join the two batteries: a battery variant is `{tag}_{arm}` (or the arch for the E31b
`len_*` / `hard_*` jobs), and should equal the variant's suite arch name.
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

OUT = ROOT / "docs" / "3_Evaluations_and_Baselines" / "capability_board.html"
PASS, HARM_POINTS = 0.75, 0.05
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
               "pchain3": "Parallel 3-hop chain", "count": "Count", "majority": "Majority"}
STAGE = {"": None, "b8k": "8k", "b16k": "16k"}

_PATTERNS = [  # (regex, how to build (variant, exam, stage))
    (re.compile(r"^len_(lookup|chain)_(.+)_s\d+(?:_(b8k|b16k))?$"), lambda m: (m[2], m[1], m[3] or "")),
    (re.compile(r"^hard_([a-z0-9]+)_(.+)_s\d+(?:_(b8k))?$"), lambda m: (m[2], m[1], m[3] or "")),
    (re.compile(r"^dense_([a-z0-9]+)_s\d+$"), lambda m: ("dense", m[1], "")),
    (re.compile(r"^([a-z0-9]+)_(lookup|chain)_([a-z0-9]+)_s\d+(?:_(b8k|b16k))?$"),
     lambda m: (f"{m[1]}_{m[3]}", m[2], m[4] or "")),
    (re.compile(r"^([a-z0-9]+)_hard_([a-z0-9]+)_([a-z0-9]+)_s\d+(?:_(b8k))?$"),
     lambda m: (f"{m[1]}_{m[3]}", m[2], m[4] or "")),
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
            out[slot][variant] = {
                str(L): {"first": _med(c["first"]), "mean": _med(c["mean"]),
                         "seeds": sum(x is not None for x in c["mean"]),
                         "first_seeds": sum(x is not None for x in c["first"])}
                for L, c in sorted(by_len.items())}
    return out


def suite(ledgers: list[dict]):
    versions = sorted({led["suite_version"] for led in ledgers if led.get("suite_version")})
    current = versions[-1] if versions else None
    by = defaultdict(lambda: {"acc": [], "p0": [], "sources": set()})
    for led in ledgers:
        if led.get("suite_version") != current:
            continue
        for r in suite_rows(led):
            k = (r["size"], r["cell"], r["level"], r["arch"])
            by[k]["acc"].append(r["acc"])
            by[k]["p0"].append(r["p0"])
            by[k]["sources"].add(r["source"])
    cells = []
    for (size, cell, level, arch), v in sorted(by.items()):
        multi = any(t in cell for t in MULTI_CANDIDATE)
        acc, p0 = _med(v["acc"]), _med(v["p0"])
        cells.append({"size": size, "cell": cell, "level": level, "arch": arch, "acc": acc, "p0": p0,
                      "honest": p0 if multi else acc, "multi": multi, "seeds": len(v["acc"]),
                      "pass": acc is not None and acc >= PASS, "sources": sorted(v["sources"])})
    return current, [v for v in versions if v != current], cells


def no_harm(bat: dict, suite_cells: list[dict], champion: str, variants: list[str]) -> dict:
    out = {}
    suite_by = defaultdict(dict)
    for c in suite_cells:
        suite_by[(c["size"], c["cell"])][c["arch"]] = c
    for v in variants:
        if v in (champion, "dense"):
            continue
        kept, lost, missing = [], [], []
        for slot, by_v in bat.items():
            champ = by_v.get(champion)
            if not champ:
                continue
            mine = by_v.get(v)
            for L, cv in champ.items():
                if cv["first"] is None or cv["first"] < PASS:
                    continue
                item = {"where": f"{slot} @ {int(L) // 1024}k", "champion": cv["first"]}
                mv = (mine or {}).get(L)
                if not mv or mv["first"] is None:
                    missing.append(item)
                elif mv["first"] < cv["first"] - HARM_POINTS:
                    lost.append({**item, "variant": mv["first"]})
                else:
                    kept.append({**item, "variant": mv["first"]})
        for (size, cell), by_a in suite_by.items():
            cc, mc = by_a.get(champion), by_a.get(v)
            if not cc or cc["honest"] is None or cc["honest"] < PASS:
                continue
            item = {"where": f"suite {cell} ({size})", "champion": cc["honest"]}
            if not mc or mc["honest"] is None:
                missing.append(item)
            elif mc["honest"] < cc["honest"] - HARM_POINTS:
                lost.append({**item, "variant": mc["honest"]})
            else:
                kept.append({**item, "variant": mc["honest"]})
        out[v] = {"kept": len(kept), "lost": lost, "missing": len(missing)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger_dir", default=str(LEDGER_DIR))
    p.add_argument("--champion", default="e31_li_m1")
    p.add_argument("--variants", nargs="*", default=None,
                   help="variants to show (default: every variant with ≥ 8 standard battery slots or registered in BATTERY_VARIANTS, plus dense)")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    ledgers = load_ledgers(Path(args.ledger_dir))
    if not ledgers:
        raise SystemExit(f"no ledger files under {args.ledger_dir}")
    bat = battery([l for l in ledgers if l["kind"] == "study"])
    current, old_versions, suite_cells = suite([l for l in ledgers if l["kind"] == "suite"])
    std_ids = [s[0] for s in STANDARD]
    coverage = defaultdict(int)
    for slot in std_ids:
        for v in bat.get(slot, {}):
            coverage[v] += 1
    registered = set()
    try:  # variants registered for the battery are shown as soon as they have any result
        from scripts.study_plans.e30_vs_e31 import BATTERY_VARIANTS
        registered = {f"{k['tag']}_{k['arm']}" for k in BATTERY_VARIANTS.values()}
    except ImportError:
        pass
    variants = args.variants or sorted(v for v, n in coverage.items() if n >= 8 or v in registered)
    variants = [args.champion] + [v for v in variants if v != args.champion]
    if "dense" in bat.get("recall8_1k", {}) and "dense" not in variants:
        variants.append("dense")
    slots = [{"id": s, "exam": e, "stage": st, "standard": True} for s, e, st in STANDARD if s in bat]
    for s in sorted(set(bat) - set(std_ids)):
        exam, length = s.rsplit("_", 1)
        slots.append({"id": s, "exam": EXTRA_NAMES.get(exam, exam), "stage": f"trained at {length}",
                      "standard": False})
    data = {
        "generated": datetime.date.today().isoformat(), "champion": args.champion, "variants": variants,
        "pass": PASS, "harm_points": HARM_POINTS, "slots": slots,
        "battery": {s["id"]: {v: bat[s["id"]][v] for v in bat[s["id"]] if v in variants} for s in slots},
        "suite_version": current, "suite_old_versions": old_versions,
        "suite": [c for c in suite_cells if c["arch"] in variants or c["arch"] in ("dense",)],
        "suite_all_arches": sorted({c["arch"] for c in suite_cells}),
        "no_harm": no_harm(bat, suite_cells, args.champion, variants),
        "sources": [{"file": str(Path(l["_file"]).relative_to(ROOT)), "kind": l["kind"], "name": l["name"],
                     "host": l["host"], "collected": l["collected"], "archive": l["archive_path"],
                     "done": sum(j["status"] == "done" for j in l["jobs"]), "jobs": len(l["jobs"]),
                     "suite_version": l.get("suite_version")} for l in ledgers],
    }
    template = (Path(__file__).parent / "capability_board_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out}: {len(variants)} variants, {len(slots)} battery slots, "
          f"{len(data['suite'])} suite cells (suite {current})")
    for v, r in data["no_harm"].items():
        print(f"  {v}: kept {r['kept']}, lost {len(r['lost'])}, not run {r['missing']} (vs {args.champion})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

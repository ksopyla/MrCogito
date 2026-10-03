#!/usr/bin/env python3
"""Capability ledger: compact, committed copies of capability-suite and study results.

Raw results live on the server that ran them (`Cache/capability/<run>`, `Cache/study/<study>`) and
are archived to the NAS; this module turns one such folder into a small JSON file that is
committed under `docs/2_Experiments_Registry/results/capability/`, so every past result can be
compared, scored and plotted from the repository even when the server is off.
Process: docs/engineering_specs/capability_checks.md (skill `capability-checks`).

Standard library only (Python ≥ 3.8): it runs on any server checkout without syncing code,
fed through ssh stdin by `scripts/pull_capability_results.sh`:

    ssh odra 'python3 - collect ~/dev/MrCogito-e31/Cache/study/e30_vs_e31 --host odra' \
        < analysis/capability_ledger.py > docs/2_Experiments_Registry/results/capability/study/e30_vs_e31.odra.json

Library use (scorecard, board): `load_ledgers()`, `suite_rows()`, `study_jobs()`.

Schema 1 — one file per (folder, host):
  {schema, kind: "suite"|"study", name, host, source_path, archive_path, collected, git: [...],
   suite_version, tier, jobs: [...]}
  suite job: {job_id, size, cell, level, seed, lr, status, cmd, results: {arch: METRICS}}
  study job: {name, status, args, init, results: {arch: METRICS},
              ladders: {"ladder": {arch: {length: LADDER}}, "ladder_lookup": {...}, ...}}
  METRICS: acc, acc_se, p0 (first answer letter), ppa (accuracy per answer letter), bits,
           prize_bits, flow, best_acc, extended, step, examples_to_75, sec_per_step, tokens_per_sec,
           params
  LADDER:  acc, acc_se, first_acc, first_acc_se, rows, ce_nats, sec_per_row, peak_gb, memory_slots
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

SCHEMA = 1
NAS_RESULTS = "/nas/ml_data/mrcogito/results"
LEDGER_DIR = Path(__file__).resolve().parents[1] / "docs" / "2_Experiments_Registry" / "results" / "capability"
_NOT_RUNG = ("job.json", "summary.json", "cost_bench.json")
_LADDER_KEYS = ("acc", "acc_se", "first_acc", "first_acc_se", "rows", "ce_nats", "sec_per_row", "peak_gb",
                "memory_slots")


def _r(x, nd=5):
    return round(x, nd) if isinstance(x, float) else x


def _status(d: Path, results: dict) -> str:
    """DONE / FAILED markers; a job without a marker counts as done once its rung JSON has results
    (the probe writes it at the end; runs from before the markers have no marker at all)."""
    if (d / "DONE").exists():
        return "done"
    if (d / "FAILED").exists():
        return "failed"
    return "done" if results else "running"


def _metrics(r: dict) -> dict:
    fin, info = r.get("final") or {}, r.get("info") or {}
    etc, thr = r.get("examples_to_criterion") or {}, r.get("throughput") or {}
    ppa = fin.get("per_position_acc") or []
    confirmed = etc.get("confirmed", etc.get("step") is not None)
    return {
        "acc": _r(fin.get("acc")), "acc_se": _r(fin.get("acc_se")), "p0": _r(ppa[0]) if ppa else None,
        "ppa": [_r(x, 4) for x in ppa], "bits": _r(info.get("recovered_bits"), 3),
        "prize_bits": _r(info.get("prize_bits"), 3), "flow": _r(info.get("information_flow"), 4),
        "best_acc": _r(r.get("best_acc")), "extended": r.get("k1_extended"), "step": fin.get("step"),
        "examples_to_75": etc.get("examples") if confirmed else None,
        "sec_per_step": _r(thr.get("sec_per_step"), 4), "tokens_per_sec": _r(thr.get("tokens_per_sec"), 1),
        "params": r.get("params"),
    }


def _rung(d: Path) -> dict:
    rungs = sorted(p for p in d.glob("*.json") if p.name not in _NOT_RUNG and not p.name.startswith("ladder"))
    if not rungs:
        return {}
    res = json.loads(rungs[0].read_text()).get("results") or {}
    return {arch: _metrics(r) for arch, r in res.items()}


def _ladder(p: Path) -> dict:
    res = json.loads(p.read_text()).get("results") or {}
    return {arch: {str(L): {k: _r(v.get(k)) for k in _LADDER_KEYS if k in v}
                   for L, v in by_len.items() if isinstance(v, dict)}
            for arch, by_len in res.items()}


def collect_suite(root: Path) -> dict:
    jobs, versions, tiers = [], set(), set()
    for jp in sorted(root.rglob("job.json")):
        job = json.loads(jp.read_text())
        versions.add(job.get("suite_version"))
        tiers.add(job.get("tier"))
        results = _rung(jp.parent)
        jobs.append({
            "job_id": job.get("job_id"), "size": job["size"], "cell": job["cell"], "level": job["level"],
            "seed": job["seed"], "lr": job["lr"], "status": _status(jp.parent, results), "cmd": job.get("cmd"),
            "results": results,
        })
    plan = root / "plan.json"
    git = []
    if plan.exists():
        git = [json.loads(plan.read_text()).get("git")]
    return {"kind": "suite", "suite_version": "/".join(sorted(v for v in versions if v)),
            "tier": "/".join(sorted(t for t in tiers if t)), "git": [g for g in git if g], "jobs": jobs}


def collect_study(root: Path) -> dict:
    planned, git = {}, []
    for pp in sorted(root.glob("plan_*.json")):
        plan = json.loads(pp.read_text())
        if plan.get("git") and plan["git"] not in git:
            git.append(plan["git"])
        for j in plan.get("jobs") or []:
            planned[j["name"]] = {"args": j.get("args"), "init": j.get("init"), "phase": plan.get("phase")}
    jobs = []
    for d in sorted(x for x in root.iterdir() if x.is_dir() and x.name not in ("launch", "logs")):
        ladders = {p.stem: _ladder(p) for p in sorted(d.glob("ladder*.json"))}
        meta, results = planned.get(d.name, {}), _rung(d)
        jobs.append({"name": d.name, "status": _status(d, results or ladders), "phase": meta.get("phase"),
                     "args": meta.get("args"), "init": meta.get("init"), "results": results, "ladders": ladders})
    return {"kind": "study", "git": git, "jobs": jobs}


def collect(root: Path, host: str) -> dict:
    root = root.expanduser().resolve()
    is_suite = any(root.rglob("job.json"))
    body = collect_suite(root) if is_suite else collect_study(root)
    kind = body["kind"]
    return {"schema": SCHEMA, "kind": kind, "name": root.name, "host": host, "source_path": str(root),
            "archive_path": f"{NAS_RESULTS}/{'capability' if kind == 'suite' else 'study'}/{root.name}",
            "collected": datetime.date.today().isoformat(), **body}


# ---- library: read the committed ledger -------------------------------------------------------

def load_ledgers(ledger_dir: Path = LEDGER_DIR, kind: str | None = None) -> list[dict]:
    out = []
    for p in sorted(Path(ledger_dir).rglob("*.json")):
        d = json.loads(p.read_text())
        if d.get("schema") == SCHEMA and (kind is None or d.get("kind") == kind):
            d["_file"] = str(p)
            out.append(d)
    return out


def suite_rows(ledger: dict) -> list[dict]:
    """One row per (job, arch) in the shape `analysis/capability_scorecard.load_runs` returns."""
    rows = []
    for j in ledger.get("jobs", []):
        if j["status"] == "running":
            continue
        for arch, m in (j.get("results") or {}).items():
            rows.append({
                "size": j["size"], "cell": j["cell"], "level": j["level"], "seed": j["seed"], "lr": j["lr"],
                "arch": arch, "bits": m.get("bits"), "prize_bits": m.get("prize_bits"), "flow": m.get("flow"),
                "acc": m.get("acc"), "acc_se": m.get("acc_se"), "params": m.get("params"),
                "examples_to_75": m.get("examples_to_75"), "tokens_per_sec": m.get("tokens_per_sec"),
                "per_position_acc": m.get("ppa"), "steps": m.get("step"), "p0": m.get("p0"),
                "source": f"{ledger['name']}@{ledger['host']}",
            })
    return rows


def study_jobs(ledger: dict) -> list[dict]:
    """Flattened finished study jobs: name, arch, seed, recipe, seq_len, metrics and ladders."""
    out = []
    for j in ledger.get("jobs", []):
        args = j.get("args") or []

        def flag(name, default=None):
            return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else default

        for arch, m in (j.get("results") or {}).items():
            out.append({"job": j["name"], "status": j["status"], "phase": j.get("phase"), "arch": arch,
                        "seed": flag("--seed"), "recipe": flag("--recipe"), "seq_len": flag("--seq_len"),
                        "init": j.get("init"), **m,
                        "ladders": {k: v.get(arch, {}) for k, v in (j.get("ladders") or {}).items()},
                        "source": f"{ledger['name']}@{ledger['host']}"})
    return out


def dumps(led: dict) -> str:
    """Readable and diff-friendly: header keys one per line, one job per line."""
    head = {k: v for k, v in led.items() if k != "jobs" and not k.startswith("_")}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()] + [' "jobs": [']
    jobs = [f"  {json.dumps(j, separators=(',', ':'))}" for j in led.get("jobs", [])]
    return "\n".join(lines + [",\n".join(jobs), " ]", "}"]) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="print the ledger JSON of one suite or study folder")
    c.add_argument("root")
    c.add_argument("--host", required=True)
    c.add_argument("--out", default=None, help="write here instead of stdout")
    s = sub.add_parser("summary", help="list the committed ledger files")
    s.add_argument("--dir", default=str(LEDGER_DIR))
    args = p.parse_args()
    if args.cmd == "collect":
        led = collect(Path(args.root), args.host)
        text = dumps(led)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
        else:
            sys.stdout.write(text)
        n = len(led["jobs"])
        done = sum(j["status"] == "done" for j in led["jobs"])
        print(f"{led['kind']} {led['name']}@{led['host']}: {done}/{n} jobs done", file=sys.stderr)
        return 0
    for led in load_ledgers(Path(args.dir)):
        n = len(led["jobs"])
        done = sum(j["status"] == "done" for j in led["jobs"])
        print(f"{led['kind']:5s} {led['name']}@{led['host']}  {done}/{n} done  collected {led['collected']}"
              f"  archive {led['archive_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

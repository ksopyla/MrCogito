#!/usr/bin/env python
"""Compact ledger of one text-checks run folder (scripts/run_text_checks.py output) — runs on the
server, prints JSON; `scripts/pull_text_checks_results.sh` saves it under
docs/2_Experiments_Registry/results/capability/text/<run>.<host>.json (committed), and
`analysis/text_board.py` draws every ledger on the text capability board.

Holds every job's state (pending / running / done / over_budget / failed), its settings card, a thinned
loss curve and throughput, and every evaluated cell (no per-item rows), so the board needs nothing else.

    python analysis/text_checks_ledger.py --in_dir Cache/text_checks/r1_screen --host polonez > ledger.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.text_checks_scorecard import collect  # noqa: E402
from evaluation.text_checks import TEXT_CHECKS_VERSION  # noqa: E402

_EXIT = re.compile(r"^EXIT (\S+) (\d+)")
CARD_KEYS = ("job", "kind", "arch", "role", "tier", "params", "params_in_band", "lr", "tokens", "steps", "global_rows",
             "mean_row_tokens", "n_gpus", "cap_gpu_hours", "seed", "data_version", "git_commit", "planned", "extra_env")


def _thin(points: list, n: int = 60) -> list:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def _exits(run: Path) -> dict:
    last = {}
    for log in sorted((run / "launch").glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = _EXIT.match(line.strip())
            if m:
                last[m[1]] = int(m[2])
    return last


def job_state(d: Path, exits: dict) -> str:
    status = (d / "status").read_text().strip() if (d / "status").exists() else ""
    if (d / "DONE").exists():
        return "over_budget" if status == "over_budget" else "done"
    if exits.get(d.name, 0) not in (0, 3):
        return "failed"
    if (d / "status.run").exists() or (d.name == "data" and (d / "build.log").exists()):
        return "running"
    return "pending"


def build(run: Path, host: str) -> dict:
    exits = _exits(run)
    jobs = []
    data_meta = None
    for jj in sorted((run / "jobs").glob("*/job.json")):
        d = jj.parent
        card = json.loads(jj.read_text())
        row = {k: card.get(k) for k in CARD_KEYS if k in card}
        row["state"] = job_state(d, exits)
        if (d / "active_seconds").exists():
            row["active_seconds"] = int((d / "active_seconds").read_text().strip() or 0)
        res = d / "result.json"
        if res.exists():
            r = json.loads(res.read_text())
            row["eval_loss"] = _thin(r.get("eval_loss") or [])
            row["train_loss"] = _thin(r.get("train_loss") or [])
            tps = r.get("tokens_per_second") or []
            row["tokens_per_second"] = sum(tps) / len(tps) if tps else None
        elif card.get("kind") == "train" and (d / "train.log").exists():
            # live run: last logged loss and throughput from the log
            txt = (d / "train.log").read_text(errors="replace")[-200_000:]
            losses = re.findall(r"'loss': ([0-9.]+).*?'epoch'", txt)
            steps = re.findall(r"'perf/real_tokens_per_second': ([0-9.]+)", txt)
            row["live_loss"] = float(losses[-1]) if losses else None
            row["tokens_per_second"] = float(steps[-1]) if steps else None
        if card.get("kind") == "eval":
            row["files"] = sorted(p.name for p in d.glob("*.json") if p.name != "job.json")
        if card.get("kind") == "data" or (data_meta is None and card.get("data")):
            mp = Path(card.get("data", "")) / "text_checks_meta.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                data_meta = {k: m.get(k) for k in ("version", "tokenizer_sha256", "eval_sha256", "eval_items", "mix",
                                                   "stories", "built_s")}
        jobs.append(row)
    models = {}
    for key, m in collect([run]).items():
        m = dict(m)
        for k in ("cells", "off"):
            m[k] = {f"{s}|{t}|{L}": c for (s, t, L), c in m[k].items()}
        m["curve"] = {str(k): v for k, v in m["curve"].items()}
        models[key] = m
    return {"kind": "text_checks", "version": TEXT_CHECKS_VERSION, "run": run.name, "host": host,
            "collected": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": data_meta,
            "jobs": jobs, "models": models}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--host", default="local")
    a = p.parse_args()
    print(json.dumps(build(Path(a.in_dir), a.host), separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()

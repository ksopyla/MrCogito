#!/usr/bin/env python
"""Validate W&B lineage coverage for recent evaluation runs.

Gate script for staged rollout:
- scans recent eval/benchmark runs,
- checks canonical lineage fields,
- reports linked coverage,
- exits non-zero if coverage drops below threshold.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check W&B eval lineage contract coverage.")
    parser.add_argument("--entity", type=str, default="ksopyla")
    parser.add_argument("--project", type=str, default="MrCogito")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days.")
    parser.add_argument("--max_runs", type=int, default=200)
    parser.add_argument(
        "--min_linked_coverage",
        type=float,
        default=1.0,
        help="Required fraction of linked eval runs (1.0 = 100%%).",
    )
    return parser.parse_args()


def is_eval_job(run: Any) -> bool:
    job_type = (run.job_type or "").lower()
    return "benchmark" in job_type or "evaluation" in job_type


def is_linked(run: Any) -> bool:
    cfg = run.config or {}
    run_id = cfg.get("source_training_run_id")
    group = cfg.get("source_training_group")
    step = cfg.get("source_checkpoint_step")
    status = cfg.get("lineage_status")
    return bool(run_id and group and step is not None and status == "linked")


def main() -> int:
    args = parse_args()
    api = wandb.Api(timeout=45)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    eval_runs: list[Any] = []
    for run in api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=200):
        if len(eval_runs) >= args.max_runs:
            break
        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
        if created < cutoff:
            break
        if is_eval_job(run):
            eval_runs.append(run)

    if not eval_runs:
        print("No recent eval runs in scope; lineage gate skipped.")
        return 0

    linked = sum(1 for run in eval_runs if is_linked(run))
    coverage = linked / len(eval_runs)
    print(
        f"Lineage coverage: linked={linked}/{len(eval_runs)} "
        f"({coverage:.2%}), required={args.min_linked_coverage:.2%}"
    )

    if coverage < args.min_linked_coverage:
        print("Lineage contract check FAILED.")
        return 1
    print("Lineage contract check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

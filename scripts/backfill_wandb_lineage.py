#!/usr/bin/env python
"""Backfill canonical W&B lineage fields for recent evaluation runs.

Additive migration utility:
- scans recent eval/benchmark runs,
- infers canonical lineage fields from model_path + optional parent run lookup,
- updates config/tags/group in W&B (when --apply is set),
- always emits a CSV report for auditability.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import wandb

# Add project root for local package imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.wandb_identity import (
    build_namespaced_eval_tags,
    lineage_to_wandb_config,
    resolve_eval_lineage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill W&B eval lineage metadata.")
    parser.add_argument("--entity", type=str, default="ksopyla")
    parser.add_argument("--project", type=str, default="MrCogito")
    parser.add_argument("--days", type=int, default=60, help="Backfill window in days.")
    parser.add_argument("--max_runs", type=int, default=500)
    parser.add_argument("--apply", action="store_true", help="Apply updates to W&B (default: dry-run).")
    parser.add_argument("--report_dir", type=str, default="Cache/Evaluation_reports")
    return parser.parse_args()


def is_eval_job(run: Any) -> bool:
    job_type = (run.job_type or "").lower()
    return "benchmark" in job_type or "evaluation" in job_type


def _merge_tags(existing: list[str], generated: list[str]) -> list[str]:
    return list(dict.fromkeys([*(existing or []), *generated]))


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=45)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    runs = api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=200)

    rows: list[dict[str, Any]] = []
    processed = 0

    for run in runs:
        if processed >= args.max_runs:
            break
        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
        if created < cutoff:
            break
        if not is_eval_job(run):
            continue
        processed += 1

        config = dict(run.config or {})
        model_path = config.get("model_path")
        if not model_path:
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "job_type": run.job_type,
                    "status": "skipped_no_model_path",
                    "updated": False,
                }
            )
            continue

        try:
            lineage = resolve_eval_lineage(
                model_path=model_path,
                source_training_run_id=config.get("source_training_run_id"),
                source_training_group=config.get("source_training_group"),
                source_training_experiment_id=config.get("source_training_experiment_id"),
                source_checkpoint_step=config.get("source_checkpoint_step"),
                source_checkpoint_epoch=config.get("source_checkpoint_epoch"),
                allow_unlinked_eval=True,
                wandb_entity=args.entity,
                wandb_project=args.project,
                resolve_with_wandb=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "job_type": run.job_type,
                    "status": f"error:{exc}",
                    "updated": False,
                }
            )
            continue

        update_payload = lineage_to_wandb_config(lineage)
        changed_keys = [k for k, v in update_payload.items() if config.get(k) != v]
        model_family = config.get("model_family") or config.get("checkpoint_family") or config.get("model_type") or "unknown"
        params_m = int(round((config.get("total_params") or config.get("model/num_parameters") or 0) / 1_000_000))
        generated_tags = build_namespaced_eval_tags(
            benchmark=config.get("benchmark") or run.job_type or "unknown",
            model_family=model_family,
            objective_family=config.get("objective_family") or config.get("pretraining_objective"),
            params_m=params_m,
            tokenizer_name=config.get("tokenizer_name") or "unknown",
            lineage=lineage,
            extra_tags=run.tags or [],
        )
        merged_tags = _merge_tags(run.tags or [], generated_tags)
        tags_changed = merged_tags != (run.tags or [])
        group_changed = bool(lineage.source_training_group and run.group != lineage.source_training_group)

        should_update = bool(changed_keys or tags_changed or group_changed)
        if args.apply and should_update:
            run.config.update(update_payload, allow_val_change=True)
            if tags_changed:
                run.tags = merged_tags
            if group_changed:
                run.group = lineage.source_training_group
            run.update()

        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "job_type": run.job_type,
                "status": lineage.lineage_status,
                "changed_keys": ",".join(changed_keys),
                "tags_changed": tags_changed,
                "group_changed": group_changed,
                "updated": bool(args.apply and should_update),
            }
        )

    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.report_dir, f"wandb_lineage_backfill_{stamp}.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "run_id",
                "run_name",
                "job_type",
                "status",
                "changed_keys",
                "tags_changed",
                "group_changed",
                "updated",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] processed={processed} report={report_path}")


if __name__ == "__main__":
    main()

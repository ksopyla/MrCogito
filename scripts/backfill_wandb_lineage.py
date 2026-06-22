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
import re
import shlex
import subprocess
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
    parser.add_argument(
        "--remote_check_hosts",
        type=str,
        default="odra,polonez",
        help="Comma-separated SSH hosts used to verify checkpoint path existence remotely.",
    )
    parser.add_argument(
        "--remote_project_root",
        type=str,
        default="/home/ksopyla/dev/MrCogito",
        help="Remote project root used to resolve relative checkpoint paths.",
    )
    parser.add_argument(
        "--skip_remote_checkpoint_check",
        action="store_true",
        help="Disable SSH checkpoint existence probing on remote servers.",
    )
    return parser.parse_args()


def is_eval_job(run: Any) -> bool:
    job_type = (run.job_type or "").lower()
    return "benchmark" in job_type or "evaluation" in job_type


def checkpoint_step_from_name(name: str) -> int | None:
    m = re.search(r"checkpoint-(\d+)", name or "")
    if not m:
        return None
    return int(m.group(1))


def normalize_training_run_id(config: dict[str, Any]) -> str | None:
    candidate = config.get("source_training_run_id") or config.get("source_run_id")
    if not candidate:
        return None
    if re.fullmatch(r"checkpoint-\d+", str(candidate)):
        return None
    return str(candidate)


def normalize_model_path_for_remote_check(model_path: str, remote_project_root: str) -> str | None:
    candidate = (model_path or "").strip()
    if not candidate:
        return None
    if "://" in candidate and not candidate.startswith("file://"):
        return None

    if candidate.startswith("file://"):
        candidate = candidate[len("file://") :]

    if candidate.startswith("~"):
        return None

    if os.path.isabs(candidate):
        return candidate

    candidate = candidate.lstrip("./")
    if candidate.startswith("Cache/"):
        return os.path.join(remote_project_root, candidate)
    return None


def remote_checkpoint_host(
    *,
    remote_path: str | None,
    hosts: list[str],
    cache: dict[str, str | None],
) -> str | None:
    if not remote_path:
        return None
    if remote_path in cache:
        return cache[remote_path]
    for host in hosts:
        if not host:
            continue
        cmd = f"test -e {shlex.quote(remote_path)}"
        try:
            result = subprocess.run(
                ["ssh", host, cmd],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0:
            cache[remote_path] = host
            return host
    cache[remote_path] = None
    return None


def _merge_tags(existing: list[str], generated: list[str]) -> list[str]:
    return list(dict.fromkeys([*(existing or []), *generated]))


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=45)
    remote_hosts = [h.strip() for h in args.remote_check_hosts.split(",") if h.strip()]
    remote_probe_cache: dict[str, str | None] = {}
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
        model_path = config.get("model_path") or config.get("source_checkpoint_path")
        if not model_path:
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "job_type": run.job_type,
                    "status": "skipped_no_model_path",
                    "updated": False,
                    "remote_path": None,
                    "remote_host": None,
                    "remote_exists": False,
                }
            )
            continue

        remote_path = normalize_model_path_for_remote_check(model_path, args.remote_project_root)
        remote_host = None
        remote_exists = False
        if not args.skip_remote_checkpoint_check:
            remote_host = remote_checkpoint_host(
                remote_path=remote_path,
                hosts=remote_hosts,
                cache=remote_probe_cache,
            )
            remote_exists = remote_host is not None

        try:
            source_checkpoint_step = config.get("source_checkpoint_step")
            if source_checkpoint_step is None:
                source_checkpoint_step = checkpoint_step_from_name(run.name)
            source_training_run_id = normalize_training_run_id(config)
            lineage = resolve_eval_lineage(
                model_path=model_path,
                source_training_run_id=source_training_run_id,
                source_training_group=config.get("source_training_group"),
                source_training_experiment_id=config.get("source_training_experiment_id"),
                source_checkpoint_step=source_checkpoint_step,
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
                    "remote_path": remote_path,
                    "remote_host": remote_host,
                    "remote_exists": remote_exists,
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
                "remote_path": remote_path,
                "remote_host": remote_host,
                "remote_exists": remote_exists,
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
                "remote_path",
                "remote_host",
                "remote_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mode = "APPLY" if args.apply else "DRY-RUN"
    remote_checked = sum(1 for row in rows if row.get("remote_path"))
    remote_found = sum(1 for row in rows if row.get("remote_exists"))
    print(
        f"[{mode}] processed={processed} report={report_path} "
        f"remote_checked={remote_checked} remote_found={remote_found}"
    )


if __name__ == "__main__":
    main()

"""Shared W&B lineage + comparability helpers for evaluation runs."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

import wandb

logger = logging.getLogger(__name__)

LINEAGE_SCHEMA_VERSION = 1
_CHECKPOINT_STEP_RE = re.compile(r"(?:^|/)checkpoint-(\d+)(?:$|/)")
_RUN_ID_RE = re.compile(r".+_\d{8}_\d{6}$")


@dataclass(frozen=True)
class EvalLineage:
    source_training_run_id: Optional[str]
    source_training_group: Optional[str]
    source_training_experiment_id: Optional[str]
    source_checkpoint_path: str
    source_checkpoint_step: Optional[int]
    source_checkpoint_epoch: Optional[float]
    lineage_status: str
    lineage_schema_version: int = LINEAGE_SCHEMA_VERSION


def _safe_tag_value(raw: str, *, max_len: int = 64) -> str:
    value = raw.strip().lower()
    value = value.replace("/", "-")
    value = re.sub(r"[^a-z0-9._:-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:max_len] or "unknown"


def parse_checkpoint_step_from_path(model_path: str) -> Optional[int]:
    match = _CHECKPOINT_STEP_RE.search(model_path or "")
    if not match:
        return None
    return int(match.group(1))


def parse_training_run_id_from_model_path(model_path: str) -> Optional[str]:
    if not model_path:
        return None
    normalized = model_path.rstrip("/")
    checkpoint_match = _CHECKPOINT_STEP_RE.search(normalized)
    if checkpoint_match:
        checkpoint_prefix = normalized[: checkpoint_match.start()].rstrip("/")
        candidate = os.path.basename(checkpoint_prefix)
        if candidate:
            return candidate

    candidate = os.path.basename(normalized)
    if _RUN_ID_RE.fullmatch(candidate):
        return candidate
    return None


def _resolve_run_metadata_from_wandb(
    *,
    entity: str,
    project: str,
    run_id: str,
) -> Dict[str, Optional[str]]:
    api = wandb.Api(timeout=30)
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        return {
            "group": run.group,
            "experiment_id": run.config.get("experiment_id"),
        }
    except Exception:
        logger.warning("Could not fetch run '%s' directly; falling back to recent-run scan.", path)

    # Fallback path for environments where run() cannot resolve display name aliases.
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=200)
        for run in runs:
            display_name = getattr(run, "display_name", None)
            if run.id == run_id or run.name == run_id or display_name == run_id:
                return {
                    "group": run.group,
                    "experiment_id": run.config.get("experiment_id"),
                }
    except Exception as exc:
        logger.warning("Fallback run scan failed while resolving lineage: %s", exc)

    return {"group": None, "experiment_id": None}


def validate_eval_lineage(lineage: EvalLineage, *, allow_unlinked_eval: bool) -> None:
    if allow_unlinked_eval:
        return
    if lineage.lineage_status != "linked":
        raise ValueError(
            "Strict lineage is enabled but this eval run is unlinked. "
            "Pass --allow_unlinked_eval only for intentional external/no-parent evaluations."
        )
    if not lineage.source_checkpoint_step and lineage.source_checkpoint_step != 0:
        raise ValueError(
            "Strict lineage requires a concrete checkpoint step. "
            "Use a checkpoint path (.../checkpoint-<step>) or pass --source_checkpoint_step."
        )


def resolve_eval_lineage(
    *,
    model_path: str,
    source_training_run_id: Optional[str],
    source_training_group: Optional[str],
    source_training_experiment_id: Optional[str],
    source_checkpoint_step: Optional[int],
    source_checkpoint_epoch: Optional[float],
    allow_unlinked_eval: bool,
    wandb_entity: str,
    wandb_project: str,
    resolve_with_wandb: bool = True,
) -> EvalLineage:
    resolved_run_id = source_training_run_id or parse_training_run_id_from_model_path(model_path)
    resolved_group = source_training_group
    resolved_experiment_id = source_training_experiment_id
    resolved_step = source_checkpoint_step
    if resolved_step is None:
        resolved_step = parse_checkpoint_step_from_path(model_path)

    if resolve_with_wandb and resolved_run_id and not resolved_group:
        metadata = _resolve_run_metadata_from_wandb(
            entity=wandb_entity,
            project=wandb_project,
            run_id=resolved_run_id,
        )
        resolved_group = resolved_group or metadata.get("group")
        resolved_experiment_id = resolved_experiment_id or metadata.get("experiment_id")

    status = "linked" if resolved_run_id and resolved_group else "unlinked"
    lineage = EvalLineage(
        source_training_run_id=resolved_run_id,
        source_training_group=resolved_group,
        source_training_experiment_id=resolved_experiment_id,
        source_checkpoint_path=model_path,
        source_checkpoint_step=resolved_step,
        source_checkpoint_epoch=source_checkpoint_epoch,
        lineage_status=status,
    )
    validate_eval_lineage(lineage, allow_unlinked_eval=allow_unlinked_eval)
    return lineage


def build_eval_compare_fields(
    *,
    model_family: str,
    params_m: int,
    objective_family: Optional[str],
    tokenizer_name: str,
    architecture_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "compare_family": model_family,
        "compare_params_m": params_m,
        "compare_objective": objective_family,
        "compare_tokenizer": tokenizer_name,
        "compare_architecture": architecture_id,
    }


def build_namespaced_eval_tags(
    *,
    benchmark: str,
    model_family: str,
    params_m: int,
    tokenizer_name: str,
    lineage: EvalLineage,
    objective_family: Optional[str] = None,
    extra_tags: Optional[Iterable[str]] = None,
) -> list[str]:
    tags = list(extra_tags or [])
    tags.extend(
        [
            "job:eval",
            f"benchmark:{_safe_tag_value(benchmark)}",
            f"family:{_safe_tag_value(model_family)}",
            f"size:{params_m}M",
            f"tokenizer:{_safe_tag_value(tokenizer_name)}",
            f"lineage:{lineage.lineage_status}",
        ]
    )
    if objective_family:
        tags.append(f"objective:{_safe_tag_value(objective_family)}")
    if lineage.source_training_experiment_id:
        tags.append(f"exp:{_safe_tag_value(lineage.source_training_experiment_id)}")
    if lineage.source_checkpoint_step is not None:
        tags.append(f"ckpt_step:{lineage.source_checkpoint_step}")
    if lineage.source_checkpoint_epoch is not None:
        tags.append(f"ckpt_epoch:{lineage.source_checkpoint_epoch:g}")
    return list(dict.fromkeys(tags))


def lineage_to_wandb_config(lineage: EvalLineage) -> Dict[str, Any]:
    config = asdict(lineage)
    # Legacy compatibility: old dashboards/scripts expected `source_run_id`.
    config["source_run_id"] = lineage.source_training_run_id
    if lineage.source_checkpoint_step is not None:
        config["source_checkpoint_id"] = f"checkpoint-{lineage.source_checkpoint_step}"
    else:
        config["source_checkpoint_id"] = None
    return config

"""Post-hoc compute audit for W&B training runs.

Reads already-logged W&B system metrics + config and computes, per run:
  * ``compute/gpu_hours``       — runtime x world_size / 3600
  * ``compute/energy_kwh``      — trapezoidal integral of per-GPU powerWatts over time,
                                  summed across the world_size GPUs (kWh)
  * ``compute/max_tokens``      — global_step x grad_accum x pbs x world_size x max_seq_length
                                  (positions processed, objective-agnostic upper bound)
  * ``compute/loss_tokens_est`` — max_tokens x per-family loss fraction (flagged approximate)
  * derived ratios: tokens_per_gpu_hour, energy_per_gpu_hour_kw,
                    gpu_hours_per_billion_tokens, energy_per_billion_tokens
  * ``compute/audit_state`` (finished | running-partial | flagged | failed) and ``compute/flag``

For finished runs the scalars are written back into the run's W&B summary so a
native W&B custom panel can render them (group by ``wandb_group``). For running
runs the local artifact is emitted but summary write-back is deferred (the live
wandb process can drop keys it did not log).

No GPU, no checkpoint, no training-loop change. See the engineering spec:
``docs/engineering_specs/compute_audit_wandb_panel.md``.

Usage:
    uv run python analysis/run_compute_audit.py \\
        --run-id concept_ar_prefix_H768L6C128D4_20260614_101305 \\
        --run-id concept_ar_prefix_H768L6C128D4_20260627_192407 \\
        [--group <wandb_group>] [--tag <tag>] \\
        --entity ksopyla --project MrCogito \\
        --out-dir Cache/Evaluation_reports/compute_audit/ \\
        [--dry-run]
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np

load_dotenv()

import wandb  # noqa: E402

ENTITY_DEFAULT = "ksopyla"
PROJECT_DEFAULT = "MrCogito"

# Energy integration
SAMPLES_CAP = 1_000_000        # full-res system history (default 500 downsamples)
POWER_GAP_MAX_S = 60.0         # split the integral across gaps larger than this
WATTS_S_TO_KWH = 3.6e6         # 1 kWh = 3.6e6 W*s

# Plausibility gate thresholds (RTX 3090: idle ~30 W, training ~150-320 W, TDP 350 W)
POWER_FLOOR_W = 80.0
POWER_TDP_W = 350.0            # hard ceiling sanity (enforcedPowerLimitWatts is preferred)
TRAPEZOID_AVG_TOL = 0.05       # |energy - avg_power*runtime| / energy
GPUH_TS_TOL = 0.01             # |gpu_h_summary - gpu_h_ts_span| / gpu_h_summary

# Regexes
GPU_POWER_RE = re.compile(r"^system\.gpu\.(\d+)\.powerWatts$")
GPU_PLIMIT_RE = re.compile(r"^system\.gpu\.(\d+)\.enforcedPowerLimitWatts$")
NUM_PROC_RE = re.compile(r"Num processes:\s*(\d+)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def trapezoid_energy(timestamps: np.ndarray, power: np.ndarray) -> Tuple[float, float]:
    """Trapezoidal integral of power over time, splitting across gaps.

    Args:
        timestamps: 1D array of unix timestamps (seconds), sorted ascending.
        power: 1D array of power values in Watts, same length, NaN-free.

    Returns:
        (energy_J, active_duration_s) where energy is in W*s (Joules) and
        active_duration is the sum of inter-sample dt that were not split out
        as gaps (dt <= 0 or dt > POWER_GAP_MAX_S).
    """
    if len(timestamps) != len(power) or len(timestamps) < 2:
        return 0.0, 0.0
    dt = np.diff(timestamps)
    energy = 0.0
    active = 0.0
    seg_start = 0
    for i in range(len(dt)):
        bad = dt[i] <= 0 or dt[i] > POWER_GAP_MAX_S
        if bad:
            e, a = _trap_segment(power[seg_start:i + 1], dt[seg_start:i])
            energy += e
            active += a
            seg_start = i + 1
    e, a = _trap_segment(power[seg_start:], dt[seg_start:] if seg_start < len(dt) else np.array([]))
    energy += e
    active += a
    return float(energy), float(active)


def _trap_segment(p: np.ndarray, dt: np.ndarray) -> Tuple[float, float]:
    if len(p) < 2 or len(dt) == 0:
        return 0.0, 0.0
    return float(np.sum((p[:-1] + p[1:]) / 2.0 * dt)), float(np.sum(dt))


def parse_world_size(config: Dict[str, Any]) -> Optional[int]:
    """Extract world_size from the HF Trainer `distributed_state` config string."""
    ds = config.get("distributed_state")
    if not isinstance(ds, str):
        return None
    m = NUM_PROC_RE.search(ds)
    return int(m.group(1)) if m else None


def cfg_get(config: Dict[str, Any], key: str) -> Any:
    """Unwrap the `{"value": v}` form if present (wandb Api normally unwraps already)."""
    v = config.get(key)
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return v["value"]
    return v


def compute_max_tokens(global_step: int, grad_accum: int, pbs: int,
                       world_size: int, seq_len: int) -> int:
    """Positions processed = optimizer steps x grad_accum x per-device batch x world x seq_len."""
    return int(global_step) * int(grad_accum) * int(pbs) * int(world_size) * int(seq_len)


def infer_family_from_name(run_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback family/objective inference from the run-name prefix.

    Older runs (pre `build_perceiver_wandb_identity`) may lack `model_family` /
    `objective_family` config keys; their run name still starts with the family.
    Order matters: `concept_ar_prefix` before `concept_ar`.
    """
    if run_name.startswith("concept_ar_prefix"):
        return "concept_ar_prefix", "prefix_suffix"
    if run_name.startswith("concept_ar"):
        return "concept_ar", "reconstruction"
    if run_name.startswith("perceiver_denoise"):
        return "perceiver_denoise", "reconstruction"
    if run_name.startswith("weighted_mlm"):
        return "weighted_mlm", "mlm"
    if run_name.startswith("diffusion_mlm"):
        return "diffusion_mlm", "reconstruction"
    if run_name.startswith("prefix_diffusion"):
        return "prefix_diffusion", "reconstruction"
    if run_name.startswith("recursive_mlm"):
        return "recursive_mlm", "mlm"
    return None, None


def arch_prefix_from_name(run_name: str) -> Optional[str]:
    """Strip the trailing `_YYYYMMDD_HHMMSS` timestamp -> architecture prefix."""
    m = re.match(r"^(.+)_\d{8}_\d{6}$", run_name)
    return m.group(1) if m else None


def loss_fraction(config: Dict[str, Any], run_name: str = "") -> Tuple[Optional[float], Optional[str]]:
    """Per-family fraction of positions that are loss targets (approximate).

    Returns (fraction, flag) where flag is None when exact, or a string reason
    when the fraction is unknown/approximate (informational, not a plausibility
    warning — see ``is_plausibility_flag``).
    """
    family = cfg_get(config, "model_family")
    obj_family = cfg_get(config, "objective_family")
    obj_variant = cfg_get(config, "objective_variant")
    # Fallback for older runs without the identity config keys.
    if family is None:
        fam_inferred, obj_inferred = infer_family_from_name(run_name)
        family = fam_inferred
        if obj_family is None:
            obj_family = obj_inferred

    is_prefix = (obj_family == "prefix_suffix"
                 or (isinstance(obj_variant, str) and obj_variant == "prefix_suffix")
                 or family == "concept_ar_prefix")
    if is_prefix:
        pmin = cfg_get(config, "prefix_ratio_min")
        pmax = cfg_get(config, "prefix_ratio_max")
        if pmin is None or pmax is None:
            return None, "loss_fraction:prefix_ratio_missing"
        frac = 1.0 - (float(pmin) + float(pmax)) / 2.0
        return frac, "loss_fraction:prefix_suffix_approx"

    is_recon = (obj_family == "reconstruction"
                or (isinstance(obj_variant, str) and obj_variant.startswith("reconstruction"))
                or family == "perceiver_denoise"
                or family == "concept_ar")
    if is_recon:
        # TSDAE/denoising reconstruction predicts the full clean sequence -> loss on ~all positions.
        return 1.0, "loss_fraction:reconstruction_approx"

    if family == "weighted_mlm":
        for k in ("mlm_probability", "masking_probability", "masking_rate"):
            rate = cfg_get(config, k)
            if rate is not None:
                return float(rate), "loss_fraction:weighted_mlm_approx"
        return None, "loss_fraction:weighted_mlm_no_rate"

    return None, "loss_fraction:unknown"


def is_plausibility_flag(flag: str) -> bool:
    """True for flags that mark a computed number as suspect (-> audit_state=flagged).

    Informational flags (loss_fraction:*, writeback_*) are recorded in
    ``compute/flag`` but do not alone mark the run flagged.
    """
    if re.match(r"^gpu\d+:", flag):
        return True
    return flag.startswith(("energy:retrieval_unsupported",
                            "energy:trapezoid_vs_avg",
                            "gpu_hours:summary_vs_ts"))


# ---------------------------------------------------------------------------
# W&B retrieval
# ---------------------------------------------------------------------------

def resolve_run(api, entity: str, project: str, name: str):
    """Resolve a run by id/name. In this project run.id == the timestamped name."""
    path = f"{entity}/{project}/{name}"
    try:
        return api.run(path)
    except Exception:
        pass
    proj = f"{entity}/{project}"
    for filt in ({"displayName": {"$eq": name}}, {"name": {"$eq": name}}):
        runs = list(api.runs(proj, filters=filt))
        if runs:
            return runs[0]
    raise ValueError(f"could not resolve run {name!r} in {proj}")


def load_power_series(run) -> Tuple[Optional[Any], List[int], Dict[int, str], Dict[int, str]]:
    """Full-res system history DataFrame + GPU index sets.

    Returns (df, gpu_indices, power_col_by_gpu, plimit_col_by_gpu) or
    (None, [], {}, {}) if the system stream could not be retrieved.
    """
    try:
        df = run.history(stream="system", samples=SAMPLES_CAP)
    except Exception:
        return None, [], {}, {}
    if df is None or len(df) == 0 or "_timestamp" not in df.columns:
        return None, [], {}, {}

    power_col: Dict[int, str] = {}
    plimit_col: Dict[int, str] = {}
    for col in df.columns:
        m = GPU_POWER_RE.match(str(col))
        if m:
            power_col[int(m.group(1))] = str(col)
            continue
        m = GPU_PLIMIT_RE.match(str(col))
        if m:
            plimit_col[int(m.group(1))] = str(col)
    gpus = sorted(power_col.keys())
    return df, gpus, power_col, plimit_col


def get_runtime(summary: Dict[str, Any], state: str) -> Tuple[Optional[float], Optional[str]]:
    """Prefer train_runtime for finished runs; fall back to _runtime (live for running)."""
    for key in ("train/train_runtime", "train_runtime"):
        v = summary.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), key
    v = summary.get("_runtime")
    if isinstance(v, (int, float)) and v > 0:
        return float(v), "_runtime"
    return None, None


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_run(run, write_back: bool) -> Dict[str, Any]:
    """Compute the full compute-audit record for one W&B run."""
    name = run.name
    state = run.state
    summary = dict(run.summary)
    config = dict(run.config)

    family_cfg = cfg_get(config, "model_family")
    obj_family_cfg = cfg_get(config, "objective_family")
    if family_cfg is None:
        fam_inf, obj_inf = infer_family_from_name(name)
        family_cfg = family_cfg or fam_inf
        obj_family_cfg = obj_family_cfg or obj_inf
    wandb_group = cfg_get(config, "wandb_group")
    group_for_panel = wandb_group or arch_prefix_from_name(name)

    rec: Dict[str, Any] = {
        "run_name": name,
        "run_id": run.id,
        "state": state,
        "wandb_group": wandb_group,
        "group_for_panel": group_for_panel,
        "model_family": family_cfg,
        "objective_family": obj_family_cfg,
        "objective_variant": cfg_get(config, "objective_variant"),
        "dataset_name": cfg_get(config, "dataset_name"),
        "git_commit": cfg_get(config, "git_commit"),
        "max_seq_length": cfg_get(config, "max_seq_length"),
        "num_train_epochs": cfg_get(config, "num_train_epochs"),
        "url": run.url,
        "scalars": {},
        "flags": [],
        "gate_failed_structural": False,
        "error": None,
    }

    # --- world_size ---
    world_size = parse_world_size(config)

    # --- runtime ---
    runtime_s, runtime_src = get_runtime(summary, state)

    # --- global_step ---
    global_step = summary.get("global_step")
    if global_step is None:
        global_step = summary.get("train/global_step")

    # --- config knobs for token math ---
    pbs = cfg_get(config, "per_device_train_batch_size")
    grad_accum = cfg_get(config, "gradient_accumulation_steps")
    seq_len = cfg_get(config, "max_seq_length")
    if seq_len is None:
        seq_len = cfg_get(config, "max_sequence_length")

    # --- structural gates ---
    missing = []
    if world_size is None:
        missing.append("world_size")
    if runtime_s is None:
        missing.append("runtime")
    if global_step is None:
        missing.append("global_step")
    for k, v in (("per_device_train_batch_size", pbs),
                 ("gradient_accumulation_steps", grad_accum),
                 ("max_seq_length", seq_len)):
        if v is None:
            missing.append(k)

    # GPU power series + gpu-count cross-check
    df, gpus, power_col, plimit_col = load_power_series(run)
    gpu_count = len(gpus)
    energy_kwh: Optional[float] = None
    per_gpu_stats: List[Dict[str, Any]] = []
    ts_span_s: Optional[float] = None

    if df is not None and gpu_count > 0:
        ts = df["_timestamp"].to_numpy(dtype=float)
        finite_ts = ts[np.isfinite(ts)]
        if len(finite_ts) >= 2:
            ts_span_s = float(finite_ts.max() - finite_ts.min())
        total_energy_J = 0.0
        total_avg_power_W = 0.0
        for gi in gpus:
            col = power_col[gi]
            arr = df[col].to_numpy(dtype=float)
            mask = np.isfinite(arr) & np.isfinite(ts)
            t = ts[mask]
            p = arr[mask]
            order = np.argsort(t)
            t = t[order]
            p = p[order]
            if len(t) < 2:
                continue
            e_J, active_s = trapezoid_energy(t, p)
            avg_W = float(np.mean(p))
            total_energy_J += e_J
            total_avg_power_W += avg_W
            plimit = None
            if gi in plimit_col:
                pl = df[plimit_col[gi]].to_numpy(dtype=float)
                pl_finite = pl[np.isfinite(pl)]
                if len(pl_finite):
                    plimit = float(pl_finite.max())
            per_gpu_stats.append({
                "gpu": gi, "n_samples": int(len(t)),
                "avg_power_W": avg_W, "energy_J": e_J,
                "active_duration_s": active_s,
                "enforced_power_limit_W": plimit,
            })
        if total_energy_J > 0:
            energy_kwh = total_energy_J / WATTS_S_TO_KWH
        rec["per_gpu"] = per_gpu_stats
        rec["ts_span_s"] = ts_span_s
        # gpu-count cross-check
        if world_size is not None and gpu_count != world_size:
            missing.append(f"gpu_count_mismatch:{gpu_count}_vs_{world_size}")
    else:
        rec["flags"].append("energy:retrieval_unsupported")

    if missing:
        rec["gate_failed_structural"] = True
        rec["error"] = "structural:" + ",".join(missing)
        rec["scalars"] = {
            "compute/audit_state": "failed",
            "compute/flag": missing,
        }
        return rec

    # --- compute scalars ---
    gpu_hours = runtime_s * world_size / 3600.0
    max_tokens = compute_max_tokens(int(global_step), int(grad_accum),
                                    int(pbs), int(world_size), int(seq_len))
    frac, frac_flag = loss_fraction(config, name)
    loss_tokens_est = int(max_tokens * frac) if frac is not None else None
    if frac_flag:
        rec["flags"].append(frac_flag)

    tokens_per_gpu_hour = max_tokens / gpu_hours if gpu_hours > 0 else None
    energy_per_gpu_hour_kw = energy_kwh / gpu_hours if (energy_kwh is not None and gpu_hours > 0) else None
    gpu_hours_per_billion = gpu_hours / (max_tokens / 1e9) if max_tokens > 0 else None
    energy_per_billion = energy_kwh / (max_tokens / 1e9) if (energy_kwh is not None and max_tokens > 0) else None

    scalars: Dict[str, Any] = {
        "compute/gpu_hours": gpu_hours,
        "compute/energy_kwh": energy_kwh,
        "compute/max_tokens": max_tokens,
        "compute/max_tokens_b": (max_tokens / 1e9) if max_tokens else None,
        "compute/loss_tokens_est": loss_tokens_est,
        "compute/tokens_per_gpu_hour": tokens_per_gpu_hour,
        "compute/energy_per_gpu_hour_kw": energy_per_gpu_hour_kw,
        "compute/gpu_hours_per_billion_tokens": gpu_hours_per_billion,
        "compute/energy_per_billion_tokens": energy_per_billion,
        "compute/world_size": world_size,
        "compute/group_for_panel": group_for_panel,
        "compute/runtime_source": runtime_src,
    }

    # --- plausibility gates ---
    for gs in per_gpu_stats:
        plimit = gs["enforced_power_limit_W"] or POWER_TDP_W
        avg = gs["avg_power_W"]
        if avg < POWER_FLOOR_W:
            rec["flags"].append(f"gpu{gs['gpu']}:avg_power_low:{avg:.1f}W")
        if avg > plimit + 1.0:
            rec["flags"].append(f"gpu{gs['gpu']}:avg_power_over_limit:{avg:.1f}>{plimit:.1f}W")

    if energy_kwh is not None and runtime_s is not None and total_avg_power_W > 0:
        approx = total_avg_power_W * runtime_s / WATTS_S_TO_KWH
        if abs(approx - energy_kwh) / energy_kwh >= TRAPEZOID_AVG_TOL:
            rec["flags"].append(
                f"energy:trapezoid_vs_avg:{abs(approx - energy_kwh) / energy_kwh:.3f}"
            )

    if ts_span_s is not None and runtime_s is not None and ts_span_s > 0:
        gpu_h_ts = ts_span_s * world_size / 3600.0
        if abs(gpu_h_ts - gpu_hours) / gpu_hours >= GPUH_TS_TOL:
            rec["flags"].append(
                f"gpu_hours:summary_vs_ts:{abs(gpu_h_ts - gpu_hours) / gpu_hours:.4f}"
            )

    # --- state ---
    if state == "running":
        audit_state = "running-partial"
    elif any(is_plausibility_flag(f) for f in rec["flags"]):
        audit_state = "flagged"
    else:
        audit_state = "finished"
    scalars["compute/audit_state"] = audit_state
    scalars["compute/flag"] = rec["flags"]
    rec["scalars"] = scalars
    rec["audit_state"] = audit_state

    # --- summary write-back (finished runs only, unless dry-run) ---
    rec["writeback"] = False
    if write_back and state != "running":
        try:
            for k, v in scalars.items():
                if v is not None:
                    run.summary[k] = v
            # Best-effort cleanup of stale cohort-relative _pct keys from a prior
            # audit version. Done AFTER the base write so a cleanup failure can't
            # block the real scalars; per-key so one missing key is harmless.
            for stale in ("compute/gpu_hours_pct", "compute/energy_kwh_pct",
                          "compute/max_tokens_pct"):
                try:
                    del run.summary[stale]
                except Exception:
                    pass
            run.summary.update()
            rec["writeback"] = True
        except Exception as e:
            rec["flags"].append(f"writeback_failed:{e!r}")
    elif state == "running":
        rec["flags"].append("writeback_deferred:running")

    return rec


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_name", "state", "audit_state", "wandb_group", "group_for_panel",
    "model_family", "objective_family", "dataset_name", "world_size",
    "max_seq_length", "num_train_epochs", "git_commit", "url",
    "compute/gpu_hours", "compute/energy_kwh", "compute/max_tokens",
    "compute/max_tokens_b", "compute/loss_tokens_est",
    "compute/tokens_per_gpu_hour", "compute/energy_per_gpu_hour_kw",
    "compute/gpu_hours_per_billion_tokens", "compute/energy_per_billion_tokens",
    "compute/runtime_source", "compute/flag", "error",
]


def emit_csv(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for rec in records:
            sc = rec.get("scalars", {})
            cfg = {}
            # pull config tags from the record (audit_run stored them at top level)
            row = {c: "" for c in CSV_COLUMNS}
            row["run_name"] = rec.get("run_name", "")
            row["state"] = rec.get("state", "")
            row["audit_state"] = sc.get("compute/audit_state", rec.get("audit_state", ""))
            row["wandb_group"] = rec.get("wandb_group") or ""
            row["group_for_panel"] = rec.get("group_for_panel") or ""
            row["model_family"] = rec.get("model_family") or ""
            row["objective_family"] = rec.get("objective_family") or ""
            row["dataset_name"] = rec.get("dataset_name") or ""
            row["world_size"] = sc.get("compute/world_size", "")
            row["max_seq_length"] = rec.get("max_seq_length") or ""
            row["num_train_epochs"] = rec.get("num_train_epochs") or ""
            row["git_commit"] = rec.get("git_commit") or ""
            row["url"] = rec.get("url", "")
            for k in ("compute/gpu_hours", "compute/energy_kwh", "compute/max_tokens",
                      "compute/max_tokens_b", "compute/loss_tokens_est",
                      "compute/tokens_per_gpu_hour",
                      "compute/energy_per_gpu_hour_kw",
                      "compute/gpu_hours_per_billion_tokens",
                      "compute/energy_per_billion_tokens",
                      "compute/runtime_source"):
                v = sc.get(k)
                row[k] = "" if v is None else v
            row["compute/flag"] = ";".join(rec.get("flags", []))
            row["error"] = rec.get("error") or ""
            w.writerow([row[c] for c in CSV_COLUMNS])


def emit_chart(records: List[Dict[str, Any]], png_path: str) -> None:
    """Matplotlib grouped bars: raw panel + ratios panel, colored by wandb_group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in records if not r.get("gate_failed_structural")]
    if not ok:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no successful runs to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    names = [r["run_name"] for r in ok]
    groups = [r.get("wandb_group") or "" for r in ok]
    uniq_groups = sorted(set(groups))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(uniq_groups.index(g) % 10) for g in groups]

    def vals(key):
        return [r["scalars"].get(key) if r.get("scalars") else None for r in ok]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    raw = [("compute/gpu_hours", "GPU-hours"),
           ("compute/energy_kwh", "Energy (kWh)"),
           ("compute/max_tokens", "Max tokens (B)")]
    for ax, (key, title) in zip(axes[0], raw):
        v = vals(key)
        v_plot = [x / 1e9 if key == "compute/max_tokens" and x is not None else x for x in v]
        bars = ax.bar(range(len(names)), v_plot, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        for b, x in zip(bars, v_plot):
            if x is not None:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{x:.2f}" if x < 100 else f"{x:.1f}",
                        ha="center", va="bottom", fontsize=7)

    ratios = [("compute/tokens_per_gpu_hour", "Tokens / GPU-h (M)"),
              ("compute/energy_per_gpu_hour_kw", "Energy / GPU-h (kW)"),
              ("compute/gpu_hours_per_billion_tokens", "GPU-h / B-tok")]
    for ax, (key, title) in zip(axes[1], ratios):
        v = vals(key)
        v_plot = [x / 1e6 if key == "compute/tokens_per_gpu_hour" and x is not None else x for x in v]
        ax.bar(range(len(names)), v_plot, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(uniq_groups.index(g) % 10))
                      for g in uniq_groups]
    fig.legend(legend_handles, uniq_groups, loc="upper center",
               ncol=min(len(uniq_groups), 4), fontsize=8)
    fig.suptitle("Compute audit comparison (grouped by wandb_group)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def emit_profile_chart(records: List[Dict[str, Any]], png_path: str) -> None:
    """Grouped 'compute profile' bar chart: the three absolute headline metrics per
    run, with ``max_tokens`` rescaled to billions so it shares the GPU-hours / energy
    range (raw tokens are ~1e9 and would dominate a shared linear axis). Stable
    absolute values — comparable across past and future runs. Mirrors the W&B
    grouped panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in records
          if not r.get("gate_failed_structural")
          and r.get("scalars", {}).get("compute/gpu_hours") is not None]
    if not ok:
        return
    names = [r["run_name"] for r in ok]
    # max_tokens rescaled to billions; gpu_hours and energy_kwh in their native units.
    metrics = [("compute/gpu_hours", "GPU-hours", lambda x: x),
               ("compute/energy_kwh", "Energy (kWh)", lambda x: x),
               ("compute/max_tokens", "Max tokens (B)", lambda x: x / 1e9)]
    n = len(names)
    m = len(metrics)
    width = 0.8 / m
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * n), 5))
    cmap = plt.get_cmap("tab10")
    for j, (key, label, scale) in enumerate(metrics):
        raw = [r["scalars"].get(key) for r in ok]
        v = [scale(r) if r is not None else None for r in raw]
        bars = ax.bar(x + (j - (m - 1) / 2) * width, v, width, label=label,
                      color=cmap(j))
        for b, val in zip(bars, v):
            if val is not None:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GPU-hours · kWh · tokens (B)  [absolute, native units]")
    ax.set_title("Compute profile (absolute; max_tokens in billions)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def expand_run_ids(api, entity: str, project: str,
                   run_ids: List[str], group: Optional[str],
                   tag: Optional[str]) -> List[str]:
    ids = list(run_ids or [])
    if group or tag:
        filt: Dict[str, Any] = {}
        if group:
            filt["config.wandb_group.value"] = group
        if tag:
            filt["tags"] = {"$in": [tag]}
        proj = f"{entity}/{project}"
        for r in api.runs(proj, filters=filt):
            ids.append(r.name)
    # de-dup, preserve order
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Dashboard run: long-form wandb.Table for the Custom Chart grouped panel
# ---------------------------------------------------------------------------
#
# The W&B built-in Bar Chart cannot cluster N metrics per run, and a Custom
# Chart with a Vega-Lite ``fold`` transform over summary scalars is fragile
# (the ``${field:...}`` substitution often fails inside ``fold``). The robust
# path (the one W&B's own docs use for grouped bars) is to log a long-form
# ``wandb.Table`` — one row per (run, metric) — to a dedicated dashboard run,
# then point a Custom Chart (Vega grouped-bar preset) at it.

DASHBOARD_RUN_NAME = "compute_profile_dashboard"
TABLE_KEY = "compute_profile"
PROFILE_METRICS = (
    ("GPU-hours", "compute/gpu_hours"),
    ("Energy (kWh)", "compute/energy_kwh"),
    ("Max tokens (B)", "compute/max_tokens_b"),
)


def log_profile_table(records: List[Dict[str, Any]],
                      entity: str, project: str,
                      dashboard_name: str = DASHBOARD_RUN_NAME) -> str:
    """Log a long-form ``compute_profile`` wandb.Table to a dashboard run.

    Columns: run_name, wandb_group, metric, value, audit_state.
    The dashboard run is tagged ``compute-dashboard``; re-running overwrites the
    table (so the panel always reflects the latest audit). A Custom Chart in the
    workspace with the grouped-bar preset + ``summaryTable: compute_profile``
    renders the 3-metrics-per-run cluster directly.
    """
    rows: List[List[Any]] = []
    for rec in records:
        if rec.get("gate_failed_structural"):
            continue
        sc = rec.get("scalars", {})
        run_name = rec.get("run_name", "")
        group = rec.get("group_for_panel") or rec.get("wandb_group") or ""
        state = sc.get("compute/audit_state", "")
        for label, key in PROFILE_METRICS:
            v = sc.get(key)
            if v is None:
                continue
            rows.append([run_name, group, label, float(v), state])

    run = wandb.init(
        project=project, entity=entity,
        name=dashboard_name, id=dashboard_name,
        resume="allow", tags=["compute-dashboard"], group=None,
        config={"kind": "compute_profile_dashboard"},
    )
    table = wandb.Table(
        columns=["run_name", "wandb_group", "metric", "value", "audit_state"],
        data=rows,
    )
    wandb.log({TABLE_KEY: table})
    url = run.url
    run.finish()
    return url


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Post-hoc compute audit for W&B training runs.")
    p.add_argument("--run-id", action="append", default=[], dest="run_ids")
    p.add_argument("--group", default=None, help="Expand to all runs in this wandb_group.")
    p.add_argument("--tag", default=None, help="Expand to all runs with this tag.")
    p.add_argument("--entity", default=ENTITY_DEFAULT)
    p.add_argument("--project", default=PROJECT_DEFAULT)
    p.add_argument("--out-dir", default="Cache/Evaluation_reports/compute_audit/")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + local artifact only; no W&B summary write-back.")
    p.add_argument("--log-table", action="store_true",
                   help="Log a long-form compute_profile wandb.Table to a dashboard run "
                        "(compute_profile_dashboard) so a Custom Chart grouped-bar panel "
                        "renders the 3 metrics per run. Use when the built-in bar chart "
                        "can't cluster metrics per run.")
    args = p.parse_args(argv)

    api = wandb.Api()
    run_ids = expand_run_ids(api, args.entity, args.project,
                             args.run_ids, args.group, args.tag)
    if not run_ids:
        print("No runs to audit. Pass --run-id, --group, or --tag.")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    records: List[Dict[str, Any]] = []
    for name in run_ids:
        print(f"\n=== {name} ===")
        try:
            run = resolve_run(api, args.entity, args.project, name)
        except Exception as e:
            print(f"  RESOLVE FAILED: {e!r}")
            records.append({"run_name": name, "state": "unknown",
                            "scalars": {"compute/audit_state": "failed"},
                            "flags": [f"resolve_failed:{e!r}"],
                            "gate_failed_structural": True,
                            "error": f"resolve_failed:{e!r}",
                            "writeback": False})
            continue
        rec = audit_run(run, write_back=not args.dry_run)
        records.append(rec)
        sc = rec.get("scalars", {})
        if rec.get("gate_failed_structural"):
            print(f"  STRUCTURAL FAIL: {rec.get('error')}")
        else:
            print(f"  state={rec.get('audit_state')}  gpu_h={sc.get('compute/gpu_hours')}")
            print(f"  energy_kwh={sc.get('compute/energy_kwh')}  "
                  f"max_tokens={sc.get('compute/max_tokens')}  "
                  f"loss_tokens_est={sc.get('compute/loss_tokens_est')}")
        if rec.get("flags"):
            print(f"  flags: {rec['flags']}")
        if rec.get("writeback"):
            print("  summary write-back: OK")
        if run.state == "running":
            print("  NOTE: run is still running — re-run this script after it finishes "
                  "to persist summary scalars.")

    csv_path = os.path.join(args.out_dir, f"{stamp}_summary.csv")
    png_path = os.path.join(args.out_dir, f"{stamp}_comparison.png")
    profile_path = os.path.join(args.out_dir, f"{stamp}_profile.png")
    json_path = os.path.join(args.out_dir, f"{stamp}_per_run.json")
    emit_csv(records, csv_path)
    emit_chart(records, png_path)
    emit_profile_chart(records, profile_path)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nWrote:\n  {csv_path}\n  {png_path}\n  {profile_path}\n  {json_path}")

    if args.log_table:
        try:
            url = log_profile_table(records, args.entity, args.project)
            print(f"\nLogged compute_profile wandb.Table to dashboard run:\n  {url}")
            print(f"Table key: '{TABLE_KEY}'. In the W&B workspace, add a Custom Chart, "
                  "pick the 'Grouped bar chart' preset, set summaryTable to "
                  f"'{TABLE_KEY}', X=run_name, Group=metric, Y=value.")
        except Exception as e:
            print(f"\n--log-table failed: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

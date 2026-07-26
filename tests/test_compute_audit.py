"""Tests for analysis/run_compute_audit.py.

Covers the synthetic integrator falsification anchor, gate logic, token math,
per-family loss fraction, the run-name fallback for older runs, and the
running-run write-back deferral. The wandb.Api run object is faked so the suite
runs offline without network.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.run_compute_audit import (
    WATTS_S_TO_KWH,
    arch_prefix_from_name,
    audit_run,
    cfg_get,
    compute_max_tokens,
    infer_family_from_name,
    is_plausibility_flag,
    loss_fraction,
    parse_world_size,
    trapezoid_energy,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSummary(dict):
    """dict that records item assignment (mirrors wandb SummarySubDict writes)."""

    def update(self, *a, **k):  # noqa: D401 - mimic wandb summary persistence
        dict.update(self, *a, **k)


class FakeRun:
    def __init__(self, name, state, summary, config, df):
        self.name = name
        self.id = name
        self.state = state
        self.summary = FakeSummary(summary)
        self.config = config
        self.url = "https://wandb.ai/x/y/runs/" + name
        self._df = df

    def history(self, stream="default", samples=500, **_):
        return self._df


def make_power_df(duration_s, dt_s, ngpu, power_w, start_ts=1_000_000.0, plimit=350.0):
    n = int(duration_s / dt_s) + 1
    ts = start_ts + np.arange(n) * dt_s
    data = {"_timestamp": ts}
    for i in range(ngpu):
        data["system.gpu.%d.powerWatts" % i] = np.full(n, float(power_w))
        data["system.gpu.%d.enforcedPowerLimitWatts" % i] = np.full(n, float(plimit))
    return pd.DataFrame(data)


def prefix_suffix_cfg(world=4, pbs=40, grad_accum=1, seq=512, epochs=5):
    return {
        "model_family": "concept_ar_prefix",
        "objective_family": "prefix_suffix",
        "objective_variant": "prefix_suffix",
        "wandb_group": "E02_concept_ar_prefix_H768L6C128D4",
        "per_device_train_batch_size": pbs,
        "gradient_accumulation_steps": grad_accum,
        "max_seq_length": seq,
        "num_train_epochs": epochs,
        "prefix_ratio_min": 0.35,
        "prefix_ratio_max": 0.45,
        "distributed_state": "Distributed environment: ... Num processes: %d\n" % world,
        "dataset_name": "ds",
        "git_commit": "abc1234",
    }


# ---------------------------------------------------------------------------
# Synthetic integrator falsification anchor
# ---------------------------------------------------------------------------

def test_trapezoid_constant_power_exact():
    ts = np.arange(0, 1001) * 10.0          # 0..10000 s, dt=10
    p = np.full_like(ts, 200.0)
    energy_J, active = trapezoid_energy(ts, p)
    # constant 200 W over 10000 s -> 2,000,000 J exactly (trapezoid is exact for constants)
    assert abs(energy_J - 2_000_000.0) < 1e-6
    assert abs(active - 10_000.0) < 1e-6


def test_trapezoid_kwh_conversion():
    ts = np.arange(0, 3601) * 1.0           # 3600 s
    p = np.full_like(ts, 250.0)
    energy_J, _ = trapezoid_energy(ts, p)
    # 250 W * 3600 s = 900,000 J = 0.25 kWh
    assert abs(energy_J / WATTS_S_TO_KWH - 0.25) < 1e-9


def test_trapezoid_linear_ramp_exact():
    # linear ramp 0->100 W over 100 s; trapezoid is exact for linear functions
    ts = np.arange(0, 101, 10.0)
    p = ts.copy()                            # 0,10,...,100 W
    energy_J, _ = trapezoid_energy(ts, p)
    # integral of p=t over [0,100] = 0.5*100*100 = 5000 J
    assert abs(energy_J - 5000.0) < 1e-6


def test_trapezoid_gap_splits_not_bridges():
    # two 100 s segments at 100 W, separated by a 100 s gap (> 60 s threshold)
    ts = np.concatenate([np.arange(0, 101, 10.0), np.arange(200, 301, 10.0)])
    p = np.full_like(ts, 100.0)
    energy_J, active = trapezoid_energy(ts, p)
    # 2 segments * (100 W * 100 s) = 20,000 J; active duration 200 s (gap excluded)
    assert abs(energy_J - 20_000.0) < 1e-6
    assert abs(active - 200.0) < 1e-6


def test_trapezoid_short_series_zero():
    ts = np.array([0.0, 1.0])
    p = np.array([100.0])
    energy_J, active = trapezoid_energy(ts, p)  # mismatched lengths
    assert energy_J == 0.0
    assert active == 0.0


# ---------------------------------------------------------------------------
# Token math + per-family loss fraction
# ---------------------------------------------------------------------------

def test_compute_max_tokens_run2():
    # run2: global_step=299130, grad_accum=1, pbs=40, world=4, seq=512
    assert compute_max_tokens(299130, 1, 40, 4, 512) == 299130 * 40 * 4 * 512


def test_compute_max_tokens_grad_accum():
    # run1: grad_accum=2 multiplies micro-batches
    assert compute_max_tokens(27600, 2, 8, 3, 2048) == 27600 * 2 * 8 * 3 * 2048


def test_loss_fraction_prefix_suffix():
    frac, flag = loss_fraction({
        "model_family": "concept_ar_prefix",
        "objective_family": "prefix_suffix",
        "prefix_ratio_min": 0.35, "prefix_ratio_max": 0.45,
    })
    assert abs(frac - 0.6) < 1e-9
    assert flag == "loss_fraction:prefix_suffix_approx"


def test_loss_fraction_reconstruction_perceiver():
    frac, flag = loss_fraction({"model_family": "perceiver_denoise",
                                "objective_family": "reconstruction"})
    assert frac == 1.0
    assert flag == "loss_fraction:reconstruction_approx"


def test_loss_fraction_reconstruction_variant_substring():
    # older perceiver run: objective_variant='reconstruction+contrastive'
    frac, flag = loss_fraction({"objective_variant": "reconstruction+contrastive"},
                               run_name="perceiver_denoise_H512L6C128D3_20260314_224319")
    assert frac == 1.0
    assert flag == "loss_fraction:reconstruction_approx"


def test_loss_fraction_unknown():
    frac, flag = loss_fraction({}, run_name="mystery_run_20260101_000000")
    assert frac is None
    assert flag == "loss_fraction:unknown"


def test_loss_fraction_prefix_ratio_missing():
    frac, flag = loss_fraction({"model_family": "concept_ar_prefix",
                                "objective_family": "prefix_suffix"})
    assert frac is None
    assert flag == "loss_fraction:prefix_ratio_missing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_parse_world_size():
    ds = "Distributed environment: DistributedType.MULTI_GPU  Backend: nccl\nNum processes: 4\nProcess index: 0\n"
    assert parse_world_size({"distributed_state": ds}) == 4
    assert parse_world_size({}) is None
    assert parse_world_size({"distributed_state": "no processes here"}) is None


def test_cfg_get_unwraps_value_dict():
    assert cfg_get({"x": {"value": 7}}, "x") == 7
    assert cfg_get({"x": 7}, "x") == 7
    assert cfg_get({}, "x") is None


def test_infer_family_from_name_ordering():
    # concept_ar_prefix must match before concept_ar
    assert infer_family_from_name("concept_ar_prefix_H768L6C128D4_20260627_192407") == (
        "concept_ar_prefix", "prefix_suffix")
    assert infer_family_from_name("concept_ar_H768L6C128D4_20260614_164206") == (
        "concept_ar", "reconstruction")
    assert infer_family_from_name("perceiver_denoise_H512L6C128D3_20260314_224319") == (
        "perceiver_denoise", "reconstruction")
    assert infer_family_from_name("weighted_mlm_X_20260101_000000") == ("weighted_mlm", "mlm")
    assert infer_family_from_name("unknown_20260101_000000") == (None, None)


def test_arch_prefix_from_name():
    assert arch_prefix_from_name("perceiver_denoise_H512L6C128D3_20260314_224319") == \
        "perceiver_denoise_H512L6C128D3"
    assert arch_prefix_from_name("concept_ar_prefix_H768L6C128D4_20260614_101305") == \
        "concept_ar_prefix_H768L6C128D4"
    assert arch_prefix_from_name("no_timestamp_here") is None


def test_is_plausibility_flag_separates_informational():
    assert is_plausibility_flag("gpu0:avg_power_low:50.0W")
    assert is_plausibility_flag("gpu2:avg_power_over_limit:360.0>350.0W")
    assert is_plausibility_flag("energy:trapezoid_vs_avg:0.12")
    assert is_plausibility_flag("energy:retrieval_unsupported")
    assert is_plausibility_flag("gpu_hours:summary_vs_ts:0.05")
    # informational only -> not plausibility
    assert not is_plausibility_flag("loss_fraction:prefix_suffix_approx")
    assert not is_plausibility_flag("loss_fraction:reconstruction_approx")
    assert not is_plausibility_flag("writeback_deferred:running")
    assert not is_plausibility_flag("writeback_failed:...")


# ---------------------------------------------------------------------------
# audit_run end-to-end with FakeRun
# ---------------------------------------------------------------------------

def test_audit_run_finished_clean_writes_summary():
    df = make_power_df(duration_s=3600, dt_s=7.5, ngpu=4, power_w=250.0)
    cfg = prefix_suffix_cfg(world=4)
    summary = {"_runtime": 3605, "train_runtime": 3600, "train/train_runtime": 3600,
               "global_step": 1000}
    run = FakeRun("concept_ar_prefix_H768L6C128D4_20260601_000000", "finished", summary, cfg, df)

    rec = audit_run(run, write_back=True)
    sc = rec["scalars"]

    assert rec["audit_state"] == "finished"
    assert rec["gate_failed_structural"] is False
    # gpu_hours = 3600 * 4 / 3600 = 4.0
    assert abs(sc["compute/gpu_hours"] - 4.0) < 1e-9
    # energy = 4 * 250W * 3600s / 3.6e6 = 1.0 kWh
    assert abs(sc["compute/energy_kwh"] - 1.0) < 1e-6
    # max_tokens = 1000 * 1 * 40 * 4 * 512
    assert sc["compute/max_tokens"] == 1000 * 40 * 4 * 512
    assert sc["compute/loss_tokens_est"] == int(1000 * 40 * 4 * 512 * 0.6)
    assert sc["compute/world_size"] == 4
    assert sc["compute/runtime_source"] == "train/train_runtime"
    # summary write-back happened
    assert rec["writeback"] is True
    assert run.summary["compute/gpu_hours"] == sc["compute/gpu_hours"]
    assert run.summary["compute/audit_state"] == "finished"
    # informational flag recorded but did not trigger flagged state
    assert "loss_fraction:prefix_suffix_approx" in rec["flags"]


def test_audit_run_running_defers_writeback():
    df = make_power_df(3600, 7.5, 3, 250.0)
    cfg = prefix_suffix_cfg(world=3, pbs=8, grad_accum=2, seq=2048, epochs=1)
    cfg["prefix_ratio_min"] = 0.3
    cfg["prefix_ratio_max"] = 0.5
    summary = {"_runtime": 3600, "global_step": 500}  # no train_runtime
    run = FakeRun("concept_ar_prefix_H768L6C128D4_20260627_192407", "running", summary, cfg, df)

    rec = audit_run(run, write_back=True)
    assert rec["audit_state"] == "running-partial"
    assert rec["writeback"] is False
    # scalars NOT written to the live run's summary
    assert "compute/gpu_hours" not in run.summary
    assert any("writeback_deferred:running" in f for f in rec["flags"])
    # but the record still carries computed numbers for the local artifact
    assert rec["scalars"]["compute/gpu_hours"] is not None
    assert rec["scalars"]["compute/runtime_source"] == "_runtime"


def test_audit_run_structural_fail_gpu_count_mismatch():
    # config says 4 processes but the power series only has 2 GPUs
    df = make_power_df(3600, 7.5, 2, 250.0)
    cfg = prefix_suffix_cfg(world=4)
    summary = {"_runtime": 3600, "global_step": 1000}
    run = FakeRun("concept_ar_prefix_X_20260601_000000", "finished", summary, cfg, df)

    rec = audit_run(run, write_back=True)
    assert rec["gate_failed_structural"] is True
    assert rec["scalars"]["compute/audit_state"] == "failed"
    assert "gpu_count_mismatch" in rec["error"]
    # no compute scalars written on structural fail
    assert "compute/gpu_hours" not in run.summary


def test_audit_run_structural_fail_missing_config():
    df = make_power_df(3600, 7.5, 4, 250.0)
    cfg = {"distributed_state": "Num processes: 4"}  # missing batch/seq_len/global_step etc
    summary = {"_runtime": 3600}  # no global_step
    run = FakeRun("concept_ar_prefix_X_20260601_000000", "finished", summary, cfg, df)

    rec = audit_run(run, write_back=True)
    assert rec["gate_failed_structural"] is True
    assert "global_step" in rec["error"]
    assert "per_device_train_batch_size" in rec["error"]


def test_audit_run_plausibility_flag_low_power():
    df = make_power_df(3600, 7.5, 4, 50.0)  # 50 W avg < 80 W floor -> idle-ish GPU
    cfg = prefix_suffix_cfg(world=4)
    summary = {"_runtime": 3600, "global_step": 1000}
    run = FakeRun("concept_ar_prefix_X_20260601_000000", "finished", summary, cfg, df)

    rec = audit_run(run, write_back=False)
    assert rec["audit_state"] == "flagged"
    assert any("avg_power_low" in f for f in rec["flags"])
    # scalars still written to the record (and would be written to summary if enabled)
    assert rec["scalars"]["compute/energy_kwh"] is not None


def test_audit_run_energy_retrieval_unsupported():
    # history() returns an empty DataFrame -> energy withheld, flagged
    run = FakeRun("concept_ar_prefix_X_20260601_000000", "finished",
                  {"_runtime": 3600, "global_step": 1000}, prefix_suffix_cfg(world=4),
                  pd.DataFrame())
    rec = audit_run(run, write_back=False)
    assert rec["audit_state"] == "flagged"
    assert any("energy:retrieval_unsupported" in f for f in rec["flags"])
    assert rec["scalars"]["compute/energy_kwh"] is None
    # gpu_hours / tokens still computed
    assert rec["scalars"]["compute/gpu_hours"] is not None
    assert rec["scalars"]["compute/max_tokens"] is not None


def test_audit_run_group_for_panel_falls_back_to_arch_prefix():
    # older run with no wandb_group config
    df = make_power_df(3600, 7.5, 3, 250.0)
    cfg = {
        "model_family": None,  # forces name-based inference
        "objective_variant": "reconstruction+contrastive",
        "per_device_train_batch_size": 16, "gradient_accumulation_steps": 1,
        "max_seq_length": 512, "num_train_epochs": 20,
        "distributed_state": "Num processes: 3", "dataset_name": "JeanKaddour/minipile",
        "git_commit": "6c7061a",
    }
    summary = {"_runtime": 3600, "train_runtime": 3600, "global_step": 1000}
    run = FakeRun("perceiver_denoise_H512L6C128D3_20260314_224319", "finished", summary, cfg, df)

    rec = audit_run(run, write_back=False)
    assert rec["model_family"] == "perceiver_denoise"
    assert rec["group_for_panel"] == "perceiver_denoise_H512L6C128D3"
    assert rec["scalars"]["compute/group_for_panel"] == "perceiver_denoise_H512L6C128D3"
    # reconstruction -> loss_fraction 1.0
    assert rec["scalars"]["compute/loss_tokens_est"] == rec["scalars"]["compute/max_tokens"]
    # max_tokens rescaled to billions for the grouped profile panel
    assert abs(rec["scalars"]["compute/max_tokens_b"]
               - rec["scalars"]["compute/max_tokens"] / 1e9) < 1e-6


def test_audit_run_writeback_removes_stale_pct_keys():
    """Re-auditing a run that has stale _pct keys (from a prior version) drops them."""
    df = make_power_df(3600, 7.5, 4, 250.0)
    cfg = prefix_suffix_cfg(world=4)
    summary = {"_runtime": 3605, "train_runtime": 3600, "global_step": 1000,
               # simulate stale keys written by the previous cohort-pct version
               "compute/gpu_hours_pct": 100.0, "compute/energy_kwh_pct": 100.0,
               "compute/max_tokens_pct": 100.0}
    run = FakeRun("concept_ar_prefix_X_20260601_000000", "finished", summary, cfg, df)
    rec = audit_run(run, write_back=True)
    assert rec["writeback"] is True
    assert "compute/gpu_hours_pct" not in run.summary
    assert "compute/energy_kwh_pct" not in run.summary
    assert "compute/max_tokens_pct" not in run.summary
    assert "compute/gpu_hours" in run.summary


# ---------------------------------------------------------------------------
# (cohort-relative _pct removed — kept absolute rescaled units instead, so the
# panel stays comparable across future runs without re-normalization)
# ---------------------------------------------------------------------------

# Compute Audit + W&B Compute Panel — engineering implementation plan

- **Type:** engineering (tracking/instrumentation/analysis), not an `E0NN` experiment
- **Status:** implemented (2026-06-28) — `analysis/run_compute_audit.py` + `tests/test_compute_audit.py` + `experiment-evaluate`/`experiment-track` skill wiring; audited 5 runs
- **Owner:** Krzysztof Sopyla
- **Serves:** comparable cross-run compute accounting (GPU-hours, total energy, training tokens) so runs of the **same experiment** (`wandb_group`) but with different data mix / optimization / hyperparameters can be compared on compute spent; surfaced as a native W&B custom panel the user filters and compares 2–3 runs in.

## Problem statement (evidence from the two example runs + W&B data)

Inspected runs in `ksopyla/MrCogito`:

- `concept_ar_prefix_H768L6C128D4_20260627_192407` — E05, **running**, 3× RTX 3090 (Odra), seq 2048, target 1 epoch (at epoch 0.13), `_runtime` ≈ 50,438 s.
- `concept_ar_prefix_H768L6C128D4_20260614_101305` — E02, **finished**, 4× RTX 3090 (Polonez), seq 512, 5 epochs, `train/train_runtime` ≈ 261,631 s.
- `concept_ar_H768L6C128D4_20260614_164206` 
- `concept_ar_prefix_H768L6C128D4_20260614_101305`
- `concept_ar_H768L6C128D4_20260613_185955`
- `perceiver_denoise_H512L6C128D3_20260314_224319`

What W&B already logs automatically (no code change needed to obtain):
- Wall-clock runtime: `_runtime` (always), `train/train_runtime` (finished runs).
- Per-GPU power time series: `system/gpu.{i}.powerWatts`, `system/gpu.{i}.enforcedPowerLimitWatts`, utilisation, memory, temp — sampled every **~7.5 s** (6,735 pts / 50,438 s on run1; 34,884 pts / 261,636 s on run2).
- Config: `per_device_train_batch_size`, `gradient_accumulation_steps`, world size (`distributed_state` → "Num processes: N"), `max_seq_length`, `prefix_ratio_min/max`, `deletion_rate`, `objective_family`, `model_family`, `wandb_group`, `global_step`.

What is **not** logged and blocks a compute comparison today:
- No single `compute/gpu_hours` scalar (derivable: `runtime × world_size / 3600`).
- No total energy scalar (requires integrating the per-GPU power time series).
- No token count: `include_num_input_tokens_seen` = `"no"` on every run; `train/total_flos` = **0** on the finished run (the custom `ConceptEncoder` does not report FLOPs to the HF Trainer).
- The two example runs are different regimes (3 vs 4 GPU, 2048 vs 512 seq, 0.13 vs 5 epochs, running vs finished). A bare "total energy" bar across them is near-meaningless; the real use case is **within-`wandb_group`** comparison (same architecture/objective, varying data mix / optimization / hyperparameters).

Derived numbers (from currently logged data, upper-bound estimate):
- GPU-hours: run1 ≈ **42.0** (partial), run2 ≈ **290.7**.
- Max tokens (positions processed) = `global_step × grad_accum × pbs × world_size × max_seq_length`: run1 ≈ **2.71 B** (partial), run2 ≈ **24.5 B**.

## Grill-me decisions (resolved)

1. **Scope & instrumentation:** post-hoc reusable script, user supplies W&B run ids (one or more); covers ≥5 past runs; **no training-loop change** (no live callback, no throughput tax, no `include_*` flags). Outputs: estimated GPU-hours, total energy across N×GPU, estimated max training tokens. *(Post-hoc only was chosen over a live callback because the data is already in W&B and the built-in `include_num_input_tokens_seen` adds a per-micro-step DDP all-gather + `.item()` sync (~1–4% throughput loss on saturated DDP) and `include_tokens_per_second` enumerates the whole dataloader at startup.)*
2. **Energy retrieval & integration:** full-resolution via the `wandb` Python API (`run.history(stream='system')` or `scan_history`), **trapezoidal integral using the real per-sample `dt` from `_timestamp`**, summed across the `world_size` GPUs, kWh = Σ(W·s)/3.6e6. (Rejected the MCP `get_run_history_tool` path: it downsamples to ~500 points — a ~70× compression that silently biases the integral.)
3. **Verification gate policy:** hard-fail on **structural** gates (refuse to write the scalar, emit an error row); **write-with-`compute/flag`** on **plausibility** gates; **mandatory synthetic unit test** for the integrator. Never write silently.
4. **"Max training tokens" definition:** report **both** — `compute/max_tokens` = positions processed (objective-agnostic upper bound, the Chinchilla "D") as the headline, plus a flagged `compute/loss_tokens_est` = loss-target estimate using per-family loss fraction. (Rejected tokens-as-loss-only: not a "max", approximate, forces per-family branching for the headline; rejected `num_input_tokens_seen`: absent on every existing run.)
5. **Comparability framing:** raw bars + per-run config tags (family, `world_size`, `seq_len`, epochs, state, dataset, `wandb_group`) + a derived-ratios panel (`tokens_per_gpu_hour`, `energy_per_gpu_hour_kw`, `gpu_hours_per_billion_tokens`, `energy_per_billion_tokens`). Primary use is within-`wandb_group` comparison; ratios + tags are the safety net when regimes differ. (`wandb_group` is the panel grouping axis — already logged.)
6. **Surfacing:** write `compute/*` scalars into each run's summary (enables a native W&B custom panel grouped by `wandb_group`, which respects the workspace run-set filter so the user can pick 2–3 runs to compare) **and** save a local CSV + matplotlib chart artifact to `Cache/Evaluation_reports/compute_audit/` for registry citation. For **running** runs, emit the local artifact now but **defer summary write-back** until the run finishes (the live wandb process can drop summary keys it did not log). **The agent cannot create the W&B panel** (no panel-creation MCP tool / supported API); the user builds the panel once from the spec in Step 5.
7. **Reminder / integration:** wire a **run-level "Compute audit" preamble** into the `experiment-evaluate` skill (runs before Tier 0; W&B-only, no GPU, once per training run) so the audit fires automatically when a run is evaluated, and add a one-line reminder in `experiment-track` to cite `compute/gpu_hours` / `compute/energy_kwh` / `compute/max_tokens` in the registry row + run report.

## Target design

### New script: `analysis/run_compute_audit.py`

CLI:
```
uv run python analysis/run_compute_audit.py \
  --run-id concept_ar_prefix_H768L6C128D4_20260614_101305 \
  --run-id concept_ar_prefix_H768L6C128D4_20260627_192407 \
  [--group <wandb_group>] [--tag <tag>] \
  --entity ksopyla --project MrCogito \
  --out-dir Cache/Evaluation_reports/compute_audit/ \
  [--dry-run]        # compute + local artifact, no W&B summary write-back
  [--no-writeback]   # same, explicit
```
`--run-id` accepts the run display name (the project's run names are the timestamped ids); `--group`/`--tag` expand to run sets. No GPU required — runs on macOS or the server.

Per audited run, computed scalars (all written to `run.summary` for finished runs; all written to the local CSV always):

| Scalar | Definition |
|---|---|
| `compute/gpu_hours` | `runtime × world_size / 3600` (`train/train_runtime` if present else `_runtime`) |
| `compute/energy_kwh` | `Σ_gpu trapezoid(powerWatts over dt) / 3.6e6` |
| `compute/max_tokens` | `global_step × grad_accum × per_device_train_batch_size × world_size × max_seq_length` |
| `compute/max_tokens_b` | `compute/max_tokens / 1e9` (billions — rescaled so it shares the GPU-hours/energy range; raw tokens ~1e9 dominate a shared axis) |
| `compute/loss_tokens_est` | `compute/max_tokens × loss_fraction` (per-family, flagged) |
| `compute/tokens_per_gpu_hour` | `compute/max_tokens / compute/gpu_hours` |
| `compute/energy_per_gpu_hour_kw` | `compute/energy_kwh / compute/gpu_hours` (= avg kW per GPU; sanity metric) |
| `compute/gpu_hours_per_billion_tokens` | `compute/gpu_hours / (compute/max_tokens / 1e9)` |
| `compute/energy_per_billion_tokens` | `compute/energy_kwh / (compute/max_tokens / 1e9)` |
| `compute/audit_state` | `finished` \| `running-partial` \| `flagged` \| `failed` |
| `compute/flag` | list of plausibility warnings (empty when clean) |
| `compute/runtime_source` | `train_runtime` \| `_runtime` |
| `compute/world_size` | int (from config `distributed_state`) |

Per-family `loss_fraction` (flagged approximate; exact loss tokens are a separate, deferred plan — see Non-goals):
- `prefix_suffix` → `1 - midpoint(prefix_ratio_min, prefix_ratio_max)` (≈0.6 for both example runs).
- `reconstruction` (E01/E04) → `1.0` (predict full sequence; loss on all non-pad positions).
- `weighted_mlm` → masking rate from config.
- unknown family → skip `compute/loss_tokens_est`, append `"loss_fraction:unknown"` to `compute/flag`.

Grouped profile scale: the three headline metrics have different units and raw
magnitudes differ by ~1e7 (tokens ~1e9 vs GPU-hours ~1e2). For a single grouped
bar per run, use the **absolute** scalars with `max_tokens` rescaled to billions
(`compute/max_tokens_b`): GPU-hours (~35–290), energy (~11–61 kWh), tokens (~1.5–24.5 B)
all land in a comparable numeric range on one linear axis. These are **stable
absolute values** — comparable across past and future runs without re-normalization
(rejected cohort-relative `%` because a future heavier run would rescale
everything). The grouped bar is a pragmatic "compute profile at a glance"; for
exact per-metric reads use the per-metric panels (each on its own correct axis).

Retrieval (Step 1 spike — verify the exact call):
- `api = wandb.Api(); run = api.run(f"{entity}/{project}/{run_id}")`.
- System metrics: `run.history(stream='system')` (preferred) → DataFrame with `_timestamp` + `system/gpu.{i}.powerWatts` per GPU; fallback `run.scan_history(keys=["system/gpu.{i}.powerWatts", "_timestamp"], ...)` per GPU if the DataFrame path lacks timestamps.
- Summary + config: `run.summary`, `run.config`, `run.state`.
- GPU index set = sorted `i` such that `system/gpu.{i}.powerWatts` exists; cross-check against `world_size`.

Energy integration:
- For each GPU `i`: build `(t, p)` pairs sorted by `t`; `dt = diff(t)`; trapezoid `Σ (p[k]+p[k+1])/2 × dt[k]`; energy_i_J = that sum (W·s). Guard against `dt <= 0` or huge gaps (gap > 60 s → split the integral, don't bridge).
- `compute/energy_kwh = (Σ_i energy_i_J) / 3.6e6`.

Verification gates (enforced before any `run.summary` write):

- **Structural — hard-fail (set `compute/audit_state=failed`, do NOT write compute/* scalars, emit error row in the CSV/report):**
  - `len(gpu_index_set) == world_size` (else a subset-GPU run would be mis-counted).
  - `global_step`, runtime, `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_seq_length`, `world_size` all present and non-null.
  - The integrator synthetic unit test passes (enforced in CI, not at runtime — see tests).
- **Plausibility — write-with-flag (set `compute/audit_state=flagged`, append reason to `compute/flag`, still write the scalars):**
  - per-GPU `avg_power` outside `[80, enforcedPowerLimitWatts]` W (RTX 3090 idle ~30 W, training ~150–320 W, TDP 350 W). A GPU averaging ~idle was likely not used.
  - `|compute/energy_kwh − avg_power_total × runtime / 3.6e6| / compute/energy_kwh ≥ 0.05` (trapezoid vs rectangle-from-avg must agree ~5% for a smooth signal; disagreement ⇒ dt/timestamp-pairing bug).
  - `|gpu_hours_from_summary − gpu_hours_from_ts_span| / gpu_hours_from_summary ≥ 0.01` (truncated series / wrong runtime).

Running-run handling:
- If `run.state == 'running'`: use current `_runtime` and current `global_step`, integrate power up to the last sample, set `compute/audit_state=running-partial`, write the local CSV/JSON row, but **skip `run.summary` write-back** (the live process can drop our keys). Print a reminder to re-run after the run finishes. `--dry-run`/`--no-writeback` forces this for all runs.

Local artifacts (`Cache/Evaluation_reports/compute_audit/`):
- `<timestamp>_summary.csv` — one row per audited run with every scalar + config tags (`wandb_group`, `model_family`, `objective_family`, `world_size`, `max_seq_length`, `num_train_epochs`, `state`, `dataset_name`, `git_commit`).
- `<timestamp>_comparison.html` + `.png` — matplotlib grouped bars: raw panel (GPU-h, energy kWh, max-tokens) and ratios panel (tokens/GPU-h, energy/GPU-h, GPU-h per B-tok), grouped/colored by `wandb_group`, per-run labels.
- `<timestamp>_per_run.json` — full per-run record (scalars, flags, gate outcomes, series stats) for auditing.

Dependencies: `wandb` (explicit dep), `matplotlib` (explicit dep), `numpy` (transitive via torch). **No new dependencies.**

### New test: `tests/test_compute_audit.py`

- **Synthetic integrator falsification anchor:** feed `integrate_energy([(t, P), ...])` a constant-power series of known P and known duration; assert `energy_kwh == P × duration / 3.6e6` exactly (within float tol). Feed a piecewise-constant ramp with a known closed-form integral. Feed a series with a >60 s gap and assert the integral splits (doesn't bridge).
- **Gate logic:** `len(gpus) != world_size` → structural hard-fail (no scalars written, `audit_state=failed`). Missing config key → hard-fail. avg_power out of bounds → flagged (scalars written, `compute/flag` populated). Trapezoid-vs-avg disagreement → flagged.
- **Token math:** `max_tokens == global_step × grad_accum × pbs × world_size × seq_len` for a synthetic config; per-family `loss_fraction` for `prefix_suffix` / `reconstruction` / `weighted_mlm` / unknown.
- **Running-run path:** `run.state == 'running'` → `audit_state=running-partial`, summary write-back skipped.
- Mock the `wandb.Api` run object (summary/config/history) so the test runs offline without network.

### Skill edit: `experiment-evaluate` (`.claude/skills/experiment-evaluate/SKILL.md`)

Insert a new section **before "### Tier 0 — Health / sanity"**, titled:

```
### Run-level preamble — Compute audit (W&B-only, no GPU, once per training run)
```

Content: run once per training run (not per checkpoint) before the per-checkpoint tiers. Writes `compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens`, `compute/loss_tokens_est` + ratios into the run's W&B summary so the compute panel (see `docs/engineering_specs/compute_audit_wandb_panel.md`) populates automatically. For still-running runs it emits the local artifact only (re-run after finish).

```
uv run python analysis/run_compute_audit.py --run-id <run_id> \
  --out-dir Cache/Evaluation_reports/compute_audit/
```

Gate: structural hard-fail → `compute/audit_state=failed` (inspect the per-run JSON); plausibility flag → `compute/audit_state=flagged` (scalars still written, read `compute/flag`). This tier needs no checkpoint and no GPU; it is orthogonal to the per-checkpoint Tier 0–3.

### Skill edit: `experiment-track` (`.claude/skills/experiment-track/SKILL.md`)

- In **"Core Workflow → 1. Reconstruct the run facts"**, add a bullet: `compute: compute/gpu_hours, compute/energy_kwh, compute/max_tokens (from the run's W&B summary, written by the compute audit; see experiment-evaluate run-level preamble)`.
- In **"How To Judge Results"** (the "training protocol, compute budget, and checkpoint maturity" factor), add: prefer the audited `compute/*` scalars over hand-estimates; note `compute/audit_state` (`flagged`/`running-partial` numbers are approximate).
- In the **run-report template** `## Configuration` table, add a `| Compute | GPU-h, kWh, max-tokens (compute audit) |` row.

### W&B panel spec (user builds once — the agent cannot)

In the `ksopyla/MrCogito` workspace:
1. **Compute profile (grouped, absolute, one scale)** — Bar Chart with the three metrics as the Metric set, using `compute/max_tokens_b` (not raw `compute/max_tokens`): `compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens_b`. **Group by run name / run id** so the three bars cluster per run. `max_tokens` is rescaled to billions so it shares the GPU-hours/energy range (raw tokens ~1e9 would dominate). These are stable absolute values — comparable across past and future runs. This is the "how much compute · how many tokens · how efficient (power)" trio per run.
2. **Per-metric bar panels (absolute, each on its own correct scale)** — Bar Chart panels, one each with y-axis:
   - `compute/gpu_hours`
   - `compute/energy_kwh`
   - `compute/max_tokens_b` (billions)
   - (optional) `compute/loss_tokens_est`, `compute/tokens_per_gpu_hour`, `compute/energy_per_gpu_hour_kw`
   - Group/color by `compute/group_for_panel` (handles old runs without `wandb_group`).
3. **Table panel** — columns: run name, `compute/group_for_panel`, `compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens_b`, `compute/loss_tokens_est`, the four ratios, `compute/audit_state`, `compute/flag`.
4. **Filtering / compare 2–3:** use the workspace run-set filter — filter by `compute/group_for_panel` (e.g. `E02_concept_ar_prefix_H768L6C128D4`) or multi-select specific run names; all panels update to show exactly those runs. Save as a named view (e.g. "Compute comparison").
5. Hover shows config tags (`world_size`, `max_seq_length`, `num_train_epochs`, `state`, `dataset_name`) automatically because they are in each run's config.

Caveat to document in the panel/view description: raw bars are meaningful within a `wandb_group` (matched setup); across regimes use the ratios and read `compute/audit_state`/`compute/flag`.

## Implementation plan (repo-rooted, ordered)

### Step 1 — `analysis/run_compute_audit.py` (core)
Spike first: confirm `run.history(stream='system')` returns full-res `(timestamp, powerWatts)` for all GPUs on the two example runs; fall back to `scan_history` if not. Then implement retrieval, trapezoidal energy integration (with gap handling), GPU-hours, token math, per-family loss fraction, ratios, gates, summary write-back (finished only), and the CSV + matplotlib artifacts.

### Step 2 — `tests/test_compute_audit.py`
Synthetic integrator falsification test + gate-logic + token-math + running-run-path tests, with a mocked `wandb.Api` so they run offline. Add to the `uv run pytest tests/ -v` suite.

### Step 3 — `experiment-evaluate` skill edit
Insert the run-level "Compute audit" preamble before Tier 0 (text above).

### Step 4 — `experiment-track` skill edit
Add the compute-scalars bullet to "reconstruct run facts", the audit-state nudge to "compute budget" judgment, and the `Compute` row to the run-report template.

### Step 5 — W&B panel spec
Hand the user the panel spec (above) to build once in the UI. No code.

### Step 6 — Traceability
- `CHANGELOG.md` entry under `## [2026-06-28] - Compute audit + W&B compute panel`, prefix `feat:` (new capability) with `eval:` impact (eval-pipeline skill change). Follow the `engineering-change-tracking` template (Why / Impact / What changed / Git tag / Related).
- `docs/1_Strategy_and_Plans/agenda.md`: one-line note that compute audit + panel is available for within-group compute comparison.
- Commit message: `feat: add post-hoc compute audit (GPU-h, energy, tokens) + W&B panel spec + eval/track skill wiring`.

## Validation and falsification

Primary gate (falsifiable):
- The **synthetic integrator unit test** must assert `energy_kwh == P × duration / 3.6e6` exactly for constant power, and the closed-form value for a piecewise ramp. If this fails, the integrator is wrong — **do not run on real data**.
- On the two example runs: `compute/gpu_hours` must equal `_runtime × world_size / 3600` within 1e-3 (run1 ≈ 42.0, run2 ≈ 290.7); `compute/max_tokens` must equal `global_step × grad_accum × pbs × world_size × max_seq_length` (run1 ≈ 2.71e9 partial, run2 ≈ 24.5e9).

Secondary acceptance (plausibility, must hold or flag):
- Per-GPU `avg_power ∈ [80, enforcedPowerLimitWatts]` W.
- `|energy_kwh − avg_power_total × runtime / 3.6e6| / energy_kwh < 0.05`.
- `|gpu_hours_summary − gpu_hours_ts_span| / gpu_hours_summary < 0.01`.

Failure/kill conditions:
- Structural gate trips on a run ⇒ that run's `compute/*` scalars are **not written**; `compute/audit_state=failed` and an error row emitted. Do not silently write a wrong number.
- Synthetic test fails ⇒ integrator bug; block merge.
- W&B system-metric retrieval can't yield full-res timestamps ⇒ fall back to `scan_history`; if neither works, the energy scalar is withheld (`compute/flag=["energy:retrieval_unsupported"]`) and GPU-hours/tokens are still written.

## Risks / open spikes

- **System-metric retrieval API:** verify `run.history(stream='system')` vs `scan_history` for full-res `(timestamp, power)` pairs across all GPUs. Small spike at the start of Step 1 on the two example runs.
- **Running-run summary write-back overwrite:** mitigated by deferring write-back until `run.state != 'running'`; the local artifact always carries the current numbers.
- **Per-family `loss_fraction` is approximate** (config midpoints, not actual non-pad counts); flagged via `compute/flag`. Exact loss tokens come from the separate, deferred **live token-counting callback** plan (the "Action 2" from the prior compute-callback discussion), which would later add an exact `compute/loss_tokens_seen` and a `6 × N × D` FLOP estimate. This plan does **not** implement that callback.
- **Heterogeneous-run comparability:** raw bars mislead across regimes; mitigated by grouping on `wandb_group` and providing ratios + tags. Documented in the panel/view description.
- **`--run-id` resolution by display name:** the project's W&B run `name` equals the timestamped run id (confirmed in the two example runs); resolution uses `runs(filters: {displayName: {$eq: ...}})`. If a future run's display name diverges, fall back to the 8-char run id.

## Non-goals

- **No training-loop change** — no live callback, no `include_*` flags, no throughput tax. (Live token counting is a separate, deferred plan.)
- **No `train/total_flos` fix** — the custom model's FLOPs reporting stays broken; `6ND` FLOP estimation is deferred to the live-token-callback plan.
- **No W&B report generation via MCP** — the `create_wandb_report_tool` / `log_analysis_to_wandb` tools are not in the installed MCP set; the panel is built manually once by the user from the Step 5 spec.
- **No backfill** of `num_input_tokens_seen` or `train/total_flos` on historical runs.
- **No `master_experiment_log.md` schema change** — compute numbers are cited in row text / run reports, not new columns.
- **No model architecture, training-objective, or benchmark-metric changes.**

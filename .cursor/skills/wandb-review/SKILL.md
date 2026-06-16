---
name: wandb-review
description: Query, group, compare, and interpret W&B training runs for the MrCogito Concept Encoder project via the W&B MCP server. Use when reviewing experiment history, finding best runs by metric, comparing configs/objectives, diagnosing training curves, clustering runs by family, or preparing conclusions before experiment-track. Requires the wandb MCP server (see .cursor/mcp.json).
---

# W&B Experiment Review

## Mission

Use the **W&B MCP server** to pull live run data from W&B and turn it into structured
comparisons and tentative conclusions. This skill is for **analysis and synthesis** —
not for launching training (`experiment-run`) or writing registry rows (`experiment-track`).

Pair with `experiment-track` when findings should be persisted to
`docs/2_Experiments_Registry/master_experiment_log.md`.

## Project defaults

| Field | Value |
|---|---|
| Entity | `ksopyla` |
| Project | `MrCogito` |
| W&B URL | https://wandb.ai/ksopyla/MrCogito |

Always include `ksopyla/MrCogito` in MCP prompts unless the user specifies otherwise.

## Run naming and grouping (from code)

Training runs log via `init_wandb()` in `training/utils_training.py`:

- **Project:** `MrCogito`
- **Group:** stable experiment/architecture identity, no timestamp
  (e.g. `E01_concept_ar_H768L6C128D4`, `E02_concept_ar_prefix_H768L6C128D4`,
  `perceiver_denoise_H512L6C128D3`)
- **Name / id:** `run_identifier` (timestamped run id)
- **Job type:** training/eval stage and objective
  (e.g. `train_concept_ar_reconstruction`, `train_concept_ar_prefix_suffix`,
  `train_perceiver_denoise_reconstruction`)
- **Tags:** stage, model family, checkpoint family, decoder type, objective,
  experiment id when known, hostname, dataset, concept-loss tags
- **Config identity keys:** `experiment_id`, `model_family`, `objective_family`,
  `architecture_id`, `checkpoint_family`, `pretraining_objective`

Eval runs (GLUE, STS-B, benchmarks) also log to `MrCogito` with `job_type` eval tags.

When grouping runs, prefer these dimensions:

1. **Model family** — tag or config `model_type` / run-name prefix:
   `concept_ar`, `concept_ar_prefix`, `perceiver_denoise`, `weighted_mlm`,
   `diffusion_mlm`, `prefix_diffusion`, `recursive_mlm`
2. **Architecture** — config keys: `hidden_size`, `num_hidden_layers`, `concept_num`,
   `token_embedding_dim`, `intermediate_size`
3. **Objective** — config `objective_variant`, `objective_family`,
   `pretraining_objective`, `concept_losses`
4. **Data** — config `dataset_name`, tag matching dataset
5. **Machine** — hostname tag (`odra`, `polonez`)
6. **Stage** — training vs eval (`job_type`)

## MCP tool workflow

Follow W&B's schema-first pattern. Chain tools; do not guess metric keys.

### 1. Discover (unfamiliar scope)

```
list_entities_tool                          → confirm ksopyla
query_wandb_entity_projects                 → confirm MrCogito
probe_project_tool on ksopyla/MrCogito        → metric keys, config keys, tags
```

### 2. Query runs

Use `query_wandb_tool` for filters, top-k by summary metric, config sweeps.

Example prompts (adapt metric keys from probe output):

- "List finished training runs in `ksopyla/MrCogito` tagged `concept_ar`, newest first."
- "Top 10 runs by lowest `eval/loss` with config `hidden_size=768`."
- "Runs where summary `train/loss` diverged or config contains `concept_losses`."

For **time series** (loss curves, LR, grad norm): use `get_run_history_tool`, not GraphQL.

### 3. Compare and diagnose

| Goal | Tool |
|---|---|
| Config + metric diff between two runs | `compare_runs_tool` |
| Convergence, NaN, overfitting signal | `diagnose_run_tool` |
| Training curve shape | `get_run_history_tool` |

Always compare like-with-like: same eval metric definition, similar epoch count, same
dataset unless the question is explicitly cross-dataset.

### 4. Persist findings (optional)

- `create_wandb_report_tool` — shareable W&B report with panels
- `log_analysis_to_wandb` — log computed aggregates as an analysis run first

## Comparison framework (MrCogito-specific)

When drawing conclusions, judge runs against **their experiment's own gates** (see
`docs/experiments_specs/<ID>.md`) and these cross-cutting signals:

| Signal | Where in W&B | Notes |
|---|---|---|
| Train / eval loss | history + summary | Primary trainability |
| Concept effective rank | summary or eval logs | Collapse if ≪ `concept_num` |
| Concept-ablation ΔCE | E01/E02 eval runs | Is AR decoder using concepts? |
| Zero-shot STS-B Pearson | eval run summaries | Semantic grounding gate (~0.50+) |
| GLUE (MRPC F1, etc.) | eval run summaries | Downstream; compare ViaDecoder route |
| Step speed | summary `train/steps_per_second` | Compute efficiency |

**Fair comparison rules** (from `experiment-track`):

- Match model size (params), objective difficulty, data regime, and checkpoint maturity.
- Do not rank a 5-epoch probe against a 40-epoch baseline without noting the gap.
- Treat early E01 signals as directional, not final verdicts.

## Recommended review sessions

### A. Family overview

1. Probe project → list distinct tags / config `model_type` values.
2. For each family, query best run by primary metric (eval loss or target benchmark).
3. Output a table: family · best run id · key metrics · concept rank · one-line takeaway.

### B. Head-to-head (two runs)

1. `compare_runs_tool` for config diff + metric delta.
2. `get_run_history_tool` for aligned loss curves.
3. `diagnose_run_tool` if either run looks unhealthy.
4. Verdict: what changed, what improved, what's still blocked.

### C. Active experiment status (e.g. E01)

1. Filter runs by group/name prefix for the active `base_id`.
2. Pull latest eval loss, concept-ablation ΔCE, effective rank from history/summary.
3. Compare to kill/success gates in the experiment spec.
4. Flag whether the run is on track, needs more epochs, or should be killed.

### D. Historical audit (pre-registry gaps)

1. Query all finished runs in a date range.
2. Cross-check against `docs/2_Experiments_Registry/master_experiment_log.md`.
3. List W&B runs missing from the ledger or ledger rows with stale metrics.

## Output format

Deliver reviews as:

1. **Scope** — entity/project, filters used, run count retrieved (confirm completeness
   for broad queries).
2. **Groups** — clusters with shared config/tags.
3. **Ranked highlights** — top/bottom runs per group with metrics.
4. **Comparisons** — paired diffs for the most informative contrasts.
5. **Tentative conclusions** — evidence-backed, not final verdicts; cite run ids.
6. **Suggested next actions** — e.g. hand off to `experiment-track`, run eval tier,
   kill/extend training.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: wandb_gql` | Upstream dep clash (`wandb>=0.27.1` vs MCP). Wrapper pins `wandb<0.27.1`; restart Cursor |
| MCP tools not available | Restart Cursor; confirm `.cursor/mcp.json` wandb server is enabled in **Tools and MCP** |
| `WANDB_API_KEY is not set` | Fill `WANDB_API_KEY` in project `.env` (from https://wandb.ai/authorize) |
| Empty query results | Verify entity `ksopyla`, project `MrCogito`; probe first for correct metric keys |
| Auth errors | Regenerate API key; ensure `.env` is saved |

## Do not use this skill for

- Remote training launch/monitor → `experiment-run`
- Running local eval scripts → `experiment-evaluate`
- Writing registry/agenda updates → `experiment-track` (use after review)
- W&B product how-to → MCP `search_wandb_docs_tool`

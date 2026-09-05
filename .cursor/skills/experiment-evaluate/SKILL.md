---
name: experiment-evaluate
description: The single source of truth for evaluating MrCogito Concept Encoder checkpoints. Knows every evaluation script and runs a tiered pipeline (health → concept geometry + AR concept-ablation ΔCE + generation samples → generation vibe-check metrics → zero-shot STS-B → supervised SICK/PAWS/GLUE) via uv. Use after training finishes, when comparing best vs last checkpoints, checking concept health, running the generation vibe check, running any benchmark, or preparing evaluation evidence before experiment-track. Supports concept_ar (E01/E02), backbone_concept (E10/E16), and the older perceiver_denoise / weighted_mlm families.
---

# Experiment Evaluate

The **one place** that defines *how to evaluate* a Concept Encoder checkpoint and *which
script covers which aspect*. It runs the evaluation; it does not launch training
(`experiment-run`) and does not write conclusions into the registry (`experiment-track`,
which links back here for the "how").

## Default Behavior (read this first)
By **default, run all tiers in cost/signal order** (Tier 0 → Tier 3) and stop early only
when a kill gate trips. **Honor explicit narrow requests literally**: if the user asks for
"only concept analysis" run just Tier 1; if they ask for "generation quality", "vibe check",
or "generation metrics" run just Tier 1.5; if they ask for "MRPC only" run just that GLUE
task; if they ask for "STS-B" run just Tier 2. Never expand a narrow request into the full
suite, and never silently skip a tier on a full run.

**The run-level compute audit is a gate, not optional** — run it first (before Tier 0) on any
training run whose W&B summary lacks `compute/audit_state` (see the run-level preamble below).
It is idempotent and macOS-runnable, and `experiment-track` treats its output as a hard
precondition. A narrow request may skip the per-checkpoint tiers, but do not hand off to
`experiment-track` while the run still lacks `compute/audit_state`.

## When To Use
After training produced checkpoints and the user asks to evaluate, compare best/last
checkpoints, check concept health/collapse, run the generation vibe check (diversity /
repetition / length-binned metrics), run STS-B / SICK / PAWS / GLUE, measure whether
the AR decoder uses its concepts (E01/E02), or reproduce a prior evaluation protocol.

## Script Inventory (the complete map)
| Aspect | Script | Covers | Families |
|---|---|---|---|
| **Compute audit (run-level)** | `analysis/run_compute_audit.py` | GPU-hours, total energy (kWh, trapezoidal integral of per-GPU powerWatts), max training tokens + per-family loss-token estimate, derived ratios; writes `compute/*` to the run's W&B summary for the compute panel | all (reads W&B, no checkpoint) |
| **Concept geometry** + **AR ablation ΔCE** + **generation samples** | `analysis/run_concept_analysis.py` | effective rank, collapse, diversity; for `concept_ar` and `backbone_concept` also concept ablations (short qualitative generation extras remain `concept_ar` only) | all |
| **Generation vibe check** (metrics + samples) | `analysis/run_generation_quality.py` | free-run / continuation diversity (distinct-n, REP-3, length-binned profile, by-length cutoffs); optional suffix-CE-by-position (`concept_ar`); concept_mode ablations (`backbone_concept`) | `concept_ar`, `backbone_concept` |
| Generation metric library | `evaluation/generation_quality.py` | `distinct_n` / `repetition_rate` / `repetition_conditional` / `summarize_generation` / `generate_free_running` / `generate_continuation` / `compute_suffix_ce_by_position` (imported by the runner + notebooks) | — (library) |
| **E10 paired mechanism evaluation** | `analysis/run_e10_comparison.py` | matched-token concept-vs-control CE, static/one-block/shuffle recurrence attribution, local regression, 2K/8K recovery and paired bootstrap CIs | `backbone_concept` |
| Geometry metric library | `analysis/concept_analysis.py` | `compute_concept_geometry_metrics` (imported by the runner) | — (library) |
| Weight/health sanity | `analysis/check_model_health.py` | NaN/Inf, weight stats, dead units before fine-tuning | all |
| Zero-shot STS-B + SICK + PAWS | `evaluation/evaluate_on_benchmark.py` | `stsb_zero_shot`, `sick_relatedness`, `sick_entailment`, `paws` | all |
| GLUE fine-tune | `evaluation/evaluate_model_on_glue.py` | cola/mrpc/stsb/sst2/qnli/qqp/rte/mnli | all |
| Eval routing | `evaluation/concept_eval_routing.py` | maps `checkpoint_family` → sentence-pair / weighted-pool / via-decoder route | — (library) |
| Checkpoint loader | `evaluation/concept_checkpoint_loader.py` | safetensors/bin load + encoder-only/full weight copy | — (library) |
| SICK launcher | `scripts/evaluate_concept_encoder_sick.sh` | thin `uv` wrapper over `evaluate_on_benchmark.py` | all |
| PAWS launcher | `scripts/evaluate_concept_encoder_paws.sh` | thin `uv` wrapper over `evaluate_on_benchmark.py` | all |
| GLUE launcher | `scripts/evaluate_concept_encoder_glue.sh` | thin `uv` wrapper over `evaluate_model_on_glue.py` | all |
| Report sync | `scripts/sync_evaluation_reports.sh` | pull `Cache/Evaluation_reports/*` from remote | — |

**Out of scope of this skill** (do not pull in): tokenizer studies
(`analysis/evaluate_tokenizers_comprehensive.py`, `scripts/evaluate_tokenizers.sh`) and the
exploratory `analysis/concept_analysis_notebook.ipynb`.

## Model-Type Rules
- `concept_ar` (E01/E02) is now a **first-class `--model_type`** in every CLI
  (`run_concept_analysis.py`, `evaluate_on_benchmark.py`, `evaluate_model_on_glue.py`).
  Pass `--model_type concept_ar` directly — the old `MODEL_TYPE_OVERRIDE=perceiver_denoise`
  workaround is gone. Routing is chosen from the checkpoint's `config.json`
  (`checkpoint_family`, `canonical_pair_eval_mode`, `canonical_single_eval_mode`,
  `evaluation_contract_version`), so no override is needed.
- Older maintained families use their native type: `perceiver_denoise`, `weighted_mlm`.
  (`diffusion_mlm` / `prefix_diffusion` model code is parked; revive before evaluating.)
- `backbone_concept` (E10) uses `run_concept_analysis.py --model_type backbone_concept` for
  Tier-1 geometry/within-arm ablations. Its decisive result is the paired
  `run_e10_comparison.py` protocol at matched 50% and 100% token checkpoints; generic
  STS-B/SICK/GLUE routing is not part of the E10 mechanism gate.

## Checkpoints To Evaluate
For every serious run, evaluate at least:
- **Best checkpoint** — selected by `load_best_model_at_end` / lowest `eval_loss` in the log.
- **Last checkpoint** — the final training checkpoint.
- Final saved model is optional (usually duplicates best).

Use exact paths under `Cache/Training/<run_id>/checkpoint-<step>`. If distributed training
left a tiny duplicate rank-artifact directory, prefer the one named in the shell log's
`Output directory`.

## Remote Setup (uv, not poetry/venv)
Run evaluations on the GPU server (usually Polonez). Python is invoked through `uv run`.
```bash
ssh polonez
cd /home/ksopyla/dev/MrCogito
git fetch origin && git checkout dev && git pull --ff-only
uv sync                                   # ensure the env matches uv.lock
export HF_HOME=/home/ksopyla/dev/MrCogito/../hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
```
Use byobu for long suites: `byobu new-session -s EVAL_<ID>`. The bash launchers set
`HF_HOME`/`HF_DATASETS_CACHE` themselves and call `uv run python` internally.

Set checkpoint variables once:
```bash
RUN_ROOT="Cache/Training/<run_id>"
BEST="$RUN_ROOT/checkpoint-<best_step>"
LAST="$RUN_ROOT/checkpoint-<last_step>"
```

## The Tiered Pipeline (cost/signal order)

### Run-level preamble — Compute audit (W&B-only, no GPU, once per training run)
**This is a gate, not a preamble in name only.** A run that was never audited has *no*
`compute/*` keys at all (not even `compute/audit_state=failed`) — the absence is silent, which
is exactly how a finished run can get tracked without a compute profile. So check first, then
run. The audit is **idempotent** (re-running overwrites with identical values), so when in
doubt just run it rather than reasoning about whether you already did.
```bash
# Preflight: has the audit run on this run? MISSING => run it now.
uv run python -c "import wandb; print(wandb.Api().run('ksopyla/MrCogito/<run_id>').summary.get('compute/audit_state','MISSING'))"
```

Runs once per **training run** (not per checkpoint), before the per-checkpoint
tiers. Reads already-logged W&B system metrics + config and writes
`compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens`,
`compute/loss_tokens_est` + ratios into the run's W&B summary so the compute
panel (see `docs/engineering_specs/compute_audit_wandb_panel.md`) populates
automatically. No checkpoint, no GPU — runs on macOS or the server.
```bash
uv run python analysis/run_compute_audit.py --run-id <run_id> \
  --out-dir Cache/Evaluation_reports/compute_audit/
```
- Structural hard-fail → `compute/audit_state=failed` (inspect the per-run JSON;
  do **not** cite that run's compute numbers).
- Plausibility flag → `compute/audit_state=flagged` (scalars still written; read
  `compute/flag`).
- Still-running runs → `compute/audit_state=running-partial`; the local artifact
  is emitted but summary write-back is deferred (re-run after the run finishes).
`--dry-run` computes + writes the local artifact without touching W&B. Batch via
`--group <wandb_group>` or `--tag <tag>`.

### Tier 0 — Health / sanity (seconds, do once per checkpoint)
Catch numerical corruption before spending GPU time.
```bash
uv run python analysis/check_model_health.py --model_path "$BEST" --model_type concept_ar
```
Gate: no NaN/Inf, no all-dead layers. If it fails, stop and inspect the checkpoint.

### Tier 1 — Concept geometry + AR concept-ablation + samples (fast, the primary gate)
One command produces geometry for any family, plus (for `concept_ar`) the concept-ablation
ΔCE and qualitative generation samples — the **decisive E01/E02 evidence**.

**Data protocol (2026-07-07):** Tier 1 now runs on **genuinely held-out, length-stratified,
seeded** data at **seq 2048** by default. For 2K-mix runs (E05+), point it at the run's
pretokenized manifest (the exact training eval split); for single-dataset runs use
`--eval_source holdout` with the run's split seed. The legacy first-N-of-train-stream
protocol is `--eval_source stream` — **train-contaminated**, only for reproducing old
numbers. Numbers produced before this upgrade (all E01–E05 evals up to 2026-07-07) are
**not comparable** with post-upgrade numbers.
```bash
# 2K-mix runs (E05+): the pretokenized eval split is the authoritative held-out source
uv run python analysis/run_concept_analysis.py \
  --model_path "$BEST" \
  --model_type concept_ar \
  --eval_source pretokenized \
  --pretokenized_manifest <the run's pretokenize manifest.json> \
  --output_json "Cache/Evaluation_reports/<run_id>_best_concept_analysis.json" \
  --num_batches 24 --batch_size 8 --max_seq_length 2048 \
  --ablation_batches 8 --num_samples 4

# single-dataset runs (E01/E02): reproduce the training holdout split
uv run python analysis/run_concept_analysis.py \
  --model_path "$BEST" \
  --model_type concept_ar \
  --dataset HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
  --eval_source holdout --split_seed 42 --test_size_percent 0.1 \
  --output_json "Cache/Evaluation_reports/<run_id>_best_concept_analysis.json"
```
Batches are stratified over token-length buckets (`--length_buckets`, default
`256,512,1024` → buckets up to 2048), so geometry, ΔCE, and the L3 compression curve
are measured per length regime; deltas are reported **± per-batch std** (a gate cleared
by less than one std is not decisively cleared). For `perceiver_denoise` / `weighted_mlm`,
drop the AR-only flags and use their dataset (geometry is computed from the encoder for
every family).

**E10 paired checkpoint gate** (run at matched ~50% and 100% token checkpoints):
```bash
uv run python analysis/run_e10_comparison.py \
  --concept_checkpoint "$CONCEPT_CKPT" \
  --control_checkpoint "$CONTROL_CKPT" \
  --eval "2048:$E10_EVAL8K_MANIFEST:0.2840" \
  --eval "8192:$E10_EVAL8K_MANIFEST:0.3176" \
  --num_docs 64 --batch_size 1 \
  --output "Cache/Evaluation_reports/e10_paired_<exposure>.json"
```
Use the frozen, train-disjoint 8K eval-only manifest; both lengths must use the same documents
truncated to length. Replace the Stage-0 G values if the paired held-out Stage-0 rerun changes
them. A utility win over control is insufficient by itself: real recurrence must also beat the
learned-static and previous-block-only states with paired CIs excluding zero.

Gates:
- **Rank — read the right number** (three distinct objects, do not conflate; see
  `docs/engineering_specs/concept_information_eval_upgrade.md`):
  - **PRIMARY de-collapse = within-sample concept-set RankMe** (`within_sample_rankme_mean`):
    how many independent directions ONE input's `C` concepts span. This is what "collapse"
    actually means. Judge de-collapse on this. Read it together with the **centered**
    variant (`within_sample_rankme_centered_mean`): raw low + centered high = shared-offset
    anisotropy (not collapse); low on both = genuine collapse.
  - **SECONDARY diagnostic = slot-mean effective rank** (`global_effective_rank`, the old
    "rank N/128"): SVD of the *batch-averaged* slots → slot redundancy, not per-input rank.
    Keep as a diagnostic only; collapsed history ~5–10/128.
  - **Cross-sample embedding RankMe** (`manifold_rankme`): diversity of pooled embeddings
    *across inputs* — a downstream-retrieval property, can exceed `C`. Never quote it as
    "concept rank".
- Concept-ablation: E01 needs **Δzero AND Δshuffle ≥ 0.5 nats** (decoder genuinely uses
  concepts); E02 needs **Δshuffle ≥ 1.0** and **Δzero ≥ 2.0** on suffix CE. Shuffle is the
  stronger test. The intact model must beat the zero/no-concept floor by the same margin.
- Samples: held-out generations should be qualitatively coherent (E01 #4 / E02 #4).
  These short samples from `run_concept_analysis.py` are a coherence sniff only — the
  **quantitative generation vibe check** is Tier 1.5.

> Note: `run_concept_analysis.py` ablation uses the **reconstruction** contract (encoder
> sees the clean sequence). For a prefix→suffix (E02) run, the authoritative suffix-CE
> Δzero/Δshuffle are logged by the training `evaluate` step (`concept_ablation/*` in W&B);
> read those for the final E02 verdict.

### Tier 1.5 — Generation vibe check (metrics + free-run samples)
The **vibe check of generation**: does the model keep producing novel tokens past the
decoder window, or does it fall into fluent-local / semantically-empty repetition loops?
Tier 1's short samples answer "does it look coherent?"; this tier answers with numbers —
`distinct-n`, `repetition_rate`, REP-3 (Welleck), length-binned diversity, and (for
`concept_ar`) optional suffix-CE-by-position. Library: `evaluation/generation_quality.py`;
CLI: `analysis/run_generation_quality.py`. Run for every serious `concept_ar` /
`backbone_concept` checkpoint after Tier 1 clears (or as a narrow request on its own).

```bash
# concept_ar (E01/E02/E05): free-run from BOS + diversity; skip suffix-CE unless probing
# the E09 Stage-0 "frozen snapshot + K-window" wall
uv run python analysis/run_generation_quality.py \
  --model_path "$BEST" \
  --model_type concept_ar \
  --no_suffix_ce \
  --free_generation_max_new_tokens 512 \
  --length_cutoffs 64 128 256 512 \
  --output_json "Cache/Evaluation_reports/<run_id>_best_generation_quality.json"

# backbone_concept (E10/E16): true causal continuation + concept_mode ablations +
# chat-template probe; long cutoffs match the long-context claim
uv run python analysis/run_generation_quality.py \
  --model_path "$BEST" \
  --model_type backbone_concept \
  --no_suffix_ce \
  --free_generation_max_new_tokens 2048 \
  --length_cutoffs 512 1024 2048 4096 8192 16384 \
  --prompt_styles continuation chat \
  --concept_modes real zero shuffle static \
  --output_json "Cache/Evaluation_reports/<run_id>_best_generation_quality.json"

# E09 Stage-0 only (concept_ar): suffix-CE-by-position without free-gen cost
uv run python analysis/run_generation_quality.py \
  --model_path "$BEST" \
  --model_type concept_ar \
  --no_free_generation \
  --num_batches 10 --batch_size 4 --max_seq_length 2048 \
  --output_json "Cache/Evaluation_reports/<run_id>_best_suffix_ce.json"
```

Read the report:
- **`aggregate_by_length`** — mean `distinct_1` / `rep_3` at each cutoff. A profile that
  *falls* (distinct↓, REP-3↑) past the decoder K-window is the repetition-loop signature
  (E05/E02-long failure mode). Flat-or-rising distinct-n is healthy novelty.
- **`free_generation[].summary.length_binned_diversity`** — per-window distinct-n aligned
  to `decoder_context_window` / `concept_block` (E05: 128). Same signal, finer grain.
- **`aggregate_by_condition`** (`backbone_concept`) — compare `continuation|real` vs
  `zero` / `shuffle` / `static` at matched lengths; if real ≈ zero on diversity *and*
  the text is empty loops, concepts are not carrying generation.
- **`suffix_ce_by_position`** (`concept_ar`, optional) — rising `ce_intact_by_bin` past
  `window_k`, or growing `delta_shuffle_by_bin`, is the frozen-memory wall (E09 Stage-0
  kill gate); a flat curve falsifies the need for writable memory at no training cost.
- Always quote **one short text snippet** next to the metrics — numbers without a sample
  are not a vibe check.

Gates (interpretive, not hard STS-B-style thresholds unless the experiment spec sets them):
- High `rep_3` (≥ ~0.3–0.5) with collapsing length-binned distinct-1 → treat as a generation
  kill / strong regression signal for long-decode claims.
- `backbone_concept`: `real` must beat `zero`/`shuffle` on usable text *or* the concept
  ablation already failed in Tier 1 — do not claim long-context generation without both.
- Skip this tier for `perceiver_denoise` / `weighted_mlm` (no free-run decode path here).

### Tier 2 — Zero-shot STS-B (fast semantic gate, no fine-tuning)
```bash
uv run python evaluation/evaluate_on_benchmark.py \
  --benchmark stsb_zero_shot \
  --model_type concept_ar \
  --model_name_or_path "$BEST" --tokenizer_name "$BEST" \
  --batch_size 128 --max_length 128
```
Gate: E01 success `Pearson ≥ 0.62` (≥ prior best 0.607); E02 `≥ 0.65`. This is the cheapest
semantic-quality signal — compare against prior bests before any fine-tuning.

**Always anchor the number with the trivial floors** (otherwise STS-B is uninterpretable — a
mean of word embeddings already scores ~0.4–0.6). Run once per study (no checkpoint needed):
```bash
# bag-of-embeddings floor (model's tokenizer family) and frozen-teacher floor
uv run python evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot \
  --model_type concept_ar --model_name_or_path "$BEST" \
  --baseline token_embed_mean --baseline_model HuggingFaceTB/SmolLM2-135M
uv run python evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot \
  --model_type concept_ar --model_name_or_path "$BEST" \
  --baseline teacher_hidden_mean --baseline_model HuggingFaceTB/SmolLM2-135M
```
Interpret: if the model's STS-B is within ~0.05 of `token_embed_mean`, the concepts add ~nothing
over averaging. Reference **ceiling** (cited, not run): SimCSE-unsup ≈ 0.76, SBERT ≈ 0.84 Spearman.

### Tier 2.5 — Frozen-encoder readout probe (mean vs attention pooling)
The decisive test of whether information is **distributed across the C concepts**. Freeze the
encoder, train only a tiny head, and compare mean-pool vs a single-learned-query attention pool
on the pair tasks (STS-B-train / SICK / PAWS). If attention-pool ≫ mean-pool, the info is spread
across slots and mean-pool was hiding it; if they tie, the set is genuinely collapsed.
```bash
for POOL in mean attention; do
  uv run python evaluation/evaluate_on_benchmark.py --benchmark sick_relatedness \
    --model_type concept_ar --model_name_or_path "$BEST" --tokenizer_name "$BEST" \
    --freeze_encoder --pool_mode "$POOL"
done
```
Report the mean-vs-attention delta (the delta is the signal, not the absolute). Same flags work
on `evaluate_model_on_glue.py`.

### Tier 3 — Supervised pair tasks (expensive, run last)
SICK (relatedness + entailment):
```bash
MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=concept_ar TOKENIZER_NAME_OVERRIDE="$BEST" \
  bash scripts/evaluate_concept_encoder_sick.sh sick_all
```
PAWS (adversarial paraphrase — meaning vs word overlap):
```bash
MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=concept_ar TOKENIZER_NAME_OVERRIDE="$BEST" \
  bash scripts/evaluate_concept_encoder_paws.sh
```
GLUE semantic subset (`mrpc`, `stsb`, `qqp`, `mnli-matched`, `mnli-mismatched`):
```bash
MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=concept_ar TOKENIZER_NAME_OVERRIDE="$BEST" \
  bash scripts/evaluate_concept_encoder_glue.sh all          # or a single task, e.g. mrpc
```

> **GLUE is demoted as concept-content evidence (2026-06-15).** Full fine-tuning unfreezes the
> encoder and trains a head, so it re-routes *around* the bottleneck and measures fine-tuning
> capacity, not what the concepts store. Do **not** cite full-finetune GLUE as evidence that
> "concepts store information." If you want a GLUE number for concept quality, run it as the
> **frozen-encoder probe** (`--freeze_encoder`, Tier 2.5) — otherwise treat full-finetune GLUE as
> at most a downstream-utility footnote. STS-B/SICK/PAWS remain the cheap semantic gates.

Repeat the whole pipeline for `$LAST`, changing the output JSON / report labels.

## Outputs To Collect
- Shell log path, usually `Cache/logs/eval_<id>_<timestamp>.log`.
- Concept-analysis JSON (geometry + `concept_ablation` + `generation_samples`) in `Cache/Evaluation_reports/`.
- Generation-quality JSON (diversity metrics + free-run / continuation samples + optional
  `suffix_ce_by_position`) in `Cache/Evaluation_reports/<run_id>_*_generation_quality.json`.
- Benchmark CSVs in `Cache/Evaluation_reports/`.
- Compute-audit CSV + chart in `Cache/Evaluation_reports/compute_audit/` and `compute/*` scalars on the run's W&B summary (from the run-level preamble).
- W&B run URLs for STS-B / SICK / PAWS / GLUE.
- Checkpoint paths and `checkpoint_family` metadata.

For the handoff to `experiment-track`, report: best + last checkpoint paths; concept
effective rank and key collapse/diversity metrics; concept-ablation Δzero/Δshuffle (and the
no-concept floor); **generation vibe-check** (`distinct_1` / `rep_3` at key length cutoffs,
plus one short sample snippet — and for `backbone_concept`, real-vs-zero/shuffle condition
deltas); STS-B zero-shot Pearson/Spearman; SICK relatedness Pearson/Spearman;
SICK entailment accuracy; PAWS accuracy/F1; GLUE semantic-subset scores; **compute scalars**
(`compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens`,
`compute/loss_tokens_est` + `compute/audit_state`/`compute/flag` from the run-level preamble) —
this is a **hard precondition**: if `compute/audit_state` is absent the audit was never run, so
run the preamble now and do not hand off without it; and any failed task with its first traceback line.

## Interpreting Results
- **Within-sample concept RankMe** is the collapse gate (see Tier 1); slot-mean rank is only a
  diagnostic and cross-sample RankMe is an embedding-diversity number, not concept rank. Better
  geometry with near-random STS-B = diversity without semantics.
- STS-B is only interpretable next to the trivial floors (Tier 2). A number near
  `token_embed_mean` means the concepts add little over averaging.
- The mean-vs-attention probe delta (Tier 2.5) tells you whether information is *distributed*
  across concepts; a flat delta corroborates genuine collapse.
- Concept-ablation ΔCE is the E01/E02 *primary* signal: small Δ on reconstruction can mean
  "task too easy from left context", which is exactly why E02 (prefix→suffix) is the decisive
  semantic test — judge E01 ablation against its own threshold, not E02's.
- Generation vibe check (Tier 1.5) catches the *other* failure mode ablation misses: fluent
  local text that is a repetition loop. Collapsing `distinct_1` / rising `rep_3` past K is
  a generation kill even when CE looks fine. Always pair the numbers with one sample snippet.
- Zero-shot STS-B is the fastest semantic gate; clear it before expensive fine-tuning.
- If train loss improves but eval CE worsens, GLUE/SICK/PAWS become essential before calling
  a run useful.
- Partial spot-checks → partial conclusions; do not issue a track-wide verdict from one tier.

## Common Pitfalls
- Pass `--model_type concept_ar` directly now; do **not** resurrect the old
  `MODEL_TYPE_OVERRIDE=perceiver_denoise` workaround (the CLIs accept `concept_ar`, and the
  bash launchers forward `MODEL_TYPE_OVERRIDE=concept_ar`).
- Do not evaluate a tiny duplicate rank-artifact directory from distributed training.
- Do not run multiple fine-tuning suites on one GPU unless you set `CUDA_VISIBLE_DEVICES` per session.
- `run_concept_analysis.py` may abort during Python finalization *after* writing its JSON
  (`PyGILState_Release ... finalizing`). If the report and `--output_json` are complete,
  treat the metrics as usable and continue under the failure-tolerant wrapper.
- AR ablation / generation are wrapped in try/except inside the runner: if they error, the
  geometry report and JSON still complete — check the printed skip reason.
- `run_generation_quality.py` loads with `local_files_only=True` first; if the tokenizer
  fetch fails offline, retry after confirming the checkpoint directory is complete. For
  `backbone_concept`, `--concept_modes` other than `real` are ignored on `concept_ar`.
  Suffix-CE is `concept_ar`-only — other families print a skip and continue free-gen.
- A `trust_remote_code` warning that then loads via parquet is usually harmless.
- If a task fails, read the first traceback before rerunning; do not blindly relaunch.

## Failure-Tolerant Wrapper
For long best-vs-last suites, wrap each command so one failure does not discard later evidence:
```bash
FAILED=()
run_cmd() {
  local name="$1"; shift
  set +e
  "$@"
  local code=$?
  set -e
  [ "$code" -eq 0 ] || FAILED+=("$name:$code")
}
```
At the end, print `${FAILED[*]}` and inspect the first traceback for each failed task.

## Handoff
When the suite is done, hand the collected evidence to **`experiment-track`** to record the
verdict in `master_experiment_log.md`, flip the experiment spec `Status`/`Result`, and update
the `agenda.md` learnings. This skill owns *how to run*; `experiment-track` owns *what it means*.

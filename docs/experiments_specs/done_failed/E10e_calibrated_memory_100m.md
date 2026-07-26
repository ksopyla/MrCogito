# E10e — Calibrated concept memory at the original 100M budget

- **Status:** killed 2026-07-13 — full 100M run completed without persistent-memory utility
- **Serves:** E10's unresolved question of whether the fully calibrated recurrent-memory path develops persistent use only with the original 100M-token exposure
- **Implementation plan:** [E10e_calibrated_memory_100m_plan.md](E10e_calibrated_memory_100m_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-12 · closed 2026-07-13

> This is a configuration-only duration diagnostic. It does not reinterpret the failed E10d
> 25M gate as success: it asks whether E10d's largest, but still sub-threshold, recurrence signal
> compounds under the original E10 budget and its matched 1,449-step cosine schedule.

## Hypothesis

If E10d's small positive beyond-local signal is an early, cumulative memory-use trajectory rather
than noise, then training its complete calibrated configuration for the original 100M non-padding
token budget will raise both `delta_static_beyond` and `delta_shuffle_beyond` to at least 0.01
nats by the matched final checkpoint, while retaining local CE and healthy concept geometry.

## Builds-on

- **Foundation:** E10d's reusable backbone-concept path: `nn/backbone_concept_lm.py`,
  `training/train_concept_pretraining.py`, optional concept-memory AdamW groups, and
  `scripts/launch_e10.sh`.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42; fresh
  initialization, not a resume from E10d.
- **Baseline to beat:** E10d
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_173115`, whose 25M checkpoint gave
  `delta_static_beyond=+0.001103` and `delta_shuffle_beyond=+0.001161` with RankMe 108.23.
  E10's original 100M concept pilot
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847` ended at
  −0.000324 / −0.000378 respectively.

## The single change

Increase the E10d training budget from 50M to **100M non-padding target tokens**, restoring the
original E10 1,449-step schedule and the corresponding cosine-LR phase. Everything else equals
E10d: normalized concept reads, read/write gate initialization 0.01, concept-memory LR `3e-4`,
LoRA LR `1e-4`, AdamW, weight decay 0, seq 2048, K=512, C=128, manifest, effective batch 72,
and seed 42.

## Success criteria (set BEFORE running)

- At 100M tokens and positions ≥1024, both `delta_static_beyond` and
  `delta_shuffle_beyond` are ≥0.01 nats.
- Both deltas exceed E10d's 25M values by ≥0.005 nats.
- Local CE at positions <512 is within +0.02 nats of E10d at comparable exposure.
- Within-sample RankMe remains ≥38.4/128; centered RankMe is reported.
- The full 100M trajectory is stable: no non-finite loss/gradients and no three consecutive
  eval-loss rises.

## Kill criteria (set BEFORE running)

- At the matched ~50M-token checkpoint (approximately step 720), both beyond-local deltas remain
  <0.002 nats: stop rather than spend the remaining budget.
- Within-sample RankMe <19.2, non-finite loss/gradients, or three consecutive eval-loss rises.
- Local CE regresses by >0.05 nats at the first two evaluations.

## Plan

- **Data:** unchanged Gemma-tokenized `smollm3_inspired_2k_e05` manifest; seq 2048;
  `DataCollatorForCausalLM`.
- **Compute:** Odra (3× RTX 3090); approximately 19 GPU-h / 6.5 wall-clock hours at the observed
  E10 throughput, including periodic ablation evaluation.
- **Steps / epochs:** 100M target tokens; approximately 1,449 optimizer steps; warmup 50.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E10e READ_CONCEPT_NORM=true \
  READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  CONCEPT_MEMORY_LR=3e-4 LEARNING_RATE=1e-4 OPTIMIZER=adam \
  WEIGHT_DECAY=0 TARGET_TOKENS=100000000 WARMUP_STEPS=50 \
  SKIP_PRETOKENIZE=1 MAX_SEQ_LENGTH=2048 CONCEPT_NUM=128 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** none — configuration only.

## Result

- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506`
- WandB: [training run](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506)
- Run report: [e10e_calibrated_memory_100m_20260713.md](../../2_Experiments_Registry/run_reports/e10e_calibrated_memory_100m_20260713.md)
- Verdict: **killed** — stable, healthy geometry and lower CE, but at 100M
  `delta_static_beyond=+0.000962` and `delta_shuffle_beyond=+0.001613`, both below the
  0.01-nat success gate; the ~50M midpoint also met the pre-registered kill criterion.

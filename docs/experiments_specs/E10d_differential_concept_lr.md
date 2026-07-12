# E10d — Differential concept-memory learning rate

- **Status:** active — approved 2026-07-12; queued after E10c
- **Serves:** E10's concept-path update-scale mismatch: test whether newly initialized memory parameters need a higher AdamW LR than pretrained-backbone LoRA
- **Implementation plan:** [E10d_differential_concept_lr_plan.md](E10d_differential_concept_lr_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-12 · closed —

> E10d keeps E10c's normalized read and 0.01 gate initialization. The only changed
> training variable is the learning rate assigned to concept-memory parameters.

## Hypothesis

If E10c's memory path is connected but under-updated relative to LoRA, then training all
concept-memory parameters at `3e-4` while retaining LoRA at `1e-4` will improve both
`delta_static_beyond` and `delta_shuffle_beyond` by at least 0.005 nats over E10c at 50M
tokens, without causing local CE regression or concept-rank collapse.

## Builds-on

- **Foundation:** E10c configuration: read-side RMSNorm enabled and read/write gate init 0.01.
- **Init / checkpoint:** fresh `google/gemma-3-1b-pt` + LoRA r=16, seed 42; not a resume.
- **Baseline to beat:** completed E10c 50M checkpoint and its recurrence deltas, local CE,
  RankMe, and parameter update/weight ratios. Exact run id/values are filled before launch.

## The single change

Set `CONCEPT_MEMORY_LR=3e-4` while the ordinary trainer `LEARNING_RATE=1e-4` remains the
LoRA LR.

The concept-memory group contains:

- `concept_init`
- `write_head.*` (BiXT, write norms, sandwich norm, and write gate)
- global-layer read gates
- enabled `read_branch.concept_norm.*`

All LoRA A/B parameters remain at `1e-4`. Weight decay stays zero for this experiment.

## Success criteria (set BEFORE running)

- **Primary:** at 50M tokens,
  `delta_static_beyond ≥ max(0.01, E10c + 0.005)`.
- **Content attribution:** `delta_shuffle_beyond ≥ max(0.01, E10c + 0.005)`.
- **Optimization evidence:** W&B records the effective concept-memory LR as `3e-4` and LoRA LR
  as `1e-4`; the pre-launch first-step probe confirms finite gradients in both groups.
- **No local regression:** local CE is within +0.02 nats of E10c at matched exposure.
- **Geometry guard:** within-sample RankMe remains ≥38.4/128.

## Kill criteria (set BEFORE running)

- At ~25M tokens, both recurrence deltas improve by <0.002 nats over E10c.
- Local CE regresses by >0.05 nats at the first two evaluations.
- Non-finite loss/gradients, within-sample RankMe <19.2, or three consecutive eval-loss rises.

## Plan

- **Data:** unchanged E10c Gemma manifest, seq 2048.
- **Compute:** Odra (3× RTX 3090), approximately 9.5 GPU-h / 3.2 wall-clock hours for 50M.
- **Steps / epochs:** approximately 725 optimizer steps; exact value from manifest stats.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E10d READ_CONCEPT_NORM=true \
  READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  CONCEPT_MEMORY_LR=3e-4 LEARNING_RATE=1e-4 OPTIMIZER=adam \
  TARGET_TOKENS=50000000 WARMUP_STEPS=50 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** optional AdamW concept-memory parameter groups in the shared trainer,
  preserving the existing HF optimizer path when unset. Muon rejects this option explicitly.

## Result

<Filled in AFTER, by experiment-track.>
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

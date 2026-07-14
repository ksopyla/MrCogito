# E10c — Small nonzero memory gates

- **Status:** killed 2026-07-12 at the pre-registered ~25M-token gate
- **Serves:** E10's null recurrent-memory diagnosis: test whether the serial zero-gate bootstrap prevents the normalized memory path from learning
- **Implementation plan:** [E10c_nonzero_memory_gates_plan.md](E10c_nonzero_memory_gates_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-12 · closed 2026-07-12

> E10c is cumulative but attributable: it keeps E10b's normalized concept read and changes
> only the gate-initialization policy. Data, objective, backbone, LoRA, optimizer, LR, and
> budget remain fixed.

## Hypothesis

If E10b's normalized read remains limited by the serial read/write bootstrap, then initializing
both bounded memory gates at `0.01` instead of exact zero will make the recurrent write/read
parameters receive useful gradients from the first update and improve beyond-local recurrence
by at least 0.005 nats over E10b at 50M tokens, without materially perturbing local Gemma CE.

## Builds-on

- **Foundation:** the E10b shared foundation with `READ_CONCEPT_NORM=true`;
  `BackboneConceptConfig.read_gate_init/write_gate_init`,
  `GlobalLayerWithConceptRead.gate`, and `ConceptWriteHead.alpha`.
- **Init / checkpoint:** fresh `google/gemma-3-1b-pt` + LoRA r=16, seed 42. This is not a
  resume from E10b; only the configuration is cumulative.
- **Baseline to beat:** the completed E10b 50M checkpoint and its
  `concept_ablation/delta_static_beyond`, `delta_shuffle_beyond`, local CE, and RankMe.
  The exact run id/values are filled before E10c launches.

## The single change

Change the **memory gate initialization policy** from fully closed to small-live:

```text
read_gate_init:  0.0 → 0.01  (all four global-layer read gates)
write_gate_init: 0.0 → 0.01  (the recurrent BiXT write gate)
```

Both are one coordinated policy for the serial memory path. The `tanh` parameterization remains,
so the initial effective residual scale is approximately 1%.

Everything else equals E10b: read-side RMSNorm enabled, seq 2048, K=512, C=128, current Gemma
manifest, plain next-token CE, AdamW, one LR of 1e-4, weight decay 0, effective batch 72, seed 42.

## Success criteria (set BEFORE running)

- **Primary:** at 50M tokens,
  `delta_static_beyond ≥ max(0.01, E10b + 0.005)`.
- **Content attribution:** `delta_shuffle_beyond ≥ max(0.01, E10b + 0.005)`.
- **Immediate gradient reach:** in the pre-launch gradient probe, `concept_init`, BiXT write
  projections, read RMSNorm gains, and LoRA-B all have finite nonzero gradients on the first
  backward pass.
- **No local regression:** local CE is within +0.02 nats of E10b at the matched checkpoint.
- **Geometry guard:** within-sample RankMe remains ≥38.4/128.

## Kill criteria (set BEFORE running)

- At ~25M tokens, both recurrence deltas improve by <0.002 nats over E10b.
- Local CE regresses by >0.05 nats at the first two evaluations.
- Within-sample RankMe <19.2, non-finite loss/gradients, or three consecutive eval-loss rises.

## Plan

- **Data:** unchanged E10b Gemma manifest, seq 2048.
- **Compute:** Odra (3× RTX 3090), approximately 9.5 GPU-h / 3.2 wall-clock hours for 50M.
- **Steps / epochs:** approximately 725 optimizer steps; exact value from manifest stats.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E10c READ_CONCEPT_NORM=true \
  READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  TARGET_TOKENS=50000000 WARMUP_STEPS=50 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** expose the two existing config fields through the shared
  args/factory/launcher and log them to W&B. Defaults remain zero.

## Result

- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_153028`
- WandB: [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_153028)
- Run report: no dedicated report; canonical summary is in `agenda.md`
- Verdict: **killed** — at the ~25M gate, beyond-local static and shuffle deltas were
  only +0.000426/+0.000353 nats, less than 0.002 above E10b and below the 0.01
  success floor. Geometry remained healthy (RankMe 115.1; centered 125.1) and eval CE
  was 1.8154, so small-live gates alone did not recover persistent memory use.

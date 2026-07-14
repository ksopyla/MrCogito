# E16 — Shared depth-recurrent concept workspace (E13 variant)

- **Status:** done / failed mechanism gate — 50M pilot completed 2026-07-14
- **Serves:** the post-E10 platform pivot toward concepts that are formed, repeatedly refined, and carried as one coherent latent reasoning state
- **Implementation plan:** [E16_shared_depth_recurrent_concepts_plan.md](E16_shared_depth_recurrent_concepts_plan.md) *(authored after this spec is approved)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-13 · closed 2026-07-14

> E16 is the smaller, shared-state variant of E13. It does not create 26 independent
> layer memories. It changes one variable versus E10e: **when the one shared concept
> state is updated**. The concept read interface, backbone, objective, data, concept
> count, block size, LoRA policy, optimization, and evaluation protocol stay fixed.
>
> This experiment tests a prerequisite for recursive concept reasoning: whether one
> coherent concept workspace can survive repeated depth-wise refinement and remain
> causally useful across token blocks. It does not yet add extra test-time reasoning
> iterations or reasoning-trace supervision.

## Hypothesis

If E10e's one shared concept state is updated with one tied write operator after each
of Gemma's four concept-reading layers, rather than only once after the complete
26-layer block, then the minimum of `delta_static_beyond` and
`delta_shuffle_beyond` will reach at least **0.01 nats at 50M tokens**, because later
layers can read concepts already refined by earlier depths instead of every depth
reading the same stale state.

## Builds-on

- **Foundation:** E10e's reusable `BackboneConceptLM` path in
  `nn/backbone_concept_lm.py`, the canonical
  `training/train_concept_pretraining.py` entrypoint, and `scripts/launch_e10.sh`.
  The new mode is a reusable config value on the same model family, never a new
  training or model fork.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42,
  with E10e's read RMSNorm, 0.01 read/write gate initialization, and differential
  concept-memory LR. This is a fresh matched initialization, not a resume.
- **Baseline to beat:** E10e
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506`. At its matched
  ~50M checkpoint (step 720), `delta_static_beyond=+0.00033` and
  `delta_shuffle_beyond=+0.00186`; at 100M they were only
  `+0.000962/+0.001613`.

## The single change

Change the **shared concept-state update schedule**:

```text
E10e:
  all four global layers read the same z_b
  z_b is written once after the complete 26-layer block

E16:
  layer 6  reads z_b   → tied write → z_b^(6)
  layer 12 reads z_b^(6)  → tied write → z_b^(12)
  layer 18 reads z_b^(12) → tied write → z_b^(18)
  layer 24 reads z_b^(18) → tied write → z_b^(24)
  z_(b+1) = z_b^(24)
```

There is still exactly one state `z [B,128,1152]`. The existing
`ConceptWriteHead` weights are shared across all four update points; each point has
its own scalar write gate so utilization can be measured. E10e's final post-block
write is disabled in this mode. Concept reads remain at the same four Gemma layers
and use the same native Q/K/V/O projections as E10e.

Everything else is held fixed: frozen Gemma-3-1B, LoRA targets/rank, raw causal-LM
objective, dataset manifest, sequence length 2048, K=512, C=128, one-block token
carry, read normalization, learning rates, optimizer, effective batch, seed, and
recurrent-state ablations.

## Success criteria (set BEFORE running)

- **Primary causal-use gate:** at 50M non-padding tokens and positions ≥1024,
  `min(delta_static_beyond, delta_shuffle_beyond) >= 0.01` nats.
- **Improvement over E10e:** each of those two deltas improves over E10e's matched
  ~50M value by at least 0.005 nats.
- **Depth utilization:** at least three of four depth-write gates have
  `|tanh(alpha)| >= 0.005`; all four read-gate magnitudes are reported.
- **No local regression:** real-state CE at positions <512 is no more than
  +0.02 nats above E10e at matched exposure.
- **Geometry guard:** final shared-state within-sample RankMe remains ≥38.4/128;
  centered RankMe is reported.
- **Stability:** no non-finite loss/gradients and no three consecutive eval-loss
  rises.

## Kill criteria (set BEFORE running)

- At the ~25M checkpoint, stop if both `delta_static_beyond` and
  `delta_shuffle_beyond` are ≤0.002 nats.
- Stop if fewer than two of four depth-write gates have
  `|tanh(alpha)| >= 0.005` at 25M tokens.
- Stop if within-sample RankMe falls below 19.2/128, local CE regresses by more
  than 0.05 nats at two consecutive evaluations, or loss/gradients become
  non-finite.
- Do not extend beyond 50M unless every success criterion is met.

## Plan

- **Architecture:** frozen Gemma-3-1B, 26 decoder layers, hidden/concept dimension
  1152; C=128 shared concepts; K=512 token blocks; concept reads and tied writes at
  human-numbered layers 6/12/18/24.
- **Tokenizer:** the `google/gemma-3-1b-pt` tokenizer, unchanged from E10e.
- **Data:** unchanged Gemma-tokenized `smollm3_inspired_2k_e05` immutable manifest;
  raw causal-LM objective at sequence length 2048.
- **Optimization:** E10e settings: LoRA r=16 over q/k/v/o, LoRA LR 1e-4,
  concept-memory LR 3e-4, AdamW, weight decay 0, read RMSNorm, read/write gate
  initialization 0.01, effective batch 72, seed 42.
- **Compute:** Odra, 3× RTX 3090; calibrate microbatch first because four
  differentiable writes increase activation memory. Expected upper bound is
  approximately 12–18 GPU-hours for the 50M pilot.
- **Steps / epochs:** 50M non-padding target tokens, approximately 725 optimizer
  steps, with a mandatory ~25M decision checkpoint and warmup 50.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E16 CONCEPT_IO_MODE=shared_depth_recurrent \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  CONCEPT_MEMORY_LR=3e-4 LEARNING_RATE=1e-4 OPTIMIZER=adam \
  WEIGHT_DECAY=0 TARGET_TOKENS=50000000 WARMUP_STEPS=50 \
  SKIP_PRETOKENIZE=1 MAX_SEQ_LENGTH=2048 CONCEPT_NUM=128 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** add the config-selectable
  `concept_io_mode="shared_depth_recurrent"` path to
  `nn/backbone_concept_lm.py`, with tied interleaved writes, four write gates,
  checkpoint round-tripping, per-depth gate/update telemetry, and existing
  real/static/zero/shuffle/one-block ablation compatibility. Extend the shared
  argument/config/launcher plumbing and tests; do not add an E16-specific training
  script.

## Result

- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260714_075403`
- WandB: [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260714_075403)
- Run report: no dedicated report yet; W&B is the source for this lifecycle update
- Verdict: **failed mechanism gate** — the 50M run finished with healthy geometry
  (within-sample RankMe 62.2; centered 125.0) and eval CE 1.8122, but final
  beyond-local static/shuffle deltas were only +0.000499/+0.001018 nats. Both are
  below the 0.01 success gate, so depth-wise shared-state updates did not establish
  persistent causal concept use.

# E16a — Shared depth-recurrent workspace optimizer A/B at 100M

- **Status:** active — Muon calibration passed; unattended Adam→Muon pair launched on Odra 2026-07-14
- **Serves:** test the user's compute/optimization hypothesis for E16 before changing context length or data
- **Implementation plan:** [E16a_muon_optimizer_ab_plan.md](E16a_muon_optimizer_ab_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-14 · closed —

> E16a is a matched 2K, 100M-token optimizer comparison. Both arms use the
> implemented E16 architecture and the same fresh initialization, data order, token
> budget, and evaluation schedule. The single research variable is the practical
> optimizer recipe: calibrated differential AdamW versus stabilized Muon.
>
> Context length and data remain unchanged. A separate E16b may scale the winning
> optimizer to 4K and 500M/1B tokens with a long-document mix.

## Hypothesis

At 100M non-padding tokens, stabilized Muon will produce a larger causally attributable
shared-concept signal than calibrated AdamW on E16—specifically improving
`min(delta_static_beyond, delta_shuffle_beyond)` by at least 0.005 nats—because its
orthogonalized matrix updates may train the tied concept writer and LoRA attention
projections more efficiently, without collapsing concept geometry.

## Builds-on

- **Foundation:** E16's `BackboneConceptLM` with
  `concept_io_mode="shared_depth_recurrent"` in `nn/backbone_concept_lm.py`;
  `PerceiverDenoiseTrainer` optimizer selection in
  `training/concept_pretraining_trainer.py`; the canonical
  `training/train_concept_pretraining.py` →
  `scripts/train_concept_pretraining_multigpu.sh` route; and the thin E10 protocol
  wrapper.
- **Init / checkpoint:** both arms start fresh from frozen
  `google/gemma-3-1b-pt` + LoRA r=16, seed 42. They do not resume E16, so optimizer
  state and LR schedules are matched from token zero within each recipe.
- **Historical reference:** E16 run
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260714_075403` at 50M:
  `delta_static_beyond=+0.000499`, `delta_shuffle_beyond=+0.001018`,
  within-sample RankMe 62.2, eval loss 1.8122.
- **Matched baseline to beat:** E16a Adam arm at the same 100M exposure. Its run id
  and final metrics are filled into the result ledger after launch.

## The single change

Change only the optimizer recipe between two fresh, token-matched arms:

```text
Adam control:
  optimizer=adam
  LoRA LR=1e-4
  concept-memory LR=3e-4
  weight decay=0

Muon treatment:
  optimizer=muon
  matrix LR=0.01
  AdamW-fallback LR=2e-4
  momentum=0.95
  weight decay=0.1
  concept-memory differential LR unset (Muon routes by tensor shape)
```

The Muon LR, fallback LR, momentum, and weight decay are one practical optimizer
recipe, not separately interpreted variables. The Muon recipe must first pass a
short sustained-LR stability calibration; if 0.01 fails only the pre-registered
fallback `matrix LR=0.005` may be used, and the selected value must be recorded.

Held fixed: E16 architecture and gates, C=128, H=1152, four tied depth writes at
Gemma layers 6/12/18/24, K=512, sequence length 2048, raw causal-LM objective,
Gemma tokenizer, immutable `smollm3_inspired_2k_e05` manifest, LoRA targets/rank,
effective batch 72, seed 42, 100M non-padding tokens, warmup fraction, cosine
schedule, checkpoint/eval cadence, and all concept ablations.

## Success criteria (set BEFORE running)

- **Primary optimizer lift:** at 100M and positions ≥1024,
  `min(delta_static_beyond, delta_shuffle_beyond)` for Muon is at least **0.005
  nats above** the matched Adam arm.
- **Absolute mechanism gate:** Muon reaches
  `min(delta_static_beyond, delta_shuffle_beyond) >= 0.01` nats.
- **Geometry:** Muon final within-sample RankMe is ≥38.4/128 and no more than 20%
  below Adam; centered RankMe is reported.
- **Depth utilization:** at least three of four Muon write gates and three of four
  read gates have magnitude ≥0.005.
- **Optimization:** Muon eval loss is no more than +0.02 nats above Adam at matched
  exposure; no non-finite loss or gradients.

The primary and absolute mechanism gates are co-required before selecting Muon for
E16b. Lower eval loss alone is not success.

## Kill criteria (set BEFORE running)

- **Muon sustained-LR calibration:** stop immediately on non-finite loss/gradient;
  stop if grad norm exceeds 10 for three consecutive logs, eval loss rises by
  >0.10 nats from its running minimum, or within-sample RankMe falls below 19.2.
  Retry once at the pre-registered matrix LR 0.005; no further tuning in E16a.
- **Full arms:** stop on non-finite loss/gradient, three consecutive eval-loss
  rises totaling >0.05 nats, or within-sample RankMe <19.2 at two consecutive
  evaluations.
- Do not select Muon for E16b if it misses either causal-use success gate, even if
  it lowers CE.

## Plan

- **Architecture:** unchanged E16: frozen 26-layer Gemma-3-1B, H=1152, C=128,
  K=512, one shared recurrent state, concept reads and tied writes at human-numbered
  layers 6/12/18/24.
- **Data:** unchanged immutable Gemma-tokenized
  `smollm3_inspired_2k_e05` train/eval manifest, sequence length 2048. This keeps
  data and context out of the optimizer comparison.
- **Compute:** Odra, 3× RTX 3090. Run the Adam arm first and Muon arm second to
  avoid GPU contention. Expected total is approximately 35–45 GPU-hours plus Muon
  calibration.
- **Steps / epochs:** 100M non-padding tokens per arm, approximately 1,449 optimizer
  steps at effective batch 72; warmup 100 steps; evaluations/checkpoints at each
  10% budget interval.
- **Muon calibration:** fresh short run with
  `LR_SCHEDULER_TYPE=constant_with_warmup`, matrix LR 0.01, fallback LR 2e-4,
  weight decay 0.1, and enough steps to sustain peak LR beyond warmup. Calibration
  is stability-only and is not included in the A/B result.
- **Launch:**
  ```bash
  # Adam arm
  EXPERIMENT_ID=E16a OPTIMIZER=adam \
  TARGET_TOKENS=100000000 bash scripts/launch_e16a.sh

  # Muon arm, after Adam completes
  EXPERIMENT_ID=E16a OPTIMIZER=muon \
  TARGET_TOKENS=100000000 bash scripts/launch_e16a.sh
  ```
- **New foundation code:** no model/trainer changes. Add only a thin E16a protocol
  wrapper and unattended sequential pipeline over the existing generic launcher;
  add parameter-flow tests to pin the two optimizer recipes.

## Result

<Filled in AFTER, by experiment-track.>
- Adam run id: —
- Muon run id: —
- WandB: —
- Run report: —
- Verdict: —

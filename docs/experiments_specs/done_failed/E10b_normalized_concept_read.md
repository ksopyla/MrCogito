# E10b — Normalized concept read interface

- **Status:** killed 2026-07-12 at the pre-registered ~25M-token gate
- **Serves:** E10's null recurrent-memory diagnosis: repair the measured concept-value scale mismatch without changing Gemma, the objective, data, sequence length, or optimizer
- **Implementation plan:** [E10b_normalized_concept_read_plan.md](E10b_normalized_concept_read_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-12 · closed 2026-07-12

> One experiment = one hypothesis = one changed variable. The broader follow-up ideas
> (4K training, a new long-document mix, nonzero gates, differential concept LR, and Muon)
> are deliberately excluded so this run can identify whether read-side normalization fixes
> the measured scale starvation.

## Hypothesis

If E10's null recurrence is primarily caused by feeding low-RMS concept values directly into
Gemma's frozen value projections, then applying a dedicated RMSNorm to `z` before the concept
read's `k_proj` and `v_proj` will make recurrent state content measurably useful by 50M tokens
(`static CE − recurrent CE ≥ 0.01` and `Δshuffle ≥ 0.01` at positions ≥1024), because the
read branch will receive a transformer-scale value stream while preserving the same fixed-memory
O(C·N) architecture and plain causal-LM objective.

## Builds-on

- **Foundation:** `nn/backbone_concept_lm.py:BackboneConceptLM`,
  `ConceptReadBranch`, `GlobalLayerWithConceptRead`, and the shared
  `training/train_concept_pretraining.py` →
  `scripts/train_concept_pretraining_multigpu.sh` → `scripts/launch_e10.sh` path.
- **Init / checkpoint:** fresh `google/gemma-3-1b-pt` backbone initialization, frozen base
  weights + LoRA r=16, seed 42. This is a fresh matched pilot, not a resume from the collapsed
  optimization state.
- **Baseline to beat:** E10 concept pilot
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847` at the matched
  ~50M-token checkpoint (`checkpoint-720`): all beyond-local recurrent-state ablations
  remained <0.001 nats; final 100M values were `Δshuffle=-0.00038`,
  `static−real=-0.00032`.

## The single change

Set `READ_CONCEPT_NORM=true`: before the four global-layer concept reads, transform
`z [B,C,H]` with a dedicated RMSNorm and feed the normalized state to both `k_proj` and
`v_proj`.

Everything else is held fixed from the E10 pilot: seq 2048, K=512, C=128,
`smollm3_inspired_2k_e05` Gemma manifest, raw causal text, plain next-token CE,
AdamW, LR 1e-4, weight decay 0, LoRA r=16 q/k/v/o, zero-init read/write gates,
effective batch 72, seed 42. In particular, this experiment does **not** change
concept initialization, gate initialization, parameter-group LRs, chat formatting,
data mix, or optimizer family.

## Success criteria (set BEFORE running)

- **Primary recurrence gate:** at 50M tokens and positions ≥1024,
  `static CE − recurrent CE ≥ 0.01 nats`.
- **Content attribution:** at the same checkpoint/positions,
  `Δshuffle = shuffled CE − real CE ≥ 0.01 nats`.
- **No local regression:** real concept-arm CE at positions <512 is no more than
  +0.02 nats above the original E10 pilot's matched checkpoint.
- **Geometry guard:** within-sample RankMe remains ≥0.3·C = 38.4; centered RankMe is
  reported alongside it.

## Kill criteria (set BEFORE running)

- At the ~25M-token checkpoint, both `static−real ≤ 0.002` and
  `Δshuffle ≤ 0.002` at positions ≥1024: stop rather than spend the full 50M.
- Within-sample RankMe <0.15·C = 19.2, non-finite loss/gradients, or eval CE rises
  across three consecutive evals.
- Local CE regression >0.05 nats at positions <512 at the first two evaluations.

## Plan

- **Data:** unchanged Gemma-tokenized `smollm3_inspired_2k_e05`, seq 2048,
  existing pretokenized manifest and `DataCollatorForCausalLM`.
- **Compute:** Odra (3× RTX 3090); estimated ~9.5 GPU-h / ~3.2 wall-clock hours for
  50M tokens, with the 25M kill checkpoint at approximately half that cost.
- **Steps / epochs:** exact target derived by `manifest_token_stats.py`; approximately
  725 optimizer steps at effective batch 72. Preserve automatic ~10% checkpoints.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E10b READ_CONCEPT_NORM=true \
  TARGET_TOKENS=50000000 WARMUP_STEPS=50 \
  SKIP_PRETOKENIZE=1 bash scripts/launch_e10.sh
  ```
- **New foundation code:** one backward-compatible, config-selectable read normalization
  option in `nn/backbone_concept_lm.py`, wired through the shared args/factory/launcher.
  Default `false` preserves E10 and old checkpoints.

## Result

- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258`
- WandB: [training run](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258)
- Run report: [e10b_normalized_concept_read_20260712.md](../../2_Experiments_Registry/run_reports/e10b_normalized_concept_read_20260712.md)
- Verdict: **killed** — at step 360, beyond-local static−real was +0.000371 and Δshuffle +0.000179, both below the 0.002-nat kill threshold; geometry and local CE remained healthy, so read-side RMSNorm alone did not repair recurrent usage.

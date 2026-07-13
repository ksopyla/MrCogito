# E15 — Supervision-calibrated delayed recall

- **Status:** active
- **Serves:** establish that the E14 forced-recall task is learnable before using it to judge the E10e recurrent memory interface
- **Implementation plan:** [E15_supervision_calibrated_delayed_recall_plan.md](E15_supervision_calibrated_delayed_recall_plan.md) *(authored after spec approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-07-13 · closed —

> One experiment = one hypothesis = one changed variable. The spec is frozen once
> training starts; results belong in the registry and run report.

## Hypothesis
If E14 failed because 984 answer labels were insufficient rather than because the unchanged
E10e memory path is incapable, then continuing the same checkpoint to 12,000 supervised
answer labels will raise held-out block-2 explicit-carry accuracy to at least 80% and will make
real recurrent memory beat static, zero, and conflicting-donor memory by at least 0.10 nats on
held-out block-4 answers.

## Builds-on
- **Foundation:** unchanged `nn/backbone_concept_lm.py:BackboneConceptLM` with E10e's C=128,
  K=512 `global_kv` read/write path; unchanged E14 deterministic dataset, sparse-label
  collator, evaluator, and generic `training/train_concept_pretraining.py` entrypoint through
  `scripts/launch_e10.sh`.
- **Init / checkpoint:** resume E14 checkpoint
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219/checkpoint-164`, including its
  optimizer state. The checkpoint has seen 984 supervised answers and 2,015,232 input tokens.
- **Baseline to beat:** E14 run
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219`: held-out block-2 accuracy
  1.95% (chance 1.56%); block-4 static/zero/donor CE margins
  +0.00182/+0.00350/+0.00237 nats; block-4 real accuracy 0.78%.

## The single change
Increase **supervised answer exposure before the architecture decision** from E14's
984-label/2.015M-input-token gate to 12,000 labels (24.576M total input tokens, 2,000 total
optimizer steps at effective batch 6).

The model, checkpoint state, dataset rows, one-answer-per-row masking, train/eval splits,
architecture, LoRA rank, gate initialization, optimizer state, base learning rates, effective
batch, schedule family, and all causal ablations remain unchanged. The cosine horizon extends
mechanically to the larger frozen budget. The gate is explicitly budgeted in supervised labels;
input tokens are only the derived compute cost.

## Success criteria (set BEFORE running)
- **Required task positive control:** at 12,000 total supervised labels, real-memory top-1
  accuracy on the 256 held-out block-2 counterfactual pairs is **≥80%**. The fact remains in
  Gemma's explicit K=512 token carry in this view, so this establishes task competence without
  requiring recurrent memory.
- **Primary memory result:** on at least 2,048 held-out block-4 counterfactual pairs, each
  paired margin `CE(static) − CE(real)`, `CE(zero) − CE(real)`, and
  `CE(conflicting donor) − CE(real)` is **≥0.10 nats**, with paired document-bootstrap
  95% confidence-interval lower bound **>0**.
- **Memory task competence:** block-4 real-memory answer accuracy is **≥50%**, while static,
  zero, and conflicting-donor accuracy each remain **≤20%**.

Passing all criteria means the E10e mechanism is trainable under a supervision-calibrated
forced-use signal. It does not establish transfer to natural long-context language modeling.

## Kill criteria (set BEFORE running)
- **Midpoint task gate:** pause at 6,000 total labels (step 1,000 / 12.288M total input
  tokens). Kill the objective protocol if held-out block-2 real-memory accuracy is **<20%**.
  Otherwise resume to the frozen 12,000-label ceiling.
- **Final task gate:** at 12,000 labels, kill the objective protocol if held-out block-2
  accuracy is **<80%**. In that case no memory-interface conclusion is allowed.
- **Architecture gate:** if block-2 accuracy is ≥80% but all three block-4 memory-margin point
  estimates are **<0.01 nats**, kill E10e's global→concept interface for this forced-recall
  task. Use the block-3 memory-age view only to route the next interface experiment; do not
  extend E15.
- Stop immediately for non-finite loss, repeated OOM after one safe batch reduction, data
  validation failure, or checkpoint-resume mismatch.

## Plan
- **Data:** reuse E14's immutable Gemma-tokenized manifest and splits exactly: 4,608
  2,048-token train rows with one answer label per row; 2,048 held-out block-4
  counterfactual pairs; 256-pair held-out block-2/3 diagnostic views. Repeated train exposure
  is allowed; held-out key/value assignments remain train-disjoint.
- **Compute:** Odra, 3× RTX 3090; approximately 3 GPU-hours total based on E14's measured
  throughput, including the midpoint pause and causal evaluations.
- **Steps / epochs:** resume at global step 164; pause at step 1,000 (6,000 total labels);
  conditionally continue to step 2,000 (12,000 total labels / approximately 2.60 train
  epochs). No extension beyond step 2,000.
- **Launch:** direct env-var configuration over `bash scripts/launch_e10.sh`, with the exact
  command frozen in the implementation plan; no experiment-specific training script.
- **New foundation code (if any):** none — config-only continuation using the implemented E14
  data, sparse-label, and causal-evaluation foundation.

## Result
Pending.
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

# E14 — Forced delayed-recall concept memory

- **Status:** draft — approved 2026-07-13; freezes active at launch
- **Serves:** determine whether E10e's recurrent global→concept mechanism can learn and causally use information that has no local-token bypass before changing the memory interface
- **Implementation plan:** [E14_forced_delayed_recall_memory_plan.md](E14_forced_delayed_recall_memory_plan.md) *(authored after spec approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-07-13

> One experiment = one hypothesis = one changed variable. The spec is frozen once
> training starts; results belong in the registry and run report.

## Hypothesis
If the unchanged E10e global→concept architecture is trained with answer-only CE on
counterfactual four-block delayed key/value recall, then real recurrent memory will lower
held-out answer-token CE by at least 0.10 nats versus static, zero, and conflicting-donor
memory, because paired examples have identical local query context but incompatible answers
whose only identifying evidence occurs in block 1.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py:BackboneConceptLM` with E10e's C=128,
  K=512 `global_kv` read/write path; `training/train_concept_pretraining.py` and the generic
  launcher via `scripts/launch_e10.sh`; no model or training-entrypoint fork.
- **Init / checkpoint:** fresh `google/gemma-3-1b-pt` backbone initialization with LoRA r16;
  E10e calibration retained (`read_concept_norm=true`, read/write gate init 0.01,
  task LR 1e-4, concept-memory LR 3e-4). Do not resume E10e's exhausted optimizer schedule.
- **Baseline to beat:** E10e
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506`, whose final beyond-local
  static−real and shuffle−real CE margins were only +0.000962 and +0.001613 nats.

## The single change
Replace E10e's plain natural-text next-token objective with a **forced distant-information
signal**: deterministic synthetic 2,048-token examples place a randomly paired key/value fact
in block 1, two full 512-token distractor blocks between fact and query, and a query plus
single-token answer in block 4. Only the answer token contributes CE.

Each held-out evaluation item has a counterfactual twin with the same key and byte-identical
blocks 2–4 through the token immediately before the answer, but a different block-1 value and
target. Thus local
tokens, position, and query form cannot identify the answer. All E10e architecture and
optimizer hyperparameters remain fixed apart from the smaller effective batch used to obtain
enough optimizer updates inside the diagnostic token budget.

## Success criteria (set BEFORE running)
- **Primary, held-out block-4 answer CE:** each paired margin
  `CE(static) − CE(real)`, `CE(zero) − CE(real)`, and
  `CE(conflicting donor) − CE(real)` is **≥0.10 nats**, with a paired document-bootstrap
  95% confidence-interval lower bound **>0**.
- **Task competence:** real-memory single-token answer top-1 accuracy is **≥50%** on at least
  2,048 held-out counterfactual pairs. The value vocabulary is frozen at dataset-build time
  and contains at least 64 equiprobable, tokenizer-verified single-token values (chance ≤1.6%).
- **Leakage/causality audit:** static, zero, and conflicting-donor answer accuracy each remain
  **≤20%**; counterfactual twins are train-disjoint and byte-identical after block 1 through
  the answer boundary.

Passing means the E10 mechanism is trainable and the natural plain-CE signal was insufficient;
it does **not** establish transfer to natural long-context language modeling.

## Kill criteria (set BEFORE running)
- At the first checkpoint after **2M non-padding input tokens**, stop if **all three**
  held-out point-estimate CE margins (static, zero, donor versus real) are **<0.01 nats**.
- Stop immediately for non-finite loss, repeated OOM after one safe batch reduction, or
  counterfactual/leakage validation failure.
- At the full budget, kill the E10 interface for this task if any primary 0.10-nat margin or
  the 50% real-memory accuracy gate is unmet. Do not extend training to rescue a failed gate.

## Plan
- **Data:** generated, immutable Gemma-tokenized delayed-recall dataset: 4,608 train examples
  (9.44M input tokens) plus at least 2,048 held-out counterfactual pairs; N=2,048 exactly,
  K=512, query in block 4, answer-only labels. Additional held-out query-at-block-2/3 views
  diagnose memory decay but are not success gates.
- **Compute:** Odra, 3× RTX 3090; expected <5 GPU-hours including the 2M-token gate and full
  run, subject to measured throughput.
- **Steps / epochs:** one pass over 4,608 train rows; effective batch 6
  (`2/GPU × 3 GPUs × accumulation 1`) gives 768 optimizer steps and 9.44M input tokens.
  Run the causal evaluation at the first checkpoint after ~2M tokens and at completion.
- **Launch:** first build the frozen dataset/manifest with the command in the implementation
  plan, then `bash scripts/launch_e14.sh`; the thin wrapper pins the approved values and
  delegates through `scripts/launch_e10.sh` to the shared generic trainer.
- **New foundation code (if any):** reusable deterministic delayed-recall dataset builder;
  opt-in preservation of precomputed label masks in `DataCollatorForCausalLM`; paired
  answer-token memory-attribution evaluator with real/static/zero/conflicting-donor modes.
  No E14-specific model class or training entrypoint.

## Result
*Pending.*
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

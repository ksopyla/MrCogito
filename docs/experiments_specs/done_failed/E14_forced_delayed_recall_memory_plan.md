# E14 — Forced Delayed-Recall Concept Memory: Implementation Plan

- **Spec:** [E14_forced_delayed_recall_memory.md](E14_forced_delayed_recall_memory.md) · **Status:** implemented and run; killed at the 2026-07-13 gate
- **Authored by:** `implementation-plan` · for → `research-implement`

> Implement the approved spec's one change: replace E10e's natural plain-CE signal
> with counterfactual delayed recall. The real model path and E10e parameters stay unchanged.

## 1. Source & fit
- **Origin:** E10/E10b–e result sequence in
  [the registry](../../2_Experiments_Registry/master_experiment_log.md): healthy concept geometry
  and improved local CE never produced more than 0.0016 nats of persistent-memory attribution.
- **Synthesis verdict:** **Adapt** — retain the calibrated E10e recurrent interface for one
  falsification test, but remove the local-information shortcut in the data/loss. Do not spend
  more natural plain-CE budget or introduce E12/E13 before this mechanism test.
- **Architecture mapping (ONE):** this touches **data + target masking**. The only model changes
  are backward-compatible evaluation instrumentation for sparse labels and an explicit
  batch permutation; neither changes `concept_mode="real"` training/inference.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptLM` / `BackboneConceptConfig` | reuse E10e real read/write path as-is | `nn/backbone_concept_lm.py` |
| `_forward_blocks` / `_per_position_ce_from_hidden` | extend only for sparse-label top-1 metrics and explicit eval-only concept permutation | `nn/backbone_concept_lm.py` |
| `DataCollatorForCausalLM` | extend with opt-in precomputed-label preservation; default remains full-token causal labels | `data/data_collators.py` |
| `DataTrainingArguments` / `build_pretraining_collators` | add and forward the opt-in masking flag | `training/concept_pretraining_args.py`, `training/concept_pretraining_factories.py` |
| `load_pretokenized_mix` / `manifest_token_stats.py` | reuse as-is | `data/dataset_preprocess.py`, `scripts/manifest_token_stats.py` |
| delayed-recall row builder | new reusable deterministic data component | `data/delayed_recall.py` |
| dataset/manifest CLI | new reusable artifact builder, not a trainer | `scripts/build_delayed_recall_dataset.py` |
| paired causal evaluator | new reusable checkpoint evaluator | `analysis/run_delayed_recall_eval.py` |
| E10 protocol wrapper | reuse through a thin E14 profile wrapper | `scripts/launch_e10.sh`, new `scripts/launch_e14.sh` |

No model class, training entrypoint, optimizer, or checkpoint family is added.

## 3. Forward pass and metrics

Symbols: `B`=batch, `N`=2048, `K`=512, `C`=128, `H`=1152, `V`=262144,
`L`=26 Gemma layers.

```text
input_ids, labels       (B,N), (B,N); labels=-100 except one answer token
z0                      (B,C,H)

for block b=0..3:
  explicit token input  previous block + current block, at most (B,2K)
  token hidden          frozen Gemma + LoRA + concept reads -> (B,<=2K,H)
  sparse answer CE      lm_head only for valid label rows/positions -> (M,V), M<=B
  recurrent write       z <- E10e BiXT write(z, current hidden) -> (B,C,H)

loss                    mean CE over the B supervised answer tokens
```

Evaluation reruns the same block loop under:
- `real`: normal E10e recurrence.
- `static`: read `z0`, disable recurrent writes.
- `zero`: disable concept reads/writes.
- `conflicting donor`: order rows as adjacent counterfactual twins and pass permutation
  `[1,0,3,2,...]`; each concept read uses its twin's current state while writes remain on the
  recipient row, matching the existing shuffle intervention but with an explicit deterministic
  donor instead of `torch.roll`.

Add `BackboneConceptLM.per_position_metrics(...) -> {"ce": [B,N], "predictions": [B,N]}`
with `-100` predictions at unsupervised positions. Compute `F.linear` only for valid sparse
targets, in chunks, so E14 never materializes `(B,N,V)`. Keep `per_position_ce()` and all
existing mode defaults byte-compatible.

## 4. Inputs & data
- **Tokenizer:** `google/gemma-3-1b-pt`; call
  `configure_text_tokenizer_for_model_vocab()` so all ids are `<262144`.
- **Train:** 2,304 counterfactual pairs = 4,608 rows, each exactly 2,048 tokens
  (9,437,184 input tokens).
- **Held-out:** 2,048 counterfactual pairs = 4,096 rows. Pair ids and random seeds are
  disjoint from train; key/value assignments are newly sampled.
- **Template:** block 1 contains a marked target key and one of two incompatible value tokens;
  blocks 2–3 are pair-identical distractors; block 4 contains a pair-identical query prefix
  and then the answer token. The answer position is beyond the explicit one-block carry.
- **Vocabulary:** select and freeze at least 64 model-valid single-token word values; balance
  their frequency exactly within each split. Keys use a separate tokenizer-verified token pool.
- **Row fields:** `input_ids`, `labels`, `pair_id`, `variant`, `answer_index`,
  `answer_token_id`, `donor_answer_token_id`, `query_block`. Only the first two reach the model.
- **Masking:** labels are `-100` everywhere except `labels[answer_index]=answer_token_id`.
  `DataCollatorForCausalLM(preserve_precomputed_labels=True)` pads/truncates them in lockstep;
  the default `False` continues to mirror `input_ids` for E10 and prior runs.
- **Manifest:** one source with weight 1.0 and ordinary `train_path`/`eval_path`, compatible
  with `load_pretokenized_mix()` and exact token budgeting. Save optional smaller block-2/3
  eval views in manifest metadata for memory-age diagnosis; they are not gates.
- **Builder validation:** exact lengths, one supervised token, balanced values, valid ids,
  twin equality from token 512 through the token immediately before the answer, different
  answers, and disjoint train/eval pair ids. Fail before writing a manifest if any invariant
  is violated.

## 5. Loss & training objective
- **Loss:** existing `BackboneConceptLM` next-token CE; no `LossManager` component.
- **Objective:** `objective_variant="causal_lm"` with answer-only labels supplied by data.
- **Weighting:** one answer token per row, therefore uniform example weighting.
- **Optimization:** fresh Gemma init + E10e LoRA/read/write configuration; task LR 1e-4,
  concept-memory LR 3e-4, read/write gate init 0.01, read RMSNorm on, warmup 50.

## 6. Config, build, launch, and evaluation
- **New backward-compatible config:** `DataTrainingArguments.preserve_precomputed_labels=False`.
- **Generic launcher knob:** `PRESERVE_PRECOMPUTED_LABELS="${...:-false}"`, forwarded to the
  canonical parser.
- **Model eval API:** optional `concept_permutation=None`; reject malformed/non-bijective
  permutations and ignore it unless the eval mode explicitly requests permutation.
- **Registry/eval routing:** unchanged (`backbone_concept`, evaluation contract v1).

Build on Odra:

```bash
source scripts/remote_paths.sh
uv run python scripts/build_delayed_recall_dataset.py \
  --tokenizer google/gemma-3-1b-pt \
  --output_dir "$DATASETS_TOK_DIR/e14_delayed_recall" \
  --manifest "$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json" \
  --train_rows 4608 --eval_pairs 2048 \
  --sequence_length 2048 --block_size 512 --value_count 64 --seed 42
```

Launch through a thin `scripts/launch_e14.sh` that pins:

```bash
EXPERIMENT_ID=E14
MANIFEST="$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json"
SKIP_PRETOKENIZE=1
TARGET_TOKENS=9437184
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
PRESERVE_PRECOMPUTED_LABELS=true
READ_CONCEPT_NORM=true
READ_GATE_INIT=0.01
WRITE_GATE_INIT=0.01
CONCEPT_MEMORY_LR=3e-4
WARMUP_STEPS=50
AUTO_INTERVALS=0
EVAL_STEPS=164
SAVE_STEPS=164
SAVE_TOTAL_LIMIT=3
MAX_EVAL_SAMPLES=256
```

Then `exec bash scripts/launch_e10.sh`. At effective batch 6, checkpoint 164 follows
2,015,232 tokens; checkpoints 328/492/656 and the final export cover the remaining budget.

Gate evaluation:

```bash
uv run python analysis/run_delayed_recall_eval.py \
  --checkpoint "Cache/Training/<run>/checkpoint-164" \
  --manifest "$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json" \
  --num_pairs 256 --batch_size 16 \
  --output "Cache/Evaluation_reports/E14_checkpoint-164_delayed_recall.json"
```

If the 2M kill gate passes, evaluate all 2,048 pairs on the final checkpoint with the same
command and `--num_pairs 2048`. The evaluator bootstraps counterfactual **pairs**, reports
real/static/zero/donor CE and top-1 accuracy, all three paired margins with 95% intervals,
donor-target following accuracy, gates, checkpoint/config identity, and manifest hash.

## 7. Tests & smoke
- `tests/test_delayed_recall.py`
  - deterministic tiny builder with a tokenizer stub;
  - exact length/block/query/one-label invariants;
  - balanced values, counterfactual prefix equality, and split disjointness;
  - manifest round-trip through `load_pretokenized_mix`.
- `tests/test_backbone_concept_lm.py`
  - default collator behavior unchanged;
  - opt-in sparse labels preserved under padding/truncation;
  - sparse CE/prediction API returns one finite CE and one prediction per row;
  - explicit adjacent-pair permutation matches batch-size-2 shuffle and rejects invalid maps.
- `tests/test_concept_pretraining_parameter_flow.py`
  - parser/factory maps `--preserve_precomputed_labels true`.
- `tests/test_training_launcher_parameter_flow.py`
  - E14 wrapper reaches the canonical parser with E10e gates/LR, sparse labels, batch 6,
  target 9,437,184, and checkpoint step 164.
- Targeted local smoke:

```bash
uv run pytest \
  tests/test_delayed_recall.py \
  tests/test_backbone_concept_lm.py \
  tests/test_concept_pretraining_parameter_flow.py \
  tests/test_training_launcher_parameter_flow.py -q
```

Build a tiny local dataset with the tokenizer cache and run the evaluator against the tiny
random model in unit tests. Do not load/train Gemma-1B on macOS beyond a forward smoke.

## 8. Risks & tradeoffs
- **Synthetic task too hard for 9.4M tokens.** Cheapest signal: checkpoint-164 real-vs-ablation
  margins. Fallback: none inside E14; a failed frozen gate branches to writer/interface diagnosis.
- **Synthetic task passes without natural transfer.** This is expected and explicitly outside
  E14's claim. A pass only authorizes a later mixed-curriculum spec.
- **Local leakage.** Builder-level counterfactual equality and ≤20% ablated accuracy gate make
  leakage visible before interpreting the architecture.
- **`shuffle` is not guaranteed to select the intended twin in larger batches.** Use the new
  explicit permutation and validate adjacent pair ids before every evaluator batch.
- **Sparse-label evaluation still projects into V=262144.** Project only the B selected hidden
  states, never all N positions; batch-size calibration remains part of remote preflight.
- **The full held-out gate may outlast training.** Run the 256-pair gate at checkpoint 164 and
  stop the Byobu training process if all margins are below 0.01; reserve 2,048-pair bootstrap
  evaluation for the final checkpoint.

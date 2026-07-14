# E02 — Implementation Plan

- **Spec:** [E02_ar_prefix_suffix.md](E02_ar_prefix_suffix.md) · **Status:** implemented and run; closed positive
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change: keep the E01 concept-conditioned AR foundation fixed and change only the objective/data view from reconstruction to prefix-to-suffix generation. Reuse-first; no new training script.

## 1. Source & Fit
- **Origin:** E02 follows the agenda's AR series and the SODA principle already explored in the parked prefix-diffusion track: force the bottleneck to generate content the encoder did not see. Prior prefix-diffusion failed with low rank, but E01 adds the materially new ingredient: a real causal AR decoder conditioned on concepts. The current E01 `0.3`-epoch run is a warmup/plumbing run; final E02 interpretation must compare against a matched-budget full E01 baseline.
- **Synthesis verdict:** Adapt. Take the prefix/suffix data asymmetry; drop the parked diffusion decoder and old train fork. Implement it as a config-selectable objective over `ConceptEncoderForConditionalLM`.
- **Architecture mapping (ONE):** this touches the **data/loss objective** only. Encoder, bottleneck, causal decoder architecture, modern blocks, optimizer family, tokenizer, and eval route stay fixed vs E01.

## 2. Reuse Map
| Component | Action | Where |
|---|---|---|
| `ConceptEncoder`, `BiConceptEncoderLayer`, `BiXTCrossAttention` | reuse as-is | `nn/concept_encoder.py` |
| `ConceptCausalDecoderLayer`, `ConceptCausalDecoderStack` | reuse as-is | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderForConditionalLM` | extend forward + ablation helpers to accept prefix/suffix batches while preserving current reconstruction contract | `nn/concept_encoder_perceiver.py` |
| `DataCollatorForPrefixGeneration` | adapt: support eos-only causal tokenizers (`sep_token_id=None`) and validate no information leak | `data/data_collators.py` |
| `DataCollatorForTSDAE` | reuse for E01 / reconstruction path | `data/data_collators.py` |
| `PerceiverDenoiseTrainer` | extend concept-ablation batch handling for `prefix_suffix`; leave contrastive path perceiver-only | `training/train_perceiver_denoise.py` |
| `ModelArguments`, `DataTrainingArguments`, `build_perceiver_denoise_config`, `main()` | extend with `objective_variant="prefix_suffix"` and prefix split knobs | `training/train_perceiver_denoise.py` |
| `train_perceiver_denoise_multigpu.sh` | extend env knobs only; no copy | `scripts/train_perceiver_denoise_multigpu.sh` |
| `analysis/run_concept_analysis.py`, `evaluation/concept_eval_routing.py` | reuse unchanged: family remains `concept_ar`; geometry/eval probe the encoder | `analysis/`, `evaluation/` |

## 3. Forward Pass
Symbols: `B`=batch, `N`=full tokenized document length, `P`=prefix length, `S`=suffix length, `C`=128 concepts, `H`=768 hidden, `Ht`=256 token embedding, `V`=SmolLM2 vocab.

```text
full tokenized row (B, N)                 # produced by load_and_preprocess_text_dataset, eos appended
  -> DataCollatorForPrefixGeneration
prefix_input_ids      (B, P)              # clean prefix + eos boundary, padded
prefix_attention_mask (B, P)
suffix_input_ids      (B, S)              # held-out suffix + eos stop, padded
labels                (B, S)              # suffix ids, -100 at pad

prefix_input_ids (B, P)
  -> ConceptEncoder token embeddings Ht->H + BiXT x6
concepts (B, C, H)

suffix_input_ids (B, S)
  -> shift_right(suffix_input_ids)         # prepend bos/eos start, remove final token
decoder_input_ids (B, S)
  -> ConceptCausalDecoderStack x4          # causal self-attn over suffix + cross-attn to concepts
hidden (B, S, H)
  -> lm_head
logits (B, S, V)
  -> next-token CE(logits[:, :-1], labels[:, 1:])
loss scalar
```

Important invariants:
- Encoder reads **prefix only**; suffix tokens never enter `ConceptEncoder`.
- Decoder cross-attends only to concepts; no encoder-token skip.
- AR decoder keeps the E01 behavior: RoPE on decoder self-attn q/k, no RoPE on orderless concepts.
- The only O(S^2) operation is the causal AR decoder over the suffix; the encoder remains O(C*P).

## 4. Inputs & Data
- **Dataset:** `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`, same loader `load_and_preprocess_text_dataset()` in `data/dataset_preprocess.py`.
- **Tokenizer / format:** `HuggingFaceTB/SmolLM2-135M`; `training/train_perceiver_denoise.py` already aliases pad to eos when needed and appends eos for causal AR. The cached tokenizer exposes `<|endoftext|>` as bos/eos/unk (id 0) and has `<|im_start|>` / `<|im_end|>` special tokens, but no chat template. E02 deliberately uses **raw document continuation**, not ChatML: do not inject `<|im_start|>`, `<|im_end|>`, role strings, or fake user/assistant turns into FineWeb-Edu rows.
- **Collator:** adapt `DataCollatorForPrefixGeneration` rather than adding a new collator. Current version assumes `sep_token_id` and ModernBERT-style `[CLS]/[SEP]`; E02 needs:
  - `boundary_token_id = sep_token_id if not None else eos_token_id`;
  - no hard failure when `sep_token_id is None` if eos exists;
  - optional prefix terminator using `boundary_token_id` so the encoder sees an explicit end of prefix;
  - suffix always ends with `boundary_token_id` / eos so the decoder learns to stop;
  - dynamic padding retained for both prefix and suffix;
  - labels equal suffix ids at real positions, `-100` at padding.
- **Split:** expose `prefix_ratio_min`, `prefix_ratio_max`, `min_prefix_content`, `min_suffix_content`, and `split_strategy` in `DataTrainingArguments`; default to existing collator values for backward compatibility. E02 sets a tight first target of **`0.35-0.45`**, `sentence_boundary`: roughly 40% prefix / 60% suffix. This keeps the continuation logical and stable for the first run; wider/harder split curricula are later experiments.
- **Context length:** keep `max_seq_length=512` for the E02 warmup and first full E01/E02 comparison. Longer contexts (`1024+`) are scientifically important but should be a later one-variable experiment after the AR objective itself is stable; increasing context now would confound the objective change and increase AR decoder cost.
- **Short examples:** keep existing minimum-length fallback, but make it tokenizer-agnostic. For too-short rows, produce at least one real prefix token and one real suffix/stop token, or skip/filter only if impossible.

## 5. Loss & Objective
- **Objective variant:** add `OBJECTIVE_PREFIX_SUFFIX = "prefix_suffix"` to `training/train_perceiver_denoise.py` and `VALID_OBJECTIVES`.
- **Loss:** plain suffix next-token CE in `ConceptEncoderForConditionalLM`, using existing `_shift_right()` and `_next_token_ce()` against suffix labels.
- **Concept losses:** keep `LossManager` available but off for E02 (`CONCEPT_LOSSES=none`). If enabled later, apply it to the prefix-derived `concept_repr` exactly like reconstruction.
- **Word dropout:** E02 sets `DECODER_WORD_DROPOUT=0.0`; the suffix target is already absent from the encoder, so adding word dropout would be a second experimental variable. Keep the implementation available for other objectives.
- **Concept ablation:** update `concept_ablation_ce()` or add a helper so it can compute CE deltas for both contracts:
  - reconstruction: `input_ids`, `attention_mask`, `labels`;
  - prefix/suffix: `prefix_input_ids`, `prefix_attention_mask`, `suffix_input_ids`, `labels`.

## 6. Config & Launch
- **No new `ConceptEncoderConfig` fields** are needed. Existing config already carries `decoder_type`, `decoder_pos_type`, `decoder_word_dropout`, `hidden_act`, `norm_type`, `checkpoint_family`, and eval contract metadata.
- **New CLI/data fields** in `training/train_perceiver_denoise.py`:
  - `DataTrainingArguments.prefix_ratio_min: float = 0.3`
  - `DataTrainingArguments.prefix_ratio_max: float = 0.5`
  - `DataTrainingArguments.min_prefix_content: int = 5`
  - `DataTrainingArguments.min_suffix_content: int = 10`
  - `DataTrainingArguments.split_strategy: str = "sentence_boundary"`
- **Validation:** allow `decoder_type="causal_ar"` with `objective_variant in {"reconstruction", "prefix_suffix"}`. Keep `reconstruction+contrastive` perceiver-only.
- **Config metadata:** set `pretraining_objective="ar_prefix_suffix_generation"` when `is_causal_ar and objective_variant == "prefix_suffix"`; family remains `concept_ar`, routes unchanged.
- **Collator selection:** in `main()`, choose `DataCollatorForPrefixGeneration` when `objective_variant == "prefix_suffix"`, otherwise `DataCollatorForTSDAE`.
- **Run naming/logging:** keep architecture family `concept_ar`; optionally set run prefix to `concept_ar_prefix` for E02 logs/checkpoints so results are distinguishable without changing eval routing.
- **Run budget:** first Odra launch is a `NUM_EPOCHS=0.3` warmup to prove objective plumbing, loss, concept ablation, and sample generation. The full E02 run should use the same epoch/step budget as the later full E01 baseline (likely 1-2 epochs), not the E01 warmup.
- **Launcher knobs:** add:
  - `PREFIX_RATIO_MIN="${PREFIX_RATIO_MIN:-0.3}"`
  - `PREFIX_RATIO_MAX="${PREFIX_RATIO_MAX:-0.5}"`
  - `MIN_PREFIX_CONTENT="${MIN_PREFIX_CONTENT:-5}"`
  - `MIN_SUFFIX_CONTENT="${MIN_SUFFIX_CONTENT:-10}"`
  - `SPLIT_STRATEGY="${SPLIT_STRATEGY:-sentence_boundary}"`
  and pass them to the existing `training/train_perceiver_denoise.py`.
- **Launch:**
  ```bash
  DECODER_TYPE=causal_ar \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=prefix_suffix DELETION_RATE=0.0 DECODER_WORD_DROPOUT=0.0 \
  PREFIX_RATIO_MIN=0.35 PREFIX_RATIO_MAX=0.45 SPLIT_STRATEGY=sentence_boundary \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  NUM_EPOCHS=0.3 TRAIN_NUM_PROC=8 TEST_NUM_PROC=4 DATALOADER_NUM_WORKERS=4 SAVE_TOTAL_LIMIT=5 \
  bash scripts/train_perceiver_denoise_multigpu.sh
  ```

## 7. Tests & Smoke
- **Collator tests** in `tests/test_data_collators.py`:
  - existing ModernBERT prefix tests still pass;
  - new eos-only fake tokenizer or SmolLM2-like fixture works with `sep_token_id=None`, `bos=eos=unk`, `pad=eos`;
  - suffix ends with eos and labels include eos;
  - no suffix token appears in prefix content, excluding boundary/pad ids;
  - dynamic padding and minimum lengths still hold.
- **Model tests** in `tests/test_concept_ar_decoder.py`:
  - `ConceptEncoderForConditionalLM.forward(prefix_input_ids=..., prefix_attention_mask=..., suffix_input_ids=..., labels=...)` returns finite loss and logits `(B, S, V)`;
  - changing suffix future tokens does not affect earlier logits when concepts are held fixed;
  - concept zero/shuffle ablation changes suffix CE for prefix/suffix batches;
  - old reconstruction forward path still works unchanged.
- **Training/config tests**:
  - `build_perceiver_denoise_config()` sets `pretraining_objective="ar_prefix_suffix_generation"` for E02;
  - parser accepts `--objective_variant prefix_suffix` with `--decoder_type causal_ar`;
  - invalid `reconstruction+contrastive` with `causal_ar` still errors.
- **Local smoke:** tiny CPU/MPS run on a small dataset slice after implementation:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python training/train_perceiver_denoise.py \
    --decoder_type causal_ar --objective_variant prefix_suffix \
    --hidden_size 64 --token_embedding_dim 32 --num_hidden_layers 2 --decoder_num_layers 1 \
    --concept_num 8 --intermediate_size 128 --hidden_act silu --norm_type rmsnorm --decoder_pos_type rope \
    --max_seq_length 64 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --max_steps 5 --eval_strategy steps --eval_steps 5 --save_strategy no --logging_steps 1 \
    --tokenizer_name HuggingFaceTB/SmolLM2-135M --dataset_name JeanKaddour/minipile \
    --output_dir Cache/Training/e02_smoke --logging_dir Cache/logs --report_to none --remove_unused_columns False
  ```
  Prefer a direct `uv run python training/train_perceiver_denoise.py` command for smoke rather than the multi-GPU launcher on macOS; keep it to a few steps.
- **Analysis smoke:** run `analysis/run_concept_analysis.py --model_type concept_ar` on the smoke checkpoint to confirm encoder-only geometry analysis still loads.

## 8. Risks & Tradeoffs
- **Risk: old prefix collator silently encodes BERT assumptions.** Cheapest signal: unit tests with eos-only tokenizer and a real SmolLM2 tokenizer. Fallback: keep one collator but make boundary behavior explicit through `boundary_token_id`.
- **Risk: loss shift is off by one for suffix generation.** Cheapest signal: tiny deterministic batch where labels include eos and CE ignores only padding. Fallback: centralize suffix CE through `_next_token_ce()` and test shapes/ignored positions.
- **Risk: Trainer ablation metrics break because batch keys changed.** Cheapest signal: unit test or smoke eval logs `concept_ablation/delta_*` for prefix/suffix. Fallback: add model-level `concept_ablation_ce_for_batch(batch)` to avoid duplicated trainer logic.
- **Risk: task too hard from random init on Odra budget.** Cheapest signal: spec kill gate, suffix eval CE by 25% checkpoint. Fallback is not a code fallback; if it fails, record and move to E04 warm-start rather than adding another variable.
- **Risk: E02 accidentally changes architecture.** Guard: do not touch decoder layers, norm/activation/rope settings, concept count, tokenizer, optimizer, or eval routing except where required to pass prefix/suffix tensors.
- **Risk: context length becomes a hidden second variable.** Cheapest signal: E01/E02 are both run at `max_seq_length=512` for the first comparison. Fallback: create a later context-scaling spec rather than changing E02.

## 9. Code Sketches
```python
# sketch: data/data_collators.py
class DataCollatorForPrefixGeneration:
    def __init__(..., split_strategy="sentence_boundary"):
        self.boundary_token_id = tokenizer.sep_token_id
        if self.boundary_token_id is None:
            self.boundary_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.boundary_token_id is None:
            raise ValueError("Prefix generation requires sep_token_id or eos_token_id.")

    def __call__(self, features):
        content = self._extract_content(raw_ids)      # strip pad and one trailing boundary/eos
        split = self._choose_split(content)
        prefix_ids = content[:split] + [self.boundary_token_id]
        suffix_ids = content[split:] + [self.boundary_token_id]
        return {
            "prefix_input_ids": ...,
            "prefix_attention_mask": ...,
            "suffix_input_ids": ...,
            "suffix_attention_mask": ...,
            "labels": labels,  # suffix ids, -100 at pad
        }
```

```python
# sketch: nn/concept_encoder_perceiver.py
class ConceptEncoderForConditionalLM(PreTrainedModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        prefix_input_ids=None,
        prefix_attention_mask=None,
        suffix_input_ids=None,
        suffix_attention_mask=None,
        **kwargs,
    ):
        if prefix_input_ids is not None:
            concepts = self.encode_concepts(prefix_input_ids, prefix_attention_mask, return_dict=True).last_hidden_state
            decoder_input_ids = self._shift_right(suffix_input_ids)
            logits = self.decode_logits(concepts, decoder_input_ids, word_dropout_p=0.0)
            task_loss = self._next_token_ce(logits, labels) if labels is not None else None
            ...
        else:
            # existing reconstruction path unchanged
            ...
```

```python
# sketch: training/train_perceiver_denoise.py
OBJECTIVE_PREFIX_SUFFIX = "prefix_suffix"
VALID_OBJECTIVES = {OBJECTIVE_RECONSTRUCTION, OBJECTIVE_RECONSTRUCTION_CONTRASTIVE, OBJECTIVE_PREFIX_SUFFIX}

if is_causal_ar and model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
    raise ValueError("causal_ar supports reconstruction or prefix_suffix, not reconstruction+contrastive")

data_collator = (
    DataCollatorForPrefixGeneration(...)
    if model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX
    else DataCollatorForTSDAE(...)
)
```

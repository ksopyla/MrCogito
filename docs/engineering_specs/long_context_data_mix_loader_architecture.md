# Long-context pretraining data-mix architecture (E05 + follow-ups)

- **Type:** engineering foundation (data loader + processor architecture), not a standalone `E0NN` experiment.
- **Status:** partially implemented (2026-06-21): recipe loader + training-arg wiring done; long-context preflight guardrails still TODO.
- **Serves:** `E05_windowed_decoder_concept_memory` and the next long-context series (2K and beyond).
- **Primary goal:** plug-and-play dataset mixes (Nemotron-inspired and SmolLM3-inspired) that are easy to edit by changing recipe files, not Python code.

## Why this is needed now

`E05` is already implemented at model/decoder level, but long-context training quality now depends on better data mixing and preprocessing control.

Current code is close, but still too rigid for iterative mix work:
- mixes are hardcoded in `data/dataset_preprocess.py` (`DATASET_MIXES`),
- switching weights/sources requires code edits,
- no first-class recipe files for versioned, reviewable mix definitions,
- no built-in policy checks for long-context coverage (for 2K+).

## Current reusable foundation (already in repo)

- Loader entrypoint: `data/dataset_preprocess.py:load_and_preprocess_dataset_mix()`
- Single-source path: `data/dataset_preprocess.py:load_and_preprocess_text_dataset()`
- Training integration: `training/train_perceiver_denoise.py` (`DataTrainingArguments.dataset_mix`, `main()` mix routing)
- Existing long-context mix: `DATASET_MIXES["long_2k_base_v1"]`
- Existing collators:
  - `DataCollatorForTSDAE` (reconstruction)
  - `DataCollatorForPrefixGeneration` (prefix→suffix)

So this is a **reuse-first extension**, not a new training script.

## Target architecture

### 1) Recipe-first mixes (file-driven)

Add recipe files under `data/mix_recipes/` (JSON):
- one file per mix,
- explicit `sources[]` with `hf_id/subset/split/text_columns/weight/max_samples`,
- optional gating metadata and fallback datasets,
- expected long-context profile targets (`>2k`, `>4k`).

This decouples data design from loader code and supports fast A/B edits.

### 2) Registry and loader API (implemented)

Implemented in `data/dataset_preprocess.py`:
- `load_mix_recipe(mix_name_or_path)` with recipe-id/path resolution (`data/mix_recipes/*.json` + explicit paths).
- mix-source normalization for HF compatibility (`hf_id` or `dataset`, `text_column` or `text_columns`).
- unified resolver for registry mixes, recipe mixes, and inline mix objects.
- runtime weight override parser + applier (`mix_weight_override`) keyed by source `name` or `hf_id`.
- fallback source loading (`fallback_hf_id`/`fallback_data_files`) for gated/unavailable datasets.

`load_and_preprocess_dataset_mix()` remains the single execution path.

### 3) Training args (implemented)

Implemented in `training/train_perceiver_denoise.py` `DataTrainingArguments`:
- `dataset_mix_recipe: Optional[str] = None` (path or mix id)
- `dataset_mix_weight_override: Optional[str] = None` (JSON string, optional quick sweeps)

Implemented routing order in `main()`:
1. `dataset_mix_recipe` (new)
2. `dataset_mix` (existing)
3. `dataset_name`/`dataset_name_subset` (existing single dataset path)

### 4) Long-context guardrails (still TODO)

Add optional preflight in loader:
- estimate expected long-doc support from prior measured stats file
  (`Cache/Evaluation_reports/seqlen_model_mix_1k_shuffle/seqlen_dist_summary.json`),
- warn if projected `>2k` support is below target for `max_seq_length >= 2048`.

This does not block runs; it avoids accidental short-doc mixes when training 2K.

## Prepared mix recipes (already added)

Machine-readable files:
- `data/mix_recipes/nemotron_nano_v3_inspired_2k.json`
- `data/mix_recipes/smollm3_inspired_2k.json`

Both are:
- objective-compatible with `reconstruction` and `prefix_suffix`,
- designed for `max_seq_length=2048`,
- weighted to preserve pretraining breadth while forcing non-trivial >2K tails.

### Mix A — Nemotron-inspired (2K)

- Main idea: Nemotron-family web quality + long-doc backbone + moderate reasoning tail.
- Gate note: `nvidia/Nemotron-CC-v2` is still manual-gated; recipe includes fallback to `OptimalScale/ClimbMix`.
- Projected from measured stats: ~23.5% docs >2K, ~10.1% >4K.

### Mix B — SmolLM3-inspired (2K)

- Main idea: SmolLM-style web+code+math proportions plus explicit long-tail boosters for E05.
- Includes low-weight reasoning tails to avoid purely short-doc behavior at 2K.
- Projected from measured stats: ~21.3% docs >2K, ~8.8% >4K.

## How this supports E05 and next runs

- `E05` needs concept-only cross-window memory pressure; that pressure appears only if the mix has enough beyond-window content.
- These recipes explicitly target that for 2K now, and can be progressively shifted for 4K/8K later by editing weights and source list (no loader rewrites).
- Same training pipeline, same collators, same launch script family.

## Remaining sequence (next coding step)

1. **Loader wiring**
   - add recipe parsing + validation to `data/dataset_preprocess.py`,
   - preserve backward compatibility with `DATASET_MIXES`.
2. **Training arg wiring**
   - add `--dataset_mix_recipe` and optional weight override.
3. **Recipe smoke tests**
   - new tests in `tests/` for parsing, weight normalization, objective compatibility, and fallback behavior.
4. **Small run smoke**
   - local tiny pass (`max_samples` very small) for each recipe with both objectives:
     - reconstruction + `DataCollatorForTSDAE`
     - prefix_suffix + `DataCollatorForPrefixGeneration`
5. **E05 launch prep**
   - run matched full-context vs windowed-context with the selected recipe and fixed seed.

## Risks and mitigations

- **Manual gating delays (Nemotron-CC-v2):** use explicit fallback sources (already encoded in recipe).
- **Mix drift vs long-context goals:** add projected `>2k` preflight warnings.
- **Too much reasoning/SFT contamination:** keep reasoning-tail weights low and explicit in recipe metadata.
- **Reproducibility:** version recipes as files under git; include recipe id/path in run config and W&B.

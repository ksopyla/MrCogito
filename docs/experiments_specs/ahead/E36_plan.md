# E36 — Implementation Plan

- **Spec:** [E36_frozen_reader_writer.md](E36_frozen_reader_writer.md) · **Status:** implemented (0da9be4)
- **Authored by:** `implementation-plan` · for → `research-implement`

## 1. Source & fit
- **Origin:** research note [e36_frozen_reader_writer_20261018.md](../../4_Research_Notes/e36_frozen_reader_writer_20261018.md)
  (ARC-Encoder, REFRAG, Cramming / Progressive Cramming, EPL, xRAG; our E35 / E33f).
- **Verdict:** Adapt — chunk-parallel pooled writer for a frozen reader, at an extreme ratio (512 tokens per vector,
  the author's choice) that no published trained compressor uses.
- **Mapping:** encoder (writer), bottleneck (one vector per span), decoder (frozen, untouched), loss (continuation +
  reconstruction).
- **Boldness check:** the reader stays frozen and the ratio stays at 512; fallbacks only via the spec's ladder.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| shared entrypoint, HF Trainer, W&B, run dirs | reuse; new `model_family="frozen_reader"` branch | `training/train_concept_pretraining.py`, `training/concept_pretraining_factories.py`, `training/concept_pretraining_args.py` |
| pretokenizer with long-document chunking | reuse (`chunk_long_docs`, `min_tokens`) | `scripts/pretokenize_mix.py`, recipe `data/mix_recipes/e36_longdoc_4k6_v1.json` |
| causal collator | reuse (rows are exactly 4,610 tokens, no padding) | `data/data_collators.py:DataCollatorForCausalLM` |
| shared launcher | extend (FR_* knobs) | `scripts/train_concept_pretraining_multigpu.sh`, thin wrapper `scripts/launch_e36.sh` |
| `SpanWriter`, `InputProjector`, `KVExpander`, `FrozenReaderMemory` | new, config-selectable | `nn/frozen_reader_memory.py` |
| evaluation (S0 / S1) | new | `evaluation/frozen_reader_eval.py` |

## 3. Forward pass (B rows, S = 512, M = 8, D = 576 writer width, d = 960 reader width, L = 32 reader layers, V = 49,152)
```
row (B, 4610) → drop BOS → spans 0..7 (B, 8·512), target span 8 (B, 512)
(B·8, 512)      → borrowed frozen embedding → 3 bidirectional SmolLM2-135M layers → mean → RMSNorm → (B, 8, D)
way input: (B, 8, D) → MLP → RMS-norm × gain → (B, 8, d) ; [entries | target embeddings] → frozen reader → (B, 512, V)
way kv:    (B, 8, D) → Linear → (B, 8, L, 2, 5, 64) → RMS-norm × per-layer gain → keys rotated at span middles
           → a 8-position prefix cache in every layer → frozen reader over the target → (B, 512, V)
positions: entry i at 512·i + 256; target at 4096 + t.  Reconstruction: entry j → target = span j at 512·j + t.
```

## 4. Inputs & data
PG-19 train (6 shards) + FinePDFs (4 shards), SmolLM2 tokenizer, whole documents chunked into rows of exactly
4,610 tokens; tails dropped; 10 % of documents held out (eval split).

## 5. Loss
Cross-entropy of the frozen reader on target tokens 1–511 (continuation; 80 % of steps) or on the rebuilt span
(reconstruction; 20 %, the same draw on every DDP rank). No auxiliary losses.

## 6. Config & launch
`--model_family frozen_reader --fr_reader --fr_writer_init --fr_writer_layers --fr_span --fr_mem_spans --fr_way
--fr_recon_frac --fr_entry_positions --fr_proj_hidden` (defaults: SmolLM2-360M, SmolLM2-135M, 3, 512, 8, input, 0.2,
span_mid, 2048).
```
RUN=Cache/e36/s0_smollm2_360m S0_READER=HuggingFaceTB/SmolLM2-360M GPU_IDS=3 bash scripts/launch_e36.sh
RUN=Cache/e36/s1_input_s512 GPU_IDS=0,1 NUM_GPUS=2 FR_WAY=input bash scripts/launch_e36.sh
RUN=Cache/e36/s1_kv_s512 GPU_IDS=2,3 NUM_GPUS=2 FR_WAY=kv bash scripts/launch_e36.sh
```

## 7. Tests & smoke
`tests/test_frozen_reader_memory.py` (10): loss finite and gradients only in writer + adapter, the writer is
bidirectional, the reader stays causal over the target and reads the entries, no-memory / raw-prefix paths equal the
plain reader, checkpoints hold only trained parts, the reconstruction draw is rank-independent. Local smoke: 6 steps
through the entrypoint + the evaluation on SmolLM2-135M (18 Oct).

## 8. Risks
- 512 tokens per vector is ~8× what an optimised vector carries into SmolLM2-360M (61 tokens): the S0 probe
  measures the ceiling first; the kill / ladder is in the spec.
- The writer starts from 3 layers of a causal LM; mean pooling may wash out facts. Cheapest signal: the "other"
  condition equals "memory" (nothing document-specific crosses).

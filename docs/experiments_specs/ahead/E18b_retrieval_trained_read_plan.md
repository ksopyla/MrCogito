# E18b — Implementation Plan

- **Spec:** [E18b_retrieval_trained_read.md](E18b_retrieval_trained_read.md) · **Status:** approved (user go 2026-09-10)
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for: *5% dense-label synthetic retrieval rows in the 32k LM mix turn the single global read
> into a general, length-extrapolating retriever (passkey 0% → ≥ 90% @32k, ≥ 80% @128k) at ≤ 0.5% LM
> cost.* No model changes. Everything is data, collator, probes, launcher knobs and a chain script.

## 1. Source & fit
- **Origin:** E18 pilot evidence — P2 (the read retrieves under dense labels), arm A (used and extrapolating
  when the stack cannot reach), arm C (no net LM value: natural text never supervises long-range
  addressing) — [report](../../2_Experiments_Registry/run_reports/e18_reach_ablation_20260909.md). Prior
  art for synthetic long-context data in pretraining mixes: ProLong / OLMo-3 style synthetic tasks.
- **Synthesis verdict:** Adapt — take "synthetic retrieval in the mix", drop needle-style rows (5 labels per
  row is ~3,000× too sparse vs the copy rows that made P2 converge); use *dense* keyed-recall rows.
- **Architecture mapping:** data + objective masking only (`perceiver_ar` family untouched); probes.
- **Boldness check:** the claim (generality + 4× extrapolation, transfer to a never-seen format) is tested
  exactly as specified; the frame-token diagnostic added below *separates* failure modes, it does not
  soften the gate.

## 2. Reuse map (read first)
| Component | Action | Where |
|---|---|---|
| `load_pretokenized_mix` — weighted all-exhausted interleave via `concatenate_datasets(...).select(idx)` | extend: honour an optional per-source `"in_eval": false` (skip that source's eval rows in the trainer eval) | `data/dataset_preprocess.py:797` |
| `_fast_weighted_all_exhausted_interleave` | reuse as-is — **forces identical columns across sources** (`input_ids`, `attention_mask`, `special_tokens_mask` on the LM shards; no `labels`) | `data/dataset_preprocess.py:17` |
| `DataCollatorForCausalLM` | extend: `loss_span_markers=(start_id, end_id)` — per-row rule, labels only between markers for rows that contain `start_id`; mutually exclusive with `preserve_precomputed_labels` | `data/data_collators.py:163` |
| `labels_from_span_markers(ids, start, end)` | **new pure function** (used by the collator and the tasks probe) | `data/data_collators.py` |
| `DataArguments.preserve_precomputed_labels` | add sibling `loss_span_markers: str = ""` | `training/concept_pretraining_args.py:363` |
| collator construction | pass the new kwarg | `training/concept_pretraining_factories.py:495` |
| generic launcher | `LOSS_SPAN_MARKERS` env → `--loss_span_markers` | `scripts/train_concept_pretraining_multigpu.sh:451` |
| `scripts/build_copy_task_dataset.py` (`iter_rows` → `Dataset.from_generator`, manifest writer) | pattern for the new builder | `scripts/` |
| `scripts/build_retrieval_mix_dataset.py` | **new reusable builder**: keyed-recall rows with real-text filler + merged manifest | `scripts/` |
| `evaluation/long_context_probes.py` — `probe_passkey`, `load_eval_rows`, `argmax_tokens` | extend: filler by concatenating rows beyond the tokenized row length; `--probe tasks`; `--frame_token` diagnostic | `evaluation/` |
| `scripts/manifest_token_stats.py`, `data/length_cache.py` | reuse; both must be **precomputed out-of-band** for the merged manifest (~35 min each single-process on Polonez, see §8) | — |
| `scripts/launch_e18.sh` (`E18_STAGE=32k`, warm start, stage-aware warmup) | reuse; **`MANIFEST=` must be overridden** (the 32k branch derives it from `PRETOKENIZE_MIX` and the generic launcher re-exports it over `PRETOKENIZED_MANIFEST`) | `scripts/launch_e18.sh:102` |

## 3. Data flow (shapes)
Symbols: `S`=32768 row length, `R`=rows, `V`=128,256, markers `START=128103`, `END=128104`
(`<|reserved_special_token_100/101|>`; BOS `128000`, row terminator `128012` = the mix's `append_eos_token_id`).
```
retrieval row (exactly S tokens, LM schema: input_ids / attention_mask=1 / special_tokens_mask):
  BOS  filler₀  [KEY₁ v₁]  filler₁ … [KEY_i v_i] … [KEY₁ START v₁ END] … filler_n  128012
        └ real text from the LM *train* shards (never eval)        └ target: labels = v₁ … END
  items per row 8–24; value length ∈ {8–16 (lookup-like), 64–512 (span copy)} sampled per item;
  KEY = 3 random ids from [1000, 2000); target follows its source by ≥ 1024 tokens, ≤ S−2
collator (loss_span_markers set):  row contains START  → labels = ids between each START..END (END included),
                                                            -100 elsewhere; unmatched START → labels to row end
                                    row has no START     → labels = input_ids (plain LM), -100 at pad
```
Supervised tokens per retrieval row ≈ 2–4k. Row weight in the merged manifest is **derived from token
share**: weights are per *row*, and a 32k row vs a mean 2,858-token LM row means 5% tokens ⇔ row weight
≈ 0.0046 (`w = f·m̄ / (S − f·(S − m̄))` with `m̄` = base mean row tokens); the builder computes it from the
base manifest's length cache (fallback `--base_mean_row_tokens`). Over the 0.5B budget (~175k rows) the
model sees ≈ 800 retrieval rows ≈ 26M tokens ≈ 2M+ supervised targets.

## 4. Inputs & data
- **Base manifest:** `datasets_tok_smollm3_32k/e18_pilot_longdoc_v1_manifest.json` (4 sources).
- **Builder:** `uv run python scripts/build_retrieval_mix_dataset.py --base_manifest $M32 --fraction 0.05
  --n_train 6000 --n_eval 200 --context 32768 --out_dir $TOK/e18b_retrieval_32k
  --out_manifest $TOK/e18b_lm_ret05_manifest.json --seed 0` → writes `train/`, `eval/` (LM schema) and the
  merged manifest: base sources × (1−w), `{"name":"retrieval_keyed_recall","weight":w,"in_eval":false,…}`,
  plus a `retrieval_meta` block (marker ids, item params, seed) for provenance.
- **Arm 0** uses the base manifest unchanged. **Arm D** uses the merged manifest with `PAR_MODE=dense`.
- **Collator:** `DataCollatorForCausalLM(loss_span_markers=(128103,128104))` for all three arms (harmless
  on arm 0: no row contains START). `preserve_precomputed_labels` stays `false`.
- **Eval:** the trainer eval = the base sources' eval rows only (`in_eval:false`), so eval loss is
  comparable across arms. Retrieval eval rows feed only `--probe tasks`.

## 5. Loss & objective
Next-token CE as in E18 (chunked soft-capped / Liger), unchanged; the marker rule only changes which
positions carry −100. No auxiliary loss.

## 6. Config & launch
- **New fields:** `DataArguments.loss_span_markers: str = ""` (comma pair; validated: two distinct ints in
  `[0, V)`, exclusive with `preserve_precomputed_labels`). Manifest source key `in_eval` (default true).
- **Launcher:** generic gains `LOSS_SPAN_MARKERS="${LOSS_SPAN_MARKERS:-}"` → `--loss_span_markers`;
  `launch_e18.sh` passes it through (no default change).
- **Arm R:**
  ```bash
  TOK=/home/ksopyla/dev/hf_home/datasets_tok_smollm3_32k
  E18_STAGE=32k EXPERIMENT_ID=E18b SKIP_PRETOKENIZE=1 \
  MANIFEST=$TOK/e18b_lm_ret05_manifest.json PRETOKENIZED_MANIFEST=$TOK/e18b_lm_ret05_manifest.json \
  LOSS_SPAN_MARKERS=128103,128104 \
  MODEL_NAME_OR_PATH=Cache/Training/perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943/checkpoint-9030 \
  LEARNING_RATE=0.002 MUON_ADAMW_LR=4e-5 WARMUP_STEPS=100 LR_SCHEDULER_TYPE=cosine \
  TARGET_TOKENS=500000000 PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/launch_e18.sh
  ```
  Arm 0: same without `MANIFEST/PRETOKENIZED_MANIFEST` overrides (base manifest), `LOSS_SPAN_MARKERS` kept.
  Arm D: arm R env + `PAR_MODE=dense MODEL_NAME_OR_PATH=Cache/Training/perceiver_ar_dense_H768L1g1s12N2048_20260907_193351/final`.
- **Probes per arm (`final`):** `--probe passkey --context_lengths 8192,32768,65536,131072 --trials 8`;
  the same with `--frame_token 128103` (diagnostic: question ends with START — separates "no circuit"
  from "frame-bound circuit"); `--probe buckets` on the 32k (8 rows) and 16k (64 rows) sets;
  `--probe tasks --tasks_dataset $TOK/e18b_retrieval_32k/eval --markers 128103,128104`; passkey at 32k
  with `--reach_window 8192` (must collapse on arm R if the read is the retrieval channel).
- **Chain:** `Cache/jobs/e18b_chain.sh` (R → 0 → D, exit markers, probes after each), launched only after
  E18 arm B finishes and the prep job (§8) is done.

## 7. Tests & smoke
- `tests/test_retrieval_mix_dataset.py`: rows are exactly `S` long; sources precede targets by ≥ 1024;
  items never overlap; `labels_from_span_markers` recovers exactly the values + END; supervised fraction
  within [0.04, 0.15]; filler ids come only from the fake base manifest's *train* dirs; merged manifest:
  weights sum to 1, retrieval token share within 10% of `--fraction`, `in_eval` false, base entries intact.
- `tests/test_data_collators.py` (new): marker rule on a mixed batch (rows with/without START), END included,
  unmatched START labels to row end, START-less rows are plain LM, `preserve_precomputed_labels` +
  markers raises, pad positions −100, `doc_ids`/packed path unaffected.
- `tests/test_dataset_preprocess_manifest.py` (new, tiny tmp datasets): `in_eval:false` skips the source
  in eval and keeps it in train; default unchanged.
- `tests/test_long_context_probes.py`: filler concatenation reaches lengths beyond any single row; `tasks`
  probe accuracy on a tiny synthetic set is 1.0 for an oracle and in [0,1] otherwise; `--frame_token`
  appends the id to the question.
- `tests/test_launch_e18.py`: `LOSS_SPAN_MARKERS` and `MANIFEST=` override flow to the parser.
- Local smoke (CPU): build a 4-row retrieval set from a fake 2-source manifest, run 2 training steps of
  the tiny `perceiver_ar` config with the marker collator, assert finite loss and that only marker spans
  contribute (compare against a manual −100 mask).
- Remote smoke (Polonez, before the chain): 20 steps of arm R at 32k on 1 GPU; assert the log shows the
  merged manifest, 5 sources, `loss_span_markers` echoed, and `real_tokens_per_second` ≈ stage B's.

## 8. Risks & tradeoffs
- **Frame gap (K1):** the trained trigger is START; the passkey question has none. Cheapest signal: the
  `--frame_token` diagnostic — if passkey passes *with* START but fails without, the circuit exists and is
  frame-bound → the K1 fix (more task families, incl. frame-free ones such as `KEY value … KEY value`
  with labels on the repeat) is the right one; if both fail, the read did not learn to address.
- **Extrapolation (K2):** RoPE on the read at 4× its trained span. Cheapest signal: passkey 64k vs 128k on
  arm R; fix iteration = `PAR_GLOBAL_LOGIT_SCALE=log` rerun (learned scale, needs training).
- **Protocol (K3):** arm 0 vs stage A buckets, first thing checked when arm 0 lands; a regression halts
  the chain (D is not run).
- **Row-weight mistake:** weights are per row; the builder derives `w` from token share and writes both
  numbers into `retrieval_meta`; the tasks/LM ratio is verified from the first 200 logged batches
  (`train/data/mean_sequence_length` rises from ~3.3k by ≈ 5% × 32k ≈ +1.5k when correct).
- **Prep time on Polonez:** the merged manifest needs `manifest_token_stats` and the length cache
  (~35 min each, single-process because forked workers die on this host) — run in `Cache/jobs/e18b_prep.sh`
  on CPU *while arm B trains*, so the chain launches without the NCCL-timeout risk stage B hit.
- **Reserved-id collisions:** `split_special_tokens=false` in the mix means a literal
  `<|reserved_special_token_100|>` in web text would tokenize to START; probability negligible, effect
  benign (that LM row gets few/no labels). Logged by the collator once if it ever happens.

## 9. Code sketches (`# sketch`)
```python
# sketch — data/data_collators.py
def labels_from_span_markers(ids: list[int], start: int, end: int) -> list[int]:
    """-100 everywhere except tokens strictly after a START up to and including the next END;
    an unmatched START labels to the row end. Rows without START are the caller's business."""

class DataCollatorForCausalLM:
    def __init__(..., preserve_precomputed_labels=False, loss_span_markers: tuple[int, int] | None = None): ...
    # in __call__: if loss_span_markers and start in ids: labels[i,:L] = labels_from_span_markers(ids, *markers)

# sketch — scripts/build_retrieval_mix_dataset.py
def iter_retrieval_rows(filler_iter, n_rows, context, rng, *, start, end, bos, eos,
                        items=(8, 24), short_len=(8, 16), span_len=(64, 512), min_gap=1024,
                        key_ids=(1000, 2000), key_len=3): ...   # yields LM-schema rows
def row_weight_for_token_share(fraction, base_mean_row_tokens, context) -> float: ...
def merge_manifest(base, retrieval_src, weight) -> dict: ...  # base weights * (1-w), in_eval False

# sketch — evaluation/long_context_probes.py
def build_filler(rows, budget): ...            # concatenate rows cyclically until >= budget
def probe_tasks(model, args, device): ...      # argmax accuracy on labels_from_span_markers positions
# passkey: question += [frame_token] when args.frame_token is set
```

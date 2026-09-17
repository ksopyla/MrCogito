# E28 — Implementation Plan

- **Spec:** [E28_exclusive_cogitoprobe_bits.md](E28_exclusive_cogitoprobe_bits.md) · **Status:** done_failed (K1)
- **Authored by:** `implementation-plan` · for → `research-implement`

> CogitoProbe-bits on exclusive E21 at 1k then 4k. Generate locally. Do not Hub upload.
> Do not `length_group` pack. Do not train 32k first.

## 1. Source & fit
- **Origin:** diagnosis packing/32k claim · literature 32k protocol · CogitoProbe
  [concept_compression_probe_suite.md](../../engineering_specs/concept_compression_probe_suite.md)
  (PR 39).
- **Synthesis verdict:** Adapt `fixed` vs `scaled` length ladder; Reject NIAH/PPL gates.
- **Architecture mapping:** exclusive read + packed-answer CE on a known prize.
- **Boldness check:** paying unique-bit exam, not FineWeb continuation.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `PerceiverARLM` exclusive inplace | reuse | `nn/perceiver_ar_lm.py` |
| CogitoProbe generator / schema | reuse as-is (PR 39) | `data/concept_probes/`, `scripts/build_concept_probe_datasets.py` |
| `DataCollatorForCausalLM` | reuse `preserve_precomputed_labels` | `data/data_collators.py` |
| `launch_e18.sh` | extend `PAR_MESSAGE_*` env | `scripts/launch_e18.sh`, `scripts/train_concept_pretraining_multigpu.sh` |
| `evaluation/bapo_metrics.py` recovered bits | Adapt to `prize_bits` column | eval helper (reusable) |

## 3. Forward pass
Same exclusive E21 as E25 inplace. `S ∈ {1024, 4096}`. Boundary token =
`atom_table.json` → `markers.query.id` (surface `Q`). Receiver = tokens at/after `Q`.
Wave A compressor flags (identity mean / AE / `key_spans`) copied from the winner.

## 4. Inputs & data
- **Dataset:** `Cache/concept_probes/full_1k4k/bits/{train,validation,test}` from
  `--scale full --lengths 1024 4096 --families bits --seed 20260916`.
- Hub id **name**: `ksopyla/cogito-probe-bits`. Load with `load_from_disk`, not
  `load_dataset("ksopyla/...")`.
- **Collator:** `preserve_precomputed_labels=True` (labels already `-100` except answer).
- **Packing:** `BATCH_PACKING_MODE=none` / off. Each row `len(input_ids)==seq_len`.
- Filter by `seq_len` and `variant` so 1k and 4k, `fixed` vs `scaled`, are separate runs
  or clearly tagged eval slices.

## 5. Loss & training objective
Teacher-forced CE on `labels ≠ -100` only. Score recovered bits against `prize_bits`
as the card specifies.

## 6. Config & launch
New launcher knobs (defaults keep E18 loadable):

| env | config field | default |
|---|---|---|
| `PAR_MESSAGE_BOUNDARY_TOKEN_ID` | `message_boundary_token_id` | `-1` |
| `PAR_MESSAGE_COMPRESS_RATIO` | `message_compress_ratio` | `16` |
| `PAR_MESSAGE_SLOTS_INPLACE` | `message_slots_inplace` | `False` |
| `PAR_MESSAGE_IDENTITY_SLOTS` | `message_identity_slots` | `False` |
| `PAR_MESSAGE_PREFIX_AE` | `message_prefix_ae` | `False` |
| `PAR_MESSAGE_GLOBAL_ANCHORS` | `message_global_anchors` | `none` |

`TOKENIZER_NAME=HuggingFaceTB/SmolLM3-3B`. `MAX_SEQ_LENGTH` = rung length.
Generate command in the spec. Dense: `PAR_MODE=dense`, boundary `-1`.

## 7. Tests & smoke
- Row `len(input_ids)==seq_len`; labels ignore filler; `gap` > local window.
- Message mask: no raw prefix token after `Q` except slots/anchors.
- Local generate `--scale pilot --lengths 1024 --families bits` (tiny) then 2 train steps.

## 8. Risks & tradeoffs
- **Risk:** hashed embed + SmolLM3 vocab vs DNA alphabet — optimization, not the claim.
  **Cheapest signal:** dense S0 @1024.
- **Risk:** PR 39 not merged — Wave B blocked on that generator; Wave A is not.
- **Fallback:** do not silently download Hub; do not upload.

## 9. Code sketches
```python
# sketch — eval slice
ds = load_from_disk("Cache/concept_probes/full_1k4k/bits/test")
ds = ds.filter(lambda r: r["seq_len"] == 1024 and r["variant"] == "fixed")
# recovered = max(0, prize_bits + sum log2 p(gold_t) for t where labels != -100)
```

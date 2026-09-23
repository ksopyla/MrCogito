# E27 — Implementation Plan

- **Spec:** [E27_hybrid_key_anchors.md](E27_hybrid_key_anchors.md) · **Status:** implemented and run 2026-09-16 · experiment killed (S1 miss; see spec Result)
- **Authored by:** `implementation-plan` · for → `research-implement`

> Hybrid `(a,b)` on exclusive E21: identity raw keys on DNA **key spans**, r=16 frozen
> mean for the rest. Do not substitute type_marks-at-r=1 (E25: extra count 0).

## 1. Source & fit
- **Origin:** diagnosis mean-pool smear (MATCH r=16 @512 = 0 bits) · literature lever 2
  (BAPO MATCH needs count of addressable items) ·
  [e21_levers_from_literature.md](../../4_Research_Notes/e21_levers_from_literature.md).
- **Synthesis verdict:** Adapt hybrid anchors; Reject width sweeps and type_marks-at-r=1.
- **Architecture mapping:** bottleneck K/V mask only (sparse `b` + pooled `a`).
- **Boldness check:** compression of values with identity keys, not “set r=1”.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `KVCompressor` / exclusive inplace mask | reuse as-is | `nn/perceiver_ar_lm.py` |
| `build_message_anchors` / `MESSAGE_GLOBAL_ANCHORS` | extend: add `key_spans` | `nn/perceiver_ar_lm.py` |
| `ArchSpec.message_global_anchors` | extend allowed set | `evaluation/bapo_models.py` |
| probe CLI `--message_global_anchors` | reuse (already a string) | `verification/bapo_capability_probe.py` |
| DNA `recall` blocks | reuse `[keymark, *key, *val]` | `data/symbolic_tasks.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `S`=512, `r`=16, `H`=256, `g`=1 kv-head, `dh`=32.
```
ids [B,S]
  → embed + SWA (QUERY = local doc start)
  → global read:
       replace[j] = inplace r=16 frozen mean of block j          # a
       anchor[t]  = True iff t in a sender key span              # b  (new)
       vis = replace ∪ anchor
       answer tokens attend vis only
  → SWA stack → CE on packed y
```
Key span: after each sender `keymark`, the next `cfg.key_len` *symbol* tokens
(`bridge` key_len=2). Do **not** mark `*val`. `keymark` itself may be marked or not
(prefer not: type_marks already tested marks).

## 4. Inputs & data
- **Dataset:** on-the-fly `recall_single` (`n_distractors=0`).
- **Collator:** probe packed y. No HF shard.

## 5. Loss & training objective
Packed answer CE only. No AE (E26).

## 6. Config & launch
- **New:** `MESSAGE_GLOBAL_ANCHORS` includes `"key_spans"`. Optional
  `message_anchor_key_len: int = 0` meaning “use DNA `key_len` from the batch meta /
  a probe flag `--key_len` already present.”
- **Launch:** see spec. Anchors-only control: evaluate with slots zeroed / invalid
  `slot_doc=-1` while anchors stay (specify `--message_override` or a probe flag
  `slots_off`; default off).

## 7. Tests & smoke
- `tests/test_perceiver_ar_message.py`: `key_spans` marks exactly the key tokens after
  `keymark`, not values; extra non-slot count > 0 at r=16; r=1 identity extra count
  may be 0 (keys already replace slots) — skip hybrid scoring at r=1.
- Probe smoke: 20 steps, loss finite.

## 8. Risks & tradeoffs
- **Risk:** key and value sit in the same 16-token block so the mean still smears
  values. **Cheapest signal:** S1 miss + S2 pass (anchors real, values still 0).
  **Fallback:** E26 AE on the value block, not more marks.
- **Risk:** K4 leak if value tokens are accidentally marked. Unit-test the span.

## 9. Code sketches
```python
# sketch
MESSAGE_GLOBAL_ANCHORS = (..., "key_spans")

def build_message_anchors(..., key_len: int = 2):
    if mode == "key_spans":
        is_mark = input_ids == keymark_id  # sender only
        # for d in 1..key_len: positions keymark+d on the same side/doc
        ...
```

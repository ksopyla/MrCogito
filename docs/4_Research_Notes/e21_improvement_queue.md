# E21 improvement queue — exclusive slots that address, then a short length ladder

Undated mutable note (research ideas + launch order). **Not** a living roadmap.
Frozen specs: [E27](../experiments_specs/done_failed/E27_hybrid_key_anchors.md) ·
[E26](../experiments_specs/done_failed/E26_prefix_ae_exclusive_slots.md) ·
[E28](../experiments_specs/done_failed/E28_exclusive_cogitoprobe_bits.md) ·
[E29](../experiments_specs/done_failed/E29_exclusive_cogitoprobe_bind.md) ·
[E19](../experiments_specs/ahead/E19_looped_slot_refinement.md).
Inputs (do not re-diagnose):
[E18/E21 vs dense](e18_e21_dense_baseline_diagnosis_20260916.md) ·
[levers from literature](e21_levers_from_literature.md) ·
CogitoProbe suite (PR 39, not on Hub):
[engineering spec](../engineering_specs/concept_compression_probe_suite.md).

> **Wave A (2026-09-16).** DNA MATCH @512 r=16: E27 and E26 both **S1 miss** (3.03 and 1.03 bits
> vs ~36). Recorded in the ledger; INDEX skipped.

> **Wave B (2026-09-16, recorded 2026-09-17).** CogitoProbe @1024: E28 **K1** (dense 6.5% /
> 0.21 bits; exclusive `fixed` 0 bits). E29 hop **0.078 bits** / 4.4% (no dense S0). 4k
> skipped. GPUs idle. Do not relaunch. E19 stays gated.

> **Reconciled headline.** The long-range object is not a content-addressable memory;
> dense learns the same DNA/language tasks; E21 has no 32k-only story (DNA walls at
> 696–1536; the 125M run saw mean length ~3.2k). Literature: count-starved for
> MATCH-class addressing, not width-starved; reject one reasoning hop; prefer
> exclusive slots + weak prefix AE; pay for tokens whose label is in the prefix;
> loop the set only after the channel carries content.

## Rejected as the next bet

- One extra exclusive attend / `message_extra_slot_attends` / pause tokens as a reasoner (E25: hops-only).
- `global_layers=2` billed as latent reasoning (E25 SELECT 1024 still 0).
- Width-first / `head_dim` / H sweeps; SVD or pretrained token init.
- Learned `u`/`delta` under answer-only CE (E25: document mean, 0 bits).
- Retrain the 125M E21 LM at 32k with `length_group` packing.
- DNA 8k/16k/32k hunts until MATCH-under-`r≥4` passes at the current walls.
- Arith/brackets as a *semantic richness* exam (CogitoProbe verdict: structure control only).

## Run order (rank ≠ ID)

IDs follow the literature lever numbers (E26 = prefix AE, E27 = hybrid `(a,b)`).
**Launch E27 first** — one new anchor mode, DNA we already have.

| Rank | ID | Wave | Claim (one line) | Attacks | GPU-h |
|---|---|---|---|---|---|
| 1 | **E27** hybrid key anchors | A | r=16 frozen-mean gist + identity keys on DNA key spans recovers MATCH | diagnosis mean-pool smear; lit lever 2 (count / `(a,b)`) | ~2 |
| 2 | **E26** prefix AE exclusive slots | A | local block reconstruction, not answer CE, makes r=16 slots MATCH-addressable | diagnosis learned-pool wipe + smear; lit lever 1 | ~4 |
| 3 | **E28** CogitoProbe-bits | B | exclusive r=16 is a *capacity* story (`scaled`) and a *length* story (`fixed`) at 1k→4k | diagnosis “no 32k-only length story”; lit count-starved | ~8 |
| 4 | **E29** CogitoProbe-bind | B | exclusive slots bind `(entity, attribute, value)`, not a bag/gist | diagnosis document embedding; lit MATCH-class on language atoms | ~8 |
| 5 | **E19** looped slot refinement | B, gated | `R≥4` tied loops over load-bearing slots lift AST/Dyck structure; `R=1` does not | lit lever 4 (LOTUS); diagnosis extra-hop-is-hops-only | ~4 |

Wave A uses on-the-fly DNA (`verification/bapo_capability_probe.py`). Wave B generates
CogitoProbe **locally** (`scripts/build_concept_probe_datasets.py`); Hub ids name the
families; **do not `hf upload`**. E23 (perceiver_concept exclusive LM) stays a parallel
language bet, not this queue.

## Wave A — data we already have

Primary rung (both E27 and E26): packed DNA `recall_single`, seq=512, r=16 inplace
frozen-mean, H=256 SSMax log — E25 scored **0 bits** for E21 vs live E18 **47.94**
(gate 0.75× ≈ 36 bits) and identity r=1 **43.08 bits**. Dense S0 ≥ 75% is already
true on this recipe. `e18_local` stays the leak check.

Do not grow DNA to 8k/16k/32k here. INDEX @1024 r=16 (53.82 bits) is a don’t-regress
control, not a second hypothesis.

**P0 (0 GPU-h, CPU, no spec).** Exclusive-mask dump at the first answer token for
SELECT 692 vs 696, r=1 — diagnosis leftover cliff. Unit test, not an architecture bet.

## Wave B — CogitoProbe (PR 39; nothing on the Hub)

Generate on Odra/Polonez, seed `20260916`, tokenizer `HuggingFaceTB/SmolLM3-3B`:

```bash
uv run python scripts/build_concept_probe_datasets.py \
  --scale full --seed 20260916 --lengths 1024 4096 \
  --families bits bind arith props \
  --out_dir Cache/concept_probes/full_1k4k --overwrite
```

Do **not** pass `--lengths 8192 16384 32768` on the first train. Those rungs exist in
the generator for a later grow-up after 4k `fixed` passes. Message boundary = CogitoProbe
marker `query` (surface `Q`; id from `atom_table.json`). Rows are already padded to
`seq_len` with packed `labels`; **do not** `BATCH_PACKING_MODE=length_group`.

| Hub id (proposed) | Family claim | Role in this queue |
|---|---|---|
| `ksopyla/cogito-probe-bits` | C-slot unique-bit capacity vs haystack (`fixed` vs `scaled`, 1k→32k) | **E28** train + gate |
| `ksopyla/cogito-probe-bind` | `(entity, attribute, value)` vs bag-of-tokens / gist | **E29** train + gate |
| `ksopyla/cogito-probe-arith` | AST + Dyck-3 vs eval-only calculator | **E19** structure eval. Keep as structure control; reject as semantic richness. Primary `subexpr`, stack `match`, shortcut `eval`. Bare digit/op tokens are 1 SmolLM3 token; glued strings merge — inject atom ids. |
| `ksopyla/cogito-probe-props` | proposition set vs filler n-grams | **E29** gist control on the same checkpoint (filler shuffle holds; proposition shuffle kills) |

Dense S0 ≥ 75% on seq=1024 packed answers before any exclusive number is scored.
`message_override ∈ {real, none, swapped, raw}` on every Wave B checkpoint.
Report 1k and 4k on the same prize (`variant=fixed`). Do not gate on NIAH, last-256
PPL, or arith `eval`.

## What each run falsifies

- **E27 fail:** identity keys plus pooled values still cannot MATCH → smear is in the *value* pool; E26 AE is the next write objective, not more anchors.
- **E26 fail** (AE saturates, MATCH flow < 0.05): r=16 cannot hold MATCH even with the published compressor repair → stop pooling keys; keep hybrid or r=1 for `b`.
- **E28 `fixed` fail at 1k:** exclusive channel still dead on a paying packed exam (not a length story).
- **E28 `fixed` pass at 1k and 4k, `scaled` also passes:** not count-starved at this prize; grow length, do not grow C.
- **E28 `fixed` pass, `scaled` collapse:** capacity result DNA’s 64-bit copy cannot state.
- **E29 `attr_color` pass, `hop_friend_place` chance:** labels not bindings (gist).
- **E19 `eval` only:** calculator; not a reasoner. `R=1` ≈ `R=4`: extra hop, already rejected.

## Depends on

- PRs #37 (literature) and #38 (diagnosis) for the input notes.
- PR #39 for CogitoProbe generators and cards.
  Wave A does not wait on 39. Wave B does not wait on a Hub publish.

# E28 — Exclusive r=16 on CogitoProbe-bits: capacity vs haystack length

- **Status:** draft 2026-09-16 (queue rank 3, Wave B; awaiting KS approval) · not launched
- **Serves:** the honest length/capacity story E21’s 125M LM never instantiated. Queue:
  [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md).
- **Implementation plan:** [E28_exclusive_cogitoprobe_bits_plan.md](E28_exclusive_cogitoprobe_bits_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-16 · closed —

> One claim, the bits family’s claim: a C-slot exclusive read recovers at most ~C·k
> unique prefix facts; `fixed` isolates haystack distance, `scaled` isolates capacity.
> DNA stays the ln(4) bandwidth exam. **Do not train 32k until 4k `fixed` passes.**
> Dataset is generated locally from PR 39; **not uploaded** to the Hub.

## Hypothesis
If the Wave A winner (E27 hybrid and/or E26 prefix AE; else frozen-mean r=16 inplace
exclusive) is trained on packed-answer CE from **`ksopyla/cogito-probe-bits`** at
seq=1024 then 4096, with the CogitoProbe `query` marker as the E21 boundary, then
**`variant=fixed` recovered bits ≥ 0.75 × dense** at **both** 1k and 4k (length: same
prize, longer haystack) **because** the labels are determined by far key→value maps
and the exclusive mask makes slots the only route — and **`variant=scaled` at 4k
falls** once `n_query·log2(|V|)` exceeds the slot budget (capacity: DNA’s 64-bit
copy cannot state this).

## Builds-on
- **Foundation:** `perceiver_ar` exclusive path; CogitoProbe generator
  `data/concept_probes/` + `scripts/build_concept_probe_datasets.py` (PR 39);
  collator `preserve_precomputed_labels`; launcher `scripts/launch_e18.sh` /
  `scripts/train_concept_pretraining_multigpu.sh`. No Hub download.
- **Init / checkpoint:** random, small E25-scale model (H=256, hashed embed,
  SmolLM3 tokenizer). Wave A compressor flags copied, not 125M E18 weights (deleted).
- **Baseline to beat:** dense S0 on the same shards (must be ≥ 75% at 1024). Uncompressed
  E18 (r=1 / no boundary) is the raw-message ceiling. E25 DNA r=16 MATCH @512 = **0 bits**.
- **Materially new:** paying packed unique-bit exam at 1k/4k with `fixed` vs `scaled`.
  Not FineWeb CE. Not a 32k `length_group` rerun.

## The architectural bet
Exclusive compressed prefix (Wave A write) + CogitoProbe-bits labels (every supervised
token is a far fact). `fixed`: 16 facts at every length (1k prize 40 bits; 4k still ~40–60).
`scaled`: fact count grows with length (4k prize mean 60, 32k scaled 640 — later).
Boundary = marker `query` (`Q`). **Out of scope:** bind/arith/props as *this* spec’s
gates (score the checkpoint as transfer; E29/E19 own those claims). Extra hop. 32k train.

## Why this is not a safe retread
Diagnosis: E21 never saw 32k rows and DNA walls sit inside 1.5k, so “potential only at
32k” is untested *and* undermined. This is the missing length/capacity split on a
**known prize**, at a budget Odra can run tomorrow (1k then 4k). Literature: ICAE/Fine-KV
count starvation; HELMET/100-LongBench “same prize at two lengths.”

## Success criteria (set BEFORE running)
Ceilings are **this replica’s dense** recovered bits (card: teacher-forced acc +
`max(0, prize_bits + Σ log2 p(gold))`).
- **S0:** dense token-acc ≥ 75% on seq=1024 `recall_packed` (both variants).
- **S1 length:** E21 `fixed` recovered bits ≥ **0.75 × dense** at seq=1024 **and** seq=4096.
- **S2 load-bearing:** `real − none` > 0 on the answer span; `swapped ≤ none` (not steering).
- **S3 capacity (reported, kills only if inverted):** `scaled`@4096 recovered fraction of
  prize **strictly below** `fixed`@4096 if prize(scaled) > prize(fixed). If `scaled` matches
  `fixed` at higher prize, count-starvation is false at this C — grow length, not C.

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% @1024 — generator/tokenizer/labels; do not score E21.
- **K2:** `e18_local` / window `< gap` solves the prize — leak.
- **K3:** S1 miss at 1024 `fixed` after the step budget while dense saturates — exclusive
  r=16 still not a memory on a paying exam. Do not launch 4k, 8k, or 32k.
- **K4:** `swapped > none` — gist/steering (E21 K3 / StateBridge). Stop.

## Plan
- **Data:** local `Cache/concept_probes/full_1k4k/bits` from
  `build_concept_probe_datasets.py --scale full --lengths 1024 4096 --families bits`.
  Hub id **name only**: `ksopyla/cogito-probe-bits`. Seed 20260916. **No upload.**
- **Compute:** Odra 1–2×3090. **~8 GPU-h** (dense + E18 + E21 at 1k, then 4k if S1@1k).
- **Steps:** packed-answer CE, ~10–20 epochs on full-1k train (4096 rows), then 4k
  if S1 passes. `BATCH_PACKING_MODE` off (rows already length `seq_len`).
- **Launch (after generate):**
  ```bash
  EXPERIMENT_ID=E28 MODEL_FAMILY=perceiver_ar PAR_MODE=perceiver \
  HIDDEN_SIZE=256 PAR_GLOBAL_LOGIT_SCALE=log \
  PAR_MESSAGE_BOUNDARY_TOKEN_ID=<atom_table.json markers.query.id> \
  PAR_MESSAGE_COMPRESS_RATIO=16 PAR_MESSAGE_SLOTS_INPLACE=True \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM3-3B MAX_SEQ_LENGTH=1024 \
  PRESERVE_PRECOMPUTED_LABELS=true BATCH_PACKING_MODE=none \
  bash scripts/launch_e18.sh
  ```
  Dense: `PAR_MODE=dense` and message off. Copy Wave A compressor flags
  (`PAR_MESSAGE_IDENTITY_SLOTS` / prefix AE / `key_spans`) from the winner.
- **New foundation code:** wire `PAR_MESSAGE_*` env knobs through the existing
  launcher (reusable; defaults off). Dataset loader for CogitoProbe disk shards.

## Result
- Run id: `<run_id>`
- WandB: —
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: —

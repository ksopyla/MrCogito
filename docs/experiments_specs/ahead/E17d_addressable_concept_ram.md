# E17d — Addressable concept RAM (sparse location write/read)

- **Status:** draft — spec + plan written; no code until go-ahead
- **Serves:** Priority 1 / the E17 family platform bet: make Gemma's concept
  banks a *working memory*, now with the inductive bias E17c lacked — slots are
  **addresses**, not a densely mixed latent soup. Keep exploring this
  architecture; do not retune E17c with more tokens.
- **Implementation plan:** [E17d_addressable_concept_ram_plan.md](E17d_addressable_concept_ram_plan.md)
- **ID note:** drafted as E18, renamed to **E17d**. No other experiment occupied
  E18. This is the same family as E17–E17c (`per_layer_banks` + carry pressure on
  Gemma-3-1B); letter suffixes are the project's family-variant scheme (E10b–e,
  E16a–b, E17a–c). E18 stays free for a later new family. GitHub PR #18 is not
  an experiment ID.
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-15 · closed —

> One experiment = one coherent architectural hypothesis: **concept banks are
> RAM**. Sparse addressed write + sparse addressed read + unused-slot invariance,
> trained under the same causal carry pressure that already forced use in E17c.
> Not a gate-init A/B, not E11 mem-tokens, not GSA/Titans associative writes.
> Implementation is args/config over the shared entrypoint; new knobs are
> reusable foundation, not a fork.

## Hypothesis
If each of E17c's four per-layer concept banks becomes an **addressable RAM** —
slot index `i` is a location with a learned address embedding; a few write heads
emit a hybrid location / content / allocation address and **hard-top-k
erase-then-add** so unaddressed rows are exactly unchanged; reads retrieve only
top-k locations — and we keep E17c's 50% causal carry dropout, then by 300M
tokens carryless first-64 Δpermutation stays **≥0.20** while every bank's
within-sample RankMe stays **≥38.4** and write-mass entropy sits in
`(ln 4, 0.7 ln C)`, because information is stored at sparse addresses instead of
being Hopfield-averaged into one bank.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` `BackboneConceptLM` with
  `concept_io_mode="per_layer_banks"`; E17c's dedicated reads, untied writers,
  block-causal loop, and carry-pressure CE; shared
  `training/train_concept_pretraining.py` →
  `scripts/train_concept_pretraining_multigpu.sh` → `scripts/launch_e10.sh`.
  New reusable modes on the existing config (`concept_write_mode`,
  `concept_read_mode`), not a new model class or trainer.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed
  42, C=128, K=512, four banks, seq 4096. Do **not** warm-start E17c: the state
  transition and the meaning of a slot change (contents start blank; addresses
  are a new embedding). Unambiguous training verdict required.
- **Baseline to beat:** E17c
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241`
  (`checkpoint-2370`): carryless first-64 Δpermutation **0.594** (bank 0 **0.38**,
  others ≤0.03), RankMe **6.75** (bank 1 **1.84**), Δpermutation_beyond **0.013**,
  free-run `real` greedy @256 **0.23/0.53**. E17b remains the dense-additive
  no-pressure control (Δbeyond 0.0055, writes closed, RankMe 68).
- **Materially new:** E17c privatized the cell and forced the path, but still
  **updated all 128 slots every block** (`gated_replace` interpolates every
  row). E17d gives slots *positions*, sparse write, sparse read, and the unused-
  slot invariant. That is a different memory datatype, not a retune of α, k, or
  pressure weight.

## The architectural bet

Keep E17c's outer loop (four depth-private banks at Gemma global layers
5/11/17/23; bank `z_g^b` used to predict block `b` contains only blocks `<b`;
50% post-first-block carry dropout with first-64 CE ×4). Replace the dense
read/write *inside* each bank.

Each bank is a RAM of C=128 rows:

```text
address_emb : [C, H]     learned embedding of slot index 0..C-1  (the address)
z           : [B, C, H]  memory CONTENTS, initialized at 0       (the tape)
usage       : [B, C]     EMA of write mass, init 0               (DNC-lite)
```

**Write (per bank, once per 512-token block), NTM erase-add + SAM top-k + DNC
allocation — one controller, not three experiments:**

```text
pool  = attention_pool(h_block; n_heads = write_num_heads)     # [B, H, H]
for each write head h = 1..H_w:                                 # H_w = 4
  k_loc, k_cnt, erase, add, g_mix, β = Linear(pool_h)
  scores = g_mix · (z · k_cnt) + (1-g_mix) · (address_emb · k_loc)
           - λ · usage                                          # prefer unused
  w      = topk_renorm(softmax(β · scores), k = K_write)        # K_write = 4
  z      = z ⊙ (1 - w ⊙ erase) + w ⊙ add                        # unaddressed: Δz = 0
  usage  = (1-μ) · usage + μ · w
```

`topk_renorm` zeros all but the k largest weights, then renormalizes. The mask
is treated as a straight-through estimator so unselected rows get exact zero
write (SAM unused-slot invariance). Softmax leakage onto the other C−k rows is
not allowed.

**Read (per token, at the wrapped global layer):** replace dense
`scaled_dot_product_attention` over all C slots.

```text
for each token q in the current block:
  scores = g_r · (z · k_cnt(q)) + (1-g_r) · (address_emb · k_loc(q))
  w      = topk_renorm(softmax(β_r · scores), k = K_read)       # K_read = 8
  read   = w · V(z)
h = GemmaLayer(h) + tanh(read_gate) · read
```

Default knobs (registered, not a sweep): `K_write=4`, `K_read=8`, `H_w=4`
(reuse existing `write_num_heads`), `λ=1`, `μ=0.1`, `g_mix` init 0.5 with
location-heavy bias allowed to learn, `β` init 1 and learned ≥1, `z` contents
**zero-init**, `address_emb` learned, `READ_GATE_INIT=0.1` as E17c.

**Why location + allocation, not content-only.** Collapsed or blank rows are
indistinguishable under cosine (DNC's reason for allocation; SAM rejected
content-write into empty memory). Address embeddings make index `i` a position;
usage bias implements "write to unused positions." Content lookup is for
*updating* a row that already holds a matching key.

**Causal carry pressure is in scope.** E17/E17b showed that without it, writes
die under plain CE. E17c showed that with it, dense writes collapse. The claim
is that **the same pressure on a RAM does not collapse**, because the mixer
cannot dump the whole suffix into every slot. Dropping pressure would re-run
the E17b failure mode and would not test the RAM hypothesis.

**Out of scope (follow-ups only if E17d passes):** DNC temporal links; NTM shift
convolution; GSA/DeltaNet/Titans associative matrices; E11 in-sequence
mem-tokens; changing C/K/backbone/LoRA/optimizer/data mix; synthetic copy/bAbI
curriculum; E08 latent reasoning; raising write-init or pressure weight; 1B
tokens on this cosine.

## Why this is not a safe retread
E17c already had per-slot gates, dedicated reads, untied writers, and pressure —
and still wrote *all* slots. Literature on NTM/DNC/SAM vs GSA/Infini/Hopfield
says that distinction is the collapse mechanism, not a knob
([addressable_memory.md](../../literature_review/addressable_memory.md)).
E11 prepends all C tokens into the sequence (dense self-attn over memory).
E17a only unties writers (E17c already did). This experiment changes the
datatype of a concept from "one more query vector in a set" to "row `i` of a
tape." Analogy: RAM vs a fully connected hidden state; sparse coding vs dense
Hebbian overwrite; Kanerva locations vs a Hopfield global average.

## Success criteria (set BEFORE running)
Evaluate at 100M (kill gate), 300M (mechanism verdict), and only then consider
a separate 1B cosine.

- **Primary (300M, both required):**
  1. Carryless first-64 `CE(batch-permuted all banks) − CE(real) ≥ 0.20` nats
     on blocks 2–7; 95% CI lower bound > 0.10 (E17c's mechanism win must hold).
  2. Within-sample RankMe **≥ 38.4/128 for every bank** (the geometry E17c
     failed). Also report RankMe of **written** slots (usage > 1/C) vs
     unwritten.
- **Addressing (300M, the RAM-specific gate):**
  - Write-mass entropy per bank, mean over holdout, in **`(ln 4, 0.7 ln 128)` ≈
    (1.39, 3.40)** — not WTA, not dense.
  - Median number of slots with write mass > 1/C per sample in **[4, 48]**.
  - Unused-slot invariance: on a diagnostic forward, `max |Δz|` on the
    complement of the top-k write mask **< 1e-6**.
  - At least **2/4** banks have carryless first-64 Δpermutation CI lower > 0
    (E17c was bank-0-only; RAM should not recreate a single absorbing cell).
- **Transfer (300M, required to justify 1B):** normal-context
  `Δpermutation_beyond ≥ 0.02` at positions ≥1024 (E17c's 0.013 stop).
- **Generation (300M, non-regression):** `real` greedy @256 distinct-1 **≥ 0.20**
  and REP-3 **≤ 0.60** (do not fall back to E16b). Absolute 1B bar stays
  d1≥0.25 / REP-3≤0.50 with `real ≥ zero`, evaluated only if 1B launches.
- **No broad LM regression:** held-out eval loss **≤ 2.36** (E17c 2.276, E17b
  2.264 + 0.10).

## Kill criteria (set BEFORE running)
- **Before GPU training:** unit tests must prove (a) unaddressed `Δz == 0`,
  (b) no intra-block or cross-block future leakage, (c) E17c legacy path still
  numerically equivalent at default config, (d) nonzero gradients through
  address embeddings, write heads, and top-k reads.
- **Any checkpoint:** non-finite loss/gradients; three consecutive eval-loss
  increases; any bank RankMe **< 19.2/128**.
- **100M:** stop if carryless first-64 Δpermutation **< 0.05**, **or** write-mass
  entropy **< ln 2** on all banks (single-slot WTA), **or** all four banks'
  mean write mass on the top-1 index **> 0.80**.
- **300M:** stop rather than spend 1B if the primary pair fails (carryless
  <0.20 **or** any-bank RankMe <38.4), **or** write entropy **> 0.95 ln C**
  (addressing did not sparsify — this was dense mix in disguise), **or**
  `Δpermutation_beyond < 0.02`, **or** `real` REP-3 **> 0.80** and worse than
  `zero`.

## Plan
- **Data:** immutable `e16b_long_4k_v1` Gemma-tokenized manifest; raw causal LM;
  max sequence 4096. Pressure inside the block-recurrent model after collation,
  same as E17c.
- **Compute:** Polonez, 4× RTX 3090, effective batch 72, **300M non-padding
  tokens** as the mechanism verdict. A later 1B run is a separate cosine and
  launches only if this 300M gate passes. Do not continue E17c's cosine.
- **Steps / epochs:** exact 300M non-padding-token ceiling; warmup 500;
  `AUTO_INTERVALS=1` (~every 10% / ~30M). 100M kill is read at the nearest
  90–120M checkpoint.
- **Launch (after implementation):**
  ```bash
  SKIP_PRETOKENIZE=1 bash scripts/launch_e17d.sh
  ```
  Wrapper pins the E17d cell, then delegates to `launch_e10.sh`. Equivalent
  explicit env:
  ```bash
  EXPERIMENT_ID=E17d CONCEPT_IO_MODE=per_layer_banks \
  CONCEPT_READ_MODE=addressed_topk CONCEPT_WRITE_MODE=addressed_erase_add \
  TIE_CONCEPT_WRITER=false ADDRESS_WRITE_TOPK=4 ADDRESS_READ_TOPK=8 \
  ADDRESS_ALLOCATION=1 CONCEPT_STATE_INIT=zeros \
  MEMORY_CARRY_DROPOUT=0.5 MEMORY_PRESSURE_TOKENS=64 MEMORY_PRESSURE_WEIGHT=4.0 \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.1 WRITE_NUM_HEADS=4 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 MUON_MOMENTUM=0.95 \
  WEIGHT_DECAY=0.1 CONCEPT_MEMORY_LR= \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 \
  TARGET_TOKENS=300000000 WARMUP_STEPS=500 AUTO_INTERVALS=1 \
  SAVE_TOTAL_LIMIT=12 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code (reusable, when implemented — NOT now):**
  - `concept_write_mode="addressed_erase_add"` on `ConceptWriteHead` (or a
    sibling `ConceptAddressedWriter`): address embeddings, usage, top-k
    erase-add. Defaults keep `additive` / `gated_replace` bit-identical.
  - `concept_read_mode="addressed_topk"` on `ConceptReadBranch`: hybrid
    location/content scores, top-k, then weighted V. Legacy `backbone_qkv` /
    `dedicated` unchanged.
  - Config fields + launcher env: `ADDRESS_WRITE_TOPK`, `ADDRESS_READ_TOPK`,
    `ADDRESS_ALLOCATION`, `CONCEPT_STATE_INIT`.
  - W&B diagnostics: per-bank write entropy, occupancy, top-1 mass, usage
    histogram, RankMe-written vs RankMe-unwritten, unused-slot `max|Δz|`.
  - Tests for invariance, causality, legacy numerical equivalence, and
    gradient flow through the addressor.
  No new training script, collator, or architecture fork.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: —
- WandB: —
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: —

## References
- E17c mixed close: [spec](../done_failed/E17c_depth_private_working_memory.md) ·
  [report](../../2_Experiments_Registry/run_reports/e17c_depth_private_working_memory_20260815.md)
- E17b mid-init (dense writes die without pressure):
  [spec](../done_failed/E17b_per_layer_mid_write_init.md)
- Addressable-memory review:
  [addressable_memory.md](../../literature_review/addressable_memory.md)
- Dense writable contrast:
  [recurrent_memory_transformers.md](../../literature_review/recurrent_memory_transformers.md)
- NTM https://arxiv.org/abs/1410.5401 · DNC
  https://www.nature.com/articles/nature20101 · SAM
  https://arxiv.org/abs/1610.09027 · Hopfield
  https://arxiv.org/abs/2008.02217 · product keys
  https://arxiv.org/abs/1907.05242 · GSA (dense-slot caution)
  https://arxiv.org/abs/2409.07146
- E11 (different graft, still design-only):
  [E11_memtoken_concept_memory.md](E11_memtoken_concept_memory.md)

# E17d — Depth-private concept layers as global-attention replacement

- **Status:** active (implemented; 300M Polonez running)
- **Serves:** Priority 1 / SG1–SG2: make Gemma's four former global layers assimilate
  long-range context through concepts, at every position, the way full attention used to.
- **Implementation plan:** [E17d_global_concept_assimilation_plan.md](E17d_global_concept_assimilation_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-17 · closed —

> E17d is one coherent bet: **the four global concept layers keep their depth-private
> banks, but they stop being optional memory sidecars and start doing the job Gemma's
> global attention layers did — mix current tokens with a compressed view of the rest
> of the document.** The previous "collapse to one bank" recommendation is withdrawn.
> E17c did not falsify depth abstraction; it never gave later banks a global-attention
> job.

## Hypothesis
If each of Gemma's four former global layers (5 / 11 / 17 / 23) mixes current tokens
with its own concept bank **inside the attention residual** (same place full attention
used to mix the full sequence), and previous blocks exist **only** as those banks at
both train and generate, then by 300M tokens permuting the banks will raise CE in the
**late** part of a block (tokens 256–512), not only the first 64, and at least **3 of 4**
banks will show their own positive permutation Δ — because each depth must assimilate a
different compression of history, the way stacked global-attention layers mix different
representations of the same long context.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` `BackboneConceptLM` with
  `concept_io_mode="per_layer_banks"`; `GlobalLayerWithConceptRead`, `ConceptReadBranch`,
  `ConceptWriteHead`; shared `training/train_concept_pretraining.py` →
  `scripts/launch_e10.sh`. E17c remains the default checkpoint contract for the cell
  knobs this experiment does *not* change.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42,
  C=128, K=512, four banks, seq 4096. Do **not** warm-start E17c: the read sits in a
  different residual and the token carry is gone; a resume would confound the verdict.
- **Baseline to beat:** E17c
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241`
  (`checkpoint-2370`). Carryless first-64 Δperm **0.594** (almost all bank 0 / 0.38);
  bins 256–512 Δperm **0.026**; with-carry every bin ~0.013; RankMe **6.75** (bank 1
  **1.84**); free-run `real` greedy @256 **0.23/0.53**.
- **Materially new vs E17c:** (1) concept cross-attn is the **global-attention
  sublayer**, not a tanh-gated add after the whole Gemma layer including the FFN;
  (2) previous-block tokens are **never** in the token stream (train and `generate()`);
  (3) no first-64 ×4 pressure (that loss paid for a block-start gist); (4) remember-biased
  **additive** writes so a page cannot overwrite 80% of a bank. Four banks stay.

## The architectural bet

Gemma-3 interleaves five sliding-window layers with one global layer. The global layer's
job is not "store a notebook." It is: **every token at this depth mixes with the rest of
the sequence, then the FFN sees that mix.** Low global layers mix more local/lexical
structure; high ones mix more abstract structure. That is the friend's claim, and it is
also why vanilla transformers have more than one full-attention layer.

E17c kept four banks but gave them a different job:

```text
E17c (memory sidecar, after the layer has already finished):
  h' = WindowedGemmaLayer(h)              # token attn + FFN, window 512, plus 512-token carry
  r  = DedicatedCrossAttn(Q=h', KV=z_g)   # query is post-FFN
  h  = h' + tanh(g) * r                   # skippable peek
  z_g(next) = (1-u)*z_g + u*BiXT(...)     # u ≈ 0.8 replace
  # train: hide the 512-token carry with p=0.5, and ×4 the first 64 tokens
  # generate: always keep the carry

E17d (global-attention replacement, inside the layer):
  # at each former global layer g ∈ {5,11,17,23}
  x      = InputLN(h)
  local  = WindowedTokenAttn(x)           # still local, like Gemma's hybrid
  global = DedicatedCrossAttn(Q=x, KV=z_g)  # THIS is the missing full-attention mix
  h      = h + local + global             # FFN then runs on the assimilated stream
  z_g(next) = z_g + tanh(α_g) * BiXT(z_g, h_block)   # additive, remember-biased
  # train and generate: no 512-token carry. Previous blocks exist only as z_*
  # uniform CE: no first-64 upweight
```

**When banks update (same schedule as E17c; E17d does not move the write).**
A 4096-token document is eight windows of K=512. Each of the four banks is a notebook
for one global layer. Updates are **after each window**, not at the end of the document.

Inside window `b` (tokens `b·512 … (b+1)·512`):

1. Layer 5 **reads** bank 5 as it was after window `b-1` (empty on window 0).
2. It mixes that into the current tokens, then the FFN runs.
3. It **writes** bank 5 from this window's layer-5 hidden states. That new notebook is
   **not** used again until window `b+1`.
4. Local layers 6–10 never read a bank. They see the *hidden stream* that layer 5 already
   mixed — that is assimilation for the next layer.
5. Layer 11 **reads its own** bank 11 from window `b-1`. It does **not** read the bank-5
   write that just happened. Then it writes bank 11 for window `b+1`.
6. Same pattern at 17 and 23.

So: four notebooks, each updated once per 512-token window, after its layer finishes
that window. Next layer in the *same* window uses the mixed hidden state, not the other
layer's new notebook. Next *window* is when each layer rereads its own updated notebook.

**Why later banks can still have a job.** After layer 5 assimilates `z_5`, local layers
6–10 process that mix (this is wanted: it *is* assimilation). Layer 11 then mixes
*higher-level current tokens* with `z_11`, which was written from layer-11 hidden states
of previous windows. That is the same reason Gemma's second global layer is not redundant
with the first: it attends to a transformed sequence, not to a gist already dumped into
the residual. E17c's bank 0 monopoly happened because the sidecar wrote a collapsed
topic summary into `h` after layer 5, and banks 1–3 were asked to store the same summary.

**Why the token carry must go.** With a 512-token carry, every "global" layer already
sees the previous page as raw tokens. Concept mixing then competes with a better cheat
sheet. E17c measured that: with carry present, Δperm ≈ 0.01 at every offset. Removing
the carry is not a second experiment; it is what makes concepts the global path.

**Why we do not collapse to one bank.** Block-Recurrent Transformers found that one
*identical* recurrent cell beats a stack of copies. Gemma's global layers are not that:
they are interleaved with local processing and sit at different depths. E17c never tested
that hypothesis fairly. Deleting banks 1–3 would abandon it.

**Out of scope:** MQAR / synthetic needles (E14/E15 protocol; follow-up *if* this still
only learns a topic gist on FinePDFs CE); collapsing to one bank; shrinking the sliding
window below K=512; VICReg (insurance if RankMe dies, not the bet); E08 latent
reasoning; 1B until the 300M gate passes; gate-init A/Bs.

## Why this is not a safe retread
E17c already had four dedicated banks, carry dropout, and a gated cell. It failed as a
**sidecar memory** trained to patch the first sentence of a new page. E17d changes the
*role* of those layers to the inductive bias Gemma already uses for long range: per-depth
global mixing. Analogy: cortex at several depths, each mixing current sensory stream with
its own compressed context, versus a single sticky note on the first layer.

## Success criteria (set BEFORE running)
Evaluate at 100M (kill), 300M (mechanism verdict), 1B only if 300M passes.
Same immutable held-out split as E17c. Default forward is already carryless, so
"carryless" and "normal" are the same path; still log a with-carry ablation for
comparison.

- **Primary (300M):** intra-block tokens **256–512**, `CE(permuted all banks) − CE(real)
  ≥ 0.10` nats; 95% CI lower bound **> 0.05**. E17c was **0.026** here. This is the
  "assimilation throughout the page, not only the first sentence" number.
- **Depth abstraction (300M):** single-bank permutation has a positive 95% CI for at
  least **3/4** banks on that same late bin. Report all four. E17c was 0.38 / 0.03 /
  0.01 / 0.03 on the *first-64* task and ~0 on the late bin.
- **Geometry (300M):** within-sample RankMe **≥ 19.2/128 for every bank** (floor, not
  the old 38.4 target). Collapse to a 7-direction gist cannot be depth-specific.
- **Generation (300M, matched no-carry decode):** `real` greedy @256 distinct-1 **≥ 0.20**
  and REP-3 **≤ 0.60**, and `real` beats `shuffle` on at least one of those. E17c was
  `real ≈ shuffle`.
- **No broad LM crash:** held-out eval loss **≤ 2.50** (E17c 2.276; some rise is
  expected without the token carry).

## Kill criteria (set BEFORE running)
- **Before GPU training:** unit tests for (a) no future leakage, (b) concept mix sits in
  the attention residual *before* FFN, (c) `generate()` uses the same no-carry policy as
  train, (d) E17b/E17c legacy configs still load.
- **Any checkpoint:** non-finite loss/grads, three consecutive eval-loss increases, or
  any bank RankMe **< 10**.
- **100M:** stop if late-bin (256–512) all-bank Δperm **< 0.03** (E17c replay) **or**
  only bank 0 has a positive Δ. That would mean we still have a layer-5 gist, just
  mixed a bit earlier in the layer.
- **300M:** do not launch 1B if the primary late-bin gate misses 0.10 or fewer than 3
  banks participate.

## Plan
- **Data:** immutable `e16b_long_4k_v1` Gemma-tokenized mix; causal LM; seq 4096.
  Pressure/upweight is **off**.
- **Compute:** Polonez, 4× RTX 3090, **300M** non-padding tokens. Microbatch is
  `PER_DEVICE_BATCH_SIZE=8` × accum `2` (effective batch **64**) after a 2026-08-17
  `length_group` sweep ranked by real tokens/sec (bs8 9089 tok/s beat bs3 8771 and
  bs10 8981). Token budget stays E17c's 300M; do not resume the aborted bs=3 run.
  Same 100M / 300M cadence as E17c. Not 1B.
- **Launch (after approval + implementation):** wrapper `scripts/launch_e17d.sh` pinning
  the knobs below, then `launch_e10.sh`.
  ```bash
  EXPERIMENT_ID=E17d CONCEPT_IO_MODE=per_layer_banks \
  CONCEPT_READ_MODE=dedicated CONCEPT_READ_PLACEMENT=attn_residual \
  TIE_CONCEPT_WRITER=false CONCEPT_WRITE_MODE=additive WRITE_GATE_INIT=0.1 \
  MEMORY_CARRY_DROPOUT=1.0 INFERENCE_CARRY_POLICY=drop_after_first \
  MEMORY_PRESSURE_TOKENS=0 MEMORY_PRESSURE_WEIGHT=1.0 \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.1 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 \
  TARGET_TOKENS=300000000 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
  `generate()` and Trainer eval honor `inference_carry_policy="drop_after_first"`
  (implemented; no longer hard-coded to `"normal"`).
- **New foundation code:** config-selectable read placement (`attn_residual` vs today's
  `post_layer`); generate-time carry policy; additive writes already exist. No new
  model class or training fork. Defaults keep E17c loadable.

## Result
<Filled in AFTER, by experiment-track. Launch identity only for now.>
- Aborted: `…20260817_124945` (checkpoint replay) and `…20260817_125416` (bs=3 underfilled).
- Run id: pending relaunch at bs=8 accum=2, TARGET_TOKENS=300000000
- Run report: pending
- Verdict: pending

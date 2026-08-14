# E17c — Depth-private gated working memory under causal carry pressure

- **Status:** implemented and smoke-verified; full training not launched
- **Serves:** Priority 1 / SG1–SG2: make Gemma's concept banks carry content-bearing
  cross-block working state that remains useful for generation.
- **Implementation plan:** [E17c_depth_private_working_memory_plan.md](E17c_depth_private_working_memory_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-14

> E17c is one coherent bet about what a usable recurrent working memory requires:
> a depth-private state transition in its own representation space, trained where the
> local token path cannot always bypass it. It is intentionally more than the missing
> init-0.3 cell; E17b showed that gate priors do not make writes persist.

## Hypothesis
If E17's four private concept banks become **depth-private selective memory cells**—each
with dedicated read projections, an untied BiXT writer, and content-dependent gated
replacement—and training causally removes the explicit previous-token carry on 50% of
post-first blocks, then by 300M tokens the real banks will beat batch-permuted banks by
**≥0.20 nats** on the first 64 carryless target tokens and by 1B will reach normal-context
`Δpermutation_beyond ≥0.05`, because useful distant context must now cross a stable,
depth-appropriate concept state rather than compete with a local CE shortcut.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` `BackboneConceptLM` with
  `concept_io_mode="per_layer_banks"`; `ConceptReadBranch`, `ConceptWriteHead`, and
  `BiXTCrossAttention`; the shared `training/train_concept_pretraining.py` →
  `scripts/train_concept_pretraining_multigpu.sh` → `scripts/launch_e10.sh` path.
  E17b remains the default implementation and checkpoint contract.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42,
  C=128, K=512, four banks, seq 4096. Do **not** warm-start E17b: E17c has new
  depth-specific projection spaces and must receive an unambiguous training verdict.
- **Baseline to beat:** E17b
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260810_135711`
  (`checkpoint-17780`): `Δshuffle_beyond=0.0055`,
  `Δstatic_beyond=0.0033`, writes closed to ≤0.049, and free-run `real` greedy
  distinct-1/REP-3@256 = **0.196/0.601**. E17 low-init is the secondary free-run
  baseline (**0.208/0.593**).
- **Materially new:** E17/E17b privatized only the **state tensor**. E17c privatizes
  the entire recurrent cell per depth, replaces unbounded additive writes with
  content-dependent retain/replace dynamics, gives concept reads projections separate
  from Gemma token self-attention, and trains with a causal local-bypass intervention.
  This is not another gate-init or optimizer retune.

## The architectural bet

For each global layer `g ∈ {5,11,17,23}` and sequence block `b`:

```text
E17b:
  h'              = GemmaLayer_g(h)
  h               = h' + tanh(read_scalar_g) · GemmaQKVO(h_pre, z_g)
  z_g(next block) = z_g + tanh(write_scalar_g) · SharedBiXT(z_g, h_current)

E17c:
  h'              = GemmaLayer_g(h)
  read_g           = DedicatedCrossAttention_g(query=Norm(h'), kv=Norm(z_g))
  h               = h' + tanh(read_scalar_g) · read_g
  candidate_g      = Norm(UntiedBiXT_g(Norm(z_g), Norm(h_current)))
  update_g         = sigmoid(Gate_g[Norm(z_g), candidate_g])       # per slot
  z_g(next block)  = (1-update_g) · z_g + update_g · candidate_g
```

The update-gate projection starts with zero weights and bias `logit(0.25)`: the writer is
live at initialization but can learn slot-specific retention. There is no global write
scalar that can close the whole path. Reads and writers are independent across depths;
the four banks still have C=128 each. Every bank writes once per 512-token block, after
its global layer, and the same bank is not read again until the next block.

**Strict block causality:** bank `z_g^b` used to predict block `b` contains only blocks
`<b`. The current block updates `z_g^(b+1)` after its read, so no current/future token
can affect an earlier prediction. E16/E16b's shared-depth same-block reread is absent.

**Causal memory pressure:** for each example and block `b>0`, with probability 0.5,
replace the explicit K-token carry with masked padding plus one BOS sentinel while
preserving the clean concept state written from earlier blocks. Teacher forcing remains
causal within the current block. CE on the first 64 target tokens of a pressured block
gets weight 4; all other valid tokens retain weight 1. Thus the model still learns normal
LM fluency, but a material share of the loss can recover prior context only through
concept memory.

This is analogous to a bank of depth-specific leaky integrators: the retention gate
controls memory timescale, the candidate carries new evidence, and context dropout
acts like training a biological working-memory circuit while intermittently removing
the sensory trace.

**Out of scope:** changing C/K/backbone/LoRA/optimizer/data mix; concept self-attention
reasoning; E08-style iterative latent refinement; synthetic recall curricula; a second
decoder; changing generation sampling. If E17c works, dedicated-read, untied-writer,
cell-dynamics, and pressure ablations become follow-up experiments rather than being
split into timid preconditions.

## Why this is not a safe retread
E17c attacks the observed failure as a coupled dynamical-system problem, not as another
scalar-init A/B. E17b briefly opened its additive write valves and then learned to close
them because local causal CE rarely rewarded durable state. E17c changes the state
transition, projection topology, and information constraint under one claim: **working
memory learns when it has a stable private cell and is sometimes the only causal path
for prior-block information.**

## Success criteria (set BEFORE running)
Evaluate at 100M (kill gate), 300M (mechanism verdict), and 1B (final quality verdict)
on the immutable held-out split.

- **Primary mechanism (decisive at 300M and retained at 1B):** on blocks 2–7 with the
  explicit carry removed, first-64-token
  `CE(batch-permuted all banks) - CE(real banks) ≥ 0.20 nats`; 95% bootstrap CI lower
  bound must exceed **0.10**. This is the single-number E17c verdict.
- **Normal-context content use (1B):** aggregate `Δpermutation_beyond ≥0.05` and
  `Δstatic_beyond ≥0.05` at positions ≥1024 (about 10× E17b), plus
  `Δone_block_beyond ≥0.02` to show accumulation beyond only the previous block.
- **Depth participation (1B):** single-bank permutation has a positive 95% CI for at
  least **3/4** banks; report every bank rather than only `encode_concepts()`' last bank.
- **Healthy dynamics:** each bank's mean content-dependent update gate lies in
  `[0.05, 0.80]`; no bank is globally frozen or replaced wholesale. Report update/state
  RMS and read gates by depth.
- **Geometry:** within-sample RankMe **≥38.4/128 for every bank**; report min/median/max.
- **Generation utility (1B):** matched continuation `real` greedy @256 reaches
  distinct-1 **≥0.25** and REP-3 **≤0.50**, with `real` no worse than `zero` on either
  metric. This improves materially over E17b without demanding that greedy decoding
  outperform sampled base Gemma.
- **No broad LM regression:** normal held-out eval loss **≤2.36** (E17b 2.264 +0.10).

## Kill criteria (set BEFORE running)
- **Before GPU training:** do not launch unless unit tests prove no intra-block or
  cross-block future leakage, E17b legacy-mode numerical equivalence, checkpoint
  round-trip, and nonzero gradients through all four dedicated readers/writers.
- **Any checkpoint:** stop for non-finite loss/gradients, three consecutive eval-loss
  increases, or any bank RankMe `<19.2/128`.
- **100M:** stop if carryless first-64 `Δpermutation <0.05` **or** all four observed mean
  update gates are `<0.02`. Unlike E17b, the pressure objective is designed to produce
  an early signal; a null result here falsifies the mechanism rather than just its scale.
- **300M:** stop rather than spend the remaining budget if the primary carryless gate
  is `<0.20`, normal-context `Δpermutation_beyond <0.02`, or `real` free-run REP-3
  exceeds **0.80** while being worse than `zero`.

## Plan
- **Data:** immutable `e16b_long_4k_v1` Gemma-tokenized manifest; raw causal LM,
  max sequence 4096. `DataCollatorForCausalLM` unchanged; pressure is applied inside
  the block-recurrent model after collation.
- **Compute:** Polonez, 4× RTX 3090. Run a 50-step VRAM/throughput calibration; full
  1B is expected to exceed E17b's 280 GPU-h because four readers and writers are
  independent. The 100M/300M gates cap a negative result.
- **Steps / epochs:** exact 1B non-padding-token ceiling; warmup 500; reports at
  100M, 300M, and 1B. Batch size is selected by calibration and logged; token budget,
  not optimizer-step count, is matched.
- **Launch:**
  ```bash
  EXPERIMENT_ID=E17c CONCEPT_IO_MODE=per_layer_banks \
  CONCEPT_READ_MODE=dedicated TIE_CONCEPT_WRITER=false \
  CONCEPT_WRITE_MODE=gated_replace WRITE_UPDATE_GATE_INIT=0.25 \
  MEMORY_CARRY_DROPOUT=0.5 MEMORY_PRESSURE_TOKENS=64 MEMORY_PRESSURE_WEIGHT=4.0 \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.1 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 MUON_MOMENTUM=0.95 \
  WEIGHT_DECAY=0.1 CONCEPT_MEMORY_LR= \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 \
  TARGET_TOKENS=1000000000 WARMUP_STEPS=500 AUTO_INTERVALS=1 \
  SAVE_TOTAL_LIMIT=12 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** config-selectable extensions to `ConceptReadBranch`,
  `ConceptWriteHead`, and the existing per-layer block loop; a `ModuleList` of the
  existing writer class when weights are untied; pressure-aware CE and per-bank
  diagnostics. No new model class, training entrypoint, collator, or architecture fork.
  All defaults reproduce E17b.

## Result
To be filled by `experiment-track`.
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

## Implementation verification (2026-08-14; not an experiment verdict)
- **Code:** `a30f0f5` on the E17c implementation branch. Defaults retain E17b's
  `backbone_qkv` + tied additive writer path and state-dict keys.
- **Local:** full suite `381 passed, 9 skipped`; a three-step H=64/C=4/two-bank smoke
  produced finite losses `5.581 → 5.542 → 5.507`, all-bank/permutation diagnostics,
  checkpoint round-trip, and generation.
- **Polonez GPU guard:** the first attempt exposed a BF16-candidate/FP32-state interpolation
  mismatch before step 1. Commit `809661b` fixes it and adds an autocast regression; a
  one-step 4×3090 guard then completed and saved normally.
- **50-step matched smoke:** run
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_084543`
  (W&B, project configured by the training environment)
  finished at effective batch 72 with 51.41M trainable parameters. W&B API comparison
  confirms the same immutable `e16b_long_4k_v1` manifest, Gemma tokenizer, seq=4096,
  causal-LM objective, C=128/K=512, LoRA r=16/alpha=32, seed 42, Muon
  `0.01 / 2e-4 / 0.95 / wd=0.1`, job type, and standard eval/ablation keys as
  E16b/E17/E17b. The smoke additionally logged carryless, permutation, per-bank, and
  update/state metrics.
- **Telemetry guard:** eval initially overwrote the last sampled pressure fraction with
  zero. Commit `a30f0f5` preserves training telemetry; follow-up run
  `..._20260814_091224` logged `memory_pressure/observed_fraction=0.4167`.
- **Smoke observations only:** normal eval loss was 2.822 on two held-out samples and
  all metrics were finite. Dynamic update gates after 50 steps were
  `0.0022 / 0.0083 / 0.0439 / 0.0041`; this rapid closing is a risk signal, not a
  kill verdict at this tiny budget. Use the preregistered 100M gate and ≥24 held-out
  batches for the scientific decision.
- **Pending full-run efficiency path:** the E17c launcher now enables cached bounded
  length grouping over the same immutable 4K manifest. No rows, source weights, document
  boundaries, labels, or token budget change. Padding and real-token throughput are logged
  globally; a 4-GPU Polonez comparison remains required before claiming a speedup.

## References
- [E17b failed mid-init run](../done_failed/E17b_per_layer_mid_write_init.md) ·
  [report](../../2_Experiments_Registry/run_reports/e17b_per_layer_mid_write_init_20260813.md)
- [E17 per-layer baseline](../done_success/E17_four_bank_concept_memory.md) ·
  [report](../../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md)
- [E16b shared init-0.3 control](../../2_Experiments_Registry/run_reports/e16b_shared_init030_1b_20260810.md)
- [Recurrent memory review](../../literature_review/recurrent_memory_transformers.md)
- [Canonical evaluation protocol](../../3_Evaluations_and_Baselines/evaluation_protocol.md)

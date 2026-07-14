# E13 — Layer-wise recurrent KV-memory concepts

- **Status:** draft (design-only — gated behind E12; no implementation plan)
- **Serves:** the pretrained-backbone fixed-memory platform bet. E13 tests whether a
  per-layer compressed memory can better replace the layer-specific KV histories that a normal
  decoder would maintain during long-context attention.
- **Implementation plan:** *(not yet written — author only if E12's trigger is met)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-12 · closed —

> **One experiment = one hypothesis = one changed variable vs E12.** E12 establishes native
> dynamic KV-prefix reading at all 26 layers while retaining one shared recurrent state and one
> write. E13 changes only the **memory-state topology**: one state/write becomes one independent
> state/update path per layer. It inherits E12's attention-prefix interface, data, objective,
> backbone, LoRA policy, sequence length, and budget.
>
> **Activation trigger:** run E13 only after E12 demonstrates that all-layer native KV-prefix
> reading is live (`Δshuffle ≥ 0.01 nats` and `static − real ≥ 0.01 nats` at positions ≥1024
> in its 50M pilot), but its recovery remains below the E10 primary target. If E12's memory is
> unused, adding 26 memories is premature; repair the shared interface first.

## Hypothesis
If every Gemma layer ℓ has its own recurrent concept state `zℓ,b [B,128,1152]`, read as that
layer's native dynamic K/V prefix and updated after each block from that same layer's
current-block hidden states, then positions ≥1024 will improve by at least 0.05 nats over E12
and retain a positive real-vs-shuffled-memory margin — because each depth can preserve and retrieve
long-range information in its own representation space rather than forcing all 26 layers to share
one 1152-dimensional memory representation.

## Builds-on

- **Foundation:** E12's reusable `concept_io_mode="kv_prefix"` path in
  `nn/backbone_concept_lm.py`; the shared
  `training/train_concept_pretraining.py` → `scripts/train_concept_pretraining_multigpu.sh` →
  `scripts/launch_e10.sh` route; Gemma's mask-dict attention interface; causal-LM collator and
  recurrent-state ablation/evaluation contract.
- **Init / checkpoint:** fresh `google/gemma-3-1b-pt`, frozen base weights + the same E12 LoRA
  configuration, fresh layer-wise concept states, seed 42. This is not a resume from E12 because
  the state topology and checkpoint contract differ.
- **Baseline to beat:** the matched E12 checkpoint at the same token exposure and held-out
  documents. The nearest existing evidence is E10's 100M concept pilot
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847`, whose recurrent margins were
  null (`Δshuffle=-0.00038`, `static−real=-0.00032`); E12 must establish a non-null interface
  before E13 is justified.

## The single change

Replace E12's shared state/update:

```text
one z_b [B,128,H]
one recurrent write after the entire 26-layer block
```

with layer-indexed state and writes:

```text
for layer ℓ = 1..26:
    Qℓ = qℓ(local token states)
    K,V = concat(local-token K,V, Pkℓ(zℓ,b), Pvℓ(zℓ,b))
    token statesℓ = attention(Qℓ, K, V)

after each block:
    zℓ,b+1 = zℓ,b + tanh(αℓ) · RMSNorm(Write(zℓ,b, hℓ,block))
```

Each layer's 128 concepts are visible only to that same layer's attention. Each write consumes
only that layer's final hidden states for the **current** block, not carry-token states. The
concept count remains 128 per layer; total recurrent-state capacity is therefore
`26 × 128 × 1152` activations per sequence.

**Parameter discipline:** reuse one weight-tied `ConceptWriteHead` across layers, with
layer-specific recurrent states, write gates `αℓ`, and E12's already layer-specific K/V prefix
projections. Tying the write transform prevents a second confound—26× new writer parameters—while
still testing whether memories must be separated by depth. An untied-writer capacity expansion is
out of scope for E13.

## Success criteria (set BEFORE running)

- **Primary:** at the matched 50M-token checkpoint and positions ≥1024, E13 real-state CE is at
  least **0.05 nats lower** than E12 real-state CE on the same held-out documents; paired-bootstrap
  95% CI excludes zero.
- **Content attribution:** `Δshuffle = shuffled CE − real CE ≥ 0.01 nats` and
  `static CE − real CE ≥ 0.01 nats` at the same checkpoint/region.
- **Layer utilization:** at least **13 of 26** layer write gates have `|tanh(αℓ)| ≥ 0.005`, and
  at least **13 of 26** prefix-read gates have magnitude ≥0.005 by 50M tokens. This prevents
  declaring success from only the E10-style four-layer pathway.
- **No local regression:** positions <512 are no more than +0.02 nats above E12 at matched exposure.
- **Geometry guard:** the final-layer state has within-sample RankMe ≥38.4/128; report the same
  metric for all layer states, summarized by minimum and median.

## Kill criteria (set BEFORE running)

- At the ~25M-token checkpoint, stop if both primary recurrence signals
  (`static−real` and `Δshuffle`) are ≤0.002 nats at positions ≥1024 **and** fewer than four
  layer read gates have magnitude ≥0.005.
- Stop if fewer than four layer write gates have magnitude ≥0.005 at 25M tokens: the layer-wise
  memories are not being written.
- Stop on non-finite loss/gradients, eval CE rising for three consecutive evaluations, or any
  final-layer within-sample RankMe <19.2/128.
- Do not run if E12 does not meet its activation trigger; that is a decision gate, not a failure
  of E13.

## Plan

- **Data:** inherit E12's Gemma-tokenized `smollm3_inspired_2k_e05` manifest, raw causal text,
  sequence length 2048, and the frozen train-disjoint 2K/8K evaluation documents.
- **Compute:** Odra 3× RTX 3090, subject to a memory calibration before the pilot. Layer-wise
  states require retaining 26 recurrent trajectories through four blocks, so activation memory and
  speed must be measured before fixing batch/accumulation. Preserve effective batch 72 by lowering
  microbatch or increasing accumulation only if calibration requires it.
- **Steps / epochs:** 50M non-padding Gemma-token pilot with the ~25M kill checkpoint; compare
  exactly against E12 at matched token exposure. Extend only after this pilot meets all criteria.
- **Launch:** intended reusable configuration:
  `EXPERIMENT_ID=E13 CONCEPT_IO_MODE=layerwise_kv_memory TARGET_TOKENS=50000000 WARMUP_STEPS=50 SKIP_PRETOKENIZE=1 bash scripts/launch_e10.sh`
  Exact flag names and calibrated batch settings are deferred to the implementation plan.
- **New foundation code (if triggered):** a config-selectable
  `concept_io_mode="layerwise_kv_memory"` on the shared backbone family: per-layer state holder,
  per-layer prefix K/V read, tied write-head application after every layer's block hidden states,
  per-layer gates, checkpoint serialization, and per-layer ablation/gate telemetry. No new training
  script or experiment-specific model fork.

## Explicitly not E13

- Four independent memory banks only at Gemma's four former-global layers (6/12/18/24): a cheaper
  intermediary topology recorded in E12's deferred design note. It needs its own spec because it
  changes number of memory banks and injection depth together.
- Untying 26 write heads.
- Changing C, sequence length, training data, objective, LoRA targets/LR, gate initialization, or
  optimizer.

## Result

<Filled in AFTER, by experiment-track.>
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

## References

- E10 — [pretrained-backbone concept memory](../done_failed/E10_gemma_backbone_concept_memory.md).
- E12 — [per-layer dynamic KV-prefix read](E12_perlayer_kv_prefix_concepts.md).
- Transformer KV caches: each transformer layer naturally maintains its own token K/V history;
  E13 tests a fixed-size recurrent approximation to that structure.

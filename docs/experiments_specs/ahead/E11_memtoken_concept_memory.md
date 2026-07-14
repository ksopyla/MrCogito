# E11 — In-sequence memory-token concept read/write (Design A; backbone-agnostic)

- **Status:** ahead / design-only — E10's global→concept interface is closed; E11 remains
  an unimplemented candidate interface and is not currently scheduled
- **Serves:** same platform bet as [E10](../done_failed/E10_gemma_backbone_concept_memory.md) (pretrained-backbone
  concept memory). E11 is the **read/write-mechanism variant**: instead of injecting concepts into
  the global layers' KV (E10 / Design C), concepts live **in the decoder's own token sequence** as
  soft memory tokens (RMT / ICAE / Gist lineage) — the simplest, most published-precedent graft, and
  the only one of the three designs that is **backbone-agnostic** (works on Qwen3/Llama, which have
  no local/global layer split).
- **Implementation plan:** *(not yet written — authored after E10's verdict)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-08 · closed —

> One experiment = one changed variable **vs E10**: the concept read/write mechanism. Backbone,
> data, masks (sliding-window tokens), block size, LoRA config, budget, and the E10 metric protocol
> (Stage-0 gap G, beyond-local CE, extrapolation at 8K, Δshuffle, RankMe guard) are all inherited.

## Hypothesis
If the C concept slots are prepended to each block's sequence as soft tokens (read: every layer's
ordinary self-attention sees them — not just 4 global layers) and re-written via R write-slot tokens
appended after the block (write: the backbone itself computes the update; outputs of the write slots
become the next state, gated + zero-init), then beyond-local CE recovery will **match or beat E10's**
— because concepts processed *through* all 26 layers are a richer read path than KV injection at 4,
at the cost of C+R extra positions per block.

## The single change (vs E10)
Concept I/O mechanism: `global-layer KV read + BiXT write` → `in-sequence soft tokens (read at all
layers) + write-slot tokens (write by the backbone itself)`. Layout per block:
`[c_1..c_C | carry | block tokens | w_1..w_R]`; tokens attend concepts + sliding window; write slots
attend everything; `z(b+1) = z(b) + tanh(α)·RMSNorm(W_out·h[w])`, α zero-init.

## Builds-on
- **Foundation:** everything E10 builds (`nn/backbone_concept_lm.py` grows a second
  `concept_io_mode="mem_tokens"` next to E10's `"global_kv"` — config-selectable, not a fork).
- **Init / checkpoint:** same frozen backbone + fresh LoRA; no warm-start from E10 (mechanism A/B
  needs matched init).
- **Baseline to beat:** the **E10 concept arm** (same budget/data/metrics) + the shared
  `CONCEPT_NUM=0` control. E11 is justified if it beats E10's beyond-local CE recovery, or matches
  it on a backbone without a local/global split (portability value).

## Success criteria (set BEFORE running; thresholds finalized once E10's numbers exist)
- Beyond-local CE recovery ≥ E10's measured recovery (same eval protocol, seq 2048 + 8192).
- E10's must-not-regress guards inherited (short-range CE tax ≤ +0.02 nats; state RankMe ≥ 0.3·C).

## Kill criteria
- E10's kill gates inherited (gap-not-opened at 50% budget; RankMe collapse; divergence).
- If E10 was itself killed at Stage 0 (G too small at 2048), E11 inherits the re-scoped protocol.

## Plan (sketch — frozen only when activated)
- Data/compute/budget identical to E10 (one arm, ~2B tokens, Odra or Polonez).
- Launch: `CONCEPT_IO_MODE=mem_tokens bash scripts/launch_e10.sh` (shared launcher; E10's wrapper
  gains the mode knob).
- New foundation code: the `mem_tokens` path in `nn/backbone_concept_lm.py` (soft-token injection,
  write-slot readout, per-block mask); no new module.

## Result
<Filled in AFTER, by experiment-track.>

## References
- RMT (arXiv:2207.06881) + AAAI-2024 pretrained-backbone follow-up; ICAE (arXiv:2307.06945);
  AutoCompressor (arXiv:2305.14788) — the mem-token lineage.
- `docs/literature_review/recurrent_memory_transformers.md` §B.

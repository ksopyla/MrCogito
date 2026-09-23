# Exclusive-slot <10M working law — copy works, compression binds, composition does not

**Date:** 2026-09-15 (campaign 2026-09-13 → 2026-09-15)
**Machine:** CPU Cloud Agent (4 cores, no CUDA)
**Instrument:** `verification/symbolic_channel_probe.py` on `data/symbolic_tasks.py`
**Measurement note:** [concept_slot_scaling_frontier_20260913](../../4_Research_Notes/concept_slot_scaling_frontier_20260913.md)
**Inventory:** [exclusive_slot_law_inventory.json](../../4_Research_Notes/exclusive_slot_law_inventory.json)
**Plots:** [task comparisons](../../4_Research_Notes/figures/exclusive_slot_task_comparisons.png) · [working law](../../4_Research_Notes/figures/exclusive_slot_working_law.png) · [steps / sizes / acc](../../4_Research_Notes/figures/exclusive_slot_law_steps_sizes_acc.png)

Not an E-numbered language-model run. CPU symbolic campaign at frozen hidden=256 (5.11M concept model). Pass mark **95%** accuracy on supervised tokens. Chance **25%** (4-letter alphabet). Model was **not** shrunk.

---

## Goal

Measure what a <10M exclusive-scope concept-slot model can do versus a no-notebook leak check and a full-book dense transformer, at the same compute, data, and parameter band. Push difficulty (longer copy, more tokens per notebook page, multi-hop lookup) rather than hunting a tiny model or 100%.

Three students take the same exam:

- **concept model** — notebook of summaries; the reader may only look at pages that ended before the current paragraph (exclusive scope). 5.11M.
- **no-notebook control** — same short reading window, notebook taken away. If this student scores above chance, the exam leaks. 2.27M.
- **dense baseline** — allowed to reread any page. Width-matched 4 layers = 2.27M. Param-matched 9 layers (D9) = 4.97M.

## Verdict

**promising as memory, not as reasoning.** The notebook carries a 32-letter fact across 1024 tokens at 5M parameters. Taking the notebook away returns chance. Length is cheap. Packing more letters onto one page binds (16 tokens/page works; 32 misses). Chained lookups stay at chance for the notebook *and* for the dense student — an exam kill, not a concept-only failure.

## Closed grid (95% bar)

### Copy a 32-letter span the local window cannot see (`far_copy`, 8 tokens per notebook page unless noted)

| exam | concept 5.11M | dense | no-notebook |
|---|---|---|---|
| seq128 | 95% @ 51k (1.35M toy) | 99.7% @ 25k (0.60M) | 25% |
| seq256 | **95.0% @ 96k / 36 min** | 99.2% @ 72k (4-layer) | 25% |
| seq512 padded | **97.1% @ 72k / 76 min** | 4-layer **~30% miss**; 9-layer **98.5% @ 120k / 121 min** | 25% |
| seq512 true-reach (span ≥128 tokens back) | **95.0% @ 48k / 40 min** | 9-layer **98.3% @ 32k / 32 min** | 25% |
| seq1024 true-reach (span ≥256 tokens back, 128 pages) | **95.9% @ 96k / 5.5 h** | 9-layer **97.1% @ 136k / 15 h** | 26% (floor) |

On seq=1024 the concept model uses **0.71× examples and 0.37× wall** versus the 9-layer dense student. Removing the notebook from a trained concept model returns ~24–26%. The skill is in the pages.

Width-matched 4-layer dense **cannot** solve padded seq512. Param-matching is the fair gate. Efficiency ranking flips by exam: dense is cheaper on true-reach seq512; concepts are cheaper on padded seq512 and seq1024.

### Compression (same copy, seq256, more letters per page)

| tokens per page | concept acc | 95%? |
|---|---|---|
| 8 (span uses 4 pages) | 95.0% @ 96k | yes |
| 16 (span uses 2 pages) | 95.7% @ 224k (~2.3× data) | yes |
| 32 (span uses 1 page) | 87.9% @ 256k | **no** |

Dense ignores page size (full-book attention). The write into a single page is the bottleneck.

### Composition (scattered `a→b`, `b→c` lookups)

| exam | who | acc @ 128k | 95%? |
|---|---|---|---|
| 3 hops, seq256 | concept and 4-layer dense | 24% | no |
| 2 hops packed, seq512 | 9-layer dense | 26% | no |

The 9-layer dense student that scored 98.5% on copy stays at chance on packed 2-hop. Concept model was not run on that cell (exam too hard for dense first).

### Learning rate

Not portable. 3e-3 trains the 1.35M toy; 1e-3 trains seq256; 3e-4 trains seq512; 1e-4 trains seq1024. Killing a run at 800 steps that is still at chance is a false negative (dense seq512 and seq1024 both sat at chance then jumped).

## What a <10M model can handle

- **Length:** copy a 32-letter span through seq=1024 with the span 256 tokens back, at ~10^5 examples, if the learning rate drops with length.
- **Compression:** 16 tokens per notebook page, not 32, at this budget.
- **Composition:** nothing at this budget, including for param-matched dense.

## Pros / cons

Pros: exclusive scope is doing the job (no-notebook never leaves chance; ablation kills the concept model). Length scaling is gentle. Against a fair ~5M dense decoder, the notebook is competitive and often cheaper in wall at long sequences (local reader + 128 pages vs full 1024² attention). The instrument is small and falsifiable.

Cons: this is copy, not language and not reasoning. E22 already showed a live, diverse notebook that next-word prediction treats as a document embedding. One page cannot yet hold a 32-letter span. Efficiency vs dense depends on the exam. Training takeoff is late and sharp.

## Next (justified by this grid)

Do not shrink the model. Do not hunt 100%. Do not grow copy length further.

1. **Fix the write, then retry 32 tokens/page.** Mean-pool into one page is the miss. Kill if still under 90% well past 256k examples at 5.11M.
2. **Invent a composition exam dense can pass, then put the concept model on it.** Packed 2-hop teaches nothing until the 9-layer dense student hits 95%.
3. **Only then return to E23 (exclusive channel on language).** The notebook must compress harder than 16 tokens/page and do at least one compositional lookup dense can also do. Otherwise next-word prediction will again pay for a document embedding.

## Banked

Frozen 5.11M exclusive-slot reference; param-matched 9-layer dense gate (4-layer width-match is a false kill at seq512); `write_result_json` dual-write after the artifact store wiped seq1024 concept JSON; durable inventory and three comparison plots.

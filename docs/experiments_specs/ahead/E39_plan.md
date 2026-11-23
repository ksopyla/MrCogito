# E39 — Implementation Plan

- **Spec:** [E39_living_concepts.md](E39_living_concepts.md) · **Status:** built 2026-11-23 (written 2026-11-22
  while P2 ran; K0 tripped and the author overrode it on 2026-11-23, see the spec). Brought in line with the spec's
  later changes: no 30M screen, the `living_score` switch, one spot H100 first.
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's bet: the reading layer's memory regrouped into contiguous spans and document-wide
> concepts, searched in two levels, with one summary per concept for what was not opened. The spec's hypothesis,
> marks, recipe and budget are frozen; this file fixes kernels, starting values, the controllers' step and the
> file layout.

## 1. Source & fit
- **Origin:** the E38 150M results (concept memory ties the topic index on single needles and long text, loses
  when one answer needs several far items) and the design session's spec of 2026-11-22.
- **Architecture mapping:** the reading layer only (layer 3 of 13): how its memory is grouped (encoder side), how
  a reader chooses what to open and what stands for the rest (decoder read). No reasoner, no tree, no new weights.
- **Boldness check:** all three changes of the bet are built as specified: variable-length contiguous spans cut by
  one backward comparison, concepts formed online for the whole document with one causal version per stride, and
  the two-level choice by match with remaining-count summaries. Nothing is replaced by the existing k-means
  grouping or by a positional page. The spans-only arm is the attribution arm, not a fallback.

## 2. Reuse map (read: `nn/concept_memory.py`, `nn/exact_read.py`, `nn/lsa_buckets.py`, `nn/perceiver_ar_lm.py`
`Attention._concept_read` / `_run_layers` / `_band_block_mask` / `attend_list`, `nn/incremental.py`,
`scripts/run_e33n.py`, `infra/aws/jobs/e38_full.sh`, `e33n_run.sh`, `e37_evals_long.sh`, `training/flow_callback.py`,
`training/concept_pretraining_trainer.py`)

| Component | Action | Where |
|---|---|---|
| `GroupIndex`, `groups_from_spans`, `OpenTable`, `open_term`, `counts` | reuse as-is: a group is a span slot (128 per 512-token stride), member rank < 16 | `nn/exact_read.py` |
| `choose` | extend: `return_counts=True` also returns each chosen group's member count (for the tokens opened per concept) | `nn/exact_read.py` |
| `build_concepts` | extend: `labels=` takes a given labelling instead of the k-means (spans-only arm, and the equality test) | `nn/concept_memory.py` |
| `concept_open_table`, `attend_concepts`, `summary_logit`, `summary_log_scale` | reuse as-is for `living_levels 1`; the length scale also for level 2 | `nn/concept_memory.py` |
| `_attend_concepts_flex` | refactor without change of behaviour into `flex_raw_call` (window ‖ sinks ‖ opened, returns output and log-sum-exp) and `merge_calls` (the guarded log-sum-exp merge); the living read calls both | `nn/concept_memory.py` |
| `living_concepts` | **new**, reusable, config-selected: span cut, leader rule, statistics and versions, two-level choice, the flex summary call, the streamed reference read, the controllers | `nn/living_concepts.py` |
| `PerceiverARConfig`, `Attention`, `PerceiverARLM._run_layers` | extend: `par_global_read = "living"`, the `living_*` fields, `Attention._living_read`, the per-forward grouping reset, `pop_train_stats()` | `nn/perceiver_ar_lm.py` |
| `IncrementalDecoder` | extend: kind `living` (grouping rebuilt at stride ends, one query through the streamed read) | `nn/incremental.py` |
| `ModelArguments`, `_plain_loop_kwargs` | extend: pass the `living_*` fields | `training/concept_pretraining_args.py`, `training/concept_pretraining_factories.py` |
| `PerceiverDenoiseTrainer.log` | extend: merge the model's `pop_train_stats()` into the periodic log (the grouping's numbers, readout R4) | `training/concept_pretraining_trainer.py` |
| `ARMS`, `TINY_OPEN`, `model_args` | extend: arms `living`, `living_spans`; `concept_sum_len` default as for `concept` | `scripts/run_e33n.py` |
| `e38_full.sh` | extend: experiment tag and S3 folder from `E38_EXPERIMENT` (default E38), plain names for the two arms | `infra/aws/jobs/e38_full.sh` |
| `e39_board.sh` | new thin wrapper (E39 tag, `runs/e39/`, saves every 400 steps on an H100) | `infra/aws/jobs/` |
| `memory_mode`, `set_concept_sum_len` | extend: off / no-summaries / flat-search switches and the length scale for the living read | `evaluation/long_context_use.py`, `nn/concept_memory.py` |

## 3. Forward pass (tensor shapes)
Symbols for a 16k row: `B` rows, `S` = 16,384 tokens, `d` = 64, `h` = 14 query heads, one key/value head,
stride 512, `n_st` = (S − 1) // 512 − 1 = 30 memory strides (those some reader sees two strides later),
`K` = 128 span slots per stride, `G` = n_st · K = 3,840 span slots, `C` = S / 16 rounded up to a multiple of 128 =
1,024 concept slots. At 64k: n_st 126, G 16,128, C 4,096. At 4k: n_st 6, G 768, C 256.

**Once per forward, first loop pass, no gradient** (`build_living`, the reading layer's own keys after `k_norm`):
```
k [B,S,1,d] → k̂ = unit keys [B, n_st·512, d]
member [B, n_st·512]: not a sink, not padding, and its document still runs at the first token of stride s + 2
cut:   cos_t = k̂_t·k̂_{t−1};  start = first member of a run | cos_t < θ (the stride's 96 lowest-cosine cuts) | every 16th
       labels [B, n_st, 512] ∈ {−1, 0 … 127}                      # slot within the stride, position order
groups: groups_from_spans(labels, strides, per_span=128)          # GroupIndex: codes [1,B,S] int32, prefix int32
U [B, n_st, 128, d] = Σ k̂ per span;  û = normalise(U)
leader rule, strides in order (loop of n_st steps, batched over B):
       sim [B,128,C] = û · normalise(Σ_concept k̂) of the document's concepts after the stride before
       join if best ≥ λ, else found: 3 parallel leader rounds on [B,128,128]; slots allocated in order; a new
       document restarts at slot 0; when the slots are full a founder joins its nearest concept
       cof [B, G] concept slot of each span;  n_c [B, n_st] slots in use after each stride;  ver_doc [B, n_st]
controllers (training only): θ and λ move one small step from the counters summed over all cards (section 5)
```
**Every pass, with gradient to the member tokens' keys and values** (`living_stats`):
```
span:    n [B,G], Σk [B,G,d], Σ|k|² [B,G], Σv [B,G,d]              # scatter-adds by span slot
stride increments by concept [B, n_st, C, ·]  →  versions = M · increments, M [B, n_st, n_st] = 1 for an earlier or
         equal stride of the same document (an exact, differentiable cumulative sum that restarts with the document)
versions: n [B,n_st,C], μ_k [B,n_st,C,d], σ² [B,n_st,C], μ_v [B,n_st,C,d]   # 2.0M numbers each for μ at 16k
```
**Every pass, the choice, no gradient** (`living_open_table`; queries in chunks inside one stride τ, version τ − 2):
```
concepts: s_c = max_h( q·μ_c/√d + |q|²σ_c²/(2d) ) [B,ch,C] → the best 32 of the reader's document
spans:    s_g, the same score on the span's own statistics [B,ch,G_visible]; candidates = spans two strides back or
          more, of the reader's document, whose concept was kept (flat search: every such span)
choose(…, m = 256, cap = 512, return_counts) → OpenTable T [B,S,G] int16 (126 MB per row at 16k, 2.1 GB at 64k)
n_open [B,S,C] int16 = tokens opened per concept (scatter-add of min(count, 512 − places used) by cof)
```
**Every pass, the read** (`attend_living`), one softmax as two flex calls merged by their log-sum-exp:
```
raw call      sinks ‖ own stride + the one before ‖ opened tokens       # flex_raw_call, the concept read's kernel
summary call  keys μ_k, values μ_v of all versions, KV length n_st·C (30,720 at 16k; 516,096 at 64k);
              mask: version = reader's stride − 2, version's document = reader's, n − n_open > 0
              score + log(n − n_open) + |q|²σ²/(2d); block list built directly: each 128-query block lists the
              ⌈n_c/128⌉ key blocks of its version (≤ 8 at 16k, ≤ 32 at 64k), no dense mask evaluation
merge         log-sum-exp with the length scale (summary_log_scale, unchanged); padding readers read zero
```
`living_levels 1` (arm `living_spans`, as the spec): every span is its own concept (concept slot = span slot,
C = G), searched flat by the same match score, one summary per span for what was not opened; the same code path
as level 2 without the leader rule. `living_score mass` adds log n to the scores (the concept read's rule); with
level 1 the model then sends the span labels through `build_concepts(labels=…)`, `concept_open_table` and
`attend_concepts` unchanged, the regression test against the existing read (tests L6: equal also through the
living path when whole spans open). Found by that test: the existing concept read ranks other documents' concepts
too (they open nothing, `choose` counts the query's document only), so in packed rows it can spend choice slots
on them; the living read ranks the reader's own document only.

**The streamed read** (`living_read_stream`), the second implementation of the same read: per chunk of queries of
one stride it makes the choice and computes one fp32 softmax over window ‖ sinks ‖ the ≤ 512 opened tokens
(gathered) ‖ the C summaries of the version, with no [B,S,·] table. It is differentiable, so it serves as (1) the
reference path of the tests and the `sdpa` backend, (2) the long-length path (chosen by itself in evaluation when
T would pass flex's 2³¹-element buffer limit, i.e. from 128k, and by `open_impl("list")`), and (3) the cached
decoding's one-query read.

**Cached decoding:** `decode_kind` → `living`; the grouping is built from the cached first-pass keys after the
prefill and again when a stride closes with answer tokens in it (after the new position's document id is written:
the membership rule looks at the first token of stride s + 2); each step reads through `living_read_stream`.

**Gradient checkpointing:** `_run_layers` clears the reading layer's grouping at its start; the first reading-layer
call builds it; the recomputation in the backward finds it and neither rebuilds nor moves the thresholds.

## 4. Inputs & data
Unchanged: `m21mix_v2` through `scripts/run_e33n.py --mix`, the same shards, order and rows as the other four
150M models. The home pipeline check and P4 use the small `m21_check` build of E38's pipeline check.

## 5. Loss, objective and the controllers
- Next-token loss, unchanged. No auxiliary loss.
- **Controllers** (buffers `living_theta`, `living_lam` on the reading layer's attention; saved with the model;
  frozen in evaluation). After the grouping of each training forward, with counters all-reduced over the cards:
  `θ ← θ − η · clip((spans/tokens − 1/8) · 8, ±1)`, `λ ← λ − η · clip((founders/spans − 1/8) · 8, ±1)`, η = 1e-3,
  clamped to [−1, 1]; a forward with fewer than 256 member tokens (1 span) moves nothing. From P1 the cut rate
  changes by about 1.4 per unit of θ around the target, so a relative error decays with a time constant of about
  90 forwards; λ is of the same order (P3: founders 8 % → 37 % of spans from λ 0.6 to 0.8 at 16k).
- **Starting values** (P1, P3 on the released concept-memory model, 2026-11-22): θ = 0.49, λ = 0.65. The keys of a
  fresh model differ; the controllers carry the thresholds from there.

## 6. Config & launch
- **New config fields** (`PerceiverARConfig`, defaults keep every other read unchanged): `living_stride 512`,
  `living_span_max 16`, `living_span_cap 128`, `living_span_target 8.0`, `living_found_target 0.125`,
  `living_keep 32`, `living_width 256`, `living_levels 2`, `living_search "two_level"`, `living_score "match"`,
  `living_servo True`, `living_theta 0.49`, `living_lam 0.65`, `living_servo_step 1e-3`; the read shares
  `concept_sinks`, `concept_top_k`, `concept_variance`, `concept_summaries`, `concept_sum_len`.
- **Checks at config time:** as for `concept` (no message boundary, no position code on the reading layer, one
  key head, perceiver mode); `living_stride ≤ 512`, `living_span_max` divides it, `living_span_cap >
  living_stride / living_span_max`; with flex `living_stride % 128 == 0`.
- **Arms:** `living` and `living_spans` in `scripts/run_e33n.py`; both count 141,506,208 parameters (checked).
- **Launch (150M, AWS, the spec's compute rule):** one spot H100 first, never on demand:
  `MC_SPOT=1 MC_BUDGET_USD=5000 bash Cache/keepers/e39-living.sh e39-living <hours>
  "infra/aws/jobs/e39_board.sh 42 muon:0.01 living" claude/e39-living-concepts-52f8b8 p5.4xlarge
  runs/e39/e33n_m120t_living_muon0.01_tok8.0e9_seed42/eval_core_long.json`; the hours keep the run's own cost
  under $450. After 300 optimizer steps the speed-per-dollar rule decides between staying and moving to the
  4 × L40S spot node (`g6e.12xlarge`, the same launch), which also takes the run if no spot H100 comes in 6 hours.

## 7. Tests & smoke
- `tests/test_living_concepts.py` (CPU, tiny sizes; the reference grouping of `scripts/e39_preflight.py` is the
  oracle for the batched kernels): the bounds (span ≤ 16, ≤ 128 per stride, none across a stride or a document,
  concepts ≤ C, overflow joins); the batched cut and leader rule equal the reference row by row; statistics and
  versions equal brute-force sums, with a document change inside a row; every past token counted exactly once
  (window ∪ opened ∪ remaining counts = the document's past, for every reader); identical member keys → the summary
  read equals full attention; opening everything equals full attention; causality (a changed token moves no
  earlier output, and no reader two strides back or less sees it through the memory); padding readers read zero;
  spans-only with given labels equals the concept read with the same labels; cached decoding equals the full
  forward at every step; thresholds frozen in evaluation, moving toward their targets in training, and identical
  for two replicas fed the same summed counters; gradients reach the members' keys and values through the
  summaries.
- `tests/test_living_concepts_cuda.py` (home GPU): flex equals the streamed read, forward and backward, at tiny
  sizes and at 4k / 16k with the 150M shapes; cached decoding equals the full forward at 4k and 16k; a two-card
  DDP run of a few steps ends with the same θ and λ on both ranks.
- Tiny pipeline check: `scripts/run_e33n.py cmd --arm living --size tiny --phase trunk … --local`, then decay.
- **P4:** `infra/aws/jobs/speed_check.sh`-style timing of `living` against `e37a` at the same size on one home
  card, 4k and 16k.

## 8. Risks & tradeoffs
- **One very large concept.** P3 on concept-memory keys: with 1 span in 8 founding, the largest concept holds 41 %
  of a 16k document's tokens and 20 % at 64k, and about half of the concepts hold one span. The leader rule's
  centres drift toward the average key. The read still opens by span score, but the first level filters less and
  the largest summary stands for a mixed bag. **Signal:** `living/largest_share` and `living/single_span_share`
  in training (R4), the flat-search switch (R2), summaries off (R3). Not changed here: it is the design's to
  decide; reported to the author with the pre-flight result.
- **Kernel cost** (S7, K3): the open table doubles (128 slots per stride), the span scoring is over 16k slots at
  64k, the leader loop adds ~40 small kernels per stride. **Signal:** P4, then the first 500 steps on the node.
  **Levers that do not change the model:** score only visible span slots, compile the leader loop, chunk sizes.
- **Thresholds in a moving geometry:** the controllers act per forward; a stage change (4k → 16k → 64k) shifts
  the founding rate at a fixed λ (P3: 18 % at 4k, 12.5 % at 16k, 8 % at 64k for λ 0.65), so λ rises through the
  stages. **Signal:** `living/found_rate` and `living/lam` in W&B.
- **Stage 1 rows of short documents** give few member tokens (a document must run two strides further): the
  controllers then move on few rows per step. The minimum-token rule keeps them from moving on noise.

## 9. Code sketches (`# sketch`)
```python
# sketch — nn/living_concepts.py
@dataclass
class LivingIndex:            # one forward's grouping, from the first pass's keys
    groups: GroupIndex        # span slots as groups (per_span = span_cap, span_len = stride)
    labels: Tensor            # [B, n_st, stride] slot of each token in its stride, −1 = not a member
    span_first: Tensor        # [B, G] first position (spans are contiguous)
    span_doc: Tensor          # [B, G] document, −1 = empty slot
    cof: Tensor               # [B, G] concept slot, −1 = empty slot
    n_c: Tensor               # [B, n_st] concept slots in use after each stride
    ver_doc: Tensor           # [B, n_st] the document a version belongs to, −1 = none
    seg: Tensor               # [B, n_st, n_st] 1 where stride t ≤ s is of the same document as s
    stride: int; span_cap: int; c_max: int

def cut_spans(khat, member, doc, theta, *, stride, span_max, span_cap) -> tuple[Tensor, dict]: ...
def form_concepts(u_sum, valid, sdoc, lam, *, c_max, rounds=3) -> tuple[Tensor, Tensor, dict]: ...
def build_living(k, doc, sink, cfg, theta, lam) -> Optional[LivingIndex]: ...          # no grad
def living_stats(k, v, lidx) -> LivingStats: ...                                        # grad
def living_open_table(q, lidx, st, doc, cfg) -> tuple[OpenTable, Tensor]: ...            # T, n_open
def attend_living(q, k, v, *, lidx, st, opened, n_open, doc, sink, ds, key_valid, cfg, backend) -> Tensor: ...
def living_read_stream(q, k, v, *, lidx, st, doc, ds, key_valid, cfg, q_offset=0) -> Tensor: ...
def servo_step(value, observed, target, step) -> Tensor: ...
```

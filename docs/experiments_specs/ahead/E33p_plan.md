# E33p — Implementation Plan (note-indexed exact read, per-query choice)

- **Spec:** [E33p_note_indexed_exact_read.md](E33p_note_indexed_exact_read.md) · **Status:** ready for
  `research-implement` (2026-11-06). The author chose **per-query selection** on 2026-11-06 ("Per query"); the block
  vote of the first draft is dropped.
- **Alignment page:** [e33p_architecture.html](../../3_Evaluations_and_Baselines/e33p_architecture.html). Its nine
  "my reading" choices and the two settled questions (skip window-only notes, hard cap of 512 opened tokens) are
  taken as approved; the per-query choice replaces its "one set per 128-query block".
- **Authored by:** `implementation-plan` · for → `research-implement` · branch `e33n-lsa-testbed`.

## 0. Decisions and corrections to the spec

1. **Each query chooses its own notes (author, 2026-11-06).** The approved block score s_b(n) = Σ_{i∈b} softmax_n(q_i·κ_n)
   chose one set for 128 queries, so query i's read depended on later queries of its block (q_{i+1} carries token
   i+1, position i's target): a leak. Per query, query i ranks the closed notes of its document by its own score
   s_i(n) = Σ_h softmax_n(q_{i,h}·κ_n / 8) (summed over its 9 heads, one choice shared by them, as LSA shares one
   choice across a query's heads) and opens the member tokens of its 64 best eligible notes. Exact and causal; no lag.
   LSA does the same (`select_indices`: per query, max over heads).
2. **The kernel is exact, not compute-sparse.** One flex softmax over raw keys ‖ notes, with a mask that also lets
   query i see raw token j outside its window when j is a member of a note i chose (within the cap). The lookup is a
   per-query table over groups (§3), not a gather into new slots. 128 queries of a block choose different notes, so
   the union of key blocks a query block touches is close to every closed block: at 4k the reading layer costs about
   what the dense arm's does; at 16k more. LSA's own fallback path does the same (flex over a dense [N,T,T] read
   table). **K2 becomes a cost measurement** (§7): the read budget per query stays 512 + 512 + notes, the compute is
   near-dense until a gather kernel exists (out of scope; LSA's Triton kernels are Hopper-only).
3. **The `raw` override is not "the whole prefix" in these models.** In the closed-read configs `message_raw_window =
   512` applies in every override, so `message_override("raw")` is the window-only read. The stage-0 test "open
   everything = raw read" is built against a twin with `message_raw_window = 0` (T6).
4. **"Notes off" has three readings** (§6): the notebook off (`message_override("none")`, window only: E33n's
   `--notes_off`), opened tokens off (gist entries kept), gist entries off (opened tokens kept).
5. **Which A.** The assignment over notes is the writer's competition softmax `a` (softmax over latents per head and
   token) **before** the renormalisation over tokens, from the last latent round, averaged over the 8 writer heads, in
   fp32. `lm_null_latent` must be off in tied arms (validation).
6. **Pinned choices the spec leaves open.**
   - Visible to query i (scored, and read as gist): notes of i's document whose page closed at or before i (E33's
     closed rule). κ of a visible note uses only keys at positions ≤ i, so the choice sees no future key.
   - Eligible (may take one of the 64 places): visible and holding at least one member outside i's window
     (position ≤ i − 512). A window-only note is skipped and the next best takes its place.
   - Members of note n: picked tokens of its page whose largest assignment is n. A token in two overlapping pages
     (stride 192, page 256) is a member of one note per page; it opens once if either is chosen.
   - Cap: the chosen notes fill 512 places best note first, each note's members oldest first; the note that crosses
     the cap is cut. A token in two chosen notes is counted in both (so a query opens ≤ 512 distinct tokens).
   - Scores are fp32 under `no_grad`; ties are broken by `torch.topk` (fp32 ties are vanishingly rare).
7. **Budget wording.** Opened tokens per query are flat (≤ 512). The choice costs O(S·C) logits per round, the same
   order as E33's full note read, chunked over queries; the attention kernel is near-dense (item 2).
8. **LSA-native port: what changes for our rows** (marked as changes in the ported file):
   - Our trunk rows pack documents with no cross-document attention. A key is taken only if it is in the query's
     document; the first 4 tokens of each **document** are its sinks (LSA: of the row). Clustering is unchanged
     (pools and merges may span a document boundary; such keys are simply never taken by the other document's
     queries, as LSA's window-only keys are never taken).
   - Cap 512 keys (LSA's 4k default is 1,024), so `selection_width` = 2·512/8 = 128 buckets (LSA's own rule).
   - Our reading layer runs 4 times (loop); the buckets are rebuilt from each round's keys (LSA rebuilds every
     forward).

## 1. Source & fit
- **Origin:** E33n's copy split ([run report](../../2_Experiments_Registry/run_reports/e33n_s30_lsa_testbed_20261105.md)):
  97 % of G is far copying; notes close about 2 % of it.
- **LSA** (arXiv 2610.42147; code github.com/Prefrontal-AI/latent-sparse-attention @ 5ebea73, Apache-2.0, read
  2026-11-06): `src/lsa/core/ward.py` (`pool_keys`, `ward_rounds`, `ward_merge`), `src/lsa/training/native/lsa.py`
  (`build_flat_buckets`, `_members`), `src/lsa/training/native/kernels/reference.py` (`scores`, `select_indices`),
  `layers.py` (`lsa_attention`: flex over a dense read table), `lsa_indexer.py` (`query_head_reduction: max`: unit
  per-head queries, unit keys).
- **Verdict:** Adapt LSA's addressing with our trained writer as the clusterer (A from the E31c slot competition) and
  the address key tied to the reader's key space; port LSA itself as the reference arm.
- **Boldness check:** the notebook becomes an index into kept text, trained only through the gist entries' κ. No gate,
  no auxiliary loss.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `PerceiverARConfig` | extend: `lm_note_key`, `lm_open`, `lm_open_k`, `lm_open_cap`, `par_global_read` + `lsa_*`, validation | `nn/perceiver_ar_lm.py` |
| `MessageCtx` | extend: `note_key`, `open_mode`, `open_k`, `open_cap`, `open_eval`, `index` | `nn/perceiver_ar_lm.py` |
| `LatentMemoryWriter._latent_read` / `forward` | extend: keep `a` of the last round; return `assign` [B, n_w, K, W] (fp32, grad) when asked | `nn/latent_memory.py` |
| `PerceiverARLM._run_layers` | extend: build the note index once after the writer | `nn/perceiver_ar_lm.py` |
| `Attention.forward` closed-read branch | extend: κ, per-query choice, open table, `attend_message(..., opened=…, gist=…)` | `nn/perceiver_ar_lm.py` |
| `Attention.forward` plain branch | extend: `par_global_read="lsa"` on the global layer → buckets, choice, `attend_lsa` | `nn/perceiver_ar_lm.py` |
| `make_message_mask_pred`, `dense_message_mask` | extend: optional opened term (OR into the raw part) | `nn/perceiver_ar_lm.py` |
| `_band_block_mask`, `_message_extra_blocks`, `_get_flex`, `_CLOSE_STRIDE` | reuse (the opened key blocks go in as `extra_blocks`) | `nn/perceiver_ar_lm.py` |
| `GroupIndex`, `tied_note_keys`, `note_groups`, `page_groups`, `select_groups`, `open_table`, `open_blocks`, `open_term`, `dense_open` | **new**, reusable for any "groups of raw keys, chosen per query" read | `nn/exact_read.py` |
| `pool_keys`, `ward_rounds`, `ward_merge`, `flat_buckets`, `bucket_groups` | **new**, ported from LSA (Apache-2.0 header, changes marked) | `nn/lsa_buckets.py` + `third_party/LSA_LICENSE` |
| `PerceiverARLM.open_override(mode)` | **new** eval context manager next to `message_override` | `nn/perceiver_ar_lm.py` |
| flow statistics | **new**: on-GPU recorder armed one micro-batch per log step | `nn/flow_stats.py` |
| `FlowCallback`, `SnapshotCallback` | **new** trainer callbacks; the trainer's `log()` merges pending flow metrics | `training/flow_callback.py`, `training/concept_pretraining_trainer.py`, `training/train_concept_pretraining.py` |
| `ModelArguments`, `_latent_memory_kwargs` | extend: the new fields passed through | `training/concept_pretraining_args.py`, `training/concept_pretraining_factories.py` |
| `ARMS` | extend: `notes_open`, `pages_open`, `lsa` | `scripts/run_e33n.py` |
| launcher / pipeline check | extend: `EXPERIMENT_ID=E33p`, flow flags, arm list | `infra/aws/jobs/e33n_run.sh`, `pipeline_check.sh` |
| long-doc loss / copy split | extend: `--open_off`, `--gist_off` next to `--notes_off` | `evaluation/long_doc_loss.py`, `evaluation/copy_split.py` |
| cross-arm probe | **new** | `evaluation/flow_probe.py`, `scripts/compare_flow.py` |

**No new parameters.** κ uses the reading layer's `wk`/`k_norm`; opened tokens use its raw k/v; choices have no
weights; LSA adds none. A notes-open model has the e33 arm's parameter count and state-dict keys, and with the same
seed its initial weights equal the e33 arm's. The writer's key head (`to_k`, `to_k_read`, `k_out_norm`) stays but
is unused in tied arms (`0·kk.sum()` keeps DDP happy). The LSA arm has the dense arm's parameters exactly.

## 3. Forward pass (s30, one 16k decay row; 4k trunk in brackets)
Symbols: B = 1 [4], S = 16,384 [4,096], d = 576, h = 9 query heads, one key/value head, dh = 64, pages n_w = 85 [21]
(256 tokens, stride 192), K = 32 notes per page, C = n_w·K = 2,720 [672], m = 64 notes, cap = 512, RW = 512.

**Notes-open (and pages-open):**
```
ids [B,S] → embed → x0 [B,S,576] → 2 local layers (window 128, RoPE) → x [B,S,576]
WRITER (once, unchanged): pages [B·n_w,256,256] → 2 bidir layers → 2 latent rounds
   → v̄ [B,C,1,64]; kk [B,C,1,64] (unused when tied); A = a_last.mean(heads) [B,n_w,K,256] fp32 (grad)
INDEX (once, no grad) → GroupIndex: member note per (page, slot) = argmax_k A on picked tokens;
   prefix counts P [B,n_w,257,K]; per-token codes code_a / code_b [B,S] int32 (= gid·512 + ord, −1 none);
   key-block range of each note [B,C,2]
LOOP × 4 (tied): reading layer (no RoPE)
   q [B,S,9,64]; k = k_norm(wk x) [B,S,1,64]; v [B,S,1,64]
   κ = Σ_t A[t,n] k_t / Σ_t A[t,n]  (fp32 einsum over page slots) → [B,C,1,64]        (grad → A, wk, x)
   CHOICE (no grad, query chunks ≤ 1,024): logits q·κ/8 [B,ch,9,C] → softmax over visible notes per head → sum
      over heads → s [B,ch,C]; eligible = visible ∧ count_i(n) > 0; top-64 → ids [B,ch,64]
      (pages: s summed per page [B,ch,n_w], top-2 eligible pages)
      count_i(n) = members of n in [lo_i, i − 512] (two lookups in P); off = exclusive cumsum over rank
      → open table T [B,S,G] int16 (G = C, or n_w): off for chosen groups, cap elsewhere (fresh tensor per round)
      → opened key blocks per query block [B,128,≤3·64·128 → dedup] (union of chosen notes' key-block ranges)
   ONE softmax (flex), KV = raw S ‖ κ/v̄ C = 19,104 [4,768] keys:
      raw j: same document, j ≤ i, and (i − j < 512  or  opened(i, j))
      note n: E33's closed rule (key κ_n, value v̄_n); masked under `gist_off`
      opened(i, j) = i − j ≥ 512 ∧ ∃ code ∈ {code_a[j], code_b[j]} ≥ 0: T[i, code>>9] + (code & 511) < cap
   → o [B,S,9,64] → wo → [B,S,576]; + 1 local layer
4 local layers (U-net skips) → final norm → lm_head 576→50,257 → chunked soft-capped CE (+ z-loss)
```

**LSA-native** (dense arm backbone, no writer, no notes):
```
LOOP × 4: reading layer: q [B,S,9,64], k [B,S,1,64], v
   BUCKETS (no grad): index keys = unit(k) [B,S,64]; blocks of 512 fully before the row end: 32 [8];
      pool_keys(4) → 128 pools/block → ward_rounds(64 buckets, cap 16 keys) → bucket of each key
      → centres = unit(key sum) [B,2048 [512],64]; GroupIndex: spans = blocks (512), 64 groups each,
      one membership per key (code_a), sinks = first 4 tokens of each document (code −2)
   CHOICE (no grad): index queries = unit(q_h) [B,S,9,64]; score = max_h q̂_h·centre [B,ch,G];
      visible = bucket's block < i // 512 (LSA's rule); top-128 visible (no eligibility filter: LSA's rule);
      keys taken best bucket first, position order, valid = same document ∧ not a sink ∧ ≤ i − 512, until 512
      → open table T [B,S,G] int16 + opened key blocks
   ONE softmax (flex), KV = raw S: same document, j ≤ i, and (i − j < 512 or sink(j) or opened(i, j)) → o
```
`count_i(g)` and `ord(j)` both start at lo = (document start + sinks), so a group's earlier members from another
document or the sinks never use a place; both are lookups in P, so the mask stays 1-D along the key axis plus one
2-D gather into T. Members inside the window also pass the T test only when they are in the window anyway (OR).

**Long rows (eval at 64k–128k).** The choice is chunked over queries (h·chunk·G ≤ 2²⁵ fp32). T is [B,S,G] int16:
22 MB at 4k (micro 4), 89 MB at 16k, 1.4 GB at 64k, 5.7 GB at 128k (one round at a time under `no_grad`). If 128k
does not fit, the eval runs the reading layer's attention in query chunks of 16k (flex takes Q ≠ KV lengths).

## 4. Inputs & data
Unchanged from E33n: `data/bin_token_dataset.py` via `--bin_manifest`. Trunk: FineWeb-Edu GPT-2 shards, 4,096 rows
with documents packed (the same-document test applies to opened tokens). Decay: LongDocs-lite, one document per
16,384 row. Manifests, seeds (42 / 1042, trainer seed 42) and data order as E33n (`scripts/run_e33n.py:manifests`).

## 5. Loss & objective
Unchanged: next-token chunked soft-capped CE + z-loss 1e-4, no auxiliary term. Gradient paths:
- **κ → A and wk:** the gist logits q·κ_n train the writer's competition and the reading layer's `wk`/`k_norm`.
- **Opened tokens:** raw keys and values in place (no gather), so their gradient is the dense arm's.
- **Gist values:** v̄ from the writer, as in E33.
- **None** through scores, top-k, membership, T, codes, buckets: integer tensors built under `no_grad` from detached
  q, κ, k.

## 6. Config & launch
| field | default | values / meaning |
|---|---|---|
| `lm_note_key` | `"writer"` | `"tied"`: gist key = κ |
| `lm_open` | `"none"` | `"notes"` (members of the best notes) · `"pages"` (the best whole pages) |
| `lm_open_k` | `0` | notes (or pages) chosen per query; ≥ 1 when `lm_open ≠ none` |
| `lm_open_cap` | `512` | opened tokens per query, at most (≤ 32,767) |
| `par_global_read` | `"dense"` | `"lsa"`: LSA's flat read on the global layer |
| `lsa_window`, `lsa_sinks`, `lsa_block`, `lsa_bucket`, `lsa_pool`, `lsa_bucket_cap`, `lsa_top_k`, `lsa_width` | 512, 4, 512, 8, 4, 16, 512, 0 | LSA's knobs; `lsa_width 0` = 2·top_k/bucket |

**Validation** (raise). Tied/open needs `message_write = latent_memory`, `lm_read = closed`, `lm_competition = True`,
`lm_null_latent = False`, `lm_reader_tokens = 1`, `lm_slot_pos = reader`, `global_nope = True`, `lm_read_topk = 0`,
`lm_note_age = none`, `lm_count_heads = 0`, `message_read_rounds = 1`, backend ≠ flash; `lm_open ≠ none` also needs
`message_raw_window > 0` and `lm_note_key = tied`. External notebooks (`_notes_override`, `write_notes`, `prefix_kv`)
raise `NotImplementedError` with tied keys. `par_global_read = lsa` needs no writer, `global_nope = True`, one key
head, `lsa_block % lsa_bucket == 0`, `lsa_bucket % lsa_pool == 0`, `lsa_bucket_cap ≥ 2·lsa_bucket`, backend ≠ flash.
Defaults keep every old config bit-identical (no new modules, no RNG use; guarded code paths).

**Eval switches** (like `message_override`): `message_override("none")` = window only (E33n's `--notes_off`);
`open_override("off")` = gist + window, no opened tokens; `open_override("gist_off")` = opened + window, notes masked
(the index still scores with κ); `raw` / `swapped` = no opened tokens (under `swapped`, κ and v̄ roll along the batch).
LSA arm: `open_override("off")` = sinks + window only.

**Arms** (`scripts/run_e33n.py`):
```python
"notes_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 64},
"pages_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "pages", "lm_open_k": 2},
"lsa":        {**ARMS["dense"], "par_global_read": "lsa"},
# tiny: lm_open_k 4 / 1, lm_open_cap 64, lsa_* scaled (window 64, block 64, bucket 8, cap 64)
```
Stage 1 trains these three and retrains `dense`, `window`, `e33` with the instrumentation, all with
`EXPERIMENT_ID=E33p` and `RUN_ID_SUFFIX` per arm. Launch (Polonez also `E33N_BURST_H=5.5 E33N_COOL_MIN=20`):
```
E33N_DATA=… E33N_RUNS=… E33N_MICRO_TRUNK=4 E33N_MICRO_DECAY=1 E33N_FLOW=1 CUDA_VISIBLE_DEVICES=<gpu> \
  bash infra/aws/jobs/e33n_run.sh notes_open s30        # pages_open, lsa, dense, window, e33
```
Evals per arm: long-doc loss (+ `--open_off --gist_off` for opening arms), copy split (+ `--notes_off --open_off`),
RULER 4k/16k, core (SDPA), then `evaluation/flow_probe.py` on the trunk and decay finals and the snapshots.

## 7. Tests (`tests/test_note_open.py`, `tests/test_lsa_read.py`) & stage 0
Tiny geometry from `tests/test_latent_memory.lm_cfg`: pages 16 / stride 12, 4 notes per page, `message_raw_window
32`, `lm_open_cap 32`, `global_nope True`, S = 96–192, packed multi-document rows where noted; LSA: window 32,
block 32, bucket 8, pool 4, cap 16 keys, top_k 32.
- **T1 config:** defaults; every invalid combination in §6 raises.
- **T2 old configs bit-identical:** before touching code, save golden logits at the current HEAD for the tiny window
  / dense / e33 / e33n configs; afterwards `torch.equal`, same state-dict keys. notes_open has e33's parameter count
  and, same seed, e33's initial weights; lsa has dense's. Existing suites pass: `test_note_topk`, `test_text_memory`,
  `test_latent_memory`, `test_perceiver_ar_message`, `test_bin_token_dataset`, `test_long_doc_loss`,
  `test_copy_split`.
- **T3 assignment:** `assign` = last round's competition softmax averaged over heads (recomputed by hand), sums to 1
  over notes on picked tokens, 0 elsewhere.
- **T4 tied key:** κ = Σ_t A k_t / Σ_t A by a loop over pages; q·κ_n = the A-weighted mean of q·k_t.
- **T5 membership & codes:** member = argmax A per page; every picked token has one code per covering page; ord =
  members of its note before it (from document start + sinks); counts from P equal a brute-force count.
- **T6 open-all = dense:** with `lm_open_k ≥ C`, cap ≥ S, one document and `gist_off`, the read equals a twin with
  `message_raw_window 0`, same weights (the `raw` override). With gist on it equals a hand-built SDPA reference: all
  causal raw keys ‖ visible notes keyed by κ.
- **T7 choice = brute force:** a Python loop per query over its visible notes: per-head softmax sum, eligibility (a
  window-only note is skipped and the next best takes its place), top-m order, cap filling (best note first, members
  oldest first, crossing note cut), overlap opened once. Pages: page score = sum of its notes' scores, 2 best
  eligible pages.
- **T8 causality & window:** perturbing token t leaves every logit at positions < t unchanged (whole model, single
  and packed rows; notes_open, pages_open, lsa). Every opened entry has p ≤ i − 512, the same document, and a page
  closed ≤ i.
- **T9 flex = SDPA (CPU forward):** the flex mask (`open_term`) against the dense reference mask (`dense_open`) on the
  same tables, element by element, and the attention outputs within 1e-5.
- **T10 gradients:** `assign.grad` and `wk.grad` non-zero through κ with opening off; the writer's key head gets no
  gradient in tied arms; selection outputs have no `grad_fn`; `v.grad` at a position reachable only as an opened
  token (loss on the last positions only) is non-zero, and exactly 0 under `open_override("off")`.
- **T11 switches:** `open_override("off")` = a `lm_open none` + tied twin; `gist_off` masks every note column;
  overrides restore on exit.
- **T12 LSA port = LSA:** `ward_rounds` / `flat_buckets` / the choice reproduce a verbatim copy of LSA's
  `build_flat_buckets` + `reference.select_indices` (kept in the test file under its licence header) on random unit
  keys, single document, sinks at the row start: identical labels, members and selected positions. With documents
  packed, a key of another document is never taken and does not use a place.
- **T13 trains / chunking:** tiny notes_open, pages_open, lsa: finite loss and two optimizer steps; chunked choice =
  unchunked; `run_e33n.model_args(arm, "s30")` counts as e33 / dense.
- **T14 instrumentation is passive:** a training step with the flow recorder armed gives bit-identical loss and
  gradients to one without; every logged key is finite; the callback logs only on log steps.
- **CUDA-only (stage 0, one 3090):** whole-model flex vs SDPA forward and backward (bf16 tolerances, as in
  `test_message_flex_matches_sdpa_cuda`); the open path under `torch.cuda.set_sync_debug_mode("warn")` lists no sync
  outside the per-forward setup (LSA's Ward loop syncs once per merge round, as in their code: counted, not removed).

**Stage 0 steps:**
1. Mac: `uv run pytest tests/test_note_open.py tests/test_lsa_read.py tests/test_note_topk.py tests/test_text_memory.py
   tests/test_latent_memory.py tests/test_perceiver_ar_message.py -q`, then `bash infra/aws/jobs/pipeline_check.sh
   DATA OUT --local` (tiny, all arms).
2. One RTX 3090: the CUDA tests, then `CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/speed_check.sh D OUT
   s30:e33:trunk:2.1e7:4 s30:notes_open:trunk:2.1e7:4 s30:pages_open:trunk:2.1e7:4 s30:lsa:trunk:2.1e7:4
   s30:notes_open:decay:6.5e6:1 s30:lsa:decay:6.5e6:1` (tokens/s, peak memory, flow overhead on a log step).
3. **K2 (cost, as revised in the spec):** from the measured speeds, the stage-1 hours of each opening arm (trunk +
   decay). Above 40 RTX 3090 hours per arm → kernel work first (in order: full-block marking for interior window and
   closed note blocks; a bf16 score matmul; 64×64 flex tiles; a gather read). Peak memory < 23.5 GB at micro 4,
   else micro 2 × accumulation 32 (same tokens per step).

**Speed estimate (not a measurement).** At 4k the reading layer's union of opened key blocks is near-dense (avg ~17
of 32 key blocks per query block, plus 6 note blocks, vs ~5 + 6 for e33), and flex does a 2-D lookup per entry.
Expect notes-open around 15–20k tokens/s at 4k (e33: 26.7k), lsa similar (no writer, but Ward and 4 re-clusterings
per forward), and the decay at 16k about half that. Stage 1 then needs about 25–35 RTX 3090 hours per opening arm
and 13–17 h per control: ~140 in all, about 1.5 days on Odra + Polonez, or roughly $300–350 of on-demand H100.

**Host syncs.** The per-forward `side.max().item()` and `slot_doc.max().item()` stay; the base predicate and the
band/note block lists are built once per forward (memo). Nothing in κ / choice / T / codes / blocks / mask calls
`.item()`, `bool(t)`, `nonzero` or boolean indexing; all shapes are static (m = min(lm_open_k, C), cap fixed, chunk
fixed). T and the block mask are rebuilt per round as fresh tensors (backward re-runs `mask_mod` with the tensors it
captured). LSA's `ward_rounds` keeps its own per-round `any()` checks (ported as is).

## 8. Instrumentation (spec "Instrumentation", identical in every arm)

**A. During training** — `nn/flow_stats.py` (`FlowRecorder`), `training/flow_callback.py` (`FlowCallback`), every
200 optimizer steps in the trunk and every 50 in the decay (763 steps; 200 would give 4 points). The callback arms
the recorder for the **first micro-batch** of a log step (`on_step_begin`); the forward fills on-GPU scalars; one
`torch.stack(...).tolist()` at log time; the trainer's `log()` merges them (rank 0). Overhead: one micro-batch in
3,200 (trunk), so ≪ 3 %.
1. `flow/<layer>/x_rms`, `flow/<layer>/attn_rel`, `flow/<layer>/ffn_rel`: residual RMS and ‖Δh‖/‖h‖ of each sublayer,
   for the pre layers, the reading layer and its local partner in each loop round (`r0`–`r3`), the stack layers.
2. `grad/<group>/norm` (`on_pre_optimizer_step`, after clipping) and `upd/<group>/ratio` = ‖ΔW‖/‖W‖
   (`on_optimizer_step`, from a copy of the group's weights taken before the step, on log steps only). Groups: embed,
   pre.i, read.attn, read.ffn (and its local partner), stack.i.attn / .ffn, writer, lm_head (tied: embed).
3. `read/<round>/<bucket>/mass_<source>` and `.../entropy`: for 64 sampled query positions per position bucket
   (0–512, 512–2k, 2k–8k, 8k–16k; at 4k the first three), the reading layer's exact attention probabilities
   recomputed in fp32 from the same q, K, V and the dense reference mask (`dense_open` / `dense_message_mask`)
   restricted to those rows; mass per source: window, notes (gist), opened, sinks (LSA), raw beyond the window (dense).
4. `select/<round>/cap_full` (share of queries whose counts reach the cap), `select/<round>/pages` (distinct pages
   opened per query, notes arms), `select/<round>/skipped` (visible notes ranked above the 64th chosen but ineligible),
   `select/<round>/overlap` (Jaccard of a query's chosen groups in round r vs r − 1), `select/<round>/opened` (mean
   distinct opened tokens per query).
5. `writer/assign_entropy` (mean H(A[t,·]) / log K over picked tokens), `writer/empty_notes` (share of notes with no
   members), `writer/members_max_mean` (largest / mean member count), `writer/kappa_ratio/<round>` (‖κ‖ / mean ‖k‖ of
   its members).
6. `lens/<layer>`: next-token loss read from the residual after each layer and round through the final norm and
   `lm_head`, on the last 1,024 positions of row 0.

**B. Across arms** — `evaluation/flow_probe.py --arm A --ckpt PATH --rows 16 --length 16384 --out JSON`, on the same 16
held-out LongDocs-lite rows for every arm (fixed seed, taken from the long-doc eval rows), at the trunk final (4k
rows: the first 4,096 tokens), the decay final and the snapshots (`SnapshotCallback`: model weights every 1,000
steps, `snapshots/step_N/model.safetensors`, plus the decay at step 500).
1. **Same choices:** the dense arm runs first and stores, per sampled query (512 per row at positions ≥ 2k) and
   round, its top-32 raw keys outside the window; each opening arm stores its opened set → recall@32 of the dense
   choice and the dense attention mass on the opened set; notebook arms: dense mass on the pages holding the top-mass
   notes.
2. **Copy-source hit rate:** on far-copy tokens (4-gram test, `copy_split.copy_categories`), the source position
   (the token after the most recent earlier occurrence outside the window) is in the opened set (any round / last
   round).
3. **CKA:** each layer's output (and each loop round's) on 2,048 fixed tokens per row, fp16 → linear CKA between
   every pair of arms layer by layer, and within an arm across rounds (`scripts/compare_flow.py`).
4. **Logit lens and attention by distance** for the reading layer (log₂ distance buckets), all probe rows.
Outputs: `docs/2_Experiments_Registry/results/evaluations/E33p_s30/flow/<arm>_<ckpt>.json`; `compare_flow.py` draws
the curves (training-time W&B keys and the probe JSONs) into one HTML page.

## 9. Risks & tradeoffs
- **The index may never learn to point at copy sources.** Its only signal is the gist logits (q·κ), weak on far
  copies in E33n. *Cheapest signal:* at the trunk's end, the copy split at 4k for notes_open vs the retrained window
  and dense, plus `--open_off`; and probe B2 (copy-source hit rate) against LSA's.
- **Speed (K2).** Near-dense reading layer plus a 2-D lookup per entry; sm86 shared memory for the captured tables
  (E33e's 2-D gather in a `score_mod` overflowed at 128×128 and needed 64×64). *Signal:* the first compile on the
  3090. *Fix order:* §7 step 3.
- **κ's norm shrinks** when a note's keys disagree, so its gist logit is flatter (by design: q·κ = mean q·k).
  *Signal:* `writer/kappa_ratio`, `read/*/mass_notes`, the `gist_off` delta.
- **Memory at 4k micro 4** is 21.2 GB of 24 for e33. E33p adds T (22 MB × 4 rounds), the choice chunks (~25 MB) and
  index tables (a few MB). *Fallback:* micro 2.
- **Hard membership** gets no gradient (spec risk); measured by `writer/*`.
- **LSA port fidelity.** T12 holds our port to LSA's own code on one document; the document and loop adaptations are
  ours and are marked. LSA's paper weights came from an unreleased trainer; we compare against the released code.
- **Length (S3).** The choice is O(S·C) per round and T grows as S²/6: chunked at 128k (§3).
- **K3 bookkeeping:** the retrained controls replace E33n's mixed W&B curves (window and e33n shared a run id).

## 10. Code sketches (`# sketch`: interfaces that pin decisions)
```python
# nn/exact_read.py — sketch
@dataclass
class GroupIndex:                    # per row: groups of raw keys, each living in one span (page / LSA block)
    span_start: Tensor               # [n_s] long
    span_len: int                    # L (256 pages, 512 LSA blocks)
    per_span: int                    # groups per span (K notes, 1 page, 64 buckets)
    prefix: Tensor                   # [B, n_s, L+1, per_span] int32 — members of each group before offset o
    code_a: Tensor                   # [B, S] int32 — gid·512 + ord, −1 none, −2 sink
    code_b: Tensor                   # [B, S] int32 — second covering span (overlapping pages), −1 none
    lo: Tensor                       # [B, S] long — document start + sinks (first position a query may open)
    g_doc: Tensor; g_close: Tensor   # [B, G] — document, and the position at which the group becomes visible
    g_kb: Tensor                     # [B, G, 2] — first and last key block holding a member

def counts(index, gid, hi, lo) -> Tensor: ...          # members of gid in [lo, hi], via two prefix lookups
@torch.no_grad()
def select_groups(score, visible, eligible, index, *, m, cap, hi, lo) -> tuple[Tensor, Tensor]:
    """per query chunk: top-m among `eligible` (LSA: `visible`), off = exclusive cumsum of counts → (ids, off)"""
def open_table(B, S, G, ids, off, cap) -> Tensor: ...  # [B,S,G] int16, cap where not chosen (fresh tensor)
def open_blocks(ids, index, *, S, block=128) -> Tensor: ...   # [B, nqb, E] key blocks to list, −1 unused
def open_term(T, index, *, cap, raw_window) -> Callable: ...  # (b, q, j) → bool, 1-D lookups + one gather in T
def dense_open(T, index, *, cap, raw_window, S) -> Tensor: ...  # [B,1,S,S] bool — SDPA / tests / flow stats

def tied_note_keys(assign, k, page_tok) -> Tensor: ...   # κ [B, C, 1, dh], fp32 einsum outside autocast
def note_groups(assign, pick, page_starts, doc, S) -> GroupIndex: ...   # once per forward
def page_groups(pick, page_starts, doc, S) -> GroupIndex: ...

# nn/lsa_buckets.py — ported (Apache-2.0, Copyright 2026 Prefrontal AI; changes marked)
def pool_keys(keys, pool) -> Tensor: ...
def ward_rounds(pool_keys, pool_sizes, bucket_count, bucket_cap, max_rounds=256) -> tuple[Tensor, Tensor]: ...
@torch.no_grad()
def flat_buckets(index_keys, *, block, per_block, pool, cap) -> tuple[Tensor, Tensor]:  # centres, key labels
def bucket_groups(labels, doc, *, block, per_block, sinks, S) -> GroupIndex: ...

# nn/perceiver_ar_lm.py — Attention.forward, closed-read branch (sketch)
if message.index is not None and message.override in ("real", "swapped"):
    k_bar = tied_note_keys(message.index.assign, k, message.index.page_tok).to(k.dtype)
    T = None
    if message.open_mode != "none" and message.override == "real" and message.open_eval != "off":
        T, extra = choose_notes(q, k_bar, message, ...)          # no grad
    o = attend_message(q, k, v, k_bar, v_bar, ctx=message, key_valid=key_valid, backend=self.backend,
                       block_masks=block_masks, opened=(T, extra) if T is not None else None,
                       gist=message.open_eval != "gist_off")
```

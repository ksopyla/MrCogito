# E33t — Implementation Plan (the catalogue read)

- **Spec:** [E33t_catalogue_read.md](E33t_catalogue_read.md) (approved 2026-11-07, alignment page
  `docs/3_Evaluations_and_Baselines/e33t_architecture.html`).
- **Builds on:** [E33p_plan.md](E33p_plan.md) — the per-query exact read (`nn/exact_read.py`), the tied note key, the
  flex `open_term`, the LSA arm's mask pattern (`attend_lsa`), the instrumentation (`nn/flow_stats.py`).
- **Bet, restated:** the notes leave the read and only choose what to open (4 learned empty slots take the
  "nothing here" role), every token is filed under one balanced note, the catalogue is trained by an index loss on
  where the read's attention lands, the budget is filled by tokens, and each loop pass opens notes the query has not
  opened before; a second arm searches pages first. Nothing here weakens that: no summaries return, the index loss is
  the catalogue's only learning signal besides the LM loss through the keys.

## 0. Decisions made while planning (inside the spec's frame)
1. **Budget by tokens = `lm_open_k = 128`, no new code.** `exact_read.choose` already takes the top-m notes and
   fills `cap` best note first, cutting the note that crosses it; E33p under-filled because m = 64 notes with 30 %
   empty notes and uneven sizes. With balanced notes (~8 tokens) and m = 128 the 512-token cap fills whenever 64+
   eligible notes exist.
2. **Pass exclusion state:** a per-forward `[B, S, G]` bool union of the groups each query opened in earlier passes
   (`T < cap` of each pass's table), kept on the `MessageCtx` (`open_seen`), reset per forward. 33 MB per 16k row,
   8 MB per 4×4k micro-batch; 2.1 GB at a 128k eval row (under `no_grad`).
3. **Empty slots** live in the reading layer: `sink_k` [n, dh] (passed through the layer's `k_norm`, so they sit in
   the raw keys' space) and `sink_v` [n, dh] initialised to zero (an empty slot adds nothing to the output at the
   start). KV = raw S ‖ n slots; slot n is readable by every valid query. They replace the note columns of E33p's
   KV, so `ctx.n_slots` stays the note count for everything else (writer, flow, eval).
4. **Index loss computed densely on sampled queries.** For 256 sampled query positions per row and pass (evenly
   spaced over positions ≥ the window), the read's own logits over all S raw keys + the slots are recomputed in fp32
   (`[h, 256, S + n]`, 151 MB at 16k); `allowed = window | opened (dense_open) | slots` gives the read's softmax and
   its log-normaliser; each token's group comes from `codes[0] >> 9` (one layer: pages do not overlap). Target mass
   of a chosen group = its opened tokens' probability (head mean); of a negative = Σ exp(logit − lse) over its
   eligible members (same document, older than the window), head mean. 32 negatives per query, sampled uniformly
   among choosable, unchosen notes. Prediction = softmax over the set of τ · max_h cos(q̂_{i,h}, κ̂_g); q detached,
   κ with gradient; τ = exp(log_tau), log_tau a learned scalar initialised at log 10. Loss = KL(t ‖ p) averaged over
   queries with ≥ 2 groups in the set, and over passes. Training only.
5. **Two-level search = exact masking at 16k.** Page key π_p = mean of the page's valid tokens' keys (this pass's
   k); a query keeps its 16 best visible pages (max-head cosine) that hold an eligible note, and only their notes stay
   candidates. At 16k all note scores are still computed (cheap); the compute saving (scoring only kept pages' notes,
   the choice as a list) is stage C's gather read. S4 is a quality test, which this implements exactly.
6. **Balanced filing.** Â = 3 Sinkhorn iterations on the head-mean assignment A [B, n_w, K, W] restricted to the
   page's valid (picked) tokens: columns (notes) scaled to sum to n_valid / K, rows (tokens) renormalised to 1. Hard
   filing (no gradient): tokens propose notes in order of Â, best first; a note accepts proposers by Â until 16;
   up to 4 proposal rounds, then any note with room. κ uses Â (gradient flows through Sinkhorn into A).
7. **Eval switches:** `open_override("off")` = slots + window (no opened tokens); new `open_override("reuse")` = a
   pass may re-open earlier passes' notes (pass exclusion off). `message_override("none")` = window only, as before.

## 1. Source & fit
E33p stage 1 (spec table, L1–L7); LSA (arXiv 2610.42147) as the reference read; SwAV's equipartition for the
filing; distillation of a read's attention into a router (as in learned-routing sparse attention) for the index
loss, here applied to a catalogue written by a trained writer. Encode = the writer (unchanged weights, new filing);
reason = the 4-pass loop (passes as successive lookups); decode / loss unchanged plus the index loss.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `PerceiverARConfig` | extend: `lm_note_read`, `lm_filing`, `lm_filing_iters`, `lm_filing_cap`, `lm_open_score`, `lm_open_new_per_pass`, `lm_open_levels`, `lm_open_pages`, `lm_sinks`, `lm_index_loss`, `lm_index_queries`, `lm_index_negatives`; validation | `nn/perceiver_ar_lm.py` |
| `MessageCtx` | extend: `open_seen`, `index_losses`, `index_stats` | `nn/perceiver_ar_lm.py` |
| `PerceiverARLM._note_index` | extend: balanced filing → Â, labels, member lists | `nn/perceiver_ar_lm.py` |
| `Attention.__init__` | extend: `sink_k`, `sink_v`, `index_log_tau` (reading layer, catalogue arms only) | `nn/perceiver_ar_lm.py` |
| `Attention.forward` closed branch | extend: catalogue path (κ from Â, choice with score / exclusion / levels, `_attend_catalogue`, index loss) | `nn/perceiver_ar_lm.py` |
| `_attend_catalogue` | **new**: raw window ‖ opened ‖ empty slots, flex (`make_message_mask_pred(gist=False)` + slot term, extra blocks) and SDPA reference | `nn/perceiver_ar_lm.py` |
| `PerceiverARLM.forward` | extend: `loss += lm_index_loss · mean(index_losses)` | `nn/perceiver_ar_lm.py` |
| `open_override` | extend: `"reuse"` | `nn/perceiver_ar_lm.py` |
| `balance_assign`, `file_capped`, `page_keys`, `index_loss` | **new**, reusable | `nn/exact_read.py` |
| `note_groups` | extend: `member=` override (filed labels) | `nn/exact_read.py` |
| `note_open_table` | extend: `score` (`sum_softmax` / `max_cos`), `exclude`, `page_key` + `keep_pages` | `nn/exact_read.py` |
| `FlowRecorder.reading_open` | extend: `sink_k` (mass on slots), index KL / rank correlation, filing balance | `nn/flow_stats.py` |
| `ModelArguments`, factory pass-through | extend | `training/concept_pretraining_args.py`, `training/concept_pretraining_factories.py` |
| `ARMS` | extend: `catalogue`, `catalogue_2l` (+ tiny) | `scripts/run_e33n.py` |
| long-doc loss | extend: `--open_reuse` | `evaluation/long_doc_loss.py` |
| `infra/aws/jobs/e33p_s1_multi.sh` | reuse with `E33N_EXPERIMENT=E33t` (wrapper `e33t_a.sh`) | `infra/aws/jobs/` |

New parameters: `sink_k`, `sink_v` (4 × 64 each) and `log_tau` (1) in the reading layer: 513 in all. Everything
else has the E33p notes-open shape; the window control's FFN fit is unchanged (`run_e33n.fit`).

## 3. Forward pass (s30, one 16k decay row; 4k trunk in brackets)
B = 1 [4], S = 16,384 [4,096], d = 576, h = 9, one KV head, dh = 64, pages n_w = 64 [16] (256 tokens, stride 256),
K = 32, C = G = 2,048 [512], m = 128, cap = 512, RW = 512, n_sinks = 4.
```
ids → embed → 2 local layers → x [B,S,576]
WRITER (once): pages [B·64,256,·] → A (head mean, last round) [B,64,32,256] fp32 (grad)
INDEX (once): Â = sinkhorn(A, valid, 3)            (grad)
              labels = file_capped(Â.detach(), valid, cap 16) [B,64,256] (−1 invalid)
              GroupIndex from labels (one layer of codes); members [B,2048,16] (positions, −1 pad)
LOOP × 4 (tied):
   q [B,S,9,64]; k [B,S,1,64]; v
   κ = Σ Â k / Σ Â  [B,2048,1,64] (grad → Â, wk); π = page mean of k [B,64,64] (2-level arm)
   CHOICE (no grad): s = max_h cos(q_h, κ) [B,ch,2048]; cand = visible ∧ eligible ∧ ¬open_seen
        (2-level: ∧ note's page in the query's 16 best pages by max_h cos(q_h, π))
        top-128 → fill 512 tokens → T [B,S,2048] int16; open_seen |= T < cap   (skipped under "reuse")
   READ (flex): KV = raw S ‖ 4 slots; raw j readable iff same doc ∧ j ≤ i ∧ (i − j < 512 ∨ opened(i, j));
        slot readable iff i is a valid token → o [B,S,9,64] → wo
   INDEX LOSS (training): 256 sampled queries → t (read mass per chosen group + 32 negatives), p = softmax(τ·s)
        → KL(t‖p) appended to ctx.index_losses
→ 5 local layers → lm_head → CE + z-loss + 0.1 · mean(index_losses)
```

## 4. Inputs & data
Unchanged (E33p / E33n manifests). Seed 1 on the AWS node with micro 4 × accumulation 4 × 4 GPUs (trunk) and
1 × 4 × 4 (decay): 64 rows of 4k per optimizer step, as the seed-1 controls.

## 5. Loss & objective
CE + z-loss 1e-4 + `lm_index_loss` (0.1) × index KL. Gradient paths: κ → Â → A (writer competition) and wk; opened
raw keys/values as the dense arm; slots' k/v; τ. None through choice, filing labels, T, exclusion.

## 6. Config & launch
| field | default | catalogue value |
|---|---|---|
| `lm_note_read` | `"gist"` | `"index"`: notes are not keys of the read |
| `lm_filing` / `lm_filing_iters` / `lm_filing_cap` | `"argmax"` / 3 / 16 | `"balanced"` |
| `lm_open_score` | `"sum_softmax"` | `"max_cos"` |
| `lm_open_new_per_pass` | `False` | `True` |
| `lm_open_levels` / `lm_open_pages` | 1 / 16 | 1 (main) · 2 (two-level) |
| `lm_sinks` | 0 | 4 (needs `lm_note_read = index`) |
| `lm_index_loss` / `lm_index_queries` / `lm_index_negatives` | 0.0 / 256 / 32 | 0.1 |
| existing: `lm_open`, `lm_open_k`, `lm_open_cap`, `lm_note_key`, `lm_stride` | — | `notes`, 128, 512, `tied`, 256 |

Validation (raise): `lm_note_read = index` needs `lm_open = notes`, `lm_note_key = tied`; `lm_filing = balanced`
needs `lm_open = notes`; `lm_open_levels = 2` needs `lm_open = notes`; `lm_sinks > 0` needs `index`;
`lm_index_loss > 0` needs `index`; `lm_filing_cap ≥ lm_window / lm_latents`. Defaults keep every older config
bit-identical (no new modules or RNG use when off).

Arms (`scripts/run_e33n.py`):
```python
"catalogue":    {**_WRITER, **_LOOP, "lm_stride": 256, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 128,
                 "lm_open_cap": 512, "lm_note_read": "index", "lm_filing": "balanced", "lm_open_score": "max_cos",
                 "lm_open_new_per_pass": True, "lm_sinks": 4, "lm_index_loss": 0.1},
"catalogue_2l": {**ARMS["catalogue"], "lm_open_levels": 2, "lm_open_pages": 16},
# tiny: lm_open_k 8, lm_open_cap 64, lm_open_pages 2, lm_index_queries 32, lm_index_negatives 4, lm_stride = window
```
Launch (stage A, after the seed-1 controls): `MC_SPOT=1 bash infra/aws/launch.sh e33t-a 14
"infra/aws/jobs/e33t_a.sh" e33n-lsa-testbed g6e.12xlarge`, where `e33t_a.sh` runs `e33p_s1_multi.sh 1 catalogue
catalogue_2l` with `E33N_EXPERIMENT=E33t` (results under `runs/e33t/`).

## 7. Tests (`tests/test_catalogue_read.py`) & stage 0
1. Defaults: the E33p golden logits stay bit-identical (`tests/test_note_open.py` golden test runs unchanged).
2. Sinkhorn: column sums of Â ≈ n_valid / K (± 1 %), rows sum to 1.
3. Filing: every valid token filed once; 1 ≤ members ≤ 16 on full pages; the cap holds on adversarial A (all tokens
   prefer note 0).
4. Choice: `max_cos` equals a brute-force cosine-max top-k; exclusion: no group opened twice by a query across 4
   passes; `levels = 2` with `keep_pages ≥ pages` equals `levels = 1`.
5. Read: flex predicate (evaluated on index grids) equals the dense mask; with cap ≥ all eligible tokens the opened
   set equals all closed-page tokens beyond the window (the dense ceiling restricted to closed pages).
6. No future leak (content), with `_future_leaks` on the catalogue configs, packed and single-document.
7. Index loss: finite; gradient reaches the writer's competition parameters, `wk`, `log_tau`; none reaches `wq`
   (q detached) through it; zero when `lm_index_loss = 0`.
8. Slots: with `sink_v = 0` and slot logits forced to −∞ the read equals the slot-free read.
9. Eval switches: `off` and `reuse` run and differ from `on` where expected.
Stage 0: `tests/test_note_open_cuda.py`-style GPU test for the catalogue read (flex = SDPA reference, forward and
backward) on one GPU; speed and memory at 4k (micro 4) and 16k (micro 1) for both arms next to LSA-native (K4 early
check); tiny pipeline check on the Mac (`scripts/run_e33n.py cmd --size tiny --local`).

## 8. Risks & tradeoffs
- **Index loss self-reinforcement** (only chosen groups have read mass): negatives' would-be mass breaks it; K2
  (rank correlation < 0.2 at 25 % of the trunk) stops a run that is not learning.
- **Sinkhorn on a sharp A** (E33p's assignment entropy 0.44) needs more iterations to balance; the filing cap
  guarantees ≤ 16 regardless, and the balance is logged.
- **Memory at 128k eval:** `open_seen` 2.1 GB + T 4.3 GB (2,048 notes per 16k → 16,384 at 128k; T chunked by the
  2³¹ rule) on a 48 GB L40S: fits; on a 24 GB 3090 the 128k eval may need the query-chunked read (E33p plan §3).
- **Two levels at 16k save no compute yet** (decision 5); the saving is stage C.

## 9. Code sketches (`# sketch`)
```python
# nn/exact_read.py — sketch
def balance_assign(A, valid, iters=3):              # A [B,n_w,K,W] fp32 (grad), valid [B,n_w,W] bool → Â
def file_capped(A_hat, valid, cap=16, rounds=4):    # → labels [B,n_w,W] long (−1 invalid), no grad
def page_keys(k, page_tok, valid):                  # → π [B,n_w,dh] (mean of valid tokens' keys)
def note_open_table(q, kappa, no, *, doc, slot_close, slot_doc, raw_window, chunk=0,
                    score="sum_softmax", exclude=None, page_key=None, keep_pages=0) -> OpenTable
def index_loss(q, k, sink_k, kappa, opened, *, doc, raw_window, rows, negatives, log_tau, gen) -> (loss, stats)

# nn/perceiver_ar_lm.py — sketch
def _attend_catalogue(q, k, v, sink_k, sink_v, *, ctx, key_valid, backend, opened): ...  # KV = raw ‖ slots
```

## 10. Stage C — long documents (design 2026-11-08, approved 2026-11-11, run the same day; results in the spec)
**Why new code.** The read stores each query's choice as the table T [B, S, G] int16 with G ∝ S, so T grows with
S²: 4.3 GB at 128k (G = 16,384), 17 GB at 256k, 275 GB at 1M (G = 131,072). Everything else in the forward is
linear in S at eval (chunked choice, chunked CE, block masks of (S/128)² bits, writer per page), so T is the one wall.
`open_seen` is not needed: the reuse arms have no pass exclusion.

**Measured data (val split, 2026-11-08):** 17 documents ≥ 128k (gutenberg 4, arxiv 1, doab 8, usgpo 4), 4 ≥ 256k
(doab 2, usgpo 2), 1 ≥ 512k and ≥ 1M (one usgpo document of 1,295,611 tokens). The long-document loss at 1M is
therefore one document: an illustration, not a statistic. The planted password at 1M (filler = that document,
20 depths) is the 1M test with rows enough to count.

**Decisions.**
1. **The choice as a list** (`lm_open_impl = "list"`, eval only, no gradient): `note_open_table` keeps its loop and
   writes each chunk's `ids` [B, ch, m] and `off` [B, ch, m] (and the counts) into a `ChoiceList` instead of T:
   [B, S, m] int32 + int16, 0.8 GB at 1M with m = 128. Same collector for the LSA port's buckets
   (`nn/lsa_buckets.py` calls the same `choose`).
2. **Opened positions without T.** For a query, chosen group g takes places [off_g, off_g + cnt_g) ∩ [0, cap); the
   opened tokens are g's members in [doc_start(i), i − RW], oldest first. Each of the cap slots t finds its group by
   `searchsorted` over off + cnt and reads `members[g, r0_g + t − off_g]` (r0_g = g's members before doc_start(i),
   from the prefix counts). members [B, G, F] (F = largest group: 16 for the catalogue, up to 512 for LSA buckets) is
   built once per forward from the codes. Exactly the table rule when every token is in at most one group per
   forward (one code layer: the catalogue's filed notes, LSA's closed blocks); overlapping spans are refused.
3. **The read per query chunk** (1,024 queries): one softmax over the raw window (keys [s0 − RW + 1, s1), band and
   same-tag test, key_valid as the flex predicate) ‖ the ≤ cap gathered tokens (same tag) ‖ the learned slots (or
   LSA's sinks). One KV head, so the gather is [ch, cap, 64]: about 130 MB per chunk.
4. **Exactness test before any long run:** at 16k (tiny and s30 shapes, fp32), the list read equals the table read
   (SDPA reference) to 1e-5, for the catalogue and the LSA port, with and without document packing.
5. **Training stays on the table** (flex, gradients). The 64k stage's T is 1 GB per row: fits.
6. **The 64k stage** (new phase `long` in `scripts/run_e33n.py`, `e33n_run.sh` after the decay): from the 16k
   decay's final weights, LongDocs-lite rows of 65,536, 0.1B tokens, picking up at the rate the decay ended on
   (1e-4, 10 % of the peak) and decaying 1−sqrt to 1e-5, 4 rows per optimizer step (the decay's 262k tokens per
   step; ~380 steps). Given to the stage-B winner,
   the LSA port and the window control alike (the controls get the same tokens; dense is not a ceiling past 16k:
   at 64k the reuse catalogue is already at 152 % of its gap).
7. **Two levels as an eval switch first.** The page key is the mean of a page's raw keys (no parameters), so the
   two-level search can run on the flat-trained winner at eval (`levels = 2`, 16 pages). If it matches the flat
   search at 128k–1M, no two-level model is trained (saves ~$60); if it loses and the compute matters, the second
   training run of the spec is the two-level reuse arm.
8. **Evals:** long-document loss at 128k, 256k (all rows that exist), 1M (one document), each with the notebook
   on / off; planted password at 128k and 1M; the window control as the floor, the LSA port (list read) as the
   reference, the 16k-only model to show what the 64k stage adds.

**Cost:** three 64k stages (~1 h each on an H100) + evals (~1 h per model at 1M) ≈ $30–50, against the spec's $80.

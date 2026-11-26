# E41 — implementation plan: the tree concept model

Spec: [E41_tree_concept_model.md](E41_tree_concept_model.md). Written 2026-11-26 with the build (the author asked for
build and training in one go); it records what the code does and why, so the next change starts from here.

## 0. Shape of the change
The tree is an **extension of the concept read**, not a new read type: `par_global_read` stays `concept` and
`concept_tree=True` adds the levels. Everything that knows the concept read keeps working unchanged: the eval
switches (`concept_off`, summaries off, nothing opened, the length scale), `--fast`'s compiled grouping, the Hub
export, cached decoding's kind `concept`. With every `tree_*` option off the code path is the concept read's.

| part | file | new |
|---|---|---|
| levels, choice, read, reasoner, losses | `nn/concept_tree.py` | new module |
| options, `_tree_read`, aux output of the reading block, train stats | `nn/perceiver_ar_lm.py` | `concept_tree`, `tree_*` |
| cached decoding | `nn/incremental.py` | tree build per pass, one-query read, plan |
| trainer arguments and pass-through | `training/concept_pretraining_args.py`, `…_factories.py` | 19 fields |
| arms | `scripts/run_e33n.py` | `tree`, `tree_noaux`, `tree_read` (+ tiny sizes) |
| eval switches | `evaluation/long_context_use.py` | memory modes `tree<N>`, `descent<N>`, `cut<M>` |
| checks | `scripts/e41_tree_view.py` | a tree view of a concept-memory checkpoint; identity and bounded-read table |
| AWS job | `infra/aws/jobs/e41_tree.sh` | stages 2–3 from concept memory's stage 1, experiment E41, `m21mix_v4` |
| tests | `tests/test_concept_tree.py`, `tests/test_concept_tree_cuda.py`, 3 arms in `test_incremental_decode.py` | 32 + 17 + 3 arms |

## 1. The tree (`build_tree`)
Input: the pass's `ConceptIndex` (C1 = closed blocks × 64 concepts; n, μ_k, σ², μ_v, z; member links). Output:
`TreeIndex`, every level's entries in one array (level 1 first), E ≈ S/7.

- Level 2: per block, the 64 concepts' unit mean keys → `concept_labels` (the same balanced k-means, compiled under
  `--fast`) → 8 nodes. Level ℓ + 1: the 64 nodes of 8 neighbouring level-ℓ containers → 8 nodes. Containers span
  512 · 8^(ℓ−2) tokens, aligned to the row.
- A node's statistics are member-weighted sums of its children's: n, μ_k, μ_v, z exactly; σ² from the mean squared
  key norm (so it is the true per-dimension variance of the node's member keys). Gradient flows through the
  statistics, never through the grouping.
- `frm` = (container's last block + 2) · 512: the first query position that may read the entry (the concept read's
  rule, applied to the container). `pfrm`, `par`: the parent's `frm` and entry. A container with two documents is
  not built: `frm = BIG`, its children's `pfrm = BIG` (roots for ever).
- Shapes at 64k: C1 8,064; level 2 1,008; level 3 120; level 4 8. At 10M: 1.25M / 156k / 19.5k / 2.4k / 305 / 38 / 5.

## 2. The read
- **Choice** (`tree_choice` → `choose_chunk`, no gradient), per query, top-down. Candidates of a level: readable
  (`frm ≤ q`, same document) and (a root: `q < pfrm`, or the parent was followed). The best `follow` candidates are
  followed. Level 1: the concept read's `choose` among concepts whose parent was followed (or that are roots), up to
  `top_k` tokens.
  - Score `lookahead`: log-sum-exp of the concepts' estimates under the node (scatter by parent links, level by
    level). It is the flat read's own routing; it costs the concept read's dense concept scores (fine to 128k).
  - Score `descent`: the node's own estimate log n + q·key/√d + |q|²σ²/2d. Needed where dense concept scores are too
    many (1M and beyond); the route loss trains it toward `lookahead`.
  - `RowReads`: per row, `follow` per level (ALL / N / 0) and a token cap. Outside training: `fixed_reads`
    (`tree_follow`, eval switch `cut`). In training with `tree_row_mix`: `row_reads`, drawn from the row's first
    min(256, 2 blocks) tokens (no query among them has a memory, so the draw leaks nothing; the same in every loop
    pass and in a recomputation).
- **Attention** (`attend_tree`): one softmax over sinks ‖ block-aligned window ‖ opened members ‖ visible summary
  entries. A summary entry is visible iff readable, (root or parent followed), not followed itself (level 1: not
  opened). Flex path: the concept read's raw call + one call over all entries (visibility in `mask_mod`, bias
  log n + |q|²σ²/2d in `score_mod`, detached as in the concept read), merged by log-sum-exp. Reference path: dense
  mask + bias (CPU tests). With every row flat the summary call covers level 1 only: the concept read, same cost.
- Every past token is represented once (test T3). Known inherited gap: a concept opened in part drops its unopened
  members (the released model's behaviour; kept so that step 0 is the concept read).

## 3. The reasoner (`TreeReasoner`, 7.4M weights at width 384)
Tokens: the nodes of level ≥ 2 (state z = member-mean of the reading layer's input, + a level embedding). Mask:
node i sees node j iff same document, both readable, `frm_j ≤ frm_i`. Two layers looped twice with input injection
(E37's reasoner block). Outputs, all zero at the start:
- `dk`, `dv` for nodes of level ≥ 2; for concepts an MLP of (own state + parent's reasoned state) → `dk1`, `dv1`.
  The read uses key = μ_k + dk, value = μ_v + dv everywhere (choice, logits, values).
- plan: the pooled state of a block's level-2 nodes → `Linear(→ 896)`; token t gets the plan of block t//512 − 2
  (the newest block it may read), added to the reading layer's output on every pass.
- optional `tree_token_update`: opened tokens read v + their concept's `dv1` (two raw copies, as E37's reasoner).

## 4. Losses
- **Next concept.** Codes: `e = normalize(z · R)` with R a fixed random 896 × 64 matrix (a buffer), code = nearest of
  1,024 (level-2 nodes) / 4,096 (concepts) codebook entries. Codebooks: moving average of assigned vectors (decay
  0.99), unused codes re-seeded from a spare batch vector; the sums of one forward are applied at the start of the
  next (so a forward and its recomputation see the same codebook), all-reduced over the cards. Prediction: after
  block b, `head(silu(pred(pooled_b)))` → cross-entropy against the histogram of block b + 1's codes, coarse and
  fine. Block b's pooled state sees only nodes with `frm ≤ frm_b`: no leak.
- **Route.** 256 queries per row; student: node's estimate with key = μ_k.detach() + dk; teacher: log-sum-exp of
  its concepts' estimates; smooth-L1 over readable (query, node) pairs.
- Both leave the reading block as a second output (`_call_block_aux`): under gradient checkpointing (the 64k stage)
  their gradient comes from the recomputation like the stream's. Averaged over the 4 passes; total loss =
  next word + 0.1 · (coarse + fine) + 0.1 · route. Logged as `tree/…` through `pop_train_stats`.
- DDP: every reasoner parameter is in the graph in every forward (zero-weight terms on short rows).

## 5. Run recipe (run 1)
`infra/aws/jobs/e41_tree.sh tree`: run folder `runs/e41/e33n_m120t_tree_muon0.01_tok8.0e9_seed42`, seeded with
concept memory's `trunk/…/final`; `e33n_run.sh` then runs decay (16k, 3.0B, 5,722 steps) and long (64k, 0.5B, 954
steps, gradient checkpointing) on `m21mix_v4`, then the job's tests. Row draw 0.4 / 0.2 / 0.4, bounded follow 16.
The new weights (reasoner) are matrices ≤ 8:1 aspect → Muon, like the body.

## 6. Checks before the paid run
1. `uv run pytest tests/test_concept_tree.py` (CPU) and `tests/test_concept_tree_cuda.py` (GPU): done, 49 pass.
2. Identity on the real weights: `scripts/e41_tree_view.py check` (released concept memory vs its tree view).
3. Bounded-read table on the released model (free readout: what the untrained tree read costs).
4. Speed: `speed_check.sh` m120t tree vs e37a at 16k on one home card (K2: ≥ 0.70 ×).
5. Pipeline: `e38_pipeline_check.sh 0.01 tree m120t` (three stages, eval, resume, quick tests).

## 7. Risks seen while building
- The bounded read's flex mask is not block-sparse (a query block's followed nodes scatter), so a bounded row is
  not faster than a flat row in training. Speed for 1M+ rows needs a gathered read (follow-up).
- `lookahead` scores all concepts per query: fine to 128k, not at 1M+; `descent` is the path there and is the
  least-tested part until the route loss has trained.
- The reasoner's mask is dense over nodes (1,150² at 64k; 18k² at 1M needs chunking).
- Codes of a fixed random projection may track surface content; the readout is next-concept accuracy against the
  unigram baseline of the codes.

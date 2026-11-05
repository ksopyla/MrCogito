# E33n — E33 on the LSA testbed: natural long text at 30M → 100M → 300M, with a top-k note read

- **Status:** **killed at stage 1 (K1), 2026-11-05.** Approved by the author's request of 2026-11-04 ("write spec,
  implement, build the evaluation like the LSA paper with the same datasets, build AWS infrastructure, start, train and
  stop; do not spend more than $2k"). Stage 1 (30M) ran on the home servers (no H100 capacity); stages 2–3 not run.
  AWS spend $0.17.
- **Serves:** vision priority 2 (long context through the notebook) on the way to priority 4 (agent-to-agent). First
  test of E33 on **natural** long text with a published protocol, so our numbers can be read next to a published
  architecture (Latent Sparse Attention, LSA).
- **Implementation plan:** [E33n_plan.md](E33n_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-04 · closed 2026-11-05
- **Research note behind it:** [hierarchy_lessons_hnet_ncm_mai_20261030.md](../../4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md)
  §1.4 (LSA) and §4 step 3; LSA review in
  [learned_kv_context_compression.md](../../literature_review/learned_kv_context_compression.md#latent-sparse-attention-lsa).

## Hypothesis
If E33's reading layer reads, per query and head, only its **top-k notes** (k = 64, chosen by its own score with no
gradient through the choice) instead of a softmax over every note of every closed page, then on held-out long
documents E33 keeps the far-context value it has at the training length (16k) and **keeps it at 4–8× the training
length** (64k–128k), where the full note read dilutes — because a fixed-size softmax cannot be swamped by a notebook
that grows with the document (our own 1M-token measurement: ~20 fetched notes 97–100 %, the whole notebook 42–50 %).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (perceiver_ar, the closed-window latent notebook `lm_read="closed"`, the E33a
  loop), `nn/latent_memory.py` (the writer), the shared trainer `training/train_concept_pretraining.py` and
  `scripts/train_concept_pretraining_multigpu.sh`. No fork.
- **Init:** random init for every arm (the trunk); each long decay starts from its own arm's trunk checkpoint with a
  fresh optimizer (LSA's branch rule).
- **Baseline to beat:** E33 itself, trained in this protocol with the full note read (arm "E33, full note read"
  below), plus the two controls that fix the ceiling and the floor. There is no earlier E33 run on natural text at
  these sizes; the text-check runs (30M, synthetic world documents, our 4,096-token vocabulary) are not comparable.
- **Materially new vs E33 as tested on text:**
  1. the top-k note read (LSA's selection, applied to compressed notes instead of raw keys — LSA does not do this);
  2. the LSA testbed layout and protocol: local windows 128 (was 16), raw reading window 512 (was 256), GPT-2
     tokenizer, FineWeb-Edu 4k trunk, LongDocs decay at 16k, WSD schedule, LSA's evaluation;
  3. scale: 30M → 100M → (300M), one H100 per run.

## The architectural bet
Same stack for every arm (only the reading layer differs):

```
tokens → input layer (factorised table, GPT-2 vocabulary, tied output)
       → prelude: 2 local layers (sliding window 128, RoPE)
       → core × 4 rounds, weights tied (E33's loop): [reading layer + 1 local layer], zero-init round marker
       → coda: local layers (window 128) → output head
reading layer (one KV head, no position code on notes):
       window control       : the last 512 raw tokens only                     (student sees the last pages)
       dense ceiling        : every earlier raw token, full causal attention    (student may reread any page)
       E33, full note read  : the last 512 raw tokens + every note of every closed 256-token page
       E33n, top-k note read: the last 512 raw tokens + its 64 best notes (per head, by score, no gradient
                              through the choice; the scores of the kept notes keep their gradient)
writer (E33 / E33n only): E31c page writer, 32 notes per 256-token page, stride 192 (~6 tokens per note)
```

- **Why top-k and not LSA's full index.** At 16k there are ~2.7k notes; scoring all of them is cheap, so the
  selection itself is the change under test. LSA's content-clustered index (Ward buckets of ~8 notes) only changes
  the *cost* of finding the top-k at 1M+; it is a later step, measured read-only on these checkpoints.
- **Loop exits off** (E33x setting) in every arm: on natural text an exit loss would put three extra 50k-way output
  passes on every token, and E33x showed exits make the rounds interchangeable.
- **Sizes** (all arms of a size identical except the reading layer; parameters counted with embeddings):

  | size | width | layers (unique) | passes per token | run tokens (trunk + decay) |
  |---|---|---|---|---|
  | 30M | 576 | 2 + 2 + 4 = 8 | 14 | 1.5B + 0.2B |
  | 100M | 896 | 3 + 2 + 6 = 11 | 17 | 3.0B + 0.4B |
  | 300M (stage 3, gated) | 1280 | 4 + 2 + 8 = 14 | 20 | 5.0B + 0.6B |

  Exact widths are fitted by `fit_params`-style counting in the plan; tokens per parameter fall with size (50 → 30 →
  17), all far below LSA's 200 because of the budget; the comparison between arms is at equal tokens, which is what
  the claim needs.

## Training protocol (LSA's, scaled down)
- **Trunk:** FineWeb-Edu (GPT-2 tokens, the 10B-token release `kjj0/finewebedu10B-gpt2`), rows of 4,096 tokens,
  documents packed with document ids (no attention across documents), WSD schedule: warmup 1 %, constant peak,
  decay over the last 10 % with a square-root shape to 10 % of peak (LSA's FineWeb trunk decays 1 %; we decay 10 %
  so the trunk checkpoint is itself a usable model). AdamW (0.9, 0.95), weight decay 0.1 on matrices, clip 1.0,
  bf16. Peak learning rate from a 3-point sweep at 30M (§Plan stage 0); larger sizes scale it by width.
- **Long decay:** from the arm's trunk checkpoint at the start of its decay, fresh optimizer, LongDocs-lite at rows of
  16,384 tokens, warmup 3 %, decay over 97 % (LSA's 32k decay shape), ~12 % of the trunk tokens (LSA: 12.3B / 100B).
- **LongDocs-lite:** LSA's LongDocs recipe (Common Pile sources: Project Gutenberg books, arXiv papers, DOAB books,
  US government publications, UK Hansard; documents of at least 16k GPT-2 tokens; code repositories from The Stack
  are left out because the dataset is gated), equal token share per source where available, 1 % of documents held out
  per source for evaluation. LSA filters at 64k tokens for 128k training; we filter at 16k for 16k training.

## Evaluation (LSA's suite, plus the far-context checks we need)
1. **Validation loss** on held-out FineWeb-Edu at 4k (the trunk) and on held-out LongDocs-lite at 16k, **by token
   position** (0–512, 512–2k, 2k–8k, 8k–16k): far positions are where the reading layer must earn its keep.
2. **Length transfer:** the same held-out long documents read at 32k, 64k and 128k (eval only).
3. **RULER**, LSA's 13 tasks (needle-in-a-haystack single 1–3, multikey 1–3, multiquery, multivalue, variable tracking,
   common and frequent word extraction, two QA tasks), 100 documents each, GPT-2-sized prompts, string-match grading,
   at 4k, 8k and 16k (LSA's column is the mean at 4k), plus 32k and 64k past the training length.
4. **Short benchmarks** (LSA's "core"): HellaSwag, ARC-easy, ARC-challenge, PIQA, WinoGrande, LAMBADA, WikiText, SciQ —
   the no-harm check (they never close a notebook page, so they measure the backbone).
5. **Notebook value inside a model:** E33 / E33n with notes switched off at eval (the far-position loss gap).
PaperTrail (LSA's 512k–10M benchmark) is out of scope at these sizes.

## Success criteria (set BEFORE running)
Let **G** be the far-context value measured by the two controls at the same size: window control loss − dense
ceiling loss on LongDocs-lite positions 2k–16k at 16k (the measured ceiling; LSA's 0.55B shows window-only 1.154 vs
full 1.085 at 128k, a gap of 0.069 nats per token).
- **S1 (the notebook carries far context on natural text):** E33 or E33n recovers ≥ 40 % of G at 16k, at 30M and at 100M.
- **S2 (top-k does not cost at the training length):** E33n within 0.005 nats of E33 (full note read) on positions
  2k–16k at 16k, and within 3 points on RULER at 4k–16k.
- **S3 (top-k holds where the full read dilutes):** at 64k and 128k, E33n's loss on positions ≥ 2k is at least
  0.01 nats below E33's, or E33n's RULER needle mean at 64k is ≥ 10 points above E33's.
- **S4 (no harm):** E33n's core-benchmark mean within 1 point of the window control at the same size.
- **Scale gate to 300M:** S1 and (S2 or S3) hold at 100M.

## Kill criteria (set BEFORE running)
- **K1:** at 30M after the long decay, both notebook arms recover < 15 % of G → stop the notebook arms; the 100M
  stage runs only the two controls if the author wants a reference.
- **K2:** E33n is > 0.02 nats worse than E33 on positions 2k–16k at 16k at 30M → the selection hides content; stop E33n.
- **K3 (budget):** cumulative spend reaches $1,900 → stop every instance and report.
- **K4 (health):** loss spikes > 0.5 that do not recover within 1k steps, or a trunk loss more than 0.1 above the
  window control at 25 % of the run → stop that run and debug.

## Plan
- **Stage 0 (≤ $60):** smoke tests and throughput on one H100 for the four arms at 30M (100 steps each), plus a
  learning-rate check at 30M (three peaks, 0.1B tokens each, window control) — decides the peak learning rate.
- **Stage 1 (≤ $250):** 30M, four arms, trunk + decay, evaluation 1–5. Gate: K1, K2.
- **Stage 2 (≤ $850):** 100M, four arms (two in parallel on separate instances where quota allows). Gate: S1–S3.
- **Stage 3 (≤ $700, only if the scale gate holds):** 300M, E33n and the dense ceiling (the window control at 300M only
  if budget is left).
- **Compute:** p5.4xlarge (1 × H100 80 GB, 16 vCPU, 3.8 TB local disk), us-east-1, on-demand $6.88/h; quota allows
  four at once. Data preparation on the same machine type only when the GPU is busy; otherwise on a CPU instance.
- **Budget:** $2,000 cap including storage; running total kept in the run log (`infra/aws/README.md`).
- **Launch:** `infra/aws/` scripts start an instance, clone this branch over HTTPS, `uv sync`, fetch data from S3,
  run the stage script, sync checkpoints and results to S3, and terminate the instance; the instance also terminates
  itself after a fixed number of hours.
- **New foundation code:** a top-k option for the closed note read (`lm_read_topk`), a memory-mapped token-shard
  dataset for the trainer (GPT-2 `.bin` shards), a LongDocs-lite builder, a position-bucketed long-document loss eval,
  a RULER runner on lm-eval for perceiver_ar, and the AWS scripts. All config-selectable and reusable.

## Result
- **Runs (2026-11-05, 30M, home servers):** dense `perceiver_ar_perceiver_H576L2g1s5N128_20261105_123824`, window
  `…_155729`, e33 `…_163042`, e33n `…_182321` (decay runs; trunks in the run report) · W&B
  https://wandb.ai/ksopyla/MrCogito · report
  [e33n_s30_lsa_testbed_20261105.md](../../2_Experiments_Registry/run_reports/e33n_s30_lsa_testbed_20261105.md) ·
  results `docs/2_Experiments_Registry/results/evaluations/E33n_s30/`.
- **Verdict: killed (K1).** At 16k, positions 2k–16k, G = 0.134 nats (95 % 0.099–0.176); the full read recovers 5.5 %
  (3.9–7.5) and the top-k read 6.1 % (4.7–7.8) of it. K2 not met, S2 met (top-k − full = −0.0008). S1 failed. Copy split:
  97 % of G sits on the 9.6 % of tokens whose 4-gram occurred earlier only beyond the 512-token window; the notebook
  closes ~2 % of that part — the far-context value of natural text at 30M is exact copying, which compressed notes
  cannot return. Stages 2–3 not run.

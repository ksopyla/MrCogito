# E39a — living concepts with every span competing (stages 2–3 again)

- **Status:** approved by the author 2026-11-26 ("Rerun stages 2–3"); trained on AWS 2026-11-26/27 ($148.32); card
  protocol 2026-11-27: **the flavour fails** (A1 missed at 16k and 64k, A2 missed at 32k; see "Verdict"). By the
  spec's own rule E39's line closes; awaiting the author's word. Three reported-only tests (board, samples, list
  exam) did not run: Polonez's `/home` disk froze at about 06:03 UTC.
- **Serves:** the 10M-context goal: whether E39's grouping keeps long text once the reader's choice no longer hides
  the right span, before any work on a choice that scales (the concept tree).
- **Implementation plan:** none of its own: one arm over E39's build ([E39_plan.md](E39_plan.md)); the arm
  `living_flat` in `scripts/run_e33n.py`, the job `infra/aws/jobs/e39a_flat.sh`.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-26 · closed —

## Hypothesis
E39 failed its bet (spec, "Verdict"): single lookups past 8k and long text. With every visible span competing at
test time (R2, same weights) the lookups came back (one number in an essay at 16k 0.48 → 1.00), but 64k documents
stayed 0.038 nats per token behind concept memory (2.331 against 2.293), because the model had trained behind the
narrow first level (32 concepts kept of about 210 at 16k and 880 at 64k). **Trained with every span competing in
stages 2 and 3, the grouping keeps long text as well as concept memory with the same recall recipe, and keeps the
single lookups.** If it does not, the grouping itself costs long text and E39's line closes.

## Builds on
- **Weights:** E39's stage-1 final (`runs/e39/e33n_m120t_living_muon0.01_tok8.0e9_seed42/trunk/…/final`). At 4k rows
  a reader read all of a document's concepts (about 23 against 32 kept), so the first level cost nothing in stage 1:
  E39's stage 1 tied concept memory (2.138 against 2.135).
- **Recipe:** E39's and the E38 recall run's stages 2–3 on `m21mix_v4` (recall rows at 3 % of stage 2, 5 % of
  stage 3), the same steps, optimizer (Muon 0.01), node type and tests.
- **Baselines:** the E38 recall run (concept memory, same stages 2–3: card protocol 64k documents 2.291, 16k 2.125,
  RULER 13-task 0.567 / 0.527 / 0.392 / 0.362) and E39 itself (2.388, 2.161, 0.436 / 0.387 / 0.269 / 0.201).
- **The one change:** `living_search = "flat"` in training and tests: a reader scores every visible span of its
  document and opens the best until 512 tokens are held; concepts still give one summary each for what is not
  opened and still come from the same grouping. Same parameters (105.1M non-embedding), same cost of the choice (both
  searches score every span; the two-level one masks spans outside the 32 kept concepts).

## Pass marks (set before the run; card protocol, `scripts/release_evals.sh`)
- **A1 (long text kept, the question):** 64k documents ≤ **2.305** and 16k documents ≤ **2.137** nats per token
  (the recall run's 2.291 and 2.125, concept memory's 2.293 and 2.126, + 0.5 %; E39 2.388 and 2.161).
- **A2 (single lookups kept):** one number in an essay (niah_single_2) ≥ 0.95 at 4k, 8k, 16k and 32k; one number among
  distractor keys (multikey 1) ≥ 0.80 at 16k (the recall run 1.00 / 1.00 / 1.00 / 0.92 and 0.79; E39 0.31 at 32k and
  0.55; E39 with the flat search at test time 1.00 and 0.84 at 16k).
- The flavour **succeeds** with A1 and A2. Reported, not marks: the RULER 13-task mean and the several-far-items cells
  against the recall run; our list exam; summaries switched off (R3); 128k documents; the 7 core tasks and BLiMP;
  the grouping's own numbers (R4).

## Stop marks
- **KA1:** held-out loss at step 1,000 of stage 2 above **2.227** (the recall run 2.177 + 0.05; E39 2.206) → stop the
  node: it learns worse than E39 did. The step-1,000 value against E39's is also reported.
- **KA2:** stage-2 speed below 59k tokens per second over its first 300 steps (0.85 × the recall run's 69.5k; E39
  67.5k) → report before going on.
- **KA3 (money):** $200 for this run's nodes (expected about $150: E39's stages 2–3 and tests took about 20 h on one
  spot g6e.12xlarge at about $7.3 an hour); the family cap of $5,000 stands ($3,578.44 spent on 2026-11-26 06:05 UTC).

## What follows
- A1 and A2 met → the grouping is worth a choice that scales: a first level whose kept count grows with the text, or
  the concept tree (R1 on E39's training numbers: concepts grow linearly, 23 → 210 → 881 at 4k / 16k / 64k). The flat
  search itself scores every span per reader and does not reach 10M tokens.
- A1 missed → the grouping's summaries and spans cost long text against concept memory's fixed blocks; close E39.

## Verdict (card protocol, 2026-11-27, Polonez card 1; against the recall run / E39)
- **A1 missed at both lengths.** 64k documents **2.337** nats per token (mark ≤ 2.305; 2.291 / 2.388), 16k documents
  **2.140** (mark ≤ 2.137; 2.125 / 2.161). About half of E39's gap closed (53 % at 64k, 58 % at 16k), not all of it.
- **A2 missed at 32k only.** One number in an essay 1.00 / 1.00 / 0.99 / **0.91** at 4k / 8k / 16k / 32k (mark ≥ 0.95
  at each; 1.00 / 1.00 / 1.00 / 0.92 and 0.98 / 0.98 / 0.48 / 0.31). The 32k bar was above the recall run's own 0.92,
  so on this cell the two tie. One number among distractor keys at 16k 0.89 (mark ≥ 0.80; 0.79 / 0.55): met.
- **Reported, not marks:**
  - RULER 13-task mean 0.514 / 0.491 / 0.399 / 0.331 (0.567 / 0.527 / 0.392 / 0.362; 0.436 / 0.387 / 0.269 / 0.201):
    a tie with the recall run at 16k, behind at 4k, 8k and 32k, above E39 everywhere.
  - Several far items: all values of one key (multivalue) 0.66 / 0.78 / 0.56 / 0.54, above the recall run at every
    length (0.59 / 0.43 / 0.35 / 0.36); several questions at once (multiquery) 0.61 / 0.62 / 0.30 / 0.21 (0.54 /
    0.60 / 0.36 / 0.16); two distractor keys (multikey 2) 0.67 / 0.46 / 0.30 / 0.22 (0.72 / 0.63 / 0.26 / 0.20);
    the hardest distractors (multikey 3) 0.08 / 0.02 / 0.00 / 0.02 (0.33 / 0.15 / 0.05 / 0.05).
  - Weak: a UUID in an essay (niah_single_3) 0.40 / 0.42 / 0.13 / 0.13 (0.97 / 0.88 / 0.46 / 0.53; E39 0.11 / 0.16 /
    0.06 / 0.02): copying a long exact string out of a span stays poor.
  - Use of far text: removing the memory costs 0.237 nats per token at 64k (recall run 0.285, E39 0.184); cutting
    the 64k text to its last 16k costs 0.034 (0.060 / 0.006). The model leans on the far text less than the recall
    run does.
  - Summaries switched off (R3): +0.059 at 64k (+0.067 / +0.051), +0.028 at 16k (+0.022 / +0.025), +0.072 at 128k
    (+0.030 / +0.068).
  - 128k documents 2.453 (2.461 / 2.487; 15 documents): a tie with the recall run.
  - Compare test (55 documents, bits per byte, lower is better): whole document 0.810 (0.794 / 0.827); saved from
    1k to most context 0.037 (0.051 / 0.020); another document's far past 0.840 (0.831 / 0.850).
  - 7 core tasks 45.79 (45.72 / 45.73): a tie.
  - Not run: the board (BLiMP and the leaderboard tier), samples and the list exam (Polonez's disk, see the log).
- **What it says.** Training with every span competing brings back the single lookups and closes about half of E39's
  long-text gap, but documents at 16k and 64k stay behind concept memory's fixed blocks. The spec's rule applies: the
  grouping's spans and summaries cost long text, so E39's line closes. One property is worth carrying over: finding
  every value of one key, which every other model fails, is better here at every length.

## Plan
- Seed `runs/e39a/e33n_m120t_living_flat_muon0.01_tok8.0e9_seed42/trunk/` with E39's stage-1 final from the laptop
  (admin profile), then launch the job on spot g6e.12xlarge (us-east-1, cheapest zones first) through the keeper
  (`infra/aws/keep_running.sh`, frozen copy `Cache/keepers/e39a-flat.sh`) with a money guard for KA3 and a waiter
  that reads KA1 and KA2 and stops the node on KA1 by itself.
- After the job's tests: the card protocol on Polonez (cards 0, 1, 3; card 2 holds a stuck process), the list exam
  and R3, against the recall run and E39.
- W&B: experiment E39a, "150M living concepts, every span competing with recall questions, Muon, stage 2 of 3".

## Log
- **2026-11-26 08:11 UTC launched.** E39's stage-1 final copied server side to the E39a folder (08:10 UTC, laptop
  admin profile); keeper (frozen copy `Cache/keepers/e39a-flat.sh`, at most 8 launches, 40 h per node) launched spot
  g6e.12xlarge `i-04c1102b95b220027` in us-east-1d at 08:12 UTC (1b and 1a had no capacity); money guard at $200;
  the waiter `Cache/ops/e39a_wait.sh` reads KA2 and KA1 from the stage-2 log and stops the node on KA1 by itself.
  Spend before the launch $3,578.44 of $5,000.
- **2026-11-26 09:14 UTC KA2 met:** stage-2 speed 71.4k tokens per second over its first 300 steps (mark 59k; the
  recall run 69.5k, E39 67.5k). **10:29 UTC KA1 met:** held-out loss at step 1,000 2.188 (stop above 2.227; the
  recall run 2.177, E39 2.206).
- **2026-11-27 03:18 UTC training and the job's own tests done.** Stage 2 08:14–20:05 UTC, stage 3 20:05–01:58 UTC,
  job tests to 03:18 UTC; the node powered itself off, the keeper exited at 03:23 UTC on the done marker, the guard
  was stopped by its PID. **KA3 met:** nodes $148.32 of $200.
- **The job's own tests** (not the card protocol; against the recall run / E39):
  - held-out loss at the end of stage 2 2.007 (1.996 / 2.017) and of stage 3, on 64k rows, 1.969 (1.938 / 2.003):
    half of E39's gap closed;
  - long documents, mean over sources, 16k 1.900 (1.890 / 1.910), 64k 1.908 (1.881 / 1.936), 128k 1.800 (1.775 /
    1.840);
  - RULER 13-task mean at 4k / 8k / 16k / 32k 0.510 / 0.489 / 0.403 / 0.338 (0.572 / 0.526 / 0.405 / 0.360 and
    0.436 / 0.384 / 0.266 / 0.194);
  - single lookups back: one number in an essay 1.00 / 1.00 / 0.99 / 0.95 (1.00 / 1.00 / 1.00 / 0.94), one number
    among distractor keys at 16k 0.88 (0.84);
  - all values of one key (multivalue) 0.66 / 0.77 / 0.50 / 0.58, above the recall run (0.59 / 0.41 / 0.35 / 0.38);
  - weak: a UUID in an essay (niah_single_3) 0.38 / 0.40 / 0.13 / 0.09 (0.97 / 0.91 / 0.44 / 0.59; E39 0.12 / 0.15
    / 0.02 / 0.02), the hardest distractor keys (multikey 3) 0.08 at 4k (0.34);
  - 7 core tasks 46.63 (46.28 / 46.52), WikiText 0.965 bits per byte (0.964 / 0.965).
  - Early read on the marks: at 64k the job test's gap to the recall run (+0.027) is about twice A1's margin
    (+0.014), so A1 at 64k is likely missed; 16k is close; A2 met on the job test. The card protocol decides.
- **2026-11-27 03:23 UTC card protocol started** on Polonez card 1 alone (cards 0 and 3 run another session's jobs,
  card 2 holds a stuck process), the list exam after it; waiter `Cache/ops/e39a_release_wait.sh`.
- **2026-11-27 03:23–06:23 UTC** ten of the twelve card jobs done (real documents 64k, 128k and 16k, RULER 4k to 32k,
  the compare test, 7 core tasks). **06:23 UTC** the board job started and froze. Polonez's `/home` (LVM volume
  `vg0-nvme` over both NVMe drives) has had one write stuck in the LVM layer since about 06:03 UTC: both drives show
  nothing in flight, the ext4 journal thread waits, and every process that writes to `/home` hangs (this run's board
  job and another session's job among them). The kernel log needs root. The waiter stopped at its 8 h limit (11:23
  UTC).
- **2026-11-27 11:31 UTC** results read without access-time writes (`O_NOATIME`) and copied to
  `Cache/results/e39a_release`. The queue's own processes on Polonez (the board job in a disk wait, the queue and the
  list-exam waiter) were left as they are: if the disk recovers, they finish the three remaining tests on card 1.
  Verdict above.

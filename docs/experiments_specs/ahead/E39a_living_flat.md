# E39a — living concepts with every span competing (stages 2–3 again)

- **Status:** approved by the author 2026-11-26 ("Rerun stages 2–3"); launched on AWS (see the log at the end).
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

## Plan
- Seed `runs/e39a/e33n_m120t_living_flat_muon0.01_tok8.0e9_seed42/trunk/` with E39's stage-1 final from the laptop
  (admin profile), then launch the job on spot g6e.12xlarge (us-east-1, cheapest zones first) through the keeper
  (`infra/aws/keep_running.sh`, frozen copy `Cache/keepers/e39a-flat.sh`) with a money guard for KA3 and a waiter
  that reads KA1 and KA2 and stops the node on KA1 by itself.
- After the job's tests: the card protocol on Polonez (cards 0, 1, 3; card 2 holds a stuck process), the list exam
  and R3, against the recall run and E39.
- W&B: experiment E39a, "150M living concepts, every span competing with recall questions, Muon, stage 2 of 3".

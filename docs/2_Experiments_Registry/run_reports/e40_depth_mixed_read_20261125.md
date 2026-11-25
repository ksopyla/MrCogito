# E40 — depth-mixed read (DeepCrossAttention as a retrofit of the 150M concept-memory model), 2026-11-24 → 25

Spec: [E40](../../experiments_specs/done_failed/E40_depth_mixed_read.md) · literature:
[depth-wise residual connections](../../literature_review/depth_wise_residual_connections.md) · saved results:
`docs/2_Experiments_Registry/results/evaluations/E40_*` · machine: Odra (3 × RTX 3090), $0 of AWS.

## Verdict

**Watch; K3 tripped.** Neither arm moved the model's long-document loss: the depth-mixed read (three GRN-v3 mixes
for the reading layer's query, keys and values, on all four loop passes) and Google's 2-DCA on every layer run both
land within ±0.005 nats per token of a paired control on whole-document loss and on the far-context gain at 64k. The
concepts do not carry more of the far past with keys and values drawn from other depths. No harm and no cost either:
the depth-mixed read keeps RULER, BLiMP and the core tasks level (S4) and runs at 0.96× the control's speed (S5).
The author's question ("are the concepts the main source?") is answered by the pre-flight: no — the concepts'
summaries alone carry 18 % of what the memory read is worth at 64k (24 % at 16k); the word-for-word reads carry the
rest, and the U-net skips amplify the read instead of routing around it.

## Pre-flight (released `concept-memory-140m-11.5bt`, Odra, 2026-11-24 18:01–18:49 UTC)

| Check | Result |
|---|---|
| P1 summaries only (nothing opened), 64k / 16k | loss 2.529 / 2.332 against 2.293 / 2.126 as trained; **share of the read's benefit carried by the summaries alone 18 % [15, 21] / 24 % [21, 27]**; worse than the last-1k cut (2.441 at 64k); the card's numbers reproduce exactly |
| P2 skip weights σ | layer 9 ← reading layer **+0.738**; layer 8 ← its looped partner −0.296; others \|σ\| < 0.08; reading layer α 0.53 |
| P3 identity at step 0 | exact (0 difference, fp32, 16k and 64k), both arms |
| P4 speed, one card, 16k | depth-mixed read **0.964×** the control (7,426 vs 7,704 tokens/s), 2-DCA 0.884× |

Two fixes before the run (18:21–18:49 UTC): the gate read the sum over 896 dimensions, so one Adam step moved it by
~0.7 and the mixed arms ran 0.01–0.025 above the control from step 10 (now the mean: level within 0.0005); the
uncompiled mixes ran at 0.85× / 0.67× (now one shared, compiled kernel per run).

## Runs (same rows, order, seed and schedule; 154M tokens each)

From the released weights: 1,430 steps of 4 × 16k rows (stage-2 mix) then 920 steps of one 64k row (stage-3 mix),
both rebuilt at home from the public sources (`scripts/e40_data_build.sh`, a subset of what the release run read).
Rates constant: Muon 1e-4, AdamW 6e-6 (the released stage 3's final rates); the mixes' own AdamW group at 1e-3 (a
deviation from the spec's 1e-4: Adam moves a weight at most its rate per step, 0.24 over the run at 1e-4).

| Arm | Card, UTC | Held-out loss, 16k part (steps 286 / 572 / 858 / 1,144 / end) |
|---|---|---|
| control | 0, 11-24 19:08 → 11-25 03:41 | 1.9275 / 1.9257 / 1.9241 / 1.9228 / 1.92146 |
| depth-mixed read | 1, 11-24 18:50 → 11-25 03:35 | 1.9278 / 1.9262 / 1.9244 / 1.9231 / 1.92148 |
| 2-DCA everywhere | 2, 11-25 03:41 → 13:03 | 1.9285 / 1.9262 / 1.9246 / 1.9237 / 1.92132 |

K2 (halfway, 21:52 UTC): +0.0003 (stop above +0.02), passed. Two cards at a time: next to a busy card 1, card 2
climbed from 69 to 81 °C within five seconds of work, so the first three-card start was stopped at 19:07 UTC and the
control restarted (`scripts/heat_guard.sh` kept every card under 80 °C by pausing our own jobs).

The mixes stayed near the identity: per-column means of b − 1 within ±0.035, single weights within 0.13 (the rate
allowed ~2), gates ≤ ~0.15; the one steady change is the values' mix taking ~2 % less of the embedding column.

## Release tests (`scripts/release_evals.sh`, modes off / summaries off / summaries only)

The retrofit itself (control against the released model): 64k whole document +0.008 worse, far-context gain −0.006;
RULER 13-task mean 0.478 / 0.431 / 0.374 / 0.336 at 4k–32k (card 0.488 / 0.439 / 0.373 / 0.319); BLiMP 80.18
(80.56); core 7 46.52 (46.22). Every arm is judged against this retrained control.

Arm − control, paired by document (nats per token, 95 % interval):

| Measure | depth-mixed read | 2-DCA everywhere |
|---|---|---|
| 64k whole document (55 docs) | −0.0005 [−0.0017, +0.0006] | −0.0001 [−0.0013, +0.0010] |
| 64k far-context gain (1k cut − whole) | +0.0013 [+0.0003, +0.0025] | +0.0024 [+0.0013, +0.0034] |
| 64k cost of summaries off | −0.0005 [−0.0029, +0.0020] | −0.0027 [−0.0052, +0.0002] |
| 64k summaries only | −0.0016 [−0.0024, −0.0008] | +0.0006 [−0.0003, +0.0015] |
| 16k whole document (120 docs) | +0.0008 [+0.0004, +0.0012] | +0.0028 [+0.0022, +0.0034] |
| 128k whole document (15 docs) | −0.0080 [−0.0125, −0.0037] | −0.0053 [−0.0092, −0.0016] |
| RULER mean 4k / 8k / 16k / 32k (control 0.478 / 0.431 / 0.374 / 0.336) | 0.490 / 0.463 / 0.385 / 0.335 | 0.503 / 0.470 / 0.388 / 0.331 |
| BLiMP / core 7 (control 80.18 / 46.52) | 80.17 / 46.44 | 80.23 / 46.44 |

Both arms' RULER leads have the same shape: the 8k lead is mostly one cell, the UUID needle (0.16 → 0.42 / 0.39), an
answer-start effect seen before (E38); several keys asked at once (multi-query) rises in both at every length to 16k
(+0.03 / +0.10 / +0.06 for the depth-mixed read, +0.08 / +0.10 / +0.12 for 2-DCA at 4k / 8k / 16k), about two
standard errors each at 100 prompts per task; the rest are within about one.

## Marks

| Mark | Result |
|---|---|
| S1 far-context gain at 64k ≥ control + 0.010 | **missed** (+0.0013) |
| S2 summaries matter more | **missed** (summaries-off cost −0.0005; summaries only 0.0016 better) |
| S3 2-DCA whole document ≥ 0.010 better at 16k and 64k | **missed** (64k −0.0001, 16k +0.0028 worse) |
| S4 no harm | depth-mixed read **met** (RULER +0.018); 2-DCA: BLiMP and core 7 level, RULER **+0.026**, outside the ±0.02 band on the good side (no harm) |
| S5 depth-mixed read ≥ 0.95× speed | **met** (0.964×) |
| K1 identity | not tripped (exact) |
| K2 halfway | not tripped (+0.0003) |
| K3 both arms within ±0.005 (whole and gain, 64k) | **tripped**: record as Watch, no from-scratch run |

## What it means

- At width 896, 13 layers and 19 layer runs, a trained concept-memory model gains nothing from depth mixes as a
  retrofit of 154M tokens; the paper's own retrofit (−0.19 perplexity) used 60k steps on 128-token text. The mixes
  barely left the identity, so the trained wiring is already near what the mixes can express: P2's skip weights
  showed the same depth pattern DCA learns.
- The bet that keys and values from early depths would make the concepts carry more is not supported: the summaries'
  share of the read moved from 16.8 % to 17.6 % at 64k.
- Open hints, no marks, the same in both mixed arms: past the training length (128k, 15 documents) they are 0.008 /
  0.005 better and use their memory a little more; on RULER they find more answers when several keys are asked at
  once (multi-query +0.06 / +0.12 at 16k). Retrieval with several targets is the skill the E38 recall rows fixed
  with data; a from-scratch run with the recall mix could test whether depth mixes add to it, but the document
  losses give no reason to spend on it now.

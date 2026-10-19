# RL post-training as a capability probe: pros, cons, novelty (2026-10-10)

Question (author, 2026-10-10): should the capability protocol gain an RL stage (GRPO or similar) after from-scratch
supervised training, e.g. on one of the top reasoning tasks? Judge efficiency, novelty and research interest.
Sources: research-scout scan (abstracts fetched; 2026 IDs recent, not replicated), our capability results
(`e33_second_hop_diagnosis_20261006.md` §5–8), the probe (`verification/bapo_capability_probe.py`), the text checks
(`evaluation/text_checks.py`, spec `docs/engineering_specs/text_capability_checks.md`).

## 1. Where we stand (the state RL would start from)

- All three models learn the atomic skill: 1-hop edge lookup 97–100 % (E33, E31, dense; seeds 0–2).
- Composition without a scratchpad (C5.pchain2, final answer only): E33 94 / 94 / 4 %, E31 7 / 9 / 9 %, dense
  15 / 23 / 9 % (floor 8.3 %). The non-passing runs sit on one plateau: teacher-forced 82–86 %, i.e. "answer with some
  candidate" — the right shape, the wrong chain.
- With the path written (C5.path2, recipe B) every model composes: E31 94, E33 94, dense 90.
- Saved weights exist for every run (Odra, `Cache/study/e33_recipe_a`, `e33_reasoning_fair`, `e33_v4_path2*`).

So we hold exactly the precondition the composition-by-RL paper names (atomic skills present, composition absent),
in tiny from-scratch models, with three architectures and a stuck E33 seed as a bonus.

## 2. What the literature says (verified abstracts)

| claim | evidence | scale | relevance |
|---|---|---|---|
| RL sharpens what the base can already sample; base wins at large k | Yue 2504.13837 (NeurIPS 25 oral) | PT ≥ 1.5B | caps expectations |
| Prolonged RL can go beyond the base | ProRL 2505.24864 | PT 1.5B | counter-evidence |
| RL adds capability only at the "edge of competence" (pass@1 fails, pass@128 > 0); fails with ~0 exposure, works from ≥ 1 % | Interplay 2512.07783 (ICML 26 spotlight) | **scratch, 100M**, synthetic DAG math | closest controlled study; our models are 3× smaller |
| RL collapses onto one pretraining format within an epoch | Echo Chamber 2504.07912 (COLM 25) | scratch, 150M–1B | a scratchpad must be in pretraining |
| RL composes atomic skills: f(g(x)) 64 % at level 2 vs 15 % for iterative rejection fine-tuning; fails if atomics missing | 2509.25123 (ICLR 26) | PT 8B | **our exact precondition** |
| Parity: next-token stays at chance, GRPO/REINFORCE after pretraining → ~100 % with CoT growing 1 → 50 tokens; needs ≥ ~1/3 long demonstrations in pretraining | Tsilivis 2510.11495 (ICLR 26) | **scratch, ~0.4–6M** | the only tiny from-scratch RL-reasoning result |
| Graph path-finding: SFT memorises co-occurrence, RL generalises via exploration; policy gradient suffers diversity collapse | 2509.22613 (ICLR 26) | 1-layer toy | multi-hop analogue |
| Outcome-reward PG learns traversal that extends to longer chains only with enough short instances | 2601.15158 | 1-layer theory | supports a hop curriculum (we have one) |
| GRPO rank bias reinforces already-likely answers | 2506.02355 | PT | danger with a 12-candidate answer set |
| RL forgets less than SFT (KL-minimal) | RL's Razor 2509.04259 | PT | no-harm on the ladder |
| RL updates 5–30 % of parameters, off principal directions | 2505.11711, 2511.08567 | PT | dynamics we could test at 30M / weight-tied |
| Looped LM: GRPO/DAPO after SFT gave no significant gain; exit gate is not RL-trained | Ouro 2510.25741 | 1.4–2.6B | loop + RL is unproven |
| Reward spread over the latent loop trajectory beats GRPO by 5.8–10.9 pts | RLTT 2602.10520 (ICML 26) | Ouro-1.4/2.6B | loop-aware credit assignment helps |
| Latent CoT + plain GRPO: no gain (rollouts deterministic); sampling noise needed | 2512.11816, 2606.10184, HRPO 2505.18454 | PT | E33's loop is deterministic; only the answer tokens are sampled |
| Small models often fail RL (TinyZero: 0.5B "fails to learn reasoning") | TinyZero README, 2503.16219, 2503.18892 | PT 0.5–1.5B | fragility warning |

Gaps no paper fills (scout + my reading):
1. RL on from-scratch models of 5–50M on multi-hop with a controlled guessing floor.
2. Architecture × RL: does RL help a dense, a latent-memory or a looped model more? Untested.
3. RL when the base sits at the guess floor of a small candidate set (any candidate wins by chance).
4. RL for loop count / halting in a small from-scratch looped transformer.

## 3. The mechanism, honestly (why plain RL on our exam may do nothing)

Our exams train the final answer with the gold label (supervised). An RL step on the same answer channel uses the
model's own samples: the gradient on a correct sample is the supervised gradient on that answer scaled by its
advantage; wrong samples are pushed down. With no tokens the model is free to choose (no scratchpad, fixed loop
count), RL sees **less** information per row than supervised training already gives (the gold answer for every row,
sampled or not). The only new ingredient is the negative push on wrong candidates. Prediction for "RL on the stuck
checkpoints, answer only": little or no gain (consistent with Yue, Interplay's zero-exposure failure, Ouro).

Second trap: pass@k is meaningless here. With 12 candidates a random guesser reaches pass@64 ≈ 99.6 %, so pass@k
cannot show hidden competence. Use instead: the probability mass on the right candidate vs the mean of the other
candidates (above 1/12 = latent competence exists), measured before RL. And GRPO's rank bias will happily reinforce
lucky guesses of the commonest candidate.

RL becomes interesting only where it gives the model a freedom supervised training does not:
- **free tokens** (a scratchpad it writes without labels; reward only on the final node): can it discover writing the
  bridge node, turning recipe A into recipe B on its own?
- **free compute** (E33: how many loop rounds; reward = correct − cost × rounds);
- **robustness** (RL generalises / forgets less): longer documents, 3 hops, unseen names.

## 4. Pros and cons

**For:**
- Scientifically open: all four gaps above are ours to fill; the setting (tiny, from scratch, three architectures,
  frozen exams with floors, dense ceiling, length ladder to 128k) is unusually well controlled.
- Cheap: a run is hours on one 3090; many seeds are affordable, so RL dynamics (jumps, entropy, update sparsity)
  can be measured with error bars — a microscope LLM-scale papers cannot afford.
- Directly tests the f(g(x)) and edge-of-competence claims in a regime where the "atomic skill" is measured, not
  assumed.
- Tells us what the loop is worth: if RL lets dense/E31 discover an explicit scratchpad and reach ~90 %, E33's latent
  composition is a compute/latency advantage (4 rounds vs 8+ written tokens), not a capability monopoly. If they
  cannot, the loop's advantage is deeper. Either result shapes the architecture story.
- Text later: T5 Compose (floor 1/8) has an exact, verifiable answer; the world-document generator makes unlimited
  fresh items.

**Against:**
- Plain answer-only RL is predicted to do little (§3); the interesting variants need new exam layouts (free
  scratchpad slots) or a stochastic exit policy — real engineering, not a flag.
- Fairness: an RL stage is a second recipe with its own knobs (group size, temperature, KL, steps); it must be one
  recipe for all models or it becomes a tuning contest. Keep it out of the gating protocol.
- Small-model RL is fragile (entropy collapse, diversity collapse in path-finding, TinyZero 0.5B failing).
- Sampling cost: our models have no KV cache, so each sampled letter re-runs the full forward. Fine for 8–24 letters
  at 1k tokens (≈ 0.01 s/row forward), heavier for text at 4k.
- Interpretation risk: an RL gain on the training distribution can be sharpening; only held-out splits (3 hops,
  longer books, new names, evidence-removed control) count as capability.
- Compute is busy: Odra runs the text round now; Polonez only in < 10 h bursts.

## 5. Verdict and the bet

**Adapt, as a separate DNA-level track ("R"), not inside the gating protocol yet.** Port to text T5 only after a DNA
signal.

The coherent bet: *in tiny from-scratch models, RL adds composition only where it gives the model a new degree of
freedom (free tokens or free compute); and which architecture benefits depends on whether composition can stay
latent.*

Three steps, cheapest first, each with a kill signal:

| step | what | prediction | kill / pass |
|---|---|---|---|
| R0 diagnostic (no training) | on the 9 saved recipe-A checkpoints: probability on the right candidate vs others; temperature samples | dense/E31 ≈ 1/12 (no hidden competence); E33 seed 2 ≈ 1/12 | if dense/E31 already put > 2× chance on the right node, plain RL (R1) becomes promising |
| R1 plain GRPO on the answer | from the stuck checkpoints (dense, E31, E33 s2), recipe A, same RL recipe for all, seeds 0–2 | no gain (literature); a positive would be surprising | pass: held-out greedy ≥ 50 % on 2/3 seeds; kill: < floor + 10 after the budget |
| R2 free scratchpad (the main bet) | pretraining mix of recipe A and written-path rows (1/3, per Tsilivis; plus a 5 % arm, per Interplay's ≥ 1 % exposure); RL on recipe A with free unlabelled think slots, reward on the final node | dense/E31 learn to write the bridge and approach their path2 score (~90 %); E33 may keep it latent (shorter scratchpads) | pass: recipe A greedy ≥ 75 % (median of 3) on held-out names; kill: no model above floor + 10 and the scratchpad never holds the bridge |
| R3 learned halting for E33 (later) | stochastic exit per round, reward = correct − λ·rounds | E33 learns 2 rounds for 2 hops, 3 for 3 | novel either way; run after R2 |

Diagnostics on every RL run: greedy score with floor, evidence-removed control, length ladder (does RL change 128k
retention?), no-harm on 1-hop lookup (RL's Razor), answer entropy, fraction of weights changed, scratchpad content
(does it contain node 1?), jump step.

Cost: R0 ≈ 1 GPU-hour; R1 ≈ 9 runs × 2–3 h; R2 ≈ 3 models × 3 seeds × (pretrain 6 h + RL 3 h) ≈ 80 GPU-hours.
Engineering: a GRPO stage in the probe (`--rl_*`, sampled decode reusing `_answer_exact`, reward from `_candidates`,
group-normalised advantage, optional KL to the start weights), plus free think slots in `data/symbolic_tasks.py`
for R2. Needs a spec (`experiment-design`) before code.

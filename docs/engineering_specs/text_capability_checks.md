# Text capability checks — trainability and reasoning on simple language (protocol)

- **Type:** engineering foundation (training + evaluation protocol). Not an `E0NN` experiment.
- **Status:** **first draft 2026-10-05**, to be calibrated after the first server runs (token budget,
  compute caps, mix shares and task dials are expected to move). Data generator, builder, scorer
  (2026-10-06), the runner, scorecard, recipe cards and the `text-checks` skill (2026-10-06, §16) are
  implemented and smoke-tested locally end to end; first server runs are calibration. Version string
  once frozen: `text-v1`; the current draft data is `text-world-v0`.
- **Owner:** Krzysztof Sopyla
- **Relation to the DNA checks:** the [capability checks](capability_checks.md) (C0–C7, random-letter
  books, one small model per exam) stay the **first** gate and are not replaced. These text checks are
  the **next step**: one from-scratch language model per architecture, trained on a fixed simple-language
  corpus with a fact layer, then examined on a fixed ladder of questions at lengths 1k → 128k. They fill
  the "C8 real text" slot of the DNA checks with their own protocol, package and ledger.
- **Skill:** `text-checks` (`.cursor/skills/text-checks/SKILL.md`): the step-by-step for agents.

## 1. What question this answers

> *Trained from scratch on the same data, with the same compute and the same parameter budget, can
> this architecture learn language and learn to look up, track and combine facts written in that
> language — how fast does it learn it, and does it keep working when the text is far longer than
> anything it was trained on?*

The DNA checks answer "can this mechanism carry and address information at all" on exams with an exact
information content. They cannot answer three things this protocol targets:

1. **Learning alongside language.** In text the model must learn fluent language *and* fact use from one
   next-token objective; the memory must earn its place against everything else the loss rewards (the
   E22 failure: the notebook was ignored on natural text).
2. **Trainability.** How many tokens until each skill appears, how stable the run is, what it costs per
   token — compared at an identical budget.
3. **Train short, read long.** Every model trains at 4k tokens and is read at 1k … 128k.

## 2. Rules carried over from the DNA checks (lessons already paid for)

| rule | from |
|---|---|
| Everything from random init; no checkpoint of another run, no pretrained weights or embeddings. | checks v4 rule 1 |
| A dense model trains on the same data and budget: it is the ceiling at the training length. A task the dense model cannot pass at 4k is **uncalibrated** and gates nothing. | checks rule 5; E22 gate lesson |
| Every task states its **guessing floor** (best score without doing the task), and every score is printed beside it. | E33a diagnosis 2026-10-05 |
| Exams with several same-shaped candidates also report the **picked candidate**; the primary score is the **exact answer**. | checks rule 4; E31 recall16 (right fact 88 %, exact copy 24 %) |
| A task with a shortcut is **flawed**: kept with its reason, never evidence. Each task has a shortcut audit (§10.3). | shuffled-chain / pchain-v1 flaws |
| Length is a separate dial from the skill; reading at longer lengths is evaluation, not training. | checks rule 3 |
| Results go to a committed ledger, raw folders to the NAS; W&B holds the training curves. | storage contract |

## 3. Prior art (how others did it, what we take)

| work | what it does | what we take |
|---|---|---|
| bAbI (Weston et al. 2015), **BABILong** (Kuratov et al. 2024, arXiv 2406.10149) | simple fact stories (who moved where, who holds what) hidden in PG19 filler, 0k → 10M tokens; generator open | facts inside filler, length as a dial, "latest location" and counting tasks; we swap PG19 for simple stories so a <100M model can learn the language |
| **Physics of Language Models 3.1** (Allen-Zhu & Li, arXiv 2309.14316) | synthetic biographies; fact extraction only works with several phrasings per fact | ≥ 4 paraphrase templates per relation in training, held-out phrasings as an out-of-distribution split |
| Physics of LM 2.1 (iGSM) | fresh generated reasoning data each step, depth dial | unlimited generated fact documents; hop depth as a dial with a held-out deeper setting |
| Grokked Transformers (arXiv 2405.15071) | 2-hop composition is late and fragile | 3-hop held out, reported not gated, until calibrated |
| ProntoQA / FLD / ProofWriter | rule deduction with made-up words, depth 1–5 | the frontier deduction level (T7) |
| MQAR / Zoology (Arora et al. 2023) | recall difficulty grows with the number of keys, not only length | number of entities is its own dial (T3) |
| RULER (Hsieh et al. 2024), lost-in-the-middle | variable tracking, aggregation; report by position | depth bins for every item |
| **TinyStories** (Eldan & Li 2023), **SimpleStories** (2025, 4k-token tokenizer) | fluent text with a ~1.5–4k word vocabulary; models of 1–35M are fluent | the language substrate and the filler |

Gap: no published small-vocabulary testbed combines fluent language, a fact layer, multi-hop questions
and 1k → 128k lengths. This protocol builds one. The author's
[cogito-mill](https://github.com/meridian21lab/cogito-mill) (solver-checked mystery stories, tuned so a
strong LLM scores ≤ 30 %) is too hard and too slow to generate for training a <150M model; it becomes a
**held-out hard exam** later, outside the gating ladder.

## 4. The setup in one screen

```
 CORPUS (identical token stream for every architecture)
   ~45 %  simple stories (TinyStories + SimpleStories train splits)        → language
   ~55 %  "world documents": stories about a small invented world, with   → fact use
          fact sentences woven in and 1–4 questions + answers at the end
 TOKENIZER  one frozen 4,096-token BPE trained on the corpus (§8.1)

 LEVELS (what the model must do)                    × LENGTH                 × SIZE
 T0 Language    held-out story loss                  train at 4k;             screen 30M,
 T1 Quote       copy a sentence seen earlier         read at 1k, 2k, 4k, 8k,  main 100M
 T2 Lookup      one fact by its subject              16k, 32k, 64k, 128k
 T3 Keyed       1 fact among N similar ones (+ look-alike names)
 T4 Latest      where is X now, after several moves
 T5 Compose     2-hop (3-hop held out)
 T6 Count       how many times …                     (frontier)
 T7 Deduce      rule chains with made-up words       (frontier)
```

## 5. Budgets (identical for every architecture)

| | **screen** | **main** | optional **large** |
|---|---|---|---|
| parameters (all trainable, incl. embeddings, head, memory) | 30M ± 5 % | 100M ± 5 % | 150M ± 5 % (hard max) |
| training tokens (data budget) | 0.6B | 2.0B | 3.0B |
| training length | 4,096 | 4,096 | 4,096 |
| tokens per optimizer step | 262,144 (64 rows × 4,096) | 262,144 | 524,288 |
| data order | the same pre-tokenized shards in the same order for every run | same | same |
| compute cap (active training time, Polonez 4 × 3090) | 6 h | 30 h (= 120 GPU-hours) | 48 h |
| expected time for the dense model | ~1.5–2 h | ~15–20 h | ~30–40 h |
| tuning budget (§7) | 4 runs × 60M tokens | 3 runs × 100M tokens | as main |
| seeds | 1 (references: 3, once per version) | 1 | 1 |

- **Expected times** extrapolate the E18 pilot (125M, 8k, 128k-token vocabulary: 28k tokens/s on
  4 × 3090); a 4k vocabulary and 4k context are cheaper. The calibration pilot (§16) measures them and
  the caps are fixed then.
- **Compute fairness.** Data budget is the primary equaliser: every run sees exactly the same tokens.
  The cap is the compute equaliser: a run that has not consumed its tokens when its active-time cap is
  reached is stopped and **scored at that checkpoint**, labeled `over-budget`. Slowness is a
  trainability result, not a free pass. Measured FLOPs per token are reported for every run.
- **Hardware.** Polonez (4 GPUs) or Odra (3 GPUs). The global batch is fixed via gradient accumulation;
  the cap is in GPU-hours, so Odra gets 4/3 of the wall-clock. Polonez runs in bursts under 10 h with
  cooldowns: every run must be resumable, and the cap counts active training time only.
- **Parameter count** is printed by the builder and checked against the band before any GPU time; an
  architecture that cannot fit says so and is not run at that tier.

## 6. Which architectures, and in what role

Round 1 (the references every later variant is compared with):

| model | role |
|---|---|
| **dense** (Llama-style decoder: RoPE, SwiGLU, RMSNorm, grouped KV heads) | the ceiling at 4k; the language-quality reference |
| **local** (the same decoder, every layer sees only the last 512 tokens) | the "no long memory" control: what is reachable without any long-range read |
| **E31c** (latent notebook, closed-window read) | the current memory champion's text version |
| **E31c, notebook switched off at evaluation** | evaluation-only twin of E31c: how much its answers depend on the notebook |
| **E31c + reread loop (E33a)** | the reasoning variant, once E33a's DNA results justify it |

Later variants enter the same way. **Prerequisite:** an architecture enters the text checks only after
passing the DNA **screen** package (C0–C2 at 10M). A text day is too expensive to discover that a
mechanism cannot carry a fact at all.

## 7. Per-architecture recipe card (each architecture gets its own best settings)

Architectures differ in what initialization and step size suit them; forcing one recipe on all would
measure the recipe, not the architecture. So each architecture brings a **recipe card**, chosen by a
fixed procedure with the **same tuning budget for everyone**, and frozen before its scored run.

**Card fields** (one file per architecture and tier, committed before the scored run):

- initialization: scheme per parameter group, with the reason or citation (e.g. normal σ 0.02; output
  projections scaled by 1/√(2·layers); zero-init of new gates or memory read-outs so training starts as
  the plain model);
- optimizer and settings: AdamW by default (β, ε, weight decay, gradient clip); another optimizer is
  allowed if declared;
- learning-rate schedule: warmup, peak, shape (warmup–stable–decay or cosine), ending exactly at the
  token budget;
- precision (bf16), attention kernel;
- architecture knobs (window, notebook geometry, loop rounds …), **frozen from the DNA checks** — not
  tuned on text unless the tuning runs below are spent on them;
- length-reading method for evaluation beyond 4k (e.g. none, RoPE base scaling), declared before the
  scored run and never changed after seeing long-length scores.

**Tuning procedure** (identical counts for every architecture):

1. *Screen tier:* 4 runs, peak step size on a ×2 grid around the card's prior, 60M tokens each, seed 0.
   Up to 1 of the 4 may instead test an alternative initialization.
2. *Main tier:* 3 runs at {½, 1, 2} × the transferred step size, 100M tokens each.
3. **Selection is on the development loss of the training mix** (held-out stories + held-out world
   documents, answers included). **Never on ladder scores**: the ladder is the test set.
4. Tuning tokens and GPU-hours are recorded on the card; they are identical in count for every
   architecture by construction.

## 8. Data

### 8.1 Tokenizer
One BPE with 4,096 tokens, trained once on a sample of the corpus (stories + world documents), frozen
and hashed. Name and place syllables are single tokens, so invented names are 2–3 tokens and
look-alike names differ by exactly one token. Every architecture uses this tokenizer.

### 8.2 Language part
TinyStories and SimpleStories train splits, deduplicated. Their validation splits are reserved for the
development set and for **evaluation filler**: no evaluation document contains a filler story seen in
training.

### 8.3 World documents (the generator)
A pure function of `(version, split, task, length, index)` → one document with its questions, answers,
candidate sets, evidence spans and floors. Same inputs, same bytes, on any machine.

**The world.** Each document samples a small cast (2–64 people) with invented names from a syllable
grammar, and gives each person relations from closed sets:

| relation | values | example sentence (one of ≥ 4 templates) |
|---|---|---|
| lives in | invented places (48) | "Lumo lived in Redhill." / "Redhill was where Lumo made his home." |
| pet + pet name | 16 common animals + invented names | "Lumo had a fox called Pim." |
| sibling, friend | another cast member | "Tavi was Lumo's sister." |
| job | 24 common jobs | "Mira worked as a baker." |
| sign / note text | 4–6 common words in a random order | "The sign on Lumo's door said: blue frogs sing at noon." |
| moved to (state) | invented places, ordered events | "Later that spring, Lumo moved to Ashford." |
| visited (events) | places, countable | "Lumo visited the market again." |

**The filler.** Real stories from the language part. The protagonist of a filler story is renamed to a
cast member, so names appear everywhere and finding the name alone is not enough; the relation must be
read.

**Answer hygiene (validator, unit-tested).**
- No filler sentence may state a reserved relation about a cast member (pattern check after renaming;
  such sentences are dropped).
- Answers to lives-in, moved-to and pet-name questions are invented words, which never occur in filler.
- The generator computes every answer from its own world state and proves it unique: exactly one value
  satisfies the question.

**Questions.** At the end of the document, after all evidence:

```
Question: Where does Lumo's sister live?
Answer: Ashford.
```

The answer ends with a period, which is part of the scored answer.

### 8.4 Training mix
- Language part ~45 %; world documents ~55 %.
- World documents in training use log-uniform lengths 256–4,096 and only in-distribution dials (§9):
  T1–T7 task types, cast 2–16, hops 1–2, moves 1–4, deduction depth 1–2.
- Plain next-token loss on every token, answers included, with no special weighting, identical for
  every architecture.
- The shares are frozen in `text-v1` after the calibration pilot (§16).

### 8.5 Splits

| split | what differs from training | role |
|---|---|---|
| **dev** | new seeds, same settings | recipe selection, loss curves |
| **in-distribution (ID)** | new seeds, **held-out names**, held-out filler stories | the gating scores |
| **length** | ID settings at 1k … 128k (more filler, the same facts) | length reading |
| **harder** | cast 64, 3 hops, 8 moves, deduction depth 3 | generalization, reported |
| **paraphrase** | 2 held-out sentence templates per relation | reported, not gating |

## 9. Levels and tasks

All evaluation items place their evidence in one of three **depth bins** (first 10 %, middle 10 %, last
10 % of the document before the question), balanced. Floors are computed by the generator per item and
averaged.

| level | task (eval, ID settings) | answer | picked-candidate floor | dials (train → harder split) | maps to DNA level |
|---|---|---|---|---|---|
| **T0 Language** | loss on held-out stories; loss on world documents excluding answers | — | — | — | — |
| **T1 Quote** | "What did the sign on Lumo's door say?", 4 signs in the document | 4–6 words | 1/4 | signs 2–8 → 16 | C0 Carry |
| **T2 Lookup** | "Where does Lumo live?", cast 4, only Lumo's home is stated | 1 place | — (one place in the document: exact answer only) | — | C1 Address |
| **T3 Keyed** | the same question, cast 16, every person's home stated, a quarter of names are look-alikes | 1 place | 1/16 | cast 4–16 → 64 | C2 + C3 |
| **T4 Latest** | "Where does Lumo live now?" after 4 moves; other people move after Lumo's last move | 1 place | 1/5 (any of Lumo's places); the most-recently-mentioned place is wrong by construction | moves 1–4 → 8 | new |
| **T5 Compose** | "Where does Lumo's sister live?", cast 8, every person has a sibling and a home (same-shaped decoy chains) | 1 place | 1/8 | hops 1–2 → 3 | C4 + C5 |
| **T6 Count** | "How many times did Lumo visit the market?", answers 0–6 balanced, look-alike names also visit | 1 number word | 1/7 | events 4–12 → 24 | C6 |
| **T7 Deduce** | made-up category rules, "Is Pim shiny?", balanced yes/no, distractor rules | yes / no | 1/2 | depth 1–2 → 3 | C5 (rules) |

**Packages:**

| package | levels | lengths | use |
|---|---|---|---|
| **text-core** | T0–T5 | ID at 1k … 128k | every architecture that enters the text checks |
| **text-frontier** | T6, T7, harder and paraphrase splits | 4k, 16k, 64k | the current research question; joins core once calibrated |

## 10. Scoring (one forward pass per item, fully deterministic)

### 10.1 Exact answer (primary)
Teacher-forced over the gold answer plus the closing period: the item is correct when the model's top
token is the gold token at **every** answer position. This is exactly the condition under which greedy
decoding writes the gold answer, so it equals generation-based exact match, but needs one forward pass,
no sampling and no string normalisation.

### 10.2 Picked candidate (secondary)
Each item carries its candidate set (all values of the asked type planted in the document). Candidate
sets are built so their **first tokens are distinct**. The pick is the candidate whose first token gets
the highest probability at the answer position. This uses the same forward pass. It shows "found the
right fact" separately from "copied it exactly".

### 10.3 Controls on every model (evaluation only)
- **Evidence removed.** A twin of each item with the evidence sentences replaced by filler of the same
  length. The score must fall to the guessing floor (within the larger of 10 points and 2 standard
  errors).
  - If it does not, that model's cell is marked `shortcut` and is not evidence.
  - If the dense model shows it, the task is `flawed` (§13).
  - Run at 4k and at the longest length the model passes.
- **Notebook off** (architectures with a switchable memory): the same items with the memory channel
  removed. The difference is the memory's contribution.
- **Floors printed.** Every score is printed as `score (floor)`.

### 10.4 Pass
A (task, length) cell **passes** when exact-answer accuracy ≥ 75 % and the evidence-removed control is at
the floor. Items per cell: 400 up to 16k, 200 at 32k–128k (standard error ≈ 2.2 and 3.1 points at 75 %).

## 11. Length: train short, read long
- Training length is 4,096 for every model. Evaluation lengths are 1k, 2k, **4k**, 8k, 16k, 32k, 64k
  and 128k.
- Longer documents add filler, not facts: the cast and the evidence are the same as at 4k, so length is
  the only change. T3's cast dial covers "more facts".
- **Reach** of a model on a task = the longest length at which the cell still passes.
- **Retention** = score at length L ÷ score at 4k, reported at 16k and 128k.
- The length-reading method is the one on the recipe card; it is never changed after seeing scores.
- **Optional long-train track** (declared per round, identical for all): the same budget with the last
  10 % of tokens at 16k. It is a different track, never mixed with the 4k track in one comparison.

## 12. Trainability metrics
- **Learning curve:** ID 4k scores (200 items per task) at 10, 25, 50, 75 and 100 % of the tokens.
- **Tokens to pass:** the first checkpoint at which each task passes. The headline trainability number.
- **Development loss** at the same checkpoints.
- **Cost:** measured tokens per second at the training length, FLOPs per token, peak memory, and
  evaluation time per item at 128k.
- **Stability:** loss spikes (more than 3 × the running standard deviation), any non-finite step. One
  restart from the last checkpoint with an unchanged recipe is allowed and recorded. A second failure
  marks the run `unstable`.

## 13. Verdicts and comparisons
- **Calibration of a task:** a task gates only if the dense model passes it at 4k (ID). Otherwise it is
  `uncalibrated`: reported, gating nothing, and a candidate's pass there is a reported win.
- **Text frontier** of a model = the highest level T1–T5 such that every level up to it passes at 4k.
- **Reach profile:** per level, the longest passing length. Memory architectures are expected to beat
  dense here; matching dense at 4k is enough there.
- **Language no-harm:** held-out story loss within 2 % of the dense model at the same tier. A memory
  that buys recall by losing language is reported as such.
- **Comparisons between models:**
  - use the same version, tier and track;
  - differences under 2 standard errors (or under the reference seed spread, whichever is larger) are
    ties;
  - a claim resting on a smaller margin needs a second seed.
- **Text champion:** chosen after round 1 by the author. From then on, the DNA no-harm rule applies
  here too: more than 5 points below the champion on a cell where the champion passes is a lost
  capability.
- **Flawed:** a task whose evidence-removed control stays above the floor for the dense model, or where
  another shortcut is found. It stays defined with its reason and is never evidence.

## 14. Running a check: the steps an agent follows

```
- [ ] 0. Preconditions: the architecture passed the DNA screen; it builds from config on the shared
         trainer (no fork); parameter count within the tier band; the recipe card exists
- [ ] 1. CPU smoke (< 1 min): build, a forward pass and a backward pass on a 1k world document; the
         causality test (no logit moves when a later token changes)
- [ ] 2. Tuning runs (§7) on Polonez or Odra; write the chosen settings and costs into the card; commit
- [ ] 3. Screen run (30M, 0.6B tokens) → text-core at 1k–16k. Go to main only if T2 passes at 4k and
         language loss is within 5 % of dense
- [ ] 4. Main run (100M, 2B tokens); checkpoint evaluations at 10/25/50/75/100 %
- [ ] 5. Final evaluation: text-core at all lengths + controls (§10.3); text-frontier when in scope
- [ ] 6. Pull results into the ledger, regenerate the board, hand the result to experiment-track
- [ ] 7. Tell the author per research-comms: frontier, reach, tokens-to-pass, language loss vs dense
```

- **Never:** hand-edit the frozen data, change the recipe after the scored run starts, tune on ladder
  scores, or compare across versions or tracks without saying so.
- **Server use follows the standing rules:** code to servers by git only, ask before using Odra while
  the DNA checks hold it, and wake Polonez only with the author's go-ahead.

## 15. Versions, storage, determinism
- `text-v1` freezes:
  - tokenizer (hash);
  - generator version;
  - corpus shards and their order (manifest hash);
  - evaluation sets, materialised to files with recorded hashes;
  - task table, floors, budgets, caps.
  Any change → `text-v2`; old results stay labeled with their version.
- **Storage** follows the [storage contract](capability_checks.md#where-results-live-the-storage-contract):
  - raw → server `Cache/text_checks/<run>` → NAS;
  - ledger → `docs/2_Experiments_Registry/results/capability/text/`;
  - training curves → W&B project MrCogito, group `text-checks-v1`.
- **Determinism:** the generator and the scoring are exact. Training is seeded; GPU kernels are not
  bit-deterministic, which the reference seed spread covers.

## 16. What has to be built (not done yet) and the calibration pilot

| piece | where (proposed) | reuses |
|---|---|---|
| world generator + answer validators + floors | `data/text_world.py` | the DNA generator's patterns (pure functions, frozen recipes) |
| tokenizer build + corpus shards + frozen eval sets | `scripts/build_text_checks_data.py` | pretokenization and manifest scripts |
| task table, versions, packages | `evaluation/text_checks.py` | `capability_tasks.py` structure |
| runner (tune / screen / main / eval, scripts per GPU, resumable) | `scripts/run_text_checks.py` | `run_capability_suite.py` launcher pattern; training through `training/train_concept_pretraining.py` and the multi-GPU launcher (recipe card → launcher settings) |
| one-pass scorer (exact + pick + controls) | `evaluation/text_checks_eval.py` | long-context probe loading |
| scorecard + board section | `analysis/text_checks_scorecard.py`, `capability_board.py` | existing board |
| local and dense baselines at the tier sizes | config entries | existing perceiver_ar dense / window modes |
| skill | `.cursor/skills/text-checks/SKILL.md` | `capability-checks` skill |

**Built 2026-10-06 (draft `text-world-v0`):**
- `data/text_world.py`: the generator (seven tasks, held-out names, renamed filler with reserved-relation
  sentences dropped, evidence-removed twins, floors); tests `tests/test_text_world.py`.
- `scripts/build_text_checks_data.py`: downloads the stories (anonymously: an expired saved Hub login
  otherwise makes public datasets look missing), trains the 4,096-token BPE, writes the two-source
  pretokenized manifest the trainer already reads, and freezes `eval/<split>.jsonl` with hashes.
- `evaluation/text_checks_eval.py`: the one-pass scorer (exact, pick, evidence-removed, notebook off).
- `evaluation/text_checks.py`: tiers, the round-1 models as trainer arguments (FFN width fitted into
  the parameter band per model), budgets, pass rules; recipe cards in `evaluation/text_checks_recipes/`.
- `scripts/run_text_checks.py`: plans the data / tune / train / eval phases as resumable job scripts
  with byobu starters (servers) or runs them here (`--mode local`, smoke tier); counts active training
  time across resumes and stops at the GPU-hour cap (scored at the newest checkpoint, `over_budget`);
  `select` writes each model's step size, chosen by development loss, into its recipe card.
- `analysis/text_checks_scorecard.py`: calibration by dense, frontier, reach, retention, tokens to pass,
  language gap vs dense, notebook contribution, shortcut flags; `--ledger` writes the committed copy.

Draft choices to calibrate:
- Tier shapes: 30M = width 576, 2 local + 1 notebook read + 5 local layers; 100M = width 896,
  3 + 1 + 8.
- E31's DNA-tested input layer and 16-token local windows, for every model.
- A global batch of 96 rows, so Odra's 3 GPUs and Polonez's 4 run the same optimizer steps.

**Local smoke result (2M-parameter models, 60 steps at 1k, scored at 512–2k):** all three train and
score, including at twice the training length and with the notebook off. Two bugs were found and
fixed, both only in the text read mode, so DNA exams are unchanged:
- On Apple GPUs, attention with a wider query–key than value returned the wrong width; the plain-attention
  path now pads the values and slices (exact).
- The reread loop added its training-only loop loss to its evaluation loss (about twice the others);
  in the text read mode the evaluation loss is now the plain next-token loss every arm is compared on.

Notes for the first server runs:
- World rows average about half the training length, so use length-grouped batching (padding was
  20–30 % with none).
- A 2M dense model after 6M tokens learns the language (loss 8.3 → 3.6) but no task leaves its floor:
  expected at that size, and exactly what the calibration pilot measures.

**Calibration pilot** (before freezing `text-v1`, about 2 days on one server):

1. Dense at screen size, 3 seeds: measures throughput, the seed spread, and which tasks dense passes at
   4k.
2. Adjust the mix share and the task dials until dense passes T1–T4 at 4k; T5 may stay `uncalibrated`.
3. Dense at main size, once: confirms the cap and the curve.
4. Local at screen size: confirms the controls and floors behave (local must fail T2 at 16k with the
   evidence early).
5. Freeze `text-v1`.

## 17. Open decisions for the author
1. **Main tier:** 100M parameters and 2B tokens, about one Polonez day (recommended), or 150M and 3B,
   about two days.
2. **Training length:** 4k (recommended: cheaper, and leaves 32× of room to test reading longer), or 8k.
3. **Round-1 set:** dense, local, E31c (+ notebook-off evaluation); add the reread loop when its DNA
   results are in (recommended), or include it from the start.

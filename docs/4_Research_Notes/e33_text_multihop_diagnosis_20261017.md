# Why compose and deduce stay at guessing on text, and what to try (17 Oct 2026)

Diagnosis and fix catalogue for the text checks' multi-step tasks. The author asked on 17 Oct to review the code
and the results as an experienced researcher, find why multi-hop does not work, and list fixes that stay aligned
with E33, the Vision and agent-to-agent latent messages, each judged by how likely it is to help and how easy it is
to build.

Task names on the board:
- **T5 compose:** "Where does the teacher of Lumo live?" One step means two lookups, two steps mean three.
- **T7 deduce:** "Lumo is a kavi. Every kavi is shiny. What is Lumo like?" Depth 1 means two lookups.
- **T6 count:** "How many times did Lumo visit the market?" This is aggregation, not multi-hop. It fails for a
  different reason and gets its own section (3.5, 5.6).

Builds on:
- [e33_second_hop_diagnosis_20261006.md](e33_second_hop_diagnosis_20261006.md), the synthetic chains (C5). The
  same loop learns 2 and 3 hops there. Most of its literature is reused below.
- The E33 page, sections 13.4–13.5 (the bridge-question plan, round targets as the next step) and 14.
- The text board: runs to 17 Oct.

## Brief

- **Step 0 result (17 Oct, measured; §7).** Link 1 breaks, in both models. Asked a two-step question, neither
  makes the first lookup:
  - E33's reads go to all home facts about equally (the answer's type), not to the teacher fact, so the answer is
    the uniform guess;
  - dense never holds the bridge in any layer, although it answers each single step when asked for it (77 % and
    68 %).
  - E33 also cannot read a teacher fact through its notes even when asked for it directly (8 %). Its notes answer
    when the asked name sits in one fact and fail when it sits in two facts with different roles.

  So the first fix is to teach the first read where to look (§6, run L3 first). For E33, its notes must also be
  able to read two-name facts.
- **What the results say.** Every model sits exactly on the uniform guess over the document's candidates:
  - compose and deduce, one step and two steps, 1k to 16k;
  - dense included, with and without the bridge questions;
  - the graded pick is 8 % = 1/12 in every cell.

  There is no partial skill to build on. The models know that the answer is one of the document's places, and
  nothing about which one.
- **Why.** An answer needs four links, and training rewards none of them directly:
  1. read 1 finds the right fact (the teacher of Lumo);
  2. read 1 carries the bridge ("Pim") out as an identity;
  3. the think step turns "Pim" into the next question;
  4. read 2 finds "Pim lived in …".

  On text the join (links 2–3) gets about 0.04 % of the training loss, with no curriculum. E33's four loop rounds
  have become interchangeable: they score the same after round 1 as after round 4, on every task. The two
  differences that matter most between E33's working synthetic recipe and text are task-dense, staged data and
  losses on the answer only.
- **What I recommend.**
  1. **No training first:** five measurements on existing checkpoints, to find which link breaks.
  2. **Then three runs at 30M, warm-started from E33's bridge-data checkpoint (its notebook already reads):**
     - **(a)** a short reasoning phase: task-dense, staged data (one step first), loop exits on answers only;
     - **(b)** (a) plus a target for round 1: name the bridge. This is the recipe that made the synthetic chains
       work.
     - **(c)** (a) plus a target for where each round reads (the page of fact r). Supervising each hop's retrieval
       is the largest measured lever in the multi-hop retrieval literature, and it trains exactly the per-round
       request that agent B sends.

     Supervision is switched off for the last 40 % of each run, so the exam measures latent reasoning. Add a dense
     control on (a)'s data. About 20 GPU-hours in all.
  3. **The bold arm, if these runs work:** linked notes. Each note carries a pointer to what to read next. This
     moves reasoning into the notebook, and it is exactly what an agent-to-agent request needs.
  4. **Count is separate.** A softmax read averages; it cannot add. Fix: a summing read over the notes.

## 1. What the results show

Exact answer / graded pick (the gold candidate's share of the probability over the document's candidates), at 4k.
Guessing is 8.3 % for compose and deduce (12 candidates) and 9.1 % for count (11). Runs: the 8-questions recipe
(3 seeds each) and the bridge-question data (one run each). Far lookup is read on facts placed early or in the middle.

| task | dense 30M | dense, bridge data | E31c (notebook) | E33 (notebook + loop) | E33, bridge data | E33f (split) |
|---|---|---|---|---|---|---|
| compose, 2 steps | 7–9 / 8 | 8 / 8 | 6–9 / 8–9 | 8 / 8 | 8 / 8 | 8–10 / 8 |
| compose, 1 step | 8–11 / 8–9 | 12 / 9 | 8–11 / 8–9 | 5–10 / 8 | 9 / 9 | 10 / 9 |
| deduce, depth 2 | 8–10 / 8 | 8 / 8 | 8–11 / 8 | 8–9 / 8 | 10 / 8 | 4–6 / 8 |
| deduce, depth 1 | 8–11 / 8 | 8 / 8 | 8 / 8 | 6–8 / 8 | 7 / 8 | 6 / 8 |
| count | 17–38 / 16–34 | 50 / 38 | 24–28 / 22 | 24–27 / 21–22 | 26 / 23 | 29–33 / 24 |
| *for comparison:* far lookup | 56–63 | 81 | 75–97 | 70–76 (one seed 13) | 72 | 74–77 |

- The plateau is total. The graded pick sits at 1/12 to two digits in every compose and deduce cell, at 1k as well
  as at 16k. An exam the model half-solves (for example by finding the bridge but mixing up the second fact) would
  lift the graded pick first. The evidence-removed twin drops the answer probability sharply (answer surprise
  0.68 → 3.87 nats on E33): the models learned the answer type, not the answer.
- **The steps are learnable on their own.** Dense on the bridge data answers "Who is the teacher of X?" 69 % at 4k
  and "What is a kavi like?" 45 %, lookup 81 % and keyed 80 %. It still guesses on one-step compose (12 %). It knows
  both steps and cannot join them. E33 was not asked the bridge questions in its exam.
- **The loop does no task work.** E33 run 1 stopped after 1, 2 or 3 rounds scores like 4 rounds on every task,
  within 2 points (board check, 16 Oct).
- **Scale is untested.** The one 50M run (dense, 1.5B tokens) was on the older 1-question data and learned almost
  nothing (lookup 20 %, quote 0 %), so it says nothing about compose.
- **Count is learned partly.** Every model gets "0 visits" right most often and lands near the true count. None
  counts exactly. The best dense run (50 %, bridge data) beats every notebook run (24–33 %).

## 2. What a one-step compose answer needs inside E33

The question ends "… teacher of Lumo live?\nAnswer:". At the answer position E33 runs:
- the prelude: 2 local layers that see 16 tokens;
- the loop core, 4 times with tied weights: the reading layer (last 256 raw tokens plus all closed notes, one
  softmax) and 1 local layer (16 tokens);
- the coda: 4 local layers (16 tokens).

Only the reading layer sees more than 16 tokens back, and only it sees the notes.

```
link 1  read 1   query "teacher-of(Lumo)"  →  note "Pim was the teacher of Lumo"
                 (not "Lumo was the teacher of Kavi", not "Lumo lived in Ravimoor": Lumo is in 3 facts)
link 2  carry    read-1 output (a mix of 64-number values) must hold "Pim" as an identity
link 3  think    2 feed-forward blocks + 1 local layer (16 tokens) turn that into a round-2 query "home-of(Pim)"
link 4  read 2   query "home-of(Pim)"  →  note "Pim lived in Tavimere"  →  first token " T"
```

- **E31c (one read) cannot do link 4 for far facts.** The layers after its reading layer see only the last 16
  tokens. Its compose score can only come from facts that happen to sit near the question.
- **Dense can do every link** in principle, since all its layers attend to everything. It fails by learning, not by
  capacity.
- **Deduce has the same shape.** The bridge is the invented category word ("kavi"), and it appears in two facts in
  two roles: the object of "Lumo is a kavi" and the subject of "Every kavi is shiny".

## 3. Why it fails: five causes, most important first

### 3.1 Training never pays for the join (every model)
- **Share of the loss.**
  - 1.5B tokens; world documents are 55 % of tokens at a mean of about 1.4k tokens each, so about 600k documents.
  - Compose is 1 task in 7: about 85k documents with up to 8 questions each, about 600k compose answers, half of
    them one-step.
  - The decision is the answer's first token: about 0.04 % of all training tokens. Deduce is the same.
- **No curriculum.** One and two steps are mixed from step 0, over documents of 256–4,096 tokens.
- **The join has no gradient of its own.** Read 2 can only learn to ask about "Pim" once read 1 hands it "Pim", and
  read 1 is only rewarded for fetching Pim once read 2 uses it. Each half needs the other before it gets any
  signal. The bridge questions teach the two lookups but not the hand-over between them (dense: 69 % on the bridge,
  12 % on the join).
- **The literature documents this exact plateau and what ends it.**
  - Guo et al. 2025 (in-context two-hop with distractor chains, 3-layer models, one-token entities): the model first
    attends to every candidate and outputs a uniform guess over the end entities, our 1/12. Then it switches
    abruptly to a sequential mechanism, one read for the bridge and one for the answer. 7–8B models fall back to
    about 1/K with K chains; fine-tuning on 1,000 two-hop prompts fixes it.
  - Yao et al. 2025: data for implicit multi-hop grows exponentially with the number of hops. A staged hop
    curriculum cuts it (3 hops ×5 → ×2, 4 hops ×20 → ×5). Mixing lower-hop questions into training gave only modest
    gains, which matches our bridge-question run.
  - Wang, Nichani et al. 2025 (theory plus a 5-layer model): composition from hard examples alone needs exponential
    samples. An easy-to-hard curriculum, or easy and hard mixed, makes it polynomial.
  - Kim et al. 2024: an easier sibling task shortens a plateau (180k → 3.2k steps).
  - Abbe et al. 2024: unsupervised scratchpads do not break the barrier; supervised ones do.
  - On BABILong's two- and three-fact tasks, recurrent memory models reach 73–100 % at 64k only when fine-tuned on
    the task with a curriculum over segment counts (RMT, ARMT).
- **The same architecture learns the join when the data pays for it.** On the synthetic chains (C5), E33's loop
  went from guessing to 94 % in 2 of 3 seeds, from random init. The two setups differ like this:

| | synthetic chains (C5, recipe A): learned | text compose: not learned |
|---|---|---|
| loss | answer tokens only (every other label −100) | every token; compose answers ≈ 0.04 % |
| curriculum | 1-hop stage to 96–100 %, then 2 hops with half the rows at 1 hop | 1–2 steps mixed from the start |
| hop count | stated in the question | implied by "the teacher of the teacher of" |
| loop exits | trained on answer tokens only | trained on every token (3 exits × 0.3) |
| when it learned | a sudden jump after 6k–10k steps of the 2-hop stage | 7.8k steps in all, about 60 compose answers per 190k tokens |
| entities | 8-letter names, one relation, one fact per edge | 2–3-token names, 4 sentence shapes, each name in 3 facts |

Confidence: high that this is the main cause for dense; high that it is necessary for E33, medium that it is
sufficient.

### 3.2 E33's rounds have become interchangeable (E33 only)
- Rounds 1–4 score the same. Three things push toward that:
  - every non-final round runs the whole coda and must predict every next token (3 × 0.3 = 0.9 of loss weight
    against 1.0 for the last round);
  - weights are tied;
  - the round embeddings start at zero.

  On stories round 1 is already enough, so a fixed point is the cheapest solution. A second round that reads
  something different never pays on 99.96 % of tokens.
- On C5 the exits saw only answers, and the rounds split the work: round 1 picked the first letter at 47–49 % (the
  first-letter guessing rate is about 42 %), rounds 2–4 at 91–93 %.
- Confidence: medium. The fixed point is measured; that the exits cause it is inferred. It is cheap to test (fix
  5.2-a).

### 3.3 A read hands out the next token, not the bridge (E33, E31c)
- **What a value is trained to carry.**
  - A note's value is a linear map of what its latent read on the page: 64 numbers, one head. The 9 query heads mix
    the values, and one output projection adds the mix to the residual.
  - Next-token loss trains it to deliver what the next token needs. That is the head of a name (the capital letter
    " P"); the rest is fetched again at each following position.
  - The C5 probe showed exactly this. After round 1 the state held node 1's first letter at 99 %, its 4th at 54 %
    and its 8th at chance.
- **Why the head is not enough here.** Person names are 2–3 tokens, and the first token is the capital letter. The
  name syllables start with about 17 consonants, while a cast has 12–16 people, so initials collide often. A value
  that carries " P" cannot address Pim's home.
- **No tie between value space and query space.** In E33 the round-2 query is W_q(x + W_o v̄): any route from
  "what I read" to "what I ask" must be learned from compose and deduce answers alone (3.1). The literature closes
  this gap in several ways:
  - End-to-end memory networks tie hop r's output embedding to hop r+1's key embedding.
  - Key–value memory networks map the read into query space (q ← R(q + o)).
  - Multi-hop dense retrieval forms the hop-2 query by re-encoding the question plus the retrieved passage with the
    same encoder. With separate encoders, chain recall@2 falls from 65.9 to 59.9.
  - DiscoLoop (2026): after loop 1 the bridge is almost perfectly decodable, but its hidden state has cosine of only
    about 0.3 with the bridge token's embedding. Re-injecting the decoded token's embedding fixes out-of-distribution
    two-hop (8 % → near 100 %). Caveat: the facts are in the weights there, not in context.
- Confidence: medium-high (strong on C5; on text, measurement D2 below).

### 3.4 One key per note cannot say which role a name plays
- **Compose.** Each person appears in three facts: as a student, as a teacher, and with a home. The four training
  shapes put the roles in different word orders. Read 1 must match "Lumo as student", not "Lumo as teacher".
- **Deduce.** Read 2 must match "kavi as subject" ("Every kavi is …"), not "kavi as object" ("Lumo is a kavi",
  which read 1 just fetched and which matches "kavi" just as well).
- **Capacity.** One 64-number key per note, shared by all 9 query heads. Crowded pages are already E33's weakest
  retrieval case (lookup 44 % at 4k against 75 % for E31c run 2).
- **The literature agrees that compression with independent reads breaks multi-hop.**
  - Soft prompt compression drops HotpotQA to the no-context level (exact match 0.30–0.32 vs 0.28 with no context
    and 0.66 uncompressed). One embedding block per paragraph recovers 0.51.
  - Retrieval with independent reads scores below random on BABILong's two- and three-fact tasks.
- Confidence: raised to medium-high for E33 by Step 0 (§7). E33 reads a fact through its notes when the asked
  name appears in one fact (home 74 %, class 65 %). It fails when the name or word appears in two facts with
  different roles: teacher-of 8 % (dense 77 %), class → property 15 % (dense 42 %).

### 3.5 Count is not multi-hop: a softmax read averages
- **Why one read cannot count.** One read returns a weighted average of the matching notes' values. If those notes
  win the softmax, the average no longer depends on how many there are. Count shows only in how much of the
  softmax the matches take from everything else, which is a saturating, noisy signal. That fits "0 right, the rest
  roughly".
- **Notes merge visits.** Visits that fall in one page share notes (6 tokens per note; visit sentences are about 5
  tokens), so a note must also count its own visits. This may be why no notebook run (24–33 %) reaches the best
  dense run (50 %).
- **The literature confirms it.**
  - Behrens et al. 2024: softmax normalization removes the count direction. A start-of-sequence sink makes its
    attention score a count proxy (99.9 % on a small histogram task), and unnormalized attention reaches about
    100 %.
  - Yehudai et al. 2024: an attention-based count gives about 1/c, and turning that back into an integer needs an
    MLP of width about n (fine for n ≤ 10).
- Confidence: high on the mechanism.

### Not the cause
- **The exams.** The rule reader scores 100 %, and the evidence-removed twins are at guessing.
- **Length.** The plateau holds at 1k as well.
- **The notebook alone.** Dense fails as well.
- **The bridge facts missing from the notes.** Lookup through the notes works at 80–90 %, and E33f answers far
  lookup from about 20 fetched notes.

## 4. Measurements first (no training; on the E33 bridge checkpoint and dense bridge)

| # | measurement | tells | tool |
|---|---|---|---|
| D1 | per-round note mass at the answer position of one-step compose / deduce items: on the page(s) of fact 1 and of fact 2 | does read 1 find the teacher fact? does read 2 move to the home fact? | `capture_note_mass` (exists) |
| D2 | linear probe per round at the answer position: the bridge's identity (which of the 12 cast members), and its 1st / 2nd token | link 2: does the bridge survive the read? | `probe_round1.py` pattern (C5) |
| D3 | the bridge written into the question ("Where does Pim live?" on compose documents) and the bridge questions themselves, on E33 | links 1 and 4 alone; the upper bound if the join worked | exam builder, 1 split |
| D4 | where the wrong mass goes: the asked person's own home (the lookup shortcut), the reverse-direction teacher, the 1-step answer on 2-step items | is any partial chain there? | per-candidate probabilities from the scorer |
| D5 | count: accuracy by true count and by how many pages hold the visits | is it the merge (3.5) or the read? | existing exam records |

Cost: under 2 GPU-hours on one server GPU. D1 + D2 decide between fix 5.2 (the read is right, the hand-over fails)
and fix 5.3 (the read itself is wrong).

## 5. Candidate fixes, assessed

**P** = my estimate that the fix, in its natural form at 30M, lifts one-step compose and depth-1 deduce to
≥ 30 % exact at 4k (guessing 8 %), with supervision switched off before the exam. **Ease:** 1 = a setting, 2 = a
builder option or a small loss, 3 = a new module, 4 = a new training mode. **Fit:** E33 / Vision (reasoning in
latents) / agent-to-agent (A2A).

### 5.1 Data and curriculum (every model)
| fix | what changes | evidence | P | ease | fit |
|---|---|---|---|---|---|
| a. reasoning phase, task-dense and staged | warm start; 0.15–0.3B tokens with compose + deduce ≈ 40 % of tokens, lookup / keyed 20 %, stories 40 %; one step only, documents 256–2k, until the graded pick passes 50 %; then 1–2 steps and 256–4k | C5 recipe A; staged hop curriculum cuts data ×5 → ×2 at 3 hops (Yao); easy-to-hard makes composition polynomial (Wang, Nichani); mixing lower hops alone is modest (Yao) | 0.35 (E33), 0.2 (dense) | 2 | all |
| b. written chain answers, then internalize | "Answer: Pim. Tavimere." (bridge first); then drop the written bridge stage by stage | C5: written path 95 % for E31 and E33; Coconut and stepwise internalization (GPT-2 small: 9×9 multiplication 99 % after removing the steps gradually) | 0.8 for the written form; 0.2 that it becomes latent (C5: written → loop 11–18 %) | 2 (+ a chain-scoring exam mode) | E33 yes; Vision partly (explicit); A2A: interpretable requests |
| c. hop count in the question | "(2 steps)" marker | needed in C5's 1-hop stage | 0.05 alone | 1 | — |
| d. single-token names (diagnostic only) | cast names drawn from names that are one token | every positive multi-hop result uses atomic entities | not a fix; isolates 3.3 | 2 | — |

### 5.2 The loop: give the rounds a job (E33)
| fix | what changes | evidence | P | ease | fit |
|---|---|---|---|---|---|
| a. exits on answer tokens only (or off) | exit labels masked to answer spans | C5 used answer-only; rounds 1 = 4 on text | 0.1 alone; removes the fixed point; saves ~2.5× compute if off | 1–2 | all |
| b. round targets: round 1 names the bridge | progress exits (round 1 → "Pim", round 2 → the place) plus the whole-name head on round 1 | C5: 97.8 % with progress + whole-name; code exists (`set_round_targets`, name head) | 0.55 with 5.1-a | 2 (builder emits round targets) | E33 high; Vision: latent steps with a target (LeCun: "a target for the iteration"); A2A: neutral |
| c. retrieval supervision per round | aux loss −log(note mass on fact r's page) for round r at the answer position, weight annealed to 0 | the biggest measured lever in the literature: multi-hop dense retrieval without hop-order supervision, chain recall@2 65.9 → 17.6; Baleen without hop labels, 4-hop recall 85.1 → 65.6; memory networks on bAbI's two-fact task (1k examples), strong supervision 0 % error vs weak 42.8 %. Counterpoint: an activation-level bridge loss did not generalize for facts held in weights (Balesni) | 0.6 alone; 0.65 with b | 2–3 (differentiable note mass at answer positions; builder emits fact spans) | **A2A highest**: it trains the per-round request directly |
| d. bridge feedback (decode → embed → add) | round r's logit-lens distribution → soft token embedding → added to round r+1's input, gated from 0 | DiscoLoop: bridge state at cosine ≈ 0.3 with its embedding; decode → re-embed takes out-of-distribution 2-hop from 8 % to near 100 % (facts in weights, symbolic, numbers partly from plots) | 0.3 alone; +0.1 on b | 3 | Vision: latent but grounded |
| e. inhibition of return | round r+1's note logits −= β · round r's note mass (β learned, init 0) | deduce's object/subject clash (3.4); visual-search analogy | 0.1 alone; enabler for deduce | 2 | A2A: "send me different notes" |
| f. per-round query adapters | a small per-round matrix on the slot query (beyond the round embedding) | no direct evidence; cheap | 0.1 | 2 | — |

### 5.3 The notebook: make notes linkable (E33, the bold arm)
| fix | what changes | evidence | P | ease | fit |
|---|---|---|---|---|---|
| a. linked notes (pointer channel) | each note gets a third 64-number vector in key space, "what to read next"; the reader adds the read's pointer mix to round r+1's slot query, gated from 0. Self-supervised on any text: a note's pointer should find the notes that mention the same name elsewhere (contrastive), plus the task loss | memory networks' adjacent tying and multi-hop retrieval's shared encoder put "what I read" in the next query's space; the identity-bridge result (a zero-hop task that shares an entity's representation across roles unlocks two-hop, facts in weights); no neural pointer-memory paper with multi-hop numbers found, so this would be new | 0.35 | 3 | **Vision + A2A highest**: notes become a latent graph; A can follow a link before replying |
| b. notebook consolidation | 1–2 attention layers among the notes after the document is written (legal in E33f's split mode: the notebook is complete before the questions) | write-time composition works elsewhere: Recurrent Entity Networks solve bAbI's two-fact task with one read (0.1 % error); HippoRAG 2's write-time graph beats dense retrieval on 2Wiki (F1 71.0 vs 61.5); "Language models need sleep" (2026) unlocks deeper hops by looping at write time (loss curves only) | 0.35 | 4 | Vision high (reasoning in concept space before any question); cost O(notes²), needs sparsity at 1M |
| c. role-aware keys | 2–3 key/value entries per note (`lm_reader_tokens` 2–3) or 2–3 key/value heads | 3.4; crowded-page weakness | 0.1 alone; enabler | 1 | E33 yes |

### 5.4 Answer-side latent thinking
| fix | what changes | evidence | P | ease | fit |
|---|---|---|---|---|---|
| a. continuous thought positions (Coconut-style) | after "Answer:", k positions whose input is the previous position's last hidden state; each reads the notes | Coconut ProsQA 97.0 with its curriculum, 76.1 without, 76.7 with no chain of thought | 0.25 after 5.1-b | 4 | Vision high |
| b. pause tokens | k filler tokens before the answer | adds reads side by side, not one after another; Coconut's comparison: pause tokens 75.9 on ProsQA, the same as no chain of thought | 0.05 | 1 | — |

### 5.5 Not now
| fix | why not |
|---|---|
| scale (100M, 2B tokens) on today's data | untested, but the bottleneck in 3.1 is signal, not parameters; at 30M the same loop composes when the data pays for it (C5); cost high (P 0.1) |
| RL on final answers | RL raises pass@1 but not pass@k beyond the base model (Yue et al. 2025); there is no skill to sharpen yet (P 0.05) |
| query-conditioned rewriting of pages | breaks write-once notes and the cheap A2A reply |

### 5.6 Count (T6)
| fix | what changes | evidence | P (exact ≥ 50 % at 4k) | ease |
|---|---|---|---|---|
| a. summing read | 1–2 heads on the note read with sigmoid (unnormalized) weights: out = Σ σ(q·k̄ + b) v̄ | unnormalized attention counts at about 100 % in small models (Behrens); sigmoid attention matches softmax on language at scale (Ramapuram); CoPE's sigmoid-gated counts: 0 % error with one variable, values to 10 | 0.45 | 3 |
| b. expose E33e's soft count | E33e already computes a soft count of the notes matching the best one; add it to the residual through a learned projection | the count is computed today and thrown away | 0.35 | 2 (needs E33e's result first) |
| c. a reader sink | one always-visible "nothing" key/value in the read, so the mass on real notes means "how many" | a start-of-sequence sink as a count proxy: 99.9 % (Behrens); `lm_null_latent` is a writer-side sink, not this; visits merged inside a note still need the note to count | 0.35 | 1–2 |
| d. count curriculum | counts 0–3 first, then 0–10; count-dense phase | plateau literature | 0.2 alone | 2 |

## 6. Recommended plan

**Step 0 — measure (D1–D5), no training, under 2 GPU-hours.** It decides which link breaks:
- read 1 finds the teacher fact but the probe does not see the bridge → start with 5.2-b;
- read 1 does not find it → start with 5.2-c;
- both hold but read 2 does not move → 5.2-d or 5.3-a.

**Done 17 Oct (§7): read 1 does not find it.** So L3 (5.2-c) leads, L2 follows. E33 also gets a gate before any
chaining claim: the single step "Who is the teacher of X?" must reach ≥ 60 % through the notes (today 8 %; dense
77 %). If L1 does not get there, the next run adds two keys per note (5.3-c, `lm_reader_tokens` 2) before L2 / L3
are read.

**Step 1 — three E33 runs and a dense control, 30M, warm start from the bridge-data checkpoints (E33's notebook
already reads there).**
- 0.3B tokens each.
- The last 40 % of each run with every supervision weight at 0 (answer-only), so the exam measures latent
  composition.
- L2 and L3 are siblings on top of L1, so each form of supervision is measured alone.

| run | changes | answers |
|---|---|---|
| L1 | reasoning phase (5.1-a) + exits on answers (5.2-a) | is it only data and a stuck loop? |
| L2 | L1 + round targets: round 1 names the bridge (5.2-b) | supervising *what* round 1 holds: the C5 recipe, ported |
| L3 | L1 + retrieval supervision per round (5.2-c) | supervising *where* each round reads: the A2A request |
| dense | L1's data on the dense bridge checkpoint | is the join a loop advantage? (C5: dense 15 %, E33 94 %) |

If L2 and L3 each give part of the answer, L2 + L3 combined is the fourth run.

- **Pass:**
  - one-step compose and depth-1 deduce ≥ 40 % exact at 4k (guessing 8 %), with supervision off;
  - two-step compose ≥ 25 %;
  - far lookup at 16k within 5 points of the start;
  - story loss within 0.03.
- **Kill this route:** L2 and L3 both still below 20 % on one-step compose at the end of their supervised part.
  Then each round is told what to hold or where to read, and still cannot ask with what it read. The value→query
  link is the blocker, and the next step is 5.2-d or 5.3-a.
- **Cost.** E33 runs at about 65k tokens/s on 4 GPUs, so one run takes about 1.3 h × 4 GPUs ≈ 5 GPU-hours. The
  ladder plus dense comes to about 15–20 GPU-hours, and fits Polonez's burst rule. Build work: about 2 days (builder
  dials and spans, the answer mask, round-target emission, a differentiable note mass).

**Step 2 — if the ladder passes:**
- two steps and depth 2;
- 16k documents;
- the A2A demo on compose. Agent B's round-2 request is built from the notes it got in round 1. Count the bytes,
  and compare with B asking once with the whole notebook.

Then the bold arm, linked notes (5.3-a): the same exam, the pointer used in place of the round targets.

**Count, in parallel or after:** the summing read (5.6-a) as one 30M arm on the 8-questions recipe. Pass: exact
count ≥ 50 % at 4k; stop: far lookup at 16k more than 5 points under E33.

## 7. Step 0 results (17 Oct 2026, Polonez GPUs 0–1; no training)

Script `analysis/multihop_links.py`. Checkpoints:
- E33 on the bridge-question data (checkpoint 7038, the one its exams used);
- dense on the same data (final).

Exams: the frozen one-step items (`steps`, 4k, 300 per task), then compose / deduce two-step and count at 1k and 4k
(250 per cell). Runs:
- pass 1: 08:25–08:35 UTC;
- pass 2: 08:39–08:47 UTC, with fact groups and a probe control.

Summaries: `docs/2_Experiments_Registry/results/capability/study/multihop_links_step0.polonez.json`.

**The single steps work in isolation, the compound question does not** (exact pick, 4k, guessing 8 %):

| question on the same documents | E33 | dense |
|---|---|---|
| one-step compose: "Where does the teacher of Lumo live?" | 8 | 9 |
| step 1 asked directly: "Who is the teacher of Lumo?" | **8** | 77 |
| step 2 with the bridge written in: "Where does Pim live?" | 74 | 68 |
| one-step deduce: "What is Lumo like?" | 8 | 8 |
| step 1 asked directly: "What is Lumo?" | 65 | 96 |
| step 2 written in: "What is a kavi like?" | 15 | 42 |
| count, exact | 25 | 51 |

**Where E33's reads go on one-step compose.** The table gives the share of each read round's attention (mean over
heads) on the pages that hold the facts named, at the answer position; "uniform" is what a read spread evenly over
the visible notes would give.

| pages holding | round 1 | round 2 | round 3 | round 4 | uniform (round 1) |
|---|---|---|---|---|---|
| the teacher fact (fact 1) | 5.3 % | 5.3 | 4.5 | 4.1 | 5.2 |
| the bridge's home (fact 2, the answer) | 10.4 | 10.3 | 7.8 | 6.5 | 5.2 |
| the asked person's own home | 9.5 | 9.3 | 7.1 | 6.0 | 5.0 |
| any home fact (12 of them) | **76** | 74 | 58 | 49 | 46 |
| any teacher fact | 46 | 46 | 39 | 35 | 45 |
| *reference: "Where does Pim live?", Pim's home* | **27** | 9 | 6 | 6 | 5 |

- **The first read targets the answer's type, not the first fact.** Three quarters of round 1's attention is on
  home facts. Every home gets about twice its uniform share: the gold one, the asked person's own, any other.
  The teacher fact gets exactly its uniform share. That is the plateau mechanism Guo et al. 2025 describe (the
  query attends to every candidate, so the output is uniform over them), here in the notes.
- **When the name is in the question, the read is sharp:** 27 % on one page, 5.5× uniform, and 74 % right.
- **Rounds 2–4 read the same places, less.** Note attention falls from 80 % to 59 % of the read. They never
  re-target.
- **Deduce looks the same, weaker:** the property and class facts get 1.2–1.3× their uniform share in round 1,
  and the chain's facts 1.2–1.4×.

**The bridge is never there.** A ridge probe on the state at the answer position (state → the name's mean token
embedding, nearest of the 12 cast members; guessing 8 %), after every layer call and loop round, finds the bridge
at 4–13 % in both models on compose and deduce. The positive control is dense asked "Who is the teacher of
Lumo?", where the bridge is decodable from layer 3 on (48–57 %). So the probe sees a bridge when one is there.
The deduce control is not valid: the answer position there predicts " a", not the class word.

**Where the compose answer's probability goes:** gold 8.4 %, own home 8.5 %, the reverse teacher's home 8.3 %, the
1-step answer on 2-step items 8.4 %. It is spread flat; no partial chain shows.

**Count.**
- E33 is half as accurate as dense: 25 % against 51 % exact at 4k, a mean error of 1.4 visits against 0.6.
- Both get 0 visits right most often.
- At equal counts (2–5 visits; 37 items share a stride, 63 do not), sharing a 192-token stride costs both models
  10–16 points: E33 19 % against 29 %, dense 43 % against 59 %. E33 trails dense by about 30 points either way, so
  the read is the main limit (3.5) and merged visits a smaller one.

**Which link breaks, and what follows.**
1. **Link 1, in both models.** Under the compound question the first lookup is never made. Dense has the skill
   (77 %) and does not invoke it. E33 reads the answer's type. Links 2–3 cannot be tested while link 1 is off; link
   4 works once the bridge is written (74 % / 68 %).
2. **E33 has an extra gap at link 1:** a two-name fact is unreadable through its notes even when asked directly
   (8 %). This is cause 3.4, now the first E33-specific blocker. Lookup works because a home fact names one person.
3. **For the plan (§6):**
   - supervising where round 1 reads (L3) is the most direct fix for (1);
   - a gate on the teacher question, and two keys per note if the gate fails, handle (2);
   - the dense control now isolates the classic gap: steps known, join not invoked.
4. **One run per model.** Pages hold about 1.5 facts each at 4k (the uniform column accounts for page size).

## 8. Fix runs (17 Oct 2026, Polonez; the author's goal: multi-hop > 70 % on the board, ≤ 100M parameters)

All runs are 30M, warm-started from E33's bridge-data checkpoint (7038), with LR 3.3e-4 and cosine decay. Scores are
exact at 4k / 16k on the frozen exams.

**Gate (09:58–10:22 UTC; 0.1B tokens of compose documents asking only "Who is the teacher of X?").**
- The teacher question rose from 8 % to 47 % (dense 77 %).
- Round 1 now puts 26 % of its attention on the teacher fact's page, 5.7× uniform: as sharp as a home lookup.
- The bridge became decodable (probe 22–34 %).
- Lookup was forgotten (74 % → 8 %), since nothing else was asked.

Reading: E33's notes can hold a two-name fact. The 8 % was missing signal, not the single key, so two keys per note
are not needed.

**E33r (10:31–11:19 UTC; 0.3B tokens).** Exits off, read target 0.5 on fact pages. Data: compose and deduce are 6 of
every 11 world documents, 30 % stories, 8 questions and 4 bridges per document.

| exam (id split) | E33 (bridge data) | E33r |
|---|---|---|
| compose, two steps | 8 / 9 % | **60 / 76 %** |
| deduce, depth 2 | 10 / 7 % | 13 / 9 % |
| far lookup | 72 / 90 % | 67 / 89 % |
| far keyed | 69 / 88 % | 61 / 85 % |
| latest · count | 22 / 4 · 26 / 19 % | 20 / 2 · 21 / 18 % |
| story loss | 1.525 | 1.516 |

- **The reads follow the chain** (two-step compose, share of each round's attention): round 1 on fact 1 97 %,
  round 2 on fact 2 90 %, round 3 on fact 3 84 % (uniform about 6 %).
- **The bridges are present:** bridge 1 is decodable after round 1 (91 %), bridge 2 after round 2 (70–74 %).
- **The answer:** first-token pick 47 % at 4k.
- **The single steps asked directly:** teacher 52 %, the bridge written in 77 %.

**Why deduce did not move.** E33r's reads reach the property fact (95 % of round 1 on its page, 32 % on the notes
holding the property word, 100× uniform), but the notes do not say which property it is.

A ridge probe from the notes on the word to the word itself, among the item's 12 candidates (guessing 8 %):

| notes on | from the value | from the key |
|---|---|---|
| a place name (compose home facts) | 68 % | 59 % |
| a property word (deduce property facts) | **11 %** | 11 % |

The writer keeps rare names and places, which far questions need, and drops common words, which local context
usually predicts. A read cannot return what was never written.

**Control: E33x** (exits off, no read target, same data and budget): compose 7 / 11 %, far lookup 75 / 91 %, far
keyed 69 / 90 %. The read target is what moved compose. It costs about 8 points of lookup at 4k.

**Aiming at the word emptied the notes (E33o, E33b run 2; 12:53–14:30 UTC, from E33r, 0.3B tokens each).**
- **E33o:** each round is pushed onto the notes holding the hop's output word (the teacher, the place, the class
  word, the property), not any note of the fact's page.
- **E33b run 2:** the same, plus a note read-back: a training-only head, zero-init, weight 0.1, that names from
  each note's value the tokens it read.
- **E33b run 1:** a random-init head at 0.2. It scrambled the values within half a run (compose first-token pick
  47 → 12 %) and was stopped.

| exam (id split) | E33r | E33o | E33b run 2 | notebook off |
|---|---|---|---|---|
| compose, two steps | 60 / 76 % | 7 / 1 % | 12 / 10 % | 5 / 2 % |
| deduce, depth 2 | 13 / 9 % | 13 / 8 % | 14 / 7 % | 10 / 10 % |
| lookup (whole exam) | 74 / 87 % | 23 / 0 % | 34 / 18 % | 24 / 0 % |
| keyed (whole exam) | 70 / 83 % | 24 / 0 % | 28 / 14 % | 24 / 0 % |
| story loss | 1.516 | 1.514 | 1.513 | |

E33o ends at the notebook-off scores. Its lookup at 4k fell steadily over the run (74 → 56 → 45 → 23 → 24 % at
checkpoints 0 / 178 / 445 / 890 / 1335), so this is a slow drift, not a step-0 shock.

The reads still aim right, but the notes they reach are empty:
- **Reads:** round 1 lands on fact 1 89 %, round 2 on fact 2 81–83 %, round 3 on fact 3 77 %. On the output word's
  own notes, round 2 puts 21–32 %, about 100× uniform.
- **Note content:** the place probe above falls from 68 % to **14 %** (E33o) and **12 %** (E33b run 2); the
  property probe stays at 12 % / 10 %.
- **Single steps:** "Where does Pim live?" asked directly falls from 77 % to 12 % / 26 %. The teacher step holds
  (65 % / 69 %).

**Why (working explanation).** A note's key and value come from the same latent's read of its page:
k̄ = RMSNorm(to_k(z) + to_k_read(read)) and v̄ = to_v(read) + to_v_state(z).
- The note-level target asks the note at "Ostrava" to answer a query about Pim.
- The cheapest way is for that latent to read Pim, so the place drops out of its value.
- The page-level target is satisfied by the note at "Pim", which can keep the place in its value.

So a loss on addressing leaks into content whenever both share one read. Read-back at 0.1 did not hold it back.

**Not a ceiling either.** The dense 30M model, trained without the reasoning-phase data, is at guessing on compose
(7.8 %) and deduce (8.5 %) at 4k, though it sees every fact.

**Where note content stands (note-content probe; place recovered from the note's value / key, guessing 8 %).**

| checkpoint | place | property word |
|---|---|---|
| E33 bridge data (7038) | 71 / 72 % | 8 % |
| E33x (exits off) | 80 / 77 % | 12 % |
| E33r (page read target) | 68 / 59 % | 11 % |
| E33rb at a third / at the end | 63 / 57 % → 56 / 50 % | 13 % → 12 % |
| E33o / E33b run 2 (note-level target) | 14 / 11 %, 12 / 11 % | 12 %, 10 % |

- The read target costs content even at page level (80 → 68 %), mostly in the keys, and keeps costing it while
  training continues.
- No model has ever written the property word.

**E33rb (18:24–20:05 UTC; E33r + zero-init read-back 0.1, page target, 0.3B tokens from the E33r final).**
- No harm, small gains: compose two steps 61 / 82 % (E33r 60 / 76 %); deduce 17 / 12 % at 4k / 16k (13 / 9 %);
  lookup 74 / 89 % (74 / 87 %). Part of this is simply 0.3B more tokens.
- Read-back did not write the property word (12 %).
- The read-back head itself learned token frequency only: its top guess at a note's own position is right 0 % of
  the time for names, places, class words and properties, and 8 % for function words.
- Checked: the targets are right. Rows are not packed with document ids, so note positions are row positions.

**Dense on the reasoning data (18:24–20:05 UTC; from the q8b dense model, same data and budget).**
- Compose 8 % and deduce 6.5 % at 4k; one-step compose 11 %, one-step deduce 8 %.
- Lookup 84 %, keyed 83 %, latest 92 %, count 46 % at 4k.
- With every fact visible, the 30M dense model learns neither join in 0.3B tokens. Dense is not a ceiling for T5 /
  T7. E33r's compose comes from the loop plus the read target.

**Deduce gate (21:04–21:34 UTC; from the E33rb final, 0.1B tokens of deduce documents asking only their one-fact
questions, fact-level read targets).**
- "What is a kavi like?" asked alone: 12 → 44 %. "What is Pim?": 79 %.
- The property word in the notes: 12 → 20 %.
- The joined deduce question did not move (12 % depth 2, 16 % one step), since the gate never asked it.
- Compose was forgotten (8 %), as the compose gate forgot lookup.

Reading: the writer can store a property when training asks for it densely. The reasoning mix asked about 6× less
often per token.

**E33ok (20:04–22:37 UTC): E33o + the key-gradient stop (lm_key_grad = detach), from the E33r final, 0.3B tokens on
the output-word data.**

| exam (id split, 4k / 16k) | E33rb | E33ok |
|---|---|---|
| deduce, depth 2 | 17 / 12 % | **36 / 35 %** |
| deduce, one step | 15 / 10 % | 40 / 35 % |
| compose, two steps | 61 / 82 % | 42 / 48 % |
| lookup | 74 / 89 % | 65 / 52 % |
| keyed | 73 / 86 % | 56 / 51 % |
| story loss | 1.5135 | 1.5111 |

- **Note content is back:** the place probe is 80 / 75 % (value / key), up from E33o's 14 / 11 %.
- **The property word is written for the first time:** 33 / 31 %; every earlier model was at 8–13 %.
- **Why it works (working explanation):** the keys learn addressing only, and the answer's gradient trains the
  values of exactly the notes the read lands on.
- **What it cost:** compose and long lookup. The run began with a shock (lookup 74 → 37 % at checkpoint 178) and
  was still climbing at the end (lookup 37 → 52 → 54 → 60 %, deduce 14 → 17 → 26 → 32 %).
- **Structural part of the cost:** with the stop, a key can only reuse the content read. A note on a place name
  can no longer carry the person's name in its key, and compose and long lookup need exactly that.

**Address read (built 18 Oct, lm_key_grad = address).**
- Each latent reads its page a second time, with its own query and position prior, for its key alone.
- The token states, latent states and every shared projection enter that read with the gradient stopped, so it
  cannot move the content read.
- It joins the key through a zero-init projection, so a warm start from a "detach" model is unchanged at step 0.
- +0.36M parameters, FFN width unchanged.
- Arches: e33ak (E33ok + address read), e33gak (E33gk + address read).

**Next (Polonez, queued, ~23:30 UTC; both from the E33ok final, 0.6B tokens on the denser output-word mix
v2reason2_out: deduce 4 of 12 tasks, 8 one-fact questions per document):**
- GPUs 0–1: e33ok, the longer run.
- GPUs 2–3: e33gk = e33ok + E33g's fixed order heads. On Odra, over its matched control E33rb, E33g gained compose
  +8 / +6 and deduce +7.
- Then e33gak (address read) if compose or long lookup stay below E33rb.

## Uncertainty
- The probabilities are my judgement from C5, the board and the literature. They are not measured.
- C5's names, relations and distractors are simpler than text's, so the port may need more than the C5 recipe
  (3.4).
- "Supervision off for the last 40 %" may let the skill decay. If it does, that is a finding: the latent route
  cannot hold it without targets at 30M.
- Dense failing everything does not prove the join is learnable on this data. The ladder's dense control and C5's
  rule 5 (a model change is proposed only where some model shows the skill) apply.
- E33e's result (Odra, state unknown) may change 5.6-b.
- Much of the 2024–26 work on the bridge (DiscoLoop, identity bridge, back-patching, grokked composition, the
  two-hop "curse") holds the facts in the weights, not in context. Whether those fixes carry over to notes is
  untested. Our setting, with the facts in context, is closer to Guo et al., memory networks and multi-hop
  retrieval.
- No paper we found loops a reader over an external latent notebook and measures in-context multi-hop. A positive
  E33 result would be new; it also means there is no recipe to copy.
- No study compares one-token and multi-token entities directly (cause 3.3 rests on our own C5 probe).

## Sources
Verified by the research scout on 17 Oct 2026 (abstracts and full text); numbers as the papers state them. Marked
(W) when the facts are held in the weights rather than in context.

Learning dynamics and curricula
- Guo et al. 2025, two-hop reasoning in context, https://arxiv.org/abs/2502.13913 (code GuoTianYu2000/twohopIC)
- Yao et al. 2025 (W), implicit multi-hop needs lots of data, https://arxiv.org/abs/2505.17923
- Wang, Nichani et al. 2025, compositional functions from easy-to-hard data, https://arxiv.org/abs/2505.23683
- Kim et al. 2024, task diversity shortens the ICL plateau, https://arxiv.org/abs/2410.05448
- Wang et al. 2024 (W), grokked transformers are implicit reasoners, https://arxiv.org/abs/2405.15071
- Abbe et al. 2024, globality barrier and inductive scratchpads, https://arxiv.org/abs/2406.06467; leap complexity
  https://arxiv.org/abs/2302.11055
- Balesni et al. (W), two-hop latent reasoning, https://arxiv.org/abs/2411.16353
- Sanford, Hsu, Telgarsky 2024, k-hop needs log-k depth, https://arxiv.org/abs/2402.09268

Memory and multi-hop
- Sukhbaatar et al. 2015, end-to-end memory networks, https://arxiv.org/abs/1503.08895
- Miller et al. 2016, key–value memory networks, https://arxiv.org/abs/1606.03126
- Henaff et al. 2016, recurrent entity networks, https://arxiv.org/abs/1612.03969
- Kuratov et al. 2024, BABILong, https://arxiv.org/abs/2406.10149; Rodkin et al. 2024, ARMT,
  https://arxiv.org/abs/2407.04841
- Łajewska et al. 2025, information preservation in prompt compression, https://arxiv.org/abs/2503.19114
- Deng et al. 2025, gist-token context compression, https://arxiv.org/abs/2412.17483

Retrieval per hop
- Xiong et al. 2021, multi-hop dense retrieval, https://arxiv.org/abs/2009.12756
- Khattab et al. 2021, Baleen, https://arxiv.org/abs/2101.00436
- Trivedi et al. 2023, IRCoT, https://arxiv.org/abs/2212.10509

The bridge
- Fu et al. 2026 (W), DiscoLoop, https://arxiv.org/abs/2607.00341
- Lin, Chen, Xu 2025 (W), identity bridge, https://arxiv.org/abs/2509.24653
- Biran et al. 2024 (W), hopping too late, https://arxiv.org/abs/2406.12775; back attention
  https://arxiv.org/abs/2502.10835

Write-time composition
- Gutiérrez et al. 2024, HippoRAG, https://arxiv.org/abs/2405.14831; HippoRAG 2, https://arxiv.org/abs/2502.14802
- Lee et al. 2026, language models need sleep, https://arxiv.org/abs/2605.26099
- Lin et al. 2025, sleep-time compute, https://arxiv.org/abs/2504.13171

Latent reasoning and loops
- Hao et al. 2024, Coconut, https://arxiv.org/abs/2412.06769
- Deng et al. 2024, stepwise internalization, https://arxiv.org/abs/2405.14838
- Saunshi et al. 2025, looped transformers, https://arxiv.org/abs/2502.17416
- Kohli et al. 2026 (W), loop, think and generalize, https://arxiv.org/abs/2604.07822
- Goyal et al. 2024, pause tokens, https://arxiv.org/abs/2310.02226
- Yue et al. 2025, RL and pass@k, https://arxiv.org/abs/2504.13837

Counting
- Behrens, Biggio, Zdeborová 2024, counting in small transformers, https://arxiv.org/abs/2407.11542
- Yehudai et al. 2024, when can transformers count to n, https://arxiv.org/abs/2407.15160
- Golovneva et al. 2024, CoPE, https://arxiv.org/abs/2405.18719
- Ramapuram et al. 2024, sigmoid attention, https://arxiv.org/abs/2409.04431
- Barbero et al. 2024, transformers need glasses, https://arxiv.org/abs/2406.04267

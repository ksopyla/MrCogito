# Roadmap to 70 % on T4–T7 (17 Oct 2026)

Goal set by the author on 17 Oct 2026: **70 % exact on T4 Latest, T5 Compose, T6 Count and T7 Deduce** with a
notebook model (30M, the current v2 exams, 4k documents first, 16k next). This note reviews where each task stands,
assesses the author's seven ideas with the evidence we have, adds what an experienced researcher would add, and
lays out the experiments by owner. The text board tracks the goal under its headline.

Sessions consulted on 17 Oct (18:15–18:45 UTC): the E33 training session (A2A / E35 owner), the agent-memory
(memory tape) session, the E33 capability-testing session, and the multi-hop session (E33r owner). Their replies
are summarised in "Coordination" below.

## Brief

- **Where we are (best notebook model, 4k / 16k; dense at 4k):**
  - latest 24 / 10 % (dense 92 %);
  - compose 60 / 76 % (E33r; dense 9 %);
  - count 33 / 22 % (dense 50 %);
  - deduce 14 / 11 % (dense 10 %). Guessing is 8–9 %, and latest's effective floor is 20 %.
- **One cause behind all four: the skill gets no gradient.** Order and number are not content. A softmax read
  returns an average, which can't pick the newest match or add matches up. The parameters that could do it start at
  0 and never grow, because 99.96 % of training tokens are story text that never needs them. E33r is the proof that
  this is fixable: as soon as each loop round was told where to read, compose jumped from 7 % to 60 % in 0.3B
  tokens.
- **Four bets, one per task. Each mechanism is separable and gets a training signal from step 0:**
  1. *Order heads* for latest: the rank among matching notes, at fixed strengths on four of the nine read heads.
     Order comes from the notes' page numbers as a score term, not mixed into the key.
  2. *Tied read-back* for deduce: notes must be readable by the decoder's own read path and output head. This is
     the author's "symmetric writer" (ideas 1, 2 and 5) with no new parameters.
  3. *Query from the read* for compose: what round r read becomes round r+1's question through the shared token
     space (idea 7), on top of E33r.
  4. *A summing read* for count: two read heads with sigmoid weights that add matches up instead of averaging them.
- **How to get there fast:** stack each bet as a 0.3B-token warm start on E33r's checkpoint (about 1 hour on 2–4
  GPUs each). Once three of them work, train the combination from scratch, 3 seeds, so the result holds without
  warm starts.

## 1. Where each task stands and what blocks it

| task | best notebook 4k / 16k | dense 4k | what blocks it (evidence) |
|---|---|---|---|
| T4 latest | 24 / 10 % (E31c, E33f) | 92 % (bridge data) | Notes carry no order. E33f, which reads only its notes, fell from 95 % to 15 % at 1k: E33 keeps recency in the raw window, not in the notes. Every order parameter started at 0 and stayed there: E33d's slopes reached at most 0.14, E33e's rank strengths 0.15. With no training, a rank strength of 2 lifts E33 run 1 from 5 % to 41 % at 16k without hurting lookup. |
| T5 compose | 60 / 76 % (E33r) | 9 % | The first read did not find the first fact (Step 0). A per-round read target fixes that. What is still missing: the loop's next query is not built from what it read (§3.3 of the diagnosis). One run. |
| T6 count | 33 / 22 % (E33f) | 50 % | A softmax read averages, so count shows up only as a saturating share of attention. Visits that fall in one page merge into the same notes. |
| T7 deduce | 14 / 11 % (E33b, E33) | 10 % | E33r finds the right page 95 % of the time and answers 12 %. The writer keeps names and places but drops common words: a probe on a property-word note is 11 % vs 68 % for a place name. |

Single runs are marked as such on the board. E31c and E33 lose one run in three on lookup, so every claim needs a
second seed (capability-testing session, DNA ladder).

## 2. The author's ideas, assessed

| # | idea | what it means in our model today | verdict | targets |
|---|---|---|---|---|
| 1–2 | Writer and decoder share one token encoding and get the same signal | Half true already: the writer reads the same token table (`embed.tok`). It has its own input projection and page encoder, though, and its only training signal is what the reader's next-token loss sends back through the read. The output head is not tied. | **Adopt in two steps.** (a) The *signal* is the high-value part: make every note decodable by the decoder's own read path and output head (tied read-back, §4.2). This needs no new head, so unlike E33b's random head it can't scramble the reader at step 0. (b) Then share the *encoding*: the writer takes the full input layer `embed(ids)` instead of the bare table. That is a one-line switch, but it changes every run's starting point, so it belongs in the from-scratch confirmation. | T7, T5, A2A |
| 3 | Notes carry global positions (beginning vs end) | Notes have no position on purpose, so they transfer to longer documents. Order inside the key cost lookup at length on DNA (E31b: 47 vs 94 % at 128k). | **Adopt as a separable score term over the notes' page numbers:** newest-match and oldest-match ranks (§4.1), never a rotation of the key. "Beginning vs end" becomes two of these heads. | T4 |
| 4 | Append the decoded message to memory | The writer already works on a stream; the model's own answer tokens are text like any other. | **Adopt, oracle first** (A2A session): append the gold first-hop sentence before the question. If even the oracle stays below 70 % on compose, the idea is dead for T5. Then append the model's own decoded first hop, tagged as its own output. It turns a hidden hop into a written one, like the written chain answers C5 (the DNA multi-step task) solved at 95 %. | T5, T7 |
| 5 | A much stronger compression signal for a fast, simple writer (VAE-like head) | E35's learned 16-vector messages carried nothing (3 runs). The reader takes about 85 bits per vector. A note stores about 10× the raw text, so the problem is what the notes keep, not how much room they have. | **Adapt.** Use read-back through the shared output head (§4.2) instead of a VAE: a KL term invites collapse, and plain reconstruction rewards keeping filler. The A2A session adds a page vector distilled from a strong embedding model, for A2A retrieval. | T7, A2A |
| 6 | JEPA (predict the next page's notes in latent space) | Not tried. | **Later.** Facts are the unpredictable part of a page, so a predictive latent loss rewards exactly what we don't need for T4–T7, and it needs an EMA target and anti-collapse terms. Worth a run for gist notes and planning once the four tasks pass. | gist / planning |
| 7 | The loop refines its query to memory and links distant facts | E33r supervises *where* each round reads. Round r+1's query is W_q(x + W_o·v̄), so the step from "what I read" to "what I ask next" has to be learned from answers alone. | **Adopt:** decode round r's read through the output head, re-embed it with the shared token table, and add it to round r+1's query input, gated from 0 (DiscoLoop, memory networks' tied hops). Ideas 1–2 and 7 then meet: one token space for writing, reading and asking. | T5, T7 |

## 3. Three lenses on the problem

- **LeCun (energy-based models, JEPA, world models).** A memory is useful when its states predict what will be
  asked. Order and number are properties of the *set* of notes, not of any one note, so the read has to compute
  over the set: a rank or a sum, not one more average. Prediction in latent space (idea 6) belongs to planning, not
  to exact recall.
- **Goodfellow (adversarial training, regularisation, privileged information).** The exams are adversarial by
  design: lookalike names, other people's last moves, other chains. Train against those same distractors, and use
  the privileged information the generator already knows (which fact, which order, how many) as auxiliary targets
  that are annealed away before the exam. E33r already showed this works for compose.
- **Karpathy (data first, small loops, look at the data).** Before building, check that the training data asks the
  skill often enough at the distance the exam asks it; overfit one batch of latest items; iterate in 1-hour warm
  starts, then confirm from scratch. A parameter that sits at 0 is a data-and-gradient bug before it is an
  architecture bug.

## 4. The bets, ranked by expected gain per GPU-hour

### 4.1 Order heads (T4; idea 3), owner: this session
- **Change.** E33e's rank term, score −= s_h·log(1 + soft count of newer notes that match about as well as the best
  note), with **fixed per-head strengths** instead of learned ones from 0. Five heads at 0 serve lookup, keyed and
  quote; four heads at 1, 2, 4 and 8 serve order.
- **Variant: "earliest" heads**, which count *older* matching notes, for beginning-vs-end questions.
- **Why fixed.** One scalar per head can't serve both story text (no order wanted) and latest questions. E33e's
  strengths climbed to 0.15 and fell back to 0.09. With heads at several fixed strengths the network picks the
  right head downstream, which is the same logic as ALiBi.
- **Tests.**
  - (a) No training, done 17 Oct 18:20–18:35 UTC on Odra: E33r with the order term, exam set, 200 items per cell.
    - Strength 2 on every head: latest at 4k 20 → 28 %, at 16k 2 → 7 %; far lookup at 16k 87 → 76 %, compose at
      4k 60 → 50 %.
    - Four order heads (1, 2, 4, 8): latest 24 / 4 %; lookup and compose within 2 points except lookup at 16k
      (87 → 81 %).
    - Much weaker than the same check on E33 run 1 (5 → 41 % at 16k). E33r's note scores were not trained to be
      counted, so only training can show whether order heads work.
  - (b) E33g: warm start from E33r, 0.3B tokens of its recipe, plus the order heads (Odra GPUs 0–1).
- **Pass:**
  - latest ≥ 60 % at 4k and ≥ 40 % at 16k;
  - compose and far lookup within 5 points of E33r.
- **Kill:** latest ≤ 30 % at 4k after the warm start. In that case order is not what blocks latest, and the next
  step is a write-time update (a note marks older matching notes as superseded).

### 4.2 Tied read-back: a symmetric writer (T7; ideas 1, 2 and 5), owner: the multi-hop session (offered)
- **Change.** In training only, each note's value (64 numbers; one key/value head) is copied to the 9 query heads and
  goes through the read layer's output projection, then the final norm, then the LM head. Its one distribution must
  name the tokens at its position −1, 0 and +1. The decoder weights are detached, so only the writer learns. The
  loss weight is 0.05–0.1 and there are no new parameters.
- **Why it beats E33b's head.**
  - It is meaningful from step 0, so it can't send random gradients into the notes.
  - It trains the notes in exactly the space the reader uses.
  - It keeps property words in the notes because they are tokens of the page.
- **Test.** Warm start from E33r, 0.3B tokens. The multi-hop session runs the untied zero-init head first (E33rb,
  17 Oct). The tied version is the follow-up if E33rb keeps lookup but the property probe stays low.
- **Address/content split (the multi-hop session's bet).** On E33o and E33b, the read targets on single notes
  emptied the notes: far lookup at 16k fell from 87 % to 0 % and 18 %. A note's key and value come from one latent
  read, so a loss on addressing pulled the content away. Splitting each latent's read into an address read (key)
  and a content read (value) lets a read-back loss and a read target act on separate parts.
- **Pass:**
  - deduce ≥ 40 % at 4k;
  - compose and far lookup within 5 points of E33r;
  - a property-word probe ≥ 50 % (now 11 %).
- **Kill:** far lookup falls more than 10 points.

### 4.3 Query from the read (T5 → 70 %, T7; idea 7), owner: the multi-hop session (E33r), offered
- **Change.** Round r's read output goes through the LM head to a soft token distribution, which the shared token
  embedding turns back into a vector. That vector is added to round r+1's query input through a gate that starts
  at 0.
- **Also:** a second E33r seed, and a longer continuation (0.6B tokens).
- **Pass:** compose ≥ 70 % at 4k on two seeds; deduce ≥ 40 % together with §4.2.

### 4.4 Decode and append (T5 / T7 upper bound; idea 4), owner: the A2A session. Result: dead as a reasoning fix
- **Oracle, no training (Odra, 17 Oct 18:28–18:52 UTC, 150 items per cell, exact).**
  - E33r compose at 4k / 16k: as is 58 / 79 %.
  - Gold first fact appended before the question: 64 / 84 %.
  - The same fact about 2k tokens back: 64 / 79 %.
  - Its own decoded steps appended: 53 / 63 %; own step accuracy is 55–58 % for step 1 and 36–39 % for step 2.
  - Gold bridge written into the question: 81 / 89 %. The E33 bridge model goes from 10 % to 76 / 93 % this way.
  - Deduce: 7–15 % in every mode, gold bridge included.
- **Reading.**
  - The memory already holds the bridge; the missing link is forming the second question. That makes §4.3 (round 1
    names the bridge, and the name is re-embedded into round 2's query) the lever for compose. The written-question
    ceiling, about 80 %, is above the goal.
  - Deduce is a storage problem (§4.2), not a join problem.
- Append stays useful as a memory-tape feature, not as a reasoning fix.

### 4.5 Summing read (T6), owner: this session (no other capacity)
- **Change.** Count heads beside the read use sigmoid weights with a learned bias and no normalisation. They sum
  the gates, n = Σ σ(a·q·k + b), instead of averaging values.
- **Training signal:** the count answer's own token loss reaches the gates through the projection. No auxiliary
  label is needed at first; an auxiliary loss on the true count the generator knows is the fallback.
- **Built (17 Oct):** `lm_count_heads` (E33h: E33r + 2 count heads). Each head has its own query from the read
  layer's input and uses the notes' keys and the layer's raw keys. Every token is counted once: a page owns its first
  192 tokens, and tokens no closed page owns yet are counted from the raw keys. The sum is coded as a soft one-hot
  over 0..10 and added through a zero projection. Tests cover a brute-force reference and once-only counting across
  a page close.
- **Test.** Warm start from E33r, 0.3B tokens.
- **Pass:** count ≥ 50 % at 4k (dense's best); then 70 % in the combined model.
- **Kill:** count ≤ 35 % at 4k.

### 4.6 Data checks before any training (all tasks; the Karpathy lens), owner: this session
- How many latest, count and deduce questions per 1B tokens, at which distance from their facts, and how much
  answer-token loss each gets.
- If far latest is rare, the reasoning-phase mix gets a latest share as well.

### 4.7 Integration and confirmation
- Each bet that passes its warm-start test joins the spine: E33r, then + order heads, then + read-back, and so on.
- When three pass, train the combination from scratch at 1.5B tokens, 3 seeds, with the shared input layer (idea 1b)
  switched on.
- **Goal check:** T4–T7 at 4k and 16k, plus the no-harm rule (far lookup, keyed and quote within 5 points of E33;
  story loss within 2 % of dense).
- **Cost:** about 7 h × 3 GPUs per E33 seed. If the loop's cost is the bottleneck, E31c with the same pieces is the
  cheaper spine for T4, T6 and T7. Compose still needs the loop.

### Not now
- 100M scale: the bottleneck is signal, not parameters (E33r learned compose at 30M within 0.3B tokens).
- RL on final answers: there is no skill to sharpen yet.
- Query-conditioned rewriting of pages: it breaks the write-once notebook that A2A relies on.

## 5. Coordination (17 Oct, 18:15–18:45 UTC)

- **E33 training session (A2A / E35 owner).**
  - No GPUs held; next steps wait for the author: shorter requests in the notes protocol, and a second E33f seed.
  - Best idea for latest: order as a feature of the note, or a rank among matches, never a recency penalty. E33f
    needs order inside the notes anyway.
  - Facts: the writer already shares `embed.tok`; E35's learned compressor carried nothing.
- **Agent-memory session.**
  - Takes ideas 4–6. It bets on decode-and-append, with the oracle first, and on a teacher-distilled page vector for
    A2A. It is skeptical of JEPA for T4–T7 and of a VAE head.
  - Cautions: keep order separable from the key; measure any writer loss against the ~85 bits-per-vector reader
    ceiling.
- **E33 capability-testing session.**
  - Nothing scheduled.
  - DNA ladder: once the read has learned keyed lookup, the notebook holds 8–16 facts out to 128k. Composition costs
    notebook models about 30 points even in order (dense 100, E31 70, E33a 67). One seed in three fails lookup.
- **Multi-hop session (E33r owner).**
  - Takes T5 and T7, on Polonez only; Odra stays free for T4 and T6.
  - Next ~24 h: E33rb (E33r + zero-init read-back 0.1, warm start, GPUs 0–1), and dense on the reasoning data (GPUs
    2–3), to check whether deduce is learnable at 30M when every fact is visible.
  - Bet: hop supervision at page level plus a separate content objective, with the latent's read split into address
    and content.
  - Cautions:
    - never aim the read target at single notes with today's writer;
    - exit losses on every token make the rounds interchangeable;
    - a random-init read-back head scrambles a warm start.
  - Dense on the 8-question bridge recipe is at guessing on compose and deduce, but it has never seen the reasoning
    data, so it is not yet a ceiling there.

GPU plan (agreed 17 Oct):
- Polonez (4 GPUs): the multi-hop session's T5/T7 ladder, in 2-GPU pairs under the burst rule.
- Odra (3 GPUs): this session's T4 and T6 warm starts (order heads, summing read), then the from-scratch
  confirmations.
- The A2A session asks the author for one Polonez GPU for the decode-and-append oracle.

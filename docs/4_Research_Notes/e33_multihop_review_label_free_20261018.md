# E33 multi-hop on text: what worked, why, and a path without generator labels (18 Oct 2026)

Review of the 17–18 Oct 2026 multi-hop campaign: T5 compose and T7 deduce on the text capability board, plus the
board session's T4 latest and T6 count. All runs are 30M parameters.

The author's constraint for what comes next: no tricks and no extra labels. Only the data that exists for text
training, instruction following and reasoning texts.

Run details and numbers: `e33_text_multihop_diagnosis_20261017.md` §7–§8, and the board session's roadmap
`text_t4_t7_roadmap_20261017.md`.

## Brief

- **What we reached.** Compose 94–95 % and deduce 98–99 % at 4k on the board's id split, against 8 % and 13 %
  before. Two recipes get there, both replicated or matched:
  - E33q / E33qn: each loop round names its hop's word.
  - The key-stop line: a writer key-gradient stop, a note-level read target and a denser mix.
  - The board session took latest to 88 % (E33j). Count is still 23–28 %.
- **What it relied on.** The generator's own labels: which fact each round must read, and which word each round
  must name. Real text does not come with these. So the result shows that the architecture can do multi-hop over
  compressed notes. It does not yet show that it learns to.
- **What generalises.**
  - Deduce at depth 3 holds (85–94 %), because its chain stops on content.
  - Compose with 3 hops does not (0–3 %): the routine stops at the trained hop count.
  - Compose on held-out templates fails for every model (5–7 %).
- **What I recommend.** Keep the architecture lessons, which are label-free, and move the supervision into text:
  - **Per-hop targets from written reasoning:** the chain-of-thought steps, which a schedule then removes. This is
    the iCoT / Coconut / SIM-CoT / LOTUS line.
  - **A writer pretrained without a task:** continuation plus a little reconstruction.
  - **A reader trained on top**, with keys and values on separate gradient paths.
- **Writer and decoder separately?** Yes, in stages, not fully apart.
  - The values (what a note holds) can be learned without any task.
  - The keys (how a note is found) must learn from the reader.
  - The literature agrees that pure autoencoding does not predict downstream quality, and that freezing the decoder
    costs a lot.

## 1. What was tested

Scores are exact on the frozen exams, id split, at 4k (16k in brackets where it matters). "Gen. labels" means the
training signal comes from the task generator, not from the text.

| run | change | signal | compose | deduce | lookup | verdict |
|---|---|---|---|---|---|---|
| E33 (bridge data) | baseline loop, exit losses on every token | answers | 8 % | 10 % | 72 % | rounds interchangeable |
| E33x | exit losses off | answers | 7 % | 10 % | 75 % | not enough alone |
| E33r | per-round read target: round r reads fact r's page | gen. labels | 60 % (76) | 13 % | 74 % | the read becomes a chain |
| E33o / E33b | read target on the output word's own notes (± read-back) | gen. labels | 7 / 12 % | 13 % | 23 / 34 % | **emptied the notes** |
| E33rb | E33r + linear read-back of note content | gen. labels + text | 61 % (82) | 17 % | 74 % | read-back head learned word frequency only |
| dense, reasoning data | full attention, no notebook | answers | 8 % | 6.5 % | 84 % | 30M dense learns neither join |
| deduce gate | 0.1B tokens of one-fact deduce questions | answers (dense) | — | step 12 → 44 % | — | storage follows demand |
| E33ok | note-level target + **key-gradient stop** | gen. labels | 42 % | 36 % | 65 % | content kept, property written |
| E33ok, denser mix | 0.6B more tokens, 2.5× the one-fact questions | gen. labels | **95 % (97)** | **98 % (99)** | 97 % | the goal |
| E33g / e33gk | + fixed order heads (rank among matching notes) | label-free term | +8 over control | +7 | = | helps selection; nothing on top of e33ok |
| E33q / E33qn | rounds **name their hop** through the LM head | gen. labels | **94 % (98)** | **99 % (100)** | 98 % | the goal; seed 1 and naming-only agree |
| E33j (board) | E33q + order on demand, latest supervised | gen. labels | 96 % | 99 % | 100 % | latest 88 % (95) |
| E33h / E33k (board) | count heads (a summing read) | answers / gen. labels | — | — | — | count 23 % / sums within ~0.7 when told |
| E33c | per-round control read over the question (gate from 0) | gen. labels | — | — | — | **unused**: gate 0.0013 at checkpoint 445; stopped early |

## 2. Main facts

**F1. The join has to be paid for per hop.**
- Answer-only training never found the two-step join at 30M in 0.3B tokens. That held for the notebook model and
  the full-attention model alike (compose 7–8 %).
- Any per-hop signal unlocked it: where round r reads (E33r: 8 → 60 %), or what round r names (E33q: → 94 %).
- This matches the literature: Coconut's latent rounds learn only with a staged language curriculum (without it,
  GSM8K is no better than no CoT, 14.4 %). SIM-CoT finds that unsupervised latent steps collapse.

**F2. The writer stores what a downstream loss asks for, not what is there.**
- Under language-model loss it kept rare names and places (probe 68–80 %) and dropped common words: the property
  in "Every kavi is shiny." stayed at 8–13 % in every model.
- The property got written only when a loss reached exactly those notes, often enough:
  - the dense gate: the property step 12 → 44 %;
  - the note-level target with the key stop: probe 33 %, then 72 % on the denser mix;
  - the naming loss: deduce 99 %.
- A weak auxiliary, a linear read-back at weight 0.1, did nothing.
- Capacity was not the limit. One 64-d note per ~6 tokens held the property once asked to.
- **The cause is the raw-window shortcut, and removing it is label-free** (measured 18 Oct 15:30 UTC). E33f answers
  questions with the document readable only through the notes (the split read; no raw tokens across the boundary).
  Its notes hold the property word at 69 / 64 % (value / key) and places at 74 / 76 %. It had no read targets, no
  naming and no extra questions.
  - With a raw window, nearby words are always visible directly, so the notes specialise in far names.
  - When everything must pass through the notes, the writer keeps common words too.
  - E33f's deduce is still 6 %. Storage is necessary, not sufficient: the join must still be learned (F1).

**F3. Addressing and content must not share a gradient path.**
- A note's key and value came from one latent read. A loss on where reads land then pulled what the note held
  toward what it was asked about: the place probe fell 68 → 14 %.
- Stopping the key gradient at the writer restored it (80 %) and let the property be written.
- Key-value memory networks build keys and values from different views for this reason. Gershman et al. (2025)
  argue the same trade-off for biological memory.
- This lesson is label-free.

**F4. Supervising the output of each hop is stronger than supervising where it reads.**
- The naming loss trains each round's state to say its hop's word. Its gradient reaches both the read and the
  content of the notes read, so it fixes F1 and F2 at once.
- Round-to-round feedback of the named word added nothing: the gate stayed at 0.009, and removing it at exam time
  changed nothing.

**F5. Order has to be relative to the matching notes.**
- Latest needed "the newest among the asked person's notes": E33j 88 %.
- A recency penalty over all notes picks anyone's newest move: E33jd 37 %. It also needed the last-move label.

**F6. Hop counts do not extrapolate; stops defined by content do.**
- On 3-hop compose the strong models answer with bridge 2's home 80–89 % of the time: they stop at the trained
  maximum of two teacher hops.
- Deduce at depth 3 works because its chain ends when a property sentence appears.
- The per-round embeddings are not the cause: they are ~2 % of the state, and remapping them changes nothing.
- Kohli et al. (2026) see looped models extrapolate hops only when trained with a hop curriculum. Saunshi et al.
  (2025) show that m loops can carry m steps.

**F7. Each new task needed its own signal.**
- Latest needed order plus a label. Count needed a summing read plus a label.
- Compose on held-out templates fails for every model, so the relation words are learned as surface forms.

## 3. Which levers survive without generator labels

| lever | label-free? | how to get its signal from available data |
|---|---|---|
| key-gradient stop / separate address read (F3) | yes, architecture | — |
| order among matching notes (F5) | yes, architecture | the order term as built (E33g) |
| summing read for count | yes, architecture | — |
| question density (F2) | yes, data mixture | QA and instruction data where answers depend on far context |
| split read, notes-only after a boundary (F2) | yes, a mask | random cut points in plain text: what follows sees earlier text only through the notes (E33f's writer keeps common words, 69 %) |
| per-round **read** target (F1) | no | (a) maybe not needed beside naming: test R1; (b) distant supervision: the context sentence that contains a written reasoning step's entities |
| per-round **naming** (F4) | no | **the written steps of a reasoning trace**: round r predicts step r's words, and a schedule then removes the steps (iCoT, Coconut, SIM-CoT, LOTUS) |
| last-move / count labels | no | the same: reasoning text that writes "moved last to", or "1, 2, 3 … so three" |
| hop count beyond training (F6) | — | data with varied hop counts (MuSiQue 2–4 hops); E33c's question control stays unused without that pressure |

The common thread: every privileged label we used is something a reasoning trace writes down. So a label-free
version replaces "the generator tells round r the answer word" with "round r predicts the r-th step of the written
reasoning".

- That is self-supervision on text that exists: chain-of-thought corpora (GSM8K-Aug, PrOntoQA, OpenThoughts) and
  multi-hop QA with decompositions (MuSiQue, HotpotQA supporting facts, StrategyQA).
- A schedule then moves the steps from text into the loop, which is iCoT's stepwise removal.
- The board session reached the same conclusion independently.

## 4. Should the writer and the decoder be trained separately?

**What our runs say.**
- What the writer keeps depends on what must pass through the notes. Under the split read it keeps common words
  with no labels at all: E33f, property 69 % (F2). Notes-only continuation is therefore the natural stage-W
  objective.
- The writer is pulled toward whatever the reader currently needs (F2). Sometimes that is destructive (F3).
- The cleanest gain came from partial separation: values trained only through what is read (content), keys only
  through addressing (E33ok).
- A weak separate objective, the linear read-back, failed: its head learned word frequency.

**What the literature says** (verified 18 Oct; links below).
- Compressors pretrained without a task and then read by a decoder work at scale (ICAE, AutoCompressors, Gisting,
  xRAG). But:
  - Reconstruction is a poor proxy. PISCO's autoencoding pretraining reaches ROUGE-L > 0.99 yet barely tracks QA
    quality. In ARC-Encoder, 20 % reconstruction in the pretraining mix is best (41.6), and 100 % is worse than
    none (37.5 vs 39.8). SAC argues that autoencoding does not fit downstream tasks.
  - Freezing the decoder costs a lot: COCOM 0.519 → 0.421 exact match on NQ; PISCO 0.48 → 0.42 on HotpotQA.
  - Compressed notes lose exact detail on surprising content. The gist-token study's "lost if surprise" is the
    common-word problem in reverse. HotpotQA falls with compression: 0.50 → 0.43 at 4×.
  - The working recipe is staged: train the encoder against a frozen decoder with a curriculum, then unfreeze both
    (REFRAG).

**Recommendation: train them separately in stages, and split keys from values.**
1. **Stage W, the writer, label-free.**
   - Objective: the notes of page p must let the decoder continue the text after page p, with no raw tokens of p
     visible (AutoCompressors). A small share (~20 %) is page reconstruction (ARC-Encoder's best mix).
   - Data: plain text plus instruction / QA text.
   - This is what makes notes keep common words when the next text needs them, without any question labels.
   - The decoder is the current coda / reader. It is either frozen (warm from E33) or trained at a low learning
     rate.
2. **Stage R, the reader and loop.**
   - The writer's value path is frozen; its key path (address read) and the reader train.
   - Data: reasoning traces whose steps give per-round targets (§3), then the iCoT removal schedule.
3. **Stage J, joint.** A short joint fine-tune at a low writer learning rate, with the key-gradient stop kept.

Why not fully separate: how a note is found is defined by the reader's questions, so keys must learn from the
reader. What a note holds can be learned without a task. For agent-to-agent use, a stage-W writer is also the
stable message format: notes another model can read without co-training.

## 5. Proposed label-free runs (30M, then 100M), cheapest first

| # | run | question | success / kill | cost |
|---|---|---|---|---|
| R1 | E33qn with the read target off (naming only) | Are read targets needed once each hop is named? | id compose / deduce ≥ 90 % → read targets dropped | 2 GPUs, 2.5 h |
| R2 | **written-chain data**: each answer writes its chain ("Lumo's teacher is Pim. Pim lives in Corvo. So Corvo."). Naming targets are read off the written steps by a rule; then iCoT removal over the last 40 %. | Does text-derived per-hop supervision match the generator labels? | id compose / deduce ≥ 90 % with the steps removed; harder ≥ E33r | data build + 4 GPUs, 4 h |
| R3 | **stage-W writer**: from E33x (property 12 %), only the writer trains, on task-free text (stories, and world documents without their questions) cut at random points, after which earlier text is readable only through the notes (arch e33w) | Does a task-free, notes-only continuation make a writer keep common words? | property probe ≥ 50 % (E33f, trained on questions in split mode: 69 %); place ≥ 74 % | 2 GPUs, 3 h |
| R4 | R3's writer + R2's reader (stage R), then stage J | Does the staged recipe reach the goal without generator labels? | compose / deduce ≥ 70 % at 4k (the original goal) | 4 GPUs, 5 h |
| R5 | real data at 100M: MuSiQue / HotpotQA long-context with decompositions written as chains; GSM8K-Aug | Does it transfer off the synthetic world? | beats a dense 100M model on MuSiQue answer F1 at matched tokens | multi-day |

R2 still uses our synthetic world, but the supervision reaches the model only as text. The same pipeline then
runs on real chain-of-thought and decomposed-QA data (R5).

## 6. What I am not sure about

- **Scale and depth.** Coconut, CODI and LOTUS are GPT-2 scale and up. Quiet-STaR-style learning from plain text
  shows gains only at 1B+. A 30M model may need the written-chain stage for longer.
- **Self-supervised read targets.** Distant supervision by entity match can point at the wrong sentence when names
  repeat. R1 tells us first whether read targets are needed at all.
- **The writer's continuation objective.** Notes that help predict the next page are not the same as notes that
  answer far questions. Stage W may again favour what the next page uses. R3's probe is the check.
- **Hop generalisation.** It remains open for compose. E33c's per-round question control went unused: its gate
  was 0.0013 at checkpoint 445, and the arm was stopped early. With only 1–2 training hops, the rounds solve the
  task by position, and nothing pays for a pointer into the question. Data with varied hop counts (MuSiQue 2–4) is
  the label-free way to create that pressure; whether the control then opens is untested.

## Sources (verified 18 Oct 2026)

Internalising written reasoning:
- iCoT, https://arxiv.org/abs/2405.14838
- Coconut, https://arxiv.org/abs/2412.06769
- CODI, https://arxiv.org/abs/2502.21074
- SIM-CoT, https://arxiv.org/abs/2509.20317
- LOTUS, https://arxiv.org/abs/2606.31779
- Reasoning in Memory, https://arxiv.org/abs/2605.30343
- lens probing of a depth-recurrent model, https://arxiv.org/abs/2507.02199

Reasoning from plain text:
- Quiet-STaR, https://arxiv.org/abs/2403.09629
- RLP, https://arxiv.org/abs/2510.01265
- pause tokens, https://arxiv.org/abs/2310.02226

Compressors:
- ICAE, https://arxiv.org/abs/2307.06945
- AutoCompressors, https://arxiv.org/abs/2305.14788
- Gisting, https://arxiv.org/abs/2304.08467
- xRAG, https://arxiv.org/abs/2405.13792
- COCOM, https://arxiv.org/abs/2407.09252
- PISCO, https://arxiv.org/abs/2501.16075
- ARC-Encoder, https://arxiv.org/abs/2510.20535
- gist-token study, https://arxiv.org/abs/2412.17483
- SAC, https://arxiv.org/abs/2510.08907
- REFRAG, https://arxiv.org/abs/2509.01092

Key-value memory:
- Key-Value Memory Networks, https://arxiv.org/abs/1606.03126
- Gershman, Fiete, Irie 2025, https://arxiv.org/abs/2501.02950

Loops and hop generalisation:
- MAC, https://arxiv.org/abs/1803.03067
- recurrent depth, https://arxiv.org/abs/2502.05171
- looped transformers for length generalisation, https://arxiv.org/abs/2409.15647
- latent thoughts, https://arxiv.org/abs/2502.17416
- grokked composition, https://arxiv.org/abs/2405.15071
- Loop, Think & Generalize, https://arxiv.org/abs/2604.07822

Data:
- HotpotQA, https://arxiv.org/abs/1809.09600
- MuSiQue, https://arxiv.org/abs/2108.00573
- 2WikiMultiHopQA, https://arxiv.org/abs/2011.01060
- StrategyQA, https://arxiv.org/abs/2101.02235
- PrOntoQA, https://arxiv.org/abs/2210.01240
- GSM8K-Aug (from https://arxiv.org/abs/2311.01460)

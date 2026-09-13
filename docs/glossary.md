# Glossary — plain language

One page for the vocabulary that keeps coming back. Written for a human reading quickly, not for the
ledger. Every entry says **what it measures**, **which way is good**, and **what a number roughly means**.

Rule for agents: if a term is used in chat and is not here, either gloss it inline or add it here.
See the `research-comms` skill.

---

## The standing picture

Almost everything in this project is the same story: **a student reading a long book for an exam.**

- The **book** is the input text — too long to keep in front of you.
- The **notebook** is the concept array (also called latents, slots, concept memory): a small set of
  vectors the model writes while reading.
- **Writing** is the encoder — deciding what goes in the notebook.
- **Reading** is cross-attention — looking something up in the notebook.
- The **dense model** is a student allowed to reread any page of the book at any moment. Expensive,
  loses nothing. It is the thing we must beat, or at least match more cheaply.
- A **control** is the same student with something removed — usually the notebook. If his score does
  not change, the notebook was decoration.

The project's recurring failure, in one sentence: **if the model can still see the raw text, it will
never bother to use the notebook.**

---

## Words we use for experiments

| term | plain meaning |
|---|---|
| **arm** | one version of a run in a comparison. Always name it by what it *is* ("the no-memory control"), never by a letter. |
| **control** | the arm with the thing under test removed or replaced, so a difference can be attributed. |
| **dense control / dense baseline** | a normal transformer, same size and data, full attention everywhere. The "reread any page" student. |
| **gate** | a pass mark written down *before* the run. **Success gate** = what would make us continue; **kill gate** = what would make us stop. |
| **ablation** | break one part on a trained model and re-measure. The drop is that part's contribution. |
| **probe** | a small targeted test run after training (e.g. hide a password, ask for it later). |
| **teacher-forced** | the model is shown the correct previous words while scoring. Easy mode; flatters memory. |
| **free-run / generation** | the model writes from its own output. Hard mode; where weak models fall apart. |
| **warm start** | continuing from an existing checkpoint instead of from scratch. |
| **pilot / main run** | small cheap version to decide; the full-budget version afterwards. |

---

## Numbers we quote

| metric | what it measures | good direction | scale |
|---|---|---|---|
| **loss / cross-entropy (CE), in nats** | how surprised the model is by the true next token | lower | 0 = certain. 0.69 = a coin flip. 1.39 = a blind guess among 4. Typical small-LM values: 2.5–4.5. A 0.05 difference is small; 0.5 is large. |
| **perplexity (ppl)** | the same thing as "how many options is it choosing between" | lower | `exp(loss)`. ppl 50 ≈ picking among 50 words. |
| **bits per byte (BPB)** | loss converted to a per-byte unit, so two models with different tokenizers can be compared | lower | ~1.5 = weak small model, ~1.0 = decent small model. The only fair ruler against public models. |
| **Δ (delta) of an ablation** | how much worse the model gets when we break something — the notebook's rent | higher = the part matters | in nats. Under 0.01 means "not used". 0.25 is real but modest. Over 1.0 is load-bearing. |
| **Δzero** | Δ when the concepts are replaced with zeros | higher | |
| **Δshuffle / Δperm** | Δ when the concepts are shuffled between examples or positions — harder than zeroing, because it removes *this* document's information while keeping the general statistics | higher | the honest version of the test. |
| **reach ablation** | Δ when we shorten only how far back a layer is allowed to look, on an already-trained model | higher = long range matters | our headline instrument for "is long context actually used?" |
| **RankMe** | how many genuinely different directions the notebook uses (`exp` of the entropy of the singular values) | higher | out of the slot count or width. Near 1 = **collapse**: every page says the same thing. 200 of 2048 = healthy. |
| **passkey accuracy** | hide a password early in a long text, ask for it at the end | higher | 0.0 means the model cannot retrieve at all. |
| **RULER / RULER-lite** | a small suite of long-context retrieval tasks: passkey, multi-key lookup, variable tracking, frequent-word counting | higher | |
| **STS-B (zero-shot)** | do two sentences the model considers similar match human similarity ratings — our semantic-quality reference | higher | 0.714 is our best from-scratch result. |
| **σ (sigma)** | how many standard errors away from zero a difference is | higher = more certain it is real | under 2 = noise. Over 10 = definitely real (but possibly tiny). |

**Real and tiny are different things.** A difference can be 25σ and still be worth nothing. Always
report both: is it real, and is it big enough to matter.

---

## Architecture words

| term | plain meaning |
|---|---|
| **concepts / latents / slots** | the small set of vectors the long input is compressed into. The notebook pages. |
| **cross-attention** | the mechanism for looking things up in the notebook; the "read". |
| **self-attention** | tokens looking at each other. |
| **sliding-window attention (SWA)** | each token can only see the last *K* tokens. Cheap; limits how far back a single layer sees. |
| **reach** | how far back information can actually travel through a stack of windows: roughly `layers × window`. |
| **bypass** | the raw-text path around the notebook. Its existence is why the notebook gets ignored. |
| **KV cache** | what a model must keep in memory per token while generating. Our compression story is mostly about shrinking this. |
| **encoder–decoder** | one stack summarises, another stack writes. Most of our designs are this, even when they look like one model. |
| **Perceiver / Perceiver AR** | architecture family where a long input is read into a small fixed set of latents by cross-attention. |
| **BiXT** | a Perceiver variant where tokens and latents refine each other both ways. |

---

## Where things are written down

| you want | look in |
|---|---|
| what we are doing now | `docs/1_Strategy_and_Plans/agenda.md` |
| the plan for one experiment, frozen before the run | `docs/experiments_specs/<lifecycle>/<ID>.md` |
| what every run produced | `docs/2_Experiments_Registry/master_experiment_log.md` |
| the story of one run and what it meant | `docs/2_Experiments_Registry/run_reports/` |
| what changed in the code | `CHANGELOG.md` |

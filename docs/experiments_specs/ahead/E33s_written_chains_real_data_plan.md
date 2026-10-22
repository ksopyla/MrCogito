# E33s — Implementation Plan

- **Spec:** [E33s_written_chains_real_data.md](E33s_written_chains_real_data.md) · **Status:** approved (author aligned
  2026-10-21) · building P0 and P1
- **Authored by:** `implementation-plan` · for → `research-implement`

## 1. Source & fit

- **Origin:** R2 of the multi-hop review (`docs/4_Research_Notes/e33_multihop_review_label_free_20261018.md` §5).
  - 30M, synthetic world, written-chain data (`build_text_checks_data.py --written_chains`).
  - The rule `parse_written_chain` reads each step's output word off the text.
  - iCoT step removal over three stages.
  - Compose 92 %, deduce 99 % at 4k, with no generator labels.
- **Synthesis verdict:** Adopt the internalisation line (iCoT 2405.14838; SIM-CoT 2509.20317's per-step decoder ≈
  our naming lens; LOTUS 2606.31779's looped model with per-loop LM-head targets), adapted to a notebook read.
  We drop latent-token replacement: the loop rounds are the latent steps.
- **Architecture mapping:** data and loss only. The model is the existing `e33qn_plain` (E33 loop + naming lens +
  per-round read target) and `dense_plain`, at the main tier with a 16k vocabulary. No new nn module.
- **Boldness check:** the plan keeps the claim intact. Real reasoning text is the only supervision, and the
  notebook loop is set against dense under identical chains. Nothing is replaced by a generator label or a safer
  model.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `PerceiverARLM`, latent-memory writer, loop, naming lens (`message_loop_name_lens`), read target (`message_loop_read_aux`) | reuse as-is | `nn/perceiver_ar_lm.py`, `nn/latent_memory.py` |
| arches `e33qn_plain`, `e33x_plain`, `dense_plain`; tiers `screen`, `main` | reuse as-is | `evaluation/text_checks.py` |
| `read_targets` column, collator `read_targets_tensor` | reuse as-is | `data/data_collators.py` |
| `_save`, `_row`, `train_tokenizer`, `language_rows`, `_mean_len`, `_sha` | reuse via import | `scripts/build_text_checks_data.py` |
| `fit_params` vocabulary | extend: take the data's tokenizer size | `scripts/run_text_checks.py:plan_train_job` |
| eval job | extend: real-QA data runs the new exam | `scripts/run_text_checks.py:plan_eval_job` |
| teacher-forced exact scoring | reuse pattern | `evaluation/text_checks_eval.py:_answer_ids/_score` |
| **real multi-hop data** | **new, reusable** | `data/real_multihop.py` + `scripts/build_real_multihop_data.py` |
| **real-QA exam** | **new, reusable** | `evaluation/real_multihop_eval.py` |
| phase-P long-document mix (100M) | new recipe + `wikipedia_en` source | `data/mix_recipes/e33s_realtext_4k_v1.json`, `scripts/pretokenize_mix.py` |

## 3. Forward pass (unchanged model; shapes at the main tier)

Symbols: `B` = 4 rows per GPU, `N` = 4096 tokens, `H` = 896, `V` = 16384, `P` = pages, `C` = 32 × P notes.

```
(B, N)        → table V×256, up-projection           → (B, N, H)
              → prelude: 3 local layers (window 16)   → (B, N, H)
(B, N, H)     → writer: P = 21 pages of 256 (stride 192), 32 latents each → keys/values (B, C=672, 64)
(B, N, H)     → loop × 4: global read [raw window 256 ‖ closed notes], one softmax, 14 heads × 64; then 1 local layer
                 at answer positions: read target (round r → step r's span) and naming lens (round r's state →
                 frozen final_norm / lm_head → step r's word tokens)
              → coda: 7 local layers → logits (B, N, V)
```

Dense: the same embedding, 12 full-attention layers, logits. No notes, no loop, no per-round targets.

## 4. Inputs & data

### `data/real_multihop.py` (library, no I/O beyond the HF loaders)

- **`QAItem`:** dataset, id, question, answer, aliases, paragraphs `[(title, text, supporting)]`, and `steps`
  (the chain the dataset writes: sentences, or equations for maths).
- **Loaders** (`datasets.load_dataset`, verified 21 Oct):

  | dataset | splits | chain steps |
  |---|---|---|
  | `dgslibisey/MuSiQue` | train 19.9k / validation 2.4k | per hop, the sentence of the supporting paragraph (`paragraph_support_idx`) that contains the hop's sub-answer |
  | `hotpotqa/hotpot_qa` distractor | train 90.4k / validation 7.4k | the `supporting_facts` sentences |
  | `framolfese/2WikiMultihopQA` | train 167.5k / validation 12.6k | the `supporting_facts` sentences |
  | `whynlp/gsm8k-aug` | train 385.6k / test 1.3k | `<<a*b=c>>` → `a*b=c` |

  - The datasets' annotations are used only here, to write the chain text.
  - Every training target is derived from that text by the rule.
- **`parse_chain(steps, answer, question, body)`, the rule, generalised from `parse_written_chain`:**
  - A step's output is the longest run of up to 5 consecutive words it shares with the next step. The run must
    contain a word that is neither a function word nor in the question. Ties go to the rarest in the body.
  - The last step's output is the answer string.
  - Equations: the output is the number after `=`.
  - Steps are reordered so the step containing the answer comes last. This is a text rule.
  - Questions where no output is found (e.g. comparisons) keep only the answer target.
- **`assemble(item, target_tokens, ntok, rng, pool, depth=None, remove_evidence=False)`:**
  - The paragraphs are joined as `"{title}\n{text}"` with blank lines between them.
  - Distractors from `pool` (the same dataset and split) pad to `target_tokens`.
  - The supporting paragraphs go at random positions, or at `depth` for exams.
  - `remove_evidence` swaps them for distractors, giving the floor twin.
  - It returns the prompt ending in `"\n\nQuestion: {q}\nAnswer:"`.
- **`written_answer(steps, answer, keep)`:** `keep` = all | drop1 | none.
  - all / drop1 → `" {s1}. {s2}. … So {answer}."`, with the first step dropped for drop1;
  - none → `" {answer}."`.
- **`chain_targets(text, offsets, question, derived)`:** the quads `[answer_pos, r, t0, t1]` in row positions
  (BOS = 0), the same format as `read_targets` in `build_text_checks_data.py`.
  - Steps are found before the question (facts in the context).
  - Maths: in the written answer after the answer position. That gives naming targets only while the steps are
    written (stages 1–2); a read target on a later span is masked by causality, which is harmless.

### `scripts/build_real_multihop_data.py` (CLI; writes the text-checks data layout, so the runner is unchanged)

- **`--rule_check`** (P0): build the chains for all training questions and print coverage per dataset and question
  type. For MuSiQue, also the rule-vs-sub-answer agreement. Writes `rule_check.json`. CPU only.
- **`--train_tokenizer N`:** a 16,384-token BPE (`train_tokenizer`) on a sample of Wikipedia + FineWeb-Edu text,
  QA paragraphs and GSM8K-Aug questions.
- **`--pretrain_tokens T`:** phase-P rows for the pilot.
  - Wikipedia (`wikimedia/wikipedia` 20231101.en) and FineWeb-Edu (`HuggingFaceFW/fineweb-edu` sample-10BT),
    streamed, in rows of 4,096.
  - Writes `text/{train,eval}`, `manifest.json` and `text_checks_meta.json`.
  - The 100M run instead uses `pretokenize_mix.py` with the 4-source recipe.
- **`--stage all|drop1|none|answers --qa_tokens T --replay 0.3`:** phase-M rows.
  - `qa/{train,eval}` with `read_targets`, and `text/` replay with empty `read_targets`.
  - The manifest weights match the token shares.
  - `answers` = the no-chain control: plain answers and no quads.
- **`--exams --lengths 4096 16384 [65536] --per_dataset n`:** frozen `eval/realqa.jsonl`.
  - Fields per item: id, dataset, length, depth bin, prompt, answer, aliases, removed prompt.
  - Built from dev / test only, with a dev distractor pool.
- **Meta:** `text_checks_meta.json` gets `kind: "real_multihop"`, `vocab`, `mix.mean_row_tokens`, the sha256s and
  the stage.

## 5. Loss & training objective

- **Loss:** unchanged (`PerceiverARLM.forward`).
  - Next-token CE on all text, written chains included in stages 1–2;
  - `+ 0.5 ·` per-round read target;
  - `+ 0.3 ·` naming lens (decoder weights frozen in the lens).
- **Dense:** next-token CE only. The `read_targets` column is ignored.
- **No-chain control:** `e33x_plain` (no read target, no naming) on `--stage answers` data.
- **Schedule:** three warm-started stages of `--tokens` each, at the main-tier learning rate / 3, cosine, as in R2.

## 6. Config & launch

- **No new model config.**
- **Runner, two backward-compatible changes:**
  - `plan_train_job` sizes the FFN with `vocab = len(data tokenizer)`. Text-checks data has 4,096, so it is
    unchanged.
  - `plan_eval_job` runs `evaluation/real_multihop_eval.py` when `meta.kind == "real_multihop"`:
    - finals: `final_realqa.json` (the 4k / 16k / 64k cells, exact over aliases, with the removed twin);
    - `final_realqa_notebook_off.json` for notebook arches;
    - curves at 4k on a 300-item subsample;
    - no copy probe.
- **Pilot P1, Polonez, 2 GPUs, screen tier:**
  ```
  uv run python scripts/run_text_checks.py plan --phase train --tier screen --arches e33qn_plain --host polonez \
    --gpus 2 3 --data $T/e33s_pilot_pretrain --out Cache/text_checks/e33s_p1_pre --tokens 3e8
  # then the three stages from its final: --data $T/e33s_pilot_{all,drop1,none} --tokens 5e7 \
  #   --extra_env MODEL_NAME_OR_PATH=<prev final> LEARNING_RATE=0.000333 --eval_lengths 4096 16384
  ```
- **Main run:** `--tier main`, phase P `--tokens 1.5e9`, stages `--tokens 1.5e8`; dense with `--arches dense_plain`.

## 7. Tests & smoke

`tests/test_real_multihop.py`, offline, on hand-built items:
- the rule on a MuSiQue-style chain (bridge "Steve Hillage", answer last);
- the reorder that puts the answer step last;
- a comparison question (answer target only);
- equations;
- `assemble` (length, supporting paragraphs present or removed, prompt ends in `Answer:`);
- `chain_targets` spans decode to the expected words;
- the three `keep` modes;
- eval: teacher-forced exact over aliases, and token F1 / EM normalisation on SQuAD cases.

Smoke: build a tiny pilot (2k rows) locally, run the runner plan with `--tier smoke`, and train a few steps on the
laptop's MPS.

## 8. Risks & tradeoffs

- **Real steps are long sentences.** The naming lens skips spans over 4 tokens, but output words are entities of
  1–5 words, so most fit. Coverage is measured in P0.
- **Comparison questions have no bridge**, so they get the answer target only, by design. Coverage is reported by
  type.
- **A 16k vocabulary at 30M** takes ~14M parameters for the embedding and LM head. The pilot is a pipeline check,
  not a result. At 100M the FFN keeps 1,728.
- **Exact match on real answers is strict.** F1 is computed by greedy generation on a 300-item subsample at 4k. The
  16k / 64k gates use exact (amend the spec's S3 wording before the main run).
- **Cheapest early kill:** P0 coverage (K1), then P1's MuSiQue 4k exact above its removed floor.

## 9. Code sketches

```python
# sketch: data/real_multihop.py
@dataclass
class QAItem:
    dataset: str; id: str; question: str; answer: str; aliases: list
    paragraphs: list            # [(title, text, supporting: bool)]
    steps: list                 # the dataset's chain, as text (sentences or equations)

def parse_chain(steps, answer, question, body) -> list[tuple[str, str]]: ...   # (step, output) or []
def assemble(item, target_tokens, ntok, rng, pool, depth=None, remove_evidence=False) -> str: ...
def written_answer(steps, answer, keep: str) -> str: ...
def chain_targets(text, offsets, question, derived, after_answer=False) -> list[int]: ...  # flat quads
```

## 10. Pre-flight results (21–22 Oct 2026, Polonez)

### P0 — rule coverage (CPU, full training sets)

| dataset | questions | coverage, sentences only | coverage, with the next paragraph's title |
|---|---|---|---|
| MuSiQue | 19,598 | 68 % (2 / 3 / 4 hops: 76 / 52 / 28 %) | **83 %** (90 / 69 / 44 %) |
| HotpotQA | 90,425 | 58 % (bridge 56 %) | **67 %** (bridge 68 %) |
| 2WikiMultiHopQA | 167,454 | 67 % | **74 %** (compositional 98 %, inference 99 %, bridge-comparison 38 %) |
| GSM8K-Aug | 385,531 | 100 % | 100 % |

- **The kill measure (MuSiQue + HotpotQA bridge):** 58.7 % with the sentence-only rule, just under the 60 % line.
  With the title-aware rule (829fa69), 70.8 %, which is above the kill line but under the 80 % target.
  - The title is text the model sees above each paragraph. A later fact often names its subject only by that title,
    or by a pronoun.
- **MuSiQue's bridge agrees with its own per-hop sub-answer** 90 % of the time (20,222 bridges).

### P1 — 30M pipeline pilot (21:12–23:08 UTC, GPUs 2–3)

- **Setup:** `e33qn_plain`, screen tier with a 16k vocabulary (FFN 448).
  - Pretraining: 0.3B tokens of Wikipedia + FineWeb-Edu.
  - Then stages all → drop1 → none, 0.05B tokens each: 17.5k / 17.8k / 18.4k QA rows of ~2k tokens, 30 % replay.
- **Eval loss:** 3.95 after pretraining, then 3.46, 3.32 and 3.24 after the three stages.
- **The pipeline ran end to end:** every stage trained, and every exam and control ran. Pass.
- **No measurable skill:** every score sits on its no-evidence floor.

  | exam at 4k, after stage 3 | exact | evidence removed | notebook off | greedy F1 |
  |---|---|---|---|---|
  | MuSiQue | 0 % | 0 % | 0.7 % | 2.8 % |
  | HotpotQA | 2.7 % | 3.0 % | 3.0 % | 5.0 % |
  | 2WikiMultiHopQA | 4.7 % | 4.7 % | 3.3 % | 6.2 % |
  | GSM8K | 2.0 % | — | 2.0 % | — |

  The model answers with typed guesses ("The New York Times", "no", "London", "4"), so the second pass condition
  (MuSiQue above its floor) fails.
- **Reading:** at 30M and 0.35B tokens of real text, real multi-hop QA is far out of reach. The main run's gates are
  absolute and have no measured ceiling yet (design rule 4). Before spending ~280 GPU-h, a ceiling at the 100M class
  is needed: a pretrained ~100M model fine-tuned on the same written-chain data.

### Ceiling probe — SmolLM2-135M (22 Oct 2026, 13:20–16:32 UTC, Polonez GPUs 2–3)

- **Setup:** a pretrained ~100M-class model (SmolLM2-135M, 2T pretraining tokens) is fine-tuned on the same
  written-chain stages: all → drop1 → none, 0.05B tokens each, with SmolLM2's tokenizer
  (`scripts/hf_ceiling_finetune.py`).
- **Exams:** the pilot's frozen exams at 4k, 300 items per dataset.
- **Scoring:** stages 1–2 in chain mode (the answer after "So"); stage 3 answer-only, with the evidence-removed
  floor.

| exam at 4k | untuned, greedy F1 | stage 3: exact | evidence removed | stage 3: greedy F1 | chain mode after stage 1 / stage 2: greedy F1 |
|---|---|---|---|---|---|
| MuSiQue | 6.4 % | 4.7 % | 1.0 % | 11.0 % | 4.5 / 6.5 % |
| HotpotQA | 17.0 % | 18.3 % | 7.0 % | 29.0 % | 13.6 / 16.6 % |
| 2WikiMultiHopQA | 19.8 % | 22.0 % | 23.7 % | 26.9 % | 25.2 / 26.1 % |
| GSM8K | 1.0 % | 3.0 % | — | 3.2 % | 0.6 / 1.1 % |

**Stage 3 by question type:**
- **HotpotQA bridge** (n = 240): exact 16.2 % against a 2.9 % floor, F1 27.2 %. This is the only exam with clear
  evidence use.
- **HotpotQA comparison:** 26.7 % against 23.3 %.
- **2Wiki comparison / bridge-comparison:** 47.5 / 43.2 % exact against 60.7 / 43.2 %. This is the yes/no prior.
- **2Wiki compositional:** 4.1 % against 1.6 %. **2Wiki inference:** 0 %.
- **MuSiQue 2 / 3 / 4 hops:** 4.6 / 5.8 / 2.4 % against 1.7 / 0 / 0 %.

**Reading:**
- At the 100M class, real multi-hop QA is near the floor even with 2T tokens of pretraining. Only HotpotQA
  bridge questions show a usable signal: +13 points of exact over the floor.
- Writing the chain out does not help at this size. The chain-mode answers are below the answer-only stage 3.
- The from-scratch E33s main run gets 2B tokens, three orders of magnitude fewer than the ceiling model. Its MuSiQue
  gates (S1–S3) cannot be met, or even resolved from noise (300 items: SE ≈ 1–2.5 points).
- **As specced, E33s would not be informative.** Decision for the author: re-scope or put on hold before the 100M
  run.

---
pretty_name: "Cogito Text World: can a small language model use the facts it reads? (1k–128k)"
license: cdla-sharing-1.0
language:
- en
task_categories:
- text-generation
- question-answering
annotations_creators:
- machine-generated
language_creators:
- machine-generated
multilinguality:
- monolingual
source_datasets:
- extended|roneneldan/TinyStories
- extended|SimpleStories/SimpleStories
size_categories:
- 100K<n<1M
tags:
- cogito-capability-checks
- synthetic
- benchmark
- evaluation
- long-context
- length-generalization
- needle-in-a-haystack
- fact-retrieval
- multi-hop
- state-tracking
- reasoning
- small-language-models
- pretraining
- tinystories
configs:
- config_name: eval_id
  default: true
  data_files:
  - split: test
    path: eval/id/*.parquet
- config_name: eval_harder
  data_files:
  - split: test
    path: eval/harder/*.parquet
- config_name: eval_paraphrase
  data_files:
  - split: test
    path: eval/paraphrase/*.parquet
- config_name: train_world
  data_files:
  - split: train
    path: train/world/*.parquet
- config_name: train_stories
  data_files:
  - split: train
    path: train/stories/*.parquet
- config_name: dev_world
  data_files:
  - split: validation
    path: dev/world/*.parquet
- config_name: dev_stories
  data_files:
  - split: validation
    path: dev/stories/*.parquet
---

# Cogito Text World

> **Part of [Cogito Capability Checks](https://huggingface.co/collections/ksopyla/cogito-capability-checks-6aca2db88ba9d2714727489f)** — *does your model actually use what it read?*
> Small synthetic tests with exact answers, guessing floors and a length ladder (1k → 128k tokens).
> Project page: [ai.ksopyla.com/projects/cogito-capability-datasets](https://ai.ksopyla.com/projects/cogito-capability-datasets/)

**TL;DR.** Train a small language model **from scratch** on simple English stories with facts woven in,
then ask it about those facts in documents of **1k to 128k tokens** (trained at 4k). Seven question
types, from copying a sign to chaining rules, each with an exact answer, a guessing rate below 10 %
and an evidence-removed twin that exposes shortcuts. Built to compare architectures (attention,
memory, recurrence) at equal data and size.

| | |
|---|---|
| **Tests** | quote · look up · pick among look-alikes · latest of many moves · two-hop chain · count · deduce |
| **Eval** | 29,400 frozen items: `eval_id` 18,200 at 1k–128k · `eval_harder` 5,600 · `eval_paraphrase` 5,600 (4k, 16k) |
| **Train** | ≈1.0B tokens of token ids: `train_world` (documents with a question and answer) + `train_stories` (plain stories) |
| **Scoring** | exact answer, one forward pass; guessing rate per item; evidence-removed twin per item |
| **Tokenizer** | bundled 4,096-token byte-level BPE (`tokenizer/`) |
| **Version** | `text-world-v2` (2026-10-10) · older: `revision="text-world-v0"` |
| **License** | CDLA-Sharing-1.0 (inherits TinyStories) |

## The questions

Every item is a story-world document: real simple stories whose characters are renamed to an invented
cast, fact sentences woven in, one question at the end. Places and rules are invented words, so the
answer never comes from prior knowledge.

| level | `task` | example question | answer | guessing |
|---|---|---|---|---|
| T1 | `quote` | What did the sign of Lumo say? (12 signs) | 4–6 words, copied exactly | 1/12 |
| T2 | `lookup` | Where does Lumo live? (12 homes stated) | an invented place | 1/12 |
| T3 | `keyed` | Where does Lumo live? (16 homes, a quarter look-alike names) | an invented place | 1/16 |
| T4 | `latest` | Where does Lumo live now? (10 moves each, 5 people) | the last place | 1/11 |
| T5 | `compose` | Where does the teacher of the teacher of Lumo live? | follow two links, then the home | 1/12 |
| T6 | `count` | How many times did Lumo visit the market? | zero … ten | 1/11 |
| T7 | `deduce` | Pim is a wump. Every wump is a tove. Every tove is shiny. … What is Pim like? | one of 12 properties | 1/12 |

```
… Once upon a time Lumo found a red ball … Lumo lived in Tavimere. …
… Later that spring, Lumo moved to Kodaford. … (filler up to 1k–128k tokens) …
Question: Where does Lumo live now?
Answer: Kodaford.
```

## Quick start

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

exams = load_dataset("ksopyla/cogito-text-world", "eval_id", split="test")
item = exams[0]
print(item["task"], item["length"], item["answer"], item["floor"])
print(item["prompt"][-300:])

repo = snapshot_download("ksopyla/cogito-text-world", repo_type="dataset", allow_patterns="tokenizer/*")
tok = AutoTokenizer.from_pretrained(f"{repo}/tokenizer")          # 4,096-token BPE, <|bos|> <|eos|> <|pad|>
train = load_dataset("ksopyla/cogito-text-world", "train_world", split="train", streaming=True)
print(tok.decode(next(iter(train))["input_ids"])[-300:])
```

## How to score

1. Feed `BOS + prompt + answer` once (teacher forcing). The item is **correct** when the model's top
   token equals the gold token at every answer position, period included — exactly when greedy
   decoding would write the answer.
2. Optional **picked candidate**: of `candidates`, the one whose first token is most likely at the
   answer position (first tokens are distinct, except `four`/`five` in `count` and the 64-person
   harder `keyed`). Shows *found the right fact* apart from *copied it exactly*.
3. Score `prompt_removed` too: a model that really does the task falls to the guessing rate there.

**Read the scores with four rules** (they decide most conclusions):

- **Far facts are the headline.** In `quote`, `lookup` and `keyed` the fact sits early, in the middle
  or late (`depth`: 5 %, 50 %, 95 % of the document, balanced). At 1k–4k a late fact is 50–200 tokens
  before the question, inside any local attention window, so report **far** (`early` + `middle`) as
  the main score and **near** (`late`) separately. A model can score 49 % on near facts and 0 % on far ones.
- **Use the effective floor.** The floor is the best score without doing the task: `floor` for most
  tasks, but **20 %** for `latest` (follow every move, ignore whose it is).
- **Learned vs passed.** *Learned*: the headline is above the effective floor by 10 points or 3 standard
  errors, whichever is larger. *Passed*: ≥ 75 % exact, with the evidence-removed twin at the floor.
- **Past the training length**, a model with no long-text method measures position extrapolation, not
  memory. Compare against a local-window model of the same size, not just against itself at 4k.

**Train short, test long:** train at 4,096 tokens, read `eval_id` at every length; longer documents
add filler, not facts. Reference scorer: `evaluation/text_checks_eval.py` in
[github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito).

## What to expect (findings so far)

From the first 18 small models trained from scratch on this data (mostly 30M parameters, 1.5B tokens;
Oct 2026, draft protocol). Only findings that held up are listed:

- **Run several seeds.** At 30M / 1.5B tokens a dense Transformer learns retrieval in some runs only: of four runs, one
  passed `quote`, one learned four tasks (quote 86 %, lookup 39 %, keyed 35 %, latest 32 % at 4k), two
  learned nothing. Skills switch on abruptly late in training. Answers are ≈ 0.1 % of training tokens
  (one question per document); a mix with more questions per document is being tested.
- **`compose` and `deduce` are open frontier:** every run is at guessing, including the probability
  on the right candidate.
- **The paraphrase split separates meaning from surface:** dense lookup drops from 37–48 % to 0–1 %
  when the fact uses a sentence shape never seen in training.
- **Compressed memories lose exact wording:** two memory architectures we tested scored 0 % on far
  `quote` items; their quote scores came only from signs inside the local window.
- **Without a long-text method, dense reads 0 % at 16k** on every task.

Live results and the protocol: [project page](https://ai.ksopyla.com/projects/cogito-capability-datasets/).

## The family: Cogito Capability Checks

One goal: **check whether a model stores and uses facts from its context, not just predicts nearby
words.** Each dataset plants facts with known answers, adds filler, asks a question at the end, and
labels how hard guessing is. Length is a ladder, so you see *where* a model stops working.

| dataset | input | the question it answers | lengths |
|---|---|---|---|
| [`cogito-probe-bits`](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | symbol keys and values | How many key→value facts did it keep? (memory capacity in bits) | 1k–32k |
| [`cogito-probe-bind`](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | people with attributes | Does it keep *who has what* together, and follow one link? | 1k–32k |
| [`cogito-probe-arith`](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | nested bracket arithmetic | Did it store the expression tree, not only the result? | 1k–32k |
| [`cogito-probe-props`](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | short facts in fluent filler | Does it keep the facts and ignore the padding? | 1k–32k |
| [`cogito-text-world`](https://huggingface.co/datasets/ksopyla/cogito-text-world) | simple English stories | Can a small LM trained from scratch quote, look up, track, chain, count and deduce facts in a story? | 1k–128k |

**Start with** a probe if you test a memory or compression module on a pretrained tokenizer;
**start with** `cogito-text-world` if you compare architectures trained from scratch on language.

## Related work and inspirations

- **Needle in a Haystack** (Kamradt, 2023) and **passkey retrieval** (Mohtashami & Jaggi, 2023, [arXiv:2305.16300](https://arxiv.org/abs/2305.16300)) — one fact in long filler; here: many facts, labelled bits, decoys.
- **RULER** (Hsieh et al., 2024, [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)) — synthetic long-context tasks (multi-key, variable tracking) for pretrained LMs; these sets add from-scratch training data and evidence-removed controls.
- **BABILong** (Kuratov et al., 2024, [arXiv:2406.10149](https://arxiv.org/abs/2406.10149)) and **bAbI** (Weston et al., 2015, [arXiv:1502.05698](https://arxiv.org/abs/1502.05698)) — facts hidden in long natural text.
- **Zoology / MQAR** (Arora et al., 2023, [arXiv:2312.04927](https://arxiv.org/abs/2312.04927)) — multi-query associative recall as an architecture test.
- **Lost in the Middle** (Liu et al., 2023, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) — answers depend on where the evidence sits.
- **Binding IDs** (Feng & Steinhardt, 2023, [arXiv:2310.17191](https://arxiv.org/abs/2310.17191)), **ListOps** (Nangia & Bowman, 2018, [arXiv:1804.06028](https://arxiv.org/abs/1804.06028)), **PrOntoQA** (Saparov & He, 2022, [arXiv:2210.01240](https://arxiv.org/abs/2210.01240)) — binding, tree structure and rule chains.
- **TinyStories** (Eldan & Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) and **SimpleStories** (Finke et al., 2025, [arXiv:2504.09184](https://arxiv.org/abs/2504.09184)) — the simple language under `cogito-text-world`.

<details>
<summary><b>Configs, fields, how it was built, limitations</b></summary>

**Configs**

| config | split | rows | content |
|---|---|---|---|
| `eval_id` | test | 18,200 | in-distribution exams at 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k (400 per task and length up to 16k, 200 above) |
| `eval_harder` | test | 5,600 | harder dials (24 signs, 16 homes, 64 people in keyed, 16 moves, 3 hops, depth-3 rules) at 4k and 16k |
| `eval_paraphrase` | test | 5,600 | fact sentences in templates never used in training, at 4k and 16k |
| `train_world` | train | 394,287 | world documents with question + answer, token ids |
| `train_stories` | train | 109,864 | plain stories packed to 4,096 tokens, token ids |
| `dev_world`, `dev_stories` | validation | 128 / 64 | held-out rows for development loss |

Intended mix: ≈45 % story tokens, ≈55 % world tokens (row weights 0.218 / 0.782; mean row 1,981 tokens).

**Eval fields:** `prompt` (ends in `Question: …\nAnswer:`), `answer` (leading space, answer, period:
`" Kodaford."`), `candidates` (all answers of the asked type in the document), `floor` (guessing rate),
`prompt_removed` (evidence-removed twin), `task`, `level`, `split`, `length` (target tokens), `depth`
(`early` / `middle` / `late` / `spread`), `seed`, `id`, `version`, `n_tokens` (BOS + prompt + answer, bundled tokenizer), `meta` (JSON: task dials,
assembly).

**Train fields:** `input_ids` (int32, `<|bos|> … <|eos|>`), `attention_mask`, `special_tokens_mask`.

**Build**
- Language: TinyStories and SimpleStories train splits (3.9M stories); their validation/test splits
  (36,757 stories) are used only as exam filler and for `dev_stories`.
- Cast: invented names from a syllable grammar; one name in five is reserved for the exams, so no exam
  name appears in training. Filler characters are renamed to cast members; filler sentences that pair a
  cast member with a fact word (lives, moved, teacher, sign, …) are dropped.
- Facts: ≥ 4 templates per relation in training, one more held out for `eval_paraphrase`.
- Audit (model-free, all 29,400 items): a rule reader answers 100 %; shortcut readers sit at the
  guessing rate, except `latest`, where tracking moves without reading the name gives 1 in 5.
- Deterministic: each item is a pure function of (version, split, task, length, index) and the story
  pool; code in `generator/`, hashes in `text_checks_meta.json`.

**Limitations**
- Children's-story language and templated facts: this tests the *mechanisms* of retrieval and
  composition in small models, not world knowledge or open-domain QA.
- Training rarely asks far back: ≈ 20 % of training questions have their fact more than 1k tokens
  before the question, ≈ 4.5 % more than 2k (documents are 256–4,096 tokens, facts placed uniformly).
- Fact sentences differ in style from the filler, so finding *a* fact is easier than in natural text;
  decoys and the evidence-removed twins guard against that.
- Stories were written by LLMs (TinyStories: GPT-3.5/4; SimpleStories: other LLMs) and inherit their biases.
- The protocol is a calibrated draft; future versions get a new tag, old ones stay loadable.

**Attribution.** CDLA-Sharing-1.0, the license of TinyStories (share-alike). Contains text derived from
TinyStories (Eldan & Li, 2023, CDLA-Sharing-1.0) and SimpleStories (Finke et al., 2025, MIT).
</details>

## Changelog

| date | version | change |
|---|---|---|
| 2026-10-13 | card | Reading rules (far facts, effective floor, learned vs passed) and findings from the first 18 models. Data unchanged. |
| 2026-10-10 | `text-world-v2` (current) | Every guessing rate below 10 % (12 signs, 12 homes, 10 moves, 12-person teacher chain, counts 0–10, deduce asks *which* property); candidate places start with different letters, so the picked-candidate score is clean; cut-off filler sentences dropped; train mix ≈1.0B tokens. |
| 2026-10-08 | `text-world-v1` | Not published. Fixed the five shortcuts found in v0 (below). |
| 2026-10-06 | `text-world-v0` | First release, kept as `revision="text-world-v0"`. **Known shortcuts, do not use for conclusions:** `lookup` solvable 98 % by copying the only place word; `latest` 100 % by the busiest mover's last place; `count` 51 % by counting all visits; two-way "sister" made `compose` ambiguous; filler stated the `deduce` property in 19 % of items. |

## Citation

Please cite the family, and the dataset you used:

```bibtex
@misc{sopyla2026cogitochecks,
  title        = {Cogito Capability Checks: synthetic tests of whether models store and use facts from long context},
  author       = {Sopyła, Krzysztof},
  year         = {2026},
  howpublished = {\url{https://ai.ksopyla.com/projects/cogito-capability-datasets/}},
  note         = {Datasets: cogito-probe-bits, -bind, -arith, -props; cogito-text-world}
}

@misc{cogito_text_world,
  title        = {Cogito Text World: a story-world dataset for capability checks of small language models},
  author       = {Sopyła, Krzysztof},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/ksopyla/cogito-text-world}},
  note         = {Cogito Capability Checks; data version text-world-v2}
}
```

Author: [Krzysztof Sopyła](https://ai.ksopyla.com) · code: [github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito).

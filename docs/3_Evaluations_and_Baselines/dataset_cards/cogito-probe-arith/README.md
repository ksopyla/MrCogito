---
pretty_name: "CogitoProbe-Arith: nested arithmetic and bracket matching in long context"
license: apache-2.0
language:
- en
task_categories:
- question-answering
- text-generation
annotations_creators:
- machine-generated
language_creators:
- machine-generated
multilinguality:
- monolingual
source_datasets:
- original
size_categories:
- 10K<n<100K
tags:
- cogito-capability-checks
- synthetic
- benchmark
- evaluation
- long-context
- needle-in-a-haystack
- length-generalization
- memory
- arithmetic
- bracket-matching
- dyck
- structure
- cogito-probe
- arith
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# CogitoProbe-Arith: nested arithmetic with mixed brackets

> **Part of [Cogito Capability Checks](https://huggingface.co/collections/ksopyla/cogito-capability-checks-6aca2db88ba9d2714727489f)** — *does your model actually use what it read?*
> Small synthetic tests with exact answers, guessing floors and a length ladder (1k → 128k tokens).
> Project page: [ai.ksopyla.com/projects/cogito-capability-datasets](https://ai.ksopyla.com/projects/cogito-capability-datasets/)

**TL;DR.** Several nested `+ - *` expressions with mixed brackets `()[]{}` up front, filler, then a question about the first (farthest) one: an inner node's value or which bracket closes which. Tests whether a model stored the **expression tree**, not just the final number.

| | |
|---|---|
| **Lengths** | 1k, 4k, 8k, 16k, 32k tokens (`seq_len`) |
| **Size** | 10,240 rows: train 8,448 · validation 896 · test 896 |
| **Tests** | structure memory: sub-expression values, bracket matching |
| **Tasks** | `subexpr` 5,175 · `match` 3,028 (primary) · `eval` 2,037 (control) |
| **Guessing** | depends on the task; see `prize_bits` |
| **Format** | text + ready ids for `HuggingFaceTB/SmolLM3-3B` (Llama-3 vocab) |
| **License** | Apache-2.0 |

## Example

```
( 3 + [ 2 * 4 ] ) - { 5 - 1 } . …more expressions…
… filler …
Q sub 0 . 2 .   →   values of internal nodes          (subexpr, primary)
Q match 2       →   index of the matching closer      (match, primary)
Q eval          →   the root value                    (eval, control only)
```

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("ksopyla/cogito-probe-arith")            # train / validation / test
row = ds["validation"][0]
print(row["seq_len"], row["task"], "|", row["query"], "->", row["answer"])

# Already tokenized for HuggingFaceTB/SmolLM3-3B (Llama-3 vocab); loss only where labels != -100.
# Or ignore the ids and use the text fields: context / query / answer.
small = ds["train"].filter(lambda r: r["seq_len"] == 1024 and r["variant"] == "fixed")  # laptop-sized start
```

## How to score

Teacher-forced token accuracy on the answer span, per `task`. Report `subexpr` and `match`;
`eval` is a single number a calculator shortcut can produce, so treat it as a control, not as
evidence that the model understands arithmetic.

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
<summary><b>Schema, build statistics, rebuild, limitations</b></summary>

**Columns**

| column | meaning |
|---|---|
| `text`, `context`, `query`, `answer` | readable surfaces; `text` is the full padded row |
| `input_ids`, `attention_mask`, `labels` | SmolLM3-3B ids, length = `seq_len`; `labels` = -100 except on the answer |
| `seq_len`, `rung` | 1024, 4096, 8192, 16384 or 32768 |
| `variant` | `fixed`: same facts as at 1k, longer filler (a drop = **length** failure); `scaled`: more facts with length (a drop = **capacity** failure) |
| `task` | question type (see above) |
| `prize_bits` | information in the gold answer (counting lower bound) |
| `gap` | tokens from the last evidence token to the answer |
| `answer_start`, `answer_end`, `evidence_end` | token indices into `input_ids` |
| `meta` | JSON: fact table, fingerprints, node values |

**Build** (seed `20260916`, 2026-09-16): 10,240 rows = train 8,448 / validation 896 / test 896.
Rows per length: 1k 4,608 · 4k 2,560 · 8k 1,280 · 16k 1,024 · 32k 768; half `fixed`, half `scaled`.
Mean `prize_bits` 12.9 (3.3–30.5).
Splits use disjoint random streams; fact fingerprints, `input_ids`, text and answer strings have zero
overlap between train, validation and test (shared answer *values*, such as colours, are expected).

**Rebuild** (deterministic):

```bash
uv run python scripts/build_concept_probe_datasets.py --scale full --seed 20260916 \
  --tokenizer HuggingFaceTB/SmolLM3-3B --families arith --out_dir Cache/concept_probes/full
```

Generator: `data/concept_probes/` in [github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito).

**Limitations**
- Not natural language: symbols are single tokens of the SmolLM3 / Llama-3 vocabulary
  (real rows use pieces like `gonzalez`, `oslo`, `validators`; the examples above use toy names).
- Answers are packed (several values per answer), so a small memory gets a dense learning signal.
- `prize_bits` is a counting lower bound, not a cross-entropy floor.
- The filler is a repeating cycle, so gzip of the whole row is tiny; compare `prize_bits`, not compressibility.
- Bracket shapes colour the tree; they do not change the value. Digits and brackets are injected as single ids, not BPE-encoded from a glued string.
</details>

## Changelog

| date | version | change |
|---|---|---|
| 2026-10-10 | card | Card rewritten: one-line goal, quick start, family table, related work, citations. Data unchanged. |
| 2026-09-16 | v0 | First public release, seed `20260916`, 10,240 rows. |

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

@misc{cogitoprobe_arith,
  title        = {CogitoProbe-Arith: nested arithmetic with mixed brackets},
  author       = {Sopyła, Krzysztof},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/ksopyla/cogito-probe-arith}},
  note         = {Cogito Capability Checks; seed 20260916, data v0}
}
```

Author: [Krzysztof Sopyła](https://ai.ksopyla.com) · code: [github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito).

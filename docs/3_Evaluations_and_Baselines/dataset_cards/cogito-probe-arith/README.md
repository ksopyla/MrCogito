---
language:
  - en
license: apache-2.0
pretty_name: CogitoProbe-Arith: nested arithmetic with mixed brackets
size_categories:
  - 10K<n<100K
task_categories:
  - question-answering
  - text-generation
tags:
  - synthetic
  - long-context
  - retrieval
  - needle-in-haystack
  - question-answering
  - cogito-probe
  - arith
  - research
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: validation
        path: validation.parquet
      - split: test
        path: test.parquet
dataset_info:
  dataset_name: ksopyla/cogito-probe-arith
---

# CogitoProbe-Arith: nested arithmetic with mixed brackets

Synthetic nested `+ - *` expressions with mixed brackets `()[]{}`. Three question types: the final number (`eval`, an easy shortcut), internal-node values (`subexpr`, the real test), and which closer matches an opener (`match`). Use it to test whether a model stored the *tree*, not just a calculator.

**Author:** Krzysztof Sopyła · **License:** Apache-2.0 · **Seed:** `20260916` · **Tokenizer:** `HuggingFaceTB/SmolLM3-3B`

## In 60 seconds

Several independent expressions are written up front, then filler, then a question about the **first** (farthest) expression.

| `task` | What it asks | Treat as |
|---|---|---|
| `eval` | `Q eval` → the root number | **Control only.** One integer; a calculator shortcut. |
| `subexpr` | `Q sub 0 . 2 .` → values of internal nodes | **Primary.** Needs the expression tree. |
| `match` | `Q match <opener-index>` → matching closer index | **Primary.** Bracket matching, not arithmetic. |

Mixed `()[]{}` do **not** change the numeric value. They colour the brackets so matching is a real question. Do not report `eval` accuracy as “the model understands arithmetic.”

## Load it

```python
from datasets import load_dataset

ds = load_dataset("ksopyla/cogito-probe-arith")
row = ds["validation"][0]

print(row["seq_len"], row["task"], row["variant"])
print("query: ", row["query"])
print("answer:", row["answer"])
print("prize bits:", row["prize_bits"], "gap:", row["gap"])

# Loss only on the answer span (already marked).
# input_ids / labels are lists of int, length == seq_len.
loss_tokens = [t for t in row["labels"] if t != -100]

# Start small on a laptop: 1,024-token rows, fixed fact count.
small = ds.filter(lambda r: r["seq_len"] == 1024 and r["variant"] == "fixed")
```

You can ignore `input_ids` and train from `context` / `query` / `answer` as text.
If you do use the provided ids, they are already tokenized for
`HuggingFaceTB/SmolLM3-3B` (Llama-3 vocab) and must not be re-tokenized.

## The four CogitoProbe datasets

| Dataset | Job in one line | Typical use |
|---|---|---|
| [`ksopyla/cogito-probe-bits`](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | Recall values for keys buried in a haystack | Memory / retrieval / compression capacity |
| [`ksopyla/cogito-probe-bind`](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | Who has which colour, who lives where, friend's city | Compositional binding vs bag-of-words |
| [`ksopyla/cogito-probe-arith`](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | Nested arithmetic + bracket matching | Did it store the expression tree? |
| [`ksopyla/cogito-probe-props`](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | Object colours amid fluent filler | Facts vs padding statistics |

Length ladder (every family): **1024 → 4096 → 8192 → 16384 → 32768**.
Half the rows are `fixed` (same fact count as at 1k, longer haystack), half are
`scaled` (more facts as the row grows).

## Why these exist

Web text is locally predictable: a language model can look strong by guessing
nearby words without remembering a fact from thousands of tokens earlier.
These four datasets hide a **known set of facts** in a long padded haystack so
you can measure whether a model (or a small latent memory) actually stored them.

Each row tells you how many bits the answer is worth (`prize_bits`) and how far
the question sits from the last fact (`gap`). That is the whole point: the
information content is labelled, the distractor text is not the prize, and the
length is a ladder rather than a single context size.

Real rows use random **single-token** English-ish pieces from the Llama-3 /
SmolLM3 vocabulary (`gonzalez`, `oslo`, `validators`, …), not the toy names
`alice` / `bob` in the examples above. The grammar of the task is the same.

## How to score

Train or evaluate **only on the answer span**. Teacher-forced token accuracy on
`labels != -100` is the main number. Recovered bits against the labelled prize:

`max(0, prize_bits + Σ log2 p(gold_t))`

A decoder that cannot see tokens more than `gap` away must sit at chance — the
evidence is that far from the answer.

Score `subexpr` and `match` as the real tasks. Score `eval` only as a shortcut control (~one integer). If only `eval` moves, the model is a calculator, not a structure memory. Digit strings are space-separated (`3 9` for 39, `- 7` for −7).

## Schema

| column | meaning |
|---|---|
| `text` / `context` / `query` / `answer` | Readable surfaces. `text` is the full padded row. |
| `input_ids`, `attention_mask`, `labels` | Ready for causal LM training. `labels` is `-100` everywhere except the answer. |
| `seq_len`, `rung` | Padded length: 1024, 4096, 8192, 16384, or 32768. |
| `variant` | `fixed` = same number of facts as at 1k, longer haystack. `scaled` = more facts as the row gets longer. |
| `task` | Question type inside this family (see above). |
| `prize_bits` | Known information content of the gold answer (combinatorial lower bound). |
| `gap` | Tokens from the last evidence token to the start of the answer. |
| `answer_start` / `answer_end` / `evidence_end` | Character-free token indices into `input_ids`. |
| `meta` | JSON string: fact table, fingerprints, node values. |

## This build

| split | rows |
|---|---|
| `train` | 8448 |
| `validation` | 896 |
| `test` | 896 |

| metric | value |
|---|---|
| total rows | 10240 |
| token length (all padded) | min 1024 / p50 4096.0 / max 32768 |
| mean prize bits | 12.914 (min 3.322, max 30.546) |
| mean gzip ratio (text) | 0.026 (n=160 stratified sample) |
| mean gzip ratio (int32 ids) | 0.037 |
| mean unigram entropy (bits) | 6.184 |
| mean bigram entropy (bits) | 6.286 |
| answer entropy (bits) | 8.604 over 2612 strings |
| tasks | `{'eval': 2037, 'subexpr': 5175, 'match': 3028}` |
| variants | `{'scaled': 5120, 'fixed': 5120}` |
| rungs | `{'seq1024': 4608, 'seq4096': 2560, 'seq8192': 1280, 'seq16384': 1024, 'seq32768': 768}` |

Per-rung means:

| seq_len | n | mean prize bits | mean gap | mean gzip(text) |
|---|---|---|---|---|
| 1024 | 4608 | 11.576 | 1001.9 | 0.068 |
| 4096 | 2560 | 12.086 | 4074.2 | 0.023 |
| 8192 | 1280 | 15.548 | 8168.9 | 0.016 |
| 16384 | 1024 | 15.303 | 16360.7 | 0.012 |
| 32768 | 768 | 16.129 | 32744.7 | 0.010 |

Example rows (truncated):

- `arith/seq1024/scaled/train/00000` task=`eval` prize=6.64 bits gap=1004 query=`Q eval` answer=`3 9`
- `arith/seq1024/scaled/train/00001` task=`subexpr` prize=15.27 bits gap=1005 query=`Q sub 2 . 1 .` answer=`1 8 . 1 4`
- `arith/seq1024/scaled/train/00002` task=`subexpr` prize=15.27 bits gap=988 query=`Q sub 5 . 2 .` answer=`- 1 . - 1 2`

### Split leakage

| pair | fingerprint overlap | input_ids overlap | text overlap | answer-string overlap |
|---|---|---|---|---|
| train∩validation | 0 | 0 | 0 | 248 |
| train∩test | 0 | 0 | 0 | 245 |
| validation∩test | 0 | 0 | 0 | 95 |

Within-split duplicate `input_ids` counts: `{'train': 0, 'validation': 0, 'test': 0}`.

Train / validation / test use disjoint random streams. A fingerprint of the
*facts* is checked for overlap. Shared answer *strings* (for example the same
8 colours) are expected and are **not** a leak.

## Rebuild

Deterministic rebuild (does not upload):

```bash
uv run python scripts/build_concept_probe_datasets.py \
  --scale full --seed 20260916 \
  --tokenizer HuggingFaceTB/SmolLM3-3B \
  --families arith \
  --out_dir Cache/concept_probes/full
```

Ids are composed from a verified 1-token atom table of `HuggingFaceTB/SmolLM3-3B`.
Arithmetic rows **inject** bare digit and bracket ids; they do not BPE-encode a
glued string such as `(1+2)*[3-4]` (that merge path is not a well-defined alphabet).

## Limitations

- Not natural language. Atoms are verified 1-token pieces of the SmolLM3 / Llama-3
  vocab, chosen so each symbol is one id. Do not treat this as a human corpus.
- Answers are **packed** (several values in one span). Single-token labels are too
  sparse for a small latent channel to learn from.
- `prize_bits` is a counting lower bound on the answer, not a cross-entropy floor
  of a local language-model window.
- Arithmetic mixed brackets colour the tree; they do not change `+ - *` meaning.
  `eval`-only accuracy is not evidence of rich structure.
- Rows are padded with a repeating filler cycle, so gzip of the full `text` looks
  tiny. Compare `prize_bits`, not compressibility of the padded row.

## Origin

These files were built for a research project on compressing long
context into a small set of latent vectors (“concepts”), so the author
could ask *what those vectors actually store*. You do not need that project,
its training code, or its internal experiment log to use the datasets.

Project page: [ai.ksopyla.com](https://ai.ksopyla.com) ·
author: [Krzysztof Sopyła](https://github.com/ksopyla).
Generator: `data/concept_probes/` in the public research repo (MIT).

## License

Apache-2.0 for this synthetic dataset. No web scrapes, no personal data.
Generator code is MIT.

## Citation

```
@misc{cogitoprobe2026,
  title  = {CogitoProbe: synthetic long-haystack probes for memory and compression},
  author = {Sopyła, Krzysztof},
  year   = {2026},
  url    = {https://huggingface.co/datasets/ksopyla/cogito-probe-arith},
  note   = {Seed 20260916. Four families: bits, bind, arith, props.},
}
```

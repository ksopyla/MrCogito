---
language:
  - en
license: apache-2.0
pretty_name: CogitoProbe-Bits: key–value recall in a long haystack
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
  - bits
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
  dataset_name: ksopyla/cogito-probe-bits
---

# CogitoProbe-Bits: key–value recall in a long haystack

Synthetic needle-in-a-haystack QA: random `key X val Y` facts sit at the start of a 1,024–32,768 token sequence, filler pads the middle, and the model must emit the values for a list of keys asked at the end. Use it to test memory, retrieval, or any compressed latent — no project background required.

**Author:** Krzysztof Sopyła · **License:** Apache-2.0 · **Seed:** `20260916` · **Tokenizer:** `HuggingFaceTB/SmolLM3-3B`

## In 60 seconds

1. Prefix lists facts: `key alice val red . key bob val blue .`
2. The rest of the sequence is filler (the haystack), padded to 1k, 4k, 8k, 16k, or 32k tokens.
3. The query asks for several keys at once: `Q alice bob`
4. The gold answer is the packed values, in the same order: `red blue`

Score the **answer span only** (`labels != -100`). A model that only models fluent filler will fail.

## Load it

```python
from datasets import load_dataset

ds = load_dataset("ksopyla/cogito-probe-bits")
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

Plot teacher-forced token accuracy (and recovered bits, see below) against how much unique information the answer carries: roughly `n_query × log2(32)` because there are 32 possible values. `variant=fixed` keeps ~16 facts at every length — if accuracy falls from 1k to 32k, that is a **length** failure. `variant=scaled` grows the fact table with length (up to 256 facts at 32k) — a drop there is a **capacity** failure. A dense Transformer should exceed ~75% packed-answer accuracy at `seq_len=1024` before you interpret any compressed-memory number.

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
| mean prize bits | 89.000 (min 40.000, max 640.000) |
| mean gzip ratio (text) | 0.035 (n=160 stratified sample) |
| mean gzip ratio (int32 ids) | 0.046 |
| mean unigram entropy (bits) | 6.215 |
| mean bigram entropy (bits) | 6.322 |
| answer entropy (bits) | 13.322 over 10240 strings |
| tasks | `{'recall_packed': 10240}` |
| variants | `{'scaled': 5120, 'fixed': 5120}` |
| rungs | `{'seq1024': 4608, 'seq4096': 2560, 'seq8192': 1280, 'seq16384': 1024, 'seq32768': 768}` |

Per-rung means:

| seq_len | n | mean prize bits | mean gap | mean gzip(text) |
|---|---|---|---|---|
| 1024 | 4608 | 40.000 | 939.5 | 0.080 |
| 4096 | 2560 | 60.000 | 3967.6 | 0.030 |
| 8192 | 1280 | 100.000 | 7975.6 | 0.025 |
| 16384 | 1024 | 180.000 | 15991.5 | 0.020 |
| 32768 | 768 | 340.000 | 32023.7 | 0.018 |

Example rows (truncated):

- `bits/seq1024/scaled/train/00000` task=`recall_packed` prize=40.00 bits gap=940 query=`Q gonzalez operator tcb lover qq trois deprecated onset` answer=`ase contrario hart greene keyboardtype validators scratch greene`
- `bits/seq1024/scaled/train/00001` task=`recall_packed` prize=40.00 bits gap=935 query=`Q initialization regimes zastav parte paw completes purchases evening` answer=`implicated ram volatility volatility seventh hanna conceal prick`
- `bits/seq1024/scaled/train/00002` task=`recall_packed` prize=40.00 bits gap=935 query=`Q jeh alma privileges inputs qq issuccess pendingintent regimes` answer=`ram differentiation timestamps rendered localization tolerated mathematic acceleration`

### Split leakage

| pair | fingerprint overlap | input_ids overlap | text overlap | answer-string overlap |
|---|---|---|---|---|
| train∩validation | 0 | 0 | 0 | 0 |
| train∩test | 0 | 0 | 0 | 0 |
| validation∩test | 0 | 0 | 0 | 0 |

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
  --families bits \
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
  url    = {https://huggingface.co/datasets/ksopyla/cogito-probe-bits},
  note   = {Seed 20260916. Four families: bits, bind, arith, props.},
}
```

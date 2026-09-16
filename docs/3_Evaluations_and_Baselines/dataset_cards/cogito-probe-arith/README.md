---
language:
  - en
license: apache-2.0
pretty_name: CogitoProbe-Arith — nested arithmetic / mixed brackets
size_categories:
  - n<1K
task_categories:
  - question-answering
  - text-generation
tags:
  - concept-compression
  - concept-bottleneck
  - synthetic
  - long-context
  - cogito-probe
  - arith
  - research
configs:
  - config_name: default
    data_files:
      train: train.parquet
      validation: validation.parquet
      test: test.parquet
dataset_info:
  dataset_name: ksopyla/cogito-probe-arith
---

# CogitoProbe-Arith — nested arithmetic / mixed brackets

**Hub id (proposed, not published until approved):** `ksopyla/cogito-probe-arith`
**Family:** `arith` · **Series:** CogitoProbe · **Default seed:** `20260916` · **Tokenizer:** `HuggingFaceTB/SmolLM3-3B`

## What it is

The bottleneck stores a compositional AST (or a sufficient set of node values + Dyck-3 match state), not a bag of digits. Eval-only is a shortcut control (~log2|result| bits) and must not be the success metric.

This is one dataset in a *series* that separates capability from confound:

| Dataset | Claim it can falsify |
|---|---|
| `ksopyla/cogito-probe-bits` | A C-slot bottleneck's unique-bit capacity vs haystack length |
| `ksopyla/cogito-probe-bind` | Tuple binding vs bag-of-tokens / document embedding |
| `ksopyla/cogito-probe-arith` | AST / Dyck-3 structure vs eval-only calculator shortcut |
| `ksopyla/cogito-probe-props` | Proposition set vs fluent-filler n-grams |

DNA A=4 (`data/symbolic_tasks.py`) remains the exact-floor *bandwidth* instrument. These
datasets exist because DNA cannot produce semantically rich concepts: iid 4-symbol
copy/recall has no entities, attributes, or propositions.

## Why it exists

Current training mixes in this repo (FineWeb-Edu, DCLM, PG-19, FinePDFs, stack-edu) are
locally predictable: E18's reach ablation paid ~0.05 nats for far text, so a concept
channel can look "used" while storing a document gist. DNA/Glyph then proved the
*channel* can copy bits, but those alphabets have no compositional semantics. This
family is the missing probe: **controlled information** + **a length ladder to 32k**
(the regime where E21 is supposed to show its potential).

## How it was generated

Exact command (deterministic):

```bash
uv run python scripts/build_concept_probe_datasets.py \
  --scale pilot --seed 20260916 \
  --tokenizer HuggingFaceTB/SmolLM3-3B \
  --families arith \
  --out_dir Cache/concept_probes/pilot
```

Generator: `data/concept_probes/` · atom table: verified 1-token pieces of `HuggingFaceTB/SmolLM3-3B`
(SmolLM3 = Llama-3 vocab). Arithmetic rows **inject** bare digit/operator ids; they do
**not** BPE-encode glued strings (that merge path is 0.76 tokens/atom and is not an
instrument).

Length ladder: **1024 → 4096 → 8192 → 16384 → 32768**.
Variants: `scaled` (item count grows with length) and `fixed` (1k item count, longer haystack).

## Schema

| column | type | meaning |
|---|---|---|
| `id` | string | `family/seqL/variant/split/index` |
| `family` | string | `arith` |
| `task` | string | query type inside the family |
| `variant` | string | `scaled` or `fixed` |
| `seq_len` | int | padded length |
| `rung` | string | `seq1024` … `seq32768` |
| `split` | string | train / validation / test |
| `input_ids` | list[int] | composed atom ids, length `seq_len` |
| `labels` | list[int] | `-100` except the answer span |
| `attention_mask` | list[int] | 1 on content |
| `text` | string | space-joined atom surfaces (readable) |
| `context` | string | prefix before the query |
| `query` | string | question atoms |
| `answer` | string | gold packed answer |
| `prize_bits` | float | known information content of the answer |
| `gap` | int | tokens from last evidence to answer start |
| `meta` | JSON string | fingerprints, node values, entity tables |

## Splits and statistics (this build)

| split | rows |
|---|---|
| `train` | 296 |
| `validation` | 84 |
| `test` | 84 |

| metric | value |
|---|---|
| total rows | 464 |
| token length (all padded) | min 1024 / p50 4096.0 / max 32768 |
| mean prize bits | 13.088 (min 3.322, max 30.546) |
| mean gzip ratio (text) | 0.037 |
| mean gzip ratio (int32 ids) | 0.052 |
| mean unigram entropy (bits) | 6.177 |
| mean bigram entropy (bits) | 6.277 |
| answer entropy (bits) | 7.242 over 270 strings |
| tasks | `{'eval': 102, 'subexpr': 244, 'match': 118}` |
| variants | `{'scaled': 232, 'fixed': 232}` |
| rungs | `{'seq1024': 192, 'seq4096': 96, 'seq8192': 80, 'seq16384': 56, 'seq32768': 40}` |

Per-rung means:

| seq_len | n | mean prize bits | mean gap | mean gzip(text) |
|---|---|---|---|---|
| 1024 | 192 | 11.928 | 1001.2 | 0.067 |
| 4096 | 96 | 12.173 | 4073.8 | 0.021 |
| 8192 | 80 | 15.614 | 8168.1 | 0.014 |
| 16384 | 56 | 14.406 | 16361.2 | 0.010 |
| 32768 | 40 | 13.961 | 32745.8 | 0.008 |

Example rows (truncated):

- `arith/seq1024/scaled/train/00000` task=`eval` prize=6.64 bits gap=1004 query=`Q eval` answer=`3 9`
- `arith/seq1024/scaled/train/00001` task=`subexpr` prize=15.27 bits gap=1005 query=`Q sub 2 . 1 .` answer=`1 8 . 1 4`
- `arith/seq1024/scaled/train/00002` task=`subexpr` prize=15.27 bits gap=988 query=`Q sub 5 . 2 .` answer=`- 1 . - 1 2`

### Leakage

| pair | fingerprint overlap | input_ids overlap | text overlap | answer-string overlap |
|---|---|---|---|---|
| train∩validation | 0 | 0 | 0 | 24 |
| train∩test | 0 | 0 | 0 | 16 |
| validation∩test | 0 | 0 | 0 | 13 |

Within-split duplicate `input_ids` counts: `{'train': 0, 'validation': 0, 'test': 0}`.

Train / validation / test use disjoint `SeedSequence` streams. A fingerprint of the
*content* (facts / entities / propositions) is checked for overlap. Answer-string
overlap is expected when the answer vocab is small (e.g. 8 colours) and is **not** a
leak.

## Intended use

- Train a concept-bottleneck / compressed-read model **only** on the answer span
  (`labels` or `--loss_span_markers` equivalent: the `answer`/`end` marker ids).
- Score **teacher-forced token accuracy** and **recovered bits**
  `max(0, prize_bits + sum log2 p(gold_t))` on `validation`/`test`.
- Necessity ablations: zero/shuffle concepts; segment-confined decoder with window
  `< gap` must sit at chance.
- Length-ladder plot: same prize (`variant=fixed`) vs same density (`variant=scaled`).

### Family-specific eval

**Primary:** `subexpr` packed internal-node values and `match` (Dyck-3). **Control, not success:** `eval` (root scalar; ~log2|result| bits — a 1-slot calculator). Kill the family as a *semantic* claim if only eval moves. Keep it as a *structure* claim if subexpr+match require the AST and survive concept ablation poorly when slots are shuffled.

## Known limitations and biases

- Closed single-token English-ish vocab from Llama-3, not a human language sample.
- Packed answers are required so the channel is not starved (E25: 8-letter copy stayed
  at chance; 32-letter copy hit 99%).
- `arith` does **not** replace DNA for closed-form `ln(A)` floors. Prize bits are
  combinatorial lower bounds, not CE floors of a local window (except where `gap` is
  recorded).
- Arithmetic mixed brackets `()[]{}` are Dyck-3 *colouring*; they do not change
  arithmetic meaning. Do not treat eval-only accuracy as evidence of rich concepts.
- Not a substitute for the deferred Deductive Stories corpus
  (`docs/engineering_specs/deductive_stories_synthetic_dataset.md`).

## Licensing

- **Code:** MIT (this repository).
- **This synthetic dataset:** Apache-2.0. No web scrapes, no personal data.

## Citation

```
@misc{cogitoprobe2026,
  title  = {CogitoProbe: controlled datasets for concept/latent compression},
  author = {Sopyła, Krzysztof},
  year   = {2026},
  url    = {https://huggingface.co/datasets/ksopyla/cogito-probe-arith},
  note   = {Synthetic length-ladder probes for concept bottlenecks. Seed 20260916.},
}
```

Project: [ai.ksopyla.com](https://ai.ksopyla.com) · code on GitHub under the author's namespace.

---
language:
  - en
license: apache-2.0
pretty_name: CogitoProbe-Bind — compositional entity binding
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
  - bind
  - research
configs:
  - config_name: default
    data_files:
      train: train.parquet
      validation: validation.parquet
      test: test.parquet
dataset_info:
  dataset_name: ksopyla/cogito-probe-bind
---

# CogitoProbe-Bind — compositional entity binding

**Hub id (proposed, not published until approved):** `ksopyla/cogito-probe-bind`
**Family:** `bind` · **Series:** CogitoProbe · **Default seed:** `20260916` · **Tokenizer:** `HuggingFaceTB/SmolLM3-3B`

## What it is

Concepts bind (entity, attribute, value) tuples. A bag-of-tokens or single document embedding cannot answer who-has-X or one-hop friend queries when entities share the same attribute vocabulary.

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
  --families bind \
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
| `family` | string | `bind` |
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
| mean prize bits | 62.448 (min 24.000, max 160.000) |
| mean gzip ratio (text) | 0.049 |
| mean gzip ratio (int32 ids) | 0.069 |
| mean unigram entropy (bits) | 6.269 |
| mean bigram entropy (bits) | 6.468 |
| answer entropy (bits) | 8.858 over 464 strings |
| tasks | `{'attr_color': 151, 'who_place': 159, 'hop_friend_place': 154}` |
| variants | `{'scaled': 232, 'fixed': 232}` |
| rungs | `{'seq1024': 192, 'seq4096': 96, 'seq8192': 80, 'seq16384': 56, 'seq32768': 40}` |

Per-rung means:

| seq_len | n | mean prize bits | mean gap | mean gzip(text) |
|---|---|---|---|---|
| 1024 | 192 | 34.500 | 887.0 | 0.088 |
| 4096 | 96 | 54.500 | 3891.0 | 0.030 |
| 8192 | 80 | 96.400 | 7851.0 | 0.023 |
| 16384 | 56 | 98.286 | 16043.0 | 0.014 |
| 32768 | 40 | 97.600 | 32427.0 | 0.009 |

Example rows (truncated):

- `bind/seq1024/scaled/train/00000` task=`attr_color` prize=40.00 bits gap=887 query=`Q color sanit listnode huckabee xss larg uif lobby attendee` answer=`green coral mouse dog oslo bird amber yellow`
- `bind/seq1024/scaled/train/00001` task=`attr_color` prize=40.00 bits gap=887 query=`Q color highlander aujourd deltax comput prosperous uif bye xss` answer=`deer dog bird yellow pink white brown silver`
- `bind/seq1024/scaled/train/00002` task=`who_place` prize=24.00 bits gap=887 query=`Q who in map miner box judge rope door actor rider` answer=`concentr where aujourd webhook elsif cargo apprent jord`

### Leakage

| pair | fingerprint overlap | input_ids overlap | text overlap | answer-string overlap |
|---|---|---|---|---|
| train∩validation | 0 | 0 | 0 | 0 |
| train∩test | 0 | 0 | 0 | 0 |
| validation∩test | 0 | 0 | 0 | 0 |

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

Break out `attr_color` / `who_place` / `hop_friend_place`. If attr is solved and hop stays at chance, the latents are labels not bindings. Counterfactual: swap one entity's colour in `meta.entities` and require the corresponding answer to flip.

## Known limitations and biases

- Closed single-token English-ish vocab from Llama-3, not a human language sample.
- Packed answers are required so the channel is not starved (E25: 8-letter copy stayed
  at chance; 32-letter copy hit 99%).
- `bind` does **not** replace DNA for closed-form `ln(A)` floors. Prize bits are
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
  url    = {https://huggingface.co/datasets/ksopyla/cogito-probe-bind},
  note   = {Synthetic length-ladder probes for concept bottlenecks. Seed 20260916.},
}
```

Project: [ai.ksopyla.com](https://ai.ksopyla.com) · code on GitHub under the author's namespace.

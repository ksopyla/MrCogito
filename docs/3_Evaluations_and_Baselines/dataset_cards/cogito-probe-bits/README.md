---
language:
  - en
license: apache-2.0
pretty_name: CogitoProbe-Bits — information-budget length ladder
size_categories:
  - 10K<n<100K
task_categories:
  - question-answering
  - text-generation
tags:
  - concept-compression
  - concept-bottleneck
  - synthetic
  - long-context
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

# CogitoProbe-Bits — information-budget length ladder

**Hub:** [`ksopyla/cogito-probe-bits`](https://huggingface.co/datasets/ksopyla/cogito-probe-bits)  
**Author:** Krzysztof Sopyła · **Family:** `bits` · **Series:** CogitoProbe · **Default seed:** `20260916` · **Tokenizer:** `HuggingFaceTB/SmolLM3-3B`

## What it is

A fixed latent set of C slots recovers at most ~C·k bits of unique prefix facts; accuracy falls once n_query·log2(|V|) exceeds that budget. Length at matched prize (variant=fixed) isolates haystack distance from capacity.

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
  --scale full --seed 20260916 \
  --tokenizer HuggingFaceTB/SmolLM3-3B \
  --families bits \
  --out_dir Cache/concept_probes/full
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
| `family` | string | `bits` |
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

Report accuracy vs `n_query · log2(|V|)` (default |V|=32). A C-slot model that matches dense at 16 facts and collapses at 256 facts (scaled 32k) is a *capacity* result. Matching at `fixed` 16 facts from 1k through 32k is a *length* result. Kill: dense < 75% on seq=1024 packed recall.

## Known limitations and biases

- Closed single-token English-ish vocab from Llama-3, not a human language sample.
- Packed answers are required so the channel is not starved (E25: 8-letter copy stayed
  at chance; 32-letter copy hit 99%).
- `bits` does **not** replace DNA for closed-form `ln(A)` floors. Prize bits are
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
  url    = {https://huggingface.co/datasets/ksopyla/cogito-probe-bits},
  note   = {Synthetic length-ladder probes for concept bottlenecks. Seed 20260916.},
}
```

Project: [ai.ksopyla.com](https://ai.ksopyla.com) · code on GitHub under the author's namespace.

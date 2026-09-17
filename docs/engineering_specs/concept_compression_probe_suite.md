# CogitoProbe — a series of datasets for concept / latent compression

- **Type:** engineering foundation (synthetic data series + HF cards + pretok builder). **Not** an `E0NN` experiment.
- **Status:** implemented 2026-09-16. **Published** 2026-09-16 to the Hugging Face Hub
  (public v0, `--scale full`, seed `20260916`):
  [`ksopyla/cogito-probe-bits`](https://huggingface.co/datasets/ksopyla/cogito-probe-bits),
  [`ksopyla/cogito-probe-bind`](https://huggingface.co/datasets/ksopyla/cogito-probe-bind),
  [`ksopyla/cogito-probe-arith`](https://huggingface.co/datasets/ksopyla/cogito-probe-arith),
  [`ksopyla/cogito-probe-props`](https://huggingface.co/datasets/ksopyla/cogito-probe-props).
  Generators `data/concept_probes/`, builder `scripts/build_concept_probe_datasets.py`,
  tests `tests/test_concept_probes.py`, cards
  `docs/3_Evaluations_and_Baselines/dataset_cards/`.
- **Owner:** Krzysztof Sopyła
- **Serves:** any memory / retrieval / compressed-latent experiment that needs
  labelled facts in a 1k–32k haystack. Hub cards are written for **external
  readers** (what the task is, a 60-second example, `load_dataset` snippet).
  DNA A=4 stays the exact-floor *bandwidth* instrument; this series is the
  *content* instrument. Internal experiment ids do not belong on the Hub README.

## Problem

The live training mixes cannot tell us whether concepts are semantically rich, and the
DNA-type exam cannot either.

### What we train and evaluate on today

| Source | Where | Weakness for a concept bottleneck |
|---|---|---|
| FineWeb-Edu | E01–E05, SmolLM3-inspired mixes, E18 fluency tier | Short, locally predictable web/edu prose. Next-token CE pays ~0.05 nats for far context (E18 reach ablation). A document embedding saturates the prize. |
| DCLM-baseline | `smollm3_inspired_*`, `e16b_long_4k_v1` | Same local-statistics attractor; no known unique-information budget per document. |
| PG-19 books | `e18_pilot_longdoc_v1`, `e22_longdoc_32k_v1` | Long *documents*, but narrative redundancy. Chunking into 32k rows (`chunk_long_docs`) still leaves highly compressible prose. No labelled fact set. |
| FinePDFs | E18 / E22 long tier | Long coherent PDFs; same CE-doesn't-pay-for-memory problem. |
| stack-edu python | E18 fluency | Local syntax + API patterns. Compressible by a language model without storing a concept array. |
| DNA A=4 (`data/symbolic_tasks.py`) | E24 / E25 / exclusive-slot law | **Exact floor, no semantics.** iid 4-symbol filler, INDEX/copy/MATCH2/REACHABILITY. E25 mapped E21 walls on this exam. A model that copies a 32-letter span has not learned a concept. |
| Glyph (`data/glyph_tasks.py`) | specified, uncalibrated | Typed 16/32-id alphabet + Markov/Dyck/arith *noise*. Still a closed algorithmic family, not propositions. |
| Delayed recall / keyed-recall rows | E14/E15, E18b | Sparse one-token answers starve the channel (E25: span-8 copy stayed at chance). English wrappers over single-token pools, not compositional structure. |
| Deductive Stories | `docs/engineering_specs/deductive_stories_synthetic_dataset.md` | **Deferred** to a separate repo after a failed LLM-expansion prototype. Not available as a mix. |

E21 (exclusive compressed read, run as E25 on DNA) only showed copy/INDEX potential at
some lengths and died on lookup/SELECT/hops. That is a *channel* map, not a semantics
map. The author's suspicion is right: **DNA cannot produce semantically rich concepts.**
Natural text cannot *measure* them. We need a series that separates those confounds.

## Design rules for the series

1. **One claim per dataset**, falsifiable without a language leaderboard.
2. **Shared length ladder** 1024 → 4096 → 8192 → 16384 → 32768 so the same probe scales
   to the E21 "only at length" regime.
3. **Two variants inside each dataset:** `fixed` (1k item count, longer haystack = length
   confound) and `scaled` (item count grows with length = capacity confound).
4. **Packed answers.** Short labels starve the channel (E25). Prize bits are
   `n_query · log2(\|answer class\|)` unless noted.
5. **Verified atoms, not BPE of a surface string.** Rows inject 1-token ids from
   `HuggingFaceTB/SmolLM3-3B` (Llama-3 vocab, the E18/E22 tokenizer).
6. **DNA is not replaced.** CogitoProbe answers "what is stored?". DNA answers "how many
   bits of a known alphabet survive?". Glyph stays the typed-noise family.

## The series (four Hub ids, one collection)

Proposed collection (not created): `ksopyla/cogito-probes`. All four dataset repos are **public**.

| Hub id | Claim it can falsify | What a C-slot bottleneck must encode | Kill if |
|---|---|---|---|
| [`ksopyla/cogito-probe-bits`](../3_Evaluations_and_Baselines/dataset_cards/cogito-probe-bits/README.md) | A C-slot array of width H recovers at most ~C·k unique bits of prefix facts; accuracy falls once `n_query·log2(\|V\|)` exceeds that budget. `fixed` 16 facts at 1k vs 32k isolates haystack distance. | The queried key→value map (not the filler). | Dense < 75% on seq=1024 packed recall. A model that ignores `n_query` and always emits a document gist. |
| [`ksopyla/cogito-probe-bind`](../3_Evaluations_and_Baselines/dataset_cards/cogito-probe-bind/README.md) | Concepts bind `(entity, attribute, value)` tuples. Bag-of-tokens and a single document embedding cannot answer `who_place` or `hop_friend_place` when entities share the attribute vocabulary. | Entity-centric slots (or an addressable tuple table). | `attr_color` solved and `hop_friend_place` at chance after a dense S0. |
| [`ksopyla/cogito-probe-arith`](../3_Evaluations_and_Baselines/dataset_cards/cogito-probe-arith/README.md) | The bottleneck stores a compositional AST (node values + Dyck-3 match state), not a bag of digits. **Eval-only is a shortcut control**, not the success metric. | Internal-node values and bracket match pointers for the queried expression. | Only `eval` moves; `subexpr`/`match` stay at chance. That is a calculator, not a concept. |
| [`ksopyla/cogito-probe-props`](../3_Evaluations_and_Baselines/dataset_cards/cogito-probe-props/README.md) | A fixed latent set carries the atomic propositions of a document, not the n-gram statistics of fluent filler. Shuffling filler must not change answers; shuffling proposition colours must. | The proposition set `{ (agent, colour, object, place) }`. | Accuracy tracks gzip/n-gram entropy of the filler rather than `n_items`. |

Cards, schema, and per-build tables live under
[`docs/3_Evaluations_and_Baselines/dataset_cards/`](../3_Evaluations_and_Baselines/dataset_cards/).

## Verdict on arithmetic / nested brackets

**Keep as Family 3 (structure control). Reject as a path to semantically rich concepts.**

The author's idea — synthetic token streams of numbers with `+ - * ( ) [ ] { }` — was
implemented as a prototype (`verification/probe_arith_tokenization.py`, seed 20260916,
200 expressions, max depth 4, SmolLM3 tokenizer) rather than accepted on faith.

| Measurement | Result |
|---|---|
| Bare atoms `0-9 + - * ( ) [ ] { }` | **1 token each** (atomic rate 1.0) |
| Leading-space digits `" 7"` | **Not atomic** (SmolLM3 splits them) |
| Leading-space operators/brackets | Atomic |
| Glued surface `((7+4)*[3-{4}])` | **96% of expressions merge**; 0.76 tokens/atom. Example `(1+2)*[3-{4}]` is 13 atoms, **10 BPE tokens**. |
| Space-separated atoms | 1.28 tokens/atom (spaces + split digits) |
| Eval-only information | **~6.6 bits** (one integer; max \|value\| exploded to 4.5e6 at unbounded `*`) |
| All distinct internal-node values | **~41.8 bits** (~6.3× the eval shortcut) |

So:

1. **Do not BPE-encode the glued string.** That is not a well-defined alphabet. The kept
   variant *injects* the 13 bare-atom ids (1:1). The readable `text` column may look glued
   after decode; `input_ids` are the instrument.
2. **Eval-only is a calculator.** One slot holding a scalar solves it. That is the
   opposite of rich concepts — it is a document embedding of dimension 1.
3. **Mixed brackets do not change arithmetic meaning.** `()[]{}` is Dyck-3 colouring on
   top of an AST. Useful as a *stack* probe (`match`), not as semantics.
4. **`*` without a bound is unusable** (values in the millions). The kept generator
   retries until every node value sits in `[-99, 99]`, with `*` probability 0.2.
5. **This is still algorithmic**, in the Delétang / RASP / ListOps family. It tests
   whether a latent set can hold a *tree*, not whether it can hold *meaning*. Semantic
   richness is `bind` + `props`. DNA remains the bit-exact copy exam.

**Kept arith recipe**

- Atoms: bare `0-9 + - * ( ) [ ] { }` plus `Q/A/END/sub/match/eval` markers.
- Haystack: `n_items` independent bracketed expressions (4 at 1k `fixed`; scales on
  `scaled`), query the earliest (farthest) one.
- Depth 3, values in `[-99, 99]`.
- Tasks (row-level mix): **`subexpr` ~50%** (packed internal-node values, primary),
  **`match` ~25%** (opener → closer index), **`eval` ~25%** (root scalar, *shortcut
  control*).
- What the bottleneck must encode to beat chance on the primary tasks: the AST of the
  target expression (or a sufficient set of node values plus Dyck-3 match pointers).
  Eval needs only the root integer.

Pilot mix on this build: 244 `subexpr` / 118 `match` / 102 `eval` of 464 rows.

## How to generate (deterministic)

Public v0 (what is on the Hub):

```bash
uv run python scripts/build_concept_probe_datasets.py \
  --scale full --seed 20260916 \
  --tokenizer HuggingFaceTB/SmolLM3-3B \
  --out_dir Cache/concept_probes/full \
  --hub_only \
  --stats_out docs/3_Evaluations_and_Baselines/dataset_cards/cogito-probe-stats.json \
  --cards_out docs/3_Evaluations_and_Baselines/dataset_cards
```

`--scale full` is the production recipe (10,240 rows/family; 8448/896/896;
see `FULL_COUNTS` in `data/concept_probes/schema.py`). `--scale pilot` is the
464-row sample used to calibrate the first spec tables (not the Hub v0 cut).

Tokenizer probe (the arith verdict):

```bash
uv run python verification/probe_arith_tokenization.py \
  --seed 20260916 --out Cache/concept_probes/arith_tokenization_probe.json
```

## Pilot statistics (seed 20260916, SmolLM3)

464 rows per family (train 296 / val 84 / test 84). Every row is padded to its rung
length. Content fingerprints and `input_ids` do **not** overlap across splits (0/0/0).
Arith answer-*string* overlap is expected (small integers) and is not a leak.

| family | mean prize bits | 32k prize | 32k gap | gzip(text) | H1 / H2 (bits) | distinct answers |
|---|---|---|---|---|---|---|
| bits | 97.2 (40–640) | 340 | 32024 | 0.043 | 6.21 / 6.32 | 464 |
| bind | 62.4 (24–160) | 97.6 | 32427 | 0.049 | 6.27 / 6.47 | 464 |
| arith | 13.1 (3.3–30.5) | 14.0 | 32746 | 0.037 | 6.18 / 6.28 | 270 |
| props | 54.3 (30–160) | 95.0 | 32570 | 0.042 | 6.20 / 6.27 | 464 |

Gzip ratios ≪ 1 because rows are padded with a repeating filler cycle — that is the
haystack. Unique information is the prize-bits column, not the gzip of the padded row.
Compare prize-bits across `fixed` vs `scaled` at the same `seq_len` to see capacity vs
length.

Bits at 32k: `fixed` ≈ 40 bits (8 values × log2(32)), `scaled` ≈ 640 bits (128 values).
A C=128, 256-dim slot array that solves `fixed`@32k and fails `scaled`@32k is a
capacity result, which DNA copy at 64 bits cannot state.

**v0 Hub cut (this publish, `--scale full`, same seed):** 10,240 rows/family
(8448/896/896). Mean prize bits: bits 89.0, bind 58.4, arith 12.9, props 50.8.
Per-family leakage fingerprints/`input_ids`/`text` = 0. Full tables:
[`cogito-probe-stats.json`](../3_Evaluations_and_Baselines/dataset_cards/cogito-probe-stats.json).

## How to evaluate

Teacher-forced accuracy on `labels ≠ -100`, plus recovered bits against `prize_bits`.
Necessity: zero/shuffle concepts; a decoder whose raw window is `< gap` must sit at
chance (gaps are ~seq_len minus a short query tail). Dense S0 ≥ 75% on seq=1024
before any compressed-read number is interpreted (same protocol as
[`bapo_capability_ladder.md`](bapo_capability_ladder.md)).

Do not score STS-B or FineWeb CE as a CogitoProbe result.

## Hugging Face publication (public v0, 2026-09-16)

Uploaded under `ksopyla/` as **public** Apache-2.0 datasets. Load:

```python
from datasets import load_dataset
ds = load_dataset("ksopyla/cogito-probe-bits")
row = ds["test"][0]
assert len(row["input_ids"]) == row["seq_len"]
```

v0 Hub counts (seed `20260916`, SmolLM3 tokenizer, `--scale full`): **10,240 rows
per family** (train 8448 / validation 896 / test 896). Content fingerprints,
`input_ids`, and `text` do not overlap across splits. Arith answer-*string*
overlap is expected (small integers) and is not a leak. Gzip ratios on the cards
are a 32-row-per-rung sample.

Staging layout (gitignored `Cache/`):

```text
Cache/concept_probes/full/hub/{bits,bind,arith,props}/
  README.md train.parquet validation.parquet test.parquet
```

Re-upload after a rebuild:

```bash
hf repos create ksopyla/cogito-probe-bits --type dataset --public --exist-ok
hf upload ksopyla/cogito-probe-bits Cache/concept_probes/full/hub/bits \
  --repo-type dataset --commit-message "Add CogitoProbe bits full (seed 20260916)"
# repeat for bind, arith, props
```

## Relation to other in-repo synthetics

| Suite | Role after this lands |
|---|---|
| DNA `symbolic_tasks.py` | Keep. Exact `ln(4)` floor. Bandwidth / INDEX. |
| Glyph `glyph_tasks.py` | Keep. Typed noise, uncalibrated. Not a CogitoProbe family. |
| Delayed recall | Keep for E14-style block diagnostics. Too sparse as a concept exam. |
| Deductive Stories | Still deferred. CogitoProbe-props is the *cheap closed-vocab* stand-in, not the narrative corpus. |

## Files

| Path | Role |
|---|---|
| `data/concept_probes/` | Generators, atom table, stats, card renderer |
| `scripts/build_concept_probe_datasets.py` | Deterministic build + Hub staging (no upload) |
| `verification/probe_arith_tokenization.py` | Tokenizer / shortcut measurement |
| `tests/test_concept_probes.py` | Solvability, determinism, leakage, SmolLM3 1:1 check |
| `docs/3_Evaluations_and_Baselines/dataset_cards/` | Four cards + `cogito-probe-stats.json` |

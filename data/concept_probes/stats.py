"""Statistics, leakage checks, and dataset-card rendering for CogitoProbe."""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

from data.concept_probes.schema import ProbeRow


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def ngram_entropy(ids: Iterable[int], n: int) -> float:
    seq = list(ids)
    if len(seq) < n:
        return 0.0
    grams = Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    return _entropy(grams)


def gzip_ratio_bytes(data: bytes) -> float:
    if not data:
        return 1.0
    return len(gzip.compress(data, compresslevel=9)) / len(data)


def gzip_ratio_text(text: str) -> float:
    return gzip_ratio_bytes(text.encode("utf-8"))


def gzip_ratio_ids(ids: list[int]) -> float:
    arr = np.asarray(ids, dtype=np.int32).tobytes()
    return gzip_ratio_bytes(arr)


def length_summary(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"n": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0}
    a = np.asarray(lengths, dtype=np.int32)
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": int(a.max()),
        "mean": float(a.mean()),
    }


def _fingerprint(row: ProbeRow) -> str:
    meta = row.meta
    if "content_fingerprint" in meta:
        return json.dumps(meta["content_fingerprint"], sort_keys=True)
    return row.text


def leakage_report(splits: dict[str, list[ProbeRow]]) -> dict[str, Any]:
    fps: dict[str, set[str]] = {}
    idsets: dict[str, set[tuple[int, ...]]] = {}
    texts: dict[str, set[str]] = {}
    answers: dict[str, set[str]] = {}
    for name, rows in splits.items():
        fps[name] = {_fingerprint(r) for r in rows}
        idsets[name] = {tuple(r.input_ids) for r in rows}
        texts[name] = {r.text for r in rows}
        answers[name] = {r.answer for r in rows}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    out: dict[str, Any] = {"pairs": {}}
    for a, b in pairs:
        if a not in fps or b not in fps:
            continue
        out["pairs"][f"{a}∩{b}"] = {
            "fingerprint": len(fps[a] & fps[b]),
            "input_ids": len(idsets[a] & idsets[b]),
            "text": len(texts[a] & texts[b]),
            "answer_only": len(answers[a] & answers[b]),
        }
    out["within_split_duplicate_ids"] = {
        name: len(rows) - len(idsets[name]) for name, rows in splits.items()
    }
    return out


def summarize_family(rows_by_split: dict[str, list[ProbeRow]]) -> dict[str, Any]:
    all_rows = [r for rs in rows_by_split.values() for r in rs]
    if not all_rows:
        return {}
    token_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    prize = []
    gaps = []
    n_supervised = []
    gzip_text = []
    gzip_ids = []
    h1 = []
    h2 = []
    tasks: Counter = Counter()
    variants: Counter = Counter()
    rungs: Counter = Counter()
    lengths = []
    for r in all_rows:
        token_counts.update(r.input_ids)
        answer_counts.update([r.answer])
        prize.append(r.prize_bits)
        gaps.append(r.gap)
        n_supervised.append(sum(1 for x in r.labels if x != -100))
        gzip_text.append(gzip_ratio_text(r.text))
        gzip_ids.append(gzip_ratio_ids(r.input_ids))
        h1.append(ngram_entropy(r.input_ids, 1))
        h2.append(ngram_entropy(r.input_ids, 2))
        tasks[r.task] += 1
        variants[r.variant] += 1
        rungs[f"seq{r.seq_len}"] += 1
        lengths.append(r.n_tokens)
    top_tokens = token_counts.most_common(12)
    split_sizes = {k: len(v) for k, v in rows_by_split.items()}
    by_rung = {}
    for r in all_rows:
        by_rung.setdefault(r.seq_len, []).append(r)
    rung_stats = {
        str(seq): {
            "n": len(rs),
            "mean_prize_bits": float(np.mean([x.prize_bits for x in rs])),
            "mean_gap": float(np.mean([x.gap for x in rs])),
            "mean_gzip_text": float(np.mean([gzip_ratio_text(x.text) for x in rs])),
            "tasks": dict(Counter(x.task for x in rs)),
        }
        for seq, rs in sorted(by_rung.items())
    }
    return {
        "n_rows": len(all_rows),
        "splits": split_sizes,
        "rungs": dict(rungs),
        "tasks": dict(tasks),
        "variants": dict(variants),
        "token_length": length_summary(lengths),
        "supervised_tokens": length_summary(n_supervised),
        "gap": length_summary(gaps),
        "prize_bits": {
            "min": float(min(prize)),
            "mean": float(np.mean(prize)),
            "max": float(max(prize)),
        },
        "gzip_ratio_text_mean": float(np.mean(gzip_text)),
        "gzip_ratio_ids_mean": float(np.mean(gzip_ids)),
        "unigram_entropy_mean": float(np.mean(h1)),
        "bigram_entropy_mean": float(np.mean(h2)),
        "answer_entropy": _entropy(answer_counts),
        "n_distinct_answers": len(answer_counts),
        "top_token_ids": [{"id": int(i), "count": int(c)} for i, c in top_tokens],
        "leakage": leakage_report(rows_by_split),
        "per_rung": rung_stats,
        "samples": [
            {
                "id": r.id,
                "task": r.task,
                "variant": r.variant,
                "seq_len": r.seq_len,
                "prize_bits": r.prize_bits,
                "gap": r.gap,
                "query": r.query,
                "answer": r.answer,
                "text_head": r.text[:400],
            }
            for r in all_rows[:3]
        ],
    }


CARD_TEMPLATE = """---
language:
  - en
license: apache-2.0
pretty_name: {pretty}
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
  - {family}
  - research
configs:
{configs_yaml}
dataset_info:
  dataset_name: {hub_id}
---

# {pretty}

**Hub id (proposed, not published until approved):** `{hub_id}`
**Family:** `{family}` · **Series:** CogitoProbe · **Default seed:** `{seed}` · **Tokenizer:** `{tokenizer}`

## What it is

{claim}

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
uv run python scripts/build_concept_probe_datasets.py \\
  --scale {scale} --seed {seed} \\
  --tokenizer {tokenizer} \\
  --families {family} \\
  --out_dir Cache/concept_probes/{scale}
```

Generator: `data/concept_probes/` · atom table: verified 1-token pieces of `{tokenizer}`
(SmolLM3 = Llama-3 vocab). Arithmetic rows **inject** bare digit/operator ids; they do
**not** BPE-encode glued strings (that merge path is 0.76 tokens/atom and is not an
instrument).

Length ladder: **1024 → 4096 → 8192 → 16384 → 32768**.
Variants: `scaled` (item count grows with length) and `fixed` (1k item count, longer haystack).

## Schema

| column | type | meaning |
|---|---|---|
| `id` | string | `family/seqL/variant/split/index` |
| `family` | string | `{family}` |
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

{stats_md}

### Leakage

{leakage_md}

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

{eval_md}

## Known limitations and biases

- Closed single-token English-ish vocab from Llama-3, not a human language sample.
- Packed answers are required so the channel is not starved (E25: 8-letter copy stayed
  at chance; 32-letter copy hit 99%).
- `{family}` does **not** replace DNA for closed-form `ln(A)` floors. Prize bits are
  combinatorial lower bounds, not CE floors of a local window (except where `gap` is
  recorded).
- Arithmetic mixed brackets `()[]{{}}` are Dyck-3 *colouring*; they do not change
  arithmetic meaning. Do not treat eval-only accuracy as evidence of rich concepts.
- Not a substitute for the deferred Deductive Stories corpus
  (`docs/engineering_specs/deductive_stories_synthetic_dataset.md`).

## Licensing

- **Code:** MIT (this repository).
- **This synthetic dataset:** Apache-2.0. No web scrapes, no personal data.

## Citation

```
@misc{{cogitoprobe2026,
  title  = {{CogitoProbe: controlled datasets for concept/latent compression}},
  author = {{Sopyła, Krzysztof}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{hub_id}}},
  note   = {{Synthetic length-ladder probes for concept bottlenecks. Seed {seed}.}},
}}
```

Project: [ai.ksopyla.com](https://ai.ksopyla.com) · code on GitHub under the author's namespace.
"""


EVAL_MD = {
    "bits": (
        "Report accuracy vs `n_query · log2(|V|)` (default |V|=32). A C-slot model that "
        "matches dense at 16 facts and collapses at 256 facts (scaled 32k) is a *capacity* "
        "result. Matching at `fixed` 16 facts from 1k through 32k is a *length* result. "
        "Kill: dense < 75% on seq=1024 packed recall."
    ),
    "bind": (
        "Break out `attr_color` / `who_place` / `hop_friend_place`. If attr is solved and "
        "hop stays at chance, the latents are labels not bindings. Counterfactual: swap one "
        "entity's colour in `meta.entities` and require the corresponding answer to flip."
    ),
    "arith": (
        "**Primary:** `subexpr` packed internal-node values and `match` (Dyck-3). "
        "**Control, not success:** `eval` (root scalar; ~log2|result| bits — a 1-slot "
        "calculator). Kill the family as a *semantic* claim if only eval moves. Keep it as a "
        "*structure* claim if subexpr+match require the AST and survive concept ablation poorly "
        "when slots are shuffled."
    ),
    "props": (
        "Packed object→colour over unique objects. Control: shuffle filler tokens in `context` "
        "(answers must hold) vs shuffle proposition colours in `meta.propositions` (answers "
        "must change). A model that tracks gzip/n-gram statistics of filler will fail the "
        "first control."
    ),
}


def _configs_yaml(family: str) -> str:
    # Hub staging layout is one parquet per split (rung is a column, not a repo config).
    return "\n".join(
        [
            "  - config_name: default",
            "    data_files:",
            "      train: train.parquet",
            "      validation: validation.parquet",
            "      test: test.parquet",
        ]
    )


def _stats_md(stats: dict[str, Any]) -> str:
    pb = stats.get("prize_bits", {})
    tl = stats.get("token_length", {})
    rows = [
        "| split | rows |",
        "|---|---|",
    ]
    for k, v in stats.get("splits", {}).items():
        rows.append(f"| `{k}` | {v} |")
    rows += [
        "",
        "| metric | value |",
        "|---|---|",
        f"| total rows | {stats.get('n_rows', 0)} |",
        f"| token length (all padded) | min {tl.get('min')} / p50 {tl.get('p50')} / max {tl.get('max')} |",
        f"| mean prize bits | {pb.get('mean', 0):.3f} (min {pb.get('min', 0):.3f}, max {pb.get('max', 0):.3f}) |",
        f"| mean gzip ratio (text) | {stats.get('gzip_ratio_text_mean', 0):.3f} |",
        f"| mean gzip ratio (int32 ids) | {stats.get('gzip_ratio_ids_mean', 0):.3f} |",
        f"| mean unigram entropy (bits) | {stats.get('unigram_entropy_mean', 0):.3f} |",
        f"| mean bigram entropy (bits) | {stats.get('bigram_entropy_mean', 0):.3f} |",
        f"| answer entropy (bits) | {stats.get('answer_entropy', 0):.3f} over {stats.get('n_distinct_answers', 0)} strings |",
        f"| tasks | `{stats.get('tasks', {})}` |",
        f"| variants | `{stats.get('variants', {})}` |",
        f"| rungs | `{stats.get('rungs', {})}` |",
        "",
        "Per-rung means:",
        "",
        "| seq_len | n | mean prize bits | mean gap | mean gzip(text) |",
        "|---|---|---|---|---|",
    ]
    for seq, rs in stats.get("per_rung", {}).items():
        rows.append(
            f"| {seq} | {rs['n']} | {rs['mean_prize_bits']:.3f} | {rs['mean_gap']:.1f} | {rs['mean_gzip_text']:.3f} |"
        )
    samples = stats.get("samples", [])
    if samples:
        rows += ["", "Example rows (truncated):", ""]
        for s in samples:
            rows.append(
                f"- `{s['id']}` task=`{s['task']}` prize={s['prize_bits']:.2f} bits "
                f"gap={s['gap']} query=`{s['query']}` answer=`{s['answer']}`"
            )
    return "\n".join(rows)


def _leakage_md(stats: dict[str, Any]) -> str:
    leak = stats.get("leakage", {})
    lines = [
        "| pair | fingerprint overlap | input_ids overlap | text overlap | answer-string overlap |",
        "|---|---|---|---|---|",
    ]
    for pair, v in leak.get("pairs", {}).items():
        lines.append(
            f"| {pair} | {v['fingerprint']} | {v['input_ids']} | {v['text']} | {v['answer_only']} |"
        )
    dup = leak.get("within_split_duplicate_ids", {})
    lines.append("")
    lines.append(f"Within-split duplicate `input_ids` counts: `{dup}`.")
    return "\n".join(lines)


def render_card(
    family: str,
    stats: dict[str, Any],
    *,
    seed: int,
    tokenizer: str,
    scale: str,
    hub_id: str,
) -> str:
    pretty = {
        "bits": "CogitoProbe-Bits — information-budget length ladder",
        "bind": "CogitoProbe-Bind — compositional entity binding",
        "arith": "CogitoProbe-Arith — nested arithmetic / mixed brackets",
        "props": "CogitoProbe-Props — proposition gist vs filler",
    }[family]
    from data.concept_probes.schema import FAMILY_CLAIMS

    return CARD_TEMPLATE.format(
        pretty=pretty,
        hub_id=hub_id,
        family=family,
        seed=seed,
        tokenizer=tokenizer,
        scale=scale,
        claim=FAMILY_CLAIMS[family],
        configs_yaml=_configs_yaml(family),
        stats_md=_stats_md(stats),
        leakage_md=_leakage_md(stats),
        eval_md=EVAL_MD[family],
    )

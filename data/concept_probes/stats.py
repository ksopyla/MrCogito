"""Statistics, leakage checks, and dataset-card rendering for CogitoProbe."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
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


def size_category(n_rows: int) -> str:
    """Hugging Face `size_categories` bucket for a dataset card."""
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    if n_rows < 1_000_000:
        return "100K<n<1M"
    if n_rows < 10_000_000:
        return "1M<n<10M"
    return "10M<n<100M"


def _ids_hash(ids: list[int]) -> bytes:
    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).digest()


def _str_hash(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _leakage_from_hash_sets(
    fp_hashes: dict[str, set[bytes]],
    id_hashes: dict[str, set[bytes]],
    text_hashes: dict[str, set[bytes]],
    answers: dict[str, set[str]],
    split_n: dict[str, int],
) -> dict[str, Any]:
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    out: dict[str, Any] = {"pairs": {}}
    for a, b in pairs:
        if a not in fp_hashes or b not in fp_hashes:
            continue
        out["pairs"][f"{a}∩{b}"] = {
            "fingerprint": len(fp_hashes[a] & fp_hashes[b]),
            "input_ids": len(id_hashes[a] & id_hashes[b]),
            "text": len(text_hashes[a] & text_hashes[b]),
            "answer_only": len(answers[a] & answers[b]),
        }
    out["within_split_duplicate_ids"] = {
        name: int(split_n.get(name, 0) - len(id_hashes.get(name, ())))
        for name in split_n
    }
    return out


def assert_no_content_leakage(leakage: dict[str, Any]) -> None:
    """Fingerprints and full `input_ids` must not cross splits; answer strings may."""
    for pair, v in leakage.get("pairs", {}).items():
        if v["fingerprint"] != 0:
            raise ValueError(f"content fingerprint leak on {pair}: {v['fingerprint']}")
        if v["input_ids"] != 0:
            raise ValueError(f"input_ids leak on {pair}: {v['input_ids']}")
        if v["text"] != 0:
            raise ValueError(f"text leak on {pair}: {v['text']}")
    dups = leakage.get("within_split_duplicate_ids", {})
    bad = {k: n for k, n in dups.items() if n}
    if bad:
        raise ValueError(f"within-split duplicate input_ids: {bad}")


class FamilyStatsAccumulator:
    """Online stats + leakage hashes. Does not retain row tensors."""

    def __init__(self, *, gzip_per_rung: int | None = None):
        self.gzip_per_rung = gzip_per_rung
        self.token_counts: Counter = Counter()
        self.answer_counts: Counter = Counter()
        self.prize: list[float] = []
        self.gaps: list[int] = []
        self.n_supervised: list[int] = []
        self.gzip_text: list[float] = []
        self.gzip_ids: list[float] = []
        self.h1: list[float] = []
        self.h2: list[float] = []
        self.tasks: Counter = Counter()
        self.variants: Counter = Counter()
        self.rungs: Counter = Counter()
        self.lengths: list[int] = []
        self.split_n: Counter = Counter()
        self.fp_hashes: dict[str, set[bytes]] = {}
        self.id_hashes: dict[str, set[bytes]] = {}
        self.text_hashes: dict[str, set[bytes]] = {}
        self.answers: dict[str, set[str]] = {}
        self.rung_gzip_n: Counter = Counter()
        self.rung_prize: dict[int, list[float]] = {}
        self.rung_gap: dict[int, list[float]] = {}
        self.rung_gzip: dict[int, list[float]] = {}
        self.rung_tasks: dict[int, Counter] = {}
        self.samples: list[dict[str, Any]] = []
        self.bad_len = 0

    def add(self, row: ProbeRow) -> None:
        if len(row.input_ids) != row.seq_len:
            self.bad_len += 1
        split = row.split
        self.split_n[split] += 1
        self.token_counts.update(row.input_ids)
        self.answer_counts.update([row.answer])
        self.prize.append(row.prize_bits)
        self.gaps.append(row.gap)
        self.n_supervised.append(sum(1 for x in row.labels if x != -100))
        self.h1.append(ngram_entropy(row.input_ids, 1))
        self.h2.append(ngram_entropy(row.input_ids, 2))
        self.tasks[row.task] += 1
        self.variants[row.variant] += 1
        self.rungs[f"seq{row.seq_len}"] += 1
        self.lengths.append(row.n_tokens)
        self.fp_hashes.setdefault(split, set()).add(_str_hash(_fingerprint(row)))
        self.id_hashes.setdefault(split, set()).add(_ids_hash(row.input_ids))
        self.text_hashes.setdefault(split, set()).add(_str_hash(row.text))
        self.answers.setdefault(split, set()).add(row.answer)
        take_gzip = self.gzip_per_rung is None or self.rung_gzip_n[row.seq_len] < self.gzip_per_rung
        if take_gzip:
            gt = gzip_ratio_text(row.text)
            self.gzip_text.append(gt)
            self.gzip_ids.append(gzip_ratio_ids(row.input_ids))
            self.rung_gzip.setdefault(row.seq_len, []).append(gt)
            self.rung_gzip_n[row.seq_len] += 1
        self.rung_prize.setdefault(row.seq_len, []).append(row.prize_bits)
        self.rung_gap.setdefault(row.seq_len, []).append(float(row.gap))
        self.rung_tasks.setdefault(row.seq_len, Counter())[row.task] += 1
        if len(self.samples) < 3:
            self.samples.append(
                {
                    "id": row.id,
                    "task": row.task,
                    "variant": row.variant,
                    "seq_len": row.seq_len,
                    "prize_bits": row.prize_bits,
                    "gap": row.gap,
                    "query": row.query,
                    "answer": row.answer,
                    "text_head": row.text[:400],
                }
            )

    def leakage(self) -> dict[str, Any]:
        return _leakage_from_hash_sets(
            self.fp_hashes,
            self.id_hashes,
            self.text_hashes,
            self.answers,
            dict(self.split_n),
        )

    def finalize(self) -> dict[str, Any]:
        if self.bad_len:
            raise ValueError(f"{self.bad_len} rows have len(input_ids) != seq_len")
        if not self.prize:
            return {}
        leakage = self.leakage()
        assert_no_content_leakage(leakage)
        rung_stats = {
            str(seq): {
                "n": len(self.rung_prize[seq]),
                "mean_prize_bits": float(np.mean(self.rung_prize[seq])),
                "mean_gap": float(np.mean(self.rung_gap[seq])),
                "mean_gzip_text": float(np.mean(self.rung_gzip[seq])) if self.rung_gzip.get(seq) else 0.0,
                "gzip_n": len(self.rung_gzip.get(seq, [])),
                "tasks": dict(self.rung_tasks[seq]),
            }
            for seq in sorted(self.rung_prize)
        }
        split_order = ("train", "validation", "test")
        splits = {k: int(self.split_n[k]) for k in split_order if k in self.split_n}
        for k, v in self.split_n.items():
            splits.setdefault(k, int(v))
        return {
            "n_rows": int(sum(self.split_n.values())),
            "splits": splits,
            "rungs": dict(self.rungs),
            "tasks": dict(self.tasks),
            "variants": dict(self.variants),
            "token_length": length_summary(self.lengths),
            "supervised_tokens": length_summary(self.n_supervised),
            "gap": length_summary(self.gaps),
            "prize_bits": {
                "min": float(min(self.prize)),
                "mean": float(np.mean(self.prize)),
                "max": float(max(self.prize)),
            },
            "gzip_ratio_text_mean": float(np.mean(self.gzip_text)) if self.gzip_text else 0.0,
            "gzip_ratio_ids_mean": float(np.mean(self.gzip_ids)) if self.gzip_ids else 0.0,
            "gzip_sampled": self.gzip_per_rung is not None,
            "gzip_n": len(self.gzip_text),
            "gzip_per_rung": self.gzip_per_rung,
            "unigram_entropy_mean": float(np.mean(self.h1)),
            "bigram_entropy_mean": float(np.mean(self.h2)),
            "answer_entropy": _entropy(self.answer_counts),
            "n_distinct_answers": len(self.answer_counts),
            "top_token_ids": [{"id": int(i), "count": int(c)} for i, c in self.token_counts.most_common(12)],
            "leakage": leakage,
            "per_rung": rung_stats,
            "samples": self.samples,
        }


def summarize_family(
    rows_by_split: dict[str, list[ProbeRow]],
    *,
    gzip_per_rung: int | None = None,
) -> dict[str, Any]:
    acc = FamilyStatsAccumulator(gzip_per_rung=gzip_per_rung)
    for split in ("train", "validation", "test"):
        for row in rows_by_split.get(split, []):
            acc.add(row)
    for split, rows in rows_by_split.items():
        if split in {"train", "validation", "test"}:
            continue
        for row in rows:
            acc.add(row)
    return acc.finalize()


def verify_hub_parquets(
    hub_root: Path,
    expected_splits: dict[str, int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Scan staged parquet without loading the whole family into RAM."""
    import pyarrow.parquet as pq

    id_hashes: dict[str, set[bytes]] = {}
    fp_hashes: dict[str, set[bytes]] = {}
    text_hashes: dict[str, set[bytes]] = {}
    answers: dict[str, set[str]] = {}
    split_n: dict[str, int] = {}
    bad_len = 0
    bad_seed = 0
    for split in ("train", "validation", "test"):
        path = Path(hub_root) / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        expected = expected_splits.get(split)
        if expected is not None and n != expected:
            raise ValueError(f"{path} has {n} rows, expected {expected}")
        split_n[split] = n
        idh: set[bytes] = set()
        fph: set[bytes] = set()
        txh: set[bytes] = set()
        ans: set[str] = set()
        cols = ["input_ids", "seq_len", "seed", "meta", "text", "answer"]
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            ids_col = batch.column("input_ids").to_pylist()
            seqs = batch.column("seq_len").to_pylist()
            seeds = batch.column("seed").to_pylist()
            metas = batch.column("meta").to_pylist()
            texts = batch.column("text").to_pylist()
            anss = batch.column("answer").to_pylist()
            for ids, sl, row_seed, meta_s, text, answer in zip(
                ids_col, seqs, seeds, metas, texts, anss, strict=True
            ):
                if len(ids) != int(sl):
                    bad_len += 1
                if int(row_seed) != int(seed):
                    bad_seed += 1
                idh.add(_ids_hash(ids))
                try:
                    meta = json.loads(meta_s)
                    fp = json.dumps(meta.get("content_fingerprint", text), sort_keys=True)
                except (TypeError, json.JSONDecodeError):
                    fp = text
                fph.add(_str_hash(fp))
                txh.add(_str_hash(text))
                ans.add(answer)
        id_hashes[split] = idh
        fp_hashes[split] = fph
        text_hashes[split] = txh
        answers[split] = ans
    if bad_len:
        raise ValueError(f"{bad_len} parquet rows have len(input_ids) != seq_len")
    if bad_seed:
        raise ValueError(f"{bad_seed} parquet rows have seed != {seed}")
    leakage = _leakage_from_hash_sets(fp_hashes, id_hashes, text_hashes, answers, split_n)
    assert_no_content_leakage(leakage)
    return {"splits": split_n, "leakage": leakage, "seed": seed}


CARD_TEMPLATE = """---
language:
  - en
license: apache-2.0
pretty_name: {pretty}
size_categories:
  - {size_cat}
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

{hub_header}

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
    gzip_note = ""
    if stats.get("gzip_sampled"):
        gzip_note = f" (n={stats.get('gzip_n')} stratified sample)"
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
        f"| mean gzip ratio (text) | {stats.get('gzip_ratio_text_mean', 0):.3f}{gzip_note} |",
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

    hub_header = (
        f"**Hub:** [`{hub_id}`](https://huggingface.co/datasets/{hub_id})  \n"
        f"**Author:** Krzysztof Sopyła · **Family:** `{family}` · **Series:** CogitoProbe · "
        f"**Default seed:** `{seed}` · **Tokenizer:** `{tokenizer}`"
    )
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
        size_cat=size_category(int(stats.get("n_rows", 0))),
        hub_header=hub_header,
    )

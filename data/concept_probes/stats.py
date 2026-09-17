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


# Hub / dataset cards are written for an external reader. Internal research
# nicknames (experiment ids, DNA/Glyph) stay out of the 60-second lead.

FAMILY_PRETTY = {
    "bits": "CogitoProbe-Bits: key–value recall in a long haystack",
    "bind": "CogitoProbe-Bind: who-has-what entity binding",
    "arith": "CogitoProbe-Arith: nested arithmetic with mixed brackets",
    "props": "CogitoProbe-Props: remember the facts, ignore the filler",
}

FAMILY_LEAD = {
    "bits": (
        "Synthetic needle-in-a-haystack QA: random `key X val Y` facts sit at the "
        "start of a 1,024–32,768 token sequence, filler pads the middle, and the "
        "model must emit the values for a list of keys asked at the end. Use it to "
        "test memory, retrieval, or any compressed latent — no project background required."
    ),
    "bind": (
        "Synthetic people-and-attributes QA: each name gets a job, a city, a colour, "
        "and a friend. The model must answer who has which colour, who lives where, "
        "or where a person's *friend* lives. A bag-of-words embedding is not enough "
        "when everyone shares the same attribute vocabulary."
    ),
    "arith": (
        "Synthetic nested `+ - *` expressions with mixed brackets `()[]{}`. Three "
        "question types: the final number (`eval`, an easy shortcut), internal-node "
        "values (`subexpr`, the real test), and which closer matches an opener "
        "(`match`). Use it to test whether a model stored the *tree*, not just a calculator."
    ),
    "props": (
        "Synthetic fact-vs-filler QA: short sentences like `the baker dropped the red "
        "cup in paris`, then a long run of unrelated filler words. The model must "
        "return each object's colour. Shuffling filler must not change answers; "
        "shuffling the fact colours must."
    ),
}

FAMILY_SIXTY = {
    "bits": """1. Prefix lists facts: `key alice val red . key bob val blue .`
2. The rest of the sequence is filler (the haystack), padded to 1k, 4k, 8k, 16k, or 32k tokens.
3. The query asks for several keys at once: `Q alice bob`
4. The gold answer is the packed values, in the same order: `red blue`

Score the **answer span only** (`labels != -100`). A model that only models fluent filler will fail.""",
    "bind": """Each entity is a bundle of attributes, written as:

```
alice is baker . alice in paris . alice color red . alice friend bob .
bob is miner . bob in oslo . bob color green . bob friend alice .
```

Then filler, then one of three questions (packed over several names):

| `task` | Query looks like | Answer |
|---|---|---|
| `attr_color` | `Q color alice bob` | `red green` |
| `who_place` | `Q who in paris oslo` | `alice bob` |
| `hop_friend_place` | `Q hop place alice` | `oslo` (Bob's city) |

If colour lookup works but the friend-hop stays at chance, the model stored labels, not bindings.""",
    "arith": """Several independent expressions are written up front, then filler, then a question about the **first** (farthest) expression.

| `task` | What it asks | Treat as |
|---|---|---|
| `eval` | `Q eval` → the root number | **Control only.** One integer; a calculator shortcut. |
| `subexpr` | `Q sub 0 . 2 .` → values of internal nodes | **Primary.** Needs the expression tree. |
| `match` | `Q match <opener-index>` → matching closer index | **Primary.** Bracket matching, not arithmetic. |

Mixed `()[]{}` do **not** change the numeric value. They colour the brackets so matching is a real question. Do not report `eval` accuracy as “the model understands arithmetic.”""",
    "props": """Facts are atomic propositions:

```
the baker dropped the red cup in paris .
the miner dropped the blue hat in oslo .
```

Then filler, then `Q color cup hat` → `red blue`.

Two cheap sanity checks you can run without a special model:

- Shuffle filler tokens between the last fact and `Q`. Gold answers must stay the same.
- Shuffle colours inside `meta.propositions`. Gold answers must change.

A model that only tracks n-grams of the padding will fail the first check.""",
}

FAMILY_SCORE = {
    "bits": (
        "Plot teacher-forced token accuracy (and recovered bits, see below) against "
        "how much unique information the answer carries: roughly "
        "`n_query × log2(32)` because there are 32 possible values. "
        "`variant=fixed` keeps ~16 facts at every length — if accuracy falls from "
        "1k to 32k, that is a **length** failure. `variant=scaled` grows the fact "
        "table with length (up to 256 facts at 32k) — a drop there is a **capacity** "
        "failure. A dense Transformer should exceed ~75% packed-answer accuracy at "
        "`seq_len=1024` before you interpret any compressed-memory number."
    ),
    "bind": (
        "Report `attr_color`, `who_place`, and `hop_friend_place` separately. "
        "Colour-only success with hop at chance means the model stored a list of "
        "colours, not who-has-what. A strong extra check: swap one entity's colour "
        "in `meta` and require that entity's answer token to flip."
    ),
    "arith": (
        "Score `subexpr` and `match` as the real tasks. Score `eval` only as a "
        "shortcut control (~one integer). If only `eval` moves, the model is a "
        "calculator, not a structure memory. Digit strings are space-separated "
        "(`3 9` for 39, `- 7` for −7)."
    ),
    "props": (
        "Packed object→colour over the queried objects. The filler-shuffle vs "
        "proposition-shuffle pair above is the claim: the latent (or the hidden "
        "state) must carry the proposition set, not the n-gram statistics of filler."
    ),
}

SERIES_TABLE = """| Dataset | Job in one line | Typical use |
|---|---|---|
| [`ksopyla/cogito-probe-bits`](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | Recall values for keys buried in a haystack | Memory / retrieval / compression capacity |
| [`ksopyla/cogito-probe-bind`](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | Who has which colour, who lives where, friend's city | Compositional binding vs bag-of-words |
| [`ksopyla/cogito-probe-arith`](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | Nested arithmetic + bracket matching | Did it store the expression tree? |
| [`ksopyla/cogito-probe-props`](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | Object colours amid fluent filler | Facts vs padding statistics |"""

WHY_EXISTS = """Web text is locally predictable: a language model can look strong by guessing
nearby words without remembering a fact from thousands of tokens earlier.
These four datasets hide a **known set of facts** in a long padded haystack so
you can measure whether a model (or a small latent memory) actually stored them.

Each row tells you how many bits the answer is worth (`prize_bits`) and how far
the question sits from the last fact (`gap`). That is the whole point: the
information content is labelled, the distractor text is not the prize, and the
length is a ladder rather than a single context size.

Real rows use random **single-token** English-ish pieces from the Llama-3 /
SmolLM3 vocabulary (`gonzalez`, `oslo`, `validators`, …), not the toy names
`alice` / `bob` in the examples above. The grammar of the task is the same."""


def _quick_start(hub_id: str) -> str:
    return f"""```python
from datasets import load_dataset

ds = load_dataset("{hub_id}")
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
`HuggingFaceTB/SmolLM3-3B` (Llama-3 vocab) and must not be re-tokenized."""


SCHEMA_TABLE = """| column | meaning |
|---|---|
| `text` / `context` / `query` / `answer` | Readable surfaces. `text` is the full padded row. |
| `input_ids`, `attention_mask`, `labels` | Ready for causal LM training. `labels` is `-100` everywhere except the answer. |
| `seq_len`, `rung` | Padded length: 1024, 4096, 8192, 16384, or 32768. |
| `variant` | `fixed` = same number of facts as at 1k, longer haystack. `scaled` = more facts as the row gets longer. |
| `task` | Question type inside this family (see above). |
| `prize_bits` | Known information content of the gold answer (combinatorial lower bound). |
| `gap` | Tokens from the last evidence token to the start of the answer. |
| `answer_start` / `answer_end` / `evidence_end` | Character-free token indices into `input_ids`. |
| `meta` | JSON string: fact table, fingerprints, node values. |"""


LIMITATIONS = """- Not natural language. Atoms are verified 1-token pieces of the SmolLM3 / Llama-3
  vocab, chosen so each symbol is one id. Do not treat this as a human corpus.
- Answers are **packed** (several values in one span). Single-token labels are too
  sparse for a small latent channel to learn from.
- `prize_bits` is a counting lower bound on the answer, not a cross-entropy floor
  of a local language-model window.
- Arithmetic mixed brackets colour the tree; they do not change `+ - *` meaning.
  `eval`-only accuracy is not evidence of rich structure.
- Rows are padded with a repeating filler cycle, so gzip of the full `text` looks
  tiny. Compare `prize_bits`, not compressibility of the padded row."""


ORIGIN = """These files were built for a research project on compressing long
context into a small set of latent vectors (“concepts”), so the author
could ask *what those vectors actually store*. You do not need that project,
its training code, or its internal experiment log to use the datasets.

Project page: [ai.ksopyla.com](https://ai.ksopyla.com) ·
author: [Krzysztof Sopyła](https://github.com/ksopyla).
Generator: `data/concept_probes/` in the public research repo (MIT)."""


def _generation_block(family: str, scale: str, seed: int, tokenizer: str) -> str:
    return f"""Deterministic rebuild (does not upload):

```bash
uv run python scripts/build_concept_probe_datasets.py \\
  --scale {scale} --seed {seed} \\
  --tokenizer {tokenizer} \\
  --families {family} \\
  --out_dir Cache/concept_probes/{scale}
```

Ids are composed from a verified 1-token atom table of `{tokenizer}`.
Arithmetic rows **inject** bare digit and bracket ids; they do not BPE-encode a
glued string such as `(1+2)*[3-4]` (that merge path is not a well-defined alphabet)."""


EVAL_MD = FAMILY_SCORE  # kept name: tests / callers may import FAMILY_SCORE instead



def _configs_yaml(family: str) -> str:
    # Hub YAML linter requires data_files as a list of {split, path} objects, not a mapping.
    del family
    return "\n".join(
        [
            "  - config_name: default",
            "    data_files:",
            "      - split: train",
            "        path: train.parquet",
            "      - split: validation",
            "        path: validation.parquet",
            "      - split: test",
            "        path: test.parquet",
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
    pretty = FAMILY_PRETTY[family]
    size_cat = size_category(int(stats.get("n_rows", 0)))
    yaml = "\n".join(
        [
            "---",
            "language:",
            "  - en",
            "license: apache-2.0",
            f'pretty_name: "{pretty}"',
            "size_categories:",
            f"  - {size_cat}",
            "task_categories:",
            "  - question-answering",
            "  - text-generation",
            "tags:",
            "  - synthetic",
            "  - long-context",
            "  - retrieval",
            "  - needle-in-haystack",
            "  - question-answering",
            "  - cogito-probe",
            f"  - {family}",
            "  - research",
            "configs:",
            _configs_yaml(family),
            "dataset_info:",
            f"  dataset_name: {hub_id}",
            "---",
        ]
    )
    citation = "\n".join(
        [
            "```",
            "@misc{cogitoprobe2026,",
            "  title  = {CogitoProbe: synthetic long-haystack probes for memory and compression},",
            "  author = {Sopyła, Krzysztof},",
            "  year   = {2026},",
            f"  url    = {{https://huggingface.co/datasets/{hub_id}}},",
            f"  note   = {{Seed {seed}. Four families: bits, bind, arith, props.}},",
            "}",
            "```",
        ]
    )
    body = f"""# {pretty}

{FAMILY_LEAD[family]}

**Author:** Krzysztof Sopyła · **License:** Apache-2.0 · **Seed:** `{seed}` · **Tokenizer:** `{tokenizer}`

## In 60 seconds

{FAMILY_SIXTY[family]}

## Load it

{_quick_start(hub_id)}

## The four CogitoProbe datasets

{SERIES_TABLE}

Length ladder (every family): **1024 → 4096 → 8192 → 16384 → 32768**.
Half the rows are `fixed` (same fact count as at 1k, longer haystack), half are
`scaled` (more facts as the row grows).

## Why these exist

{WHY_EXISTS}

## How to score

Train or evaluate **only on the answer span**. Teacher-forced token accuracy on
`labels != -100` is the main number. Recovered bits against the labelled prize:

`max(0, prize_bits + Σ log2 p(gold_t))`

A decoder that cannot see tokens more than `gap` away must sit at chance — the
evidence is that far from the answer.

{FAMILY_SCORE[family]}

## Schema

{SCHEMA_TABLE}

## This build

{_stats_md(stats)}

### Split leakage

{_leakage_md(stats)}

Train / validation / test use disjoint random streams. A fingerprint of the
*facts* is checked for overlap. Shared answer *strings* (for example the same
8 colours) are expected and are **not** a leak.

## Rebuild

{_generation_block(family, scale, seed, tokenizer)}

## Limitations

{LIMITATIONS}

## Origin

{ORIGIN}

## License

Apache-2.0 for this synthetic dataset. No web scrapes, no personal data.
Generator code is MIT.

## Citation

{citation}
"""
    return yaml + "\n\n" + body.rstrip() + "\n"


def rewrite_cards_from_stats(
    stats_path: Path,
    cards_out: Path,
    *,
    families: list[str] | None = None,
) -> list[Path]:
    """Re-render Hub cards from an existing `cogito-probe-stats.json` (no parquet rebuild)."""
    payload = json.loads(Path(stats_path).read_text())
    seed = int(payload.get("seed", 20260916))
    tokenizer = str(payload.get("tokenizer", "HuggingFaceTB/SmolLM3-3B"))
    scale = str(payload.get("scale", "full"))
    fams = payload.get("families", {})
    wanted = list(families) if families else list(fams)
    written: list[Path] = []
    cards_out = Path(cards_out)
    cards_out.mkdir(parents=True, exist_ok=True)
    from data.concept_probes.schema import HUB_IDS

    for family in wanted:
        if family not in fams:
            raise KeyError(f"{family} missing from {stats_path}")
        card = render_card(
            family,
            fams[family],
            seed=seed,
            tokenizer=tokenizer,
            scale=scale,
            hub_id=HUB_IDS[family],
        )
        dest = cards_out / f"cogito-probe-{family}" / "README.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(card)
        written.append(dest)
    return written


#!/usr/bin/env python
"""Publish a text-checks data build (scripts/build_text_checks_data.py) to the Hugging Face Hub.

Exports the build to Parquet plus the tokenizer, the build metadata and the generator code, writes a
dataset card (README.md) and uploads it as one dataset repo:

    eval/{id,harder,paraphrase}/*.parquet   frozen exam items (text; prompt + evidence-removed twin)
    train/{world,stories}/*.parquet          the training mix as token ids (decode with tokenizer/)
    dev/{world,stories}/*.parquet            held-out rows for development loss
    tokenizer/                               the frozen 4,096-token byte-level BPE
    text_checks_meta.json                    versions, hashes, counts
    generator/                               the code that generated it (pure functions, seeded)

The token is read from $HF_TOKEN or from an env file (`HF_TOKEN=...`); it is never printed.

    uv run python scripts/publish_text_checks_dataset.py --data $TOK/text_checks_v0 --whoami
    uv run python scripts/publish_text_checks_dataset.py --data $TOK/text_checks_v0 --dry_run --export_dir /tmp/x
    uv run python scripts/publish_text_checks_dataset.py --data $TOK/text_checks_v0 --repo_id <user>/cogito-text-world
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

EVAL_SPLITS = ("id", "harder", "paraphrase")
TASK_ROWS = [
    ("T1", "quote", "What did the sign of Lumo say?", "4–6 common words, copied exactly", "1/#signs (1/4)"),
    ("T2", "lookup", "Where does Lumo live?", "an invented place; only the asked home is stated", "≈0"),
    ("T3", "keyed", "Where does Lumo live?", "an invented place among 16 homes, 1/4 look-alike names", "1/16"),
    ("T4", "latest", "Where does Lumo live now?", "the last of 4 moves; others move later", "1/5"),
    ("T5", "compose", "Where does the sister of the sister of Lumo live?", "follow 1–3 sister links, then the home", "1/8"),
    ("T6", "count", "How many times did Lumo visit the market?", "a number word zero–six (balanced)", "1/7"),
    ("T7", "deduce", "Is Pim shiny?", "yes/no from made-up category rules (balanced)", "1/2"),
]


def load_token(env_file: str | None) -> str:
    tok = os.environ.get("HF_TOKEN")
    candidates = [env_file] if env_file else [str(ROOT / ".env"), str(Path.home() / "dev" / "MrCogito" / ".env")]
    for f in candidates:
        if tok or not f or not Path(f).exists():
            continue
        for line in Path(f).read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("no HF_TOKEN in the environment or the env file")
    return tok


def export_eval(data: Path, out: Path, rows_per_file: int = 2000) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    counts = {}
    for split in EVAL_SPLITS:
        src = data / "eval" / f"{split}.jsonl"
        if not src.exists():
            continue
        d = out / "eval" / split
        d.mkdir(parents=True, exist_ok=True)
        buf, k, n = [], 0, 0

        def flush():
            nonlocal buf, k
            if not buf:
                return
            for r in buf:
                r["meta"] = json.dumps(r.get("meta", {}), sort_keys=True)
            pq.write_table(pa.Table.from_pylist(buf), d / f"{split}-{k:05d}.parquet", compression="zstd")
            buf, k = [], k + 1

        with src.open() as fh:
            for line in fh:
                buf.append(json.loads(line))
                n += 1
                if len(buf) >= rows_per_file:
                    flush()
        flush()
        counts[split] = n
    return counts


def export_rows(data: Path, out: Path, shard_rows: int = 20000) -> dict:
    from datasets import load_from_disk

    counts = {}
    for src_name, split, dst in (("world", "train", "train/world"), ("stories", "train", "train/stories"),
                                 ("world", "eval", "dev/world"), ("stories", "eval", "dev/stories")):
        p = data / src_name / split
        if not p.exists():
            continue
        ds = load_from_disk(str(p))
        d = out / dst
        d.mkdir(parents=True, exist_ok=True)
        n_shards = max(1, -(-len(ds) // shard_rows))
        for i in range(n_shards):
            ds.shard(n_shards, i, contiguous=True).to_parquet(str(d / f"{dst.replace('/', '-')}-{i:05d}.parquet"))
        counts[dst] = len(ds)
    return counts


def card(repo_id: str, meta: dict, eval_counts: dict, row_counts: dict) -> str:
    mix = meta.get("mix", {})
    a = meta.get("args", {})
    tot_world = row_counts.get("train/world", 0) * mix.get("mean_world_row_tokens", 0)
    tot_story = row_counts.get("train/stories", 0) * mix.get("mean_story_row_tokens", 0)
    n_total = sum(eval_counts.values())
    size = "10K<n<100K" if n_total < 100_000 else "100K<n<1M"
    lengths = ", ".join(f"{L // 1024}k" for L in a.get("eval_lengths", []))
    extra = ", ".join(f"{L // 1024}k" for L in a.get("extra_lengths", []))
    configs = [("eval_id", "eval/id/*.parquet"), ("eval_harder", "eval/harder/*.parquet"),
               ("eval_paraphrase", "eval/paraphrase/*.parquet"), ("train_world", "train/world/*.parquet"),
               ("train_stories", "train/stories/*.parquet"), ("dev_world", "dev/world/*.parquet"),
               ("dev_stories", "dev/stories/*.parquet")]
    cfg_yaml = "\n".join(f"- config_name: {n}\n  data_files:\n  - split: {'test' if n.startswith('eval') else ('validation' if n.startswith('dev') else 'train')}\n    path: {p}"
                         for n, p in configs)
    tasks_md = "\n".join(f"| {lv} | `{t}` | {q} | {a_} | {f} |" for lv, t, q, a_, f in TASK_ROWS)
    return f"""---
pretty_name: Cogito Text World (text capability checks)
license: cdla-sharing-1.0
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- synthetic
- long-context
- reasoning
- multi-hop
- fact-retrieval
- small-language-models
- tinystories
- evaluation
size_categories:
- {size}
configs:
{cfg_yaml}
---

# Cogito Text World

A from-scratch **training + evaluation** set for testing whether small language models (≈2M–150M parameters)
can learn simple language *and* learn to use facts written in it: copy, look up, pick among look-alikes,
track a changing fact, combine two facts, count, and chain rules — at document lengths from
{lengths} after training at {a.get('seq_len', 4096) // 1024}k.

Every exam item is a **story-world document**: real simple stories (TinyStories, SimpleStories) whose characters
are renamed to an invented cast, with templated fact sentences woven in, and one question at the end. Every
answer is computed by the generator, unique, and checkable by exact match.

It is the data of the *text capability checks* of the MrCogito project (a protocol for comparing new
long-context architectures — memory, recurrence, attention variants — at equal data, compute and parameter
budgets, trained from scratch). Data version **`{meta.get('version')}`**.

## The question types (levels)

| level | `task` | example question | what the answer is | guessing floor |
|---|---|---|---|---|
{tasks_md}

Each eval item also carries an **evidence-removed twin** (`prompt_removed`): the same document with the
sentences that answer the question replaced by filler. A model that really does the task must fall to the
guessing floor on the twin; a model that still answers has found a shortcut.

## Configs

| config | split | rows | content |
|---|---|---|---|
| `eval_id` | test | {eval_counts.get('id', 0):,} | in-distribution exams at {lengths} (400 items per task and length up to 16k, 200 above) |
| `eval_harder` | test | {eval_counts.get('harder', 0):,} | harder settings (64-person cast, 3 hops, 8 moves, depth-3 rules) at {extra} |
| `eval_paraphrase` | test | {eval_counts.get('paraphrase', 0):,} | sentence templates never used in training, at {extra} |
| `train_world` | train | {row_counts.get('train/world', 0):,} | world documents with question + answer, token ids (≈{tot_world / 1e9:.2f}B tokens) |
| `train_stories` | train | {row_counts.get('train/stories', 0):,} | plain stories packed to {a.get('seq_len', 4096)} tokens, token ids (≈{tot_story / 1e9:.2f}B tokens) |
| `dev_world`, `dev_stories` | validation | {row_counts.get('dev/world', 0):,} / {row_counts.get('dev/stories', 0):,} | held-out rows for development loss (recipe tuning) |

### Eval fields

| field | meaning |
|---|---|
| `prompt` | the document, ending in `Question: …\\nAnswer:` |
| `answer` | the scored continuation: a leading space, the answer, a closing period (e.g. `" Kodaford."`) |
| `candidates` | every answer of the asked type planted in the document (same format as `answer`) |
| `floor` | the guessing floor of this item (best score without doing the task) |
| `prompt_removed` | the evidence-removed twin |
| `task`, `level`, `split`, `length`, `depth`, `seed` | task id (T1–T7), split, target length in tokens, where the evidence sits (`early`/`middle`/`late`, or `spread`), generator seed |
| `n_tokens` | BOS + prompt + answer, in tokens of the bundled tokenizer |
| `meta` | JSON string: task dials (hops, moves, count, depth) and assembly info |

### Train fields
`input_ids` (int32, `<|bos|> … <|eos|>`), `attention_mask`, `special_tokens_mask`. One document per row; decode
with `tokenizer/`. The intended mix is ≈{100 * mix.get('lang_token_share', 0.45):.0f} % story tokens and
≈{100 - 100 * mix.get('lang_token_share', 0.45):.0f} % world tokens (row weights {', '.join(f'{w:.3f}' for w in mix.get('row_weights', []))},
mean row length {mix.get('mean_row_tokens', 0):.0f} tokens).

## Use

```python
from datasets import load_dataset
from transformers import AutoTokenizer

exams = load_dataset("{repo_id}", "eval_id", split="test")
tok = AutoTokenizer.from_pretrained("{repo_id}", subfolder="tokenizer")
train = load_dataset("{repo_id}", "train_world", split="train", streaming=True)
print(tok.decode(next(iter(train))["input_ids"]))
```

**Scoring (one forward pass per item).** Feed `BOS + prompt + answer` (teacher forcing). The item is correct
when the model's top token equals the gold token at every answer position, including the period — exactly
the condition under which greedy decoding writes the gold answer. Optionally also score the *picked
candidate*: the candidate whose first token is most probable at the answer position. Report every score next
to its guessing floor and next to the evidence-removed twin. A reference scorer is in the MrCogito repository
(`evaluation/text_checks_eval.py`).

**Train short, test long.** Train at {a.get('seq_len', 4096)} tokens; read the `eval_id` items at every length.
Longer documents add filler, not facts, so length is the only change.

## How it was built

- **Language**: TinyStories and SimpleStories train splits ({meta.get('stories', {}).get('train', 0):,} unique stories);
  their validation/test splits ({meta.get('stories', {}).get('heldout', 0):,} stories) are used **only** as filler
  in the exams and in `dev_stories`.
- **Cast and world**: invented names from a syllable grammar; one name in five (a fixed hash class) is
  reserved for the exams, so no exam name appears in training. Places are invented words, so place answers
  never occur in filler.
- **Filler hygiene**: common character names in filler stories are renamed to cast members (a name alone
  never locates a fact); any filler sentence that mentions a cast member together with a reserved relation
  word (lives, moved, sister, visited, sign, …) is dropped, so filler never states or contradicts a fact.
- **Facts**: ≥ 4 sentence templates per relation in training (one more held out for `eval_paraphrase`).
- **Questions**: one per document, at the end, after all evidence; answers are computed from the generator's
  own world state and unique.
- **Tokenizer**: a 4,096-token byte-level BPE trained on stories + world documents (`<|bos|>`, `<|eos|>`, `<|pad|>`).
- **Determinism**: every item is a pure function of (version, split, task, length, index) and the story pool;
  the code is in `generator/` (`text_world.py`, `build_text_checks_data.py`). Hashes are in `text_checks_meta.json`
  (tokenizer `{str(meta.get('tokenizer_sha256', ''))[:16]}…`).

## Limitations

- The language is deliberately simple (children's-story vocabulary) and the fact sentences are templated:
  this tests the *mechanisms* of retrieval and composition in small models, not knowledge or open-domain QA.
- Fact sentences differ in style from the filler; finding *a* fact is easier than in natural text. The keyed,
  latest and compose tasks therefore plant same-shaped decoy facts, and the twins expose shortcuts.
- Stories were generated by GPT-3.5/4 (TinyStories) and other LLMs (SimpleStories) and inherit their biases.
- The draft protocol is being calibrated; later versions may change task dials or the mix.

## License and attribution

Released under **CDLA-Sharing-1.0**, the license of TinyStories (share-alike). It contains text derived from:
- **TinyStories** — Eldan & Li, 2023, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
  (arXiv:2305.07759), CDLA-Sharing-1.0.
- **SimpleStories** — Finke et al., 2025 (arXiv:2504.09184), MIT.

## Citation

```bibtex
@misc{{cogito_text_world,
  title  = {{Cogito Text World: a story-world dataset for text capability checks of small language models}},
  author = {{Sopyła, Krzysztof}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note   = {{Data version {meta.get('version')}}}
}}
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--repo_id", default=None, help="default: <account>/cogito-text-world")
    p.add_argument("--env_file", default=None)
    p.add_argument("--export_dir", default=None, help="default: <data>/hub_export")
    p.add_argument("--private", action="store_true")
    p.add_argument("--whoami", action="store_true", help="only print the token's account")
    p.add_argument("--dry_run", action="store_true", help="export and write the card, do not upload")
    a = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=load_token(a.env_file))
    who = api.whoami()["name"]
    if a.whoami:
        print(f"token account: {who}")
        return
    data = Path(a.data)
    meta = json.loads((data / "text_checks_meta.json").read_text())
    repo_id = a.repo_id or f"{who}/cogito-text-world"
    out = Path(a.export_dir or data / "hub_export")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    eval_counts = export_eval(data, out)
    row_counts = export_rows(data, out)
    shutil.copytree(data / "tokenizer", out / "tokenizer")
    shutil.copy(data / "text_checks_meta.json", out / "text_checks_meta.json")
    (out / "generator").mkdir()
    for f in ("data/text_world.py", "scripts/build_text_checks_data.py"):
        shutil.copy(ROOT / f, out / "generator" / Path(f).name)
    (out / "README.md").write_text(card(repo_id, meta, eval_counts, row_counts))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"exported {out}: eval {eval_counts}, rows {row_counts}, {size / 1e9:.2f} GB")
    if a.dry_run:
        return
    api.create_repo(repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out))
    print(f"published https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()

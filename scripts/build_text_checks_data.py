#!/usr/bin/env python
"""Build the text capability checks data (draft `text-v0`): tokenizer, training mix, eval sets.

Spec: docs/engineering_specs/text_capability_checks.md (§8, §15). Generator: data/text_world.py.

Writes, under --out_dir:
  tokenizer/                 a frozen byte-level BPE (--vocab tokens) with <|bos|> <|eos|> <|pad|>
  stories/{train,eval}       language rows: whole stories concatenated to ~--seq_len tokens
  world/{train,eval}         world documents with a question and answer, log-uniform 256..seq_len
  manifest.json              the pretokenized-mix manifest the trainer reads (--pretokenized_manifest)
  eval/<split>.jsonl         frozen exam items (prompt, answer, candidates, floor, evidence-removed twin)
  text_checks_meta.json      versions, hashes, counts, token shares

Rows keep the LM shard schema (input_ids / attention_mask / special_tokens_mask), one document
per row, BOS … EOS, so they go through the normal causal-LM path with no special labels.

Smoke (local, minutes; filler from the small validation/test story files):
  uv run python scripts/build_text_checks_data.py --stories smoke --out_dir Cache/text_checks/smoke \
      --seq_len 1024 --n_lang_rows 300 --n_world_rows 600 --eval_lengths 512 1024 2048 --eval_items 6
Full (servers):
  uv run python scripts/build_text_checks_data.py --stories full --out_dir $DATA/text_checks_v0 \
      --seq_len 4096 --n_lang_rows ... --n_world_rows ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.text_world import (  # noqa: E402
    TASKS,
    TEXT_WORLD_VERSION,
    clean_story,
    make_eval_record,
    make_train_document,
)

STORY_FILES = {
    # repo: (train files, held-out files, text column candidates)
    "roneneldan/TinyStories": (
        [f"data/train-0000{i}-of-00004-{h}.parquet" for i, h in
         enumerate(["2d5a1467fff1081b", "5852b56a2bd28fd9", "a26307300439e943", "d243063613e5a057"])],
        ["data/validation-00000-of-00001-869c898b519ad725.parquet"],
    ),
    "SimpleStories/SimpleStories": (
        [f"data/train-0000{i}-of-00007.parquet" for i in range(7)],
        ["data/test-00000-of-00001.parquet"],
    ),
}
SPECIALS = ("<|bos|>", "<|eos|>", "<|pad|>")
SINGLE_EVIDENCE_TASKS = ("quote", "lookup", "keyed")


def _read_parquet_texts(repo: str, filename: str) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # token=False: public datasets; an expired saved login would otherwise make the Hub say 404.
    path = hf_hub_download(repo, filename, repo_type="dataset", token=False)
    table = pq.read_table(path)
    col = "text" if "text" in table.column_names else ("story" if "story" in table.column_names else table.column_names[0])
    return [clean_story(t) for t in table.column(col).to_pylist() if t]


def load_stories(mode: str, max_train_files: int, seed: int) -> tuple[list[str], list[str]]:
    """(train pool, held-out pool). Held-out stories are only ever used as evaluation filler
    and in the language eval rows."""
    train, held = [], []
    for repo, (tr_files, ho_files) in STORY_FILES.items():
        if mode == "smoke":
            texts = []
            for f in ho_files:
                texts += _read_parquet_texts(repo, f)
            for t in texts:  # split the small files 80/20 by hash
                (held if zlib.crc32(t.encode()) % 5 == 0 else train).append(t)
        else:
            for f in tr_files[:max_train_files]:
                train += _read_parquet_texts(repo, f)
            for f in ho_files:
                held += _read_parquet_texts(repo, f)
    train = sorted(set(t for t in train if 40 <= len(t) <= 4000))
    held = sorted(set(t for t in held if 40 <= len(t) <= 4000) - set(train))
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(held)
    return train, held


def train_tokenizer(texts, vocab: int, out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=list(SPECIALS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=SPECIALS[0], eos_token=SPECIALS[1],
                                   pad_token=SPECIALS[2], model_max_length=1 << 20)
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out))
    return fast


def _row(ids: list[int], bos: int, eos: int) -> dict:
    full = [bos] + ids + [eos]
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "special_tokens_mask": [1] + [0] * len(ids) + [1]}


def language_rows(stories: list[str], n: int, seq_len: int, encode, bos: int, eos: int) -> list[dict]:
    rows, cur, i = [], [], 0
    sep = encode("\n\n")
    while len(rows) < n and i < len(stories) * 4:
        ids = encode(stories[i % len(stories)])
        i += 1
        cur = cur + (sep if cur else []) + ids
        if len(cur) >= seq_len - 2:
            rows.append(_row(cur[: seq_len - 2], bos, eos))
            cur = []
    return rows


def world_rows(stories: list[str], n: int, seq_len: int, encode, bos: int, eos: int, seed: int,
               min_len: int = 256) -> list[dict]:
    rows = []
    ntok = lambda s: len(encode(s))  # noqa: E731
    rng = random.Random(seed)
    k = 0
    while len(rows) < n:
        target = int(math.exp(rng.uniform(math.log(min_len), math.log(seq_len - 2))))
        for shrink in (1.0, 0.9, 0.8, 0.6):
            text = make_train_document(seed * 1_000_003 + k, stories, int(target * shrink), ntok)
            ids = encode(text)
            if len(ids) <= seq_len - 2:
                rows.append(_row(ids, bos, eos))
                break
        k += 1
    return rows


FEATURES = None


def _features():
    from datasets import Features, Sequence, Value

    return Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
                     "special_tokens_mask": Sequence(Value("int8"))})


def _save(rows, path: Path):
    """Stream rows (a list or a generator) to disk with compact integer types."""
    from datasets import Dataset

    rows = list(rows) if not callable(rows) else rows
    gen = rows if callable(rows) else (lambda: (r for r in rows))
    Dataset.from_generator(gen, features=_features(), cache_dir=str(path) + ".cache").save_to_disk(str(path))
    shutil.rmtree(str(path) + ".cache", ignore_errors=True)


_WORKER: dict = {}


def _world_worker(job):
    """One shard of world rows (forked workers share the tokenizer and the story pool)."""
    shard, n, seq_len, seed, out = job
    tok, stories = _WORKER["tok"], _WORKER["stories"]
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    path = Path(out) / f"shard_{shard:04d}"
    _save(lambda: iter(world_rows(stories, n, seq_len, encode, tok.bos_token_id, tok.eos_token_id,
                                  seed=seed * 100_003 + shard)), path)
    return str(path)


def build_world_split(tok, stories, n_rows, seq_len, seed, out: Path, num_proc: int):
    from datasets import concatenate_datasets, load_from_disk

    _WORKER.update(tok=tok, stories=stories)
    per = max(1, math.ceil(n_rows / max(1, num_proc * 4)))
    jobs, left, k = [], n_rows, 0
    while left > 0:
        jobs.append((k, min(per, left), seq_len, seed, str(out) + ".shards"))
        left -= per
        k += 1
    if num_proc > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(num_proc) as pool:
            paths = pool.map(_world_worker, jobs)
    else:
        paths = [_world_worker(j) for j in jobs]
    ds = concatenate_datasets([load_from_disk(p) for p in paths])
    ds.save_to_disk(str(out))
    shutil.rmtree(str(out) + ".shards", ignore_errors=True)
    return ds


def _mean_len(ds) -> float:
    """Mean row length without materialising the rows (arrow list lengths)."""
    import pyarrow.compute as pc

    col = ds.data.column("input_ids")
    return float(pc.sum(pc.list_value_length(col)).as_py()) / max(1, len(ds))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stories", choices=("smoke", "full"), default="smoke")
    p.add_argument("--max_train_files", type=int, default=99, help="story train parquet files per source (full)")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--tokenizer_sample", type=int, default=200_000, help="stories used to train the BPE")
    p.add_argument("--n_lang_rows", type=int, default=1000)
    p.add_argument("--n_world_rows", type=int, default=2000)
    p.add_argument("--n_eval_lang_rows", type=int, default=64)
    p.add_argument("--n_eval_world_rows", type=int, default=128)
    p.add_argument("--target_tokens", type=float, default=0,
                   help="size the training mix to this many tokens (overrides --n_lang_rows/--n_world_rows)")
    p.add_argument("--num_proc", type=int, default=1, help="processes generating world rows")
    p.add_argument("--lang_token_share", type=float, default=0.45)
    p.add_argument("--eval_lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--extra_lengths", type=int, nargs="+", default=[4096, 16384],
                   help="lengths of the non-id splits (harder, paraphrase)")
    p.add_argument("--eval_items", type=int, default=400, help="items per (split, task, length) up to 16k")
    p.add_argument("--eval_items_long", type=int, default=200, help="items per cell above 16k")
    p.add_argument("--eval_splits", nargs="+", default=["id", "harder", "paraphrase"])
    p.add_argument("--eval_tasks", nargs="+", default=list(TASKS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_st, held_st = load_stories(args.stories, args.max_train_files, args.seed)
    print(f"stories: {len(train_st):,} train / {len(held_st):,} held-out ({time.time() - t0:.0f}s)")

    # tokenizer: stories + world documents (approximate lengths; names and places must be in its data)
    approx = lambda s: max(1, len(s) // 4)  # noqa: E731
    tok_texts = train_st[: args.tokenizer_sample] + [
        make_train_document(10_000_000 + i, train_st, int(math.exp(random.Random(i).uniform(math.log(256), math.log(4096)))), approx)
        for i in range(max(500, args.tokenizer_sample // 50))
    ]
    tok = train_tokenizer(tok_texts, args.vocab, out / "tokenizer")
    bos, eos = tok.bos_token_id, tok.eos_token_id
    cache: dict[str, list[int]] = {}

    def encode(s: str) -> list[int]:
        r = cache.get(s)
        if r is None:
            r = tok.encode(s, add_special_tokens=False)
            if len(s) < 400:
                if len(cache) > 2_000_000:
                    cache.clear()
                cache[s] = r
        return r

    print(f"tokenizer: {len(tok)} tokens ({time.time() - t0:.0f}s)")

    # training mix
    n_lang, n_world = args.n_lang_rows, args.n_world_rows
    if args.target_tokens:
        probe = world_rows(train_st, 200, args.seq_len, encode, bos, eos, seed=99)
        m_probe = sum(len(r["input_ids"]) for r in probe) / len(probe)
        n_lang = math.ceil(args.lang_token_share * args.target_tokens / args.seq_len)
        n_world = math.ceil((1 - args.lang_token_share) * args.target_tokens / m_probe)
        print(f"target {args.target_tokens:.3g} tokens → {n_lang:,} story rows + {n_world:,} world rows "
              f"(world rows ≈ {m_probe:.0f} tokens)")
    _save(lambda: iter(language_rows(train_st, n_lang, args.seq_len, encode, bos, eos)), out / "stories" / "train")
    _save(language_rows(held_st, args.n_eval_lang_rows, args.seq_len, encode, bos, eos), out / "stories" / "eval")
    world_tr = build_world_split(tok, train_st, n_world, args.seq_len, args.seed * 7 + 1, out / "world" / "train",
                                 args.num_proc)
    _save(world_rows(train_st, args.n_eval_world_rows, args.seq_len, encode, bos, eos, seed=args.seed * 7 + 2),
          out / "world" / "eval")
    from datasets import load_from_disk

    lang_tr = load_from_disk(str(out / "stories" / "train"))
    m_lang = _mean_len(lang_tr)
    m_world = _mean_len(world_tr)
    # row weights so the language part is `lang_token_share` of the TOKENS
    f = args.lang_token_share
    w_lang = f / m_lang
    w_world = (1 - f) / m_world
    w_lang, w_world = w_lang / (w_lang + w_world), w_world / (w_lang + w_world)
    manifest = {
        "mix_id": f"text_checks_{TEXT_WORLD_VERSION}",
        "max_seq_length": args.seq_len,
        "objective": "causal_lm",
        "seed": args.seed,
        "tokenizer": str((out / "tokenizer").resolve()),
        "sources": [
            {"name": "stories", "train_path": str((out / "stories" / "train").resolve()),
             "eval_path": str((out / "stories" / "eval").resolve()), "weight": w_lang, "in_eval": True},
            {"name": "world", "train_path": str((out / "world" / "train").resolve()),
             "eval_path": str((out / "world" / "eval").resolve()), "weight": w_world, "in_eval": True},
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"mix: {len(lang_tr)} story rows (mean {m_lang:.0f} tok), {len(world_tr)} world rows "
          f"(mean {m_world:.0f} tok), row weights {w_lang:.3f}/{w_world:.3f} ({time.time() - t0:.0f}s)")

    # frozen eval sets (held-out filler, held-out names)
    ntok = lambda s: len(encode(s))  # noqa: E731
    (out / "eval").mkdir(exist_ok=True)
    eval_hashes, eval_counts = {}, {}
    for split in args.eval_splits:
        path = out / "eval" / f"{split}.jsonl"
        n = 0
        with path.open("w") as fh:
            lengths = args.eval_lengths if split == "id" else args.extra_lengths
            for task in args.eval_tasks:
                for length in lengths:
                    base = zlib.crc32(f"{split}/{task}/{length}".encode()) * 10_000
                    n_items = args.eval_items if length <= 16384 else args.eval_items_long
                    for i in range(n_items):
                        depth = ("early", "middle", "late")[i % 3] if task in SINGLE_EVIDENCE_TASKS else None
                        rec = make_eval_record(task, base + i, split, length, depth, held_st, ntok)
                        rec["n_tokens"] = 1 + len(encode(rec["prompt"])) + len(encode(rec["answer"]))
                        fh.write(json.dumps(rec) + "\n")
                        n += 1
        eval_hashes[split] = _sha(path)
        eval_counts[split] = n
        print(f"eval {split}: {n} items ({time.time() - t0:.0f}s)")

    meta = {
        "version": TEXT_WORLD_VERSION,
        "args": vars(args),
        "tokenizer_sha256": _sha(out / "tokenizer" / "tokenizer.json"),
        "eval_sha256": eval_hashes,
        "eval_items": eval_counts,
        "stories": {"train": len(train_st), "heldout": len(held_st)},
        "mix": {"story_rows": len(lang_tr), "world_rows": len(world_tr), "mean_story_row_tokens": m_lang,
                "mean_world_row_tokens": m_world, "row_weights": [w_lang, w_world],
                "mean_row_tokens": w_lang * m_lang + w_world * m_world,
                "lang_token_share": f},
        "built_s": round(time.time() - t0, 1),
    }
    (out / "text_checks_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("version", "eval_items", "mix")}, indent=2))


if __name__ == "__main__":
    main()

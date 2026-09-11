#!/usr/bin/env python
"""Build dense-label keyed-recall rows (E18b) and merge them into a pretokenized LM manifest.

Each row is exactly `--context` tokens of *real text* (drawn from the base manifest's TRAIN
splits, never eval) with embedded retrieval items. An item is a random key followed by a value
(a natural-text span); later in the row the key repeats, framed by two reserved ids, and the
value must be reproduced:

    BOS  filler  KEY value  filler ... KEY value ... filler  KEY START value END  filler ...  EOS

Labels are not stored: the collator derives them from the markers (`loss_span_markers`), so the
rows keep the LM shard schema (input_ids / attention_mask / special_tokens_mask) and interleave
with ordinary sources. Every target follows its source by at least `--min_gap` tokens; sources
and targets are otherwise randomly placed. Values mix short (lookup-like) and long (span-copy)
lengths. No passkey / needle format is generated — that is the held-out transfer probe.

E21 (`--boundary_id`): one reserved boundary id is placed between the last source and the first
target (row length unchanged), so all sources are *sender* tokens and all targets *receiver*
tokens — the recall must pass through the compressed message slots of the Perceiver AR global read.

  uv run python scripts/build_retrieval_mix_dataset.py --base_manifest $M32 --fraction 0.05 \
      --n_train 6000 --n_eval 200 --out_dir $TOK/e18b_retrieval_32k \
      --out_manifest $TOK/e18b_lm_ret05_manifest.json

The merged manifest keeps the base sources (weights scaled by 1-w), adds the retrieval source
with `in_eval: false`, and records the derivation of the row weight `w` (weights are per ROW;
a 32k retrieval row vs a ~2.9k-token LM row means 5% of tokens is w ≈ 0.0046).
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

DEFAULT_START, DEFAULT_END = 128103, 128104  # <|reserved_special_token_95|>, <|reserved_special_token_96|> in the SmolLM3 tokenizer
DEFAULT_BOS, DEFAULT_EOS = 128000, 128012  # <|begin_of_text|>, <|im_end|> (the mix's append_eos_token_id)
LM_COLUMNS = ("input_ids", "attention_mask", "special_tokens_mask")


def row_weight_for_token_share(fraction: float, base_mean_row_tokens: float, context: int) -> float:
    """Row-sampling weight w such that retrieval rows make up `fraction` of TOKENS.

    Token share = w·S / (w·S + (1−w)·m̄)  ⇒  w = f·m̄ / (S − f·(S − m̄)).
    """
    f, m, S = float(fraction), float(base_mean_row_tokens), float(context)
    if not 0.0 < f < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    if m <= 0 or S <= 0:
        raise ValueError("base_mean_row_tokens and context must be positive")
    return f * m / (S - f * (S - m))


def token_share_for_row_weight(w: float, base_mean_row_tokens: float, context: int) -> float:
    return w * context / (w * context + (1.0 - w) * base_mean_row_tokens)


class FillerStream:
    """Deterministic stream of real-text ids from the base sources' train splits.

    Rows are visited in a seeded random order; a leading BOS is stripped from every piece so the
    only BOS in a built row is its first token. Pieces are concatenated as needed.
    """

    def __init__(self, train_paths, rng: np.random.Generator, bos: int, rows_cap: int | None = None):
        self.parts = [load_from_disk(p) for p in train_paths]
        self.rng = rng
        self.bos = bos
        self.order = []
        for pi, part in enumerate(self.parts):
            n = len(part) if rows_cap is None else min(len(part), rows_cap)
            idx = rng.permutation(len(part))[:n]
            self.order.extend((pi, int(i)) for i in idx)
        rng.shuffle(self.order)
        if not self.order:
            raise ValueError("no filler rows available")
        self.pos = 0
        self.buf: list[int] = []

    def _refill(self):
        pi, ri = self.order[self.pos % len(self.order)]
        self.pos += 1
        ids = list(self.parts[pi][ri]["input_ids"])
        if ids and ids[0] == self.bos:
            ids = ids[1:]
        self.buf.extend(int(t) for t in ids)

    def take(self, n: int) -> list[int]:
        while len(self.buf) < n:
            self._refill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


def sample_items(rng, n_items, short_len, span_len, key_lo, key_hi, key_len, p_short=0.5):
    items = []
    for _ in range(n_items):
        if rng.random() < p_short:
            L = int(rng.integers(short_len[0], short_len[1] + 1))
        else:
            L = int(rng.integers(span_len[0], span_len[1] + 1))
        key = [int(k) for k in rng.integers(key_lo, key_hi, size=key_len)]
        items.append((key, L))
    return items


def build_row(rng, filler: FillerStream, *, context, items, short_len, span_len, min_gap, key_lo, key_hi,
              key_len, start, end, bos, eos, min_chunk=16, max_item_frac=0.6, boundary=None):
    """One row of exactly `context` ids + its special_tokens_mask. Returns (ids, special).

    `boundary` (E21): reserved id placed at a random offset inside chunk n — the filler between the
    last source and the first target — replacing one filler token, so every source is a *sender*
    token and every target a *receiver* token and recall must flow through the message slots.
    """
    S = context
    for _ in range(50):
        n = int(rng.integers(items[0], items[1] + 1))
        its = sample_items(rng, n, short_len, span_len, key_lo, key_hi, key_len)
        T = sum(2 * L + 2 * key_len + 2 for _, L in its)  # source key+value, target key+START+value+END
        F = S - 2 - T
        if T <= max_item_frac * S and F >= min_gap + (2 * n + 1) * min_chunk:
            break
    else:
        raise ValueError("cannot fit items into the context; lower --items / --span_len or raise --context")
    # 2n+1 filler chunks; chunk n (between the last source and the first target) carries the min_gap
    k = 2 * n + 1
    w = rng.random(k) + 0.05
    w /= w.sum()
    spare = F - min_gap - k * min_chunk
    chunks = (np.floor(w * spare).astype(np.int64) + min_chunk).tolist()
    chunks[n] += min_gap
    chunks[-1] += F - sum(chunks)
    assert sum(chunks) == F and min(chunks) >= min_chunk
    values = [filler.take(L) for _, L in its]
    ids: list[int] = [bos]
    special: list[int] = [1]

    def add_filler(m):
        ids.extend(filler.take(m))
        special.extend([0] * m)

    def add_chunk(idx):
        if boundary is not None and idx == n:
            before = int(rng.integers(0, chunks[idx]))
            add_filler(before)
            ids.append(boundary); special.append(1)
            add_filler(chunks[idx] - 1 - before)
        else:
            add_filler(chunks[idx])

    ci = 0
    add_chunk(ci); ci += 1
    for i in rng.permutation(n):
        key, _ = its[i]
        ids.extend(key + values[i]); special.extend([0] * (len(key) + len(values[i])))
        add_chunk(ci); ci += 1
    for i in rng.permutation(n):
        key, _ = its[i]
        ids.extend(key); special.extend([0] * len(key))
        ids.append(start); special.append(1)
        ids.extend(values[i]); special.extend([0] * len(values[i]))
        ids.append(end); special.append(1)
        add_chunk(ci); ci += 1
    ids.append(eos); special.append(1)
    assert len(ids) == S and len(special) == S, (len(ids), S)
    # filler pieces keep their internal document terminators (and a value span may cross one):
    # the mask must flag every special id wherever it sits, exactly as the tokenizer would.
    specials = {bos, eos, start, end} | ({boundary} if boundary is not None else set())
    special = [1 if t in specials else 0 for t in ids]
    return ids, special


def iter_rows(n_rows, train_paths, seed, **kw):
    rng = np.random.default_rng(seed)
    filler = FillerStream(train_paths, rng, kw["bos"], rows_cap=kw.pop("rows_cap", None))
    for _ in range(n_rows):
        ids, special = build_row(rng, filler, **kw)
        yield {"input_ids": ids, "attention_mask": [1] * len(ids), "special_tokens_mask": special}


def base_mean_row_tokens(base_manifest: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    lengths_dir = Path(f"{base_manifest}.lengths")
    if lengths_dir.exists():
        ds = load_from_disk(str(lengths_dir))
        return float(np.asarray(ds["length"], dtype=np.float64).mean())
    stats = Path(f"{base_manifest}.token_stats.json")
    if stats.exists():
        return float(json.loads(stats.read_text())["average_tokens_per_row"])
    raise SystemExit(
        f"no length cache ({lengths_dir}) or token stats next to {base_manifest}; pass --base_mean_row_tokens"
    )


def merge_manifest(base: dict, retrieval_src: dict, weight: float, meta: dict) -> dict:
    out = json.loads(json.dumps(base))
    for src in out["sources"]:
        src["weight"] = float(src.get("weight", 1.0)) * (1.0 - weight)
    out["sources"].append({**retrieval_src, "weight": float(weight), "in_eval": False})
    out["mix_id"] = f"{base.get('mix_id', 'mix')}+{retrieval_src['name']}"
    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["retrieval_meta"] = meta
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_manifest", required=True)
    p.add_argument("--fraction", type=float, default=0.05, help="target share of TOKENS from retrieval rows")
    p.add_argument("--n_train", type=int, default=6000)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_manifest", required=True)
    p.add_argument("--name", default="retrieval_keyed_recall")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_id", type=int, default=DEFAULT_START)
    p.add_argument("--end_id", type=int, default=DEFAULT_END)
    p.add_argument("--bos", type=int, default=DEFAULT_BOS)
    p.add_argument("--eos", type=int, default=DEFAULT_EOS)
    p.add_argument("--key_lo", type=int, default=1000)
    p.add_argument("--key_hi", type=int, default=2000)
    p.add_argument("--key_len", type=int, default=3)
    p.add_argument("--items", type=int, nargs=2, default=(8, 24))
    p.add_argument("--short_len", type=int, nargs=2, default=(8, 16))
    p.add_argument("--span_len", type=int, nargs=2, default=(64, 512))
    p.add_argument("--min_gap", type=int, default=1024)
    p.add_argument("--filler_sources", default="", help="comma list of base source names (default: all)")
    p.add_argument("--filler_rows_cap", type=int, default=20000, help="rows visited per filler source")
    p.add_argument("--base_mean_row_tokens", type=float, default=None)
    p.add_argument("--boundary_id", type=int, default=None,
                   help="E21: reserved sender|receiver boundary id placed between the last source and the first "
                        "target (default: none; 128105 = <|reserved_special_token_97|>)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.boundary_id is not None and args.boundary_id in (args.start_id, args.end_id, args.bos, args.eos):
        raise SystemExit("--boundary_id must differ from --start_id/--end_id/--bos/--eos")

    base_path = Path(args.base_manifest)
    base = json.loads(base_path.read_text())
    names = [x.strip() for x in args.filler_sources.split(",") if x.strip()]
    sources = [s for s in base["sources"] if not names or s["name"] in names]
    if not sources:
        raise SystemExit("no filler sources selected")
    train_paths = [s["train_path"] for s in sources]
    feats = load_from_disk(train_paths[0]).features
    if set(feats.keys()) != set(LM_COLUMNS):
        raise SystemExit(f"base shards must carry exactly {LM_COLUMNS}, got {list(feats.keys())}")

    out = Path(args.out_dir)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    kw = dict(
        context=args.context, items=tuple(args.items), short_len=tuple(args.short_len),
        span_len=tuple(args.span_len), min_gap=args.min_gap, key_lo=args.key_lo, key_hi=args.key_hi,
        key_len=args.key_len, start=args.start_id, end=args.end_id, bos=args.bos, eos=args.eos,
        rows_cap=args.filler_rows_cap, boundary=args.boundary_id,
    )
    for split, n, seed in (("train", args.n_train, args.seed), ("eval", args.n_eval, args.seed + 1)):
        # NOTE: list-valued gen_kwargs are treated as shards by `from_generator` (the generator
        # would run once per path and multiply the rows); pass the paths as a tuple.
        ds = Dataset.from_generator(
            iter_rows, features=feats, keep_in_memory=False,
            gen_kwargs=dict(n_rows=n, train_paths=tuple(train_paths), seed=seed, **kw),
        )
        ds.save_to_disk(str(out / split))
        print(f"{split}: {len(ds):,} rows × {args.context} tokens -> {out / split}")

    m_bar = base_mean_row_tokens(base_path, args.base_mean_row_tokens)
    w = row_weight_for_token_share(args.fraction, m_bar, args.context)
    meta = {
        "start_id": args.start_id, "end_id": args.end_id, "bos": args.bos, "eos": args.eos,
        "key_range": [args.key_lo, args.key_hi], "key_len": args.key_len, "items": list(args.items),
        "short_len": list(args.short_len), "span_len": list(args.span_len), "min_gap": args.min_gap,
        "seed": args.seed, "context": args.context, "filler_sources": [s["name"] for s in sources],
        "boundary_id": args.boundary_id,
        "base_mean_row_tokens": m_bar, "target_token_fraction": args.fraction, "row_weight": w,
        "achieved_token_fraction": token_share_for_row_weight(w, m_bar, args.context),
    }
    src = {
        "name": args.name, "path": str(out), "train_path": str(out / "train"), "eval_path": str(out / "eval"),
        "num_train_rows": args.n_train, "num_eval_rows": args.n_eval,
    }
    merged = merge_manifest(base, src, w, meta)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(json.dumps(merged, indent=2))
    print(json.dumps({"row_weight": w, "base_mean_row_tokens": m_bar, "manifest": args.out_manifest}, indent=2))


if __name__ == "__main__":
    main()

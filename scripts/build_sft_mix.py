"""E37 instruction tuning (SFT): build the mix as GPT-2 token shards plus parallel loss-mask shards.

Spec: docs/experiments_specs/ahead/E37_sft_sensitivity.md (Data, Format). Output is read by data/bin_token_dataset.py
(per-source "masks") through scripts/run_e33n.py --phase sft and scripts/lsa_orig.py --phase sft.

Format (GPT-2's existing tokens, no new embeddings). A conversation is
    <|endoftext|>  then, per message,  "<|{role}|>\\n" + content + "\\n<|end|>\\n"      (role: system / user / assistant)
Each piece is tokenized on its own and the ids concatenated, so the mask boundaries are exact. Literal special-token
strings inside a message are split as ordinary text, so <|endoftext|> (50256) only ever opens a conversation or pads
a row. Loss mask: 1 on the assistant content and its "\\n<|end|>\\n" (the model learns to stop), 0 on everything else
(the opening <|endoftext|>, every role header, system and user turns, padding). Prompting a tuned model uses the same
pieces: ChatFormat.encode(messages, generation_prompt=True) ends with "<|assistant|>\\n".

Train split: conversations longer than seq_len are dropped (never cut) and counted; every LongAlign conversation that
fits is kept; then the longest prefix of a seeded permutation of the other conversations that packs, together with
LongAlign, into round(tokens / seq_len) rows (tokens counts padding). Test split: every conversation that fits.
Packing: best-fit decreasing (longest conversation first, into the row with the least room that still holds it) into
rows of exactly seq_len, the row tail padded with <|endoftext|> (mask 0); then the order of conversations inside each
row and the order of rows are shuffled (seeded). Every shard holds whole rows, so the "stream" rule of
BinRowDataset reads exactly the packed rows.

Writes to --out:
  smol_smoltalk_{train,test}_NNN.bin    token shards (modded-nanogpt layout, data.bin_token_dataset.write_shard)
  smol_smoltalk_{train,test}_NNN.mask   the loss mask, same layout (uint16 0/1), token for token
  manifest_sft.json                     rule "stream"; train = smol_smoltalk (masks, weight 1 - replay_share) +
                                        longdocs_replay (LongDocs-lite train text, no masks = loss on every token,
                                        the decay's one-document rows, weight replay_share); eval = the test rows
  stats.json                            conversations kept / dropped per source, tokens, answer share, rows

Usage (on the server, CPU only):
  uv run python scripts/build_sft_mix.py --out $DATA/sft_smoltalk_16k --tokenizer $DATA/tokenizer_gpt2 \\
      --replay_glob "$DATA/longdocs_lite_gpt2/*_train_*.bin"
  (offline: --parquet_dir DIR with the dataset's parquet files; names holding "train" / "test" pick the split)
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.bin_token_dataset import load_bin_manifest, write_shard  # noqa: E402

EOT = 50256
ROLES = ("system", "user", "assistant")
LONG_SOURCE = "longalign"
END = "\n<|end|>\n"


class ChatFormat:
    """The SFT chat format on a GPT-2 tokenizer: ids and loss mask of whole conversations, piece by piece."""

    def __init__(self, tokenizer):
        if tokenizer.convert_tokens_to_ids("<|endoftext|>") != EOT:
            raise ValueError("the SFT format needs the GPT-2 tokenizer (<|endoftext|> = 50256)")
        tokenizer.model_max_length = int(1e12)  # long LongAlign contents: no length warning, nothing is cut here
        self.tok = tokenizer
        self.header = {r: self._ids([f"<|{r}|>\n"])[0] for r in ROLES}
        self.end = self._ids([END])[0]

    def _ids(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self.tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def encode_batch(self, conversations: Sequence[Sequence[dict]]) -> list[tuple[np.ndarray, np.ndarray] | str]:
        """Per conversation (ids uint16 [n], mask uint8 [n]), or the reason it is unusable ("bad_role", "no_answer")."""
        flat = [(m.get("content") or "") for c in conversations for m in c]
        content = iter(self._ids(flat))
        out: list[tuple[np.ndarray, np.ndarray] | str] = []
        for conv in conversations:
            ids, mask, bad = [EOT], [0], False
            for m in conv:
                c = next(content)
                role = m.get("role")
                if role not in self.header:
                    bad = True
                    continue
                a = int(role == "assistant")
                ids += self.header[role] + c + self.end
                mask += [0] * len(self.header[role]) + [a] * (len(c) + len(self.end))
            if bad:
                out.append("bad_role")
            elif not any(mask):
                out.append("no_answer")
            else:
                out.append((np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)))
        return out

    def encode(self, messages: Sequence[dict], generation_prompt: bool = False) -> tuple[list[int], list[int]]:
        """One conversation; with generation_prompt the ids end with "<|assistant|>\\n" (prompting a tuned model)."""
        r = self.encode_batch([messages])[0]
        if isinstance(r, str) and not (generation_prompt and r == "no_answer"):
            raise ValueError(f"conversation unusable: {r}")
        if isinstance(r, str):  # a prompt without an answer yet
            ids, mask = [EOT], [0]
            for m, c in zip(messages, self._ids([(m.get("content") or "") for m in messages])):
                ids += self.header[m["role"]] + c + self.end
                mask += [0] * (len(self.header[m["role"]]) + len(c) + len(self.end))
        else:
            ids, mask = r[0].tolist(), r[1].tolist()
        if generation_prompt:
            ids += self.header["assistant"]
            mask += [0] * len(self.header["assistant"])
        return ids, mask


def pack_rows(lengths: Sequence[int], cap: int) -> list[list[int]]:
    """Best-fit decreasing: item indices per row; every row's items sum to <= cap (items longer than cap: error)."""
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    rows: list[list[int]] = []
    room: list[tuple[int, int]] = []  # (free tokens, row), sorted; full rows leave
    for i in order:
        n = int(lengths[i])
        if n > cap:
            raise ValueError(f"item {i} has {n} tokens > row {cap}")
        j = bisect.bisect_left(room, (n, -1))
        if j < len(room):
            free, r = room.pop(j)
        else:
            free, r = cap, len(rows)
            rows.append([])
        rows[r].append(i)
        if free - n > 0:
            bisect.insort(room, (free - n, r))
    return rows


def select_prefix(fixed: Sequence[int], candidates: Sequence[int], cap: int, target_rows: int) -> int:
    """The largest k such that fixed + candidates[:k] packs into at most target_rows rows (0 when the fixed items
    alone need more)."""
    def rows(k: int) -> int:
        return len(pack_rows(list(fixed) + list(candidates[:k]), cap))

    if rows(0) >= target_rows:
        return 0
    lo, hi = 0, len(candidates)
    if rows(hi) <= target_rows:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rows(mid) <= target_rows:
            lo = mid
        else:
            hi = mid - 1
    return lo


def write_rows(out: Path, prefix: str, convs: list[tuple[np.ndarray, np.ndarray]], rows: list[list[int]], S: int,
               rng: np.random.Generator, shard_rows: int) -> tuple[list[str], list[str]]:
    """Rows of whole conversations (order inside a row and of rows shuffled), padded to S with <|endoftext|> (mask 0),
    into token shards and mask shards of at most shard_rows rows each."""
    rows = [rows[i] for i in rng.permutation(len(rows))]
    files, masks = [], []
    for s, a in enumerate(range(0, len(rows), shard_rows)):
        chunk = rows[a: a + shard_rows]
        ids = np.full((len(chunk), S), EOT, dtype=np.uint16)
        msk = np.zeros((len(chunk), S), dtype=np.uint16)
        for r, items in enumerate(chunk):
            t = 0
            for i in (items[j] for j in rng.permutation(len(items))):
                c_ids, c_mask = convs[i]
                ids[r, t: t + len(c_ids)] = c_ids
                msk[r, t: t + len(c_ids)] = c_mask
                t += len(c_ids)
        f, m = out / f"{prefix}_{s:03d}.bin", out / f"{prefix}_{s:03d}.mask"
        write_shard(f, ids.reshape(-1))
        write_shard(m, msk.reshape(-1))
        files.append(str(f))
        masks.append(str(m))
    return files, masks


def load_split(args, split: str):
    """The split as a pyarrow Table with columns messages, source."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.parquet_dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.parquet_dir, "**", "*.parquet"), recursive=True)
                       if split in Path(p).name)
        if not paths:
            raise FileNotFoundError(f"no parquet file with '{split}' in its name under {args.parquet_dir}")
        return pa.concat_tables([pq.read_table(p, columns=["messages", "source"]) for p in paths])
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=split)
    t = getattr(ds.data, "table", ds.data)
    return t.select(["messages", "source"])


def encode_rows(fmt: ChatFormat, table, idx: Sequence[int], batch: int = 2048) -> Iterable[tuple[int, object]]:
    """(row index, ChatFormat result) in the order of idx."""
    import pyarrow as pa

    for a in range(0, len(idx), batch):
        part = list(idx[a: a + batch])
        msgs = table.take(pa.array(part, type=pa.int64())).column("messages").to_pylist()
        yield from zip(part, fmt.encode_batch(msgs))


def _split_stats(convs, srcs, rows, S, counts) -> dict:
    content = sum(len(c[0]) for c in convs)
    answer = sum(int(c[1].sum()) for c in convs)
    per = defaultdict(lambda: {"kept": 0, "tokens": 0, "answer_tokens": 0})
    for (ids, mask), s in zip(convs, srcs):
        per[s]["kept"] += 1
        per[s]["tokens"] += len(ids)
        per[s]["answer_tokens"] += int(mask.sum())
    sources = {}
    for s in sorted(set(counts) | set(per)):
        c = counts.get(s, Counter())
        # encoded: rows of the source read in the sample (kept + dropped); the rest were not drawn
        sources[s] = {**per[s], "available": c["available"], "encoded": c["encoded"],
                      "dropped_too_long": c["too_long"], "dropped_no_answer": c["no_answer"],
                      "dropped_bad_role": c["bad_role"]}
    return {"rows": len(rows), "row_tokens": len(rows) * S, "conversations": len(convs), "content_tokens": content,
            "pad_tokens": len(rows) * S - content, "fill": content / max(1, len(rows) * S),
            "answer_tokens": answer, "answer_share_of_content": answer / max(1, content),
            "answer_share_of_row_tokens": answer / max(1, len(rows) * S), "sources": sources}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="openai-community/gpt2", help="path or hub id of the GPT-2 tokenizer")
    ap.add_argument("--seq_len", type=int, default=16384)
    ap.add_argument("--tokens", type=float, default=180e6, help="instruction tokens in the train rows, padding included")
    ap.add_argument("--seed", type=int, default=42)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    src.add_argument("--parquet_dir", help="offline: a folder with the dataset's parquet files")
    ap.add_argument("--replay_glob", help="LongDocs-lite train shards, e.g. DATA/longdocs_lite_gpt2/*_train_*.bin")
    ap.add_argument("--replay_share", type=float, default=0.1, help="share of train rows drawn from the replay")
    ap.add_argument("--test_rows", type=int, default=256, help="eval rows the manifest asks for (of the test split)")
    ap.add_argument("--shard_tokens", type=float, default=1e8, help="tokens per shard (rounded down to whole rows)")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    t0 = time.time()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    S = int(a.seq_len)
    target_rows = int(round(a.tokens / S))
    shard_rows = max(1, int(a.shard_tokens) // S)
    fmt = ChatFormat(AutoTokenizer.from_pretrained(a.tokenizer))
    stats: dict = {"seq_len": S, "seed": a.seed, "tokens_target": a.tokens, "target_rows": target_rows,
                   "tokenizer": a.tokenizer, "dataset": a.parquet_dir or a.dataset, "format": {
                       "open": "<|endoftext|>", "turn": "<|{role}|>\\n{content}\\n<|end|>\\n",
                       "loss": "assistant content + its \\n<|end|>\\n"}}

    def encode_split(table, idx, keep: list, kept_src: list, stop_tokens: float | None = None) -> list:
        """Encode rows of `table` in the order of idx; whole conversations that fit go to keep / kept_src. Returns
        (source, outcome) per row read, in order (outcome: kept / too_long / no_answer / bad_role). With stop_tokens,
        stops after the batch in which the kept tokens reach it."""
        got, events = 0, []
        srcs = table.column("source")
        for i, r in encode_rows(fmt, table, idx):
            s = srcs[i].as_py()
            if isinstance(r, str):
                events.append((s, r))
            elif len(r[0]) > S:
                events.append((s, "too_long"))
            else:
                keep.append(r)
                kept_src.append(s)
                got += len(r[0])
                events.append((s, "kept"))
            if stop_tokens is not None and got >= stop_tokens and len(events) % 2048 == 0:
                break
        return events

    def count(counts, events) -> None:
        for s, what in events:
            counts[s]["encoded"] += 1
            counts[s][what] += 1

    # ---- train: every LongAlign conversation that fits, then a seeded prefix of the others
    train = load_split(a, "train")
    sources = train.column("source").to_pylist()
    counts: dict = defaultdict(Counter)
    for s in sources:
        counts[s]["available"] += 1
    long_idx = [i for i, s in enumerate(sources) if s == LONG_SOURCE]
    other_idx = np.asarray([i for i, s in enumerate(sources) if s != LONG_SOURCE], dtype=np.int64)
    rng = np.random.default_rng(a.seed)
    other_idx = other_idx[rng.permutation(len(other_idx))]
    fixed, fixed_src = [], []
    count(counts, encode_split(train, long_idx, fixed, fixed_src))
    cand, cand_src, ev = [], [], []
    # enough candidates to fill target_rows even at 100 % fill (then the prefix search cuts back)
    need = target_rows * S - sum(len(c[0]) for c in fixed)
    if need > 0:
        ev = encode_split(train, other_idx.tolist(), cand, cand_src, stop_tokens=need)
    k = select_prefix([len(c[0]) for c in fixed], [len(c[0]) for c in cand], S, target_rows)
    if k == 0 and fixed and len(pack_rows([len(c[0]) for c in fixed], S)) > target_rows:
        print(f"warning: LongAlign alone needs more than {target_rows} rows; no other conversation added",
              file=sys.stderr)
    # the sample: the permutation's prefix up to its k-th kept conversation (drops counted inside it only)
    kept_pos = [j for j, (_, what) in enumerate(ev) if what == "kept"]
    count(counts, ev[: kept_pos[k - 1] + 1] if k else [])
    convs, conv_src = fixed + cand[:k], fixed_src + cand_src[:k]
    rows = pack_rows([len(c[0]) for c in convs], S)
    tr_files, tr_masks = write_rows(out, "smol_smoltalk_train", convs, rows, S, np.random.default_rng(a.seed + 1),
                                    shard_rows)
    stats["train"] = _split_stats(convs, conv_src, rows, S, counts)
    del convs, cand, fixed

    # ---- test: every conversation that fits
    test = load_split(a, "test")
    tcounts: dict = defaultdict(Counter)
    for s in test.column("source").to_pylist():
        tcounts[s]["available"] += 1
    tconvs, tsrc = [], []
    count(tcounts, encode_split(test, list(range(test.num_rows)), tconvs, tsrc))
    trows = pack_rows([len(c[0]) for c in tconvs], S)
    te_files, te_masks = write_rows(out, "smol_smoltalk_test", tconvs, trows, S, np.random.default_rng(a.seed + 2),
                                    shard_rows)
    stats["test"] = _split_stats(tconvs, tsrc, trows, S, tcounts)

    # ---- manifest
    share = float(a.replay_share) if a.replay_glob else 0.0
    train_src = [{"name": "smol_smoltalk", "files": tr_files, "masks": tr_masks, "weight": 1.0 - share}]
    if a.replay_glob:
        # the decay's rows (one contiguous span of one document), loss on every token (no masks)
        train_src.append({"name": "longdocs_replay", "files": [os.path.abspath(a.replay_glob)], "weight": share,
                          "rule": "doc"})
    manifest = {"seq_len": S, "rule": "stream", "eot": EOT, "seed": a.seed, "train": train_src,
                "eval": [{"name": "smol_smoltalk_test", "files": te_files, "masks": te_masks,
                          "rows": int(a.test_rows)}],
                "sft": {"builder": "scripts/build_sft_mix.py", "stats": str(out / "stats.json"),
                        "format": stats["format"]}}
    (out / "manifest_sft.json").write_text(json.dumps(manifest, indent=2))

    # ---- what the trainer will actually draw (BinRowDataset's weight → row arithmetic)
    tr_ds, ev_ds, _ = load_bin_manifest(str(out / "manifest_sft.json"))
    per = tr_ds.rows_per_source
    stats["dataset"] = {"train_rows_per_source": per, "train_rows": len(tr_ds), "eval_rows": len(ev_ds or []),
                        "replay_share_of_rows": per.get("longdocs_replay", 0) / max(1, len(tr_ds)),
                        "train_tokens_per_epoch": len(tr_ds) * S}
    # BinRowDataset takes floor(w_i * min_j(rows_j / w_j)) rows per source: a replay with fewer than
    # rows * share / (1 - share) rows would silently cut the instruction rows. One row may go to the floor.
    short = per["smol_smoltalk"] < len(rows) - 1
    if a.replay_glob and (short or abs(stats["dataset"]["replay_share_of_rows"] - share) > 0.01):
        stats["error"] = (f"the trainer would draw {per} rows: the replay ({a.replay_glob}) is too small for share "
                          f"{share} of {len(rows)} instruction rows; add replay shards or lower --replay_share")
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if "error" in stats:
        sys.exit(f"error: {stats['error']}")
    tr, te, d = stats["train"], stats["test"], stats["dataset"]
    print(f"train: {tr['rows']} rows x {S} ({tr['conversations']} conversations, fill {tr['fill']:.4f}, answer share "
          f"{tr['answer_share_of_content']:.3f}); test: {te['rows']} rows ({te['conversations']} conversations); "
          f"trainer rows {d['train_rows_per_source']} (replay {d['replay_share_of_rows']:.3f}); {stats['seconds']} s")


if __name__ == "__main__":
    main()

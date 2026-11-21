"""Build LongDocs-lite: LSA's long-document corpus recipe (arXiv 2610.42147, repo data/longdocs), scaled down.

Sources are the Common Pile "filtered" datasets LSA collects from (gutenberg, arxiv, doab, usgpo, hansard; LSA's
`code` source, The Stack, is gated and left out; wikimedia and the agent traces are optional). A document is one row
of the hub dataset; it is kept when it has at least `--min_tokens` GPT-2 tokens (LSA: 64k for 128k training; here
16k for 16k training). Tokens are GPT-2 ids (tiktoken "gpt2", identical to the 50,257-token GPT-2 BPE), each document
prefixed with <|endoftext|> (50256) — the same layout as the FineWeb-Edu GPT-2 shards (`kjj0/finewebedu10B-gpt2`,
modded-nanogpt format: 256 int32 header [20240520, 1, n_tokens, ...] then uint16 tokens).

Per source: up to `--tokens_per_source` training tokens, and every 100th kept document (1 %) goes to the held-out
split. Writes <out>/<source>_{train,val}_NNNNNN.bin and <out>/manifest.json (documents and tokens per source/split).

Usage: uv run python scripts/build_longdocs_lite.py --out /data/longdocs_lite_gpt2 --tokens_per_source 300e6
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np

SOURCES = {  # LSA source card name → Common Pile dataset (filtered), as in LSA's src/lsa/assets/datasets/sources
    "gutenberg": "common-pile/project_gutenberg_filtered",
    "arxiv": "common-pile/arxiv_papers_filtered",
    "doab": "common-pile/doab_filtered",
    "usgpo": "common-pile/usgpo_filtered",
    "hansard": "common-pile/uk_hansard_filtered",
    "wikimedia": "common-pile/wikimedia_filtered",
}
EOT = 50256
MAGIC, VERSION = 20240520, 1


class ShardWriter:
    def __init__(self, out: Path, prefix: str, shard_tokens: int):
        self.out, self.prefix, self.cap = out, prefix, shard_tokens
        self.buf: list[np.ndarray] = []
        self.n = 0
        self.idx = 0
        self.total = 0
        self.files: list[str] = []

    def add(self, toks: np.ndarray) -> None:
        self.buf.append(toks)
        self.n += len(toks)
        self.total += len(toks)
        if self.n >= self.cap:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        arr = np.concatenate(self.buf).astype(np.uint16)
        header = np.zeros(256, dtype=np.int32)
        header[0], header[1], header[2] = MAGIC, VERSION, len(arr)
        path = self.out / f"{self.prefix}_{self.idx:06d}.bin"
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(arr.tobytes())
        self.files.append(path.name)
        self.idx += 1
        self.buf, self.n = [], 0


def iter_docs(repo: str):
    """Yield (doc_id, text) over the dataset's .json.gz / .jsonl.gz shards, in file order."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.endswith((".json.gz", ".jsonl.gz")))
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset")
        with gzip.open(path, "rb") as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                row = json.loads(line)
                yield str(row.get("id", "")), row.get("text") or ""


def build_source(name: str, repo: str, out: Path, args, enc) -> dict:
    train = ShardWriter(out, f"{name}_train", int(args.shard_tokens))
    val = ShardWriter(out, f"{name}_val", int(args.shard_tokens))
    kept = seen = 0
    t0 = time.time()
    batch: list[str] = []

    def drain(batch: list[str]) -> bool:
        nonlocal kept
        for ids in enc.encode_ordinary_batch(batch, num_threads=args.threads):
            if len(ids) < args.min_tokens:
                continue
            toks = np.asarray([EOT] + ids, dtype=np.uint16)
            (val if kept % 100 == 99 else train).add(toks)
            kept += 1
            if train.total >= args.tokens_per_source:
                return True
        return False

    done = False
    for _, text in iter_docs(repo):
        seen += 1
        # a document shorter than ~2.5 characters per kept token can never reach min_tokens (LSA's pre-filter)
        if len(text) < 2.5 * args.min_tokens:
            continue
        batch.append(text)
        if len(batch) >= args.batch_docs:
            done = drain(batch)
            batch = []
            print(f"[{name}] seen {seen:,} kept {kept:,} train {train.total/1e6:.0f}M val {val.total/1e6:.1f}M "
                  f"({time.time()-t0:.0f} s)", flush=True)
            if done:
                break
    if batch and not done:
        drain(batch)
    train.flush()
    val.flush()
    return {"repo": repo, "docs_seen": seen, "docs_kept": kept, "train_tokens": int(train.total),
            "val_tokens": int(val.total), "train_files": train.files, "val_files": val.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="gutenberg,arxiv,doab,usgpo,hansard")
    ap.add_argument("--tokens_per_source", type=float, default=300e6)
    ap.add_argument("--min_tokens", type=int, default=16384)
    ap.add_argument("--shard_tokens", type=float, default=100e6)
    ap.add_argument("--batch_docs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"recipe": "LongDocs-lite (LSA arXiv 2610.42147 LongDocs, Common Pile filtered sources)",
                "tokenizer": "gpt2 (tiktoken)", "eot": EOT, "min_tokens": args.min_tokens,
                "format": "modded-nanogpt .bin: 256 int32 header [20240520, 1, n], uint16 tokens, EOT starts a doc",
                "sources": {}}
    for name in args.sources.split(","):
        manifest["sources"][name] = build_source(name, SOURCES[name], out, args, enc)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("docs_kept", "train_tokens", "val_tokens")}
                      for k, v in manifest["sources"].items()}, indent=2))


if __name__ == "__main__":
    main()

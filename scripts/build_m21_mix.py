"""Build the m21mix_v1 pretraining mix (E38): open Hugging Face text datasets -> token shards, no human step.

Reads a recipe (data/mix_recipes/m21_mix_v1.json), and for every source: lists the Hub files anonymously, reads them
in a fixed order one at a time, tokenizes with the SmolLM2 tokenizer (HuggingFaceTB/SmolLM2-135M, 49,152 ids,
<|endoftext|> = 0), and writes token shards in the layout of data/bin_token_dataset.py (256 int32 header
[20240520, 1, n_tokens], then uint16 ids; <|endoftext|> STARTS every document and appears nowhere else):

  DIR/<name>_train_NNNNNN.bin   at most 100,000,000 tokens each; a document is never split across shards
  DIR/<name>_val_000000.bin     the held-out documents (never in train)
  DIR/<name>.done.json          written LAST: tokens, documents, shards (with token counts), raw files read, UTF-8
                                bytes of the text (for bits per byte), length histogram, rows available per row rule
  DIR/tokenizer/                the tokenizer, pad_token = eos_token
  DIR/mix.json                  every source's done.json + the recipe's phases, each with a "supply" report

Resumable: a source with a done.json is skipped; a source without one is rebuilt from scratch (same order, same
result). A source that fails does not stop the others; the exit code is then non-zero.

Source kinds (recipe "kind"):
  docs       one row = one document (column "text"); optional row filter on a column, min_tokens / max_tokens per
             document (outside -> dropped; min_tokens is how the 16k and 64k tiers are made), and a cheap pre-filter:
             a text shorter than min_tokens x 2 characters is not tokenized.
  repo_pack  rows are source files with a repository name and a path; the files of one repository (within one
             reading round) are sorted by path and rendered
             "# Repository: <name>\\n\\n## File: <path>\\n<code>\\n\\n..."; a pack holds at most max_tokens (then a
             new pack of the same repository starts) and is kept from min_tokens up.
  chat       rows with a "messages" list; one conversation = one document, rendered with the tokenizer's chat template
             (SmolLM2-135M has none: ChatML, "<|im_start|>role\\ncontent<|im_end|>\\n", is used and saved with the
             tokenizer).
  external   built by another script; never built here, but listed in mix.json when its done.json is in DIR.

Held-out split: a document is held out when a stable hash of its id (or its text, or its repository) falls in the
lowest `val_fraction` (0.5 %). Held-out documents go to the val shard until it holds `val_tokens` (2M) and, for
sources with min_tokens, 40 documents; later ones are dropped, so a held-out document is never trained on.
"heldout": "longdocs_lite" replays scripts/build_longdocs_lite.py's own rule instead (every 100th document that has
>= 40,960 characters and >= 16,384 GPT-2 tokens, in file order), so the documents that were validation there stay
validation here.

Usage:
  uv run python scripts/build_m21_mix.py --recipe data/mix_recipes/m21_mix_v1.json --out DIR --list
  uv run python scripts/build_m21_mix.py --recipe data/mix_recipes/m21_mix_v1.json --out DIR --scale 0.001   # smoke
  uv run python scripts/build_m21_mix.py --recipe data/mix_recipes/m21_mix_v1.json --out DIR --only books --no-mix
  uv run python scripts/build_m21_mix.py --recipe data/mix_recipes/m21_mix_v1.json --out DIR --mix-only
On a node: infra/aws/jobs/m21_data_build.sh (sources in parallel processes, each synced to S3 when it is done).
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256
SHARD_TOKENS = 100_000_000
HIST = (4096, 16384, 65536, 131072)  # length histogram: documents and tokens at >= these many tokens
ROW_LENS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)  # rows available are recorded for these seq_len
DEFAULTS = {"val_fraction": 0.005, "val_tokens": 2_000_000, "val_min_docs": 40, "min_chars_per_token": 2.0,
            "round_chars": 12e9}  # a recipe's "defaults" override these; a source's own field overrides both
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
          "'<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
RETRIES = 6
BATCH_DOCS, BATCH_CHARS, BATCH_ROWS = 512, 16_000_000, 50_000
_sleep = time.sleep  # tests replace it


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------
# Hub access: retries, listing, one raw file at a time
# ----------------------------------------------------------------------------------------------------------------
def is_transient(e: BaseException) -> bool:
    """A network or server error worth another try (not: missing / gated / private, nor a bug in a recipe)."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 425, 429) or status >= 500
    if type(e).__name__ == "LocalEntryNotFoundError":  # huggingface_hub: the Hub did not answer (no cached copy)
        return True
    if isinstance(e, (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError)):
        return False
    if isinstance(e, (ConnectionError, TimeoutError, EOFError, OSError)):
        return True
    return type(e).__module__.split(".")[0] in ("requests", "urllib3", "httpx", "httpcore", "aiohttp", "zlib",
                                                "zstandard")


def retry(fn, what: str, tries: int = RETRIES):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_transient(e) or k == tries - 1:
                raise
            wait = min(120.0, 2.0 * 2 ** k)
            log(f"retry {k + 1}/{tries - 1} in {wait:.0f} s: {what}: {type(e).__name__}: {str(e)[:200]}")
            _sleep(wait)


class Raw:
    """Raw files of a source: a public Hub dataset repo read anonymously, or a local directory ("local").

    mode "download": a Hub file is downloaded whole to raw_dir, read, then deleted (bounded disk).
    mode "stream":   a Hub file is read over HTTP as far as needed (small builds: nothing is downloaded whole).
    """

    def __init__(self, raw_dir: Path, mode: str = "download"):
        self.raw_dir, self.mode = Path(raw_dir), mode
        self._lists: dict[str, dict[str, int]] = {}
        self._fs = None

    def listing(self, spec: dict) -> dict[str, int]:
        """{path: bytes} of every file of the source's repo (or local directory)."""
        if spec.get("local"):
            root = Path(spec["local"])
            if not root.is_dir():
                raise FileNotFoundError(f"local source directory {root} does not exist")
            return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}
        repo = spec["repo"]
        if repo not in self._lists:
            from huggingface_hub import HfApi

            def go():
                info = HfApi(token=False).dataset_info(repo, files_metadata=True, token=False)
                if getattr(info, "gated", False) or getattr(info, "private", False):
                    raise PermissionError(f"{repo} is gated or private: not readable without a token")
                return {s.rfilename: int(s.size or 0) for s in info.siblings}

            self._lists[repo] = retry(go, f"list {repo}")
        return self._lists[repo]

    @contextlib.contextmanager
    def open(self, spec: dict, path: str):
        """A seekable binary file object for one raw file."""
        if spec.get("local"):
            with open(Path(spec["local"]) / path, "rb") as f:
                yield f
        elif self.mode == "stream":
            if self._fs is None:
                from huggingface_hub import HfFileSystem

                self._fs = HfFileSystem(token=False)
            # parquet: the reader asks for the byte ranges it needs (pyarrow buffers them); else: read ahead
            kw = ({"block_size": 1 << 20, "cache_type": "none"} if path.endswith(".parquet")
                  else {"block_size": 8 << 20, "cache_type": "readahead"})
            f = retry(lambda: self._fs.open(f"datasets/{spec['repo']}/{path}", "rb", **kw), f"open {path}")
            try:
                yield f
            finally:
                f.close()
        else:
            from huggingface_hub import hf_hub_download

            local_dir = self.raw_dir / spec["name"]
            local = retry(lambda: hf_hub_download(spec["repo"], path, repo_type="dataset", token=False,
                                                  local_dir=str(local_dir)), f"download {spec['repo']}/{path}")
            try:
                with open(local, "rb") as f:
                    yield f
            finally:
                with contextlib.suppress(OSError):
                    os.remove(local)


def parse_rows(f, path: str, columns: list[str]):
    """Rows of one raw file as dicts: parquet (only `columns`, in small batches) or JSON lines (.gz / .zst / plain)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        # buffer_size: a column chunk is read in pieces as rows are asked for (some files are ONE 3 GB row group)
        pf = pq.ParquetFile(f, buffer_size=4 << 20, pre_buffer=False)
        names = pf.schema_arrow.names
        missing = [c for c in columns if c not in names]
        if missing:
            raise KeyError(f"{path}: no column {missing}; the file has {names}")
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()
        return
    if path.endswith(".gz"):
        stream = gzip.open(f, "rb")
    elif path.endswith(".zst"):
        import zstandard

        stream = zstandard.ZstdDecompressor().stream_reader(f)
    elif path.endswith((".jsonl", ".json")):
        stream = f
    else:
        raise ValueError(f"{path}: unknown raw file type (parquet, .jsonl, .json.gz, .jsonl.gz, .jsonl.zst)")
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def read_rows(raw: Raw, spec: dict, path: str, columns: list[str]):
    """parse_rows with retries: after a transient error the file is opened again and the rows already given skipped."""
    given, fails, given_at_fail = 0, 0, -1
    while True:
        try:
            with raw.open(spec, path) as f:
                for i, row in enumerate(parse_rows(f, path, columns)):
                    if i < given:
                        continue
                    given += 1
                    yield row
            return
        except Exception as e:  # noqa: BLE001
            if given > given_at_fail:
                fails, given_at_fail = 0, given
            if not is_transient(e) or fails >= RETRIES - 1:
                raise
            fails += 1
            wait = min(120.0, 2.0 * 2 ** (fails - 1))
            log(f"[{spec['name']}] retry {fails}/{RETRIES - 1} in {wait:.0f} s after {given:,} rows of {path}: "
                f"{type(e).__name__}: {str(e)[:200]}")
            _sleep(wait)


def get_path(row, dotted: str):
    cur = row
    for k in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def as_filters(f) -> list[dict]:
    return [] if not f else ([f] if isinstance(f, dict) else list(f))


def passes(row: dict, filters: list[dict]) -> bool:
    """Row filter: {"column": c, "min": x} (value >= x), "max" (value <= x), "in" (value in list). None fails."""
    for flt in filters:
        v = get_path(row, flt["column"])
        if v is None:
            return False
        if "min" in flt and not v >= flt["min"]:
            return False
        if "max" in flt and not v <= flt["max"]:
            return False
        if "in" in flt and v not in flt["in"]:
            return False
    return True


def resolve_parts(spec: dict, listing: dict[str, int]) -> list[dict]:
    """The source's parts: [{"files": [(path, bytes)], "cum_share": float, "filter": [...], "label": str}].

    A source has either "files" (one part) or "parts": [{"share": s, "files": ...}]. "files" is a glob or a list of
    globs over the repo's file paths; each glob's matches are sorted and the globs are read round-robin (one file of
    each in turn), so several directories mix evenly. Part i is read until the train split holds
    target x (share_1 + ... + share_i) tokens.
    """
    parts = spec.get("parts") or [{"share": 1.0, "files": spec.get("files")}]
    total = float(sum(float(p.get("share", 1.0)) for p in parts))
    out, cum = [], 0.0
    for k, p in enumerate(parts):
        globs = p.get("files")
        if not globs:
            raise ValueError(f"source {spec['name']}: part {k} names no files")
        globs = [globs] if isinstance(globs, str) else list(globs)
        columns = []
        for g in globs:
            hits = sorted(f for f in listing if fnmatch.fnmatchcase(f, g))
            if not hits:
                raise FileNotFoundError(f"source {spec['name']}: no file matches {g!r}")
            columns.append(hits)
        files, seen = [], set()
        for i in range(max(len(c) for c in columns)):
            for c in columns:
                if i < len(c) and c[i] not in seen:
                    seen.add(c[i])
                    files.append((c[i], listing[c[i]]))
        cum += float(p.get("share", 1.0)) / total
        out.append({"files": files, "cum_share": min(1.0, cum), "filter": as_filters(p.get("filter")),
                    "label": p.get("label", f"part {k + 1}/{len(parts)}")})
    out[-1]["cum_share"] = 1.0
    return out


def effective_spec(spec: dict, raw: Raw) -> tuple[dict, dict[str, int], bool]:
    """(spec, listing, used_fallback): the source itself, or its "fallback" when the repo cannot be listed."""
    try:
        return spec, raw.listing(spec), False
    except Exception as e:  # noqa: BLE001
        if not spec.get("fallback"):
            raise
        log(f"[{spec['name']}] !! {spec.get('repo')} is not readable ({type(e).__name__}: {str(e)[:200]}); "
            f"using the fallback {spec['fallback'].get('repo')}")
        fb = {k: v for k, v in spec.items() if k not in ("fallback", "parts", "files", "filter")}
        fb.update(spec["fallback"])
        return fb, raw.listing(fb), True


# ----------------------------------------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------------------------------------
class Tok:
    """The fast tokenizer. Documents are encoded with special-token strings split as ordinary text, so
    <|endoftext|> written inside a web page never becomes id 0; chats are encoded with them as single ids."""

    def __init__(self, name_or_dir: str):
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        self.name = name_or_dir
        self.hf = retry(lambda: AutoTokenizer.from_pretrained(name_or_dir, token=False), f"tokenizer {name_or_dir}")
        if self.hf.eos_token_id is None:
            raise ValueError(f"{name_or_dir}: the tokenizer has no eos token")
        self.eot, self.vocab_size = int(self.hf.eos_token_id), len(self.hf)
        if self.vocab_size > 65536:
            raise ValueError(f"{name_or_dir}: {self.vocab_size} ids do not fit uint16 shards")
        if not self.hf.chat_template:
            self.hf.chat_template = CHATML
        self._special = self.hf.backend_tokenizer
        self._plain = Tokenizer.from_str(self._special.to_str())
        self._plain.encode_special_tokens = True

    def encode_batch(self, texts: list[str], special: bool = False) -> list[np.ndarray]:
        t = self._special if special else self._plain
        fn = getattr(t, "encode_batch_fast", t.encode_batch)
        return [np.asarray(e.ids, dtype=np.uint16) for e in fn(texts, add_special_tokens=False)]

    def render_chat(self, messages: list[dict]) -> str:
        return self.hf.apply_chat_template(messages, tokenize=False)

    def decode(self, ids) -> str:
        return self._special.decode([int(i) for i in ids], skip_special_tokens=False)

    def save(self, path: Path) -> None:
        """DIR/tokenizer with pad_token = eos_token; safe when several builder processes start together."""
        path = Path(path)
        if (path / "tokenizer.json").exists():
            return
        self.hf.pad_token = self.hf.eos_token
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        self.hf.save_pretrained(str(tmp))
        try:
            os.rename(tmp, path)
        except OSError:  # another process saved it first
            shutil.rmtree(tmp, ignore_errors=True)


def utf8(text: str) -> tuple[str, int]:
    """(text the tokenizer accepts, its UTF-8 bytes): lone surrogates from broken JSON escapes become U+FFFD."""
    try:
        return text, len(text.encode("utf-8"))
    except UnicodeEncodeError:
        text = text.encode("utf-8", "replace").decode("utf-8")
        return text, len(text.encode("utf-8"))


# ----------------------------------------------------------------------------------------------------------------
# Held-out rules
# ----------------------------------------------------------------------------------------------------------------
def stable_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


class HashHeldout:
    """Held out when the stable hash of the document's key is in the lowest `fraction` — a property of the document
    alone, whatever the order, the target or the tokenizer."""

    name = "hash"

    def __init__(self, fraction: float):
        self.per_million = int(round(float(fraction) * 1_000_000))

    def is_val(self, key: str) -> bool:
        return stable_hash(key) % 1_000_000 < self.per_million

    def mark(self, keys: list[str], texts: list[str]) -> list[bool]:
        return [self.is_val(k) for k in keys]


class LongdocsLiteHeldout:
    """scripts/build_longdocs_lite.py's rule, replayed: among the rows in file order, a row with at least
    2.5 x min_tokens characters and at least min_tokens GPT-2 tokens is "kept" there, and kept number 99, 199, ...
    (kept % 100 == 99) is validation. `mark` must see every row of the source, in order."""

    name = "longdocs_lite"

    def __init__(self, count_tokens, min_tokens: int = 16384, every: int = 100):
        self.count, self.min_tokens, self.every, self.kept = count_tokens, int(min_tokens), int(every), 0

    def mark(self, keys: list[str], texts: list[str]) -> list[bool]:
        cand = [i for i, t in enumerate(texts) if not len(t) < 2.5 * self.min_tokens]
        out = [False] * len(texts)
        for i, n in zip(cand, self.count([texts[i] for i in cand]) if cand else []):
            if n < self.min_tokens:
                continue
            out[i] = self.kept % self.every == self.every - 1
            self.kept += 1
        return out


def gpt2_counter(threads: int):
    """GPT-2 token counts exactly as build_longdocs_lite.py takes them (tiktoken "gpt2", encode_ordinary_batch)."""
    import tiktoken

    enc = retry(lambda: tiktoken.get_encoding("gpt2"), "tiktoken gpt2")
    return lambda texts: [len(ids) for ids in enc.encode_ordinary_batch(texts, num_threads=max(1, threads))]


# ----------------------------------------------------------------------------------------------------------------
# Shards
# ----------------------------------------------------------------------------------------------------------------
def write_shard(path: Path, arr: np.ndarray) -> None:
    """The layout of data.bin_token_dataset.write_shard, written to a temporary name first."""
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(arr)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(arr, dtype=np.uint16).tobytes())
    os.replace(tmp, path)


def shard_tokens(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: not a token shard")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(int(header[2]),))


class Split:
    """One split of one source: its shards (a document is never split across two) and its counters.
    Token counts include the <|endoftext|> that starts each document; the length histogram does not."""

    def __init__(self, out: Path, prefix: str, cap: int | None, row_lens: tuple[int, ...]):
        self.out, self.prefix, self.cap = Path(out), prefix, cap
        self.buf: list[np.ndarray] = []
        self.buf_tokens = 0
        self.shards: list[dict] = []
        self.tokens = self.docs = self.bytes = 0
        self.hist = {t: [0, 0] for t in HIST}
        self.doc_rows = {s: 0 for s in row_lens}
        self.last_bytes = self.last_tokens = 0

    def _count(self, n: int, nbytes: int, sign: int) -> None:
        self.tokens += sign * n
        self.docs += sign
        self.bytes += sign * nbytes
        for t in HIST:
            if n - 1 >= t:
                self.hist[t][0] += sign
                self.hist[t][1] += sign * (n - 1)
        for s in self.doc_rows:
            self.doc_rows[s] += sign * (n // s)

    def add(self, doc: np.ndarray, nbytes: int) -> None:
        if self.cap is not None and self.buf and self.buf_tokens + len(doc) > self.cap:
            self.flush()
        self.buf.append(doc)
        self.buf_tokens += len(doc)
        self.last_bytes, self.last_tokens = nbytes, len(doc)
        self._count(len(doc), nbytes, +1)

    def pop(self) -> tuple[np.ndarray, int]:
        """Take back the document added last (it is still in memory: a flush only happens before an add)."""
        doc = self.buf.pop()
        self.buf_tokens -= len(doc)
        self._count(len(doc), self.last_bytes, -1)
        return doc, self.last_bytes

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"{self.prefix}_{len(self.shards):06d}.bin"
        write_shard(self.out / name, np.concatenate(self.buf))
        self.shards.append({"file": name, "tokens": int(self.buf_tokens), "documents": len(self.buf)})
        self.buf, self.buf_tokens = [], 0

    def report(self) -> dict:
        text_tokens = self.tokens - self.docs
        return {
            "tokens": int(self.tokens), "documents": int(self.docs), "text_bytes": int(self.bytes),
            "bytes_per_token": round(self.bytes / text_tokens, 4) if text_tokens else None,
            "shards": self.shards,
            "length_hist": {str(t): {"documents": int(d), "tokens": int(n)} for t, (d, n) in self.hist.items()},
            "rows": {"stream": {str(s): int(sum(sh["tokens"] // s for sh in self.shards)) for s in self.doc_rows},
                     "doc": {str(s): int(n) for s, n in self.doc_rows.items()}},
        }


class Sink:
    """Sends each finished document to train or val, and says when the train target is reached."""

    def __init__(self, name: str, out: Path, target: int, shard_cap: int, val_cap: int, val_min_docs: int,
                 val_min_rows: dict[int, int], row_lens: tuple[int, ...]):
        self.train = Split(out, f"{name}_train", shard_cap, row_lens)
        self.val = Split(out, f"{name}_val", None, row_lens)
        self.target, self.quota, self.share = int(target), int(target), 1.0
        self.val_cap, self.val_min_docs, self.val_min_rows = val_cap, val_min_docs, val_min_rows
        self.val_dropped = [0, 0]
        self.val_fallback = False

    def set_share(self, cum_share: float) -> None:
        """Reading part i: stop at target x (share_1 + ... + share_i); the val budget grows the same way."""
        self.share = cum_share
        self.quota = self.target if cum_share >= 1.0 else int(math.ceil(self.target * cum_share))

    @property
    def done(self) -> bool:
        if self.val.docs > 0:
            return self.train.tokens >= self.quota
        # no held-out document yet (a tiny build): one more document is read, to stand in as validation (see close)
        return self.train.docs >= 2 and self.train.tokens - self.train.last_tokens >= self.quota

    def _val_wants(self, n: int) -> bool:
        if self.val.tokens < self.val_cap * self.share or self.val.docs < math.ceil(self.val_min_docs * self.share):
            return True
        return any(n >= s and self.val.doc_rows[s] < need for s, need in self.val_min_rows.items())

    def add(self, doc: np.ndarray, nbytes: int, is_val: bool) -> None:
        if not is_val:
            self.train.add(doc, nbytes)
        elif self._val_wants(len(doc)):
            self.val.add(doc, nbytes)
        else:  # held out, but the val shard is full: dropped, never trained on
            self.val_dropped[0] += 1
            self.val_dropped[1] += len(doc)

    def close(self) -> None:
        if self.val.docs == 0 and self.train.docs >= 2:
            # too few documents for the held-out rule to pick one (a --scale build): the last train document
            self.val.add(*self.train.pop())
            self.val_fallback = True
        self.train.flush()
        self.val.flush()


class Progress:
    def __init__(self, name: str, sink: Sink, counters: Counter, every: float = 30.0):
        self.name, self.sink, self.c, self.every = name, sink, counters, every
        self.t0 = self.last = time.time()

    def tick(self, where: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        dt = max(now - self.t0, 1e-6)
        tr, va = self.sink.train, self.sink.val
        log(f"[{self.name}] {where} | rows {self.c['rows']:,} | train {tr.tokens / 1e6:,.1f}M of "
            f"{self.sink.target / 1e6:,.1f}M tokens ({100 * tr.tokens / max(1, self.sink.target):.0f} %), "
            f"{tr.docs:,} docs | val {va.tokens / 1e6:.2f}M, {va.docs:,} docs | "
            f"{self.c['tokenized'] / dt / 1e6:.2f}M tokens/s tokenized, {dt:.0f} s")


# ----------------------------------------------------------------------------------------------------------------
# Source kinds
# ----------------------------------------------------------------------------------------------------------------
def top_columns(*dotted) -> list[str]:
    out = []
    for d in dotted:
        if d and d.split(".")[0] not in out:
            out.append(d.split(".")[0])
    return out


def make_doc(ids: np.ndarray, eot: int) -> np.ndarray:
    doc = np.empty(len(ids) + 1, dtype=np.uint16)
    doc[0] = eot
    doc[1:] = ids
    return doc


def run_docs(spec: dict, parts: list[dict], sink: Sink, tok, raw: Raw, heldout, c: Counter, raw_files: list[dict],
             prog: Progress) -> None:
    """kinds "docs" and "chat": one row = one document."""
    chat = spec["kind"] == "chat"
    text_col = spec.get("messages_column", "messages") if chat else spec.get("text_column", "text")
    id_col = spec.get("id_column")
    min_tok, max_tok = int(spec.get("min_tokens", 0)), int(spec.get("max_tokens", 0))
    min_chars = min_tok * float(spec["min_chars_per_token"])
    keys: list[str] = []
    texts: list[str] = []
    cand: list[bool] = []
    load = [0, 0]  # candidates, candidate characters

    def flush() -> None:
        if not texts:
            return
        marks = heldout.mark(keys, texts)
        idx = [i for i, ok in enumerate(cand) if ok]
        c["short_chars"] += len(texts) - len(idx)
        safe = [utf8(texts[i]) for i in idx]
        ids_list = tok.encode_batch([s for s, _ in safe], special=chat) if idx else []
        c["tokenized"] += int(sum(len(x) for x in ids_list))
        for i, ids, (_, nbytes) in zip(idx, ids_list, safe):
            if sink.done:
                break
            if (ids == tok.eot).any():  # only a chat can hold the literal token; it must not start a "document"
                ids = ids[ids != tok.eot]
                c["eot_inside_removed"] += 1
            if len(ids) < max(min_tok, 1):
                c["below_min_tokens"] += 1
            elif max_tok and len(ids) > max_tok:
                c["above_max_tokens"] += 1
            else:
                sink.add(make_doc(ids, tok.eot), nbytes, marks[i])
        reset()

    def reset() -> None:
        keys.clear()
        texts.clear()
        cand.clear()
        load[0] = load[1] = 0

    for part in parts:
        sink.set_share(part["cum_share"])
        filters = as_filters(spec.get("filter")) + part["filter"]
        columns = top_columns(text_col, id_col, *[f["column"] for f in filters])
        for k, (path, size) in enumerate(part["files"]):
            if sink.done:
                break
            rec = {"path": path, "bytes": size, "rows": 0, "complete": False}
            raw_files.append(rec)
            where = f"{part['label']}, file {k + 1}/{len(part['files'])} {path}"
            for row in read_rows(raw, spec, path, columns):
                rec["rows"] += 1
                c["rows"] += 1
                if not passes(row, filters):
                    c["filtered"] += 1
                    continue
                text = get_path(row, text_col)
                if chat and text:
                    text = tok.render_chat(text)
                if not isinstance(text, str) or not text:
                    c["empty"] += 1
                    continue
                key = get_path(row, id_col) if id_col else None
                keys.append(str(key) if key not in (None, "") else text)
                texts.append(text)
                ok = len(text) >= min_chars
                cand.append(ok)
                if ok:
                    load[0] += 1
                    load[1] += len(text)
                if load[0] >= BATCH_DOCS or load[1] >= BATCH_CHARS or len(texts) >= BATCH_ROWS:
                    flush()
                    prog.tick(where)
                    if sink.done:
                        break
            else:
                rec["complete"] = True
                flush()  # before the next raw file is opened (or downloaded): it may not be needed
        reset()
        prog.tick(f"{part['label']} done", force=True)


def run_repo_pack(spec: dict, parts: list[dict], sink: Sink, tok, raw: Raw, heldout, c: Counter,
                  raw_files: list[dict], prog: Progress, tmp_dir: Path, scale: float) -> None:
    """kind "repo_pack": the files of one repository, sorted by path, packed into documents.

    The rows of a part are read in rounds of `round_chars` characters (rows of a repository are scattered over the
    raw files, so a round has to be large for a repository to gather its files). A round is spilled to disk in
    buckets by a hash of the repository name; then, bucket by bucket and repository by repository in name order,
    the files are rendered, tokenized and packed. Reading stops as soon as the part's token quota is reached.
    """
    text_col = spec.get("text_column", "text")
    repo_col, path_col = spec.get("repo_column", "metadata.repo_name"), spec.get("path_column", "metadata.path")
    min_tok, max_tok = int(spec.get("min_tokens", 0)), int(spec.get("max_tokens", 0))
    min_chars = min_tok * float(spec["min_chars_per_token"])
    round_chars = float(spec["round_chars"])
    if scale < 1.0:
        round_chars = max(round_chars * scale, min(round_chars, 20e6))
    n_buckets = max(1, min(256, int(math.ceil(round_chars / 200e6))))

    def rows_of(part: dict, filters: list[dict]):
        columns = top_columns(text_col, repo_col, path_col, *[f["column"] for f in filters])
        for path, size in part["files"]:
            rec = {"path": path, "bytes": size, "rows": 0, "complete": False}
            raw_files.append(rec)
            for row in read_rows(raw, spec, path, columns):
                rec["rows"] += 1
                c["rows"] += 1
                if not passes(row, filters):
                    c["filtered"] += 1
                    continue
                repo, text = get_path(row, repo_col), get_path(row, text_col)
                if not repo or not isinstance(text, str) or not text:
                    c["empty"] += 1
                    continue
                yield str(repo), str(get_path(row, path_col) or ""), text
            rec["complete"] = True

    def spill(rows) -> int:
        """Write up to round_chars characters of rows into the bucket files; rows written."""
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        files = [open(tmp_dir / f"bucket_{b:03d}.jsonl", "w", encoding="ascii") for b in range(n_buckets)]
        n = chars = 0
        try:
            for repo, path, text in rows:
                files[stable_hash(repo) % n_buckets].write(json.dumps([repo, path, text]) + "\n")
                n += 1
                chars += len(text)
                if chars >= round_chars:
                    break
        finally:
            for f in files:
                f.close()
        return n

    def pack(pending: list[tuple[str, list[str]]]) -> None:
        """pending: (repository, [header, file segment, ...]) -> packs -> sink."""
        safe = [utf8(s) for _, segs in pending for s in segs]
        ids_all = tok.encode_batch([s for s, _ in safe]) if safe else []
        c["tokenized"] += int(sum(len(x) for x in ids_all))
        at = 0
        for repo, segs in pending:
            ids = [x[x != tok.eot] if (x == tok.eot).any() else x for x in ids_all[at: at + len(segs)]]
            nb = [b for _, b in safe[at: at + len(segs)]]
            at += len(segs)
            if sink.done:
                continue
            is_val = heldout.mark([repo], [""])[0]
            cur, cur_tok, cur_bytes = [ids[0]], len(ids[0]), nb[0]

            def close() -> None:
                if len(cur) == 1:
                    return
                if cur_tok < max(min_tok, 1):
                    c["packs_below_min_tokens"] += 1
                    return
                c["packs"] += 1
                c["packs_single_file"] += len(cur) == 2
                c["files_in_packs"] += len(cur) - 1
                sink.add(make_doc(np.concatenate(cur), tok.eot), cur_bytes, is_val)

            for seg, b in zip(ids[1:], nb[1:]):
                if max_tok and len(ids[0]) + len(seg) > max_tok:
                    c["files_above_max_tokens"] += 1  # one file larger than a whole pack: left out
                    continue
                if max_tok and len(cur) > 1 and cur_tok + len(seg) > max_tok:
                    close()
                    cur, cur_tok, cur_bytes = [ids[0]], len(ids[0]), nb[0]
                cur.append(seg)
                cur_tok += len(seg)
                cur_bytes += b
            close()

    def pack_bucket(path: Path, where: str) -> None:
        repos: dict[str, set] = defaultdict(set)
        with open(path, encoding="ascii") as f:
            for line in f:
                repo, p, text = json.loads(line)
                repos[repo].add((p, text))  # a set: the same file twice counts once
        pending, chars = [], 0
        for repo in sorted(repos):
            files = sorted(repos[repo])
            c["repos"] += 1
            total = sum(len(t) for _, t in files)
            if total < min_chars:
                c["repos_short_chars"] += 1
                continue
            pending.append((repo, [f"# Repository: {repo}\n\n"]
                            + [f"## File: {p.lstrip('/')}\n{t}\n\n" for p, t in files]))
            chars += total
            if chars >= BATCH_CHARS:
                pack(pending)
                pending, chars = [], 0
                prog.tick(where)
                if sink.done:
                    return
        pack(pending)

    try:
        for part in parts:
            sink.set_share(part["cum_share"])
            rows = rows_of(part, as_filters(spec.get("filter")) + part["filter"])
            rnd = 0
            while not sink.done:
                rnd += 1
                n = spill(rows)
                if n == 0:
                    break
                c["rounds"] += 1
                prog.tick(f"{part['label']}, round {rnd}: {n:,} files in {n_buckets} buckets", force=True)
                for b in range(n_buckets):
                    pack_bucket(tmp_dir / f"bucket_{b:03d}.jsonl",
                                f"{part['label']}, round {rnd}, bucket {b + 1}/{n_buckets}")
                    if sink.done:
                        break
            rows.close()
            prog.tick(f"{part['label']} done", force=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------
# One source
# ----------------------------------------------------------------------------------------------------------------
def recipe_row_lens(recipe: dict) -> tuple[int, ...]:
    lens = set(ROW_LENS) | {int(p["seq_len"]) for p in (recipe.get("phases") or {}).values()}
    for s in recipe.get("sources", []):
        lens |= {int(k) for k in (s.get("val_min_rows") or {})}
    return tuple(sorted(lens))


def source_files(out: Path, name: str) -> list[Path]:
    pats = (f"{name}_train_??????.bin", f"{name}_val_??????.bin", f"{name}_train_??????.bin.tmp",
            f"{name}_val_??????.bin.tmp", f"{name}.done.json.tmp")
    return [p for pat in pats for p in out.glob(pat)]


def build_source(spec: dict, recipe: dict, out: Path, tok, raw: Raw, scale: float = 1.0, workers: int = 1,
                 gpt2_count=None) -> dict:
    """Build one source from scratch; returns (and writes, last) its done.json."""
    name, kind = spec["name"], spec["kind"]
    if kind not in ("docs", "repo_pack", "chat"):
        raise ValueError(f"source {name}: unknown kind {kind!r}")
    for p in source_files(out, name):  # what an interrupted build left
        p.unlink()
    t0 = time.time()
    defaults = {**DEFAULTS, **(recipe.get("defaults") or {})}
    used, listing, used_fallback = effective_spec(spec, raw)
    used = {"min_chars_per_token": defaults["min_chars_per_token"], "round_chars": defaults["round_chars"], **used}
    parts = resolve_parts(used, listing)
    long_docs = int(used.get("min_tokens", 0)) > 0
    target = max(1, int(float(used["target_tokens"]) * scale))
    val_min_docs = int(used.get("val_min_docs", defaults["val_min_docs"] if long_docs else 0))
    sink = Sink(
        name, out, target, int(recipe.get("shard_tokens", SHARD_TOKENS)),
        val_cap=int(float(used.get("val_tokens", defaults["val_tokens"])) * scale),
        val_min_docs=val_min_docs if scale >= 1.0 else int(math.ceil(val_min_docs * scale)),
        val_min_rows={int(k): int(v) for k, v in (used.get("val_min_rows") or {}).items()},
        row_lens=recipe_row_lens(recipe))
    rule = used.get("heldout", "hash")
    if rule == "longdocs_lite":
        if kind != "docs" or used.get("filter") or any(p["filter"] for p in parts):
            raise ValueError(f"source {name}: the longdocs_lite held-out rule needs kind docs and no row filter")
        heldout = LongdocsLiteHeldout(gpt2_count or gpt2_counter(workers), int(used.get("heldout_min_tokens", 16384)))
    elif rule == "hash":
        heldout = HashHeldout(float(used.get("val_fraction", defaults["val_fraction"])))
    else:
        raise ValueError(f"source {name}: unknown held-out rule {rule!r}")
    c: Counter = Counter()
    raw_files: list[dict] = []
    prog = Progress(name, sink, c)
    log(f"[{name}] start: {kind} from {used.get('repo') or used.get('local')}, target {target:,} train tokens, "
        f"{sum(len(p['files']) for p in parts)} raw files listed in {len(parts)} part(s), held-out rule {rule}, "
        f"reading by {'local files' if used.get('local') else raw.mode}")
    if kind == "repo_pack":
        run_repo_pack(used, parts, sink, tok, raw, heldout, c, raw_files, prog, raw.raw_dir / f"{name}.rounds", scale)
    else:
        run_docs(used, parts, sink, tok, raw, heldout, c, raw_files, prog)
    sink.close()
    if sink.train.docs == 0:
        raise RuntimeError(f"source {name}: no document kept ({dict(c)})")
    seconds = time.time() - t0
    done = {
        "name": name, "kind": kind, "repo": used.get("repo") or used.get("local"),
        "tokens": int(sink.train.tokens), "documents": int(sink.train.docs),
        "shards": [s["file"] for s in sink.train.shards], "val_shards": [s["file"] for s in sink.val.shards],
        "target_tokens": target, "short": bool(sink.train.tokens < target), "scale": scale,
        "train": sink.train.report(), "val": sink.val.report(),
        "heldout": {"rule": rule, "val_fallback_last_train_document": sink.val_fallback,
                    "dropped_documents": int(sink.val_dropped[0]), "dropped_tokens": int(sink.val_dropped[1])},
        "raw_files": raw_files, "raw_bytes": int(sum(r["bytes"] for r in raw_files)),
        "counters": {k: int(v) for k, v in sorted(c.items())},
        "seconds": round(seconds, 1), "tokens_per_s": round(c["tokenized"] / max(seconds, 1e-6)),
        "tokenizer": getattr(tok, "name", None), "eot": int(tok.eot), "finished_utc": utc(),
        "used_fallback": used_fallback, "recipe": spec,
    }
    if sink.val_fallback:
        log(f"[{name}] !! no document was held out by the rule (tiny build): the last train document is the val shard")
    if done["short"]:
        log(f"[{name}] !! SHORT: {sink.train.tokens:,} train tokens of {target:,} — the raw files listed ran out")
    write_json(out / f"{name}.done.json", done)
    tr = done["train"]
    log(f"[{name}] done: {tr['tokens']:,} train tokens in {tr['documents']:,} docs, {len(tr['shards'])} shard(s); "
        f"val {done['val']['tokens']:,} tokens in {done['val']['documents']:,} docs; "
        f"{tr['bytes_per_token']} bytes/token; {len(raw_files)} raw files ({done['raw_bytes'] / 1e9:.2f} GB); "
        f"{seconds:.0f} s, {done['tokens_per_s'] / 1e6:.2f}M tokens/s tokenized")
    return done


# ----------------------------------------------------------------------------------------------------------------
# mix.json and the supply report
# ----------------------------------------------------------------------------------------------------------------
def count_rows(files: list[Path], rule: str, seq_len: int, eot: int) -> int:
    """Rows BinRowDataset would find in these shards (its _stream_rows / _doc_rows)."""
    n = 0
    for p in files:
        sh = shard_tokens(p)
        if rule == "stream":
            n += len(sh) // seq_len
        else:
            starts = np.flatnonzero(np.asarray(sh) == eot)
            ends = np.append(starts[1:], len(sh))
            n += int(((ends - starts) // seq_len).sum())
    return n


def rows_available(done: dict, split: str, rule: str, seq_len: int, out: Path, eot: int) -> int | None:
    """From the done.json when it has the count; else counted in the shard files when they are here."""
    n = (((done.get(split) or {}).get("rows") or {}).get(rule) or {}).get(str(seq_len))
    if n is not None:
        return int(n)
    files = sorted(out.glob(f"{done.get('name', '')}_{split}_??????.bin"))
    return count_rows(files, rule, seq_len, eot) if files else None


def write_mix(recipe: dict, out: Path, scale: float = 1.0) -> tuple[dict, list[str]]:
    """DIR/mix.json from the done.json files in DIR; returns (mix, names of built sources that are missing)."""
    eot = int(recipe.get("eot", 0))
    sources, missing, external_missing = {}, [], []
    for spec in recipe["sources"]:
        path = out / f"{spec['name']}.done.json"
        if path.exists():
            sources[spec["name"]] = json.loads(path.read_text(encoding="utf-8"))
            sources[spec["name"]].setdefault("name", spec["name"])
        else:
            (external_missing if spec["kind"] == "external" else missing).append(spec["name"])
    phases, budget = {}, defaultdict(dict)
    for pname, phase in (recipe.get("phases") or {}).items():
        S, tokens = int(phase["seq_len"]), float(phase["tokens"]) * scale
        wsum = float(sum(float(e.get("weight", 1.0)) for e in phase["train"])) or 1.0
        rules = {e["name"]: e.get("rule", phase.get("rule", "stream")) for e in phase["train"]}
        supply, eval_supply = [], []
        for e in phase["train"]:
            need_tokens = float(e.get("weight", 1.0)) / wsum * tokens
            need = int(math.ceil(need_tokens / S))
            done = sources.get(e["name"])
            have = rows_available(done, "train", rules[e["name"]], S, out, eot) if done else None
            supply.append({"name": e["name"], "rule": rules[e["name"]], "rows_needed": need, "rows_available": have,
                           "short": have is None or have < need, "missing": done is None})
            budget[e["name"]][pname] = int(need_tokens)
        for e in phase.get("eval") or []:
            rule = e.get("rule", rules.get(e["name"], phase.get("rule", "stream")))
            done = sources.get(e["name"])
            have = rows_available(done, "val", rule, S, out, eot) if done else None
            eval_supply.append({"name": e["name"], "rule": rule, "rows_needed": int(e.get("rows", 0)),
                                "rows_available": have, "short": have is None or have < int(e.get("rows", 0)),
                                "missing": done is None})
        phases[pname] = {**phase, "supply": supply, "eval_supply": eval_supply}
    token_budget = {}
    for name, per in budget.items():
        built = sources.get(name, {}).get("tokens")
        token_budget[name] = {"tokens_built": built, "tokens_needed": per, "tokens_needed_total": sum(per.values()),
                              "short": built is None or built < sum(per.values())}
    mix = {"name": recipe.get("name", "mix"), "tokenizer": "tokenizer", "tokenizer_source": recipe.get("tokenizer"),
           "vocab_size": int(recipe.get("vocab_size", 0)), "eot": eot, "scale": scale, "created_utc": utc(),
           "missing": missing, "external_missing": external_missing, "sources": sources, "phases": phases,
           "token_budget": token_budget}
    write_json(out / "mix.json", mix)
    return mix, missing


def print_supply(mix: dict) -> None:
    def cell(v):
        return "-" if v is None else f"{v:,}"

    for pname, phase in mix["phases"].items():
        print(f"\n== supply, phase {pname}: seq_len {phase['seq_len']:,}, "
              f"{float(phase['tokens']) * mix['scale'] / 1e9:.4g}B tokens")
        print(f"   {'source':<16}{'split':<7}{'rule':<8}{'rows needed':>13}{'rows available':>16}{'x':>8}")
        for split, rows in (("train", phase["supply"]), ("val", phase["eval_supply"])):
            for r in rows:
                ratio = "-" if not r["rows_available"] or not r["rows_needed"] else \
                    f"{r['rows_available'] / r['rows_needed']:.2f}"
                flag = ("   !! MISSING" if r["missing"] else "   !! SHORT") if r["short"] else ""
                print(f"   {r['name']:<16}{split:<7}{r['rule']:<8}{cell(r['rows_needed']):>13}"
                      f"{cell(r['rows_available']):>16}{ratio:>8}{flag}")
    print("\n== tokens needed over all phases (a source used by several phases is read once: later phases get its "
          "last shards)")
    print(f"   {'source':<16}{'built':>16}{'needed':>16}")
    for name, b in mix["token_budget"].items():
        print(f"   {name:<16}{cell(b['tokens_built']):>16}{cell(b['tokens_needed_total']):>16}"
              f"{'   !! SHORT' if b['short'] else ''}")
    short = sorted({r["name"] for p in mix["phases"].values() for r in p["supply"] + p["eval_supply"] if r["short"]}
                   | {n for n, b in mix["token_budget"].items() if b["short"]})
    if short:
        print(f"\n!! SHORT OR MISSING: {', '.join(short)} — the trainer repeats rows of a short source; "
              f"a missing source has no rows at all")
    if mix["missing"]:
        print(f"!! NOT BUILT: {', '.join(mix['missing'])}")
    if mix["external_missing"]:
        print(f"   external sources not in this folder yet: {', '.join(mix['external_missing'])}")
    sys.stdout.flush()


# ----------------------------------------------------------------------------------------------------------------
# --list
# ----------------------------------------------------------------------------------------------------------------
def list_sources(specs: list[dict], raw: Raw, tokenizer: str | None = None) -> bool:
    """Print, per source, the Hub files it would read and whether the repo answers without a token."""
    ok = True
    if tokenizer and not Path(tokenizer).exists():
        try:
            from huggingface_hub import HfApi

            info = retry(lambda: HfApi(token=False).model_info(tokenizer, token=False), f"model {tokenizer}")
            gated = bool(getattr(info, "gated", False) or getattr(info, "private", False))
            print(f"tokenizer {tokenizer}: {'GATED' if gated else 'reachable without a token'}")
            ok &= not gated
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"tokenizer {tokenizer}: !! NOT REACHABLE: {type(e).__name__}: {str(e)[:200]}")
    for spec in specs:
        cands = [("", spec)]
        if spec.get("fallback"):
            fb = {k: v for k, v in spec.items() if k not in ("fallback", "parts", "files", "filter")}
            cands.append((" (fallback)", {**fb, **spec["fallback"]}))
        for tag, s in cands:
            where = s.get("repo") or s.get("local")
            try:
                parts = resolve_parts(s, raw.listing(s))
                if not s.get("local"):  # the first file answers a request without a token
                    from huggingface_hub import get_hf_file_metadata, hf_hub_url

                    first = parts[0]["files"][0][0]
                    retry(lambda: get_hf_file_metadata(hf_hub_url(s["repo"], first, repo_type="dataset"), token=False),
                          f"head {where}/{first}")
                n = sum(len(p["files"]) for p in parts)
                gb = sum(b for p in parts for _, b in p["files"]) / 1e9
                print(f"{s['name']}{tag}: {where}: reachable without a token; {n} file(s), {gb:.2f} GB listed, "
                      f"target {float(s['target_tokens']):.3g} tokens")
                for p in parts:
                    names = [f for f, _ in p["files"]]
                    shown = names if len(names) <= 4 else names[:3] + [f"... {len(names) - 3} more"]
                    print(f"    {p['label']} (to {100 * p['cum_share']:.0f} % of the target): {', '.join(shown)}")
            except Exception as e:  # noqa: BLE001
                ok = ok and bool(tag)  # an unreachable fallback is a warning only
                print(f"{s['name']}{tag}: {where}: !! NOT REACHABLE: {type(e).__name__}: {str(e)[:300]}")
    sys.stdout.flush()
    return ok


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def run(args, tok=None, raw: Raw | None = None, gpt2_count=None) -> int:
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in recipe["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("the recipe names a source twice")
    only = [n for n in (args.only or "").split(",") if n]
    unknown = [n for n in only if n not in names]
    if unknown:
        raise ValueError(f"--only: no source named {unknown}; the recipe has {names}")
    todo = [s for s in recipe["sources"] if s["kind"] != "external" and (not only or s["name"] in only)]
    if args.names == "external":
        print("\n".join(s["name"] for s in recipe["sources"] if s["kind"] == "external"))
        return 0
    if args.names:
        print("\n".join(s["name"] for s in sorted(todo, key=lambda s: -float(s.get("target_tokens", 0)))))
        return 0
    tokenizer = args.tokenizer or recipe.get("tokenizer")
    mode = args.read if args.read != "auto" else ("stream" if args.scale < 0.05 else "download")
    raw = raw or Raw(Path(args.raw) if args.raw else out / "_raw", mode)
    if args.list:
        return 0 if list_sources(todo, raw, tokenizer) else 1

    failed: dict[str, str] = {}
    if not args.mix_only:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        os.environ["RAYON_NUM_THREADS"] = str(max(1, args.workers))
        if tok is None:
            tok = Tok(tokenizer)
        for key in ("eot", "vocab_size"):
            if key in recipe and int(recipe[key]) != int(getattr(tok, key)):
                raise ValueError(f"the recipe says {key} = {recipe[key]}, "
                                 f"the tokenizer {tokenizer} has {getattr(tok, key)}")
        tok.save(out / "tokenizer")
        for spec in todo:
            name = spec["name"]
            done_path = out / f"{name}.done.json"
            if done_path.exists():
                old = json.loads(done_path.read_text(encoding="utf-8"))
                same = old.get("recipe") == spec and float(old.get("scale", 1.0)) == float(args.scale)
                log(f"[{name}] already built ({old.get('tokens', 0):,} train tokens): skipped"
                    + ("" if same else " — !! its done.json was built from a different recipe entry or scale; "
                                       "delete it to rebuild"))
                continue
            try:
                build_source(spec, recipe, out, tok, raw, args.scale, args.workers, gpt2_count)
            except Exception as e:  # noqa: BLE001  one source's failure must not stop the others
                failed[name] = f"{type(e).__name__}: {e}"
                log(f"[{name}] !! FAILED: {failed[name]}")
                traceback.print_exc()
                sys.stderr.flush()
    missing: list[str] = []
    if not args.no_mix:
        if tok is None and not (out / "tokenizer" / "tokenizer.json").exists():
            Tok(tokenizer).save(out / "tokenizer")
        mix, missing = write_mix(recipe, out, args.scale)
        print_supply(mix)
        log(f"wrote {out / 'mix.json'}: {len(mix['sources'])} source(s)")
    if not args.mix_only:
        built = [s["name"] for s in todo if s["name"] not in failed]
        log(f"summary: {len(built)} of {len(todo)} source(s) built or already there: {', '.join(built) or '-'}")
    for name, err in failed.items():
        log(f"summary: !! {name} FAILED: {err}")
    if failed:
        return 1
    return 1 if (missing and not only and not args.no_mix) else 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True, help="the mix folder (shards, done.json files, tokenizer, mix.json)")
    ap.add_argument("--only", default="", help="comma-separated source names (default: every source)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every token target (0.001: a smoke build)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="tokenizer threads")
    ap.add_argument("--tokenizer", default=None, help="Hub id or folder (default: the recipe's)")
    ap.add_argument("--list", action="store_true", help="print the Hub files per source and whether each repo "
                                                        "answers without a token; build nothing")
    ap.add_argument("--names", nargs="?", const="build", choices=("build", "external"),
                    help="print the source names this run would build, largest target first (or: the external ones)")
    ap.add_argument("--read", choices=("auto", "download", "stream"), default="auto",
                    help="download: whole raw files to --raw, deleted after use; stream: read over HTTP only as far "
                         "as needed; auto: stream when --scale < 0.05")
    ap.add_argument("--raw", default=None, help="folder for raw downloads and temporary files (default: OUT/_raw)")
    ap.add_argument("--no-mix", action="store_true", help="do not write mix.json (parallel per-source runs)")
    ap.add_argument("--mix-only", action="store_true", help="only write mix.json from the done.json files in OUT")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

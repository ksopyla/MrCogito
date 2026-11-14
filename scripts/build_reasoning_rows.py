#!/usr/bin/env python
"""Reasoning rows for the 100M long-context model: rows whose answers need text that sits thousands of tokens back.

Web text rarely needs far context, so a few percent of the training mix is constructed here. Four shard sets, each
<name>_train_%06d.bin (at most 100M tokens per shard, whole rows), <name>_val_000000.bin (64 rows) and
<name>.done.json:

  multihop_16k  0.20e9 tokens   "library" rows from 2WikiMultihopQA, TRAIN split only: the context paragraphs of as
  multihop_64k  0.06e9 tokens   many questions as fit, shuffled, each `Title: ...\\n<sentences>`, and at the END one
                                `Question: ...\\nEvidence: <the chain in words>\\nAnswer: ...` block per question.
  lookup_16k    0.10e9 tokens   a contiguous span of one public-domain book (Common Pile, Project Gutenberg) with K
  lookup_64k    0.03e9 tokens   invented facts planted at sentence / paragraph boundaries (K in 4-12 at 16k, 8-24 at
                                64k), and at the END one `Question: ...\\nAnswer: ...` block per fact, shuffled.

Row format (read by data.bin_token_dataset.BinRowDataset with rule "stream"): [eot] + text tokens, right-padded with
eot to exactly seq_len. Every padding eot is its own one-token document, so the padding costs no loss. Shards are the
modded-nanogpt layout (256 int32 header [20240520, 1, n_tokens], uint16 ids), whole rows only. Plain next-token text:
no loss masks.

Multi-hop rows. A question's block names the chain (`Evidence:`, rendered from the item's evidence triples, or its
supporting sentences when it has none) and then the answer; the paragraphs the chain rests on are far back in the row.
The question block is ordered so that a question whose supporting paragraphs all sit close to the block comes last.
A 2Wiki item is used at most once per pass over the split; when the items run out before the target is reached a new
pass starts in a different seeded order (paragraphs meet different neighbours); done.json says how many passes were
used. A row is filled to about 1 % of seq_len: the last question keeps its supporting paragraphs and as many of its
other paragraphs as fit, and what is left is topped up with distractor paragraphs of the next questions. The rows of
the val shard come from 8 parquet row groups that training rows never read. Items whose answer is neither yes / no
nor a string of their own paragraphs are dropped. The validation and test splits of 2Wiki are never read; HotpotQA,
MuSiQue, SQuAD and BABILong (our evals) are not used at all.

Lookup rows. Facts are invented: fictional names assembled from syllables, random values, nine kinds (a person's
number, a place's founding year, an object's colour / material / owner, a key -> value setting, a short quoted
phrase, a date, a 6-10 character identifier) written from 53 statement and 52 question templates plus 8 revision
templates. About 15 % of the facts are stated twice, the later statement revising the earlier one (the answer is the
latest value); about 15 % of the rows ask one or two questions about something that was never planted, answered
`not stated in the text`. No template uses the words of our evals' needles (RULER, BABILong: BANNED below; a
generated statement or question that would contain one is redrawn). Books: at least seq_len tokens, licence "Public
Domain", up to 3 non-overlapping spans per book. A book is never used when the training mix would hold it out as
validation, under either rule of scripts/build_m21_mix.py: the one its `books` source uses ("longdocs_lite": every
100th document, in file order, among those with at least 40,960 characters and 16,384 GPT-2 tokens; replayed here
with tiktoken over every document that is read), or the hash rule (stable hash of the id, or of the text, with
`% 1_000_000 < 5000` or `% 200 == 0`). The remaining books are split by the hash of the id into this builder's own
val and train books, per set.

Tokens. Every piece of a row is tokenized on its own and the ids are concatenated; pieces meet only where a
non-space character is followed by white space, which is always a token boundary for the byte-level BPE used here,
so the ids equal the tokenization of the whole row text (checked on every 25th row; done.json: `joint_check`).

Downloads are anonymous (no token is sent), ranged and resumable: 2Wiki is read one parquet row group at a time (a
full build downloads both train files), the Gutenberg files front to back only as far as books are needed. Raw files
live in <out>/_raw_reasoning_rows (a folder of its own: --out can be the mix folder of scripts/build_m21_mix.py) and
are deleted when no set of the run needs them any more. A set that has its done.json is
skipped (resume); a set without one is rebuilt from scratch. A build is a pure function of --seed, --scale and the
tokenizer. done.json holds the statistics (rows, tokens, questions per row, answer distances, bytes of text, seconds,
parameters, the Hub files read) and, in the layout of scripts/build_m21_mix.py, what that script's mix.json reads
for an "external" source: `tokens`, `shards`, and train / val `rows` per rule and row length.

Usage:
  uv run python scripts/build_reasoning_rows.py --out /data/m21mix          # the full targets, into the mix folder
  uv run python scripts/build_reasoning_rows.py --out /tmp/reasoning_smoke --scale 0.002        # a complete tiny build
  uv run python scripts/build_reasoning_rows.py --out DIR --only lookup_16k,lookup_64k --seed 1
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EOT_TOKEN = "<|endoftext|>"
MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256  # the shard layout of data/bin_token_dataset.py
SHARD_TOKENS = 100_000_000
VAL_ROWS = 64
MIN_BACK = 2000        # "far": an answer at least this many tokens after the paragraph it needs
MIN_FILL = 0.99        # a row that a pass ends before it is this full is not written
TOP_UP_ITEMS = 6       # a full library row is topped up with distractor paragraphs of this many following items
CHECK_EVERY = 25       # every n-th row: ids == tokenizer(text) and decode(ids) == text
NOT_STATED = "not stated in the text"
RAW_DIR = "_raw_reasoning_rows"  # downloads in progress; not "_raw", which the mix builder owns in the same folder

WIKI_REPO = "framolfese/2WikiMultihopQA"
WIKI_TRAIN_FILES = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
WIKI_COLUMNS = ["id", "question", "answer", "type", "evidences", "supporting_facts", "context"]
WIKI_VAL_GROUPS = 8    # parquet row groups (1,000 items each) that only the val rows read
WIKI_LICENSE_NOTE = ("no license tag on the Hub repo; its README states Apache-2.0, the licence of the original "
                     "github.com/Alab-NII/2wikimultihop")
BOOK_REPO = "common-pile/project_gutenberg_filtered"
BOOK_LICENSE_NOTE = "no license tag on the Hub repo; each document carries metadata.license, used when 'Public Domain'"

# name -> what to build. `buckets`: which of the 100 hash buckets of book ids feed this set's val / train rows.
SETS: dict[str, dict] = {
    "multihop_16k": {"kind": "multihop", "seq_len": 16384, "tokens": 0.20e9},
    "multihop_64k": {"kind": "multihop", "seq_len": 65536, "tokens": 0.06e9},
    "lookup_16k": {"kind": "lookup", "seq_len": 16384, "tokens": 0.10e9, "facts": (4, 12), "spans_per_book": 3,
                   "buckets": {"val": (0, 6), "train": (56, 100)}},
    "lookup_64k": {"kind": "lookup", "seq_len": 65536, "tokens": 0.03e9, "facts": (8, 24), "spans_per_book": 3,
                   "buckets": {"val": (6, 26), "train": (26, 56)}},
}
P_REVISED = 0.15       # share of planted facts stated twice (the later statement wins)
P_MISSING_ROW = 0.15   # share of lookup rows with 1-2 questions about something never planted


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------------------------------------
class Tok:
    """The tokenizer calls the builders need (a `tokenizers.Tokenizer` underneath; the tests pass a stand-in)."""

    def __init__(self, tk, name: str):
        tk.no_truncation()
        tk.no_padding()
        self.tk, self.name = tk, name
        eot = tk.token_to_id(EOT_TOKEN)
        if eot is None:
            raise ValueError(f"tokenizer {name}: no {EOT_TOKEN} token")
        self.eot = int(eot)
        self.vocab = int(tk.get_vocab_size(with_added_tokens=True))
        if self.vocab > 65536:
            raise ValueError(f"tokenizer {name}: {self.vocab} ids do not fit the uint16 shards")

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]

    def encode_offsets(self, texts) -> list[tuple[list[int], list[tuple[int, int]]]]:
        """Per text: (ids, [(first char, one past the last char) of each token])."""
        return [(e.ids, e.offsets) for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=False)


def load_tokenizer(id_or_dir: str) -> Tok:
    from tokenizers import Tokenizer

    p = Path(id_or_dir)
    if p.is_dir():
        path = p / "tokenizer.json"
    elif p.is_file():
        path = p
    else:
        from huggingface_hub import hf_hub_download

        path = retry(lambda: hf_hub_download(id_or_dir, "tokenizer.json", token=False), f"tokenizer {id_or_dir}")
    return Tok(Tokenizer.from_file(str(path)), id_or_dir)


# ----------------------------------------------------------------------------------------------------------------
# Hub access: anonymous, ranged, retried
# ----------------------------------------------------------------------------------------------------------------
def _hub_errors() -> tuple:
    errs: list = [OSError, EOFError, TimeoutError, zlib.error]
    for mod, name in (("huggingface_hub.errors", "HfHubHTTPError"), ("httpx", "HTTPError")):
        try:
            errs.append(getattr(__import__(mod, fromlist=[name]), name))
        except Exception:  # noqa: BLE001 - an optional library, or a version without the name
            pass
    return tuple(errs)


def retry(fn, what: str, tries: int = 6, base: float = 2.0, sleep=time.sleep):
    """fn(), again after 2, 4, 8, 16, 32 s when the Hub or the network fails; the last error is raised."""
    errors = _hub_errors()
    for k in range(tries):
        try:
            return fn()
        except errors as e:
            if k == tries - 1:
                raise
            wait = min(60.0, base * 2 ** k)
            log(f"[hub] {what}: {type(e).__name__}: {str(e)[:200]} - retry {k + 1}/{tries - 1} in {wait:.0f} s")
            sleep(wait)


def http_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of a public Hub file. No credentials are sent."""
    import requests

    def get() -> bytes:
        r = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "reasoning-rows-builder"},
                         timeout=(20, 180))
        if r.status_code not in (200, 206):
            raise OSError(f"HTTP {r.status_code}")
        data = r.content if r.status_code == 206 else r.content[start:end]  # 200: the server ignored the range
        if len(data) != end - start:
            raise OSError(f"{len(data)} bytes for a range of {end - start}")
        return data

    return retry(get, f"bytes {start:,}-{end:,} of {url.rsplit('/', 1)[-1]}")


def hub_file_url(repo: str, name: str, revision: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo, name, repo_type="dataset", revision=revision)


def hub_info(repo: str) -> dict:
    """Revision, license tag and file sizes of a public dataset repo."""
    from huggingface_hub import HfApi

    info = retry(lambda: HfApi(token=False).dataset_info(repo, files_metadata=True), f"info of {repo}")
    card = info.card_data
    return {"sha": info.sha, "license_tag": getattr(card, "license", None) if card is not None else None,
            "sizes": {s.rfilename: int(s.size or 0) for s in info.siblings}}


class HubPrefix:
    """A public Hub file read front to back in ranges and kept as a growing local copy, so a second reader (the other
    lookup set, a resumed run) does not download the same bytes again, and a build that needs little downloads only
    the head of the file."""

    CHUNK = 8 << 20

    def __init__(self, url: str, path: Path, size: int):
        self.url, self.path, self.size = url, Path(path), int(size)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def have(self) -> int:
        return min(self.size, self.path.stat().st_size) if self.path.exists() else 0

    def chunks(self):
        pos = 0
        while pos < self.size:
            have = self.have()
            if pos >= have:
                data = http_range(self.url, have, min(self.size, have + self.CHUNK))
                with open(self.path, "ab") as f:
                    f.write(data)
                have += len(data)
            with open(self.path, "rb") as f:
                f.seek(pos)
                data = f.read(min(self.CHUNK, have - pos))
            pos += len(data)
            yield data

    def fetch_all(self) -> Path:
        for _ in self.chunks():
            pass
        return self.path


class HubRangeFile(io.RawIOBase):
    """A read-only, seekable view of a public Hub file through ranged GETs: pyarrow reads the parquet footer and then
    only the row groups that are asked for."""

    BLOCK = 1 << 20  # read at least this much per request (a row group is a few MB, the footer under 1 MB)

    def __init__(self, url: str, size: int, block: int | None = None):
        super().__init__()
        self.url, self.size, self.block = url, int(size), block or self.BLOCK
        self.pos, self.buf_start, self.buf, self.fetched = 0, 0, b"", 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        a = self.pos
        if not (self.buf_start <= a and a + n <= self.buf_start + len(self.buf)):
            self.buf = http_range(self.url, a, min(self.size, a + max(n, self.block)))
            self.buf_start = a
            self.fetched += len(self.buf)
        self.pos += n
        return self.buf[a - self.buf_start: a - self.buf_start + n]

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def gz_json_lines(chunks):
    """Rows of a (possibly multi-member) gzip of JSON lines, decoded as the compressed chunks arrive."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    tail = b""
    for data in chunks:
        out = [tail]
        while data:
            out.append(d.decompress(data))
            data = b""
            if d.eof:  # a gzip member ended; another may follow in the same chunk
                data = d.unused_data
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        lines = b"".join(out).split(b"\n")
        tail = lines.pop()
        for line in lines:
            if line.strip():
                yield json.loads(line)
    if tail.strip():
        yield json.loads(tail)


# ----------------------------------------------------------------------------------------------------------------
# Rows, shards, statistics
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Row:
    ids: np.ndarray          # [seq_len] uint16: eot, the text, eot padding
    n: int                   # tokens before the padding (the leading eot included)
    text: str
    q_char: int              # where the question block starts in `text`
    qa: list[dict]           # per question: question, answer, answer_pos, distance (None: nothing was planted), ...
    info: dict = field(default_factory=dict)


def finish_row(tok: Tok, seq_len: int, parts: list, texts: list[str], q_char: int, qa: list[dict], info: dict) -> Row:
    """`parts`: the token pieces after the leading eot, in order; `texts`: their text."""
    body = np.concatenate([np.asarray(p, dtype=np.uint16) for p in parts])
    n = 1 + len(body)
    if n > seq_len:
        raise RuntimeError(f"row of {n} tokens for seq_len {seq_len}")
    if (body == tok.eot).any():
        raise RuntimeError("an eot inside the text of a row")
    ids = np.full(seq_len, tok.eot, dtype=np.uint16)
    ids[1:n] = body
    return Row(ids, n, "".join(texts), q_char, qa, info)


class ShardWriter:
    """Rows of exactly seq_len uint16 tokens -> <out>/<prefix>_%06d.bin, at most `shard_tokens` per shard."""

    def __init__(self, out: Path, prefix: str, seq_len: int, shard_tokens: int = SHARD_TOKENS):
        self.out, self.prefix, self.seq_len = Path(out), prefix, int(seq_len)
        self.rows_per_shard = max(1, int(shard_tokens) // self.seq_len)
        self.rows, self.shards, self._f, self._n = 0, [], None, 0  # shards: [{"file", "tokens", "rows"}]

    @property
    def files(self) -> list[str]:
        return [s["file"] for s in self.shards]

    def add(self, ids: np.ndarray) -> None:
        if ids.dtype != np.uint16 or len(ids) != self.seq_len:
            raise ValueError("a row must be seq_len uint16 tokens")
        if self._f is None:
            self._path = self.out / f"{self.prefix}_{len(self.shards):06d}.bin"
            self._f = open(str(self._path) + ".tmp", "wb")
            self._f.write(np.zeros(HEADER_INTS, dtype=np.int32).tobytes())
            self._n = 0
        self._f.write(ids.tobytes())
        self._n += 1
        self.rows += 1
        if self._n >= self.rows_per_shard:
            self.close()

    def close(self) -> list[str]:
        if self._f is not None:
            header = np.zeros(HEADER_INTS, dtype=np.int32)
            header[0], header[1], header[2] = MAGIC, VERSION, self._n * self.seq_len
            self._f.seek(0)
            self._f.write(header.tobytes())
            self._f.close()
            os.replace(str(self._path) + ".tmp", self._path)
            self.shards.append({"file": self._path.name, "tokens": self._n * self.seq_len, "rows": self._n})
            self._f = None
        return self.files


def _shard_cap(split: str, shard_tokens: float, rows: dict, seq_len: int) -> int:
    """Tokens per shard: the val rows are always one file, <name>_val_000000.bin."""
    return int(max(shard_tokens, rows["val"] * seq_len)) if split == "val" else int(shard_tokens)


def summarize(values, edges=(500, 1000, 2000, 4000, 8000, 16000, 32000)) -> dict:
    """Mean, quantiles and a histogram of token distances (or of any list of numbers)."""
    if not len(values):
        return {"n": 0}
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    out = {"n": int(len(v)), "mean": round(float(v.mean()), 1), "min": float(v.min()), "p5": float(q[0]),
           "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "max": float(v.max())}
    if edges:
        out["share_ge_2000"] = round(float((v >= MIN_BACK).mean()), 4)
        names = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
        out["histogram"] = dict(zip(names, np.bincount(np.searchsorted(edges, v, side="right"),
                                                        minlength=len(edges) + 1).tolist()))
    return out


class SplitStats:
    """What done.json says about one split of one set."""

    def __init__(self, tok: Tok, seq_len: int):
        self.tok, self.seq_len = tok, seq_len
        self.rows = self.content = self.text_bytes = self.checked = self.mismatched = 0
        self.items, self.fills, self.dist = [], [], []
        self.counts: collections.Counter = collections.Counter()
        self.t0 = time.time()

    def add(self, row: Row) -> None:
        if self.rows % CHECK_EVERY == 0:  # the pieces, concatenated, are the tokenization of the whole text
            body = row.ids[1: row.n].tolist()
            self.checked += 1
            self.mismatched += not (self.tok.encode(row.text) == body and self.tok.decode(body) == row.text)
        self.rows += 1
        self.content += row.n
        self.text_bytes += len(row.text.encode("utf-8"))
        self.items.append(len(row.qa))
        self.fills.append(row.n / self.seq_len)
        self.dist += [q["distance"] for q in row.qa if q["distance"] is not None]
        for k, v in row.info.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.counts[k] += v

    def summary(self, writer: ShardWriter) -> dict:
        """`tokens`, `text_bytes`, `shards` and `rows` as scripts/build_m21_mix.py writes them (its mix.json reads the
        rows of an external set from here: {rule: {seq_len: rows}}); `n_rows` is the plain count."""
        sec = time.time() - self.t0
        fills = np.asarray(self.fills or [0.0])
        return {"n_rows": self.rows, "tokens": self.rows * self.seq_len, "content_tokens": int(self.content),
                "text_bytes": int(self.text_bytes), "shards": writer.shards,
                "rows": {"stream": {str(self.seq_len): self.rows}},
                "items_per_row": summarize(self.items, edges=None),
                "fill": {"mean": round(float(fills.mean()), 5), "min": round(float(fills.min()), 5),
                         "share_ge_0.99": round(float((fills >= 0.99).mean()), 4)},
                "answer_distance": summarize(self.dist), "counts": dict(sorted(self.counts.items())),
                "joint_check": {"rows": self.checked, "mismatched": int(self.mismatched)}, "seconds": round(sec, 2)}


# ----------------------------------------------------------------------------------------------------------------
# Multi-hop: 2WikiMultihopQA library rows
# ----------------------------------------------------------------------------------------------------------------
EVIDENCE_PHRASES = {  # relations that do not read as "the <relation> of <subject> is <object>"
    "award received": "{s} received the award {o}",
    "educated at": "{s} was educated at {o}",
    "founded by": "{s} was founded by {o}",
    "has part": "{s} has the part {o}",
    "child": "A child of {s} is {o}",
    "sibling": "A sibling of {s} is {o}",
}
_REL_TAILS = {"by", "at", "in", "of", "to", "from", "for", "on", "with"}


def _clean(s) -> str:
    return " ".join(str(s if s is not None else "").split())


def evidence_sentence(s: str, r: str, o: str) -> str:
    """One evidence triple (subject, relation, object) in words, without the final full stop."""
    tpl = EVIDENCE_PHRASES.get(r)
    if tpl is None:
        words = r.split()
        odd = words[-1] in _REL_TAILS or words[0] in ("has", "is", "was")
        tpl = "For {s}, {r}: {o}" if odd else "The {r} of {s} is {o}"
    return tpl.format(s=s, r=r, o=o).rstrip(".")


@dataclass
class WikiItem:
    id: str
    question: str
    answer: str
    qtype: str
    evidence: str                         # the chain, in words
    paras: list[tuple[str, str, bool]]    # (title, text, supporting), no paragraph twice

    def piece(self, i: int) -> str:
        return f"\n\nTitle: {self.paras[i][0]}\n{self.paras[i][1]}"

    def question_piece(self) -> str:
        return f"\n\nQuestion: {self.question}\nEvidence: {self.evidence}\nAnswer:"


def parse_wiki_record(rec: dict, intern: dict | None = None) -> tuple[WikiItem | None, str]:
    """A 2Wiki record (HotpotQA layout) -> (item, "") or (None, why it is dropped)."""
    q, a = _clean(rec.get("question")), _clean(rec.get("answer"))
    if not q or not a:
        return None, "empty_question_or_answer"
    ctx, sf = rec.get("context") or {}, rec.get("supporting_facts") or {}
    sup = {_clean(t) for t in sf.get("title") or []}
    paras, seen, sents = [], set(), {}
    for t, ss in zip(ctx.get("title") or [], ctx.get("sentences") or []):
        t, x = _clean(t), _clean(" ".join(ss or []))
        if not t or not x or (t, x) in seen:
            continue
        seen.add((t, x))
        sents.setdefault(t, list(ss or []))
        if intern is not None:
            t, x = intern.setdefault((t, x), (t, x))
        paras.append((t, x, t in sup))
    if not sup or sup - {p[0] for p in paras}:
        return None, "supporting_paragraph_missing"
    triples = [[_clean(x) for x in e] for e in rec.get("evidences") or []]
    triples = [e for e in triples if len(e) == 3 and all(e)]
    if triples:
        chain = [evidence_sentence(*e) for e in triples]
    else:  # no triples: the supporting sentences themselves
        chain = [_clean(sents[_clean(t)][i]).rstrip(".") for t, i in zip(sf.get("title") or [], sf.get("sent_id") or [])
                 if _clean(t) in sents and 0 <= i < len(sents[_clean(t)])]
        chain = [c for c in chain if c]
    if not chain:
        return None, "no_evidence"
    if a.lower() not in ("yes", "no") and a.lower() not in " ".join(f"{t} {x}" for t, x, _ in paras).lower():
        return None, "answer_not_in_context"
    return WikiItem(str(rec.get("id", "")), q, a, str(rec.get("type", "")), ". ".join(chain) + ".", paras), ""


class WikiSource:
    """2WikiMultihopQA TRAIN items, read one parquet row group (1,000 items) at a time. A fixed choice of
    WIKI_VAL_GROUPS row groups (the same for every seed) feeds the val rows; the others, in a seeded order, the
    train rows. `files` (tests): local parquet files instead of the Hub."""

    def __init__(self, raw_dir: Path, seed: int, whole_files: bool = False, files: dict | None = None,
                 n_val_groups: int = WIKI_VAL_GROUPS):
        self.local, self.whole = files, whole_files
        self.names = list(files) if files else list(WIKI_TRAIN_FILES)
        self.info = {"sha": "local", "license_tag": None, "sizes": {}} if files else hub_info(WIKI_REPO)
        missing = [n for n in self.names if not files and n not in self.info["sizes"]]
        if missing:
            raise FileNotFoundError(f"{WIKI_REPO}: no {missing}")
        self.raw = Path(raw_dir) / f"2wiki-{self.info['sha'][:12]}"
        self._pf, self._remote, self._items, self._intern = {}, {}, {}, {}
        self.counts: collections.Counter = collections.Counter()
        groups = [(n, g) for n in self.names for g in range(self._parquet(n).metadata.num_row_groups)]
        order = sorted(groups)
        random.Random("2wiki-val-row-groups").shuffle(order)
        self.val_groups, rest = order[:n_val_groups], order[n_val_groups:]
        random.Random(f"{seed}|2wiki-train-row-groups").shuffle(rest)
        self.train_groups = rest
        self.n_train_groups = len(rest)

    def _parquet(self, name: str):
        import pyarrow.parquet as pq

        if name not in self._pf:
            if self.local:
                src = str(self.local[name])
            else:
                url, size = hub_file_url(WIKI_REPO, name, self.info["sha"]), self.info["sizes"][name]
                if self.whole:
                    src = str(HubPrefix(url, self.raw / name.replace("/", "__"), size).fetch_all())
                else:
                    src = self._remote[name] = HubRangeFile(url, size)
            self._pf[name] = pq.ParquetFile(src)
        return self._pf[name]

    def _group(self, key: tuple[str, int]) -> list[WikiItem]:
        if key not in self._items:
            name, g = key
            cols = [c for c in WIKI_COLUMNS if c in self._parquet(name).schema_arrow.names]
            recs = retry(lambda: self._parquet(name).read_row_group(g, columns=cols).to_pylist(), f"{name} group {g}")
            items = []
            for rec in recs:
                it, why = parse_wiki_record(rec, self._intern)
                self.counts["records"] += 1
                self.counts["kept" if it else f"dropped_{why}"] += 1
                if it:
                    items.append(it)
            self._items[key] = items
        return self._items[key]

    def val_items(self) -> list[WikiItem]:
        return [it for key in self.val_groups for it in self._group(key)]

    def train_items(self, n_groups: int) -> list[WikiItem]:
        """The items of the first `n_groups` train row groups (a longer prefix extends a shorter one)."""
        return [it for key in self.train_groups[:n_groups] for it in self._group(key)]

    def describe(self) -> dict:
        files = [{"file": n, "bytes_total": self.info["sizes"].get(n),
                  "bytes_fetched": (self._remote[n].fetched if n in self._remote
                                    else self.info["sizes"].get(n) if self.whole and n in self._pf else 0),
                  "row_groups_read": sorted(g for f, g in self._items if f == n)} for n in self.names]
        return {"repo": WIKI_REPO, "revision": self.info["sha"], "license_tag": self.info["license_tag"],
                "license_note": WIKI_LICENSE_NOTE, "splits_read": ["train"], "files": files,
                "val_row_groups": [list(k) for k in self.val_groups], "records": dict(sorted(self.counts.items()))}

    def cleanup(self) -> None:
        self._pf.clear()
        self._remote.clear()
        shutil.rmtree(self.raw, ignore_errors=True)


def wiki_item_stream(source, split: str, seed_key: str, est_items: float):
    """(pass, item) for ever: every item of the split once per pass, each pass in its own seeded order. Train row
    groups are read as they are needed (a small build reads a few); a new pass starts only when all are used."""
    total = 0 if split == "val" else source.n_train_groups
    n = min(total, max(1, math.ceil(est_items / 1000 * 1.15)))
    pool = source.val_items() if split == "val" else source.train_items(n)
    if not pool:
        raise RuntimeError(f"no 2Wiki items for the {split} rows")
    p, fresh = 0, list(pool)
    while True:
        random.Random(f"{seed_key}|pass{p}|{len(pool)}").shuffle(fresh)
        for it in fresh:
            yield p, it
        if n < total:  # the items read so far ran out, not the split: read more row groups, same pass
            had, n = len(pool), min(total, n + max(2, n // 2))
            pool = source.train_items(n)
            fresh = pool[had:]
        else:
            p, fresh = p + 1, list(pool)


@dataclass
class _Entry:
    pass_idx: int
    item: WikiItem
    para_ids: list[list[int]]
    q_ids: list[int]
    a_ids: list[int]


class _Feed:
    """Items with their token pieces, tokenized a block at a time, with a few items of look-ahead."""

    def __init__(self, stream, tok: Tok, block: int = 128):
        self.stream, self.tok, self.block = iter(stream), tok, block
        self.buf: collections.deque = collections.deque()
        self.counts: collections.Counter = collections.Counter()

    def _fill(self) -> bool:
        chunk = list(itertools.islice(self.stream, self.block))
        texts = []
        for _, it in chunk:
            texts += [it.piece(i) for i in range(len(it.paras))] + [it.question_piece(), " " + it.answer]
        ids, k = self.tok.encode_batch(texts) if texts else [], 0
        for p, it in chunk:
            n = len(it.paras)
            e = _Entry(p, it, ids[k: k + n], ids[k + n], ids[k + n + 1])
            k += n + 2
            if any(self.tok.eot in x for x in e.para_ids + [e.q_ids, e.a_ids]):
                self.counts["items_with_eot_text"] += 1
                continue
            self.buf.append(e)
        return bool(chunk)

    def peek(self) -> _Entry | None:
        while not self.buf:
            if not self._fill():
                return None
        return self.buf[0]

    def pop(self) -> _Entry:
        return self.buf.popleft()

    def ahead(self, k: int) -> list[_Entry]:
        """Up to the next k items, without taking them."""
        while len(self.buf) < k and self._fill():
            pass
        return list(itertools.islice(self.buf, k))


def multihop_rows(feed: _Feed, tok: Tok, seq_len: int, seed_key: str):
    """Library rows from the feed, until it ends. Rows never mix two passes."""
    final = tok.encode("\n\n")
    budget = seq_len - 1 - len(final)
    k = 0
    while True:
        first = feed.peek()
        if first is None:
            return
        rng = random.Random(f"{seed_key}|row{k}")
        paras: dict[tuple[str, str], list] = {}  # (title, text) -> [piece text, ids, is a filler]
        qs: list[_Entry] = []
        used = 0
        while True:
            e = feed.peek()
            if e is None or e.pass_idx != first.pass_idx:
                break
            it = e.item
            new = [i for i, p in enumerate(it.paras) if p[:2] not in paras]
            sup = [i for i in new if it.paras[i][2]]
            rest = [i for i in new if not it.paras[i][2]]
            base = len(e.q_ids) + len(e.a_ids) + sum(len(e.para_ids[i]) for i in sup)
            if used + base > budget:
                if qs:
                    break  # this item opens the next row
                feed.pop()  # it does not fit an empty row
                feed.counts["items_too_long"] += 1
                continue
            feed.pop()
            used += base
            rng.shuffle(rest)
            take = list(sup)
            for i in rest:  # the item's other paragraphs, as far as they fit
                if used + len(e.para_ids[i]) <= budget:
                    take.append(i)
                    used += len(e.para_ids[i])
            for i in take:
                paras[it.paras[i][:2]] = [it.piece(i), e.para_ids[i], False]
            qs.append(e)
            if len(take) < len(new):
                break  # the row is full: the last item was trimmed to fit
        nxt = feed.peek()
        fillers = 0
        for e in feed.ahead(TOP_UP_ITEMS):  # top up with distractor paragraphs of the items that come next
            idx = [i for i, p in enumerate(e.item.paras) if not p[2] and p[:2] not in paras]
            rng.shuffle(idx)
            for i in idx:
                if used + len(e.para_ids[i]) <= budget:
                    paras[e.item.paras[i][:2]] = [e.item.piece(i), e.para_ids[i], True]
                    used += len(e.para_ids[i])
                    fillers += 1
        ended = nxt is None or nxt.pass_idx != first.pass_idx
        if not qs or (ended and (used + 1 + len(final)) / seq_len < MIN_FILL):
            feed.counts["partial_rows_dropped"] += bool(qs)  # the pass (or the feed) ended inside this row
            if nxt is None:
                return
            continue
        yield _library_row(paras, qs, fillers, tok, seq_len, rng, final)
        k += 1


def _library_row(paras: dict, qs: list[_Entry], fillers: int, tok: Tok, seq_len: int, rng: random.Random,
                 final: list[int]) -> Row:
    keys = list(paras)
    rng.shuffle(keys)
    head = paras[keys[0]][0][2:]  # the row starts with a title, not with a blank line
    parts, texts, end, pos = [tok.encode(head)], [head], {}, 1
    pos += len(parts[0])
    end[keys[0]] = pos
    for key in keys[1:]:
        text, ids, _ = paras[key]
        parts.append(ids)
        texts.append(text)
        pos += len(ids)
        end[key] = pos
    q_char = sum(len(t) for t in texts)
    # the end of the supporting paragraph that is farthest back, per question
    far = [min(end[p[:2]] for p in e.item.paras if p[2]) for e in qs]
    safe = [i for i, e in enumerate(qs) if pos + len(e.q_ids) - far[i] >= MIN_BACK]  # far enough wherever it is asked
    late = sorted(set(range(len(qs))) - set(safe), key=lambda i: (far[i], i))      # the closer, the later
    rng.shuffle(safe)
    qa = []
    for i in safe + late:
        e = qs[i]
        parts += [e.q_ids, e.a_ids]
        texts += [e.item.question_piece(), " " + e.item.answer]
        ans = pos + len(e.q_ids)
        qa.append({"id": e.item.id, "question": e.item.question, "answer": e.item.answer, "answer_pos": ans,
                   "distance": ans - far[i], "type": e.item.qtype})
        pos = ans + len(e.a_ids)
    parts.append(final)
    texts.append("\n\n")
    info = {"paragraphs": len(keys), "filler_paragraphs": fillers, "pass": qs[0].pass_idx,
            "yes_no_answers": sum(q["answer"].lower() in ("yes", "no") for q in qa)}
    return finish_row(tok, seq_len, parts, texts, q_char, qa, info)


def build_multihop_set(name: str, cfg: dict, source, tok: Tok, out: Path, seed: int, rows: dict,
                       shard_tokens: int) -> dict:
    S, res = cfg["seq_len"], {}
    for split in ("val", "train"):
        key = f"{seed}|{name}|{split}"
        feed = _Feed(wiki_item_stream(source, split, key, est_items=rows[split] * S / 900), tok)
        writer = ShardWriter(out, f"{name}_{split}", S, _shard_cap(split, shard_tokens, rows, S))
        stats, passes, last = SplitStats(tok, S), 0, 0.0
        for row in multihop_rows(feed, tok, S, key) if rows[split] > 0 else ():
            writer.add(row.ids)
            stats.add(row)
            passes = max(passes, row.info["pass"] + 1)
            if time.time() - last > 20:
                last = time.time()
                log(f"[{name}] {split} {writer.rows:,}/{rows[split]:,} rows")
            if writer.rows >= rows[split]:
                break
        writer.close()
        res[split] = {**stats.summary(writer), "passes_used": passes, "target_rows": rows[split],
                      "feed": dict(sorted(feed.counts.items()))}
        res[split]["counts"].pop("pass", None)
    return res


# ----------------------------------------------------------------------------------------------------------------
# Lookup: invented facts
# ----------------------------------------------------------------------------------------------------------------
# What our evals plant and ask (lm-eval RULER needles / variable tracking / word counts, BABILong's bAbI facts and
# instructions, the pass-key task). Checked without case, except "VAR " (RULER writes its variables in capitals).
BANNED = ["special magic", "magic number", "magic uuid", "magic word", "magic", "uuid", "pass key", "passkey",
          "one of the special", "mentioned in the provided text", "provided text", "hidden within", "hidden in",
          "the following text", "memorize", "i will quiz", "quiz", "variable", "are assigned the value",
          "most common words", "coded text", "coded word", "moved to the", "went to the", "journeyed to the",
          "travelled to the", "went back to the", "picked up the", "got the", "took the", "grabbed the",
          "dropped the", "discarded the", "put down the", "left the", "where is", "where was"]
BANNED_EXACT = ["VAR "]


def banned_in(text: str) -> str | None:
    low = text.lower()
    return next((b for b in BANNED if b in low), None) or next((b for b in BANNED_EXACT if b in text), None)


_ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "j", "k", "l", "m", "n", "p", "pr", "r", "s", "st", "t",
           "tr", "v", "w", "z", "th", "sh", "bl", "cl", "fr", "sk"]
_VOWELS = ["a", "e", "i", "o", "u"] * 3 + ["ae", "ai", "ea", "io", "ou", "y"]
_INNER = ["", "", "", "", "n", "r", "l", "s"]
_CODAS = ["", "", "n", "r", "l", "s", "th", "m", "nd", "rn", "x", "ld", "ck", "sh", "v"]
_PLACE_ENDS = ["stead", "haven", "gate", "crest", "vale", "port", "holm", "reach", "fell", "garth", "mouth", "wold"]
_PLACE_KINDS = ["town", "village", "port", "hamlet", "city", "abbey", "fort", "parish", "colony", "market town"]
_GROUPS = ["Guild", "Lodge", "Circle", "Company", "Society", "Chapter", "Brigade", "Choir", "League", "Assembly"]
_OBJECTS = ["lantern", "kettle", "satchel", "compass", "quilt", "barrel", "flute", "saddle", "mirror", "goblet",
            "cloak", "chest", "spade", "loom", "bell", "candlestick", "teapot", "basket", "stool", "drum", "helmet",
            "ladle", "trunk", "carpet"]
_COLOURS = ["crimson", "amber", "teal", "indigo", "ivory", "olive", "scarlet", "violet", "russet", "turquoise",
            "maroon", "saffron", "charcoal", "white", "black", "green", "blue", "yellow", "grey", "orange"]
_MATERIALS = ["oak", "pewter", "brass", "copper", "walnut", "leather", "marble", "tin", "silver", "birch", "slate",
              "iron", "glass", "wool", "bronze", "cedar", "linen", "granite", "clay", "wicker"]
_KEY_WORDS = ["limit", "delay", "depth", "width", "mode", "level", "rate", "count", "offset", "margin", "ratio",
              "window", "timeout", "threshold", "interval", "gain"]
_STATES = ["locked", "open", "muted", "paused", "armed", "idle", "steady", "reversed", "strict", "manual", "double",
           "narrow"]
_ADJ = ["quiet", "bright", "old", "young", "bold", "gentle", "silver", "golden", "distant", "patient", "merry",
        "humble", "swift", "steady", "northern", "mossy"]
_NOUNS = ["rivers", "lanterns", "harbours", "orchards", "sparrows", "bridges", "ploughs", "anchors", "candles",
          "meadows", "bakers", "weavers", "shepherds", "millers", "towers", "gardens"]
_VERBS = ["carry", "guard", "follow", "greet", "remember", "outlast", "welcome", "mend", "keep", "cross", "feed",
          "praise"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_THINGS = [("crate", "addressed to"), ("trunk", "stored for"), ("manuscript", "donated by"),
           ("telescope", "ordered by"), ("wagon", "hired by"), ("parcel", "sent to"), ("ledger", "kept by"),
           ("clock", "repaired for")]
_ID_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# kind -> attributes (when the kind has several), the noun phrase used by revision sentences, statement and question
# templates. Fields: {e} the subject, {E} the subject starting a sentence, {a} the attribute, {v} the value, {av} the
# value with its article, {p} a person, {o} an object, {n} an object's name, {k} a key.
FAMILIES: dict[str, dict] = {
    "person_number": {
        "attrs": ["badge number", "locker number", "ticket number", "cabin number", "permit number",
                  "membership number", "payroll number", "seat number"],
        "label": "the {a} of {e}",
        "statements": ["{e}'s {a} is {v}.",
                       "The {a} of {e} is {v}.",
                       "{e} was issued the {a} {v}.",
                       "According to the register, {e} holds the {a} {v}.",
                       "The clerk entered {v} as the {a} for {e}.",
                       "For {e}, the {a} on file is {v}.",
                       "It was noted that the {a} belonging to {e} is {v}."],
        "questions": ["What is {e}'s {a}?",
                      "What is the {a} of {e}?",
                      "Which {a} was issued to {e}?",
                      "Which {a} does {e} hold?",
                      "What {a} is on file for {e}?",
                      "Give the {a} that belongs to {e}.",
                      "Name the {a} recorded for {e}."]},
    "founding_year": {
        "label": "the founding year of {e}",
        "statements": ["{E} was founded in {v}.",
                       "The founding of {e} dates to {v}.",
                       "{E} was established in the year {v}.",
                       "{E} traces its founding to {v}.",
                       "In {v}, {e} was founded.",
                       "The charter that founded {e} was signed in {v}.",
                       "Chronicles give {v} as the year in which {e} was founded."],
        "questions": ["In what year was {e} founded?",
                      "When was {e} founded?",
                      "What is the founding year of {e}?",
                      "To what year does the founding of {e} date?",
                      "What year saw the founding of {e}?",
                      "The founding of {e} took place in which year?",
                      "Give the year in which {e} was founded."]},
    "colour": {
        "label": "the colour of {p}'s {o}",
        "statements": ["{p}'s {o} is {v}.",
                       "The {o} that {p} owns is {v}.",
                       "{p} keeps {av} {o}.",
                       "The colour of the {o} owned by {p} is {v}."],
        "questions": ["What colour is {p}'s {o}?",
                      "What is the colour of the {o} that {p} owns?",
                      "Of what colour is the {o} kept by {p}?",
                      "Which colour does {p}'s {o} have?"]},
    "material": {
        "label": "the material of {p}'s {o}",
        "statements": ["{p}'s {o} is made of {v}.",
                       "The {o} belonging to {p} was fashioned from {v}.",
                       "{p} owns a {o} made entirely of {v}.",
                       "The maker of {p}'s {o} worked it in {v}."],
        "questions": ["What is {p}'s {o} made of?",
                      "From what material was the {o} belonging to {p} fashioned?",
                      "Which material is {p}'s {o} made of?",
                      "In what material was {p}'s {o} worked?"]},
    "owner": {
        "label": "the owner of the {o} called {n}",
        "statements": ["The {o} called {n} belongs to {v}.",
                       "{v} is the owner of the {o} called {n}.",
                       "The {o} known as {n} is owned by {v}.",
                       "Ownership of the {o} named {n} rests with {v}."],
        "questions": ["Who owns the {o} called {n}?",
                      "To whom does the {o} called {n} belong?",
                      "Who is the owner of the {o} known as {n}?",
                      "With whom does ownership of the {o} named {n} rest?"]},
    "setting": {
        "label": "the value of {k}",
        "statements": ["The setting {k} is set to {v}.",
                       "In the configuration, {k} has the value {v}.",
                       "The parameter {k} was fixed at {v}.",
                       "Configuration note: {k} equals {v}.",
                       "The operators kept {k} at {v}.",
                       "The value stored under the key {k} is {v}.",
                       "For the key {k}, the configured value is {v}."],
        "questions": ["What is the setting {k} set to?",
                      "What value does {k} have in the configuration?",
                      "At what value was the parameter {k} fixed?",
                      "What value is stored under the key {k}?",
                      "What is the configured value for the key {k}?",
                      "At which value was {k} kept?",
                      "Give the value of the setting {k}."]},
    "phrase": {
        "attrs": ["motto", "watchword", "greeting", "toast", "slogan", "refrain"],
        "label": "the {a} of {e}",
        "statements": ["The {a} of {e} is \"{v}\".",
                       "{E} took as its {a} the words \"{v}\".",
                       "Members of {e} use the {a} \"{v}\".",
                       "Carved above the door of {e} is its {a}: \"{v}\".",
                       "\"{v}\" is the {a} of {e}.",
                       "The {a} adopted by {e} reads \"{v}\".",
                       "Everyone in {e} knows the {a} \"{v}\"."],
        "questions": ["What is the {a} of {e}?",
                      "Which words did {e} take as its {a}?",
                      "What {a} do members of {e} use?",
                      "How does the {a} adopted by {e} read?",
                      "Quote the {a} of {e}.",
                      "Which {a} belongs to {e}?",
                      "Give the {a} of {e} word for word."]},
    "date": {
        "attrs": ["wedding", "hearing", "inspection", "investiture", "trial", "farewell dinner", "first lecture",
                  "examination"],
        "label": "the date of the {a} of {e}",
        "statements": ["The {a} of {e} took place on {v}.",
                       "{e}'s {a} was held on {v}.",
                       "On {v}, the {a} of {e} was held.",
                       "The date of {e}'s {a} was {v}.",
                       "The diary records the {a} of {e} under {v}.",
                       "It was on {v} that the {a} of {e} took place.",
                       "The {a} of {e} is dated {v}."],
        "questions": ["On what date did the {a} of {e} take place?",
                      "When was {e}'s {a} held?",
                      "What was the date of {e}'s {a}?",
                      "On which date was the {a} of {e} held?",
                      "The {a} of {e} took place on what date?",
                      "Give the date of the {a} of {e}.",
                      "Which date is given for {e}'s {a}?"]},
    "identifier": {
        "attrs": ["serial", "shelf mark", "registration code", "tag", "catalogue code", "stock code",
                  "reference code"],
        "label": "the {a} of {e}",
        "statements": ["The {a} of {e} is {v}.",
                       "{E} bears the {a} {v}.",
                       "Stamped on {e} is the {a} {v}.",
                       "The inventory lists {e} under the {a} {v}.",
                       "{E} was entered with the {a} {v}.",
                       "The {a} {v} identifies {e}.",
                       "On {e}, the {a} reads {v}."],
        "questions": ["What is the {a} of {e}?",
                      "Which {a} does {e} bear?",
                      "Which {a} is stamped on {e}?",
                      "Under which {a} is {e} listed?",
                      "What {a} identifies {e}?",
                      "What does the {a} on {e} read?",
                      "Give the {a} of {e}."]},
}
KINDS = list(FAMILIES)
# a later statement that replaces an earlier one: {l} the fact's noun phrase, {L} the same starting a sentence
REVISIONS = ["Correction: {l} is now {v}.",
             "{L} was later changed to {v}.",
             "An amendment was entered: {l} is {v}, and the earlier entry no longer stands.",
             "Update: {l} has been revised to {v}.",
             "On review, {l} was corrected to {v}.",
             "The earlier entry was struck out; {l} is {v}.",
             "{L} has since been altered and is now {v}.",
             "A later note revises {l} to {v}."]
_RESTATED = ("person_number", "setting", "identifier", "phrase")  # a plain second statement reads as a replacement


def statement_templates() -> list[str]:
    return sorted({t for f in FAMILIES.values() for t in f["statements"]} | set(REVISIONS))


def question_templates() -> list[str]:
    return sorted({t for f in FAMILIES.values() for t in f["questions"]})


def template_vocabulary() -> list[str]:
    """Every fixed string a planted statement or question can contain."""
    words = [w for lst in (_PLACE_ENDS, _PLACE_KINDS, _GROUPS, _OBJECTS, _COLOURS, _MATERIALS, _KEY_WORDS, _STATES,
                           _ADJ, _NOUNS, _VERBS, _MONTHS) for w in lst]
    words += [w for pair in _THINGS for w in pair] + [a for f in FAMILIES.values() for a in f.get("attrs", [])]
    return words + [f["label"] for f in FAMILIES.values()] + [NOT_STATED]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class _Names:
    """Invented names, assembled from syllables, no name twice in a row of facts."""

    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def word(self, lo: int = 2, hi: int = 3) -> str:
        r = self.rng
        while True:
            n = lo if r.random() < 0.7 else r.randint(lo, hi)
            w = "".join(r.choice(_ONSETS) + r.choice(_VOWELS) + r.choice(_CODAS if i == n - 1 else _INNER)
                        for i in range(n))
            if w not in self.used and len(w) >= 4:
                self.used.add(w)
                return w.capitalize()

    def person(self) -> str:
        return f"{self.word()} {self.word()}"

    def place(self) -> str:
        return self.word(1, 2) + self.rng.choice(_PLACE_ENDS)


@dataclass
class Fact:
    kind: str
    fields: dict
    values: list[str]        # what the text states, in order; the answer is the last ([] when nothing is planted)
    statements: list[str]    # one per value
    question: str
    answer: str
    subject: str             # the invented name that only this fact carries


def _value(kind: str, rng: random.Random, names: _Names) -> str:
    if kind == "person_number":
        return str(rng.randint(10 ** 2, 10 ** rng.randint(3, 6) - 1))
    if kind == "founding_year":
        return str(rng.randint(1050, 1890))
    if kind == "colour":
        return rng.choice(_COLOURS)
    if kind == "material":
        return rng.choice(_MATERIALS)
    if kind == "owner":
        return names.person()
    if kind == "setting":
        return rng.choice([str(rng.randint(10, 9999)), f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}",
                           rng.choice(_STATES)])
    if kind == "phrase":
        pools = {"A": _ADJ, "N": _NOUNS, "V": _VERBS}
        return " ".join(rng.choice(pools[c]) for c in rng.choice(["ANVAN", "NVAN", "ANVN", "NVN"]))
    if kind == "date":
        return f"{rng.randint(1, 28)} {rng.choice(_MONTHS)} {rng.randint(1500, 1899)}"
    while True:  # identifier: 6-10 characters, letters and digits, sometimes one hyphen
        n = rng.randint(6, 10)
        s = "".join(rng.choice(_ID_CHARS) for _ in range(n))
        if rng.random() < 0.3:
            s = s[: n // 2] + "-" + s[n // 2 + 1:]
        if any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
            return s


def _subject(kind: str, rng: random.Random, names: _Names) -> tuple[dict, str]:
    """The fields that name what a fact is about, and the invented name in them."""
    if kind in ("person_number", "date"):
        e = names.person()
        return {"e": e, "E": e}, e
    if kind == "founding_year":
        place = names.place()
        e = f"the {rng.choice(_PLACE_KINDS)} of {place}"
        return {"e": e, "E": _cap(e)}, place
    if kind in ("colour", "material"):
        p = names.person()
        return {"p": p, "o": rng.choice(_OBJECTS)}, p
    if kind == "owner":
        n = names.word()
        return {"o": rng.choice(_OBJECTS), "n": n}, n
    if kind == "setting":
        stem, word = names.word(1, 2).lower(), rng.choice(_KEY_WORDS)
        k = rng.choice(["{}_{}", "{}.{}", "{}-{}"]).format(stem, word)
        return {"k": k}, k
    if kind == "phrase":
        name = names.word()
        e = f"the {name} {rng.choice(_GROUPS)}"
        return {"e": e, "E": _cap(e)}, name
    person = names.person()  # identifier
    thing, rel = rng.choice(_THINGS)
    e = f"the {thing} {rel} {person}"
    return {"e": e, "E": _cap(e)}, person


def _render(tpl: str, f: dict, kind: str, v: str) -> str:
    av = f"{'an' if v[:1] in 'aeiou' else 'a'} {v}" if kind == "colour" else v
    return tpl.format(**f, v=v, av=av)


def make_fact(rng: random.Random, names: _Names, kind: str | None = None, revised: bool = False) -> Fact:
    """One invented fact: its statement(s), a question in other words, the answer."""
    kind = kind or rng.choice(KINDS)
    fam = FAMILIES[kind]
    while True:
        f, subject = _subject(kind, rng, names)
        if fam.get("attrs"):
            f["a"] = rng.choice(fam["attrs"])
        values = [_value(kind, rng, names)]
        statements = [_render(rng.choice(fam["statements"]), f, kind, values[0])]
        if revised:
            while True:
                v2 = _value(kind, rng, names)
                if v2 != values[0]:
                    break
            values.append(v2)
            if kind in _RESTATED and rng.random() < 0.25:
                statements.append(_render(rng.choice(fam["statements"]), f, kind, v2))
            else:
                label = fam["label"].format(**f)
                shown = f'"{v2}"' if kind == "phrase" else v2
                statements.append(rng.choice(REVISIONS).format(l=label, L=_cap(label), v=shown))
        question = rng.choice(fam["questions"]).format(**f)
        if not any(banned_in(s) for s in statements + [question]):
            return Fact(kind, f, values, statements, question, values[-1], subject)


def make_missing(rng: random.Random, names: _Names, planted: list[Fact]) -> Fact:
    """A question about something the text never states: another attribute of a planted subject, or a subject the
    text never names."""
    known = [f for f in planted if f.kind in ("colour", "material") or len(FAMILIES[f.kind].get("attrs", [])) > 1]
    while True:
        if known and rng.random() < 0.5:
            f = rng.choice(known)
            if f.kind in ("colour", "material"):
                kind, fields = ("material" if f.kind == "colour" else "colour"), dict(f.fields)
            else:
                kind = f.kind
                fields = {**f.fields, "a": rng.choice([a for a in FAMILIES[kind]["attrs"] if a != f.fields["a"]])}
            question, subject = rng.choice(FAMILIES[kind]["questions"]).format(**fields), f.subject
        else:
            g = make_fact(rng, names)
            kind, fields, question, subject = g.kind, g.fields, g.question, g.subject
        if not banned_in(question):
            return Fact(kind, fields, [], [], question, NOT_STATED, subject)


def make_row_facts(rng: random.Random, k_range: tuple[int, int]) -> tuple[list[Fact], list[Fact]]:
    """(planted facts, questions about something never planted) for one row."""
    names = _Names(rng)
    planted = [make_fact(rng, names, revised=rng.random() < P_REVISED) for _ in range(rng.randint(*k_range))]
    n_missing = rng.randint(1, 2) if rng.random() < P_MISSING_ROW else 0
    missing = [make_missing(rng, names, planted) for _ in range(n_missing)]
    return planted, missing


# ----------------------------------------------------------------------------------------------------------------
# Lookup: books
# ----------------------------------------------------------------------------------------------------------------
def stable_hash(key: str) -> int:
    """The mix builder's document hash (scripts/build_m21_mix.py `stable_hash`)."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


HELD_OUT_RULES = ["longdocs_lite (the rule of the mix's `books` source): in file order, among the documents with "
                  ">= 40,960 characters and >= 16,384 GPT-2 tokens (tiktoken gpt2), every 100th (number 99, 199, ...)",
                  "hash (the mix's other rule): stable_hash(key) % 1_000_000 < 5_000, or % 200 == 0, with key = the "
                  "document id and with key = the document text"]


def book_held_out(doc_id: str, text: str) -> bool:
    """True when the mix's hash rule would hold the book out as validation (LongdocsVal is its other rule)."""
    return any(h % 200 == 0 or h % 1_000_000 < 5_000 for h in (stable_hash(doc_id), stable_hash(text)))


def gpt2_counter(threads: int | None = None):
    """GPT-2 token counts as scripts/build_longdocs_lite.py takes them (tiktoken "gpt2", encode_ordinary_batch)."""
    import tiktoken

    enc = retry(lambda: tiktoken.get_encoding("gpt2"), "tiktoken gpt2")
    n = max(1, threads or min(16, os.cpu_count() or 1))
    return lambda texts: [len(ids) for ids in enc.encode_ordinary_batch(texts, num_threads=n)]


class LongdocsVal:
    """The validation rule of scripts/build_longdocs_lite.py, as scripts/build_m21_mix.py replays it for its `books`
    source ("heldout": "longdocs_lite"): among the documents in file order, one with at least 2.5 x min_tokens
    characters and at least min_tokens GPT-2 tokens is "kept", and kept number 99, 199, ... is validation. `mark`
    must see every document of the source, in order."""

    def __init__(self, count=None, min_tokens: int = 16384, every: int = 100):
        self.count, self.min_tokens, self.every, self.kept = count, int(min_tokens), int(every), 0

    def mark(self, texts: list[str]) -> list[bool]:
        cand = [i for i, t in enumerate(texts) if not len(t) < 2.5 * self.min_tokens]
        out = [False] * len(texts)
        if cand and self.count is None:
            self.count = gpt2_counter()
        for i, n in zip(cand, self.count([texts[i] for i in cand]) if cand else []):
            if n < self.min_tokens:
                continue
            out[i] = self.kept % self.every == self.every - 1
            self.kept += 1
        return out


def book_bucket(doc_id: str) -> int:
    """0-99: which of this builder's own splits a book belongs to (SETS[...]["buckets"])."""
    return stable_hash(doc_id) // 1_000_000 % 100


class BookSource:
    """Common Pile's Project Gutenberg documents in file order, read only as far as they are consumed. Every document
    passes through the mix's longdocs_lite validation rule, in order, the first time it is read."""

    GROUP = 16  # documents whose GPT-2 tokens are counted together

    def __init__(self, raw_dir: Path, count=None):
        self.info = hub_info(BOOK_REPO)
        self.files = sorted((n, s) for n, s in self.info["sizes"].items() if n.endswith((".json.gz", ".jsonl.gz")))
        if not self.files:
            raise FileNotFoundError(f"{BOOK_REPO}: no .json.gz files")
        self.raw = Path(raw_dir) / f"gutenberg-{self.info['sha'][:12]}"
        self._prefix = {n: HubPrefix(hub_file_url(BOOK_REPO, n, self.info["sha"]), self.raw / n, s)
                        for n, s in self.files}
        self.fetched: dict[str, int] = {}
        self._val, self._marks = LongdocsVal(count), []  # per document of the stream: validation in the mix?

    def _rows(self):
        for name, _ in self.files:
            for row in gz_json_lines(self._prefix[name].chunks()):
                self.fetched[name] = max(self.fetched.get(name, 0), self._prefix[name].have())
                meta = row.get("metadata") or {}
                yield (str(row.get("id", "")), str(meta.get("title") or ""), str(meta.get("license") or ""),
                       row.get("text") or "")

    def books(self):
        """(id, title, licence, text, validation in the mix) of every document, file after file."""
        rows, i = self._rows(), 0
        while True:
            group = list(itertools.islice(rows, self.GROUP))
            if not group:
                return
            if i + len(group) > len(self._marks):  # this part of the stream is read for the first time
                self._marks += self._val.mark([r[3] for r in group[len(self._marks) - i:]])
            for r in group:
                yield (*r, self._marks[i])
                i += 1

    def describe(self) -> dict:
        return {"repo": BOOK_REPO, "revision": self.info["sha"], "license_tag": self.info["license_tag"],
                "license_note": BOOK_LICENSE_NOTE,
                "files": [{"file": n, "bytes_total": s, "bytes_fetched": self.fetched[n]} for n, s in self.files
                          if n in self.fetched]}

    def cleanup(self) -> None:
        shutil.rmtree(self.raw, ignore_errors=True)


_WS = " \t\r\n"
_BLANK = re.compile(r"\n[ \t\r]*\n[ \t\r\n]*")                   # a blank line: a paragraph starts at its end
_GAP = re.compile(r"[ \t\r\n]+")
_WORD_END = re.compile(r"(?<=\S)(?=[ \t\r\n])")
_PARA_END = re.compile(r"(?<=\S)(?=\r?\n[ \t\r]*\n)")
_SENT_END = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]]?(?=[ \t\r\n]+[\"'\u201c\u2018(\[]?[A-Z])")
_ABBREV = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Messrs|Dr|St|Rev|Hon|Col|Capt|Gen|Lieut|Lt|Sr|Jr|Prof|No|Nos|Vol|Esq|Mme|"
                     r"Mlle|vs|etc|viz|cf|pp|ch|Fig|Art)|\b[A-Za-z])\.[\"'\u201d\u2019)\]]?$")


def _cut(starts: list[int], ends: list[int], c: int) -> int | None:
    """The token index i such that tokens [0, i) end exactly at character c and token i starts at or after it."""
    i = bisect.bisect_right(ends, c)
    if i == 0 or ends[i - 1] != c or (i < len(starts) and starts[i] < c):
        return None
    return i


def _sentence_end(text: str, c: int) -> bool:
    if c < 2 or (c < len(text) and text[c] not in _WS):
        return False
    return text[c - 1] in ".!?" or (text[c - 1] in "\"'\u201d\u2019)]" and text[c - 2] in ".!?")


def _choose_span(text: str, starts: list[int], ends: list[int], lo: int, hi: int, budget: int, rng: random.Random,
                 tok: Tok):
    """About `budget` tokens of the book inside tokens [lo, hi): from the first word of a paragraph to the end of a
    sentence. Returns (head text, head ids, t0, t1): the span's text is head + text[ends[t0 - 1]: ends[t1 - 1]], its
    tokens head ids + ids[t0: t1], at most `budget` of them. None when the book has no usable boundary there."""
    n, slack = len(starts), hi - lo - budget
    if slack < 0:
        return None
    margin = min(1024, slack // 2)
    t = lo + rng.randrange(slack - margin + 1)
    c, c_max = starts[t], starts[min(n - 1, t + margin)]
    m = _BLANK.search(text, c, c_max) or _GAP.search(text, c, c_max)  # the next paragraph, else the next word
    c0 = m.end() if m else None
    if c0 is None or c0 >= c_max:
        return None
    t0, pos = None, c0
    for _ in range(8):  # the head: the first word(s), up to a white space that is a token boundary
        w = _WORD_END.search(text, pos + 1)
        if not w:
            return None
        t0, pos = _cut(starts, ends, w.start()), w.start()
        if t0 is not None:
            break
    if t0 is None:
        return None
    head = text[c0:pos]
    head_ids = tok.encode(head)
    top = min(hi, t0 + budget - len(head_ids))  # inside [lo, hi): the spans of a book never overlap
    floor = max(t0 + 1, top - 64)
    for sentence in (True, False):  # the end of a sentence if one is near, else the end of a word
        for u in range(top, floor - 1, -1):
            e = ends[u - 1]
            ok = _sentence_end(text, e) if sentence else (
                e > 0 and text[e - 1] not in _WS and (e == len(text) or text[e] in _WS))
            if ok and (u == n or starts[u] >= e):
                return head, head_ids, t0, u
    return None


def _boundaries(text: str, starts: list[int], ends: list[int], a: int, b: int) -> tuple[list[int], list[int]]:
    """Token indices inside characters (a, b) where a statement can be planted: after a paragraph, after a sentence."""
    para, sent = [], []
    for m in _PARA_END.finditer(text, a + 1, b):
        i = _cut(starts, ends, m.start())
        if i is not None:
            para.append(i)
    for m in _SENT_END.finditer(text, a + 1, b):
        if _ABBREV.search(text, max(a, m.end() - 10), m.end()):
            continue
        i = _cut(starts, ends, m.end())
        if i is not None:
            sent.append(i)
    return para, sent


def _place(kinds: list[str], para: list[int], sent: list[int], t0: int, t1: int, rng: random.Random):
    """A distinct token index for each statement: a uniform random depth in the span, moved to the nearest free
    boundary of the statement's kind (a paragraph statement after a paragraph, a sentence statement after a
    sentence; the other list when its own is empty)."""
    taken, out = set(), []
    for kind in kinds:
        cands = (para or sent) if kind == "para" else (sent or para)
        j = bisect.bisect_left(cands, rng.uniform(t0, t1))
        pick = next((cands[i] for d in range(len(cands)) for i in (j + d, j - d - 1)
                     if 0 <= i < len(cands) and cands[i] not in taken), None)
        if pick is None:
            return None
        taken.add(pick)
        out.append(pick)
    return out


def lookup_rows(doc_id: str, text: str, ids: list[int], offsets: list[tuple[int, int]], tok: Tok, cfg: dict,
                seed_key: str) -> list[Row]:
    """The rows of one book: up to cfg["spans_per_book"] spans that do not overlap, each with its own facts."""
    S, n = cfg["seq_len"], len(ids)
    if n < S or tok.eot in ids:
        return []
    rng = random.Random(f"{seed_key}|{doc_id}")
    starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]
    arr = np.asarray(ids, dtype=np.uint16)
    final = tok.encode("\n\n")
    n_spans = min(cfg["spans_per_book"], n // S)
    seg, rows = n // n_spans, []
    for j in range(n_spans):
        planted, missing = make_row_facts(rng, cfg["facts"])
        stmts = [(fi, si) for fi, f in enumerate(planted) for si in range(len(f.statements))]
        kinds = [rng.choice(("para", "sent")) for _ in stmts]
        pieces = [("\n\n" if k == "para" else " ") + planted[fi].statements[si] for (fi, si), k in zip(stmts, kinds)]
        asked = planted + missing
        rng.shuffle(asked)
        q_pieces = [f"\n\nQuestion: {f.question}\nAnswer:" for f in asked]
        enc = tok.encode_batch(pieces + q_pieces + [" " + f.answer for f in asked])
        s_ids, q_ids, a_ids = enc[: len(pieces)], enc[len(pieces): len(pieces) + len(asked)], enc[-len(asked):]
        budget = S - 1 - len(final) - sum(len(x) for x in enc)
        span = _choose_span(text, starts, ends, j * seg, (j + 1) * seg, budget, rng, tok)
        if span is None:
            continue
        head, head_ids, t0, t1 = span
        para, sent = _boundaries(text, starts, ends, ends[t0 - 1], ends[t1 - 1])
        at = _place(kinds, para, sent, t0, t1, rng)
        if at is None:
            continue
        where = dict(zip(stmts, range(len(stmts))))
        for fi, f in enumerate(planted):  # a revision comes after the statement it revises
            if len(f.statements) == 2 and at[where[fi, 0]] > at[where[fi, 1]]:
                at[where[fi, 0]], at[where[fi, 1]] = at[where[fi, 1]], at[where[fi, 0]]
        parts, texts, pos, cur, s_end = [head_ids], [head], 1 + len(head_ids), t0, {}
        for t_ins, k in sorted(zip(at, range(len(stmts)))):
            parts += [arr[cur:t_ins], s_ids[k]]
            texts += [text[ends[cur - 1]: ends[t_ins - 1]], pieces[k]]
            pos += t_ins - cur + len(s_ids[k])
            s_end[stmts[k]] = pos
            cur = t_ins
        parts.append(arr[cur:t1])
        texts.append(text[ends[cur - 1]: ends[t1 - 1]])
        pos += t1 - cur
        q_char = sum(len(x) for x in texts)
        index = {id(f): fi for fi, f in enumerate(planted)}
        qa = []
        for f, q, a, qp in zip(asked, q_ids, a_ids, q_pieces):
            parts += [q, a]
            texts += [qp, " " + f.answer]
            ans, fi = pos + len(q), index.get(id(f))
            last = None if fi is None else s_end[fi, len(f.statements) - 1]
            qa.append({"question": f.question, "answer": f.answer, "answer_pos": ans, "kind": f.kind,
                       "attr": f.fields.get("a"),
                       "distance": None if last is None else ans - last, "values": list(f.values),
                       "statements": list(f.statements), "subject": f.subject,
                       "statement_ends": [] if fi is None else [s_end[fi, si] for si in range(len(f.statements))]})
            pos = ans + len(a)
        parts.append(final)
        texts.append("\n\n")
        info = {"facts": len(planted), "statements": len(stmts), "not_stated": len(missing),
                "revised": sum(len(f.statements) == 2 for f in planted), "span_tokens": len(head_ids) + t1 - t0,
                **{f"kind_{k}": sum(f.kind == k for f in planted) for k in KINDS}}
        row = finish_row(tok, S, parts, texts, q_char, qa, info)
        row.info["book"] = doc_id
        rows.append(row)
    return rows


def build_lookup_set(name: str, cfg: dict, source, tok: Tok, out: Path, seed: int, rows: dict,
                     shard_tokens: int) -> dict:
    S, splits = cfg["seq_len"], ("val", "train")
    writers = {s: ShardWriter(out, f"{name}_{s}", S, _shard_cap(s, shard_tokens, rows, S)) for s in splits}
    stats = {s: SplitStats(tok, S) for s in splits}
    used = {s: [] for s in splits}
    c: collections.Counter = collections.Counter()
    batch, chars, last = [], 0, [0.0]

    def full(split: str) -> bool:
        return writers[split].rows >= rows[split]

    def flush() -> None:
        nonlocal batch, chars
        for (split, doc_id, text), (ids, offsets) in zip(batch, tok.encode_offsets([b[2] for b in batch])):
            if full(split):
                continue
            made = lookup_rows(doc_id, text, ids, offsets, tok, cfg, f"{seed}|{name}")
            c["books_under_seq_len" if len(ids) < S else "books_used" if made else "books_without_rows"] += 1
            for row in made[: rows[split] - writers[split].rows]:
                writers[split].add(row.ids)
                stats[split].add(row)
                if not used[split] or used[split][-1] != doc_id:
                    used[split].append(doc_id)
        batch, chars = [], 0
        if time.time() - last[0] > 20:
            last[0] = time.time()
            log(f"[{name}] books seen {c['books_seen']:,}; rows " +
                ", ".join(f"{s} {writers[s].rows:,}/{rows[s]:,}" for s in splits))

    for doc_id, _title, licence, text, mix_val in source.books():
        if all(full(s) for s in splits):
            break
        c["books_seen"] += 1
        if mix_val or book_held_out(doc_id, text):  # validation in the training mix: never used here
            c["books_held_out_longdocs_rule" if mix_val else "books_held_out_hash_rule"] += 1
            continue
        if "public domain" not in licence.lower():
            c["books_other_licence"] += 1
            continue
        b = book_bucket(doc_id)
        split = next((s for s in splits if cfg["buckets"][s][0] <= b < cfg["buckets"][s][1]), None)
        if split is None or full(split):
            continue
        if len(text) < 2 * S:  # under two characters per token: it cannot hold seq_len tokens of prose
            c["books_under_seq_len"] += 1
            continue
        batch.append((split, doc_id, text))
        chars += len(text)
        if len(batch) >= 8 or chars >= 4_000_000:
            flush()
    flush()
    res = {}
    for s in splits:
        writers[s].close()
        res[s] = {**stats[s].summary(writers[s]), "target_rows": rows[s], "target_met": full(s),
                  "books": len(used[s]), "book_ids": used[s]}
        if not full(s):
            log(f"[{name}] !! {s}: {writers[s].rows:,} of {rows[s]:,} rows: the books ran out")
    res["books"] = dict(sorted(c.items()))
    return res


# ----------------------------------------------------------------------------------------------------------------
# The build
# ----------------------------------------------------------------------------------------------------------------
def target_rows(cfg: dict, scale: float) -> int:
    return max(1, round(cfg["tokens"] * scale / cfg["seq_len"]))


def run(args, wiki=None, books=None, tok: Tok | None = None, sets: dict | None = None) -> dict:
    """Build the sets that have no done.json yet. `wiki`, `books`, `tok`, `sets`: stand-ins for the tests."""
    sets = sets or SETS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in (args.only.split(",") if args.only else list(sets)) if n]
    unknown = [n for n in names if n not in sets]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; known: {list(sets)}")
    todo = [n for n in names if not (out / f"{n}.done.json").exists()]
    for n in names:
        if n not in todo:
            log(f"[{n}] done.json exists - skipped")
    if not todo:
        return {"built": [], "skipped": names}
    tok = tok or load_tokenizer(args.tokenizer)
    raw = out / RAW_DIR
    rows = {n: {"train": target_rows(sets[n], args.scale), "val": int(args.val_rows)} for n in todo}
    for k, name in enumerate(todo):
        cfg, t0 = sets[name], time.time()
        for stale in list(out.glob(f"{name}_train_*.bin*")) + list(out.glob(f"{name}_val_*.bin*")):
            stale.unlink()  # a set without done.json is rebuilt from scratch
        log(f"[{name}] {rows[name]['train']:,} train + {rows[name]['val']} val rows of {cfg['seq_len']:,} tokens")
        if cfg["kind"] == "multihop":
            if wiki is None:
                need = max(rows[n]["train"] * sets[n]["seq_len"] for n in todo if sets[n]["kind"] == "multihop")
                wiki = WikiSource(raw, args.seed, whole_files=need / 900 / 1000 >= 40)
            res, source = build_multihop_set(name, cfg, wiki, tok, out, args.seed, rows[name], args.shard_tokens), wiki
        else:
            books = books or BookSource(raw)
            res, source = build_lookup_set(name, cfg, books, tok, out, args.seed, rows[name], args.shard_tokens), books
        tr, va, sec = res["train"], res["val"], time.time() - t0
        bad = tr["joint_check"]["mismatched"] + va["joint_check"]["mismatched"]
        if bad:
            log(f"[{name}] !! {bad} checked row(s) are not the tokenizer's own tokenization of their text: this "
                f"tokenizer splits text differently at piece boundaries (the rows are valid, their ids not canonical)")
        n_rows = tr["n_rows"] + va["n_rows"]
        speed = {"rows_per_s": round(n_rows / sec, 2), "tokens_per_s": round(n_rows * cfg["seq_len"] / sec),
                 "of": "train and val rows over the set's wall time, downloads included"}
        params = {"seed": args.seed, "scale": args.scale, "target_tokens": cfg["tokens"] * args.scale,
                  "val_rows": int(args.val_rows), "shard_tokens": int(args.shard_tokens), "min_back": MIN_BACK,
                  "min_fill_at_pass_end": MIN_FILL, "not_stated_answer": NOT_STATED,
                  **{k_: v for k_, v in cfg.items() if k_ not in ("kind", "seq_len", "tokens")}}
        if cfg["kind"] == "lookup":
            params.update({"p_revised_fact": P_REVISED, "p_row_with_unplanted_questions": P_MISSING_ROW,
                           "fact_kinds": KINDS, "statement_templates": len(statement_templates()),
                           "question_templates": len(question_templates()), "banned_strings_checked": BANNED,
                           "banned_strings_checked_case_sensitive": BANNED_EXACT,
                           "banned_check": "every template and vocabulary list (tests); every planted statement "
                                           "and question at build time (one that contains a banned string is redrawn)",
                           "books_skipped_when": HELD_OUT_RULES,
                           "book_bucket": "stable_hash(id) // 1_000_000 % 100, see `buckets`"})
        else:
            params.update({"val_row_groups": WIKI_VAL_GROUPS, "row_text": "Title: <title>\\n<sentences>\\n\\n ... "
                           "Question: ...\\nEvidence: ...\\nAnswer: ...\\n\\n"})
        done = {"name": name, "kind": cfg["kind"], "seq_len": cfg["seq_len"], "eot": tok.eot, "tokenizer": tok.name,
                "vocab": tok.vocab, "rule": "stream", "rows": tr["n_rows"], "tokens": tr["tokens"],
                "shards": [x["file"] for x in tr["shards"]], "val_shards": [x["file"] for x in va["shards"]],
                "scale": args.scale, "target_tokens": cfg["tokens"] * args.scale,
                "text_bytes": tr["text_bytes"], "items_per_row": tr["items_per_row"],
                "answer_distance": tr["answer_distance"],
                "answer_distance_is": "tokens from the end of the text an answer needs (multihop: the supporting "
                                      "paragraph that is farthest back; lookup: the latest statement of the fact) "
                                      "to the first token of the answer",
                "seconds": round(sec, 2), "speed": speed, "train": tr, "val": va, "parameters": params,
                "hub": source.describe(), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        done.update({k_: v for k_, v in res.items() if k_ not in ("train", "val")})
        tmp = out / f"{name}.done.json.tmp"
        tmp.write_text(json.dumps(done, indent=2))
        os.replace(tmp, out / f"{name}.done.json")
        log(f"[{name}] done: {tr['n_rows']:,} train rows, {tr['tokens']:,} tokens, {sec:.0f} s "
            f"({speed['rows_per_s']} rows/s, {speed['tokens_per_s']:,} tokens/s with the val rows and the downloads)")
        left = {sets[n]["kind"] for n in todo[k + 1:]}  # raw files no later set needs go now
        if wiki is not None and "multihop" not in left:
            wiki.cleanup()
        if books is not None and "lookup" not in left:
            books.cleanup()
    shutil.rmtree(raw, ignore_errors=True)
    return {"built": todo, "skipped": [n for n in names if n not in todo]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory of the shard sets")
    ap.add_argument("--only", default="", help=f"comma-separated subset of {','.join(SETS)}")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each set's token target (0.002: tiny)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M", help="Hub id, or a folder with tokenizer.json")
    ap.add_argument("--val_rows", type=int, default=VAL_ROWS)
    ap.add_argument("--shard_tokens", type=float, default=SHARD_TOKENS)
    args = ap.parse_args(argv)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # anonymous: no stored token is ever sent
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    run(args)


if __name__ == "__main__":
    main()
    sys.stdout.flush()

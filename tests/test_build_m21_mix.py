"""scripts/build_m21_mix.py: the E38 mix builder, on tiny local parquet / JSON-lines fixtures with a fake tokenizer
(no network). The shards are read back through data.bin_token_dataset.BinRowDataset, the trainer's reader."""
import gzip
import json
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.build_longdocs_lite as legacy
import scripts.build_m21_mix as m21
from data.bin_token_dataset import BinRowDataset, open_shard

EOT = 0
SPECIALS = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}


class FakeTok:
    """One id per UTF-8 byte (byte + 3); ids 0-2 are <|endoftext|>, <|im_start|>, <|im_end|> (chat mode only)."""

    name, eot, vocab_size = "fake", EOT, 259

    def __init__(self):
        self.seen: list[str] = []

    def _encode(self, text: str, special: bool) -> np.ndarray:
        if not special:
            return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + 3
        out, i = [], 0
        while i < len(text):
            for s, k in SPECIALS.items():
                if text.startswith(s, i):
                    out.append(k)
                    i += len(s)
                    break
            else:
                out.extend(b + 3 for b in text[i].encode("utf-8"))
                i += 1
        return np.asarray(out, dtype=np.uint16)

    def encode_batch(self, texts, special=False):
        self.seen.extend(texts)
        return [self._encode(t, special) for t in texts]

    def render_chat(self, messages):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def decode(self, ids) -> str:
        inv = {v: k for k, v in SPECIALS.items()}
        out = b""
        for i in ids:
            out += inv[int(i)].encode() if int(i) < 3 else bytes([int(i) - 3])
        return out.decode("utf-8")

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


def text_of(i: int, n: int) -> str:
    """A document of exactly n ASCII characters (= n fake tokens) that starts with its number."""
    head = f"doc{i:05d} "
    return (head + "abcdefghij " * (n // 11 + 1))[:n]


def write_parquet(path: Path, rows: list[dict], row_group: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group)


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(tmp_path: Path, sources: list[dict], phases: dict | None = None, out: str = "out", tok=None,
          argv: tuple = (), gpt2_count=None, **recipe_extra):
    recipe = {"name": "tiny", "eot": EOT, "vocab_size": 259, "tokenizer": "fake", "sources": sources,
              "phases": phases or {}, **recipe_extra}
    rp = tmp_path / f"{out}_recipe.json"
    rp.write_text(json.dumps(recipe))
    tok = tok or FakeTok()
    args = m21.parse_args(["--recipe", str(rp), "--out", str(tmp_path / out), "--tokenizer", str(tmp_path), *argv])
    rc = m21.run(args, tok=tok, raw=m21.Raw(tmp_path / out / "_raw"), gpt2_count=gpt2_count)
    return rc, tmp_path / out, tok


def docs_in(path: Path) -> list[np.ndarray]:
    """The documents of a shard: each starts with EOT."""
    sh = np.asarray(open_shard(str(path)))
    starts = np.flatnonzero(sh == EOT)
    assert len(sh) == 0 or starts[0] == 0
    return [sh[a:b] for a, b in zip(starts, np.append(starts[1:], len(sh)))]


def split_texts(out: Path, name: str, split: str) -> list[str]:
    tok = FakeTok()
    return [tok.decode(d[1:]) for p in sorted(out.glob(f"{name}_{split}_*.bin")) for d in docs_in(p)]


def done_of(out: Path, name: str) -> dict:
    return json.loads((out / f"{name}.done.json").read_text())


def web_source(tmp_path: Path, n: int = 120, name: str = "web", **extra) -> tuple[dict, dict[str, str]]:
    rng = np.random.default_rng(1)
    rows = [{"id": f"id{i}", "text": text_of(i, int(rng.integers(20, 200))), "score": int(i % 5)} for i in range(n)]
    write_parquet(tmp_path / "raw" / name / "data" / "a-000.parquet", rows[: n // 2])
    write_parquet(tmp_path / "raw" / name / "data" / "a-001.parquet", rows[n // 2:])
    spec = {"name": name, "kind": "docs", "local": str(tmp_path / "raw" / name), "files": "data/*.parquet",
            "id_column": "id", "target_tokens": 10 ** 9, "val_fraction": 0.2, **extra}
    return spec, {r["id"]: r["text"] for r in rows}


# ---------------------------------------------------------------------------------------------------------------
def test_shards_roundtrip_through_bin_row_dataset_both_rules(tmp_path):
    spec, texts = web_source(tmp_path, filter={"column": "score", "min": 3})
    phases = {"p16": {"seq_len": 16, "rule": "stream", "tokens": 160, "train": [{"name": "web", "weight": 1.0}]},
              "p32": {"seq_len": 32, "rule": "doc", "tokens": 320, "train": [{"name": "web", "weight": 1.0}]}}
    rc, out, _ = build(tmp_path, [spec], phases)  # rows are counted for the seq_len of every phase
    assert rc == 0
    kept = {t for i, t in enumerate(texts.values()) if i % 5 >= 3}  # the row filter
    train, val = split_texts(out, "web", "train"), split_texts(out, "web", "val")
    assert set(train) | set(val) == kept and len(train) + len(val) == len(kept)
    done = done_of(out, "web")
    assert done["counters"]["filtered"] == len(texts) - len(kept)
    files = [str(out / f) for f in done["shards"]]
    stream = np.concatenate([np.asarray(open_shard(f)) for f in files])
    for rule, S in (("stream", 16), ("doc", 32)):
        ds = BinRowDataset([{"name": "web", "files": files}], seq_len=S, rule=rule, eot=EOT, seed=0)
        assert len(ds) == done["train"]["rows"][rule][str(S)] == m21.count_rows([Path(f) for f in files], rule, S, EOT)
        whole = stream.tobytes()
        for i in range(len(ds)):
            row = ds[i]
            assert row["input_ids"].numpy().astype(np.uint16).tobytes() in whole
            if rule == "doc":  # a row is a span of ONE document, from its <|endoftext|>
                assert row["doc_ids"].max().item() == 0
    assert done["train"]["text_bytes"] == sum(len(t.encode()) for t in train)
    assert done["train"]["tokens"] == sum(len(t) + 1 for t in train) == done["tokens"]


def test_eot_starts_every_document_and_is_nowhere_else(tmp_path):
    rows = [{"id": str(i), "text": text_of(i, 40) + " <|endoftext|> literal"} for i in range(30)]
    write_parquet(tmp_path / "raw" / "w" / "x.parquet", rows)
    chats = [{"messages": [{"role": "user", "content": f"hello {i} <|endoftext|> there"},
                           {"role": "assistant", "content": "answer " * 5}]} for i in range(30)]
    write_parquet(tmp_path / "raw" / "c" / "train-0.parquet", chats)
    sources = [{"name": "w", "kind": "docs", "local": str(tmp_path / "raw" / "w"), "files": "*.parquet",
                "id_column": "id", "target_tokens": 10 ** 9},
               {"name": "c", "kind": "chat", "local": str(tmp_path / "raw" / "c"), "files": "train-*.parquet",
                "target_tokens": 10 ** 9}]
    rc, out, _ = build(tmp_path, sources)
    assert rc == 0
    for name in ("w", "c"):
        done = done_of(out, name)
        for split in ("train", "val"):
            for sh in done[split]["shards"]:
                arr = np.asarray(open_shard(str(out / sh["file"])))
                assert arr[0] == EOT and int((arr == EOT).sum()) == sh["documents"]
    assert "<|endoftext|> literal" in split_texts(out, "w", "train")[0]  # ordinary text in a document
    chat = split_texts(out, "c", "train")[0]
    assert chat.startswith("<|im_start|>user\nhello ") and chat.endswith("<|im_end|>\n") and "endoftext" not in chat
    assert done_of(out, "c")["counters"]["eot_inside_removed"] == 30


def test_min_and_max_token_filter_and_the_character_prefilter(tmp_path):
    lengths = [10, 49, 50, 51, 99, 100, 101, 150, 200, 201, 400]
    rows = [{"id": str(i), "text": text_of(i, n)} for i, n in enumerate(lengths)]
    write_parquet(tmp_path / "raw" / "d" / "x.parquet", rows)
    spec = {"name": "d", "kind": "docs", "local": str(tmp_path / "raw" / "d"), "files": "*.parquet",
            "id_column": "id", "target_tokens": 10 ** 9, "min_tokens": 50, "max_tokens": 200, "val_fraction": 0.0}
    rc, out, tok = build(tmp_path, [spec])
    assert rc == 0
    # default pre-filter: a text shorter than min_tokens x 2 characters is never tokenized
    assert sorted(len(t) for t in tok.seen) == [100, 101, 150, 200, 201, 400]
    assert sorted(len(t) for t in split_texts(out, "d", "train") + split_texts(out, "d", "val")) == [100, 101, 150, 200]
    c = done_of(out, "d")["counters"]
    assert c["short_chars"] == 5 and c["above_max_tokens"] == 2

    spec2 = {**spec, "name": "e", "min_chars_per_token": 0}
    rc, out, _ = build(tmp_path, [spec2], out="out2")
    assert sorted(len(t) for t in split_texts(out, "e", "train") + split_texts(out, "e", "val")) == \
        [50, 51, 99, 100, 101, 150, 200]
    assert done_of(out, "e")["counters"]["below_min_tokens"] == 2


def test_heldout_split_is_deterministic_and_disjoint(tmp_path):
    spec, texts = web_source(tmp_path, n=300)
    rc, out_a, _ = build(tmp_path, [spec], out="a")
    rc, out_b, _ = build(tmp_path, [spec], out="b")
    for p in sorted(out_a.glob("web_*.bin")):
        assert p.read_bytes() == (out_b / p.name).read_bytes()
    rule = m21.HashHeldout(0.2)
    expect_val = {t for k, t in texts.items() if rule.is_val(k)}
    train, val = set(split_texts(out_a, "web", "train")), set(split_texts(out_a, "web", "val"))
    assert val == expect_val and 30 < len(val) < 90
    assert not train & val and train | val == set(texts.values())

    # a full val shard: later held-out documents are dropped, never trained on
    capped = {**spec, "val_tokens": 400}
    rc, out_c, _ = build(tmp_path, [capped], out="c")
    train_c, val_c = set(split_texts(out_c, "web", "train")), set(split_texts(out_c, "web", "val"))
    done = done_of(out_c, "web")
    assert val_c < expect_val and not train_c & expect_val
    assert done["heldout"]["dropped_documents"] == len(expect_val) - len(val_c) > 0
    assert 400 <= done["val"]["tokens"] < 400 + 201


def test_longdocs_lite_rule_keeps_the_legacy_validation_documents(tmp_path, monkeypatch):
    """The five Common Pile sources: the documents build_longdocs_lite.py held out stay held out."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(700):
        words = int(rng.integers(5, 60))  # the fake GPT-2 count is the number of words
        rows.append({"id": str(i), "text": f"{i} " + " ".join("w" * int(rng.integers(1, 9)) for _ in range(words))})
    write_jsonl_gz(tmp_path / "raw" / "lp" / "p-0000.json.gz", rows[:400])
    write_jsonl_gz(tmp_path / "raw" / "lp" / "p-0001.json.gz", rows[400:])

    class Enc:  # tiktoken's interface: one id per word; the first is the document's number
        @staticmethod
        def encode_ordinary_batch(texts, num_threads=1):
            return [[int(t.split()[0])] + [7] * (len(t.split()) - 1) for t in texts]

    # the legacy builder itself, on the same rows in the same order
    monkeypatch.setattr(legacy, "iter_docs", lambda repo: ((r["id"], r["text"]) for r in rows))
    (tmp_path / "legacy").mkdir()
    largs = types.SimpleNamespace(shard_tokens=10 ** 9, tokens_per_source=10 ** 9, min_tokens=30, batch_docs=64,
                                  threads=1)
    legacy.build_source("lp", "repo", tmp_path / "legacy", largs, Enc)

    def legacy_docs(split):
        sh = np.asarray(open_shard(str(tmp_path / "legacy" / f"lp_{split}_000000.bin")))
        return {int(sh[i + 1]) for i in np.flatnonzero(sh == legacy.EOT)}

    legacy_val, legacy_train = legacy_docs("val"), legacy_docs("train")
    assert len(legacy_val) >= 3

    spec = {"name": "lp", "kind": "docs", "local": str(tmp_path / "raw" / "lp"), "files": "*.json.gz",
            "id_column": "id", "target_tokens": 10 ** 9, "min_tokens": 60, "heldout": "longdocs_lite",
            "heldout_min_tokens": 30, "val_min_docs": 0}
    rc, out, _ = build(tmp_path, [spec],
                       gpt2_count=lambda texts: [len(x) for x in Enc.encode_ordinary_batch(texts)])
    assert rc == 0
    train = {int(t.split()[0]) for t in split_texts(out, "lp", "train")}
    val = {int(t.split()[0]) for t in split_texts(out, "lp", "val")}
    kept_here = {int(r["id"]) for r in rows if len(r["text"]) >= 120}  # 60 fake tokens, 2 characters per token
    assert val == legacy_val & kept_here and val
    assert not train & legacy_val and train == kept_here - legacy_val
    assert train - legacy_train  # documents the legacy build dropped (too few GPT-2 tokens) may be trained on


def code_rows() -> list[dict]:
    def row(repo, path, n, lang="Python"):
        return {"id": f"{repo}:{path}", "text": text_of(len(path), n),
                "metadata": {"repo_name": repo, "path": path, "language": lang}}

    rows = [row("org/big", f"/src/m{i:02d}.py", 90) for i in range(12)]          # several packs
    rows += [row("org/mid", "/b.py", 80), row("org/mid", "/a.py", 80), row("org/mid", "/c/z.py", 60)]
    rows += [row("org/tiny", "/x.py", 30)]                                       # below min_tokens
    rows += [row("org/huge", "/gen.py", 900), row("org/huge", "/ok1.py", 120), row("org/huge", "/ok2.py", 120)]
    rows += [row("org/other", "/a.js", 300, "JavaScript")]                       # filtered by language
    rng = np.random.default_rng(0)
    return [rows[i] for i in rng.permutation(len(rows))]  # a repository's files are scattered


def test_repo_pack_groups_files_by_repository(tmp_path):
    rows = code_rows()
    write_jsonl_gz(tmp_path / "raw" / "code" / "s-0000.json.gz", rows[:10])
    write_jsonl_gz(tmp_path / "raw" / "code" / "s-0001.json.gz", rows[10:])
    spec = {"name": "repos", "kind": "repo_pack", "local": str(tmp_path / "raw" / "code"), "files": "s-*.json.gz",
            "filter": {"column": "metadata.language", "in": ["Python"]}, "min_tokens": 200, "max_tokens": 400,
            "target_tokens": 10 ** 9, "val_fraction": 0.0, "min_chars_per_token": 1.0}
    rc, out, _ = build(tmp_path, [spec])
    assert rc == 0
    packs = split_texts(out, "repos", "train") + split_texts(out, "repos", "val")
    by_repo: dict[str, list[str]] = {}
    for p in packs:
        assert p.startswith("# Repository: org/") and 200 <= len(p) <= 400
        by_repo.setdefault(p.split("\n")[0].removeprefix("# Repository: "), []).append(p)
    assert set(by_repo) == {"org/big", "org/mid", "org/huge"}
    by_text = {r["text"]: r for r in rows}
    for repo, ps in by_repo.items():
        paths = []
        for p in ps:
            head, *files = p.split("## File: ")
            assert head == f"# Repository: {repo}\n\n"
            for f in files:
                path, body = f.split("\n", 1)
                assert body.endswith("\n\n") and by_text[body[:-2]]["metadata"]["repo_name"] == repo
                paths.append(path)
        assert paths == sorted(paths)  # files in path order, also across the packs of one repository
    assert len(by_repo["org/big"]) >= 3  # max_tokens starts a new pack of the same repository
    assert [p.split("## File: ")[1].split("\n")[0] for p in by_repo["org/mid"]] == ["a.py"]
    assert "b.py" in by_repo["org/mid"][0] and "c/z.py" in by_repo["org/mid"][0]
    assert all("gen.py" not in p for p in packs)  # one file above max_tokens is left out
    c = done_of(out, "repos")["counters"]
    assert c["files_above_max_tokens"] == 1 and c["repos_short_chars"] == 1 and c["filtered"] == 1
    assert c["packs"] == len(packs)

    # the held-out rule hashes the repository name: every pack of a repository lands in the same split
    kept = ("org/big", "org/mid", "org/huge")
    frac = next(f for f in np.linspace(0.05, 0.95, 19) if 0 < sum(m21.HashHeldout(f).is_val(r) for r in kept) < 3)
    held = {r for r in kept if m21.HashHeldout(frac).is_val(r)}
    spec_val = {**spec, "name": "repos2", "val_fraction": float(frac)}
    rc, out, _ = build(tmp_path, [spec_val], out="out2")
    tr = {p.split("\n")[0].removeprefix("# Repository: ") for p in split_texts(out, "repos2", "train")}
    va = {p.split("\n")[0].removeprefix("# Repository: ") for p in split_texts(out, "repos2", "val")}
    assert va == held and tr == set(kept) - held


def test_target_tokens_cut_off_and_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(m21, "BATCH_DOCS", 16)
    spec, _ = web_source(tmp_path, n=400, val_fraction=0.05)
    spec["target_tokens"] = 5000
    rc, out, _ = build(tmp_path, [spec])
    done = done_of(out, "web")
    last = len(docs_in(out / done["shards"][-1])[-1])
    assert done["tokens"] >= 5000 > done["tokens"] - last and not done["short"]
    # reading stopped inside the first raw file; the second was never opened
    assert [(r["path"], r["complete"]) for r in done["raw_files"]] == [("data/a-000.parquet", False)]
    assert done["raw_files"][0]["rows"] < 200

    # the target falls exactly at the end of a raw file's rows: the next file is still not opened
    monkeypatch.setattr(m21, "BATCH_DOCS", 512)
    rc, out, _ = build(tmp_path, [spec], out="out1")
    assert [(r["path"], r["complete"]) for r in done_of(out, "web")["raw_files"]] == [("data/a-000.parquet", True)]

    # two parts, 30 % and 70 %: the first raw file is read until 30 % of the target, then the second
    parts = {**spec, "name": "web", "parts": [{"share": 0.3, "files": "data/a-000.parquet"},
                                              {"share": 0.7, "files": "data/a-001.parquet"}]}
    del parts["files"]
    rc, out, _ = build(tmp_path, [parts], out="out2")
    done = done_of(out, "web")
    first = {r["text"] for r in pq.read_table(tmp_path / "raw" / "web" / "data" / "a-000.parquet").to_pylist()}
    from_first = sum(len(t) + 1 for t in split_texts(out, "web", "train") if t in first)
    assert 1500 <= from_first < 1500 + 201 and done["tokens"] >= 5000
    assert [r["path"] for r in done["raw_files"]] == ["data/a-000.parquet", "data/a-001.parquet"]

    # raw files run out before the target: built, and marked short
    spec["target_tokens"] = 10 ** 9
    rc, out, _ = build(tmp_path, [spec], out="out3")
    assert rc == 0 and done_of(out, "web")["short"]


def test_a_document_is_never_split_across_shards(tmp_path):
    spec, texts = web_source(tmp_path, n=200)
    rc, out, _ = build(tmp_path, [spec], shard_tokens=1000)
    done = done_of(out, "web")
    assert len(done["shards"]) > 5
    for sh in done["train"]["shards"]:
        docs = docs_in(out / sh["file"])
        assert sh["tokens"] == sum(len(d) for d in docs) <= 1000 and sh["documents"] == len(docs)
    assert sorted(split_texts(out, "web", "train") + split_texts(out, "web", "val")) == sorted(texts.values())
    assert len(done["val_shards"]) == 1 and done["val_shards"][0] == "web_val_000000.bin"


def test_resume_skips_a_finished_source_and_rebuilds_an_unfinished_one(tmp_path):
    a, _ = web_source(tmp_path, name="a")
    b, _ = web_source(tmp_path, name="b")
    rc, out, _ = build(tmp_path, [a, b])
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.glob("*.bin")}
    (out / "b.done.json").unlink()                       # b was interrupted: shards, no done.json
    (out / "b_train_000007.bin").write_bytes(b"stale")
    (out / "b_train_000000.bin").write_bytes(b"half written")
    rc, out, tok = build(tmp_path, [a, b])
    assert rc == 0
    assert all(t.startswith("doc") for t in tok.seen) and len(tok.seen) == 120  # only b was tokenized again
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.glob("*.bin")}
    assert set(after) == set(before)                     # the stale shard is gone
    assert all(after[n] == before[n] for n in after if n.startswith("a_"))
    assert all(after[n][1] == before[n][1] for n in after if n.startswith("b_"))
    assert (out / "b.done.json").exists()


def test_mix_json_matches_what_was_written(tmp_path):
    web, _ = web_source(tmp_path, n=300)
    long_rows = [{"id": str(i), "text": text_of(i, 300 + 37 * (i % 9))} for i in range(60)]
    write_jsonl_gz(tmp_path / "raw" / "long" / "l.jsonl.gz", long_rows)
    long = {"name": "long", "kind": "docs", "local": str(tmp_path / "raw" / "long"), "files": "*.jsonl.gz",
            "id_column": "id", "min_tokens": 128, "target_tokens": 10 ** 9, "val_fraction": 0.2, "val_min_docs": 3}
    sources = [web, long, {"name": "ext_here", "kind": "external"}, {"name": "ext_gone", "kind": "external"}]
    phases = {"trunk": {"seq_len": 64, "rule": "stream", "seed": 1, "tokens": 6400,
                        "train": [{"name": "web", "weight": 0.75}, {"name": "ext_here", "weight": 0.25}],
                        "eval": [{"name": "web", "rows": 4}]},
              "decay": {"seq_len": 128, "rule": "doc", "seed": 2, "tokens": 10 ** 7,
                        "train": [{"name": "long", "weight": 0.5}, {"name": "web", "weight": 0.25, "rule": "stream"},
                                  {"name": "ext_gone", "weight": 0.25, "rule": "stream"}],
                        "eval": [{"name": "long", "rows": 2}]}}
    out = tmp_path / "out"
    out.mkdir()
    # an external source built by another script: its own done.json format, shards in the mix folder
    m21.write_shard(out / "ext_here_train_000000.bin", np.asarray(([EOT] + [5] * 99) * 7, dtype=np.uint16))
    (out / "ext_here.done.json").write_text(json.dumps({"tokens": 700, "whatever": True}))
    rc, out, _ = build(tmp_path, sources, phases)
    assert rc == 0
    mix = json.loads((out / "mix.json").read_text())
    assert mix["name"] == "tiny" and mix["tokenizer"] == "tokenizer" and mix["eot"] == EOT and mix["vocab_size"] == 259
    assert (out / "tokenizer" / "tokenizer.json").exists()
    assert set(mix["sources"]) == {"web", "long", "ext_here"} and mix["external_missing"] == ["ext_gone"]
    assert mix["missing"] == []
    for name in ("web", "long"):
        src = mix["sources"][name]
        assert src == done_of(out, name)
        on_disk = sorted(p.name for p in out.glob(f"{name}_train_*.bin"))
        assert [s["file"] for s in src["train"]["shards"]] == on_disk == src["shards"]
        assert [s["tokens"] for s in src["train"]["shards"]] == [len(open_shard(str(out / f))) for f in on_disk]
        assert src["tokens"] == sum(s["tokens"] for s in src["train"]["shards"])
        hist = src["train"]["length_hist"]
        assert set(hist) == {"4096", "16384", "65536", "131072"} and hist["4096"] == {"documents": 0, "tokens": 0}

    def rows(name, split, rule, S):  # what the trainer's reader finds
        files = sorted(str(p) for p in out.glob(f"{name}_{split}_*.bin"))
        return len(BinRowDataset([{"name": name, "files": files}], seq_len=S, rule=rule, eot=EOT))

    trunk = {r["name"]: r for r in mix["phases"]["trunk"]["supply"]}
    assert trunk["web"] == {"name": "web", "rule": "stream", "rows_needed": 75, "missing": False,
                            "rows_available": rows("web", "train", "stream", 64),
                            "short": rows("web", "train", "stream", 64) < 75}
    assert trunk["ext_here"]["rows_available"] == 10 and trunk["ext_here"]["rows_needed"] == 25  # 700 // 64
    assert trunk["ext_here"]["short"] and not trunk["ext_here"]["missing"]
    decay = {r["name"]: r for r in mix["phases"]["decay"]["supply"]}
    assert decay["long"]["rule"] == "doc" and decay["long"]["rows_available"] == rows("long", "train", "doc", 128)
    assert decay["long"]["rows_needed"] == 39063 and decay["long"]["short"]          # loud, not fatal
    assert decay["web"]["rule"] == "stream" and decay["web"]["rows_available"] == rows("web", "train", "stream", 128)
    assert decay["ext_gone"] == {"name": "ext_gone", "rule": "stream", "rows_needed": 19532,
                                 "rows_available": None, "short": True, "missing": True}
    ev = mix["phases"]["decay"]["eval_supply"][0]
    assert ev["rule"] == "doc" and ev["rows_available"] == rows("long", "val", "doc", 128) >= 2 and not ev["short"]
    assert mix["phases"]["trunk"]["seed"] == 1 and mix["phases"]["decay"]["train"] == phases["decay"]["train"]
    assert mix["token_budget"]["web"]["tokens_needed"] == {"trunk": 4800, "decay": 2500000}
    budget = mix["token_budget"]["web"]
    assert budget["short"] and budget["tokens_built"] == mix["sources"]["web"]["tokens"]


def test_a_failing_source_does_not_stop_the_others(tmp_path, capsys):
    good, _ = web_source(tmp_path, name="good")
    bad_column = {**good, "name": "bad_column", "text_column": "body"}
    no_files = {**good, "name": "no_files", "files": "nothing/*.parquet"}
    rc, out, _ = build(tmp_path, [bad_column, no_files, good])
    assert rc == 1
    assert (out / "good.done.json").exists() and not (out / "bad_column.done.json").exists()
    mix = json.loads((out / "mix.json").read_text())
    assert mix["missing"] == ["bad_column", "no_files"] and set(mix["sources"]) == {"good"}
    text = capsys.readouterr().out
    assert "bad_column FAILED" in text and "no_files FAILED" in text and "NOT BUILT: bad_column, no_files" in text


def test_scale_gives_a_complete_tiny_build(tmp_path):
    web, _ = web_source(tmp_path, n=300, val_fraction=0.005)
    web["target_tokens"] = 400_000
    long_rows = [{"id": str(i), "text": text_of(i, 300)} for i in range(60)]
    write_jsonl_gz(tmp_path / "raw" / "long" / "l.jsonl.gz", long_rows)
    long = {"name": "long", "kind": "docs", "local": str(tmp_path / "raw" / "long"), "files": "*.jsonl.gz",
            "id_column": "id", "min_tokens": 128, "target_tokens": 300_000, "heldout": "longdocs_lite",
            "heldout_min_tokens": 128}
    rc, out, _ = build(tmp_path, [web, long], argv=("--scale", "0.001"), gpt2_count=lambda ts: [len(t) for t in ts])
    assert rc == 0
    for name, target in (("web", 400), ("long", 300)):
        done = done_of(out, name)
        assert done["target_tokens"] == target and done["scale"] == 0.001
        assert len(done["shards"]) >= 1 and done["val_shards"] == [f"{name}_val_000000.bin"]
        assert done["val"]["documents"] >= 1 and done["tokens"] < target + 302
    assert done_of(out, "long")["heldout"]["val_fallback_last_train_document"]
    assert not set(split_texts(out, "long", "train")) & set(split_texts(out, "long", "val"))


def test_read_rows_reopens_after_a_transient_error(tmp_path, monkeypatch):
    rows = [{"id": str(i), "text": text_of(i, 30)} for i in range(50)]
    write_jsonl_gz(tmp_path / "raw" / "s" / "x.jsonl.gz", rows)
    spec = {"name": "s", "local": str(tmp_path / "raw" / "s")}
    monkeypatch.setattr(m21, "_sleep", lambda s: None)
    real, calls = m21.parse_rows, []

    def flaky(f, path, columns):
        calls.append(1)
        for i, row in enumerate(real(f, path, columns)):
            if len(calls) < 3 and i == 20 * len(calls):
                raise ConnectionError("reset by peer")
            yield row

    monkeypatch.setattr(m21, "parse_rows", flaky)
    got = list(m21.read_rows(m21.Raw(tmp_path / "rawdir"), spec, "x.jsonl.gz", ["id", "text"]))
    assert [r["id"] for r in got] == [str(i) for i in range(50)] and len(calls) == 3

    def broken(f, path, columns):
        raise KeyError("no column")
        yield

    monkeypatch.setattr(m21, "parse_rows", broken)
    with pytest.raises(KeyError):  # not a network error: no retry
        list(m21.read_rows(m21.Raw(tmp_path / "rawdir"), spec, "x.jsonl.gz", ["id"]))
    assert m21.is_transient(TimeoutError()) and not m21.is_transient(FileNotFoundError())


def test_list_and_names(tmp_path, capsys):
    web, _ = web_source(tmp_path)
    gone = {**web, "name": "gone", "local": str(tmp_path / "nowhere")}
    ext = {"name": "ext", "kind": "external"}
    web["target_tokens"], gone["target_tokens"] = 5, 9
    rc, out, _ = build(tmp_path, [web, ext], argv=("--list",))
    assert rc == 0 and "web: " in capsys.readouterr().out and not list(out.glob("*.bin"))
    rc, out, _ = build(tmp_path, [web, gone, ext], argv=("--list",))
    assert rc == 1 and "gone: " + str(tmp_path / "nowhere") + ": !! NOT REACHABLE" in capsys.readouterr().out
    rc, out, _ = build(tmp_path, [web, gone, ext], argv=("--names",))
    assert rc == 0 and capsys.readouterr().out.split() == ["gone", "web"]
    with pytest.raises(ValueError):
        build(tmp_path, [web], argv=("--only", "nope"))


def test_only_and_mix_only(tmp_path):
    a, _ = web_source(tmp_path, name="a")
    b, _ = web_source(tmp_path, name="b")
    rc, out, _ = build(tmp_path, [a, b], argv=("--only", "b", "--no-mix"))
    assert rc == 0 and (out / "b.done.json").exists() and not (out / "a.done.json").exists()
    assert not (out / "mix.json").exists()
    rc, out, tok = build(tmp_path, [a, b], argv=("--mix-only",))
    assert rc == 1 and tok.seen == []  # a is not built: loud, and nothing is tokenized by --mix-only
    assert json.loads((out / "mix.json").read_text())["missing"] == ["a"]
    rc, out, _ = build(tmp_path, [a, b], argv=("--only", "a", "--no-mix"))
    rc, out, _ = build(tmp_path, [a, b], argv=("--mix-only",))
    assert rc == 0 and set(json.loads((out / "mix.json").read_text())["sources"]) == {"a", "b"}

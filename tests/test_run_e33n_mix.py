"""scripts/run_e33n.py --mix: phase manifests and trainer commands from a built data mix (E38)."""
from __future__ import annotations

import json

import numpy as np

from data.bin_token_dataset import load_bin_manifest, write_shard
from scripts.run_e33n import mix_manifests, train_cmd


def _mix(tmp_path, eot=0):
    d = tmp_path / "mix"
    (d / "tokenizer").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def docs(n_docs, length):  # eot starts every document
        out = []
        for _ in range(n_docs):
            out += [eot] + rng.integers(5, 90, length - 1).tolist()
        return out

    for i in range(1, 7):  # web: 6 shards of 640 tokens, short documents
        write_shard(d / f"web_train_{i:06d}.bin", docs(20, 32))
    write_shard(d / "web_val_000000.bin", docs(20, 32))
    for i in range(1, 3):  # books: documents of 200 tokens
        write_shard(d / f"books_train_{i:06d}.bin", docs(8, 200))
    write_shard(d / "books_val_000000.bin", docs(4, 200))
    write_shard(d / "hop_train_000001.bin", docs(10, 64))  # whole rows of 64
    mix = {"name": "t", "tokenizer": "tokenizer", "vocab_size": 100, "eot": eot, "sources": {},
           "phases": {
               "trunk": {"seq_len": 32, "rule": "stream", "seed": 42, "tokens": 2000,
                         "train": [{"name": "web", "weight": 1.0}], "eval": [{"name": "web", "rows": 4}]},
               "decay": {"seq_len": 64, "rule": "doc", "seed": 1042, "tokens": 1000,
                         "train": [{"name": "books", "weight": 0.5}, {"name": "web", "weight": 0.3, "rule": "stream"},
                                   {"name": "hop", "weight": 0.2, "rule": "stream"}],
                         "eval": [{"name": "books", "rows": 2}]},
               "long": {"seq_len": 128, "rule": "doc", "seed": 2042, "tokens": 800,
                        "train": [{"name": "books", "weight": 0.5}, {"name": "web", "weight": 0.5, "rule": "stream"}],
                        "eval": [{"name": "books", "rows": 2}]}}}
    (d / "mix.json").write_text(json.dumps(mix))
    return d


def test_phase_manifests_split_shared_stream_sources(tmp_path):
    d = _mix(tmp_path)
    paths, mix = mix_manifests(d, tmp_path / "run")
    man = {k: json.loads(p.read_text()) for k, p in paths.items()}
    web = {k: [f.split("_")[-1] for s in m["train"] if s["name"] == "web" for f in s["files"]] for k, m in man.items()}
    # long needs 0.5 × 800 × 1.03 = 412 tokens → the last shard (640); decay 0.3 × 1000 × 1.03 = 309 → the one before
    assert web["long"] == ["000006.bin"] and web["decay"] == ["000005.bin"]
    assert web["trunk"] == [f"{i:06d}.bin" for i in range(1, 5)]
    assert not set(web["trunk"]) & set(web["decay"]) and not set(web["decay"]) & set(web["long"])
    books = {k: [s for s in m["train"] if s["name"] == "books"][0]["files"] for k, m in man.items() if k != "trunk"}
    assert books["decay"] == books["long"] and len(books["decay"]) == 2  # long documents are shared
    assert all(m["eot"] == 0 for m in man.values())
    assert [s.get("rule") for s in man["decay"]["train"]] == [None, "stream", "stream"]
    for k, S in (("trunk", 32), ("decay", 64), ("long", 128)):
        train, ev, m = load_bin_manifest(str(paths[k]))
        item = train[0]
        assert item["input_ids"].shape == (S,) and len(train) > 0 and ev is not None and len(ev) > 0
    train, _, _ = load_bin_manifest(str(paths["decay"]))
    assert set(train.rows_per_source) == {"books", "web", "hop"}


def test_train_cmd_takes_tokenizer_tokens_and_rows_from_the_mix(tmp_path):
    d = _mix(tmp_path)
    cmd = train_cmd("dense", "tiny", "decay", data=tmp_path, out=tmp_path / "run", init="X", mix=str(d), fast=True)
    v = lambda flag: cmd[cmd.index(flag) + 1]
    assert v("--tokenizer_name") == str(d / "tokenizer")
    assert v("--max_seq_length") == "64" and v("--use_liger") == "True"
    assert v("--bin_manifest").endswith("manifest_decay.json")
    rows = int(v("--per_device_train_batch_size")) * int(v("--gradient_accumulation_steps"))
    assert int(v("--max_steps")) == max(1, round(1000 / (rows * 64)))  # the mix's 1,000 decay tokens
    assert "--concept_compile" not in cmd  # not a concept arm
    cmd = train_cmd("e37a", "tiny", "trunk", data=tmp_path, out=tmp_path / "run", mix=str(d), fast=True, tokens=640)
    assert cmd[cmd.index("--concept_compile") + 1] == "True" and "--model_compile" not in cmd


def test_lenient_mode_for_small_builds(tmp_path):
    """One web shard for three phases, and a phase source with no row: strict stops, lenient shares and leaves out;
    an evaluation source with no row at the phase's length is always left out."""
    import pytest

    from scripts.run_e33n import mix_supply

    d = _mix(tmp_path)
    for i in range(2, 7):
        (d / f"web_train_{i:06d}.bin").unlink()
    mix = json.loads((d / "mix.json").read_text())
    mix["phases"]["long"]["supply"] = [{"name": "books", "rows_available": 0, "rows_needed": 3},
                                       {"name": "web", "rows_available": 5, "rows_needed": 3}]
    mix["phases"]["long"]["eval_supply"] = [{"name": "books", "rows_available": 0, "rows_needed": 2}]
    mix["phases"]["decay"]["supply"] = [{"name": "books", "rows_available": 12, "rows_needed": 8},
                                        {"name": "web", "rows_available": 10, "rows_needed": 5},
                                        {"name": "hop", "rows_available": 10, "rows_needed": 3}]
    (d / "mix.json").write_text(json.dumps(mix))
    sup = mix_supply(mix)
    assert sup["long"]["empty"] == ["books"] and sup["long"]["eval_empty"] == ["books"]
    assert sup["decay"]["limited_by"] == "books" and sup["decay"]["rows_per_pass"] == 24  # 12 rows at half the weight
    with pytest.raises(ValueError):
        mix_manifests(d, tmp_path / "strict")
    paths, _ = mix_manifests(d, tmp_path / "run", lenient=True)
    man = {k: json.loads(p.read_text()) for k, p in paths.items()}
    assert [s["name"] for s in man["long"]["train"]] == ["web"] and man["long"]["eval"] == []
    assert all(len([s for s in m["train"] if s["name"] == "web"][0]["files"]) == 1 for m in man.values())

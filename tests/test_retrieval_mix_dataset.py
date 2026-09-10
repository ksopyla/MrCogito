"""E18b: scripts/build_retrieval_mix_dataset.py — dense-label keyed-recall rows + merged manifest."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, Features, Sequence, Value

from data.data_collators import labels_from_span_markers
from scripts.build_retrieval_mix_dataset import (
    FillerStream,
    build_row,
    iter_rows,
    merge_manifest,
    row_weight_for_token_share,
    token_share_for_row_weight,
)

FEATS = Features({
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    "special_tokens_mask": Sequence(Value("int8")),
})
BOS, EOS, START, END = 1, 2, 90, 91
KEY_LO, KEY_HI, KEY_LEN = 10, 20, 3
TRAIN_RANGE, EVAL_RANGE = (100, 200), (300, 400)


def _fake_source(root: Path, name: str, n_train=12, n_eval=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for split, n, rng_range in (("train", n_train, TRAIN_RANGE), ("eval", n_eval, EVAL_RANGE)):
        rows = []
        for _ in range(n):
            L = int(rng.integers(40, 120))
            ids = [BOS] + rng.integers(*rng_range, size=L - 2).tolist() + [EOS]
            rows.append({"input_ids": ids, "attention_mask": [1] * L,
                         "special_tokens_mask": [1] + [0] * (L - 2) + [1]})
        ds = Dataset.from_list(rows, features=FEATS)
        path = root / name / split
        ds.save_to_disk(str(path))
        out[split] = str(path)
    return {"name": name, "weight": 0.5, "train_path": out["train"], "eval_path": out["eval"]}


@pytest.fixture
def base_manifest(tmp_path):
    srcs = [_fake_source(tmp_path, "a", seed=1), _fake_source(tmp_path, "b", seed=2)]
    man = {"mix_id": "fake", "objective": "causal_lm", "max_seq_length": 512, "seed": 42, "sources": srcs}
    path = tmp_path / "base_manifest.json"
    path.write_text(json.dumps(man))
    return path, man


ROW_KW = dict(context=512, items=(2, 4), short_len=(4, 6), span_len=(8, 16), min_gap=64,
              key_lo=KEY_LO, key_hi=KEY_HI, key_len=KEY_LEN, start=START, end=END, bos=BOS, eos=EOS)


def _targets(ids):
    """[(start_index, key, value)] for every START in the row."""
    out = []
    for i, t in enumerate(ids):
        if t == START:
            j = ids.index(END, i)
            out.append((i, ids[i - KEY_LEN:i], ids[i + 1:j]))
    return out


def test_rows_are_exact_length_with_consistent_specials(base_manifest):
    path, man = base_manifest
    rows = list(iter_rows(5, [s["train_path"] for s in man["sources"]], seed=0, **ROW_KW))
    for r in rows:
        ids, am, sp = r["input_ids"], r["attention_mask"], r["special_tokens_mask"]
        assert len(ids) == 512 and len(am) == 512 and len(sp) == 512
        assert ids[0] == BOS and ids[-1] == EOS and all(am)
        assert [i for i, s in enumerate(sp) if s] == [i for i, t in enumerate(ids) if t in (BOS, EOS, START, END)]
        assert ids.count(START) == ids.count(END) >= 2


def test_every_target_has_an_earlier_source_at_least_min_gap_back(base_manifest):
    path, man = base_manifest
    rows = list(iter_rows(8, [s["train_path"] for s in man["sources"]], seed=3, **ROW_KW))
    for r in rows:
        ids = r["input_ids"]
        for i, key, value in _targets(ids):
            pat = key + value
            # the source occurrence: key immediately followed by the value, at least min_gap before START
            hits = [j for j in range(0, i - len(pat)) if ids[j:j + len(pat)] == pat]
            assert hits, "source not found"
            assert i - hits[0] >= ROW_KW["min_gap"]


def test_marker_rule_recovers_exactly_the_values_and_end(base_manifest):
    path, man = base_manifest
    rows = list(iter_rows(4, [s["train_path"] for s in man["sources"]], seed=5, **ROW_KW))
    for r in rows:
        ids = r["input_ids"]
        labels = labels_from_span_markers(ids, START, END)
        expected = []
        for i, key, value in _targets(ids):
            expected.extend(value + [END])
        assert [l for l in labels if l != -100] == expected
        frac = sum(l != -100 for l in labels) / len(labels)
        assert 0.02 <= frac <= 0.6, frac


def test_filler_never_uses_eval_rows(base_manifest):
    path, man = base_manifest
    rows = list(iter_rows(6, [s["train_path"] for s in man["sources"]], seed=7, **ROW_KW))
    for r in rows:
        assert not any(EVAL_RANGE[0] <= t < EVAL_RANGE[1] for t in r["input_ids"])


def test_row_weight_matches_token_share():
    for f, m, S in [(0.05, 2858.0, 32768), (0.1, 1000.0, 8192), (0.5, 4096.0, 4096)]:
        w = row_weight_for_token_share(f, m, S)
        assert abs(token_share_for_row_weight(w, m, S) - f) < 1e-9
    assert abs(row_weight_for_token_share(0.05, 2858.0, 32768) - 0.00457) < 5e-5
    with pytest.raises(ValueError):
        row_weight_for_token_share(0.0, 2858.0, 32768)


def test_merge_manifest_scales_base_and_excludes_retrieval_from_eval(base_manifest):
    path, man = base_manifest
    src = {"name": "retrieval_keyed_recall", "train_path": "/x/train", "eval_path": "/x/eval",
           "num_train_rows": 10, "num_eval_rows": 2}
    merged = merge_manifest(man, src, 0.01, {"row_weight": 0.01})
    weights = [s["weight"] for s in merged["sources"]]
    assert abs(sum(weights) - 1.0) < 1e-9
    assert merged["sources"][-1]["in_eval"] is False and merged["sources"][-1]["weight"] == 0.01
    assert [s["name"] for s in merged["sources"][:-1]] == ["a", "b"]
    assert merged["retrieval_meta"]["row_weight"] == 0.01 and merged["mix_id"].startswith("fake+")
    assert man["sources"][0]["weight"] == 0.5  # base untouched


def test_cli_end_to_end_writes_arrow_and_manifest(base_manifest, tmp_path):
    path, man = base_manifest
    out_dir, out_man = tmp_path / "ret", tmp_path / "merged.json"
    cmd = [sys.executable, "scripts/build_retrieval_mix_dataset.py", "--base_manifest", str(path),
           "--fraction", "0.05", "--n_train", "3", "--n_eval", "2", "--context", "512",
           "--out_dir", str(out_dir), "--out_manifest", str(out_man), "--seed", "0",
           "--start_id", str(START), "--end_id", str(END), "--bos", str(BOS), "--eos", str(EOS),
           "--key_lo", str(KEY_LO), "--key_hi", str(KEY_HI), "--items", "2", "3", "--short_len", "4", "6",
           "--span_len", "8", "16", "--min_gap", "64", "--base_mean_row_tokens", "80"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert res.returncode == 0, res.stdout + res.stderr
    from datasets import load_from_disk
    tr, ev = load_from_disk(str(out_dir / "train")), load_from_disk(str(out_dir / "eval"))
    assert len(tr) == 3 and len(ev) == 2
    assert tr.features == FEATS  # exact LM-shard schema so the interleave can concatenate
    merged = json.loads(out_man.read_text())
    assert merged["sources"][-1]["name"] == "retrieval_keyed_recall"
    assert abs(merged["retrieval_meta"]["achieved_token_fraction"] - 0.05) < 1e-6
    assert abs(sum(s["weight"] for s in merged["sources"]) - 1.0) < 1e-9

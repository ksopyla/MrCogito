"""load_pretokenized_mix: per-source `in_eval: false` keeps a source out of the trainer eval."""
import json

import pytest
from datasets import Dataset

from data.dataset_preprocess import load_pretokenized_mix


def _src(root, name, n_train, n_eval, in_eval=None):
    tr = Dataset.from_dict({"input_ids": [[1, 2, 3]] * n_train})
    ev = Dataset.from_dict({"input_ids": [[4, 5]] * n_eval})
    tr.save_to_disk(str(root / name / "train")); ev.save_to_disk(str(root / name / "eval"))
    src = {"name": name, "weight": 0.5, "train_path": str(root / name / "train"), "eval_path": str(root / name / "eval")}
    if in_eval is not None:
        src["in_eval"] = in_eval
    return src


def _manifest(root, sources):
    p = root / "m.json"
    p.write_text(json.dumps({"mix_id": "t", "seed": 1, "sources": sources}))
    return p


def test_in_eval_false_excludes_only_that_source(tmp_path):
    srcs = [_src(tmp_path, "lm", 6, 3), _src(tmp_path, "ret", 4, 2, in_eval=False)]
    train, evaluation = load_pretokenized_mix(_manifest(tmp_path, srcs))
    # the all-exhausted interleave cycles the smaller source, so train >= the row sum; eval is exact
    assert len(train) >= 10 and len(evaluation) == 3
    assert set(tuple(r) for r in train["input_ids"]) == {(1, 2, 3)}


def test_in_eval_default_true_is_unchanged(tmp_path):
    srcs = [_src(tmp_path, "lm", 6, 3), _src(tmp_path, "ret", 4, 2)]
    train, evaluation = load_pretokenized_mix(_manifest(tmp_path, srcs))
    assert len(train) >= 10 and len(evaluation) == 5


def test_all_sources_excluded_is_an_error(tmp_path):
    srcs = [_src(tmp_path, "a", 2, 1, in_eval=False), _src(tmp_path, "b", 2, 1, in_eval=False)]
    with pytest.raises(ValueError):
        load_pretokenized_mix(_manifest(tmp_path, srcs))

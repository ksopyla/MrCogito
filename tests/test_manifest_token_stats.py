import json

from datasets import Dataset

from data.dataset_preprocess import load_pretokenized_mix
from scripts.manifest_token_stats import compute_stats


def test_manifest_token_stats_counts_interleaved_tokens(tmp_path):
    sources = []
    for name, rows, weight in [
        ("a", [[1, 2, 3], [4, 5]], 0.75),
        ("b", [[6, 7, 8, 9]], 0.25),
    ]:
        train_path = tmp_path / name / "train"
        eval_path = tmp_path / name / "eval"
        Dataset.from_dict({"input_ids": rows}).save_to_disk(train_path)
        Dataset.from_dict({"input_ids": [rows[0]]}).save_to_disk(eval_path)
        sources.append(
            {
                "name": name,
                "weight": weight,
                "train_path": str(train_path),
                "eval_path": str(eval_path),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mix_id": "tiny", "seed": 42, "sources": sources}))

    train_ds, _ = load_pretokenized_mix(manifest)
    expected_tokens = sum(len(row["input_ids"]) for row in train_ds)
    stats = compute_stats(manifest, target_tokens=100, effective_batch=4, num_proc=2)
    assert stats["train_rows"] > 0
    assert stats["full_epoch_tokens"] == expected_tokens
    assert stats["epochs_for_target"] == 100 / stats["full_epoch_tokens"]
    assert stats["estimated_optimizer_steps"] > 0
    assert compute_stats(manifest, 100, 4, num_proc=1) == stats  # cached result is stable
    # Changing only effective_batch must reuse the token count (no recount).
    stats_bs8 = compute_stats(manifest, 100, 8, num_proc=1)
    assert stats_bs8["full_epoch_tokens"] == stats["full_epoch_tokens"]
    assert stats_bs8["effective_batch"] == 8
    assert stats_bs8["estimated_optimizer_steps"] == __import__("math").ceil(
        stats["train_rows"] * stats_bs8["epochs_for_target"] / 8
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_token_count_fast_paths_match_the_scan(tmp_path, monkeypatch):
    import numpy as np
    import scripts.manifest_token_stats as mts
    from data.length_cache import save_length_cache

    rows = [[1] * n for n in np.random.default_rng(3).integers(1, 30, size=40).tolist()]
    train_path, eval_path = tmp_path / "s" / "train", tmp_path / "s" / "eval"
    Dataset.from_dict({"input_ids": rows}).save_to_disk(train_path)
    Dataset.from_dict({"input_ids": rows[:2]}).save_to_disk(eval_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mix_id": "t", "seed": 1, "sources": [
        {"name": "s", "weight": 1.0, "train_path": str(train_path), "eval_path": str(eval_path)}]}))
    train_ds, _ = load_pretokenized_mix(manifest)
    expected = sum(len(r) for r in rows)
    # Arrow offsets (no length cache yet)
    assert mts._fast_token_count(manifest, train_ds) == expected
    # valid length cache wins
    save_length_cache(manifest, np.array([len(r["input_ids"]) for r in train_ds], dtype=np.int32))
    monkeypatch.setattr(mts, "_lengths_from_list_offsets", lambda ds: (_ for _ in ()).throw(AssertionError("offsets not needed")))
    assert mts._fast_token_count(manifest, train_ds) == expected
    # a cache of the wrong size is ignored and offsets are used
    monkeypatch.setattr(mts, "_lengths_from_list_offsets", lambda ds: np.array([1] * len(ds), dtype=np.int32))
    monkeypatch.setattr(mts, "load_length_cache", lambda m: np.array([5, 5], dtype=np.int32))
    assert mts._fast_token_count(manifest, train_ds) == len(train_ds)
    # no fast source -> scan
    monkeypatch.setattr(mts, "_lengths_from_list_offsets", lambda ds: None)
    monkeypatch.setattr(mts, "load_length_cache", lambda m: None)
    stats = mts.compute_stats(manifest, target_tokens=1000, effective_batch=4, num_proc=1)
    assert stats["full_epoch_tokens"] == expected

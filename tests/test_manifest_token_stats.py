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

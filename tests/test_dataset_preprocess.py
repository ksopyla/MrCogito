import os
import sys

import pytest
from datasets import Dataset, DatasetDict, interleave_datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataset_preprocess as dataset_preprocess_module


@pytest.mark.parametrize(
    ("lengths", "probabilities", "seed"),
    [
        ([3, 5], [0.7, 0.3], 42),
        ([2, 4, 7], [0.2, 0.3, 0.5], 7),
        ([1, 9, 3], [0.05, 0.8, 0.15], 123),
    ],
)
def test_fast_weighted_interleave_matches_huggingface(lengths, probabilities, seed):
    parts = []
    offset = 0
    for length in lengths:
        parts.append(Dataset.from_dict({"row_id": list(range(offset, offset + length))}))
        offset += length
    expected = interleave_datasets(
        parts,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    actual = dataset_preprocess_module._fast_weighted_all_exhausted_interleave(
        parts, probabilities, seed
    )
    assert actual["row_id"] == expected["row_id"]


class FakeTokenizer:
    def __call__(
        self,
        text_batch,
        padding="max_length",
        truncation=True,
        max_length=8,
        return_special_tokens_mask=True,
    ):
        input_ids = []
        attention_mask = []
        special_tokens_mask = []

        for text in text_batch:
            token_count = max(1, min(max_length - 2, len(text.split())))
            ids = [101] + list(range(1, token_count + 1)) + [102]
            ids = ids[:max_length]
            pad_len = max_length - len(ids)

            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            special_tokens_mask.append(
                [1] + [0] * max(0, len(ids) - 2) + [1] + [1] * pad_len
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


def test_prefers_validation_split(monkeypatch, tmp_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["train one", "train two", "train three"]}),
            "validation": Dataset.from_dict({"text": ["val one", "val two"]}),
            "test": Dataset.from_dict({"text": ["test should not be used"]}),
        }
    )

    monkeypatch.setattr(
        dataset_preprocess_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    train_ds, eval_ds = dataset_preprocess_module.load_and_preprocess_text_dataset(
        tokenizer=FakeTokenizer(),
        dataset_hf_path="Salesforce/wikitext",
        dataset_name_subset="wikitext-103-v1",
        text_column_name="text",
        dataset_cache_dir=str(tmp_path),
        train_num_proc=1,
        test_num_proc=1,
    )

    assert len(train_ds) == 3
    assert len(eval_ds) == 2
    assert "input_ids" in train_ds.column_names
    assert "attention_mask" in eval_ds.column_names


def test_renames_requested_text_column(monkeypatch, tmp_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"content": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]}
            )
        }
    )

    monkeypatch.setattr(
        dataset_preprocess_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    train_ds, eval_ds = dataset_preprocess_module.load_and_preprocess_text_dataset(
        tokenizer=FakeTokenizer(),
        dataset_hf_path="custom/wiki",
        dataset_name_subset=None,
        text_column_name="content",
        dataset_cache_dir=str(tmp_path),
        train_num_proc=1,
        test_num_proc=1,
    )

    assert len(train_ds) + len(eval_ds) == 3
    assert "input_ids" in train_ds.column_names
    assert "special_tokens_mask" in eval_ds.column_names
    assert "content" not in train_ds.column_names


def test_raises_when_requested_text_column_missing(monkeypatch, tmp_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"body": ["missing requested column"]}),
            "validation": Dataset.from_dict({"body": ["still missing"]}),
        }
    )

    monkeypatch.setattr(
        dataset_preprocess_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    with pytest.raises(ValueError, match="requested text column"):
        dataset_preprocess_module.load_and_preprocess_text_dataset(
            tokenizer=FakeTokenizer(),
            dataset_hf_path="custom/wiki",
            dataset_name_subset=None,
            text_column_name="text",
            dataset_cache_dir=str(tmp_path),
            train_num_proc=1,
            test_num_proc=1,
        )

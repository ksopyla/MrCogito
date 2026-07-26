import json
from collections import Counter

from datasets import Dataset

from data.dataset_preprocess import load_pretokenized_mix
from data.delayed_recall import (
    build_delayed_recall_rows,
    select_delayed_recall_token_pools,
)


class TinyWordTokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0

    _markers = {
        "Memory record. Key": [3],
        "has value": [4],
        ".": [5],
        "Recall the stored value for key": [6],
        ". Answer:": [7],
    }

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._markers[text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        token_id = token_ids[0]
        first = chr(ord("a") + (token_id // 26) % 26)
        second = chr(ord("a") + token_id % 26)
        return f" word{first}{second}"


def _build_rows(seed=42, split="train"):
    tokenizer = TinyWordTokenizer()
    pools = select_delayed_recall_token_pools(
        tokenizer,
        model_vocab_size=len(tokenizer),
        value_count=4,
        key_count=4,
        noise_count=8,
        seed=seed,
    )
    rows = build_delayed_recall_rows(
        tokenizer,
        pools,
        split=split,
        num_rows=8,
        sequence_length=32,
        block_size=8,
        query_block=4,
        seed=seed + 1,
    )
    return pools, rows


def test_delayed_recall_rows_are_deterministic_balanced_and_counterfactual():
    pools, rows = _build_rows()
    _, rebuilt = _build_rows()
    assert rows == rebuilt
    assert len(rows) == 8
    assert Counter(row["answer_token_id"] for row in rows) == {
        value_id: 2 for value_id in pools.value_ids
    }

    for left, right in zip(rows[::2], rows[1::2], strict=True):
        answer_index = left["answer_index"]
        assert answer_index == 28
        assert left["input_ids"][8:answer_index] == right["input_ids"][8:answer_index]
        assert left["answer_token_id"] != right["answer_token_id"]
        assert sum(label != -100 for label in left["labels"]) == 1
        assert left["labels"][answer_index] == left["answer_token_id"]
        assert left["donor_answer_token_id"] == right["answer_token_id"]


def test_delayed_recall_pair_ids_are_split_disjoint():
    _, train = _build_rows(split="train")
    _, evaluation = _build_rows(split="eval")
    assert {row["pair_id"] for row in train}.isdisjoint(
        {row["pair_id"] for row in evaluation}
    )


def test_delayed_recall_manifest_round_trip(tmp_path):
    _, train_rows = _build_rows(split="train")
    _, eval_rows = _build_rows(split="eval")
    train_path = tmp_path / "train"
    eval_path = tmp_path / "eval"
    Dataset.from_list(train_rows).save_to_disk(str(train_path))
    Dataset.from_list(eval_rows).save_to_disk(str(eval_path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mix_id": "test_delayed_recall",
                "max_seq_length": 32,
                "objective": "causal_lm",
                "seed": 42,
                "sources": [
                    {
                        "name": "delayed",
                        "weight": 1.0,
                        "train_path": str(train_path),
                        "eval_path": str(eval_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    train, evaluation = load_pretokenized_mix(manifest)
    assert len(train) == len(train_rows)
    assert len(evaluation) == len(eval_rows)
    assert train.column_names == Dataset.from_list(train_rows).column_names

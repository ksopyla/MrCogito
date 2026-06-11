"""
Tests for DataCollatorForTSDAE.
Run: poetry run pytest tests/test_tsdae_collator.py -v
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import AutoTokenizer
from data.data_collators import DataCollatorForTSDAE


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


@pytest.fixture
def sample_features(tokenizer):
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Concept encoder models aim to capture semantic meaning across multiple tokens.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    features = []
    for text in texts:
        encoded = tokenizer(text, truncation=True, max_length=64)
        features.append({"input_ids": encoded["input_ids"]})
    return features


class TestDataCollatorForTSDAE:

    def test_output_keys(self, tokenizer, sample_features):
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5)
        batch = collator(sample_features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_output_shapes(self, tokenizer, sample_features):
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5)
        batch = collator(sample_features)
        B = len(sample_features)
        L = batch["input_ids"].shape[1]
        assert batch["input_ids"].shape == (B, L)
        assert batch["attention_mask"].shape == (B, L)
        assert batch["labels"].shape == (B, L)

    def test_input_ids_unchanged(self, tokenizer, sample_features):
        """input_ids must be the clean original tokens, not corrupted."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5)
        batch = collator(sample_features)
        first_ids = sample_features[0]["input_ids"]
        L = min(len(first_ids), batch["input_ids"].shape[1])
        assert batch["input_ids"][0, :L].tolist() == first_ids[:L]

    def test_deletion_reduces_attention_mask(self, tokenizer, sample_features):
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.6)
        batch = collator(sample_features)
        for i in range(len(sample_features)):
            orig_len = len(sample_features[i]["input_ids"])
            orig_len = min(orig_len, batch["input_ids"].shape[1])
            surviving = batch["attention_mask"][i, :orig_len].sum().item()
            assert surviving < orig_len, (
                f"Sequence {i}: all {orig_len} tokens survived despite 0.6 deletion rate"
            )

    def test_at_least_one_token_survives(self, tokenizer, sample_features):
        """Even at extreme deletion rate, at least 1 token must survive."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.99)
        for _ in range(10):
            batch = collator(sample_features)
            for i in range(len(sample_features)):
                surviving = batch["attention_mask"][i].sum().item()
                assert surviving >= 1, f"Sequence {i}: zero surviving tokens"

    def test_special_tokens_not_deleted(self, tokenizer, sample_features):
        """CLS, SEP, PAD tokens should never be deleted."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.99)
        batch = collator(sample_features)
        special_ids = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)

        for i in range(len(sample_features)):
            for pos in range(batch["input_ids"].shape[1]):
                token_id = batch["input_ids"][i, pos].item()
                if token_id in special_ids:
                    mask_val = batch["attention_mask"][i, pos].item()
                    is_pad = (token_id == tokenizer.pad_token_id)
                    if not is_pad:
                        assert mask_val == 1, (
                            f"Special token {token_id} at pos {pos} was deleted"
                        )

    def test_labels_cover_all_non_pad_positions(self, tokenizer, sample_features):
        """Labels should be original token ids at non-pad, -100 at padding."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5)
        batch = collator(sample_features)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for i in range(len(sample_features)):
            for pos in range(batch["labels"].shape[1]):
                token_id = batch["input_ids"][i, pos].item()
                label = batch["labels"][i, pos].item()
                if token_id == pad_id:
                    assert label == -100, f"Pad position should have label -100"
                else:
                    assert label == token_id, (
                        f"Non-pad label {label} != input_id {token_id}"
                    )

    def test_deletion_rate_approximately_correct(self, tokenizer, sample_features):
        """Over many runs, actual deletion rate should be near the configured rate."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5, max_length=128)
        total_deletable = 0
        total_deleted = 0
        for _ in range(50):
            batch = collator(sample_features)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            special_ids = collator._special_ids
            for i in range(len(sample_features)):
                for pos in range(batch["input_ids"].shape[1]):
                    tid = batch["input_ids"][i, pos].item()
                    if tid not in special_ids:
                        total_deletable += 1
                        if batch["attention_mask"][i, pos].item() == 0:
                            total_deleted += 1

        actual_rate = total_deleted / max(total_deletable, 1)
        assert 0.35 < actual_rate < 0.65, (
            f"Expected ~0.5 deletion rate, got {actual_rate:.3f}"
        )

    def test_max_length_truncation(self, tokenizer):
        long_text = "word " * 200
        encoded = tokenizer(long_text, truncation=True, max_length=512)
        features = [{"input_ids": encoded["input_ids"]}]
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.5, max_length=64)
        batch = collator(features)
        assert batch["input_ids"].shape[1] <= 64

    def test_zero_deletion_rate(self, tokenizer, sample_features):
        """With 0% deletion, attention_mask should match original non-pad mask."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.0)
        batch = collator(sample_features)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for i in range(len(sample_features)):
            for pos in range(batch["input_ids"].shape[1]):
                tid = batch["input_ids"][i, pos].item()
                expected = 0 if tid == pad_id else 1
                assert batch["attention_mask"][i, pos].item() == expected

    def test_seeded_collator_is_deterministic(self, tokenizer, sample_features):
        """A seeded collator must produce identical deletions for identical batches."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.6, seed=42)
        batch_a = collator(sample_features)
        batch_b = collator(sample_features)
        assert torch.equal(batch_a["attention_mask"], batch_b["attention_mask"])
        assert torch.equal(batch_a["labels"], batch_b["labels"])

    def test_unseeded_collator_resamples_deletions(self, tokenizer, sample_features):
        """Without a seed, deletion masks should vary across calls."""
        collator = DataCollatorForTSDAE(tokenizer, deletion_rate=0.6)
        masks = [collator(sample_features)["attention_mask"] for _ in range(10)]
        assert any(not torch.equal(masks[0], m) for m in masks[1:])


class _EosOnlyTokenizer:
    """SmolLM2-style stub: <|endoftext|> is bos=eos, pad aliases eos, no cls/sep/mask."""

    eos_token_id = 0
    bos_token_id = 0
    pad_token_id = 0  # pad aliased to eos at runtime
    cls_token_id = None
    sep_token_id = None
    mask_token_id = None
    unk_token_id = 0

    def __len__(self):
        return 100


class TestTSDAECollatorPadAliasesEos:
    """SmolLM2 contract: pad == eos, masking must stay positional, eos stays trainable."""

    def _features(self):
        # Two docs of different length, each ending with the real eos (id 0).
        return [
            {"input_ids": [11, 12, 13, 14, 15, 0]},
            {"input_ids": [21, 22, 0]},
        ]

    def test_real_eos_kept_as_label_and_pad_masked(self):
        collator = DataCollatorForTSDAE(_EosOnlyTokenizer(), deletion_rate=0.6, seed=7)
        batch = collator(self._features())
        labels = batch["labels"]

        # Row 0: full length 6, the trailing eos (pos 5) must be a real target.
        assert labels[0, 5].item() == 0
        # Row 1: content length 3, eos at pos 2 trainable; positions 3+ are padding -> -100.
        assert labels[1, 2].item() == 0
        assert (labels[1, 3:] == -100).all()

    def test_eos_never_deleted_from_encoder_view(self):
        collator = DataCollatorForTSDAE(_EosOnlyTokenizer(), deletion_rate=0.99)
        for _ in range(10):
            batch = collator(self._features())
            # eos is a special token: must remain visible at its real positions.
            assert batch["attention_mask"][0, 5].item() == 1
            assert batch["attention_mask"][1, 2].item() == 1

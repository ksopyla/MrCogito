import sys
import os
import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForWholeWordMask

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.dataset_preprocess import NeighborWordMaskCollator
    _HAS_NEIGHBOR_COLLATOR = True
except ImportError:
    _HAS_NEIGHBOR_COLLATOR = False
from data.data_collators import DataCollatorForPrefixGeneration

_skip_neighbor = pytest.mark.skipif(
    not _HAS_NEIGHBOR_COLLATOR,
    reason="NeighborWordMaskCollator no longer available",
)

@pytest.fixture
def tokenizer():
    """Load a test tokenizer (BERT)"""
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@pytest.fixture
def test_examples():
    """Create test examples for data collator testing"""
    # Sample texts with clear concepts (multi-word phrases)
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Concept encoder models aim to capture semantic meaning across multiple tokens.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    examples = []
    
    for text in texts:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_special_tokens_mask=True
        )
        examples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens_mask": encoded["special_tokens_mask"],
        })
    
    return examples

@_skip_neighbor
def test_neighbor_word_mask_collator_init(tokenizer):
    """Test NeighborWordMaskCollator initialization"""
    # Test default parameters
    collator = NeighborWordMaskCollator(tokenizer)
    assert collator.mlm_probability == 0.25  # Higher than BERT's 0.15
    assert collator.window_size == 3
    
    # Test custom parameters
    collator = NeighborWordMaskCollator(tokenizer, mlm_probability=0.4, window_size=5)
    assert collator.mlm_probability == 0.4
    assert collator.window_size == 5

@_skip_neighbor
def test_neighbor_word_mask_collator_call(tokenizer, test_examples):
    """Test full collator call functionality"""
    collator = NeighborWordMaskCollator(tokenizer, mlm_probability=0.3, window_size=2)
    batch = collator(test_examples)
    
    # Check output shape and types
    assert "input_ids" in batch
    assert "labels" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["input_ids"].shape == batch["labels"].shape
    
    # Check that the right proportion of tokens is masked
    mask_count = (batch["labels"] != -100).sum().item()
    total_tokens = torch.prod(torch.tensor(batch["input_ids"].shape)).item()
    special_tokens_count = batch["special_tokens_mask"].sum().item()
    mask_ratio = mask_count / (total_tokens - special_tokens_count)
    
    # We expect the ratio to be approximately the specified mlm_probability (with some tolerance)
    assert 0.2 <= mask_ratio <= 0.4, f"Mask ratio {mask_ratio} is not within expected range"
    
@_skip_neighbor
def test_neighbor_word_masking_pattern(tokenizer):
    """Test that tokens from neighboring words are masked together"""
    # Create a specific example with multi-token words
    text = "Artificial intelligence and machine learning are related fields."
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_special_tokens_mask=True
    )
    example = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "special_tokens_mask": encoded["special_tokens_mask"],
    }
    
    # Set a high masking probability to ensure good coverage
    collator = NeighborWordMaskCollator(tokenizer, mlm_probability=0.5, window_size=1)
    
    # Process multiple times to check for neighboring pattern
    neighboring_pattern_count = 0
    
    for _ in range(10):  # Run multiple trials
        batch = collator([example])
        labels = batch["labels"][0]
        
        # Find masked positions
        masked_positions = torch.where(labels != -100)[0]
        
        # Check if there are adjacent masked positions
        if len(masked_positions) >= 2:
            diffs = masked_positions[1:] - masked_positions[:-1]
            if (diffs == 1).any():
                neighboring_pattern_count += 1
    
    # We expect to see neighboring patterns in most runs
    assert neighboring_pattern_count >= 5, "Neighbor masking pattern not detected frequently enough"

@_skip_neighbor
def test_masking_respects_word_boundaries(tokenizer):
    """Test that masking respects word boundaries"""
    # Create a text with clear word boundaries
    text = "The concept encoder architecture for language models"
    
    # Tokenize to see word boundaries
    encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    word_ids = encoding.word_ids()
    
    # Prepare an example
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_special_tokens_mask=True
    )
    example = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "special_tokens_mask": encoded["special_tokens_mask"],
    }
    
    # Create a collator
    collator = NeighborWordMaskCollator(tokenizer, mlm_probability=0.4, window_size=1)
    
    # Run multiple trials to check for word boundary respect
    complete_word_count = 0
    
    for _ in range(10):
        batch = collator([example])
        labels = batch["labels"][0]
        
        # Find masked tokens
        masked_positions = torch.where(labels != -100)[0]
        
        # Skip special tokens (CLS, SEP)
        masked_positions = [pos for pos in masked_positions if pos > 0 and pos < len(encoded["input_ids"]) - 1]
        
        if masked_positions:
            # Get word IDs for masked positions
            masked_word_ids = []
            for pos in masked_positions:
                token = encoded["input_ids"][pos]
                token_idx = token_ids.index(token) if token in token_ids else None
                if token_idx is not None:
                    word_id = word_ids[token_idx]
                    if word_id is not None:
                        masked_word_ids.append(word_id)
            
            # Check if all tokens of at least one word are masked
            if masked_word_ids:
                word_id_counts = {}
                for word_id in masked_word_ids:
                    word_id_counts[word_id] = word_id_counts.get(word_id, 0) + 1
                
                # Count word tokens in original text
                word_token_counts = {}
                for i, word_id in enumerate(word_ids):
                    if word_id is not None:
                        word_token_counts[word_id] = word_token_counts.get(word_id, 0) + 1
                
                # Check if any word has all its tokens masked
                for word_id, count in word_id_counts.items():
                    if count == word_token_counts.get(word_id, 0):
                        complete_word_count += 1
                        break
    
    # Expect to see complete word masking in most runs
    assert complete_word_count >= 3, "Word boundary respect not detected frequently enough"

@_skip_neighbor
def test_masking_rate(tokenizer, test_examples):
    """Test that the masking rate is higher than standard BERT masking"""
    # Create both collators
    bert_collator = DataCollatorForWholeWordMask(tokenizer, mlm_probability=0.15)
    neighbor_collator = NeighborWordMaskCollator(tokenizer, mlm_probability=0.25)
    
    # Process examples with both collators
    bert_batch = bert_collator(test_examples)
    neighbor_batch = neighbor_collator(test_examples)
    
    # Calculate masking rates
    bert_mask_count = (bert_batch["labels"] != -100).sum().item()
    neighbor_mask_count = (neighbor_batch["labels"] != -100).sum().item()
    
    # Get total non-special tokens
    total_tokens = torch.prod(torch.tensor(bert_batch["input_ids"].shape)).item()
    special_tokens_count = sum(example["special_tokens_mask"].count(1) for example in test_examples)
    
    bert_mask_ratio = bert_mask_count / (total_tokens - special_tokens_count)
    neighbor_mask_ratio = neighbor_mask_count / (total_tokens - special_tokens_count)
    
    # Check that neighbor masking rate is higher
    assert neighbor_mask_ratio > bert_mask_ratio, f"Neighbor masking ratio {neighbor_mask_ratio} should be higher than BERT masking ratio {bert_mask_ratio}"


# ==========================================================================
# DataCollatorForPrefixGeneration tests
# ==========================================================================

@pytest.fixture
def modern_bert_tokenizer():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def _make_prefix_examples(tokenizer, texts, max_length=64):
    examples = []
    for text in texts:
        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        examples.append({"input_ids": encoded["input_ids"]})
    return examples


class TestDataCollatorForPrefixGeneration:

    def test_output_keys(self, modern_bert_tokenizer):
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(
            modern_bert_tokenizer,
            ["The quick brown fox jumps over the lazy dog in the park."],
        )
        batch = collator(examples)
        expected_keys = {
            "prefix_input_ids", "prefix_attention_mask",
            "suffix_input_ids", "suffix_attention_mask", "labels",
        }
        assert set(batch.keys()) == expected_keys

    def test_output_shapes(self, modern_bert_tokenizer):
        texts = [
            "Machine learning is a subset of artificial intelligence and data science.",
            "The concept encoder models aim to capture semantic meaning across tokens.",
        ]
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, texts)
        batch = collator(examples)

        B = len(texts)
        P = batch["prefix_input_ids"].shape[1]
        S = batch["suffix_input_ids"].shape[1]

        assert batch["prefix_input_ids"].shape == (B, P)
        assert batch["prefix_attention_mask"].shape == (B, P)
        assert batch["suffix_input_ids"].shape == (B, S)
        assert batch["suffix_attention_mask"].shape == (B, S)
        assert batch["labels"].shape == (B, S)

    def test_split_ratios(self, modern_bert_tokenizer):
        """Prefix/suffix content lengths should respect ratio bounds."""
        text = "A " * 50  # ~50 content tokens
        collator = DataCollatorForPrefixGeneration(
            modern_bert_tokenizer, max_length=128,
            prefix_ratio_min=0.3, prefix_ratio_max=0.5,
        )
        examples = _make_prefix_examples(modern_bert_tokenizer, [text], max_length=128)

        raw = modern_bert_tokenizer(text, padding=False, truncation=True, max_length=128)
        content_len = len(raw["input_ids"]) - 2  # minus [CLS] and [SEP]

        for _ in range(20):
            batch = collator(examples)
            # prefix has [CLS] + content + [SEP], so content = real - 2
            prefix_real = batch["prefix_attention_mask"][0].sum().item()
            prefix_content = prefix_real - 2  # minus CLS and SEP
            assert prefix_content >= int(content_len * 0.3) - 1
            assert prefix_content <= int(content_len * 0.5) + 1

    def test_no_information_leak(self, modern_bert_tokenizer):
        """Suffix content tokens should not appear in prefix content and vice versa."""
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, [text])
        batch = collator(examples)

        cls_id = modern_bert_tokenizer.cls_token_id
        sep_id = modern_bert_tokenizer.sep_token_id
        pad_id = modern_bert_tokenizer.pad_token_id

        special = {cls_id, sep_id, pad_id}

        prefix_ids = batch["prefix_input_ids"][0].tolist()
        suffix_ids = batch["suffix_input_ids"][0].tolist()

        prefix_content = set(prefix_ids) - special
        suffix_content = set(suffix_ids) - special

        # Content tokens must be disjoint
        overlap = prefix_content & suffix_content
        assert len(overlap) == 0, f"Leaked tokens: {overlap}"

    def test_special_tokens_prefix(self, modern_bert_tokenizer):
        """Prefix should start with [CLS] and end with [SEP] (before padding)."""
        text = "The quick brown fox jumps over the lazy dog in the park near the river."
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, [text])
        batch = collator(examples)

        prefix = batch["prefix_input_ids"][0]
        mask = batch["prefix_attention_mask"][0]
        real_len = mask.sum().item()

        assert prefix[0].item() == modern_bert_tokenizer.cls_token_id
        assert prefix[real_len - 1].item() == modern_bert_tokenizer.sep_token_id

    def test_special_tokens_suffix(self, modern_bert_tokenizer):
        """Suffix should end with [SEP] (before padding) and NOT start with [CLS]."""
        text = "The quick brown fox jumps over the lazy dog in the park near the river."
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, [text])
        batch = collator(examples)

        suffix = batch["suffix_input_ids"][0]
        mask = batch["suffix_attention_mask"][0]
        real_len = mask.sum().item()

        assert suffix[real_len - 1].item() == modern_bert_tokenizer.sep_token_id
        assert suffix[0].item() != modern_bert_tokenizer.cls_token_id

    def test_labels_padding(self, modern_bert_tokenizer):
        """Labels should be -100 at pad positions, actual token ids elsewhere."""
        texts = [
            "Short text here.",
            "A significantly longer piece of text to create different lengths in this batch for padding.",
        ]
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, texts)
        batch = collator(examples)

        labels = batch["labels"]
        suffix_mask = batch["suffix_attention_mask"]
        suffix_ids = batch["suffix_input_ids"]

        # Where mask is 1, labels == suffix_input_ids
        real_positions = suffix_mask == 1
        assert (labels[real_positions] == suffix_ids[real_positions]).all()

        # Where mask is 0, labels == -100
        pad_positions = suffix_mask == 0
        if pad_positions.any():
            assert (labels[pad_positions] == -100).all()

    def test_sep_in_suffix_labels(self, modern_bert_tokenizer):
        """The [SEP] end-of-sequence marker should appear in suffix labels."""
        text = "The quick brown fox jumps over the lazy dog in the park."
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=64)
        examples = _make_prefix_examples(modern_bert_tokenizer, [text])
        batch = collator(examples)

        labels = batch["labels"][0]
        sep_id = modern_bert_tokenizer.sep_token_id
        real_labels = labels[labels != -100]
        assert sep_id in real_labels.tolist()

    def test_minimum_lengths(self, modern_bert_tokenizer):
        """Very short sequences should still produce valid prefix and suffix."""
        text = "Hi."
        collator = DataCollatorForPrefixGeneration(
            modern_bert_tokenizer, max_length=64,
            min_prefix_content=2, min_suffix_content=1,
        )
        examples = _make_prefix_examples(modern_bert_tokenizer, [text])
        batch = collator(examples)

        prefix_real = batch["prefix_attention_mask"][0].sum().item()
        suffix_real = batch["suffix_attention_mask"][0].sum().item()
        # At minimum: prefix has [CLS] + 1 content + [SEP] = 3
        # suffix has 1 content + [SEP] = 2
        assert prefix_real >= 3
        assert suffix_real >= 2

    def test_dynamic_padding(self, modern_bert_tokenizer):
        """Batch padding should adapt to actual lengths, not always max_length."""
        texts = [
            "Short text.",
            "Another short text here.",
        ]
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=512)
        examples = _make_prefix_examples(modern_bert_tokenizer, texts, max_length=512)
        batch = collator(examples)

        P = batch["prefix_input_ids"].shape[1]
        S = batch["suffix_input_ids"].shape[1]
        assert P < 512, f"Prefix padded to max_length ({P}) instead of dynamic"
        assert S < 512, f"Suffix padded to max_length ({S}) instead of dynamic"

    def test_variable_length_batch(self, modern_bert_tokenizer):
        """Different-length sequences in a batch should all produce valid splits."""
        texts = [
            "Hi there.",
            "The quick brown fox jumps over the lazy dog in the park near the river bank.",
            "Medium length sentence for testing purposes only.",
        ]
        collator = DataCollatorForPrefixGeneration(modern_bert_tokenizer, max_length=128)
        examples = _make_prefix_examples(modern_bert_tokenizer, texts, max_length=128)
        batch = collator(examples)

        for i in range(len(texts)):
            prefix_real = batch["prefix_attention_mask"][i].sum().item()
            suffix_real = batch["suffix_attention_mask"][i].sum().item()
            assert prefix_real >= 3, f"Example {i}: prefix too short ({prefix_real})"
            assert suffix_real >= 2, f"Example {i}: suffix too short ({suffix_real})"

    def test_requires_sep_token(self):
        """Collator should raise ValueError if tokenizer has no sep_token_id."""

        class FakeTokenizer:
            pad_token_id = 0
            cls_token_id = 1
            sep_token_id = None

        with pytest.raises(ValueError, match="sep_token_id"):
            DataCollatorForPrefixGeneration(FakeTokenizer())

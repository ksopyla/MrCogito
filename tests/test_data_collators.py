import sys
import os
import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForWholeWordMask

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset_preprocess import NeighborWordMaskCollator

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

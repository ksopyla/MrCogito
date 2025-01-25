import pytest
import torch

from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoder,
    ConceptEncoderForMaskedLM,
)

@pytest.fixture
def small_encoder_config():
    """
    Returns a ConceptEncoderConfig with small sizes for quick tests.
    By centralizing the config here, we avoid duplication across different tests.
    """
    
    
        # Adjust special tokens to be within vocab_size
    PAD_TOKEN_ID = 0    # [PAD]
    CLS_TOKEN_ID = 1    # [CLS] [bos]
    SEP_TOKEN_ID = 2    # [SEP] [eos]
    MASK_TOKEN_ID = 3   # [MASK]
    return ConceptEncoderConfig(
        vocab_size=16,
        concept_size=4,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=SEP_TOKEN_ID,
        bos_token_id=CLS_TOKEN_ID,
        cls_token_id=CLS_TOKEN_ID,
        sep_token_id=SEP_TOKEN_ID,
        mask_token_id=MASK_TOKEN_ID,
    )

@pytest.fixture
def masked_lm_input_data(small_encoder_config):
    """
    Fixture that prepares minimal masked language modeling inputs:
     - Uses token IDs within vocab_size range
     - Special tokens at the start of vocab ([PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3)
     - Proper attention masking for padded sequences
     - Labels representing the original unmasked sequence
    Returns: (input_ids, attention_mask, labels)
    """
    # Adjust special tokens to be within vocab_size
    PAD_TOKEN_ID = small_encoder_config.pad_token_id    # [PAD]
    CLS_TOKEN_ID = small_encoder_config.cls_token_id    # [CLS]
    SEP_TOKEN_ID = small_encoder_config.sep_token_id    # [SEP]
    MASK_TOKEN_ID = small_encoder_config.mask_token_id   # [MASK]
    
    # Create labels with tokens in range [4, vocab_size-1] for normal tokens
    # First create the labels (original sequence)
    labels = torch.tensor([
        [CLS_TOKEN_ID, 4, 5, 6, 7, 8, 9, SEP_TOKEN_ID],      # First sequence uses tokens 4-9
        [CLS_TOKEN_ID, 5, 6, 7, SEP_TOKEN_ID, 0, 0, 0],      # Second sequence, shorter with padding
    ], dtype=torch.long)
    
    # Create model inputs by masking one token from each sequence
    model_input_ids = labels.clone()  
    # Mask the third token (index 2) in each sequence
    model_input_ids[:, 2] = MASK_TOKEN_ID  # Replace with [MASK] token

    # Create attention mask: 1 for real tokens (including [CLS], [SEP], [MASK])
    # and 0 for padding tokens
    attention_mask = (labels != PAD_TOKEN_ID).long()

    return model_input_ids, attention_mask, labels

def test_concept_encoder_returns_last_hidden_state_shape(small_encoder_config, masked_lm_input_data):
    """
    Test that the ConceptEncoder (with a small config) returns a last_hidden_state
    of shape [batch_size, concept_size, hidden_size].
    """
    model = ConceptEncoder(small_encoder_config)
    model.eval()

    input_ids, attention_mask, labels = masked_lm_input_data
    
    batch_size = input_ids.size(0)
    seq_length = input_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state

    assert last_hidden_state.shape == (
        batch_size, 
        small_encoder_config.concept_size, 
        small_encoder_config.hidden_size
    ), (
        f"Expected shape ({batch_size}, {small_encoder_config.concept_size}, "
        f"{small_encoder_config.hidden_size}), got {last_hidden_state.shape}"
    )
    
@pytest.mark.parametrize(
    "batch_size,seq_length,vocab_size,concept_size, num_attn_heads",
    [
        (1, 6, 10, 4, 1), # check concept size vs attention heads
        (1, 6, 10, 4, 2), # check concept size vs attention heads
        (1, 6, 10, 4, 4), # check concept size vs attention heads
        (2, 6, 10, 4, 1), # check concept size vs attention heads, batch size 
        (2, 6, 10, 4, 2), # check concept size vs attention heads, batch size 
        (2, 6, 10, 4, 4), # check concept size vs attention heads, batch size   
    ]
)
def test_concept_encoder_attention_heads_configurations(batch_size, seq_length, vocab_size, concept_size, num_attn_heads):
    """
    Test the ConceptEncoder with various configurations to ensure the basic shape is produced for one layer encoder:
      (batch_size, concept_size, hidden_size).
    This test is useful for verifying that our shaping logic is correct, 
    even with different config parameters.
    """
    config = ConceptEncoderConfig(
        vocab_size=vocab_size,
        concept_size=concept_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=num_attn_heads,
        intermediate_size=16,
    )
    model = ConceptEncoder(config)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

    expected_shape = (batch_size, concept_size, config.hidden_size)
    assert last_hidden_state.shape == expected_shape, (
        f"Expected {expected_shape}, got {last_hidden_state.shape}"
    )

def test_concept_encoder_for_masked_lm_small(small_encoder_config, masked_lm_input_data):
    """
    Test that ConceptEncoderForMaskedLM runs properly with:
      - A small config
      - Minimal batch/sequence dimension
      - Special tokens included for LM
    Verifies (loss, logits) shape when labels exist and 
    verifies (logits,) shape when labels are None.
    """
    model = ConceptEncoderForMaskedLM(small_encoder_config)
    model.eval()

    input_ids, attention_mask, labels = masked_lm_input_data

    # With labels => (loss, logits)
    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
        assert loss is not None, "Loss should be returned when labels are provided."
        assert logits.shape == (
            input_ids.size(0),
            small_encoder_config.concept_size,
            small_encoder_config.vocab_size,
        ), "Logits shape mismatch with labels provided."

    # Without labels => (logits,)
    with torch.no_grad():
        outputs_no_labels = model(input_ids, attention_mask=attention_mask)
        assert len(outputs_no_labels) == 1, "Should return just a single-element tuple with logits."
        logits_only = outputs_no_labels[0]
        assert logits_only.shape == (
            input_ids.size(0),
            small_encoder_config.concept_size,
            small_encoder_config.vocab_size,
        ), "Logits shape mismatch when no labels are provided."



@pytest.mark.parametrize("num_heads", [2, 4])
def test_multihead_attention_variants(num_heads, small_encoder_config, masked_lm_input_data):
    """
    Test that multi-head attention runs with 2 and 4 heads
    in a single-layer ConceptEncoder, confirming the final outputs shape.
    """
    config = small_encoder_config
    config.num_attention_heads = num_heads
    
    
    model = ConceptEncoder(config)
    model.eval()

    input_ids, attention_mask, labels = masked_lm_input_data
    batch_size = input_ids.size(0)
    seq_length = input_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

    expected_shape = (batch_size, config.concept_size, config.hidden_size)
    assert last_hidden_state.shape == expected_shape, (
        f"Got shape {last_hidden_state.shape}, expected {expected_shape}"
    )

import pytest
import torch

from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoder,
    ConceptEncoderForMaskedLM,
)

@pytest.mark.parametrize("batch_size, seq_length", [(1, 2), (2, 3)])
def test_concept_encoder_small_shapes(batch_size, seq_length):
    """
    A simplified test that uses a very small ConceptEncoderConfig and checks
    that the final hidden state has shape [batch_size, concept_size, hidden_size].
    """
    config = ConceptEncoderConfig(
        vocab_size=16,      # small vocab
        concept_size=4,     # few concept tokens
        hidden_size=8,      # small hidden dimension
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
    )
    model = ConceptEncoder(config)
    model.eval()

    # Random input of shape [batch_size, seq_length]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    # Provide a simple 2D mask of 1's
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64)

    with torch.no_grad():
        # model(...) returns a BaseModelOutput
        outputs = model(input_ids, attention_mask=attention_mask)
        # Grab the last_hidden_state explicitly
        hidden_states = outputs.last_hidden_state

    assert hidden_states.shape == (batch_size, config.concept_size, config.hidden_size), (
        f"Expected shape {(batch_size, config.concept_size, config.hidden_size)}, "
        f"got {hidden_states.shape}"
    )

def test_concept_encoder_for_masked_lm_small():
    """
    A simplified masked LM test that verifies shape of (loss, logits).
    Uses minimal batch/sequence size and small config.
    """
    config = ConceptEncoderConfig(
        vocab_size=16,
        concept_size=4,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
    )
    model = ConceptEncoderForMaskedLM(config)
    model.eval()

    # Minimal batch and seq
    batch_size, seq_length = 2, 3
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64)
    labels = torch.randint(0, config.vocab_size, (batch_size, config.concept_size))

    # When labels are provided => (loss, logits)
    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
        assert loss is not None, "Loss should be returned when labels are given."
        assert logits.shape == (batch_size, config.concept_size, config.vocab_size), (
            "Logits shape should match [batch_size, concept_size, vocab_size]."
        )

    # Without labels => (logits,)
    with torch.no_grad():
        outputs_no_labels = model(input_ids, attention_mask=attention_mask)
        assert len(outputs_no_labels) == 1, "Should return just a tuple with logits."
        logits_only = outputs_no_labels[0]
        assert logits_only.shape == (batch_size, config.concept_size, config.vocab_size), (
            "Logits shape should match [batch_size, concept_size, vocab_size] with no labels."
        )

@pytest.mark.parametrize(
    "batch_size,seq_length,vocab_size,concept_size",
    [
        (1, 2, 10, 4),
        (2, 3, 12, 2),
    ]
)
def test_small_config_encoder_outputs(batch_size, seq_length, vocab_size, concept_size):
    """
    Test the ConceptEncoder with small input sizes and custom vocab_size/concept_size
    so we can manually verify the encoder output.
    """
    config = ConceptEncoderConfig(
        vocab_size=vocab_size,
        concept_size=concept_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
    )
    model = ConceptEncoder(config)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # "outputs" is a BaseModelOutput, so let's get its last_hidden_state:
        last_hidden_state = outputs.last_hidden_state

    # Check shape => [batch_size, concept_size, hidden_size]
    assert last_hidden_state.shape == (batch_size, concept_size, config.hidden_size), (
        f"Expected output shape {(batch_size, concept_size, config.hidden_size)}, "
        f"got {last_hidden_state.shape}"
    )

    # Placeholder for future numeric checks, e.g.:
    # expected_output = torch.tensor([...], dtype=outputs.dtype)
    # assert torch.allclose(outputs, expected_output, atol=1e-4), "Outputs differ from expected reference"

@pytest.mark.parametrize("num_heads", [2, 4])
def test_multihead_attention_variants(num_heads):
    """
    Test that multi-head attention runs with 2 and 4 heads,
    verifying the final shape is correct.
    """
    config = ConceptEncoderConfig(
        vocab_size=16,
        concept_size=4,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        intermediate_size=16,
    )
    model = ConceptEncoder(config)
    model.eval()

    # Minimal random input
    batch_size, seq_length = 2, 3
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

    expected_shape = (batch_size, config.concept_size, config.hidden_size)
    assert last_hidden_state.shape == expected_shape, (
        f"Got shape {last_hidden_state.shape}, expected {expected_shape}"
    )

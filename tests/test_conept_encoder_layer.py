import pytest
import torch

from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoder,
    ConceptEncoderForMaskedLM,
)

@pytest.mark.parametrize("batch_size,seq_length", [(2, 5), (3, 8)])
def test_concept_encoder_outputs_consistency(batch_size, seq_length):
    """
    Test that the ConceptEncoder produces the same outputs when attention_mask
    is logically identical but dimensionally reorganized (e.g. 2D vs expanded
    3D or post-reshaped).
    """
    config = ConceptEncoderConfig()
    model = ConceptEncoder(config)
    model.eval()

    # Generate random input and a binary attention mask
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_length)
    )
    attention_mask_2d = torch.ones((batch_size, seq_length), dtype=torch.int64)

    # Forward pass #1 with straightforward 2D mask
    with torch.no_grad():
        out1 = model(input_ids, attention_mask=attention_mask_2d)

    # Reorganize attention_mask: individually expand dims to (batch_size, 1, seq_length)
    attention_mask_3d = attention_mask_2d.unsqueeze(1).clone()
    # Forward pass #2
    with torch.no_grad():
        out2 = model(input_ids, attention_mask=attention_mask_3d)

    # Check shape and numerical equivalence
    assert out1.shape == out2.shape, "Outputs differ in shape"
    assert torch.allclose(out1, out2, atol=1e-6), "Outputs differ numerically"

@pytest.mark.parametrize("batch_size,seq_length", [(2, 5), (2, 10)])
def test_concept_encoder_for_masked_lm(batch_size, seq_length):
    """
    Simple test to ensure ConceptEncoderForMaskedLM can run forward with random data
    and produce the correct shape for logits, as well as a valid loss if labels are provided.
    """
    config = ConceptEncoderConfig()
    model = ConceptEncoderForMaskedLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    labels = torch.randint(0, config.vocab_size, (batch_size, config.concept_size))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # outputs => (loss, logits)
        loss, logits = outputs
        assert loss is not None, "Loss should not be None when labels are provided."
        assert logits.shape == (
            batch_size,
            config.concept_size,
            config.vocab_size,
        ), "Unexpected logits shape"

        # Also check the no-labels scenario
        outputs_without_labels = model(input_ids, attention_mask=attention_mask)
        # outputs_without_labels => (logits,)
        logits_only = outputs_without_labels[0]
        assert logits_only.shape == (
            batch_size,
            config.concept_size,
            config.vocab_size,
        ), "Unexpected logits shape with no labels."

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
    so we can manually verify the encoder output. This test checks:
    - Output shape matches (batch_size, concept_size, hidden_size)
    - (Optional placeholder) We can later compare against known expected values.
    """
    # Use a very small hidden_size and minimal layers for easy inspection
    config = ConceptEncoderConfig(
        vocab_size=vocab_size,
        concept_size=concept_size,
        hidden_size=8,          # keep it small for easy manual comparison
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,   # small feedforward layer
    )
    model = ConceptEncoder(config)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    # Use a simple full-1 attention mask (not testing attention_mask logic here)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Check shape => [batch_size, concept_size, hidden_size]
    assert outputs.shape == (batch_size, concept_size, config.hidden_size), (
        f"Expected output shape {(batch_size, concept_size, config.hidden_size)}, "
        f"got {outputs.shape}"
    )

    # Placeholder for future numeric checks, e.g.:
    # expected_output = torch.tensor([...], dtype=outputs.dtype)
    # assert torch.allclose(outputs, expected_output, atol=1e-4), "Outputs differ from expected reference"

"""
Verify backward compatibility and new features for Dimension Inversion + Concept Position Encoding.

Tests:
1. Backward compat: Config without new fields defaults correctly
2. Backward compat: Model with default config produces same structure as before
3. Dimension Inversion: token_embedding_dim < hidden_size works correctly
4. Concept Position Encoding: sinusoidal and learned variants work
5. Weight tying: correctly disabled when token_dim != hidden_size
6. All model variants (perceiver_mlm, posonly, weighted, ViaDecoder) work with new config

Usage:
    python verification/verify_dimension_inversion.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoder
from nn.concept_encoder_perceiver import (
    ConceptEncoderForMaskedLMPerceiver,
    ConceptEncoderForMaskedLMPerceiverPosOnly,
    ConceptEncoderForSequenceClassificationPerceiver,
    ConceptEncoderForSequenceClassificationViaDecoder,
)
from nn.concept_encoder_weighted import (
    ConceptEncoderForMaskedLMWeighted,
    ConceptEncoderForSequenceClassificationWeighted,
)


# Common test dimensions
VOCAB_SIZE = 100
CONCEPT_NUM = 8
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_SEQ_LEN = 50
INTERMEDIATE_SIZE = 64
BATCH_SIZE = 2
SEQ_LEN = 20


def make_config(**overrides):
    """Create a test config with sensible defaults, allowing overrides."""
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        concept_num=CONCEPT_NUM,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        pad_token_id=0,
        mask_token_id=3,
    )
    defaults.update(overrides)
    return ConceptEncoderConfig(**defaults)


def make_input(batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    """Create dummy input tensors."""
    input_ids = torch.randint(4, VOCAB_SIZE, (batch_size, seq_len))  # Avoid special tokens 0-3
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    # Mask ~15% of positions
    mask = torch.rand(batch_size, seq_len) < 0.15
    labels[~mask] = -100
    return input_ids, attention_mask, labels


def test_backward_compat_config():
    """Config without new fields should default to hidden_size and 'none'."""
    print("Test 1: Backward compat config defaults...")
    config = make_config()
    
    assert config.token_embedding_dim == HIDDEN_SIZE, \
        f"Expected token_embedding_dim={HIDDEN_SIZE}, got {config.token_embedding_dim}"
    assert config.concept_position_type == "none", \
        f"Expected concept_position_type='none', got {config.concept_position_type}"
    
    # Explicit None should also default
    config2 = make_config(token_embedding_dim=None)
    assert config2.token_embedding_dim == HIDDEN_SIZE
    
    print("  PASSED: Config defaults are backward compatible")


def test_backward_compat_model_structure():
    """Model with default config should have NO projection layers."""
    print("Test 2: Backward compat model structure...")
    config = make_config()
    
    encoder = ConceptEncoder(config)
    assert encoder.token_projection is None, "Default config should not create token_projection"
    assert not hasattr(encoder, 'concept_position_emb') or \
           (config.concept_position_type == "none"), \
        "Default config should not create concept_position_emb"
    
    # Token embeddings should be in hidden_size space
    assert encoder.token_embeddings.embedding_dim == HIDDEN_SIZE
    assert encoder.token_position_embeddings.embedding_dim == HIDDEN_SIZE
    
    # Test forward pass
    input_ids, attention_mask, _ = make_input()
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    assert output.last_hidden_state.shape == (BATCH_SIZE, CONCEPT_NUM, HIDDEN_SIZE)
    
    print("  PASSED: No projection layers with default config")


def test_dimension_inversion():
    """token_embedding_dim < hidden_size should create projection layers."""
    print("Test 3: Dimension Inversion...")
    TOKEN_DIM = 8  # Much smaller than HIDDEN_SIZE=32
    config = make_config(token_embedding_dim=TOKEN_DIM)
    
    assert config.token_embedding_dim == TOKEN_DIM
    
    encoder = ConceptEncoder(config)
    assert encoder.token_projection is not None, "Should create token_projection when dims differ"
    assert encoder.token_embeddings.embedding_dim == TOKEN_DIM
    assert encoder.token_position_embeddings.embedding_dim == TOKEN_DIM
    assert encoder.token_projection.in_features == TOKEN_DIM
    assert encoder.token_projection.out_features == HIDDEN_SIZE
    
    # Concept embeddings should still be in hidden_size space
    assert encoder.concept_embeddings.embedding_dim == HIDDEN_SIZE
    
    # Forward pass
    input_ids, attention_mask, _ = make_input()
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    assert output.last_hidden_state.shape == (BATCH_SIZE, CONCEPT_NUM, HIDDEN_SIZE)
    
    print(f"  PASSED: Dimension Inversion works (token_dim={TOKEN_DIM}, hidden_size={HIDDEN_SIZE})")


def test_concept_position_sinusoidal():
    """Sinusoidal concept positions should add fixed embeddings."""
    print("Test 4a: Sinusoidal concept positions...")
    config = make_config(concept_position_type="sinusoidal")
    
    encoder = ConceptEncoder(config)
    assert hasattr(encoder, 'concept_position_emb'), "Should have concept_position_emb buffer"
    assert encoder.concept_position_emb.shape == (CONCEPT_NUM, HIDDEN_SIZE)
    # Sinusoidal embeddings should be a buffer, not a parameter
    assert 'concept_position_emb' not in dict(encoder.named_parameters()), \
        "Sinusoidal should be a buffer, not a parameter"
    
    # Forward pass
    input_ids, attention_mask, _ = make_input()
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    assert output.last_hidden_state.shape == (BATCH_SIZE, CONCEPT_NUM, HIDDEN_SIZE)
    
    print("  PASSED: Sinusoidal concept positions work")


def test_concept_position_learned():
    """Learned concept positions should add trainable embeddings."""
    print("Test 4b: Learned concept positions...")
    config = make_config(concept_position_type="learned")
    
    encoder = ConceptEncoder(config)
    assert isinstance(encoder.concept_position_emb, torch.nn.Embedding)
    assert encoder.concept_position_emb.num_embeddings == CONCEPT_NUM
    assert encoder.concept_position_emb.embedding_dim == HIDDEN_SIZE
    
    # Forward pass
    input_ids, attention_mask, _ = make_input()
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    assert output.last_hidden_state.shape == (BATCH_SIZE, CONCEPT_NUM, HIDDEN_SIZE)
    
    print("  PASSED: Learned concept positions work")


def test_perceiver_mlm_with_inversion():
    """Perceiver MLM should work with Dimension Inversion."""
    print("Test 5: Perceiver MLM with Dimension Inversion...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, tie_word_embeddings=False)
    
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.eval()
    
    # Check decoder input projection exists
    assert model.decoder_input_projection is not None, \
        "Should have decoder_input_projection when token_dim != hidden_size"
    assert model.decoder_input_projection.in_features == TOKEN_DIM
    assert model.decoder_input_projection.out_features == HIDDEN_SIZE
    
    # Forward pass with labels (sparse decoding)
    input_ids, attention_mask, labels = make_input()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert output.loss is not None
    assert output.loss.dim() == 0  # scalar
    
    # Forward pass without labels (full decoding)
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    assert output.logits is not None
    assert output.logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    print(f"  PASSED: Perceiver MLM works with token_dim={TOKEN_DIM}")


def test_posonly_mlm_with_inversion():
    """PosOnly MLM should work with Dimension Inversion (no decoder input projection needed)."""
    print("Test 6: PosOnly MLM with Dimension Inversion...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, tie_word_embeddings=False)
    
    model = ConceptEncoderForMaskedLMPerceiverPosOnly(config)
    model.eval()
    
    # PosOnly doesn't use input embeddings in decoder, so no decoder_input_projection
    assert not hasattr(model, 'decoder_input_projection') or model.decoder_input_projection is None
    
    # Forward pass
    input_ids, attention_mask, labels = make_input()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert output.loss is not None
    
    print(f"  PASSED: PosOnly MLM works with token_dim={TOKEN_DIM}")


def test_weighted_mlm_with_inversion():
    """Weighted MLM should work with Dimension Inversion."""
    print("Test 7: Weighted MLM with Dimension Inversion...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, tie_word_embeddings=False)
    
    model = ConceptEncoderForMaskedLMWeighted(config)
    model.eval()
    
    # Forward pass
    input_ids, attention_mask, labels = make_input()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert output.loss is not None
    
    print(f"  PASSED: Weighted MLM works with token_dim={TOKEN_DIM}")


def test_via_decoder_classification():
    """ViaDecoder classification should work with Dimension Inversion."""
    print("Test 8: ViaDecoder classification with Dimension Inversion...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, num_labels=3)
    
    model = ConceptEncoderForSequenceClassificationViaDecoder(config)
    model.eval()
    
    # Check decoder input projection
    assert model.decoder_input_projection is not None
    
    # Forward pass
    input_ids, attention_mask, _ = make_input()
    cls_labels = torch.randint(0, 3, (BATCH_SIZE,))
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=cls_labels)
    assert output.loss is not None
    assert output.logits.shape == (BATCH_SIZE, 3)
    
    print(f"  PASSED: ViaDecoder classification works with token_dim={TOKEN_DIM}")


def test_weight_tying_disabled_with_inversion():
    """Weight tying should be disabled when token_dim != hidden_size."""
    print("Test 9: Weight tying disabled with Dimension Inversion...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, tie_word_embeddings=True)
    
    model = ConceptEncoderForMaskedLMPerceiver(config)
    
    # lm_head should NOT be tied to token_embeddings (different dims)
    assert model.lm_head.weight.shape != model.encoder.token_embeddings.weight.shape, \
        "lm_head and token_embeddings should have different shapes"
    assert model.lm_head.weight.data_ptr() != model.encoder.token_embeddings.weight.data_ptr(), \
        "Weights should NOT be tied when dims differ"
    
    print("  PASSED: Weight tying correctly disabled with Dimension Inversion")


def test_weight_tying_works_without_inversion():
    """Weight tying should still work when token_dim == hidden_size."""
    print("Test 10: Weight tying works without Dimension Inversion...")
    config = make_config(tie_word_embeddings=True)  # token_dim == hidden_size
    
    model = ConceptEncoderForMaskedLMPerceiver(config)
    
    # lm_head SHOULD be tied to token_embeddings
    assert model.lm_head.weight.data_ptr() == model.encoder.token_embeddings.weight.data_ptr(), \
        "Weights should be tied when token_dim == hidden_size"
    
    print("  PASSED: Weight tying works correctly without Dimension Inversion")


def test_combined_inversion_and_position():
    """Both Dimension Inversion and concept position encoding together."""
    print("Test 11: Combined Dimension Inversion + concept positions...")
    TOKEN_DIM = 8
    config = make_config(token_embedding_dim=TOKEN_DIM, concept_position_type="sinusoidal")
    
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.eval()
    
    input_ids, attention_mask, labels = make_input()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert output.loss is not None
    
    print(f"  PASSED: Combined features work (token_dim={TOKEN_DIM}, pos=sinusoidal)")


def test_param_count_reduction():
    """Dimension Inversion should reduce total parameters."""
    print("Test 12: Parameter count reduction with Dimension Inversion...")
    
    config_baseline = make_config()
    config_inverted = make_config(token_embedding_dim=8)
    
    model_baseline = ConceptEncoderForMaskedLMPerceiver(config_baseline)
    model_inverted = ConceptEncoderForMaskedLMPerceiver(config_inverted)
    
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_inverted = sum(p.numel() for p in model_inverted.parameters())
    
    print(f"  Baseline params:  {params_baseline:,}")
    print(f"  Inverted params:  {params_inverted:,}")
    print(f"  Reduction:        {params_baseline - params_inverted:,} ({100*(1-params_inverted/params_baseline):.1f}%)")
    
    # token_embeddings go from [100, 32] to [100, 8] = saving 2400
    # position_embeddings go from [50, 32] to [50, 8] = saving 1200
    # But we add projection [8, 32] = +256 and decoder_input_projection [8, 32] = +256
    # Net savings should be > 0
    assert params_inverted < params_baseline, \
        "Dimension Inversion should reduce total params"
    
    print("  PASSED: Dimension Inversion reduces parameters")


if __name__ == "__main__":
    print("=" * 60)
    print("Dimension Inversion & Concept Position Encoding Tests")
    print("=" * 60)
    
    tests = [
        test_backward_compat_config,
        test_backward_compat_model_structure,
        test_dimension_inversion,
        test_concept_position_sinusoidal,
        test_concept_position_learned,
        test_perceiver_mlm_with_inversion,
        test_posonly_mlm_with_inversion,
        test_weighted_mlm_with_inversion,
        test_via_decoder_classification,
        test_weight_tying_disabled_with_inversion,
        test_weight_tying_works_without_inversion,
        test_combined_inversion_and_position,
        test_param_count_reduction,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! Backward compatibility verified.")

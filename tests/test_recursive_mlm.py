"""Verify recursive_mlm model builds, runs forward pass, and is registered in mlm_training."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nn.concept_encoder_recursive_mlm import RecursiveConceptEncoderForMaskedLM
from nn.concept_encoder_recursive import RecursiveConceptEncoderConfig


def test_model_creation():
    cfg = RecursiveConceptEncoderConfig(
        num_hidden_layers=6, hidden_size=512, intermediate_size=2048,
        concept_num=128, vocab_size=50368, max_sequence_length=512,
    )
    m = RecursiveConceptEncoderForMaskedLM(cfg)
    total_p = sum(pp.numel() for pp in m.parameters())
    encoder_p = sum(pp.numel() for pp in m.encoder.parameters())
    print(f"recursive_mlm total params: {total_p:,}")
    print(f"  encoder params: {encoder_p:,}")
    print(f"  decoder+head params: {total_p - encoder_p:,}")
    assert hasattr(m.encoder, "shared_layer"), "Encoder must have shared_layer"
    assert encoder_p < 35_000_000, f"Expected <35M encoder params (embeddings+1 shared layer), got {encoder_p:,}"
    return m, cfg


def test_forward_pass(m):
    m.eval()
    ids = torch.randint(0, 50368, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.int)
    labels = ids.clone()
    labels[labels != 4] = -100  # mask most positions
    labels[:, 5:10] = ids[:, 5:10]

    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, labels=labels)
    print(f"Loss: {out.loss.item():.4f}")
    print(f"Logits shape: {out.logits.shape}")
    assert out.loss is not None
    print("Forward pass OK")


def test_registry():
    from training.mlm_training import MODEL_REGISTRY
    assert "recursive_mlm" in MODEL_REGISTRY, "recursive_mlm not in MODEL_REGISTRY"
    entry = MODEL_REGISTRY["recursive_mlm"]
    assert entry["class"] is RecursiveConceptEncoderForMaskedLM
    assert entry["config_class"] is RecursiveConceptEncoderConfig
    print("MODEL_REGISTRY registration OK")


def test_test_time_scaling(m, cfg):
    m.eval()
    ids = torch.randint(0, 50368, (1, 32))
    mask = torch.ones(1, 32, dtype=torch.int)
    labels = ids.clone()
    labels[:, :5] = -100

    cfg.num_iterations = 12
    with torch.no_grad():
        out12 = m(input_ids=ids, attention_mask=mask, labels=labels)
    print(f"K=12 loss: {out12.loss.item():.4f}")

    cfg.num_iterations = None  # reset to default (num_hidden_layers=6)
    with torch.no_grad():
        out6 = m(input_ids=ids, attention_mask=mask, labels=labels)
    print(f"K=6  loss: {out6.loss.item():.4f}")
    print("Test-time scaling OK")


if __name__ == "__main__":
    m, cfg = test_model_creation()
    test_forward_pass(m)
    test_registry()
    test_test_time_scaling(m, cfg)
    print("\nAll recursive_mlm tests passed!")

"""Quick verification that standard and recursive encoders are fully independent."""
import torch
from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoder
from nn.concept_encoder_recursive import RecursiveConceptEncoderConfig, RecursiveConceptEncoder


def test_standard_encoder_unchanged():
    cfg = ConceptEncoderConfig(num_hidden_layers=6)
    m = ConceptEncoder(cfg)
    p = sum(p.numel() for p in m.parameters())
    print(f"Standard:  {len(m.layers)} layers, {p:,} params")
    assert len(m.layers) == 6
    assert not hasattr(cfg, "weight_tied_layers")
    assert not hasattr(cfg, "num_iterations")
    print("  Config is clean (no recursive fields)")

    ids = torch.randint(0, 30522, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.int)
    out = m(ids, mask)
    assert out.last_hidden_state.shape == (2, 128, 512)
    print(f"  Forward OK: {out.last_hidden_state.shape}")


def test_recursive_encoder():
    cfg = RecursiveConceptEncoderConfig(num_hidden_layers=6)
    m = RecursiveConceptEncoder(cfg)
    p = sum(p.numel() for p in m.parameters())
    print(f"Recursive: 1 shared layer, {p:,} params")
    assert hasattr(m, "shared_layer")
    assert not hasattr(m, "layers")

    ids = torch.randint(0, 30522, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.int)
    out = m(ids, mask)
    assert out.last_hidden_state.shape == (2, 128, 512)
    print(f"  Forward OK (6 iters): {out.last_hidden_state.shape}")

    cfg.num_iterations = 12
    out2 = m(ids, mask)
    assert out2.last_hidden_state.shape == (2, 128, 512)
    print(f"  Forward OK (12 iters): {out2.last_hidden_state.shape}")


def test_param_savings():
    cfg1 = ConceptEncoderConfig(num_hidden_layers=6)
    cfg2 = RecursiveConceptEncoderConfig(num_hidden_layers=6)
    m1 = ConceptEncoder(cfg1)
    m2 = RecursiveConceptEncoder(cfg2)
    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    savings = (p1 - p2) / p1 * 100
    print(f"Savings: {p1:,} -> {p2:,} ({savings:.1f}%)")
    assert p2 < p1


if __name__ == "__main__":
    test_standard_encoder_unchanged()
    test_recursive_encoder()
    test_param_savings()
    print("\nAll tests passed!")

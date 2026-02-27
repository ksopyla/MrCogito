"""
Verify that sparse MLM decoding produces identical loss to full decoding.

This test catches issues where:
1. Sparse decoding (masked positions only) diverges from full decoding (all positions + ignore_index)
2. need_weights=False changes attention output values
3. Gradient flow through sparse indexing is correct

Run: python verification/verify_sparse_decoding.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


def test_sparse_vs_full_loss():
    """Verify sparse and full CrossEntropyLoss produce identical results."""
    print("Test 1: Sparse vs Full CrossEntropyLoss...")
    
    torch.manual_seed(42)
    B, L, V = 4, 32, 1000
    
    logits_full = torch.randn(B, L, V, requires_grad=True)
    labels = torch.randint(0, V, (B, L))
    # Mask ~15% positions (like MLM)
    mask_positions = torch.rand(B, L) < 0.15
    labels[~mask_positions] = -100
    
    # Full decoding: CrossEntropyLoss with ignore_index=-100
    loss_fct_full = CrossEntropyLoss(ignore_index=-100)
    loss_full = loss_fct_full(logits_full.view(-1, V), labels.view(-1))
    
    # Sparse decoding: gather masked positions, then CrossEntropyLoss without ignore_index
    mask = (labels != -100)
    flat_logits = logits_full.detach().clone().requires_grad_(True).reshape(-1, V)
    flat_mask = mask.reshape(-1)
    masked_logits = flat_logits[flat_mask]
    flat_labels = labels.view(-1)
    masked_labels = flat_labels[flat_mask]
    
    loss_fct_sparse = CrossEntropyLoss()
    loss_sparse = loss_fct_sparse(masked_logits, masked_labels)
    
    diff = abs(loss_full.item() - loss_sparse.item())
    print(f"  Full loss:   {loss_full.item():.6f}")
    print(f"  Sparse loss: {loss_sparse.item():.6f}")
    print(f"  Difference:  {diff:.8f}")
    
    assert diff < 1e-5, f"Loss difference too large: {diff}"
    print("  PASSED: Sparse and full loss are identical")


def test_sparse_gradient_flow():
    """Verify gradients flow correctly through sparse indexing."""
    print("\nTest 2: Gradient flow through sparse indexing...")
    
    torch.manual_seed(42)
    B, L, H, V = 2, 16, 32, 100
    
    # Simulate decoder output -> lm_head -> sparse loss
    decoder_output = torch.randn(B, L, H, requires_grad=True)
    lm_head = nn.Linear(H, V, bias=False)
    
    labels = torch.randint(0, V, (B, L))
    mask_positions = torch.rand(B, L) < 0.15
    # Ensure at least 1 masked position
    mask_positions[0, 0] = True
    labels[~mask_positions] = -100
    
    # Sparse path (as in our code)
    mask = (labels != -100)
    flat_decoder = decoder_output.reshape(-1, H)
    flat_mask = mask.reshape(-1)
    masked_decoder = flat_decoder[flat_mask]
    masked_logits = lm_head(masked_decoder)
    flat_labels = labels.view(-1)
    masked_labels = flat_labels[flat_mask]
    
    loss = CrossEntropyLoss()(masked_logits, masked_labels)
    loss.backward()
    
    assert decoder_output.grad is not None, "No gradient on decoder_output!"
    assert not torch.isnan(decoder_output.grad).any(), "NaN in gradients!"
    assert not torch.isinf(decoder_output.grad).any(), "Inf in gradients!"
    
    # Only masked positions should have non-zero gradients
    grad_magnitudes = decoder_output.grad.abs().sum(dim=-1)  # [B, L]
    masked_grad = grad_magnitudes[mask]
    unmasked_grad = grad_magnitudes[~mask]
    
    assert masked_grad.sum() > 0, "Masked positions should have non-zero gradients"
    assert unmasked_grad.sum().item() < 1e-7, "Unmasked positions should have zero gradients"
    
    print(f"  Masked positions grad magnitude: {masked_grad.mean().item():.6f}")
    print(f"  Unmasked positions grad magnitude: {unmasked_grad.sum().item():.8f}")
    print("  PASSED: Gradients flow correctly through sparse indexing")


def test_need_weights_false():
    """Verify need_weights=False doesn't change attention output values."""
    print("\nTest 3: need_weights=False output equivalence...")
    
    torch.manual_seed(42)
    B, S, H = 2, 16, 32
    
    attn = nn.MultiheadAttention(embed_dim=H, num_heads=4, batch_first=True)
    attn.eval()
    
    Q = torch.randn(B, 8, H)   # concepts
    K = torch.randn(B, S, H)   # tokens
    V = K.clone()
    
    with torch.no_grad():
        out_with_weights, weights = attn(Q, K, V, need_weights=True)
        out_no_weights, _ = attn(Q, K, V, need_weights=False)
    
    diff = (out_with_weights - out_no_weights).abs().max().item()
    print(f"  Max output difference: {diff:.10f}")
    
    assert diff < 1e-5, f"Outputs differ with need_weights=False: max diff={diff}"
    print("  PASSED: need_weights=False produces identical outputs")


def test_full_model_sparse_vs_full():
    """Test the actual model forward pass: sparse vs full decoding match."""
    print("\nTest 4: Full model sparse vs full decoding loss comparison...")
    
    from nn.concept_encoder import ConceptEncoderConfig
    from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
    
    config = ConceptEncoderConfig(
        vocab_size=100, concept_num=8, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_sequence_length=50,
        pad_token_id=0, mask_token_id=3,
    )
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.eval()
    
    B, L = 2, 20
    input_ids = torch.randint(4, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = input_ids.clone()
    mask = torch.rand(B, L) < 0.15
    mask[:, 0] = True  # Ensure at least one masked
    labels[~mask] = -100
    
    with torch.no_grad():
        # Current sparse path (labels provided)
        output_sparse = model(input_ids, attention_mask=attention_mask, labels=labels)
        sparse_loss = output_sparse.loss.item()
        
        # Compare: full logits path (no labels -> get full logits, then compute loss manually)
        output_full = model(input_ids, attention_mask=attention_mask)
        full_logits = output_full.logits  # [B, L, V]
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        full_loss = loss_fct(full_logits.view(-1, config.vocab_size), labels.view(-1)).item()
    
    diff = abs(sparse_loss - full_loss)
    print(f"  Sparse path loss: {sparse_loss:.6f}")
    print(f"  Full path loss:   {full_loss:.6f}")
    print(f"  Difference:       {diff:.8f}")
    
    assert diff < 1e-4, f"Model sparse vs full loss mismatch: {diff}"
    print("  PASSED: Model sparse and full decoding produce same loss")


def test_training_step_gradients():
    """Simulate a training step and check gradients are healthy."""
    print("\nTest 5: Training step gradient health...")
    
    from nn.concept_encoder import ConceptEncoderConfig
    from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
    
    config = ConceptEncoderConfig(
        vocab_size=100, concept_num=8, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_sequence_length=50,
        pad_token_id=0, mask_token_id=3,
    )
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.train()
    
    B, L = 4, 20
    input_ids = torch.randint(4, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = input_ids.clone()
    mask = torch.rand(B, L) < 0.15
    mask[:, 0] = True
    labels[~mask] = -100
    
    # Forward + backward
    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    
    max_grad = 0
    has_nan = False
    has_inf = False
    zero_grad_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            g = param.grad
            if torch.isnan(g).any():
                has_nan = True
                print(f"  NaN gradient in {name}")
            if torch.isinf(g).any():
                has_inf = True
                print(f"  Inf gradient in {name}")
            grad_norm = g.abs().max().item()
            if grad_norm > max_grad:
                max_grad = grad_norm
            if grad_norm < 1e-10:
                zero_grad_params += 1
    
    print(f"  Max gradient magnitude: {max_grad:.6f}")
    print(f"  Params with zero grad: {zero_grad_params}/{total_params}")
    
    assert not has_nan, "NaN gradients detected!"
    assert not has_inf, "Inf gradients detected!"
    assert max_grad < 100, f"Gradient too large: {max_grad}"
    assert max_grad > 1e-8, f"Gradient too small: {max_grad}"
    
    print("  PASSED: Training step gradients are healthy")


def test_multi_step_training():
    """Run multiple training steps to check for instability."""
    print("\nTest 6: Multi-step training stability (20 steps)...")
    
    from nn.concept_encoder import ConceptEncoderConfig
    from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
    
    config = ConceptEncoderConfig(
        vocab_size=100, concept_num=8, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_sequence_length=50,
        pad_token_id=0, mask_token_id=3,
    )
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    losses = []
    for step in range(20):
        optimizer.zero_grad()
        
        B, L = 4, 20
        input_ids = torch.randint(4, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        labels = input_ids.clone()
        mask = torch.rand(B, L) < 0.15
        mask[:, 0] = True
        labels[~mask] = -100
        
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    # Loss should decrease
    first_5_avg = sum(losses[:5]) / 5
    last_5_avg = sum(losses[-5:]) / 5
    
    print(f"  First 5 avg loss: {first_5_avg:.4f}")
    print(f"  Last 5 avg loss:  {last_5_avg:.4f}")
    
    assert not any(l != l for l in losses), "NaN loss detected!"  # NaN check
    assert all(l < 100 for l in losses), "Loss exploded!"
    
    print("  PASSED: Multi-step training is stable")


if __name__ == "__main__":
    print("=" * 60)
    print("Sparse Decoding & Engineering Fix Verification")
    print("=" * 60)
    
    tests = [
        test_sparse_vs_full_loss,
        test_sparse_gradient_flow,
        test_need_weights_false,
        test_full_model_sparse_vs_full,
        test_training_step_gradients,
        test_multi_step_training,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! Sparse decoding and engineering fixes are correct.")

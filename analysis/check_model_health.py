"""
Model Health Check Script

This script performs sanity checks on a trained Concept Encoder model
to identify numerical issues before fine-tuning on downstream tasks.

Usage:
    python analysis/check_model_health.py --model_path ./Cache/Training/MODEL_NAME --model_type weighted_mlm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from transformers import AutoTokenizer
import numpy as np
from nn.concept_encoder import (
    ConceptEncoderForMaskedLMWeighted,
    ConceptEncoderConfig
)

def check_parameter_health(model):
    """Check for NaN, Inf, or extreme values in model parameters."""
    print("\n" + "="*80)
    print("PARAMETER HEALTH CHECK")
    print("="*80)
    
    issues_found = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        data = param.data
        
        # Check for NaN
        if torch.isnan(data).any():
            issues_found.append(f"[X] NaN values in {name}")
        
        # Check for Inf
        if torch.isinf(data).any():
            issues_found.append(f"[X] Inf values in {name}")
        
        # Check for extreme values
        abs_max = data.abs().max().item()
        if abs_max > 1e3:
            issues_found.append(f"[!] Large values in {name}: max={abs_max:.2e}")
        
        # Check variance
        std = data.std().item()
        if std < 1e-6:
            issues_found.append(f"[!] Very low variance in {name}: std={std:.2e}")
        if std > 1e2:
            issues_found.append(f"[!] Very high variance in {name}: std={std:.2e}")
    
    print(f"Total parameters checked: {total_params:,}")
    
    if not issues_found:
        print("[OK] All parameters are healthy!")
    else:
        print(f"\n[!] Found {len(issues_found)} potential issues:")
        for issue in issues_found:
            print(f"  {issue}")
    
    return len(issues_found) == 0

def check_forward_pass(model, tokenizer):
    """Check if forward pass produces reasonable outputs."""
    print("\n" + "="*80)
    print("FORWARD PASS CHECK")
    print("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create sample input
    sample_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Check logits
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"Logits mean: {logits.mean().item():.2f}")
            print(f"Logits std: {logits.std().item():.2f}")
            
            # Check for NaN/Inf
            if torch.isnan(logits).any():
                print("[X] NaN values in output logits!")
                return False
            if torch.isinf(logits).any():
                print("[X] Inf values in output logits!")
                return False
            
            # Check if predictions are reasonable (not all same token)
            predictions = logits.argmax(dim=-1)
            unique_preds = torch.unique(predictions).numel()
            print(f"Unique predictions: {unique_preds} out of {predictions.numel()}")
            
            if unique_preds == 1:
                print("[!] Model is predicting the same token for all positions!")
                return False
            elif unique_preds < predictions.numel() * 0.1:
                print("[!] Model has very low prediction diversity!")
            
            print("[OK] Forward pass produces reasonable outputs")
            return True
            
        except Exception as e:
            print(f"[X] Forward pass failed with error: {str(e)}")
            return False

def check_loss_computation(model, tokenizer):
    """Check if loss computation is stable."""
    print("\n" + "="*80)
    print("LOSS COMPUTATION CHECK")
    print("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create sample input with labels
    sample_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)
    
    # Create labels (same as input_ids, with -100 for non-masked)
    labels = inputs['input_ids'].clone()
    # Mask 15% of tokens
    mask_prob = torch.rand(labels.shape) < 0.15
    labels[~mask_prob] = -100
    
    with torch.no_grad():
        try:
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            print(f"Loss value: {loss.item():.4f}")
            
            if torch.isnan(loss):
                print("[X] Loss is NaN!")
                return False
            if torch.isinf(loss):
                print("[X] Loss is Inf!")
                return False
            if loss.item() > 100:
                print(f"[!] Loss is very high: {loss.item():.2f}")
                return False
            if loss.item() < 0:
                print(f"[X] Loss is negative: {loss.item():.4f}")
                return False
            
            print("[OK] Loss computation is stable")
            return True
            
        except Exception as e:
            print(f"[X] Loss computation failed with error: {str(e)}")
            return False

def check_concept_embeddings(model):
    """Check concept embedding statistics."""
    print("\n" + "="*80)
    print("CONCEPT EMBEDDING CHECK")
    print("="*80)
    
    concept_emb = model.encoder.concept_embeddings.weight.data
    
    print(f"Concept embeddings shape: {concept_emb.shape}")
    print(f"Concept embeddings range: [{concept_emb.min().item():.4f}, {concept_emb.max().item():.4f}]")
    print(f"Concept embeddings mean: {concept_emb.mean().item():.4f}")
    print(f"Concept embeddings std: {concept_emb.std().item():.4f}")
    
    # Check concept diversity (pairwise distances)
    with torch.no_grad():
        # Normalize embeddings
        concept_emb_norm = concept_emb / concept_emb.norm(dim=1, keepdim=True)
        # Compute pairwise similarities
        similarities = torch.mm(concept_emb_norm, concept_emb_norm.t())
        # Get off-diagonal elements (exclude self-similarity)
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool, device=similarities.device)
        off_diag_sim = similarities[mask]
        
        print(f"Pairwise concept similarity (cosine):")
        print(f"  Mean: {off_diag_sim.mean().item():.4f}")
        print(f"  Std: {off_diag_sim.std().item():.4f}")
        print(f"  Min: {off_diag_sim.min().item():.4f}")
        print(f"  Max: {off_diag_sim.max().item():.4f}")
        
        # Check if concepts are too similar (collapsed)
        if off_diag_sim.mean().item() > 0.9:
            print("[!] Concepts are very similar (possible mode collapse)")
            return False
        
    print("[OK] Concept embeddings look healthy")
    return True

def main():
    parser = argparse.ArgumentParser(description="Check Concept Encoder model health")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="weighted_mlm", help="Model type")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer to use")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CONCEPT ENCODER MODEL HEALTH CHECK")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    
    # Load model
    print("\nLoading model...")
    try:
        if args.model_type == "weighted_mlm":
            model = ConceptEncoderForMaskedLMWeighted.from_pretrained(args.model_path)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"[OK] Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"[X] Failed to load model: {str(e)}")
        return
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print("[OK] Tokenizer loaded successfully")
    
    # Run checks
    all_checks_passed = True
    
    all_checks_passed &= check_parameter_health(model)
    all_checks_passed &= check_concept_embeddings(model)
    all_checks_passed &= check_forward_pass(model, tokenizer)
    all_checks_passed &= check_loss_computation(model, tokenizer)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_checks_passed:
        print("[OK] MODEL IS HEALTHY - Ready for fine-tuning on downstream tasks!")
    else:
        print("[X] MODEL HAS ISSUES - DO NOT use for fine-tuning yet!")
        print("\nRecommendations:")
        print("  1. Re-train with lower learning rate (1e-4 instead of 5e-4)")
        print("  2. Train for longer (at least 3 epochs)")
        print("  3. Use gradient clipping (max_grad_norm=1.0)")
        print("  4. Consider using FP32 instead of BF16 for stability")
    
    print("="*80)

if __name__ == "__main__":
    main()


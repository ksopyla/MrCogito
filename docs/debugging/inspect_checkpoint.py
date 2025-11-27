#!/usr/bin/env python
import torch
import os
import sys
import argparse
from transformers import AutoConfig, AutoModel
import numpy as np

# Add parent directory to path to allow importing nn.concept_encoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoderForMaskedLMWeighted, ConceptEncoderForSequenceClassification
except ImportError:
    print("Could not import ConceptEncoder classes. Make sure you are running this from the project root or scripts directory.")
    sys.path.append(".")
    try:
        from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoderForMaskedLMWeighted
    except ImportError as e:
        print(f"Error importing: {e}")
        sys.exit(1)

def inspect_model(model_path):
    print(f"Inspecting model at: {model_path}")
    
    try:
        # Load config
        config = ConceptEncoderConfig.from_pretrained(model_path)
        print("Config loaded successfully.")
        
        # Try loading state dict directly to avoid class mismatch issues initially
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            state_dict_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(state_dict_path):
                print(f"No model file found at {model_path}")
                return

        print(f"Loading weights from {state_dict_path}...")
        # We use safe loading if available, else torch.load
        if state_dict_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            
        print(f"Loaded {len(state_dict)} tensors.")
        
        has_nan = False
        has_inf = False
        max_val = -float('inf')
        min_val = float('inf')
        
        print("\n--- Weight Statistics ---")
        print(f"{'Layer Name':<60} | {'Shape':<20} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'NaN/Inf'}")
        print("-" * 140)
        
        for name, param in state_dict.items():
            # Skip integer tensors (like position_ids if saved)
            if not torch.is_floating_point(param):
                continue
                
            p_min = param.min().item()
            p_max = param.max().item()
            p_mean = param.mean().item()
            p_std = param.std().item()
            
            p_has_nan = torch.isnan(param).any().item()
            p_has_inf = torch.isinf(param).any().item()
            
            if p_has_nan: has_nan = True
            if p_has_inf: has_inf = True
            
            if p_max > max_val: max_val = p_max
            if p_min < min_val: min_val = p_min
            
            status = ""
            if p_has_nan: status += "NaN "
            if p_has_inf: status += "Inf"
            if abs(p_max) > 100 or abs(p_min) > 100: status += "Large"
            
            # Only print if interesting or every N layers
            if p_has_nan or p_has_inf or abs(p_max) > 10 or abs(p_min) > 10 or "concept_embeddings" in name or "concept_weights" in name:
                print(f"{name:<60} | {str(list(param.shape)):<20} | {p_min:.4f}     | {p_max:.4f}     | {p_mean:.4f}     | {p_std:.4f}     | {status}")
        
        print("-" * 140)
        print(f"\nOverall Status:")
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        print(f"  Global Max: {max_val}")
        print(f"  Global Min: {min_val}")
        
        if has_nan or has_inf:
            print("\nCRITICAL: Model contains NaN or Inf values. It is corrupted.")
        elif max_val > 1000 or min_val < -1000:
            print("\nWARNING: Model contains very large values. This might cause instability during fine-tuning.")
        else:
            print("\nModel weights seem statistically healthy (no NaNs/Infs, reasonable range).")

    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_checkpoint.py <model_path>")
        sys.exit(1)
    
    inspect_model(sys.argv[1])


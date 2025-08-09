#!/usr/bin/env python3
"""
ConceptEncoder Parameter Analysis and Model Size Calculator

This script analyzes the parameter breakdown for ConceptEncoderForMaskedLM
and provides model size recommendations based on literature review.

Updated to support:
- Different token and concept embedding dimensions
- Power-of-2 constraints for efficiency
- Modern vocabulary sizes from current models
- Systematic parameter exploration

Based on:
- DistilBERT paper (Sanh et al., 2019): 66M params with 40% reduction
- TinyBERT paper (Jiao et al., 2019): 28% params of original BERT
- Parameter calculation methodology from transformer literature
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for ConceptEncoder model.
    
    Architecture uses:
    - token_embedding_dim: Small dimension for token embeddings (efficiency)
    - concept_embedding_dim: Main model dimension (replaces "hidden_size")
    - All transformer operations use concept_embedding_dim
    """
    vocab_size: int
    token_embedding_dim: int      # Token embedding dimension (small for efficiency)
    concept_embedding_dim: int    # Main model dimension (the true "hidden_size")
    num_hidden_layers: int
    num_attention_heads: int
    concept_size: int             # Number of concept tokens
    intermediate_size: int        # FFN intermediate size
    max_position_embeddings: int = 512
    tie_word_embeddings: bool = True


def calculate_concept_encoder_parameters(config: ModelConfig) -> Dict[str, Any]:
    """
    Calculate parameter count for optimized ConceptEncoder architecture.
    
    Architecture:
    1. Token Embeddings: vocab_size × token_embedding_dim
    2. Position Embeddings: max_position_embeddings × token_embedding_dim  
    3. Concept Embeddings: concept_size × concept_embedding_dim
    4. N × ConceptEncoderLayer (cross_attn + self_attn + 3×LayerNorm + gated_ffn)
    5. Final LayerNorm: concept_embedding_dim × 2
    6. concept_to_sequence: LayerNorm + Linear(concept_embedding_dim -> concept_embedding_dim)
    7. lm_head: concept_embedding_dim × vocab_size (tied or untied)
    
    Cross-attention uses different Q, K, V dimensions:
    - Q: concept_embedding_dim (concepts as queries)
    - K, V: token_embedding_dim (tokens as keys/values)
    """
    
    # Use concept_embedding_dim as the main model dimension
    d_model = config.concept_embedding_dim
    d_token = config.token_embedding_dim
    
    # 1. Token Embeddings - optimized size
    token_embeddings = config.vocab_size * d_token
    
    # 2. Position Embeddings - use token dimension for efficiency
    position_embeddings = config.max_position_embeddings * d_token
    
    # 3. Concept Embeddings - main model dimension
    concept_embeddings = config.concept_size * d_model
    
    # 4. Per-layer parameters (ConceptEncoderLayer)
    
    # Cross-attention: concepts (Q) attend to tokens (K, V)
    # Q: concept_embedding_dim, K,V: token_embedding_dim
    cross_attention_params = (
        d_model * d_model +          # Q projection
        d_token * d_model * 2 +      # K, V projections  
        d_model + d_model * 2 +      # Q, K, V biases
        d_model * d_model +          # Output projection
        d_model                      # Output bias
    )
    
    # Self-attention: concepts attend to concepts (all concept_embedding_dim)
    self_attention_params = (
        d_model * d_model * 3 +      # Q, K, V projections
        d_model * 3 +                # Q, K, V biases
        d_model * d_model +          # Output projection
        d_model                      # Output bias
    )
    
    # Layer Normalization (3 per layer) - all use concept_embedding_dim
    layer_norm_params = d_model * 2 * 3  # 3 norms × 2 params each (weight + bias)
    
    # Gated FFN - uses concept_embedding_dim throughout
    gated_ffn_params = (
        d_model * (config.intermediate_size * 2) +  # Wi: concept_embedding_dim -> intermediate_size*2
        (config.intermediate_size * 2) +           # Wi bias
        config.intermediate_size * d_model +       # Wo: intermediate_size -> concept_embedding_dim
        d_model                                    # Wo bias
    )
    
    # Total per layer
    per_layer_params = (
        cross_attention_params +
        self_attention_params + 
        layer_norm_params +
        gated_ffn_params
    )
    
    total_transformer_params = config.num_hidden_layers * per_layer_params
    
    # 5. Final LayerNorm
    final_layer_norm = d_model * 2  # weight + bias
    
    # 6. ConceptEncoderForMaskedLM: concept_to_sequence module
    concept_to_sequence_params = (
        d_model * 2 +               # LayerNorm (weight + bias)
        d_model * d_model +         # Linear weight
        d_model                     # Linear bias
    )
    
    # 7. LM head: concept_embedding_dim -> vocab_size
    if config.tie_word_embeddings:
        # Need projection layer: concept_embedding_dim -> token_embedding_dim for tying
        lm_head_params = d_model * d_token  # Projection layer to match token embeddings
    else:
        lm_head_params = d_model * config.vocab_size  # Direct projection
    
    # Total parameters
    total_params = (
        token_embeddings +
        position_embeddings +
        concept_embeddings +
        total_transformer_params +
        final_layer_norm +
        concept_to_sequence_params +
        lm_head_params
    )
    
    return {
        "total_params": total_params,
        "total_params_M": total_params / 1e6,
        "breakdown": {
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "concept_embeddings": concept_embeddings,
            "transformer_layers": total_transformer_params,
            "final_layer_norm": final_layer_norm,
            "concept_to_sequence": concept_to_sequence_params,
            "lm_head": lm_head_params
        },
        "per_layer_params": per_layer_params,
        "config": config,
        "architecture_details": {
            "main_model_dim": d_model,
            "token_dim": d_token,
            "cross_attn_q_dim": d_model,
            "cross_attn_kv_dim": d_token,
            "self_attn_dim": d_model,
            "ffn_dim": d_model
        }
    }


def get_modern_vocab_sizes() -> Dict[str, Dict[str, Any]]:
    """Get vocabulary sizes from modern models with examples and ranges."""
    return {
        "tiny_4k": {
            "size": 4096,
            "range": "4K",
            "description": "Rapid prototyping",
            "examples": ["Custom minimal", "Domain-specific"]
        },
        "small_8k": {
            "size": 8192, 
            "range": "8K",
            "description": "Mobile/edge efficient",
            "examples": ["Specialized domains", "Mobile models"]
        },
        "compact_16k": {
            "size": 16384,
            "range": "16K", 
            "description": "Compact but diverse",
            "examples": ["DistilBERT variants", "Efficient models"]
        },
        "standard_32k": {
            "size": 32000,
            "range": "32K",
            "description": "Modern standard",
            "examples": ["LLaMA 1/2", "Mixtral", "T5"]
        },
        "bert_30k": {
            "size": 30522,
            "range": "30K",
            "description": "BERT-family standard",
            "examples": ["BERT", "DeBERTa", "ELECTRA"]
        },
        "medium_50k": {
            "size": 50257,
            "range": "50K", 
            "description": "Balanced efficiency",
            "examples": ["GPT-2", "RoBERTa", "CodeT5"]
        },
        "large_64k": {
            "size": 65536,
            "range": "64K",
            "description": "Rich tokenization", 
            "examples": ["GPT-3.5 variants", "Claude"]
        },
        "xlarge_128k": {
            "size": 128256,
            "range": "128K",
            "description": "Extra-large modern",
            "examples": ["LLaMA 3", "Qwen 1.5", "Mistral"]
        },
        "xxlarge_256k": {
            "size": 256000,
            "range": "256K", 
            "description": "Multilingual/code max",
            "examples": ["Gemma", "PaLM 2", "Multilingual"]
        }
    }





def interactive_model_designer():
    """Interactive model configuration designer."""
    print("=" * 60)
    print("🧠 ConceptEncoder Interactive Model Designer")
    print("=" * 60)
    
    # Show vocabulary size options
    vocab_sizes = get_modern_vocab_sizes()
    print("\n📚 Available Vocabulary Sizes:")
    print("=" * 85)
    print(f"{'#':<3} {'Name':<12} {'Range':<6} {'Tokens':<8} {'Description':<20} {'Examples'}")
    print("=" * 85)
    for i, (name, vocab_info) in enumerate(vocab_sizes.items(), 1):
        examples_str = ", ".join(vocab_info["examples"][:2])  # Show first 2 examples
        if len(vocab_info["examples"]) > 2:
            examples_str += "..."
        print(f"{i:<3} {name:<12} {vocab_info['range']:<6} {vocab_info['size']:<8,} {vocab_info['description']:<20} {examples_str}")
    print("=" * 85)
    
    while True:
        try:
            print("\n" + "=" * 50)
            print("🎯 Model Configuration Input")
            print("=" * 50)
            
            # Get user inputs
            print("\n1. Vocabulary Configuration:")
            vocab_choice = input(f"   Enter vocab number (1-{len(vocab_sizes)}) or custom size: ").strip()
            
            if vocab_choice.isdigit() and 1 <= int(vocab_choice) <= len(vocab_sizes):
                vocab_name, vocab_info = list(vocab_sizes.items())[int(vocab_choice) - 1]
                vocab_size = vocab_info["size"]
                print(f"   Selected: {vocab_name} ({vocab_info['range']}) - {vocab_size:,} tokens")
                print(f"   Examples: {', '.join(vocab_info['examples'])}")
            else:
                try:
                    vocab_size = int(vocab_choice)
                    vocab_name = "custom"
                    print(f"   Custom vocab size: {vocab_size:,} tokens")
                except ValueError:
                    print("   Invalid input. Using BERT default (30,522)")
                    vocab_size = 30522
                    vocab_name = "bert_30k"
            
            print("\n2. Embedding Dimensions (powers of 2):")
            token_dim = int(input("   Token embedding dim (64, 128, 256, 512, 1024): ") or "256")
            concept_dim = int(input("   Concept embedding dim (128, 256, 512, 1024, 2048): ") or "512")
            
            print("\n3. Architecture Parameters:")
            concept_size = int(input("   Number of concept tokens (32, 64, 128, 256): ") or "128")
            ffn_size = int(input("   FFN intermediate size (1024, 2048, 4096): ") or "2048")
            
            print("\n4. Target Parameters:")
            target_params_input = input("   Target total parameters (e.g., '30M', '100M', '500M', '1B','3B'): ").strip().upper()
            
            # Parse target parameters
            if target_params_input.endswith('M'):
                target_params = float(target_params_input[:-1]) * 1e6
            elif target_params_input.endswith('B'):
                target_params = float(target_params_input[:-1]) * 1e9
            else:
                target_params = float(target_params_input)
            
            print(f"\n🔍 Finding optimal configuration for {target_params/1e6:.1f}M parameters...")
            
            # Find optimal layer and head configurations
            best_configs = find_optimal_config(
                vocab_size=vocab_size,
                token_dim=token_dim,
                concept_dim=concept_dim,
                concept_size=concept_size,
                ffn_size=ffn_size,
                target_params=target_params,
                tolerance=0.1  # 10% tolerance
            )
            
            if best_configs:
                print(f"\n✅ Found {len(best_configs)} optimal configurations:")
                print("\n" + "=" * 80)
                print(f"{'ConceptDim':>10} {'Layers':>7} {'Heads':>6} {'Params':>8} {'Head Dim':>8} {'Tok%':>6} {'Conc%':>6} {'Trans%':>7}")
                print("=" * 80)
                
                for config in best_configs[:10]:  # Show top 10
                    result = calculate_concept_encoder_parameters(config['config'])
                    tok_pct = (result["breakdown"]["token_embeddings"] / result["total_params"]) * 100
                    conc_pct = (result["breakdown"]["concept_embeddings"] / result["total_params"]) * 100
                    trans_pct = (result["breakdown"]["transformer_layers"] / result["total_params"]) * 100
                    head_dim = config['config'].concept_embedding_dim // config['config'].num_attention_heads
                    
                    print(f"{config['config'].concept_embedding_dim:>10} "
                          f"{config['config'].num_hidden_layers:>7} "
                          f"{config['config'].num_attention_heads:>6} "
                          f"{result['total_params_M']:>7.1f}M "
                          f"{head_dim:>8} "
                          f"{tok_pct:>5.1f}% "
                          f"{conc_pct:>5.1f}% "
                          f"{trans_pct:>6.1f}%")
                
                # Show detailed breakdown for best config
                best_config = best_configs[0]
                best_result = calculate_concept_encoder_parameters(best_config['config'])
                
                print(f"\n🏆 RECOMMENDED CONFIGURATION:")
                print("=" * 50)
                print(f"Vocabulary: {vocab_name} ({vocab_size:,} tokens)")
                print(f"Token embedding dim: {token_dim}")
                print(f"Concept embedding dim: {concept_dim} (main model dimension)")
                print(f"Number of layers: {best_config['config'].num_hidden_layers}")
                print(f"Number of heads: {best_config['config'].num_attention_heads}")
                print(f"Concept tokens: {concept_size}")
                print(f"FFN size: {ffn_size}")
                print(f"Total parameters: {best_result['total_params_M']:.2f}M")
                
                print(f"\n📊 Parameter Breakdown:")
                for component, params in best_result['breakdown'].items():
                    pct = (params / best_result['total_params']) * 100
                    print(f"  {component.replace('_', ' ').title():20}: {params/1e6:>6.2f}M ({pct:>5.1f}%)")
                
                print(f"\n💻 CLI Command:")
                print("=" * 50)
                print(f"--concept_embedding_dim {best_config['config'].concept_embedding_dim} \\")
                print(f"--token_embedding_dim {best_config['config'].token_embedding_dim} \\")
                print(f"--num_hidden_layers {best_config['config'].num_hidden_layers} \\")
                print(f"--num_attention_heads {best_config['config'].num_attention_heads} \\")
                print(f"--concept_size {concept_size} \\")
                print(f"--intermediate_size {ffn_size} \\")
                if vocab_name != "custom":
                    tokenizer_map = {
                        "bert_30k": "bert-base-uncased",
                        "standard_32k": "meta-llama/Llama-2-7b-hf", 
                        "xlarge_128k": "meta-llama/Llama-3.2-1B",
                        "medium_50k": "gpt2",
                        "xxlarge_256k": "google/gemma-2b"
                    }
                    
                    if vocab_name in tokenizer_map:
                        print(f"--tokenizer_name {tokenizer_map[vocab_name]}")
                    else:
                        print(f"# Tokenizer for {vocab_name}: You'll need to find/create appropriate tokenizer")
                        print(f"# Vocab size: {vocab_size:,} tokens")
                else:
                    print(f"# Custom vocab size: {vocab_size:,} tokens")
                    print(f"# You'll need to create/provide appropriate tokenizer")
                
                print(f"\n🔧 Architecture Notes:")
                print(f"• Main model dimension: {concept_dim} (concept_embedding_dim)")
                print(f"• Token dimension: {token_dim} (for efficiency)")
                print(f"• Cross-attention: Q={concept_dim}, K,V={token_dim}")
                print(f"• Self-attention: Q,K,V={concept_dim}")
                print(f"• All LayerNorm/FFN use: {concept_dim}")
                
            else:
                print(f"\n❌ No valid configurations found for {target_params/1e6:.1f}M parameters")
                
                # Calculate what the parameters would be with minimum viable config
                min_config = ModelConfig(
                    vocab_size=vocab_size,
                    token_embedding_dim=token_dim,
                    concept_embedding_dim=concept_dim,
                    num_hidden_layers=1,  # Minimum layers
                    num_attention_heads=1,  # Minimum heads (must divide concept_embedding_dim)
                    concept_size=concept_size,
                    intermediate_size=ffn_size,
                    max_position_embeddings=512,
                    tie_word_embeddings=True
                )
                
                min_result = calculate_concept_encoder_parameters(min_config)
                print(f"\n📊 Minimum possible parameters with your constraints:")
                print(f"   Estimated: {min_result['total_params_M']:.2f}M parameters")
                print(f"   Your target: {target_params/1e6:.1f}M parameters")
                print(f"   Difference: {min_result['total_params_M'] - target_params/1e6:+.2f}M ({((min_result['total_params_M'] / (target_params/1e6)) - 1) * 100:+.1f}%)")
                
                if min_result['total_params_M'] > target_params/1e6:
                    print(f"\n💡 Suggestions to reduce parameters:")
                    print(f"   • Use smaller vocabulary (current: {vocab_size:,} tokens)")
                    print(f"   • Reduce token embedding dim (current: {token_dim})")
                    print(f"   • Reduce concept embedding dim (current: {concept_dim})")
                    print(f"   • Reduce concept tokens (current: {concept_size})")
                    print(f"   • Reduce FFN size (current: {ffn_size})")
                else:
                    print(f"\n💡 Your target is achievable with very small models!")
                    print(f"   Consider increasing target parameters for better performance.")
            
            # Ask to continue
            print("\n" + "=" * 50)
            continue_choice = input("🔄 Try another configuration? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid inputs.")


def find_optimal_config(vocab_size, token_dim, concept_dim, concept_size, ffn_size, target_params, tolerance=0.1):
    """Find optimal layer and head configurations for given constraints."""
    valid_configs = []
    
    # Try different layer counts
    for num_layers in range(1, 25):  # 1 to 24 layers
        
        # Try different head counts (must divide concept_embedding_dim evenly)
        possible_heads = [h for h in [1, 2, 4, 6, 8, 12, 16, 20, 24, 32] if concept_dim % h == 0]
        
        for num_heads in possible_heads:
            
            config = ModelConfig(
                vocab_size=vocab_size,
                token_embedding_dim=token_dim,
                concept_embedding_dim=concept_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                concept_size=concept_size,
                intermediate_size=ffn_size,
                max_position_embeddings=512,
                tie_word_embeddings=True
            )
            
            result = calculate_concept_encoder_parameters(config)
            params_diff = abs(result["total_params"] - target_params) / target_params
            
            # Check if within tolerance
            if params_diff <= tolerance:
                valid_configs.append({
                    'config': config,
                    'result': result,
                    'params_diff': params_diff,
                    'efficiency': concept_size / result["total_params_M"]
                })
    
    # Sort by parameter difference (closest to target first), then by efficiency
    valid_configs.sort(key=lambda x: (x['params_diff'], -x['efficiency']))
    
    return valid_configs





if __name__ == "__main__":
    interactive_model_designer()
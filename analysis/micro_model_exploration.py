#!/usr/bin/env python3
"""
Micro Model Exploration for ConceptEncoder
Exploring unconventional hyperparameter spaces around 40M parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.concept_encoder_parameter_analysis import (
    ModelConfig, calculate_concept_encoder_parameters
)
from typing import List, Dict, Any
import pandas as pd

def explore_micro_models() -> List[Dict[str, Any]]:
    """
    Explore diverse micro model configurations around 40M parameters.
    Focus on unconventional architectures not explored in mainstream research.
    """
    
    configurations = []
    
    # Configuration 1: Extreme Asymmetry - Ultra-small token embeddings
    # Inspired by: Hash embeddings (Svenstrup et al. 2017) and ALBERT factorization
    configs_to_test = [
        {
            "name": "extreme_asymmetry_4d",
            "vocab_size": 32000,  # LLaMA-like
            "token_embedding_dim": 4,  # EXTREME: 4-dim token embeddings
            "concept_embedding_dim": 1024,  # Large concept space
            "concept_num": 256,
            "intermediate_size": 4096,
            "rationale": "Extreme factorization: 4-dim token embeddings with 1024-dim concepts. Tests if semantic meaning can be fully captured by concept layer."
        },
        {
            "name": "extreme_asymmetry_8d",
            "vocab_size": 50257,  # GPT-2 vocab
            "token_embedding_dim": 8,  # EXTREME: 8-dim token embeddings
            "concept_embedding_dim": 768,
            "concept_num": 192,
            "intermediate_size": 3072,
            "rationale": "8-dim tokens inspired by Bloom filters and hash embeddings. Tests minimal token representation hypothesis."
        },
        {
            "name": "extreme_asymmetry_16d",
            "vocab_size": 128256,  # LLaMA-3 large vocab
            "token_embedding_dim": 16,  # Very small embeddings for large vocab
            "concept_embedding_dim": 512,
            "concept_num": 128,
            "intermediate_size": 2048,
            "rationale": "16-dim tokens with 128K vocab. Tests if small embeddings can handle massive vocabularies efficiently."
        },
        
        # Configuration 2: Concept-Heavy Architecture
        # Inspired by: Mixture of Experts and routing networks
        {
            "name": "concept_heavy_512",
            "vocab_size": 30522,  # BERT vocab
            "token_embedding_dim": 64,
            "concept_embedding_dim": 512,
            "concept_num": 512,  # LARGE number of concepts
            "intermediate_size": 1536,
            "rationale": "512 concepts acting as 'semantic experts'. Inspired by MoE architectures but using concepts as routing mechanism."
        },
        {
            "name": "concept_heavy_1024",
            "vocab_size": 16384,  # Smaller vocab
            "token_embedding_dim": 32,
            "concept_embedding_dim": 384,
            "concept_num": 1024,  # VERY LARGE number of concepts
            "intermediate_size": 1536,
            "rationale": "1024 concepts with small model dim. Tests if many specialized concepts can compensate for smaller hidden dimensions."
        },
        
        # Configuration 3: Skinny-Deep Architecture
        # Inspired by: EfficientNet scaling laws and depth vs width research
        {
            "name": "skinny_deep",
            "vocab_size": 32000,
            "token_embedding_dim": 128,
            "concept_embedding_dim": 256,  # Very narrow
            "concept_num": 64,
            "intermediate_size": 768,  # Narrow FFN
            "rationale": "Narrow 256-dim with more layers. Tests depth over width hypothesis for concept learning."
        },
        
        # Configuration 4: Wide-Shallow Architecture
        # Inspired by: Wide ResNets and shallow but wide transformers
        {
            "name": "wide_shallow",
            "vocab_size": 30522,
            "token_embedding_dim": 256,
            "concept_embedding_dim": 1536,  # Very wide
            "concept_num": 96,
            "intermediate_size": 6144,  # Wide FFN
            "rationale": "1536-dim width with few layers. Tests if wide concept space can learn complex patterns in shallow networks."
        },
        
        # Configuration 5: Bottleneck Architecture
        # Inspired by: Funnel Transformer and hourglass architectures
        {
            "name": "bottleneck_small_concepts",
            "vocab_size": 50257,
            "token_embedding_dim": 512,
            "concept_embedding_dim": 768,
            "concept_num": 32,  # VERY FEW concepts (bottleneck)
            "intermediate_size": 3072,
            "rationale": "Only 32 concepts forcing extreme compression. Tests information bottleneck theory for representation learning."
        },
        
        # Configuration 6: Fibonacci-inspired dimensions
        # Unconventional: Using Fibonacci sequence for dimension sizing
        {
            "name": "fibonacci_dims",
            "vocab_size": 32768,  # Power of 2
            "token_embedding_dim": 89,  # Fibonacci number
            "concept_embedding_dim": 610,  # Fibonacci number
            "concept_num": 144,  # Fibonacci number
            "intermediate_size": 2584,  # Fibonacci number
            "rationale": "Non-power-of-2 dimensions based on Fibonacci sequence. Tests if natural proportions improve learning dynamics."
        },
        
        # Configuration 7: Prime number architecture
        # Unconventional: Prime numbers for all dimensions
        {
            "name": "prime_architecture",
            "vocab_size": 32003,  # Prime
            "token_embedding_dim": 97,  # Prime
            "concept_embedding_dim": 509,  # Prime
            "concept_num": 127,  # Prime
            "intermediate_size": 2039,  # Prime
            "rationale": "Prime number dimensions to avoid resonance patterns. Inspired by hash table design and number theory."
        },
        
        # Configuration 8: Tiny vocabulary with rich concepts
        # Testing extreme vocabulary compression
        {
            "name": "tiny_vocab_rich",
            "vocab_size": 4096,  # TINY vocab
            "token_embedding_dim": 256,
            "concept_embedding_dim": 1024,
            "concept_num": 256,
            "intermediate_size": 4096,
            "rationale": "4K vocab with rich embeddings. Tests if small, curated vocabulary with rich concepts can match larger vocab performance."
        },
        
        # Configuration 9: Massive vocabulary with minimal embeddings
        {
            "name": "massive_vocab_minimal",
            "vocab_size": 256000,  # MASSIVE vocab
            "token_embedding_dim": 8,  # Tiny embeddings
            "concept_embedding_dim": 384,
            "concept_num": 96,
            "intermediate_size": 1536,
            "rationale": "256K vocab with 8-dim embeddings. Tests extreme vocabulary scaling with minimal per-token parameters."
        },
        
        # Configuration 10: Golden ratio proportions
        # Mathematical harmony in architecture
        {
            "name": "golden_ratio",
            "vocab_size": 30522,
            "token_embedding_dim": 128,
            "concept_embedding_dim": 512,  # ~4x token dim
            "concept_num": 207,  # 128 * 1.618
            "intermediate_size": 2073,  # 512 * 4.05 (close to golden ratio squared)
            "rationale": "Dimensions following golden ratio proportions. Tests if mathematical harmony improves optimization dynamics."
        },
        
        # Configuration 11: Unbalanced attention
        # Different head counts to test attention patterns
        {
            "name": "unbalanced_attention_3h",
            "vocab_size": 32000,
            "token_embedding_dim": 96,
            "concept_embedding_dim": 384,  # Divisible by 3
            "concept_num": 96,
            "intermediate_size": 1536,
            "rationale": "3 attention heads (unusual). Tests if fewer heads with larger dimension per head improves concept attention."
        },
        
        # Configuration 12: Many heads, small dim
        {
            "name": "many_heads_32h",
            "vocab_size": 30522,
            "token_embedding_dim": 64,
            "concept_embedding_dim": 512,  # Divisible by 32
            "concept_num": 128,
            "intermediate_size": 2048,
            "rationale": "32 attention heads with 16-dim per head. Tests if many small attention heads improve concept diversity."
        }
    ]
    
    # Find optimal layers and heads for each configuration
    for config_spec in configs_to_test:
        # Try different layer and head combinations
        best_config = None
        best_diff = float('inf')
        
        for num_layers in range(2, 12):  # 2 to 11 layers
            # Find valid head counts
            concept_dim = config_spec["concept_embedding_dim"]
            possible_heads = [h for h in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32] if concept_dim % h == 0]
            
            for num_heads in possible_heads:
                config = ModelConfig(
                    vocab_size=config_spec["vocab_size"],
                    token_embedding_dim=config_spec["token_embedding_dim"],
                    concept_embedding_dim=config_spec["concept_embedding_dim"],
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    concept_num=config_spec["concept_num"],
                    intermediate_size=config_spec["intermediate_size"],
                    max_sequence_length=512,
                    tie_word_embeddings=True
                )
                
                result = calculate_concept_encoder_parameters(config)
                target = 40e6  # 40M parameters
                diff = abs(result["total_params"] - target)
                
                # Check if this is closer to 40M and within reasonable range (35M-45M)
                if diff < best_diff and 35e6 <= result["total_params"] <= 45e6:
                    best_diff = diff
                    best_config = {
                        "name": config_spec["name"],
                        "config": config,
                        "result": result,
                        "rationale": config_spec["rationale"]
                    }
        
        if best_config:
            configurations.append(best_config)
    
    return configurations


def format_results(configurations: List[Dict[str, Any]]) -> None:
    """Format and display the exploration results."""
    
    print("\n" + "="*100)
    print("🔬 UNCONVENTIONAL MICRO MODEL ARCHITECTURES (~40M Parameters)")
    print("="*100)
    
    # Create summary table
    summary_data = []
    
    for i, cfg in enumerate(configurations, 1):
        config = cfg["config"]
        result = cfg["result"]
        
        # Calculate key metrics
        tok_emb_pct = (result["breakdown"]["token_embeddings"] / result["total_params"]) * 100
        concept_emb_pct = (result["breakdown"]["concept_embeddings"] / result["total_params"]) * 100
        transformer_pct = (result["breakdown"]["transformer_layers"] / result["total_params"]) * 100
        
        # Calculate asymmetry ratio
        asymmetry_ratio = config.concept_embedding_dim / config.token_embedding_dim
        
        summary_data.append({
            "Name": cfg["name"],
            "Params (M)": f"{result['total_params_M']:.1f}",
            "Vocab": f"{config.vocab_size:,}",
            "Token Dim": config.token_embedding_dim,
            "Concept Dim": config.concept_embedding_dim,
            "Asymmetry": f"{asymmetry_ratio:.1f}x",
            "Concepts": config.concept_num,
            "Layers": config.num_hidden_layers,
            "Heads": config.num_attention_heads,
            "FFN": config.intermediate_size,
            "Tok Emb%": f"{tok_emb_pct:.1f}",
            "Concept%": f"{concept_emb_pct:.1f}",
            "Trans%": f"{transformer_pct:.1f}"
        })
    
    # Display summary table
    df = pd.DataFrame(summary_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\n📊 CONFIGURATION SUMMARY:")
    print(df.to_string(index=False))
    
    # Detailed configuration cards
    print("\n" + "="*100)
    print("📋 DETAILED CONFIGURATION CARDS")
    print("="*100)
    
    for i, cfg in enumerate(configurations, 1):
        config = cfg["config"]
        result = cfg["result"]
        
        print(f"\n{'='*60}")
        print(f"Configuration #{i}: {cfg['name'].upper()}")
        print(f"{'='*60}")
        
        print(f"\n📖 Rationale:")
        print(f"   {cfg['rationale']}")
        
        print(f"\n🔧 Architecture:")
        print(f"   Vocabulary:        {config.vocab_size:,} tokens")
        print(f"   Token Embedding:   {config.token_embedding_dim} dim")
        print(f"   Concept Embedding: {config.concept_embedding_dim} dim (main model dimension)")
        print(f"   Asymmetry Ratio:   {config.concept_embedding_dim/config.token_embedding_dim:.1f}x")
        print(f"   Concept Tokens:    {config.concept_num}")
        print(f"   Hidden Layers:     {config.num_hidden_layers}")
        print(f"   Attention Heads:   {config.num_attention_heads}")
        print(f"   Head Dimension:    {config.concept_embedding_dim // config.num_attention_heads}")
        print(f"   FFN Size:          {config.intermediate_size}")
        print(f"   Total Parameters:  {result['total_params_M']:.2f}M")
        
        print(f"\n📊 Parameter Distribution:")
        for component, params in result['breakdown'].items():
            pct = (params / result['total_params']) * 100
            print(f"   {component.replace('_', ' ').title():25}: {params/1e6:>6.2f}M ({pct:>5.1f}%)")
        
        print(f"\n💻 Training Command:")
        print(f"   python training/mlm_training.py \\")
        print(f"     --model_name {cfg['name']} \\")
        print(f"     --vocab_size {config.vocab_size} \\")
        print(f"     --token_embedding_dim {config.token_embedding_dim} \\")
        print(f"     --concept_embedding_dim {config.concept_embedding_dim} \\")
        print(f"     --num_hidden_layers {config.num_hidden_layers} \\")
        print(f"     --num_attention_heads {config.num_attention_heads} \\")
        print(f"     --concept_num {config.concept_num} \\")
        print(f"     --intermediate_size {config.intermediate_size}")
    
    # Research insights
    print(f"\n{'='*100}")
    print("🔬 RESEARCH INSIGHTS & COMPARISONS")
    print("="*100)
    
    print("\n📚 Literature Context:")
    print("""
    1. **Extreme Asymmetry (4-16 dim tokens)**:
       - Novel: No published work uses <32-dim token embeddings
       - Related: ALBERT factorization (128-dim), Hash embeddings
       - Hypothesis: Semantic meaning fully captured by concept layer
    
    2. **Concept-Heavy (512-1024 concepts)**:
       - Novel: Treating concepts as semantic routers/experts
       - Related: Mixture of Experts, Routing Transformers
       - Hypothesis: Many specialized concepts > deep layers
    
    3. **Prime/Fibonacci Dimensions**:
       - Novel: Non-power-of-2 dimensions unexplored in transformers
       - Related: Hash table design principles
       - Hypothesis: Avoid resonance, improve hash distribution
    
    4. **Massive Vocab + Minimal Embeddings (256K vocab, 8-dim)**:
       - Novel: Extreme vocabulary scaling with <32-dim embeddings
       - Related: Bloom filters, compressed sensing
       - Hypothesis: Vocabulary size independent of embedding quality
    
    5. **Bottleneck Concepts (32 concepts)**:
       - Novel: Extreme semantic compression via few concepts
       - Related: Information bottleneck theory, VAE
       - Hypothesis: Forces discovery of fundamental semantic primitives
    """)
    
    print("\n🎯 Experimental Priority:")
    print("""
    Phase 1 - Validation (Week 1):
    ├── extreme_asymmetry_4d - Test if 4-dim embeddings are learnable
    ├── bottleneck_small_concepts - Test 32-concept bottleneck
    └── massive_vocab_minimal - Test 256K vocab feasibility
    
    Phase 2 - Exploration (Week 2):
    ├── concept_heavy_512 - Test many-concept hypothesis
    ├── fibonacci_dims - Test natural proportions
    └── wide_shallow vs skinny_deep - Architecture comparison
    
    Phase 3 - Optimization (Week 3):
    └── Best performers from Phase 1-2 with masking strategies
    """)


if __name__ == "__main__":
    configurations = explore_micro_models()
    format_results(configurations)




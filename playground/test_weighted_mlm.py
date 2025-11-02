"""
Quick test script to verify the weighted MLM model works and loss decreases.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoderForMaskedLMWeighted

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Small config for quick testing
config = ConceptEncoderConfig(
    vocab_size=30522,
    concept_size=32,  # Small for testing
    hidden_size=256,
    num_hidden_layers=2,  # Shallow for quick testing
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=512,
    tie_word_embeddings=True  # Test with tied embeddings
)

# Initialize model and tokenizer
model = ConceptEncoderForMaskedLMWeighted(config).to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
model.train()

print("\nStarting training loop...")
losses = []

for step in range(100):
    # Create dummy data with more variety
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text."
    ] * 2  # Batch size 8
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    # Filter out token_type_ids if present (our model doesn't use them)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    
    # Create masked labels (mask 15% of tokens)
    labels = inputs["input_ids"].clone()
    mask_prob = torch.rand_like(labels.float()) < 0.15
    # Don't mask special tokens
    batch_size, seq_len = labels.shape
    special_tokens_mask = []
    for i in range(batch_size):
        mask = tokenizer.get_special_tokens_mask(labels[i].tolist(), already_has_special_tokens=True)
        special_tokens_mask.append(mask)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    mask_prob = mask_prob & ~special_tokens_mask
    
    labels[~mask_prob] = -100  # Ignore non-masked tokens
    inputs["input_ids"][mask_prob] = tokenizer.mask_token_id
    
    # Forward pass
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    losses.append(loss.item())
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 10 == 0:
        avg_loss = sum(losses[-10:]) / len(losses[-10:]) if len(losses) >= 10 else loss.item()
        print(f"Step {step:3d}, Loss: {loss.item():.4f}, Avg Loss (last 10): {avg_loss:.4f}")

# Analyze position weights
print("\n\nAnalyzing learned position weights...")
position_weights = model.get_position_weights_analysis()
print(f"Position weights shape: {position_weights.shape}")

# Show which concepts are most used for first 10 positions
print("\nTop 3 concepts for first 10 positions:")
for pos in range(min(10, position_weights.shape[0])):
    top_concepts = position_weights[pos].argsort()[-3:][::-1]
    top_weights = position_weights[pos][top_concepts]
    print(f"Position {pos}: concepts {top_concepts} with weights {top_weights}")

# Check if loss is decreasing
if len(losses) > 20:
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    improvement = (early_avg - late_avg) / early_avg * 100
    print(f"\nLoss improvement: {improvement:.1f}% (early avg: {early_avg:.4f}, late avg: {late_avg:.4f})")

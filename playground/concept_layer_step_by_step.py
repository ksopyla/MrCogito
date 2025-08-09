# %% testing the transformer Bert layer on a single sentence input
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer
from rich import print

# 1) Fix random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Create a sample sentence
sentence = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy cat.",
]

# Tokenize the sentence and convert to tensor
inputs = tokenizer(sentence, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


# print original sentend, the list of tokens and the tokens id

print("Tokenize sentence, tokens and ids")
print(sentence)
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
print(input_ids)
print("----------")

# %%

#shared token and concept embeddings dimensions
representation_dim = 128
ff_dim = 256
vocab_size = tokenizer.vocab_size
concept_length = 64

# batch size is equal to the number of the sentences, based on the input data seq_len dimension
batch_size = input_ids.shape[0]
num_heads = 2
num_layers = 1  

hidden_dropout_prob=0.0
attention_probs_dropout_prob=0.0

float_type = torch.bfloat16

#%%


token_emb_layer = torch.nn.Embedding(vocab_size, representation_dim, dtype=float_type)

# Convert tokens to embeddings
token_embeddings = token_emb_layer(input_ids)

# initalize the concept embeddings
concept_emb_layer = torch.nn.Embedding(concept_length, representation_dim, dtype=float_type)

concept_embeddings = concept_emb_layer(torch.arange(concept_length))
# Add batch dimension to match token_embeddings batch size
concept_embeddings = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, concept_length, representation_dim]

# debug print the shapes of the concept representations and the token embeddings,
print(f"concept_embeddings shape: {concept_embeddings.shape} of type {concept_embeddings.dtype}")
print(f"token_embeddings shape: {token_embeddings.shape} of type {token_embeddings.dtype}")


#%% define the layer normalization layers
pre_cross_attn_norm = nn.LayerNorm(representation_dim, dtype=float_type)
pre_self_attn_norm = nn.LayerNorm(representation_dim, dtype=float_type)
pre_ff_norm = nn.LayerNorm(representation_dim, dtype=float_type)


# %% deffine attention layers, cross attention between the concept representations and the token embeddings, and concept self attention

# cross attention between the concept representations and the token embeddings
concept_seq_attn = nn.MultiheadAttention(
    representation_dim, num_heads=num_heads, batch_first=True,
    dropout=attention_probs_dropout_prob,
    dtype=float_type,
)

# self attention on the concept representations
concept_self_attn = nn.MultiheadAttention(
    representation_dim, num_heads=num_heads, batch_first=True,
    dropout=attention_probs_dropout_prob,
    dtype=float_type,
)

# input to the concept attention layers is the concept embeddings
concept_representations = concept_embeddings

# %% compute the cross attention between the concept representations and the token embeddings

# apply the layer normalization (nn.LayerNorm) with residual connection
normed_concepts = pre_cross_attn_norm(concept_representations)


# Option1 = float-based, expand the attention mask to the shape [batch_size×num_heads, target_seq_len,source_seq_len] - if we need more flexibility
# at the beginning attention_mask shape => [batch_size, seq_len]
# we want a float-based mask of shape [batch_size×num_heads, target_seq_len, source_seq_len]

# expanded_attn_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
# expanded_attn_mask = expanded_attn_mask.expand(
#     batch_size,
#     concept_embeddings.size(1),  # concept_length
#     input_ids.size(1)           # seq_len
# )  # now => [batch_size, concept_length, seq_len]
# expanded_attn_mask = expanded_attn_mask.to(dtype=token_embeddings.dtype)
# expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(expanded_attn_mask.dtype).min

# # Insert a dimension for num_heads => [batch_size, 1, concept_length, seq_len]
# expanded_attn_mask = expanded_attn_mask.unsqueeze(1)
# expanded_attn_mask = expanded_attn_mask.repeat(1, num_heads, 1, 1)
# # => [batch_size, num_heads, concept_length, seq_len]

# # Flatten batch_size * num_heads => [batch_size * num_heads, concept_length, seq_len]
# expanded_attn_mask = expanded_attn_mask.view(
#     batch_size * num_heads,
#     concept_embeddings.size(1),
#     input_ids.size(1)
# )

# concept_seq_attn_output, concept_seq_attn_weights = concept_seq_attn(
#     normed_concepts,     # query => [batch_size, concept_length, hidden_dim]
#     token_embeddings,    # key   => [batch_size, seq_len, hidden_dim]
#     token_embeddings,    # value => [batch_size, seq_len, hidden_dim]
#     attn_mask=expanded_attn_mask
# )

# Option2 = boolean-based, “padding tokens should not contribute to attention
# Suppose attention_mask has shape [batch_size, seq_len] with 1=nonpad, 0=pad
# We need the opposite for key_padding_mask, which requires True at positions that should be masked out
key_padding_mask = (attention_mask == 0)  # bool of shape [batch_size, seq_len]
concept_seq_attn_output, concept_seq_attn_weights = concept_seq_attn(
    normed_concepts,     # query => [batch_size, concept_length, hidden_dim]
    token_embeddings,    # key   => [batch_size, seq_len, hidden_dim]
    token_embeddings,    # value => [batch_size, seq_len, hidden_dim]
    key_padding_mask=key_padding_mask
)


# residual connection
concept_representations = concept_representations + concept_seq_attn_output

# apply the layer normalization (nn.LayerNorm) with residual connection
normed_concepts = pre_self_attn_norm(concept_representations)

# compute the self concept attention on the output of the cross attention
concept_self_attn_output, concept_self_attn_weights = concept_self_attn(
    normed_concepts, normed_concepts, normed_concepts,
    attn_mask=None # no mask needed for concept self attention
)

concept_representations = concept_representations + concept_self_attn_output


# print the shapes of the output of the cross attention and the self attention
print(f"concept_seq_attn_output shape: {concept_seq_attn_output.shape}")
print(f"concept_self_attn_output shape: {concept_self_attn_output.shape}")


# %% compute the layer normalization (nn.LayerNorm) for the output of concept self attention with residual connection

normed_concepts = pre_ff_norm(concept_representations)

# apply the feed forward layer (nn.Linear) with GeLU activation and dropout with Wi,Wo weights with gate and dropout

# apply the feed forward layer (nn.Linear) with GeLU activation and dropout with Wi,Wo weights with gate and dropout
# https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/modernbert/modular_modernbert.py#L503
Wi = nn.Linear(representation_dim, ff_dim * 2, dtype=float_type)
Wo = nn.Linear(ff_dim, representation_dim, dtype=float_type)

wi_dropout = nn.Dropout(hidden_dropout_prob)
act_fn = nn.GELU()

ff_input, ff_gate = Wi(normed_concepts).chunk(2, dim=-1)

ff_output = Wo(wi_dropout(act_fn(ff_input) * ff_gate))

#residual connection
concept_representations = concept_representations + ff_output


# print the shape of the final concept representations
print(f"concept_representations shape: {concept_representations.shape}")

#%% mlm head
lm_concept2vocab = nn.Linear(representation_dim, vocab_size, bias=False, dtype=float_type)

lm_concept2vocab_output = lm_concept2vocab(concept_representations)

# print the shape of the final lm head output
print(f"lm_concept2vocab_output shape: {lm_concept2vocab_output.shape}")

lm_concept2seq = nn.Linear(representation_dim, input_ids.shape[1], bias=False, dtype=float_type)

lm_concept2seq_output = lm_concept2seq(concept_representations)

# print the shape of the final lm head output
print(f"lm_concept2seq_output shape: {lm_concept2seq_output.shape}")



# %% Compare with ConceptEncoder implementation
print("\n=== Comparing Step-by-Step vs ConceptEncoder Implementation ===")

import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add to path only if not already there
if project_root not in sys.path:
    sys.path.append(project_root)

# Then use the import
from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig

# Create config with matching dimensions
config = ConceptEncoderConfig(
    vocab_size=vocab_size,
    concept_size=concept_length,
    hidden_size=representation_dim,  # 128
    num_hidden_layers=num_layers,  # test with one layer
    num_attention_heads=2, #num_heads,  # test with one head
    intermediate_size=ff_dim,  # 256
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    cls_token_id=tokenizer.cls_token_id,
    sep_token_id=tokenizer.sep_token_id,
    mask_token_id=tokenizer.mask_token_id,
)

# Initialize ConceptEncoder with the config
concept_encoder = ConceptEncoder(config)

# 2) Disable dropout explicitly by eval() - ensures no random drops
concept_encoder.eval()


# Get concept representations from ConceptEncoder
with torch.no_grad():
    encoder_concept_output = concept_encoder(input_ids, attention_mask)

encoder_concept_representations = encoder_concept_output.last_hidden_state

# Print shapes for comparison
print("\nShape Comparison:")
print(f"Step-by-step concept_representations shape: {concept_representations.shape}")
print(f"ConceptEncoder output shape: {encoder_concept_representations.shape}")

# Compare the outputs (note: values will differ due to random initialization)
print("\nOutput Statistics Comparison:")
print("Step-by-step computation:")
print(f"Mean: {concept_representations.mean().item():.6f}")
print(f"Std: {concept_representations.std().item():.6f}")
print(f"Min: {concept_representations.min().item():.6f}")
print(f"Max: {concept_representations.max().item():.6f}")

print("\nConceptEncoder computation:")
print(f"Mean: {encoder_concept_representations.mean().item():.6f}")
print(f"Std: {encoder_concept_representations.std().item():.6f}")
print(f"Min: {encoder_concept_representations.min().item():.6f}")
print(f"Max: {encoder_concept_representations.max().item():.6f}")




#
# %%

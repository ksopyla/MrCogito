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

representation_dim = 128
ff_dim = 256
vocab_size = tokenizer.vocab_size
concept_length = 64

# batch size is equal to the number of the sentences, based on the input data seq_len dimension
batch_size = input_ids.shape[0]
num_heads = 1
num_layers = 1  

hidden_dropout_prob=0.0
attention_probs_dropout_prob=0.0

#%%
CONCEPT_ENCDEC_DOC_STRING = """
The example below is a test of the concept encoder-decoder approach. 
This tries to organize the computation step by step before it will be implemented as a full model. 

The model encoder is similar to the BERT or ModernBert encoder, it consists of the following layers:
1. The token embeddings and concept embeddings layers as Rotary embeddings
2. the position embeddings layer as Ro

3. The concept attention layer - cross attention between the concept tokens and the word tokens
4. The tokens attention layer - self attention on the word tokens
5. the layer normalization blocks - for almost all layers
6. The residual connections
5. The feed forward layer (MLP with GeLU activation and dropout with Wi,Wo weights) with ff_dim

The input data flow is as follows:
1. The input data is tokenized and converted to the input ids and attention mask.
2. the concept embeddings are randomly initialized
2. The input id are transformed to the token embeddings.
3. then cross attention is applied to the concept embeddings and the token embeddings
4. then self attention is applied to the output of the cross attention
5. the layer normalization is applied to the output of the self attention with residual connection
6. the feed forward layer is applied to the output of the layer normalization with residual connection
7. the layer normalization is applied to the output of the feed forward layer with residual connection


The cross attention between the concept tokens and the sequence tokens should be faster than the self attention on the word tokens due to the smaller
matrices dimensions.

Its better to multiply the concept tensor [concept_length, batch_size, embed_dim] with the sequence tensor [sequence_length, batch_size, embed_dim  ]
than the sequence tensor [sequence_length, batch_size, embed_dim] with itself because concept_length << sequence_length.
While computing the attention we need to mulitply (Q*K^T)*V 
* in the case of the self attention we have Q = K = V of shape [sequence_length, embed_dim], we aim to sequence_length be ~128K - 2M tokens, so the Q*K^T is very expensive operation in terms of time and memory [128K*embed_dim]*[embed_dim*128K] - as a result we got [128K*128K] matrix
* however in the case of the cross attention with concept tokens we have Q = [concept_length, embed_dim] where concept_length is 8-256, K = V = [sequence_length, embed_dim] this leads to [concept_length*embed_dim]*[embed_dim*sequence_length] - as a result we got [concept_length*sequence_length] matrix which is much smaller and can be stored in memory.

"""




token_emb_layer = torch.nn.Embedding(vocab_size, representation_dim)

# Convert tokens to embeddings
token_embeddings = token_emb_layer(input_ids)

# initalize the concept embeddings
concept_emb_layer = torch.nn.Embedding(concept_length, representation_dim)

concept_embeddings = concept_emb_layer(torch.arange(concept_length))
# Add batch dimension to match token_embeddings batch size
concept_embeddings = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, concept_length, representation_dim]

# debug print the shapes of the concept representations and the token embeddings,
print(f"concept_embeddings shape: {concept_embeddings.shape}")
print(f"token_embeddings shape: {token_embeddings.shape}")


#%% define the layer normalization layers
pre_cross_attn_norm = nn.LayerNorm(representation_dim)
pre_self_attn_norm = nn.LayerNorm(representation_dim)
pre_ff_norm = nn.LayerNorm(representation_dim)


# %% deffine attention layers, cross attention between the concept representations and the token embeddings, and concept self attention
concept_seq_attn = nn.MultiheadAttention(
    representation_dim, num_heads=num_heads, batch_first=True,
    dropout=attention_probs_dropout_prob,   
)
concept_self_attn = nn.MultiheadAttention(
    representation_dim, num_heads=num_heads, batch_first=True,
    dropout=attention_probs_dropout_prob,
)

# input to the concept attention layers is the concept embeddings
concept_representations = concept_embeddings

# %% compute the cross attention between the concept representations and the token embeddings

# apply the layer normalization (nn.LayerNorm) with residual connection
normed_concepts = pre_cross_attn_norm(concept_representations)

# Convert the 2D attention_mask into a float-based 3D mask, the same way ConceptEncoder does it.
# Shape => (batch_size, concept_seq_len, seq_len)
expanded_attn_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
expanded_attn_mask = expanded_attn_mask.expand(
    batch_size, concept_embeddings.size(1), input_ids.size(1)
)
expanded_attn_mask = expanded_attn_mask.to(dtype=token_embeddings.dtype)
expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(expanded_attn_mask.dtype).min

concept_seq_attn_output, concept_seq_attn_weights = concept_seq_attn(
    normed_concepts,
    token_embeddings,
    token_embeddings,
    attn_mask=expanded_attn_mask
)
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
Wi = nn.Linear(representation_dim, ff_dim * 2)
Wo = nn.Linear(ff_dim, representation_dim)

wi_dropout = nn.Dropout(hidden_dropout_prob)
act_fn = nn.GELU()

ff_input, ff_gate = Wi(normed_concepts).chunk(2, dim=-1)

ff_output = Wo(wi_dropout(act_fn(ff_input) * ff_gate))

#residual connection
concept_representations = concept_representations + ff_output


# print the shape of the final concept representations
print(f"concept_representations shape: {concept_representations.shape}")


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
    num_attention_heads=num_heads,  # test with one head
    intermediate_size=ff_dim,  # 256
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
)

# Initialize ConceptEncoder with the config
concept_encoder = ConceptEncoder(config)

# 2) Disable dropout explicitly by eval() - ensures no random drops
concept_encoder.eval()


# Get concept representations from ConceptEncoder
with torch.no_grad():
    encoder_concept_representations = concept_encoder(input_ids, attention_mask)

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
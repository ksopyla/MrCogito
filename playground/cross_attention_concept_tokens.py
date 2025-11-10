"""The script compares the time of the self attention and the cross attention with concept tokens.
The cross attention between the concept tokens and the sequence tokens should be faster than the self attention on the word tokens due to the smaller
matrices dimensions.

Its better to multiply the concept tensor [concept_length, batch_size, embed_dim] with the sequence tensor [sequence_length, batch_size, embed_dim  ]
than the sequence tensor [sequence_length, batch_size, embed_dim] with itself because concept_length << sequence_length.
While computing the attention we need to mulitply (Q*K^T)*V 
* in the case of the self attention we have Q = K = V of shape [sequence_length, embed_dim], we aim to sequence_length be ~128K - 2M tokens, so the Q*K^T is very expensive operation in terms of time and memory [128K*embed_dim]*[embed_dim*128K] - as a result we got [128K*128K] matrix
* however in the case of the cross attention with concept tokens we have Q = [concept_length, embed_dim] where concept_length is 8-256, K = V = [sequence_length, embed_dim] this leads to [concept_length*embed_dim]*[embed_dim*sequence_length] - as a result we got [concept_length*sequence_length] matrix which is much smaller and can be stored in memory.

"""

#%%
import torch
import torch.nn as nn

import time

#%% - the dim and size of the input
token_embed_dim = 256
concept_embed_dim = 512
num_heads = 1
batch_size = 2

concept_length = 64
sequence_length = 16384


# (batch_size, concept_length, embed_dim)
concept_embeddings = torch.rand(batch_size, concept_length, concept_embed_dim)

# (batch_size, sequence_length, embed_dim)
seq_x = torch.rand(batch_size, sequence_length, token_embed_dim)

#%% - Self attention 

# Initialize the MultiheadAttention module between the token embeddings
multihead_attn = nn.MultiheadAttention(token_embed_dim, num_heads, batch_first=True)

# Define the input tensors (query, key, value)

# Compute the selfattention output and time it
start_time = time.time()    
attn_output, attn_output_weights = multihead_attn(seq_x, seq_x, seq_x)
end_time = time.time()

print(f"Self attention Time taken: {end_time - start_time} seconds")

#%% - print the output
# print("Attention Output Shape:", attn_output.shape)
# print("Attention Output:", attn_output)

# print("--------------------------------")
# print("Attention Weights Shape:", attn_output_weights.shape)
# print("Attention Weights:", attn_output_weights)


#%% concept self attention

concept_self_attn = nn.MultiheadAttention(concept_embed_dim, num_heads, batch_first=True)
start_time = time.time()
concept_self_attn_output, concept_self_attn_weights = concept_self_attn(concept_embeddings, concept_embeddings, concept_embeddings)
end_time = time.time()
print(f"Concept self attention Time taken: {end_time - start_time} seconds")

# %% Concept tokens cross attention
# For cross attention with different dimensions:
# - embed_dim: dimension of queries (concepts) = concept_embed_dim
# - kdim: dimension of keys (tokens) = token_embed_dim  
# - vdim: dimension of values (tokens) = token_embed_dim
concept_seq_attn = nn.MultiheadAttention(
    embed_dim=concept_embed_dim, 
    num_heads=num_heads, 
    kdim=token_embed_dim, 
    vdim=token_embed_dim, 
    batch_first=True
)


# measure the wall clock time of the execution 
start_time = time.time()
concept_seq_attn_output, concept_seq_attn_weights = concept_seq_attn(concept_embeddings, seq_x, seq_x)
end_time = time.time()
print(f"Cross sequence attention Time taken: {end_time - start_time} seconds")



# %%


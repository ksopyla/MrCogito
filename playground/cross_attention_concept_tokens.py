#%%
import torch
import torch.nn as nn

# Define the model parameters
embed_dim = 4
num_heads = 1
batch_size = 2

# Initialize the MultiheadAttention module
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

# Define the input tensors (query, key, value)

# (concept_length, batch_size, embed_dim)
query = torch.rand(3, batch_size, embed_dim)  


# (sequence_length, batch_size, embed_dim)
key = torch.rand(5, batch_size, embed_dim)    
value = torch.rand(5, batch_size, embed_dim)  

# Compute the attention output
attn_output, attn_output_weights = multihead_attn(query, key, value)

#%%
print("Attention Output Shape:", attn_output.shape)
print("Attention Output:", attn_output)

print("--------------------------------")
print("Attention Weights Shape:", attn_output_weights.shape)
print("Attention Weights:", attn_output_weights)

# %%

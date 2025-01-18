#%% testing the transformer Bert layer on a single sentence input
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer
from rich import print

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Create a sample sentence
sentence = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy cat."]

# Tokenize the sentence and convert to tensor
inputs = tokenizer(sentence, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]



# print original sentend, the list of tokens and the tokens id
print(sentence)
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
print(input_ids)
#%%

#%% Create embeddings layer (we need this to convert tokens to embeddings)
config = BertConfig()
embedding_dim = config.hidden_size
vocab_size = config.vocab_size
token_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

# Convert tokens to embeddings, the shape is [batch_size, seq_len, hidden_size]
hidden_states = token_embeddings(input_ids)

# Initialize a single BERT layer
bert_layer = BertLayer(config)

# Create attention mask in the correct format
extended_attention_mask = attention_mask[:, None, None, :]
extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

# Pass the embeddings through the BERT layer
layer_outputs = bert_layer(hidden_states, extended_attention_mask)

# the shape of the output is [batch_size, seq_len, hidden_size]
layer_output = layer_outputs[0]  # Get the main output

# Print shapes and sample outputs
print("\nInput shapes:")
print(f"Input tokens shape: {input_ids.shape}")
print(f"Input embeddings shape: {hidden_states.shape}")
print(f"Layer output shape: {layer_output.shape}")

print("\nSample of layer output (first token, first 5 dimensions):")
print(layer_output[0, 0, :5])

# Decode back the input tokens for reference
print("\nOriginal tokens:")
print(tokenizer.convert_ids_to_tokens(input_ids[0]))

# %%

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

representation_dim = 128
ff_dim = 256
vocab_size = tokenizer.vocab_size
concept_length = 64

#batch size is equal to the number of the sentences, based on the input data seq_len dimension
batch_size = input_ids.shape[0] 


token_emb_layer = torch.nn.Embedding(vocab_size, representation_dim)

# Convert tokens to embeddings
token_embeddings = token_emb_layer(input_ids)

# initalize the concept embeddings tensor with random values
concept_representations = torch.rand(batch_size, concept_length, representation_dim)

# debug print the shapes of the concept representations and the token embeddings,
print(f"concept_representations shape: {concept_representations.shape}")
print(f"token_embeddings shape: {token_embeddings.shape}")

#%% deffine attention layers, cross attention between the concept representations and the token embeddings, and concept self attention
concept_seq_attn = nn.MultiheadAttention(representation_dim, num_heads=1, batch_first=True)
concept_self_attn = nn.MultiheadAttention(representation_dim, num_heads=1, batch_first=True)

#%% compute the cross attention between the concept representations and the token embeddings
concept_seq_attn_output, concept_seq_attn_weights = concept_seq_attn(concept_representations, token_embeddings, token_embeddings)

# compute the self attention on the output of the cross attention
concept_self_attn_output, concept_self_attn_weights = concept_self_attn(concept_seq_attn_output, concept_seq_attn_output, concept_seq_attn_output)


# print the shapes of the output of the cross attention and the self attention
print(f"concept_seq_attn_output shape: {concept_seq_attn_output.shape}")
print(f"concept_self_attn_output shape: {concept_self_attn_output.shape}")


#%% compute the layer normalization (nn.LayerNorm) for the output of concept self attention with residual connection 

attn_output_norm = nn.LayerNorm(representation_dim)
concept_representations = attn_output_norm(concept_seq_attn_output + concept_self_attn_output)

# apply the feed forward layer (nn.Linear) with GeLU activation and dropout with Wi,Wo weights with gate and dropout
# https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/modernbert/modular_modernbert.py#L503
Wi = nn.Linear(representation_dim, ff_dim*2)
Wo = nn.Linear(ff_dim, representation_dim)

wi_dropout = nn.Dropout(0.1)
act_fn = nn.GELU()

ff_input, ff_gate = Wi(concept_representations).chunk(2, dim=-1)

ff_output = Wo(wi_dropout(act_fn(ff_input) * ff_gate))

# apply the layer normalization (nn.LayerNorm) with residual connection

ff_output_norm = nn.LayerNorm(representation_dim)
concept_representations = ff_output_norm(ff_output + concept_representations)


# print the shape of the final concept representations
print(f"concept_representations shape: {concept_representations.shape}")











# %%


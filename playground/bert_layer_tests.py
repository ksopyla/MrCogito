# %% testing the transformer Bert layer on a single sentence input
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer
from rich import print

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
print(sentence)
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
print(input_ids)
# %%

# %% Create embeddings layer (we need this to convert tokens to embeddings)
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
extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
    torch.float32
).min

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


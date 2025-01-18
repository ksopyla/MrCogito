

# Research ideas

This work tries to address the problem of generating the coherent text and chat responses with use of encoder-decoder approach. The main idea lies in intuition that just predicting the next word is not enough and written text has as deeper structure that could be represent better with encoder model which will try to uncover this deeper meaning by using the concept tokens instead word tokens. Then the decoder will try to generate the text based on the concept tokens.


In addition to properly represent the underlying meaning of the text, we want to increase the context length of the model by using the concept tokens.

Its better to multiply the concept tensor [concept_length, batch_size, embed_dim] with the sequence tensor [sequence_length, batch_size, embed_dim  ]
than the sequence tensor [sequence_length, batch_size, embed_dim] with itself because concept_length << sequence_length.
While computing the attention we need to mulitply (Q*K^T)*V 
* in the case of the self attention we have Q = K = V of shape [sequence_length, embed_dim], we aim to sequence_length be ~128K - 2M tokens, so the Q*K^T is very expensive operation in terms of time and memory [128K*embed_dim]*[embed_dim*128K] - as a result we got [128K*128K] matrix
* however in the case of the cross attention with concept tokens we have Q = [concept_length, embed_dim] where concept_length is 8-256, K = V = [sequence_length, embed_dim] this leads to [concept_length*embed_dim]*[embed_dim*sequence_length] - as a result we got [concept_length*sequence_length] matrix which is much smaller and can be stored in memory.


This work is inspired by the papers:

* "Memory Transformer"
*  "ConceptBERT: A Concept-based Framework for Pre-training Language Models"
* "Large Concept Models" by Meta


## Attending to the concept tokens

Define the concept tokens and train the encoder with added concept tokens to original sequence. 

The concept tokens could be added at the beginning of the sequence with different attention schemes. 

The number of concept tokens will be fixed from 8 to 128, they could act as register (this was mentioned in one of the paper with vision encoder, that dino model with more registers could be better and attention is more explainable)





## How to replace unused tokens in a model


I want to replace the name of the unused tokens in a model with a new name, the idea is to use those unused tokens as concept tokens. 

Vocab unused tokens replacement: 


* https://github.com/huggingface/transformers/issues/31475 
https://github.com/huggingface/transformers/issues/27974
* https://discuss.huggingface.co/t/change-gemma-tokenizer-unused-token/80867/2 
* 


Untested code, by copilot:
```
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the mapping of old tokens to new tokens
token_mapping = {
    "<unused0>": "<NEW_TOKEN1>",
    "<unused1>": "<NEW_TOKEN2>"
}

# Update the tokenizer's vocabulary
for old_token, new_token in token_mapping.items():
    if old_token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(old_token)
        tokenizer.add_tokens([new_token])
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        tokenizer.vocab[new_token] = new_token_id
        del tokenizer.vocab[old_token]

# Save the updated tokenizer
tokenizer.save_pretrained("./updated_tokenizer")
```


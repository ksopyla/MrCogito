

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



## Prompts 

** Analys and improve the concept encoder layer **

You are a machine learning research and ai engineer. With deep knowledge of current AI neural networks architecuture.
You have access to @Hugging-Face-Transformers and @Pytorch documentation, you can use your knowledge about research articles from arxiv.
You are helping me to invent new architecture, some of my reserch notes are in reserch_notes.md  . My base idea is to use concepts insted of tokens, concept is more abstract mental model based on group of the tokens, each concept attent to tokens (via cross-attention). I'm building the encoder decoder architecture, but now focus on encoder part. I 

Please read and analyse the code, help me improve idea from the teoretical point of view as well as practical by fixing the code errors, wrong use of function, tensor shapes mismatch. 

Give me a llist of further improvements for consideration. 




### How to compute the logits for MLM - ideas

Now I seen the flows in my thinking while computing the logits for MLM. 
Enoder computes the concept_representation of shape [batch_size, concept_size, hidden_size] and I should map it in some way to [batch_size, sequence_len, vocab_size]. 

One of my idea is to do as follow: 
1. multiply the sequence token embeddings of [batch_size, sequence_len, hidden_dim] by concept_representation transpose  [batch_size, hidden_dim, concept_len] -> [batch_size, sequence_len, concept_len] - this allows to match tokens from sequence to conepts
2. make an linear projecttion to [batch_size, sequence_len, vocab_size] to get the sequence logits


# multiple masks 


* [Iterative Mask Filling: An Effective Text Augmentation Method Using Masked Language Modeling](https://arxiv.org/abs/2401.01830) (2024)
* [Blank language model](https://arxiv.org/abs/2002.03079) (2020)
* transformer pr - https://github.com/huggingface/transformers/pull/10222

* PMI-Masking: Principled masking of correlated spans
* SpanBERT: Improving Pre-training by Representing and Predicting Spans
* 


Morphological aware tokenization: 
* MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies

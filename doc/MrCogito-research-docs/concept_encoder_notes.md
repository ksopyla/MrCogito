# Research ideas

This work tries to address the problem of generating the coherent text and chat responses with use of encoder-decoder or diffusion basedapproach. 
The main idea lies in intuition that just predicting the next word is not enough and written text has as deeper structure that could be represent better with encoder model which will try to uncover this deeper meaning by using the concept tokens instead word tokens. Then the decoder (autoregressive or diffusion) will try to generate the text based on the concept tokens.


In addition to properly represent the underlying meaning of the text, we want to increase the context length of the model by using the concept tokens.

Its better to multiply the concept tensor by the sequence tensor than the sequence tensor by itself because $\text{concept\_length} \ll \text{sequence\_length}$.
While computing the attention we need to multiply $(Q \cdot K^T) \cdot V$

**Self attention computation** between the sequence "token" tensors, ends up with matrix multiplication: 

* In the case of the self attention we have $Q = K = V$ of shape $[sequence\_length, embed\_dim]$, we aim to have sequence_length be ~128K - 2M tokens, so the $Q \cdot K^T$ is very expensive operation in terms of time and memory $[128K \times embed\_dim] \cdot [embed\_dim \times 128K]$ - as a result we get $[128K \times 128K]$ matrix, 
* for 128k context length this needs storing $128*1024*128*1024/(1024*1024*1024)=16G$ float numbers


**Cross attention computation** between the concept and sequence "token" tensors, ends up with less expensive operation: 
* concept tokens are stored as $Q = [concept\_length, embed\_dim]$ where concept_length could be in a range of 32-2048, sequence "tokens" are stored as $K = V = [sequence\_length, embed\_dim]$ 
* this leads to matrix multiplication $[concept\_length \times embed\_dim] \cdot [embed\_dim \times sequence\_length]$ - as a result we get $[concept\_length \times sequence\_length]$ matrix which is **much smaller**

When we add batch dimension to the tensors we get:

$Q=[concept\_length, batch\_size, embed\_dim]$ 

$K=V=[sequence\_length, batch\_size, embed\_dim]$ 



This work was initially inspired by the papers:

* "Memory Transformer"
*  "ConceptBERT: A Concept-based Framework for Pre-training Language Models"
* "Large Concept Models" by Meta
* "LLaDA diffusion model"


## Evaluation

[evaluation_strategies](evaluation_strategies.md)


## Research log 

Notes, ideas, thoughts, questions and material gathered during the research.

### Idea 1 - use the unsued BERT or ModernBERT tokens as concept tokens

Do not crate new architecture, just add additional concept tokens at the begining of the sequence, without changing how the attention is computed. Stay with the self attention - those added tokens could act like registers - Memory Transformer idea - there is also paper Vision transformer need registers - https://arxiv.org/abs/2309.16588 
Those registers could form the concepts while training, as they are not explicitly used, previous papers suggest that during training the model learns to use those registers for some tasks.

Some idea variants:
1. User unsed tokens and treat them as the concept tokens and train the encoder with artificcaly added concept tokens to original sequence. Do not change the self attention mechnism, don't have to use any mask in attention.
2. We could experiment with differnet atttention masks, "concept" tokens could only attend to the sequence tokens, or they could attend to each other.
3. The number of concept tokens will be fixed from 8 to 128, they could act as register (this was mentioned in one of the paper with vision encoder, that dino model with more registers could be better and attention is more explainable)


#### Conclusions

This idea was explored in the "memory transformer" paper, but was not evaluated in a larger scale, with more complex tasks and modern data. Worth to get back to this idea. It is simple and could be a good baseline.

[Idea 1 - details](experiments/idea1_unused_tokens_as_concepts.md)



## Idea 2 - Concept Encoder with MLM traning - concept-sequence tokens cross attention

Start with the concept encoder model, like BERT, ModernBERT or LLaDA. Train the model with MLM task and use cross attention to connect the concept tokens to the sequence tokens. 


### How to compute the logits for MLM - ideas

Now I seen the flows in my thinking while computing the logits for MLM. 
Enoder computes the concept_representation of shape $[batch\_size, concept\_size, hidden\_size]$ and I should map it in some way to $[batch\_size, sequence\_len, vocab\_size]$. 

One of my idea is to do as follow: 
1. Multiply the sequence token embeddings of $[batch\_size, sequence\_len, hidden\_dim]$ by concept_representation transpose  $[batch\_size, hidden\_dim, concept\_len]$ → $[batch\_size, sequence\_len, concept\_len]$ - this allows to match tokens from sequence to concepts
2. Make a linear projection to $[batch\_size, sequence\_len, vocab\_size]$ to get the sequence logits


## Idea 3 - Multiple token masking for proper concept encoding

My intuition tells me that to train the proper concepts we can't use the token masking, we need to use the multiple token masking. Concept is more abstract and it is not just one token, it is a group of the tokens. 
Similar to META paper Large Concept Models - they treat each sentence as a concept.
I see this a little different, the concept encoding should grab the information from a neighborhood group of the tokens.


* [Iterative Mask Filling: An Effective Text Augmentation Method Using Masked Language Modeling](https://arxiv.org/abs/2401.01830) (2024)
* [Blank language model](https://arxiv.org/abs/2002.03079) (2020)
* transformer pr - https://github.com/huggingface/transformers/pull/10222

* PMI-Masking: Principled masking of correlated spans
* SpanBERT: Improving Pre-training by Representing and Predicting Spans


## Idea 4 - Morphological aware tokenization

Following the idea 3, that for better concept encoding we need to use the multiple token masking, we should use the morphological aware tokenization. 
Current tokenizers are not able to properly tokenize the words with morphological variations and split the words for root and morphemes.
Intuition behind this: root and morphemes contains meaning

Experiments 
1. Test the morfessor tokenization - experiments shows that it is able to split the words for root and morphemes better than the other tokenizers BPE, Unigram, WordPiece. However the vocabulary size is too big, this lead to idea to train unigram model based on the morfessor segments. 

2. To check: train the morfessor model not on words but on sentences, in theory this should learn how to split the sentences for concepts - should be validated

3. Evaluate the quality of the tokenization alg. by using the BLEU score between the words and morphological segments 
```python
ground_truth_morphems = {
    "windsurfing" : ["wind surf ing", "wind surfing"],
    "kitesurfing" : ["kite surf ing", "kite surfing"],
    "unfortunately" : ["un unfortunately", "un fortunate ly"],
}
```
Some words could be taken from [MorphoLex-en](https://github.com/hugomailhot/MorphoLex-en/tree/master)

### Conclusions and Recommendations - TL;DR

Detailed analysis is in the [experiment_morphological_aware_tokenization.md](experiment_morphological_aware_tokenization.md)

Considering both performance and practical aspects, I have decided to train the custom unigram model on the modern data, the XLNet tokenizer is a good choice for first "concept encoding" experiments but due to quite old training procedure without any modern text pile (code, slang, chat, multilanguage, etc) we should train the custom tokenizer.
The unigram model trained in similar fashion as **uni_wikipedia_words_1M_min_7_nltk** with mixed data.
The procedure: 
1. Unigram model trained on filtered set of moderately frequent words (≥7 occurrences)
2. Training should be done on words (meaningfull tokens or conepts ). NLTK's word_tokenize is enough for word extraction, w
3. It is sufficient to train directly on words without Morfessor preprocessing
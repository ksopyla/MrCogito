

# Research ideas

This work tries to address the problem of generating the coherent text and chat responses with use of encoder-decoder approach. The main idea lies in intuition that just predicting the next word is not enough and written text has as deeper structure that could be represent better with encoder model which will try to uncover this deeper meaning by using the concept tokens instead word tokens. Then the decoder will try to generate the text based on the concept tokens.


In addition to properly represent the underlying meaning of the text, we want to increase the context length of the model by using the concept tokens.

Its better to multiply the concept tensor $[concept_length, batch_size, embed_dim]$ with the sequence tensor $[sequence_length, batch_size, embed_dim  ]$ than the sequence tensor $[sequence_length, batch_size, embed_dim]$ with itself because $concept_length << sequence_length$.
While computing the attention we need to mulitply $(Q*K^T)*V$ :
* in the case of the self attention we have $Q = K = V$ of shape $[sequence_length, embed_dim]$, we aim to sequence_length be ~128K - 2M tokens, so the $Q*K^T$ is very expensive operation in terms of time and memory $[128K*embed\_dim]*[embed\_dim*128K]$ - as a result we got $[128K*128K]$ matrix
* however in the case of the cross attention with concept tokens we have $Q = [concept\_length, embed\_dim]$ where concept_length is 8-256, $K = V = [sequence\_length, embed\_dim]$ this leads to $[concept\_length*embed\_dim]*[embed\_dim*sequence\_length]$ - as a result we got $[concept\_length*sequence\_length]$ matrix which is **much smaller and bigger context** can be stored in memory.


This work is inspired by the papers:

* "Memory Transformer"
*  "ConceptBERT: A Concept-based Framework for Pre-training Language Models"
* "Large Concept Models" by Meta


## Prompts for AI assistant

** Analys and improve the concept encoder layer **

You are a machine learning research and ai engineer. With deep knowledge of current AI neural networks architecuture.
You have access to @Hugging-Face-Transformers and @Pytorch documentation, you can use your knowledge about research articles from arxiv.
You are helping me to invent new architecture, some of my reserch notes are in reserch_notes.md  . My base idea is to use concepts insted of tokens, concept is more abstract mental model based on group of the tokens, each concept attent to tokens (via cross-attention). I'm building the encoder decoder architecture, but now focus on encoder part. I 

Please read and analyse the code, help me improve idea from the teoretical point of view as well as practical by fixing the code errors, wrong use of function, tensor shapes mismatch. 

Give me a list of further improvements for consideration. 




## Research log 

Notes, ideas, thoughts, questions and material gathered during the research.


### Idea 1 - use the unsued BERT or ModernBERT tokens as concept tokens

Do not crate new architecture, just add additional concept tokens at the begining of the sequence, withou changing how the attention is computed. Stay with the self attention - those added tokens could act like registers - Memory Transformer idea - there is alos paper Vision transformer need registers - https://arxiv.org/abs/2309.16588 

Define the concept tokens and train the encoder with added concept tokens to original sequence. 

The concept tokens could be added at the beginning of the sequence with different attention schemes. 

The number of concept tokens will be fixed from 8 to 128, they could act as register (this was mentioned in one of the paper with vision encoder, that dino model with more registers could be better and attention is more explainable)


#### How to replace unused tokens in a model


I want to replace the name of the unused tokens in a model with a new name, the idea is to use those unused tokens as concept tokens. 

Vocab unused tokens replacement: 

* https://github.com/huggingface/transformers/issues/31475 
https://github.com/huggingface/transformers/issues/27974
* https://discuss.huggingface.co/t/change-gemma-tokenizer-unused-token/80867/2 
* 

Untested code, by copilot:

```python
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

## Idea 2 - Concept Encoder with MLM traning - concept-sequence tokens cross attention

Start with the concept encoder model, like BERT. Train the model with MLM task and use cross attention to connect the concept tokens to the sequence tokens. 


### How to compute the logits for MLM - ideas

Now I seen the flows in my thinking while computing the logits for MLM. 
Enoder computes the concept_representation of shape $[batch\_size, concept\_size, hidden\_size]$ and I should map it in some way to $[batch\_size, sequence\_len, vocab\_size]$. 

One of my idea is to do as follow: 
1. multiply the sequence token embeddings of $[batch\_size, sequence\_len, hidden\_dim]$ by concept_representation transpose  $[batch\_size, hidden\_dim, concept\_len]$ -> $[batch\_size, sequence\_len, concept\_len]$ - this allows to match tokens from sequence to conepts
2. make an linear projecttion to $[batch\_size, sequence\_len, vocab\_size]$ to get the sequence logits




## Idea 3 - Multiple token masking for proper concept encoding

My intuition tellm me that to train the proper concepts we can't use the token masking, we need to use the multiple token masking. Concept is more abstract and it is not just one token, it is a group of the tokens. 
Similar to META paper Large Concept Models - they treat each sentence as a concept.
I see this a little different, the concept encoding should grab the information from a neighborhood group of the tokens.


* [Iterative Mask Filling: An Effective Text Augmentation Method Using Masked Language Modeling](https://arxiv.org/abs/2401.01830) (2024)
* [Blank language model](https://arxiv.org/abs/2002.03079) (2020)
* transformer pr - https://github.com/huggingface/transformers/pull/10222

* PMI-Masking: Principled masking of correlated spans
* SpanBERT: Improving Pre-training by Representing and Predicting Spans
* 


## Idea 4 - Morphological aware tokenization

Following the idea 3, that for better concept encoding we need to use the multiple token masking, we should use the morphological aware tokenization. 
Current tokenizers are not able to properly tokenize the words with morphological variations and split the words for root  and morphemes.
Intuition behind this: root and morphemes contains meaning

Experiments 
1. test the morfessor tokenization - experiments shows that it is able to split the words for root and morphemes better than the other tokenizers BPE, Unigram, WordPiece. However the vocabulary size is too big, this lead to idea to train unigram model based on the morfessor segments. 

2. to check: train the morfessor model not on words but on sentences, in theory this should learn how to split the sentences for concepts - should be validated

3. Evalute the quality of the tokenization alg. by using the BLEU score between the  words and morphological segments 
```python
ground_truth_morphems = {
    "windsurfing" : ["wind surf ing", "wind surfing"],
    "kitesurfing" : ["kite surf ing", "kite surfing"],
    "unfortunately" : ["un fortunately", "un fortunate ly"],
}
```
Some words could be takene from [MorphoLex-en](https://github.com/hugomailhot/MorphoLex-en/tree/master)


After the first experiments I have found that Morfessor and XLM tokenizer give the best results. But Morfessor is not able to prune the vocabulary to desired size. 
Thats why I have decided to use the Morfessor to split the words into the root and morphemes and then train the unigram model on the morphological segments.




* MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies
* Byte Pair Encoding is Suboptimal for Language Model Pretraining 2020 - https://arxiv.org/pdf/2004.03720 - tl;dr: BPE is not good for morphologically rich languages, unigram is better
* Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidate - 2018 - https://arxiv.org/pdf/1804.10959 - tl;dr: adding regularization as subsumpling give better results for unigram model for translation task, this is similar in nature what I try to do, train the unigram model on the morphological segments (eg from morfessor) 

* Morfessor - https://github.com/aalto-speech/morfessor
* https://aayushsanghavi.blogspot.com/2018/03/morphological-segmentation-of-words.html



### Morfessor package 

Package for training the morphological segmentation of the words
After some experiments with Morfessor package, I found that it is able to train the model to split the words for root and morphemes. Very good results.
Sample code form https://aayushsanghavi.blogspot.com/2018/03/morphological-segmentation-of-words.html


### Experiments 

While morfessor is not able to generate the vocabulary of desired size, I have tried to train custom unigram tokenizer. 
I want to check if the unigram tokenizer trained on the morphological segments will give better results than the unigram tokenizer trained on the words.



#### Exp 1. 

Date: 24.02.2025 - 18:00
Commit: f337749d189b1d9f764b05d57f741e393e2b0a01
Files: 
* [train_tokenizer_playground.py](../playground/train_tokenizer_playground.py) 
* [train_morfessor_models.py](../training/train_morfessor_models.py)

Description:
Initial experiment with a few morfessor models, should not be used for deciding which approch is better. This was just to start and see some first results. 
The resulta were evaluated on [ground_truth.py](../playground/ground_truth.py) dictionary with arbitrary choosen words and their ground truth morphological segments.
This is not a big dataset, but it is enough to see some first results. 
This evaluation procedure does not take into account the results on 'programming code', 'latex' text. 

Loaded some standard tokenizers from Hugging Face and trained the Morfessor models on the sentences.


morfessor_* - models are pure morfessor models trained on different corpora
morferssor_nltk - is traind on nltk words corpus - it contains 236736 words
morfessor_wiki - is trained on wikitext dataset that was alredy preprocessed and words are splited corpus - it contains 101425671 non unique words
morfessor_sent - is trained on the sentences from the wikitext dataset - it contains 4290307 sentences
morfessor_wikipedia_300M - is trained on the wikipedia dataset that was alredy preprocessed and words are splited corpus - model trainined on 300M articles from wikipedia, giving the 314572800 words
uni_wiki - unigram tokenizer trained on prepreocess wikitext dataset with use of morfessor_wiki model
uni_wikipedia - unigram tokenizer trained on prepreocess wikipedia dataset with use of morfessor_wikipedia_300M model (later called uni_wikipedia_300M)


BLEU Scores for each tokenizer     

| Tokenizer                | BLEU Score | 1-gram | 2-gram | 3-gram |
|--------------------------|------------|--------|--------|--------|
| bert                     | 0.3550     | 0.5177 | 0.3457 | 0.2500 |
| modernbert               | 0.2903     | 0.5725 | 0.3846 | 0.1111 |
| llama32                  | 0.3041     | 0.6462 | 0.4571 | 0.0952 |
| gpt2                     | 0.2761     | 0.6183 | 0.4085 | 0.0833 |
| **xlnet**                | **0.6412** | 0.7119 | 0.6552 | 0.5652 |
| albert                   | 0.4332     | 0.5081 | 0.4219 | 0.3793 |
| morfessor_nltk           | 0.4882     | 0.6029 | 0.5263 | 0.3667 |
| morfessor_wiki           | 0.6948     | 0.6782 | 0.8889 | 1.0000 |
| morfessor_sent           | 0.7105     | 0.6932 | 0.8929 | 1.0000 |
| morfessor_wikipedia_300M | 0.0000     | 0.4928 | 0.7778 | 0.0000 |
| uni_wiki                 | 0.2723     | 0.5247 | 0.3333 | 0.1154 |
| uni_wikipedia            | 0.0000     | 0.5245 | 0.2410 | 0.0000 |


#### Exp 2. 

Date: 25.02.2025 - 19:00
Commit: 
Files: 
* [train_tokenizer_playground.py](../playground/train_tokenizer_playground.py) 
* [train_morfessor_models.py](../training/train_morfessor_models.py)

Description:
More reliable experiment then previous one (Exp 1). The ground truth [ground_truth.py](../playground/ground_truth.py) was extended with more words and morphological segments based on the MorphoLex-en.

Loaded some standard tokenizers from Hugging Face and trained the Morfessor models on the sentences.
morfessor_* - models are pure morfessor models trained on different corpora
morferssor_nltk - is traind on nltk words corpus - it contains 236736 words
morfessor_wiki - is trained on wikitext dataset that was alredy preprocessed and words are splited corpus - it contains 101425671 non unique words
morfessor_sent - is trained on the sentences from the wikitext dataset - it contains 4290307 sentences
morfessor_wikipedia_300M - is trained on the wikipedia dataset that was alredy preprocessed and words are splited corpus - model trainined on 300M articles from wikipedia, giving the 314572800 words (morfessor_wikipedia_en_train_words_300M.txt)

morfessor_wikipedia_10M - is trained on the wikipedia dataset that was alredy preprocessed and words are splited corpus - model trainined on 10M words subset of morfessor_wikipedia_en_train_words_300M



uni_wiki - unigram tokenizer trained on prepreocess wikitext dataset with use of morfessor_wiki model
uni_wikipedia_300M - unigram tokenizer trained on prepreocess wikipedia dataset with use of morfessor_wikipedia_300M model (later called uni_wikipedia_300M)




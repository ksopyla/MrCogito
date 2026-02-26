# Idea 4 - Morphological aware tokenization 


* [Idea 4 - Morphological aware tokenization](#idea-4---morphological-aware-tokenization)
        * [Recommended Approach - TL;DR](#recommended-approach---tldr)
    * [Morphological aware tokenization intuitions and hypothesis](#morphological-aware-tokenization-intuitions-and-hypothesis)
    * [Experiments](#experiments)
        * [Exp 1.](#exp-1)
        * [Exp 2.](#exp-2)
    * [Conclusions and Recommendations](#conclusions-and-recommendations)
        * [Analysis and Practical Considerations](#analysis-and-practical-considerations)

## Recommended Approach - TL;DR

Considering both performance and practical aspects, I have decided to train the custom unigram model on the modern data, the XLNet tokenizer is a good choice for first "concept encoding" experiments but due to quite old training procedure without any modern text pile (code, slang, chat, multilanguage, etc) we should train the custom tokenizer.
The unigram model trained in similar fashion as **uni_wikipedia_words_1M_min_7_nltk** with mixed data.
The procedure: 
1. Unigram model trained on filtered set of moderately frequent words (≥7 occurrences)
2. Training should be done on words (meaningfull tokens or conepts ). NLTK's word_tokenize is enough for word extraction, w
3. It is sufficient to train directly on words without Morfessor preprocessing 

## Morphological aware tokenization intuitions and hypothesis

Following the [concept_encoder_notes](../concept_encoder_notes.md#idea-4---morphological-aware-tokenization) that for better concept encoding we need to use the multiple token masking, we should use the morphological aware tokenization. 
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
    "unfortunately" : ["un unfortunately", "un fortunate ly"],
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

**Morfessor package**

Package for training the morphological segmentation of the words
After some experiments with Morfessor package, I found that it is able to train the model to split the words for root and morphemes. Very good results.
Sample code form https://aayushsanghavi.blogspot.com/2018/03/morphological-segmentation-of-words.html


## Experiments 

While morfessor is not able to generate the vocabulary of desired size, I have tried to train custom unigram tokenizer. 
I want to check if the unigram tokenizer trained on the morphological segments will give better results than the unigram tokenizer trained on the words.



### Exp 1. 

Date: 24.02.2025 - 18:00
Commit: f337749d189b1d9f764b05d57f741e393e2b0a01
Files: 
* [train_tokenizer_playground.py](../playground/train_tokenizer_playground.py) 
* [train_morfessor_models.py](../training/train_morfessor_models.py)

Description:
Initial experiment with a few morfessor models, should not be used for deciding which approch is better. This was just to start and see some first results. 
The resulta were evaluated on [ground_truth.py](../playground/data/ground_truth.py) dictionary with arbitrary choosen words and their ground truth morphological segments.
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


### Exp 2. 

Date: 25.02.2025 - 19:00
Commit: f337749d189b1d9f764b05d57f741e393e2b0a01
Files: 
* [train_tokenizer_playground.py](../playground/train_tokenizer_playground.py) 
* [train_morfessor_models.py](../training/train_morfessor_models.py)

**Description and methodology:**

More reliable experiment than previous one (Exp 1). The ground truth [ground_truth.py](../playground/ground_truth.py) was extended with more words and morphological segments based on the MorphoLex-en.

**Preprocessing methods:**

Several preprocessing approaches were used in this experiment:
1. **Direct tokenization**: Using standard tokenizers without modification
2. **Morfessor segmentation**: Pure morphological segmentation using various Morfessor models
3. **Two-stage processing**: First segmenting with Morfessor, then training a unigram tokenizer on the morphological segments

For Morfessor models, preprocessing included:
- Loading text from various sources (NLTK, WikiText, Wikipedia)
- Word extraction using either simple split or NLTK's word_tokenize
- Optional frequency filtering (min_3, min_7 occurrences)
- Optional count modification using logarithmic scaling (log_func)

The dataset processing utilized multiprocessing (up to 60 processes) and batch sizes of 1000-5000 for efficiency. Processed datasets were cached to avoid redundant processing.

**Tokenizers description, training procedure and results:**

Loaded some standard tokenizers from Hugging Face and trained the Morfessor models on the sentences.
morfessor_* - models are pure morfessor models trained on different corpora, training was done with use of [train_morfessor_models.py](../training/train_morfessor_models.py) script


* **morfessor_nltk** - trained on NLTK words corpus - it contains 236,736 words, [morfessor_wikipedia_en_train_words_10M.txt](../Cache/Morfessor/morfessor_wikipedia_en_train_words_10M.txt)
* **morfessor_wiki** - trained on WikiText dataset that was already preprocessed and words are split corpus - it contains 101,425,671 non-unique words, [morfessor_wiki_en_train.txt](../Cache/Morfessor/morfessor_wiki_en_train.txt)
* **morfessor_sent** - trained on the sentences from the WikiText dataset - it contains 4,290,307 sentences, [morfessor_wiki_en_train_sentences.txt](../Cache/Morfessor/morfessor_wiki_en_train_sentences.txt)
* **morfessor_wikipedia_300M** - trained on the Wikipedia dataset that was preprocessed with words split - model trained on 300M articles from Wikipedia, containing 314,572,800 words
* **morfessor_wikipedia_10M** - trained on a 10M word subset of the morfessor_wikipedia_en_train_words_300M dataset
* **morfessor_wikipedia_1M_unique** - trained on unique words extracted from 1M Wikipedia articles, words split using Python's split() function, count_modifier=lambda x: 1, containing 10,585,465 unique words
* **morfessor_wikipedia_1M_unique_nltk** - trained on unique words extracted from 1M Wikipedia articles, words split using NLTK's word_tokenize, count_modifier=lambda x: 1, containing 5,838,892 unique words
* **morfessor_wikipedia_1M_min_3_nltk** - trained on words that occur at least 3 times across 1M Wikipedia articles, tokenized with NLTK, containing 1,690,998 words
* **morfessor_wikipedia_1M_min_7_nltk** - trained on words that occur at least 7 times across 1M Wikipedia articles, tokenized with NLTK, containing 884,821 words
* **morfessor_wikipedia_1M_min_7_nltk_log** - same dataset as morfessor_wikipedia_1M_min_7_nltk but using logarithmic count modifier (log_func: log₂(frequency+1))

Unigram tokenizers were trained using the SentencePiece Unigram algorithm with a vocabulary size of 32,768 tokens:

* **uni_wiki** - unigram tokenizer trained on WikiText dataset (1M articles) preprocessed by morfessor_wiki model, trained on created dataset [wikitext_wikitext-103-v1_1000000_wiki_morphems](../Cache/Dataset/wikitext_wikitext-103-v1_1000000_wiki_morphems) with use of `get_preprocessed_morfessor_dataset` function
* **uni_wikipedia_300M** - unigram tokenizer trained on preprocessed Wikipedia 1M articles, text preprocessed with use of [morfessor_wikipedia_en_train_words_300M](../Cache/Morfessor/morfessor_wikipedia_en_train_words_300M.bin) model and `get_preprocessed_morfessor_dataset`, trained on created dataset [wikipedia_20231101.en_1000000_wikipedia300m_morphems](../Cache/Datasets/wikipedia_20231101.en_1000000_wikipedia300m_morphems/)
* **uni_wikipedia_10M** - unigram tokenizer trained on preprocessed Wikipedia 1M articles, text preprocessed with use of [morfessor_wikipedia_en_train_words_10M](../Cache/Morfessor/morfessor_wikipedia_en_train_words_10M.bin) model and `get_preprocessed_morfessor_dataset`, trained on created dataset [wikipedia_20231101.en_10000000_wikipedia10m_morphems](../Cache/Datasets/wikipedia_20231101.en_10000000_wikipedia10m_morphems/) 
* **uni_wikipedia_10M_unique** - unigram tokenizer trained on preprocessed Wikipedia 1M articles, text preprocessed with use of [morfessor_wikipedia_en_train_words_unique_split_1M_art](../Cache/Morfessor/morfessor_wikipedia_en_train_words_unique_split_1M_art.bin) model and `get_preprocessed_morfessor_dataset`, trained on created dataset [wikipedia_20231101.en_1000000_wikipedia1m_unique_morphems](../Cache/Datasets/wikipedia_20231101.en_1000000_wikipedia1m_unique_morphems/)
* **uni_wikipedia_10M_unique_tok** - unigram tokenizer trained on 1M Wikipedia articles preprocessed with morfessor_wikipedia_1M_unique_nltk model, dataset [wikipedia_20231101.en_1000000_wikipedia1m_unique_tok_morphems](../Cache/Datasets/wikipedia_20231101.en_1000000_wikipedia1m_unique_tok_morphems/)
* **uni_wikipedia_1M_unique_3M_words** - unigram tokenizer trained on 1M Wikipedia articles preprocessed with morfessor_wikipedia_1M_min_3_nltk model, dataset [wikipedia_20231101.en_1000000_wikipedia1m_min_3_nltk_morphems](../Cache/Datasets/wikipedia_20231101.en_1000000_wikipedia1m_min_3_nltk_morphems/)
* **uni_wikipedia_1M_min_7_nltk** - unigram tokenizer trained on 1M Wikipedia articles preprocessed with morfessor_wikipedia_1M_min_7_nltk model, dataset [wikipedia_20231101.en_1000000_wikipedia1m_min_7_nltk_morphems](../Cache/Datasets/wikipedia_20231101.en_1000000_wikipedia1m_min_7_nltk_morphems/)
* **uni_normal_wikipedia** - unigram tokenizer trained directly on raw Wikipedia articles without any morphological preprocessing
* **uni_normal_wikitext** - unigram tokenizer trained directly on raw WikiText dataset without any morphological preprocessing
* **uni_wikipedia_words_1M_min_7_nltk** - unigram tokenizer trained on raw words that appear at least 7 times in 1M Wikipedia articles, without morfessor preprocessing
* **uni_wikipedia_words_1M_min_7_nltk_morphems** - unigram tokenizer trained on morphological segments created by applying morfessor_wikipedia_1M_min_7_nltk model to each word in the 1M min_7 dataset
* **uni_wikipedia_words_1M_min_7_nltk_log** - unigram tokenizer trained on morphological segments created by applying morfessor_wikipedia_1M_min_7_nltk_log model to Wikipedia text

All unigram tokenizers were initialized with:
- Standard normalizers (NFKD, StripAccents, quote normalization, whitespace normalization)
- Metaspace pre-tokenizer
- Vocabulary size of 32,768 (2^15)
- Special tokens ["\<cls>", "\<sep>", "\<unk>", "\<pad>", "\<mask>", "\<s>", "\</s>"]

The training used the UnigramTrainer with:
- shrinking_factor=0.85
- n_sub_iterations=5
- max_piece_length=10

#### Evaluation results:

The tokenizers were evaluated using BLEU scores against a ground truth morphological segmentation dataset with expanded entries from MorphoLex-en. The evaluation computed 1-gram, 2-gram, and 3-gram precision as well as the overall BLEU score.

**BLEU Scores for each tokenizer (Exp 2)**                        

| Tokenizer | BLEU Score | 1-gram | 2-gram | 3-gram |
|-----------|------------|--------|--------|--------|
| bert | 0.2468 | 0.4138 | 0.2299 | 0.1579 |
| modernbert | 0.2587 | 0.4736 | 0.2924 | 0.1250 |
| llama32 | 0.2691 | 0.4842 | 0.3059 | 0.1316 |
| gpt2 | 0.2703 | 0.4873 | 0.3004 | 0.1348 |
| **xlnet** | **0.4076**  | 0.5469 | 0.4257 | 0.3333 |
| albert | 0.3939 | 0.4839 | 0.4094 | 0.3500 |
| morfessor_nltk | 0.5331 | 0.6560 | 0.5514 | 0.4189 |
| morfessor_wiki | 0.3394 | 0.4788 | 0.4933 | 0.5000 |
| morfessor_sent | 0.3343 | 0.4703 | 0.4800 | 0.5000 |
| morfessor_wikipedia_300M | 0.0000 | 0.3542 | 0.4839 | 0.0000 |
| morfessor_wikipedia_10M | 0.0000 | 0.4786 | 0.4110 | 0.0000 |
| morfessor_wikipedia_1M_unique | 0.0000 | 0.3981 | 0.4222 | 0.0000 |
| morfessor_wikipedia_1M_unique_nltk | 0.4497 | 0.5633 | 0.5595 | 0.7500 |
| **morfessor_wikipedia_1M_min_3_nltk** | **0.5985**  | 0.6902 | 0.6176 | 0.6538 |
| **morfessor_wikipedia_1M_min_7_nltk** | **0.6092**  | 0.7217 | 0.6265 | 0.5000 |
| **morfessor_wikipedia_1M_min_7_nltk_log** | **0.6035**  | 0.7138 | 0.6159 | 0.5000 |
| uni_wiki | 0.3164 | 0.5157 | 0.3254 | 0.1887 |
| uni_wikipedia_300M | 0.3350 | 0.5543 | 0.3586 | 0.1892 |
| uni_wikipedia_10M | 0.2506 | 0.5012 | 0.2713 | 0.1157 |
| uni_wikipedia_10M_unique | 0.3007 | 0.5354 | 0.3227 | 0.1573 |
| uni_wikipedia_10M_unique_tok | 0.2728 | 0.4884 | 0.2807 | 0.1481 |
| uni_wikipedia_1M_unique_3M_words | 0.2017 | 0.4737 | 0.2218 | 0.0781 |
| uni_wikipedia_1M_min_7_nltk | 0.2159 | 0.4744 | 0.2528 | 0.0840 |
| uni_normal_wikipedia | 0.3489 | 0.5527 | 0.3789 | 0.2027 |
| uni_normal_wikitext | 0.2800 | 0.4908 | 0.3028 | 0.1477 |
| **uni_wikipedia_words_1M_min_7_nltk** | **0.4570** | 0.6605 | 0.4954 | 0.2917 |
| uni_wikipedia_words_1M_min_7_nltk_morphems | 0.2205 | 0.4604 | 0.2410 | 0.0966 |
| uni_wikipedia_words_1M_min_7_nltk_log | 0.2275 | 0.5012 | 0.2689 | 0.0873 |
| minipile_32k | 0.2926 | 0.5352 | 0.3292 | 0.1475 |
| minipile_64k | 0.3341 | 0.5288 | 0.4375 | 0.2857 |


## Conclusions and Recommendations



### Analysis and Practical Considerations

1. **Morfessor models are good for morphological segmentation**. 
The three best performing models (`morfessor_wikipedia_1M_min_7_nltk`, `morfessor_wikipedia_1M_min_7_nltk_log`, and `morfessor_wikipedia_1M_min_3_nltk`) all significantly outperform standard subword tokenizers on our morphological evaluation dataset.
However, the **Morfessor models are not able to provide the desired vocabulary size**, that is why I have decided to train the custom unigram tokenizer on the morphological segments.
2. **Frequency filtering is crucial for quality**. The best models were trained on moderately frequent words (occurring at least 3-7 times)
3. **Two-stage tokenization degrades morphological accuracy**. The unigram tokenizers trained on Morfessor-segmented text performed worse than direct Morfessor models.


Considering real-world implementation, several factors beyond BLEU scores must be addressed:

1. **Integration with 🤗 Transformers**: While Morfessor excels at morphological segmentation, it doesn't integrate natively with the Hugging Face ecosystem. Models like XLNet (BLEU: 0.4076) offer much better integration while still providing reasonable morphological awareness. 

2. **Handling non-natural, structured and special text**: Our evaluation focused on English words, but real applications must process:
   1. **Programming code**: Special characters, camelCase, snake_case, code blocks
   2. **Structured formats**: LaTeX, Markdown, HTML, XML, JSON
   3. **Numerical data**: Dates, times, measurements, scientific notation.
   4. **Slang and neologisms**: Standard tokenizers trained on recent web data generally handle these better
   5. **Emojis**: Require special handling that modern tokenizers often include
   6. **Non-English text**: Modern multilingual tokenizers support a wide range of scripts and languages   
   Standard tokenizers like BERT, GPT-2 and Llama have been trained on diverse data including code and structured text.

3. **Multilingual support**: Morfessor is primarily designed for single-language morphological analysis.

4. **Vocabulary size management**: Morfessor doesn't naturally limit vocabulary size, while the unigram models maintain a fixed vocabulary size (32K tokens), which is essential for transformer model efficiency.




## Publications: 

* [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/pdf/2004.03720)
* [MorphPiece : A Linguistic Tokenizer for Large Language Model](https://arxiv.org/pdf/2307.07262)
* [Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline](https://aaltodoc.aalto.fi/server/api/core/bitstreams/78f1f8d4-c7a4-49e5-992e-85bd70f06ed4/content)
* [MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies](https://arxiv.org/pdf/2502.00894)
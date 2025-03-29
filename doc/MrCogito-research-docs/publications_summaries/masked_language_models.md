* [XLnet](#xlnet)
* [ModernBert](#modernbert)
* [Albert](#albert)



## XLnet

**Title:** XLNet: Generalized Autoregressive Pretraining for Language Understanding  
**Publish Date:** 2 Jan 2020  
**Authors:** Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le  
**URL:** [https://arxiv.org/pdf/1906.08237](https://arxiv.org/pdf/1906.08237)  
**Extracted tags:** [#autoregressive](app://obsidian.md/index.html#autoregressive) [#pretraining](app://obsidian.md/index.html#pretraining) [#languagemodeling](app://obsidian.md/index.html#languagemodeling) [#bidirectionalcontexts](app://obsidian.md/index.html#bidirectionalcontexts) [#Transformer-XL](app://obsidian.md/index.html#Transformer-XL)

### TL;DR

XLNet is a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.

### The problem that authors want to solve

With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. Faced with the pros and cons of existing language pretraining objectives, in this work, we propose XLNet, a generalized autoregressive method that leverages the best of both AR language modeling and AE while avoiding their limitations.

### The solution, main idea on the intuition level and strong points

The key idea of XLNet is to maximize the expected log likelihood of a sequence w.r.t. all possible permutations of the factorization order. Thanks to the permutation operation, the context for each position can consist of tokens from both left and right. In expectation, each position learns to utilize contextual information from all positions, i.e., capturing bidirectional context. As a generalized AR language model, XLNet does not rely on data corruption, hence, it does not suffer from the pretrain-finetune discrepancy that BERT is subject to. Meanwhile, the autoregressive objective also provides a natural way to use the product rule for factorizing the joint probability of the predicted tokens, eliminating the independence assumption made in BERT. XLNet integrates the segment recurrence mechanism and relative encoding scheme of Transformer-XL into pretraining, which empirically improves the performance especially for tasks involving a longer text sequence.

### The detailed solution, training process, data preparation

The permutation language modeling objective can be expressed as follows:  
.  
To avoid the problem that the representation  does not depend on which position it will predict, the authors propose to re-parameterize the next-token distribution to be target position aware:  
, where  denotes a new type of representations which additionally take the target position  as input. The authors propose to use two sets of hidden representations instead of one: The content representation , which serves a similar role to the standard hidden states in Transformer, and the query representation , which only has access to the contextual information  and the position , but not the content .

For pretraining, the authors use the BooksCorpus and English Wikipedia as part of our pretraining data, which have 13GB plain text combined. In addition, they include Giga5 (16GB text), ClueWeb 2012-B and Common Crawl for pretraining. The largest model XLNet-Large has the same architecture hyperparameters as BERT-Large, which results in a similar model size. During pretraining, the authors always use a full sequence length of 512.

### The evaluation procedure, evaluation datasets and results

The authors use a variety of natural language understanding datasets to evaluate the performance of their method, including GLUE language understanding tasks, reading comprehension tasks like SQuAD and RACE, text classification tasks such as Yelp and IMDB, and the ClueWeb09-B document ranking task. Under comparable experiment setting, XLNet consistently outperforms BERT on a wide spectrum of problems.

### Previous attempts to solve this problem

Among unsupervised pretraining objectives, autoregressive (AR) language modeling and autoencoding (AE) have been the two most successful pretraining objectives. AR language modeling seeks to estimate the probability distribution of a text corpus with an autoregressive model. Since an AR language model is only trained to encode a uni-directional context (either forward or backward), it is not effective at modeling deep bidirectional contexts. In comparison, AE based pretraining does not perform explicit density estimation but instead aims to reconstruct the original data from corrupted input. A notable example is BERT, which has been the state-of-the-art pretraining approach. However, the artificial symbols like [MASK] used by BERT during pretraining are absent from real data at finetuning time, resulting in a pretrain-finetune discrepancy. Moreover, since the predicted tokens are masked in the input, BERT is not able to model the joint probability using the product rule as in AR language modeling.

### Max 5 top most relevant to the problem publication from bibliography

Extracted from bibliography section

1. Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860 , 2019.
2. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 ,2018.
3. Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, and Hugo Larochelle. Neural autoregressive distribution estimation. The Journal of Machine Learning Research , 17(1):7184– 7220, 2016.
4. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems , pages 5998–6008, 2017.
5. Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. arXiv preprint arXiv:1901.11504 , 2019.


### XLnet model architecture

| Hparam                  | Value   |
|-------------------------|---------|
| Number of layers        | 24      |
| Hidden size             | 1024    |
| Number of attention heads | 16      |
| Attention head size     | 64      |
| FFN inner hidden size   | 4096    |
| Hidden Dropout          | 0.1     |
| GeLU Dropout            | 0.0     |
| Attention dropout       | 0.1     |
| Partial prediction K    | 6       |
| Max sequence length     | 512     |
| Batch size              | 8192    |
| Learning rate           | 4e-4    |
| Number of steps         | 500K    |
| Warmup steps            | 40,000  |
| Learning rate decay     | Linear  |
| Adam epsilon            | 1e-6    |
| Weight decay            | 0.01    |


| Hparam              | RACE   | SQuAD  | MNLI   | Yelp-5 |
| ------------------- | ------ | ------ | ------ | ------ |
| Dropout             | 0.1    | 0.1    | 0.1    | 0.1    |
| Attention dropout   | 0.1    | 0.1    | 0.1    | 0.1    |
| Max sequence length | 512    | 512    | 128    | 512    |
| Batch size          | 32     | 48     | 128    | 128    |
| Learning rate       | 2e-5   | 3e-5   | 2e-5   | 1e-5   |
| Number of steps     | 12K    | 8K     | 10K    | 10K    |
| Learning rate decay | Linear | Linear | Linear | Linear |
| Weight decay        | 0.01   | 0.01   | 0.01   | 0.01   |
| Adam epsilon        | 1e-6   | 1e-6   | 1e-6   | 1e-6   |
| Layer-wise lr decay | 1.0    | 0.75   | 1.0    | 1.0    |


## ModernBert


**Title:** Smarter, Better, Faster, Longer : A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference  

**Publish Date:** 19 Dec 2024  
**Authors:** Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, Iacopo Poli  
**URL:** [https://arxiv.org/pdf/2412.13663](https://arxiv.org/pdf/2412.13663)  
Extracted tags (with hash): [#encoder](app://obsidian.md/index.html#encoder) [#transformer](app://obsidian.md/index.html#transformer) [#ModernBERT](app://obsidian.md/index.html#ModernBERT) [#NLP](app://obsidian.md/index.html#NLP) [#retrieval](app://obsidian.md/index.html#retrieval) [#classification](app://obsidian.md/index.html#classification) [#long_context](app://obsidian.md/index.html#long_context) [#finetuning](app://obsidian.md/index.html#finetuning) [#inference](app://obsidian.md/index.html#inference)

### The problem that authors want to solve

The authors aim to address the limitations of existing encoder-only transformer models like BERT, which, despite their widespread use, haven't seen significant improvements in recent years. These limitations include:

- Sequence length limited to 512 tokens.
- Suboptimal model design and vocabulary sizes.
- Inefficient architectures in terms of downstream performance and computational efficiency.
- Limited training data in volume, restricted to narrow domains, lacking code data, or lacking knowledge of recent events.

### The solution, main idea on the intuition level and strong points

The solution is ModernBERT, a modernized encoder-only transformer model. The main idea is to bring modern model optimizations to encoder-only models, resulting in a Pareto improvement over older encoders.

Strong points:

- Trained on 2 trillion tokens with a native 8192 sequence length.
- State-of-the-art results on diverse classification tasks and single/multi-vector retrieval.
- Speed and memory efficient, designed for inference on common GPUs.

### The detailed solution, training process, data preparation

**Architectural Improvements:**

- **Modern Transformer Bias Terms:** Bias terms are disabled in all linear layers except the final decoder linear layer and in all Layer Norms.
- **Positional Embeddings:** Rotary positional embeddings (RoPE) are used instead of absolute positional embeddings.
- **Normalization:** Pre-normalization block with standard layer normalization is used. A LayerNorm is added after the embedding layer, and the first LayerNorm in the first attention layer is removed.
- **Activation:** GeGLU activation function is adopted.
- **Efficiency Improvements:**
    - Alternating Attention: Attention layers alternate between global and local attention.
    - Unpadding: Employs unpadding for both training and inference.
    - Flash Attention: Uses Flash Attention 3 for global attention layers and Flash Attention 2 for local attention layers.
    - torch.compile: PyTorch's built-in compiling is leveraged to improve training efficiency.
- **Model Design:** Models are designed to maximize the utilization of common GPUs, with 22 and 28 layers for the base and large models, respectively.

**Training:**

- **Data Mixture:** Trained on 2 trillion tokens of primarily English data from various sources, including web documents, code, and scientific literature.
- **Tokenizer:** Uses a modified version of the OLMo tokenizer. Vocabulary size is set to 50,368.
- **Sequence Packing:** Sequence packing is adopted with a greedy algorithm to avoid high minibatch-size variance.
- **MLM:** Masked Language Modeling (MLM) setup is used with a masking rate of 30 percent.
- **Optimizer:** StableAdamW optimizer is used.
- **Learning Rate Schedule:** A modified trapezoidal Learning Rate (LR) schedule (Warmup-Stable-Decay) is used.
- **Batch Size Schedule:** Batch size scheduling starts with smaller gradient accumulated batches, increasing over time to the full batch size.
- **Weight Initialization and Tiling:** ModernBERT-base is initialized with random weights following the Megatron initialization. For ModernBERT-large, weights are initialized from ModernBERT-base using center tiling and Gopher layer scaling.
- **Context Length Extension:** The native context length of ModernBERT is extended to 8192 tokens by increasing the global attention layer’s RoPE theta to 160,000 and training for an additional 300 billion tokens.

### Previous attempts to solve this problem

The article mentions several previous efforts to improve encoder-only models:

- **MosaicBERT, CrammingBERT, and AcademicBERT:** Focused on matching BERT performance with better training efficiency.     
                                                                                        

| Task  | Base        |       | Large       |       |
|-------|-------------|-------|-------------|-------|
|       | LR          | WD    | LR          | WD    | Ep  |
| CoLA  | 8e−5        | 1e−6  | 3e−5        | 8e−6  | 5   |
| MNLI  | 5e−5        | 5e−6  | 3e−5        | 1e−5  | 1   |
| MRPC  | 5e−5        | 5e−6  | 8e−5        | 5e−6  | 2   |
| QNLI  | 8e−5        | 5e−6  | 3e−5        | 5e−6  | 2   |
| QQP   | 5e−5        | 5e−6  | 5e−5        | 8e−6  | 2   |
| RTE   | 5e−5        | 1e−5  | 5e−5        | 8e−6  | 3   |
| SST-2 | 8e−5        | 1e−5  | 1e−5        | 1e−6  | 3   |
| STSB  | 8e−5        | 5e−6  | 8e−5        | 1e−5  | 10  |


### Table 4: ModernBERT Model Design

| Parameter | Base | Large |
|:----------|:-----|:------|
| Vocabulary | 50,368 | 50,368 |
| Unused Tokens | 83 | 83 |
| Layers | 22 | 28 |
| Hidden Size | 768 | 1024 |
| Transformer Block | Pre-Norm | Pre-Norm |
| Activation Function | GeLU | GeLU |
| Linear Bias | False | False |
| Attention | Multi-head | Multi-head |
| Attention Heads | 12 | 16 |
| Global Attention | Every three layers | Every three layers |
| Local Attention Window | 128 | 128 |
| Intermediate Size | 1,152 | 2,624 |
| GLU Expansion | 2,304 | 5,248 |
| Normalization | LayerNorm | LayerNorm |
| Norm Epsilon | 1e-5 | 1e-5 |
| Norm Bias | False | False |
| RoPE theta | 160,000 | 160,000 |
| Local Attn RoPE theta | 10,000 | 10,000 |


## Albert

ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
https://arxiv.org/pdf/1909.11942



 | Model        | Parameters | Layers | Hidden | Embedding | Parameter-sharing |
|--------------|------------|--------|--------|-----------|-------------------|
| BERT base    | 108M       | 12     | 768    | 768       | False             |
| BERT large   | 334M       | 24     | 1024   | 1024      | False             |
| ALBERT base  | 12M        | 12     | 768    | 128       | True              |
| ALBERT large | 18M        | 24     | 1024   | 128       | True              |
| ALBERT xlarge| 60M        | 24     | 2048   | 128       | True              |
| ALBERT xxlarge| 235M       | 12     | 4096   | 128       | True              |
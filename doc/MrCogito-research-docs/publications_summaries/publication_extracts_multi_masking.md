



This file is a collection of articles that I find interesting and worth summarising. it contains a summaries by AI, its main purpose is to serve as a reference, guide and ground for further AI generation to combine the ideas, generate new ones, check if my ideas are already known and align.  

[[prompt_for_extraction]]

Table of contents:

* [Masked language modelling](#masked-language-modelling)
   * [Blank language model](#blank-language-model)
   * [Large Language Diffusion Models](#large-language-diffusion-models)
      * [The problem that authors want to solve](#the-problem-that-authors-want-to-solve)
      * [The solution, main idea on the intuition level and strong points](#the-solution-main-idea-on-the-intuition-level-and-strong-points)
      * [The detailed solution, training process, data preparation](#the-detailed-solution-training-process-data-preparation)
      * [Previous attempts to solve this problem](#previous-attempts-to-solve-this-problem)
      * [Max 5 top most relevant to the problem follow-up publications](#max-5-top-most-relevant-to-the-problem-follow-up-publications)
   * [Insertion Transformer: Flexible Sequence Generation via Insertion Operations](#insertion-transformer-flexible-sequence-generation-via-insertion-operations)
   * [Towards More Efficient Insertion Transformer with Fractional Positional Encoding](#towards-more-efficient-insertion-transformer-with-fractional-positional-encoding)
   * [Smarter, Better, Faster, Longer : A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](#smarter-better-faster-longer--a-modern-bidirectional-encoder-for-fast-memory-efficient-and-long-context-finetuning-and-inference)
      * [The problem that authors want to solve](#the-problem-that-authors-want-to-solve-1)
      * [The solution, main idea on the intuition level and strong points](#the-solution-main-idea-on-the-intuition-level-and-strong-points-1)
      * [The detailed solution, training process, data preparation](#the-detailed-solution-training-process-data-preparation-1)
      * [Previous attempts to solve this problem](#previous-attempts-to-solve-this-problem-1)


# Masked language modelling 

Mask filling techniques as learning objectives. I think this is not fully explored topic. 


There are still some unexplored paths and problems,
1.  how to handle the multi-token words, entities and concepts, current classical masking pipeline return only tokens that form a complete word, they are not able to substitute the whole entity or concept by [mask] token.
2.  what part of the input should be masked? Up to this day, the most common approach is to mask 15% of the input tokens. But why? Recent LLaDA paper shows that masking more for diffusion process could lead to superior performance. 
3.  Generation by masking could led to faster, more efficient in terms of computational resources, generation. But how to handle the expansion of the text produced by the model? How to handle the long generation?
4.  Training with masked language modelling could take into account the bidirectional context, which in many real-world use cases is crucial: code infilling, rewriting, 




## Blank language model

Link [Blank language model](https://arxiv.org/abs/2002.03079) (2020)

**Title, Publish Date, and Authors**
   - **Title**: Blank Language Models
   - **Publish Date**: November 17, 2020
   - **Authors**: Tianxiao Shen, Victor Quach, Regina Barzilay, Tommi Jaakkola

**Problem the Authors Want to Solve**
   The authors aim to address the challenge of generating text that involves filling in missing fragments within partially specified text. This includes tasks like text editing, information fusion, and ancient text restoration, where the input may have multiple missing spans of varying lengths.

**Literature Review and Previous Attempts**
   - Existing methods adapt left-to-right language models for text infilling, relying on intricate inference algorithms like dynamic programming or gradient search. However, these approaches face limitations such as high decoding complexity and inability to handle variable infilling lengths.
   - Other models like the Insertion Transformer and Levenshtein Transformer attempt to optimize generation order but are not tailored for text rewriting tasks.
   - Masked Language Models (MLMs) like BERT require predefined insertion lengths and generation orders, which can lead to suboptimal solutions.

**Solution and Main Idea**
   The authors propose the **Blank Language Model (BLM)**, which dynamically creates and fills blanks in text. The model uses a special blank symbol ("_") to control token placement and iteratively determines the generation location and content. BLM supports variable-length infilling and ensures consistency with the surrounding context. It is trained by maximizing a lower bound of the marginal data likelihood and uses simple decoding strategies like greedy decoding or beam search.

**Top 5 Relevant Follow-Up Publications**
   - Zhu et al. (2019): Explored text infilling tasks and provided foundational insights into the challenges of missing text reconstruction.
   - Assael et al. (2019): Focused on ancient text restoration, introducing datasets like PHI-ML for character-level restoration.
   - Stern et al. (2019): Developed the Insertion Transformer, a precursor to dynamic canvas-based text generation.
   - Gu et al. (2019): Investigated adaptive generation orders and insertion mechanisms in sequence models.
   - Shen et al. (2017): Studied style transfer tasks, showcasing the flexibility of models in diverse text generation conditions.



## Large Language Diffusion Models

Autoregressive models (ARMs) are widely regarded as the cornerstone of large language models (LLMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA models distributions through a forward data masking process and a reverse process, parameterized by a vanilla Transformer to predict masked tokens. By optimizing a likelihood bound, it provides a principled generative approach for probabilistic inference. Across extensive benchmarks, LLaDA demonstrates strong scalability, outperforming our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings establish diffusion models as a viable and promising alternative to ARMs, challenging the assumption that key LLM capabilities discussed above are inherently tied to ARMs. Project page and codes: this https URL.


**Title:** Large Language Diffusion Models  
**Publish Date:** February 18, 2025  
**Authors:** Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li  
**URL:** https://arxiv.org/pdf/2502.09992
**Article extracted Tags:** #masked-lm #diffusion-model #gen-ai #transformers  #in-context-learning #instruction-following 

Other resources: 
* [LLaDA: The Diffusion Model That Could Redefine Language Generation | by Maxime Wolf | Data Science Collective | Feb, 2025 | Medium](https://medium.com/data-science-collective/llada-explained-how-diffusion-could-revolutionize-language-models-950bcce4ec09)*


### The problem that authors want to solve

The paper addresses the question of whether the autoregressive paradigm is the *only* viable path to achieving the intelligence exhibited by Large Language Models (LLMs). The authors challenge the notion that autoregressive models (ARMs) are the cornerstone of LLMs, pointing out limitations such as high computational costs due to sequential token generation and ineffectiveness in reversal reasoning tasks.

### The solution, main idea on the intuition level and strong points

The authors introduce LLaDA (Large Language Diffusion with Masking), a diffusion model trained from scratch. Instead of predicting the next token sequentially, LLaDA models distributions through a forward data masking process and a reverse process, parameterized by a Transformer to predict masked tokens.

**Main Idea:** To demonstrate that the generative modeling principles, rather than the autoregressive formulation itself, underpin the essential properties of LLMs. LLaDA leverages a masked diffusion model (MDM) to construct a model distribution with bidirectional dependencies.

**Strong Points:**
*   **Scalability:** LLaDA scales effectively, achieving comparable results to ARM baselines.
*   **In-Context Learning:** LLaDA 8B surpasses LLaMA2 7B on nearly all 15 standard zero/few-shot learning tasks and performs on par with LLaMA3 8B.
*   **Instruction-Following:** LLaDA significantly enhances the ability to follow instructions after SFT.
*   **Reversal Reasoning:** LLaDA effectively breaks the reversal curse, outperforming GPT-4o in a reversal poem completion task.

### The detailed solution, training process, data preparation

**Probabilistic Formulation:** LLaDA defines a model distribution $p_\theta(x_0)$ through a forward process (gradually masking tokens) and a reverse process (iteratively predicting masked tokens). The core is a mask predictor $p_\theta(\cdot | x_t)$ trained using a cross-entropy loss computed only on the masked tokens:

$L(\theta) = -E_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbb{1}[x_{it} = M] \log p_\theta(x_i^0 | x_t) \right]$

**Pre-training:** LLaDA uses a Transformer as the mask predictor, pre-trained on 2.3 trillion tokens. The data is derived from online corpora, encompassing high-quality code, math, and multilingual data. A fixed sequence length of 4096 tokens is used, with a computational cost of 0.13 million H800 GPU hours.

**Supervised Fine-Tuning (SFT):** LLaDA is fine-tuned on 4.5 million pairs of prompts and responses. The implementation involves masking tokens in the response independently and feeding both the prompt and the masked response to the pre-trained mask predictor.

**Inference:** LLaDA samples new text and evaluates the likelihood of candidate text. Given a prompt, the reverse process is discretized to sample from the model distribution, starting from a fully masked response.

### Previous attempts to solve this problem

The paper mentions previous work on diffusion models in the context of LLMs, noting that they have been underexplored despite extensive efforts in visual domains. Previous approaches include:

*   Continuousizing text data and applying diffusion models directly.
*   Modeling continuous parameters of discrete distributions.
*   Replacing continuous diffusion with discrete processes featuring new forward and reverse dynamics.
*   Fine-tuning ARMs in the MDM formulation.

The authors highlight that previous attempts often face challenges in scalability and have not achieved performance comparable to strong LLMs under comprehensive evaluation.

### Max 5 top most relevant to the problem follow-up publications

Based on the "Related Work" section of the provided paper, here are 5 relevant follow-up publications:

1.  **Discrete diffusion language modeling by estimating the ratios of the data distribution (Lou et al., 2023):** This paper explores masked diffusion, showing it can achieve perplexity comparable to or surpassing ARMs at the GPT-2 scale.
2.  **Your absorbing discrete diffusion secretly models the conditional distributions of clean data (Ou et al., 2024):** This work establishes fundamental theoretical results that motivated the model design, training, and inference in the LLaDA paper.
3.  **Scaling up masked diffusion models on text (Nie et al., 2024):** This paper explores how MDM can be leveraged for language tasks such as question answering at the GPT-2 scale.
4.  **Scaling diffusion language models via adaptation from autoregressive models (Gong et al., 2024):** This paper looks at fine-tuning ARMs in the MDM formulation.
5.  **All are worth words: A vit backbone for diffusion models (Bao et al., 2023):** This paper introduces a Vision Transformer (ViT) backbone for diffusion models, which could be relevant for improving the performance of LLaDA.



-----------------------------------------------

## Insertion Transformer: Flexible Sequence Generation via Insertion Operations  

**Title:** Insertion Transformer: Flexible Sequence Generation via Insertion Operations  
**Publish Date:** February 8, 2019  
**Authors:** Mitchell Stern, William Chan, Jamie Kiros, Jakob Uszkoreit  
**URL:** [Link to the article](https://arxiv.org/pdf/1902.03249)

**The problem that authors want to solve**

The authors aim to address the limitations of traditional autoregressive sequence generation models, which rely on fixed left-to-right ordering. These models struggle with parallel token generation and dynamic sequence length prediction, leading to inefficiencies in certain applications like machine translation.

**The solution, main idea on the intuition level and strong points**

The Insertion Transformer introduces a flexible sequence generation framework based on insertion operations. Unlike traditional models, it allows tokens to be inserted at arbitrary positions in the sequence during decoding. This flexibility enables:
- Dynamic growth of the output sequence.
- Accommodation of various generation orderings, such as left-to-right or binary tree traversal.
- Both fully and partially autoregressive generation, enhancing robustness and efficiency.

**The detailed solution, training process, data preparation**

The Insertion Transformer modifies the original Transformer architecture by:
1. Removing the causal self-attention mask in the decoder to allow full context access.
2. Introducing slot representations for insertion locations.
3. Supporting joint content-location distributions for insertion operations.

Training involves optimizing loss functions tailored to specific generation orders, such as left-to-right, balanced binary tree, or uniform order. The model is trained on the WMT 2014 English-German translation dataset using TensorFlow, with experiments conducted on eight P100 GPUs.

**Previous attempts to solve this problem**

Previous non-autoregressive models like the Non-Autoregressive Transformer (NAT) and Iterative Refinement Model allowed parallel generation but faced challenges such as:
- Fixed sequence length prediction.
- Conditional independence assumptions limiting expressive power.
- Separate decoders for initial hypothesis generation and refinement.

**Max 5 top most relevant to the problem follow-up publications**

1. Vaswani et al., 2017 - Original Transformer architecture.
2. Gu et al., 2018 - Non-Autoregressive Transformer (NAT).
3. Lee et al., 2018 - Iterative Refinement Model.
4. Hinton et al., 2015 - Knowledge distillation techniques.
5. Yang et al., 2018 - Mixture-of-softmaxes for language modeling.




## Towards More Efficient Insertion Transformer with Fractional Positional Encoding

**Title:** Towards More Efficient Insertion Transformer with Fractional Positional Encoding  
**Publish Date:** January 31, 2023  
**Authors:** Zhisong Zhang (Carnegie Mellon University), Yizhe Zhang (Apple Inc.), Bill Dolan (Microsoft Research)  
**URL:** [arXiv:2112.06295](https://arxiv.org/pdf/2112.06295)

**The Problem That Authors Want to Solve**

The authors aim to address the computational inefficiency in insertion-based sequence generation models, specifically the Insertion Transformer. The issue arises from the need to re-encode all previously generated tokens at each decoding step due to the incompatibility of absolute positional encoding with insertion-based generation schemes.

**Literature Review and Previous Attempts to Solve This Problem**

- **Auto-regressive Models:** These models generate sequences in a left-to-right manner, which limits parallelization but allows for reusable positional encodings.
- **Insertion-based Models:** Introduced as an alternative to auto-regressive models, they allow flexible generation orders but face computational overhead due to re-encoding requirements.
- **Relative Positional Encoding (REL):** Previously adopted to mitigate re-encoding but involves complex modifications to attention mechanisms.

**The Solution: Main Idea on the Intuition Level**

The authors propose a novel **Fractional Positional Encoding (FPE)** scheme. FPE dynamically calculates positional representations for tokens based on their left and right neighbors during insertion. This approach ensures that positional representations remain unchanged throughout the decoding process, enabling computational reusability similar to auto-regressive models. FPE is lightweight, requiring only a simple linear function for embedding calculations.

**Top 5 Most Relevant to the Problem Follow-Up Publications**

1. Stern et al. (2019) - Introduction of the Insertion Transformer.
2. Shaw et al. (2018) - Relative positional encoding for self-attention mechanisms.
3. Gu et al. (2019a) - Insertion-based decoding with inferred generation order.
4. Lu et al. (2022) - Efficient insertion-based text generation models.









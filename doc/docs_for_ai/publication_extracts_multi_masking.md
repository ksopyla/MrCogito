

This file is a collection of articles that I find interesting and worth summarising. it contains a summaries by AI, its main purpose is to serve as a reference, guide and ground for further AI generation to combine the ideas, generate new ones, check if my ideas are already known and align.  

[prompt_for_extraction.md](./prompt_for_extraction.md)

Table of contents:

* [Masked language modelling](#masked-language-modelling)
   * [Blank language model](#blank-language-model)
   * [Large Language Diffusion Models](#large-language-diffusion-models)
   * [Insertion Transformer: Flexible Sequence Generation via Insertion Operations](#insertion-transformer-flexible-sequence-generation-via-insertion-operations)
   * [Towards More Efficient Insertion Transformer with Fractional Positional Encoding](#towards-more-efficient-insertion-transformer-with-fractional-positional-encoding)
* [Concept models, combining tokens into concepts](#concept-models-combining-tokens-into-concepts)
   * ["Memory Transformer"](#memory-transformer)
   * ["ConceptBERT: A Concept-based Framework for Pre-training Language Models"](#conceptbert-a-concept-based-framework-for-pre-training-language-models)
   * ["Large Concept Models" by Meta](#large-concept-models-by-meta)


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
**URL:** [Link to the article](https://arxiv.org/pdf/2502.09992)

**The Problem That Authors Want to Solve**

The authors aim to challenge the dominance of autoregressive models (ARMs) in large language models (LLMs) by investigating whether diffusion models can act as viable alternatives. They focus on scalability, instruction-following, and in-context learning.

**Literature Review and Previous Attempts to Solve This Problem**

- **ARMs:** Widely adopted but computationally expensive and limited in certain tasks like reversal reasoning.  
- **Diffusion Models:** Mainly explored in visual tasks but known for computational challenges in language tasks.  
- **Masked Diffusion Models (MDMs):** Showed early promise (e.g., Nie et al., 2024) but lacked rigorous evaluation against strong LLM baselines.


**The Solution, Main Idea on the Intuition Level**

The authors propose **LLaDA (Large Language Diffusion with mAsking)**, a diffusion-based language model.  
Key ideas:
- Utilizes a forward data masking and reverse prediction process.  
- Incorporates bidirectional dependencies for greater flexibility.  
- Achieves strong scalability and benchmarks performance comparable to leading LLMs (e.g., LLaMA).  
- Demonstrates enhanced instruction-following, multi-turn dialogue ability, and reversal reasoning, surpassing GPT-4o in reversal poetry tasks.

**Top 5 Most Relevant to the Problem Follow-Up Publications**
1. **Nie et al., 2024:** Initial experiments with MDMs for language tasks at GPT-2 scale.  
2. **Lou et al., 2023:** Foundational work showing MDMs achieving ARM-level perplexity.  
3. **Austin et al., 2021a:** Introduced discrete masking processes used in LLaDA.  
4. **Berglund et al., 2023:** Studied reversal reasoning, which LLaDA addresses effectively.  
5. **Ou et al., 2024:** Provided theoretical underpinnings for MDM-based approaches.


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






# Concept models, combining tokens into concepts



## "Memory Transformer"


## "ConceptBERT: A Concept-based Framework for Pre-training Language Models"


## "Large Concept Models" by Meta


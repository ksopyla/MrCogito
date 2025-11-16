


#transformers #gen-ai #masked-lm 

This is an AI extracted summary for an article

# Insertion Transformer Flexible Sequence Generation via Insertion Operations 


**Publish Date:** 8 Feb 2019
**Authors:** Mitchell Stern, William Chan, Jamie Kiros, Jakob Uszkoreit
**URL:** https://arxiv.org/pdf/1902.03249

### The problem that authors want to solve

The authors address the limitations of autoregressive sequence models, which generate tokens in a fixed, often left-to-right order, making it difficult to accommodate parallel token generation or more elaborate generation orderings. Non-autoregressive models allow for parallel generation but have drawbacks such as needing to predefine the target sequence length and strong conditional independence assumptions between output tokens.

### The solution, main idea on the intuition level and strong points

The authors propose the Insertion Transformer, an iterative, partially autoregressive model that uses insertion operations to generate sequences. Instead of generating tokens in a fixed order, the Insertion Transformer allows tokens to be inserted anywhere in the sequence during decoding. This flexibility enables the model to be trained to follow specific orderings or to maximize entropy over all valid insertions for robustness. It also seamlessly accommodates both fully autoregressive (one insertion at a time) and partially autoregressive (simultaneous insertions at multiple locations) generation.

**Strong points:**

*   Bypasses the problem of needing to predict the target sequence length ahead of time.
*   Permits deviation from classic left-to-right generation, allowing for more exotic orderings.
*   Can be used in an autoregressive manner for serial decoding or in a partially autoregressive manner for parallel decoding.

### The detailed solution, training process, data preparation

**Detailed Solution:**

The Insertion Transformer is based on a modified version of the original Transformer architecture. Key changes include:

*   **Full Decoder Self-Attention:** The causal self-attention mask is removed from the decoder, allowing all positions to attend to all other positions.
*   **Slot Representations via Concatenated Outputs:** The model produces n+1 vectors for a sequence of length n, one for each of the n-1 slots between words plus 2 for the beginning and end slots. This is achieved by adding special marker tokens at the beginning and end of the decoder input and then concatenating each adjacent pair of vectors to obtain slot representations.

**Training Process:**

The Insertion Transformer can be trained with different loss functions to accommodate arbitrary generation orders. The paper discusses three order loss functions:

*   **Left-to-Right:** Imitates the conventional left-to-right ordering.
*   **Balanced Binary Tree:** Encourages maximal parallelism by producing the centermost token first, then the center tokens of the spans on either side, and so on.
*   **Uniform:** Assigns equal probability mass to each correct action with no special preference.

The authors also experiment with two termination conditions: slot finalization and sequence finalization.

**Data Preparation:**

The models are trained on the WMT 2014 English-German translation dataset, using newstest2013 for development and newstest2014 for testing.

### Previous attempts to solve this problem

The paper mentions Non-Autoregressive Transformer (NAT) and Iterative Refinement model as previous attempts to solve the problem of parallel sequence generation.

### Max 5 top most relevant to the problem follow-up publications

The article mentions the following relevant publications:

1.  Gu et al. (2018): Non-Autoregressive Neural Machine Translation.
2.  Lee et al. (2018): Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement.
3.  Stern et al. (2018): Blockwise Parallel Decoding for Deep Autoregressive Models.
4.  Wang et al. (2018): Semi-Autoregressive Neural Machine Translation.
5.  Gu et al. (2019): Insertion-based Decoding with Automatically Inferred Generation Order.
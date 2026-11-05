# Concept models, combining tokens into concepts

Concept learning and encoding, list of extracts from articles. How to model concept learning, the learning objective, training protocol, etc

* [Concept models, combining tokens into concepts](#concept-models-combining-tokens-into-concepts)
    * [Memory Transformer](#memory-transformer)
        * [TL;DR](#tldr)
        * [The problem that authors want to solve](#the-problem-that-authors-want-to-solve)
        * [The solution, main idea on the intuition level and strong points](#the-solution-main-idea-on-the-intuition-level-and-strong-points)
        * [The detailed solution, training process, data preparation](#the-detailed-solution-training-process-data-preparation)
        * [The evaluation procedure, evaluation datasets and results](#the-evaluation-procedure-evaluation-datasets-and-results)
        * [Previous attempts to solve this problem](#previous-attempts-to-solve-this-problem)
    * [ConcEPT: Concept-Enhanced Pre-Training for Language Models](#concept-concept-enhanced-pre-training-for-language-models)
        * [TL;DR](#tldr-1)
        * [The problem that authors want to solve](#the-problem-that-authors-want-to-solve-1)
        * [The solution, main idea on the intuition level and strong points](#the-solution-main-idea-on-the-intuition-level-and-strong-points-1)
        * [The detailed solution, training process, data preparation](#the-detailed-solution-training-process-data-preparation-1)
        * [The evaluation procedure, evaluation datasets and results](#the-evaluation-procedure-evaluation-datasets-and-results-1)
        * [Previous attempts to solve this problem](#previous-attempts-to-solve-this-problem-1)
        * [Max 5 top most relevant to the problem publication from bibliography](#max-5-top-most-relevant-to-the-problem-publication-from-bibliography)
    * [Large Concept Models: Language Modeling in a Sentence Representation Space](#large-concept-models-language-modeling-in-a-sentence-representation-space)
        * [TL;DR](#tldr-2)
        * [The problem that authors want to solve](#the-problem-that-authors-want-to-solve-2)
        * [The solution, main idea on the intuition level and strong points](#the-solution-main-idea-on-the-intuition-level-and-strong-points-2)
        * [The detailed solution, training process, data preparation](#the-detailed-solution-training-process-data-preparation-2)
        * [The evaluation procedure, evaluation datasets and results](#the-evaluation-procedure-evaluation-datasets-and-results-2)
        * [Previous attempts to solve this problem](#previous-attempts-to-solve-this-problem-2)
        * [Max 5 top most relevant to the problem follow-up publications](#max-5-top-most-relevant-to-the-problem-follow-up-publications)
    * [Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation](#next-concept-modeling-fast-parallel-coarse-to-fine-text-generation)
    * [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net)](#dynamic-chunking-for-end-to-end-hierarchical-sequence-modeling-h-net)



## Memory Transformer 2020

https://arxiv.org/abs/2006.11527

**Title:** Memory Transformer  
**Publish Date:** 20 June 2020 (v1), 16 February 2021 (v2)  
**Authors:** Mikhail S. Burtsev, Yuri Kuratov, Anton Peganov, Grigory V. Sapunov  
**URL:** [https://arxiv.org/abs/2006.11527](https://arxiv.org/abs/2006.11527)  
**Extracted tags (with hash):** [#ComputationAndLanguage](app://obsidian.md/index.html#ComputationAndLanguage) [#MachineLearning](app://obsidian.md/index.html#MachineLearning) [#NeuralAndEvolutionaryComputing](app://obsidian.md/index.html#NeuralAndEvolutionaryComputing)

### TL;DR

Transformer-based models have achieved state-of-the-art results in many natural language processing tasks. However, the self-attention architecture's context storage may limit the processing of sequence-wide properties. This work proposes adding trainable memory to store non-local representations, enhancing the Transformer model's ability to process global context.

### The problem that authors want to solve

The self-attention architecture of Transformers stores information about context mostly within element-wise representations, which can limit the model's ability to process properties related to the sequence as a whole.

### The solution, main idea on the intuition level and strong points

The authors propose extending the Transformer by:

1. Adding memory tokens to store non-local representations.
2. Creating a memory bottleneck for global information.
3. Controlling memory updates with a dedicated layer.

These memory-augmented Transformers demonstrate improved performance on machine translation and language modeling tasks by better handling global context.

### The detailed solution, training process, data preparation

The paper introduces three extensions to the Transformer baseline:

1. **Memory Tokens:** Incorporating additional tokens specifically designed to store non-local representations of the input sequence.
2. **Memory Bottleneck:** Designing a bottleneck mechanism to manage global information efficiently.
3. **Dedicated Memory Update Layer:** Implementing a specialized layer to control how and when the memory is updated during training.

The models were trained using standard backpropagation techniques on datasets relevant to machine translation and language modeling.

### The evaluation procedure, evaluation datasets and results

The memory-augmented Transformers were evaluated on:

- **Machine Translation:** Demonstrated a positive correlation between the presence of memory and improved translation quality.
- **Language Modeling:** Showed enhanced performance in predicting subsequent words in a sequence.
- **GLUE Benchmark:** When augmenting pre-trained masked language models with memory tokens, the results were mixed, indicating improvements in some tasks but not uniformly across all.

Additionally, visualization of attention patterns over the memory revealed that the models were better at processing global context information.

### Previous attempts to solve this problem

Memory-augmented neural networks (MANNs) extend traditional neural architectures with general-purpose memory, enabling models to learn simple algorithms like Copy or Reverse. MANNs have been successfully trained via backpropagation on various tasks, including question answering and language modeling, often outperforming RNNs and LSTMs of comparable complexity.





## ConcEPT: Concept-Enhanced Pre-Training for Language Models  

**Publish Date:** January 11, 2024  
**Authors:** Xintao Wang, Zhouhong Gu, Jiaqing Liang, Dakuan Lu, Yanghua Xiao, Wei Wang  
**URL:** [https://arxiv.org/pdf/2401.05669](https://arxiv.org/pdf/2401.05669)  
**Extracted tags (with hash):** [#NLP](app://obsidian.md/index.html#NLP) [#LanguageModels](app://obsidian.md/index.html#LanguageModels) [#PLMs](app://obsidian.md/index.html#PLMs) [#ConceptualKnowledge](app://obsidian.md/index.html#ConceptualKnowledge) [#PreTraining](app://obsidian.md/index.html#PreTraining) [#EntityTyping](app://obsidian.md/index.html#EntityTyping) [#KEPLMs](app://obsidian.md/index.html#KEPLMs)

### TL;DR

_Pre-trained language models (PLMs) have been prevailing in state-of-the-art methods for natural language processing, and knowledge-enhanced PLMs are further proposed to promote model performance in knowledge-intensive tasks. However, conceptual knowledge, one essential kind of knowledge for human cognition, still remains understudied in this line of research. This limits PLMs’ performance in scenarios requiring human-like cognition, such as understanding long-tail entities with concepts. In this paper, we propose ConcEPT, which stands for Concept-Enhanced Pre-Training for language models, to infuse conceptual knowledge into PLMs._

### The problem that authors want to solve

_Conceptual knowledge, an essential type of knowledge for human cognition, remains understudied in knowledge-enhanced pre-trained language models (KEPLMs). This gap limits PLMs’ performance in scenarios requiring human-like cognition, such as understanding long-tail entities with concepts._

### The solution, main idea on the intuition level and strong points

_The authors propose ConcEPT (Concept-Enhanced Pre-Training for language models), which infuses conceptual knowledge into PLMs by exploiting external taxonomies with a novel pre-training objective called entity concept prediction (ECP). Unlike previous concept-enhanced methods, ConcEPT can be readily adapted to various downstream applications without requiring entity linking or concept mapping._

### The detailed solution, training process, data preparation

_ConcEPT leverages external taxonomies, specifically a Wikidata-based taxonomy named WikiTaxo, to enhance PLMs with conceptual knowledge. The novel pre-training objective, entity concept prediction (ECP), involves predicting the concepts of entity mentions based on their contexts. Entities in the input documents are annotated via entity linking, their concepts are retrieved from the taxonomy, and the model is trained to predict these concepts using a binary classification approach. The model is implemented based on the BERT-base architecture, with an additional ECP head. Pre-training is conducted on the Wikipedia corpus linked with Wikidata entities, using a combination of ECP and masked language modeling (MLM) objectives._

### The evaluation procedure, evaluation datasets and results

_ConcEPT is evaluated on four knowledge-intensive tasks: entity typing, conceptual knowledge probing, relation classification, and knowledge graph completion. The datasets used include Open Entity, FIGER, COPEN, TACRED, FB15k-237, and Wiki-CKT. Experimental results demonstrate that ConcEPT outperforms vanilla BERT and existing KEPLMs across these tasks, validating the effectiveness of concept-enhanced pre-training. For instance, in entity typing, ConcEPT achieves significant improvements in micro F1 scores on Open Entity and FIGER compared to BERT._

### Previous attempts to solve this problem

_Previous efforts in enhancing PLMs with knowledge have primarily focused on integrating various types of knowledge such as entities, facts from knowledge graphs, syntax, retrieved texts, and logical rules. Examples include ERNIE, KEPLER, and KnowBERT. However, these approaches have largely overlooked the infusion of conceptual knowledge, which is crucial for human-like cognition and understanding of long-tail entities._

### Max 5 top most relevant to the problem publication from bibliography

1. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).** _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding._  
    [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
    
2. **Zhang, Y., et al. (2019).** _ERNIE: Enhanced Representation through Knowledge Integration._  
    [arXiv:1905.07129](https://arxiv.org/abs/1905.07129)
    
3. **Wang, X., et al. (2021).** _KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation._  
    Transactions of the Association for Computational Linguistics, 9:176–194.
    
4. **Speer, R., Chin, J., & Havasi, C. (2017).** _ConceptNet 5.5: An Open Multilingual Graph of General Knowledge._  
    [AAAI-17](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14617)
    
5. **Peng, H., Gao, T., Han, X., Lin, Y., Li, P., Liu, Z., Sun, M., & Zhou, J. (2020a).** _Learning from Context or Names? An Empirical Study on Neural Relation Extraction._  
    Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).


##  Large Concept Models: Language Modeling in a Sentence Representation Space
**Publish Date:** December 12, 2024
**Authors:** LCM team, Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R. Costa-jussà, David Dale, Hady Elsahar, Kevin Heffernan, João Maria Janeiro, Tuan Tran, Christophe Ropers, Eduardo Sánchez, Robin San Roman, Alexandre Mourachko, Safiyyah Saleem, Holger Schwenk
**URL:** https://arxiv.org/pdf/2412.08821
**Extracted tags (with hash):** #LargeConceptModel #LanguageModeling #SentenceRepresentation #SemanticRepresentation #SONAR #DiffusionModels #Summarization #SummaryExpansion #ZeroShotGeneralization

### TL;DR
The paper introduces Large Concept Models (LCMs), an architecture that operates on higher-level semantic representations ("concepts") instead of tokens. LCMs are trained to perform autoregressive sentence prediction in an embedding space (SONAR), achieving impressive zero-shot generalization performance across many languages and modalities.

### The problem that authors want to solve
Current Large Language Models (LLMs) process input and generate output at the token level, missing a crucial characteristic of human intelligence: explicit reasoning and planning at multiple levels of abstraction. The authors aim to move away from processing at the token level and closer to (hierarchical) reasoning in an abstract embedding space.

### The solution, main idea on the intuition level and strong points
The solution is a Large Concept Model (LCM) that operates on an explicit higher-level semantic representation called a "concept." The main idea is to model the underlying reasoning process at a purely semantic level, independent of any instantiation in a specific language or modality.

Strong points:
*   Reasoning at an abstract language- and modality-agnostic level beyond tokens.
*   Explicit hierarchical structure for better readability and interactive edits.
*   Handling of long context and long-form output efficiently.
*   Unparalleled zero-shot generalization to new languages and modalities.
*   Modularity and extensibility, allowing independent development of concept encoders and decoders.

### The detailed solution, training process, data preparation

*   **SONAR Embedding Space:** The LCM uses the SONAR sentence embedding space, which supports text input and output in 200 languages, speech input in 76 languages, and speech output in English.
*   **Data Preparation:** Raw text datasets are converted into sequences of SONAR embeddings. Sentence segmentation is performed using SaT Capped.
*   **LCM Variants:**
    *   **Base-LCM:** A standard decoder-only Transformer trained to minimize the MSE loss between predicted and ground truth sentence embeddings.
    *   **Diffusion-based LCMs (One-Tower and Two-Tower):** Models that use a diffusion process to generate sentence embeddings. They involve a forward noising process and a reverse denoising process.
    *   **Quantized LCM:** Models that quantize the SONAR space and then model these discrete units.
*   **Training Process:** Models are trained on the task of next concept prediction. Different noise schedules (Cosine, Quadratic, Sigmoid) and loss weighting strategies are explored for diffusion-based LCMs.

### The evaluation procedure, evaluation datasets and results

*   **Pre-training Evaluation:** Models are evaluated on the quality of next sentence prediction using metrics like L2 distance, round-trip L2 distance, contrastive accuracy, paraphrasing, and mutual information. Datasets used include ROC-stories, C4, Wikipedia-en, and Gutenberg.
*   **Instruction-tuning Evaluation:** Models are instruction-tuned on the stories subset of Cosmopedia and evaluated on a held-out subset of Cosmopedia. Metrics include ROUGE-L and coherence.
*   **Zero-shot Generalization:** Models are evaluated on the XLSum corpus, a multilingual abstractive news summarization benchmark covering 45 languages.
*   **Results:** Diffusion-based LCMs generally outperform other variants. The LCM exhibits strong zero-shot generalization performance to languages it has never seen, outperforming Llama-3.1-8B-IT on English and on average over foreign languages officially supported by the LLM.

### Previous attempts to solve this problem
*   **Sentence embeddings for language modeling:** Ippolito et al. (2020) proposed a sentence-level language model operating by choosing the next sentence from a finite set of candidates. The INSET architecture (Huang et al., 2020) solves the sentence infilling task. Marfurt and Henderson (2021) and Cornille et al. (2024) used predicted next sentence embeddings in a fully generative setting. An et al. (2024) proposed the SentenceVAE architecture.
*   **Language modeling with diffusion:** The PLANNER architecture (Zhang et al., 2023) consists of a variational autoencoder for paragraphs and a diffusion model. Lovelace et al. (2024) augmented a decoder-only language model with an encoded semantic proposal. A TEncDM model (Shabalin et al., 2024) performs diffusion in the space of contextual token embeddings.

### Max 5 top most relevant to the problem follow-up publications
Due to the limitations of my current capabilities, I am unable to provide a list of the top 5 most relevant follow-up publications.




## Long-Context Language Modeling with Parallel Context Encoding

Howard Yen  Tianyu Gao  Danqi Chen
Princeton Language and Intelligence (PLI), Princeton University {hyen,tianyug,danqic}@cs.princeton.edu
11 Jun 2024

paper: https://arxiv.org/html/2402.16617v2

code: https://github.com/princeton-nlp/CEPE


### TL;DR

Abstract
Extending large language models (LLMs) to process longer inputs is crucial for a wide range of applications. However, the substantial computational cost of transformers and limited generalization of positional encoding restrict the size of their context window. We introduce Context Expansion with Parallel Encoding (CEPE [Uncaptioned image]), a framework that can be applied to any existing decoder-only LLMs to extend their context window. CEPE employs a small encoder to process long inputs chunk by chunk, enabling the frozen decoder to utilize additional contexts via cross-attention. CEPE is efficient, generalizable, and versatile: trained with 8K-token documents, it extends the context window of LLAMA-2 to 128K tokens, offering 10× the throughput with only 1/6 of the memory. CEPE yields strong performance on language modeling and in-context learning. CEPE also excels in retrieval-augmented applications, while existing long-context models degenerate with retrieved contexts. We further introduce a CEPE variant that can extend the context window of instruction-tuned models using only unlabeled data, and showcase its effectiveness on LLaMA-2-Chat, leading to a strong instruction-following model that can leverage very long contexts on downstream tasks.1

32k / exclusive-channel reading (PPL-last-256 is a weak gate; compare Activation Beacon):
[`learned_kv_context_compression.md`](learned_kv_context_compression.md).



## Can Memory-Augmented Language Models Generalize on Reasoning-in-a-Haystack Tasks?

https://arxiv.org/pdf/2503.07903

Abstract
Large language models often expose their brittleness in reasoning tasks, especially while executing long chains of reasoning over context. We propose MemReasoner, a new and simple memory-augmented LLM architecture, in which 
the memory learns the relative order of facts in context, and enables hopping over them, while the
decoder selectively attends to the memory. MemReasoner is trained end-to-end, with optional supporting fact supervision of varying degrees. We train MemReasoner, along with existing memoryaugmented transformer models and a state-space
model, on two distinct synthetic multi-hop reasoning tasks. Experiments performed under a variety of challenging scenarios, including the presence of long distractor text or target answer changes in test set, show strong generalization of MemReasoner on both single- and two-hop tasks. This generalization of MemReasoner is achieved using
none-to-weak supporting fact supervision (using none and 1% of supporting facts for one- and twohop tasks, respectively). In contrast, baseline models overall struggle to generalize and benefit far less from using full supporting fact supervision. The results highlight the importance of explicit memory mechanisms, combined with additional weak supervision, for improving large language model’s context processing ability toward reasoning tasks



## Perceiver: General Perception with Iterative Attention
Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira

https://arxiv.org/pdf/2103.03206


Biological systems perceive the world by simultaneously processing high-dimensional inputs from modalities as diverse as vision, audition, touch, proprioception, etc. The perception models used in deep learning on the other hand are designed for individual modalities, often relying on domain-specific assumptions such as the local grid structures exploited by virtually all existing vision models. These priors introduce helpful inductive biases, but also lock models to individual modalities. In this paper we introduce the Perceiver - a model that builds upon Transformers and hence makes few architectural assumptions about the relationship between its inputs, but that also scales to hundreds of thousands of inputs, like ConvNets. The model leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck, allowing it to scale to handle very large inputs. We show that this architecture is competitive with or outperforms strong, specialized models on classification tasks across various modalities: images, point clouds, audio, video, and video+audio. The Perceiver obtains performance comparable to ResNet-50 and ViT on ImageNet without 2D convolutions by directly attending to 50,000 pixels. It is also competitive in all modalities in AudioSet.



## DeepCrossAttention: Supercharging Transformer Residual Connections

Date: 2025 Feb 10
Link: https://arxiv.org/pdf/2502.06785
Authors: Mike Heddes, Adel Javanmard, Kyriakos Axiotis, Gang Fu, MohammadHossein Bateni, Vahab Mirrokni

Transformer networks have achieved remarkable success across diverse domains, leveraging a variety of architectural innovations, including residual connections. However, traditional residual connections, which simply sum the outputs of previous layers, can dilute crucial information. This work introduces DeepCrossAttention (DCA), an approach that enhances residual learning in transformers. DCA employs learnable, input-dependent weights to dynamically combine layer outputs, enabling the model to selectively focus on the most relevant information in any of the previous layers. Furthermore, DCA incorporates depth-wise cross-attention, allowing for richer interactions between layers at different depths. Our language modeling experiments show that DCA achieves improved perplexity for a given training time. Moreover, DCA obtains the same model quality up to 3x faster while adding a negligible number of parameters. Theoretical analysis confirms that DCA provides an improved trade-off between accuracy and model size when the ratio of collective layer ranks to the ambient dimension falls below a critical threshold.


## Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation

**Title:** Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation  
**Publish Date:** 9 October 2026  
**Authors:** Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré (Stanford / Together AI)  
**URL:** [https://arxiv.org/abs/2610.14162](https://arxiv.org/abs/2610.14162) · code [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT; evaluation-only, no training loop; read at `c702742`) · weights [hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints) · data [hazyresearch/ncm-datasets](https://huggingface.co/datasets/hazyresearch/ncm-datasets)  
**Project fit:** [hierarchy_lessons_hnet_ncm_mai_20261030.md](../4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md) (Adapt: ordered / residual notes; Sudoku-style exam)

### TL;DR

Text is first encoded by a frozen **multiscale residual VQ-VAE** into K levels of discrete codes, short and
coarse first, full length last; a transformer then generates **one level per step, all codes of a level in
parallel**, conditioned on the cumulative reconstruction of the earlier levels. With K ≪ L steps and short early
levels, ~100M models match or beat diffusion language models at 10–31× less wall-clock and ~4–5× fewer training
tokens, and a 25M model solves 86 % of hard Sudoku boards where autoregressive models solve 1 %. "Concept" here
means *what the coarse levels carry*. Read from the code and appendices: the hierarchy is **not** a compression
of the text overall (TinyStories: 641 codes of 12 bits for 128 tokens, ~4.6× the bits of the BPE ids); only the
first levels compress (levels 0–2: 25 codes, ~5 tokens per code), the "down" step is average pooling over fixed
equal regions, and the coarse levels keep a story's **theme but not its names or objects** (App. H).

### The problem that authors want to solve

Diffusion language models promise parallel generation but (i) lose quality quickly with fewer denoising steps,
(ii) process the full sequence at every step with no KV cache, and (iii) need many passes over the data (DUO and
MDLM see > 500B tokens of OpenWebText). Distillation to few steps costs extra training and diversity.

### The solution, main idea on the intuition level and strong points

Borrow next-scale prediction from images (VAR) and residual quantization from audio codecs: describe a sequence
at increasing resolution, each level coding only what the coarser ones missed, and generate resolution by
resolution. Coarse levels fix the "idea" (the paper's translation trajectory shows the gist first, words later);
fine levels fill in and correct. Early steps are cheap because coarse levels are short.

### The detailed solution, training process, data preparation

- **Tokenizer.** `z = E(x) ∈ R^{L×d}`. Level k: `u_k = Down_k(z^{(k-1)}) ∈ R^{ℓ_k×d}` (residual downsampled to
  `ℓ_k` vectors, `ℓ_1 < … < ℓ_K = L`), nearest entry of codebook `V_k`, a reconstruction operator `R_k` maps the
  chosen vectors back to length L: `ẑ^{(k)} = ẑ^{(k-1)} + R_k(e_k(r_k))`, `z^{(k)} = z − ẑ^{(k)}`. Decoder maps
  `ẑ^{(K)}` to token distributions.
- **Nested dropout.** Each training step samples a cutoff k and computes reconstruction CE plus VQ commitment and
  codebook MSE at `ẑ^{(k)}`, so every prefix of levels is a usable reconstruction (earlier levels cannot defer
  information to later ones). Straight-through estimator through quantization.
- **Generator.** Tokenizer frozen. Input to level k = `ẑ^{(k-1)}` resized to `ℓ_k` positions and projected; all
  levels trained in one forward pass with block-diagonal attention; loss `Σ_k w_k Σ_i −log p(r_k^i | z̃^{(k-1)})`
  with `w_k ∝ ℓ_k^{-α}` (α = 0 equal per code, α = 1 equal per level). **Conditioning corruption:** a level-
  dependent fraction of conditioning codes is replaced by random codes of the same codebook (labels stay true), for
  robustness to the generator's own errors. Codes within a level are sampled independently given the earlier
  levels.
- **Conditional generation.** Prefix-LM mask: prompt tokens (embedded with the tokenizer's table) attend to each
  other; each target block attends to the prompt and itself.
- **Scale.** Generators ~100M (88M XSum, 120M WMT, 25M Sudoku); one tokenizer per dataset; scaling run 97M / 317M /
  725M on LM1B with a shared tokenizer.
- **As implemented** (`models/quant.py`, `config/eval/*.yaml`):
  - Levels: TinyStories 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128 codes (4,096 per level); OpenWebText
    1, 4, 9, 16, … (squares); Sudoku 1 → 89 codes with 16 codes per level (4 bits).
  - Down = `F.interpolate(mode="area")`, up = linear, plus a VAR-style residual conv; only level 0 has learned
    conv down/up-samplers.
  - Codebooks: moving-average updates (decay 0.95–0.99) and dead-code restarts; residuals on the stop-gradient
    encoder output. Code comment: straight-through "at level 0 only; at higher levels it diverges within ~100
    steps".
  - Nested dropout (`fine_drop`): probability 0.5 per batch, cutoff level drawn uniformly, minimum 1 kept.
  - Generator: block-diagonal attention (best quality, 1.4× faster than block-causal); codebook-tied heads; level 0
    from a unigram (unconditional) or predicted from the prompt (Sudoku, greedy decoding).
  - Code corruption: mean rate ~0.5, linearly decreasing over levels; moderate rates "greatly improve" quality,
    excessive ones hurt.
  - Level-loss weighting barely matters (Gen. PPL 18.9–19.4 for α ≤ 1.5).
  - More levels help, with diminishing returns (4 → 32).

### The evaluation procedure, evaluation datasets and results

- Unconditional (TinyStories, LM1B, OpenWebText): generative perplexity under GPT-2-large, reported with mean
  unigram entropy (diversity). Matches or beats the strongest diffusion LMs with ~4–11× fewer function
  evaluations; LM1B 21× faster than DUO at better perplexity; OWT 10× faster, competitive.
- Conditional: XSum ROUGE-1/2/L 35.0 / 12.8 / 28.0 at 8 steps vs AURORA-LM 34.7 / 12.2 / 26.6 at 128 (31×
  faster); with MBR over candidates 37.3 / 14.0 / 29.6. WMT14 De–En 22.6 BLEU at 11 steps (AR 25.2 at 64; E2D2
  24.8 at 64).
- Reasoning: Sudoku (40 / 35 / 30 clues) 99.5 / 96.5 / 86.1 % solved at 12 steps vs AR 14.6 / 5.1 / 1.0 %, MDLM 92.0
  / 77.1 / 30.2 %, FMLM+ 97.7 / 92.8 / 81.3 % at 16 steps.
- Efficiency: LM1B tokenizer + generator see 13.4B token positions vs 66B for the baselines (4.9×). Tokenizer exact
  reconstruction near 100 % (100 % on Sudoku), codebook use 62–90 %.
- Probes (App. H): fixing the first 3 of 12 levels and resampling the rest keeps the story's broad theme while names,
  objects and wording change; mixing two stories' coarse codes blends their plots; swapping two level-1 codes
  changes characters and events.
- Authors' limitations (App. I): fixed depth and level lengths (an input-dependent schedule would adapt compute),
  separately trained tokenizer (end-to-end could help), ≤ 725M, artifacts at coarse-level transitions.

### Previous attempts to solve this problem

Few-step diffusion by distillation (SDTT, DUO distillation) and flow maps (FMLM); block-autoregressive diffusion
(KV cache, but blocks are sequential); latent diffusion over continuous text representations. Hierarchical /
concept-level LMs: MEGABYTE (byte patches), H-Net (learned chunks), Large Concept Models (sentence embeddings),
ConceptLM, Dynamic Large Concept Models (variable-length concepts), HDLM (hierarchy of vocabularies).

### Max 5 top most relevant to the problem publications from bibliography

- VAR, next-scale prediction for images: [arXiv 2404.02905](https://arxiv.org/abs/2404.02905)
- RQ-VAE, residual quantization: [arXiv 2203.01941](https://arxiv.org/abs/2203.01941)
- H-Net, dynamic chunking: [arXiv 2507.07955](https://arxiv.org/abs/2507.07955)
- Dynamic Large Concept Models: [arXiv 2512.24617](https://arxiv.org/abs/2512.24617)
- MDLM, masked diffusion LMs: [arXiv 2406.07524](https://arxiv.org/abs/2406.07524)



## Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net)

**Title:** Dynamic Chunking for End-to-End Hierarchical Sequence Modeling  
**Publish Date:** July 2025 (v2 used here); scaled follow-up in Cartesia's blog *From Tokens to Bytes* I–II (28 Oct 2026, not read at source)  
**Authors:** Sukjun Hwang, Brandon Wang, Albert Gu (CMU, Cartesia AI)  
**URL:** [https://arxiv.org/abs/2507.07955](https://arxiv.org/abs/2507.07955) · code [goombalab/hnet](https://github.com/goombalab/hnet) (read at `3673fe1`, Sep 2025; the improved Oct 2026 architecture is not released) · weights [cartesia-ai](https://huggingface.co/cartesia-ai)  
**Project fit:** [hierarchy_lessons_hnet_ncm_mai_20261030.md](../4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md) (Adapt: content-placed notes, zero-start raw shortcut, recurrent writer)

### TL;DR

A U-Net-like byte-level LM that **learns where chunks end** (a cosine-dissimilarity router between adjacent encoder
states), runs a deep Transformer only on the chunk vectors, and expands back with a confidence-weighted moving
average. With a ratio loss holding the compression, a 1-stage H-Net matches a compute- and data-matched BPE
Transformer and a 2-stage one beats it (1.3B: 0.715 vs 0.730 bits per byte; downstream 58.2 vs 55.5), is ~2× more
robust to noisy text, and helps more on Chinese, code and DNA. Ablations: fixed-width pooling clearly loses;
how a chunk is squeezed does not matter; recurrent (Mamba) encoders compress better than attention.

### The problem that authors want to solve

Tokenizers are a handcrafted, non-end-to-end compression step; byte-level isotropic models are too expensive,
static pooling (MegaByte, Hourglass) ignores variable information rate, and heuristic or external boundaries
(SpaceByte, BLT) are modality-specific. Learning discrete boundaries end to end had been unstable.

### The solution, main idea on the intuition level and strong points

Let the model chunk its own input: where the representation changes, start a new chunk. Put most parameters in a
main network that sees only chunks; keep a cheap full-resolution path for detail. Recurse: the main network can
itself be an H-Net (bytes → sub-words → words / phrases).

### The detailed solution, training process, data preparation

- Encoder (4 Mamba-2 layers) → router `p_t = ½(1 − cos(W_q x_t, W_k x_{t−1}))`, boundary if `p_t ≥ 0.5`
  (`W_q = W_k = I` at init; first position always a boundary) → keep boundary vectors (no pooling) → main network
  → smoothing EMA `z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}` → repeat each chunk vector until the next boundary,
  multiplied by a straight-through confidence (`p` at boundaries, `1 − p` elsewhere) → add `Linear(x̂)` from the
  encoder (weight **zero-initialised**) → decoder (4 Mamba-2 layers).
- Ratio loss `N/(N−1)·((N−1)FG + (1−F)(1−G))`, F = share selected, G = mean probability; weight 0.03; targets 6
  bytes per chunk (1 stage) or (3, 3) (2 stages).
- Signal propagation: RMSNorm at the end of every sub-network; projection only on the shortcut; per-stage
  learning-rate multipliers (outer stages higher, by positions and width); width monotone inward
  (e.g. 1024 / 1024 / 1536).
- 100B FineWeb-Edu tokens; 8,192 bytes per sequence; AdamW with warmup-stable-decay; models matched to GPT-3 Large
  (760M) and XL (1.3B) FLOPs per byte.

### The evaluation procedure, evaluation datasets and results

- Bits per byte (FineWeb-Edu, Large / XL): BPE Transformer 0.756 / 0.730; H-Net pool 0.780 / –; space 0.755 / 0.726;
  1-stage 0.755 / 0.728; **2-stage 0.743 / 0.715**. 2-stage crosses the Transformer after 30B bytes, gap widening.
- Zero-shot average of 7 tasks: XL 2-stage 58.2 vs Transformer 55.5; Large 2-stage 55.5 = XL Transformer.
- Noisy HellaSwag robustness score (XL): 42.8 vs 22.2 for BPE.
- Chinese (XWinograd-zh 66.3 vs 59.9) and code beat Llama-3-tokenized Transformers; DNA: isotropic perplexity with
  3.6× less data.
- Ablations: without smoothing the ratio swings and quality drops; cosine routing and straight-through help;
  Mamba encoders beat Transformer encoders even on BPE-token inputs; selection ≈ mean ≈ max ≈ cross-attention for
  the squeeze (App. E.1); hybrid main network scales better; beats a FLOP-matched MoE.
- Learned chunks: stage 1 at spaces and word starts; stage 2 at words and multi-word groups.
- Cost: up to 2× slower training; dynamic memory per batch.
- Cartesia's scaled 2-stage model (blog excerpts, unverified): 3.67B reaches SmolLM3's six-task average after ~1T
  vs ~11T tokens (different sizes and data); loss-to-loss advantage grows out of domain.

### Previous attempts to solve this problem

Isotropic byte models (ByT5, MambaByte); static pooling (Funnel, Hourglass, MegaByte, Block Transformer);
external boundaries (SpaceByte by spaces, BLT by a small model's entropy); soft or stochastic dynamic boundaries
(MANTa, MrT5, DPT) — unstable or single-stage.

### Max 5 top most relevant to the problem publications from bibliography

- Byte Latent Transformer: [arXiv 2412.09871](https://arxiv.org/abs/2412.09871)
- SpaceByte: [arXiv 2404.14408](https://arxiv.org/abs/2404.14408)
- MegaByte: [arXiv 2305.07185](https://arxiv.org/abs/2305.07185)
- Hourglass Transformer: [arXiv 2110.13711](https://arxiv.org/abs/2110.13711)
- Dynamic Token Pooling (DPT): [arXiv 2211.09761](https://arxiv.org/abs/2211.09761)

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
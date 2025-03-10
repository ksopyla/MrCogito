


# Concept models, combining tokens into concepts

Concept learning and encoding, list of extracts from articles. How to model concept learning, the learning objective, training protocol, etc. 




## "Memory Transformer"


## "ConceptBERT: A Concept-based Framework for Pre-training Language Models"



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

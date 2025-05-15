Training Concept Encoders via Masked Language Modeling: Strategies, Loss Functions, and Implementation1. IntroductionPre-trained language models (PLMs) like BERT and RoBERTa have demonstrated remarkable capabilities in learning language representations through self-supervised objectives, primarily Masked Language Modeling (MLM).1 These models typically operate at the token level, predicting masked tokens based on their surrounding context. However, there is growing interest in models that explicitly learn and operate on higher-level semantic representations, often termed "concepts".1 Such concepts, potentially corresponding to entities, sentences, or abstract ideas, could enable more robust reasoning, better generalization across languages and modalities, and improved handling of long-range dependencies and coherence.3Training models to learn these concept representations effectively presents unique challenges. One promising approach involves adapting the MLM objective: instead of directly predicting tokens from token-level context, the model first encodes the input into concept representations and then uses these concepts to predict the masked tokens. This report provides guidance on training such a concept encoder using MLM, focusing specifically on:
Appropriate masking strategies, including those potentially employed by models like LLaDA 19 and Blank Language Models (BLM).20
The suitability of standard loss functions like Cross-Entropy (CE) loss for architectures where concepts indirectly map to token logits.
Alternative loss functions that might provide a more direct learning signal for concept representations.
Implementation details within the Hugging Face Transformers library for custom architectures (ConceptEncoderForMaskedLM and ConceptEncoderWithSimMatrixForMaskedLM).
The implications of employing large-percentage masking strategies.
The goal is to provide a comprehensive technical overview to facilitate the effective pre-training of concept encoders using masked language modeling objectives.2. Masking Strategies for Concept Encoder TrainingThe choice of masking strategy significantly influences what the model learns during MLM pre-training. Different strategies create different learning signals, potentially biasing the model towards local context, span-level understanding, or more global, conceptual inference.2.1. Standard MLM (BERT-style)The original BERT model 2 employs a simple random token masking strategy:
Mechanism: Approximately 15% of input tokens are randomly selected for masking.
Replacement: Of the selected tokens, 80% are replaced with a special `` token, 10% are replaced with a random token from the vocabulary, and 10% remain unchanged.
Objective: Predict the original identity of the masked tokens based on the surrounding unmasked context.
Bias: This strategy primarily encourages the model to learn local contextual dependencies and token co-occurrence patterns. It provides a relatively dense learning signal focused on individual token prediction.
2.2. Span Masking (SpanBERT-style)SpanBERT 23 proposed modifications to better represent and predict text spans:
Mechanism: Instead of masking random individual tokens, contiguous random spans of text are masked. Span lengths are typically sampled from a distribution (e.g., geometric, mean length ~3.8 tokens in SpanBERT).23 The total percentage of masked tokens remains around 15%.
Objective: SpanBERT introduces a Span Boundary Objective (SBO). It trains the representations of the tokens at the boundaries of the masked span to predict the entire content of the masked span, without relying on the individual token representations within the span.23 This forces the boundary tokens to encapsulate information about the span's content.
Bias: This strategy encourages the model to learn representations that are more aware of span-level information, which has proven beneficial for tasks like question answering and coreference resolution that rely on identifying and reasoning about text spans.23
2.3. Large-Span / Diffusion-Inspired / BLM MaskingModels aiming for higher-level understanding or generation sometimes employ strategies involving masking larger portions of the text.
LLaDA: While specific details on the LLaDA diffusion model's masking strategy for text generation are limited in the provided materials, diffusion models often involve iterative refinement or denoising processes.19 In a text context, this could translate to predicting large missing parts of a sequence, potentially conditioned on diffusion timesteps, analogous to reversing a noise-addition process.19 This might involve masking large, potentially random, contiguous or non-contiguous blocks of text, forcing the model to rely more heavily on global context or latent representations rather than immediate local cues.
Blank Language Models (BLM): BLMs explicitly generate text by creating and filling in "blanks".20 Training involves predicting words to fill blanks and deciding whether to insert new blanks. This inherently deals with predicting potentially large missing segments within a partially generated canvas, pushing the model towards infilling capabilities and potentially learning strong contextual representations to handle significant missing information.20
Iterative Mask Filling: Techniques like Iterative Mask Filling 26 use MLM iteratively, masking and predicting words multiple times within a sentence. While primarily used for data augmentation, this process involves the model making predictions in contexts where multiple tokens might be masked simultaneously or sequentially, requiring robust contextual understanding.
Bias: Masking large portions drastically reduces the available local context for predicting any given masked token. This forces the model to rely more on global context, long-range dependencies, and potentially higher-level conceptual understanding of the entire sequence.20 This aligns well with the goal of training a concept encoder but introduces significant challenges discussed in Section 5.
2.4. Other Masking Variations
Whole Word Masking: If a selected token is part of a larger word (e.g., subword tokenization), all tokens corresponding to that word are masked. This prevents the model from trivially predicting parts of a word based on other parts. The DataCollatorForWholeWordMask implements this.
Neighbor Word Masking: The NeighborWordMaskCollator likely implements a strategy related to span masking or masking words adjacent to each other, encouraging the model to learn relationships between neighboring words or within local phrases.
Table 1: Comparison of Masking StrategiesStrategyMasking UnitPercentageObjective(s)Primary BiasPotential for Concept LearningStandard MLM (BERT)Random Tokens~15%Predict masked tokensLocal Context, Token Co-occurrenceModerateSpan Masking (SpanBERT)Contiguous Spans~15%Predict masked tokens + SBOSpan-level RepresentationEnhanced (via SBO)Large-Span / BLMLarge Spans/Blanks>15% (Variable)Predict masked tokens / Fill blanksGlobal Context, InfillingHigh (if stable training)Whole Word MaskingFull Words~15%Predict masked tokens (full words)Word-level SemanticsModerateFor training a concept encoder, strategies that move beyond individual random tokens, such as SpanBERT's SBO or carefully implemented large-span masking, appear more aligned with the goal of capturing higher-level information. However, the increased difficulty of large-span masking necessitates careful consideration of training stability and loss function design.3. Loss Functions for Concept-based MLMThe standard loss function for MLM is Cross-Entropy (CE) loss, calculated between the model's predicted logits for masked positions and the true token IDs.3.1. Standard Cross-Entropy Loss
Mechanism: For each masked token position i, the model outputs a logit vector li​∈R∣V∣, where ∣V∣ is the vocabulary size. The CE loss compares the softmax probability distribution derived from li​ with a one-hot vector representing the true token yi​. The total loss is averaged over all masked tokens.
$$ \mathcal{L}{CE} = - \sum{i \in \text{masked}} \log P(y_i | \text{context}) = - \sum_{i \in \text{masked}} \log \frac{\exp(l_{i, y_i})}{\sum_{j=1}^{|\mathcal{V}|} \exp(l_{i, j})} $$
Suitability for ConceptEncoderForMaskedLM: In this architecture, concept embeddings ci​∈Rdconcept​ are mapped to vocabulary logits li​∈R∣V∣, likely via an attention mechanism or a linear projection.29 CE loss can train this system end-to-end. However, the learning signal for the concept encoder itself is indirect. The gradients must flow back through the concept-to-logit mapping mechanism. If this mapping is complex or inefficient, or if dconcept​≪∣V∣, the signal might become weak or noisy, potentially hindering the learning of high-quality concept representations. The model might find it easier to optimize the mapping layer rather than improving the underlying concept embeddings.
Suitability for ConceptEncoderWithSimMatrixForMaskedLM: This architecture introduces further complexity with a similarity matrix and a gating mechanism combining concept-based logits (lconcept​) and token-based logits (ltoken​) [Outline]. The final logits are li​=gi​⋅lconcept,i​+(1−gi​)⋅ltoken,i​. While CE loss can still be applied to li​, the signal backpropagating to the concept encoder is even more indirect. It's modulated by the gate gi​ and potentially diluted by the parallel token-based pathway. If the token-based pathway is strong (e.g., derived from a standard transformer), the model might learn to minimize CE loss primarily through that path, potentially neglecting or even shutting off the concept pathway via the gate, thus failing to train the concept encoder effectively.
3.2. Alternative Loss Functions / Training ObjectivesGiven the indirect nature of CE loss for training the concept encoder in these architectures, alternative objectives that provide a more direct signal to the latent concept representations are worth considering.
Variational Autoencoder (VAE) Objective (ELBO): VAEs are generative models designed to learn latent representations.31 In this context, the concept encoder could act as the VAE encoder, mapping input text to a distribution over latent concept vectors (e.g., mean μ and variance σ2). The decoder part would reconstruct the masked tokens conditioned on a sample z from this latent distribution. The training objective is the Evidence Lower Bound (ELBO):
$$ \mathcal{L}{ELBO} = \mathbb{E}{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) |
| p(z)) $$
where the first term is the reconstruction loss (equivalent to CE loss for predicting masked tokens, conditioned on z) and the second term is the Kullback-Leibler (KL) divergence between the learned posterior distribution q(z∣x) (output by the concept encoder) and a prior distribution p(z) (often a standard Gaussian N(0,I)).

Pros: Directly optimizes the latent representation (z) via the KL term, encouraging it to capture meaningful variations while being constrained by the prior. Provides a principled way to model uncertainty in concept representations. Has been used successfully for text generation, sometimes with hierarchical latent variables for better coherence.35
Cons: Can suffer from the "KL vanishing" problem, where the KL term goes to zero, and the decoder ignores the latent variable, especially with powerful autoregressive decoders.36 Requires careful tuning of the KL weight (often using annealing). Adds complexity with sampling during training and defining the VAE decoder structure.


Contrastive Learning: Contrastive objectives aim to learn representations where similar inputs are mapped closer together in the embedding space, while dissimilar inputs are pushed apart. In this context, one could define "similar" based on semantic relatedness or conceptual overlap.

Mechanism: An auxiliary contrastive loss could be added. For a given input sequence (anchor), create positive examples (e.g., augmentations of the same sequence, sequences discussing the same underlying concept) and negative examples (unrelated sequences). The loss encourages the concept embeddings of the anchor and positives to be closer than the embeddings of the anchor and negatives.
Pros: Directly shapes the concept embedding space based on semantic similarity. Can lead to highly discriminative representations. Does not necessarily require a generative decoder component.
Cons: Requires careful definition and sourcing of positive/negative examples, which can be challenging for abstract concepts. Performance is sensitive to the choice of negative samples and temperature scaling. May not directly optimize for the MLM reconstruction task itself.


Perceiver-style Latent Bottleneck: The Perceiver architecture uses a fixed-size latent array as an attentional bottleneck.45 Inputs attend to latents, and latents attend to each other through transformer blocks. A similar idea could be applied here: the concept encoder produces a set of latent "concept" tokens. These latents then attend to the input token representations (or vice-versa) and are processed internally before being used for prediction.

Pros: Decouples the main computation from the input sequence length, potentially improving efficiency for long inputs. Forces information compression through the latent bottleneck.
Cons: The connection between these architectural latents and interpretable "concepts" might be weak. Training dynamics can be complex. May still rely on CE loss for the final prediction, inheriting some of its indirectness issues.


Table 2: Comparison of Loss Functions for Concept Encoder Training via MLMLoss Function / ObjectiveDirectness of Signal to Concept EncoderReconstruction TaskComplexityPotential IssuesCross-Entropy (CE)Indirect (via mapping/gating)YesLowWeak signal, potential pathway neglect (gated)VAE (ELBO)Direct (KL term) + Indirect (Recon.)YesMediumKL vanishing, tuning complexityContrastive LossDirect (similarity/dissimilarity)No (Auxiliary)MediumSample selection, may not align with MLMPerceiver-styleIndirect (via bottleneck attention)YesHighInterpretability, complex dynamicsStandard CE loss remains the simplest starting point. However, if analysis suggests the concept encoder is not learning effectively due to the indirect signal, incorporating a VAE objective (ELBO) appears the most theoretically grounded alternative for directly optimizing the latent concept representations within an MLM framework.4. Implementation Details with Hugging Face / PyTorchTraining MLM models, including custom architectures, can leverage the Hugging Face Trainer API and associated components like data collators.4.1. Standard MLM Training Loop (Hugging Face)The typical workflow involves:
Dataset Preparation: Tokenize raw text data into input_ids and attention_mask.
Data Collator: Use DataCollatorForLanguageModeling. This collator takes a batch of tokenized examples and performs the masking strategy (e.g., 15% random token masking by default). It returns a dictionary containing:

input_ids: Tensor with some tokens replaced by `` or random tokens.
attention_mask: Standard attention mask.
labels: A tensor of the same shape as input_ids. It contains the original token IDs for the masked positions and -100 elsewhere. The value -100 is the default ignore_index for PyTorch's CrossEntropyLoss.


Model Forward Pass: The model (e.g., BertForMaskedLM) receives input_ids, attention_mask, and optionally labels. It computes hidden states and projects the representations at all positions (including unmasked ones) to vocabulary logits, typically of shape [batch_size, seq_length, vocab_size].
Loss Calculation: If labels are provided, the model calculates the CE loss. Internally, this involves:

Selecting the logits corresponding to the masked positions (where labels!= -100).
Reshaping logits and labels to [num_masked_tokens, vocab_size] and [num_masked_tokens], respectively.
Applying torch.nn.CrossEntropyLoss(ignore_index=-100) which automatically handles the reshaping and ignores the -100 entries in the labels tensor.


Trainer API: The Trainer orchestrates the process, feeding batches from the data collator to the model, collecting the loss, performing backpropagation, and updating model weights.
4.2. Implementation for ConceptEncoderForMaskedLMThis model uses concept embeddings to predict masked tokens.

Model Definition: Define a class inheriting from transformers.PreTrainedModel. It should contain the ConceptEncoder module and a mechanism (e.g., an attention layer or linear layer) to map concept embeddings to vocabulary logits. The configuration should store vocab_size and concept_dim.


Forward Pass Implementation:
Pythonimport torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
# Assuming ConceptEncoder is defined elsewhere and returns embeddings of shape [batch, seq_len, concept_dim]

class ConceptEncoderForMaskedLM(PreTrainedModel):
    def __init__(self, config, concept_encoder):
        super().__init__(config)
        self.concept_encoder = concept_encoder
        # Option 1: Simple Linear Projection
        self.lm_head = nn.Linear(config.concept_dim, config.vocab_size)
        # Option 2: More complex mapping (e.g., attention-based) - requires more parameters
        # self.vocab_embeddings = nn.Embedding(config.vocab_size, config.concept_dim) # Example
        # self.attention_mapper = SomeAttentionMechanism(...) # Example

        self.init_weights() # Initialize weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        # Add other inputs required by concept_encoder if necessary
        **kwargs,
    ):
        # 1. Get Concept Embeddings
        # Assuming concept_encoder takes input_ids and attention_mask
        outputs = self.concept_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Pass other necessary args
            output_hidden_states=True, # Ensure embeddings are returned
            **kwargs
        )
        # Assuming the last hidden state is the concept embedding
        concept_embeddings = outputs.last_hidden_state # Shape: [batch_size, seq_length, concept_dim]

        # 2. Map Concept Embeddings to Logits
        # Using Option 1 (Linear Projection)
        logits = self.lm_head(concept_embeddings) # Shape: [batch_size, seq_length, vocab_size]
        # If using Option 2 (Attention), implement the mapping logic here

        # 3. Calculate Loss (if labels provided)
        loss = None
        if labels is not None:
            # Standard Hugging Face loss calculation for MLM
            loss_fct = CrossEntropyLoss() # ignore_index defaults to -100
            # Reshape logits and labels for CrossEntropyLoss
            # Logits: [batch_size, seq_length, vocab_size] -> [batch_size * seq_length, vocab_size]
            # Labels: [batch_size, seq_length] -> [batch_size * seq_length]
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 4. Return Output
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states, # Pass through hidden states from encoder
            attentions=outputs.attentions, # Pass through attentions from encoder
        )




Mapping Concept Dimension to Vocabulary Size: The critical step is mapping the concept_embeddings (shape [batch, seq_len, concept_dim]) to logits (shape [batch, seq_len, vocab_size]). A simple nn.Linear(config.concept_dim, config.vocab_size) is shown above. This assumes a direct, position-wise mapping. However, if concept_dim is significantly smaller than vocab_size, or if the relationship is highly non-linear, this linear projection might be insufficient. A more sophisticated approach could involve using the concept embeddings as queries to attend over the vocabulary embeddings (treated as keys/values), allowing a more flexible mapping but increasing computational complexity and parameters.29 The choice directly impacts how effectively the learned concept representation can be translated back into token predictions for the MLM task.

4.3. Implementation for ConceptEncoderWithSimMatrixForMaskedLMThis model uses a similarity matrix and gating to combine concept-based and token-based predictions.

Model Definition: Define the class inheriting from PreTrainedModel. Include the ConceptEncoder, modules for the similarity matrix calculation, the gating mechanism, and potentially a parallel pathway for generating token-based logits.


Forward Pass Implementation:
Pythonimport torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
# Assuming ConceptEncoder and similarity/gating components are defined

class ConceptEncoderWithSimMatrixForMaskedLM(PreTrainedModel):
    def __init__(self, config, concept_encoder, sim_matrix_module, gate_module, token_logit_module):
        super().__init__(config)
        self.concept_encoder = concept_encoder
        self.sim_matrix_module = sim_matrix_module # Module to compute concept_logits
        self.gate_module = gate_module # Module to compute gate values
        self.token_logit_module = token_logit_module # Module to compute token_logits

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        # Add other inputs if needed
        **kwargs,
    ):
        # 1. Get Concept Embeddings
        outputs = self.concept_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        concept_embeddings = outputs.last_hidden_state # Shape: [batch, seq_len, concept_dim]

        # 2. Calculate Concept-based Logits
        # This depends heavily on the sim_matrix_module implementation
        concept_logits = self.sim_matrix_module(concept_embeddings) # Shape: [batch, seq_len, vocab_size]

        # 3. Calculate Token-based Logits
        # This depends heavily on the token_logit_module implementation
        # Example: Could be another encoder, or simpler projection from input embeddings
        token_logits = self.token_logit_module(input_ids, attention_mask) # Shape: [batch, seq_len, vocab_size]

        # 4. Calculate Gate Values
        # Gate could depend on concept_embeddings or other features
        gate_values = self.gate_module(concept_embeddings) # Shape: [batch, seq_len, 1] or [batch, seq_len, vocab_size]
        gate = torch.sigmoid(gate_values)

        # 5. Combine Logits
        # Ensure broadcasting if gate shape is [batch, seq_len, 1]
        final_logits = gate * concept_logits + (1 - gate) * token_logits

        # 6. Calculate Loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() # ignore_index defaults to -100
            loss = loss_fct(final_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 7. Return Output
        return MaskedLMOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # Optionally return gate values, concept_logits, token_logits for debugging
        )



Training the Gate: The gating mechanism adds complexity. The model must learn when to prioritize the concept-based predictions versus the token-based ones. The standard CE loss provides only an indirect signal for training the gate parameters. The gradient flowing to the gate depends on how changes in the gate value affect the final CE loss via the combined logits. If one pathway (e.g., the token-based one) consistently provides a lower-loss prediction, the gate might learn to predominantly favor that pathway, potentially hindering the training of the concept encoder. If the gate fails to learn a meaningful balancing strategy, auxiliary loss functions specifically targeting the gate's behavior (e.g., encouraging openness based on concept confidence or relevance metrics) might be necessary.


Defining the "Token-based Logits": The design of the token_logit_module is crucial. If it implements a powerful standard transformer encoder (like BERT), its predictions (token_logits) might be very accurate, potentially overshadowing the concept_logits. In this scenario, the concept pathway must offer a significant advantage (e.g., better handling of specific conceptual nuances or long-range dependencies) for the gate to learn to utilize it. Conversely, if the token_logit_module is simpler (e.g., a linear projection from input token embeddings), the MLM task becomes harder for this pathway alone, potentially creating more pressure and opportunity for the concept pathway (and the concept encoder) to learn and contribute meaningfully. This design choice creates a trade-off affecting the learning dynamics and the demands placed on the concept encoder.

5. Implications of Large-Percentage Masking StrategiesEmploying masking strategies that cover a large percentage of the input sequence (e.g., > 50%, as potentially used in LLaDA-like diffusion models or BLMs 19) has significant implications compared to standard 15% masking.5.1. Training Dynamics and StabilityMasking a large fraction of tokens substantially increases the difficulty of the prediction task for each masked token, as the available context is drastically reduced. This can lead to:
Instability: The increased difficulty may result in unstable training dynamics, requiring smaller learning rates, longer warm-up periods, or more careful hyperparameter tuning to prevent divergence.
Slower Convergence: While each step might force stronger learning, the overall convergence time might increase due to the task's difficulty and potential instability.
Gradient Quality: The gradients derived from predicting tokens with very limited context might be noisier or sparser, especially early in training. This could make it harder for the model to discern meaningful patterns initially. In contrast, standard 15% masking provides ample local context, generally leading to more stable training and smoother convergence for established architectures.
5.2. Computational Efficiency
Per-Step Cost: Calculating the loss function over a larger number of masked tokens increases the computational cost of the backward pass. If 50% of tokens are masked versus 15%, the loss computation involves over three times as many predictions contributing to the loss gradient. The forward pass cost might remain similar depending on the architecture, but the gradient computation and weight updates will be heavier. Techniques for efficient attention computation become increasingly relevant as sequence lengths or computational demands grow.47
Overall Cost: The total training cost depends on the interplay between per-step cost and convergence speed. If large masking significantly slows convergence, the overall computational budget required might be higher, even if the model potentially learns more robust representations per step.
5.3. Quality of Learning Signal for Concept Encoder
Potential Benefit: The core motivation for large masking in this context is to force the model beyond local patterns. By severely limiting local context, the model must rely on understanding the global structure, semantics, and underlying concepts of the sequence to make reasonable predictions.20 This could provide a strong, albeit challenging, learning signal for the concept encoder, driving it to capture more abstract and holistic information. Models like BLMs demonstrate that generation is possible even with substantial missing information.20
Potential Drawback: If the prediction task becomes overwhelmingly difficult due to insufficient context, the learning signal might become too weak or noisy to effectively train the concept encoder. The model might resort to learning superficial statistics, fail to converge, or produce incoherent outputs.11 There is an inherent information bottleneck trade-off: large masking creates a bottleneck, forcing compression and abstraction (potentially into concepts) if successful, but risking failure if the remaining information is fundamentally insufficient for meaningful prediction.
Comparison to 15% Masking: Standard 15% masking provides a dense, primarily local learning signal, leading to stable training but potentially weaker learning of global concepts. Large-percentage masking provides a sparser, potentially more global signal, but is harder and riskier.
Given these factors, a pragmatic approach would be to start with more conservative masking strategies (e.g., SpanBERT-style at 15-25%) and incrementally increase the masking percentage or span length while carefully monitoring training stability, convergence speed, and downstream performance indicative of concept understanding.6. Conclusion and RecommendationsTraining concept encoders using Masked Language Modeling presents a promising avenue for developing models with deeper semantic understanding. However, success requires careful consideration of masking strategies, loss functions, architectural choices, and implementation details.6.1. Synthesis of Findings
Masking: Standard random token masking (BERT) primarily captures local context. Span-based masking (SpanBERT) improves span representation. Large-span or BLM-style masking forces reliance on global context, potentially ideal for concept learning but introduces significant training challenges (stability, signal quality).
Loss Functions: Standard Cross-Entropy loss is simple to implement but provides an indirect signal to the concept encoder, especially in complex architectures involving attention mapping or gating. VAE-based objectives (ELBO) offer a more direct way to optimize latent concept representations but add complexity and potential KL vanishing issues. Contrastive losses can shape the embedding space but require careful sample design.
Implementation: Hugging Face Trainer and DataCollatorForLanguageModeling provide a standard framework. Custom model implementations must correctly handle the flow from input to concept embeddings, map these embeddings (directly or indirectly) to vocabulary logits, and compute the loss using the standard -100 ignore index for unmasked tokens. Architectures with gating mechanisms require careful design of the parallel token-based pathway and monitoring of gate training.
Large Masking: While potentially beneficial for learning global concepts by creating an information bottleneck, high masking percentages (>50%) increase task difficulty, risk training instability, raise computational costs per step, and may yield noisy gradients. The trade-off between forcing global understanding and providing sufficient signal is critical.
6.2. Actionable Recommendations
Masking Strategy Selection: Begin with established, robust strategies. Use DataCollatorForWholeWordMask or implement SpanBERT-style masking (potentially using NeighborWordMaskCollator as a starting point or implementing custom span sampling and the Span Boundary Objective). Consider increasing the masking percentage moderately (e.g., 20-30%) from the standard 15%. Only explore very large masking percentages (>50%) or BLM-like dynamic masking if initial results indicate a need for a stronger global signal and if training stability can be maintained.
Loss Function Selection: Start with the standard CrossEntropyLoss. It is the simplest baseline and works well for many MLM tasks. Monitor the learning progress of the concept encoder's parameters specifically (e.g., gradient norms, weight changes). If the concept encoder appears undertrained or the concept representations lack quality (assessed via probing or downstream tasks), consider implementing a VAE-based objective (ELBO loss with KL divergence on concept embeddings) as the next step, as it directly optimizes the latent space. An auxiliary contrastive loss could also be explored.
Architecture-Specific Considerations (ConceptEncoderWithSimMatrixForMaskedLM):

Token Logit Source: Carefully design the module generating token_logits. Using a very simple baseline (e.g., linear projection from input embeddings) might initially encourage the model to rely more on the concept pathway.
Gate Training: Monitor the gate's behavior during training (e.g., average activation values). If the gate consistently favors one pathway or fails to learn a dynamic strategy, consider adding an auxiliary loss term to guide its learning based on prediction confidence or other heuristics.


Evaluation Beyond Perplexity: MLM perplexity measures how well the model predicts masked tokens but not necessarily the quality of the underlying concept representations. Define and track additional metrics:

Probing Tasks: Design tasks to explicitly probe the learned concept embeddings for desired properties (e.g., clustering concepts, predicting concept attributes).1
Downstream Task Performance: Evaluate the fine-tuned concept encoder on downstream tasks that heavily rely on conceptual understanding or reasoning (e.g., knowledge-intensive QA 2, relation extraction 23, commonsense reasoning 10).
Coherence Metrics: Assess the coherence of generated text or representations if applicable.11


6.3. Future Considerations
Hierarchical Concepts: Explore architectures that learn hierarchical concepts (e.g., global and local latent variables) for improved long-text coherence and structure.35
Alternative Architectures: Investigate the integration of concept learning with architectures potentially better suited for long sequences or different computational paradigms, such as State Space Models (Mamba) 75 or Memory Transformers.72
Concept Grounding: Explore methods to ground learned concepts in external knowledge sources (like taxonomies or knowledge graphs) during pre-training.1
By systematically exploring masking strategies, evaluating loss functions, carefully implementing the custom architectures, and rigorously evaluating the learned representations, it is possible to effectively train concept encoders using the powerful framework of masked language modeling.
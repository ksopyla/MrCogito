"""
ConceptEncoder with Perceiver IO style decoding.

This module provides ConceptEncoder models using Perceiver IO cross-attention
for decoding concept representations back to sequence/classification outputs.

Models:
- ConceptEncoderForDenoisingPerceiver: canonical denoising-first perceiver
- ConceptEncoderForMaskedLMPerceiver: MLM with Input+Position queries (hybrid approach)
- ConceptEncoderForMaskedLMPerceiverPosOnly: legacy alias for the denoising perceiver
- ConceptEncoderForSequenceClassificationPerceiver: Sequence classification via concept mean pooling
- ConceptEncoderForSequenceClassificationViaDecoder: Sequence classification via pretrained decoder
- ConceptEncoderForSentencePairClassification: Sentence-pair tasks with separate encoding

Decoder Query Strategies:
- Input+Position (default): Query = token_embedding + position_embedding
  Provides a "hint" about what token was at each position.
- Position-only: Query = position_embedding only
  Pure Perceiver IO style used by the maintained denoising stack.

Loss Management:
- Uses LossManager for clean, extensible loss handling
- Supports MLM-only, MLM + concept loss, MLM + multiple concept losses
- Supports fixed, learnable, and uncertainty-based weighting

Example:
    >>> from nn.loss_manager import LossConfig, LossManager
    >>> 
    >>> # Legacy MLM baseline with fixed orthogonality weight
    >>> loss_config = LossConfig(
    ...     concept_losses=["orthogonality"],
    ...     weighting_strategy="fixed"
    ... )
    >>> model = ConceptEncoderForMaskedLMPerceiver(model_config, loss_config=loss_config)
    >>> 
    >>> # Position-only variant (pure Perceiver IO)
    >>> model = ConceptEncoderForMaskedLMPerceiverPosOnly(model_config, loss_config=loss_config)
    >>> 
    >>> # For inference (no concept loss)
    >>> model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("path/to/model")
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig, build_norm, embedding_padding_idx
from nn.loss_manager import LossManager, LossConfig

logger = logging.get_logger(__name__)


class PerceiverDecoderLayer(nn.Module):
    """Canonical linear Perceiver-IO decoder block: position queries cross-attend the C
    concepts, then a gated FFN. There is deliberately **no self-attention over the N output
    queries** — that would be O(N^2) and break the project's O(C*N) bottleneck invariant,
    which is incompatible with the long-context vision. Output positions are therefore
    conditionally independent given the concepts (the standard non-autoregressive parallel
    decode); all cross-position information must flow through the concept bottleneck.
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.pre_cross_norm = nn.LayerNorm(config.hidden_size)
        self.pre_ff_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_in = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.ffn_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = nn.GELU()

    def forward(
        self,
        query_states: torch.Tensor,
        concept_repr: torch.Tensor,
    ) -> torch.Tensor:
        cross_normed_queries = self.pre_cross_norm(query_states)
        cross_attn_output, _ = self.cross_attn(
            query=cross_normed_queries,
            key=concept_repr,
            value=concept_repr,
            need_weights=False,
        )
        query_states = query_states + cross_attn_output

        ff_input, ff_gate = self.ffn_in(self.pre_ff_norm(query_states)).chunk(2, dim=-1)
        ff_output = self.ffn_out(self.ffn_dropout(self.act_fn(ff_input) * ff_gate))
        return query_states + ff_output


class PerceiverDecoderStack(nn.Module):
    """Position-only Perceiver decoder used by all maintained perceiver checkpoints."""

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.query_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [PerceiverDecoderLayer(config) for _ in range(config.decoder_num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def build_queries(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.query_embeddings(position_ids).expand(batch_size, -1, -1)

    def forward(
        self,
        concept_repr: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = query_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, concept_repr)
        return self.output_norm(hidden_states)


class ConceptEncoderForDenoisingPerceiver(PreTrainedModel):
    """Canonical denoising-first perceiver pretraining model."""

    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None,
    ):
        super().__init__(config)
        self.config = config
        self.config.decoder_posonly = True
        self.encoder = ConceptEncoder(config)
        self.decoder = PerceiverDecoderStack(config)
        self.set_loss_config(loss_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        if config.tie_word_embeddings and config.token_embedding_dim == config.hidden_size:
            self._tie_or_clone_weights(self.lm_head, self.encoder.token_embeddings)

    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        self.loss_manager = LossManager.create_for_model(
            concept_num=self.config.concept_num,
            hidden_size=self.config.hidden_size,
            loss_config=loss_config,
        )
        self._loss_config = loss_config

    def encode_concepts(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def decode_from_concepts(
        self,
        concept_repr: torch.Tensor,
        seq_length: int,
    ) -> torch.Tensor:
        queries = self.decoder.build_queries(
            batch_size=concept_repr.size(0),
            seq_length=seq_length,
            device=concept_repr.device,
        )
        return self.decoder(concept_repr=concept_repr, query_states=queries)

    def pool_concepts(self, concept_repr: torch.Tensor) -> torch.Tensor:
        return concept_repr.mean(dim=1)

    def reconstruction_loss(
        self,
        decoder_output: torch.Tensor,
        labels: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.lm_head(decoder_output)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        task_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, task_loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        del token_type_ids, special_tokens_mask
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state
        decoder_output = self.decode_from_concepts(concept_repr, seq_length=input_ids.size(1))

        loss = None
        logits = None
        if labels is not None:
            logits, task_loss = self.reconstruction_loss(decoder_output, labels)
            if self.training:
                loss = self.loss_manager(task_loss=task_loss, concept_repr=concept_repr)
            else:
                loss = task_loss
        else:
            logits = self.lm_head(decoder_output)

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ConceptEncoderForMaskedLMPerceiver(PreTrainedModel):
    """
    ConceptEncoder with Perceiver IO style decoding for Masked Language Modeling.
    
    Architecture:
    1. Encoder: Tokens -> Concepts via cross-attention
    2. Decoder: Concepts -> Token predictions via Perceiver IO cross-attention
    
    Training:
    - For training with concept regularization, pass `loss_config` to __init__
    - For inference or baseline (no concept loss), omit `loss_config`
    
    This follows SOLID principles:
    - Model config (ConceptEncoderConfig) = architecture only, saved with model
    - Loss config (LossConfig) = training behavior, NOT saved with model
    - Loss computation delegated to LossManager (Single Responsibility)
    
    Example:
        >>> from nn.loss_manager import LossConfig
        >>> 
        >>> # For training with concept loss
        >>> loss_config = LossConfig(
        ...     concept_losses=["orthogonality", "uniformity"],
        ...     weighting_strategy="fixed"
        ... )
        >>> model = ConceptEncoderForMaskedLMPerceiver(model_config, loss_config=loss_config)
        >>> 
        >>> # For inference (no concept loss)
        >>> model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("path/to/model")
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"
    # Declare tied weight pairs so safetensors handles them correctly.
    # lm_head.weight is tied to encoder.token_embeddings.weight when
    # config.tie_word_embeddings=True and token_embedding_dim==hidden_size.
    # Without this, safetensors >=0.4.3 raises RuntimeError on checkpoint save.
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self, 
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None
    ):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # === Loss Management (Delegated to LossManager) ===
        # Initialize via set_loss_config to avoid duplication
        self.set_loss_config(loss_config)
        
        # === Decoder Architecture ===
        # Decoder Queries: Position embeddings (always in hidden_size space)
        self.decoder_query_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length, 
            embedding_dim=config.hidden_size
        )
        
        # Dimension Inversion: when token_embedding_dim < hidden_size, the input embeddings
        # used in decoder queries are in token_dim space and need projection to hidden_size.
        # This projection is separate from the encoder's token_projection because it handles
        # the decoder-specific input contribution to queries.
        if config.token_embedding_dim != config.hidden_size:
            self.decoder_input_projection = nn.Linear(config.token_embedding_dim, config.hidden_size)
        else:
            self.decoder_input_projection = None
        
        # Cross-Attention: Query=Position/Input, Key=Concepts, Value=Concepts
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        
        # Decoder Layer Norms
        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        self.post_cross_norm = nn.LayerNorm(config.hidden_size)
        
        # FFN after attention
        self.decoder_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # MLM Head: projects from hidden_size to vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
        # Optionally tie embeddings (only when token_embedding_dim == hidden_size,
        # otherwise shapes mismatch and tying is not possible)
        if config.tie_word_embeddings and config.token_embedding_dim == config.hidden_size:
            self._tie_or_clone_weights(self.lm_head, self.encoder.token_embeddings)
    
    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        """
        Update loss configuration (e.g., for ablation studies mid-training).
        
        Args:
            loss_config: New loss configuration, or None to disable concept loss
        """
        self.loss_manager = LossManager.create_for_model(
            concept_num=self.config.concept_num,
            hidden_size=self.config.hidden_size,
            loss_config=loss_config
        )
        self._loss_config = loss_config

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = input_ids.shape
        
        # 1. Encode: Tokens -> Concepts
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        # 2. Decode: Concepts -> Sequence using Perceiver IO (Cross Attention)
        # Construct Queries: Input Embeddings + Position Embeddings
        # This gives the decoder a hint about what was at the position (especially for unmasked tokens)
        # and allows the model to focus on filling in the [MASK] tokens using concepts.
        
        # A. Position Embeddings (always in hidden_size space)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.decoder_query_embeddings(position_ids).expand(batch_size, -1, -1)
        
        # B. Input Embeddings (Reuse encoder's token embeddings)
        # When Dimension Inversion is active (token_dim < hidden_size), project to hidden_size
        input_embeddings = self.encoder.token_embeddings(input_ids)
        if self.decoder_input_projection is not None:
            input_embeddings = self.decoder_input_projection(input_embeddings)
        
        # Combine: Query = Input + Position (both now in hidden_size space)
        decoder_queries = input_embeddings + pos_embeddings
        
        # Norm queries before attention (Pre-LN)
        decoder_queries_norm = self.decoder_norm(decoder_queries)
        
        # Cross Attention
        # Query: Input+Pos [B, L, H]
        # Key/Value: Concepts [B, C, H]
        # need_weights=False enables SDPA/Flash Attention fast path on PyTorch 2.x
        # (returning attn_weights forces the slow O(N^2) materialized attention path)
        attn_output, _ = self.decoder_cross_attn(
            query=decoder_queries_norm,
            key=concept_repr,
            value=concept_repr,
            need_weights=False
        )
        
        # Residual Connection 1 (Add attention result to original queries)
        decoder_latents = decoder_queries + attn_output
        
        # Feed Forward Network with Residual 2
        # Note: We apply norm before FFN (Pre-LN style)
        decoder_output = decoder_latents + self.decoder_ffn(self.post_cross_norm(decoder_latents))
        
        # 3. Compute logits and loss
        # Use sparse decoding for memory efficiency whenever labels are provided
        # (both training AND evaluation). Only compute logits for masked positions
        # (~15% of sequence), saving ~7x memory on the logits tensor.
        # The loss is mathematically identical to full decoding with ignore_index=-100.
        # Full logits are only materialized for pure inference (labels=None).
        loss = None
        logits = None
        
        if labels is not None:
            # SPARSE MLM DECODING: Only compute logits for masked positions
            # labels != -100 indicates positions where we need predictions
            mask = (labels != -100)  # [B, L]
            
            # Gather decoder outputs only at masked positions
            # Use reshape instead of view - decoder_output may be non-contiguous
            flat_decoder_output = decoder_output.reshape(-1, decoder_output.size(-1))  # [B*L, H]
            flat_mask = mask.reshape(-1)  # [B*L]
            
            # Select only masked positions
            masked_decoder_output = flat_decoder_output[flat_mask]  # [num_masked, H]
            
            # Project only masked positions to vocabulary
            masked_logits = self.lm_head(masked_decoder_output)  # [num_masked, V]
            
            # Get corresponding labels
            flat_labels = labels.view(-1)  # [B*L]
            masked_labels = flat_labels[flat_mask]  # [num_masked]
            
            # Compute MLM loss on sparse predictions
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(masked_logits, masked_labels)
            
            # Apply loss manager for concept regularization (only during training)
            if self.training:
                loss = self.loss_manager(
                    task_loss=mlm_loss,
                    concept_repr=concept_repr
                )
            else:
                loss = mlm_loss
            
            # Don't return full logits - saves ~6GB from [B,L,V] tensor + fp32 conversion
            logits = None
            
        else:
            # FULL DECODING: For pure inference without labels (e.g., generation)
            logits = self.lm_head(decoder_output)  # [B, L, V]
        
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ConceptEncoderForSequenceClassificationPerceiver(PreTrainedModel):
    """
    ConceptEncoder with weighted concept pooling for Sequence Classification.
    
    Uses simple mean pooling over concepts to avoid introducing another learned
    pooling bottleneck into concept-space evaluation.
    
    Example:
        >>> model = ConceptEncoderForSequenceClassificationPerceiver(config)
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss = outputs.loss
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        self.pool_norm = nn.LayerNorm(config.hidden_size)
        self.pool_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        pooled = self.pool_norm(concept_repr.mean(dim=1))
        
        logits = self.classifier(self.pool_dropout(pooled))  # [B, num_labels]
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# =============================================================================
# Classification via Pretrained MLM Decoder (Experiment 3.1)
# =============================================================================

class ConceptEncoderForSequenceClassificationViaDecoder(PreTrainedModel):
    """
    ConceptEncoder with sequence classification using the pretrained denoising decoder.
    
    KEY INSIGHT: Instead of discarding the MLM decoder and using a single CLS query,
    this model REUSES the full pretrained decoder to reconstruct a sequence representation,
    then pools and classifies. This loads ALL pretrained weights (encoder + decoder)
    rather than just encoder weights.
    
    Architecture:
    1. Encoder: Tokens -> Concepts via cross-attention (pretrained)
    2. Decoder: Concepts -> Full sequence via the shared position-only decoder stack
    3. Pool: Mean pool decoder output over non-padding positions
    4. Classify: Linear head on pooled representation
    
    Weight Loading:
    - encoder.* weights loaded from MLM checkpoint (pretrained)
    - decoder_* weights loaded from MLM checkpoint (pretrained) 
    - classifier.* randomly initialized (trained during fine-tuning)
    
    Example:
        >>> model = ConceptEncoderForSequenceClassificationViaDecoder(config)
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss = outputs.loss
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.encoder = ConceptEncoder(config)
        self.decoder = PerceiverDecoderStack(config)
        
        # === Classification Head (new, randomly initialized) ===
        self.pre_pool_norm = nn.LayerNorm(config.hidden_size)
        
        self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = input_ids.shape
        
        # 1. Encode: Tokens -> Concepts (pretrained)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        decoder_queries = self.decoder.build_queries(batch_size, seq_length, input_ids.device)
        decoder_output = self.decoder(concept_repr=concept_repr, query_states=decoder_queries)
        
        # 3. Pool: Mean pool over non-padding positions (like BERT mean pooling)
        # Apply LayerNorm before pooling for stable representations
        decoder_output = self.pre_pool_norm(decoder_output)
        
        # Mask padding positions
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (decoder_output * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = decoder_output.mean(dim=1)  # [B, H]
        
        # 4. Classify
        logits = self.classifier(self.classifier_dropout(pooled))  # [B, num_labels]
        
        # 5. Compute task loss (no concept regularization for classification)
        loss = None
        if labels is not None:
            # Determine problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            # Compute task loss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# =============================================================================
# Position-Only Decoder Variants (Pure Perceiver IO Style)
# =============================================================================

# =============================================================================
# Sentence-Pair Classification (separate encoding, concept-space comparison)
# =============================================================================

class AttentionPool(nn.Module):
    """Permutation-aware pooling over the C concepts via a single learned query.

    Unlike mean-pooling, a learned query can attend to the slots that carry the
    instance-specific information, so information *distributed across concepts*
    becomes recoverable. One query, single-head cross-attention: tiny by design
    (a frozen-encoder probe — the delta vs mean-pool is the signal, not capacity).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.query, std=hidden_size ** -0.5)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

    def forward(self, concepts: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # concepts: [B, C, H] -> [B, H]
        b = concepts.shape[0]
        q = self.query.expand(b, -1, -1)  # [B, 1, H]
        pooled, _ = self.attn(q, concepts, concepts, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


class ConceptEncoderForSentencePairClassification(PreTrainedModel):
    """
    ConceptEncoder for sentence-pair tasks with separate encoding.
    
    Each sentence is encoded independently into concept representations,
    then compared in concept space.  This avoids the distribution mismatch
    between pretraining (single spans) and GLUE evaluation (concatenated pairs).
    
    Architecture:
        sentence_a → shared encoder → concepts_a → mean pool → z_a
        sentence_b → shared encoder → concepts_b → mean pool → z_b
        [z_a; z_b; |z_a - z_b|; z_a * z_b] → classifier → logits
    
    Supports zero-shot STS-B evaluation via cosine similarity of z_a, z_b
    (no classifier needed, no fine-tuning required).
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        self.pool_norm = nn.LayerNorm(config.hidden_size)
        self.pool_mode = getattr(config, "pool_mode", "mean")
        if self.pool_mode == "attention":
            self.attn_pool = AttentionPool(config.hidden_size)
        
        # Classifier on concatenated features: [z_a; z_b; |z_a-z_b|; z_a*z_b]
        self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _pool_concepts(self, concept_repr: torch.Tensor) -> torch.Tensor:
        """Pool the C concepts to one vector: [B, C, H] -> [B, H].

        ``pool_mode='mean'`` (default) is byte-identical to the original behaviour;
        ``pool_mode='attention'`` uses the learned-query AttentionPool so that
        information distributed across slots is recoverable.
        """
        if self.pool_mode == "attention":
            return self.pool_norm(self.attn_pool(concept_repr))
        return self.pool_norm(concept_repr.mean(dim=1))

    def forward(
        self,
        input_ids_a: Optional[torch.LongTensor] = None,
        attention_mask_a: Optional[torch.Tensor] = None,
        input_ids_b: Optional[torch.LongTensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cosine_only: bool = False,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Args:
            input_ids_a, attention_mask_a: First sentence.
            input_ids_b, attention_mask_b: Second sentence.
            labels: Task labels.
            cosine_only: If True, return cosine similarity as logits
                (for zero-shot STS-B, bypassing the classifier).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encode each sentence independently (shared weights)
        concepts_a = self.encoder(
            input_ids=input_ids_a, attention_mask=attention_mask_a, return_dict=True
        ).last_hidden_state  # [B, C, H]
        
        concepts_b = self.encoder(
            input_ids=input_ids_b, attention_mask=attention_mask_b, return_dict=True
        ).last_hidden_state  # [B, C, H]
        
        z_a = self._pool_concepts(concepts_a)  # [B, H]
        z_b = self._pool_concepts(concepts_b)  # [B, H]
        
        if cosine_only:
            cos_sim = F.cosine_similarity(z_a, z_b, dim=-1)  # [B]
            logits = cos_sim.unsqueeze(-1)  # [B, 1] for compatibility
            loss = None
            if labels is not None:
                loss = F.mse_loss(cos_sim, labels.float())
            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)
            return SequenceClassifierOutput(loss=loss, logits=logits)
        
        # Feature engineering: [z_a; z_b; |z_a-z_b|; z_a*z_b]
        diff = torch.abs(z_a - z_b)
        prod = z_a * z_b
        combined = torch.cat([z_a, z_b, diff, prod], dim=1)  # [B, 4*H]
        
        logits = self.classifier(self.classifier_dropout(combined))  # [B, num_labels]
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(loss=loss, logits=logits)


# =============================================================================
# Position-Only Decoder Variants (Pure Perceiver IO Style)
# =============================================================================

class ConceptEncoderForMaskedLMPerceiverPosOnly(ConceptEncoderForDenoisingPerceiver):
    """Legacy alias kept as a thin wrapper around the denoising perceiver."""


# =============================================================================
# Autoregressive concept-conditioned decoder (E01)
# =============================================================================
# A from-scratch AR Transformer decoder conditioned on the concept bottleneck:
# causal self-attention over target tokens + cross-attention to the C concepts as
# memory, trained with next-token cross-entropy. This is the generative counterpart
# to the parallel PerceiverDecoderStack (which models p(x|concepts) as independent
# per-position predictions and cannot generate). Reusable + config-selectable via
# ConceptEncoderConfig.decoder_type == "causal_ar".


def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute rotary cos/sin tables of shape [seq_len, head_dim] (Su et al. 2021).

    head_dim must be even. Returns (cos, sin) ready to broadcast over [B, n_heads, T, head_dim].
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)              # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)               # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def build_sliding_window_causal_mask(
    seq_len: int, window: int, device, dtype=torch.bool
) -> torch.Tensor:
    """Boolean SDPA mask [T, T] for a last-K sliding-window causal decoder (E05).

    ``True`` = token i may attend token j. Allowed iff ``i - window < j <= i`` — i.e.
    causal AND within the last ``window`` tokens (the current token plus ``window-1``
    predecessors). Out-of-window predecessors are masked, so any dependency further than
    ``window`` back must be served through the concept bottleneck instead of local context.
    Broadcasts over [B, n_heads, T, T] inside scaled_dot_product_attention.
    """
    idx = torch.arange(seq_len, device=device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)              # j <= i
    in_window = idx.unsqueeze(1) - idx.unsqueeze(0) < window   # i - j < window
    mask = causal & in_window
    return mask.to(dtype) if dtype == torch.bool else mask


def _combine_self_attn_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    batch: int,
    seq_len: int,
    device,
) -> Tuple[Optional[torch.Tensor], bool]:
    """Fold the sliding-window causal mask and the [B, T] padding mask into one SDPA mask.

    SDPA accepts a single ``attn_mask``; there is no separate key-padding arg. Both
    constraints here are key-side, so they AND as a bool mask (True = attend).

    - ``attn_mask``: [T, T] bool sliding-window causal (True = attend), or None (full causal).
    - ``key_padding_mask``: [B, T] bool (True = pad/IGNORE), or None.
    Returns ``(mask, is_causal)``:
    - neither set  -> (None, True)  -> the cheap flash is_causal kernel (full causal).
    - only window  -> (attn_mask, False) as before (E05 windowed path).
    - padding set  -> a [B, 1, T, T] bool mask = (causal-or-full) AND (not pad); is_causal=False.
    """
    if key_padding_mask is None:
        return attn_mask, attn_mask is None
    # keep: [B, T] bool, True = real key (not padded)
    keep = ~key_padding_mask.bool().view(batch, 1, 1, seq_len)  # [B,1,1,T]
    if attn_mask is not None:
        win = attn_mask.bool().view(1, 1, seq_len, seq_len)     # [1,1,T,T]
        mask = win & keep
    else:
        # Full-causal upper bound via is_causal=True would conflict with an explicit mask,
        # so materialise a causal mask and AND with the padding keep-mask.
        idx = torch.arange(seq_len, device=device)
        causal = idx.view(seq_len, 1) >= idx.view(1, seq_len)   # [T,T]
        mask = causal.view(1, 1, seq_len, seq_len) & keep
    return mask, False


def _chunked_window_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window: int, chunk_size: int = 2048,
    key_padding_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """O(N*K) memory sliding-window causal attention for long context.

    Computes causal + last-``window`` attention in blocks over the query axis. Each
    query chunk only loads its union key window ``[s-window+1, e)`` (width <= chunk+K-1),
    so the attention matrix materialised is O(chunk * (chunk+K)) — independent of N —
    and total memory is O(N*K). Numerically equivalent to the full bool-mask SDPA path
    within bf16 precision (verified). Hardware-agnostic (no Hopper/flex needed).

    q, k, v: [B, h, N, d]. ``key_padding_mask`` [B, N] (True = pad/ignore) is folded
    per chunk (only the loaded key window is masked). Returns [B, h, N, d].
    """
    B, h, N, d = q.shape
    out = torch.empty_like(q)
    kpm = key_padding_mask
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        qch = q[:, :, s:e, :]                                  # [B,h,qc,d]
        ks = max(0, s - window + 1)                            # union of query windows in [s,e)
        kch = k[:, :, ks:e, :]                                 # [B,h,kc,d]
        vch = v[:, :, ks:e, :]
        qi = torch.arange(s, e, device=q.device)[:, None]      # [qc,1]
        kj = torch.arange(ks, e, device=q.device)[None, :]     # [1,kc]
        mask = (kj <= qi) & (qi - kj < window)                 # [qc,kc] causal+last-K
        if kpm is not None:
            keep = ~kpm[:, ks:e].bool()                        # [B,kc]
            mask = mask[None, None, :, :] & keep[:, None, None, :]
        out[:, :, s:e, :] = F.scaled_dot_product_attention(
            qch, kch, vch, attn_mask=mask, is_causal=False, dropout_p=dropout_p,
        )
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x of shape [B, n_heads, T, head_dim]. cos/sin: [T, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class ConceptCausalDecoderLayer(nn.Module):
    """One AR decoder block: causal self-attn (+RoPE) → cross-attn to concepts → gated FFN.

    Shapes: h [B, T, H] (target hidden), concepts [B, C, H] (memory). Pre-norm
    residual structure mirrors the rest of the foundation. Self-attention uses
    manual q/k/v so RoPE can be applied to q,k; SDPA runs with is_causal=True.
    Cross-attention reads concepts (orderless) — no positional encoding there.
    Keeps decoding O(T*C); the only O(T^2) term is the causal self-attention over
    the *output* length, which is intrinsic to autoregression.
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_heads})"
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.use_rope = config.decoder_pos_type == "rope"
        self.attn_dropout_p = config.attention_probs_dropout_prob
        self.attn_impl = getattr(config, "decoder_attn_impl", "sdpa")
        self.attn_chunk_size = int(getattr(config, "decoder_attn_chunk_size", 2048) or 2048)
        self.context_window = getattr(config, "decoder_context_window", None)

        # --- causal self-attention (manual q/k/v for RoPE) ---
        self.pre_self_norm = build_norm(config.norm_type, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_out = nn.Linear(config.hidden_size, config.hidden_size)

        # --- cross-attention to concepts (no RoPE) ---
        self.pre_cross_norm = build_norm(config.norm_type, config.hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # --- gated FFN (SwiGLU when hidden_act="silu") ---
        self.pre_ff_norm = build_norm(config.norm_type, config.hidden_size)
        self.ffn_in = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.ffn_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = ACT2FN[config.hidden_act]

    def _self_attention(
        self, x: torch.Tensor, rope,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, h, d).transpose(1, 2)   # [B, h, T, d]
        k = self.k_proj(x).view(B, T, h, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, d).transpose(1, 2)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        # Long-context path: O(N*K) memory chunked windowed attention. Used when a
        # sliding window is set AND the chunked impl is selected; the full bool SDPA
        # mask would materialise O(N^2) and OOM past ~16K on a 3090. Numerically
        # equivalent to the SDPA-math path within bf16 precision.
        if self.attn_impl == "chunked_window" and self.context_window is not None:
            attn = _chunked_window_causal_attention(
                q, k, v, window=self.context_window,
                chunk_size=self.attn_chunk_size,
                key_padding_mask=key_padding_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
            )                                                     # [B, h, T, d]
            attn = attn.transpose(1, 2).reshape(B, T, self.hidden_size)
            return self.self_out(attn)
        # SDPA takes a single mask; fold the [B, T] padding mask (True = ignore) into the
        # attention mask. Both the sliding-window causal pattern (E05) and the padding mask
        # are key-side constraints, so we AND them as a bool mask broadcastable to
        # [B, h, T, T]. When neither is set we keep the cheap flash-friendly is_causal path.
        sdpa_attn_mask, is_causal = _combine_self_attn_mask(attn_mask, key_padding_mask, B, T, x.device)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )                                                     # [B, h, T, d]
        attn = attn.transpose(1, 2).reshape(B, T, self.hidden_size)
        return self.self_out(attn)

    def forward(
        self,
        h: torch.Tensor,
        concepts: torch.Tensor,
        rope=None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = h + self._self_attention(
            self.pre_self_norm(h), rope, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # Cross-attention keys are the C concepts (all valid — no concept-side padding), so
        # only the self-attention needs the suffix padding mask. Padded query positions'
        # cross-attn outputs are discarded by labels=-100 downstream, so masking them on the
        # query side is unnecessary (and MHA has no query-mask arg).
        cross_out, _ = self.cross_attn(
            self.pre_cross_norm(h), concepts, concepts, need_weights=False
        )
        h = h + cross_out
        ff_in, ff_gate = self.ffn_in(self.pre_ff_norm(h)).chunk(2, dim=-1)
        h = h + self.ffn_out(self.ffn_dropout(self.act_fn(ff_in) * ff_gate))
        return h


class ConceptCausalDecoderStack(nn.Module):
    """AR decoder: embed shifted target tokens → N causal layers (cross-attend to concepts) → norm.

    Token embeddings live in token_embedding_dim (asymmetry); a projection lifts them
    to hidden_size when dims differ. With decoder_pos_type="rope" no absolute position
    embedding is added (RoPE injects position inside self-attention); with "learned" an
    absolute position embedding is added. A single learned "dropout" embedding implements
    decoder-input word-dropout (posterior-collapse guard) at the embedding level.
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        self.use_rope = config.decoder_pos_type == "rope"
        token_dim = config.token_embedding_dim

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=token_dim,
            padding_idx=embedding_padding_idx(config.pad_token_id, config.eos_token_id),
        )
        self.input_projection = (
            nn.Linear(token_dim, config.hidden_size) if token_dim != config.hidden_size else None
        )
        if not self.use_rope:
            self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        else:
            self.position_embeddings = None
        # Learned "dropout" embedding substituted for word-dropped decoder-input positions.
        self.dropout_embedding = nn.Parameter(torch.zeros(config.hidden_size))

        self.layers = nn.ModuleList(
            [ConceptCausalDecoderLayer(config) for _ in range(config.decoder_num_layers)]
        )
        self.output_norm = build_norm(config.norm_type, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        self._head_dim = config.hidden_size // config.num_attention_heads
        self._rope_theta = config.rope_theta
        # E05: sliding-window causal context (None = full causal). Built lazily per
        # forward and cached by (T, device) since the boolean mask is content-independent.
        self.context_window = getattr(config, "decoder_context_window", None)
        self.attn_impl = getattr(config, "decoder_attn_impl", "sdpa")
        self.attn_chunk_size = int(getattr(config, "decoder_attn_chunk_size", 2048) or 2048)
        self.gradient_checkpointing = False
        self._window_mask_cache: dict = {}

    def embed(
        self,
        decoder_input_ids: torch.LongTensor,
        word_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        B, T = decoder_input_ids.shape
        emb = self.token_embeddings(decoder_input_ids)
        if self.input_projection is not None:
            emb = self.input_projection(emb)                       # [B, T, H]
        # Applied whenever explicitly requested (callers gate on training mode;
        # eval-time diagnostics pass the train rate to measure the matched condition).
        if word_dropout_p > 0.0:
            drop = (torch.rand(B, T, device=emb.device) < word_dropout_p).unsqueeze(-1)
            emb = torch.where(drop, self.dropout_embedding.to(emb.dtype), emb)
        elif self.training:
            # Keep the otherwise-unused parameter in the autograd graph so DDP with
            # find_unused_parameters=False doesn't error on objectives that disable
            # word-dropout (e.g. E02 prefix->suffix). Numerically a no-op.
            emb = emb + 0.0 * self.dropout_embedding.to(emb.dtype)
        if self.position_embeddings is not None:
            position_ids = torch.arange(T, device=decoder_input_ids.device).unsqueeze(0)
            emb = emb + self.position_embeddings(position_ids)
        return self.embed_dropout(emb)

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        concepts: torch.Tensor,
        word_dropout_p: float = 0.0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed(decoder_input_ids, word_dropout_p=word_dropout_p)   # [B, T, H]
        T = h.size(1)
        rope = None
        if self.use_rope:
            rope = _build_rope_cache(
                T, self._head_dim, self._rope_theta, h.device, h.dtype
            )
        # Chunked windowed attention builds its mask per-chunk internally (O(N*K)), so
        # skip the O(N^2) full mask materialisation — at long N the int64 [N,N] diff in
        # build_sliding_window_causal_mask alone is ~32 GB at N=65536.
        if self.attn_impl == "chunked_window" and self.context_window is not None:
            attn_mask = None
        else:
            attn_mask = self._sliding_window_mask(T, h.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def _dec_fwd(hh, conc, _rope, _amask, _kpm, _layer=layer):
                    return _layer(hh, conc, rope=_rope, attn_mask=_amask, key_padding_mask=_kpm)
                h = torch.utils.checkpoint.checkpoint(
                    _dec_fwd, h, concepts, rope, attn_mask, key_padding_mask,
                    use_reentrant=False,
                )
            else:
                h = layer(h, concepts, rope=rope, attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask)
        return self.output_norm(h)

    def _sliding_window_mask(self, seq_len: int, device) -> Optional[torch.Tensor]:
        """Return the [T, T] window-causal mask, or None for full causal (flash path).

        None when no window is set OR the window already covers the whole sequence
        (window >= T), so short sequences keep the cheaper is_causal kernel.
        """
        window = self.context_window
        if window is None or window >= seq_len:
            return None
        key = (seq_len, device)
        mask = self._window_mask_cache.get(key)
        if mask is None:
            mask = build_sliding_window_causal_mask(seq_len, window, device)
            self._window_mask_cache[key] = mask
        return mask


class AnchorDistillHead(nn.Module):
    """E03 de-collapse head: regenerate a frozen teacher's per-token hidden states from concepts.

    Position queries cross-attend to the C concepts (reusing PerceiverDecoderLayer), then a linear
    projects to the teacher's hidden size — an MSE target computed by the trainer. Keeps the
    bottleneck advantage O(C*N): no token self-attention. Deliberately LEAN (config.anchor_head_layers,
    default 2) so the de-collapse pressure lands on the concepts, not on an expressive head that could
    reconstruct the teacher from a low-rank (collapsed) concept set and hide the collapse.

    Shapes: concepts [B, C, H] + seq_length N -> per-token predictions [B, N, teacher_hidden].
    """

    def __init__(self, config: ConceptEncoderConfig, teacher_hidden: int):
        super().__init__()
        self.query_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layers = nn.ModuleList(
            [PerceiverDecoderLayer(config) for _ in range(config.anchor_head_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, teacher_hidden)

    def forward(self, concept_repr: torch.Tensor, seq_length: int) -> torch.Tensor:
        batch_size = concept_repr.size(0)
        position_ids = torch.arange(seq_length, device=concept_repr.device).unsqueeze(0)
        hidden = self.query_embeddings(position_ids).expand(batch_size, -1, -1)  # [B, N, H]
        for layer in self.layers:
            hidden = layer(hidden, concept_repr)                                  # cross-attn to concepts
        return self.proj(self.output_norm(hidden))                                # [B, N, teacher_hidden]


def masked_standardized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    standardize: bool = True,
) -> torch.Tensor:
    """Per-token masked MSE for the E03 anchor (single source of the loss policy).

    Shapes: pred / target [B, N, D]; target_mask [B, N] (True/1 = real token). When ``standardize``
    the target is per-token layer-normed (zero-mean/unit-var over D) before the MSE, so the teacher's
    raw scale is irrelevant and the head learns the shape (Cosmos/LDLM practice). Error is averaged
    over masked positions AND feature dims. Pure (no teacher) → unit-testable in isolation.
    """
    if standardize:
        target = F.layer_norm(target.float(), (target.size(-1),)).to(pred.dtype)
    else:
        target = target.to(pred.dtype)
    mask = target_mask.unsqueeze(-1).to(pred.dtype)            # [B, N, 1]
    sq = ((pred - target) ** 2) * mask
    denom = mask.sum().clamp(min=1) * pred.size(-1)
    return sq.sum() / denom


class ConceptEncoderForConditionalLM(PreTrainedModel):
    """Encoder → concepts → autoregressive concept-conditioned decoder.

    Reconstruction forward shapes (E01):
        input_ids       [B, N]  clean token ids (encoder sees them through attention_mask)
        attention_mask  [B, N]  1 = visible to encoder, 0 = TSDAE-deleted/pad
        labels          [B, N]  reconstruction targets (−100 at pad); next-token shifted internally
      → concepts        [B, C, H]
      → decoder_input   [B, N]  shift-right of input_ids (prepend bos)
      → logits          [B, N, V]
      → loss            scalar next-token CE on labels[:, 1:]

    Prefix/suffix forward shapes (E02):
        prefix_input_ids      [B, P]  encoder-visible prefix only
        prefix_attention_mask [B, P]
        suffix_input_ids      [B, S]  decoder target sequence
        labels                [B, S]  suffix targets (−100 at pad)
      → concepts              [B, C, H]
      → decoder_input         [B, S]  shift-right of suffix_input_ids
      → logits                [B, S, V]
      → loss                  scalar suffix next-token CE

    Selected when ConceptEncoderConfig.decoder_type == "causal_ar". Keeps the encoder
    O(C*N); the decoder is deliberately lean (decoder_num_layers < encoder layers) and
    uses decoder-input word-dropout to prevent posterior collapse.
    """

    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None,
    ):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        self.decoder = ConceptCausalDecoderStack(config)
        self.set_loss_config(loss_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # E03: lean auxiliary head built ONLY when anchor_loss is enabled, so anchor_loss=False
        # leaves the E01 state_dict byte-for-byte unchanged (old checkpoints load as before).
        # The frozen teacher that produces MSE targets lives on the trainer, not here.
        self.anchor_head = None
        if getattr(config, "anchor_loss", False):
            teacher_hidden = config.anchor_teacher_hidden or config.hidden_size
            self.anchor_head = AnchorDistillHead(config, teacher_hidden)
        self.post_init()

        if config.tie_word_embeddings and config.token_embedding_dim == config.hidden_size:
            self._tie_or_clone_weights(self.lm_head, self.decoder.token_embeddings)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        # Propagate the HF Trainer's gradient_checkpointing flag to the custom encoder
        # and decoder stacks, whose layer loops wrap each layer in torch.utils.checkpoint
        # when the flag is on (long-context activation-memory control).
        self.encoder.gradient_checkpointing = bool(enable)
        self.decoder.gradient_checkpointing = bool(enable)

    def anchor_predict(self, concept_repr: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Per-token teacher-hidden-state predictions from concepts (E03 anchor). [B,C,H]->[B,N,Ht]."""
        if self.anchor_head is None:
            raise RuntimeError(
                "anchor_predict called but config.anchor_loss is False (no anchor head built)."
            )
        return self.anchor_head(concept_repr, seq_length)

    def compute_anchor_loss(
        self,
        concept_repr: torch.Tensor,
        teacher_hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        standardize: bool = True,
    ) -> torch.Tensor:
        """E03 anchor loss: distil the (precomputed, detached) teacher per-token hidden states through
        the concept bottleneck. Pure w.r.t. the teacher (the caller supplies its states), so the loss
        is unit-testable without downloading a teacher; the model owns the predict+compare. seq_length
        is read from the targets."""
        pred = self.anchor_predict(concept_repr, teacher_hidden_states.size(1))
        return masked_standardized_mse(pred, teacher_hidden_states, target_mask, standardize=standardize)

    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        self.loss_manager = LossManager.create_for_model(
            concept_num=self.config.concept_num,
            hidden_size=self.config.hidden_size,
            loss_config=loss_config,
        )
        self._loss_config = loss_config

    def encode_concepts(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def pool_concepts(self, concept_repr: torch.Tensor) -> torch.Tensor:
        return concept_repr.mean(dim=1)

    def _shift_right(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Prepend a start token (bos if available else eos) and drop the last token."""
        start_id = self.config.bos_token_id
        if start_id is None:
            start_id = self.config.eos_token_id
        if start_id is None:
            start_id = self.config.pad_token_id or 0
        shifted = input_ids.new_full(input_ids.shape, start_id)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        return shifted

    def decode_logits(
        self,
        concept_repr: torch.Tensor,
        decoder_input_ids: torch.LongTensor,
        word_dropout_p: float = 0.0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.decoder(
            decoder_input_ids, concept_repr,
            word_dropout_p=word_dropout_p, key_padding_mask=key_padding_mask,
        )
        return self.lm_head(hidden)

    def decode_hidden(
        self,
        concept_repr: torch.Tensor,
        decoder_input_ids: torch.LongTensor,
        word_dropout_p: float = 0.0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decoder output BEFORE the lm_head — used by the chunked-CE path (F2) so the
        lm_head+CE can run in N-blocks without materialising [B,N,V] logits."""
        return self.decoder(
            decoder_input_ids, concept_repr,
            word_dropout_p=word_dropout_p, key_padding_mask=key_padding_mask,
        )

    def _chunked_teacher_forced_ce(
        self, hidden: torch.Tensor, labels: torch.LongTensor, block_size: int,
    ) -> torch.Tensor:
        """O(N*V) -> O(block*V) memory cross-entropy for long context (F2).

        Computes lm_head + CE in blocks over the N axis so the full [B,N,V] logits
        tensor is never materialised (and the fp32 CE upcast stays block-sized).
        Equivalent to _teacher_forced_ce(decode_logits(...), labels) up to floating
        point order. Used only in the training forward (encode_decode_loss) when
        config.chunked_ce_block_size > 0; ablation/eval keep the full-logits path.
        """
        B, T, H = hidden.shape
        total = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        counted = 0
        for s in range(0, T, block_size):
            e = min(s + block_size, T)
            logits_blk = self.lm_head(hidden[:, s:e, :])                 # [B, blk, V]
            lbl_blk = labels[:, s:e]                                     # [B, blk]
            ce_blk = F.cross_entropy(
                logits_blk.reshape(-1, logits_blk.size(-1)),
                lbl_blk.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total = total + ce_blk
        # mean over non-ignored positions = total_sum / count
        count = (labels != -100).sum().clamp(min=1)
        return total.to(torch.float32) / count

    @staticmethod
    def _teacher_forced_ce(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Next-token CE for logits produced from ALREADY shift-right-ed decoder inputs.

        decoder_input = [bos, x0..x_{N-2}], so logits[t] is conditioned on
        [bos, x0..x_{t-1}] and predicts labels[t] = x_t directly — NO additional
        logits/labels shift here (T5 convention). Shifting again would pair
        logits[t] with x_{t+1}, silently training a skip-one objective where the
        decoder never sees the immediately preceding token (the E01-warmup bug).
        """
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _teacher_forced_ce_early(
        logits: torch.Tensor, labels: torch.LongTensor, k: int
    ) -> torch.Tensor:
        """Next-token CE restricted to the FIRST k target positions.

        Suffix targets are left-aligned (real tokens first, pad after), so the
        first k columns are the earliest suffix tokens. Concept reliance is
        strongest there: later positions are increasingly predictable from the
        teacher-forced suffix context regardless of the concepts, which dilutes
        the all-position ablation delta. The early-position delta is therefore
        the sharper instrument for "does the decoder use the concepts?".
        """
        k = max(1, min(k, labels.size(1)))
        return F.cross_entropy(
            logits[:, :k, :].reshape(-1, logits.size(-1)),
            labels[:, :k].reshape(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _teacher_forced_ce_window(
        logits: torch.Tensor, labels: torch.LongTensor, window_k: int, beyond: bool
    ) -> torch.Tensor:
        """Next-token CE on positions BEYOND (t >= window_k) or WITHIN (t < window_k) the
        sliding window (E05 long-range memory gate).

        With a last-K windowed decoder, a position t >= window_k cannot reach tokens before
        t - window_k through local self-attention, so any dependency further back than the
        window MUST flow through the concepts. The intact-vs-ablated CE gap on beyond-window
        positions is therefore the direct test of "are concepts used as cross-window memory?".
        Within-window positions are the local-fluency control (the window still serves them).
        """
        T = labels.size(1)
        window_k = max(1, min(window_k, T))
        if beyond:
            sl = slice(window_k, T)
        else:
            sl = slice(0, window_k)
        return F.cross_entropy(
            logits[:, sl, :].reshape(-1, logits.size(-1)),
            labels[:, sl].reshape(-1),
            ignore_index=-100,
        )

    def _loss_from_logits(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.LongTensor],
        concept_repr: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        task_loss = self._teacher_forced_ce(logits, labels)
        if self.training and self.loss_manager.is_enabled:
            return self.loss_manager(task_loss=task_loss, concept_repr=concept_repr)
        return task_loss

    def encode_decode_loss(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor],
        target_input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        target_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, BaseModelOutput]:
        """Single source of the encode → (shift target) → decode → loss recipe.

        Used by BOTH forward() branches (reconstruction: ``target == encoder input``;
        prefix→suffix: ``target == suffix``) AND auxiliary-loss callers (the E03 anchor in
        PerceiverDenoiseTrainer). Centralising it keeps the decoder loss from drifting between
        call sites — the class of bug behind the E01 double-shift.

        ``target_attention_mask`` [B, T] (1 = real, 0 = pad) optionally masks padded decoder
        positions out of self-attention so real queries don't attend pad noise at long seq
        lengths (E05/2K). Labels=-100 already drops them from the loss; this keeps the forward
        clean. None falls back to the prior unmasked behaviour.

        Returns ``(loss, logits, encoder_outputs)``; ``loss`` is None when ``labels`` is None.
        """
        encoder_outputs = self.encode_concepts(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        concept_repr = encoder_outputs.last_hidden_state
        decoder_input_ids = self._shift_right(target_input_ids)
        word_dropout_p = self.config.decoder_word_dropout if self.training else 0.0
        dec_key_padding = None
        if target_attention_mask is not None:
            dec_key_padding = target_attention_mask == 0  # [B, T] True = ignore (SDPA convention)
        ce_block = int(getattr(self.config, "chunked_ce_block_size", 0) or 0)
        if ce_block > 0 and self.training and labels is not None:
            # F2: chunked lm_head + CE — never materialise [B,N,V] (the O(N*V) spike,
            # ~6 GB at N=16384, V=49152). Compute hidden, then loss in N-blocks.
            hidden = self.decode_hidden(
                concept_repr, decoder_input_ids,
                word_dropout_p=word_dropout_p, key_padding_mask=dec_key_padding,
            )
            task_loss = self._chunked_teacher_forced_ce(hidden, labels, ce_block)
            if self.loss_manager.is_enabled:
                loss = self.loss_manager(task_loss=task_loss, concept_repr=concept_repr)
            else:
                loss = task_loss
            return loss, None, encoder_outputs
        logits = self.decode_logits(
            concept_repr, decoder_input_ids,
            word_dropout_p=word_dropout_p, key_padding_mask=dec_key_padding,
        )
        loss = self._loss_from_logits(logits, labels, concept_repr)
        return loss, logits, encoder_outputs

    @torch.no_grad()
    def concept_ablation_ce(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prefix_input_ids: Optional[torch.LongTensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        early_k: int = 16,
        window_k: Optional[int] = None,
    ) -> dict:
        """Posterior-collapse diagnostic: next-token CE with intact vs ablated concepts.

        Returns {ce_intact, ce_zero, ce_shuffle, delta_zero, delta_shuffle} and, when the
        model was trained with decoder word-dropout, {ce_intact_wd, gap_clean_vs_wd}.
        A decoder that genuinely uses the concepts shows ce_zero/ce_shuffle >> ce_intact
        (large positive deltas). "zero" replaces concepts with zeros (the no-concept
        floor); "shuffle" permutes concepts across the batch (breaks instance-specific
        info while preserving concept statistics — the stronger test).

        ce_intact is measured with CLEAN decoder inputs (word_dropout=0). When training
        used word-dropout, that clean condition is out-of-distribution for the decoder, so
        ce_intact_wd re-measures intact CE under the TRAIN-matched word-dropout rate.
        gap_clean_vs_wd = ce_intact - ce_intact_wd: a large positive gap means the
        decoder is specialized to word-dropped inputs and the clean-input eval CE
        understates the model's quality (train/eval protocol mismatch, E01 diagnostic).
        """
        if prefix_input_ids is not None:
            if suffix_input_ids is None or labels is None:
                raise ValueError("prefix/suffix ablation requires suffix_input_ids and labels.")
            encoder_input_ids = prefix_input_ids
            encoder_attention_mask = prefix_attention_mask
            decoder_input_ids = self._shift_right(suffix_input_ids)
        else:
            if input_ids is None or labels is None:
                raise ValueError("reconstruction ablation requires input_ids and labels.")
            encoder_input_ids = input_ids
            encoder_attention_mask = attention_mask
            decoder_input_ids = self._shift_right(input_ids)

        # Match training's decoder padding mask: labels=-100 marks padded positions, so a
        # real-position mask is labels != -100. Keeps the ablation forward consistent with
        # the training forward (E05/2K) so the deltas aren't confounded by pad noise.
        dec_key_padding = labels == -100

        was_training = self.training
        self.eval()
        concepts = self.encode_concepts(
            input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, return_dict=True
        ).last_hidden_state
        logits_intact = self.decode_logits(concepts, decoder_input_ids, key_padding_mask=dec_key_padding)
        logits_zero = self.decode_logits(torch.zeros_like(concepts), decoder_input_ids, key_padding_mask=dec_key_padding)
        perm = torch.randperm(concepts.size(0), device=concepts.device)
        logits_shuffle = self.decode_logits(concepts[perm], decoder_input_ids, key_padding_mask=dec_key_padding)

        ce_intact = self._teacher_forced_ce(logits_intact, labels)
        ce_zero = self._teacher_forced_ce(logits_zero, labels)
        ce_shuffle = self._teacher_forced_ce(logits_shuffle, labels)

        # Early-position deltas: where concepts matter most (see _teacher_forced_ce_early).
        ce_intact_early = self._teacher_forced_ce_early(logits_intact, labels, early_k)
        ce_zero_early = self._teacher_forced_ce_early(logits_zero, labels, early_k)
        ce_shuffle_early = self._teacher_forced_ce_early(logits_shuffle, labels, early_k)

        metrics = {
            "ce_intact": ce_intact.item(),
            "ce_zero": ce_zero.item(),
            "ce_shuffle": ce_shuffle.item(),
            "delta_zero": (ce_zero - ce_intact).item(),
            "delta_shuffle": (ce_shuffle - ce_intact).item(),
            "ce_intact_early": ce_intact_early.item(),
            "delta_zero_early": (ce_zero_early - ce_intact_early).item(),
            "delta_shuffle_early": (ce_shuffle_early - ce_intact_early).item(),
        }

        # E05 long-range memory gate: split CE by sliding-window boundary. Beyond-window
        # positions (t >= window_k) cannot reach far-back tokens via local context, so a
        # large intact-vs-ablated gap there means the concepts carry cross-window memory.
        if window_k is not None and window_k < labels.size(1):
            ci_beyond = self._teacher_forced_ce_window(logits_intact, labels, window_k, beyond=True)
            cz_beyond = self._teacher_forced_ce_window(logits_zero, labels, window_k, beyond=True)
            cs_beyond = self._teacher_forced_ce_window(logits_shuffle, labels, window_k, beyond=True)
            ci_within = self._teacher_forced_ce_window(logits_intact, labels, window_k, beyond=False)
            metrics.update({
                "window_k": int(window_k),
                "ce_intact_beyond_window": ci_beyond.item(),
                "ce_intact_within_window": ci_within.item(),
                "delta_zero_beyond_window": (cz_beyond - ci_beyond).item(),
                "delta_shuffle_beyond_window": (cs_beyond - ci_beyond).item(),
            })

        train_wd = float(getattr(self.config, "decoder_word_dropout", 0.0) or 0.0)
        if train_wd > 0.0:
            ce_intact_wd = self._teacher_forced_ce(
                self.decode_logits(concepts, decoder_input_ids, word_dropout_p=train_wd,
                                   key_padding_mask=dec_key_padding),
                labels,
            )
            metrics["ce_intact_wd"] = ce_intact_wd.item()
            metrics["gap_clean_vs_wd"] = (ce_intact - ce_intact_wd).item()

        if was_training:
            self.train()
        return metrics

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prefix_input_ids: Optional[torch.LongTensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        del token_type_ids, special_tokens_mask
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if prefix_input_ids is not None:
            if suffix_input_ids is None:
                raise ValueError("prefix/suffix forward requires suffix_input_ids.")
            loss, logits, encoder_outputs = self.encode_decode_loss(
                prefix_input_ids,
                prefix_attention_mask,
                suffix_input_ids,
                labels,
                target_attention_mask=suffix_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if not return_dict:
                output = (logits,) + encoder_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        loss, logits, encoder_outputs = self.encode_decode_loss(
            input_ids,
            attention_mask,
            input_ids,
            labels,
            target_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

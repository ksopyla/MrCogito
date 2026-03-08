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
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig
from nn.loss_manager import LossManager, LossConfig

logger = logging.get_logger(__name__)


class PerceiverDecoderLayer(nn.Module):
    """Decoder block shared by denoising pretraining and ViaDecoder evaluation."""

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.pre_self_norm = nn.LayerNorm(config.hidden_size)
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
        normed_queries = self.pre_self_norm(query_states)
        self_attn_output, _ = self.self_attn(
            normed_queries,
            normed_queries,
            normed_queries,
            need_weights=False,
        )
        query_states = query_states + self_attn_output

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
        input_ids: Optional[torch.LongTensor] = None,
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
        input_ids: Optional[torch.LongTensor] = None,
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
        """Mean pooling over concepts: [B, C, H] -> [B, H]."""
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

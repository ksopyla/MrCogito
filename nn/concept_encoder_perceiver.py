"""
ConceptEncoder with Perceiver IO style decoding.

This module provides ConceptEncoder models using Perceiver IO cross-attention
for decoding concept representations back to sequence/classification outputs.

Models:
- ConceptEncoderForMaskedLMPerceiver: MLM pretraining
- ConceptEncoderForSequenceClassificationPerceiver: Sequence classification (GLUE, etc.)

Loss Management:
- Uses LossManager for clean, extensible loss handling
- Supports MLM-only, MLM + concept loss, MLM + multiple concept losses
- Supports fixed, learnable, and uncertainty-based weighting

Example:
    >>> from nn.loss_manager import LossConfig, LossManager
    >>> 
    >>> # MLM + orthogonality with learnable weights
    >>> loss_config = LossConfig(
    ...     concept_losses=["orthogonality"],
    ...     weighting_strategy="kendall_gal"
    ... )
    >>> model = ConceptEncoderForMaskedLMPerceiver(model_config, loss_config=loss_config)
    >>> 
    >>> # For inference (no concept loss)
    >>> model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("path/to/model")
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig
from nn.loss_manager import LossManager, LossConfig

logger = logging.get_logger(__name__)


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
        ...     weighting_strategy="kendall_gal"
        ... )
        >>> model = ConceptEncoderForMaskedLMPerceiver(model_config, loss_config=loss_config)
        >>> 
        >>> # For inference (no concept loss)
        >>> model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("path/to/model")
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

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
        # Decoder Queries: Position embeddings
        self.decoder_query_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length, 
            embedding_dim=config.hidden_size
        )
        
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

        # MLM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
        # Optionally tie embeddings
        if config.tie_word_embeddings:
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
        
        # A. Position Embeddings
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.decoder_query_embeddings(position_ids).expand(batch_size, -1, -1)
        
        # B. Input Embeddings (Reuse encoder's embeddings)
        # We access the embeddings directly from the encoder instance
        input_embeddings = self.encoder.token_embeddings(input_ids)
        
        # Combine: Query = Input + Position
        # This is standard Transformer input construction, but used here as the Decoder Query
        decoder_queries = input_embeddings + pos_embeddings
        
        # Norm queries before attention (Pre-LN)
        decoder_queries_norm = self.decoder_norm(decoder_queries)
        
        # Cross Attention
        # Query: Input+Pos [B, L, H]
        # Key/Value: Concepts [B, C, H]
        attn_output, attn_weights = self.decoder_cross_attn(
            query=decoder_queries_norm,
            key=concept_repr,
            value=concept_repr
        )
        
        # Residual Connection 1 (Add attention result to original queries)
        decoder_latents = decoder_queries + attn_output
        
        # Feed Forward Network with Residual 2
        # Note: We apply norm before FFN (Pre-LN style)
        decoder_output = decoder_latents + self.decoder_ffn(self.post_cross_norm(decoder_latents))
        
        # 3. Project to Vocabulary
        logits = self.lm_head(decoder_output) # [B, L, V]
        
        # 4. Compute Loss (delegated to LossManager)
        loss = None
        if labels is not None:
            # Compute MLM loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # Apply loss manager only during training (concept losses are regularization)
            # During evaluation, use only task loss for fair comparison
            if self.training:
                loss = self.loss_manager(
                    task_loss=mlm_loss,
                    concept_repr=concept_repr
                )
            else:
                loss = mlm_loss
            
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
    ConceptEncoder with Perceiver IO decoding for Sequence Classification.
    
    Uses a single learnable [CLS] query to aggregate concept information
    directly into a classification token via cross-attention.
    
    This model is for fine-tuning on classification tasks (e.g., GLUE).
    It uses only task loss (CrossEntropy/MSE/BCE) - no concept regularization.
    
    Example:
        >>> model = ConceptEncoderForSequenceClassificationPerceiver(config)
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss = outputs.loss  # Task loss only
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # === Classification Head (cls_ prefix to avoid name collision with MLM decoder) ===
        # Learnable [CLS] query for classification
        self.cls_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_query.data.normal_(mean=0.0, std=config.initializer_range)
        
        # Cross-Attention to aggregate concepts into [CLS] token
        self.cls_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        
        self.cls_norm = nn.LayerNorm(config.hidden_size)
        
        self.cls_ffn = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Final LayerNorm before classifier to stabilize training
        self.cls_final_norm = nn.LayerNorm(config.hidden_size)
        
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
        batch_size = input_ids.shape[0]
        
        # 1. Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        # 2. Decode with Single Query
        # Expand CLS query to batch: [1, 1, H] -> [B, 1, H]
        cls_hidden = self.cls_query.expand(batch_size, -1, -1)
        
        # Pre-LN before Attention (apply norm to copy, preserve residual stream)
        cls_hidden_norm = self.cls_norm(cls_hidden)
        
        # Cross Attention: aggregate concepts into [CLS] token
        attn_output, _ = self.cls_cross_attn(
            query=cls_hidden_norm,
            key=concept_repr,
            value=concept_repr
        )
        
        # Residual connection
        cls_hidden = cls_hidden + attn_output
        
        # FFN with residual (cls_ffn includes LayerNorm at the start)
        cls_hidden = cls_hidden + self.cls_ffn(cls_hidden)  # [B, 1, H]
        
        # Apply final normalization to stabilize logits/loss
        cls_hidden = self.cls_final_norm(cls_hidden)
        
        # 3. Classify
        # Squeeze sequence dim (1)
        logits = self.classifier(cls_hidden.squeeze(1))  # [B, num_labels]
        
        # 4. Compute task loss only (no concept regularization for classification)
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

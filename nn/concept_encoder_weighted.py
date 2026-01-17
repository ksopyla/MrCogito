"""
ConceptEncoder with weighted concept combination for decoding.

This module provides a simpler alternative to attention-based decoding,
using learned position-specific weights to combine concepts.

Models:
- ConceptEncoderForMaskedLMWeighted: MLM pretraining
- ConceptEncoderForSequenceClassificationWeighted: Sequence classification

Loss Management:
- Uses LossManager for clean, extensible loss handling
- Default: No concept loss (task loss only) - configure via LossConfig
- Concept losses are only applied during training, not evaluation
- Configurable via LossConfig for experimentation
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig
from nn.loss_manager import LossManager, LossConfig, check_loss_feasibility


class ConceptEncoderForMaskedLMWeighted(PreTrainedModel):
    """
    Simplified ConceptEncoder MLM using learned weights to combine concepts.
    
    This is a simpler approach than attention-based decoding, suitable for
    initial experiments to verify the concept bottleneck works.
    
    Each sequence position has learned weights to combine concepts into a
    position-specific representation, which is then projected to vocabulary.
    
    Training:
    - Default: No concept loss (task loss only) - configure via loss_config
    - For concept losses, pass `loss_config` to __init__
    - Concept losses are only applied during training, not evaluation
    
    Example:
        >>> # Default behavior (task loss only)
        >>> model = ConceptEncoderForMaskedLMWeighted(config)
        >>> 
        >>> # With concept loss (orthogonality + learnable weights)
        >>> from nn.loss_manager import LossConfig
        >>> loss_config = LossConfig(
        ...     concept_losses=["orthogonality"],
        ...     weighting_strategy="kendall_gal"
        ... )
        >>> model = ConceptEncoderForMaskedLMWeighted(config, loss_config=loss_config)
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
        # If loss_config is None, LossManager defaults to disabled (task loss only)
        # This ensures consistent behavior with perceiver_mlm
        self._setup_loss_manager(config, loss_config)
        
        # === Decoder Architecture ===
        # Learn a weight matrix for combining concepts per sequence position
        self.concept_weights = nn.Parameter(
            torch.randn(config.max_sequence_length, config.concept_num) / math.sqrt(config.concept_num)
        )
        
        # Simple MLM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Projection layer before lm_head for more capacity
        self.pre_lm_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Initialize weights
        self.post_init()
    
    def _setup_loss_manager(
        self, 
        model_config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig]
    ) -> None:
        """
        Setup the loss manager based on configuration.
        
        Args:
            model_config: Model architecture config (for feasibility check)
            loss_config: Training loss config (None = no concept loss)
        """
        # Validate loss feasibility
        if loss_config is not None and loss_config.is_enabled:
            warnings = check_loss_feasibility(
                model_config.concept_num,
                model_config.hidden_size,
                loss_config.concept_losses
            )
            for warning in warnings:
                from transformers.utils import logging
                logger = logging.get_logger(__name__)
                logger.warning(warning)
        
        # Create loss manager (handles None as disabled)
        self.loss_manager = LossManager(loss_config)
        self._loss_config = loss_config
    
    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        """
        Update loss configuration (e.g., for ablation studies mid-training).
        
        Args:
            loss_config: New loss configuration, or None to disable concept loss
        """
        self._setup_loss_manager(self.config, loss_config)
        
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
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        """
        Forward pass using weighted concept combination.
        
        The key idea: each position learns which concepts to use through
        trainable weights, avoiding complex attention mechanisms.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = input_ids.shape
        
        # Get concept representations from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        # Validate sequence length
        if seq_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}."
            )
        
        # Get position-specific weights and normalize
        position_weights = self.concept_weights[:seq_length, :].contiguous()
        position_weights = F.softmax(position_weights, dim=-1)
        
        # Combine concepts using learned weights (einsum for CUDA compatibility)
        # Formula: einsum('sc,bcd->bsd') where s=seq, c=concepts, b=batch, d=hidden
        sequence_repr = torch.einsum('sc,bcd->bsd', position_weights, concept_repr)
        
        # Apply projection before final LM head
        sequence_repr = self.pre_lm_projection(sequence_repr)
        
        # Project to vocabulary
        logits = self.lm_head(sequence_repr)  # [B, L, V]
        
        # Compute loss (delegated to LossManager)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # Apply loss manager (only during training)
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
    
    def get_position_weights_analysis(self):
        """
        Helper method to analyze which concepts are used for which positions.
        Useful for interpretability and debugging.
        """
        weights = F.softmax(self.concept_weights, dim=-1)
        return weights.detach().cpu().numpy()


class ConceptEncoderForSequenceClassificationWeighted(PreTrainedModel):
    """
    ConceptEncoder Model with a sequence classification head,
    using the weighted combination strategy.
    
    Training:
    - Default: No concept loss (classification-focused)
    - For concept loss, pass `loss_config` to __init__
    
    Example:
        >>> # Default: no concept loss
        >>> model = ConceptEncoderForSequenceClassificationWeighted(config)
        >>> 
        >>> # With concept loss
        >>> from nn.loss_manager import LossConfig
        >>> loss_config = LossConfig(concept_losses=["uniformity"])
        >>> model = ConceptEncoderForSequenceClassificationWeighted(config, loss_config=loss_config)
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(
        self, 
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # The underlying ConceptEncoder
        self.encoder = ConceptEncoder(config)
        
        # === Loss Management ===
        # Default: no concept loss for classification (different from MLM)
        if loss_config is None:
            loss_config = LossConfig.disabled()
        self._setup_loss_manager(config, loss_config)
        
        # Learned concept weights (position-specific)
        self.concept_weights = nn.Parameter(
            torch.randn(config.max_sequence_length, config.concept_num) / math.sqrt(config.concept_num)
        )
        
        # Projection and classification
        self.pre_classifier_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size), 
            nn.Linear(config.hidden_size, config.hidden_size), 
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()
    
    def _setup_loss_manager(
        self, 
        model_config: ConceptEncoderConfig,
        loss_config: LossConfig
    ) -> None:
        """Setup the loss manager based on configuration."""
        if loss_config.is_enabled:
            warnings = check_loss_feasibility(
                model_config.concept_num,
                model_config.hidden_size,
                loss_config.concept_losses
            )
            for warning in warnings:
                from transformers.utils import logging
                logger = logging.get_logger(__name__)
                logger.warning(warning)
        
        self.loss_manager = LossManager(loss_config)
        self._loss_config = loss_config
    
    def set_loss_config(self, loss_config: LossConfig) -> None:
        """Update loss configuration."""
        self._setup_loss_manager(self.config, loss_config)
    
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
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        if seq_length > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max.")

        # Apply Concept Weights
        position_weights = self.concept_weights[:seq_length, :].contiguous()
        position_weights = F.softmax(position_weights, dim=-1)
        
        # [B, L, H]
        sequence_repr = torch.einsum('sc,bcd->bsd', position_weights, concept_repr)
        
        # Projection
        sequence_repr = self.pre_classifier_projection(sequence_repr)
        
        # Pooling for Classification (mean pooling with mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_repr.size()).float()
            sum_embeddings = torch.sum(sequence_repr * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(sequence_repr, dim=1)
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss (delegated to LossManager)
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
                    task_loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    task_loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                task_loss = loss_fct(logits, labels)
            
            # Apply loss manager (only during training)
            if self.training:
                loss = self.loss_manager(
                    task_loss=task_loss,
                    concept_repr=concept_repr
                )
            else:
                loss = task_loss
        
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

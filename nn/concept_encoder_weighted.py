from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig

def compute_orthogonality_loss(concept_repr):
    """
    Encourage concept vectors to be orthogonal to each other.
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
    Returns:
        orthogonality_loss: scalar
    """
    # Normalize concepts to unit vectors
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
    
    # Compute concept similarity matrix
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
    
    # Create identity matrix (target: concepts should be orthogonal)
    batch_size, concept_num = concept_sim.shape[:2]
    eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
    eye = eye.expand(batch_size, -1, -1)
    
    # Compute loss: penalize non-diagonal elements
    off_diagonal_mask = 1.0 - eye
    orthogonality_loss = (concept_sim * off_diagonal_mask).pow(2).sum() / (batch_size * concept_num * (concept_num - 1))
    
    return orthogonality_loss

class ConceptEncoderForMaskedLMWeighted(PreTrainedModel):
    """
    Simplified ConceptEncoder MLM using learned weights to combine concepts.
    This is a much simpler approach than attention-based decoding, suitable for
    initial experiments to verify the concept bottleneck works.
    
    Each sequence position has learned weights to combine concepts into a
    position-specific representation, which is then projected to vocabulary.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # Learn a weight matrix for combining concepts per sequence position
        # Initialize with small random values to break symmetry
        self.concept_weights = nn.Parameter(
            torch.randn(config.max_sequence_length, config.concept_num) / math.sqrt(config.concept_num)
        )
        
        # Simple MLM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Optional: add a projection layer before lm_head for more capacity
        self.pre_lm_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size), # [batch_size, seq_length, concept_dim]
            nn.Linear(config.hidden_size, config.hidden_size), # [batch_size, seq_length, concept_dim] -> [batch_size, seq_length, concept_dim]
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Initialize weights
        self.post_init()
        
    def _init_weights(self, module):
        """Initialize the weights"""
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

        # Get the concept representations (batch_size, concept_num, concept_dim)
        concept_repr = encoder_outputs.last_hidden_state  # [batch_size, concept_num, concept_dim]
        
        # CRITICAL FIX: Ensure seq_length doesn't exceed max_sequence_length to prevent out-of-bounds access
        if seq_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_sequence_length {self.config.max_sequence_length}. "
                f"This should be prevented by the data preprocessing."
            )
        
        # Get position-specific weights and normalize them
        # Use .contiguous() to ensure memory layout is correct for distributed training
        position_weights = self.concept_weights[:seq_length, :].contiguous()  # [seq_length, concept_num]
        position_weights = F.softmax(position_weights, dim=-1)  # Normalize over concepts
        
        #Combine concepts using learned weights: [Batch_size, seq_length, concept_dim] = broadcast:[seq_length, concept_num] x [batch_size, concept_num, concept_dim]
        # CRITICAL FIX for RTX 5090 / CUDA 12.8 compatibility:
        # Use einsum instead of bmm + repeat to avoid CUDA 12.8 compiler bug
        # The CUDA 12.8 compiler has a known miscompilation issue on SM120 (RTX 5090)
        # that causes illegal memory access errors. This is fixed in CUDA 12.9.1.
        # einsum with implicit broadcasting avoids the buggy code path.
        # einsum is more explicit and handles broadcasting safely
        # We broadcast position_weights to batch dimension implicitly
        # Formula: einsum('sc,bcd->bsd') where:
        #   s = seq_length, c = concept_num, b = batch_size, d = hidden_dim
        # This implicitly broadcasts position_weights across the batch dimension
        sequence_repr = torch.einsum('sc,bcd->bsd', position_weights, concept_repr)
        
        # OLD METHOD (triggers CUDA 12.8 bug on RTX 5090):
        # position_weights_expanded = position_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        # sequence_repr = torch.bmm(position_weights_expanded, concept_repr)
        
        # Optional: apply projection before final LM head
        sequence_repr = self.pre_lm_projection(sequence_repr) # [batch_size, seq_length, concept_dim]
        
        # Project to vocabulary
        logits = self.lm_head(sequence_repr)  # [batch_size, seq_length, vocab_size]
        
        # Compute MLM loss if labels provided
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # Optional: Add orthogonality loss to encourage diverse concepts
            if self.training: # removed hasattr check as we import function directly
                ortho_loss = compute_orthogonality_loss(concept_repr)
                masked_lm_loss = masked_lm_loss + 0.01 * ortho_loss
        
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return MaskedLMOutput(
            loss=masked_lm_loss,
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
    ConceptEncoder Model with a sequence classification head on top,
    using the weighted combination strategy from ConceptEncoderForMaskedLMWeighted.
    
    Args:
        config (ConceptEncoderConfig): Model configuration defining hidden sizes, embeddings, etc.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # The underlying ConceptEncoder
        self.encoder = ConceptEncoder(config)
        
        # Learned concept weights (position-specific)
        # We initialize these with the same shape as the MLM model to allow loading weights
        self.concept_weights = nn.Parameter(
            torch.randn(config.max_sequence_length, config.concept_num) / math.sqrt(config.concept_num)
        )
        
        # Pooling / Projection
        # In MLMWeighted, we have: sequence_repr = pre_lm_projection(weighted_sum)
        # Here we want a single vector for classification.
        # Strategy:
        # 1. Compute weighted sum -> [batch, seq_len, concept_dim]
        # 2. Pool across sequence length (Mean/Max/CLS?)
        # 3. Classify
        
        # To reuse weights from MLMWeighted, we should probably include the pre_lm_projection if possible,
        # but that maps to concept_dim. 
        
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
    
    def _init_weights(self, module):
        """Initialize the weights"""
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
        
        # [batch_size, concept_num, concept_dim]
        concept_repr = encoder_outputs.last_hidden_state
        
        if seq_length > self.config.max_sequence_length:
             raise ValueError(f"Sequence length {seq_length} exceeds max.")

        # 1. Apply Concept Weights (same as MLM Weighted)
        position_weights = self.concept_weights[:seq_length, :].contiguous() # [seq_len, concept_num]
        position_weights = F.softmax(position_weights, dim=-1)
        
        # [batch_size, seq_len, concept_dim]
        sequence_repr = torch.einsum('sc,bcd->bsd', position_weights, concept_repr)
        
        # 2. Projection (matches pre_lm_projection structure)
        sequence_repr = self.pre_classifier_projection(sequence_repr)
        
        # 3. Pooling for Classification
        # Now we have a sequence of representations. For classification, we usually need one vector.
        # Options: Mean pooling over sequence, or use the first token (CLS-like).
        # Since we don't have a dedicated CLS token mechanism in the weighted sum, Mean Pooling is safe.
        # Mask out padding tokens!
        
        if attention_mask is not None:
            # attention_mask is [batch, seq_len] (1 for keep, 0 for pad)
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_repr.size()).float()
            sum_embeddings = torch.sum(sequence_repr * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(sequence_repr, dim=1)
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
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


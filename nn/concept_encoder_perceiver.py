from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

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

class ConceptEncoderForMaskedLMPerceiver(PreTrainedModel):
    """
    ConceptEncoder with Perceiver IO style decoding.
    
    Improved Version:
    1. Supports Input Embeddings as Decoder Queries (better gradient flow).
    2. Includes Orthogonality Loss (prevents concept collapse).
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # Decoder Queries
        # We keep the learned position embeddings as a base or fallback
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
        
        # Learnable Loss Weights (Kendall & Gal, CVPR 2018)
        # parameter 0: MLM Loss, parameter 1: Orthogonality Loss
        # initialized to 0.0 (which corresponds to weight = 1.0 = exp(-0.0))
        self.loss_log_vars = nn.Parameter(torch.zeros(2))
        
        # Initialize weights
        self.post_init()
        
        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.encoder.token_embeddings)

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
        concept_repr = encoder_outputs.last_hidden_state # [B, C, H]
        
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
        
        # 4. Compute Loss
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # ADDED: Orthogonality Loss with Uncertainty Weighting
            ortho_loss = compute_orthogonality_loss(concept_repr)
            
            # Retrieve learned variances
            # precision = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
            # Loss = precision * task_loss + log_var
            # We add 0.5 * log_var to the loss (standard derivation)
            
            w_mlm = torch.exp(-self.loss_log_vars[0])
            w_ortho = torch.exp(-self.loss_log_vars[1])
            
            # Combined loss
            # Note: we multiply by 0.5 as per Kendall & Gal formulation, 
            # but simple weighting works too. Sticking to the paper:
            # L = (1/2sigma^2) * L_task + log(sigma)
            # log(sigma) = 0.5 * log_var
            
            masked_lm_loss = (0.5 * w_mlm * masked_lm_loss) + (0.5 * self.loss_log_vars[0]) + \
                             (0.5 * w_ortho * ortho_loss) + (0.5 * self.loss_log_vars[1])
            
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ConceptEncoderForSequenceClassificationPerceiver(PreTrainedModel):
    """
    ConceptEncoder with Perceiver IO decoding for Sequence Classification.
    
    Uses a single learnable [CLS] query to aggregate concept information
    directly into a classification token, avoiding mean pooling.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # Learnable [CLS] query for classification
        # Shape: [1, 1, hidden_size]
        self.cls_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_query.data.normal_(mean=0.0, std=config.initializer_range)
        
        # Cross-Attention (same arch as MLM decoder, but 1 query)
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        
        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Learnable Loss Weights (Kendall & Gal)
        # param 0: Task Loss, param 1: Ortho Loss
        self.loss_log_vars = nn.Parameter(torch.zeros(2))

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
        batch_size = input_ids.shape[0]
        
        # 1. Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state # [B, C, H]
        
        # 2. Decode with Single Query
        # Expand CLS query to batch: [1, 1, H] -> [B, 1, H]
        decoder_queries = self.cls_query.expand(batch_size, -1, -1)
        
        # Pre-LN before Attention (Fixed: Apply norm to copy, preserve residual stream)
        decoder_queries_norm = self.decoder_norm(decoder_queries)
        
        # Cross Attn
        attn_output, _ = self.decoder_cross_attn(
            query=decoder_queries_norm,
            key=concept_repr,
            value=concept_repr
        )
        
        # Residual (Add to original queries)
        decoder_queries = decoder_queries + attn_output
        
        # FFN (Note: decoder_ffn already includes a LayerNorm at the start)
        decoder_output = decoder_queries + self.decoder_ffn(decoder_queries) # [B, 1, H]
        
        # 3. Classify
        # Squeeze sequence dim (1)
        logits = self.classifier(decoder_output.squeeze(1)) # [B, num_labels]
        
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
            
            # ADDED: Orthogonality Loss with Uncertainty Weighting
            if self.training:
                ortho_loss = compute_orthogonality_loss(concept_repr)
                
                w_task = torch.exp(-self.loss_log_vars[0])
                w_ortho = torch.exp(-self.loss_log_vars[1])
                
                # Kendall & Gal weighting
                loss = (0.5 * w_task * loss) + (0.5 * self.loss_log_vars[0]) + \
                       (0.5 * w_ortho * ortho_loss) + (0.5 * self.loss_log_vars[1])
                
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

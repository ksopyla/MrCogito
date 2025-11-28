from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig

class ConceptEncoderForMaskedLMPerceiver(PreTrainedModel):
    """
    ConceptEncoder with Perceiver IO style decoding.
    
    Instead of using static weights to combine concepts (like in Weighted model),
    this model uses a Cross-Attention mechanism where Position Embeddings 'query'
    the Concept representations to reconstruct the sequence.
    
    This allows for Dynamic Routing: the model can choose which concepts to use
    at each position based on the concept content, not just fixed position-concept pairs.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # Learnable queries for the decoder (representing positions)
        # We use an embedding layer to retrieve queries for the specific sequence length
        self.decoder_query_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length, 
            embedding_dim=config.hidden_size
        )
        
        # Cross-Attention: Query=Position, Key=Concepts, Value=Concepts
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        
        # Decoder Layer Norm (Pre-LN style for the decoder block)
        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        
        # Optional: FFN after attention (Standard Transformer Decoder Block)
        # Perceiver IO uses: CrossAttn -> MLP.
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
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
        
        # Generate queries for the current sequence length
        # [1, seq_len] -> [1, seq_len, H] -> [B, seq_len, H]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        decoder_queries = self.decoder_query_embeddings(position_ids).expand(batch_size, -1, -1)
        
        # Norm queries (Pre-LN) - helpful for training stability
        decoder_queries = self.decoder_norm(decoder_queries)
        
        # Cross Attention
        # Query: Position Embeddings [B, L, H]
        # Key/Value: Concept Representations [B, C, H]
        # No mask needed for keys (concepts are fully visible)
        attn_output, attn_weights = self.decoder_cross_attn(
            query=decoder_queries,
            key=concept_repr,
            value=concept_repr
        )
        
        # Residual connection (if we had an input to residual add to... but here queries are "new")
        # Usually Perceiver Decoder is: Output = CrossAttn(Query, Latents) + Query (Residual)
        # Adding residual to Position Embeddings preserves position information
        decoder_queries = decoder_queries + attn_output
        
        # Feed Forward Network
        decoder_output = decoder_queries + self.decoder_ffn(decoder_queries)
        
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
        self.cls_query = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
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
        
        self.post_init()
        
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
        decoder_queries = self.decoder_norm(decoder_queries)
        
        # Cross Attn
        attn_output, _ = self.decoder_cross_attn(
            query=decoder_queries,
            key=concept_repr,
            value=concept_repr
        )
        
        # Residual & FFN
        decoder_queries = decoder_queries + attn_output
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
                
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

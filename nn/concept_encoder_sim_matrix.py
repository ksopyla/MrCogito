from typing import Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig

class ConceptEncoderWithSimMatrixForMaskedLM(PreTrainedModel):
    """
    ConceptEncoder Model with a masked language modeling head on top (for masked language modeling).
    This model uses a encoded concepts to predict the [masked] tokens in the sequence.

    Args:
        config (ConceptEncoderConfig): Model configuration defining hidden sizes, embeddings, etc.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ConceptEncoder(config)
        
        # Concept->Vocab projection
        self.concept_vocab_projection = nn.Linear(
            config.hidden_size, # concept_dim
            config.vocab_size,
            bias=False
        )
        
        # Token-level LM head
        self.lm_token_head = nn.Linear(
            config.hidden_size, # token_embedding_dim
            config.vocab_size,
            bias=False
        )
        
        # Dynamic gating mechanism
        # Improved gating mechanism with more capacity
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),  # Takes concatenated input
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize weights
        self.post_init()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, special_tokens_mask=None, labels=None):
        # Encoder forward
        encoder_out = self.encoder(input_ids, attention_mask)
        concept_repr = encoder_out.last_hidden_state  # [Batch_size, concept_num, concept_dim]
        

        # Get token embeddings with position information
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_emb = self.encoder.token_embeddings(input_ids) + self.encoder.token_position_embeddings(position_ids) # [Batch_size, seq_length, hidden_size=token_embedding_dim]
        
        # Compute similarity with learnable temperature
        similarity = torch.einsum("bsh,bch->bsc", token_emb, concept_repr)
        similarity = similarity * self.temperature  # Learnable scaling

        # Dynamic sparsity based on attention mask
        if attention_mask is not None:
            # Mask out padding tokens in similarity computation
            similarity = similarity.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        
        # Take only top-k similarity values (routing mechanism)
        # topk_val, _ = similarity.topk(k=4, dim=-1)
        # mask = similarity >= topk_val[..., -1].unsqueeze(-1)
        # similarity = similarity.masked_fill(~mask, 0)
        
        # Project to vocab space, question is concept_vocab_projection trained? why use its weights?
        concept_logits = torch.einsum("bsc,cv->bsv", similarity, 
                                   self.concept_vocab_projection.weight)
        
        # Gated combination with token logits
        token_logits = self.lm_token_head(token_emb)

        # gated combination current token representations with average concept representations
        gate_input = torch.cat([token_emb, torch.mean(concept_repr, dim=1, keepdim=True).expand(-1, seq_length, -1)], dim=-1)
        gate = self.gate(gate_input)

        
        logits = gate * concept_logits + (1 - gate) * token_logits
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_out.hidden_states,
            attentions=encoder_out.attentions
        )


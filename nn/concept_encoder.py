from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ConceptEncoderConfig:
    """Configuration class for ConceptEncoder.
    
    This configuration class controls all the parameters needed for the ConceptEncoder model.
    It defines the architecture (number of layers, dimensions, etc.) and training parameters
    (dropout rates, initialization ranges, etc.).
    
    Args:
        vocab_size (int): Size of the token vocabulary.
        concept_size (int): Number of concept tokens to learn.
        hidden_size (int): Dimension of hidden layers and embeddings.
        num_hidden_layers (int): Number of transformer layers in the encoder.
        num_attention_heads (int): Number of attention heads in each layer.
        intermediate_size (int): Dimension of the feedforward network in each layer.
        hidden_act (str): Activation function for the hidden layers ("gelu", "relu", etc.).
        hidden_dropout_prob (float): Dropout probability for all fully connected layers.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        max_position_embeddings (int): Maximum sequence length supported by the model.
        type_vocab_size (int): Size of the token type vocabulary.
        initializer_range (float): Standard deviation for initializing model weights.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        concept_size: int = 128,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.concept_size = concept_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

class ConceptEncoderLayer(nn.Module):
    """A single layer of the concept encoder.
    
    This layer implements the core computation of the concept encoder, consisting of:
    1. Cross-attention between concepts and input tokens
    2. Self-attention between concepts
    3. Feed-forward network with gating mechanism
    
    The layer uses Pre-LN (Layer Normalization) architecture for better training stability.
    
    Args:
        config (ConceptEncoderConfig): Configuration object defining the layer parameters.
    """
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        # Cross attention between concepts and tokens
        self.concept_token_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        # Self attention for concepts
        self.concept_self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        
        # Pre-LN normalization layers
        self.pre_cross_attn_norm = nn.LayerNorm(config.hidden_size)
        self.pre_self_attn_norm = nn.LayerNorm(config.hidden_size)
        self.pre_ff_norm = nn.LayerNorm(config.hidden_size)
        
        # Feed Forward Network with Wi and Wo matrices and gating mechanism
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2)  # *2 for gating
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.wi_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = nn.GELU()

    def forward(
        self,
        concept_representations: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process input through the encoder layer.
        
        Args:
            concept_representations: Tensor of shape (batch_size, concept_length, hidden_size)
                Current concept representations to be updated.
            token_embeddings: Tensor of shape (batch_size, sequence_length, hidden_size)
                Token embeddings to attend to.
            attention_mask: Optional tensor of shape (batch_size, 1, sequence_length)
                Mask to avoid attending to padding tokens. Values should be 0 or 1.
        
        Returns:
            torch.Tensor: Updated concept representations of shape (batch_size, concept_length, hidden_size)
        """
        # Cross Attention between concept and token embeddings
        # Pre-LN
        normed_concepts = self.pre_cross_attn_norm(concept_representations)
        concept_token_attn_output, _ = self.concept_token_attn(
            normed_concepts, token_embeddings, token_embeddings, 
            attn_mask=attention_mask 
        )
        concept_representations = concept_representations + concept_token_attn_output

        # Self Attention on concept representations
        # Pre-LN
        normed_concepts = self.pre_self_attn_norm(concept_representations)
        concept_self_attn_output, _ = self.concept_self_attn(
            normed_concepts, normed_concepts, normed_concepts,
            attn_mask=None  # No mask needed for concept self-attention
        )
        concept_representations = concept_representations + concept_self_attn_output

        # Feed Forward Network with gating mechanism
        # Pre-LN
        normed_concepts = self.pre_ff_norm(concept_representations)
        ff_input, ff_gate = self.Wi(normed_concepts).chunk(2, dim=-1)
        ff_output = self.Wo(self.wi_dropout(self.act_fn(ff_input) * ff_gate))
        concept_representations = concept_representations + ff_output

        return concept_representations

class ConceptEncoder(nn.Module, ModuleUtilsMixin):
    """Concept Encoder model.
    
    This model learns concept representations by attending to input token sequences.
    It uses a stack of transformer layers with both cross-attention (between concepts
    and tokens) and self-attention (between concepts) mechanisms.
    
    The model follows a Pre-LN architecture for better training stability.
    
    Args:
        config (ConceptEncoderConfig): Configuration object defining the model architecture.
    """
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.concept_embeddings = nn.Embedding(config.concept_size, config.hidden_size)

        self.layers = nn.ModuleList([ConceptEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights.
        
        Applies the initialization strategy from Hugging Face Transformers:
        - Linear layers: truncated normal initialization for weights, zeros for biases
        - Embedding layers: truncated normal initialization
        - LayerNorm: ones for weights, zeros for biases
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Process input tokens through the encoder.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
                Indices of input sequence tokens in the vocabulary.
            attention_mask: Optional tensor of shape (batch_size, sequence_length)
                Mask to avoid attending to padding tokens. 1 for tokens to attend to, 0 for tokens to ignore.
        
        Returns:
            torch.Tensor: Final concept representations of shape (batch_size, concept_size, hidden_size)
        """
        batch_size, seq_length = input_ids.size(0), input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, sequence_length)

        token_embeddings = self.token_embeddings(input_ids) + self.token_position_embeddings(position_ids)
        token_embeddings = self.dropout(token_embeddings)

        if attention_mask is not None:
            # Convert from 2D mask (batch_size, seq_length) to 3D float-based mask
            attention_mask = attention_mask.unsqueeze(1).to(dtype=token_embeddings.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Initialize concept embeddings
        concept_representations = self.concept_embeddings(
            torch.arange(self.config.concept_size, device=input_ids.device)
        ).unsqueeze(0)  # shape (1, concept_size, hidden_size)
        concept_representations = concept_representations.expand(batch_size, -1, -1)
        # shape (batch_size, concept_size, hidden_size)

        # Now that concept_representations is created, expand attention_mask to match
        if attention_mask is not None:
            # [batch_size, 1, seq_length] -> [batch_size, concept_size, seq_length]
            attention_mask = attention_mask.expand(
                batch_size, concept_representations.size(1), seq_length
            )

        # Pass through each layer
        for layer in self.layers:
            concept_representations = layer(concept_representations, token_embeddings, attention_mask)

        concept_representations = self.output_layer_norm(concept_representations)
        return concept_representations

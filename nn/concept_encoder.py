from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

logger = logging.get_logger(__name__)

class ConceptEncoderConfig(PretrainedConfig):
    """
    Configuration class for ConceptEncoder model architecture.
    
    Inherits from PretrainedConfig for seamless integration with the
    HuggingFace Transformers library (from_pretrained, save_pretrained).

    This configuration class controls ONLY model architecture parameters.
    Training-specific parameters (like loss configuration) should be passed
    separately via ConceptLossConfig to follow Single Responsibility Principle.

    Args:
        vocab_size (int): Size of the token vocabulary.
        concept_num (int): Number of concept tokens to learn.
        hidden_size (int): Dimension of hidden layers, concept embeddings, and attention.
        token_embedding_dim (int or None): Dimension of token embeddings. When smaller than
            hidden_size, a projection layer bridges the gap (Dimension Inversion).
            None defaults to hidden_size for backward compatibility with existing checkpoints.
        num_hidden_layers (int): Number of transformer layers in the encoder.
        num_attention_heads (int): Number of attention heads in each layer.
        intermediate_size (int): Dimension of the feedforward network in each layer.
        hidden_act (str): Activation function for the hidden layers.
        hidden_dropout_prob (float): Dropout probability for fully connected layers.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        max_sequence_length (int): Maximum sequence length supported by the model.
        concept_position_type (str): Type of positional encoding for concept embeddings.
            "none" = no position (current default, concepts are orderless).
            "sinusoidal" = fixed sinusoidal positions (no extra params).
            "learned" = learned position embeddings (extra params).
        type_vocab_size (int): Size of the token type vocabulary.
        initializer_range (float): Standard deviation for initializing model weights.
        is_decoder (bool): Whether the model acts as a decoder. Defaults to False.
        tie_word_embeddings (bool): Whether to tie input/output embeddings. Defaults to True.
    
    Note:
        For training with concept regularization losses, use ConceptLossConfig
        from nn.concept_losses module. This separation follows SOLID principles:
        - Model config = what the model IS (architecture)
        - Loss config = how the model is TRAINED (behavior)
    """

    model_type = "concept_encoder"

    def __init__(
        self,
        vocab_size: int = 30522,
        concept_num: int = 128,
        hidden_size: int = 512,
        token_embedding_dim: Optional[int] = None,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_act: str = "gelu",
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        cls_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_sequence_length: int = 2048,
        concept_position_type: str = "none",
        type_vocab_size: int = 2,
        initializer_range: float = 0.1,
        is_decoder: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.concept_num = concept_num
        self.hidden_size = hidden_size
        # Dimension Inversion: token_embedding_dim can be smaller than hidden_size.
        # None defaults to hidden_size for backward compatibility with existing checkpoints.
        self.token_embedding_dim = token_embedding_dim if token_embedding_dim is not None else hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.concept_position_type = concept_position_type
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.is_decoder = is_decoder
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.use_bixt = kwargs.pop("use_bixt", False)

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
        self.act_fn = nn.GELU() # TODO: might need to try other activation functions


    def forward(
        self,
        concept_representations: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Process input through the encoder layer.
        
        Args:
            concept_representations: Tensor of shape (batch_size, concept_length, hidden_size=concept_dim)
                Current concept representations to be updated.
            token_embeddings: Tensor of shape (batch_size, sequence_length, hidden_size=token_embedding_dim)
                Token embeddings to attend to.
            attention_mask: Optional tensor of shape (batch_size, 1, sequence_length)
                Mask to avoid attending to padding tokens. Values should be 0 or 1.
        
        Returns:
            torch.Tensor: Updated concept representations of shape (batch_size, concept_length, hidden_size=concept_dim)
        """
        
        # Layer Normalization - concept normalization
        normed_concepts = self.pre_cross_attn_norm(concept_representations)
        # Cross Attention between concept and token embeddings, 
        # Queries: concepts [batch_size, concept_num, concept_dim]
        # Keys: token embeddings [batch_size, sequence_length, token_embedding_dim]
        # Values: token embeddings [batch_size, sequence_length, token_embedding_dim]
        # need_weights=False enables SDPA/Flash Attention fast path on PyTorch 2.x
        # Without it, attention weights are computed even if discarded (_), disabling the fast path

        concept_token_attn_output, _ = self.concept_token_attn(
            normed_concepts, token_embeddings, token_embeddings, 
            key_padding_mask=attention_mask,
            need_weights=False
        )

        # Add residual connection, add the additional knowledge from the concept token similarities to original concept representations, (how to fuse such information?, norm could act as a fuse operation, so maybe we could also use other operations )
        concept_representations = concept_representations + concept_token_attn_output

        
        # Pre-LN, norm operation could be view as fusing the knowledge
        normed_concepts = self.pre_self_attn_norm(concept_representations)

        # Self Attention on concept representations, Q, K, V = concept_representations
        # Queries: concepts [batch_size, concept_num, concept_dim]
        # Keys: concepts [batch_size, concept_num, concept_dim]
        # Values: concepts [batch_size, concept_num, concept_dim]
        # need_weights=False enables SDPA/Flash Attention fast path on PyTorch 2.x
        concept_self_attn_output, _ = self.concept_self_attn(
            normed_concepts, normed_concepts, normed_concepts,
            attn_mask=None,  # No mask needed for concept self-attention
            need_weights=False
        )

        # Add residual connection between concepts after concept self attention
        concept_representations = concept_representations + concept_self_attn_output

        # Feed Forward Network with gating mechanism
        # Layer Normalization - concept normalization
        normed_concepts = self.pre_ff_norm(concept_representations)
        ff_input, ff_gate = self.Wi(normed_concepts).chunk(2, dim=-1)
        ff_output = self.Wo(self.wi_dropout(self.act_fn(ff_input) * ff_gate))
        concept_representations = concept_representations + ff_output

        return concept_representations


class BiConceptEncoderLayer(nn.Module):
    """BiXT-style bidirectional cross-attention encoder layer.
    
    Extends ConceptEncoderLayer with a reverse cross-attention that updates
    token representations from concepts.  This lets tokens become contextualised
    across layers WITHOUT O(N^2) token self-attention.
    
    Per-layer complexity: O(C*N) + O(N*C) + O(C^2) = O(C*N), linear in N.
    
    Reference: Hiller, Ehinger & Drummond, "BiXT: Perceiving Longer Sequences
    With Bi-Directional Cross-Attention Transformers", NeurIPS 2024.
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()

        H = config.hidden_size

        # --- concepts ← tokens (same as standard layer) ---
        self.concept_token_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob, batch_first=True,
        )
        self.pre_cross_attn_norm = nn.LayerNorm(H)

        # --- tokens ← concepts (BiXT reverse path, O(N*C)) ---
        self.token_concept_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob, batch_first=True,
        )
        self.pre_reverse_cross_norm = nn.LayerNorm(H)

        # --- concept self-attention ---
        self.concept_self_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob, batch_first=True,
        )
        self.pre_self_attn_norm = nn.LayerNorm(H)

        # --- concept FFN (gated) ---
        self.pre_ff_norm = nn.LayerNorm(H)
        self.Wi = nn.Linear(H, config.intermediate_size * 2)
        self.Wo = nn.Linear(config.intermediate_size, H)
        self.wi_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = nn.GELU()

    def forward(
        self,
        concept_representations: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (concepts, tokens) -- both updated.
        """
        # 1. concepts ← cross-attn(Q=concepts, KV=tokens)   O(C*N)
        normed_c = self.pre_cross_attn_norm(concept_representations)
        c_out, _ = self.concept_token_attn(
            normed_c, token_embeddings, token_embeddings,
            key_padding_mask=attention_mask, need_weights=False,
        )
        concept_representations = concept_representations + c_out

        # 2. tokens ← cross-attn(Q=tokens, KV=concepts)    O(N*C)
        normed_t = self.pre_reverse_cross_norm(token_embeddings)
        t_out, _ = self.token_concept_attn(
            normed_t, concept_representations, concept_representations,
            need_weights=False,
        )
        token_embeddings = token_embeddings + t_out

        # 3. concept self-attention   O(C^2)
        normed_c = self.pre_self_attn_norm(concept_representations)
        sa_out, _ = self.concept_self_attn(
            normed_c, normed_c, normed_c,
            attn_mask=None, need_weights=False,
        )
        concept_representations = concept_representations + sa_out

        # 4. concept FFN
        normed_c = self.pre_ff_norm(concept_representations)
        ff_in, ff_gate = self.Wi(normed_c).chunk(2, dim=-1)
        ff_out = self.Wo(self.wi_dropout(self.act_fn(ff_in) * ff_gate))
        concept_representations = concept_representations + ff_out

        return concept_representations, token_embeddings

class ConceptEncoder(PreTrainedModel):
    """Concept Encoder model.
    
    This model learns concept representations by attending to input token sequences.
    It uses a stack of transformer layers with both cross-attention (between concepts
    and tokens) and self-attention (between concepts) mechanisms.
    
    The model follows a Pre-LN architecture for better training stability.
    
    Args:
        config (ConceptEncoderConfig): Configuration object defining the model architecture.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config

        token_dim = config.token_embedding_dim  # May differ from hidden_size (Dimension Inversion)

        # Token embeddings [vocab_size, token_embedding_dim]
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=token_dim, padding_idx=config.pad_token_id
        )
        # Token position embeddings [max_sequence_length, token_embedding_dim]
        self.token_position_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length, embedding_dim=token_dim
        )
        
        # Dimension Inversion: project token embeddings to hidden_size when dims differ.
        # When token_embedding_dim < hidden_size, tokens are cheap (small vocab memory)
        # and concepts are rich (large hidden_size for attention and FFN).
        if token_dim != config.hidden_size:
            self.token_projection = nn.Linear(token_dim, config.hidden_size)
        else:
            self.token_projection = None
        
        # Concept embeddings [concept_num, hidden_size=concept_dim]
        # Concepts always live in hidden_size space (the "fat" dimension)
        self.concept_embeddings = nn.Embedding(
            num_embeddings=config.concept_num, embedding_dim=config.hidden_size
        )
        
        # Concept position encoding (optional, default "none" for backward compat)
        # "sinusoidal": fixed sinusoidal positions, no extra trainable params
        # "learned": trainable position embeddings for each concept slot
        # "none": concepts are orderless (original design)
        if config.concept_position_type == "sinusoidal":
            # Register as buffer (not a parameter) -- no gradient, moves with model device
            sinusoidal_emb = self._create_sinusoidal_embeddings(config.concept_num, config.hidden_size)
            self.register_buffer("concept_position_emb", sinusoidal_emb)
        elif config.concept_position_type == "learned":
            self.concept_position_emb = nn.Embedding(
                num_embeddings=config.concept_num, embedding_dim=config.hidden_size
            )
        # For "none", no concept_position_emb attribute is created

        # Concept encoder layers [num_hidden_layers]
        layer_cls = BiConceptEncoderLayer if getattr(config, "use_bixt", False) else ConceptEncoderLayer
        self.layers = nn.ModuleList([layer_cls(config) for _ in range(config.num_hidden_layers)])
        self._use_bixt = getattr(config, "use_bixt", False)
        # Dropout [hidden_dropout_prob]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output layer normalization [hidden_size=concept_dim]
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.post_init()
    
    @staticmethod
    def _create_sinusoidal_embeddings(num_positions: int, dim: int) -> torch.Tensor:
        """Create fixed sinusoidal position embeddings (Vaswani et al., 2017).
        
        These provide a fixed ordering signal to concept slots without adding
        trainable parameters. Each position gets a unique pattern of sin/cos values.
        
        Args:
            num_positions: Number of positions (concept_num)
            dim: Embedding dimension (hidden_size)
            
        Returns:
            Tensor of shape [num_positions, dim]
        """
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        embeddings = torch.zeros(num_positions, dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings

    def _init_weights(self, module):
        """
        Override _init_weights to use custom initialization for embeddings.
        
        Initialize embeddings with different variances:
        - Token and position embeddings: Normal(0, initializer_range)
        - Concept embeddings: Normal(0, 2 * initializer_range) - higher variance for diversity, capped at 1.0
        - Linear and LayerNorm: Use PyTorch defaults
        
        The higher variance for concept embeddings encourages initial diversity while staying
        within reasonable bounds to avoid gradient instability.
        """
        if module is self.concept_embeddings:
            # Concept embeddings get 2x variance for increased initial diversity (capped at 1.0)
            concept_std = min(2.0 * self.config.initializer_range, 1.0)
            module.weight.data.normal_(mean=0.0, std=concept_std)
        
        elif isinstance(module, nn.Embedding):
            # Token and position embeddings use standard initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # Ensure padding embeddings are zeros
                module.weight.data[module.padding_idx].zero_()
        
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Args:
            input_ids (torch.LongTensor): [batch_size, seq_length].
            attention_mask (Optional[torch.FloatTensor]): [batch_size, seq_length], 1=keep, 0=ignore.
            output_attentions (bool): Whether to return cross-attention probs from each layer.
            output_hidden_states (bool): Whether to return concept_representations from each layer.
            return_dict (bool): If True, return a BaseModelOutput or dict instead of a tuple.

        Returns:
            BaseModelOutput or tuple(last_hidden_state, hidden_states, attentions)
        """
        batch_size, seq_length = input_ids.size()

        # 1) Token embeddings (in token_embedding_dim space)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_embeddings = self.token_embeddings(input_ids) + self.token_position_embeddings(position_ids)
        
        # Project token embeddings to hidden_size if Dimension Inversion is active
        # (token_embedding_dim < hidden_size). This bridges the gap between cheap
        # token representations and the rich concept/attention space.
        if self.token_projection is not None:
            token_embeddings = self.token_projection(token_embeddings)
        
        token_embeddings = self.dropout(token_embeddings)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # bool of shape [batch_size, seq_len]

        # 2) Initialize concept embeddings [batch_size, concept_num, hidden_size]
        # Every item in the batch starts with the exact same set of initial concept prototypes.
        # These prototypes are then specialized for each input sequence through the layers.
        concept_ids = torch.arange(self.config.concept_num, device=input_ids.device)
        concept_representations = self.concept_embeddings(concept_ids).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add concept position encoding if configured
        # Sinusoidal: fixed positions from buffer (no gradient)
        # Learned: trainable positions from embedding layer
        if self.config.concept_position_type == "sinusoidal":
            concept_representations = concept_representations + self.concept_position_emb.unsqueeze(0)
        elif self.config.concept_position_type == "learned":
            concept_representations = concept_representations + self.concept_position_emb(concept_ids).unsqueeze(0)

        # Possibly track hidden_states/attentions
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = concept_representations

        # 4) Pass through each layer
        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self._use_bixt:
                hidden_states, token_embeddings = layer(
                    concept_representations=hidden_states,
                    token_embeddings=token_embeddings,
                    attention_mask=key_padding_mask,
                )
            else:
                hidden_states = layer(
                    concept_representations=hidden_states,
                    token_embeddings=token_embeddings,
                    attention_mask=key_padding_mask,
                )

        last_hidden_state = self.output_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (last_hidden_state,)

        if return_dict:
            return BaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        else:
            outputs = (last_hidden_state,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

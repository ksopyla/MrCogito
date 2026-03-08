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
        use_bixt: bool = False,
        bixt_token_ffn: bool = True,
        decoder_posonly: bool = False,
        decoder_num_layers: int = 3,
        checkpoint_family: Optional[str] = None,
        evaluation_contract_version: Optional[int] = None,
        canonical_pair_eval_mode: Optional[str] = None,
        canonical_single_eval_mode: Optional[str] = None,
        pretraining_objective: Optional[str] = None,
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
        self.use_bixt = use_bixt
        self.bixt_token_ffn = bixt_token_ffn
        # decoder_posonly: True for TSDAE/PosOnly checkpoints — decoder queries use position
        # embeddings only (no input token shortcut). Stored in config so ViaDecoder classification
        # loads the correct decoder variant without silently using the wrong mode.
        self.decoder_posonly = decoder_posonly
        self.decoder_num_layers = decoder_num_layers
        # Evaluation contract metadata lets downstream scripts pick the canonical
        # evaluation route directly from the checkpoint instead of relying on
        # fragile CLI conventions.
        self.checkpoint_family = checkpoint_family
        self.evaluation_contract_version = evaluation_contract_version
        self.canonical_pair_eval_mode = canonical_pair_eval_mode
        self.canonical_single_eval_mode = canonical_single_eval_mode
        self.pretraining_objective = pretraining_objective

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


class BiXTCrossAttention(nn.Module):
    """Paper-faithful bi-directional cross-attention (Hiller et al., NeurIPS 2024).

    Computes the similarity matrix R_lat @ R_tok^T ONCE and transposes it for the
    reverse direction (Eq. 2).  Uses 4 projection matrices (R_lat, V_lat, R_tok,
    V_tok) instead of 6 (Q, K, V per direction), saving ~1/3 of projection params.
    Both sides are updated simultaneously from pre-update representations (Eq. 3).

    Supports different dim_lat and dim_tok, enabling Dimension Inversion where
    tokens stay in a thin embedding space (e.g. 32) while concepts live in a
    rich space (e.g. 512).  The dimension bridging happens transiently inside
    the reference/value projections.
    """

    def __init__(
        self,
        dim_lat: int,
        dim_tok: int,
        dim_attn: int,
        num_heads: int,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        update_tokens: bool = True,
    ):
        super().__init__()
        assert dim_attn % num_heads == 0, (
            f"dim_attn ({dim_attn}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim_attn = dim_attn
        self.update_tokens = update_tokens

        # Reference + Value projections for each side (4 matrices, not 6)
        self.rv_lat = nn.Linear(dim_lat, dim_attn * 2)
        self.rv_tok = nn.Linear(dim_tok, dim_attn * 2)

        self.attn_drop_lat = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)
        self.proj_drop_lat = nn.Dropout(proj_drop)
        if self.update_tokens:
            self.attn_drop_tok = nn.Dropout(attn_drop)
            self.proj_tok = nn.Linear(dim_attn, dim_tok)
            self.proj_drop_tok = nn.Dropout(proj_drop)
        else:
            self.attn_drop_tok = None
            self.proj_tok = None
            self.proj_drop_tok = None

    def forward(
        self,
        x_lat: torch.Tensor,
        x_tok: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_lat: [B, M, dim_lat] — pre-normed concept representations.
            x_tok: [B, N, dim_tok] — pre-normed token representations.
            key_padding_mask: [B, N] bool, True = padded (ignored in lat←tok).

        Returns:
            (lat_update, tok_update): residual deltas for both sides.
        """
        B, M, _ = x_lat.shape
        _, N, _ = x_tok.shape
        h, d = self.num_heads, self.head_dim

        rv_l = self.rv_lat(x_lat).reshape(B, M, 2, h, d).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_l.unbind(0)  # each [B, h, M, d]

        rv_t = self.rv_tok(x_tok).reshape(B, N, 2, h, d).permute(2, 0, 3, 1, 4)
        r_tok, v_tok = rv_t.unbind(0)  # each [B, h, N, d]

        # Similarity computed ONCE (Eq. 2)
        S = (r_lat @ r_tok.transpose(-2, -1)) * self.scale  # [B, h, M, N]

        # --- Lat ← Tok: mask padded token positions before softmax ---
        if key_padding_mask is not None:
            S_masked = S.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B,1,1,N]
                float('-inf'),
            )
        else:
            S_masked = S

        A_lat = self.attn_drop_lat(S_masked.softmax(dim=-1))     # [B, h, M, N]
        lat_out = (A_lat @ v_tok).transpose(1, 2).reshape(B, M, self.dim_attn)
        lat_out = self.proj_drop_lat(self.proj_lat(lat_out))

        tok_out = None
        if self.update_tokens:
            # --- Tok ← Lat: transpose ORIGINAL S (latents are never padded) ---
            S_T = S.transpose(-2, -1)                                 # [B, h, N, M]
            A_tok = self.attn_drop_tok(S_T.softmax(dim=-1))           # [B, h, N, M]
            tok_out = (A_tok @ v_lat).transpose(1, 2).reshape(B, N, self.dim_attn)
            tok_out = self.proj_drop_tok(self.proj_tok(tok_out))

        return lat_out, tok_out


class BiConceptEncoderLayer(nn.Module):
    """BiXT-style bidirectional cross-attention encoder layer.

    Uses paper-faithful BiXTCrossAttention: single shared similarity matrix,
    simultaneous updates, and native support for Dimension Inversion where
    tokens stay in token_embedding_dim throughout all layers.

    Per-layer flow:
      1. BiXCA  — concepts and tokens attend to each other simultaneously  O(C*N)
      2. Token FFN (optional) — cheap non-linear refinement in dim_tok     O(N*dim_tok)
      3. Concept self-attention                                            O(C^2)
      4. Concept FFN (gated)                                               O(C*intermediate)

    Reference: Hiller, Ehinger & Drummond, "BiXT: Perceiving Longer Sequences
    With Bi-Directional Cross-Attention Transformers", NeurIPS 2024.
    """

    def __init__(self, config: ConceptEncoderConfig, update_tokens: bool = True):
        super().__init__()

        dim_lat = config.hidden_size
        dim_tok = config.token_embedding_dim
        dim_attn = config.hidden_size

        # --- BiXT cross-attention (single similarity matrix) ---
        self._update_tokens = update_tokens
        self.bixt_cross_attn = BiXTCrossAttention(
            dim_lat=dim_lat, dim_tok=dim_tok, dim_attn=dim_attn,
            num_heads=config.num_attention_heads,
            attn_drop=config.attention_probs_dropout_prob,
            proj_drop=config.hidden_dropout_prob,
            update_tokens=update_tokens,
        )
        self.pre_cross_norm_lat = nn.LayerNorm(dim_lat)
        self.pre_cross_norm_tok = nn.LayerNorm(dim_tok)

        # --- token FFN (optional, very cheap at small dim_tok) ---
        self._use_token_ffn = update_tokens and getattr(config, "bixt_token_ffn", True)
        if self._use_token_ffn:
            tok_intermediate = dim_tok * 4
            self.pre_ff_norm_tok = nn.LayerNorm(dim_tok)
            self.Wi_tok = nn.Linear(dim_tok, tok_intermediate * 2)  # *2 for gating
            self.Wo_tok = nn.Linear(tok_intermediate, dim_tok)
            self.wi_dropout_tok = nn.Dropout(config.hidden_dropout_prob)

        # --- concept self-attention ---
        self.concept_self_attn = nn.MultiheadAttention(
            embed_dim=dim_lat, num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob, batch_first=True,
        )
        self.pre_self_attn_norm = nn.LayerNorm(dim_lat)

        # --- concept FFN (gated) ---
        self.pre_ff_norm_lat = nn.LayerNorm(dim_lat)
        self.Wi_lat = nn.Linear(dim_lat, config.intermediate_size * 2)
        self.Wo_lat = nn.Linear(config.intermediate_size, dim_lat)
        self.wi_dropout_lat = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = nn.GELU()

    def forward(
        self,
        concept_representations: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concept_representations: [B, C, dim_lat]
            token_embeddings:        [B, N, dim_tok]  (may differ from dim_lat)
            attention_mask:          [B, N] bool, True = padded

        Returns:
            (concepts, tokens) — both updated.
        """
        # 1. Simultaneous bi-directional cross-attention (Eq. 2-3)
        normed_c = self.pre_cross_norm_lat(concept_representations)
        normed_t = self.pre_cross_norm_tok(token_embeddings)
        c_out, t_out = self.bixt_cross_attn(normed_c, normed_t, key_padding_mask=attention_mask)
        concept_representations = concept_representations + c_out
        if self._update_tokens:
            token_embeddings = token_embeddings + t_out

        # 2. Token FFN — cheap non-linear processing in dim_tok space
        if self._use_token_ffn:
            nt = self.pre_ff_norm_tok(token_embeddings)
            ff_in_t, ff_gate_t = self.Wi_tok(nt).chunk(2, dim=-1)
            ff_out_t = self.Wo_tok(self.wi_dropout_tok(self.act_fn(ff_in_t) * ff_gate_t))
            token_embeddings = token_embeddings + ff_out_t

        # 3. Concept self-attention   O(C^2)
        normed_c = self.pre_self_attn_norm(concept_representations)
        sa_out, _ = self.concept_self_attn(
            normed_c, normed_c, normed_c, attn_mask=None, need_weights=False,
        )
        concept_representations = concept_representations + sa_out

        # 4. Concept FFN (gated)
        normed_c = self.pre_ff_norm_lat(concept_representations)
        ff_in, ff_gate = self.Wi_lat(normed_c).chunk(2, dim=-1)
        ff_out = self.Wo_lat(self.wi_dropout_lat(self.act_fn(ff_in) * ff_gate))
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
        # BiXT handles dimension bridging internally (rv_tok projects dim_tok → dim_attn
        # transiently), so tokens stay thin throughout all layers — critical for long
        # sequences where persistent [B, N, hidden_size] storage is prohibitive.
        # Non-BiXT layers require tokens in hidden_size (nn.MHA embed_dim constraint).
        if token_dim != config.hidden_size and not config.use_bixt:
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
        if getattr(config, "use_bixt", False):
            self.layers = nn.ModuleList([
                BiConceptEncoderLayer(
                    config,
                    update_tokens=(layer_index < config.num_hidden_layers - 1),
                )
                for layer_index in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList([ConceptEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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

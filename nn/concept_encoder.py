from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

logger = logging.get_logger(__name__)

class ConceptEncoderConfig(PretrainedConfig):
    """
    Configuration class for ConceptEncoder.
    
    Inherits from PretrainedConfig, so it integrates more seamlessly with
    the Hugging Face Transformers library (e.g., from_pretrained, save_pretrained).

    This configuration class controls parameters for the ConceptEncoder model.
    It defines the architecture (number of layers, dimensions, etc.) and training
    parameters (dropout rates, initialization ranges, etc.).

    Args:
        vocab_size (int): Size of the token vocabulary.
        concept_size (int): Number of concept tokens to learn.
        hidden_size (int): Dimension of hidden layers and embeddings.
        num_hidden_layers (int): Number of transformer layers in the encoder.
        num_attention_heads (int): Number of attention heads in each layer.
        intermediate_size (int): Dimension of the feedforward network in each layer.
        hidden_act (str): Activation function for the hidden layers.
        hidden_dropout_prob (float): Dropout probability for fully connected layers.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        max_position_embeddings (int): Maximum sequence length supported by the model.
        type_vocab_size (int): Size of the token type vocabulary.
        initializer_range (float): Standard deviation for initializing model weights.
        is_decoder (bool): Whether the model acts as a decoder. Defaults to False.
    """

    model_type = "concept_encoder"

    def __init__(
        self,
        vocab_size: int = 30522,
        concept_size: int = 128,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        hidden_act: str = "gelu",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        cls_token_id: int = 3,
        sep_token_id: int = 4,
        mask_token_id: int = 5,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.is_decoder = is_decoder
        
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id

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
        attention_mask: Optional[torch.BoolTensor] = None,
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
            key_padding_mask=attention_mask 
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

        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id)
        self.token_position_embeddings = nn.Embedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size)
        self.concept_embeddings = nn.Embedding(num_embeddings=config.concept_size, embedding_dim=config.hidden_size)

        self.layers = nn.ModuleList([ConceptEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.post_init()

    def _init_weights(self, module):
        """
        Override _init_weights so that from_pretrained or .init_weights() uses
        your custom init logic. This aligns with Hugging Face patterns.
        """
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

        # 1) Token embeddings
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_embeddings = self.token_embeddings(input_ids) + self.token_position_embeddings(position_ids)
        token_embeddings = self.dropout(token_embeddings)

        key_padding_mask = (attention_mask == 0)  # bool of shape [batch_size, seq_len]

        # 3) Initialize concept embeddings [batch_size, concept_length, hidden_size]
        concept_representations = self.concept_embeddings(
            torch.arange(self.config.concept_size, device=input_ids.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)

        # Possibly track hidden_states/attentions
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = concept_representations

        # 4) Pass through each layer
        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Cross + self-attention inside the layer
            hidden_states = layer(
                concept_representations=hidden_states,
                token_embeddings=token_embeddings,
                # 3D attention_mask => [batch_size, concept_length, seq_length]
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

class ConceptEncoderForMaskedLM(PreTrainedModel):
    """
    ConceptEncoder Model with a language modeling head on top (for masked language modeling).

    Args:
        config (ConceptEncoderConfig): Model configuration defining hidden sizes, embeddings, etc.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config

        # The underlying ConceptEncoder (as defined above).
        self.encoder = ConceptEncoder(config) # []

        # Project concepts to sequence positions via attention
        self.concept_to_sequence = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Final MLM head to project to vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and tie if needed
        self.post_init()
        self.tie_weights()

    def tie_weights(self):
        # Tie the lm_head to the token_embeddings weight to share parameters if desired
        self._tie_or_clone_weights(self.lm_head, self.encoder.token_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        Perform a forward pass for masked language modeling, based on the final concept representations.
        Concept to Sequence Mapping:
        We now map concepts back to sequence positions
        This is done through an attention mechanism between token embeddings and concepts
        The attention weights determine how much each concept contributes to each sequence position
        
        Args:
            input_ids (torch.LongTensor): [batch_size, seq_length] 
                Indices of input sequence tokens.
            attention_mask (Optional[torch.FloatTensor]): [batch_size, seq_length]
                1 for tokens to attend to, 0 for tokens to ignore.
            labels (Optional[torch.LongTensor]): [batch_size, seq_length]
                MLM labels for the input sequence. pad_token_id indicates tokens to ignore.

        Returns:
            (loss, logits) if labels are provided
            or just (logits,) otherwise

            logits => [batch_size, concept_size, vocab_size]
        """
        # Get concept representations from encoder
        encoder_outputs = self.encoder(input_ids, attention_mask)
        concept_repr = encoder_outputs.last_hidden_state  # [batch_size, concept_size, hidden_size]

        # Get token embeddings for attention computation
        token_embeddings = self.encoder.token_embeddings(input_ids)  # [batch_size, seq_length, hidden_size]

        # Project concepts for attention
        projected_concepts = self.concept_to_sequence(concept_repr)  # [batch_size, concept_size, hidden_size]

        # Compute attention scores between sequence positions and concepts
        attention_scores = torch.matmul(
            token_embeddings, 
            projected_concepts.transpose(-1, -2)
        )  # [batch_size, seq_length, concept_size]
        
        # Normalize attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weight concept representations by attention
        sequence_repr = torch.matmul(
            attention_weights, 
            concept_repr
        )  # [batch_size, seq_length, hidden_size]

        # Project to vocabulary
        logits = self.lm_head(sequence_repr)  # [batch_size, seq_length, vocab_size]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            return (loss, logits)
        
        return (logits,)

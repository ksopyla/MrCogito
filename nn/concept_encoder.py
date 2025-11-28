from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput

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
        concept_num (int): Number of concept tokens to learn.
        hidden_size (int): Dimension of hidden layers and embeddings.
        num_hidden_layers (int): Number of transformer layers in the encoder.
        num_attention_heads (int): Number of attention heads in each layer.
        intermediate_size (int): Dimension of the feedforward network in each layer.
        hidden_act (str): Activation function for the hidden layers.
        hidden_dropout_prob (float): Dropout probability for fully connected layers.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        max_sequence_length (int): Maximum sequence length supported by the model.
        type_vocab_size (int): Size of the token type vocabulary.
        initializer_range (float): Standard deviation for initializing model weights.
        is_decoder (bool): Whether the model acts as a decoder. Defaults to False.
    """

    model_type = "concept_encoder"

    def __init__(
        self,
        vocab_size: int = 30522,
        concept_num: int = 128,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_act: str = "gelu",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        cls_token_id: int = 3,
        sep_token_id: int = 4,
        mask_token_id: int = 5,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_sequence_length: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.1,
        is_decoder: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.concept_num = concept_num
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_sequence_length = max_sequence_length
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
        concept_token_attn_output, _ = self.concept_token_attn(
            normed_concepts, token_embeddings, token_embeddings, 
            key_padding_mask=attention_mask 
        )

        # Add residual connection, add the additional knowledge from the concept token similarities to original concept representations, (how to fuse such information?, norm could act as a fuse operation, so maybe we could also use other operations )
        concept_representations = concept_representations + concept_token_attn_output

        
        # Pre-LN, norm operation could be view as fusing the knowledge
        normed_concepts = self.pre_self_attn_norm(concept_representations)

        # Self Attention on concept representations, Q, K, V = concept_representations
        # Queries: concepts [batch_size, concept_num, concept_dim]
        # Keys: concepts [batch_size, concept_num, concept_dim]
        # Values: concepts [batch_size, concept_num, concept_dim]
        concept_self_attn_output, _ = self.concept_self_attn(
            normed_concepts, normed_concepts, normed_concepts,
            attn_mask=None  # No mask needed for concept self-attention
        )

        # Add residual connection between concepts after concept self attention
        concept_representations = concept_representations + concept_self_attn_output

        # Feed Forward Network with gating mechanism
        # Layer Normalization - concept normalization
        normed_concepts = self.pre_ff_norm(concept_representations)

        ff_input, ff_gate = self.Wi(normed_concepts).chunk(2, dim=-1)
        ff_output = self.Wo(self.wi_dropout(self.act_fn(ff_input) * ff_gate))

        # Add residual connection between concepts after feed forward network
        concept_representations = concept_representations + ff_output

        return concept_representations # [batch_size, concept_num, concept_dim]

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


        # Token embeddings [vocab_size, hidden_size=token_embedding_dim]
        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id)   
        # Token position embeddings [max_sequence_length, hidden_size=token_embedding_dim]
        self.token_position_embeddings = nn.Embedding(num_embeddings=config.max_sequence_length, embedding_dim=config.hidden_size)
        # Concept embeddings [concept_num, hidden_size=concept_dim]
        self.concept_embeddings = nn.Embedding(num_embeddings=config.concept_num, embedding_dim=config.hidden_size)

        # Concept encoder layers [num_hidden_layers]
        self.layers = nn.ModuleList([ConceptEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Dropout [hidden_dropout_prob]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output layer normalization [hidden_size=concept_dim] - return the concept representations [batch_size, concept_num, concept_dim]
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.post_init()

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
        # From gemini deep research analysis:
        # Concept Initialization: A key step is the initialization of concept_representations. The learnable concept_embeddings (shape [concept_num, hidden_size]) are expanded to match the batch size ([batch_size, concept_num, hidden_size]). This means every item in the batch starts with the exact same set of initial concept prototypes. These prototypes are then specialized for each input sequence through the subsequent layer processing.
        concept_representations = self.concept_embeddings(
            torch.arange(self.config.concept_num, device=input_ids.device)
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

        # todo - add necessary variables here when we establisht how we want to combine the concepts and tokens


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, special_tokens_mask=None, labels=None):

        pass

 



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
        
        # Optionally tie embeddings (disabled by default for experimentation)
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
            if self.training and hasattr(self, 'compute_orthogonality_loss'):
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

    






# try add this to the loss function to learn the different concepts and make them not correlated with each other to much
def compute_orthogonality_loss(self, concept_repr):
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
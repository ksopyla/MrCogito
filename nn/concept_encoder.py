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
        max_position_embeddings (int): Maximum sequence length supported by the model.
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
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
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
        self.max_position_embeddings = max_position_embeddings
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

        # Add residual connection, add the additional knowledge from the concept token similarities to original concept representations, (how to fuse such information?, norm could act as a fuse operation, so maybe we could also use other operations )
        concept_representations = concept_representations + concept_token_attn_output

        
        # Pre-LN, norm operation could be view as fusing the knowledge
        normed_concepts = self.pre_self_attn_norm(concept_representations)

        # Self Attention on concept representations, if this is needed? leave for further experiments
        concept_self_attn_output, _ = self.concept_self_attn(
            normed_concepts, normed_concepts, normed_concepts,
            attn_mask=None  # No mask needed for concept self-attention
        )

        # Add residual connection between concepts
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
        self.concept_embeddings = nn.Embedding(num_embeddings=config.concept_num, embedding_dim=config.hidden_size)

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

        # Project concepts to sequence positions via linear layer
        self.concept_to_sequence = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size) # from concept_dim_size to token_embedding_dim
        )

        # Final MLM head to project to vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and tie if needed
        self.post_init()
        
        # Purposely not tying the weights, to allow for more flexibility in the model architecture
        #self.tie_weights()



    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

            logits => [batch_size, concept_num, vocab_size]
        """

        


        # Get concept representations from encoder
        encoder_outputs = self.encoder(input_ids, attention_mask, output_attentions, output_hidden_states)
        concept_repr = encoder_outputs.last_hidden_state  # [batch_size, concept_num, hidden_size]

        # Get token embeddings for attention computation

        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.encoder.token_embeddings(input_ids) + self.encoder.token_position_embeddings(position_ids)  # [batch_size, seq_length, hidden_size]

        # Project concepts for attention

        # from gemini deep research analysis:
        #TODO: Project Concepts: Apply the concept_to_sequence module to concept_repr to get projected_concepts. The purpose of this projection (LayerNorm + Linear) before the attention step is not immediately obvious from first principles. It might serve to transform the concept representations into a space more suitable for being attended to by token embeddings, or simply to add learnable parameters to the mapping process. Its necessity or benefit should ideally be verified through ablation studies.
        projected_concepts = self.concept_to_sequence(concept_repr)  # [batch_size, concept_num, hidden_size]



        # from gemini deep research analysis https://gemini.google.com/gem/0696cd886317/ce82c504453e4949:
        # This approach fundamentally changes the nature of the MLM task. It compels the model to 
        # **first compress the essential information of the entire sequence into the fixed-size concept_repr bottleneck**, 
        # and then reconstruct the sequence from this compressed representation. 
        # This bears resemblance to autoencoder frameworks, where the concepts act as the encoded latent code. 
        # It also relates conceptually to latent variable models where generation or reconstruction is conditioned on learned latent codes.1 
        # The potential consequence is that the model might develop stronger representations of global semantics or concepts, 
        # potentially at the cost of fine-grained local prediction accuracy compared to standard MLM
        # The attention-based mapping from concepts back to sequence positions is distinct from other methods. Some approaches map latent representations directly to vocabulary logits, sometimes using the MLM head itself. Others use dedicated decoders. This implementation introduces an intermediate attention step where tokens query concepts to reconstruct their own representations before final vocabulary projection.   
        # A potential risk is that the model might learn a trivial solution, such as copying token information into concepts and then directly retrieving it. The multi-layer structure of the ConceptEncoder should mitigate this, but the effectiveness of the information compression into the concept bottleneck remains an empirical question

        # Compute attention scores between sequence positions and concepts
        attention_scores = torch.matmul(
            token_embeddings, 
            projected_concepts.transpose(-1, -2)
        )  # [batch_size, seq_length, concept_num]
        
        # Add scaling to prevent softmax overflow
        attention_scores = attention_scores / (self.config.hidden_size ** 0.5)

        # Apply attention mask if provided (mask out padding tokens)
        if attention_mask is not None:
            # Expand mask for broadcasting [batch_size, seq_length, 1]
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(attention_mask_expanded == 0, -1e9)
        
        # Normalize attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weight concept representations by attention
        sequence_repr = torch.matmul(
            attention_weights, 
            concept_repr
        )  # [batch_size, seq_length, hidden_size]

        # Project to vocabulary
        logits = self.lm_head(sequence_repr)  # [batch_size, seq_length, vocab_size]

        mlm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            mlm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return MaskedLMOutput(
            loss=mlm_loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )



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
            config.hidden_size, # concept_dim_size
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
        concept_repr = encoder_out.last_hidden_state  # [B, C, H]
        

        # Get token embeddings with position information
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_emb = self.encoder.token_embeddings(input_ids) + self.encoder.token_position_embeddings(position_ids) # [B, S, H]
        
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
        
        # Learn a weight matrix for combining concepts per position
        # Initialize with small random values to break symmetry
        self.concept_weights = nn.Parameter(
            torch.randn(config.max_position_embeddings, config.concept_num) / math.sqrt(config.concept_num)
        )
        
        # Simple MLM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Optional: add a projection layer before lm_head for more capacity
        self.pre_lm_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
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
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]
        
        # Get position-specific weights and normalize them
        position_weights = self.concept_weights[:seq_length, :]  # [S, C]
        position_weights = F.softmax(position_weights, dim=-1)  # Normalize over concepts
        
        # Expand weights for batch processing
        position_weights_expanded = position_weights.unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, C]
        
        # Combine concepts using learned weights: [B, S, H] = [B, S, C] x [B, C, H]
        sequence_repr = torch.bmm(position_weights_expanded, concept_repr)
        
        # Optional: apply projection before final LM head
        sequence_repr = self.pre_lm_projection(sequence_repr)
        
        # Project to vocabulary
        logits = self.lm_head(sequence_repr)  # [B, S, V]
        
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


class ConceptEncoderForSequenceClassification(PreTrainedModel):
    """
    ConceptEncoder Model with a sequence classification head on top
    for tasks like GLUE (MNLI, QNLI, QQP, SST-2, etc.).
    
    This class is designed to fine-tune a pretrained ConceptEncoder model
    on classification tasks, similar to BertForSequenceClassification.
    
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
        
        # Classification head - we'll use a pooling layer to get a fixed-size
        # representation followed by a classification layer
        #todo: figure out how the polling should be done, different models uses different strategies 
        # look at from transformers.modeling_utils import SequenceSummary
        # xlnet implements addtitional abstration module to use different pooling strategies

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # Add dropout for regularization during fine-tuning
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply finalizer
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
        """
        Forward pass for sequence classification.
        
        This method:
        1. Passes the inputs through the base ConceptEncoder model
        2. Pools the concept representations (averaging all concepts)
        3. Passes the pooled representation through the classification head
        4. Computes loss if labels are provided
        
        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length]
            attention_mask: Attention mask of shape [batch_size, sequence_length]
            labels: Optional labels for computing loss
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a SequenceClassifierOutput or a tuple
            
        Returns:
            SequenceClassifierOutput or tuple with logits and optional hidden_states/attentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass through the encoder to get concept representations
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the concept representations (batch_size, concept_num, concept_dim)
        concept_representations = encoder_outputs.last_hidden_state
        
        # Pool the concept representations - average pooling across the concept dimension
        # This gives us a representation of size (batch_size, concept_dim)
        pooled_output = torch.mean(concept_representations, dim=1)
        
        # Apply the pooler transformation
        pooled_output = self.pooler(pooled_output)
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Compute logits
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels are provided
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
    
    def get_input_embeddings(self):
        """Returns the token embeddings layer of the encoder."""
        return self.encoder.token_embeddings
    






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
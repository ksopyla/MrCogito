"""
Recursive Concept Encoder — TRM-style weight-tied architecture.

This is a separate model variant that reuses a SINGLE ConceptEncoderLayer
K times instead of having K separate layers. Inspired by:
  - TRM (Jolicoeur-Martineau, 2025): 7M params beats 100B+ LLMs via recursion
  - ALBERT (Lan et al., 2020): shared layers, 12M params, 90.6% MRPC F1
  - Universal Transformer (Dehghani et al., 2018): depth via iteration
  - Recurrent Depth (Geiping et al., 2025): latent-space reasoning

Key properties:
  - ~47% fewer encoder params than standard ConceptEncoder (1 layer vs L layers)
  - Test-time compute scaling: override num_iterations at inference
  - Loads encoder weights from standard ConceptEncoder (layer 0 only)
  - Same ConceptEncoderLayer, same ConceptEncoderConfig — no new layer code

This file does NOT touch concept_encoder.py. The standard ConceptEncoder
is completely unaffected and remains the default for all existing workflows.

Usage:
    from nn.concept_encoder_recursive import RecursiveConceptEncoder, RecursiveConceptEncoderConfig

    config = RecursiveConceptEncoderConfig(
        num_hidden_layers=6,   # how many iterations during training
        num_iterations=None,   # None → use num_hidden_layers; override at inference
    )
    encoder = RecursiveConceptEncoder(config)

    # Test-time compute scaling:
    encoder.config.num_iterations = 12  # more iterations for harder inputs
"""

from typing import Optional
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutput

from nn.concept_encoder import ConceptEncoderConfig, ConceptEncoderLayer

logger = logging.get_logger(__name__)


class RecursiveConceptEncoderConfig(ConceptEncoderConfig):
    """Config for the recursive (weight-tied) concept encoder.

    Extends ConceptEncoderConfig with a single extra field:
      num_iterations: how many times the shared layer is applied.
                      None → falls back to num_hidden_layers.
                      Can be changed at inference for test-time compute scaling.

    All other fields are identical to ConceptEncoderConfig, so existing
    tokenizer / hidden_size / concept_num settings are reused as-is.
    """

    model_type = "recursive_concept_encoder"

    def __init__(self, num_iterations: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations


class RecursiveConceptEncoder(PreTrainedModel):
    """Concept Encoder with a single weight-tied layer applied K times.

    Architecture is identical to ConceptEncoder except:
      - self.layers contains exactly ONE ConceptEncoderLayer
      - forward() applies that layer K times (K = num_iterations or num_hidden_layers)
      - Token/concept embeddings, projections, norms are the same

    Parameter comparison (H512, L6, C128, intermediate=2048):
      Standard ConceptEncoder:  6 × 6.3M layer params = 37.6M  → ~64.5M total
      RecursiveConceptEncoder:  1 × 6.3M layer params =  6.3M  → ~33.2M total  (−47%)
    """

    config_class = RecursiveConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: RecursiveConceptEncoderConfig):
        super().__init__(config)
        self.config = config

        token_dim = config.token_embedding_dim

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=token_dim,
            padding_idx=config.pad_token_id,
        )
        self.token_position_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length,
            embedding_dim=token_dim,
        )

        if token_dim != config.hidden_size:
            self.token_projection = nn.Linear(token_dim, config.hidden_size)
        else:
            self.token_projection = None

        self.concept_embeddings = nn.Embedding(
            num_embeddings=config.concept_num,
            embedding_dim=config.hidden_size,
        )

        if config.concept_position_type == "sinusoidal":
            sinusoidal_emb = self._create_sinusoidal_embeddings(
                config.concept_num, config.hidden_size
            )
            self.register_buffer("concept_position_emb", sinusoidal_emb)
        elif config.concept_position_type == "learned":
            self.concept_position_emb = nn.Embedding(
                num_embeddings=config.concept_num,
                embedding_dim=config.hidden_size,
            )

        # ONE shared layer — the core difference from ConceptEncoder
        self.shared_layer = ConceptEncoderLayer(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.post_init()

    @staticmethod
    def _create_sinusoidal_embeddings(num_positions: int, dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        embeddings = torch.zeros(num_positions, dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings

    def _init_weights(self, module):
        if module is self.concept_embeddings:
            concept_std = min(2.0 * self.config.initializer_range, 1.0)
            module.weight.data.normal_(mean=0.0, std=concept_std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def _num_iterations(self) -> int:
        """Effective iteration count: num_iterations if set, else num_hidden_layers."""
        return self.config.num_iterations or self.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.size()

        position_ids = (
            torch.arange(seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        token_embeddings = (
            self.token_embeddings(input_ids)
            + self.token_position_embeddings(position_ids)
        )
        if self.token_projection is not None:
            token_embeddings = self.token_projection(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        concept_ids = torch.arange(self.config.concept_num, device=input_ids.device)
        concept_representations = (
            self.concept_embeddings(concept_ids).unsqueeze(0).expand(batch_size, -1, -1)
        )
        if self.config.concept_position_type == "sinusoidal":
            concept_representations = (
                concept_representations + self.concept_position_emb.unsqueeze(0)
            )
        elif self.config.concept_position_type == "learned":
            concept_representations = (
                concept_representations
                + self.concept_position_emb(concept_ids).unsqueeze(0)
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = concept_representations

        # Apply the SAME layer K times (weight-tied recursion)
        for iteration in range(self._num_iterations):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = self.shared_layer(
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
        outputs = (last_hidden_state,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)
        return outputs

    def load_from_standard_checkpoint(self, checkpoint_path: str) -> dict:
        """Load encoder weights from a standard (non-recursive) ConceptEncoder checkpoint.

        Copies embedding/projection/norm weights directly. For the shared layer,
        loads weights from layers.0 of the checkpoint (first layer).

        Args:
            checkpoint_path: Path to directory containing model.safetensors or pytorch_model.bin.

        Returns:
            dict with keys 'loaded' and 'skipped' (counts).
        """
        import os

        ckpt_file = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            from safetensors.torch import load_file
            ckpt = load_file(ckpt_file)
        else:
            ckpt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            ckpt = torch.load(ckpt_file, map_location="cpu")

        model_sd = self.state_dict()
        loaded, skipped = 0, 0

        for ckpt_key, ckpt_val in ckpt.items():
            if not ckpt_key.startswith("encoder."):
                continue

            # Map layers.0.* → shared_layer.*
            target_key = ckpt_key.replace("encoder.", "", 1)
            if target_key.startswith("layers.0."):
                target_key = target_key.replace("layers.0.", "shared_layer.", 1)
            elif target_key.startswith("layers."):
                # Skip layers 1..N — not needed for shared layer
                skipped += 1
                continue

            if target_key in model_sd and model_sd[target_key].shape == ckpt_val.shape:
                model_sd[target_key] = ckpt_val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(model_sd)
        logger.info(
            f"Loaded {loaded} weights from standard checkpoint "
            f"(skipped {skipped}, layers.1-N ignored for weight tying)"
        )
        return {"loaded": loaded, "skipped": skipped}

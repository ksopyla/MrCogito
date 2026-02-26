"""
Recursive Concept Encoder for Masked Language Modeling.

Pairs the RecursiveConceptEncoder (weight-tied, 1 shared layer applied K times)
with the same Perceiver IO decoder used by ConceptEncoderForMaskedLMPerceiver.

The decoder is NOT weight-tied — only the encoder uses recursion. This mirrors
ALBERT's design: shared encoder layers + task-specific heads.

Usage in mlm_training.py:
    --model_type recursive_mlm

Warm-start from a standard perceiver_mlm checkpoint:
    --model_type recursive_mlm --model_name_or_path Cache/Training/perceiver_mlm_H512L6C128_.../

Weight loading from standard checkpoint:
    - encoder.layers.0.* -> encoder.shared_layer.*  (layer 0 reused as shared layer)
    - encoder.layers.1-N.* -> skipped (not needed for weight tying)
    - decoder_* -> loaded as-is (same architecture)
    - lm_head.* -> loaded as-is
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import MaskedLMOutput

from nn.concept_encoder_recursive import RecursiveConceptEncoder, RecursiveConceptEncoderConfig
from nn.loss_manager import LossManager, LossConfig

logger = logging.get_logger(__name__)


class RecursiveConceptEncoderForMaskedLM(PreTrainedModel):
    """Recursive ConceptEncoder with Perceiver IO decoding for MLM.

    Structurally identical to ConceptEncoderForMaskedLMPerceiver except
    self.encoder is a RecursiveConceptEncoder (1 shared layer, K iterations).

    The decoder uses Input+Position queries (same as perceiver_mlm).
    """

    config_class = RecursiveConceptEncoderConfig
    base_model_prefix = "concept_encoder"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: RecursiveConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None,
    ):
        super().__init__(config)
        self.config = config

        self.encoder = RecursiveConceptEncoder(config)

        self.set_loss_config(loss_config)

        self.decoder_query_embeddings = nn.Embedding(
            num_embeddings=config.max_sequence_length,
            embedding_dim=config.hidden_size,
        )

        if config.token_embedding_dim != config.hidden_size:
            self.decoder_input_projection = nn.Linear(
                config.token_embedding_dim, config.hidden_size
            )
        else:
            self.decoder_input_projection = None

        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        self.post_cross_norm = nn.LayerNorm(config.hidden_size)

        self.decoder_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        if config.tie_word_embeddings and config.token_embedding_dim == config.hidden_size:
            self._tie_or_clone_weights(self.lm_head, self.encoder.token_embeddings)

    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        self.loss_manager = LossManager.create_for_model(
            concept_num=self.config.concept_num,
            hidden_size=self.config.hidden_size,
            loss_config=loss_config,
        )
        self._loss_config = loss_config

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

        # 1. Encode: tokens -> concepts (K iterations of shared layer)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        concept_repr = encoder_outputs.last_hidden_state  # [B, C, H]

        # 2. Decode: concepts -> token predictions (Perceiver IO)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.decoder_query_embeddings(position_ids).expand(batch_size, -1, -1)

        input_embeddings = self.encoder.token_embeddings(input_ids)
        if self.decoder_input_projection is not None:
            input_embeddings = self.decoder_input_projection(input_embeddings)

        decoder_queries = input_embeddings + pos_embeddings
        decoder_queries_norm = self.decoder_norm(decoder_queries)

        attn_output, _ = self.decoder_cross_attn(
            query=decoder_queries_norm,
            key=concept_repr,
            value=concept_repr,
            need_weights=False,
        )

        decoder_latents = decoder_queries + attn_output
        decoder_output = decoder_latents + self.decoder_ffn(
            self.post_cross_norm(decoder_latents)
        )

        # 3. Sparse MLM loss (same as ConceptEncoderForMaskedLMPerceiver)
        loss = None
        logits = None

        if labels is not None:
            mask = labels != -100
            flat_decoder_output = decoder_output.reshape(-1, decoder_output.size(-1))
            flat_mask = mask.reshape(-1)
            masked_decoder_output = flat_decoder_output[flat_mask]
            masked_logits = self.lm_head(masked_decoder_output)

            flat_labels = labels.view(-1)
            masked_labels = flat_labels[flat_mask]

            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(masked_logits, masked_labels)

            loss = self.loss_manager(
                task_loss=mlm_loss, concept_repr=concept_repr
            )
            logits = masked_logits
        else:
            logits = self.lm_head(decoder_output)

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def load_from_standard_mlm_checkpoint(self, checkpoint_path: str) -> dict:
        """Load weights from a standard perceiver_mlm checkpoint.

        Maps:
          encoder.layers.0.* -> encoder.shared_layer.*
          encoder.layers.1-N.* -> skipped
          decoder_*, lm_head.* -> loaded directly (same architecture)

        Args:
            checkpoint_path: directory with model.safetensors or pytorch_model.bin

        Returns:
            dict with 'loaded' and 'skipped' counts.
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
            if ckpt_key.startswith("lm_head.") or ckpt_key.startswith("loss_manager."):
                skipped += 1
                continue

            target_key = ckpt_key
            if "encoder.layers.0." in ckpt_key:
                target_key = ckpt_key.replace("encoder.layers.0.", "encoder.shared_layer.")
            elif "encoder.layers." in ckpt_key:
                skipped += 1
                continue

            if target_key in model_sd and model_sd[target_key].shape == ckpt_val.shape:
                model_sd[target_key] = ckpt_val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(model_sd)
        logger.info(
            f"Loaded {loaded} weights from standard MLM checkpoint "
            f"(skipped {skipped}; layers.1-N ignored for weight tying)"
        )
        return {"loaded": loaded, "skipped": skipped}

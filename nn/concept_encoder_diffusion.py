"""
Concept Encoder with Masked Discrete Diffusion Decoder.

Architecture
------------
  Encoder : ConceptEncoder (tokens → K concept vectors, same as MLM models)
  Decoder : ConceptDiffusionDecoder
              - Receives: concept vectors (from encoder) + noisy token sequence
                          + continuous timestep t ∈ [0, 1]
              - Predicts: clean token ids at all masked positions
              - Training: Masked Discrete Diffusion (MDLM-style)

Why masked diffusion instead of MLM
------------------------------------
  MLM uses a fixed 15 % masking rate and predicts only at masked positions.
  This forces the concept bottleneck to preserve token-level detail so the
  decoder can fill in a handful of gaps using mostly *local* context.

  Masked diffusion samples the masking rate t ~ Uniform(0, 1) every batch:
    t ≈ 0.05–0.30  : easy denoising, decoder relies on surviving tokens
    t ≈ 0.50–0.80  : hard denoising, decoder MUST use concept vectors
    t ≈ 0.90–1.00  : all tokens masked, decoder denoises from pure concepts

  The curriculum from easy→hard naturally pushes the encoder to build *richer*
  concept representations and removes the fundamental MLM misalignment where
  15 % masking is too easy to require semantic concept-level understanding.

Inference (generation)
-----------------------
  1. Start from all-[MASK] sequence (t = 1.0)
  2. Run K denoising steps (e.g. K = 10–20)
  3. At each step: predict logits → unmask top-confidence positions
  4. Repeat until no [MASK] tokens remain

References
----------
  - MDLM: Masked Diffusion Language Models — Sahoo et al., 2024
  - LCM: Large Concept Models — Meta, 2024 (concept-space diffusion)
  - Recurrent Depth: Geiping et al., 2025 (latent-space reasoning)
  - Token Assorted: Su et al., 2025 (mixing latent + text tokens)
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig
from nn.loss_manager import LossManager, LossConfig

logger = logging.get_logger(__name__)


# ============================================================================
# Output dataclass
# ============================================================================

@dataclass
class DiffusionOutput(ModelOutput):
    """Output of ConceptEncoderForMaskedDiffusion."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None        # [B, L, V] — full vocab logits (inference)
    masked_logits: Optional[torch.Tensor] = None  # [num_masked, V] — sparse (training)
    concept_repr: Optional[torch.Tensor] = None  # [B, C, H]
    noise_level: Optional[torch.Tensor] = None   # [B] — t values used this batch


# ============================================================================
# Timestep embedding
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds a continuous noise level t ∈ [0, 1] into a fixed-dim vector.

    Sinusoidal encoding (like positional encoding) avoids learned parameters
    for the timestep conditioning, keeping the signal clean.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] float tensor in [0, 1]
        Returns:
            [B, dim] embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return self.proj(emb)


# ============================================================================
# Diffusion decoder layer
# ============================================================================

class DiffusionDecoderLayer(nn.Module):
    """
    Single transformer layer for the diffusion decoder.

    Order of operations (Pre-LN):
      1. Self-attention between token positions (coordinate denoising)
      2. Cross-attention to concept vectors (inject semantic conditioning)
      3. FFN

    The timestep embedding is injected via additive bias before self-attention,
    following the AdaLN/adaGN convention used in diffusion transformers.
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        H = config.hidden_size

        self.norm_self = nn.LayerNorm(H)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.norm_cross = nn.LayerNorm(H)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(H)
        self.ff_in = nn.Linear(H, config.intermediate_size * 2)
        self.ff_out = nn.Linear(config.intermediate_size, H)
        self.ff_drop = nn.Dropout(config.hidden_dropout_prob)

        # Timestep scale+shift (AdaLN style) applied before self-attention norm
        self.t_proj = nn.Linear(H, H * 2)

    def forward(
        self,
        x: torch.Tensor,         # [B, L, H] — noisy token representations
        concepts: torch.Tensor,  # [B, C, H] — concept conditioning
        t_emb: torch.Tensor,     # [B, H]    — timestep embedding
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:

        # Timestep conditioning: scale and shift before self-attention
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)  # each [B, H]
        x_t = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 1. Self-attention (token ↔ token)
        x_normed = self.norm_self(x_t)
        sa_out, _ = self.self_attn(
            x_normed, x_normed, x_normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + sa_out

        # 2. Cross-attention (token → concepts)
        x_normed = self.norm_cross(x)
        ca_out, _ = self.cross_attn(
            query=x_normed,
            key=concepts,
            value=concepts,
            need_weights=False,
        )
        x = x + ca_out

        # 3. Gated FFN
        x_normed = self.norm_ff(x)
        gate_inp, gate = self.ff_in(x_normed).chunk(2, dim=-1)
        ff_out = self.ff_out(self.ff_drop(F.gelu(gate_inp) * gate))
        x = x + ff_out

        return x


# ============================================================================
# Full diffusion decoder
# ============================================================================

class ConceptDiffusionDecoder(nn.Module):
    """
    Transformer decoder that denoises masked token sequences conditioned on
    concept vectors.

    Input:
        noisy_ids  : [B, L]    — token ids with some positions = mask_token_id
        concepts   : [B, C, H] — concept vectors from the encoder
        t          : [B]       — noise level for each sample (0 = clean, 1 = all masked)

    Output:
        logits     : [B, L, V] — predicted token logits at every position
    """

    def __init__(self, config: ConceptEncoderConfig, num_layers: int = 2):
        super().__init__()
        H = config.hidden_size
        token_dim = config.token_embedding_dim

        # Reuses encoder token embedding — same vocabulary, same space after projection
        self.token_embed = nn.Embedding(config.vocab_size, token_dim)
        self.pos_embed = nn.Embedding(config.max_sequence_length, H)

        # Project token dim to hidden_size when Dimension Inversion is active
        if token_dim != H:
            self.token_proj = nn.Linear(token_dim, H)
        else:
            self.token_proj = None

        self.t_embed = SinusoidalTimestepEmbedding(H)

        self.layers = nn.ModuleList(
            [DiffusionDecoderLayer(config) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

    def forward(
        self,
        noisy_ids: torch.LongTensor,  # [B, L]
        concepts: torch.Tensor,       # [B, C, H]
        t: torch.Tensor,              # [B] in [0, 1]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = noisy_ids.shape

        pos_ids = torch.arange(L, device=noisy_ids.device).unsqueeze(0)
        x = self.token_embed(noisy_ids)          # [B, L, token_dim]
        if self.token_proj is not None:
            x = self.token_proj(x)               # [B, L, H]
        x = x + self.pos_embed(pos_ids)          # [B, L, H]

        t_emb = self.t_embed(t)                  # [B, H]

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for layer in self.layers:
            x = layer(x, concepts, t_emb, key_padding_mask)

        return self.lm_head(self.out_norm(x))    # [B, L, V]


# ============================================================================
# Full encoder-decoder model
# ============================================================================

class ConceptEncoderForMaskedDiffusion(PreTrainedModel):
    """
    ConceptEncoder + Masked Discrete Diffusion Decoder.

    Training
    --------
    Each forward pass:
      1. Encode input tokens → concept vectors
      2. Sample noise level  t ~ Uniform(t_min, 1.0)   (t_min avoids trivial t≈0)
      3. Mask each token independently with probability t
         (masked tokens → mask_token_id)
      4. Decode concepts + noisy tokens → logits
      5. Compute cross-entropy at all masked positions

    The variable masking rate (vs MLM's fixed 15%) creates a natural curriculum:
      - Low t  → decoder uses surviving tokens (easy, local denoising)
      - High t → decoder must lean on concepts (hard, semantic denoising)

    Inference
    ---------
    Use `generate()` for iterative denoising (K steps from all-[MASK]).

    Args:
        config          : ConceptEncoderConfig
        loss_config     : LossConfig for concept regularization (optional)
        decoder_layers  : Number of transformer layers in the diffusion decoder
        t_min           : Minimum noise level sampled during training (default 0.05)
    """

    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(
        self,
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None,
        decoder_layers: int = 2,
        t_min: float = 0.05,
    ):
        super().__init__(config)
        self.config = config
        self.t_min = t_min

        self.encoder = ConceptEncoder(config)
        self.decoder = ConceptDiffusionDecoder(config, num_layers=decoder_layers)

        self.loss_manager = LossManager.create_for_model(
            concept_num=config.concept_num,
            hidden_size=config.hidden_size,
            loss_config=loss_config,
        )
        self._loss_config = loss_config

        self.post_init()

    def set_loss_config(self, loss_config: Optional[LossConfig]) -> None:
        self.loss_manager = LossManager.create_for_model(
            concept_num=self.config.concept_num,
            hidden_size=self.config.hidden_size,
            loss_config=loss_config,
        )
        self._loss_config = loss_config

    def _apply_noise(
        self,
        input_ids: torch.LongTensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: independently mask each token with probability t[i].

        Args:
            input_ids    : [B, L] — clean token ids
            t            : [B]   — per-sample noise level in [0, 1]
            mask_token_id: id of the [MASK] token

        Returns:
            noisy_ids : [B, L] — input_ids with masked positions replaced
            noise_mask: [B, L] — True where tokens were masked
        """
        # For each token, draw Uniform(0,1) and mask if < t[i]
        rand = torch.rand_like(input_ids, dtype=torch.float32)
        noise_mask = rand < t.unsqueeze(1)          # [B, L] bool

        noisy_ids = input_ids.clone()
        noisy_ids[noise_mask] = mask_token_id
        return noisy_ids, noise_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,           # override noise level for testing
        return_dict: Optional[bool] = None,
    ) -> DiffusionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, L = input_ids.shape

        # 1. Encode — identical to MLM pretraining
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        concepts = encoder_out.last_hidden_state  # [B, C, H]

        # 2. Sample noise level
        if t is None:
            t = torch.empty(B, device=input_ids.device).uniform_(self.t_min, 1.0)

        # 3. Apply masking (forward diffusion)
        mask_token_id = self.config.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "config.mask_token_id must be set. "
                "Pass it via ConceptEncoderConfig(mask_token_id=tokenizer.mask_token_id)."
            )
        noisy_ids, noise_mask = self._apply_noise(input_ids, t, mask_token_id)

        # 4. Decode
        logits = self.decoder(noisy_ids, concepts, t, attention_mask)  # [B, L, V]

        # 5. Sparse loss — only at masked positions (identical to MLM sparse decoding)
        loss = None
        if self.training or noise_mask.any():
            flat_logits = logits.reshape(-1, logits.size(-1))      # [B*L, V]
            flat_mask = noise_mask.reshape(-1)                      # [B*L]
            masked_logits = flat_logits[flat_mask]                  # [M, V]
            masked_targets = input_ids.reshape(-1)[flat_mask]       # [M]

            if masked_logits.numel() > 0:
                diffusion_loss = F.cross_entropy(masked_logits, masked_targets)
                # Concept regularization (optional, via LossManager)
                if self.training:
                    loss = self.loss_manager(
                        task_loss=diffusion_loss,
                        concept_repr=concepts,
                    )
                else:
                    loss = diffusion_loss
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        if not return_dict:
            return (loss, logits, concepts, t)

        return DiffusionOutput(
            loss=loss,
            logits=logits if not self.training else None,
            masked_logits=masked_logits if self.training and noise_mask.any() else None,
            concept_repr=concepts,
            noise_level=t,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.LongTensor:
        """
        Iterative masked diffusion generation.

        If input_ids is given, only the positions that are already [MASK] are
        denoised (conditional generation / infilling).  If input_ids is None,
        starts from an all-[MASK] sequence of length config.max_sequence_length.

        At each step:
          1. Predict logits for all masked positions
          2. Sample (or argmax) token from logits
          3. Unmask the `1/steps_remaining` most-confident positions
          4. Repeat until no [MASK] tokens remain

        Args:
            input_ids     : [B, L] optional seed; None → all-MASK
            attention_mask: [B, L] optional
            num_steps     : denoising steps (higher = better quality)
            temperature   : sampling temperature (1.0 = standard, <1 = sharper)
            top_k         : top-k filtering (0 = disabled)

        Returns:
            generated_ids : [B, L] fully denoised token ids
        """
        mask_id = self.config.mask_token_id
        assert mask_id is not None, "config.mask_token_id must be set for generation"

        if input_ids is None:
            B = 1
            L = self.config.max_sequence_length
            input_ids = torch.full((B, L), mask_id, dtype=torch.long, device=self.device)
            if attention_mask is None:
                attention_mask = torch.ones(B, L, dtype=torch.long, device=self.device)

        B, L = input_ids.shape
        current = input_ids.clone()

        # Encode the context once (concepts are fixed throughout generation)
        # For full generation (all-MASK), concepts are derived from the mask embedding —
        # they will be near-zero; the decoder mostly uses position embeddings at t≈1.
        encoder_out = self.encoder(current, attention_mask, return_dict=True)
        concepts = encoder_out.last_hidden_state  # [B, C, H]

        for step in range(num_steps):
            still_masked = (current == mask_id)              # [B, L] bool
            n_masked = still_masked.sum(dim=-1).max().item()
            if n_masked == 0:
                break

            # Noise level decreases linearly: 1.0 → 0 over steps
            t_val = 1.0 - step / num_steps
            t = torch.full((B,), t_val, device=current.device)

            logits = self.decoder(current, concepts, t, attention_mask)  # [B, L, V]

            # Sample / argmax at masked positions
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                values, _ = logits.topk(top_k, dim=-1)
                logits[logits < values[..., -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)               # [B, L, V]
            sampled = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).reshape(B, L)                                  # [B, L]

            # Confidence score = max probability at each position
            confidence = probs.max(dim=-1).values            # [B, L]

            # Unmask the top `fraction` of masked positions (those with highest confidence)
            steps_remaining = num_steps - step
            unmask_count = max(1, round(n_masked / steps_remaining))

            # Only consider positions that are still masked
            confidence_masked = confidence * still_masked.float()

            for b in range(B):
                masked_positions = still_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                n_unmask = min(unmask_count, len(masked_positions))
                conf_at_masked = confidence_masked[b, masked_positions]
                _, top_idx = conf_at_masked.topk(n_unmask)
                positions_to_unmask = masked_positions[top_idx]
                current[b, positions_to_unmask] = sampled[b, positions_to_unmask]

        return current

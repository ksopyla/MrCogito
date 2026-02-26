"""
Concept Encoder with Masked Discrete Diffusion Decoder.

Architecture
------------
  Encoder : ConceptEncoder (tokens → K concept vectors, same as MLM models)
  Decoder : ConceptDiffusionDecoder
              - Uses ONLY cross-attention to concept vectors (NO token self-attention)
              - Receives: concept vectors + noisy token ids + timestep t ∈ [0, 1]
              - Returns: hidden states at each position
  lm_head : Linear(H → V), applied sparsely to masked positions only

Why cross-attention only (no self-attention)
--------------------------------------------
  The whole point of the concept bottleneck is O(C*N) complexity instead of O(N²).
  Self-attention between N tokens in the decoder defeats this purpose entirely.

  With C=128 concepts:
    Self-attention at N=2M tokens : O(N²) = 4 trillion ops/layer — impossible
    Cross-attention at N=2M tokens: O(N*C) = 256M ops/layer   — trivial

  The encoder already compresses the full input into C concept vectors.  Each
  decoder position independently queries the concept bank to reconstruct its
  token.  No token needs to see other tokens — the concepts already carry the
  full semantic context.

  This is architecturally equivalent to the Perceiver IO decoder with timestep
  conditioning, and follows the same paradigm as Muse (Chang et al., 2023)
  which conditions masked image generation on T5 text embeddings via
  cross-attention.

Why masked diffusion instead of MLM
------------------------------------
  MLM uses a fixed 15% masking rate and predicts only at masked positions.
  This forces the concept bottleneck to preserve token-level detail so the
  decoder can fill in a handful of gaps using mostly *local* context.

  Masked diffusion samples the masking rate t ~ Uniform(0, 1) every batch:
    t ≈ 0.05–0.30  : easy denoising, decoder relies on surviving tokens
    t ≈ 0.50–0.80  : hard denoising, decoder MUST use concept vectors
    t ≈ 0.90–1.00  : all tokens masked, decoder denoises from pure concepts

  The curriculum from easy→hard naturally pushes the encoder to build *richer*
  concept representations.

Timestep conditioning: AdaLN-Zero (Peebles & Xie, 2023)
---------------------------------------------------------
  Scale and shift are regressed from the timestep embedding and applied to the
  normalized hidden states before cross-attention and FFN.  An output gate
  (initialized to zero) controls the residual contribution.

  Zero-initialization ensures the layer starts as identity, preventing the
  multiplicative instability that caused gradient explosion in the previous
  implementation.  The model gradually learns to use the conditioning signal
  during training.

Inference (generation)
-----------------------
  1. Start from all-[MASK] sequence (t = 1.0)
  2. Run K denoising steps (e.g. K = 10–20)
  3. At each step: predict logits → unmask top-confidence positions
  4. Repeat until no [MASK] tokens remain

References
----------
  - MDLM: Masked Diffusion Language Models — Sahoo et al., 2024
  - LLaDA: Large Language Diffusion Models — Nie et al., 2025
  - DiT: Scalable Diffusion Models — Peebles & Xie, 2023 (AdaLN-Zero)
  - Muse: Text-To-Image via Masked Generative Transformers — Chang et al., 2023
  - Perceiver IO: A General Architecture — Jaegle et al., 2021
  - LCM: Large Concept Models — Meta, 2024
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
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
    logits: Optional[torch.Tensor] = None
    masked_logits: Optional[torch.Tensor] = None
    concept_repr: Optional[torch.Tensor] = None
    noise_level: Optional[torch.Tensor] = None


# ============================================================================
# Timestep embedding
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds a continuous noise level t ∈ [0, 1] into a fixed-dim vector.

    Uses sinusoidal encoding (like positional encoding) followed by a small
    MLP to produce a rich conditioning signal for AdaLN-Zero.
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
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# ============================================================================
# Diffusion decoder layer — cross-attention only, NO self-attention
# ============================================================================

class DiffusionDecoderLayer(nn.Module):
    """
    Perceiver IO-style decoder layer with AdaLN-Zero timestep conditioning.

    Each position independently queries the concept bank via cross-attention.
    NO token-to-token self-attention — the concepts carry all inter-position
    context.

    Complexity per layer: O(N * C * H) instead of O(N² * H).
    With C=128, N=2M: 256M ops vs 4T ops (15,000x cheaper).

    AdaLN-Zero (Peebles & Xie, 2023):
      - Regresses scale, shift, gate from the timestep embedding
      - Gate is zero-initialized so the layer starts as identity
      - Prevents multiplicative instability during early training
    """

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        H = config.hidden_size

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

        # AdaLN-Zero: 6 modulation vectors from timestep
        # [scale_ca, shift_ca, gate_ca, scale_ff, shift_ff, gate_ff]
        self.adaLN = nn.Linear(H, H * 6)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self,
        x: torch.Tensor,
        concepts: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, L, H] — noisy token representations
            concepts: [B, C, H] — concept vectors from encoder
            t_emb:    [B, H]    — timestep embedding
        Returns:
            [B, L, H] updated token representations
        """
        mods = self.adaLN(t_emb).unsqueeze(1)  # [B, 1, 6*H]
        (scale_ca, shift_ca, gate_ca,
         scale_ff, shift_ff, gate_ff) = mods.chunk(6, dim=-1)

        # Cross-attention to concepts (the ONLY attention in this layer)
        x_norm = self.norm_cross(x) * (1 + scale_ca) + shift_ca
        ca_out, _ = self.cross_attn(
            query=x_norm,
            key=concepts,
            value=concepts,
            need_weights=False,
        )
        x = x + gate_ca * ca_out

        # Gated FFN
        x_norm = self.norm_ff(x) * (1 + scale_ff) + shift_ff
        gate_inp, ff_gate = self.ff_in(x_norm).chunk(2, dim=-1)
        ff_out = self.ff_out(self.ff_drop(F.gelu(gate_inp) * ff_gate))
        x = x + gate_ff * ff_out

        return x


# ============================================================================
# Full diffusion decoder
# ============================================================================

class ConceptDiffusionDecoder(nn.Module):
    """
    Perceiver IO-style diffusion decoder with concept cross-attention.

    Takes noisy token ids + concept vectors + timestep and produces hidden
    states at each position.  lm_head is NOT part of the decoder — it lives
    in the model class for sparse logits computation.

    Each decoder layer does cross-attention from token queries to concept
    key/values.  NO self-attention between tokens.

    Input:
        noisy_ids  : [B, L]    — token ids with some positions = mask_token_id
        concepts   : [B, C, H] — concept vectors from the encoder
        t          : [B]       — noise level (0 = clean, 1 = all masked)

    Output:
        hidden     : [B, L, H] — decoder hidden states (NOT logits)
    """

    def __init__(self, config: ConceptEncoderConfig, num_layers: int = 2):
        super().__init__()
        H = config.hidden_size
        token_dim = config.token_embedding_dim

        self.token_embed = nn.Embedding(config.vocab_size, token_dim)
        self.pos_embed = nn.Embedding(config.max_sequence_length, H)

        if token_dim != H:
            self.token_proj = nn.Linear(token_dim, H)
        else:
            self.token_proj = None

        self.t_embed = SinusoidalTimestepEmbedding(H)

        self.layers = nn.ModuleList(
            [DiffusionDecoderLayer(config) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(H)

    def forward(
        self,
        noisy_ids: torch.LongTensor,
        concepts: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, L = noisy_ids.shape

        pos_ids = torch.arange(L, device=noisy_ids.device).unsqueeze(0)
        x = self.token_embed(noisy_ids)
        if self.token_proj is not None:
            x = self.token_proj(x)
        x = x + self.pos_embed(pos_ids)

        t_emb = self.t_embed(t)

        for layer in self.layers:
            x = layer(x, concepts, t_emb)

        return self.out_norm(x)


# ============================================================================
# Full encoder-decoder model
# ============================================================================

class ConceptEncoderForMaskedDiffusion(PreTrainedModel):
    """
    ConceptEncoder + Masked Discrete Diffusion Decoder.

    Training
    --------
    Each forward pass:
      1. Encode CLEAN input tokens → concept vectors   (encoder sees full input)
      2. Sample noise level  t ~ Uniform(t_min, 1.0)
      3. Mask each token independently with probability t
      4. Decode concepts + noisy tokens → hidden states  (cross-attention only)
      5. Compute SPARSE cross-entropy at masked positions only

    The lm_head projection (H → V) is applied only to the ~t*L masked
    positions instead of all L positions.  This saves 2-6x compute on the
    most expensive operation (the vocab-size matmul).

    Inference
    ---------
    Use `generate()` for iterative denoising (K steps from all-[MASK]).

    Args:
        config          : ConceptEncoderConfig
        loss_config     : LossConfig for concept regularization (optional)
        decoder_layers  : Number of cross-attention layers in the decoder
        t_min           : Minimum noise level sampled during training
        label_smoothing : Label smoothing for cross-entropy (prevents overconfident
                          predictions that lead to sharp loss landscapes and
                          gradient explosion)
    """

    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(
        self,
        config: ConceptEncoderConfig,
        loss_config: Optional[LossConfig] = None,
        decoder_layers: int = 2,
        t_min: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__(config)
        self.config = config
        self.t_min = t_min
        self.label_smoothing = label_smoothing

        self.encoder = ConceptEncoder(config)
        self.decoder = ConceptDiffusionDecoder(config, num_layers=decoder_layers)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: independently mask each token with probability t[i].
        Padding positions (attention_mask == 0) are never masked.

        Returns:
            noisy_ids : [B, L] — input_ids with masked positions replaced
            noise_mask: [B, L] — True where tokens were masked
        """
        rand = torch.rand_like(input_ids, dtype=torch.float32)
        noise_mask = rand < t.unsqueeze(1)

        if attention_mask is not None:
            noise_mask = noise_mask & (attention_mask == 1)

        noisy_ids = input_ids.clone()
        noisy_ids[noise_mask] = mask_token_id
        return noisy_ids, noise_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DiffusionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, L = input_ids.shape

        # 1. Encode CLEAN tokens → concept vectors
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
        noisy_ids, noise_mask = self._apply_noise(input_ids, t, mask_token_id, attention_mask)

        # 4. Decode: concept cross-attention → hidden states
        hidden = self.decoder(noisy_ids, concepts, t)  # [B, L, H]

        # 5. SPARSE logits and loss — only at masked positions
        loss = None
        masked_logits = None
        if self.training or noise_mask.any():
            flat_hidden = hidden.reshape(-1, hidden.size(-1))  # [B*L, H]
            flat_mask = noise_mask.reshape(-1)                   # [B*L]
            masked_hidden = flat_hidden[flat_mask]               # [M, H]
            masked_logits = self.lm_head(masked_hidden)          # [M, V]  ← sparse!
            masked_targets = input_ids.reshape(-1)[flat_mask]     # [M]

            if masked_logits.numel() > 0:
                diffusion_loss = F.cross_entropy(
                    masked_logits, masked_targets,
                    label_smoothing=self.label_smoothing,
                )
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
            return (loss, None, concepts, t)

        return DiffusionOutput(
            loss=loss,
            logits=None,
            masked_logits=masked_logits,
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

        If input_ids is given, only [MASK] positions are denoised (infilling).
        If input_ids is None, starts from all-[MASK] of max_sequence_length.

        At each step:
          1. Predict logits for all masked positions (via concept cross-attention)
          2. Sample token from logits
          3. Unmask the most-confident positions
          4. Repeat until no [MASK] tokens remain

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

        encoder_out = self.encoder(current, attention_mask, return_dict=True)
        concepts = encoder_out.last_hidden_state

        for step in range(num_steps):
            still_masked = (current == mask_id)
            n_masked = still_masked.sum(dim=-1).max().item()
            if n_masked == 0:
                break

            t_val = 1.0 - step / num_steps
            t = torch.full((B,), t_val, device=current.device)

            hidden = self.decoder(current, concepts, t)
            logits = self.lm_head(hidden)  # [B, L, V] — full logits for generation

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                values, _ = logits.topk(top_k, dim=-1)
                logits[logits < values[..., -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).reshape(B, L)

            confidence = probs.max(dim=-1).values

            steps_remaining = num_steps - step
            unmask_count = max(1, round(n_masked / steps_remaining))

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

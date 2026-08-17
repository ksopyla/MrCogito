"""Pretrained-backbone concept memory (E10/E16/E17 and E17c working memory).

Grafts the MrCogito concept machinery onto a frozen pretrained decoder (Gemma-3 family):

  * READ  — the backbone's *global* attention layers (Gemma-3 interleaves 5 sliding-window
    layers : 1 global layer) lose their full-attention reach (all token↔token attention is
    windowed) and gain a gated cross-attention read of C concept slots. Legacy modes reuse
    Gemma's projections; E17c selects depth-private dedicated projections.
  * WRITE — after each K-token block (K = the backbone's sliding window), the concept state
    is updated from block hidden states through BiXT. Legacy modes use additive scalar-gated
    writes; E17c uses untied depth-private, content-gated replacement cells.
  * RECURRENT ENCODE == RECURRENT DECODE — there is no separate encoder: any input is
    consumed block-by-block through the same write op, so input length is unbounded at
    fixed memory, O(N·(K+C)) total.

Zero-init property: with all read gates and the write gate at 0 the graft is inert — the block
loop equals plain Gemma with every layer window-masked exactly for the first two blocks (RoPE
is relative, positions reset per block, one-block carry), and beyond that it truncates history
HARDER than a full-sequence windowed forward (stacked SWA layers widen the receptive field by
~(W-1) per layer; the block loop caps it at carry+block). That truncated context is exactly
what the concepts must supply. `concept_num=0` skips the concept machinery entirely and is the
matched training control arm (identical block protocol → clean A/B attribution).

Specs: docs/experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md and
docs/experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md
(+ their _plan.md files).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from nn.concept_encoder import BiXTCrossAttention
from nn.concept_encoder_perceiver import ChunkedLMHeadCE

IGNORE_INDEX = -100
# Intra-block CE bins named after the K=512 protocol; offsets scale with concept_block.
INTRA_BLOCK_BINS = (
    ("0_64", 0.0, 0.125),
    ("64_128", 0.125, 0.25),
    ("128_256", 0.25, 0.5),
    ("256_512", 0.5, 1.0),
)


class BackboneConceptConfig(PretrainedConfig):
    """Config for `BackboneConceptLM`.

    `backbone_config` (a plain dict of the backbone's HF config) makes the model structure
    reconstructible without hub access: `BackboneConceptLM(config)` always builds the
    backbone *structure* from it (random weights), and `from_pretrained_backbone` fills the
    weights from the hub for the initial training run. Checkpoints saved by the trainer
    contain the full (backbone + graft) state and round-trip through `from_pretrained`.
    """

    model_type = "backbone_concept"

    def __init__(
        self,
        backbone_model: str = "google/gemma-3-1b-pt",
        backbone_config: Optional[dict] = None,
        concept_num: int = 128,
        concept_block: int = 512,
        concept_io_mode: str = "global_kv",   # E16: "shared_depth_recurrent"
        write_num_heads: int = 4,
        read_concept_norm: bool = False,
        read_gate_init: float = 0.0,
        write_gate_init: float = 0.0,
        concept_read_mode: str = "backbone_qkv",
        tie_concept_writer: bool = True,
        concept_write_mode: str = "additive",
        write_update_gate_init: float = 0.25,
        concept_read_placement: str = "post_layer",
        inference_carry_policy: str = "normal",
        memory_carry_dropout: float = 0.0,
        memory_pressure_tokens: int = 0,
        memory_pressure_weight: float = 1.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: str = "q_proj,k_proj,v_proj,o_proj",
        global_attention_mode: str = "windowed",   # "full" = intact backbone (Stage-0 upper baseline)
        checkpoint_family: str = "backbone_concept",
        tokenizer_name: Optional[str] = None,
        **kwargs,
    ):
        self.backbone_model = backbone_model
        self.backbone_config = backbone_config
        self.concept_num = concept_num
        self.concept_block = concept_block
        self.concept_io_mode = concept_io_mode
        self.write_num_heads = write_num_heads
        self.read_concept_norm = read_concept_norm
        self.read_gate_init = read_gate_init
        self.write_gate_init = write_gate_init
        self.concept_read_mode = concept_read_mode
        self.concept_read_placement = concept_read_placement
        self.tie_concept_writer = tie_concept_writer
        self.concept_write_mode = concept_write_mode
        self.write_update_gate_init = write_update_gate_init
        self.inference_carry_policy = inference_carry_policy
        self.memory_carry_dropout = memory_carry_dropout
        self.memory_pressure_tokens = memory_pressure_tokens
        self.memory_pressure_weight = memory_pressure_weight
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets
        self.global_attention_mode = global_attention_mode
        self.checkpoint_family = checkpoint_family
        self.tokenizer_name = tokenizer_name
        # Facade: mirror the backbone's headline dims onto this config so the shared
        # training/wandb plumbing (which was written for the flat concept-encoder config
        # and reads config.hidden_size / num_attention_heads / etc. unconditionally) gets
        # real values instead of AttributeError. The authoritative backbone config stays
        # in self.backbone_config; these are the few fields the shared code touches.
        bb = backbone_config or {}
        self.hidden_size = bb.get("hidden_size")
        self.num_hidden_layers = bb.get("num_hidden_layers")
        self.num_attention_heads = bb.get("num_attention_heads")
        self.intermediate_size = bb.get("intermediate_size")
        self.vocab_size = bb.get("vocab_size")
        self.token_embedding_dim = bb.get("hidden_size")   # backbone embedding == hidden
        self.max_sequence_length = bb.get("max_position_embeddings")
        self.head_dim = bb.get("head_dim")
        self.sliding_window = bb.get("sliding_window")
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)

    def sync_backbone_facade(self, backbone_config: dict) -> None:
        """Refresh shared-plumbing dimensions after loading the hub backbone config."""
        self.backbone_config = backbone_config
        self.hidden_size = backbone_config.get("hidden_size")
        self.num_hidden_layers = backbone_config.get("num_hidden_layers")
        self.num_attention_heads = backbone_config.get("num_attention_heads")
        self.intermediate_size = backbone_config.get("intermediate_size")
        self.vocab_size = backbone_config.get("vocab_size")
        self.token_embedding_dim = backbone_config.get("hidden_size")
        self.max_sequence_length = backbone_config.get("max_position_embeddings")
        self.head_dim = backbone_config.get("head_dim")
        self.sliding_window = backbone_config.get("sliding_window")


class ConceptReadBranch(nn.Module):
    """Cross-attention read from ``z [B,C,H]`` into tokens ``x [B,Q,H]``.

    ``backbone_qkv`` preserves the legacy Gemma-projection path exactly. ``dedicated``
    owns Q/K/V/O projections and per-head Q/K norms, giving each wrapped global layer
    an independent concept-read representation space. Neither path applies RoPE because
    concept slots are a position-free set.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        mode: str = "backbone_qkv",
        num_heads: int = 4,
        normalize_concepts: bool = False,
        rms_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if mode not in {"backbone_qkv", "dedicated"}:
            raise ValueError(
                f"Unknown concept read mode {mode!r}; expected 'backbone_qkv' or 'dedicated'."
            )
        if mode == "dedicated" and hidden_size % num_heads:
            raise ValueError(
                f"Dedicated concept read hidden_size={hidden_size} must divide "
                f"num_heads={num_heads}."
            )
        self.concept_norm = (
            nn.RMSNorm(hidden_size, eps=rms_norm_eps)
            if normalize_concepts
            else nn.Identity()
        )
        if mode == "dedicated":
            self.query_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
            self.q_norm = nn.RMSNorm(hidden_size // num_heads, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(hidden_size // num_heads, eps=rms_norm_eps)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.query_norm = None
            self.q_proj = None
            self.kv_proj = None
            self.q_norm = None
            self.k_norm = None
            self.o_proj = None

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        attn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        B, Q, _ = x.shape
        C = z.shape[1]
        if self.mode == "dedicated":
            hd = self.hidden_size // self.num_heads
            x_read = self.query_norm(x)
            z_read = self.concept_norm(z).to(x_read.dtype)
            q = self.q_proj(x_read).view(B, Q, self.num_heads, hd).transpose(1, 2)
            kv = self.kv_proj(z_read)
            k, v = kv.chunk(2, dim=-1)
            k = k.view(B, C, self.num_heads, hd).transpose(1, 2)
            v = v.view(B, C, self.num_heads, hd).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            o = F.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
            o = o.transpose(1, 2).reshape(B, Q, self.hidden_size)
            return self.o_proj(o)

        if attn is None:
            raise ValueError("backbone_qkv concept read requires the wrapped attention module.")
        hd = attn.head_dim
        z_read = self.concept_norm(z).to(x.dtype)
        q = attn.q_proj(x).view(B, Q, -1, hd).transpose(1, 2)   # [B, nH, Q, hd]
        k = attn.k_proj(z_read).view(B, C, -1, hd).transpose(1, 2)     # [B, nKV, C, hd]
        v = attn.v_proj(z_read).view(B, C, -1, hd).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if attn.num_key_value_groups > 1:
            k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
            v = v.repeat_interleave(attn.num_key_value_groups, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
        o = o.transpose(1, 2).reshape(B, Q, -1)
        return attn.o_proj(o)


class _AttnWithConceptResidual(nn.Module):
    """Add a concept cross-attn mix onto the token-attention output (before FFN).

    Owns only the original token attention. Read weights stay on the parent
    ``GlobalLayerWithConceptRead`` so state_dict keys remain ``*.read_branch.*`` /
    ``*.gate``. The parent is stored without ``nn.Module`` registration (a cycle
    would otherwise form: wrapper ⊂ layer ⊂ parent).
    """

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.original_attn = original_attn
        # Gemma3DecoderLayer reads this on ``self.self_attn`` before calling it.
        self.is_sliding = getattr(original_attn, "is_sliding", False)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_attn, name)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.original_attn(hidden_states, *args, **kwargs)
        owner = getattr(self, "_owner", None)
        z = None if owner is None else owner._read_z
        if z is None:
            return out
        tok = out[0] if isinstance(out, tuple) else out
        read = owner.read_branch(
            hidden_states, z.to(hidden_states.dtype), self.original_attn
        )
        mixed = tok + torch.tanh(owner.gate) * read
        if isinstance(out, tuple):
            return (mixed,) + tuple(out[1:])
        return mixed


class GlobalLayerWithConceptRead(nn.Module):
    """Wraps one of the backbone's global decoder layers with a tanh-gated concept read.

    ``post_layer`` (default, E17c): run the original layer, then add the concept mix
    after FFN. ``attn_residual`` (E17d): wrap ``layer.self_attn`` so the mix lands in
    the attention residual and the FFN sees the assimilated stream. The concept state
    arrives through a shared mutable holder (E10) or an explicit ``concept_state``
    argument (E16/E17) so checkpoint replay never observes a later recurrent state.
    """

    def __init__(
        self,
        layer: nn.Module,
        state_holder: dict,
        gate_init: float,
        *,
        hidden_size: int,
        read_mode: str = "backbone_qkv",
        read_placement: str = "post_layer",
        read_num_heads: int = 4,
        normalize_concepts: bool = False,
        rms_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        if read_placement not in {"post_layer", "attn_residual"}:
            raise ValueError(
                f"Unknown concept_read_placement {read_placement!r}; "
                "expected 'post_layer' or 'attn_residual'."
            )
        self.layer = layer
        self.attention_type = layer.attention_type   # read by Gemma3TextModel.forward's mask routing
        self.read_placement = read_placement
        self._state = state_holder                    # plain dict, not a submodule
        self._read_z = None
        self.read_branch = ConceptReadBranch(
            hidden_size,
            mode=read_mode,
            num_heads=read_num_heads,
            normalize_concepts=normalize_concepts,
            rms_norm_eps=rms_norm_eps,
        )
        if normalize_concepts or read_mode == "dedicated":
            reference = layer.input_layernorm.weight
            self.read_branch.to(device=reference.device, dtype=reference.dtype)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        if read_placement == "attn_residual":
            wrapper = _AttnWithConceptResidual(layer.self_attn)
            layer.self_attn = wrapper
            # Avoid registering this wrapper's parent (cycle).
            object.__setattr__(wrapper, "_owner", self)

    _STATE_UNSET = object()

    def _resolve_read_state(self, concept_state):
        if concept_state is self._STATE_UNSET:
            z = self._state.get("z")
        else:
            z = concept_state
        if z is None:
            return None
        # Holder-based ablations (shared/global_kv). Banked execution permutes the
        # tensor before passing ``concept_state`` and leaves these flags unset.
        permutation = self._state.get("permutation")
        if permutation is not None:
            z = z.index_select(0, permutation)
        elif self._state.get("shuffle"):
            z = torch.roll(z, shifts=1, dims=0)
        return z

    def forward(
        self,
        hidden_states,
        *args,
        concept_state=_STATE_UNSET,
        **kwargs,
    ):
        z = self._resolve_read_state(concept_state)
        if self.read_placement == "attn_residual":
            self._read_z = z
            try:
                return self.layer(hidden_states, *args, **kwargs)
            finally:
                self._read_z = None

        outputs = self.layer(hidden_states, *args, **kwargs)
        if z is not None:
            # E17c's dedicated branch queries the post-layer representation. The legacy
            # branch deliberately keeps its pre-layer normalized query for exact E17b numerics.
            x = (
                outputs[0]
                if self.read_branch.mode == "dedicated"
                else self.layer.input_layernorm(hidden_states)
            )
            read = self.read_branch(x, z.to(x.dtype), self.layer.self_attn)
            hidden = outputs[0] + torch.tanh(self.gate) * read
            outputs = (hidden,) + tuple(outputs[1:])
        return outputs


class ConceptWriteHead(nn.Module):
    """BiXT concept writer with additive or selective-replacement dynamics.

    The BiXT and normalization weights are always shared. ``global_kv`` owns the
    checkpoint-compatible scalar ``alpha``; ``shared_depth_recurrent`` owns one
    ``depth_alphas`` entry per discovered global layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        gate_init: float = 0.0,
        num_depth_gates: int = 0,
        update_mode: str = "additive",
        update_gate_init: float = 0.25,
    ):
        super().__init__()
        if update_mode not in {"additive", "gated_replace"}:
            raise ValueError(
                f"Unknown concept write mode {update_mode!r}; expected additive or gated_replace."
            )
        if not 0.0 < update_gate_init < 1.0:
            raise ValueError("update_gate_init must lie strictly between 0 and 1.")
        self.update_mode = update_mode
        self.bixt = BiXTCrossAttention(
            dim_lat=hidden_size, dim_tok=hidden_size, dim_attn=hidden_size,
            num_heads=num_heads, update_tokens=False,
        )
        self.norm_lat = nn.RMSNorm(hidden_size)
        self.norm_tok = nn.RMSNorm(hidden_size)
        self.sandwich = nn.RMSNorm(hidden_size)   # Ouro-style anti-collapse post-norm
        if update_mode == "gated_replace":
            self.register_parameter("alpha", None)
            self.register_parameter("depth_alphas", None)
            self.update_gate = nn.Linear(2 * hidden_size, 1)
            nn.init.zeros_(self.update_gate.weight)
            nn.init.constant_(
                self.update_gate.bias,
                math.log(update_gate_init / (1.0 - update_gate_init)),
            )
        elif num_depth_gates:
            self.register_parameter("alpha", None)
            self.depth_alphas = nn.Parameter(
                torch.full((num_depth_gates,), float(gate_init))
            )
        else:
            self.alpha = nn.Parameter(torch.tensor(float(gate_init)))
            self.register_parameter("depth_alphas", None)
            self.update_gate = None
        self._last_update_gate_mean: Optional[float] = None
        self._last_update_rms: Optional[float] = None
        self._last_state_rms: Optional[float] = None

    def forward(
        self,
        z: torch.Tensor,
        h_block: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        depth_index: Optional[int] = None,
    ) -> torch.Tensor:
        # pad_mask: [B, Kb] bool, True = padding.
        if self.update_mode == "gated_replace":
            if depth_index is not None:
                raise ValueError("depth_index is invalid for gated replacement writes.")
            valid_row = (~pad_mask).any(dim=1)
            safe_pad = pad_mask.clone()
            safe_pad[~valid_row, 0] = False
            lat, _ = self.bixt(
                self.norm_lat(z),
                self.norm_tok(h_block).to(z.dtype),
                key_padding_mask=safe_pad,
            )
            # Autocast emits BF16 attention/norm outputs while the learned concept
            # state intentionally remains FP32. ``torch.lerp`` requires one dtype.
            candidate = self.sandwich(lat).to(z.dtype)
            gate_input = torch.cat([self.norm_lat(z), candidate], dim=-1)
            update_gate = torch.sigmoid(self.update_gate(gate_input)).to(z.dtype)
            next_z = torch.lerp(z, candidate, update_gate)
            next_z = torch.where(valid_row.view(-1, 1, 1), next_z, z)
            with torch.no_grad():
                valid_gate = update_gate[valid_row]
                self._last_update_gate_mean = (
                    float(valid_gate.mean().item()) if valid_gate.numel() else 0.0
                )
                delta = next_z - z
                self._last_update_rms = float(delta.float().square().mean().sqrt().item())
                self._last_state_rms = float(z.float().square().mean().sqrt().item())
            return next_z

        if self.depth_alphas is None:
            if depth_index is not None:
                raise ValueError("depth_index is only valid for depth-gated writes.")
            alpha = self.alpha
        else:
            if depth_index is None:
                raise ValueError("depth_index is required for depth-gated writes.")
            alpha = self.depth_alphas[depth_index]
        valid_row = (~pad_mask).any(dim=1)                     # [B]
        safe_pad = pad_mask.clone()
        safe_pad[~valid_row, 0] = False                        # avoid all -inf softmax rows
        lat, _ = self.bixt(
            self.norm_lat(z), self.norm_tok(h_block).to(z.dtype), key_padding_mask=safe_pad
        )
        update = torch.tanh(alpha) * self.sandwich(lat)
        update = update * valid_row.view(-1, 1, 1).to(update.dtype)
        return z + update


class BackboneConceptLM(PreTrainedModel):
    """Frozen pretrained decoder + LoRA + config-selected shared concept memory."""

    config_class = BackboneConceptConfig
    base_model_prefix = "backbone_concept"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["backbone.lm_head.weight"]

    def __init__(self, config: BackboneConceptConfig, backbone: Optional[nn.Module] = None):
        super().__init__(config)
        valid_io_modes = {"global_kv", "shared_depth_recurrent", "per_layer_banks"}
        if config.concept_io_mode not in valid_io_modes:
            raise NotImplementedError(
                f"concept_io_mode={config.concept_io_mode!r} is not implemented; "
                f"expected one of {sorted(valid_io_modes)}."
            )
        if config.concept_read_mode not in {"backbone_qkv", "dedicated"}:
            raise ValueError(
                "concept_read_mode must be 'backbone_qkv' or 'dedicated', got "
                f"{config.concept_read_mode!r}."
            )
        if config.concept_write_mode not in {"additive", "gated_replace"}:
            raise ValueError(
                "concept_write_mode must be 'additive' or 'gated_replace', got "
                f"{config.concept_write_mode!r}."
            )
        if not 0.0 <= config.memory_carry_dropout <= 1.0:
            raise ValueError("memory_carry_dropout must be in [0, 1].")
        if not 0 <= config.memory_pressure_tokens <= config.concept_block:
            raise ValueError("memory_pressure_tokens must be in [0, concept_block].")
        if config.memory_pressure_weight < 1.0:
            raise ValueError("memory_pressure_weight must be >= 1.")
        if not 0.0 < config.write_update_gate_init < 1.0:
            raise ValueError("write_update_gate_init must lie strictly between 0 and 1.")
        if config.concept_read_placement not in {"post_layer", "attn_residual"}:
            raise ValueError(
                "concept_read_placement must be 'post_layer' or 'attn_residual', got "
                f"{config.concept_read_placement!r}."
            )
        if config.inference_carry_policy not in {"normal", "drop_after_first"}:
            raise ValueError(
                "inference_carry_policy must be 'normal' or 'drop_after_first', got "
                f"{config.inference_carry_policy!r}."
            )
        pressure_active = (
            config.memory_carry_dropout > 0.0
            or config.memory_pressure_tokens > 0
            or config.memory_pressure_weight != 1.0
        )
        if config.memory_carry_dropout == 0.0 and (
            config.memory_pressure_tokens != 0
            or config.memory_pressure_weight != 1.0
        ):
            raise ValueError(
                "memory_pressure_tokens/weight must be inactive when "
                "memory_carry_dropout is zero."
            )
        e17c_feature_active = (
            config.concept_read_mode == "dedicated"
            or not config.tie_concept_writer
            or config.concept_write_mode == "gated_replace"
            or pressure_active
        )
        if e17c_feature_active and (
            config.concept_io_mode != "per_layer_banks" or config.concept_num <= 0
        ):
            raise ValueError(
                "Dedicated reads, untied/gated writers, and memory pressure require "
                "concept_io_mode='per_layer_banks' with concept_num > 0."
            )
        if backbone is None:
            if not config.backbone_config:
                raise ValueError(
                    "config.backbone_config is required to build the backbone structure "
                    "(use BackboneConceptLM.from_pretrained_backbone for the hub-weights path)."
                )
            from transformers import Gemma3ForCausalLM, Gemma3TextConfig
            backbone = Gemma3ForCausalLM(Gemma3TextConfig(**config.backbone_config))
        self.backbone = backbone
        bb_cfg = self.backbone.config
        config.sync_backbone_facade(bb_cfg.to_dict())
        if getattr(bb_cfg, "final_logit_softcapping", None):
            raise ValueError("ChunkedLMHeadCE path assumes no final_logit_softcapping.")
        if config.concept_block != bb_cfg.sliding_window:
            raise ValueError(
                f"concept_block ({config.concept_block}) must equal the backbone's sliding "
                f"window ({bb_cfg.sliding_window}) so one mask serves both layer types and "
                "the zero-init equivalence holds."
            )
        self.hidden_size = bb_cfg.hidden_size

        # --- freeze the backbone, then inject LoRA (adapter params come back trainable) ---
        self.backbone.requires_grad_(False)
        if config.lora_r and config.lora_r > 0:
            from peft import LoraConfig, inject_adapter_in_model
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=[t.strip() for t in config.lora_targets.split(",") if t.strip()],
                bias="none",
            )
            self.backbone = inject_adapter_in_model(lora_cfg, self.backbone)

        # --- concept machinery (skipped entirely for the concept_num=0 control arm) ---
        self._concept_state: dict = {
            "z": None,
            "shuffle": False,
            "permutation": None,
        }
        layers = self.backbone.model.layers
        self.global_layer_indices = tuple(
            i for i, layer in enumerate(layers)
            if layer.attention_type == "full_attention"
        )
        if config.concept_num > 0:
            if not self.global_layer_indices:
                raise ValueError(
                    "No 'full_attention' layers found in the backbone — the concept read "
                    "graft would be a silent no-op. Check the backbone's layer_types "
                    f"(got: {[layer.attention_type for layer in layers]})."
                )
            num_global = len(self.global_layer_indices)
            # per_layer_banks gives each global layer its OWN concept bank, so the learned
            # initial state is [G, C, H] (one init per bank); shared modes keep the single
            # [C, H] init. The machinery (tied ConceptWriteHead, gates) is identical.
            per_layer = config.concept_io_mode == "per_layer_banks"
            init_shape = (
                (num_global, config.concept_num, self.hidden_size)
                if per_layer
                else (config.concept_num, self.hidden_size)
            )
            self.concept_init = nn.Parameter(
                torch.randn(*init_shape) * (self.hidden_size ** -0.5)
            )
            if config.tie_concept_writer:
                self.write_head = ConceptWriteHead(
                    self.hidden_size,
                    config.write_num_heads,
                    gate_init=config.write_gate_init,
                    # Legacy per-bank/depth writes share one writer with one scalar per depth.
                    num_depth_gates=(
                        num_global
                        if (
                            config.concept_write_mode == "additive"
                            and config.concept_io_mode
                            in ("shared_depth_recurrent", "per_layer_banks")
                        )
                        else 0
                    ),
                    update_mode=config.concept_write_mode,
                    update_gate_init=config.write_update_gate_init,
                )
                self.write_heads = None
            else:
                self.write_head = None
                self.write_heads = nn.ModuleList(
                    ConceptWriteHead(
                        self.hidden_size,
                        config.write_num_heads,
                        gate_init=config.write_gate_init,
                        update_mode=config.concept_write_mode,
                        update_gate_init=config.write_update_gate_init,
                    )
                    for _ in range(num_global)
                )
            for i, layer in enumerate(layers):
                if layer.attention_type == "full_attention":
                    layers[i] = GlobalLayerWithConceptRead(
                        layer,
                        self._concept_state,
                        config.read_gate_init,
                        hidden_size=self.hidden_size,
                        read_mode=config.concept_read_mode,
                        read_placement=config.concept_read_placement,
                        read_num_heads=config.write_num_heads,
                        normalize_concepts=config.read_concept_norm,
                        rms_norm_eps=getattr(bb_cfg, "rms_norm_eps", None),
                    )
        else:
            self.concept_init = None
            self.write_head = None
            self.write_heads = None

        self._last_pressure_fraction = 0.0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained_backbone(cls, config: BackboneConceptConfig, **backbone_kwargs):
        """Initial-training path: load the backbone weights from the hub, then graft."""
        from transformers import Gemma3ForCausalLM
        backbone = Gemma3ForCausalLM.from_pretrained(config.backbone_model, **backbone_kwargs)
        config.sync_backbone_facade(backbone.config.to_dict())
        return cls(config, backbone=backbone)

    def _init_weights(self, module):
        # Only reached for missing keys during from_pretrained; keep it conservative.
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.backbone.model.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.backbone.lm_head

    # ------------------------------------------------------------------ plumbing
    @property
    def has_concepts(self) -> bool:
        return self.concept_init is not None

    def _writer_for_depth(self, depth_index: int) -> ConceptWriteHead:
        """Select a tied legacy writer or depth-private E17c writer."""
        if self.write_heads is not None:
            return self.write_heads[depth_index]
        if self.write_head is None:
            raise RuntimeError("Concept writer requested for a model without concepts.")
        return self.write_head

    def _resolve_carry_policy(self, carry_policy: Optional[str] = None) -> str:
        """Explicit ``carry_policy`` wins; else train keeps Bernoulli dropout, eval uses config."""
        if carry_policy is None:
            if self.training:
                carry_policy = "normal"
            else:
                carry_policy = getattr(self.config, "inference_carry_policy", "normal")
        if carry_policy not in {"normal", "drop_after_first"}:
            raise ValueError(
                f"Unknown carry_policy={carry_policy!r}; expected normal or drop_after_first."
            )
        return carry_policy

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            # Reentrant checkpointing registers a backward hook each time a shared LoRA
            # parameter is used. E10 reuses each backbone layer across sequence blocks, so
            # DDP sees the same LoRA parameter marked ready multiple times. The modern
            # non-reentrant path supports this shared-parameter recurrence correctly.
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        # Frozen embeddings would otherwise feed requires_grad=False inputs to the
        # checkpointed layers, silently disabling recomputation gradients.
        self.backbone.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def _windowed_causal_mask(self, dec_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """[B,1,Q,Q] additive float mask: causal ∧ (q−kv < window) ∧ (kv non-pad ∨ kv==q).

        The kv==q escape keeps fully-padded query rows finite (their outputs are discarded
        via labels=-100); matches HF sliding-window semantics (dist ∈ [0, W-1])."""
        B, Q = dec_mask.shape
        device = dec_mask.device
        idx = torch.arange(Q, device=device)
        dist = idx.view(-1, 1) - idx.view(1, -1)                       # q - kv
        allowed = (dist >= 0) & (dist < self.config.concept_block)     # [Q, Q]
        kv_ok = dec_mask.bool().view(B, 1, 1, Q)
        diag = torch.eye(Q, dtype=torch.bool, device=device).view(1, 1, Q, Q)
        allowed = allowed.view(1, 1, Q, Q) & (kv_ok | diag)
        mask = torch.zeros(B, 1, Q, Q, dtype=dtype, device=device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _full_causal_mask(self, dec_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, Q = dec_mask.shape
        device = dec_mask.device
        idx = torch.arange(Q, device=device)
        allowed = (idx.view(-1, 1) - idx.view(1, -1)) >= 0
        kv_ok = dec_mask.bool().view(B, 1, 1, Q)
        diag = torch.eye(Q, dtype=torch.bool, device=device).view(1, 1, Q, Q)
        allowed = allowed.view(1, 1, Q, Q) & (kv_ok | diag)
        mask = torch.zeros(B, 1, Q, Q, dtype=dtype, device=device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _lm_ce_sum(self, pred_hidden: torch.Tensor, targets: torch.Tensor):
        """Token-summed next-token CE without materializing [B,*,V] in the autograd graph."""
        count = (targets != IGNORE_INDEX).sum()
        if count == 0:
            return pred_hidden.new_zeros((), dtype=torch.float32), count
        mean = ChunkedLMHeadCE.apply(
            pred_hidden, self.backbone.lm_head.weight, targets, 256, IGNORE_INDEX
        )
        return mean * count, count

    @torch.no_grad()
    def _per_position_metrics_from_hidden(self, pred_hidden, targets, chunk: int = 256):
        """Eval-only CE/top-1 at targeted positions without dense ``[B,T,V]`` logits."""
        B, T, _ = pred_hidden.shape
        out_ce = pred_hidden.new_full((B, T), float("nan"), dtype=torch.float32)
        out_predictions = targets.new_full((B, T), IGNORE_INDEX)
        valid = targets != IGNORE_INDEX
        if not valid.any():
            return out_ce, out_predictions

        selected_hidden = pred_hidden[valid]
        selected_targets = targets[valid]
        selected_ce = pred_hidden.new_empty(
            selected_targets.shape, dtype=torch.float32
        )
        selected_predictions = selected_targets.new_empty(selected_targets.shape)
        weight = self.backbone.lm_head.weight
        for s in range(0, len(selected_targets), chunk):
            e = min(s + chunk, len(selected_targets))
            logits = F.linear(selected_hidden[s:e], weight).float()
            selected_ce[s:e] = F.cross_entropy(
                logits,
                selected_targets[s:e],
                reduction="none",
            )
            selected_predictions[s:e] = logits.argmax(dim=-1)
        out_ce[valid] = selected_ce
        out_predictions[valid] = selected_predictions
        return out_ce, out_predictions

    @torch.no_grad()
    def _per_position_ce_from_hidden(self, pred_hidden, targets, chunk: int = 256):
        """[B, T] CE per target position (nan where label ignored). Eval-only."""
        return self._per_position_metrics_from_hidden(
            pred_hidden, targets, chunk=chunk
        )[0]

    # ------------------------------------------------------------------ core block loop
    def _forward_shared_depth_block(
        self,
        inputs_embeds: torch.Tensor,
        attention_masks: dict[str, torch.Tensor],
        z: Optional[torch.Tensor],
        *,
        block_len: int,
        block_pad_mask: torch.Tensor,
        concept_mode: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute one Gemma block and interleave tied concept writes.

        Shapes: token embeddings ``[B,Q,H]`` and state ``[B,C,H]`` produce final
        token states ``[B,Q,H]`` plus the refined shared state ``[B,C,H]``.
        This mirrors the no-cache Gemma3TextModel layer loop, while passing ``z``
        explicitly to concept reads and mutating it only after layer forwards return.
        """
        text_model = self.backbone.model
        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings_global = text_model.rotary_emb(
            hidden_states, position_ids
        )
        position_embeddings_local = text_model.rotary_emb_local(
            hidden_states, position_ids
        )

        depth_index = 0
        for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
            layer_kwargs = {
                "position_embeddings_global": position_embeddings_global,
                "position_embeddings_local": position_embeddings_local,
                "attention_mask": attention_masks[decoder_layer.attention_type],
                "position_ids": position_ids,
                "past_key_values": None,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
            }
            if isinstance(decoder_layer, GlobalLayerWithConceptRead):
                layer_outputs = decoder_layer(
                    hidden_states,
                    concept_state=z,
                    **layer_kwargs,
                )
            else:
                layer_outputs = decoder_layer(hidden_states, **layer_kwargs)
            hidden_states = layer_outputs[0]

            if isinstance(decoder_layer, GlobalLayerWithConceptRead):
                if z is not None and concept_mode != "static":
                    write_base = (
                        self.concept_init.unsqueeze(0).expand(z.shape[0], -1, -1)
                        if concept_mode == "one_block"
                        else z
                    )
                    z = self.write_head(
                        write_base,
                        hidden_states[:, -block_len:],
                        block_pad_mask,
                        depth_index=depth_index,
                    )
                depth_index += 1

        if depth_index != len(self.global_layer_indices):
            raise RuntimeError(
                "Shared-depth execution saw a different number of global layers "
                f"({depth_index}) than construction ({len(self.global_layer_indices)})."
            )
        return text_model.norm(hidden_states), z

    def _forward_per_layer_banks_block(
        self,
        inputs_embeds: torch.Tensor,
        attention_masks: dict[str, torch.Tensor],
        z_banks: Optional[torch.Tensor],
        *,
        block_len: int,
        block_pad_mask: torch.Tensor,
        concept_mode: str,
        concept_permutation: Optional[torch.Tensor] = None,
        concept_bank_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One Gemma block where each global layer reads AND writes its OWN concept bank.

        Mirrors ``_forward_shared_depth_block`` but the state ``z_banks`` is ``[B, G, C, H]``
        (G private banks, one per global layer). Global layer g reads ``z_banks[:, g]`` and
        writes back to ``z_banks[:, g]`` through the configured tied or depth-private
        ``ConceptWriteHead``; it never sees another bank. Sliding layers are unchanged.

        Because each bank is touched by exactly one layer, the per-bank writes are
        independent within a block — bank updates are accumulated by list reassignment
        (autograd-safe) and re-stacked at the end. ``z_banks=None`` (concept_mode='zero')
        runs the block with no concept reads/writes and returns ``None``.
        """
        text_model = self.backbone.model
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings_global = text_model.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = text_model.rotary_emb_local(hidden_states, position_ids)

        G = len(self.global_layer_indices)
        banks = list(z_banks.unbind(dim=1)) if z_banks is not None else [None] * G
        depth_index = 0
        for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
            layer_kwargs = {
                "position_embeddings_global": position_embeddings_global,
                "position_embeddings_local": position_embeddings_local,
                "attention_mask": attention_masks[decoder_layer.attention_type],
                "position_ids": position_ids,
                "past_key_values": None,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
            }
            if isinstance(decoder_layer, GlobalLayerWithConceptRead):
                z_g = banks[depth_index]                       # this layer's private bank [B, C, H]
                z_read = z_g
                ablate_bank = concept_bank_index is None or concept_bank_index == depth_index
                if z_read is not None and ablate_bank:
                    if concept_mode == "permutation":
                        z_read = z_read.index_select(0, concept_permutation)
                    elif concept_mode == "shuffle":
                        z_read = torch.roll(z_read, shifts=1, dims=0)
                layer_outputs = decoder_layer(hidden_states, concept_state=z_read, **layer_kwargs)
                hidden_states = layer_outputs[0]
                if z_g is not None and concept_mode != "static":
                    write_base = (
                        self.concept_init[depth_index].unsqueeze(0).expand_as(z_g)
                        if concept_mode == "one_block"
                        else z_g
                    )
                    writer = self._writer_for_depth(depth_index)
                    banks[depth_index] = writer(
                        write_base,
                        hidden_states[:, -block_len:],
                        block_pad_mask,
                        depth_index=(
                            depth_index
                            if writer.depth_alphas is not None
                            else None
                        ),
                    )
                depth_index += 1
            else:
                layer_outputs = decoder_layer(hidden_states, **layer_kwargs)
                hidden_states = layer_outputs[0]

        if depth_index != G:
            raise RuntimeError(
                "per_layer_banks execution saw a different number of global layers "
                f"({depth_index}) than construction ({G})."
            )
        out_state = None if z_banks is None else torch.stack(banks, dim=1)  # [B, G, C, H]
        return text_model.norm(hidden_states), out_state

    def _forward_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_mode: str = "real",  # real | shuffle | zero | static | one_block | permutation
        concept_permutation: Optional[torch.Tensor] = None,
        carry_policy: Optional[str] = None,
        concept_bank_index: Optional[int] = None,
        per_position: bool = False,
        return_predictions: bool = False,
        return_last_hidden: bool = False,
        initial_concepts: Optional[torch.Tensor] = None,
    ):
        valid_modes = {"real", "shuffle", "zero", "static", "one_block", "permutation"}
        if concept_mode not in valid_modes:
            raise ValueError(
                f"Unknown concept_mode={concept_mode!r}; expected one of {sorted(valid_modes)}."
            )
        carry_policy = self._resolve_carry_policy(carry_policy)
        B, S = input_ids.shape
        if return_predictions and not per_position:
            raise ValueError("return_predictions=True requires per_position=True.")
        if return_last_hidden and (per_position or return_predictions):
            raise ValueError(
                "return_last_hidden cannot be combined with per_position/return_predictions."
            )
        if concept_mode == "permutation":
            if concept_permutation is None:
                raise ValueError(
                    "concept_mode='permutation' requires concept_permutation."
                )
            concept_permutation = concept_permutation.to(
                device=input_ids.device, dtype=torch.long
            )
            if concept_permutation.shape != (B,):
                raise ValueError(
                    f"concept_permutation must have shape ({B},), got "
                    f"{tuple(concept_permutation.shape)}."
                )
            if not torch.equal(
                concept_permutation.sort().values,
                torch.arange(B, device=input_ids.device),
            ):
                raise ValueError("concept_permutation must be a bijection over the batch.")
        elif concept_permutation is not None:
            raise ValueError(
                "concept_permutation is only valid with concept_mode='permutation'."
            )
        if concept_bank_index is not None:
            if self.config.concept_io_mode != "per_layer_banks":
                raise ValueError("concept_bank_index requires per_layer_banks mode.")
            if concept_mode not in {"shuffle", "permutation"}:
                raise ValueError(
                    "concept_bank_index is only valid for shuffle/permutation ablations."
                )
            if not 0 <= concept_bank_index < len(self.global_layer_indices):
                raise ValueError(
                    f"concept_bank_index must be in [0, {len(self.global_layer_indices)})."
                )
        K = self.config.concept_block
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        dtype = self.backbone.model.embed_tokens.weight.dtype

        use_concepts = self.has_concepts and concept_mode != "zero"
        per_layer = self.config.concept_io_mode == "per_layer_banks"
        if use_concepts:
            # ``initial_concepts`` overrides the learned init — used by the ``frozen``
            # decode path (prompt-encoded state, read-only). Shape is mode-dependent:
            # [B, C, H] for the shared modes, [B, G, C, H] for per_layer_banks.
            if initial_concepts is not None:
                z = initial_concepts
            elif per_layer:
                z = self.concept_init.unsqueeze(0).expand(B, -1, -1, -1)  # [B, G, C, H]
            else:
                z = self.concept_init.unsqueeze(0).expand(B, -1, -1)      # [B, C, H]
        else:
            z = None
        # Banked execution applies interventions explicitly so a single depth can be
        # tested. Shared/global modes retain the holder-based legacy path.
        self._concept_state["shuffle"] = concept_mode == "shuffle" and not per_layer
        self._concept_state["permutation"] = (
            concept_permutation
            if concept_mode == "permutation" and not per_layer
            else None
        )

        total_ce = input_ids.new_zeros((), dtype=torch.float32)
        total_cnt = input_ids.new_zeros((), dtype=torch.long)
        pressure_ce = input_ids.new_zeros((), dtype=torch.float32)
        pressure_cnt = input_ids.new_zeros((), dtype=torch.long)
        pressured_examples = 0
        eligible_pressure_examples = 0
        pos_ce = (
            input_ids.new_full((B, S), float("nan"), dtype=torch.float32) if per_position else None
        )
        pos_predictions = (
            input_ids.new_full((B, S), IGNORE_INDEX)
            if per_position and return_predictions
            else None
        )

        n_blocks = math.ceil(S / K)
        last_hidden = None
        for b in range(n_blocks):
            s, e = b * K, min((b + 1) * K, S)
            blk_len = e - s
            lo = s - K if b > 0 else 0
            dec_ids = input_ids[:, lo:e]
            dec_mask = attention_mask[:, lo:e]
            pressure_rows = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
            if b > 0:
                valid_current = attention_mask[:, s:e].bool().any(dim=1)
                eligible_pressure_examples += int(valid_current.sum().item())
                if carry_policy == "drop_after_first":
                    pressure_rows = valid_current
                elif self.training and self.config.memory_carry_dropout > 0.0:
                    pressure_rows = (
                        torch.rand(B, device=input_ids.device)
                        < self.config.memory_carry_dropout
                    ) & valid_current
                if bool(pressure_rows.any()):
                    carry_len = s - lo
                    if carry_len <= 0:
                        raise RuntimeError("Memory pressure requires a non-empty carry.")
                    dec_ids = dec_ids.clone()
                    dec_mask = dec_mask.clone()
                    pad_id = getattr(self.config, "pad_token_id", None)
                    if pad_id is None:
                        pad_id = getattr(self.backbone.config, "pad_token_id", 0)
                    bos_id = getattr(self.config, "bos_token_id", None)
                    if bos_id is None:
                        bos_id = getattr(self.backbone.config, "bos_token_id", None)
                    if bos_id is None:
                        raise ValueError("Memory pressure requires a bos_token_id.")
                    dec_ids[pressure_rows, :carry_len] = int(pad_id or 0)
                    dec_mask[pressure_rows, :carry_len] = 0
                    dec_ids[pressure_rows, carry_len - 1] = int(bos_id)
                    dec_mask[pressure_rows, carry_len - 1] = 1
                    pressured_examples += int(pressure_rows.sum().item())
            mask4d = self._windowed_causal_mask(dec_mask, dtype)
            self._concept_state["z"] = z
            attention_masks = {
                "full_attention": mask4d,
                "sliding_attention": mask4d,
            }
            inputs_embeds = self.backbone.model.embed_tokens(dec_ids)
            if (
                self.config.concept_io_mode in ("shared_depth_recurrent", "per_layer_banks")
                and self.has_concepts
            ):
                if self.config.concept_io_mode == "per_layer_banks":
                    h, z = self._forward_per_layer_banks_block(
                        inputs_embeds,
                        attention_masks,
                        z,
                        block_len=blk_len,
                        block_pad_mask=attention_mask[:, s:e] == 0,
                        concept_mode=concept_mode,
                        concept_permutation=concept_permutation,
                        concept_bank_index=concept_bank_index,
                    )
                else:
                    h, z = self._forward_shared_depth_block(
                        inputs_embeds,
                        attention_masks,
                        z,
                        block_len=blk_len,
                        block_pad_mask=attention_mask[:, s:e] == 0,
                        concept_mode=concept_mode,
                    )
            else:
                out = self.backbone.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_masks,
                    use_cache=False,
                )
                h = out.last_hidden_state                               # [B, Q, H]

            if return_last_hidden and b == n_blocks - 1:
                # Last new token in this block sits at the end of the (carry+)block window.
                last_hidden = h[:, -1].contiguous()

            if labels is not None:
                if b == 0:
                    pred_h = h[:, : blk_len - 1]
                    tgt = labels[:, s + 1 : e]
                else:
                    carry_len = s - lo                                  # = K
                    pred_h = h[:, carry_len - 1 : carry_len - 1 + blk_len]
                    tgt = labels[:, s:e]
                if per_position:
                    if return_predictions:
                        ce, predictions = self._per_position_metrics_from_hidden(
                            pred_h, tgt
                        )
                    else:
                        ce = self._per_position_ce_from_hidden(pred_h, tgt)
                    pos_ce[:, (s + 1 if b == 0 else s) : e] = ce
                    if return_predictions:
                        pos_predictions[:, (s + 1 if b == 0 else s) : e] = predictions
                else:
                    ce_sum, cnt = self._lm_ce_sum(pred_h, tgt)
                    total_ce = total_ce + ce_sum
                    total_cnt = total_cnt + cnt
                    if (
                        b > 0
                        and bool(pressure_rows.any())
                        and self.config.memory_pressure_tokens > 0
                        and self.config.memory_pressure_weight > 1.0
                    ):
                        pressure_len = min(
                            self.config.memory_pressure_tokens, pred_h.shape[1]
                        )
                        extra_ce, extra_cnt = self._lm_ce_sum(
                            pred_h[pressure_rows, :pressure_len],
                            tgt[pressure_rows, :pressure_len],
                        )
                        pressure_ce = pressure_ce + extra_ce
                        pressure_cnt = pressure_cnt + extra_cnt

            if (
                self.config.concept_io_mode == "global_kv"
                and use_concepts
                and concept_mode != "static"
            ):
                h_blk = h[:, -blk_len:]
                blk_pad = attention_mask[:, s:e] == 0
                write_base = (
                    self.concept_init.unsqueeze(0).expand(B, -1, -1)
                    if concept_mode == "one_block"
                    else z
                )
                z = self.write_head(write_base, h_blk, blk_pad)

        self._concept_state["z"] = None
        self._concept_state["shuffle"] = False
        self._concept_state["permutation"] = None
        # Preserve the latest *training* intervention rate for eval-time telemetry.
        # Deterministic normal/carryless analysis forwards must not overwrite it.
        if self.training and carry_policy == "normal":
            self._last_pressure_fraction = (
                pressured_examples / eligible_pressure_examples
                if eligible_pressure_examples
                else 0.0
            )

        if return_last_hidden:
            if last_hidden is None:
                raise RuntimeError("return_last_hidden requested but no blocks were run.")
            return last_hidden
        if per_position:
            if return_predictions:
                return pos_ce, pos_predictions, z
            return pos_ce, z
        loss = None
        if labels is not None:
            pressure_multiplier = self.config.memory_pressure_weight - 1.0
            # This branch must be identical on every DDP rank. A local ``pressure_cnt``
            # check can diverge when Bernoulli sampling selects no rows on one rank,
            # yielding mismatched all-reduce dtypes (long vs float).
            weighted_pressure = (
                pressure_multiplier > 0.0
                and self.config.memory_pressure_tokens > 0
                and self.config.memory_carry_dropout > 0.0
            )
            if weighted_pressure:
                numerator = total_ce + pressure_multiplier * pressure_ce
                denominator = (
                    total_cnt.to(torch.float32)
                    + pressure_multiplier * pressure_cnt.to(torch.float32)
                )
            else:
                numerator = total_ce
                denominator = total_cnt.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    denominator, op=torch.distributed.ReduceOp.SUM
                )
                world_size = torch.distributed.get_world_size()
            # DDP averages rank gradients. Multiplying each rank's local CE sum by
            # world_size/global_count makes that average equal the true global token mean,
            # even when right-padding yields unequal valid-token counts across ranks.
            loss = (
                numerator
                * world_size
                / denominator.clamp(min=1).to(numerator.dtype)
            )
            if use_concepts and torch.is_grad_enabled():
                # DDP (find_unused_parameters=False): tie the final write's params into the
                # graph even for single-block batches; contributes exactly zero.
                loss = loss + 0.0 * z.float().sum()
        return loss, z

    # ------------------------------------------------------------------ public API
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.config.global_attention_mode == "full":
            raise RuntimeError(
                "global_attention_mode='full' is an eval-only baseline; use per_position_ce()."
            )
        loss, _ = self._forward_blocks(input_ids, attention_mask, labels)
        # logits deliberately None: [B,S,V] at V=262K would be gigabytes; the trainer's
        # fast path and eval only consume .loss.
        return CausalLMOutputWithPast(loss=loss, logits=None)

    @torch.no_grad()
    def per_position_ce(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "blockwise",              # "blockwise" | "single_windowed" | "full_attention"
        concept_mode: str = "real",
        concept_permutation: Optional[torch.Tensor] = None,
        *,
        carry_policy: str = "normal",
        concept_bank_index: Optional[int] = None,
    ) -> torch.Tensor:
        """[B, S] next-token CE per position (nan where untargeted). The Stage-0 /
        extrapolation-eval workhorse. "single_windowed" (one forward, every layer window-
        masked, no concepts) is the zero-init equivalence reference for "blockwise";
        "full_attention" is the intact-backbone upper baseline."""
        if labels is None:
            labels = torch.where(
                (attention_mask if attention_mask is not None else torch.ones_like(input_ids)).bool(),
                input_ids, torch.full_like(input_ids, IGNORE_INDEX),
            )
        if mode == "blockwise":
            pos_ce, _ = self._forward_blocks(
                input_ids,
                attention_mask,
                labels,
                concept_mode=concept_mode,
                concept_permutation=concept_permutation,
                carry_policy=carry_policy,
                concept_bank_index=concept_bank_index,
                per_position=True,
            )
            return pos_ce
        if concept_permutation is not None:
            raise ValueError("concept_permutation is only supported in blockwise mode.")
        if carry_policy != "normal" or concept_bank_index is not None:
            raise ValueError(
                "carry_policy/concept_bank_index are only supported in blockwise mode."
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        dtype = self.backbone.model.embed_tokens.weight.dtype
        windowed = self._windowed_causal_mask(attention_mask, dtype)
        if mode == "single_windowed":
            masks = {"full_attention": windowed, "sliding_attention": windowed}
        elif mode == "full_attention":
            masks = {
                "full_attention": self._full_causal_mask(attention_mask, dtype),
                "sliding_attention": windowed,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self._concept_state["z"] = None
        out = self.backbone.model(
            inputs_embeds=self.backbone.model.embed_tokens(input_ids),
            attention_mask=masks,
            use_cache=False,
        )
        h = out.last_hidden_state
        B, S = input_ids.shape
        pos_ce = input_ids.new_full((B, S), float("nan"), dtype=torch.float32)
        pos_ce[:, 1:] = self._per_position_ce_from_hidden(h[:, :-1], labels[:, 1:])
        return pos_ce

    @torch.no_grad()
    def per_position_metrics(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        concept_mode: str = "real",
        concept_permutation: Optional[torch.Tensor] = None,
        *,
        carry_policy: str = "normal",
        concept_bank_index: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Blockwise next-token CE and top-1 predictions at targeted positions.

        Returns ``ce`` as ``[B,S]`` with NaN at ignored labels and ``predictions``
        as ``[B,S]`` with ``IGNORE_INDEX`` at ignored labels. Logits are projected
        only for targeted positions, which keeps sparse-answer evaluation cheap at
        large vocabularies.
        """
        if labels is None:
            labels = torch.where(
                (
                    attention_mask
                    if attention_mask is not None
                    else torch.ones_like(input_ids)
                ).bool(),
                input_ids,
                torch.full_like(input_ids, IGNORE_INDEX),
            )
        pos_ce, predictions, _ = self._forward_blocks(
            input_ids,
            attention_mask,
            labels,
            concept_mode=concept_mode,
            concept_permutation=concept_permutation,
            carry_policy=carry_policy,
            concept_bank_index=concept_bank_index,
            per_position=True,
            return_predictions=True,
        )
        return {"ce": pos_ce, "predictions": predictions}

    @torch.no_grad()
    def encode_concepts(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        """Final concept state after consuming the whole input block-recurrently — the
        [B, C, H] contract the trainer's geometry probe and run_concept_analysis expect."""
        if not self.has_concepts:
            raise RuntimeError("encode_concepts requires concept_num > 0.")
        _, z = self._forward_blocks(input_ids, attention_mask, labels=None)
        # per_layer_banks carries G banks [B, G, C, H]; expose the last bank [B, C, H] so
        # the downstream geometry/ablation probes (which expect [B, C, H]) work unchanged.
        if self.config.concept_io_mode == "per_layer_banks":
            z = z[:, -1]
        return BaseModelOutput(last_hidden_state=z)

    @torch.no_grad()
    def encode_concept_banks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return all depth-private concept banks as ``[B,G,C,H]``.

        The established ``encode_concepts`` contract remains ``[B,C,H]`` and returns
        the last bank. This explicit bank API prevents geometry probes from silently
        evaluating only one E17/E17c depth.
        """
        if not self.has_concepts or self.config.concept_io_mode != "per_layer_banks":
            raise RuntimeError(
                "encode_concept_banks requires concept_num > 0 and per_layer_banks mode."
            )
        _, banks = self._forward_blocks(input_ids, attention_mask, labels=None)
        return banks

    @torch.no_grad()
    def next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        concept_mode: str = "real",
        initial_concepts: Optional[torch.Tensor] = None,
        carry_policy: Optional[str] = None,
    ) -> torch.Tensor:
        """Logits for the token immediately after ``input_ids``. Shape ``[B, V]``.

        Re-runs the block-recurrent forward (no KV cache). Intended for short
        interactive generation / playground use, not long free-running decode.

        The LM-head projection is always computed in float32 so MPS float16
        models still yield finite sampling logits.
        """
        last_hidden = self._forward_blocks(
            input_ids,
            attention_mask,
            labels=None,
            concept_mode=concept_mode,
            carry_policy=carry_policy,
            return_last_hidden=True,
            initial_concepts=initial_concepts,
        )
        weight = self.backbone.lm_head.weight.float()
        return F.linear(last_hidden.float(), weight)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        concept_mode: str = "real",
        carry_policy: Optional[str] = None,
    ) -> torch.Tensor:
        """Greedy or sampled continuation of ``input_ids`` (returns full sequences).

        Each step re-encodes the growing prefix through the concept block loop.
        Keep ``max_new_tokens`` modest on CPU/MPS.

        On Apple MPS, load the model in ``float32`` (fp16 forwards often NaN).

        ``repetition_penalty`` (HF/CTRL style, default 1.0 = off): each step lowers
        the score of every token already in the context (positive logits divided by
        ``repetition_penalty``, negative logits multiplied). Values >1 break the
        greedy fixed-point loops that otherwise dominate long free-running decode
        from a base concept-LM (the E16b repetition symptom).

        ``concept_mode="frozen"`` encodes the prompt into the concept state WITH
        writes, then decodes read-only (no writes from the model's own tokens) — a
        zero-training-cost probe of whether self-generated concept writes drive the
        free-run repetition (the E16b Layer-0 "freeze-z" diagnostic).

        Carry policy defaults to ``config.inference_carry_policy`` (E17d:
        ``drop_after_first`` so previous windows exist only as concept banks).
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        cur = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(cur)
        finished = torch.zeros(cur.shape[0], dtype=torch.bool, device=cur.device)
        # ``frozen`` concept decode (Layer-0 free-run probe): encode the prompt into
        # the concept state WITH writes, then decode read-only (no writes from the
        # model's own tokens). Falsifies "self-generated writes poison free-run" at
        # zero training cost. Internally maps to a ``static`` (read-only) forward
        # seeded by the prompt-encoded ``z``.
        frozen_z = None
        fwd_mode = concept_mode
        resolved_carry = self._resolve_carry_policy(carry_policy)
        if concept_mode == "frozen":
            if not self.has_concepts:
                raise ValueError("concept_mode='frozen' requires concept_num > 0.")
            # Full concept state after prompt-encoding: [B, C, H] for the shared modes,
            # [B, G, C, H] for per_layer_banks (encode_concepts exposes only the last bank).
            _, frozen_z = self._forward_blocks(
                input_ids,
                mask,
                labels=None,
                concept_mode="real",
                carry_policy=resolved_carry,
            )
            fwd_mode = "static"
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(
                cur, mask, concept_mode=fwd_mode, initial_concepts=frozen_z,
                carry_policy=resolved_carry,
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "next_token_logits produced non-finite values. On Apple MPS this "
                    "usually means the model was loaded in float16 — reload with "
                    "dtype=torch.float32 (see playground/e16b_generation_playground.ipynb)."
                )
            logits = logits.float()
            # Repetition penalty (HF / CTRL style) over the full context seen so far.
            # rp>1 depresses already-used tokens; rp=1.0 reproduces prior behaviour.
            if repetition_penalty and repetition_penalty != 1.0:
                score = logits.gather(-1, cur)                       # [B, T] seen-token logits
                score = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty,
                )
                logits = logits.scatter(-1, cur, score)
            if do_sample:
                if temperature <= 0:
                    raise ValueError("temperature must be > 0 when do_sample=True")
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1]
                    logits = logits.masked_fill(logits < kth.unsqueeze(-1), float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sl_probs = torch.softmax(sorted_logits, dim=-1)
                    cum = sl_probs.cumsum(dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                probs = torch.softmax(logits, dim=-1)
                # Bulletproof sampling distribution: the prior multinomial crash lived
                # here. Strip NaN/Inf, clamp >=0, renormalise, and replace any
                # degenerate (all-masked) row with a one-hot at the argmax BEFORE
                # multinomial ever sees it.
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                sums = probs.sum(dim=-1, keepdim=True)
                probs = probs / sums.clamp_min(1e-20)
                if bool((sums <= 0).any()):
                    finite_logits = logits.masked_fill(~torch.isfinite(logits), float("-inf"))
                    fallback = finite_logits.argmax(dim=-1)
                    onehot = torch.zeros_like(probs)
                    onehot[torch.arange(probs.size(0), device=probs.device), fallback] = 1.0
                    deg = (sums <= 0).to(probs.dtype)
                    probs = deg * onehot + (1.0 - deg) * probs
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            if finished.any():
                pad_id = self.config.pad_token_id
                if pad_id is None:
                    pad_id = eos_token_id if eos_token_id is not None else 0
                next_id = torch.where(
                    finished.view(-1, 1),
                    torch.full_like(next_id, pad_id),
                    next_id,
                )
            cur = torch.cat([cur, next_id], dim=1)
            mask = torch.cat([mask, (~finished).long().view(-1, 1)], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break
        return cur

    @torch.no_grad()
    def _intra_block_bin_mean(
        self, values: torch.Tensor, start_frac: float, end_frac: float
    ) -> float:
        """Mean CE on post-first blocks in ``[start_frac, end_frac)`` of each K-block."""
        K = self.config.concept_block
        parts = []
        for block_index in range(1, math.ceil(values.shape[1] / K)):
            start = block_index * K + int(start_frac * K)
            stop = min(block_index * K + int(end_frac * K), values.shape[1])
            if stop > start:
                parts.append(values[:, start:stop].reshape(-1))
        if not parts:
            return float("nan")
        selected = torch.cat(parts)
        return float(selected.nanmean().item())

    @torch.no_grad()
    def concept_ablation_ce(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        window_k: Optional[int] = None,
    ) -> dict:
        """Does the decoder read the concept *content*? CE with real vs batch-shuffled vs
        zeroed concept state, split by protocol reach.

        Positions < K are purely local. Positions [K, 2K) still have the complete preceding
        block in the explicit carry and therefore do not require recurrent memory. Only
        positions >= 2K are beyond the one-block-carry control's direct reach; this is the
        decisive E10 region (>=1024 for K=512).
        """
        if not self.has_concepts:
            return {}
        K = self.config.concept_block
        beyond_start = 2 * K
        results = {}
        for name in ("real", "shuffle", "zero", "static", "one_block"):
            pos_ce = self.per_position_ce(
                input_ids, attention_mask, labels, mode="blockwise", concept_mode=name
            )
            early = pos_ce[:, :K]
            carry = pos_ce[:, K:beyond_start]
            beyond = pos_ce[:, beyond_start:]
            results[f"ce_{name}"] = float(pos_ce.nanmean().item())
            results[f"ce_{name}_early"] = float(early.nanmean().item())
            if carry.numel() > 0 and not torch.isnan(carry).all():
                results[f"ce_{name}_carry"] = float(carry.nanmean().item())
            if beyond.numel() > 0 and not torch.isnan(beyond).all():
                results[f"ce_{name}_beyond"] = float(beyond.nanmean().item())
        for region in ("", "_early", "_carry", "_beyond"):
            if f"ce_shuffle{region}" in results and f"ce_real{region}" in results:
                results[f"delta_shuffle{region}"] = (
                    results[f"ce_shuffle{region}"] - results[f"ce_real{region}"]
                )
            if f"ce_zero{region}" in results and f"ce_real{region}" in results:
                results[f"delta_zero{region}"] = (
                    results[f"ce_zero{region}"] - results[f"ce_real{region}"]
                )
            if f"ce_static{region}" in results and f"ce_real{region}" in results:
                results[f"delta_static{region}"] = (
                    results[f"ce_static{region}"] - results[f"ce_real{region}"]
                )
            if f"ce_one_block{region}" in results and f"ce_real{region}" in results:
                results[f"delta_one_block{region}"] = (
                    results[f"ce_one_block{region}"] - results[f"ce_real{region}"]
                )
        # E17c's decisive test is a deterministic content permutation, not a roll hidden
        # behind the historical "shuffle" name. Batch size one cannot support it.
        if (
            self.config.concept_io_mode == "per_layer_banks"
            and input_ids.shape[0] > 1
        ):
            permutation = torch.roll(
                torch.arange(input_ids.shape[0], device=input_ids.device), shifts=1
            )
            perm_ce = self.per_position_ce(
                input_ids,
                attention_mask,
                labels,
                mode="blockwise",
                concept_mode="permutation",
                concept_permutation=permutation,
            )
            results["ce_permutation"] = float(perm_ce.nanmean().item())
            results["delta_permutation"] = (
                results["ce_permutation"] - results["ce_real"]
            )
            if perm_ce.shape[1] > beyond_start:
                perm_beyond = perm_ce[:, beyond_start:]
                if not torch.isnan(perm_beyond).all():
                    results["ce_permutation_beyond"] = float(
                        perm_beyond.nanmean().item()
                    )
                    results["delta_permutation_beyond"] = (
                        results["ce_permutation_beyond"]
                        - results["ce_real_beyond"]
                    )

            # Forced carry removal uses exactly the train-time intervention but is
            # deterministic and eval-only. Aggregate the first R targets of every
            # post-first block, the region E17c explicitly pressures during training.
            pressure_real = self.per_position_ce(
                input_ids,
                attention_mask,
                labels,
                mode="blockwise",
                concept_mode="real",
                carry_policy="drop_after_first",
            )
            pressure_perm = self.per_position_ce(
                input_ids,
                attention_mask,
                labels,
                mode="blockwise",
                concept_mode="permutation",
                concept_permutation=permutation,
                carry_policy="drop_after_first",
            )

            def pressure_prefix_mean(values: torch.Tensor) -> float:
                parts = []
                R = self.config.memory_pressure_tokens or min(64, K)
                for block_index in range(1, math.ceil(values.shape[1] / K)):
                    start = block_index * K
                    stop = min(start + R, values.shape[1])
                    if stop > start:
                        parts.append(values[:, start:stop].reshape(-1))
                if not parts:
                    return float("nan")
                selected = torch.cat(parts)
                return float(selected.nanmean().item())

            results["pressure_ce_real_first64"] = pressure_prefix_mean(pressure_real)
            results["pressure_ce_permutation_first64"] = pressure_prefix_mean(
                pressure_perm
            )
            results["pressure_delta_permutation_first64"] = (
                results["pressure_ce_permutation_first64"]
                - results["pressure_ce_real_first64"]
            )

            # E17d primary: late intra-block bins under forced no-carry. Names follow
            # the K=512 protocol; offsets scale with concept_block (K=8 tests use K/2:K).
            for bin_name, start_frac, end_frac in INTRA_BLOCK_BINS:
                real_bin = self._intra_block_bin_mean(
                    pressure_real, start_frac, end_frac
                )
                perm_bin = self._intra_block_bin_mean(
                    pressure_perm, start_frac, end_frac
                )
                results[f"ce_real_block_{bin_name}"] = real_bin
                results[f"ce_permutation_block_{bin_name}"] = perm_bin
                results[f"delta_permutation_block_{bin_name}"] = perm_bin - real_bin

            for bank_index in range(len(self.global_layer_indices)):
                bank_perm = self.per_position_ce(
                    input_ids,
                    attention_mask,
                    labels,
                    mode="blockwise",
                    concept_mode="permutation",
                    concept_permutation=permutation,
                    concept_bank_index=bank_index,
                )
                if bank_perm.shape[1] > beyond_start:
                    bank_beyond = bank_perm[:, beyond_start:]
                    if not torch.isnan(bank_beyond).all():
                        bank_ce = float(bank_beyond.nanmean().item())
                        results[f"ce_permutation_bank_{bank_index}_beyond"] = bank_ce
                        results[f"delta_permutation_bank_{bank_index}_beyond"] = (
                            bank_ce - results["ce_real_beyond"]
                        )
                bank_pressure = self.per_position_ce(
                    input_ids,
                    attention_mask,
                    labels,
                    mode="blockwise",
                    concept_mode="permutation",
                    concept_permutation=permutation,
                    concept_bank_index=bank_index,
                    carry_policy="drop_after_first",
                )
                bank_pressure_ce = pressure_prefix_mean(bank_pressure)
                results[f"pressure_ce_permutation_bank_{bank_index}_first64"] = (
                    bank_pressure_ce
                )
                results[f"pressure_delta_permutation_bank_{bank_index}_first64"] = (
                    bank_pressure_ce - results["pressure_ce_real_first64"]
                )
                late_bank = self._intra_block_bin_mean(bank_pressure, 0.5, 1.0)
                results[f"ce_permutation_bank_{bank_index}_block_256_512"] = late_bank
                results[f"delta_permutation_bank_{bank_index}_block_256_512"] = (
                    late_bank - results["ce_real_block_256_512"]
                )
        return results

    @torch.no_grad()
    def concept_gate_metrics(self) -> dict[str, float]:
        """Current effective read/write gates for E10/E16 live monitoring."""
        if not self.has_concepts:
            return {}
        if self.config.concept_write_mode == "gated_replace":
            metrics = {}
            for depth_index, layer_index in enumerate(self.global_layer_indices):
                writer = self._writer_for_depth(depth_index)
                if writer._last_update_gate_mean is not None:
                    metrics[f"concept_gates/update_{depth_index}"] = (
                        writer._last_update_gate_mean
                    )
                    metrics[f"concept_gates/update_layer_{layer_index}"] = (
                        writer._last_update_gate_mean
                    )
                    metrics[f"concept_state/update_rms_{depth_index}"] = (
                        writer._last_update_rms
                    )
                    metrics[f"concept_state/state_rms_{depth_index}"] = (
                        writer._last_state_rms
                    )
        elif self.config.concept_io_mode == "global_kv":
            metrics = {
                "concept_gates/write": float(
                    torch.tanh(self.write_head.alpha).item()
                )
            }
        else:
            metrics = {}
            for depth_index, layer_index in enumerate(self.global_layer_indices):
                writer = self._writer_for_depth(depth_index)
                if writer.depth_alphas is not None:
                    value = float(
                        torch.tanh(writer.depth_alphas[depth_index]).item()
                    )
                elif writer.alpha is not None:
                    value = float(torch.tanh(writer.alpha).item())
                else:
                    continue
                metrics[f"concept_gates/write_{depth_index}"] = value
                metrics[f"concept_gates/write_layer_{layer_index}"] = value
        read_idx = 0
        for layer_idx, layer in enumerate(self.backbone.model.layers):
            if isinstance(layer, GlobalLayerWithConceptRead):
                value = float(torch.tanh(layer.gate).item())
                metrics[f"concept_gates/read_{read_idx}"] = value
                metrics[f"concept_gates/read_layer_{layer_idx}"] = value
                read_idx += 1
        if self.config.memory_carry_dropout > 0.0:
            metrics["memory_pressure/observed_fraction"] = self._last_pressure_fraction
        return metrics

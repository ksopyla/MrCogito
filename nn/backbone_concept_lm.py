"""E10 — Pretrained-backbone concept memory (Design C: global→concept read + recurrent write).

Grafts the MrCogito concept machinery onto a frozen pretrained decoder (Gemma-3 family):

  * READ  — the backbone's *global* attention layers (Gemma-3 interleaves 5 sliding-window
    layers : 1 global layer) lose their full-attention reach (all token↔token attention is
    windowed) and instead gain a zero-init-gated cross-attention read of C concept slots,
    computed with the layer's OWN q/k/v/o projections (no new attention weights).
  * WRITE — after each K-token block (K = the backbone's sliding window), the concept state
    is updated from the block's final hidden states through a gated BiXT cross-attention
    (`nn/concept_encoder.py:BiXTCrossAttention`, `update_tokens=False`) — the E09 write op.
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

Spec: docs/experiments_specs/E10_gemma_backbone_concept_memory.md (+ _plan.md).
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
        concept_io_mode: str = "global_kv",   # E11: "mem_tokens" · E12: "kv_prefix"
        write_num_heads: int = 4,
        read_gate_init: float = 0.0,
        write_gate_init: float = 0.0,
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
        self.read_gate_init = read_gate_init
        self.write_gate_init = write_gate_init
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


class ConceptReadBranch(nn.Module):
    """Cross-attention read of the concept state, reusing the wrapped Gemma3Attention's own
    q/k/v/o projections (LoRA-adapted automatically). No RoPE on either side — the concept
    memory is a position-free set, so the read is length-extrapolation-safe by construction."""

    def forward(self, x_normed: torch.Tensor, z: torch.Tensor, attn: nn.Module) -> torch.Tensor:
        B, Q, _ = x_normed.shape
        C = z.shape[1]
        hd = attn.head_dim
        q = attn.q_proj(x_normed).view(B, Q, -1, hd).transpose(1, 2)   # [B, nH, Q, hd]
        k = attn.k_proj(z).view(B, C, -1, hd).transpose(1, 2)          # [B, nKV, C, hd]
        v = attn.v_proj(z).view(B, C, -1, hd).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if attn.num_key_value_groups > 1:
            k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
            v = v.repeat_interleave(attn.num_key_value_groups, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
        o = o.transpose(1, 2).reshape(B, Q, -1)
        return attn.o_proj(o)


class GlobalLayerWithConceptRead(nn.Module):
    """Wraps one of the backbone's global decoder layers: runs the original layer unchanged,
    then adds a tanh-gated (zero-init) concept read. The concept state arrives through a
    shared mutable holder set by `BackboneConceptLM` before each block forward."""

    def __init__(self, layer: nn.Module, state_holder: dict, gate_init: float):
        super().__init__()
        self.layer = layer
        self.attention_type = layer.attention_type   # read by Gemma3TextModel.forward's mask routing
        self._state = state_holder                    # plain dict, not a submodule
        self.read_branch = ConceptReadBranch()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.layer(hidden_states, *args, **kwargs)
        z = self._state.get("z")
        if z is not None:
            if self._state.get("shuffle"):
                z = torch.roll(z, shifts=1, dims=0)   # batch derangement (ablation)
            x = self.layer.input_layernorm(hidden_states)
            read = self.read_branch(x, z.to(hidden_states.dtype), self.layer.self_attn)
            hidden = outputs[0] + torch.tanh(self.gate) * read
            outputs = (hidden,) + tuple(outputs[1:])
        return outputs


class ConceptWriteHead(nn.Module):
    """Gated recurrent concept write (E09 design): z ← z + tanh(α)·RMSNorm(BiXT_lat←tok(z, h)).
    α is zero-init ⇒ identity at step 0. Rows whose block is entirely padding are left
    untouched (and their would-be NaN softmax is neutralized before the gate)."""

    def __init__(self, hidden_size: int, num_heads: int, gate_init: float = 0.0):
        super().__init__()
        self.bixt = BiXTCrossAttention(
            dim_lat=hidden_size, dim_tok=hidden_size, dim_attn=hidden_size,
            num_heads=num_heads, update_tokens=False,
        )
        self.norm_lat = nn.RMSNorm(hidden_size)
        self.norm_tok = nn.RMSNorm(hidden_size)
        self.sandwich = nn.RMSNorm(hidden_size)   # Ouro-style anti-collapse post-norm
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, z: torch.Tensor, h_block: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # pad_mask: [B, Kb] bool, True = padding.
        valid_row = (~pad_mask).any(dim=1)                     # [B]
        safe_pad = pad_mask.clone()
        safe_pad[~valid_row, 0] = False                        # avoid all -inf softmax rows
        lat, _ = self.bixt(
            self.norm_lat(z), self.norm_tok(h_block).to(z.dtype), key_padding_mask=safe_pad
        )
        update = torch.tanh(self.alpha) * self.sandwich(lat)
        update = update * valid_row.view(-1, 1, 1).to(update.dtype)
        return z + update


class BackboneConceptLM(PreTrainedModel):
    """Frozen pretrained decoder + LoRA + concept read/write graft (Design C / `global_kv`)."""

    config_class = BackboneConceptConfig
    base_model_prefix = "backbone_concept"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["backbone.lm_head.weight"]

    def __init__(self, config: BackboneConceptConfig, backbone: Optional[nn.Module] = None):
        super().__init__(config)
        if config.concept_io_mode != "global_kv":
            raise NotImplementedError(
                f"concept_io_mode={config.concept_io_mode!r} is a follow-up spec (E11/E12); "
                "only 'global_kv' (E10) is implemented."
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
        if config.backbone_config is None:
            config.backbone_config = bb_cfg.to_dict()
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
        self._concept_state: dict = {"z": None, "shuffle": False}
        if config.concept_num > 0:
            self.concept_init = nn.Parameter(
                torch.randn(config.concept_num, self.hidden_size) * (self.hidden_size ** -0.5)
            )
            self.write_head = ConceptWriteHead(
                self.hidden_size, config.write_num_heads, gate_init=config.write_gate_init
            )
            layers = self.backbone.model.layers
            n_wrapped = 0
            for i, layer in enumerate(layers):
                if layer.attention_type == "full_attention":
                    layers[i] = GlobalLayerWithConceptRead(
                        layer, self._concept_state, config.read_gate_init
                    )
                    n_wrapped += 1
            if n_wrapped == 0:
                raise ValueError(
                    "No 'full_attention' layers found in the backbone — the concept read "
                    "graft would be a silent no-op. Check the backbone's layer_types "
                    f"(got: {[l.attention_type for l in layers]})."
                )
        else:
            self.concept_init = None
            self.write_head = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained_backbone(cls, config: BackboneConceptConfig, **backbone_kwargs):
        """Initial-training path: load the backbone weights from the hub, then graft."""
        from transformers import Gemma3ForCausalLM
        backbone = Gemma3ForCausalLM.from_pretrained(config.backbone_model, **backbone_kwargs)
        config.backbone_config = backbone.config.to_dict()
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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
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
    def _per_position_ce_from_hidden(self, pred_hidden, targets, chunk: int = 256):
        """[B, T] CE per target position (nan where label ignored). Eval-only."""
        B, T, _ = pred_hidden.shape
        out = pred_hidden.new_full((B, T), float("nan"), dtype=torch.float32)
        weight = self.backbone.lm_head.weight
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            logits = F.linear(pred_hidden[:, s:e], weight).float()
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets[:, s:e].reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="none",
            ).view(B, e - s)
            valid = targets[:, s:e] != IGNORE_INDEX
            out[:, s:e] = torch.where(valid, ce, torch.full_like(ce, float("nan")))
        return out

    # ------------------------------------------------------------------ core block loop
    def _forward_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        concept_mode: str = "real",           # "real" | "shuffle" | "zero"
        per_position: bool = False,
    ):
        B, S = input_ids.shape
        K = self.config.concept_block
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        dtype = self.backbone.model.embed_tokens.weight.dtype

        use_concepts = self.has_concepts and concept_mode != "zero"
        z = self.concept_init.unsqueeze(0).expand(B, -1, -1) if use_concepts else None
        self._concept_state["shuffle"] = concept_mode == "shuffle"

        total_ce = input_ids.new_zeros((), dtype=torch.float32)
        total_cnt = input_ids.new_zeros((), dtype=torch.long)
        pos_ce = (
            input_ids.new_full((B, S), float("nan"), dtype=torch.float32) if per_position else None
        )

        n_blocks = math.ceil(S / K)
        for b in range(n_blocks):
            s, e = b * K, min((b + 1) * K, S)
            blk_len = e - s
            lo = s - K if b > 0 else 0
            dec_ids = input_ids[:, lo:e]
            dec_mask = attention_mask[:, lo:e]
            mask4d = self._windowed_causal_mask(dec_mask, dtype)
            self._concept_state["z"] = z
            out = self.backbone.model(
                inputs_embeds=self.backbone.model.embed_tokens(dec_ids),
                attention_mask={"full_attention": mask4d, "sliding_attention": mask4d},
                use_cache=False,
            )
            h = out.last_hidden_state                                   # [B, Q, H]

            if labels is not None:
                if b == 0:
                    pred_h = h[:, : blk_len - 1]
                    tgt = labels[:, s + 1 : e]
                else:
                    carry_len = s - lo                                  # = K
                    pred_h = h[:, carry_len - 1 : carry_len - 1 + blk_len]
                    tgt = labels[:, s:e]
                if per_position:
                    ce = self._per_position_ce_from_hidden(pred_h, tgt)
                    pos_ce[:, (s + 1 if b == 0 else s) : e] = ce
                else:
                    ce_sum, cnt = self._lm_ce_sum(pred_h, tgt)
                    total_ce = total_ce + ce_sum
                    total_cnt = total_cnt + cnt

            if use_concepts:
                h_blk = h[:, -blk_len:]
                blk_pad = attention_mask[:, s:e] == 0
                z = self.write_head(z, h_blk, blk_pad)

        self._concept_state["z"] = None
        self._concept_state["shuffle"] = False

        if per_position:
            return pos_ce, z
        loss = None
        if labels is not None:
            loss = total_ce / total_cnt.clamp(min=1).to(total_ce.dtype)
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
                input_ids, attention_mask, labels, concept_mode=concept_mode, per_position=True
            )
            return pos_ce
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
    def encode_concepts(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        """Final concept state after consuming the whole input block-recurrently — the
        [B, C, H] contract the trainer's geometry probe and run_concept_analysis expect."""
        if not self.has_concepts:
            raise RuntimeError("encode_concepts requires concept_num > 0.")
        _, z = self._forward_blocks(input_ids, attention_mask, labels=None)
        return BaseModelOutput(last_hidden_state=z)

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
        for name in ("real", "shuffle", "zero"):
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
        return results

    @torch.no_grad()
    def concept_gate_metrics(self) -> dict[str, float]:
        """Current effective read/write gates for E10 live monitoring."""
        if not self.has_concepts:
            return {}
        metrics = {"concept_gates/write": float(torch.tanh(self.write_head.alpha).item())}
        read_idx = 0
        for layer_idx, layer in enumerate(self.backbone.model.layers):
            if isinstance(layer, GlobalLayerWithConceptRead):
                value = float(torch.tanh(layer.gate).item())
                metrics[f"concept_gates/read_{read_idx}"] = value
                metrics[f"concept_gates/read_layer_{layer_idx}"] = value
                read_idx += 1
        return metrics

"""Arguments and compatibility validation for concept pretraining."""

from dataclasses import dataclass, field
from typing import Optional

from nn.loss_manager import LossConfig, get_available_losses
from training.concept_pretraining_objectives import (
    DECODER_CAUSAL_AR,
    DECODER_PERCEIVER_POSONLY,
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    VALID_DECODER_TYPES,
    VALID_OBJECTIVES,
    resolve_append_eos_token_id,
)


@dataclass
class ModelArguments:
    hidden_size: int = field(default=512)
    token_embedding_dim: int = field(
        default=512,
        metadata={"help": "Token embedding dimension used by the encoder token stream."},
    )
    num_hidden_layers: int = field(default=6)
    concept_num: int = field(default=128)
    intermediate_size: int = field(default=2048)
    decoder_num_layers: int = field(
        default=3,
        metadata={"help": "Number of stacked decoder layers (position-only OR causal AR)."},
    )
    decoder_type: str = field(
        default=DECODER_PERCEIVER_POSONLY,
        metadata={
            "help": f"Decoder family: one of {sorted(VALID_DECODER_TYPES)}. "
            "'causal_ar' = autoregressive concept-conditioned decoder (E01)."
        },
    )
    decoder_pos_type: str = field(
        default="learned",
        metadata={"help": "Decoder position encoding: 'learned' or 'rope' (causal_ar only)."},
    )
    decoder_word_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of decoder-input tokens replaced by a learned dropout "
            "embedding (posterior-collapse guard for causal_ar)."
        },
    )
    decoder_context_window: Optional[int] = field(
        default=None,
        metadata={
            "help": "E05: restrict causal_ar decoder self-attention to the last K tokens "
            "(sliding-window). None = full causal context (E01/E02/E03). When set, "
            "out-of-window context is only reachable through the concepts."
        },
    )
    decoder_attn_impl: str = field(
        default="sdpa",
        metadata={
            "help": "Decoder self-attn backend. 'sdpa' (default, byte-unchanged) or "
            "'chunked_window' — O(N*K) memory windowed attention for long context. "
            "Only applies when decoder_context_window is set."
        },
    )
    decoder_attn_chunk_size: int = field(
        default=2048,
        metadata={
            "help": "Query chunk size for decoder_attn_impl='chunked_window'. Larger = "
            "fewer kernel launches but higher peak; default 2048."
        },
    )
    chunked_ce_block_size: int = field(
        default=0,
        metadata={
            "help": "F2 long-context: compute lm_head+CE in N-blocks of this size so "
            "the full [B,N,V] logits + fp32 CE upcast are never materialised (the O(N*V) "
            "spike). 0 = off (materialise full logits, legacy). Training-only; ablation/eval "
            "keep the full-logits path."
        },
    )
    hidden_act: str = field(
        default="gelu",
        metadata={
            "help": "FFN activation. 'silu' makes the gated FFN SwiGLU; "
            "'gelu' = GEGLU (legacy)."
        },
    )
    norm_type: str = field(
        default="layernorm",
        metadata={"help": "Normalization: 'layernorm' (legacy) or 'rmsnorm'."},
    )
    concept_position_type: str = field(default="none")
    use_bixt: bool = field(
        default=True,
        metadata={"help": "Use BiXT bidirectional encoder layers."},
    )
    bixt_token_ffn: bool = field(
        default=True,
        metadata={"help": "Enable the cheap token-side FFN inside BiXT layers."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkpoint used to warm-start encoder weights."},
    )
    torch_compile_dynamic: bool = field(
        default=False,
        metadata={"help": "Apply torch.compile(dynamic=True)."},
    )
    objective_variant: str = field(
        default=OBJECTIVE_RECONSTRUCTION,
        metadata={"help": "One of: reconstruction, reconstruction+contrastive, prefix_suffix."},
    )
    contrastive_weight: float = field(
        default=0.3,
        metadata={"help": "Weight applied to the contrastive loss when enabled."},
    )
    contrastive_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature used by the in-batch contrastive loss."},
    )
    anchor_loss: bool = field(
        default=False,
        metadata={
            "help": "Enable the frozen-encoder per-token hidden-state anchor auxiliary (E03)."
        },
    )
    anchor_model_name: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={
            "help": "Frozen teacher whose per-token hidden states the concepts must reconstruct. "
            "Must share the model tokenizer (1:1 token alignment)."
        },
    )
    anchor_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight lambda on the anchor MSE term added to the AR loss."},
    )
    anchor_standardize: bool = field(
        default=True,
        metadata={"help": "Per-token layer_norm of teacher targets before MSE (stable regression)."},
    )
    anchor_head_layers: int = field(
        default=2,
        metadata={
            "help": "Lean anchor head depth (PerceiverDecoderLayer blocks); keep small."
        },
    )
    backbone_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "E10: HF id of the frozen pretrained decoder to graft concepts onto "
            "(e.g. google/gemma-3-1b-pt). None = the classic concept-encoder families."
        },
    )
    concept_block: int = field(
        default=512,
        metadata={
            "help": "E10: block size = write cadence; must equal the backbone's sliding "
            "window (Gemma-3-1B: 512)."
        },
    )
    concept_io_mode: str = field(
        default="global_kv",
        metadata={
            "help": "Backbone concept read/write mechanism: 'global_kv' keeps E10's "
            "single post-block write; 'shared_depth_recurrent' (E16) applies one tied "
            "write after each global concept-reading layer."
        },
    )
    read_concept_norm: bool = field(
        default=False,
        metadata={
            "help": "E10b: RMS-normalize recurrent concepts before the global-layer "
            "concept-read K/V projections."
        },
    )
    read_gate_init: float = field(
        default=0.0,
        metadata={"help": "E10c: raw tanh-gate initialization for every concept read."},
    )
    write_gate_init: float = field(
        default=0.0,
        metadata={"help": "E10c: raw tanh-gate initialization for the recurrent BiXT write."},
    )
    concept_read_mode: str = field(
        default="backbone_qkv",
        metadata={
            "help": "Concept read projections: backbone_qkv (legacy E17b) or dedicated."
        },
    )
    concept_read_placement: str = field(
        default="post_layer",
        metadata={
            "help": "Where the concept mix is added: post_layer (E17c sidecar after FFN) "
            "or attn_residual (E17d mix inside the attention residual before FFN)."
        },
    )
    inference_carry_policy: str = field(
        default="normal",
        metadata={
            "help": "Eval/generate carry policy when carry_policy is omitted: normal "
            "(keep previous-block tokens) or drop_after_first (concepts-only history)."
        },
    )
    tie_concept_writer: bool = field(
        default=True,
        metadata={"help": "Share one concept writer across global depths (legacy behavior)."},
    )
    concept_write_mode: str = field(
        default="additive",
        metadata={"help": "Concept state transition: additive or gated_replace."},
    )
    write_update_gate_init: float = field(
        default=0.25,
        metadata={"help": "Initial sigmoid update probability for gated replacement."},
    )
    memory_carry_dropout: float = field(
        default=0.0,
        metadata={"help": "Per-example probability of dropping prior-block token carry."},
    )
    memory_pressure_tokens: int = field(
        default=0,
        metadata={"help": "Early current-block targets upweighted under carry pressure."},
    )
    memory_pressure_weight: float = field(
        default=1.0,
        metadata={"help": "CE weight for pressured early targets (must be >=1)."},
    )
    lora_r: int = field(default=16, metadata={"help": "E10: LoRA rank on the backbone (0 = off)."})
    lora_alpha: int = field(default=32, metadata={"help": "E10: LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "E10: LoRA dropout."})
    lora_targets: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "E10: comma-separated LoRA target module names."},
    )


@dataclass
class LossArguments:
    concept_losses: Optional[str] = field(
        default="none",
        metadata={
            "help": f"Space-separated concept losses or 'none'. "
            f"Available: {get_available_losses()}"
        },
    )
    loss_weight: float = field(
        default=0.02,
        metadata={"help": "Shared fixed weight distributed over enabled concept losses."},
    )
    uniformity_temperature: float = field(default=2.0)
    concept_loss_warmup_steps: int = field(default=0)

    def to_loss_config(self) -> LossConfig:
        if self.concept_losses is None or self.concept_losses.lower() == "none":
            return LossConfig.disabled()

        losses = self.concept_losses.split()
        per_loss_weight = self.loss_weight / len(losses) if losses else 0.0
        loss_weights = {"task": 1.0, **{loss_name: per_loss_weight for loss_name in losses}}
        loss_params = {}

        if "uniformity" in losses or "combined" in losses:
            loss_params["uniformity"] = {"temperature": self.uniformity_temperature}
            loss_params["combined"] = {"temperature": self.uniformity_temperature}

        return LossConfig(
            concept_losses=losses,
            weighting_strategy="fixed",
            loss_weights=loss_weights,
            loss_params=loss_params,
            warmup_steps=self.concept_loss_warmup_steps,
        )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(default="JeanKaddour/minipile")
    dataset_name_subset: Optional[str] = field(default=None)
    dataset_mix: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a registered multi-dataset mix in "
            "data.dataset_preprocess.DATASET_MIXES (e.g. 'long_2k_base_v1'). When set, "
            "overrides dataset_name/dataset_name_subset and interleaves the mix."
        },
    )
    dataset_mix_recipe: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path or id of a JSON mix recipe (data/mix_recipes/*.json). "
            "Preferred over dataset_mix for configurable long-context pretraining."
        },
    )
    dataset_mix_weight_override: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON object that overrides source weights at runtime, "
            'e.g. \'{"finepdfs_100BT":0.6,"fineweb_edu":0.2,"finemath_3plus":0.2}\'.'
        },
    )
    pretokenized_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a manifest JSON written by scripts/pretokenize_mix.py. "
            "If set, loads pre-tokenized sources via load_from_disk (instant, no download). "
            "Overrides dataset_mix/dataset_mix_recipe when present."
        },
    )
    preserve_precomputed_labels: bool = field(
        default=False,
        metadata={
            "help": "For causal_lm manifests, preserve each row's precomputed sparse labels "
            "instead of mirroring input_ids. Default false reproduces E10."
        },
    )
    batch_packing_mode: str = field(
        default="none",
        metadata={
            "help": "Training pad-reduction mode: 'none' or 'length_group'. "
            "Length grouping preserves rows and only reorders examples inside bounded "
            "shuffled windows."
        },
    )
    length_group_mega_batch_mult: int = field(
        default=20,
        metadata={
            "help": "Sortish window multiplier. Each shuffled window contains this many "
            "(per-device batch × gradient accumulation) examples before sorting by length."
        },
    )
    tokenizer_name: str = field(default="answerdotai/ModernBERT-base")
    max_seq_length: int = field(default=512)
    test_size_percent: float = field(default=0.1)
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional deterministic cap for training-time evaluation. "
            "Final evaluation should use the full frozen held-out protocol."
        },
    )
    dataset_cache_dir: Optional[str] = field(default=None)
    deletion_rate: float = field(default=0.6)
    train_num_proc: int = field(default=8)
    test_num_proc: int = field(default=4)
    prefix_ratio_min: float = field(default=0.3)
    prefix_ratio_max: float = field(default=0.5)
    min_prefix_content: int = field(default=5)
    min_suffix_content: int = field(default=10)
    split_strategy: str = field(default="sentence_boundary")

    def __post_init__(self) -> None:
        valid_modes = {"none", "length_group"}
        if self.batch_packing_mode not in valid_modes:
            raise ValueError(
                f"batch_packing_mode must be one of {sorted(valid_modes)}, "
                f"got {self.batch_packing_mode!r}."
            )
        if self.length_group_mega_batch_mult < 1:
            raise ValueError("length_group_mega_batch_mult must be positive.")
        if self.batch_packing_mode == "length_group" and not self.pretokenized_manifest:
            raise ValueError(
                "batch_packing_mode='length_group' requires --pretokenized_manifest "
                "so cached lengths stay aligned with the interleaved dataset."
            )


@dataclass
class OptimizerArguments:
    """Optimizer family and Muon-specific fallback parameters."""

    optimizer: str = field(
        default="adam",
        metadata={
            "help": "Optimizer family: 'adam' (HF adamw_torch_fused) or "
            "'muon' (nn.muon.Muon)."
        },
    )
    concept_memory_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "E10d: optional AdamW LR for concept_init, write head, read gates, "
            "and read norms. None keeps the existing single-LR optimizer path."
        },
    )
    muon_adamw_lr: float = field(
        default=2e-3,
        metadata={
            "help": "Muon only: AdamW LR for the non-orthogonalized fallback params "
            "(embeddings, lm_head, norms, biases). The matrix LR is --learning_rate."
        },
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Muon only: momentum coefficient for the Muon momentum buffer."},
    )


def validate_training_configuration(
    model_args: ModelArguments,
    loss_args: LossArguments,
) -> tuple[bool, bool]:
    """Validate objective/family compatibility and return `(is_causal_ar, is_backbone)`."""
    if model_args.objective_variant not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective_variant: {model_args.objective_variant}. "
            f"Expected one of {sorted(VALID_OBJECTIVES)}."
        )
    if model_args.decoder_type not in VALID_DECODER_TYPES:
        raise ValueError(
            f"Unknown decoder_type: {model_args.decoder_type}. "
            f"Expected one of {sorted(VALID_DECODER_TYPES)}."
        )

    is_causal_ar = model_args.decoder_type == DECODER_CAUSAL_AR
    is_backbone = model_args.backbone_model is not None
    if is_backbone and model_args.objective_variant != OBJECTIVE_CAUSAL_LM:
        raise ValueError(
            "backbone_model (E10) requires objective_variant='causal_lm'; "
            f"got {model_args.objective_variant!r}."
        )
    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM and not is_backbone:
        raise ValueError("objective_variant='causal_lm' requires --backbone_model (E10).")
    if is_backbone and model_args.anchor_loss:
        raise ValueError("anchor_loss is not supported by the backbone-concept family.")
    if is_backbone and loss_args.concept_losses and loss_args.concept_losses.lower() != "none":
        raise ValueError("concept_losses are not wired into the backbone-concept family (E10).")
    if is_backbone and model_args.model_name_or_path:
        raise ValueError(
            "model_name_or_path warm-start is the concept-encoder path; the backbone family "
            "initializes from backbone_model directly."
        )
    if is_causal_ar and model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        raise ValueError(
            "decoder_type='causal_ar' supports objective_variant='reconstruction' or "
            f"'prefix_suffix' (got {model_args.objective_variant!r}). "
            "The contrastive path is perceiver-only."
        )
    if not is_causal_ar and model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        raise ValueError("objective_variant='prefix_suffix' requires decoder_type='causal_ar'.")
    if model_args.anchor_loss:
        if not is_causal_ar:
            raise ValueError("anchor_loss=True requires decoder_type='causal_ar' (E03).")
        if model_args.objective_variant != OBJECTIVE_RECONSTRUCTION:
            raise ValueError(
                "anchor_loss=True is scoped to objective_variant='reconstruction' (E03 v1); "
                f"got {model_args.objective_variant!r}."
            )
    return is_causal_ar, is_backbone

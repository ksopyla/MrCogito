"""
Perceiver denoising pretraining.

Maintained perceiver training path:
  - BiXT encoder by default
  - position-only stacked decoder
  - full-sequence reconstruction from deleted-token inputs
  - optional stage-2 SimCSE-style contrastive objective
  - causal-AR prefix->suffix objective for concept-conditioned generation
"""

import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)
from transformers.modeling_outputs import MaskedLMOutput

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collators import (
    DataCollatorForCausalLM,
    DataCollatorForPrefixGeneration,
    DataCollatorForTSDAE,
)
from data.dataset_preprocess import (
    load_and_preprocess_dataset_mix,
    load_and_preprocess_text_dataset,
    load_pretokenized_mix,
)
from nn.backbone_concept_lm import BackboneConceptConfig, BackboneConceptLM
from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.loss_manager import ConceptLossStepCallback, LossConfig, get_available_losses
from training.utils_training import (
    WandbRunIdentity,
    broadcast_object,
    build_perceiver_wandb_identity,
    init_wandb,
    is_main_process,
    log_data_config,
    log_loss_config,
    log_model_info,
    log_system_info,
    log_training_config,
    setup_distributed,
    setup_file_logging,
    setup_run_dirs,
)

logger = logging.get_logger(__name__)

OBJECTIVE_RECONSTRUCTION = "reconstruction"
OBJECTIVE_RECONSTRUCTION_CONTRASTIVE = "reconstruction+contrastive"
OBJECTIVE_PREFIX_SUFFIX = "prefix_suffix"
OBJECTIVE_CAUSAL_LM = "causal_lm"   # E10 backbone-concept family: plain next-token CE
VALID_OBJECTIVES = {
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_CAUSAL_LM,
}


def resolve_append_eos_token_id(objective_variant, is_causal_ar, eos_token_id):
    """Decide whether preprocessing appends EOS (and so stays variable-length, padding=False).

    Both decoder families need this:
      - causal_ar: the AR decoder must learn to stop (EOS as a next-token target).
      - perceiver_posonly reconstruction: a single boundary EOS, AND — critically — variable-length
        rows so DataCollatorForTSDAE (which rebuilds the attention_mask from row LENGTH) marks
        padding correctly. The old padding="max_length" path made every pad position look real,
        so the encoder attended the pad tail and the decoder was trained to predict <eos> on
        hundreds of pad positions (a concept-free shortcut) — and put the perceiver path on a
        different data contract than its causal_ar baseline.

    Returns the eos id to append, or None to keep the legacy pad-to-max_length path.
    """
    objective_appends_eos = objective_variant in (
        OBJECTIVE_RECONSTRUCTION,
        OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
        OBJECTIVE_PREFIX_SUFFIX,
        OBJECTIVE_CAUSAL_LM,   # EOS marks the document boundary the LM must learn
    )
    if eos_token_id is not None and (is_causal_ar or objective_appends_eos):
        return eos_token_id
    return None


DECODER_PERCEIVER_POSONLY = "perceiver_posonly"
DECODER_CAUSAL_AR = "causal_ar"
VALID_DECODER_TYPES = {DECODER_PERCEIVER_POSONLY, DECODER_CAUSAL_AR}


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
        metadata={"help": f"Decoder family: one of {sorted(VALID_DECODER_TYPES)}. "
                  "'causal_ar' = autoregressive concept-conditioned decoder (E01)."},
    )
    decoder_pos_type: str = field(
        default="learned",
        metadata={"help": "Decoder position encoding: 'learned' or 'rope' (causal_ar only)."},
    )
    decoder_word_dropout: float = field(
        default=0.0,
        metadata={"help": "Fraction of decoder-input tokens replaced by a learned dropout "
                  "embedding (posterior-collapse guard for causal_ar)."},
    )
    decoder_context_window: Optional[int] = field(
        default=None,
        metadata={"help": "E05: restrict causal_ar decoder self-attention to the last K tokens "
                  "(sliding-window). None = full causal context (E01/E02/E03). When set, "
                  "out-of-window context is only reachable through the concepts."},
    )
    decoder_attn_impl: str = field(
        default="sdpa",
        metadata={"help": "Decoder self-attn backend. 'sdpa' (default, byte-unchanged) or "
                  "'chunked_window' — O(N*K) memory windowed attention for long context. "
                  "Only applies when decoder_context_window is set."},
    )
    decoder_attn_chunk_size: int = field(
        default=2048,
        metadata={"help": "Query chunk size for decoder_attn_impl='chunked_window'. Larger = "
                  "fewer kernel launches but higher peak; default 2048."},
    )
    chunked_ce_block_size: int = field(
        default=0,
        metadata={"help": "F2 long-context: compute lm_head+CE in N-blocks of this size so "
                  "the full [B,N,V] logits + fp32 CE upcast are never materialised (the O(N*V) "
                  "spike). 0 = off (materialise full logits, legacy). Training-only; ablation/eval "
                  "keep the full-logits path."},
    )
    hidden_act: str = field(
        default="gelu",
        metadata={"help": "FFN activation. 'silu' makes the gated FFN SwiGLU; 'gelu' = GEGLU (legacy)."},
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
    # E03 — concept de-collapse via a frozen-encoder hidden-state anchor (causal_ar + reconstruction).
    anchor_loss: bool = field(
        default=False,
        metadata={"help": "Enable the frozen-encoder per-token hidden-state anchor auxiliary (E03)."},
    )
    anchor_model_name: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "Frozen teacher whose per-token hidden states the concepts must reconstruct. "
                  "Must share the model tokenizer (1:1 token alignment)."},
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
        metadata={"help": "Lean anchor head depth (PerceiverDecoderLayer blocks); keep small."},
    )
    # E10 — pretrained-backbone concept memory (BackboneConceptLM). Default None keeps every
    # existing model family byte-identical; setting backbone_model selects the new family
    # (requires objective_variant='causal_lm').
    backbone_model: Optional[str] = field(
        default=None,
        metadata={"help": "E10: HF id of the frozen pretrained decoder to graft concepts onto "
                  "(e.g. google/gemma-3-1b-pt). None = the classic concept-encoder families."},
    )
    concept_block: int = field(
        default=512,
        metadata={"help": "E10: block size = write cadence; must equal the backbone's sliding "
                  "window (Gemma-3-1B: 512)."},
    )
    concept_io_mode: str = field(
        default="global_kv",
        metadata={"help": "E10: concept read/write mechanism. 'global_kv' (E10 Design C); "
                  "'mem_tokens' (E11) / 'kv_prefix' (E12) are follow-up specs."},
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
        metadata={"help": f"Space-separated concept losses or 'none'. Available: {get_available_losses()}"},
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
        metadata={"help": "Name of a registered multi-dataset mix in "
                  "data.dataset_preprocess.DATASET_MIXES (e.g. 'long_2k_base_v1'). When set, "
                  "overrides dataset_name/dataset_name_subset and interleaves the mix."},
    )
    dataset_mix_recipe: Optional[str] = field(
        default=None,
        metadata={"help": "Path or id of a JSON mix recipe (data/mix_recipes/*.json). "
                  "Preferred over dataset_mix for configurable long-context pretraining."},
    )
    dataset_mix_weight_override: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON object that overrides source weights at runtime, "
                  "e.g. '{\"finepdfs_100BT\":0.6,\"fineweb_edu\":0.2,\"finemath_3plus\":0.2}'."},
    )
    pretokenized_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a manifest JSON written by scripts/pretokenize_mix.py. "
                  "If set, loads pre-tokenized sources via load_from_disk (instant, no download). "
                  "Overrides dataset_mix/dataset_mix_recipe when present."},
    )
    tokenizer_name: str = field(default="answerdotai/ModernBERT-base")
    max_seq_length: int = field(default=512)
    test_size_percent: float = field(default=0.1)
    dataset_cache_dir: Optional[str] = field(default=None)
    deletion_rate: float = field(default=0.6)
    train_num_proc: int = field(default=8)
    test_num_proc: int = field(default=4)
    prefix_ratio_min: float = field(default=0.3)
    prefix_ratio_max: float = field(default=0.5)
    min_prefix_content: int = field(default=5)
    min_suffix_content: int = field(default=10)
    split_strategy: str = field(default="sentence_boundary")


@dataclass
class OptimizerArguments:
    """Optimizer selection. `optimizer` picks the family ("adam" = the HF default
    adamw_torch_fused, or "muon" = nn.muon.Muon); the remaining fields parameterize the Muon
    branch only (see PerceiverDenoiseTrainer.create_optimizer). The LR is `--learning_rate` for
    both families; for Muon it is the matrix LR and `muon_adamw_lr` is the fallback LR for the
    embedding / lm_head / 1D params Muon does not orthogonalize (nn.muon.Muon).

    We use our own `--optimizer` flag rather than HF's `--optim` because HF coerces `--optim` to
    its OptimizerNames enum in TrainingArguments.__post_init__, which rejects the "muon" string."""
    optimizer: str = field(
        default="adam",
        metadata={"help": "Optimizer family: 'adam' (HF adamw_torch_fused) or 'muon' (nn.muon.Muon)."},
    )
    muon_adamw_lr: float = field(
        default=2e-3,
        metadata={"help": "Muon only: AdamW LR for the non-orthogonalized fallback params "
                  "(embeddings, lm_head, norms, biases). The matrix LR is --learning_rate."},
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Muon only: momentum coefficient for the Muon momentum buffer."},
    )


class PerceiverDenoiseTrainer(Trainer):
    def __init__(
        self,
        *args,
        objective_variant: str,
        contrastive_weight: float,
        contrastive_temperature: float,
        compute_concept_ablation: bool = False,
        concept_ablation_batches: int = 5,
        eval_data_collator=None,
        anchor_loss: bool = False,
        anchor_loss_weight: float = 0.5,
        anchor_standardize: bool = True,
        anchor_model_name: Optional[str] = None,
        optimizer_choice: str = "adam",
        muon_adamw_lr: float = 2e-3,
        muon_momentum: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.objective_variant = objective_variant
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.compute_concept_ablation = compute_concept_ablation
        self.concept_ablation_batches = concept_ablation_batches
        self.eval_data_collator = eval_data_collator
        # Optimizer selection (used by create_optimizer): "adam" = HF default, "muon" = nn.muon.Muon.
        self.optimizer_choice = optimizer_choice
        self.muon_adamw_lr = muon_adamw_lr
        self.muon_momentum = muon_momentum
        # E03: frozen teacher held on the trainer (NOT a model submodule → never checkpointed).
        self.anchor_loss = anchor_loss
        self.anchor_loss_weight = anchor_loss_weight
        self.anchor_standardize = anchor_standardize
        self.anchor_teacher = None
        if anchor_loss:
            if anchor_model_name is None:
                raise ValueError("anchor_loss=True requires anchor_model_name.")
            logger.info(f"Loading frozen anchor teacher: {anchor_model_name}")
            teacher = AutoModel.from_pretrained(anchor_model_name)
            teacher.eval()
            teacher.requires_grad_(False)
            self.anchor_teacher = teacher.to(self.args.device)

    def create_optimizer(self):
        """Build the optimizer. Routes `--optimizer muon` to `nn.muon.Muon` (2D weight matrices via
        Newton-Schulz orthogonalization; AdamW fallback for embeddings/lm_head/1D params). The
        default (`--optimizer adam`) falls through to the HF Trainer default (adamw_torch_fused),
        so the non-Muon path is byte-unchanged. Overriding create_optimizer is the HF-sanctioned
        pattern for custom optimizers.

        `--learning_rate` becomes Muon's matrix LR; `muon_adamw_lr` is the fallback LR. The cosine
        LR scheduler (create_scheduler) wraps `self.optimizer` and schedules the `lr` group as usual."""
        if self.optimizer_choice == "muon":
            from nn.muon import Muon
            self.optimizer = Muon(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=self.muon_momentum,
                adamw_lr=self.muon_adamw_lr,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def _anchor_mse(
        self,
        base_model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        concept_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen-teacher forward + the model-owned anchor loss. The teacher runs on the CLEAN
        input_ids with a non-pad mask (labels != -100) — NOT the TSDAE-corrupted attention_mask —
        so the concepts must carry uncorrupted semantics. The standardize+masked-MSE policy lives in
        ConceptEncoderForConditionalLM.compute_anchor_loss (single source, unit-tested)."""
        target_mask = (labels != -100)
        with torch.no_grad():
            teacher_hidden = self.anchor_teacher(
                input_ids=input_ids, attention_mask=target_mask.long()
            ).last_hidden_state  # [B, N, Ht]
        return base_model.compute_anchor_loss(
            concept_repr, teacher_hidden, target_mask, standardize=self.anchor_standardize
        )

    def _anchor_compute_loss(self, base_model, inputs, return_outputs):
        """Reconstruction AR loss (the model's single-source recipe) + lambda * anchor MSE."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        # encode_decode_loss is the SAME recipe forward() uses → the anchor path cannot drift.
        # Pass the target mask so the decoder self-attn matches the forward() behaviour (E05/2K).
        task_loss, logits, encoder_outputs = base_model.encode_decode_loss(
            input_ids, attention_mask, input_ids, labels,
            target_attention_mask=attention_mask,
        )
        concept_repr = encoder_outputs.last_hidden_state
        anchor_mse = self._anchor_mse(base_model, input_ids, labels, concept_repr)
        total_loss = task_loss + self.anchor_loss_weight * anchor_mse

        outputs = MaskedLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def get_eval_dataloader(self, eval_dataset=None):
        """Use a separate (seeded, deterministic-corruption) collator for eval.

        The training collator samples fresh TSDAE deletions / prefix splits per
        call; reusing it at eval makes eval_loss noisy and lets best-checkpoint
        selection depend on corruption luck.
        """
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    @torch.no_grad()
    def _concept_ablation_metrics(self) -> dict:
        """Average the model's concept-ablation CE over a few eval batches.

        Posterior-collapse diagnostic for the causal-AR decoder: large positive
        delta_zero / delta_shuffle mean the decoder genuinely uses the concepts.
        """
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(base_model, "concept_ablation_ce"):
            return {}
        # E05: when a sliding-window decoder is configured, also log beyond-window
        # ablation deltas (the long-range memory gate). None for full-context decoders.
        window_k = getattr(base_model.config, "decoder_context_window", None)
        dataloader = self.get_eval_dataloader()
        device = self.args.device
        sums: dict = {}
        n = 0
        rank_metrics: dict = {}
        anchor_sum = 0.0
        anchor_n = 0
        for i, batch in enumerate(dataloader):
            if i >= self.concept_ablation_batches:
                break
            labels = batch["labels"].to(device)
            if "prefix_input_ids" in batch:
                prefix_attention_mask = batch.get("prefix_attention_mask")
                if prefix_attention_mask is not None:
                    prefix_attention_mask = prefix_attention_mask.to(device)
                encoder_input_ids = batch["prefix_input_ids"].to(device)
                encoder_attention_mask = prefix_attention_mask
                m = base_model.concept_ablation_ce(
                    prefix_input_ids=encoder_input_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_input_ids=batch["suffix_input_ids"].to(device),
                    labels=labels,
                    window_k=window_k,
                )
            else:
                encoder_input_ids = batch["input_ids"].to(device)
                encoder_attention_mask = batch.get("attention_mask")
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.to(device)
                m = base_model.concept_ablation_ce(
                    encoder_input_ids, encoder_attention_mask, labels, window_k=window_k
                )
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            # E03: held-out anchor MSE (de-collapse progress) — reconstruction batches only.
            # Kept out of eval_loss so best-checkpoint selection stays a clean AR CE.
            if self.anchor_loss and self.anchor_teacher is not None and "prefix_input_ids" not in batch:
                concepts_eval = base_model.encode_concepts(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True,
                ).last_hidden_state
                anchor_sum += self._anchor_mse(base_model, encoder_input_ids, labels, concepts_eval).item()
                anchor_n += 1
            # Concept geometry (collapse gate) from the first batch only — cheap.
            if not rank_metrics:
                rank_metrics = self._concept_effective_rank(
                    base_model, encoder_input_ids, encoder_attention_mask
                )
            n += 1
        if n == 0:
            return {}
        out = {f"concept_ablation/{k}": v / n for k, v in sums.items()}
        out.update(rank_metrics)
        if anchor_n > 0:
            out["anchor/mse_eval"] = anchor_sum / anchor_n
        return out

    @torch.no_grad()
    def _concept_effective_rank(self, base_model, input_ids, attention_mask) -> dict:
        """Effective rank (nuclear/spectral norm of the mean concept matrix).

        The collapse gate: low effective rank means the C concepts are redundant
        (occupy few dimensions). Logged each eval so de-collapse is visible live.
        Matches analysis/concept_analysis.compute_concept_geometry_metrics.
        """
        try:
            concepts = base_model.encode_concepts(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state.float()  # [B, C, H]
            concept_mean = concepts.mean(dim=0)  # [C, H]
            s = torch.linalg.svdvals(concept_mean)
            eff_rank = (s.sum() / (s.max() + 1e-8)).item()
            max_rank = min(concept_mean.shape)
            return {
                "concept_geometry/effective_rank": eff_rank,
                "concept_geometry/effective_rank_normalized": eff_rank / max_rank,
            }
        except Exception:
            return {}

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.compute_concept_ablation and is_main_process():
            ablation = self._concept_ablation_metrics()
            if ablation:
                metrics.update(ablation)
                self.log(ablation)
        return metrics

    def _contrastive_loss(
        self,
        model: ConceptEncoderForDenoisingPerceiver,
        concept_repr_a: torch.Tensor,
        concept_repr_b: torch.Tensor,
    ) -> torch.Tensor:
        pooled_a = F.normalize(model.pool_concepts(concept_repr_a), dim=-1)
        pooled_b = F.normalize(model.pool_concepts(concept_repr_b), dim=-1)
        similarity = pooled_a @ pooled_b.T / self.contrastive_temperature
        labels = torch.arange(similarity.size(0), device=similarity.device)
        return (
            F.cross_entropy(similarity, labels)
            + F.cross_entropy(similarity.T, labels)
        ) / 2.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        # Fast path: eval (pure AR CE — keeps eval_loss comparable for best-checkpoint selection),
        # or a plain forward objective with no extra manual auxiliary. The anchor and contrastive
        # objectives fall through to the manual path below (training only).
        if not model.training or (
            self.objective_variant
            in {OBJECTIVE_RECONSTRUCTION, OBJECTIVE_PREFIX_SUFFIX, OBJECTIVE_CAUSAL_LM}
            and not self.anchor_loss
        ):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Unwrap DDP wrapper to access custom methods (encode_concepts, etc.).
        # Gradient sync still works: with find_unused_parameters=False, DDP's
        # parameter-level backward hooks fire regardless of the forward path.
        base_model = model.module if hasattr(model, "module") else model

        # E03: reconstruction AR loss + frozen-teacher hidden-state anchor MSE.
        if self.anchor_loss:
            return self._anchor_compute_loss(base_model, inputs, return_outputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        encoder_outputs_a = base_model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        concept_repr_a = encoder_outputs_a.last_hidden_state
        decoder_output = base_model.decode_from_concepts(concept_repr_a, seq_length=input_ids.size(1))
        logits, task_loss = base_model.reconstruction_loss(decoder_output, labels)

        if base_model.loss_manager.is_enabled:
            total_loss = base_model.loss_manager(task_loss=task_loss, concept_repr=concept_repr_a)
        else:
            total_loss = task_loss

        encoder_outputs_b = base_model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        contrastive_loss = self._contrastive_loss(
            base_model,
            concept_repr_a=concept_repr_a,
            concept_repr_b=encoder_outputs_b.last_hidden_state,
        )
        total_loss = total_loss + self.contrastive_weight * contrastive_loss

        outputs = MaskedLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=encoder_outputs_a.hidden_states,
            attentions=encoder_outputs_a.attentions,
        )
        return (total_loss, outputs) if return_outputs else total_loss


def build_perceiver_denoise_config(
    tokenizer,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
) -> ConceptEncoderConfig:
    objective_name = "denoising_full_reconstruction"
    if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        objective_name = "denoising_full_reconstruction_contrastive"
    elif model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        objective_name = "ar_prefix_suffix_generation"
    is_causal_ar = model_args.decoder_type == DECODER_CAUSAL_AR
    # The causal-AR family probes concept quality with the encoder only (no
    # PerceiverDecoderStack exists in its checkpoint), so the canonical single-input
    # route is weighted_pool (encoder_only); pair tasks use sentence_pair (encoder_only).
    checkpoint_family = "concept_ar" if is_causal_ar else "perceiver_denoise"
    canonical_single_eval_mode = "weighted_pool" if is_causal_ar else "via_decoder"
    if is_causal_ar and model_args.objective_variant == OBJECTIVE_RECONSTRUCTION:
        objective_name = "ar_denoising_reconstruction"

    # E03: infer the frozen teacher's hidden size from its config so the anchor head is rebuildable
    # from our config alone (no teacher needed at eval/analysis), and assert a shared vocab so the
    # teacher's per-token states align 1:1 with our token ids.
    anchor_teacher_hidden = None
    if model_args.anchor_loss:
        teacher_cfg = AutoConfig.from_pretrained(model_args.anchor_model_name)
        anchor_teacher_hidden = teacher_cfg.hidden_size
        if teacher_cfg.vocab_size != len(tokenizer):
            raise ValueError(
                f"anchor_loss requires the model tokenizer to match the teacher vocab for 1:1 token "
                f"alignment: tokenizer has {len(tokenizer)} tokens but "
                f"{model_args.anchor_model_name} has {teacher_cfg.vocab_size}. "
                f"Use TOKENIZER_NAME={model_args.anchor_model_name} (or a same-vocab tokenizer)."
            )

    return ConceptEncoderConfig(
        vocab_size=len(tokenizer),
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        token_embedding_dim=model_args.token_embedding_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=8,
        intermediate_size=model_args.intermediate_size,
        hidden_act=model_args.hidden_act,
        max_sequence_length=data_args.max_seq_length,
        concept_position_type=model_args.concept_position_type,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        tie_word_embeddings=model_args.token_embedding_dim == model_args.hidden_size,
        tokenizer_name=data_args.tokenizer_name,
        use_bixt=model_args.use_bixt,
        bixt_token_ffn=model_args.bixt_token_ffn,
        decoder_posonly=not is_causal_ar,
        decoder_num_layers=model_args.decoder_num_layers,
        decoder_type=model_args.decoder_type,
        decoder_pos_type=model_args.decoder_pos_type,
        decoder_word_dropout=model_args.decoder_word_dropout,
        decoder_context_window=model_args.decoder_context_window,
        decoder_attn_impl=getattr(model_args, "decoder_attn_impl", "sdpa"),
        decoder_attn_chunk_size=getattr(model_args, "decoder_attn_chunk_size", 2048),
        chunked_ce_block_size=getattr(model_args, "chunked_ce_block_size", 0),
        norm_type=model_args.norm_type,
        checkpoint_family=checkpoint_family,
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode=canonical_single_eval_mode,
        pretraining_objective=objective_name,
        anchor_loss=model_args.anchor_loss,
        anchor_model_name=model_args.anchor_model_name if model_args.anchor_loss else None,
        anchor_loss_weight=model_args.anchor_loss_weight,
        anchor_standardize=model_args.anchor_standardize,
        anchor_head_layers=model_args.anchor_head_layers,
        anchor_teacher_hidden=anchor_teacher_hidden,
    )


def main():
    setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    parser = HfArgumentParser(
        (ModelArguments, LossArguments, DataTrainingArguments, OptimizerArguments, TrainingArguments)
    )
    model_args, loss_args, data_args, optim_args, training_args = parser.parse_args_into_dataclasses()

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
        raise ValueError(
            "objective_variant='prefix_suffix' requires decoder_type='causal_ar'."
        )
    if model_args.anchor_loss:
        if not is_causal_ar:
            raise ValueError("anchor_loss=True requires decoder_type='causal_ar' (E03).")
        if model_args.objective_variant != OBJECTIVE_RECONSTRUCTION:
            raise ValueError(
                "anchor_loss=True is scoped to objective_variant='reconstruction' (E03 v1); "
                f"got {model_args.objective_variant!r}."
            )

    set_seed(training_args.seed)
    log_system_info()

    log_data_config(
        data_args,
        extra_fields={
            "Deletion rate": data_args.deletion_rate,
            "Objective": model_args.objective_variant,
            "Prefix ratio min": data_args.prefix_ratio_min,
            "Prefix ratio max": data_args.prefix_ratio_max,
            "Split strategy": data_args.split_strategy,
            "Dataset mix (registry)": data_args.dataset_mix,
            "Dataset mix recipe": data_args.dataset_mix_recipe,
        },
    )

    logger.info(f"Loading tokenizer: {data_args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_name,
        cache_dir=data_args.dataset_cache_dir,
    )

    # SmolLM2-style causal tokenizers have <|endoftext|> (bos=eos=unk) but no distinct pad.
    # Reuse eos as pad (standard decoder-only convention). Masking is positional via
    # attention_mask / labels=-100, so pad positions never contribute to loss.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad nor eos token; cannot train.")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Tokenizer had no pad token; set pad_token=eos_token "
            f"({tokenizer.pad_token!r}, pad_id={tokenizer.pad_token_id})."
        )

    # Append EOS so preprocessing stays variable-length (padding=False) for both decoder families;
    # see resolve_append_eos_token_id for the full rationale (fixes the perceiver pad-mask bug and
    # keeps E04 on the same data contract as its E03 causal_ar baseline).
    append_eos_token_id = resolve_append_eos_token_id(
        model_args.objective_variant, is_causal_ar, tokenizer.eos_token_id
    )

    with training_args.main_process_first(desc="loading and tokenizing dataset"):
        if data_args.pretokenized_manifest:
            logger.info(f"Loading pre-tokenized mix from manifest: {data_args.pretokenized_manifest}")
            train_ds, test_ds = load_pretokenized_mix(data_args.pretokenized_manifest)
        else:
            selected_mix = data_args.dataset_mix_recipe or data_args.dataset_mix
            if data_args.dataset_mix_recipe and data_args.dataset_mix:
                logger.warning(
                    "Both dataset_mix_recipe and dataset_mix were provided. "
                    f"Using dataset_mix_recipe='{data_args.dataset_mix_recipe}' and ignoring "
                    f"dataset_mix='{data_args.dataset_mix}'."
                )

            if selected_mix:
                logger.info(f"Loading dataset mix: {selected_mix}")
                if data_args.dataset_mix_weight_override:
                    try:
                        preview_override = json.loads(data_args.dataset_mix_weight_override)
                    except Exception:
                        preview_override = data_args.dataset_mix_weight_override
                    logger.info(f"Applying mix weight override: {preview_override}")
                train_ds, test_ds = load_and_preprocess_dataset_mix(
                    tokenizer,
                    selected_mix,
                    mix_weight_override=data_args.dataset_mix_weight_override,
                    test_size_percent=data_args.test_size_percent,
                    max_seq_length=data_args.max_seq_length,
                    dataset_cache_dir=data_args.dataset_cache_dir,
                    train_num_proc=data_args.train_num_proc,
                    test_num_proc=data_args.test_num_proc,
                    append_eos_token_id=append_eos_token_id,
                    split_seed=training_args.seed,
                    interleave_seed=training_args.seed,
                )
            else:
                logger.info(f"Loading dataset: {data_args.dataset_name}")
                train_ds, test_ds = load_and_preprocess_text_dataset(
                    tokenizer,
                    data_args.dataset_name,
                    data_args.dataset_name_subset,
                    "text",
                    test_size_percent=data_args.test_size_percent,
                    max_seq_length=data_args.max_seq_length,
                    dataset_cache_dir=data_args.dataset_cache_dir,
                    train_num_proc=data_args.train_num_proc,
                    test_num_proc=data_args.test_num_proc,
                    append_eos_token_id=append_eos_token_id,
                    split_seed=training_args.seed,
                )

    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")
    logger.info("=" * 60)

    loss_config = loss_args.to_loss_config()
    log_loss_config(loss_config)

    if is_backbone:
        config = BackboneConceptConfig(
            backbone_model=model_args.backbone_model,
            concept_num=model_args.concept_num,
            concept_block=model_args.concept_block,
            concept_io_mode=model_args.concept_io_mode,
            lora_r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            lora_targets=model_args.lora_targets,
            tokenizer_name=data_args.tokenizer_name,
        )
        config.pretraining_objective = "causal_lm_block_recurrent"
        logger.info(
            f"Initializing BackboneConceptLM (backbone={model_args.backbone_model}, "
            f"C={model_args.concept_num}, K={model_args.concept_block}, "
            f"io={model_args.concept_io_mode}, lora_r={model_args.lora_r})"
        )
        model = BackboneConceptLM.from_pretrained_backbone(config)
    else:
        config = build_perceiver_denoise_config(tokenizer, model_args, data_args)
        model_class = ConceptEncoderForConditionalLM if is_causal_ar else ConceptEncoderForDenoisingPerceiver
        logger.info(f"Initializing {model_class.__name__}")
        model = model_class(config, loss_config=loss_config)

    if model_args.model_name_or_path and not is_backbone:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)
        logger.info("Loaded pretrained encoder weights. Decoder and objective head use the current config.")

    if is_backbone:
        model_type_str = "backbone_concept"
    elif is_causal_ar:
        model_type_str = "concept_ar"
        if model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
            model_type_str += "_prefix"
        if model_args.use_bixt:
            model_type_str += "_bixt"
    else:
        model_type_str = "perceiver_denoise"
        if model_args.use_bixt:
            model_type_str += "_bixt"
        if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
            model_type_str += "_contrastive"

    log_model_info(
        model,
        config=config,
        model_type=model_type_str,
        model_description="BiXT perceiver denoising pretraining",
    )

    if torch.cuda.is_available() and is_main_process() and not is_backbone:
        # Flash-Attention v2 probe: meaningful only for the concept-encoder family (it
        # checks the encoder's cross-attn shape [num_heads, concept_num, head_dim]). The
        # backbone family uses the backbone's native attention, so the probe is skipped.
        _num_heads = config.num_attention_heads
        _head_dim = config.hidden_size // _num_heads
        try:
            _q = torch.zeros(1, _num_heads, config.concept_num, _head_dim,
                             dtype=torch.bfloat16, device="cuda")
            _k = torch.zeros(1, _num_heads, 512, _head_dim,
                             dtype=torch.bfloat16, device="cuda")
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                torch.nn.functional.scaled_dot_product_attention(_q, _k, _k)
            logger.info(
                f"Flash Attention v2: ACTIVE  "
                f"(heads={_num_heads}, head_dim={_head_dim}, dtype=bf16)"
            )
        except Exception as _fa_exc:
            logger.warning(
                f"Flash Attention not available — training will use memory-efficient / math SDPA. "
                f"Reason: {_fa_exc}"
            )
        finally:
            del _q, _k

    if model_args.torch_compile_dynamic and torch.cuda.is_available():
        backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
        logger.info(f"torch.compile(dynamic=True, backend='{backend}')")
        model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)

    env_experiment_id = os.environ.get("WANDB_EXPERIMENT_ID") or os.environ.get("EXPERIMENT_ID")
    if is_backbone:
        backbone_short = model_args.backbone_model.split("/")[-1].replace("-", "_")
        # Architecture id is shared by BOTH arms (the A/B variable is concept_num, surfaced
        # as a tag below) so they land in the SAME W&B group and filter together — same
        # convention as E05's optimizer A/B sharing one group.
        architecture_id = f"backbone_concept_{backbone_short}_K{model_args.concept_block}"
        resolved_experiment = env_experiment_id or "E10"
        arm_tag = "concept-arm" if model_args.concept_num > 0 else "control-arm"
        wandb_identity = WandbRunIdentity(
            experiment_id=resolved_experiment,
            model_family="backbone_concept",
            objective_family="causal_lm",
            architecture_id=architecture_id,
            group=f"{resolved_experiment}_{architecture_id}",
            job_type="train_backbone_causal_lm",
            tags=[
                "train", "concept-encoder", "decoder:autoregressive", "task:generation",
                "backbone_concept", model_args.backbone_model,
                f"io-{model_args.concept_io_mode}", "causal_lm",
                arm_tag,
                f"lora_r{model_args.lora_r}",
                resolved_experiment,
            ],
        )
    else:
        wandb_identity = build_perceiver_wandb_identity(
            decoder_type=model_args.decoder_type,
            objective_variant=model_args.objective_variant,
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            concept_num=model_args.concept_num,
            decoder_num_layers=model_args.decoder_num_layers,
            checkpoint_family=config.checkpoint_family,
            pretraining_objective=config.pretraining_objective,
            use_bixt=model_args.use_bixt,
            anchor_loss=model_args.anchor_loss,
            experiment_id=env_experiment_id,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Backbone family: distinguish the two arms in the run_id (and so the checkpoint dir
    # and W&B run name) — they share a group but must not collide on disk/W&B.
    arm_suffix = ""
    if is_backbone:
        arm_suffix = "_concept" if model_args.concept_num > 0 else "_control"
    run_identifier = (
        f"{wandb_identity.architecture_id}{arm_suffix}_{timestamp}"
    )
    # The timestamp is wall-clock, so each DDP rank would otherwise compute its
    # own run_id (and may straddle a second boundary). Broadcast rank 0's id so
    # all ranks share ONE output directory / W&B run.
    run_identifier = broadcast_object(run_identifier)
    setup_run_dirs(training_args, run_identifier)
    training_args.use_cpu = False

    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None

    training_extra_fields = {
        "Deletion rate": data_args.deletion_rate,
        "BiXT encoder": model_args.use_bixt,
        "Decoder layers": model_args.decoder_num_layers,
        "Objective": model_args.objective_variant,
        "Anchor loss": model_args.anchor_loss,
        "W&B group": wandb_identity.group,
        "W&B job_type": wandb_identity.job_type,
        "Prefix ratio min": data_args.prefix_ratio_min,
        "Prefix ratio max": data_args.prefix_ratio_max,
        "Split strategy": data_args.split_strategy,
        "Dataset mix (registry)": data_args.dataset_mix,
        "Dataset mix recipe": data_args.dataset_mix_recipe,
        "Optimizer": optim_args.optimizer,
    }
    if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        training_extra_fields["Contrastive weight"] = model_args.contrastive_weight

    log_training_config(training_args, extra_fields=training_extra_fields)

    init_wandb(
        training_args,
        model,
        config,
        data_args,
        loss_config,
        wandb_identity.group,
        run_identifier,
        job_type=wandb_identity.job_type,
        model_type=model_type_str,
        wandb_tags=[*wandb_identity.tags, f"optim-{optim_args.optimizer}"],
        extra_config={
            **wandb_identity.to_config(),
            "deletion_rate": data_args.deletion_rate,
            "use_bixt": model_args.use_bixt,
            "bixt_token_ffn": model_args.bixt_token_ffn,
            "decoder_type": model_args.decoder_type,
            "decoder_num_layers": model_args.decoder_num_layers,
            "checkpoint_family": config.checkpoint_family,
            "pretraining_objective": config.pretraining_objective,
            "objective_variant": model_args.objective_variant,
            "anchor_loss": model_args.anchor_loss,
            "anchor_model_name": model_args.anchor_model_name if model_args.anchor_loss else None,
            "anchor_loss_weight": model_args.anchor_loss_weight,
            "anchor_standardize": model_args.anchor_standardize,
            "anchor_head_layers": model_args.anchor_head_layers,
            "contrastive_weight": model_args.contrastive_weight,
            "contrastive_temperature": model_args.contrastive_temperature,
            "prefix_ratio_min": data_args.prefix_ratio_min,
            "prefix_ratio_max": data_args.prefix_ratio_max,
            "min_prefix_content": data_args.min_prefix_content,
            "min_suffix_content": data_args.min_suffix_content,
            "split_strategy": data_args.split_strategy,
            "dataset_mix": data_args.dataset_mix,
            "dataset_mix_recipe": data_args.dataset_mix_recipe,
            "dataset_mix_weight_override": data_args.dataset_mix_weight_override,
            "optimizer": optim_args.optimizer,
            "muon_adamw_lr": optim_args.muon_adamw_lr,
            "muon_momentum": optim_args.muon_momentum,
            **(
                # E10 backbone-concept graft knobs (also surfaces trainable_params explicitly
                # — the LoRA fraction is the headline number for this family).
                {
                    "backbone_model": model_args.backbone_model,
                    "concept_block": model_args.concept_block,
                    "concept_io_mode": model_args.concept_io_mode,
                    "lora_r": model_args.lora_r,
                    "lora_alpha": model_args.lora_alpha,
                    "lora_dropout": model_args.lora_dropout,
                    "lora_targets": model_args.lora_targets,
                    "global_attention_mode": config.global_attention_mode,
                    "arm": "concept" if model_args.concept_num > 0 else "control",
                }
                if is_backbone else {}
            ),
        },
    )

    # Train collator samples fresh corruption per call; the eval collator is seeded
    # so the held-out set always sees the same deletions / split points (stable
    # eval_loss, fair best-checkpoint selection).
    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM:
        # No corruption/splitting is sampled, so the same (deterministic) collator serves
        # both train and eval.
        data_collator = DataCollatorForCausalLM(tokenizer, max_length=data_args.max_seq_length)
        eval_data_collator = DataCollatorForCausalLM(tokenizer, max_length=data_args.max_seq_length)
    elif model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        prefix_collator_kwargs = dict(
            max_length=data_args.max_seq_length,
            prefix_ratio_min=data_args.prefix_ratio_min,
            prefix_ratio_max=data_args.prefix_ratio_max,
            min_prefix_content=data_args.min_prefix_content,
            min_suffix_content=data_args.min_suffix_content,
            split_strategy=data_args.split_strategy,
        )
        data_collator = DataCollatorForPrefixGeneration(tokenizer, **prefix_collator_kwargs)
        eval_data_collator = DataCollatorForPrefixGeneration(
            tokenizer, seed=training_args.seed, **prefix_collator_kwargs
        )
    else:
        data_collator = DataCollatorForTSDAE(
            tokenizer,
            deletion_rate=data_args.deletion_rate,
            max_length=data_args.max_seq_length,
        )
        eval_data_collator = DataCollatorForTSDAE(
            tokenizer,
            deletion_rate=data_args.deletion_rate,
            max_length=data_args.max_seq_length,
            seed=training_args.seed,
        )

    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())

    trainer = PerceiverDenoiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        objective_variant=model_args.objective_variant,
        contrastive_weight=model_args.contrastive_weight,
        contrastive_temperature=model_args.contrastive_temperature,
        compute_concept_ablation=is_causal_ar or (is_backbone and model_args.concept_num > 0),
        eval_data_collator=eval_data_collator,
        anchor_loss=model_args.anchor_loss,
        anchor_loss_weight=model_args.anchor_loss_weight,
        anchor_standardize=model_args.anchor_standardize,
        anchor_model_name=model_args.anchor_model_name,
        optimizer_choice=optim_args.optimizer,
        muon_adamw_lr=optim_args.muon_adamw_lr,
        muon_momentum=optim_args.muon_momentum,
    )

    if is_backbone:
        decoder_desc = (
            f"backbone_concept ({model_args.backbone_model}, frozen+LoRA r={model_args.lora_r}, "
            f"C={model_args.concept_num}, K={model_args.concept_block}, "
            f"io={model_args.concept_io_mode})"
        )
    else:
        decoder_desc = (
            f"causal_ar (AR, {model_args.decoder_num_layers}L, pos={model_args.decoder_pos_type}, "
            f"word_dropout={model_args.decoder_word_dropout})"
            if is_causal_ar
            else f"perceiver_posonly ({model_args.decoder_num_layers}L)"
        )
    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM:
        objective_desc = (
            f"causal_lm (block-recurrent next-token CE, block K={model_args.concept_block}, "
            f"concepts={'on' if model_args.concept_num > 0 else 'OFF (control arm)'})"
        )
    elif model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        objective_desc = (
            f"prefix_suffix (encoder sees prefix {data_args.prefix_ratio_min:.2f}-"
            f"{data_args.prefix_ratio_max:.2f} via {data_args.split_strategy}, decoder generates suffix)"
        )
    elif model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        objective_desc = (
            f"reconstruction+contrastive (TSDAE deletion={data_args.deletion_rate}, "
            f"contrastive_weight={model_args.contrastive_weight})"
        )
    else:
        objective_desc = f"reconstruction (TSDAE denoising, deletion={data_args.deletion_rate})"

    logger.info("=" * 60)
    logger.info(f"STARTING TRAINING: {datetime.now()}")
    logger.info(f"  Run id          : {run_identifier}")
    logger.info(f"  W&B group       : {wandb_identity.group}")
    logger.info(f"  W&B job_type    : {wandb_identity.job_type}")
    logger.info(f"  Model type      : {model_type_str}  ({type(model).__name__})")
    logger.info(f"  Pretraining obj : {config.pretraining_objective}")
    logger.info(f"  Objective       : {objective_desc}")
    logger.info(f"  Decoder         : {decoder_desc}")
    if is_backbone:
        logger.info(
            f"  Backbone        : {model_args.backbone_model} "
            f"C{config.concept_num} K{config.concept_block} lora_r={config.lora_r} "
            f"targets={config.lora_targets}"
        )
    else:
        logger.info(
            f"  Encoder         : H{config.hidden_size} L{config.num_hidden_layers} "
            f"C{config.concept_num} token_emb={config.token_embedding_dim} "
            f"act={config.hidden_act} norm={config.norm_type} bixt={model_args.use_bixt}"
        )
    if data_args.dataset_mix_recipe or data_args.dataset_mix:
        selected_mix = data_args.dataset_mix_recipe or data_args.dataset_mix
        logger.info(
            f"  Data            : mix={selected_mix} tokenizer={data_args.tokenizer_name} "
            f"max_seq={data_args.max_seq_length}"
            + (
                f" weight_override={data_args.dataset_mix_weight_override}"
                if data_args.dataset_mix_weight_override
                else ""
            )
        )
    else:
        logger.info(
            f"  Data            : {data_args.dataset_name} {data_args.dataset_name_subset or ''} "
            f"tokenizer={data_args.tokenizer_name} max_seq={data_args.max_seq_length}"
        )
    logger.info(f"  Eval collator   : seeded={getattr(eval_data_collator, 'seed', None)} "
                f"(deterministic held-out corruption)")
    logger.info("=" * 60)
    trainer.train()

    final_path = os.path.join(training_args.output_dir, run_identifier)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved model to: {final_path}")

    if wandb.run and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()

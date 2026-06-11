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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)
from transformers.modeling_outputs import MaskedLMOutput

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collators import DataCollatorForPrefixGeneration, DataCollatorForTSDAE
from data.dataset_preprocess import load_and_preprocess_text_dataset
from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.loss_manager import ConceptLossStepCallback, LossConfig, get_available_losses
from training.utils_training import (
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
VALID_OBJECTIVES = {
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    OBJECTIVE_PREFIX_SUFFIX,
}


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.objective_variant = objective_variant
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.compute_concept_ablation = compute_concept_ablation
        self.concept_ablation_batches = concept_ablation_batches
        self.eval_data_collator = eval_data_collator

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
        dataloader = self.get_eval_dataloader()
        device = self.args.device
        sums: dict = {}
        n = 0
        for i, batch in enumerate(dataloader):
            if i >= self.concept_ablation_batches:
                break
            labels = batch["labels"].to(device)
            if "prefix_input_ids" in batch:
                prefix_attention_mask = batch.get("prefix_attention_mask")
                if prefix_attention_mask is not None:
                    prefix_attention_mask = prefix_attention_mask.to(device)
                m = base_model.concept_ablation_ce(
                    prefix_input_ids=batch["prefix_input_ids"].to(device),
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_input_ids=batch["suffix_input_ids"].to(device),
                    labels=labels,
                )
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                m = base_model.concept_ablation_ce(input_ids, attention_mask, labels)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        if n == 0:
            return {}
        return {f"concept_ablation/{k}": v / n for k, v in sums.items()}

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
        if (
            not model.training
            or self.objective_variant in {OBJECTIVE_RECONSTRUCTION, OBJECTIVE_PREFIX_SUFFIX}
        ):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Unwrap DDP wrapper to access custom methods (encode_concepts, etc.).
        # Gradient sync still works: with find_unused_parameters=False, DDP's
        # parameter-level backward hooks fire regardless of the forward path.
        base_model = model.module if hasattr(model, "module") else model

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
        norm_type=model_args.norm_type,
        checkpoint_family=checkpoint_family,
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode=canonical_single_eval_mode,
        pretraining_objective=objective_name,
    )


def main():
    setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    parser = HfArgumentParser((ModelArguments, LossArguments, DataTrainingArguments, TrainingArguments))
    model_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # For the AR decoder, append EOS to every document so the model learns to stop.
    append_eos_token_id = (
        tokenizer.eos_token_id if (is_causal_ar and tokenizer.eos_token_id is not None) else None
    )

    logger.info(f"Loading dataset: {data_args.dataset_name}")
    with training_args.main_process_first(desc="loading and tokenizing dataset"):
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
        )

    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")
    logger.info("=" * 60)

    config = build_perceiver_denoise_config(tokenizer, model_args, data_args)
    loss_config = loss_args.to_loss_config()
    log_loss_config(loss_config)

    model_class = ConceptEncoderForConditionalLM if is_causal_ar else ConceptEncoderForDenoisingPerceiver
    logger.info(f"Initializing {model_class.__name__}")
    model = model_class(config, loss_config=loss_config)

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)
        logger.info("Loaded pretrained encoder weights. Decoder and objective head use the current config.")

    if is_causal_ar:
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

    if torch.cuda.is_available() and is_main_process():
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_causal_ar and model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        run_prefix = "concept_ar_prefix"
    else:
        run_prefix = "concept_ar" if is_causal_ar else "perceiver_denoise"
    run_identifier = (
        f"{run_prefix}_H{model_args.hidden_size}"
        f"L{model_args.num_hidden_layers}"
        f"C{model_args.concept_num}"
        f"D{model_args.decoder_num_layers}_{timestamp}"
    )
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
        "Prefix ratio min": data_args.prefix_ratio_min,
        "Prefix ratio max": data_args.prefix_ratio_max,
        "Split strategy": data_args.split_strategy,
    }
    if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        training_extra_fields["Contrastive weight"] = model_args.contrastive_weight

    log_training_config(training_args, extra_fields=training_extra_fields)

    wandb_tags = ["perceiver-denoise", "concept-encoder", model_args.objective_variant]
    if model_args.use_bixt:
        wandb_tags.append("bixt")

    init_wandb(
        training_args,
        model,
        config,
        data_args,
        loss_config,
        "perceiver_denoise",
        run_identifier,
        job_type="perceiver-denoising-pretraining",
        model_type=model_type_str,
        wandb_tags=wandb_tags,
        extra_config={
            "deletion_rate": data_args.deletion_rate,
            "use_bixt": model_args.use_bixt,
            "bixt_token_ffn": model_args.bixt_token_ffn,
            "decoder_num_layers": model_args.decoder_num_layers,
            "objective_variant": model_args.objective_variant,
            "contrastive_weight": model_args.contrastive_weight,
            "contrastive_temperature": model_args.contrastive_temperature,
            "prefix_ratio_min": data_args.prefix_ratio_min,
            "prefix_ratio_max": data_args.prefix_ratio_max,
            "min_prefix_content": data_args.min_prefix_content,
            "min_suffix_content": data_args.min_suffix_content,
            "split_strategy": data_args.split_strategy,
        },
    )

    # Train collator samples fresh corruption per call; the eval collator is seeded
    # so the held-out set always sees the same deletions / split points (stable
    # eval_loss, fair best-checkpoint selection).
    if model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
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
        compute_concept_ablation=is_causal_ar,
        eval_data_collator=eval_data_collator,
    )

    logger.info("=" * 60)
    logger.info(f"Starting perceiver denoising pretraining: {datetime.now()}")
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

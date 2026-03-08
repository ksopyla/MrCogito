"""
Perceiver denoising pretraining.

Maintained perceiver training path:
  - BiXT encoder by default
  - position-only stacked decoder
  - full-sequence reconstruction from deleted-token inputs
  - optional stage-2 SimCSE-style contrastive objective
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

from data.data_collators import DataCollatorForTSDAE
from data.dataset_preprocess import load_and_preprocess_text_dataset
from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForDenoisingPerceiver
from nn.loss_manager import ConceptLossStepCallback, LossConfig, get_available_losses
from training.utils_training import (
    init_wandb,
    is_main_process,
    log_data_config,
    log_loss_config,
    log_model_info,
    log_system_info,
    log_training_config,
    setup_run_dirs,
)

logger = logging.get_logger(__name__)

OBJECTIVE_RECONSTRUCTION = "reconstruction"
OBJECTIVE_RECONSTRUCTION_CONTRASTIVE = "reconstruction+contrastive"
VALID_OBJECTIVES = {
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
}


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
        metadata={"help": "Number of stacked position-only decoder layers."},
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
        metadata={"help": "One of: reconstruction, reconstruction+contrastive."},
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


class PerceiverDenoiseTrainer(Trainer):
    def __init__(
        self,
        *args,
        objective_variant: str,
        contrastive_weight: float,
        contrastive_temperature: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.objective_variant = objective_variant
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

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
        if not model.training or self.objective_variant == OBJECTIVE_RECONSTRUCTION:
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        encoder_outputs_a = model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        concept_repr_a = encoder_outputs_a.last_hidden_state
        decoder_output = model.decode_from_concepts(concept_repr_a, seq_length=input_ids.size(1))
        logits, task_loss = model.reconstruction_loss(decoder_output, labels)

        if model.loss_manager.is_enabled:
            total_loss = model.loss_manager(task_loss=task_loss, concept_repr=concept_repr_a)
        else:
            total_loss = task_loss

        encoder_outputs_b = model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        contrastive_loss = self._contrastive_loss(
            model,
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
    objective_name = (
        "denoising_full_reconstruction"
        if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION
        else "denoising_full_reconstruction_contrastive"
    )
    return ConceptEncoderConfig(
        vocab_size=len(tokenizer),
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        token_embedding_dim=model_args.token_embedding_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=8,
        intermediate_size=model_args.intermediate_size,
        max_sequence_length=data_args.max_seq_length,
        concept_position_type=model_args.concept_position_type,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        tie_word_embeddings=model_args.token_embedding_dim == model_args.hidden_size,
        tokenizer_name=data_args.tokenizer_name,
        use_bixt=model_args.use_bixt,
        bixt_token_ffn=model_args.bixt_token_ffn,
        decoder_posonly=True,
        decoder_num_layers=model_args.decoder_num_layers,
        checkpoint_family="perceiver_denoise",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="via_decoder",
        pretraining_objective=objective_name,
    )


def main():
    parser = HfArgumentParser((ModelArguments, LossArguments, DataTrainingArguments, TrainingArguments))
    model_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.objective_variant not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective_variant: {model_args.objective_variant}. "
            f"Expected one of {sorted(VALID_OBJECTIVES)}."
        )

    set_seed(training_args.seed)
    log_system_info()

    log_data_config(
        data_args,
        extra_fields={
            "Deletion rate": data_args.deletion_rate,
            "Objective": model_args.objective_variant,
        },
    )

    logger.info(f"Loading tokenizer: {data_args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_name,
        cache_dir=data_args.dataset_cache_dir,
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
        )

    config = build_perceiver_denoise_config(tokenizer, model_args, data_args)
    loss_config = loss_args.to_loss_config()
    log_loss_config(loss_config)

    logger.info("Initializing ConceptEncoderForDenoisingPerceiver")
    model = ConceptEncoderForDenoisingPerceiver(config, loss_config=loss_config)

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained = ConceptEncoderForDenoisingPerceiver.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)
        logger.info("Loaded pretrained encoder weights. Decoder and objective head use the current config.")

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

    if model_args.torch_compile_dynamic and torch.cuda.is_available():
        backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
        logger.info(f"torch.compile(dynamic=True, backend='{backend}')")
        model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_identifier = (
        f"perceiver_denoise_H{model_args.hidden_size}"
        f"L{model_args.num_hidden_layers}"
        f"C{model_args.concept_num}"
        f"D{model_args.decoder_num_layers}_{timestamp}"
    )
    setup_run_dirs(training_args, run_identifier)

    log_training_config(
        training_args,
        extra_fields={
            "Deletion rate": data_args.deletion_rate,
            "BiXT encoder": model_args.use_bixt,
            "Decoder layers": model_args.decoder_num_layers,
            "Objective": model_args.objective_variant,
            "Contrastive weight": model_args.contrastive_weight,
        },
    )

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
        },
    )

    data_collator = DataCollatorForTSDAE(
        tokenizer,
        deletion_rate=data_args.deletion_rate,
        max_length=data_args.max_seq_length,
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

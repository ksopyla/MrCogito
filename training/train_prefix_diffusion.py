"""
SODA-inspired Prefix Generation — Pretraining Script.

Trains ConceptEncoderForPrefixDiffusion: the encoder sees clean prefix tokens
and compresses them into concept vectors; the diffusion decoder generates
suffix tokens conditioned only on those concepts.

This forces the concept bottleneck to capture semantic gist rather than
surface-level token patterns, addressing concept collapse (rank 5/128)
observed in self-reconstruction diffusion.

Usage (on Polonez / Odra via accelerate):
    accelerate launch --num_processes=4 --mixed_precision=bf16 --multi_gpu \
        training/train_prefix_diffusion.py \
        --hidden_size 512 --num_hidden_layers 6 --concept_num 128 \
        --decoder_layers 2 \
        --dataset_name JeanKaddour/minipile \
        --tokenizer_name answerdotai/ModernBERT-base \
        --num_train_epochs 20 --learning_rate 3e-4 \
        --per_device_train_batch_size 64 \
        --output_dir Cache/Training \
        --bf16

References:
    - SODA (Hudson et al., CVPR 2024) — bottleneck diffusion + novel-view synthesis
    - MDLM (Sahoo et al., NeurIPS 2024) — ELBO = weighted MLM
    - LLaDA (Nie et al., 2025) — loss/p_mask weighting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from datetime import datetime
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    logging,
)

import torch
from dataclasses import dataclass, field
from typing import Optional

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_diffusion import ConceptEncoderForPrefixDiffusion
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
from nn.loss_manager import LossConfig, ConceptLossStepCallback, get_available_losses

from data.dataset_preprocess import load_and_preprocess_text_dataset
from data.data_collators import DataCollatorForPrefixGeneration
from training.utils_training import (
    log_system_info,
    log_model_info,
    log_data_config,
    log_loss_config,
    log_training_config,
    setup_run_dirs,
    init_wandb,
)

logger = logging.get_logger(__name__)


# ============================================================================
# Argument dataclasses
# ============================================================================

@dataclass
class ModelArguments:
    hidden_size: int = field(default=512)
    token_embedding_dim: int = field(
        default=0,
        metadata={"help": "0 = same as hidden_size (Dimension Inversion disabled)"}
    )
    num_hidden_layers: int = field(default=6)
    concept_num: int = field(default=128)
    intermediate_size: int = field(default=2048)
    concept_position_type: str = field(default="none")
    use_bixt: bool = field(
        default=False,
        metadata={"help": "Use BiXT bidirectional cross-attention encoder layers."}
    )
    bixt_token_ffn: bool = field(
        default=True,
        metadata={"help": "Add token FFN in BiXT layers (cheap at small token_dim)."}
    )
    decoder_layers: int = field(
        default=2,
        metadata={"help": "Transformer layers in diffusion decoder (keep small: 1-4)"}
    )
    t_min: float = field(
        default=0.3,
        metadata={"help": "Minimum noise level sampled during training."}
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "Label smoothing for cross-entropy loss."}
    )
    elbo_weight: bool = field(
        default=True,
        metadata={"help": "ELBO-derived per-token 1/t loss weighting (MDLM/LLaDA)."}
    )
    torch_compile_dynamic: bool = field(
        default=False,
        metadata={"help": "Compile model with torch.compile(dynamic=True)."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained encoder for warm-starting."}
    )


@dataclass
class LossArguments:
    concept_losses: Optional[str] = field(
        default="none",
        metadata={"help": f"Space-separated concept loss names or 'none'. "
                          f"Available: {get_available_losses()}"}
    )
    loss_weighting: str = field(
        default="fixed",
        metadata={"choices": ["fixed", "learnable", "kendall_gal"]}
    )
    loss_weight: float = field(default=0.02)
    uniformity_temperature: float = field(default=2.0)
    concept_loss_warmup_steps: int = field(default=0)

    def to_loss_config(self) -> LossConfig:
        if self.concept_losses is None or self.concept_losses.lower() == "none":
            return LossConfig.disabled()
        losses = self.concept_losses.split()
        loss_weights = {"task": 1.0}
        if self.loss_weighting == "fixed":
            per = self.loss_weight / len(losses) if losses else 0
            for n in losses:
                loss_weights[n] = per
        loss_params = {}
        if "uniformity" in losses or "combined" in losses:
            loss_params["uniformity"] = {"temperature": self.uniformity_temperature}
            loss_params["combined"] = {"temperature": self.uniformity_temperature}
        return LossConfig(
            concept_losses=losses,
            weighting_strategy=self.loss_weighting,
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
    prefix_ratio_min: float = field(
        default=0.3,
        metadata={"help": "Minimum fraction of content tokens in prefix (0.3 = 30%)."}
    )
    prefix_ratio_max: float = field(
        default=0.5,
        metadata={"help": "Maximum fraction of content tokens in prefix (0.5 = 50%)."}
    )


# ============================================================================
# Custom Trainer
# ============================================================================

class PrefixDiffusionTrainer(Trainer):
    """Trainer subclass that extracts loss from DiffusionOutput."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((
        ModelArguments, LossArguments, DataTrainingArguments, TrainingArguments,
    ))
    model_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    log_system_info()

    log_data_config(data_args, extra_fields={
        "Prefix ratio": f"[{data_args.prefix_ratio_min}, {data_args.prefix_ratio_max}]",
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_name,
        cache_dir=data_args.dataset_cache_dir,
    )

    # Dataset
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

    # Token embedding dim
    effective_token_dim = (
        model_args.token_embedding_dim
        if model_args.token_embedding_dim > 0
        else model_args.hidden_size
    )

    # Model config
    config = ConceptEncoderConfig(
        vocab_size=len(tokenizer),
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        token_embedding_dim=effective_token_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=8,
        intermediate_size=model_args.intermediate_size,
        max_sequence_length=data_args.max_seq_length,
        concept_position_type=model_args.concept_position_type,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        use_bixt=model_args.use_bixt,
        bixt_token_ffn=model_args.bixt_token_ffn,
    )

    loss_config = loss_args.to_loss_config()

    log_loss_config(loss_config)

    logger.info("Initializing ConceptEncoderForPrefixDiffusion")
    model = ConceptEncoderForPrefixDiffusion(
        config,
        loss_config=loss_config,
        decoder_layers=model_args.decoder_layers,
        t_min=model_args.t_min,
        label_smoothing=model_args.label_smoothing,
        elbo_weight=model_args.elbo_weight,
    )

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained = ConceptEncoderForMaskedLMPerceiver.from_pretrained(
            model_args.model_name_or_path,
        )
        model.encoder.load_state_dict(pretrained.encoder.state_dict())
        logger.info("Loaded pretrained encoder weights. Decoder uses random init.")

    model_type_str = "prefix_diffusion_bixt" if model_args.use_bixt else "prefix_diffusion"
    log_model_info(
        model, config=config,
        model_type=model_type_str,
        model_description="Prefix Generation (SODA-style)",
    )

    # torch.compile
    if model_args.torch_compile_dynamic:
        if not torch.cuda.is_available():
            logger.warning("torch_compile_dynamic=True but no CUDA — skipping.")
        else:
            backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
            logger.info(f"Applying torch.compile(dynamic=True, backend='{backend}') ...")
            model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)

    # Run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bixt_tag = "BiXT" if model_args.use_bixt else ""
    base_id = (
        f"prefix_diff{bixt_tag}_H{model_args.hidden_size}"
        f"L{model_args.num_hidden_layers}"
        f"C{model_args.concept_num}D{model_args.decoder_layers}"
    )
    run_identifier = f"{base_id}_{timestamp}"

    setup_run_dirs(training_args, run_identifier)

    log_training_config(training_args, extra_fields={
        "BiXT encoder": model_args.use_bixt,
        "t_min": model_args.t_min,
        "ELBO weighting": model_args.elbo_weight,
        "Decoder layers": model_args.decoder_layers,
    })

    wandb_tags = ["prefix_diffusion", "soda-style"]
    if model_args.use_bixt:
        wandb_tags.append("bixt")

    init_wandb(
        training_args, model, config, data_args, loss_config,
        base_id, run_identifier,
        job_type="prefix-diffusion-pretraining",
        model_type=model_type_str,
        wandb_tags=wandb_tags,
        notes=f"SODA-style prefix generation, Dataset: {data_args.dataset_name}",
        extra_config={
            "decoder_layers": model_args.decoder_layers,
            "t_min": model_args.t_min,
            "label_smoothing": model_args.label_smoothing,
            "elbo_weight": model_args.elbo_weight,
            "use_bixt": model_args.use_bixt,
            "prefix_ratio_min": data_args.prefix_ratio_min,
            "prefix_ratio_max": data_args.prefix_ratio_max,
        },
    )

    # Data collator
    data_collator = DataCollatorForPrefixGeneration(
        tokenizer,
        max_length=data_args.max_seq_length,
        prefix_ratio_min=data_args.prefix_ratio_min,
        prefix_ratio_max=data_args.prefix_ratio_max,
    )

    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())

    trainer = PrefixDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("=" * 60)
    logger.info(f"Starting prefix diffusion pretraining: {datetime.now()}")
    logger.info("=" * 60)
    trainer.train()

    final_path = os.path.join(training_args.output_dir, run_identifier)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved model to: {final_path}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()

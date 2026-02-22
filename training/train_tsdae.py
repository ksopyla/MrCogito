"""
Concept Encoder + TSDAE (Denoising Auto-Encoder) — Pretraining Script.

Trains ConceptEncoderForMaskedLMPerceiverPosOnly with TSDAE objective:
  - Randomly "deletes" ~60% of input tokens (attention_mask zeroed)
  - Encoder sees only surviving tokens via key_padding_mask
  - PosOnly decoder reconstructs the FULL original sequence from concepts
  - Dense CE loss at ALL non-pad positions (not sparse MLM)

Why TSDAE instead of MLM:
  - No [MASK] token pollution (deleted tokens are invisible, not embedded)
  - No input-embedding shortcut (PosOnly decoder queries are position-only)
  - ALL positions contribute gradient (vs 15% in MLM → 83x stronger signal)
  - 60% deletion rate forces semantic concept encoding, not local co-occurrence

Reference:
  Wang et al., "TSDAE: Using Transformer-based Sequential Denoising
  Auto-Encoder for Unsupervised Sentence Embedding Learning", EMNLP 2021.

Usage (local):
    poetry run python training/train_tsdae.py ^
        --hidden_size 512 --num_hidden_layers 6 --concept_num 128 ^
        --dataset_name JeanKaddour/minipile ^
        --tokenizer_name answerdotai/ModernBERT-base ^
        --num_train_epochs 20 --learning_rate 3e-4 ^
        --per_device_train_batch_size 16 ^
        --output_dir Cache/Training --bf16

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes=4 --mixed_precision=bf16 --multi_gpu \\
        training/train_tsdae.py \\
        --hidden_size 512 --num_hidden_layers 6 --concept_num 128 \\
        --dataset_name JeanKaddour/minipile \\
        --num_train_epochs 20 --learning_rate 3e-4 \\
        --per_device_train_batch_size 64 \\
        --output_dir Cache/Training --bf16
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
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiverPosOnly
from nn.loss_manager import LossConfig, get_available_losses

from training.data_collators import DataCollatorForTSDAE
from training.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    count_parameters,
    is_main_process,
    log_system_info,
    log_model_info,
    get_git_info,
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
        metadata={"help": "Use BiXT bidirectional cross-attention layers (tokens update from concepts at each layer). "
                          "Preserves O(C*N) complexity while contextualising token representations."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained encoder checkpoint for warm-start."}
    )
    torch_compile_dynamic: bool = field(
        default=False,
        metadata={"help": "torch.compile(dynamic=True). Keep TrainingArguments.torch_compile=False."}
    )


@dataclass
class LossArguments:
    concept_losses: Optional[str] = field(
        default="none",
        metadata={"help": f"Space-separated concept loss names or 'none'. Available: {get_available_losses()}"}
    )
    loss_weighting: str = field(
        default="fixed",
        metadata={"choices": ["fixed", "learnable", "kendall_gal"]}
    )
    loss_weight: float = field(default=0.05)
    uniformity_temperature: float = field(default=2.0)

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
        )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(default="JeanKaddour/minipile")
    dataset_name_subset: Optional[str] = field(default=None)
    tokenizer_name: str = field(default="answerdotai/ModernBERT-base")
    max_seq_length: int = field(default=512)
    test_size_percent: float = field(default=0.1)
    dataset_cache_dir: Optional[str] = field(default=None)
    deletion_rate: float = field(
        default=0.6,
        metadata={"help": "Fraction of non-special tokens to delete per sample (TSDAE noise rate). "
                          "Higher = harder reconstruction, stronger concept signal."}
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, LossArguments, DataTrainingArguments, TrainingArguments))
    model_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    log_system_info()

    # Tokenizer
    logger.info(f"Loading tokenizer: {data_args.tokenizer_name}")
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

    effective_token_dim = (
        model_args.token_embedding_dim
        if model_args.token_embedding_dim > 0
        else model_args.hidden_size
    )

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
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        use_bixt=model_args.use_bixt,
    )

    loss_config = loss_args.to_loss_config()

    logger.info("Initializing ConceptEncoderForMaskedLMPerceiverPosOnly (TSDAE)")
    model = ConceptEncoderForMaskedLMPerceiverPosOnly(config, loss_config=loss_config)

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
        pretrained = ConceptEncoderForMaskedLMPerceiver.from_pretrained(model_args.model_name_or_path)
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)
        logger.info("Loaded pretrained encoder weights. Decoder uses random init.")

    log_model_info(model, config=config, model_type="tsdae_posonly",
                   model_description="Concept + TSDAE PosOnly")

    if model_args.torch_compile_dynamic:
        if torch.cuda.is_available():
            backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
            logger.info(f"torch.compile(dynamic=True, backend='{backend}')")
            model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)

    # Run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bixt_tag = "BiXT" if model_args.use_bixt else ""
    base_id = (f"tsdae_posonly_{bixt_tag}H{model_args.hidden_size}"
               f"L{model_args.num_hidden_layers}C{model_args.concept_num}")
    run_identifier = f"{base_id}_{timestamp}"

    output_dir = training_args.output_dir or "./outputs"
    if not output_dir.endswith(run_identifier):
        training_args.output_dir = os.path.join(output_dir, run_identifier)

    if not training_args.logging_dir:
        training_args.logging_dir = os.path.join(
            os.path.dirname(training_args.output_dir), "logs", run_identifier
        )
    elif not training_args.logging_dir.endswith(run_identifier):
        training_args.logging_dir = os.path.join(training_args.logging_dir, run_identifier)

    training_args.run_name = run_identifier
    training_args.report_to = ["tensorboard", "wandb"]
    training_args.push_to_hub = False
    training_args.remove_unused_columns = False
    training_args.fp16 = not training_args.bf16

    if is_main_process():
        total_params, trainable_params = count_parameters(model)
        wandb.init(
            project="MrCogito",
            id=run_identifier,
            name=run_identifier,
            job_type="tsdae-pretraining",
            config={
                "model_type": "tsdae_posonly",
                "hidden_size": model_args.hidden_size,
                "num_hidden_layers": model_args.num_hidden_layers,
                "concept_num": model_args.concept_num,
                "use_bixt": model_args.use_bixt,
                "deletion_rate": data_args.deletion_rate,
                "concept_losses": loss_args.concept_losses,
                "loss_weighting": loss_args.loss_weighting,
                "dataset": data_args.dataset_name,
                "total_params": total_params,
                **{f"git_{k}": v for k, v in get_git_info().items()},
            },
            tags=["tsdae", "concept-encoder", "posonly", data_args.dataset_name],
        )

    data_collator = DataCollatorForTSDAE(
        tokenizer,
        deletion_rate=data_args.deletion_rate,
        max_length=data_args.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("=" * 60)
    logger.info(f"Starting TSDAE pretraining: {datetime.now()}")
    logger.info(f"  Deletion rate: {data_args.deletion_rate}")
    logger.info(f"  BiXT: {model_args.use_bixt}")
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

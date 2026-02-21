"""
Concept Encoder + Masked Diffusion Decoder — Pretraining Script.

Trains ConceptEncoderForMaskedDiffusion on text corpora.  The objective is
Masked Discrete Diffusion: each batch samples a noise level t ~ Uniform(t_min, 1)
and masks that fraction of tokens independently.  The model must predict the
clean tokens at all masked positions using both surviving tokens AND concept
vectors from the encoder.

Why this instead of MLM?
  - Variable masking (0%–100%) vs fixed 15% creates a rich curriculum
  - At high mask rates the model is forced to use concept representations
  - No fundamental tension between compression and reconstruction like in MLM

Usage (on Polonez / Odra via accelerate):
    accelerate launch --num_processes=4 --mixed_precision=bf16 --multi_gpu \
        training/train_diffusion.py \
        --model_type perceiver_diffusion \
        --hidden_size 512 --num_hidden_layers 6 --concept_num 128 \
        --decoder_layers 2 \
        --dataset_name JeanKaddour/minipile \
        --tokenizer_name answerdotai/ModernBERT-base \
        --num_train_epochs 20 --learning_rate 3e-4 \
        --per_device_train_batch_size 64 \
        --concept_losses combined --loss_weighting kendall_gal \
        --output_dir Cache/Training \
        --bf16
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    logging,
)

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_diffusion import ConceptEncoderForMaskedDiffusion
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
from nn.loss_manager import LossConfig, get_available_losses

from training.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    get_parameter_breakdown,
    count_parameters,
    setup_distributed,
    is_main_process,
    get_hostname,
    log_system_info,
    log_model_info,
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
    decoder_layers: int = field(
        default=2,
        metadata={"help": "Transformer layers in diffusion decoder (keep small: 1-4)"}
    )
    t_min: float = field(
        default=0.05,
        metadata={"help": "Minimum noise level sampled during training. "
                  "Avoids trivial t≈0 steps where nothing is masked."}
    )
    # torch.compile is applied MANUALLY here (not via TrainingArguments.torch_compile) so we
    # can pass dynamic=True.  TrainingArguments.torch_compile should be kept False to avoid
    # double-compilation.
    torch_compile_dynamic: bool = field(
        default=False,
        metadata={"help": "Compile model with torch.compile(dynamic=True) for stable training "
                          "with variable-shape tensors. "
                          "Keep TrainingArguments.torch_compile=False when this is True. "
                          "Backend is read from TrainingArguments.torch_compile_backend (default: inductor)."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. "
                          "Used to warm-start the encoder weights."}
    )


@dataclass
class LossArguments:
    concept_losses: Optional[str] = field(
        default="combined",
        metadata={"help": f"Space-separated concept loss names or 'none'. "
                          f"Available: {get_available_losses()}"}
    )
    loss_weighting: str = field(
        default="kendall_gal",
        metadata={"choices": ["fixed", "learnable", "kendall_gal"]}
    )
    loss_weight: float = field(default=0.1)
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


# ============================================================================
# Data collator for masked diffusion
# ============================================================================

class DataCollatorForMaskedDiffusion:
    """
    Collates batches for masked diffusion training.

    Unlike MLM's fixed-rate masking, the noise level t is sampled PER BATCH
    inside the model's forward() so the collator only needs to return clean
    input_ids + attention_mask.  The model handles all masking internally,
    which means:
      - No variable-shape sparse tensors at the collator level
      - torch.compile works without dynamic shapes at this stage
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]

        # Pad to max length in this batch (or max_length)
        max_len = min(max(len(x) for x in input_ids), self.max_length)
        padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(input_ids), max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            ids_t = torch.tensor(ids[:max_len], dtype=torch.long)
            padded_ids[i, : len(ids_t)] = ids_t
            attention_mask[i, : len(ids_t)] = 1

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": padded_ids.clone(),  # Required so HF Trainer computes eval_loss
        }


# ============================================================================
# Custom Trainer (handles the DiffusionOutput structure)
# ============================================================================

class DiffusionTrainer(Trainer):
    """
    Minimal Trainer subclass that extracts `loss` from DiffusionOutput.

    HuggingFace Trainer expects model() to return a dict or a dataclass with
    a `loss` attribute when `labels` is present.  Since we don't have a
    `labels` field (noise is sampled inside forward()), we override
    `compute_loss` to call model directly.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss


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
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )

    loss_config = loss_args.to_loss_config()

    logger.info("Initializing ConceptEncoderForMaskedDiffusion")
    model = ConceptEncoderForMaskedDiffusion(
        config,
        loss_config=loss_config,
        decoder_layers=model_args.decoder_layers,
        t_min=model_args.t_min,
    )

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained_mlm = ConceptEncoderForMaskedLMPerceiver.from_pretrained(model_args.model_name_or_path)
        model.encoder.load_state_dict(pretrained_mlm.encoder.state_dict())
        logger.info("Successfully loaded pretrained encoder weights. Diffusion decoder uses random init.")

    log_model_info(model, config=config, model_type="diffusion", model_description="Concept + Masked Diffusion")

    # Apply torch.compile with dynamic=True AFTER model init, BEFORE Trainer creation.
    # Using dynamic=True prevents constant recompilation caused by variable masked-token counts.
    # Keep training_args.torch_compile=False so HF Trainer does NOT compile again.
    if model_args.torch_compile_dynamic:
        if not torch.cuda.is_available():
            logger.warning("torch_compile_dynamic=True but no CUDA detected — skipping compile.")
        else:
            backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
            logger.info(f"Applying torch.compile(dynamic=True, backend='{backend}') ...")
            model = torch.compile(
                model,
                dynamic=True,    # Handle variable masked-token shapes without recompilation
                fullgraph=False, # Allow graph breaks (safer for complex HF models)
                backend=backend,
            )
            logger.info("torch.compile applied successfully.")

    # Run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_id = (f"diffusion_H{model_args.hidden_size}L{model_args.num_hidden_layers}"
               f"C{model_args.concept_num}D{model_args.decoder_layers}")
    run_identifier = f"{base_id}_{timestamp}"

    # Setup directories
    output_dir = training_args.output_dir if training_args.output_dir else "./outputs"
    
    # If the user passed "--output_dir ./Cache/Training/", HF Trainer will just dump checkpoints
    # straight into that folder instead of a unique run folder. So we append the run_identifier.
    if not output_dir.endswith(run_identifier):
        training_args.output_dir = os.path.join(output_dir, run_identifier)

    # Force logging_dir to be under Cache/logs/run_identifier
    if not training_args.logging_dir:
        # Default fallback
        training_args.logging_dir = os.path.join(os.path.dirname(training_args.output_dir), "logs", run_identifier)
    else:
        # Ensure it also gets the run_identifier folder
        if not training_args.logging_dir.endswith(run_identifier):
             training_args.logging_dir = os.path.join(training_args.logging_dir, run_identifier)

    training_args.run_name = run_identifier
    training_args.report_to = ["tensorboard", "wandb"]
    training_args.push_to_hub = False
    training_args.remove_unused_columns = False  # keep input_ids + attention_mask
    training_args.fp16 = not training_args.bf16

    if is_main_process():
        total_params, trainable_params = count_parameters(model)
        wandb.init(
            project="MrCogito",
            id=run_identifier,
            name=run_identifier,
            job_type="diffusion-pretraining",
            config={
                "model_type": "concept_diffusion",
                "hidden_size": model_args.hidden_size,
                "num_hidden_layers": model_args.num_hidden_layers,
                "concept_num": model_args.concept_num,
                "decoder_layers": model_args.decoder_layers,
                "t_min": model_args.t_min,
                "concept_losses": loss_args.concept_losses,
                "loss_weighting": loss_args.loss_weighting,
                "dataset": data_args.dataset_name,
                "total_params": total_params,
            },
            tags=["diffusion", "concept-encoder", data_args.dataset_name],
        )

    data_collator = DataCollatorForMaskedDiffusion(tokenizer, max_length=data_args.max_seq_length)

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("=" * 60)
    logger.info(f"Starting diffusion pretraining: {datetime.now()}")
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

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
from typing import Dict, List, Optional, Any

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_diffusion import ConceptEncoderForMaskedDiffusion
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver
from nn.loss_manager import LossConfig, ConceptLossStepCallback, get_available_losses

from data.dataset_preprocess import load_and_preprocess_text_dataset
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
        default=0.3,
        metadata={"help": "Minimum noise level sampled during training. "
                  "0.3+ avoids the near-MLM regime where concepts are unnecessary."}
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "Label smoothing for cross-entropy loss. Prevents overconfident "
                  "predictions that create sharp loss landscapes and gradient explosion."}
    )
    elbo_weight: bool = field(
        default=True,
        metadata={"help": "ELBO-derived per-token 1/t loss weighting (MDLM/LLaDA). "
                  "Normalizes gradient magnitude across noise levels."}
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
    concept_loss_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup steps for concept losses (0 = no warmup). "
                          "Only effective with fixed weighting."}
    )

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
    setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    parser = HfArgumentParser((ModelArguments, LossArguments, DataTrainingArguments, TrainingArguments))
    model_args, loss_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    log_system_info()

    log_data_config(data_args)

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

    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")
    logger.info("=" * 60)

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
        checkpoint_family="diffusion_mlm",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="weighted_pool",
        pretraining_objective="masked_diffusion_self_reconstruction",
    )

    loss_config = loss_args.to_loss_config()

    log_loss_config(loss_config)

    logger.info("Initializing ConceptEncoderForMaskedDiffusion")
    model = ConceptEncoderForMaskedDiffusion(
        config,
        loss_config=loss_config,
        decoder_layers=model_args.decoder_layers,
        t_min=model_args.t_min,
        label_smoothing=model_args.label_smoothing,
        elbo_weight=model_args.elbo_weight,
    )

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained_mlm = ConceptEncoderForMaskedLMPerceiver.from_pretrained(model_args.model_name_or_path)
        model.encoder.load_state_dict(pretrained_mlm.encoder.state_dict())
        logger.info("Successfully loaded pretrained encoder weights. Diffusion decoder uses random init.")

    log_model_info(model, config=config, model_type="diffusion", model_description="Concept + Masked Diffusion")

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

    setup_run_dirs(training_args, run_identifier)
    training_args.use_cpu = False

    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None

    log_training_config(training_args, extra_fields={
        "Decoder layers": model_args.decoder_layers,
        "t_min": model_args.t_min,
        "ELBO weighting": model_args.elbo_weight,
    })

    init_wandb(
        training_args, model, config, data_args, loss_config,
        base_id, run_identifier,
        job_type="diffusion-pretraining",
        model_type="concept_diffusion",
        wandb_tags=["concept_diffusion", "diffusion-pretraining"],
        extra_config={
            "decoder_layers": model_args.decoder_layers,
            "t_min": model_args.t_min,
            "label_smoothing": model_args.label_smoothing,
            "elbo_weight": model_args.elbo_weight,
        },
    )

    data_collator = DataCollatorForMaskedDiffusion(tokenizer, max_length=data_args.max_seq_length)

    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())
        logger.info(f"Concept loss warmup: {loss_config.warmup_steps} steps")

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("=" * 60)
    logger.info(f"Starting diffusion pretraining: {datetime.now()}")
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

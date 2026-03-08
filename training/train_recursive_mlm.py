import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_preprocess import load_and_preprocess_text_dataset
from nn.concept_encoder_recursive import RecursiveConceptEncoderConfig
from nn.concept_encoder_recursive_mlm import RecursiveConceptEncoderForMaskedLM
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
MODEL_CLASS = RecursiveConceptEncoderForMaskedLM
MODEL_DESCRIPTION = "Recursive ConceptEncoder with shared encoder layer applied K times"


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional recursive checkpoint or standard perceiver MLM checkpoint for warm-start."},
    )
    torch_compile_dynamic: bool = field(default=False)
    hidden_size: int = field(default=512)
    token_embedding_dim: int = field(
        default=512,
        metadata={"help": "Token embedding dimension. Use explicit values; no 0 sentinel."},
    )
    intermediate_size: int = field(default=2048)
    num_hidden_layers: int = field(
        default=6,
        metadata={"help": "Number of recursive iterations (K)."},
    )
    concept_num: int = field(default=128)
    concept_position_type: str = field(default="none")


@dataclass
class LossArguments:
    concept_losses: Optional[str] = field(
        default="none",
        metadata={"help": f"Concept loss types to use, space-separated. Available: {get_available_losses()}"},
    )
    loss_weighting: str = field(
        default="fixed",
        metadata={"choices": ["fixed", "learnable", "kendall_gal"]},
    )
    loss_weight: float = field(default=0.02)
    soft_ortho_threshold: float = field(default=0.1)
    uniformity_temperature: float = field(default=2.0)
    concept_loss_warmup_steps: int = field(default=0)

    def to_loss_config(self) -> LossConfig:
        if self.concept_losses is None or self.concept_losses.lower() == "none":
            return LossConfig.disabled()

        losses = self.concept_losses.split()
        loss_weights = {"task": 1.0}
        if self.loss_weighting == "fixed":
            per_loss_weight = self.loss_weight / len(losses) if losses else 0.0
            for loss_name in losses:
                loss_weights[loss_name] = per_loss_weight

        loss_params = {}
        if "soft_orthogonality" in losses:
            loss_params["soft_orthogonality"] = {"threshold": self.soft_ortho_threshold}
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
    mlm_probability: float = field(default=0.25)
    masking_type: str = field(default="random", metadata={"choices": ["random", "whole_word"]})
    max_seq_length: int = field(default=512)
    test_size_percent: float = field(default=0.1)
    dataset_name: str = field(default="JeanKaddour/minipile")
    dataset_name_subset: Optional[str] = field(default=None)
    tokenizer_name: str = field(default="answerdotai/ModernBERT-base")
    dataset_cache_dir: Optional[str] = field(default="./Cache/Datasets")


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LossArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def _build_config(tokenizer, model_args: ModelArguments, data_args: DataTrainingArguments):
    num_attention_heads = max(1, min(8, model_args.hidden_size // 64))
    should_tie = model_args.token_embedding_dim == model_args.hidden_size
    return RecursiveConceptEncoderConfig(
        vocab_size=len(tokenizer),
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        token_embedding_dim=model_args.token_embedding_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_sequence_length=data_args.max_seq_length,
        concept_position_type=model_args.concept_position_type,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        tie_word_embeddings=should_tie,
        tokenizer_name=data_args.tokenizer_name,
        checkpoint_family="recursive_mlm",
        pretraining_objective="masked_language_modeling_recursive",
    )


def _is_recursive_checkpoint(model_name_or_path: str) -> bool:
    try:
        checkpoint_config = AutoConfig.from_pretrained(model_name_or_path)
    except Exception:
        return False

    checkpoint_family = getattr(checkpoint_config, "checkpoint_family", None)
    return (
        getattr(checkpoint_config, "model_type", None) == "recursive_concept_encoder"
        or checkpoint_family == "recursive_mlm"
    )


def main():
    setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    model_args, data_args, loss_args, training_args = parse_args()
    loss_config = loss_args.to_loss_config()
    set_seed(training_args.seed)
    log_system_info()

    log_data_config(
        data_args,
        extra_fields={
            "MLM probability": data_args.mlm_probability,
            "Masking type": data_args.masking_type,
            "Training family": "recursive_mlm",
        },
    )

    logger.info(f"Loading tokenizer: {data_args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)

    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer '{data_args.tokenizer_name}' does not define pad_token_id.")
    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer '{data_args.tokenizer_name}' does not define mask_token_id.")

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

    config = _build_config(tokenizer, model_args, data_args)
    log_loss_config(loss_config)

    logger.info(f"Initializing model: {MODEL_DESCRIPTION}")
    model = MODEL_CLASS(config, loss_config=loss_config)

    if model_args.model_name_or_path:
        logger.info(f"Warm-starting recursive model from: {model_args.model_name_or_path}")
        if _is_recursive_checkpoint(model_args.model_name_or_path):
            model = MODEL_CLASS.from_pretrained(model_args.model_name_or_path, config=config)
            model.set_loss_config(loss_config)
            logger.info("Loaded recursive checkpoint via from_pretrained.")
        else:
            if not os.path.isdir(model_args.model_name_or_path):
                raise ValueError(
                    "Standard perceiver warm-start for recursive MLM requires a local checkpoint directory "
                    "so encoder layer-0 weights can be remapped into the shared recursive layer."
                )
            logger.info("Applying standard perceiver MLM -> recursive warm-start mapping.")
            model = MODEL_CLASS(config, loss_config=loss_config)
            model.load_from_standard_mlm_checkpoint(model_args.model_name_or_path)

    if model_args.torch_compile_dynamic and torch.cuda.is_available():
        backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
        logger.info(f"Applying torch.compile(dynamic=True, backend='{backend}') ...")
        model = torch.compile(model, dynamic=True, fullgraph=False, backend=backend)

    log_model_info(
        model,
        config=config,
        model_type="recursive_mlm",
        model_description=MODEL_DESCRIPTION,
    )

    if data_args.masking_type == "whole_word":
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_id = f"recursive_mlm_H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_num}"
    if config.token_embedding_dim != config.hidden_size:
        base_id += f"_T{config.token_embedding_dim}"
    run_identifier = f"{base_id}_{timestamp}"

    setup_run_dirs(training_args, run_identifier)
    training_args.remove_unused_columns = True
    training_args.use_cpu = False

    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None

    log_training_config(training_args)

    init_wandb(
        training_args,
        model,
        config,
        data_args,
        loss_config,
        base_id,
        run_identifier,
        job_type="recursive-mlm-pretraining",
        model_type="recursive_mlm",
        wandb_tags=["recursive_mlm", "mlm-pretraining", data_args.masking_type],
        extra_config={
            "mlm_probability": data_args.mlm_probability,
            "masking_type": data_args.masking_type,
            "encoder_style": "recursive_weight_tied",
        },
    )

    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("=" * 60)
    logger.info(f"Starting recursive MLM pretraining: {datetime.now()}")
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

"""Compatibility entrypoint for multi-family concept pretraining.

Stage 1 keeps this historical command and its public imports stable while the
implementation lives in neutral concept-pretraining modules.
"""

import os
import sys
from datetime import datetime

import torch
import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    logging,
    set_seed,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_preprocess import configure_text_tokenizer_for_model_vocab
from nn.loss_manager import ConceptLossStepCallback
from training.concept_pretraining_args import (
    DataTrainingArguments,
    LossArguments,
    ModelArguments,
    OptimizerArguments,
    validate_training_configuration,
)
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
from training.concept_pretraining_factories import (
    align_special_tokens_for_training,
    build_distributed_run_identifier,
    build_perceiver_denoise_config,
    build_pretraining_collators,
    build_pretraining_model,
    build_training_wandb_identity,
    load_pretraining_datasets,
)
from training.concept_pretraining_trainer import PerceiverDenoiseTrainer
from training.utils_training import (
    init_wandb,
    is_main_process,
    log_data_config,
    log_loss_config,
    log_model_info,
    log_system_info,
    log_training_config,
    resolve_dataset_identifier,
    setup_distributed,
    setup_file_logging,
    setup_run_dirs,
)


logger = logging.get_logger(__name__)


def main():
    setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    parser = HfArgumentParser(
        (
            ModelArguments,
            LossArguments,
            DataTrainingArguments,
            OptimizerArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        loss_args,
        data_args,
        optim_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    is_causal_ar, is_backbone = validate_training_configuration(
        model_args,
        loss_args,
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

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad nor eos token; cannot train.")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Tokenizer had no pad token; set pad_token=eos_token "
            f"({tokenizer.pad_token!r}, pad_id={tokenizer.pad_token_id})."
        )

    if is_backbone:
        backbone_text_config = AutoConfig.from_pretrained(
            model_args.backbone_model,
            cache_dir=data_args.dataset_cache_dir,
        )
        if configure_text_tokenizer_for_model_vocab(
            tokenizer,
            backbone_text_config.vocab_size,
        ):
            logger.warning(
                "Tokenizer exposes ids beyond the text backbone vocabulary "
                f"({len(tokenizer)} > {backbone_text_config.vocab_size}); literal special-token "
                "strings will be split as ordinary text to prevent invalid embedding indices."
            )

    append_eos_token_id = resolve_append_eos_token_id(
        model_args.objective_variant,
        is_causal_ar,
        tokenizer.eos_token_id,
    )
    train_ds, test_ds = load_pretraining_datasets(
        tokenizer,
        data_args,
        training_args,
        append_eos_token_id,
    )

    loss_config = loss_args.to_loss_config()
    log_loss_config(loss_config)

    model, config, model_type_str = build_pretraining_model(
        tokenizer,
        model_args,
        data_args,
        training_args,
        loss_config,
        is_causal_ar=is_causal_ar,
        is_backbone=is_backbone,
    )

    log_model_info(
        model,
        config=config,
        model_type=model_type_str,
        model_description="BiXT perceiver denoising pretraining",
    )

    if torch.cuda.is_available() and is_main_process() and not is_backbone:
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        try:
            query = torch.zeros(
                1,
                num_heads,
                config.concept_num,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            key = torch.zeros(
                1,
                num_heads,
                512,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ):
                torch.nn.functional.scaled_dot_product_attention(query, key, key)
            logger.info(
                f"Flash Attention v2: ACTIVE  "
                f"(heads={num_heads}, head_dim={head_dim}, dtype=bf16)"
            )
        except Exception as flash_attention_error:
            logger.warning(
                "Flash Attention not available — training will use "
                "memory-efficient / math SDPA. "
                f"Reason: {flash_attention_error}"
            )
        finally:
            del query, key

    if model_args.torch_compile_dynamic and torch.cuda.is_available():
        backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
        logger.info(f"torch.compile(dynamic=True, backend='{backend}')")
        model = torch.compile(
            model,
            dynamic=True,
            fullgraph=False,
            backend=backend,
        )

    env_experiment_id = os.environ.get("WANDB_EXPERIMENT_ID") or os.environ.get(
        "EXPERIMENT_ID"
    )
    wandb_identity = build_training_wandb_identity(
        model_args,
        config,
        is_backbone=is_backbone,
        experiment_id=env_experiment_id,
    )

    run_identifier = build_distributed_run_identifier(
        wandb_identity,
        is_backbone=is_backbone,
        concept_num=model_args.concept_num,
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
            "anchor_model_name": (
                model_args.anchor_model_name if model_args.anchor_loss else None
            ),
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
            "pretokenized_manifest": data_args.pretokenized_manifest,
            "target_tokens": (
                int(os.environ["TARGET_TOKENS"])
                if os.environ.get("TARGET_TOKENS")
                else None
            ),
            "estimated_optimizer_steps": (
                int(os.environ["ESTIMATED_STEPS"])
                if os.environ.get("ESTIMATED_STEPS")
                else None
            ),
            "optimizer": optim_args.optimizer,
            "muon_adamw_lr": optim_args.muon_adamw_lr,
            "muon_momentum": optim_args.muon_momentum,
            **(
                {
                    "backbone_model": model_args.backbone_model,
                    "concept_block": model_args.concept_block,
                    "concept_io_mode": model_args.concept_io_mode,
                    "lora_r": model_args.lora_r,
                    "lora_alpha": model_args.lora_alpha,
                    "lora_dropout": model_args.lora_dropout,
                    "lora_targets": model_args.lora_targets,
                    "global_attention_mode": config.global_attention_mode,
                    "arm": (
                        "concept" if model_args.concept_num > 0 else "control"
                    ),
                }
                if is_backbone
                else {}
            ),
        },
    )

    data_collator, eval_data_collator = build_pretraining_collators(
        tokenizer,
        model,
        model_args,
        data_args,
        training_args,
    )

    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())

    if is_backbone:
        special_token_changes = align_special_tokens_for_training(model, tokenizer)
        if special_token_changes and is_main_process():
            logger.info(
                "Aligned backbone training special-token configs to tokenizer ids: "
                f"{special_token_changes}"
            )

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
        compute_concept_ablation=(
            is_causal_ar or (is_backbone and model_args.concept_num > 0)
        ),
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
            f"backbone_concept ({model_args.backbone_model}, "
            f"frozen+LoRA r={model_args.lora_r}, C={model_args.concept_num}, "
            f"K={model_args.concept_block}, io={model_args.concept_io_mode})"
        )
    else:
        decoder_desc = (
            f"causal_ar (AR, {model_args.decoder_num_layers}L, "
            f"pos={model_args.decoder_pos_type}, "
            f"word_dropout={model_args.decoder_word_dropout})"
            if is_causal_ar
            else f"perceiver_posonly ({model_args.decoder_num_layers}L)"
        )

    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM:
        objective_desc = (
            f"causal_lm (block-recurrent next-token CE, "
            f"block K={model_args.concept_block}, "
            f"concepts={'on' if model_args.concept_num > 0 else 'OFF (control arm)'})"
        )
    elif model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        objective_desc = (
            f"prefix_suffix (encoder sees prefix "
            f"{data_args.prefix_ratio_min:.2f}-{data_args.prefix_ratio_max:.2f} "
            f"via {data_args.split_strategy}, decoder generates suffix)"
        )
    elif model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
        objective_desc = (
            f"reconstruction+contrastive (TSDAE deletion={data_args.deletion_rate}, "
            f"contrastive_weight={model_args.contrastive_weight})"
        )
    else:
        objective_desc = (
            f"reconstruction (TSDAE denoising, deletion={data_args.deletion_rate})"
        )

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
            f"act={config.hidden_act} norm={config.norm_type} "
            f"bixt={model_args.use_bixt}"
        )
    effective_dataset = resolve_dataset_identifier(data_args)
    logger.info(
        f"  Data            : {effective_dataset} "
        f"tokenizer={data_args.tokenizer_name} max_seq={data_args.max_seq_length}"
        + (
            f" weight_override={data_args.dataset_mix_weight_override}"
            if data_args.dataset_mix_weight_override
            else ""
        )
    )
    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM:
        logger.info(
            "  Eval collator   : deterministic=True (causal LM; no corruption)"
        )
    else:
        logger.info(
            f"  Eval collator   : seeded={getattr(eval_data_collator, 'seed', None)} "
            "(deterministic held-out corruption)"
        )
    logger.info("=" * 60)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    final_path = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved model to: {final_path}")

    if wandb.run and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()

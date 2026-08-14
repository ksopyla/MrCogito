"""Model, data, collator, and identity factories for concept pretraining."""

import json
from datetime import datetime

import torch
from transformers import AutoConfig, logging

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
from training.concept_pretraining_args import (
    DataTrainingArguments,
    ModelArguments,
)
from training.concept_pretraining_objectives import (
    DECODER_CAUSAL_AR,
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
)
from training.utils_training import (
    WandbRunIdentity,
    broadcast_object,
    build_perceiver_wandb_identity,
)


logger = logging.get_logger("training.train_concept_pretraining")


def align_special_tokens_for_training(model, tokenizer) -> dict:
    """Align Trainer-facing model configs to canonical scalar tokenizer ids."""
    changes = {}
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is None:
            continue
        for config_name, config in (
            ("config", getattr(model, "config", None)),
            ("generation_config", getattr(model, "generation_config", None)),
        ):
            if config is None:
                continue
            previous = getattr(config, name, None)
            if previous != token_id:
                changes[f"{config_name}.{name}"] = {"from": previous, "to": token_id}
                setattr(config, name, token_id)
    return changes


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
    checkpoint_family = "concept_ar" if is_causal_ar else "perceiver_denoise"
    canonical_single_eval_mode = "weighted_pool" if is_causal_ar else "via_decoder"
    if is_causal_ar and model_args.objective_variant == OBJECTIVE_RECONSTRUCTION:
        objective_name = "ar_denoising_reconstruction"

    anchor_teacher_hidden = None
    if model_args.anchor_loss:
        teacher_cfg = AutoConfig.from_pretrained(model_args.anchor_model_name)
        anchor_teacher_hidden = teacher_cfg.hidden_size
        if teacher_cfg.vocab_size != len(tokenizer):
            raise ValueError(
                "anchor_loss requires the model tokenizer to match the teacher vocab for "
                f"1:1 token alignment: tokenizer has {len(tokenizer)} tokens but "
                f"{model_args.anchor_model_name} has {teacher_cfg.vocab_size}. "
                f"Use TOKENIZER_NAME={model_args.anchor_model_name} "
                "(or a same-vocab tokenizer)."
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


def load_pretraining_datasets(
    tokenizer,
    data_args,
    training_args,
    append_eos_token_id,
):
    """Load the selected pretokenized, recipe, registry-mix, or direct-Hub route."""
    with training_args.main_process_first(desc="loading and tokenizing dataset"):
        if data_args.pretokenized_manifest:
            logger.info(
                f"Loading pre-tokenized mix from manifest: "
                f"{data_args.pretokenized_manifest}"
            )
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

    if data_args.max_eval_samples and len(test_ds) > data_args.max_eval_samples:
        test_ds = test_ds.shuffle(seed=training_args.seed).select(
            range(data_args.max_eval_samples)
        )
        logger.info(
            f"Capped training-time eval to {len(test_ds):,} deterministic samples "
            "(full held-out size was larger)."
        )

    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")
    logger.info("=" * 60)
    return train_ds, test_ds


def build_pretraining_model(
    tokenizer,
    model_args,
    data_args,
    training_args,
    loss_config,
    *,
    is_causal_ar,
    is_backbone,
):
    """Construct the selected model family and preserve warm-start behavior."""
    if is_backbone:
        config = BackboneConceptConfig(
            backbone_model=model_args.backbone_model,
            concept_num=model_args.concept_num,
            concept_block=model_args.concept_block,
            concept_io_mode=model_args.concept_io_mode,
            read_concept_norm=model_args.read_concept_norm,
            read_gate_init=model_args.read_gate_init,
            write_gate_init=model_args.write_gate_init,
            concept_read_mode=model_args.concept_read_mode,
            tie_concept_writer=model_args.tie_concept_writer,
            concept_write_mode=model_args.concept_write_mode,
            write_update_gate_init=model_args.write_update_gate_init,
            memory_carry_dropout=model_args.memory_carry_dropout,
            memory_pressure_tokens=model_args.memory_pressure_tokens,
            memory_pressure_weight=model_args.memory_pressure_weight,
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
            f"io={model_args.concept_io_mode}, read={model_args.concept_read_mode}, "
            f"write={model_args.concept_write_mode}, tied={model_args.tie_concept_writer}, "
            f"lora_r={model_args.lora_r})"
        )
        backbone_load_kwargs = {}
        if training_args.bf16:
            backbone_load_kwargs["dtype"] = torch.bfloat16
        model = BackboneConceptLM.from_pretrained_backbone(
            config,
            **backbone_load_kwargs,
        )
        model_class = BackboneConceptLM
    else:
        config = build_perceiver_denoise_config(tokenizer, model_args, data_args)
        model_class = (
            ConceptEncoderForConditionalLM
            if is_causal_ar
            else ConceptEncoderForDenoisingPerceiver
        )
        logger.info(f"Initializing {model_class.__name__}")
        model = model_class(config, loss_config=loss_config)

    if model_args.model_name_or_path and not is_backbone:
        logger.info(f"Warm-starting encoder from {model_args.model_name_or_path}")
        pretrained = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)
        logger.info(
            "Loaded pretrained encoder weights. Decoder and objective head use "
            "the current config."
        )

    if is_backbone:
        model_type = "backbone_concept"
    elif is_causal_ar:
        model_type = "concept_ar"
        if model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
            model_type += "_prefix"
        if model_args.use_bixt:
            model_type += "_bixt"
    else:
        model_type = "perceiver_denoise"
        if model_args.use_bixt:
            model_type += "_bixt"
        if model_args.objective_variant == OBJECTIVE_RECONSTRUCTION_CONTRASTIVE:
            model_type += "_contrastive"
    return model, config, model_type


def build_training_wandb_identity(
    model_args,
    config,
    *,
    is_backbone,
    experiment_id=None,
):
    """Build the existing W&B identity for backbone and concept-encoder families."""
    if is_backbone:
        backbone_short = model_args.backbone_model.split("/")[-1].replace("-", "_")
        architecture_id = (
            f"backbone_concept_{backbone_short}_K{model_args.concept_block}"
        )
        resolved_experiment = experiment_id or "E10"
        arm_tag = "concept-arm" if model_args.concept_num > 0 else "control-arm"
        return WandbRunIdentity(
            experiment_id=resolved_experiment,
            model_family="backbone_concept",
            objective_family="causal_lm",
            architecture_id=architecture_id,
            group=f"{resolved_experiment}_{architecture_id}",
            job_type="train_backbone_causal_lm",
            tags=[
                "train",
                "concept-encoder",
                "decoder:autoregressive",
                "task:generation",
                "backbone_concept",
                model_args.backbone_model,
                f"io-{model_args.concept_io_mode}",
                "causal_lm",
                arm_tag,
                f"lora_r{model_args.lora_r}",
                resolved_experiment,
            ],
        )

    return build_perceiver_wandb_identity(
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
        experiment_id=experiment_id,
    )


def build_distributed_run_identifier(
    wandb_identity,
    *,
    is_backbone,
    concept_num,
    timestamp=None,
):
    """Build one run id and broadcast rank 0's value to every DDP process."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    arm_suffix = ""
    if is_backbone:
        arm_suffix = "_concept" if concept_num > 0 else "_control"
    run_identifier = f"{wandb_identity.architecture_id}{arm_suffix}_{timestamp}"
    return broadcast_object(run_identifier)


def build_pretraining_collators(
    tokenizer,
    model,
    model_args,
    data_args,
    training_args,
):
    """Return stochastic train and deterministic evaluation collators."""
    if model_args.objective_variant == OBJECTIVE_CAUSAL_LM:
        causal_collator_kwargs = {
            "max_length": data_args.max_seq_length,
            "model_vocab_size": model.backbone.config.vocab_size,
            "preserve_precomputed_labels": data_args.preserve_precomputed_labels,
        }
        data_collator = DataCollatorForCausalLM(tokenizer, **causal_collator_kwargs)
        eval_data_collator = DataCollatorForCausalLM(
            tokenizer,
            **causal_collator_kwargs,
        )
    elif model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX:
        prefix_collator_kwargs = {
            "max_length": data_args.max_seq_length,
            "prefix_ratio_min": data_args.prefix_ratio_min,
            "prefix_ratio_max": data_args.prefix_ratio_max,
            "min_prefix_content": data_args.min_prefix_content,
            "min_suffix_content": data_args.min_suffix_content,
            "split_strategy": data_args.split_strategy,
        }
        data_collator = DataCollatorForPrefixGeneration(
            tokenizer,
            **prefix_collator_kwargs,
        )
        eval_data_collator = DataCollatorForPrefixGeneration(
            tokenizer,
            seed=training_args.seed,
            **prefix_collator_kwargs,
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
    return data_collator, eval_data_collator

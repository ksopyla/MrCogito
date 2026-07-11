from types import SimpleNamespace

import pytest

from training import concept_pretraining_factories as factories
from training.concept_pretraining_args import (
    DECODER_CAUSAL_AR,
    DECODER_PERCEIVER_POSONLY,
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    DataTrainingArguments,
    LossArguments,
    ModelArguments,
    validate_training_configuration,
)
from training.train_concept_pretraining import build_argument_parser


def test_parser_maps_e05_profile_into_all_argument_groups(tmp_path):
    parser = build_argument_parser()
    model_args, loss_args, data_args, optim_args, training_args = (
        parser.parse_args_into_dataclasses(
            args=[
                "--decoder_type",
                "causal_ar",
                "--objective_variant",
                "prefix_suffix",
                "--decoder_context_window",
                "128",
                "--decoder_attn_impl",
                "chunked_window",
                "--chunked_ce_block_size",
                "512",
                "--dataset_mix_recipe",
                "smollm3_inspired_2k_e05",
                "--pretokenized_manifest",
                "/cache/hf_home/datasets_tok/e05_manifest.json",
                "--max_seq_length",
                "2048",
                "--optimizer",
                "muon",
                "--muon_adamw_lr",
                "2e-4",
                "--muon_momentum",
                "0.95",
                "--weight_decay",
                "0.1",
                "--gradient_accumulation_steps",
                "3",
                "--output_dir",
                str(tmp_path),
            ]
        )
    )

    assert model_args.decoder_type == DECODER_CAUSAL_AR
    assert model_args.objective_variant == OBJECTIVE_PREFIX_SUFFIX
    assert model_args.decoder_context_window == 128
    assert model_args.decoder_attn_impl == "chunked_window"
    assert model_args.chunked_ce_block_size == 512
    assert loss_args.concept_losses == "none"
    assert data_args.dataset_mix_recipe == "smollm3_inspired_2k_e05"
    assert data_args.pretokenized_manifest.endswith("e05_manifest.json")
    assert data_args.max_seq_length == 2048
    assert optim_args.optimizer == "muon"
    assert optim_args.muon_adamw_lr == pytest.approx(2e-4)
    assert optim_args.muon_momentum == pytest.approx(0.95)
    assert training_args.weight_decay == pytest.approx(0.1)
    assert training_args.gradient_accumulation_steps == 3


def test_parser_maps_e10_profile_into_model_data_and_training_args(tmp_path):
    parser = build_argument_parser()
    model_args, _, data_args, optim_args, training_args = (
        parser.parse_args_into_dataclasses(
            args=[
                "--decoder_type",
                "causal_ar",
                "--objective_variant",
                "causal_lm",
                "--backbone_model",
                "google/gemma-3-1b-pt",
                "--concept_num",
                "0",
                "--concept_block",
                "512",
                "--concept_io_mode",
                "global_kv",
                "--lora_r",
                "16",
                "--pretokenized_manifest",
                "/cache/hf_home/datasets_tok_gemma/e10_manifest.json",
                "--max_eval_samples",
                "2048",
                "--gradient_checkpointing",
                "True",
                "--optimizer",
                "adam",
                "--output_dir",
                str(tmp_path),
            ]
        )
    )

    assert model_args.backbone_model == "google/gemma-3-1b-pt"
    assert model_args.objective_variant == OBJECTIVE_CAUSAL_LM
    assert model_args.concept_num == 0
    assert model_args.concept_block == 512
    assert model_args.concept_io_mode == "global_kv"
    assert model_args.lora_r == 16
    assert data_args.pretokenized_manifest.endswith("e10_manifest.json")
    assert data_args.max_eval_samples == 2048
    assert optim_args.optimizer == "adam"
    assert training_args.gradient_checkpointing is True


@pytest.mark.parametrize(
    ("model_args", "loss_args", "message"),
    [
        (
            ModelArguments(decoder_type="unknown"),
            LossArguments(),
            "Unknown decoder_type",
        ),
        (
            ModelArguments(objective_variant=OBJECTIVE_CAUSAL_LM),
            LossArguments(),
            "requires --backbone_model",
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
            ),
            LossArguments(),
            "contrastive path is perceiver-only",
        ),
        (
            ModelArguments(anchor_loss=True),
            LossArguments(),
            "requires decoder_type='causal_ar'",
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_PREFIX_SUFFIX,
                anchor_loss=True,
            ),
            LossArguments(),
            "scoped to objective_variant='reconstruction'",
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_CAUSAL_LM,
                backbone_model="google/gemma-3-1b-pt",
                anchor_loss=True,
            ),
            LossArguments(),
            "anchor_loss is not supported",
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_CAUSAL_LM,
                backbone_model="google/gemma-3-1b-pt",
            ),
            LossArguments(concept_losses="uniformity"),
            "concept_losses are not wired",
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_CAUSAL_LM,
                backbone_model="google/gemma-3-1b-pt",
                model_name_or_path="/tmp/checkpoint",
            ),
            LossArguments(),
            "initializes from backbone_model directly",
        ),
    ],
)
def test_validation_covers_all_family_objective_rejections(
    model_args,
    loss_args,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_training_configuration(model_args, loss_args)


@pytest.mark.parametrize(
    ("model_args", "expected"),
    [
        (ModelArguments(), (False, False)),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_PREFIX_SUFFIX,
            ),
            (True, False),
        ),
        (
            ModelArguments(
                decoder_type=DECODER_CAUSAL_AR,
                objective_variant=OBJECTIVE_RECONSTRUCTION,
                anchor_loss=True,
            ),
            (True, False),
        ),
    ],
)
def test_validation_accepts_maintained_non_backbone_profiles(model_args, expected):
    assert validate_training_configuration(model_args, LossArguments()) == expected


class _Tokenizer:
    pad_token_id = 0
    mask_token_id = None
    cls_token_id = None
    sep_token_id = None
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = None

    def __len__(self):
        return 40


def test_config_factory_maps_e05_window_and_chunk_parameters():
    config = factories.build_perceiver_denoise_config(
        _Tokenizer(),
        ModelArguments(
            hidden_size=32,
            token_embedding_dim=16,
            num_hidden_layers=2,
            concept_num=8,
            intermediate_size=64,
            decoder_num_layers=2,
            decoder_type=DECODER_CAUSAL_AR,
            objective_variant=OBJECTIVE_PREFIX_SUFFIX,
            decoder_context_window=128,
            decoder_attn_impl="chunked_window",
            decoder_attn_chunk_size=256,
            chunked_ce_block_size=512,
        ),
        DataTrainingArguments(max_seq_length=2048, tokenizer_name="dummy"),
    )

    assert config.decoder_context_window == 128
    assert config.decoder_attn_impl == "chunked_window"
    assert config.decoder_attn_chunk_size == 256
    assert config.chunked_ce_block_size == 512
    assert config.max_sequence_length == 2048
    assert config.checkpoint_family == "concept_ar"
    assert config.pretraining_objective == "ar_prefix_suffix_generation"


def test_model_factory_routes_perceiver_and_causal_ar(monkeypatch):
    config = SimpleNamespace(checkpoint_family="test")
    monkeypatch.setattr(
        factories,
        "build_perceiver_denoise_config",
        lambda tokenizer, model_args, data_args: config,
    )

    class FakePerceiver:
        def __init__(self, built_config, loss_config):
            self.config = built_config
            self.loss_config = loss_config

    class FakeCausal:
        def __init__(self, built_config, loss_config):
            self.config = built_config
            self.loss_config = loss_config

    monkeypatch.setattr(
        factories,
        "ConceptEncoderForDenoisingPerceiver",
        FakePerceiver,
    )
    monkeypatch.setattr(factories, "ConceptEncoderForConditionalLM", FakeCausal)

    common = {
        "tokenizer": object(),
        "data_args": DataTrainingArguments(),
        "training_args": SimpleNamespace(bf16=False),
        "loss_config": object(),
        "is_backbone": False,
    }
    perceiver, _, perceiver_type = factories.build_pretraining_model(
        model_args=ModelArguments(use_bixt=True),
        is_causal_ar=False,
        **common,
    )
    causal, _, causal_type = factories.build_pretraining_model(
        model_args=ModelArguments(
            decoder_type=DECODER_CAUSAL_AR,
            objective_variant=OBJECTIVE_PREFIX_SUFFIX,
            use_bixt=True,
        ),
        is_causal_ar=True,
        **common,
    )

    assert isinstance(perceiver, FakePerceiver)
    assert perceiver_type == "perceiver_denoise_bixt"
    assert isinstance(causal, FakeCausal)
    assert causal_type == "concept_ar_prefix_bixt"


def test_model_factory_routes_backbone_and_bf16(monkeypatch):
    captured = {}

    class FakeBackboneModel:
        @classmethod
        def from_pretrained_backbone(cls, config, **kwargs):
            captured.update(config=config, kwargs=kwargs)
            return cls()

    monkeypatch.setattr(factories, "BackboneConceptLM", FakeBackboneModel)
    model, config, model_type = factories.build_pretraining_model(
        tokenizer=object(),
        model_args=ModelArguments(
            decoder_type=DECODER_CAUSAL_AR,
            objective_variant=OBJECTIVE_CAUSAL_LM,
            backbone_model="google/gemma-3-1b-pt",
        ),
        data_args=DataTrainingArguments(tokenizer_name="google/gemma-3-1b-pt"),
        training_args=SimpleNamespace(bf16=True),
        loss_config=object(),
        is_causal_ar=True,
        is_backbone=True,
    )

    assert isinstance(model, FakeBackboneModel)
    assert config.backbone_model == "google/gemma-3-1b-pt"
    assert captured["kwargs"]["dtype"].is_floating_point
    assert str(captured["kwargs"]["dtype"]) == "torch.bfloat16"
    assert model_type == "backbone_concept"

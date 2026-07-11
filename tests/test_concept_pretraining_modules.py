from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from data.data_collators import (
    DataCollatorForCausalLM,
    DataCollatorForPrefixGeneration,
    DataCollatorForTSDAE,
)
from training import concept_pretraining_args as args_module
from training import concept_pretraining_factories as factories
from training import concept_pretraining_objectives as objectives
from training import concept_pretraining_trainer as trainer_module
from training import train_concept_pretraining as canonical_entrypoint
from training import train_perceiver_denoise as legacy_entrypoint


def test_legacy_entrypoint_reexports_extracted_public_symbols():
    assert legacy_entrypoint.main is canonical_entrypoint.main
    assert legacy_entrypoint.ModelArguments is args_module.ModelArguments
    assert legacy_entrypoint.LossArguments is args_module.LossArguments
    assert legacy_entrypoint.DataTrainingArguments is args_module.DataTrainingArguments
    assert legacy_entrypoint.OptimizerArguments is args_module.OptimizerArguments
    assert (
        legacy_entrypoint.resolve_append_eos_token_id
        is objectives.resolve_append_eos_token_id
    )
    assert args_module.resolve_append_eos_token_id is objectives.resolve_append_eos_token_id
    assert (
        legacy_entrypoint.build_perceiver_denoise_config
        is factories.build_perceiver_denoise_config
    )
    assert (
        legacy_entrypoint.align_special_tokens_for_training
        is factories.align_special_tokens_for_training
    )
    assert (
        legacy_entrypoint.PerceiverDenoiseTrainer
        is trainer_module.PerceiverDenoiseTrainer
    )


def test_validate_training_configuration_returns_family_flags():
    is_causal_ar, is_backbone = args_module.validate_training_configuration(
        args_module.ModelArguments(
            decoder_type=args_module.DECODER_CAUSAL_AR,
            objective_variant=args_module.OBJECTIVE_CAUSAL_LM,
            backbone_model="google/gemma-3-1b-pt",
        ),
        args_module.LossArguments(),
    )

    assert is_causal_ar is True
    assert is_backbone is True


@pytest.mark.parametrize(
    ("model_args", "message"),
    [
        (
            args_module.ModelArguments(objective_variant="unknown"),
            "Unknown objective_variant",
        ),
        (
            args_module.ModelArguments(
                objective_variant=args_module.OBJECTIVE_PREFIX_SUFFIX,
                decoder_type=args_module.DECODER_PERCEIVER_POSONLY,
            ),
            "requires decoder_type='causal_ar'",
        ),
        (
            args_module.ModelArguments(
                objective_variant=args_module.OBJECTIVE_RECONSTRUCTION,
                backbone_model="google/gemma-3-1b-pt",
            ),
            "requires objective_variant='causal_lm'",
        ),
    ],
)
def test_validate_training_configuration_rejects_incompatible_modes(
    model_args,
    message,
):
    with pytest.raises(ValueError, match=message):
        args_module.validate_training_configuration(
            model_args,
            args_module.LossArguments(),
        )


def _training_args(seed=42):
    return SimpleNamespace(
        seed=seed,
        main_process_first=lambda **kwargs: nullcontext(),
    )


def _dataset_args(**overrides):
    values = {
        "dataset_name": "direct/dataset",
        "dataset_name_subset": "subset",
        "dataset_mix": None,
        "dataset_mix_recipe": None,
        "dataset_mix_weight_override": None,
        "pretokenized_manifest": None,
        "test_size_percent": 0.1,
        "max_seq_length": 16,
        "max_eval_samples": None,
        "dataset_cache_dir": "/cache/hf_home/datasets",
        "train_num_proc": 2,
        "test_num_proc": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_data_factory_prioritizes_pretokenized_manifest(monkeypatch):
    calls = []
    train, evaluation = [1, 2], [3]
    monkeypatch.setattr(
        factories,
        "load_pretokenized_mix",
        lambda manifest: calls.append(("pretokenized", manifest))
        or (train, evaluation),
    )
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_dataset_mix",
        lambda *args, **kwargs: pytest.fail("mix route must not run"),
    )
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_text_dataset",
        lambda *args, **kwargs: pytest.fail("direct Hub route must not run"),
    )
    data_args = _dataset_args(
        pretokenized_manifest="/cache/hf_home/datasets_tok/manifest.json",
        dataset_mix_recipe="recipe",
        dataset_mix="registry",
    )

    actual_train, actual_eval = factories.load_pretraining_datasets(
        tokenizer=object(),
        data_args=data_args,
        training_args=_training_args(),
        append_eos_token_id=2,
    )

    assert calls == [
        ("pretokenized", "/cache/hf_home/datasets_tok/manifest.json")
    ]
    assert actual_train is train
    assert actual_eval is evaluation


def test_data_factory_prioritizes_recipe_over_registry_mix(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_dataset_mix",
        lambda tokenizer, selected_mix, **kwargs: captured.update(
            selected_mix=selected_mix,
            **kwargs,
        )
        or ([1], [2]),
    )
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_text_dataset",
        lambda *args, **kwargs: pytest.fail("direct Hub route must not run"),
    )
    data_args = _dataset_args(
        dataset_mix_recipe="recipe-v1",
        dataset_mix="legacy-registry",
        dataset_mix_weight_override='{"source": 0.75}',
    )

    factories.load_pretraining_datasets(
        tokenizer=object(),
        data_args=data_args,
        training_args=_training_args(seed=7),
        append_eos_token_id=2,
    )

    assert captured["selected_mix"] == "recipe-v1"
    assert captured["mix_weight_override"] == '{"source": 0.75}'
    assert captured["dataset_cache_dir"] == "/cache/hf_home/datasets"
    assert captured["split_seed"] == 7
    assert captured["interleave_seed"] == 7


def test_data_factory_uses_registry_mix_when_no_recipe(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_dataset_mix",
        lambda tokenizer, selected_mix, **kwargs: captured.update(
            selected_mix=selected_mix,
            **kwargs,
        )
        or ([1], [2]),
    )
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_text_dataset",
        lambda *args, **kwargs: pytest.fail("direct Hub route must not run"),
    )

    factories.load_pretraining_datasets(
        tokenizer=object(),
        data_args=_dataset_args(dataset_mix="long_2k_base_v1"),
        training_args=_training_args(seed=11),
        append_eos_token_id=2,
    )

    assert captured["selected_mix"] == "long_2k_base_v1"
    assert captured["dataset_cache_dir"] == "/cache/hf_home/datasets"
    assert captured["split_seed"] == 11
    assert captured["interleave_seed"] == 11


def test_data_factory_caps_eval_deterministically(monkeypatch):
    class EvaluationDataset:
        def __init__(self):
            self.shuffle_seed = None
            self.selected = None

        def __len__(self):
            return 10 if self.selected is None else len(self.selected)

        def shuffle(self, seed):
            self.shuffle_seed = seed
            return self

        def select(self, indices):
            self.selected = list(indices)
            return self

    evaluation = EvaluationDataset()
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_text_dataset",
        lambda *args, **kwargs: ([1, 2], evaluation),
    )

    _, actual_eval = factories.load_pretraining_datasets(
        tokenizer=object(),
        data_args=_dataset_args(max_eval_samples=3),
        training_args=_training_args(seed=19),
        append_eos_token_id=2,
    )

    assert actual_eval is evaluation
    assert evaluation.shuffle_seed == 19
    assert evaluation.selected == [0, 1, 2]
    assert len(actual_eval) == 3


def test_data_factory_keeps_direct_hub_route_and_cache(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        factories,
        "load_and_preprocess_text_dataset",
        lambda tokenizer, dataset_name, subset, text_column, **kwargs: captured.update(
            dataset_name=dataset_name,
            subset=subset,
            text_column=text_column,
            **kwargs,
        )
        or ([1], [2]),
    )

    factories.load_pretraining_datasets(
        tokenizer=object(),
        data_args=_dataset_args(),
        training_args=_training_args(seed=9),
        append_eos_token_id=2,
    )

    assert captured["dataset_name"] == "direct/dataset"
    assert captured["subset"] == "subset"
    assert captured["text_column"] == "text"
    assert captured["dataset_cache_dir"] == "/cache/hf_home/datasets"
    assert captured["split_seed"] == 9


def test_backbone_wandb_identity_keeps_group_and_arm_tags():
    model_args = args_module.ModelArguments(
        objective_variant=args_module.OBJECTIVE_CAUSAL_LM,
        decoder_type=args_module.DECODER_CAUSAL_AR,
        backbone_model="google/gemma-3-1b-pt",
        concept_num=128,
        concept_block=512,
        concept_io_mode="global_kv",
        lora_r=16,
    )

    identity = factories.build_training_wandb_identity(
        model_args,
        config=SimpleNamespace(),
        is_backbone=True,
        experiment_id="E10",
    )

    assert identity.architecture_id == "backbone_concept_gemma_3_1b_pt_K512"
    assert identity.group == "E10_backbone_concept_gemma_3_1b_pt_K512"
    assert identity.job_type == "train_backbone_causal_lm"
    assert {"E10", "backbone_concept", "concept-arm", "causal_lm"}.issubset(
        identity.tags
    )


@pytest.mark.parametrize(
    ("concept_num", "arm_suffix"),
    [(128, "_concept"), (0, "_control")],
)
def test_distributed_backbone_run_identifier_keeps_arm_suffix(
    monkeypatch,
    concept_num,
    arm_suffix,
):
    broadcast_values = []
    monkeypatch.setattr(
        factories,
        "broadcast_object",
        lambda value: broadcast_values.append(value) or value,
    )
    identity = SimpleNamespace(architecture_id="backbone_concept_gemma_3_1b_pt_K512")

    run_identifier = factories.build_distributed_run_identifier(
        identity,
        is_backbone=True,
        concept_num=concept_num,
        timestamp="20260711_120000",
    )

    assert run_identifier == (
        f"backbone_concept_gemma_3_1b_pt_K512{arm_suffix}_20260711_120000"
    )
    assert broadcast_values == [run_identifier]


class _Tokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    eos_token_id = 3
    bos_token_id = 4

    def __len__(self):
        return 128


@pytest.mark.parametrize(
    ("objective", "collator_class"),
    [
        (args_module.OBJECTIVE_CAUSAL_LM, DataCollatorForCausalLM),
        (args_module.OBJECTIVE_PREFIX_SUFFIX, DataCollatorForPrefixGeneration),
        (args_module.OBJECTIVE_RECONSTRUCTION, DataCollatorForTSDAE),
    ],
)
def test_collator_factory_keeps_objective_routing_and_seeded_eval(
    objective,
    collator_class,
):
    model = SimpleNamespace(
        backbone=SimpleNamespace(config=SimpleNamespace(vocab_size=128))
    )
    model_args = args_module.ModelArguments(objective_variant=objective)
    data_args = args_module.DataTrainingArguments(
        max_seq_length=32,
        deletion_rate=0.4,
        prefix_ratio_min=0.25,
        prefix_ratio_max=0.6,
        split_strategy="token_random",
    )

    train_collator, eval_collator = factories.build_pretraining_collators(
        _Tokenizer(),
        model,
        model_args,
        data_args,
        SimpleNamespace(seed=17),
    )

    assert isinstance(train_collator, collator_class)
    assert isinstance(eval_collator, collator_class)
    if objective != args_module.OBJECTIVE_CAUSAL_LM:
        assert train_collator.seed is None
        assert eval_collator.seed == 17

import pytest

from evaluation.wandb_identity import (
    build_namespaced_eval_tags,
    infer_checkpoint_id,
    parse_checkpoint_step_from_path,
    parse_training_run_id_from_model_path,
    resolve_eval_lineage,
)


def test_parse_checkpoint_step_from_path():
    assert parse_checkpoint_step_from_path("Cache/Training/foo/checkpoint-66000") == 66000
    assert parse_checkpoint_step_from_path("/tmp/run/checkpoint-42/model.safetensors") == 42
    assert parse_checkpoint_step_from_path("Cache/Training/foo/final_model") is None


def test_parse_training_run_id_from_model_path():
    assert (
        parse_training_run_id_from_model_path(
            "Cache/Training/perceiver_denoise_H768L6C128D4_20260618_200645/checkpoint-66000"
        )
        == "perceiver_denoise_H768L6C128D4_20260618_200645"
    )
    assert (
        parse_training_run_id_from_model_path("Cache/Training/concept_ar_H768L6C128D4_20260615_211458")
        == "concept_ar_H768L6C128D4_20260615_211458"
    )


def test_resolve_eval_lineage_linked_without_api():
    lineage = resolve_eval_lineage(
        model_path="Cache/Training/concept_ar_H768L6C128D4_20260615_211458/checkpoint-10000",
        source_training_run_id=None,
        source_training_group="E03_concept_ar_H768L6C128D4",
        source_training_experiment_id="E03",
        source_checkpoint_step=None,
        source_checkpoint_epoch=0.3,
        allow_unlinked_eval=False,
        wandb_entity="ksopyla",
        wandb_project="MrCogito",
        resolve_with_wandb=False,
    )
    assert lineage.lineage_status == "linked"
    assert lineage.source_training_run_id == "concept_ar_H768L6C128D4_20260615_211458"
    assert lineage.source_checkpoint_step == 10000
    assert lineage.source_checkpoint_epoch == 0.3


def test_resolve_eval_lineage_strict_rejects_unlinked():
    with pytest.raises(ValueError):
        resolve_eval_lineage(
            model_path="HuggingFaceTB/SmolLM2-135M",
            source_training_run_id=None,
            source_training_group=None,
            source_training_experiment_id=None,
            source_checkpoint_step=None,
            source_checkpoint_epoch=None,
            allow_unlinked_eval=False,
            wandb_entity="ksopyla",
            wandb_project="MrCogito",
            resolve_with_wandb=False,
        )


def test_resolve_eval_lineage_strict_rejects_missing_checkpoint_step():
    with pytest.raises(ValueError):
        resolve_eval_lineage(
            model_path="Cache/Training/concept_ar_H768L6C128D4_20260615_211458/final_model",
            source_training_run_id="concept_ar_H768L6C128D4_20260615_211458",
            source_training_group="E03_concept_ar_H768L6C128D4",
            source_training_experiment_id="E03",
            source_checkpoint_step=None,
            source_checkpoint_epoch=None,
            allow_unlinked_eval=False,
            wandb_entity="ksopyla",
            wandb_project="MrCogito",
            resolve_with_wandb=False,
        )


def test_resolve_eval_lineage_allow_unlinked_mode():
    lineage = resolve_eval_lineage(
        model_path="HuggingFaceTB/SmolLM2-135M",
        source_training_run_id=None,
        source_training_group=None,
        source_training_experiment_id=None,
        source_checkpoint_step=None,
        source_checkpoint_epoch=None,
        allow_unlinked_eval=True,
        wandb_entity="ksopyla",
        wandb_project="MrCogito",
        resolve_with_wandb=False,
    )
    assert lineage.lineage_status == "unlinked"


def test_build_namespaced_eval_tags_contains_lineage_facets():
    lineage = resolve_eval_lineage(
        model_path="Cache/Training/concept_ar_H768L6C128D4_20260615_211458/checkpoint-10000",
        source_training_run_id=None,
        source_training_group="E03_concept_ar_H768L6C128D4",
        source_training_experiment_id="E03",
        source_checkpoint_step=None,
        source_checkpoint_epoch=1.0,
        allow_unlinked_eval=False,
        wandb_entity="ksopyla",
        wandb_project="MrCogito",
        resolve_with_wandb=False,
    )
    tags = build_namespaced_eval_tags(
        benchmark="stsb_zero_shot",
        model_family="concept_ar",
        objective_family="ar_reconstruction",
        params_m=73,
        tokenizer_name="HuggingFaceTB/SmolLM2-135M",
        lineage=lineage,
        extra_tags=["legacy-tag"],
    )
    assert "benchmark:stsb_zero_shot" in tags
    assert "family:concept_ar" in tags
    assert "exp:e03" in tags
    assert "ckpt_step:10000" in tags
    assert "lineage:linked" in tags


def test_infer_checkpoint_id_final_model():
    assert (
        infer_checkpoint_id(
            source_checkpoint_path="Cache/Training/concept_ar_prefix_H768L6C128D4_20260612_094555/concept_ar_prefix_H768L6C128D4_20260612_094555",
            source_training_run_id="concept_ar_prefix_H768L6C128D4_20260612_094555",
            source_checkpoint_step=None,
        )
        == "final_model"
    )

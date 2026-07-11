import ast
import logging
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import utils_training


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PATHS_SCRIPT = REPO_ROOT / "scripts" / "remote_paths.sh"
CANONICAL_ENTRYPOINT = REPO_ROOT / "training" / "train_concept_pretraining.py"
LEGACY_ENTRYPOINT = REPO_ROOT / "training" / "train_perceiver_denoise.py"
CANONICAL_LAUNCHER = REPO_ROOT / "scripts" / "train_concept_pretraining_multigpu.sh"
LEGACY_LAUNCHER = REPO_ROOT / "scripts" / "train_perceiver_denoise_multigpu.sh"
E10_PIPELINE = REPO_ROOT / "scripts" / "launch_e10_pipeline.sh"
WEIGHTED_MLM_TRAINER = REPO_ROOT / "parked" / "training" / "train_weighted_mlm.py"
RECURSIVE_TOMBSTONE = REPO_ROOT / "parked" / "README.md"
TRAINING_ENTRYPOINTS = [
    CANONICAL_ENTRYPOINT,
    WEIGHTED_MLM_TRAINER,
    REPO_ROOT / "parked" / "training" / "train_diffusion.py",
    REPO_ROOT / "parked" / "training" / "train_prefix_diffusion.py",
]


def test_canonical_and_legacy_entrypoints_expose_compatible_cli_help():
    required_flags = {
        "--objective_variant",
        "--decoder_type",
        "--pretokenized_manifest",
        "--dataset_mix_recipe",
        "--backbone_model",
    }

    for entrypoint in (CANONICAL_ENTRYPOINT, LEGACY_ENTRYPOINT):
        result = subprocess.run(
            [sys.executable, str(entrypoint), "--help"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert all(flag in result.stdout for flag in required_flags)


def test_generic_launcher_targets_canonical_entrypoint():
    launcher = CANONICAL_LAUNCHER.read_text(encoding="utf-8")

    assert "training/train_concept_pretraining.py" in launcher
    assert "training/train_perceiver_denoise.py" not in launcher


def test_legacy_generic_launcher_is_a_thin_compatibility_wrapper():
    launcher = LEGACY_LAUNCHER.read_text(encoding="utf-8")

    assert "train_concept_pretraining_multigpu.sh" in launcher
    assert "accelerate launch" not in launcher
    assert "training/train_" not in launcher


def test_e10_pipeline_delegates_to_protocol_wrapper_not_generic_runner():
    pipeline = E10_PIPELINE.read_text(encoding="utf-8")

    assert "exec bash scripts/launch_e10.sh" in pipeline
    assert "train_concept_pretraining_multigpu.sh" not in pipeline
    assert "train_perceiver_denoise_multigpu.sh" not in pipeline


def test_retired_training_paths_keep_reproducibility_and_tombstone_contracts():
    assert not (REPO_ROOT / "training" / "train_mlm.py").exists()
    assert WEIGHTED_MLM_TRAINER.exists()
    assert (REPO_ROOT / "nn" / "concept_encoder_weighted.py").exists()
    assert not (
        REPO_ROOT / "parked" / "training" / "train_recursive_mlm.py"
    ).exists()
    assert "Recursive MLM / TRM-style weight-tied encoder" in (
        RECURSIVE_TOMBSTONE.read_text(encoding="utf-8")
    )


def test_parked_weighted_mlm_trainer_keeps_cli_for_reproducibility():
    result = subprocess.run(
        [sys.executable, str(WEIGHTED_MLM_TRAINER), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model_type" in result.stdout
    assert "--mlm_probability" in result.stdout
    assert "--concept_losses" in result.stdout


def test_all_training_entrypoints_use_shared_logging_contract():
    required_calls = {
        "setup_file_logging",
        "log_data_config",
        "log_model_info",
        "log_training_config",
        "init_wandb",
    }

    for entrypoint in TRAINING_ENTRYPOINTS:
        tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert required_calls.issubset(calls), (
            f"{entrypoint.relative_to(REPO_ROOT)} must retain the shared console/file/W&B "
            f"logging contract; missing {sorted(required_calls - calls)}"
        )


def test_setup_file_logging_writes_to_console_and_file(monkeypatch, tmp_path, capsys):
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    monkeypatch.setattr(utils_training, "is_main_process", lambda: True)
    try:
        utils_training.setup_file_logging(str(tmp_path))
        root_logger.info("training-contract-message")
        for handler in root_logger.handlers:
            handler.flush()

        console_handlers = [
            handler
            for handler in root_logger.handlers
            if type(handler) is logging.StreamHandler
        ]
        file_handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert len(console_handlers) == 1
        assert len(file_handlers) == 1
        assert "training-contract-message" in capsys.readouterr().err

        log_files = list(tmp_path.glob("training_*.log"))
        assert len(log_files) == 1
        assert "training-contract-message" in log_files[0].read_text(encoding="utf-8")
    finally:
        for handler in root_logger.handlers:
            if handler not in original_handlers:
                handler.close()
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def test_init_wandb_preserves_project_identity_and_effective_dataset(monkeypatch):
    captured = {}
    run_identifier = "concept_ar_prefix_H32L1C4D1_20260711_120000"
    group = "E02_concept_ar_prefix_H32L1C4D1"
    manifest = "/cache/hf_home/datasets_tok/example_manifest.json"

    def fake_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(utils_training, "is_main_process", lambda: True)
    monkeypatch.setattr(utils_training, "get_git_info", lambda: {
        "commit": "abc1234",
        "commit_long": "abc1234" * 5 + "abc12",
        "tag": None,
        "dirty": False,
    })
    monkeypatch.setattr(utils_training, "get_hostname", lambda: "test-host")
    monkeypatch.setattr(utils_training.wandb, "init", fake_init)
    monkeypatch.setattr(
        utils_training.wandb,
        "run",
        SimpleNamespace(id=run_identifier, name=run_identifier),
    )

    training_args = SimpleNamespace(
        output_dir=f"Cache/Training/{run_identifier}",
        logging_dir=f"Cache/logs/{run_identifier}",
        run_name=run_identifier,
        optim="adamw_torch_fused",
        learning_rate=3e-4,
    )
    config = SimpleNamespace(
        hidden_size=32,
        token_embedding_dim=32,
        num_hidden_layers=1,
        concept_num=4,
        intermediate_size=64,
        num_attention_heads=4,
        vocab_size=128,
        max_sequence_length=16,
    )
    data_args = SimpleNamespace(
        dataset_name="JeanKaddour/minipile",
        dataset_name_subset=None,
        dataset_mix=None,
        dataset_mix_recipe=None,
        pretokenized_manifest=manifest,
        tokenizer_name="dummy-tokenizer",
        max_seq_length=16,
    )
    loss_config = SimpleNamespace(
        is_enabled=False,
        concept_losses=[],
        weighting_strategy="fixed",
    )

    utils_training.init_wandb(
        training_args,
        torch.nn.Linear(4, 4),
        config,
        data_args,
        loss_config,
        group,
        run_identifier,
        job_type="train_ar_generation_prefix_suffix",
        model_type="concept_ar_prefix_bixt",
        wandb_tags=["E02", "x" * 100],
        extra_config={
            "experiment_id": "E02",
            "model_family": "concept_ar_prefix",
            "objective_family": "prefix_suffix",
            "architecture_id": "concept_ar_prefix_H32L1C4D1",
            "wandb_group": group,
            "optimizer": "adam",
        },
    )

    assert captured["project"] == "MrCogito"
    assert captured["id"] == run_identifier
    assert captured["name"] == run_identifier
    assert captured["group"] == group
    assert captured["job_type"] == "train_ar_generation_prefix_suffix"
    assert captured["sync_tensorboard"] is True
    assert captured["config"]["dataset_name"] == manifest
    assert captured["config"]["dataset_name_hf_default"] == "JeanKaddour/minipile"
    assert captured["config"]["model_family"] == "concept_ar_prefix"
    assert captured["config"]["optim"] == "adam"
    assert "example_manifest.json" in captured["tags"]
    assert all(len(tag) <= 64 for tag in captured["tags"])


def test_setup_run_dirs_preserves_workspace_cache_layout():
    args = SimpleNamespace(
        output_dir="Cache/Training",
        logging_dir="Cache/logs",
        run_name=None,
        report_to=[],
        push_to_hub=True,
        remove_unused_columns=True,
        fp16=False,
        bf16=True,
    )

    utils_training.setup_run_dirs(args, "test_run")

    assert args.output_dir == os.path.join("Cache", "Training", "test_run")
    assert args.logging_dir == os.path.join("Cache", "logs", "test_run")
    assert args.run_name == "test_run"
    assert args.report_to == ["tensorboard", "wandb"]
    assert args.push_to_hub is False
    assert args.remove_unused_columns is False
    assert args.fp16 is False


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "pretokenized_manifest": "/cache/datasets_tok/manifest.json",
                "dataset_mix_recipe": "recipe",
                "dataset_mix": "registry",
                "dataset_name": "direct",
            },
            "/cache/datasets_tok/manifest.json",
        ),
        (
            {
                "pretokenized_manifest": None,
                "dataset_mix_recipe": "recipe",
                "dataset_mix": "registry",
                "dataset_name": "direct",
            },
            "recipe",
        ),
        (
            {
                "pretokenized_manifest": None,
                "dataset_mix_recipe": None,
                "dataset_mix": "registry",
                "dataset_name": "direct",
            },
            "registry",
        ),
        (
            {
                "pretokenized_manifest": None,
                "dataset_mix_recipe": None,
                "dataset_mix": None,
                "dataset_name": "direct",
            },
            "direct",
        ),
    ],
)
def test_resolve_dataset_identifier_matches_loader_priority(values, expected):
    assert utils_training.resolve_dataset_identifier(SimpleNamespace(**values)) == expected


def test_setup_distributed_applies_exported_timeout_to_first_process_group(
    monkeypatch,
):
    captured = {}
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("DDP_TIMEOUT", "7200")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: captured.update(init=kwargs),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda rank: captured.update(device=rank),
    )

    local_rank = utils_training.setup_distributed()

    assert local_rank == 2
    assert captured["init"]["backend"] == "nccl"
    assert captured["init"]["device_id"] == torch.device("cuda:2")
    assert captured["init"]["timeout"] == timedelta(minutes=120)
    assert captured["device"] == 2


def _source_remote_paths(tmp_path, extra_env=None):
    env = os.environ.copy()
    for name in (
        "HF_DATASETS_CACHE",
        "DATASETS_TOK_DIR",
        "DATASETS_RAW_DIR",
        "TRANSFORMERS_CACHE",
    ):
        env.pop(name, None)
    env["REMOTE_PATHS_SCRIPT"] = str(REMOTE_PATHS_SCRIPT)
    env["HF_HOME"] = str(tmp_path / "hf_home")
    env.update(extra_env or {})

    command = """
source "$REMOTE_PATHS_SCRIPT"
printf '%s\n' \
  "$PROJECT_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$DATASETS_TOK_DIR" \
  "$DATASETS_RAW_DIR" \
  "$OUTPUT_DIR" \
  "$LOGGING_DIR" \
  "${TRANSFORMERS_CACHE-unset}"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_remote_paths_preserve_hf_cache_roles_and_workspace_cache(tmp_path):
    (
        project_root,
        hf_home,
        hf_datasets_cache,
        datasets_tok_dir,
        datasets_raw_dir,
        output_dir,
        logging_dir,
        transformers_cache,
    ) = _source_remote_paths(tmp_path)

    assert project_root == str(tmp_path)
    assert hf_home == str(tmp_path / "hf_home")
    assert hf_datasets_cache == str(tmp_path / "hf_home" / "datasets")
    assert datasets_tok_dir == str(tmp_path / "hf_home" / "datasets_tok")
    assert datasets_raw_dir == str(tmp_path / "hf_home" / "datasets_raw")
    assert output_dir == str(tmp_path / "Cache" / "Training")
    assert logging_dir == str(tmp_path / "Cache" / "logs")
    assert transformers_cache == "unset"

    for directory in (
        hf_home,
        hf_datasets_cache,
        datasets_tok_dir,
        datasets_raw_dir,
        output_dir,
        logging_dir,
    ):
        assert Path(directory).is_dir()


def test_remote_paths_keep_explicit_hf_cache_overrides(tmp_path):
    overrides = {
        "HF_DATASETS_CACHE": str(tmp_path / "custom_datasets"),
        "DATASETS_TOK_DIR": str(tmp_path / "datasets_tok_gemma"),
        "DATASETS_RAW_DIR": str(tmp_path / "nas_raw"),
    }
    values = _source_remote_paths(tmp_path, overrides)

    assert values[2] == overrides["HF_DATASETS_CACHE"]
    assert values[3] == overrides["DATASETS_TOK_DIR"]
    assert values[4] == overrides["DATASETS_RAW_DIR"]

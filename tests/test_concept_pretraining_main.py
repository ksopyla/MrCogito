import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import train_concept_pretraining as entrypoint


class _Tokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 2
    eos_token = "</s>"

    def __init__(self, calls):
        self.calls = calls

    def save_pretrained(self, path):
        self.calls["tokenizer_save"] = path


class _Trainer:
    def __init__(self, calls, **kwargs):
        calls["trainer_kwargs"] = kwargs
        self.calls = calls

    def train(self, resume_from_checkpoint=None):
        self.calls["train_resume"] = resume_from_checkpoint

    def save_model(self, path):
        self.calls["model_save"] = path


@pytest.mark.parametrize(
    ("data_cli", "expected_identifier"),
    [
        (
            ["--dataset_name", "acme/direct-hub", "--dataset_name_subset", "clean"],
            "acme/direct-hub",
        ),
        (
            ["--pretokenized_manifest", "/cache/datasets_tok/manifest.json"],
            "/cache/datasets_tok/manifest.json",
        ),
    ],
)
def test_main_wires_cli_data_logs_wandb_trainer_and_final_save(
    monkeypatch,
    tmp_path,
    data_cli,
    expected_identifier,
):
    calls = {}
    tokenizer = _Tokenizer(calls)
    model = object()
    train_ds = object()
    eval_ds = object()
    config = SimpleNamespace(
        checkpoint_family="concept_ar",
        pretraining_objective="ar_prefix_suffix_generation",
        hidden_size=32,
        num_hidden_layers=2,
        concept_num=8,
        token_embedding_dim=16,
        hidden_act="gelu",
        norm_type="layernorm",
    )
    identity = SimpleNamespace(
        group="E02_test",
        job_type="concept_ar_prefix",
        tags=["family-concept_ar", "objective-prefix_suffix"],
        to_config=lambda: {"experiment_id": "E02"},
    )

    def record(name):
        return lambda *args, **kwargs: calls.setdefault(name, []).append((args, kwargs))

    monkeypatch.setattr(entrypoint, "setup_distributed", record("setup_distributed"))
    monkeypatch.setattr(entrypoint, "is_main_process", lambda: True)
    monkeypatch.setattr(entrypoint, "setup_file_logging", record("setup_file_logging"))
    monkeypatch.setattr(entrypoint.logging, "set_verbosity_info", lambda: None)
    monkeypatch.setattr(entrypoint, "set_seed", record("set_seed"))
    monkeypatch.setattr(entrypoint, "log_system_info", record("log_system_info"))
    monkeypatch.setattr(entrypoint, "log_data_config", record("log_data_config"))
    monkeypatch.setattr(entrypoint, "log_loss_config", record("log_loss_config"))
    monkeypatch.setattr(entrypoint, "log_model_info", record("log_model_info"))
    monkeypatch.setattr(entrypoint, "log_training_config", record("log_training_config"))
    monkeypatch.setattr(entrypoint.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        entrypoint,
        "load_pretraining_datasets",
        lambda tokenizer_arg, data_args, training_args, eos_id: (
            calls.update(
                dataset_load=(tokenizer_arg, data_args, training_args, eos_id)
            )
            or (train_ds, eval_ds)
        ),
    )
    monkeypatch.setattr(
        entrypoint,
        "build_pretraining_model",
        lambda *args, **kwargs: (model, config, "concept_ar_prefix_bixt"),
    )
    monkeypatch.setattr(
        entrypoint,
        "build_training_wandb_identity",
        lambda *args, **kwargs: identity,
    )
    monkeypatch.setattr(
        entrypoint,
        "build_distributed_run_identifier",
        lambda *args, **kwargs: "E02_test_20260711_120000",
    )
    monkeypatch.setattr(entrypoint, "setup_run_dirs", record("setup_run_dirs"))
    monkeypatch.setattr(entrypoint, "init_wandb", record("init_wandb"))
    monkeypatch.setattr(
        entrypoint,
        "build_pretraining_collators",
        lambda *args, **kwargs: (
            "train-collator",
            SimpleNamespace(seed=43),
        ),
    )
    monkeypatch.setattr(
        entrypoint,
        "PerceiverDenoiseTrainer",
        lambda **kwargs: _Trainer(calls, **kwargs),
    )
    monkeypatch.setattr(entrypoint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(entrypoint.wandb, "run", None)
    monkeypatch.setenv("WANDB_EXPERIMENT_ID", "E02")
    monkeypatch.setenv("TARGET_TOKENS", "1000000")
    monkeypatch.setenv("ESTIMATED_STEPS", "42")

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_concept_pretraining.py",
            "--hidden_size",
            "32",
            "--token_embedding_dim",
            "16",
            "--num_hidden_layers",
            "2",
            "--concept_num",
            "8",
            "--intermediate_size",
            "64",
            "--decoder_num_layers",
            "2",
            "--decoder_type",
            "causal_ar",
            "--objective_variant",
            "prefix_suffix",
            "--tokenizer_name",
            "acme/tokenizer",
            "--dataset_cache_dir",
            "/cache/hf_home/datasets",
            "--max_seq_length",
            "128",
            "--optimizer",
            "muon",
            "--muon_adamw_lr",
            "0.0002",
            "--muon_momentum",
            "0.95",
            "--seed",
            "17",
            "--resume_from_checkpoint",
            "/cache/checkpoint-10",
            "--output_dir",
            str(output_dir),
            "--report_to",
            "none",
            *data_cli,
        ],
    )

    entrypoint.main()

    assert calls["setup_distributed"]
    assert calls["setup_file_logging"]
    assert calls["set_seed"][0][0] == (17,)
    loaded_tokenizer, data_args, training_args, eos_id = calls["dataset_load"]
    assert loaded_tokenizer is tokenizer
    assert eos_id == 2
    assert data_args.dataset_cache_dir == "/cache/hf_home/datasets"
    assert data_args.max_seq_length == 128
    assert training_args.resume_from_checkpoint == "/cache/checkpoint-10"
    assert entrypoint.resolve_dataset_identifier(data_args) == expected_identifier

    trainer_kwargs = calls["trainer_kwargs"]
    assert trainer_kwargs["model"] is model
    assert trainer_kwargs["train_dataset"] is train_ds
    assert trainer_kwargs["eval_dataset"] is eval_ds
    assert trainer_kwargs["data_collator"] == "train-collator"
    assert trainer_kwargs["objective_variant"] == "prefix_suffix"
    assert trainer_kwargs["optimizer_choice"] == "muon"
    assert trainer_kwargs["concept_memory_lr"] is None
    assert trainer_kwargs["muon_adamw_lr"] == pytest.approx(2e-4)
    assert trainer_kwargs["muon_momentum"] == pytest.approx(0.95)
    assert calls["train_resume"] == "/cache/checkpoint-10"

    init_args, init_kwargs = calls["init_wandb"][0]
    assert init_args[5:7] == ("E02_test", "E02_test_20260711_120000")
    assert init_kwargs["job_type"] == "concept_ar_prefix"
    assert "optim-muon" in init_kwargs["wandb_tags"]
    assert init_kwargs["extra_config"]["pretokenized_manifest"] == (
        "/cache/datasets_tok/manifest.json"
        if "--pretokenized_manifest" in data_cli
        else None
    )
    assert init_kwargs["extra_config"]["target_tokens"] == 1_000_000
    assert init_kwargs["extra_config"]["estimated_optimizer_steps"] == 42
    assert init_kwargs["extra_config"]["concept_memory_lr"] is None

    final_path = str(output_dir / "final")
    assert calls["model_save"] == final_path
    assert calls["tokenizer_save"] == final_path
    setup_args, _ = calls["setup_run_dirs"][0]
    assert setup_args[1] == "E02_test_20260711_120000"

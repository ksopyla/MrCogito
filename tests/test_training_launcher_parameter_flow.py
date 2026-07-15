import os
import subprocess
from pathlib import Path

from training.train_concept_pretraining import build_argument_parser


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "train_concept_pretraining_multigpu.sh"
LEGACY_LAUNCHER = REPO_ROOT / "scripts" / "train_perceiver_denoise_multigpu.sh"
E05_LAUNCHER = REPO_ROOT / "scripts" / "launch_e05.sh"
E10_LAUNCHER = REPO_ROOT / "scripts" / "launch_e10.sh"
E14_LAUNCHER = REPO_ROOT / "scripts" / "launch_e14.sh"
E16A_LAUNCHER = REPO_ROOT / "scripts" / "launch_e16a.sh"
E16A_PIPELINE = REPO_ROOT / "scripts" / "launch_e16a_pipeline.sh"
E16B_LAUNCHER = REPO_ROOT / "scripts" / "launch_e16b.sh"


def _write_executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _run_launcher(tmp_path, extra_env, launcher=LAUNCHER):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture_path = tmp_path / "accelerate.args"
    ddp_capture_path = tmp_path / "ddp_timeout.txt"
    _write_executable(
        bin_dir / "nvidia-smi",
        "#!/bin/bash\nprintf '%s\\n' 'GPU 0: test'\n",
    )
    _write_executable(
        bin_dir / "uv",
        """#!/bin/bash
set -euo pipefail
if [ "${1:-}" = "run" ] && [ "${2:-}" = "accelerate" ]; then
    printf '%s\n' "$@" > "$CAPTURE_PATH"
    printf '%s\n' "${DDP_TIMEOUT:-unset}" > "$DDP_CAPTURE_PATH"
elif [ "${1:-}" = "run" ] && [ "${2:-}" = "python" ] && [ "${3:-}" = "scripts/manifest_token_stats.py" ]; then
    field=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--field" ]; then
            field="$2"
            break
        fi
        shift
    done
    if [ "$field" = "epochs_for_target" ]; then
        printf '%s\n' "7.5"
    else
        printf '%s\n' "1234"
    fi
else
    while IFS= read -r _line; do :; done || true
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "CAPTURE_PATH": str(capture_path),
            "DDP_CAPTURE_PATH": str(ddp_capture_path),
        }
    )
    env.update(extra_env)
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    args = (
        capture_path.read_text(encoding="utf-8").splitlines()
        if capture_path.exists()
        else []
    )
    return result, args, ddp_capture_path


def _value_after(args, flag):
    return args[args.index(flag) + 1]


def test_launcher_e05_profile_reaches_canonical_parser(tmp_path):
    manifest = tmp_path / "hf_home" / "datasets_tok" / "e05_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    result, args, ddp_capture_path = _run_launcher(
        tmp_path,
        {
            "DECODER_TYPE": "causal_ar",
            "OBJECTIVE_VARIANT": "prefix_suffix",
            "DECODER_CONTEXT_WINDOW": "128",
            "DECODER_ATTN_IMPL": "chunked_window",
            "DECODER_ATTN_CHUNK_SIZE": "256",
            "CHUNKED_CE_BLOCK_SIZE": "512",
            "PRETOKENIZED_MANIFEST": str(manifest),
            "MAX_SEQ_LENGTH": "2048",
            "MAX_EVAL_SAMPLES": "321",
            "OPTIMIZER": "muon",
            "MUON_ADAMW_LR": "0.0002",
            "MUON_MOMENTUM": "0.95",
            "WEIGHT_DECAY": "0.1",
            "TARGET_TOKENS": "1000000",
            "AUTO_INTERVALS": "1",
            "DDP_TIMEOUT": "7200",
        },
    )

    assert result.returncode == 0, result.stderr
    entrypoint_index = args.index("training/train_concept_pretraining.py")
    assert _value_after(args, "--dataset_cache_dir") == str(
        tmp_path / "hf_home" / "datasets"
    )
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)
    assert _value_after(args, "--num_train_epochs") == "7.5"
    assert _value_after(args, "--eval_steps") == "123"
    assert _value_after(args, "--save_steps") == "123"
    assert _value_after(args, "--ddp_timeout") == "7200"
    assert ddp_capture_path.read_text(encoding="utf-8").strip() == "7200"

    parser_args = args[entrypoint_index + 1 :]
    ddp_backend_index = parser_args.index("--ddp_backend")
    del parser_args[ddp_backend_index : ddp_backend_index + 2]
    parser_args.remove("--bf16")
    model_args, _, data_args, optim_args, training_args = (
        build_argument_parser().parse_args_into_dataclasses(args=parser_args)
    )
    assert model_args.decoder_type == "causal_ar"
    assert model_args.objective_variant == "prefix_suffix"
    assert model_args.decoder_context_window == 128
    assert model_args.decoder_attn_impl == "chunked_window"
    assert model_args.decoder_attn_chunk_size == 256
    assert model_args.chunked_ce_block_size == 512
    assert data_args.pretokenized_manifest == str(manifest)
    assert data_args.max_seq_length == 2048
    assert data_args.max_eval_samples == 321
    assert optim_args.optimizer == "muon"
    assert optim_args.muon_adamw_lr == 0.0002
    assert training_args.weight_decay == 0.1
    assert training_args.num_train_epochs == 7.5


def test_launcher_pretokenize_reuse_passes_only_manifest(tmp_path):
    manifest = tmp_path / "prepared" / "mix_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "PRETOKENIZE_MIX": "smollm3_inspired_2k_e05",
            "SKIP_PRETOKENIZE": "1",
            "MANIFEST": str(manifest),
            "DATASET_MIX": "ignored_registry_mix",
            "DATASET_MIX_RECIPE": "ignored_recipe",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)
    assert "--dataset_mix" not in args
    assert "--dataset_mix_recipe" not in args


def test_launcher_rejects_exact_token_budget_without_manifest(tmp_path):
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "TARGET_TOKENS": "1000000",
            "PRETOKENIZED_MANIFEST": "",
            "PRETOKENIZE_MIX": "",
        },
    )

    assert result.returncode != 0
    assert "TARGET_TOKENS requires PRETOKENIZED_MANIFEST" in result.stdout
    assert args == []


def test_legacy_launcher_generates_identical_training_arguments(tmp_path):
    canonical_result, canonical_args, _ = _run_launcher(tmp_path, {})
    legacy_result, legacy_args, _ = _run_launcher(
        tmp_path,
        {},
        launcher=LEGACY_LAUNCHER,
    )

    assert canonical_result.returncode == 0, canonical_result.stderr
    assert legacy_result.returncode == 0, legacy_result.stderr
    assert legacy_args == canonical_args


def test_e05_protocol_wrapper_pins_profile_and_delegates(tmp_path):
    manifest = tmp_path / "e05_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "SKIP_PRETOKENIZE": "1",
            "MANIFEST": str(manifest),
        },
        launcher=E05_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert "training/train_concept_pretraining.py" in args
    assert _value_after(args, "--decoder_type") == "causal_ar"
    assert _value_after(args, "--objective_variant") == "prefix_suffix"
    assert _value_after(args, "--decoder_context_window") == "128"
    assert _value_after(args, "--hidden_size") == "768"
    assert _value_after(args, "--token_embedding_dim") == "256"
    assert _value_after(args, "--max_seq_length") == "2048"
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)


def test_e10_protocol_wrapper_pins_backbone_and_delegates(tmp_path):
    data_root = tmp_path / "e10_data"
    data_root.mkdir()
    manifest = data_root / "e10_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "SKIP_PRETOKENIZE": "1",
            "MANIFEST": str(manifest),
            "TARGET_TOKENS": "",
        },
        launcher=E10_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert "training/train_concept_pretraining.py" in args
    assert _value_after(args, "--backbone_model") == "google/gemma-3-1b-pt"
    assert _value_after(args, "--objective_variant") == "causal_lm"
    assert _value_after(args, "--concept_num") == "128"
    assert _value_after(args, "--concept_block") == "512"
    assert _value_after(args, "--concept_io_mode") == "global_kv"
    assert _value_after(args, "--read_concept_norm") == "false"
    assert _value_after(args, "--read_gate_init") == "0.0"
    assert _value_after(args, "--write_gate_init") == "0.0"
    assert "--concept_memory_lr" not in args
    assert _value_after(args, "--tokenizer_name") == "google/gemma-3-1b-pt"
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)


def test_e10_protocol_wrapper_forwards_recovery_sequence_overrides(tmp_path):
    data_root = tmp_path / "e10_recovery_data"
    data_root.mkdir()
    manifest = data_root / "e10_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "SKIP_PRETOKENIZE": "1",
            "MANIFEST": str(manifest),
            "TARGET_TOKENS": "",
            "CONCEPT_IO_MODE": "shared_depth_recurrent",
            "READ_CONCEPT_NORM": "true",
            "READ_GATE_INIT": "0.01",
            "WRITE_GATE_INIT": "0.01",
            "CONCEPT_MEMORY_LR": "3e-4",
        },
        launcher=E10_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--concept_io_mode") == "shared_depth_recurrent"
    assert _value_after(args, "--read_concept_norm") == "true"
    assert _value_after(args, "--read_gate_init") == "0.01"
    assert _value_after(args, "--write_gate_init") == "0.01"
    assert _value_after(args, "--concept_memory_lr") == "3e-4"


def test_e14_protocol_wrapper_pins_forced_recall_profile(tmp_path):
    data_root = tmp_path / "e14_data"
    data_root.mkdir()
    manifest = data_root / "e14_delayed_recall_gemma_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "MANIFEST": str(manifest),
        },
        launcher=E14_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--backbone_model") == "google/gemma-3-1b-pt"
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)
    assert _value_after(args, "--preserve_precomputed_labels") == "true"
    assert _value_after(args, "--per_device_train_batch_size") == "2"
    assert _value_after(args, "--gradient_accumulation_steps") == "1"
    assert _value_after(args, "--read_concept_norm") == "true"
    assert _value_after(args, "--read_gate_init") == "0.01"
    assert _value_after(args, "--write_gate_init") == "0.01"
    assert _value_after(args, "--concept_memory_lr") == "3e-4"
    assert _value_after(args, "--warmup_steps") == "50"
    assert _value_after(args, "--eval_steps") == "164"
    assert _value_after(args, "--save_steps") == "164"


def _run_e16a_arm(tmp_path, optimizer):
    data_root = tmp_path / f"e16a_{optimizer}_data"
    data_root.mkdir()
    manifest = data_root / "smollm3_inspired_2k_e05_gemma_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "MANIFEST": str(manifest),
            "OPTIMIZER": optimizer,
        },
        launcher=E16A_LAUNCHER,
    )


def test_e16a_adam_arm_pins_matched_protocol(tmp_path):
    result, args, _ = _run_e16a_arm(tmp_path, "adam")

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--concept_io_mode") == "shared_depth_recurrent"
    assert _value_after(args, "--read_concept_norm") == "true"
    assert _value_after(args, "--read_gate_init") == "0.01"
    assert _value_after(args, "--write_gate_init") == "0.01"
    assert _value_after(args, "--max_seq_length") == "2048"
    assert _value_after(args, "--optimizer") == "adam"
    assert _value_after(args, "--learning_rate") == "1e-4"
    assert _value_after(args, "--concept_memory_lr") == "3e-4"
    assert _value_after(args, "--weight_decay") == "0.0"
    assert _value_after(args, "--warmup_steps") == "100"
    assert _value_after(args, "--num_train_epochs") == "7.5"


def test_e16a_muon_arm_pins_stabilized_recipe(tmp_path):
    result, args, _ = _run_e16a_arm(tmp_path, "muon")

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--concept_io_mode") == "shared_depth_recurrent"
    assert _value_after(args, "--optimizer") == "muon"
    assert _value_after(args, "--learning_rate") == "0.01"
    assert _value_after(args, "--muon_adamw_lr") == "2e-4"
    assert _value_after(args, "--muon_momentum") == "0.95"
    assert _value_after(args, "--weight_decay") == "0.1"
    assert "--concept_memory_lr" not in args


def test_e16a_rejects_unknown_optimizer_before_delegating(tmp_path):
    result, args, _ = _run_launcher(
        tmp_path,
        {"OPTIMIZER": "unknown"},
        launcher=E16A_LAUNCHER,
    )

    assert result.returncode == 2
    assert "must be 'adam' or 'muon'" in result.stderr
    assert args == []


def test_e16a_pipeline_orders_arms_and_stops_on_failure():
    pipeline = E16A_PIPELINE.read_text(encoding="utf-8")

    assert "set -euo pipefail" in pipeline
    assert pipeline.index("__E16A_ADAM_START__") < pipeline.index("__E16A_ADAM_COMPLETE__")
    assert pipeline.index("__E16A_ADAM_COMPLETE__") < pipeline.index("__E16A_MUON_START__")
    assert pipeline.index("__E16A_MUON_START__") < pipeline.index("__E16A_MUON_COMPLETE__")


def test_e16b_pins_longctx_muon_1b_protocol(tmp_path):
    data_root = tmp_path / "e16b_data"
    data_root.mkdir()
    manifest = data_root / "smollm3_inspired_4k_e16b_gemma_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "MANIFEST": str(manifest),
            "TARGET_TOKENS": "",
        },
        launcher=E16B_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--concept_io_mode") == "shared_depth_recurrent"
    assert _value_after(args, "--max_seq_length") == "4096"
    assert _value_after(args, "--optimizer") == "muon"
    assert _value_after(args, "--learning_rate") == "0.01"
    assert _value_after(args, "--muon_adamw_lr") == "2e-4"
    assert _value_after(args, "--weight_decay") == "0.1"
    assert _value_after(args, "--warmup_steps") == "500"
    assert _value_after(args, "--per_device_train_batch_size") == "4"
    assert _value_after(args, "--gradient_accumulation_steps") == "6"
    assert _value_after(args, "--pretokenized_manifest") == str(manifest)
    assert "--concept_memory_lr" not in args


def test_e10_accepts_max_seq_length_override(tmp_path):
    data_root = tmp_path / "e10_4k_data"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result, args, _ = _run_launcher(
        tmp_path,
        {
            "DATASETS_TOK_DIR": str(data_root),
            "DATASETS_RAW_DIR": str(tmp_path / "raw"),
            "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
            "SKIP_PRETOKENIZE": "1",
            "MANIFEST": str(manifest),
            "TARGET_TOKENS": "",
            "MAX_SEQ_LENGTH": "4096",
        },
        launcher=E10_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    assert _value_after(args, "--max_seq_length") == "4096"

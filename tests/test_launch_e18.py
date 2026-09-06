"""E18 launcher protocol tests (scripts/launch_e18.sh → generic launcher → canonical parser)."""
from pathlib import Path

from tests.test_training_launcher_parameter_flow import _run_launcher, _value_after
from training.train_concept_pretraining import build_argument_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
E18_LAUNCHER = REPO_ROOT / "scripts" / "launch_e18.sh"


def _parse(args):
    entry = args.index("training/train_concept_pretraining.py")
    parser_args = args[entry + 1 :]
    for flag in ("--ddp_backend", "--ddp_timeout"):
        if flag in parser_args:
            i = parser_args.index(flag)
            del parser_args[i : i + 2]
    return build_argument_parser().parse_args_into_dataclasses(parser_args)


def _run_stage(tmp_path, extra):
    tok_root = tmp_path / "tok"
    tok_root.mkdir(exist_ok=True)
    manifest = tok_root / "e18_pilot_longdoc_v1_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    copy_manifest = tok_root / "copy_32k_manifest.json"
    copy_manifest.write_text("{}", encoding="utf-8")
    env = {
        "DATASETS_TOK_DIR": str(tok_root),
        "DATASETS_RAW_DIR": str(tmp_path / "raw"),
        "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
        "SKIP_PRETOKENIZE": "1",
    }
    env.update(extra)
    return _run_launcher(tmp_path, env, launcher=E18_LAUNCHER)


def test_e18_stage_a_pins_perceiver_pilot_protocol(tmp_path):
    result, args, _ = _run_stage(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--model_family") == "perceiver_ar"
    assert _value_after(args, "--objective_variant") == "causal_lm"
    assert _value_after(args, "--par_mode") == "perceiver"
    assert _value_after(args, "--max_seq_length") == "8192"
    assert _value_after(args, "--par_block") == "2048"
    assert _value_after(args, "--par_pre_layers") == "1"
    assert _value_after(args, "--num_kv_heads") == "2"
    assert _value_after(args, "--attn_backend") == "flex"
    assert _value_after(args, "--optimizer") == "muon"
    assert _value_after(args, "--learning_rate") == "0.01"
    assert _value_after(args, "--muon_adamw_lr") == "2e-4"
    assert _value_after(args, "--weight_decay") == "0.1"
    assert _value_after(args, "--max_grad_norm") == "0.5"
    assert _value_after(args, "--lr_scheduler_type") == "constant_with_warmup"
    assert _value_after(args, "--prediction_loss_only") == "True"
    assert _value_after(args, "--tokenizer_name") == "HuggingFaceTB/SmolLM3-3B"
    model_args, loss_args, data_args, optim_args, training_args = _parse(args)
    assert model_args.model_family == "perceiver_ar"
    assert model_args.par_ngram_buckets == 65536
    assert model_args.par_value_embed_layers == "0,4,8"
    assert training_args.prediction_loss_only is True


def test_e18_stage_b_switches_to_32k_and_own_tok_tree(tmp_path):
    result, args, _ = _run_stage(tmp_path, {"E18_STAGE": "32k"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--max_seq_length") == "32768"
    assert _value_after(args, "--per_device_train_batch_size") == "1"
    assert _value_after(args, "--gradient_accumulation_steps") == "8"


def test_e18_dense_control_and_copy_task(tmp_path):
    result, args, _ = _run_stage(tmp_path, {"PAR_MODE": "dense"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--par_mode") == "dense"

    result, args, _ = _run_stage(tmp_path, {"E18_TASK": "copy"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--max_seq_length") == "32768"
    assert _value_after(args, "--num_hidden_layers") == "6"
    assert _value_after(args, "--preserve_precomputed_labels") == "true"
    assert _value_after(args, "--pretokenized_manifest").endswith("copy_32k_manifest.json")

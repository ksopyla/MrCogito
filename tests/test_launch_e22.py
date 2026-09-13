"""E22 launcher protocol tests (scripts/launch_e22.sh → generic launcher → canonical parser)."""
from pathlib import Path

from tests.test_training_launcher_parameter_flow import _run_launcher, _value_after
from training.train_concept_pretraining import build_argument_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
E22_LAUNCHER = REPO_ROOT / "scripts" / "launch_e22.sh"


def _parse(args):
    entry = args.index("training/train_concept_pretraining.py")
    parser_args = args[entry + 1 :]
    for flag in ("--ddp_backend", "--ddp_timeout"):
        if flag in parser_args:
            i = parser_args.index(flag)
            del parser_args[i : i + 2]
    return build_argument_parser().parse_args_into_dataclasses(parser_args)


def _run(tmp_path, extra):
    tok_root = tmp_path / "tok"
    tok_root.mkdir(exist_ok=True)
    (tok_root / "e22_longmix_32k_manifest.json").write_text("{}", encoding="utf-8")
    env = {
        "DATASETS_TOK_DIR": str(tok_root),
        "DATASETS_RAW_DIR": str(tmp_path / "raw"),
        "RAW_ARCHIVE_DIR": str(tmp_path / "raw"),
        "SKIP_PRETOKENIZE": "1",
        "E22_SMOKE": "1",   # no TARGET_TOKENS → no manifest token-stat pass on the stub manifest
    }
    env.update(extra)
    return _run_launcher(tmp_path, env, launcher=E22_LAUNCHER)


def test_e22_arm_a_pins_the_bet(tmp_path):
    result, args, _ = _run(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--model_family") == "perceiver_concept"
    assert _value_after(args, "--objective_variant") == "causal_lm"
    assert _value_after(args, "--max_seq_length") == "32768"
    assert _value_after(args, "--batch_packing_mode") == "pack"
    assert _value_after(args, "--loss_span_markers") == "128103,128104"
    assert _value_after(args, "--pcl_enc_layers") == "6"
    assert _value_after(args, "--pcl_concept_ratio") == "16"
    assert _value_after(args, "--pcl_latent_layers") == "4"
    assert _value_after(args, "--pcl_dec_layers") == "8"
    assert _value_after(args, "--pcl_dec_segment") == "1024"
    assert _value_after(args, "--pcl_dec_local") == "block"
    assert _value_after(args, "--pcl_concept_mode") == "full"
    assert _value_after(args, "--attn_backend") == "flex"
    assert _value_after(args, "--optimizer") == "muon"
    assert _value_after(args, "--learning_rate") == "0.01"
    assert _value_after(args, "--lr_scheduler_type") == "cosine"
    assert _value_after(args, "--tokenizer_name") == "HuggingFaceTB/SmolLM3-3B"
    model_args, loss_args, data_args, optim_args, training_args = _parse(args)
    assert model_args.model_family == "perceiver_concept"
    assert model_args.par_ngram_buckets == 65536
    assert model_args.pcl_enc_value_embed_layers == "0,3"
    assert data_args.pretokenized_manifest.endswith("e22_longmix_32k_manifest.json")


def test_e22_arm_c_disables_the_array(tmp_path):
    result, args, _ = _run(tmp_path, {"E22_ARM": "C"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--model_family") == "perceiver_concept"
    assert _value_after(args, "--pcl_concept_mode") == "none"


def test_e22_dense_control_is_matched_perceiver_ar(tmp_path):
    result, args, _ = _run(tmp_path, {"E22_ARM": "dense"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert _value_after(args, "--model_family") == "perceiver_ar"
    assert _value_after(args, "--par_mode") == "dense"
    assert _value_after(args, "--num_hidden_layers") == "18"
    assert _value_after(args, "--max_seq_length") == "32768"
    assert _value_after(args, "--batch_packing_mode") == "pack"


def test_e22_refuses_unknown_arm(tmp_path):
    result, _, _ = _run(tmp_path, {"E22_ARM": "Z"})
    assert result.returncode != 0

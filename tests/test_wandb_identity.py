from training.utils_training import build_perceiver_wandb_identity


def test_perceiver_denoise_reconstruction_identity():
    identity = build_perceiver_wandb_identity(
        decoder_type="perceiver_posonly",
        objective_variant="reconstruction",
        hidden_size=512,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=3,
        checkpoint_family="perceiver_denoise",
        pretraining_objective="denoising_full_reconstruction",
        use_bixt=True,
    )

    assert identity.experiment_id is None
    assert identity.model_family == "perceiver_denoise"
    assert identity.objective_family == "reconstruction"
    assert identity.architecture_id == "perceiver_denoise_H512L6C128D3"
    assert identity.group == "perceiver_denoise_H512L6C128D3"
    assert identity.job_type == "train_perceiver_denoise_reconstruction"
    assert "perceiver-denoise" not in identity.tags
    assert {"train", "perceiver_denoise", "bixt"}.issubset(identity.tags)


def test_e01_concept_ar_reconstruction_identity():
    identity = build_perceiver_wandb_identity(
        decoder_type="causal_ar",
        objective_variant="reconstruction",
        hidden_size=768,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=4,
        checkpoint_family="concept_ar",
        pretraining_objective="ar_denoising_reconstruction",
        use_bixt=True,
    )

    assert identity.experiment_id == "E01"
    assert identity.model_family == "concept_ar"
    assert identity.objective_family == "ar_reconstruction"
    assert identity.architecture_id == "concept_ar_H768L6C128D4"
    assert identity.group == "E01_concept_ar_H768L6C128D4"
    assert identity.job_type == "train_concept_ar_reconstruction"
    assert {"E01", "concept_ar", "ar_reconstruction", "causal_ar"}.issubset(identity.tags)


def test_e02_prefix_suffix_identity():
    identity = build_perceiver_wandb_identity(
        decoder_type="causal_ar",
        objective_variant="prefix_suffix",
        hidden_size=768,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=4,
        checkpoint_family="concept_ar",
        pretraining_objective="ar_prefix_suffix_generation",
        use_bixt=True,
    )

    assert identity.experiment_id == "E02"
    assert identity.model_family == "concept_ar_prefix"
    assert identity.objective_family == "prefix_suffix"
    assert identity.architecture_id == "concept_ar_prefix_H768L6C128D4"
    assert identity.group == "E02_concept_ar_prefix_H768L6C128D4"
    assert identity.job_type == "train_concept_ar_prefix_suffix"
    assert {"E02", "concept_ar_prefix", "prefix_suffix", "causal_ar"}.issubset(identity.tags)


def test_explicit_experiment_id_overrides_inferred_default():
    identity = build_perceiver_wandb_identity(
        decoder_type="causal_ar",
        objective_variant="reconstruction",
        hidden_size=768,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=4,
        checkpoint_family="concept_ar",
        pretraining_objective="ar_denoising_reconstruction",
        use_bixt=True,
        experiment_id="E03",
    )

    assert identity.experiment_id == "E03"
    assert identity.group == "E03_concept_ar_H768L6C128D4"
    assert "E03" in identity.tags

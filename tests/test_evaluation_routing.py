import pytest

from evaluation.concept_eval_routing import resolve_concept_eval_route
from nn.concept_encoder import ConceptEncoderConfig


def test_diffusion_pair_tasks_require_metadata_contract():
    config = ConceptEncoderConfig()

    with pytest.raises(ValueError, match="requires evaluation metadata"):
        resolve_concept_eval_route(
            config=config,
            requested_model_type="prefix_diffusion",
            has_pair_inputs=True,
        )


def test_diffusion_pair_tasks_route_to_sentence_pair():
    config = ConceptEncoderConfig(
        checkpoint_family="prefix_diffusion",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="weighted_pool",
    )

    route = resolve_concept_eval_route(
        config=config,
        requested_model_type="prefix_diffusion",
        has_pair_inputs=True,
    )

    assert route.model_mode == "sentence_pair"
    assert route.pair_input_mode == "separate"
    assert route.load_mode == "encoder_only"


def test_diffusion_single_tasks_route_to_weighted_pool():
    config = ConceptEncoderConfig(
        checkpoint_family="diffusion_mlm",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="weighted_pool",
    )

    route = resolve_concept_eval_route(
        config=config,
        requested_model_type="diffusion_mlm",
        has_pair_inputs=False,
    )

    assert route.model_mode == "weighted_pool"
    assert route.pair_input_mode == "concatenated"


def test_decoder_classifier_uses_encoder_decoder_loading():
    config = ConceptEncoderConfig(
        checkpoint_family="perceiver_denoise",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="via_decoder",
    )

    route = resolve_concept_eval_route(
        config=config,
        requested_model_type="perceiver_denoise",
        has_pair_inputs=False,
    )

    assert route.model_mode == "via_decoder"
    assert route.load_mode == "encoder_decoder"


def test_perceiver_pair_tasks_route_to_sentence_pair():
    config = ConceptEncoderConfig(
        checkpoint_family="perceiver_denoise",
        evaluation_contract_version=1,
        canonical_pair_eval_mode="sentence_pair",
        canonical_single_eval_mode="via_decoder",
    )

    route = resolve_concept_eval_route(
        config=config,
        requested_model_type="perceiver_denoise",
        has_pair_inputs=True,
    )

    assert route.model_mode == "sentence_pair"
    assert route.pair_input_mode == "separate"
    assert route.load_mode == "encoder_only"


def test_perceiver_family_requires_metadata_contract():
    config = ConceptEncoderConfig()

    with pytest.raises(ValueError, match="requires evaluation metadata"):
        resolve_concept_eval_route(
            config=config,
            requested_model_type="perceiver_denoise",
            has_pair_inputs=False,
        )

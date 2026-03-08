from dataclasses import dataclass
from typing import Optional


DIFFUSION_FAMILIES = {"diffusion_mlm", "prefix_diffusion"}
PERCEIVER_FAMILIES = {"perceiver_denoise"}


@dataclass(frozen=True)
class ConceptEvalRoute:
    checkpoint_family: str
    model_mode: str
    pair_input_mode: str
    load_mode: str


def _get_config_value(config, name: str, default=None):
    return getattr(config, name, default)


def resolve_checkpoint_family(config, requested_model_type: str) -> str:
    family = _get_config_value(config, "checkpoint_family")
    if family:
        return family

    if requested_model_type in DIFFUSION_FAMILIES | PERCEIVER_FAMILIES:
        raise ValueError(
            "This checkpoint family now requires evaluation metadata in the saved "
            "config. Re-export or retrain the checkpoint with the updated training "
            "scripts so evaluation can choose the canonical route automatically."
        )

    return requested_model_type


def resolve_concept_eval_route(
    config,
    requested_model_type: str,
    has_pair_inputs: bool,
) -> ConceptEvalRoute:
    family = resolve_checkpoint_family(config, requested_model_type)

    if family == "weighted_mlm":
        return ConceptEvalRoute(
            checkpoint_family=family,
            model_mode="weighted_pool",
            pair_input_mode="concatenated",
            load_mode="full",
        )

    if family in DIFFUSION_FAMILIES | PERCEIVER_FAMILIES:
        contract_version = _get_config_value(config, "evaluation_contract_version")
        pair_mode = _get_config_value(config, "canonical_pair_eval_mode")
        single_mode = _get_config_value(config, "canonical_single_eval_mode")

        if contract_version != 1:
            raise ValueError(
                f"Unsupported evaluation contract version: {contract_version!r}. "
                "Only version 1 is currently supported."
            )

        if has_pair_inputs:
            if pair_mode != "sentence_pair":
                raise ValueError(
                    "This checkpoint family must declare "
                    "canonical_pair_eval_mode='sentence_pair'."
                )
            return ConceptEvalRoute(
                checkpoint_family=family,
                model_mode="sentence_pair",
                pair_input_mode="separate",
                load_mode="encoder_only",
            )

        if single_mode == "weighted_pool":
            return ConceptEvalRoute(
                checkpoint_family=family,
                model_mode="weighted_pool",
                pair_input_mode="concatenated",
                load_mode="encoder_only",
            )

        if single_mode == "via_decoder":
            return ConceptEvalRoute(
                checkpoint_family=family,
                model_mode="via_decoder",
                pair_input_mode="concatenated",
                load_mode="encoder_decoder",
            )

        raise ValueError(
            "Unsupported canonical_single_eval_mode. Expected 'weighted_pool' or "
            f"'via_decoder', got {single_mode!r}."
        )

    raise ValueError(f"Unsupported concept checkpoint family: {family}")


def is_separate_pair_route(route: ConceptEvalRoute) -> bool:
    return route.pair_input_mode == "separate"

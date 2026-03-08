import os

import torch

from nn.concept_encoder_perceiver import (
    ConceptEncoderForSequenceClassificationPerceiver,
    ConceptEncoderForSequenceClassificationViaDecoder,
    ConceptEncoderForSentencePairClassification,
)
from nn.concept_encoder_weighted import ConceptEncoderForSequenceClassificationWeighted


def select_concept_eval_model_class(route):
    if route.model_mode == "weighted_pool" and route.load_mode == "full":
        return ConceptEncoderForSequenceClassificationWeighted
    if route.model_mode == "weighted_pool":
        return ConceptEncoderForSequenceClassificationPerceiver
    if route.model_mode == "via_decoder":
        return ConceptEncoderForSequenceClassificationViaDecoder
    if route.model_mode == "sentence_pair":
        return ConceptEncoderForSentencePairClassification
    raise ValueError(f"Unsupported concept evaluation route: {route.model_mode}")


def resolve_checkpoint_file(model_path: str) -> str | None:
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        return safetensors_path

    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return pytorch_path

    return None


def load_checkpoint_state_dict(model_path: str):
    checkpoint_path = resolve_checkpoint_file(model_path)
    if checkpoint_path is None:
        return None

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint_path)

    return torch.load(checkpoint_path, map_location="cpu")


def load_concept_checkpoint_weights(model, checkpoint_state_dict, route):
    if checkpoint_state_dict is None:
        return 0, 0

    model_state_dict = model.state_dict()
    loaded = 0
    skipped = 0

    if route.load_mode == "encoder_decoder":
        for key, value in checkpoint_state_dict.items():
            if key.startswith("lm_head.") or key.startswith("loss_manager."):
                continue
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded += 1
            else:
                skipped += 1
    elif route.load_mode == "encoder_only":
        for key, value in checkpoint_state_dict.items():
            if key.startswith("encoder.") and key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded += 1
            else:
                skipped += 1
    else:
        for key, value in checkpoint_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded += 1
            else:
                skipped += 1

    model.load_state_dict(model_state_dict)
    return loaded, skipped

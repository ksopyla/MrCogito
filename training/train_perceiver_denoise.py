"""Compatibility wrapper for the renamed concept-pretraining entrypoint.

Use ``training/train_concept_pretraining.py`` for new launchers and imports.
This path remains executable for one migration window and preserves the public
symbols imported by existing external code.
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_concept_pretraining import (
    DECODER_CAUSAL_AR,
    DECODER_PERCEIVER_POSONLY,
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    VALID_DECODER_TYPES,
    VALID_OBJECTIVES,
    DataTrainingArguments,
    LossArguments,
    ModelArguments,
    OptimizerArguments,
    PerceiverDenoiseTrainer,
    align_special_tokens_for_training,
    build_perceiver_denoise_config,
    main,
    resolve_append_eos_token_id,
)


__all__ = [
    "DECODER_CAUSAL_AR",
    "DECODER_PERCEIVER_POSONLY",
    "OBJECTIVE_CAUSAL_LM",
    "OBJECTIVE_PREFIX_SUFFIX",
    "OBJECTIVE_RECONSTRUCTION",
    "OBJECTIVE_RECONSTRUCTION_CONTRASTIVE",
    "VALID_DECODER_TYPES",
    "VALID_OBJECTIVES",
    "DataTrainingArguments",
    "LossArguments",
    "ModelArguments",
    "OptimizerArguments",
    "PerceiverDenoiseTrainer",
    "align_special_tokens_for_training",
    "build_perceiver_denoise_config",
    "main",
    "resolve_append_eos_token_id",
]


if __name__ == "__main__":
    main()

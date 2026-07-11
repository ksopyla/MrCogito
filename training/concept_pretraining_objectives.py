"""Import-light objective and decoder constants shared with preprocessing scripts."""


OBJECTIVE_RECONSTRUCTION = "reconstruction"
OBJECTIVE_RECONSTRUCTION_CONTRASTIVE = "reconstruction+contrastive"
OBJECTIVE_PREFIX_SUFFIX = "prefix_suffix"
OBJECTIVE_CAUSAL_LM = "causal_lm"
VALID_OBJECTIVES = {
    OBJECTIVE_RECONSTRUCTION,
    OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_CAUSAL_LM,
}

DECODER_PERCEIVER_POSONLY = "perceiver_posonly"
DECODER_CAUSAL_AR = "causal_ar"
VALID_DECODER_TYPES = {DECODER_PERCEIVER_POSONLY, DECODER_CAUSAL_AR}


def resolve_append_eos_token_id(objective_variant, is_causal_ar, eos_token_id):
    """Return the EOS id for variable-length preprocessing when the objective needs it."""
    objective_appends_eos = objective_variant in (
        OBJECTIVE_RECONSTRUCTION,
        OBJECTIVE_RECONSTRUCTION_CONTRASTIVE,
        OBJECTIVE_PREFIX_SUFFIX,
        OBJECTIVE_CAUSAL_LM,
    )
    if eos_token_id is not None and (is_causal_ar or objective_appends_eos):
        return eos_token_id
    return None

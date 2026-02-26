from transformers import PreTrainedModel

from nn.concept_encoder import ConceptEncoder, ConceptEncoderConfig

class ConceptEncoderForMaskedLM(PreTrainedModel):
    """
    ConceptEncoder Model with a language modeling head on top (for masked language modeling).

    Args:
        config (ConceptEncoderConfig): Model configuration defining hidden sizes, embeddings, etc.
    """
    config_class = ConceptEncoderConfig
    base_model_prefix = "concept_encoder"

    def __init__(self, config: ConceptEncoderConfig):
        super().__init__(config)
        self.config = config

        # The underlying ConceptEncoder (as defined above).
        self.encoder = ConceptEncoder(config) # []

        # todo - add necessary variables here when we establisht how we want to combine the concepts and tokens


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, special_tokens_mask=None, labels=None):

        pass


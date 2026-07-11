import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted


def test_weighted_mlm_checkpoint_round_trip_remains_supported(tmp_path):
    config = ConceptEncoderConfig(
        vocab_size=32,
        max_sequence_length=8,
        hidden_size=16,
        token_embedding_dim=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        concept_num=4,
        pad_token_id=0,
        mask_token_id=1,
    )
    model = ConceptEncoderForMaskedLMWeighted(config)
    model.save_pretrained(tmp_path)

    loaded = ConceptEncoderForMaskedLMWeighted.from_pretrained(tmp_path)
    output = loaded(
        input_ids=torch.tensor([[2, 3, 4, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )

    assert output.logits.shape == (1, 4, 32)
    assert torch.equal(model.concept_weights, loaded.concept_weights)

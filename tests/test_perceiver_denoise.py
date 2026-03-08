import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForDenoisingPerceiver,
    ConceptEncoderForSequenceClassificationViaDecoder,
    ConceptEncoderForSentencePairClassification,
)
from training.train_perceiver_denoise import (
    DataTrainingArguments,
    ModelArguments,
    build_perceiver_denoise_config,
)


class _DummyTokenizer:
    pad_token_id = 0
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    eos_token_id = 4
    unk_token_id = 5

    def __len__(self):
        return 32


def test_build_perceiver_denoise_config_stamps_eval_contract():
    config = build_perceiver_denoise_config(
        _DummyTokenizer(),
        ModelArguments(
            hidden_size=32,
            token_embedding_dim=32,
            num_hidden_layers=2,
            concept_num=8,
            intermediate_size=64,
            decoder_num_layers=2,
            use_bixt=True,
        ),
        DataTrainingArguments(max_seq_length=16, tokenizer_name="dummy"),
    )

    assert config.checkpoint_family == "perceiver_denoise"
    assert config.evaluation_contract_version == 1
    assert config.canonical_pair_eval_mode == "sentence_pair"
    assert config.canonical_single_eval_mode == "via_decoder"
    assert config.decoder_posonly is True
    assert config.decoder_num_layers == 2


def test_denoising_model_uses_stacked_decoder_layers():
    config = ConceptEncoderConfig(
        vocab_size=32,
        hidden_size=32,
        token_embedding_dim=32,
        concept_num=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        decoder_num_layers=2,
    )
    model = ConceptEncoderForDenoisingPerceiver(config)

    assert len(model.decoder.layers) == 2

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.logits.shape == (2, 16, config.vocab_size)
    assert outputs.loss is not None


def test_via_decoder_shares_decoder_stack_shapes():
    config = ConceptEncoderConfig(
        vocab_size=32,
        hidden_size=32,
        token_embedding_dim=32,
        concept_num=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        decoder_num_layers=3,
        num_labels=2,
        decoder_posonly=True,
    )
    pretrain_model = ConceptEncoderForDenoisingPerceiver(config)
    clf_model = ConceptEncoderForSequenceClassificationViaDecoder(config)

    for key in [
        "decoder.query_embeddings.weight",
        "decoder.layers.0.self_attn.in_proj_weight",
        "decoder.layers.1.cross_attn.in_proj_weight",
        "decoder.output_norm.weight",
    ]:
        assert key in pretrain_model.state_dict()
        assert key in clf_model.state_dict()
        assert pretrain_model.state_dict()[key].shape == clf_model.state_dict()[key].shape


def test_sentence_pair_classifier_supports_cosine_only():
    config = ConceptEncoderConfig(
        vocab_size=32,
        hidden_size=32,
        token_embedding_dim=32,
        concept_num=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        num_labels=1,
    )
    model = ConceptEncoderForSentencePairClassification(config)

    input_ids_a = torch.randint(0, config.vocab_size, (2, 16))
    input_ids_b = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids_a)
    labels = torch.rand(2)

    outputs = model(
        input_ids_a=input_ids_a,
        attention_mask_a=attention_mask,
        input_ids_b=input_ids_b,
        attention_mask_b=attention_mask,
        labels=labels,
        cosine_only=True,
    )

    assert outputs.logits.shape == (2, 1)
    assert outputs.loss is not None

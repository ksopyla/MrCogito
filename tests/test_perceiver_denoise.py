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
    resolve_append_eos_token_id,
)
from training.utils_training import build_perceiver_wandb_identity


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
        "decoder.layers.0.cross_attn.in_proj_weight",
        "decoder.layers.1.cross_attn.in_proj_weight",
        "decoder.output_norm.weight",
    ]:
        assert key in pretrain_model.state_dict()
        assert key in clf_model.state_dict()
        assert pretrain_model.state_dict()[key].shape == clf_model.state_dict()[key].shape


def test_perceiver_decoder_is_linear_no_output_self_attention():
    """Regression guard: the parallel Perceiver-IO decoder must have NO self-attention over
    the N output queries (that would be O(N^2) and break the O(C*N) bottleneck invariant).
    Each layer keeps only cross-attention to the C concepts + an FFN."""
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

    layer = model.decoder.layers[0]
    assert not hasattr(layer, "self_attn"), "output self-attention must be removed (O(N^2))"
    assert not hasattr(layer, "pre_self_norm")
    assert hasattr(layer, "cross_attn")

    # No self_attn weights anywhere in the decoder state dict.
    assert not any("self_attn" in k for k in model.decoder.state_dict())


def test_perceiver_reconstruction_appends_eos_like_causal_ar():
    """The perceiver reconstruction path MUST append EOS (variable-length preprocessing), the same
    contract as the causal_ar baseline. Regression guard for the pad-mask bug: the old behaviour
    (append_eos=None for the perceiver path) pre-padded to max_length and corrupted masks/labels."""
    EOS = 4
    # Parallel (perceiver) reconstruction — the E04 path — must now append EOS.
    assert resolve_append_eos_token_id("reconstruction", is_causal_ar=False, eos_token_id=EOS) == EOS
    assert resolve_append_eos_token_id("reconstruction+contrastive", is_causal_ar=False, eos_token_id=EOS) == EOS
    # Causal AR (E01/E02/E03) keeps appending EOS as before.
    assert resolve_append_eos_token_id("reconstruction", is_causal_ar=True, eos_token_id=EOS) == EOS
    assert resolve_append_eos_token_id("prefix_suffix", is_causal_ar=True, eos_token_id=EOS) == EOS
    # No EOS token in the tokenizer -> cannot append (fall back to legacy path).
    assert resolve_append_eos_token_id("reconstruction", is_causal_ar=False, eos_token_id=None) is None


def test_wandb_identity_tags_parallel_reconstruction():
    """W&B clarity: the parallel reconstruction run carries legible decoder/task tags and a
    scannable job_type, and defaults to the E04 experiment when none is passed."""
    identity = build_perceiver_wandb_identity(
        decoder_type="perceiver_posonly",
        objective_variant="reconstruction",
        hidden_size=768,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=4,
        checkpoint_family="perceiver_denoise",
        pretraining_objective="reconstruction",
        use_bixt=True,
    )
    assert "decoder:parallel" in identity.tags
    assert "task:reconstruction" in identity.tags
    assert identity.job_type == "train_parallel_reconstruction"
    assert identity.experiment_id == "E04"
    # Routing key untouched.
    assert "perceiver_denoise" in identity.tags


def test_wandb_identity_tags_ar_generation():
    identity = build_perceiver_wandb_identity(
        decoder_type="causal_ar",
        objective_variant="prefix_suffix",
        hidden_size=768,
        num_hidden_layers=6,
        concept_num=128,
        decoder_num_layers=4,
        checkpoint_family="concept_ar",
        pretraining_objective="prefix_suffix",
        use_bixt=True,
    )
    assert "decoder:autoregressive" in identity.tags
    assert "task:generation" in identity.tags
    assert identity.job_type == "train_ar_generation_prefix_suffix"


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

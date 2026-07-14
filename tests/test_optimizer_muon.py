"""Optimizer routing: `--optimizer muon` builds nn.muon.Muon via PerceiverDenoiseTrainer.create_optimizer;
the default (`--optimizer adam`) path is byte-unchanged. CPU-only; no dataloader needed
(create_optimizer only touches self.model + self.args)."""
import torch
import pytest
from transformers import TrainingArguments

from nn.backbone_concept_lm import BackboneConceptConfig, BackboneConceptLM
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from nn.muon import Muon
from training.train_concept_pretraining import (
    DataTrainingArguments,
    ModelArguments,
    PerceiverDenoiseTrainer,
    build_perceiver_denoise_config,
)


class _Tok:
    """Minimal tokenizer stand-in (same shape as the test_concept_ar_decoder pattern)."""
    pad_token_id, mask_token_id, cls_token_id = 0, None, None
    sep_token_id, bos_token_id, eos_token_id, unk_token_id = None, 1, 2, None

    def __len__(self):
        return 40


def _tiny_model():
    config = build_perceiver_denoise_config(
        _Tok(),
        ModelArguments(
            hidden_size=32,
            token_embedding_dim=16,
            num_hidden_layers=2,
            concept_num=8,
            intermediate_size=64,
            decoder_num_layers=2,
            decoder_type="causal_ar",
            decoder_pos_type="rope",
            hidden_act="silu",
            norm_type="rmsnorm",
            use_bixt=True,
        ),
        DataTrainingArguments(max_seq_length=16, tokenizer_name="dummy"),
    )
    return ConceptEncoderForConditionalLM(config)


def _trainer(tmp_path, optimizer_choice):
    # HF `optim` stays a valid default (HF coerces --optim to its enum and rejects "muon"); our
    # own `--optimizer` flag (threaded as optimizer_choice) is the real selector.
    args = TrainingArguments(
        output_dir=str(tmp_path),
        optim="adamw_torch_fused",
        learning_rate=0.02 if optimizer_choice == "muon" else 5e-5,
        weight_decay=0.0,
        per_device_train_batch_size=1,
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        logging_steps=1,
    )
    return PerceiverDenoiseTrainer(
        model=_tiny_model(),
        args=args,
        objective_variant="reconstruction",
        contrastive_weight=0.3,
        contrastive_temperature=0.05,
        optimizer_choice=optimizer_choice,
        muon_adamw_lr=2e-3,
        muon_momentum=0.95,
    )


def _tiny_backbone(concept_io_mode="global_kv"):
    config = BackboneConceptConfig(
        backbone_model="tiny-random-gemma3",
        backbone_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "query_pre_attn_scalar": 32,
            "sliding_window": 8,
            "max_position_embeddings": 64,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "attention_dropout": 0.0,
        },
        concept_num=4,
        concept_block=8,
        concept_io_mode=concept_io_mode,
        write_num_heads=2,
        read_concept_norm=True,
        read_gate_init=0.01,
        write_gate_init=0.01,
        lora_r=2,
        lora_dropout=0.0,
    )
    return BackboneConceptLM(config)


def _backbone_trainer(
    tmp_path,
    *,
    optimizer_choice="adam",
    concept_memory_lr=3e-4,
    concept_io_mode="global_kv",
):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        optim="adamw_torch",
        learning_rate=1e-4,
        weight_decay=0.1,
        per_device_train_batch_size=1,
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
    )
    return PerceiverDenoiseTrainer(
        model=_tiny_backbone(concept_io_mode),
        args=args,
        objective_variant="causal_lm",
        contrastive_weight=0.3,
        contrastive_temperature=0.05,
        optimizer_choice=optimizer_choice,
        concept_memory_lr=concept_memory_lr,
    )


def test_muon_routes_through_trainer(tmp_path):
    trainer = _trainer(tmp_path, optimizer_choice="muon")
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer, Muon)


def test_default_optim_unchanged(tmp_path):
    """The non-Muon path falls through to the HF default — not a Muon instance."""
    trainer = _trainer(tmp_path, optimizer_choice="adam")
    trainer.create_optimizer()
    assert not isinstance(trainer.optimizer, Muon)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_muon_step_runs(tmp_path):
    """A full Muon step (Newton-Schulz orthogonalization on 2D params + AdamW fallback) executes."""
    trainer = _trainer(tmp_path, optimizer_choice="muon")
    trainer.create_optimizer()
    p = next(trainer.model.parameters())
    p.grad = torch.ones_like(p)
    trainer.optimizer.step()  # must not raise
    assert torch.isfinite(p).all()


def test_backbone_differential_adamw_partitions_every_trainable_once(tmp_path):
    trainer = _backbone_trainer(tmp_path)
    optimizer = trainer.create_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)

    grouped_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_ids = [
        id(parameter)
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert {name.removesuffix("_no_decay").removesuffix("_decay") for name in groups} == {
        "lora",
        "concept_memory",
    }
    assert all(
        group["lr"] == pytest.approx(3e-4)
        for name, group in groups.items()
        if name.startswith("concept_memory")
    )
    assert all(
        group["lr"] == pytest.approx(1e-4)
        for name, group in groups.items()
        if name.startswith("lora")
    )
    assert any(group["weight_decay"] == pytest.approx(0.0) for group in groups.values())
    assert any(group["weight_decay"] == pytest.approx(0.1) for group in groups.values())


def test_shared_depth_gates_use_concept_memory_optimizer_group(tmp_path):
    trainer = _backbone_trainer(
        tmp_path,
        concept_io_mode="shared_depth_recurrent",
    )
    optimizer = trainer.create_optimizer()
    depth_gate = trainer.model.write_head.depth_alphas
    matching_groups = [
        group for group in optimizer.param_groups
        if any(parameter is depth_gate for parameter in group["params"])
    ]
    assert len(matching_groups) == 1
    assert matching_groups[0]["group_name"] == "concept_memory_no_decay"
    assert matching_groups[0]["lr"] == pytest.approx(3e-4)


def test_differential_adamw_fails_on_unknown_trainable_parameter(tmp_path):
    trainer = _backbone_trainer(tmp_path)
    trainer.model.unclassified_trainable = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="unclassified trainable"):
        trainer.create_optimizer()


def test_differential_concept_lr_rejects_muon(tmp_path):
    trainer = _backbone_trainer(tmp_path, optimizer_choice="muon")
    with pytest.raises(ValueError, match="only supported with optimizer='adam'"):
        trainer.create_optimizer()

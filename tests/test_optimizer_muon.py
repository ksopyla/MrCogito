"""Optimizer routing: `--optimizer muon` builds nn.muon.Muon via PerceiverDenoiseTrainer.create_optimizer;
the default (`--optimizer adam`) path is byte-unchanged. CPU-only; no dataloader needed
(create_optimizer only touches self.model + self.args)."""
import torch
from transformers import TrainingArguments

from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from nn.muon import Muon
from training.train_perceiver_denoise import (
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

from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from data.length_grouped_sampler import CachedLengthGroupedSampler
from training.concept_pretraining_trainer import PerceiverDenoiseTrainer


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index + 1.0]),
            "labels": torch.tensor([0.0]),
        }


class _TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_ids, labels):
        predictions = input_ids * self.weight
        loss = torch.nn.functional.mse_loss(predictions, labels)
        return SimpleNamespace(loss=loss, logits=predictions)


def test_trainer_runs_one_real_optimizer_step_and_logs_loss(tmp_path):
    model = _TinyLossModel()
    original_weight = model.weight.detach().clone()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=0.1,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
    )
    trainer = PerceiverDenoiseTrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(),
        objective_variant="reconstruction",
        contrastive_weight=0.3,
        contrastive_temperature=0.05,
        optimizer_choice="adam",
        muon_adamw_lr=2e-3,
        muon_momentum=0.95,
    )

    result = trainer.train()

    assert result.global_step == 1
    assert not torch.equal(model.weight.detach(), original_weight)
    assert any("loss" in record for record in trainer.state.log_history)
    assert trainer.args.gradient_accumulation_steps == 2
    assert trainer.args.learning_rate == 0.1


def test_trainer_uses_cached_length_sampler_without_owning_ddp_sharding(tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
        report_to="none",
        use_cpu=True,
    )
    lengths = [1, 7]
    trainer = PerceiverDenoiseTrainer(
        model=_TinyLossModel(),
        args=args,
        train_dataset=_TinyDataset(),
        objective_variant="reconstruction",
        contrastive_weight=0.3,
        contrastive_temperature=0.05,
        batch_packing_mode="length_group",
        train_lengths=lengths,
        length_group_mega_batch_mult=20,
    )

    sampler = trainer._get_train_sampler()

    assert isinstance(sampler, CachedLengthGroupedSampler)
    assert sampler.batch_size == 6
    assert sampler.mega_batch_mult == 20
    assert sorted(sampler) == [0, 1]


def test_padding_metrics_are_added_to_the_next_trainer_log(tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        use_cpu=True,
    )
    trainer = PerceiverDenoiseTrainer(
        model=_TinyLossModel(),
        args=args,
        train_dataset=_TinyDataset(),
        objective_variant="reconstruction",
        contrastive_weight=0.3,
        contrastive_temperature=0.05,
    )
    trainer._record_padding_metrics(torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))

    trainer.log({"loss": 1.0})

    logged = trainer.state.log_history[-1]
    assert logged["data/pad_ratio"] == 0.25
    assert logged["data/real_tokens_per_batch"] == 6.0
    assert logged["data/padded_tokens_per_batch"] == 2.0
    assert logged["data/mean_sequence_length"] == 3.0
    assert logged["data/mean_batch_max_length"] == 4.0
    assert logged["perf/real_tokens_per_second"] > 0

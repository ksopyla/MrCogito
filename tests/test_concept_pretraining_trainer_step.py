from types import SimpleNamespace

import torch
from transformers import TrainingArguments

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

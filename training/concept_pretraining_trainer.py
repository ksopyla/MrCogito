"""Custom Trainer behavior shared by concept-pretraining model families."""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, Trainer, logging
from transformers.modeling_outputs import MaskedLMOutput

from nn.concept_encoder_perceiver import ConceptEncoderForDenoisingPerceiver
from training.concept_pretraining_objectives import (
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_PREFIX_SUFFIX,
    OBJECTIVE_RECONSTRUCTION,
)
from training.utils_training import is_main_process


logger = logging.get_logger("training.train_concept_pretraining")


class PerceiverDenoiseTrainer(Trainer):
    def __init__(
        self,
        *args,
        objective_variant: str,
        contrastive_weight: float,
        contrastive_temperature: float,
        compute_concept_ablation: bool = False,
        concept_ablation_batches: int = 5,
        eval_data_collator=None,
        anchor_loss: bool = False,
        anchor_loss_weight: float = 0.5,
        anchor_standardize: bool = True,
        anchor_model_name: Optional[str] = None,
        optimizer_choice: str = "adam",
        concept_memory_lr: Optional[float] = None,
        muon_adamw_lr: float = 2e-3,
        muon_momentum: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.objective_variant = objective_variant
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.compute_concept_ablation = compute_concept_ablation
        self.concept_ablation_batches = concept_ablation_batches
        self.eval_data_collator = eval_data_collator
        self.optimizer_choice = optimizer_choice
        self.concept_memory_lr = concept_memory_lr
        self.muon_adamw_lr = muon_adamw_lr
        self.muon_momentum = muon_momentum
        self.anchor_loss = anchor_loss
        self.anchor_loss_weight = anchor_loss_weight
        self.anchor_standardize = anchor_standardize
        self.anchor_teacher = None
        if anchor_loss:
            if anchor_model_name is None:
                raise ValueError("anchor_loss=True requires anchor_model_name.")
            logger.info(f"Loading frozen anchor teacher: {anchor_model_name}")
            teacher = AutoModel.from_pretrained(anchor_model_name)
            teacher.eval()
            teacher.requires_grad_(False)
            self.anchor_teacher = teacher.to(self.args.device)

    def create_optimizer(self):
        """Build the configured Adam or Muon optimizer."""
        if self.optimizer_choice == "muon":
            if self.concept_memory_lr is not None:
                raise ValueError(
                    "concept_memory_lr is only supported with optimizer='adam'; "
                    "Muon routes parameters by tensor shape rather than E10 parameter role."
                )
            from nn.muon import Muon

            self.optimizer = Muon(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=self.muon_momentum,
                adamw_lr=self.muon_adamw_lr,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        if self.concept_memory_lr is not None:
            return self._create_backbone_differential_adamw()
        return super().create_optimizer()

    @staticmethod
    def _backbone_parameter_role(name: str) -> str:
        """Classify E10 trainables for differential AdamW; fail closed on future drift."""
        if name.startswith("module."):
            name = name[len("module.") :]
        if "lora_A" in name or "lora_B" in name:
            return "lora"
        if (
            name == "concept_init"
            or name.startswith("write_head.")
            or name.endswith(".gate")
            or ".read_branch.concept_norm." in name
        ):
            return "concept_memory"
        return "unknown"

    def _create_backbone_differential_adamw(self):
        if self.concept_memory_lr is None or self.concept_memory_lr <= 0:
            raise ValueError("concept_memory_lr must be a positive float when set.")
        family = getattr(self.model.config, "checkpoint_family", None)
        if family != "backbone_concept":
            raise ValueError(
                "concept_memory_lr requires the backbone_concept family; "
                f"got checkpoint_family={family!r}."
            )

        decay_names = set(self.get_decay_parameter_names(self.model))
        buckets = {
            ("lora", True): [],
            ("lora", False): [],
            ("concept_memory", True): [],
            ("concept_memory", False): [],
        }
        unknown = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            role = self._backbone_parameter_role(name)
            if role == "unknown":
                unknown.append(name)
                continue
            # Preserve the usual no-decay treatment for gains, biases, and scalar gates.
            use_decay = name in decay_names and parameter.ndim >= 2
            buckets[(role, use_decay)].append(parameter)
        if unknown:
            preview = ", ".join(unknown[:8])
            raise ValueError(
                "Differential AdamW found unclassified trainable backbone parameters: "
                f"{preview}"
            )

        grouped_parameters = []
        for (role, use_decay), parameters in buckets.items():
            if not parameters:
                continue
            grouped_parameters.append(
                {
                    "params": parameters,
                    "lr": (
                        self.concept_memory_lr
                        if role == "concept_memory"
                        else self.args.learning_rate
                    ),
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                    "group_name": f"{role}_{'decay' if use_decay else 'no_decay'}",
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        self.optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
        logger.info(
            "Differential AdamW: LoRA LR=%g, concept-memory LR=%g, groups=%s",
            self.args.learning_rate,
            self.concept_memory_lr,
            [
                (group["group_name"], len(group["params"]), group["weight_decay"])
                for group in grouped_parameters
            ],
        )
        return self.optimizer

    def _anchor_mse(
        self,
        base_model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        concept_repr: torch.Tensor,
    ) -> torch.Tensor:
        target_mask = labels != -100
        with torch.no_grad():
            teacher_hidden = self.anchor_teacher(
                input_ids=input_ids,
                attention_mask=target_mask.long(),
            ).last_hidden_state
        return base_model.compute_anchor_loss(
            concept_repr,
            teacher_hidden,
            target_mask,
            standardize=self.anchor_standardize,
        )

    def _anchor_compute_loss(self, base_model, inputs, return_outputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        task_loss, logits, encoder_outputs = base_model.encode_decode_loss(
            input_ids,
            attention_mask,
            input_ids,
            labels,
            target_attention_mask=attention_mask,
        )
        concept_repr = encoder_outputs.last_hidden_state
        anchor_mse = self._anchor_mse(base_model, input_ids, labels, concept_repr)
        total_loss = task_loss + self.anchor_loss_weight * anchor_mse

        outputs = MaskedLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def get_eval_dataloader(self, eval_dataset=None):
        """Use a separate seeded collator for deterministic evaluation."""
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    @torch.no_grad()
    def _concept_ablation_metrics(self) -> dict:
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(base_model, "concept_ablation_ce"):
            return {}
        window_k = getattr(base_model.config, "decoder_context_window", None)
        dataloader = self.get_eval_dataloader()
        device = self.args.device
        sums: dict = {}
        n = 0
        rank_metrics: dict = {}
        anchor_sum = 0.0
        anchor_n = 0
        for i, batch in enumerate(dataloader):
            if i >= self.concept_ablation_batches:
                break
            labels = batch["labels"].to(device)
            if "prefix_input_ids" in batch:
                prefix_attention_mask = batch.get("prefix_attention_mask")
                if prefix_attention_mask is not None:
                    prefix_attention_mask = prefix_attention_mask.to(device)
                encoder_input_ids = batch["prefix_input_ids"].to(device)
                encoder_attention_mask = prefix_attention_mask
                metrics = base_model.concept_ablation_ce(
                    prefix_input_ids=encoder_input_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_input_ids=batch["suffix_input_ids"].to(device),
                    labels=labels,
                    window_k=window_k,
                )
            else:
                encoder_input_ids = batch["input_ids"].to(device)
                encoder_attention_mask = batch.get("attention_mask")
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.to(device)
                metrics = base_model.concept_ablation_ce(
                    encoder_input_ids,
                    encoder_attention_mask,
                    labels,
                    window_k=window_k,
                )
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value
            if (
                self.anchor_loss
                and self.anchor_teacher is not None
                and "prefix_input_ids" not in batch
            ):
                concepts_eval = base_model.encode_concepts(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True,
                ).last_hidden_state
                anchor_sum += self._anchor_mse(
                    base_model,
                    encoder_input_ids,
                    labels,
                    concepts_eval,
                ).item()
                anchor_n += 1
            if not rank_metrics:
                rank_metrics = self._concept_effective_rank(
                    base_model,
                    encoder_input_ids,
                    encoder_attention_mask,
                )
            n += 1
        if n == 0:
            return {}
        out = {f"concept_ablation/{name}": value / n for name, value in sums.items()}
        out.update(rank_metrics)
        if hasattr(base_model, "concept_gate_metrics"):
            out.update(base_model.concept_gate_metrics())
        if anchor_n > 0:
            out["anchor/mse_eval"] = anchor_sum / anchor_n
        return out

    @torch.no_grad()
    def _concept_effective_rank(self, base_model, input_ids, attention_mask) -> dict:
        try:
            concepts = base_model.encode_concepts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state.float()
            concept_mean = concepts.mean(dim=0)
            singular_values = torch.linalg.svdvals(concept_mean)
            effective_rank = (
                singular_values.sum() / (singular_values.max() + 1e-8)
            ).item()
            max_rank = min(concept_mean.shape)
            metrics = {
                "concept_geometry/effective_rank": effective_rank,
                "concept_geometry/effective_rank_normalized": effective_rank / max_rank,
            }
            from analysis.concept_analysis import compute_within_sample_concept_rank

            within = compute_within_sample_concept_rank(concepts)
            metrics.update(
                {
                    f"concept_geometry/{name}": value
                    for name, value in within.items()
                }
            )
            return metrics
        except Exception as exc:
            if getattr(base_model.config, "checkpoint_family", None) == "backbone_concept":
                raise RuntimeError(
                    "E10 within-sample RankMe is a registered kill gate and failed to compute."
                ) from exc
            logger.warning(f"Concept geometry probe failed: {exc}")
            return {}

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.compute_concept_ablation and is_main_process():
            ablation = self._concept_ablation_metrics()
            if ablation:
                metrics.update(ablation)
                self.log(ablation)
        return metrics

    def _contrastive_loss(
        self,
        model: ConceptEncoderForDenoisingPerceiver,
        concept_repr_a: torch.Tensor,
        concept_repr_b: torch.Tensor,
    ) -> torch.Tensor:
        pooled_a = F.normalize(model.pool_concepts(concept_repr_a), dim=-1)
        pooled_b = F.normalize(model.pool_concepts(concept_repr_b), dim=-1)
        similarity = pooled_a @ pooled_b.T / self.contrastive_temperature
        labels = torch.arange(similarity.size(0), device=similarity.device)
        return (
            F.cross_entropy(similarity, labels)
            + F.cross_entropy(similarity.T, labels)
        ) / 2.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if not model.training or (
            self.objective_variant
            in {OBJECTIVE_RECONSTRUCTION, OBJECTIVE_PREFIX_SUFFIX, OBJECTIVE_CAUSAL_LM}
            and not self.anchor_loss
        ):
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        base_model = model.module if hasattr(model, "module") else model
        if self.anchor_loss:
            return self._anchor_compute_loss(base_model, inputs, return_outputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        encoder_outputs_a = base_model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        concept_repr_a = encoder_outputs_a.last_hidden_state
        decoder_output = base_model.decode_from_concepts(
            concept_repr_a,
            seq_length=input_ids.size(1),
        )
        logits, task_loss = base_model.reconstruction_loss(decoder_output, labels)

        if base_model.loss_manager.is_enabled:
            total_loss = base_model.loss_manager(
                task_loss=task_loss,
                concept_repr=concept_repr_a,
            )
        else:
            total_loss = task_loss

        encoder_outputs_b = base_model.encode_concepts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        contrastive_loss = self._contrastive_loss(
            base_model,
            concept_repr_a=concept_repr_a,
            concept_repr_b=encoder_outputs_b.last_hidden_state,
        )
        total_loss = total_loss + self.contrastive_weight * contrastive_loss

        outputs = MaskedLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=encoder_outputs_a.hidden_states,
            attentions=encoder_outputs_a.attentions,
        )
        return (total_loss, outputs) if return_outputs else total_loss

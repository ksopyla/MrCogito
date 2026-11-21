"""Wall-clock budget for one trainer invocation (--max_train_hours).

When the budget runs out the callback asks the Trainer to save a checkpoint at the current step and stop; the
entrypoint then skips the final save and exits with code 3, and the caller resumes with --resume_from_checkpoint.
Used for heat-limited bursts on Polonez and for machines that can be interrupted. The step counter, optimizer,
scheduler and data position all come back from the checkpoint, so a resumed run follows the same schedule.

Multi-GPU: every rank must stop at the same step (the checkpoint save is collective), so the ranks agree on the
decision with one tiny all-reduce per optimizer step: if any rank is past its deadline, all stop.
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback


class StopAfterHoursCallback(TrainerCallback):
    def __init__(self, hours: float):
        self.hours = float(hours)
        self.deadline = time.time() + self.hours * 3600.0
        self.stopped = False

    def _any_rank_past_deadline(self) -> bool:
        hit = time.time() >= self.deadline
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return hit
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        flag = torch.tensor([1.0 if hit else 0.0], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stopped and state.global_step < state.max_steps and self._any_rank_past_deadline():
            control.should_save = True
            control.should_training_stop = True
            self.stopped = True
        return control

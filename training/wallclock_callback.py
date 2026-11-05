"""Wall-clock budget for one trainer invocation (--max_train_hours).

When the budget runs out the callback asks the Trainer to save a checkpoint at the current step and stop; the
entrypoint then skips the final save and exits with code 3, and the caller resumes with --resume_from_checkpoint.
Used for heat-limited bursts on Polonez and for machines that can be interrupted. The step counter, optimizer,
scheduler and data position all come back from the checkpoint, so a resumed run follows the same schedule.
"""
from __future__ import annotations

import time

from transformers import TrainerCallback


class StopAfterHoursCallback(TrainerCallback):
    def __init__(self, hours: float):
        self.hours = float(hours)
        self.deadline = time.time() + self.hours * 3600.0
        self.stopped = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stopped and time.time() >= self.deadline and state.global_step < state.max_steps:
            control.should_save = True
            control.should_training_stop = True
            self.stopped = True
        return control

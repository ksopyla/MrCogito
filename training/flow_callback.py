"""E33p instrumentation callbacks (plan §8): training dynamics every N optimizer steps, and weight snapshots.

`FlowCallback` arms `model.flow_recorder` (nn/flow_stats.FlowRecorder) for the first micro-batch of every N-th step,
then, after gradient clipping, adds per-group gradient norms and, after the optimizer step, per-group update ratios
‖ΔW‖ / ‖W‖ (from a copy of the weights taken just before the step, on log steps only). The trainer's `log()` merges
the pending numbers into its next log line (`pop_flow`). `SnapshotCallback` saves the model weights every N steps to
`<output_dir>/snapshots/step_N/model.safetensors` (for evaluation/flow_probe.py), independent of the rotating
checkpoints.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import torch
from transformers import TrainerCallback

from nn.flow_stats import FlowRecorder

_LAYER = re.compile(r"^layers\.(\d+)\.(attn|mlp)\.")


def param_group(name: str) -> str:
    """layers.{i}.attn.* → L{i}.attn, layers.{i}.mlp.* → L{i}.ffn, memory_writer.* → writer, lm_head / final_norm → head,
    embed.* → embed, everything else → its top-level module name."""
    if name.startswith("module."):
        name = name[len("module."):]
    m = _LAYER.match(name)
    if m:
        return f"L{m.group(1)}.{'attn' if m.group(2) == 'attn' else 'ffn'}"
    top = name.split(".", 1)[0]
    return {"memory_writer": "writer", "lm_head": "head", "final_norm": "head"}.get(top, top)


def _base(model):
    return model.module if hasattr(model, "module") else model


class FlowCallback(TrainerCallback):
    def __init__(self, every: int, recorder: FlowRecorder | None = None):
        self.every = int(every)
        self.rec = recorder or FlowRecorder()
        self.pending: dict[str, float] = {}
        self._on = False
        self._before: dict[str, torch.Tensor] = {}

    def _groups(self, model):
        g = defaultdict(list)
        for n, p in _base(model).named_parameters():
            if p.requires_grad:
                g[param_group(n)].append((n, p))
        return g

    def on_step_begin(self, args, state, control, model=None, **kw):
        self._on = self.every > 0 and state.global_step % self.every == 0 and model is not None
        if self._on:
            m = _base(model)
            m.flow_recorder = self.rec
            self.rec.armed = True

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
        if not self._on:
            return
        m = _base(model)
        self.rec.armed = False
        out = self.rec.finish(m)
        keys, vals = [], []
        for g, ps in self._groups(model).items():
            gs = [p.grad.detach().float().pow(2).sum() for _, p in ps if p.grad is not None]
            if gs:
                keys.append(f"grad/{g}/norm")
                vals.append(torch.stack(gs).sum().sqrt())
            for n, p in ps:
                self._before[n] = p.detach().clone()
        if vals:
            out.update({k: float(v) for k, v in zip(keys, torch.stack(vals).tolist())})
        self.pending.update(out)

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, model=None, **kw):
        if not self._on:
            return
        keys, vals = [], []
        for g, ps in self._groups(model).items():
            d = [(p.detach().float() - self._before[n].float()).pow(2).sum() for n, p in ps if n in self._before]
            w = [self._before[n].float().pow(2).sum() for n, p in ps if n in self._before]
            if d:
                keys.append(f"upd/{g}/ratio")
                vals.append(torch.stack(d).sum().sqrt() / torch.stack(w).sum().sqrt().clamp_min(1e-12))
        if vals:
            self.pending.update({k: float(v) for k, v in zip(keys, torch.stack(vals).tolist())})
        self._before = {}
        self._on = False

    def pop_flow(self) -> dict[str, float]:
        out, self.pending = self.pending, {}
        return out


class SnapshotCallback(TrainerCallback):
    """Model weights only, every `every` optimizer steps (and never step 0), on the main process."""

    def __init__(self, every: int):
        self.every = int(every)

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.every <= 0 or model is None or state.global_step == 0 or state.global_step % self.every:
            return
        if not state.is_world_process_zero:
            return
        from safetensors.torch import save_file

        out = os.path.join(args.output_dir, "snapshots", f"step_{state.global_step}")
        os.makedirs(out, exist_ok=True)
        sd = {k: v.detach().contiguous().cpu() for k, v in _base(model).state_dict().items()}
        tied = _base(model).config.tie_word_embeddings
        if tied and "lm_head.weight" in sd:
            sd.pop("lm_head.weight")
        save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
        _base(model).config.to_json_file(os.path.join(out, "config.json"))

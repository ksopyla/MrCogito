"""Family-aware loading for the from-scratch Perceiver LM checkpoints (E18 `perceiver_ar`,
E22 `perceiver_concept`). The eval layer (long-context probes, lm-eval adapter, health check)
calls `load_perceiver_lm` so one runner serves both families."""
from __future__ import annotations

import json
from pathlib import Path

import torch

FAMILIES = ("perceiver_ar", "perceiver_concept")


def checkpoint_family(checkpoint: str) -> str:
    cfg = json.loads((Path(checkpoint) / "config.json").read_text())
    fam = cfg.get("checkpoint_family") or cfg.get("model_type") or "perceiver_ar"
    if fam.startswith("perceiver_concept"):
        return "perceiver_concept"
    return "perceiver_ar"


def load_perceiver_lm(checkpoint: str, device: str, attn_backend: str | None = None, dtype=None):
    """Instantiate the right family from `checkpoint`, move to `device`, eval mode. bf16 on CUDA
    unless `dtype` is given."""
    fam = checkpoint_family(checkpoint)
    if fam == "perceiver_concept":
        from nn.perceiver_concept_lm import PerceiverConceptConfig as Cfg, PerceiverConceptLM as Model
    else:
        from nn.perceiver_ar_lm import PerceiverARConfig as Cfg, PerceiverARLM as Model
    cfg = Cfg.from_pretrained(checkpoint)
    if attn_backend:
        cfg.attn_backend = attn_backend
    model = Model.from_pretrained(checkpoint, config=cfg)
    model.to(device).eval()
    if dtype is not None:
        model.to(dtype)
    elif str(device).startswith("cuda"):
        model.to(torch.bfloat16)
    return model

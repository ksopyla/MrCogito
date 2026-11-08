"""E33p T2: golden logits of the tiny window / dense / e33 / e33n models, saved before the E33p code changed the
reading layer (`python -m tests.e33p_golden` at the commit before E33p's first code change). The test in
tests/test_note_open.py rebuilds the same models and asserts bit-identical logits and the same state-dict keys."""
from __future__ import annotations

from pathlib import Path

import torch

from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg, lm_cfg
from tests.test_text_memory import TEXT

GOLDEN = Path(__file__).with_name("data") / "e33p_golden.pt"
LOOP = dict(message_loop_rounds=4, global_nope=True, message_raw_window=32)


def golden_cfgs() -> dict:
    return {
        "window": lm_cfg(**{**TEXT, **LOOP, "message_override": "none"}),
        "dense": cfg(message_boundary_token_id=-1, message_loop_rounds=4, global_nope=True),
        "e33": lm_cfg(**{**TEXT, **LOOP}),
        "e33n": lm_cfg(**{**TEXT, **LOOP, "lm_read_topk": 4}),
    }


def golden_rows() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(3, 80, (2, 160), generator=g)
    return ids


def golden_doc_ids() -> torch.Tensor:
    d = torch.zeros(2, 160, dtype=torch.long)
    d[1, 70:] = 1  # second row packs two documents
    return d


def build(name: str) -> PerceiverARLM:
    torch.manual_seed(0)
    m = PerceiverARLM(golden_cfgs()[name]).eval()
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def compute() -> dict:
    out = {}
    ids, doc = golden_rows(), golden_doc_ids()
    for name in golden_cfgs():
        m = build(name)
        with torch.no_grad():
            out[name] = {
                "logits": m(ids, doc_ids=doc).logits.clone(),
                "keys": sorted(m.state_dict().keys()),
                "n_params": sum(p.numel() for p in m.parameters()),
            }
    return out


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    torch.save(compute(), GOLDEN)
    print("saved", GOLDEN)

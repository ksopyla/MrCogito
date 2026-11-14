"""E37 stage 0: where a training step of the concept read spends its GPU time (torch.profiler, top kernels / ops).

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/profile_concept_read.py --arm e37a --size s30 --seq 16384

Builds the arm from scripts/run_e33n.py's model arguments (flex backend), runs a few bf16 forward + backward steps on
random tokens of one document, and prints the ops with the most CUDA time and the step time per arm.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build(arm: str, size: str):
    from scripts.run_e33n import EOT, VOCAB, fit
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    args, _ = fit(arm, size)
    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "flex", False
    tok = type("T", (), {"pad_token_id": EOT, "bos_token_id": EOT, "eos_token_id": EOT, "__len__": lambda self: VOCAB})()
    model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="gpt2"))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["e37a"])
    ap.add_argument("--size", default="s30")
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    for arm in a.arm:
        torch.manual_seed(0)
        m = build(arm, a.size).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        ids = torch.randint(100, 50000, (1, a.seq), device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for _ in range(2):  # compile and warm up
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        print(f"== {arm} {a.size} seq {a.seq}: {dt:.3f} s per step, {a.seq / dt:.0f} tokens/s")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,
                                                torch.profiler.ProfilerActivity.CPU]) as prof:
            step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.top, max_name_column_width=70))
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

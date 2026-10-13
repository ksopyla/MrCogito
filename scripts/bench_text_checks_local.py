#!/usr/bin/env python
"""Training throughput of the text-check models on this machine (Apple MPS / CPU / CUDA).

Times forward + backward + AdamW step on random token rows, after a warm-up, for each model at a
given width and training length, so a local tier can be sized from measured tokens per second.

  uv run python scripts/bench_text_checks_local.py --hidden 256 384 --seq_len 1024 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from evaluation.text_checks import ARCHES, VOCAB, _base, _E31C_SMOKE, _round64  # noqa: E402


def build(arch: str, hidden: int, pre: int, stack: int, device: str, small_notebook: bool):
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    t = SimpleNamespace(hidden=hidden, pre_layers=pre, global_layers=1, stack_layers=stack)
    args = {**_base(t), **ARCHES[arch].args}
    if small_notebook and "lm_latents" in args:
        args.update(_E31C_SMOKE)
    args["num_attention_heads"] = hidden // args["head_dim"]
    args["intermediate_size"] = _round64(hidden * 8 / 3)
    args.update(json.loads(os.environ.get("BENCH_OVERRIDES", "{}")))  # diagnostics: e.g. {"logit_softcap": 0}
    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "sdpa", False
    tok = type("T", (), {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "__len__": lambda self: VOCAB})()
    model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=8192, tokenizer_name="bench"))
    return model.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arches", nargs="+", default=["dense", "local", "e31c"])
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 384])
    p.add_argument("--pre_layers", type=int, default=2)
    p.add_argument("--stack_layers", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--small_notebook", action="store_true", help="the smoke tier's small notebook")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    for h in a.hidden:
        for arch in a.arches:
            torch.manual_seed(0)
            model = build(arch, h, a.pre_layers, a.stack_layers, a.device, a.small_notebook)
            n = sum(q.numel() for q in model.parameters() if q.requires_grad)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randint(3, VOCAB, (a.batch, a.seq_len), device=a.device)
            ctx = torch.autocast(a.device, dtype=torch.bfloat16) if a.device != "cpu" else torch.autocast("cpu", enabled=False)

            def step():
                with ctx:
                    out = model(input_ids=x, labels=x)
                loss = out.loss if hasattr(out, "loss") else out[0]
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                return loss

            try:
                for _ in range(2):
                    step()
                if a.device == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
                for _ in range(a.steps):
                    loss = step()
                float(loss)
                if a.device == "mps":
                    torch.mps.synchronize()
                dt = (time.time() - t0) / a.steps
                tps = a.batch * a.seq_len / dt
                print(f"{arch:<10} hidden {h:>4} params {n / 1e6:6.2f}M seq {a.seq_len} batch {a.batch}: "
                      f"{dt:.3f} s/step, {tps:,.0f} tokens/s, 100M tokens in {1e8 / tps / 3600:.1f} h", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{arch:<10} hidden {h:>4}: failed {type(e).__name__}: {str(e)[:200]}", flush=True)
            del model, opt
            if a.device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()

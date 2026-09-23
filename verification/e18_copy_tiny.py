"""E18 gate P2 de-risk: can a tiny Perceiver AR v2 learn mirrored copy at all (CPU, minutes)?

The 32k copy runs on Polonez sat at the uniform floor ln(256) for 500 steps. This script trains
tiny perceiver / dense models on the same task construction (scripts/build_copy_task_dataset.py:
[BOS] a mirror(a) [EOS], loss on the mirrored half) at a short context, where the answer is
known within minutes. If the tiny model learns it, the 32k failure is optimisation / scale /
curriculum; if it does not, the family has a copy problem.

    uv run python verification/e18_copy_tiny.py --context 130 --steps 600
    uv run python verification/e18_copy_tiny.py --mode dense
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM


def make_batch(rng, batch, context, vocab_lo, vocab_hi, bos, eos, task="mirror"):
    half = (context - 2) // 2
    a = rng.integers(vocab_lo, vocab_hi, size=(batch, half))
    mirrored = a[:, ::-1] if task == "mirror" else a  # "copy" = plain forward copy (fixed offset)
    ids = np.concatenate([np.full((batch, 1), bos), a, mirrored, np.full((batch, 1), eos)], axis=1)
    labels = np.concatenate([np.full((batch, 1 + half), -100), mirrored, np.full((batch, 1), eos)], axis=1)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["perceiver", "dense"], default="perceiver")
    p.add_argument("--context", type=int, default=130)
    p.add_argument("--block", type=int, default=32, help="stack window (perceiver mode)")
    p.add_argument("--pre_window", type=int, default=16)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--stack_layers", type=int, default=2)
    p.add_argument("--pre_layers", type=int, default=1)
    p.add_argument("--global_layers", type=int, default=1)
    p.add_argument("--value_embed_layers", default="0", help="comma list of layer indices with value embeddings")
    p.add_argument("--vocab_slice", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qk_gain", type=float, default=1.0,
                   help="init of the q_norm/k_norm RMSNorm gains (QK-norm bounds |logit| <= gain^2*sqrt(dh))")
    p.add_argument("--no_qk_norm", action="store_true", help="replace q/k RMSNorm by identity")
    p.add_argument("--task", choices=["mirror", "copy"], default="mirror")
    p.add_argument("--sink", action="store_true", help="swa_sink: windowed layers also attend to token 0")
    p.add_argument("--global_positions", default="", help="comma list: absolute layer index of each global read")
    p.add_argument("--global_nope", action="store_true", help="global read without RoPE (content-only)")
    p.add_argument("--global_logit_scale", choices=["none", "log"], default="none", help="SSMax-style log(n) query scale on the global read")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    V = 512
    bos, eos, lo, hi = 1, 2, 100, 100 + args.vocab_slice
    cfg = PerceiverARConfig(
        vocab_size=V, hidden_size=args.hidden, intermediate_size=args.hidden * 3, token_embedding_dim=32,
        par_mode=args.mode, pre_layers=args.pre_layers, pre_window=args.pre_window, global_layers=args.global_layers,
        stack_layers=args.stack_layers, block=args.block, num_attention_heads=4, num_kv_heads=2,
        head_dim=32, ngram_buckets=1024, value_embed_dim=16,
        value_embed_layers=tuple(int(x) for x in args.value_embed_layers.split(",") if x != ""),
        use_liger=False, attn_backend="sdpa", attn_pad_multiple=1, chunked_ce_block_size=64,
        pad_token_id=0, bos_token_id=bos, eos_token_id=eos, swa_sink=args.sink,
        global_positions=tuple(int(x) for x in args.global_positions.split(",") if x != "") or None,
        global_nope=args.global_nope, global_logit_scale=args.global_logit_scale, global_scale_ref=args.context,
    )
    model = PerceiverARLM(cfg)
    for layer in model.layers:
        if args.no_qk_norm:
            layer.attn.q_norm = torch.nn.Identity()
            layer.attn.k_norm = torch.nn.Identity()
        elif args.qk_gain != 1.0:
            with torch.no_grad():
                layer.attn.q_norm.weight.fill_(args.qk_gain)
                layer.attn.k_norm.weight.fill_(args.qk_gain)
    n_params = sum(p.numel() for p in model.parameters())
    if args.optimizer == "muon":
        from nn.muon import Muon

        opt = Muon(model.parameters(), lr=args.lr, momentum=0.95, adamw_lr=args.lr / 10, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    warmup = 50
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    floor = math.log(args.vocab_slice)
    print(f"task={args.task} mode={args.mode} params={n_params/1e6:.2f}M context={args.context} block={args.block} "
          f"pre={args.pre_layers}@{args.pre_window} global={args.global_layers} stack={args.stack_layers} ve={args.value_embed_layers} "
          f"uniform-floor={floor:.3f} opt={args.optimizer} lr={args.lr} qk_gain={args.qk_gain} no_qk_norm={args.no_qk_norm} sink={args.sink} "
          f"gpos={args.global_positions or 'default'} gnope={args.global_nope} gscale={args.global_logit_scale}")
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        ids, labels = make_batch(rng, args.batch, args.context, lo, hi, bos, eos, args.task)
        loss = model(input_ids=ids, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                ids, labels = make_batch(np.random.default_rng(999), 64, args.context, lo, hi, bos, eos, args.task)
                _, per, valid = model(input_ids=ids, labels=labels, return_per_token_loss=True)
                h = model.hidden_states(ids)[:, :-1]
                pred = torch.nn.functional.linear(h, model.lm_head.weight).argmax(-1)
                tgt = labels[:, 1:]
                m = tgt != -100
                acc = (pred[m] == tgt[m]).float().mean().item()
                # accuracy split: targets whose source lies inside vs outside the stack window
                half = (args.context - 2) // 2
                pos = torch.arange(tgt.shape[1])[None].expand_as(tgt)
                dist = (2 * (pos - half) + 1) if args.task == "mirror" else torch.full_like(pos, half)
                near = m & (dist < args.block)
                far = m & (dist >= args.block)
                acc_near = (pred[near] == tgt[near]).float().mean().item() if near.any() else float("nan")
                acc_far = (pred[far] == tgt[far]).float().mean().item() if far.any() else float("nan")
            model.train()
            print(f"step {step:4d} loss {loss.item():.3f} eval_ce {per[valid].mean().item():.3f} "
                  f"acc {acc:.3f} (near {acc_near:.3f} / far {acc_far:.3f}) {time.time()-t0:.0f}s")
    print("RESULT", "learned" if acc > 0.99 else ("partial" if acc > 0.2 else "floor"), f"acc={acc:.3f}")


if __name__ == "__main__":
    main()

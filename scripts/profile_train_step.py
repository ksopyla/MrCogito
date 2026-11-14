"""Where a training step's time goes, for one arm × size of scripts/run_e33n.py on one GPU (2026-11-14).

Builds the model through the trainer's own factory, reads real rows from a bin manifest (the trunk's 4k rows with
packed documents, or the decay's 16k rows), and times forward + backward + optimizer step under bf16 autocast:
  * tokens per second of the bare step (no Trainer, no dataloader) — the ceiling the trainer can reach;
  * the forward's wall time split by module type (attention of the reading layer / local attention / MLP / loss),
    from forward hooks with a CUDA synchronize around each call (so the split run is slower than the timed run);
  * optionally the top CUDA ops (torch.profiler).

Usage: CUDA_VISIBLE_DEVICES=0 uv run python scripts/profile_train_step.py --arm e37a --size m100 --phase trunk \
           --data DATA --rows 2 [--liger] [--steps 12] [--ops 25]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e33n import EOT, VOCAB, fit, manifests  # noqa: E402


def build(arm: str, size: str, liger: bool, backend: str):
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    args, n = fit(arm, size)
    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger, ma.attn_pad_multiple = backend, liger, 1
    ma.chunked_ce_block_size = 2048
    tok = type("T", (), {"pad_token_id": EOT, "bos_token_id": EOT, "eos_token_id": EOT, "__len__": lambda s: VOCAB})()
    model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="gpt2"))
    return model, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--size", default="m100")
    ap.add_argument("--phase", default="trunk", choices=("trunk", "decay", "long"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=2, help="rows per forward (the micro-batch)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--ops", type=int, default=0, help="also print this many top CUDA ops (torch.profiler)")
    a = ap.parse_args()
    from data.bin_token_dataset import load_bin_manifest

    dev = "cuda"
    with tempfile.TemporaryDirectory() as tmp:
        man = manifests(Path(a.data), Path(tmp), a.size)
        train, _, m = load_bin_manifest(str(man[a.phase]))
    S = int(m["seq_len"])
    model, n = build(a.arm, a.size, a.liger, a.backend)
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    def batch(i):
        items = [train[i * a.rows + j] for j in range(a.rows)]
        ids = torch.stack([x["input_ids"] for x in items]).to(dev)
        doc = torch.stack([x["doc_ids"] for x in items]).to(dev)
        labels = ids.clone()
        labels[:, 1:][doc[:, 1:] != doc[:, :-1]] = -100  # the collator's rule: a document's first token is no target
        return ids, doc, labels

    def step(i):
        ids, doc, labels = batch(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, doc_ids=doc, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss)

    for i in range(3):  # warm-up: kernels compiled, allocator settled
        step(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(3, 3 + a.steps):
        loss = step(i)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.steps
    out = dict(arm=a.arm, size=a.size, phase=a.phase, rows=a.rows, seq=S, liger=a.liger, non_embedding_params=n,
               step_s=round(dt, 4), tokens_per_s=round(a.rows * S / dt), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
               loss=round(loss, 4))

    # forward split by module type (synchronised hooks)
    gi = model.config.global_layer_index
    acc, t_in = defaultdict(float), {}

    def pre(name):
        def f(mod, args):
            torch.cuda.synchronize()
            t_in[name] = time.time()
        return f

    def post(name, group):
        def f(mod, args, res):
            torch.cuda.synchronize()
            acc[group] += time.time() - t_in[name]
        return f

    hs = []
    for li, blk in enumerate(model.layers):
        g = "reading-layer attention" if li == gi else "local attention"
        hs += [blk.attn.register_forward_pre_hook(pre(f"a{li}")), blk.attn.register_forward_hook(post(f"a{li}", g)),
               blk.mlp.register_forward_pre_hook(pre(f"m{li}")), blk.mlp.register_forward_hook(post(f"m{li}", "MLP"))]
    tf = tb = 0.0
    for i in range(20, 24):
        ids, doc, labels = batch(i)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_t = model(input_ids=ids, doc_ids=doc, labels=labels).loss
        torch.cuda.synchronize()
        t1 = time.time()
        loss_t.backward()
        torch.cuda.synchronize()
        tf, tb = tf + t1 - t0, tb + time.time() - t1
        opt.zero_grad(set_to_none=True)
    for h in hs:
        h.remove()
    k = 4
    split = {g: round(v / k, 4) for g, v in acc.items()}
    split["loss, embeddings, masks, the rest"] = round(tf / k - sum(acc.values()) / k, 4)
    out.update(forward_s=round(tf / k, 4), backward_s=round(tb / k, 4), forward_split_s=split)
    print(json.dumps(out, indent=1))
    if a.ops:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(30, 32):
                step(i)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.ops, max_name_column_width=60))


if __name__ == "__main__":
    main()

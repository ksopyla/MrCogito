"""E33n runner: the four arms × sizes × phases of the LSA-shaped protocol, as trainer commands (one GPU per run).

Spec: docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md · plan: E33n_plan.md.

  arms   window  — reading layer sees only its last 512 raw tokens (E31c writer built, notes switched off)
         dense   — reading layer: full causal attention over every earlier raw token (E18-style raw global read)
         e33     — E33: last 512 raw tokens + every note of every closed 256-token page (full note read)
         e33n    — E33n: last 512 raw tokens + the 64 best notes per query and head (top-k note read)
         notes_open / pages_open — E33p: tied note keys; each query opens the raw tokens of its 64 best notes'
                   members (its 2 best pages), ≤ 512 tokens, plus every note as gist and the 512-token window
         lsa     — E33p: LSA's released flat read on the dense arm (sinks + window + 512 keys from its best buckets)
         catalogue / catalogue_2l — E33t: the notes only choose what opens (4 learned empty slots, no summaries in the
                   read), balanced filing (≤ 16 per note), max-head cosine, up to 128 notes until 512 tokens, a new
                   place per loop pass, an index loss; pages of 256 without overlap; _2l searches pages first
         Note (2026-11-06): before then the trainer passed the loop only to notebook models, so E33n's dense arm
         trained with one round; it now loops 4 times like every other arm.
         All: E33's loop (prelude → [reading layer + 1 local layer] × 4 tied → coda), exits off, local windows 128,
         plain input (no n-gram tables / value embeddings), GPT-2 vocabulary (50,257), tied factorised embedding.
  sizes  s30 (30M) · m100 (100M) · l300 (300M) — FFN width fitted to the target with the trainer's own factory.
  phases trunk — FineWeb-Edu GPT-2 shards, rows of 4,096 (documents packed, no cross-document attention),
                 warmup then constant learning rate (the WSD "stable" part; LSA branches its decays from it);
         decay — from the trunk's final weights, fresh optimizer, LongDocs-lite rows of 16,384 (one document per
                 row), warmup 3 % then 1−sqrt decay over 97 % to 10 % of the peak (LSA's 32k decay shape).

Usage (on a node, from the repo root):
  uv run python scripts/run_e33n.py cmd --arm e33n --size s30 --phase trunk --data /opt/dlami/nvme/data --out OUT
  uv run python scripts/run_e33n.py params            # parameter counts of every arm × size
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOCAB = 50257
EOT = 50256

SIZES = {
    #        hidden  e    pre  stack  target   trunk tok  decay tok  trunk rows/step (×4k)  lr peak
    "s30":  dict(hidden=576,  e=256, pre=2, stack=5, target=30e6,  trunk=1.5e9, decay=0.2e9, rows=64,  lr=1.0e-3),
    "m100": dict(hidden=896,  e=384, pre=3, stack=7, target=100e6, trunk=3.0e9, decay=0.4e9, rows=128, lr=6.4e-4),
    "l300": dict(hidden=1280, e=512, pre=4, stack=9, target=300e6, trunk=5.0e9, decay=0.6e9, rows=256, lr=4.5e-4),
    "tiny": dict(hidden=128,  e=64,  pre=1, stack=2, target=8e6,   trunk=2e6,   decay=1e6,   rows=4,   lr=1.0e-3),
}

_LOOP = {"message_loop_rounds": 4, "message_loop_exit_aux": 0.0}  # loop span 1 is the default
_WRITER = {
    "par_mode": "perceiver", "message_write": "latent_memory", "message_boundary_token_id": -1,
    "message_raw_window": 512, "lm_read": "closed", "lm_context": "page_bidir", "lm_window": 256, "lm_stride": 192,
    "lm_latents": 32, "lm_latent_dim": 512, "lm_heads": 8, "lm_writer_dim": 256, "lm_enc_layers": 2, "lm_rounds": 2,
    "lm_reader_tokens": 1, "lm_addr": "none", "lm_slot_pos": "reader",
}
ARMS = {
    "window": {**_WRITER, **_LOOP, "message_override": "none"},
    "dense": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP},
    "e33": {**_WRITER, **_LOOP},
    "e33n": {**_WRITER, **_LOOP, "lm_read_topk": 64},
    # E33p (docs/experiments_specs/ahead/E33p_note_indexed_exact_read.md): the notes as an index, per-query choice
    "notes_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 64, "lm_open_cap": 512},
    "pages_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "pages", "lm_open_k": 2, "lm_open_cap": 512},
    # LSA's released flat read on the dense arm's reading layer (nn/lsa_buckets.py), budget matched at 512 keys
    "lsa": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "lsa",
            "lsa_window": 512, "lsa_sinks": 4, "lsa_block": 512, "lsa_bucket": 8, "lsa_pool": 4, "lsa_bucket_cap": 16,
            "lsa_top_k": 512},
    # E33t (docs/experiments_specs/ahead/E33t_catalogue_read.md): the notes only choose what opens; balanced filing,
    # max-head cosine, budget filled by tokens (up to 128 notes), a new place per pass, learned empty slots, index loss
    "catalogue": {**_WRITER, **_LOOP, "lm_stride": 256, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 128,
                  "lm_open_cap": 512, "lm_note_read": "index", "lm_filing": "balanced", "lm_open_score": "max_cos",
                  "lm_open_new_per_pass": True, "lm_sinks": 4, "lm_index_loss": 0.1},
}
ARMS["catalogue_2l"] = {**ARMS["catalogue"], "lm_open_levels": 2, "lm_open_pages": 16}  # pages first, then notes
TINY_OPEN = {"notes_open": {"lm_open_k": 4, "lm_open_cap": 64}, "pages_open": {"lm_open_k": 1, "lm_open_cap": 64},
             "lsa": {"lsa_window": 64, "lsa_block": 64, "lsa_top_k": 64},
             "catalogue": {"lm_open_k": 16, "lm_open_cap": 64, "lm_index_queries": 32, "lm_index_negatives": 4,
                           "lm_stride": 64},
             "catalogue_2l": {"lm_open_k": 16, "lm_open_cap": 64, "lm_index_queries": 32, "lm_index_negatives": 4,
                              "lm_stride": 64, "lm_open_pages": 2}}
TINY_WRITER = {"lm_latents": 8, "lm_latent_dim": 64, "lm_heads": 2, "lm_writer_dim": 64, "lm_enc_layers": 1,
               "lm_rounds": 1, "message_raw_window": 64, "lm_window": 64, "lm_stride": 48}


def model_args(arm: str, size: str, ffn: int | None = None) -> dict:
    z = SIZES[size]
    a = {
        "model_family": "perceiver_ar", "objective_variant": "causal_lm", "decoder_type": "causal_ar",
        "hidden_size": z["hidden"], "token_embedding_dim": z["e"], "head_dim": 64 if z["hidden"] >= 256 else 32,
        "num_kv_heads": 1, "par_pre_layers": z["pre"], "par_pre_window": 128 if size != "tiny" else 32,
        "par_global_layers": 1, "num_hidden_layers": z["stack"], "par_block": 128 if size != "tiny" else 32,
        "par_global_nope": "true", "par_ngram_orders": "", "par_value_embed_layers": "",
        "logit_softcap": 30.0, "z_loss": 1e-4,
    }
    a.update(ARMS[arm])
    if size == "tiny" and "lm_latents" in a:
        a.update(TINY_WRITER)
        if "lm_read_topk" in a:
            a["lm_read_topk"] = 4
    if size == "tiny" and arm in TINY_OPEN:
        a.update(TINY_OPEN[arm])
    a["num_attention_heads"] = z["hidden"] // a["head_dim"]
    a["intermediate_size"] = ffn or max(64, round(z["hidden"] * 8 / 3 / 64) * 64)
    return a


def count(args: dict, non_embedding: bool = True) -> int:
    """Trainable parameters through the trainer's own factory (meta device). Sizes are named by the NON-embedding
    count (Kaplan et al. 2020's convention): the GPT-2 vocabulary puts ~V·(e + H) parameters in the input table and
    the untied output layer (29M + 13M at width 576), which would otherwise be most of a "30M" model."""
    import torch
    from types import SimpleNamespace

    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "sdpa", False
    tok = type("T", (), {"pad_token_id": EOT, "bos_token_id": EOT, "eos_token_id": EOT, "__len__": lambda self: VOCAB})()
    with torch.device("meta"):
        model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="gpt2"))
    skip = ("embed.", "lm_head.") if non_embedding else ()
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith(skip))


def fit(arm: str, size: str) -> tuple[dict, int]:
    """FFN width (multiple of 64) bringing the WINDOW arm's non-embedding count closest to the target; every arm of a
    size uses that width, so arms differ only in the reading layer (the window arm carries the writer too)."""
    z = SIZES[size]
    lo, hi, best = 64, z["hidden"] * 12, None
    while lo <= hi:
        mid = max(64, round((lo + hi) / 2 / 64) * 64)
        n = count(model_args("window", size, mid))
        if best is None or abs(n - z["target"]) < abs(best[1] - z["target"]):
            best = (mid, n)
        lo, hi = (mid + 64, hi) if n < z["target"] else (lo, mid - 64)
    args = model_args(arm, size, best[0])
    return args, count(args)


def manifests(data: Path, out: Path, size: str) -> dict:
    """Write the two token-shard manifests (trunk / decay) for a run under `out`, return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    fw, ld = data / "fineweb_edu_gpt2", data / "longdocs_lite_gpt2"
    tiny = size == "tiny"
    trunk = {"seq_len": 512 if tiny else 4096, "rule": "stream", "eot": EOT, "seed": 42,
             "train": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_train_*.bin")], "weight": 1.0}],
             "eval": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_val_*.bin")], "rows": 256}]}
    srcs = json.loads((ld / "manifest.json").read_text())["sources"] if (ld / "manifest.json").exists() else {}
    decay = {"seq_len": 1024 if tiny else 16384, "rule": "doc", "eot": EOT, "seed": 1042,
             "train": [{"name": s, "files": [str(ld / f"{s}_train_*.bin")], "weight": 1.0} for s in srcs],
             "eval": [{"name": s, "files": [str(ld / f"{s}_val_*.bin")], "rows": 16} for s in srcs]}
    p = {"trunk": out / "manifest_trunk.json", "decay": out / "manifest_decay.json"}
    p["trunk"].write_text(json.dumps(trunk, indent=2))
    p["decay"].write_text(json.dumps(decay, indent=2))
    return p


def train_cmd(arm: str, size: str, phase: str, *, data: Path, out: Path, init: str | None = None,
              lr: float | None = None, tokens: float | None = None, micro: int | None = None, seed: int = 42,
              wandb: bool = False, eval_every: int = 0, backend: str = "flex", local: bool = False,
              flow: bool = False, gpus: int = 1) -> list[str]:
    z = SIZES[size]
    args, n = fit(arm, size)
    man = manifests(data, out, size)
    seq = json.loads(man[phase].read_text())["seq_len"]
    tokens = tokens or (z["trunk"] if phase == "trunk" else z["decay"])
    rows_per_step = z["rows"] * 4096 // seq  # the same tokens per step in both phases
    rows_per_step = max(1, rows_per_step)
    micro = micro or max(1, min(rows_per_step, (16 if phase == "trunk" else 4) if size == "s30" else
                                (8 if phase == "trunk" else 2)))
    # gpus > 1: one model on several GPUs (data parallel, torchrun); each optimizer step still sees rows_per_step
    # rows: micro × accum × gpus. When gpus does not divide the step (64 rows on 3 GPUs) the step is rounded down
    # and the step count rises to keep the same total tokens.
    gpus = max(1, int(gpus))
    accum = max(1, rows_per_step // (micro * gpus))
    if micro * accum * gpus != rows_per_step:
        import sys
        print(f"note: {micro}×{accum}×{gpus} = {micro * accum * gpus} rows per step instead of {rows_per_step}",
              file=sys.stderr)
    steps = max(1, int(round(tokens / (micro * accum * gpus * seq))))
    lr = lr or z["lr"]
    launch = (["torchrun", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else ["python"])
    cmd = ["uv", "run", *launch, "training/train_concept_pretraining.py"]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    if phase == "trunk":
        sched = ["--lr_scheduler_type", "constant_with_warmup", "--warmup_steps", str(max(1, int(0.01 * steps)))]
    else:
        sched = ["--lr_scheduler_type", "warmup_stable_decay", "--warmup_steps", str(max(1, int(0.03 * steps))),
                 "--lr_scheduler_kwargs", json.dumps({"num_decay_steps": int(0.97 * steps), "decay_type": "1-sqrt",
                                                      "min_lr_ratio": 0.1})]
    save_every = max(50, steps // 10)
    cmd += [
        "--attn_backend", backend, "--attn_pad_multiple", "1", "--use_liger", "False",
        "--bin_manifest", str(man[phase]), "--tokenizer_name", str(data / "tokenizer_gpt2"),
        "--max_seq_length", str(seq), "--batch_packing_mode", "none",
        "--per_device_train_batch_size", str(micro), "--per_device_eval_batch_size", str(max(1, micro // 2)),
        "--gradient_accumulation_steps", str(accum), "--learning_rate", str(lr), "--max_steps", str(steps),
        *sched, "--weight_decay", "0.1", "--adam_beta1", "0.9", "--adam_beta2", "0.95", "--max_grad_norm", "1.0",
        "--optim", "adamw_torch_fused", "--bf16", "True", "--logging_steps", "10",
        "--eval_strategy", "steps" if eval_every else "no", "--eval_steps", str(eval_every or save_every),
        "--save_strategy", "steps", "--save_steps", str(save_every), "--save_total_limit", "2",
        "--output_dir", str(out / phase), "--seed", str(seed), "--report_to", "wandb" if wandb else "none",
        "--overwrite_output_dir", "True", "--disable_tqdm", "True", "--dataloader_num_workers", "6",
        "--prediction_loss_only", "True", "--chunked_ce_block_size", "2048",
    ]
    if init:
        cmd += ["--model_name_or_path", init]
    if flow:  # E33p instrumentation: dynamics every 200 steps (decay: 50 of ~763) and weight snapshots
        cmd += ["--flow_every", "200" if phase == "trunk" else "50",
                "--snapshot_every", "1000" if phase == "trunk" else "500"]
    if local:  # laptop pipeline check (Apple GPU / CPU): reference attention, plain AdamW, no worker processes
        import torch

        swap = {"--attn_backend": "sdpa", "--optim": "adamw_torch", "--dataloader_num_workers": "0",
                "--bf16": "True" if torch.backends.mps.is_available() else "False"}
        cmd = [swap.get(cmd[i - 1], x) if i else x for i, x in enumerate(cmd)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("params")
    c = sub.add_parser("cmd")
    c.add_argument("--arm", required=True, choices=sorted(ARMS))
    c.add_argument("--size", required=True, choices=sorted(SIZES))
    c.add_argument("--phase", required=True, choices=("trunk", "decay"))
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--init")
    c.add_argument("--lr", type=float)
    c.add_argument("--tokens", type=float)
    c.add_argument("--micro", type=int)
    c.add_argument("--eval_every", type=int, default=0)
    c.add_argument("--backend", default="flex")
    c.add_argument("--wandb", action="store_true")
    c.add_argument("--local", action="store_true", help="laptop pipeline check (sdpa, adamw_torch, no workers)")
    c.add_argument("--flow", action="store_true", help="E33p instrumentation (flow/grad/upd/read/select/lens keys)")
    c.add_argument("--seed", type=int, default=42, help="trainer seed: initial weights and the order rows are drawn in")
    c.add_argument("--gpus", type=int, default=1, help="GPUs for one model (data parallel via torchrun)")
    a = ap.parse_args()
    if a.what == "params":
        for size in ("s30", "m100", "l300"):
            for arm in ARMS:
                args, n = fit(arm, size)
                print(f"{size:5} {arm:7} non-embedding {n/1e6:7.1f}M  total {count(args, False)/1e6:7.1f}M  "
                      f"ffn {args['intermediate_size']}")
        return
    cmd = train_cmd(a.arm, a.size, a.phase, data=Path(a.data), out=Path(a.out), init=a.init, lr=a.lr,
                    tokens=a.tokens, micro=a.micro, wandb=a.wandb, eval_every=a.eval_every, backend=a.backend,
                    local=a.local, flow=a.flow, seed=a.seed, gpus=a.gpus)
    print(shlex.join(cmd))


if __name__ == "__main__":
    main()

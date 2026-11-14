"""E37 stage B: the ORIGINAL Latent Sparse Attention model (Kubicek et al., arXiv 2610.42147) at ~160M, trained by THEIR
trainer (`lsa native train`'s path: torchtitan's Trainer from `lsa.training.native.trainer.trainer_config`) on OUR m100
data and token budget. Runs in their venv (their repo github.com/Prefrontal-AI/latent-sparse-attention @ 5ebea732d639
installed from its uv.lock: torch 2.13 cu129, torchtitan @56a721b, flash-linear-attention 0.5.2), one process per GPU
through torchrun. Nothing in their checkout is edited; three things are swapped in at run time, here:

  1. model — their preset `configs/training/native/qwen35_0_8b_gpt2/pretrain_fineweb_lsa.yaml` (lsa_mqa8 on the FineWeb
     trunk; 0.55B in the paper) with torchtitan's Qwen3.5 body shrunk to our m100's size: hidden 896, 12 layers
     (9 Gated DeltaNet + 3 attention: their three-to-one), Gated DeltaNet 7 key and 7 value heads of 128 (the kernel
     needs 128; key width = hidden, as in Qwen3.5-2B), SwiGLU 2240 (2.5 x hidden, chosen to land on our 162M), tied
     GPT-2 embeddings (50,304). Their GPT-2 transformation (`gpt2_model`: N(0, 0.02), depth-scaled residual writers,
     FLA's GDN decay init) and the attention layer are untouched: 8 query heads over 1 KV head of 128, no RoPE, output
     gate, QK-norm; `kind: lsa`, `coarse: false`, `fast_read: true`, window 512, block 512, pool 4, buckets of 8 (cap 16),
     4 sinks, top_k 1024 (the preset's; their 128k decays raise it to 2048, our 16k decay keeps 1024), max reduction
     over query heads.
  2. data — our shards through our BinRowDataset (data/bin_token_dataset.py, the manifests of scripts/run_e33n.py) in
     the order our HF trainer draws them (accelerate's seedable sampler: torch.randperm(N, seed 42), checked against
     transformers' Trainer.get_train_dataloader), so the token stream is the m100 runs' own. Positions restart at
     every document (torchtitan then cuts the Gated DeltaNet's document offsets there), labels that would predict the
     first token of the next document are ignored (our collator's rule), and a row's last token predicts nothing.
     Their LSA layer's fused read takes no document mask: inside a packed 4k trunk row it reads across documents,
     as in their own FineWeb trunk. (The decay rows and every eval row hold one document.) On several GPUs (torchrun,
     their FSDP over run.dp_shard) the loader deals the global micro-batch order out to the ranks as accelerate does,
     so every optimizer step reads the same 128 x 4k rows on 1 or 4 GPUs.
  3. schedule — our m100 phases: trunk 3.0B tokens = 5,722 steps of 128 x 4,096, warmup 57 steps (1 %), then constant
     (WSD's stable part; their decay_ratio 0); decay from the trunk's final weights (model only: fresh optimizer,
     schedule and stream, their `run.init_from`), 0.4B tokens = 763 steps of 32 x 16,384, warmup 22, 1 - sqrt decay
     over the last 740 steps to 0.1 x. AdamW (0.9, 0.95; torchtitan's default_adamw), weight decay 0.1 on matrices
     (their scope), clip 1.0, fp32 master weights under bf16 autocast (their runtime). Peak LR 6.4e-4: our m100 value,
     so the LR is not a difference between the arms; it lies inside the range their own seed/LR sweep trained the
     0.55B at (5e-4 and 1e-3, lsa_mqa8_lr_seeds.yaml), and their 5e-4 scaled 1/width from 1,024 to 896 is 5.7e-4.
  4. instruction tuning (E37 SFT, docs/experiments_specs/ahead/E37_sft_sensitivity.md) — phase sft: from the decay's
     final weights (model only, `run.init_from`), the manifest of scripts/build_sft_mix.py (--sft_manifest), as
     scripts/run_e33n.py --phase sft trains our arms: 200M tokens = 763 steps of 16 x 16,384, warmup 22, 1 - sqrt decay
     over the last 740 steps to 0.1 x, peak LR from --lr (the pre-flight's pick), the same row order. The items carry
     labels (loss on the answer tokens; the LongDocs-lite replay on every token): the label of a position is the
     item's label of the next token, still ignored across a document start and at the row's end (rows_to_batch).

Usage (on the node, this repo's root as the working directory, their venv's python):
  python scripts/lsa_orig.py params                                    # parameter counts of the preset
  torchrun --standalone --nproc_per_node=1 scripts/lsa_orig.py train --phase trunk --data D --out R [--steps N]
  torchrun --standalone --nproc_per_node=1 scripts/lsa_orig.py train --phase decay --data D --out R --init STEP_DIR
  torchrun --standalone --nproc_per_node=1 scripts/lsa_orig.py train --phase sft --data D --out R \
      --init R/decay/checkpoint/step-763 --sft_manifest D/sft_smoltalk_16k/manifest_sft.json --lr LR
  A phase writes R/<phase>/ (torchtitan's dump folder: checkpoint/step-N, their config.json) and resumes from its
  newest checkpoint when rerun.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LSA_REPO = Path(os.environ.get("LSA_REPO", "/opt/dlami/nvme/lsa"))
PRESET = LSA_REPO / "configs/training/native/qwen35_0_8b_gpt2/pretrain_fineweb_lsa.yaml"
EOT = 50256
IGNORE = -100

# the ~160M body (see the module docstring); everything else is their preset's
BODY = dict(dim=896, n_layers=12, n_key_heads=7, n_value_heads=7, hidden_dim=2240)
# our m100 budget (scripts/run_e33n.py SIZES["m100"]): tokens per phase, 128 rows of 4k per step, peak LR
TRUNK_TOKENS, DECAY_TOKENS, STEP_TOKENS, LR = 3.0e9, 0.4e9, 128 * 4096, 6.4e-4
# E37 SFT (scripts/run_e33n.py SFT): 200M tokens, 16 rows of 16k per step; its peak LR is a required argument
SFT_TOKENS, SFT_STEP_TOKENS = 200e6, 16 * 16384
PHASES = {
    # seq, micro rows per pass (80 GB H100; sft: 24 GB home cards), warmup share, decay share
    "trunk": dict(seq=4096, micro=8, warmup=0.01, decay=0.0, tokens=TRUNK_TOKENS, ckpt=500),
    "decay": dict(seq=16384, micro=2, warmup=0.03, decay=0.97, tokens=DECAY_TOKENS, ckpt=200),
    "sft": dict(seq=16384, micro=1, warmup=0.03, decay=0.97, tokens=SFT_TOKENS, ckpt=200, step_tokens=SFT_STEP_TOKENS),
}


def step_tokens(phase: str) -> int:
    """Tokens per optimizer step: 128 x 4k (trunk, decay) or 16 x 16k (sft)."""
    return int(PHASES[phase].get("step_tokens", STEP_TOKENS))


def _load_by_path(name: str, path: Path):
    """Import one of our modules by file path (no repo root on sys.path: our top-level packages `data`, `nn`,
    `evaluation` must not shadow anything in their environment)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # @dataclass under `from __future__ import annotations` looks its module up here
    spec.loader.exec_module(mod)
    return mod


def bin_dataset():
    return _load_by_path("mc_bin_token_dataset", ROOT / "data" / "bin_token_dataset.py")


def manifests(data: Path, out: Path) -> dict[str, Path]:
    """The two shard manifests of an m100 run, exactly as scripts/run_e33n.manifests writes them."""
    out.mkdir(parents=True, exist_ok=True)
    fw, ld = data / "fineweb_edu_gpt2", data / "longdocs_lite_gpt2"
    trunk = {"seq_len": 4096, "rule": "stream", "eot": EOT, "seed": 42,
             "train": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_train_*.bin")], "weight": 1.0}],
             "eval": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_val_*.bin")], "rows": 256}]}
    srcs = json.loads((ld / "manifest.json").read_text())["sources"]
    decay = {"seq_len": 16384, "rule": "doc", "eot": EOT, "seed": 1042,
             "train": [{"name": s, "files": [str(ld / f"{s}_train_*.bin")], "weight": 1.0} for s in srcs],
             "eval": [{"name": s, "files": [str(ld / f"{s}_val_*.bin")], "rows": 16} for s in srcs]}
    p = {"trunk": out / "manifest_trunk.json", "decay": out / "manifest_decay.json"}
    p["trunk"].write_text(json.dumps(trunk, indent=2))
    p["decay"].write_text(json.dumps(decay, indent=2))
    return p


def rows_to_batch(ids, doc, labels=None):
    """Rows [R, S] (input ids, per-row document ids) -> their trainer's flat batch contract (lsa.training.data.corpus
    .batch_of): input, labels (next token; ignored across a document start and at the row's end), positions (restart
    at each document), padding_mask, num_valid_tokens.

    `labels` [R, S] (optional): the dataset's labels in our HF convention (labels[t] is the target of position t - 1,
    -100 = no target; data/bin_token_dataset.py with loss masks). Position t then predicts labels[t + 1] instead of
    ids[t + 1], under the same two rules. Without it the batch is exactly as before (every next token)."""
    import torch

    R, S = ids.shape
    nxt = ids if labels is None else labels
    labels = torch.full_like(ids, IGNORE)
    labels[:, :-1] = nxt[:, 1:]
    labels[:, :-1] = labels[:, :-1].masked_fill(doc[:, 1:] != doc[:, :-1], IGNORE)
    start = torch.zeros_like(ids, dtype=torch.bool)
    start[:, 0] = True
    start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    ar = torch.arange(S).expand(R, S)
    first = torch.where(start, ar, torch.zeros_like(ar)).cummax(dim=1).values
    flat_labels = labels.reshape(-1)
    return {"input": ids.reshape(-1).contiguous(), "labels": flat_labels.contiguous(),
            "positions": (ar - first).reshape(-1).contiguous(),
            "padding_mask": torch.zeros(R * S, dtype=torch.bool),
            "num_valid_tokens": int((flat_labels != IGNORE).sum())}


def patch_body() -> None:
    """Make torchtitan's Qwen3.5 "0.8B" registry entry build the ~160M body (their code asks for that flavour and then
    applies the GPT-2 vocabulary, tying, inits and the run's head shape on top). Only this process sees it."""
    import torchtitan.models.qwen3_5 as q

    if getattr(q, "_mc_patched", False):
        return
    _, max_len = q.qwen3_5_configs["0.8B"]

    def lsa160m(attn_backend: str, *, seq_len: int):
        dim, vocab = BODY["dim"], 248320  # the vocabulary is replaced by gpt2_model (50,304, tied)
        return q.Qwen35Model.Config(
            vocab_size=vocab, dim=dim, norm=q._offset_norm(dim),
            tok_embeddings=q.Embedding.Config(num_embeddings=vocab, embedding_dim=dim, param_init=q._EMBEDDING_INIT),
            lm_head=q.Linear.Config(in_features=dim, out_features=vocab, param_init=q._output_linear_init(dim)),
            layers=q._build_qwen35_layers(
                # attention template as Qwen3.5-0.8B's; shape_heads then sets the run's 8Q/1KV/128, rotary 0
                rope=q.MRoPE.Config(dim=64, max_context_length=seq_len, theta=10_000_000.0, mrope_section=[11, 11, 10]),
                attn_backend=attn_backend, n_layers=BODY["n_layers"], dim=dim, n_heads=8, n_kv_heads=2, head_dim=256,
                rotary_dim=64, hidden_dim=BODY["hidden_dim"], n_key_heads=BODY["n_key_heads"],
                n_value_heads=BODY["n_value_heads"], key_head_dim=128, value_head_dim=128),
        )

    q.qwen3_5_configs["0.8B"] = (lsa160m, max_len)
    q._mc_patched = True


def fit_kernels() -> str:
    """Their LSA selection on a GPU whose shared memory its Triton kernel does not fit.

    Their code picks the selection kernel by `supported()`: lsa_rank (CUDA, sm_90 only), else lsa_select (Triton), else
    `reference.select_indices` (chunked torch, the specification both kernels are tested against). lsa_select's tiles
    need 105-168 KiB of shared memory a block at any tile size (measured on the L40S: 101,376 bytes there; the H100 has
    227 KiB), and its `supported()` does not check, so on such a GPU it is switched off and the reference selects
    (6 ms a 4k row, 39 ms a 16k row on an L40S). The read itself (gather_sparse, Triton) fits and is unchanged."""
    import torch
    from lsa.training.native.kernels import lsa_select

    if not torch.cuda.is_available():
        return "cpu"
    smem = int(getattr(torch.cuda.get_device_properties(0), "shared_memory_per_block_optin", 0))
    if smem and smem < 172032:
        lsa_select.supported = lambda *args, **kwargs: False
        return f"lsa_select off ({smem} B shared memory a block < 172,032): reference selection"
    return f"their kernels as they are ({smem} B shared memory a block)"


def phase_numbers(phase: str, steps: int | None = None, micro: int | None = None,
                  tokens: float | None = None) -> dict[str, Any]:
    p = PHASES[phase]
    seq = p["seq"]
    full = int(round((tokens or p["tokens"]) / step_tokens(phase)))  # 5722 / 763 / 763, as run_e33n.train_cmd rounds
    n = steps or full
    warm = max(1, int(p["warmup"] * full))                        # HF: int(0.01 * steps) / int(0.03 * steps)
    decay_steps = int(p["decay"] * full)                          # HF: num_decay_steps = int(0.97 * steps)
    # their schedule rounds the decay up (ceil(ratio * steps)); a ratio half a step under ours keeps our count.
    # A shortened smoke run keeps the phase's shape instead.
    ratio = ((decay_steps - 0.5) / n if decay_steps else 0.0) if n == full else p["decay"]
    m = micro or p["micro"]
    return dict(seq=seq, steps=n, warmup=min(warm, n), decay_ratio=ratio, decay_steps=math.ceil(ratio * n),
                tokens_per_rank=m * seq, micro=m, ckpt=p["ckpt"])


def experiment(phase: str, out: Path, *, init: str | None = None, steps: int | None = None,
               micro: int | None = None, extra: list[str] | None = None, lr: float | None = None,
               tokens: float | None = None):
    """Their NativeConfig for one phase: the preset plus our overrides (lr: the sft phase's peak, required there)."""
    from lsa.training.native.config import load

    if phase == "sft" and not lr:
        raise ValueError("phase sft needs its peak LR (--lr)")
    lr = lr or LR
    z = phase_numbers(phase, steps, micro, tokens)
    ov = [
        # data parallel over the GPUs torchrun gives (their default: FSDP over dp_shard); 1 for the evals
        f"run.output={out.resolve()}", f"run.name={phase}", f"run.dp_shard={int(os.environ.get('WORLD_SIZE', 1))}",
        "run.dp_replicate=1",
        f"run.optim.steps={z['steps']}", f"run.optim.warmup_steps={z['warmup']}", f"run.optim.lr={lr}",
        f"run.optim.decay_ratio={z['decay_ratio']}", "run.optim.decay_shape=sqrt", "run.optim.schedule=wsd",
        "run.optim.min_lr_factor=0.1", "run.optim.weight_decay=0.1", "run.optim.grad_clip=1.0",
        f"run.optim.tokens_per_rank={z['tokens_per_rank']}", f"run.optim.tokens_per_step={step_tokens(phase)}",
        "run.checkpoint.decay_start=false", f"run.checkpoint_every={z['ckpt']}",
        "run.runtime.seed=42", "run.runtime.autocast_bf16=true", "run.runtime.log_every=10",
        "run.runtime.activation_checkpoint=none", "run.evaluation.every=0", f"data.seq_len={z['seq']}",
    ]
    if init:
        ov.append(f"run.init_from={Path(init).resolve()}")
    exps = load(PRESET, ov + list(extra or []))
    assert len(exps) == 1, f"{len(exps)} runs resolved"
    return exps[0], z


def tokenizer_path(exp, data: Path) -> str:
    """Their pinned GPT-2 hub snapshot; our identical tokenizer files if the snapshot is missing."""
    p = exp.tokenizer_path
    return p if Path(p, "tokenizer.json").exists() else str(data / "tokenizer_gpt2")


def make_loader_class():
    import torch
    from torchtitan.components.data.loader import BaseDataLoader

    class McShardRows(BaseDataLoader):
        """Our rows in our HF trainer's order, as a torchtitan dataloader (state: micro-batches taken).

        One global order of micro-batches (rows perm[g * rows : (g + 1) * rows] for micro-batch g); with W data-parallel
        ranks, rank r takes micro-batches r, r + W, r + 2W, ... (accelerate's BatchSamplerShard), so every optimizer
        step reads the same 128 x 4k rows whatever the number of GPUs."""

        @dataclass(kw_only=True, slots=True)
        class Config(BaseDataLoader.Config):
            manifest: str = ""
            order_seed: int = 42

        def __init__(self, config: Config, *, dp_world_size: int, dp_rank: int, max_context_length: int,
                     num_tokens_per_batch: int, **kwargs: Any) -> None:
            del kwargs
            super().__init__()
            self.world, self.rank = int(dp_world_size), int(dp_rank)
            self.key = f"dp_rank_{self.rank}"
            self.ds, _, m = bin_dataset().load_bin_manifest(config.manifest)
            self.S = int(m["seq_len"])
            if num_tokens_per_batch % self.S or self.S > max_context_length:
                raise ValueError(f"{num_tokens_per_batch} tokens per pass is not whole rows of {self.S}")
            self.rows = num_tokens_per_batch // self.S
            self.seed = int(config.order_seed)
            self.batches = 0
            self._epoch, self._order = -1, None
            print(f"[mc loader] {config.manifest}: {len(self.ds):,} rows of {self.S}, {self.rows} per pass, "
                  f"per source {self.ds.rows_per_source}", flush=True)

        def _indices(self, k: int) -> list[int]:
            n = len(self.ds)
            out = []
            gm = k * self.world + self.rank   # this rank's k-th micro-batch in the global order
            for g in range(gm * self.rows, (gm + 1) * self.rows):
                epoch, i = divmod(g, n)
                if epoch != self._epoch:   # accelerate's SeedableRandomSampler: seed + epoch
                    self._order = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + epoch))
                    self._epoch = epoch
                out.append(int(self._order[i]))
            return out

        def __iter__(self):
            while True:
                items = [self.ds[i] for i in self._indices(self.batches)]
                self.batches += 1
                yield rows_to_batch(torch.stack([x["input_ids"] for x in items]),
                                    torch.stack([x["doc_ids"] for x in items]),
                                    torch.stack([x["labels"] for x in items]) if "labels" in items[0] else None)

        def state_dict(self) -> dict[str, Any]:
            return {"dp_world_size": self.world,
                    self.key: {"batches": self.batches, "seed": self.seed, "rows": self.rows}}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            if state_dict.get("dp_world_size") != self.world or self.key not in state_dict:
                raise ValueError(f"loader state saved for {state_dict.get('dp_world_size')} ranks, now {self.world}")
            s = state_dict[self.key]
            if (s["seed"], s["rows"]) != (self.seed, self.rows):
                raise ValueError(f"loader state saved with seed/rows {s['seed']}/{s['rows']}")
            self.batches = int(s["batches"])

    return McShardRows


def train(args) -> None:
    from torchtitan.tools.logging import init_logger

    from lsa.io import write_json
    from lsa.training.native.run import describe, meta
    from lsa.training.native.trainer import check_label_contract, trainer_config

    out = Path(args.out)
    man = manifests(Path(args.data), out)
    if args.phase == "sft":
        man["sft"] = Path(args.sft_manifest).resolve()
        seq = int(json.loads(man["sft"].read_text())["seq_len"])
        if seq != PHASES["sft"]["seq"]:
            raise ValueError(f"sft manifest seq_len {seq} != {PHASES['sft']['seq']}")
    patch_body()
    lr = args.lr if args.phase == "sft" else LR
    exp, z = experiment(args.phase, out, init=args.init, steps=args.steps, micro=args.micro, lr=lr,
                        tokens=args.tokens if args.phase == "sft" else None)
    init_logger()
    check_label_contract()
    print(describe(exp), flush=True)
    print(f"[mc] phase {args.phase}: {z}; kernels: {fit_kernels()}", flush=True)
    config = trainer_config(exp, check_corpora=False)
    config.hf_assets_path = tokenizer_path(exp, Path(args.data))
    Loader = make_loader_class()
    config.dataloader = Loader.Config(manifest=str(man[args.phase]), order_seed=42)
    config.checkpoint.keep_latest_k = 2
    if int(os.environ.get("RANK", 0)) == 0:
        write_json(exp.out / "config.json", {**meta(exp), "mc": {"body": BODY, "phase": z, "lr": lr,
                                                                 "manifest": str(man[args.phase])}})
    trainer = config.build()
    try:
        trainer.train()
    finally:
        trainer.close()


def params(args) -> None:
    """Parameter counts of the ~160M body under their preset (meta device, no GPU)."""
    import torch

    from lsa.training.native.trainer import trainer_config

    patch_body()
    exp, _ = experiment("trunk", Path(args.out))
    config = trainer_config(exp, check_corpora=False)
    spec = config.model_spec
    spec.model.update_from_config(config=config)
    with torch.device("meta"):
        model = spec.model.build()
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    emb = sum(p.numel() for n, p in named if n.startswith("tok_embeddings"))
    head = sum(p.numel() for n, p in named if n.startswith("lm_head"))
    kinds = {}
    for i, layer in enumerate(model.layers.values() if hasattr(model.layers, "values") else model.layers):
        mod = getattr(layer, "attn", None) or getattr(layer, "attention", None)
        kind = "gated_deltanet" if type(mod).__name__ == "GatedDeltaNet" else type(mod).__name__
        kinds.setdefault(kind, []).append(i)
    res = {"total": total, "embedding": emb, "lm_head_separate": head, "non_embedding": total - emb - head,
           "tied": bool(getattr(spec.model, "enable_weight_tying", False)), "layers": kinds, "body": BODY,
           "attention": {k: getattr(exp.attention, k) for k in ("kind", "coarse", "fast_read", "window_size",
                                                               "block_size", "pool_size", "bucket_size",
                                                               "bucket_cap", "sink_size", "top_k",
                                                               "query_head_reduction")},
           "heads": {k: getattr(exp.model, k) for k in ("query_head_count", "kv_head_count", "head_dim",
                                                        "rotary_dim")}}
    print(json.dumps(res, indent=1))
    print(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("params")
    p.add_argument("--out", default="/tmp/lsa_orig_params")
    t = sub.add_parser("train")
    t.add_argument("--phase", required=True, choices=sorted(PHASES))
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init", help="decay: the trunk's final step folder (.../trunk/checkpoint/step-N); "
                                  "sft: the decay's (.../decay/checkpoint/step-763)")
    t.add_argument("--steps", type=int, help="smoke test: stop after N steps (schedule shape of the full phase)")
    t.add_argument("--micro", type=int, help="rows per pass (gradient accumulation makes up the 128 x 4k step)")
    t.add_argument("--sft_manifest", help="sft: manifest_sft.json of scripts/build_sft_mix.py")
    t.add_argument("--lr", type=float, help="sft: peak LR (required; trunk and decay use 6.4e-4)")
    t.add_argument("--tokens", type=float, help="sft: tokens of the stage (default 200e6; the LR pre-flight 20e6)")
    a = ap.parse_args()
    if a.what == "train" and a.phase == "sft" and not (a.sft_manifest and a.lr and a.init):
        ap.error("--phase sft needs --sft_manifest, --lr and --init (the decay's final step folder)")
    params(a) if a.what == "params" else train(a)


if __name__ == "__main__":
    main()

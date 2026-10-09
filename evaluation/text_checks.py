"""Text capability checks — the definitions the runner, scorer and scorecard share (draft).

Spec: docs/engineering_specs/text_capability_checks.md. Skill: `text-checks`.

- TIERS: parameter target, token budget, training length, global batch, compute cap, layer shape.
- ARCHES: each round-1 model as trainer arguments (the same names as the training CLI) on top of
  the tier shape. FFN width is fitted per architecture so every model lands in the tier's
  parameter band (±5 %) — the one knob the protocol lets differ for parameter parity.
- Recipe cards: each architecture's own initialization / optimizer / schedule, from
  `evaluation/text_checks_recipes/<arch>.json` (priors until tuned; see §7 of the spec).
- Scoring rules (pass mark, controls, gating levels) for the scorecard.

Everything here is a draft to be calibrated on the first server runs (budgets, caps, shapes).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

TEXT_CHECKS_VERSION = "text-v0-draft"
RECIPE_DIR = Path(__file__).resolve().parent / "text_checks_recipes"
PASS = 0.75
SHORTCUT_MARGIN = 0.10            # evidence-removed score must stay within floor + 10 points (or 2 SE)
GATING_TASKS = ("quote", "lookup", "keyed", "latest", "compose")   # text-core T1–T5
FRONTIER_TASKS = ("count", "deduce")                                  # T6, T7
CURVE_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
VOCAB = 4096


@dataclass(frozen=True)
class Tier:
    name: str
    target_params: float          # all trainable params incl. embeddings, head, memory
    tokens: float                 # training tokens (data budget)
    seq_len: int
    global_rows: int              # rows per optimizer step (fixed for every model and host)
    per_device_batch: int
    cap_gpu_hours: float          # compute cap: active training GPU-hours
    hidden: int
    pre_layers: int
    global_layers: int
    stack_layers: int
    eval_lengths: tuple = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    extra_lengths: tuple = (4096, 16384)    # harder / paraphrase splits
    tune_runs: int = 4
    tune_tokens: float = 60e6
    band: float = 0.05


TIERS = {
    "screen": Tier("screen", 30e6, 0.6e9, 4096, 96, 8, 24.0, hidden=576, pre_layers=2, global_layers=1,
                   stack_layers=5),
    "main": Tier("main", 100e6, 2.0e9, 4096, 96, 4, 120.0, hidden=896, pre_layers=3, global_layers=1,
                 stack_layers=8, tune_runs=3, tune_tokens=100e6),
    # local pipeline-check tier (Apple M5 Max laptop, ~1 h per model): confirms the pipeline works and
    # catches obvious errors only; no conclusions on learnability or budgets (those runs go to the servers).
    "lab": Tier("lab", 16e6, 100e6, 1024, 32, 16, 3.0, hidden=256, pre_layers=2, global_layers=1,
                stack_layers=4, eval_lengths=(1024, 2048, 4096), extra_lengths=(1024,), tune_runs=3,
                tune_tokens=12e6, band=0.10),
    # local plumbing check only (minutes on a laptop); never a result
    "smoke": Tier("smoke", 2e6, 0.25e6, 1024, 8, 4, 0.5, hidden=128, pre_layers=1, global_layers=1,
                  stack_layers=2, eval_lengths=(512, 1024, 2048), extra_lengths=(1024,), tune_runs=2,
                  tune_tokens=0.05e6, band=0.25),
}


def _base(t: Tier) -> dict:
    """Arguments every model shares at a tier (the E31 'as tested' input layer and windows)."""
    return {
        "model_family": "perceiver_ar", "objective_variant": "causal_lm", "decoder_type": "causal_ar",
        "hidden_size": t.hidden, "token_embedding_dim": 256 if t.hidden >= 256 else t.hidden,
        "head_dim": 64 if t.hidden >= 256 else 32, "num_kv_heads": 1,
        "par_pre_layers": t.pre_layers, "par_pre_window": 16, "par_global_layers": t.global_layers,
        "num_hidden_layers": t.stack_layers, "par_block": 16,
        "par_ngram_orders": "2,3", "par_ngram_buckets": 8192 if t.hidden >= 256 else 1024,
        "par_value_embed_layers": f"0,{t.pre_layers}", "par_value_embed_dim": 64 if t.hidden >= 256 else 16,
        "logit_softcap": 30.0, "z_loss": 1e-4,
    }


_E31C = {
    "par_mode": "perceiver", "message_write": "latent_memory", "message_raw_window": 256,
    "lm_read": "closed", "lm_context": "page_bidir", "lm_window": 256, "lm_stride": 192,
    "lm_latents": 32, "lm_latent_dim": 512, "lm_heads": 8, "lm_writer_dim": 256, "lm_enc_layers": 2,
    "lm_rounds": 2, "lm_reader_tokens": 1, "lm_addr": "none", "lm_slot_pos": "reader",
}
_E31C_SMOKE = {"lm_latents": 8, "lm_latent_dim": 64, "lm_heads": 2, "lm_writer_dim": 64, "lm_enc_layers": 1,
               "lm_rounds": 1}


@dataclass(frozen=True)
class Arch:
    name: str
    role: str
    args: dict = field(default_factory=dict)
    has_notebook: bool = False


ARCHES = {
    "dense": Arch("dense", "ceiling at the training length; language reference", {"par_mode": "dense"}),
    # diagnostic control (lab tier, 2026-10-09): the dense model without the E31 input layer (no hashed
    # n-gram features, no value embeddings) — does that layer delay learning to copy from context?
    "dense_plain": Arch("dense_plain", "diagnostic: dense without the n-gram / value-embedding input layer",
                        {"par_mode": "dense", "par_ngram_orders": "", "par_value_embed_layers": ""}),
    "local": Arch("local", "no-long-memory control: E31c trained with its notebook removed (recent 256 tokens only)",
                  {**_E31C, "message_override": "none"}),
    "e31c": Arch("e31c", "E31 latent notebook, closed-window text read", dict(_E31C), has_notebook=True),
    "e31c_loop": Arch("e31c_loop", "E31c + E33a read-think-reread loop (4 rounds)",
                      {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.3,
                       "message_loop_exit_targets": "answer"}, has_notebook=True),
}

# trainer argument → launcher env var (scripts/train_concept_pretraining_multigpu.sh)
ENV_OF = {
    "hidden_size": "HIDDEN_SIZE", "intermediate_size": "INTERMEDIATE_SIZE", "token_embedding_dim": "TOKEN_EMBEDDING_DIM",
    "num_hidden_layers": "NUM_LAYERS", "objective_variant": "OBJECTIVE_VARIANT", "decoder_type": "DECODER_TYPE",
    "model_family": "MODEL_FAMILY", "par_mode": "PAR_MODE", "par_pre_layers": "PAR_PRE_LAYERS",
    "par_pre_window": "PAR_PRE_WINDOW", "par_global_layers": "PAR_GLOBAL_LAYERS", "par_block": "PAR_BLOCK",
    "num_attention_heads": "NUM_ATTENTION_HEADS", "num_kv_heads": "NUM_KV_HEADS", "head_dim": "HEAD_DIM",
    "par_ngram_orders": "PAR_NGRAM_ORDERS", "par_ngram_buckets": "PAR_NGRAM_BUCKETS",
    "par_value_embed_layers": "PAR_VALUE_EMBED_LAYERS", "par_value_embed_dim": "PAR_VALUE_EMBED_DIM",
    "logit_softcap": "LOGIT_SOFTCAP", "z_loss": "Z_LOSS", "attn_backend": "ATTN_BACKEND",
    "message_write": "PAR_MESSAGE_WRITE", "message_raw_window": "PAR_MESSAGE_RAW_WINDOW",
    "message_override": "PAR_MESSAGE_OVERRIDE", "lm_read": "PAR_LM_READ", "lm_context": "PAR_LM_CONTEXT",
    "lm_window": "PAR_LM_WINDOW", "lm_stride": "PAR_LM_STRIDE", "lm_latents": "PAR_LM_LATENTS",
    "lm_latent_dim": "PAR_LM_LATENT_DIM", "lm_heads": "PAR_LM_HEADS", "lm_writer_dim": "PAR_LM_WRITER_DIM",
    "lm_enc_layers": "PAR_LM_ENC_LAYERS", "lm_rounds": "PAR_LM_ROUNDS", "lm_reader_tokens": "PAR_LM_READER_TOKENS",
    "lm_addr": "PAR_LM_ADDR", "lm_slot_pos": "PAR_LM_SLOT_POS", "message_loop_rounds": "PAR_LOOP_ROUNDS",
    "message_loop_exit_aux": "PAR_LOOP_EXIT_AUX", "message_loop_exit_targets": "PAR_LOOP_EXIT_TARGETS",
}


def model_args(arch: str, tier: str, intermediate_size: int | None = None) -> dict:
    t = TIERS[tier]
    a = ARCHES[arch]
    args = {**_base(t), **a.args}
    if tier == "smoke" and "lm_latents" in args:
        args.update(_E31C_SMOKE)
    args["num_attention_heads"] = t.hidden // args["head_dim"]
    args["intermediate_size"] = intermediate_size or _round64(t.hidden * 8 / 3)
    return args


def _round64(x: float) -> int:
    return max(64, int(round(x / 64.0)) * 64)


def count_params(args: dict, vocab: int = VOCAB) -> int:
    """Exact trainable parameter count, built through the trainer's own factory on the meta device."""
    import torch
    from types import SimpleNamespace

    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "sdpa", False
    tok = type("T", (), {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "__len__": lambda self: vocab})()
    with torch.device("meta"):
        model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="text-checks"))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_params(arch: str, tier: str, vocab: int = VOCAB) -> tuple[dict, int]:
    """Pick the FFN width (multiple of 64) that brings this model closest to the tier's target."""
    t = TIERS[tier]
    lo, hi = 64, _round64(t.hidden * 12)
    best = None
    while lo <= hi:  # params grow monotonically with the FFN width
        mid = _round64((lo + hi) / 2)
        n = count_params(model_args(arch, tier, mid), vocab)
        if best is None or abs(n - t.target_params) < abs(best[1] - t.target_params):
            best = (mid, n)
        if n < t.target_params:
            lo = mid + 64
        else:
            hi = mid - 64
    width, n = best
    return model_args(arch, tier, width), n


def in_band(n: int, tier: str) -> bool:
    t = TIERS[tier]
    return abs(n - t.target_params) <= t.band * t.target_params


DEFAULT_RECIPE = {
    "status": "prior (untuned)",
    "init": "trainer default: normal(0.02) projections, zero-init residual outputs (perceiver_ar)",
    "optimizer": "adam", "weight_decay": 0.1, "max_grad_norm": 1.0,
    "lr": {"screen": 1e-3, "main": 6e-4, "smoke": 1e-3, "lab": 1e-3},
    "warmup_frac": 0.02, "scheduler": "cosine",
    "length_method": "none",
    "tuning": {},
}


def load_recipe(arch: str) -> dict:
    path = RECIPE_DIR / f"{arch}.json"
    card = json.loads(path.read_text()) if path.exists() else {}
    out = json.loads(json.dumps(DEFAULT_RECIPE))
    for k, v in card.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def steps_for(tokens: float, global_rows: int, mean_row_tokens: float) -> int:
    """Optimizer steps that consume `tokens`. Every model reads the same rows in the same order,
    so the same step count is the same data for all."""
    return max(1, int(math.ceil(tokens / (global_rows * mean_row_tokens))))


def per_device_batch(recipe: dict, tier: str) -> int:
    """Rows per GPU micro-batch: the tier default unless the model's recipe card lowers it to fit memory
    (the global batch, hence the data and the optimizer steps, stay the same through accumulation)."""
    return int((recipe.get("per_device_batch") or {}).get(tier, TIERS[tier].per_device_batch))


def grad_accum(tier: str, n_gpus: int, pdb: int | None = None) -> int:
    t = TIERS[tier]
    pdb = pdb or t.per_device_batch
    per_step = pdb * n_gpus
    if t.global_rows % per_step:
        raise ValueError(f"global batch {t.global_rows} rows is not a multiple of {pdb} × {n_gpus} GPUs")
    return t.global_rows // per_step

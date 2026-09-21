#!/usr/bin/env python
"""Train tiny E18 / dense / encoder-decoder models on one BAPO DNA rung.

This is the scale-up of `verification/symbolic_channel_probe.py` onto E18 / E21 and
matched dense transformers. The other agent owns the `perceiver_concept` Arm-A 100%
map at seq=128. E24 scored E18. E25 scores **E21** (message boundary + compressed slots)
on the same DNA rungs, one at a time.

Protocol
--------
1. Train `dense` first, up to `--steps * --k1_mult` (K1). If held-out accuracy < 75%, the rung
   is uncalibrated — do not interpret E18/E21 numbers.
2. Train `e18_local` on retrieval rungs. It must sit near chance / the analytic floor.
3. Train `e18` (uncompressed one-read control), `e21` (mean slots), and `e30`
   (sliding-window Perceiver banks) under `max(--steps, dense_steps_used)`.
   Small-model protocol: `docs/engineering_specs/small_model_capability_protocol.md`.
4. Write a JSON bundle (learning traces + InfoReport) and optional plots.

  uv run python verification/bapo_capability_probe.py --scale tiny --recipe far_copy --arch dense e18 e21 e18_local
"""
from __future__ import annotations

import argparse
import math
import os
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.bapo_ladder import (  # noqa: E402
    CALIBRATED_RECIPES,
    GLYPH_CORE_RECIPES,
    SCALES,
    SOLVABLE_ACC,
    TINY_PROOF_TASKS,
    config_for,
    generate_row_for,
    resolve_recipe,
    rung_card,
)
from data.glyph_tasks import GlyphTaskConfig, floor_nats as glyph_floor_nats  # noqa: E402
from data.symbolic_tasks import floor_nats  # noqa: E402
from evaluation.bapo_metrics import info_report  # noqa: E402
from evaluation.bapo_models import ARCHES, ArchSpec, arch_cache, build_model, n_params  # noqa: E402


def _message_cm(model, override: str):
    """E21 probe control: wrap forwards in `model.message_override`. Default `real` is a no-op.

    `raw` keeps the QUERY document-start on local SWA / n-grams and lets the global read see
    uncompressed prefix K/V across the boundary. Ignored when the model has no message path.
    """
    mode = override or "real"
    if mode == "real" or not hasattr(model, "message_override"):
        return contextlib.nullcontext()
    return model.message_override(mode)


def amp_ctx(device: torch.device, amp: str):
    """bf16 autocast on CUDA; off on CPU. fp16 is opt-in (no GradScaler — prefer bf16)."""
    if amp == "off" or device.type != "cuda":
        return contextlib.nullcontext()
    want = amp
    if amp == "auto":
        want = "bf16" if torch.cuda.is_bf16_supported() else "off"
    if want == "off":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if want == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _floor(cfg, window: int) -> float:
    if isinstance(cfg, GlyphTaskConfig):
        return glyph_floor_nats(cfg, window)
    return floor_nats(cfg, window)


def _boundary_token_id(cfg) -> int:
    """DNA/Glyph `query` control is the E21 message boundary. Missing → off."""
    vocab = getattr(cfg, "vocab", None)
    if vocab is None or not hasattr(vocab, "control"):
        return -1
    try:
        return int(vocab.control("query"))
    except (KeyError, TypeError):
        return -1


_TYPE_MARK_NAMES = ("keymark", "decoy", "spanmark", "hop", "mark")


def _type_mark_token_ids(cfg) -> tuple[int, ...]:
    """Control ids that open evidence / decoy blocks (sparse type-cue anchors)."""
    vocab = getattr(cfg, "vocab", None)
    if vocab is None or not hasattr(vocab, "control"):
        return ()
    ids = []
    for name in _TYPE_MARK_NAMES:
        try:
            ids.append(int(vocab.control(name)))
        except (KeyError, TypeError):
            continue
    return tuple(ids)


def _keymark_token_ids(cfg) -> tuple[int, ...]:
    vocab = getattr(cfg, "vocab", None)
    if vocab is None or not hasattr(vocab, "control"):
        return ()
    try:
        return (int(vocab.control("keymark")),)
    except (KeyError, TypeError):
        return ()


def make_batch(cfg, rng, batch: int, device):
    rows = [generate_row_for(cfg, rng) for _ in range(batch)]
    ids = torch.from_numpy(np.stack([r.input_ids for r in rows])).long().to(device)
    labels = torch.from_numpy(np.stack([r.labels for r in rows])).long().to(device)
    return ids, labels


@torch.no_grad()
def evaluate(model, batches, *, amp: str, device: torch.device, message_override: str = "real") -> dict:
    model.eval()
    ce_sum, n, hits = 0.0, 0, 0
    ctx = amp_ctx(device, amp)
    for ids, labels in batches:
        with _message_cm(model, message_override), ctx:
            out = model(ids, labels=labels, return_per_token_loss=True)
            if isinstance(out, tuple):
                _lm, per, valid = out
            else:
                raise RuntimeError("model did not return per-token loss")
            packed = model(ids, return_logits=True)
            logits = packed.logits if hasattr(packed, "logits") else packed
        ce_sum += float(per[valid].float().sum())
        n += int(valid.sum())
        pred = logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        m = tgt != -100
        hits += int((pred[m] == tgt[m]).sum())
    model.train()
    return {"ce_nats": ce_sum / max(n, 1), "acc": hits / max(n, 1), "tokens": n}


def _slot_rankme(model, batches, *, amp: str, device: torch.device) -> dict:
    """Within-batch RankMe of exclusive slot K (health diagnostic; not concept_ar)."""
    if not hasattr(model, "layers"):
        return {}
    gi = int(getattr(model.config, "global_layer_index", 0) or 0)
    attn = model.layers[gi].attn
    if getattr(attn, "compressor", None) is None:
        return {}
    chunks = []
    model.eval()
    ctx = amp_ctx(device, amp)
    with torch.no_grad():
        for ids, labels in batches:
            with ctx:
                _ = model(ids, labels=labels)
            k_bar = getattr(attn, "_last_k_bar", None)
            if k_bar is None:
                continue
            chunks.append(k_bar.detach().float().cpu().reshape(-1, k_bar.shape[-2] * k_bar.shape[-1]))
    model.train()
    if not chunks:
        return {}
    x = torch.cat(chunks, dim=0)
    x = x - x.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        return {"n": int(x.shape[0]), "dim": int(x.shape[1])}
    p = (s * s) / (s * s).sum().clamp(min=1e-12)
    rankme = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    return {
        "slot_rankme": rankme,
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "n_singular": int(s.numel()),
    }


def _write_geometry(model, seq_len: int) -> dict:
    """E30 window-bank diagnostics (and a no-op for other writes)."""
    cfg = getattr(model, "config", None)
    if cfg is None or str(getattr(cfg, "message_write", "block_mean") or "block_mean") != "sw_perceiver":
        return {}
    from nn.perceiver_ar_lm import swp_geometry

    geo = swp_geometry(cfg, seq_len)
    out = {
        "bank_size": geo.bank_size,
        "window": geo.window,
        "stride": geo.stride,
        "n_windows": geo.n_windows,
        "n_slots": geo.n_slots,
        "coverage": geo.coverage,
        "compression": geo.compression,
        "sliding": geo.sliding,
        "log_window": float(math.log(max(geo.window, 1))),
    }
    gi = int(getattr(cfg, "global_layer_index", 0) or 0)
    attn = model.layers[gi].attn if hasattr(model, "layers") else None
    ent = getattr(getattr(attn, "compressor", None), "_last_entropy", None)
    if ent is not None and ent == ent:
        out["attn_entropy"] = float(ent)
        out["entropy_over_logW"] = float(ent) / max(out["log_window"], 1e-6)
    return out


def _channel_ablations(model, eval_batches, *, amp, device, spec) -> dict:
    """MATCH analogue of concept ablation: real / none (anchors-only if key_spans) / swapped / slots-only."""
    out = {}
    for mode in ("none", "swapped"):
        try:
            out[mode] = evaluate(model, eval_batches, amp=amp, device=device, message_override=mode)
        except Exception as exc:  # noqa: BLE001 — keep the probe going
            out[mode] = {"error": str(exc)}
    prev = model.config.message_global_anchors
    if prev not in ("none",):
        model.config.message_global_anchors = "none"
        try:
            out["slots_only"] = evaluate(
                model, eval_batches, amp=amp, device=device, message_override="real"
            )
        except Exception as exc:  # noqa: BLE001
            out["slots_only"] = {"error": str(exc)}
        model.config.message_global_anchors = prev
    return out


def _ce_still_falling(trace: list, *, min_drop: float = 0.2) -> bool:
    """True when eval CE dropped ≥ `min_drop` nats in the last third of logged evals.

    Exclusive-slot <10M law: dense can sit at chance then jump. A still-falling CE at
    the K1 cap is not a kill — extend, don't skip_rest (small_model_capability_protocol).
    """
    if len(trace) < 2:
        return False
    n = len(trace)
    cut = max(0, n - max(1, n // 3) - 1)
    early = float(trace[cut].get("ce_nats", 0.0))
    late = float(trace[-1].get("ce_nats", 0.0))
    return (early - late) >= min_drop


def train_one(arch: str, cfg, args, eval_batches, spec: ArchSpec, device, *, steps: int) -> dict:
    model = build_model(
        arch,
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=args.seed,
    ).to(device)
    params = n_params(model)
    if params > args.max_params:
        raise SystemExit(f"{arch} has {params} params > --max_params {args.max_params}")
    if hasattr(model, "layers"):
        patterns = [(layer.attn.pattern, layer.attn.window) for layer in model.layers]
        print(
            f"  [{arch}] {params/1e6:.3f}M  patterns={patterns}  "
            f"kv={getattr(model.config, 'num_kv_heads', '-')}  "
            f"logit_scale={getattr(model.config, 'global_logit_scale', '-')}  "
            f"backend={getattr(model.config, 'attn_backend', '-')}  "
            f"zero_resid={getattr(model.config, 'zero_init_residuals', True)}  "
            f"msg_boundary={getattr(model.config, 'message_boundary_token_id', -1)}  "
            f"msg_r={getattr(model.config, 'message_compress_ratio', '-')}  "
            f"msg_remainder={getattr(model.config, 'message_pool_remainder', False)}  "
            f"msg_override={args.message_override if arch in ('e21', 'e30') else '-'}  "
            f"msg_inplace={getattr(model.config, 'message_slots_inplace', False) if arch == 'e21' else '-'}  "
            f"msg_rawkv={getattr(model.config, 'message_inplace_raw_kv', False) if arch == 'e21' else '-'}  "
            f"msg_idslots={getattr(model.config, 'message_identity_slots', False) if arch == 'e21' else '-'}  "
            f"msg_packstride={getattr(model.config, 'message_pack_stride', 0) if arch == 'e21' else '-'}  "
            f"msg_keepswa={getattr(model.config, 'message_keep_local_swa', False) if arch in ('e21', 'e30') else '-'}  "
            f"glob_layers={getattr(model.config, 'global_layers', '-')}  "
            f"msg_extrahops={getattr(model.config, 'message_extra_slot_attends', 0) if arch in ('e21', 'e30') else '-'}  "
            f"msg_updatekv={getattr(model.config, 'message_update_slot_kv', False) if arch in ('e21', 'e30') else '-'}  "
            f"msg_anchors={getattr(model.config, 'message_global_anchors', 'none') if arch in ('e21', 'e30') else '-'}  "
            f"msg_keylen={getattr(model.config, 'message_anchor_key_len', 0) if arch == 'e21' else '-'}  "
            f"msg_prefix_ae={getattr(model.config, 'message_prefix_ae', False) if arch == 'e21' else '-'}  "
            f"msg_write={getattr(model.config, 'message_write', '-')}",
            flush=True,
        )
    if arch == "e30":
        from nn.perceiver_ar_lm import swp_geometry

        geo = swp_geometry(model.config, cfg.seq_len)
        print(
            f"  [e30] geometry K={geo.bank_size} W={geo.window} stride={geo.stride} "
            f"n_win={geo.n_windows} C={geo.n_slots} N/C={geo.compression:.2f} "
            f"heads={geo.n_heads} qdim={geo.query_dim} sliding={geo.sliding}",
            flush=True,
        )
        if not geo.sliding:
            print("  [e30] WARNING: n_windows<2 — this length is not the sliding claim", flush=True)
    override = args.message_override if arch in ("e21", "e30") else "real"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = max(1, min(50, steps // 10))

    def lr_factor(step: int) -> float:
        return min(1.0, (step + 1) / warmup)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    rng = np.random.default_rng(args.seed + 1000 + sum(ord(c) for c in arch))
    t0, trace = time.time(), []
    best_acc = -1.0
    k1_extended = False
    sdp_cm = contextlib.nullcontext()
    if args.sdpa_math:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        sdp_cm = sdpa_kernel(SDPBackend.MATH)
    step = 0
    max_steps = steps
    with sdp_cm:
        while step < max_steps:
            step += 1
            ids, labels = make_batch(cfg, rng, args.batch, device)
            with _message_cm(model, override), amp_ctx(device, args.amp):
                out = model(ids, labels=labels)
                loss = out.loss if hasattr(out, "loss") else out[0].loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % args.eval_every == 0 or step == max_steps:
                ev = evaluate(
                    model, eval_batches, amp=args.amp, device=device, message_override=override
                )
                trace.append({"step": step, **ev, "sec": time.time() - t0})
                print(
                    f"  [{arch}] step {step:5d}  train {float(loss.detach()):.4f}  "
                    f"eval CE {ev['ce_nats']:.4f}  acc {ev['acc']:.3f}  "
                    f"({(time.time() - t0) / step:.2f} s/step)",
                    flush=True,
                )
                best_acc = max(best_acc, ev["acc"])
                if ev["acc"] >= args.early_stop_acc:
                    print(f"  [{arch}] early stop at {step} (acc {ev['acc']:.3f})", flush=True)
                    break
                if (
                    not k1_extended
                    and step >= steps
                    and ev["acc"] < SOLVABLE_ACC
                    and _ce_still_falling(trace)
                ):
                    k1_extended = True
                    max_steps = step + steps
                    print(
                        f"  [{arch}] eval CE still falling (≥0.2 nats in last third); "
                        f"extending to {max_steps} (small-model protocol)",
                        flush=True,
                    )
    cache = arch_cache(arch, spec, cfg.seq_len)
    final = trace[-1]
    report = info_report(
        ce_nats=final["ce_nats"],
        acc=final["acc"],
        n_supervised=final["tokens"],
        cfg=cfg,
        window=spec.local_window,
        nominal_b_tokens=cache["nominal_b_tokens"],
        nominal_a_bytes=cache["nominal_a_bytes"],
    )
    extra = {}
    if arch in ("e21", "e30"):
        extra["channel_ablations"] = _channel_ablations(
            model, eval_batches, amp=args.amp, device=device, spec=spec
        )
        extra["slot_geometry"] = _slot_rankme(model, eval_batches, amp=args.amp, device=device)
        extra["write_geometry"] = _write_geometry(model, cfg.seq_len)
        ae = getattr(model, "_last_prefix_ae", None)
        if ae:
            extra["prefix_ae"] = ae
            print(
                f"  [{arch}] prefix_ae loss {ae.get('loss')}  tok_acc {ae.get('tok_acc')}  "
                f"key_acc {ae.get('key_acc')}",
                flush=True,
            )
        wg = extra["write_geometry"]
        print(
            f"  [{arch}] ablations "
            + str({k: v.get("acc") if isinstance(v, dict) else v for k, v in extra["channel_ablations"].items()})
            + f"  rankme {extra['slot_geometry'].get('slot_rankme')}"
            + (f"  n_win {wg.get('n_windows')} C {wg.get('n_slots')} H/logW {wg.get('entropy_over_logW')}" if wg else ""),
            flush=True,
        )
    return {
        "arch": arch,
        "params": params,
        "cache": cache,
        "final": final,
        "best_acc": best_acc,
        "trace": trace,
        "info": report.as_dict(),
        "early_stopped": final["acc"] >= args.early_stop_acc,
        "k1_extended": k1_extended,
        **extra,
    }


def run_rung(task: str, args, *, recipe_name: str | None = None) -> dict:
    scale = SCALES[args.scale]
    recipe = resolve_recipe(recipe_name or task)
    display = recipe.name if recipe_name else task
    over = dict(recipe.overrides)
    cli = {
        "n_distractors": args.n_distractors,
        "key_len": args.key_len,
        "value_len": args.value_len,
        "hops": args.hops,
        "span_len": args.span_len,
        "min_gap": args.min_gap,
        "seq_len": args.seq_len,
    }
    if recipe.family == "glyph" or recipe.task in {
        "copy_span", "reverse", "every_k", "filter_mod", "dyck_close",
        "fact_markov", "story_fact", "chain_ordered_noise", "chain_shuffled_noise",
    }:
        cli.update({"width": args.width, "noise": args.noise, "k": args.every_k, "modulus": args.modulus})
    else:
        cli.update({"n_decoys": args.n_decoys, "evidence_align": args.evidence_align})
    over.update({k: v for k, v in cli.items() if v is not None})
    cfg = config_for(scale, recipe.task, **over)
    window = args.local_window if args.local_window is not None else scale.local_window
    spec = ArchSpec(
        name="shared",
        hidden=args.hidden,
        pre_layers=args.pre_layers,
        global_layers=args.global_layers,
        stack_layers=args.stack_layers,
        local_window=window,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        head_dim=args.head_dim,
        n_kv_heads=args.kv_heads,
        value_embed_layers=tuple(int(x) for x in args.value_embed_layers.split(",") if x.strip()),
        attn_backend=args.attn_backend,
        global_logit_scale=args.global_logit_scale,
        z_loss=args.z_loss,
        zero_init_residuals=not args.warm_residuals,
        message_compress_ratio=args.message_ratio,
        message_boundary_token_id=_boundary_token_id(cfg),
        message_pool_remainder=args.message_pool_remainder,
        message_slots_inplace=args.message_slots_inplace,
        message_inplace_raw_kv=args.message_inplace_raw_kv,
        message_identity_slots=args.message_identity_slots,
        message_pack_stride=args.message_pack_stride,
        message_keep_local_swa=args.message_keep_local_swa,
        message_extra_slot_attends=args.message_extra_slot_attends,
        message_update_slot_kv=args.message_update_slot_kv,
        message_global_anchors=args.message_global_anchors,
        message_anchor_token_ids=(
            _type_mark_token_ids(cfg)
            if args.message_global_anchors in ("type_marks", "query_nbhd+type")
            else (_keymark_token_ids(cfg) if args.message_global_anchors == "key_spans" or args.message_prefix_ae else ())
        ),
        message_anchor_key_len=int(cfg.key_len) if args.message_global_anchors == "key_spans" or args.message_prefix_ae else 0,
        message_prefix_ae=args.message_prefix_ae,
        message_prefix_ae_weight=args.message_prefix_ae_weight,
        message_prefix_ae_stopgrad_answer=args.message_prefix_ae_stopgrad_answer,
        message_write="sw_perceiver" if "e30" in args.arch else "block_mean",
        swp_bank_size=args.swp_bank_size,
        swp_coverage=args.swp_coverage,
        swp_window=args.swp_window,
        swp_stride=args.swp_stride,
        swp_n_heads=args.swp_n_heads,
        swp_query_dim=args.swp_query_dim,
        swp_auto_fit=args.swp_auto_fit,
    )
    card = rung_card(scale, recipe.task, **over)
    card["local_window"] = window
    card["floor_nats"] = _floor(cfg, window)
    eval_rng = np.random.default_rng(args.seed + 99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_batches = [make_batch(cfg, eval_rng, args.batch, device) for _ in range(max(1, args.eval_rows // args.batch))]
    gap_probe = [generate_row_for(cfg, np.random.default_rng(args.seed + 7 + i)).gap for i in range(16)]
    print(
        f"\n=== {display} ({recipe.task}) @ {scale.name}  seq={cfg.seq_len} gap={cfg.min_gap} "
        f"window={window} prize={card['prize_bits']:.2f} bits  "
        f"answer_len={cfg.answer_len} (target {card['target_answer_len']})  "
        f"floor={card['floor_nats']:.4f} nats  chance={card['chance_acc']:.3f}  "
        f"device={device} amp={args.amp}  "
        f"row_gap[min/med/max]={min(gap_probe)}/{int(np.median(gap_probe))}/{max(gap_probe)} ===",
        flush=True,
    )
    arches = list(args.arch)
    if args.dense_first and "dense" in arches:
        arches = ["dense"] + [a for a in arches if a != "dense"]
    results = {}
    dense_steps_used = args.steps
    skip_rest = False
    for arch in arches:
        if skip_rest:
            print(f"--- {arch} skipped (dense < {SOLVABLE_ACC:.0%} at K1; rung uncalibrated) ---", flush=True)
            continue
        if arch == "dense":
            arch_steps = args.steps * args.k1_mult
        else:
            arch_steps = max(args.steps, dense_steps_used)
        print(f"--- {arch}  (≤ {arch_steps} steps) ---", flush=True)
        results[arch] = train_one(arch, cfg, args, eval_batches, spec, device, steps=arch_steps)
        print(
            f"  {arch}: {results[arch]['params']/1e6:.3f}M  acc {results[arch]['final']['acc']:.3f}  "
            f"flow {results[arch]['info']['information_flow']:.3f}  "
            f"a_bits {results[arch]['info']['effective_a_bits']:.2f}  "
            f"B/tok {results[arch]['info']['bytes_per_input_token']:.4g}",
            flush=True,
        )
        if arch == "dense":
            dense_steps_used = results[arch]["final"]["step"]
            if args.skip_uncalibrated and results[arch]["final"]["acc"] < SOLVABLE_ACC:
                skip_rest = True
    calibrated = True
    notes = []
    if "dense" in results:
        dacc = results["dense"]["final"]["acc"]
        ok = dacc >= SOLVABLE_ACC
        notes.append(("dense >= 75% (task is solvable here)", ok, f"acc={dacc:.3f}"))
        calibrated = calibrated and ok
    if "e18_local" in results and recipe.task not in {"count", "majority"}:
        lacc = results["e18_local"]["final"]["acc"]
        # Local arm should not substantially beat chance on retrieval rungs.
        leak = lacc > card["chance_acc"] + 0.15
        notes.append(("e18_local near chance (task does not leak)", not leak, f"acc={lacc:.3f}"))
        if leak:
            calibrated = False
    print()
    for name, ok, detail in notes:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return {
        "task": display,
        "generator_task": recipe.task,
        "recipe": recipe.name,
        "scale": scale.name,
        "card": card,
        "calibrated": calibrated,
        "results": results,
        "hidden": args.hidden,
        "steps": args.steps,
        "k1_mult": args.k1_mult,
        "dense_steps_used": dense_steps_used,
        "batch": args.batch,
        "seed": args.seed,
        "amp": args.amp,
        "hunt": {
            "seq_len": cfg.seq_len,
            "min_gap": cfg.min_gap,
            "local_window": window,
            "kv_heads": args.kv_heads,
            "global_logit_scale": args.global_logit_scale,
            "attn_backend": args.attn_backend,
            "hidden": args.hidden,
            "head_dim": args.head_dim,
            "lr": args.lr,
            "global_layers": args.global_layers,
            "stack_layers": args.stack_layers,
            "warm_residuals": args.warm_residuals,
            "message_ratio": args.message_ratio,
            "message_override": args.message_override,
            "message_pool_remainder": args.message_pool_remainder,
            "message_slots_inplace": args.message_slots_inplace,
            "message_inplace_raw_kv": args.message_inplace_raw_kv,
            "message_identity_slots": args.message_identity_slots,
            "message_pack_stride": args.message_pack_stride,
            "message_keep_local_swa": args.message_keep_local_swa,
            "message_extra_slot_attends": args.message_extra_slot_attends,
            "message_update_slot_kv": args.message_update_slot_kv,
            "message_global_anchors": args.message_global_anchors,
            "message_prefix_ae": args.message_prefix_ae,
            "message_prefix_ae_weight": args.message_prefix_ae_weight,
            "swp_bank_size": args.swp_bank_size,
            "swp_coverage": args.swp_coverage,
            "swp_window": args.swp_window,
            "swp_stride": args.swp_stride,
            "swp_n_heads": args.swp_n_heads,
            "swp_query_dim": args.swp_query_dim,
            "swp_auto_fit": args.swp_auto_fit,
            "k1_extended": {
                a: bool(r.get("k1_extended", False)) for a, r in results.items()
            },
        },
        "pack": {
            "answer_len": cfg.answer_len,
            "key_len": cfg.key_len,
            "value_len": cfg.value_len,
            "span_len": cfg.span_len,
            "n_distractors": cfg.n_distractors,
            "n_decoys": cfg.n_decoys,
            "hops": cfg.hops,
            "prize_bits": card["prize_bits"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="tiny", choices=list(SCALES))
    p.add_argument("--task", nargs="+", default=list(TINY_PROOF_TASKS))
    p.add_argument(
        "--recipe",
        nargs="+",
        default=None,
        help="named recipes (far_copy, recall_single, … or Glyph reverse/every_k/filter_mod/…). "
        "Overrides --task. Use calibrated DNA recipes to score E18; Glyph rungs are uncalibrated until dense ≥ 75%.",
    )
    p.add_argument("--arch", nargs="+", default=["dense", "e18", "encdec"], choices=list(ARCHES))
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pre_layers", type=int, default=1)
    p.add_argument(
        "--global_layers",
        type=int,
        default=1,
        help="Sequential full Attention+FFN global blocks (E18 raw / E21 exclusive "
        "slots). Default 1 (E18-loadable). Distinct from --stack_layers (SWA) and "
        "from --message_extra_slot_attends (extra attends inside one Attention).",
    )
    p.add_argument(
        "--stack_layers",
        type=int,
        default=2,
        help="SWA local stack after the global read(s). Default 2. Not a second global.",
    )
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--dec_layers", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument(
        "--kv_heads",
        type=int,
        default=1,
        help="KV heads (GQA). 0 or a non-divisor of Q heads → full MHA. Default 1 matches the tiny ladder.",
    )
    p.add_argument(
        "--global_logit_scale",
        default="none",
        choices=("none", "log"),
        help="SSMax-style log(n_visible) query scale on full layers (anti-dilution at 4k+).",
    )
    p.add_argument("--attn_backend", default="sdpa", choices=("sdpa", "flex", "flash"))
    p.add_argument("--z_loss", type=float, default=1e-4)
    p.add_argument(
        "--warm_residuals",
        action="store_true",
        help="Do not zero-init attn.wo / mlp.down. Needed so a 512+ needle can open the residual read.",
    )
    p.add_argument(
        "--message_ratio",
        type=int,
        default=16,
        help="E21 KVCompressor ratio (prefix tokens per slot). Ignored for other arches.",
    )
    p.add_argument(
        "--message_override",
        default="real",
        choices=("real", "none", "swapped", "raw"),
        help="E21 global-read channel: real slots (default), none (floor), swapped (wrong row), "
        "raw (uncompressed prefix K/V across QUERY; local SWA still severed). Ignored for other arches.",
    )
    p.add_argument(
        "--message_pool_remainder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: pool the incomplete last sender block. Default off (experiment 1 complete-block-only).",
    )
    p.add_argument(
        "--message_slots_inplace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: write slot K/V into sender prefix positions (KV_LEN=S, no concat extra stream). "
        "Default off (concat slots). Receivers still cannot see uncompressed sender tokens.",
    )
    p.add_argument(
        "--message_inplace_raw_kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: with --message_slots_inplace, copy token K/V into replace positions "
        "(skip compressor values). Exclusive ~replace mask stays on. Default off.",
    )
    p.add_argument(
        "--message_identity_slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: bypass KVCompressor u/delta (frozen mean; r=1 is a hard token K/V copy). "
        "Still goes through the slot/scatter path. Default off.",
    )
    p.add_argument(
        "--message_pack_stride",
        type=int,
        default=0,
        help="E21: exclusive leftover vs N-token packs tiled to end at QUERY "
        "(left leftover after BOS dropped from exclusive identity slots). "
        "0=off (default; r=1 identity keeps every sender token). "
        "32=DNA packed-answer stride. Remainder-on keeps leftover as identity slots. "
        "--message_pool_remainder is a no-op at r=1 without this flag. E18-loadable.",
    )
    p.add_argument(
        "--message_keep_local_swa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: do not treat QUERY as a SWA/n-gram document start. Local window still "
        "sees raw prefix tokens inside the sliding window. Exclusive compressed/identity "
        "slots stay on the global read. Default off (severed local path; E18-loadable).",
    )
    p.add_argument(
        "--message_extra_slot_attends",
        type=int,
        default=0,
        help="E21: extra exclusive global attends over the *same* frozen slot K/V "
        "(queries update from the previous hop). Default 0. Not DNA --hops, not raw prefix KV.",
    )
    p.add_argument(
        "--message_update_slot_kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E21: rewrite exclusive slot K/V from the post-attend residual before each "
        "extra hop (queries and slot keys update). Default off (frozen slot K/V). "
        "No-op when --message_extra_slot_attends is 0. Still not raw prefix KV.",
    )
    p.add_argument(
        "--message_global_anchors",
        default="none",
        choices=("none", "query_nbhd", "query_side", "type_marks", "query_nbhd+type", "key_spans"),
        help="E21: which extra positions join exclusive slot K/V as raw keys. "
        "none (default): prior exclusive slots only, E18-loadable. "
        "type_marks: keymark/decoy/spanmark/hop/mark control tokens. "
        "query_nbhd: 4 sender tokens immediately before QUERY (already r=1 slots). "
        "query_side: QUERY plus a small window after the message boundary "
        "(receiver type request; not prefix replace slots). "
        "query_nbhd+type: union of prefix-nbhd and type_marks. "
        "key_spans: DNA key-field tokens after each sender keymark (E27 hybrid b). "
        "Sparse subset (count << seq), not the full raw prefix (that is E18).",
    )
    p.add_argument(
        "--message_prefix_ae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="E26: weak linear reconstruction of each complete sender block from its slot. "
        "Default off (E18-loadable). Pair with identity_slots off so u/delta can move under AE.",
    )
    p.add_argument(
        "--message_prefix_ae_weight",
        type=float,
        default=1.0,
        help="λ on L_AE when --message_prefix_ae is on. Default 1.0.",
    )
    p.add_argument(
        "--message_prefix_ae_stopgrad_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="E26: detach slots on the exclusive read so compressor u/delta see AE grads only. "
        "Default on.",
    )
    p.add_argument(
        "--swp_bank_size",
        type=int,
        default=32,
        help="E30: learned queries per window (K). Auto-fit may shrink this on short seq.",
    )
    p.add_argument("--swp_coverage", type=int, default=8, help="E30: W/K when --swp_window is 0.")
    p.add_argument("--swp_window", type=int, default=0, help="E30: window W in tokens (0 = coverage*K).")
    p.add_argument("--swp_stride", type=int, default=0, help="E30: stride (0 = 0.75*W).")
    p.add_argument("--swp_n_heads", type=int, default=0, help="E30: write heads (0 = max(4, Q heads)).")
    p.add_argument("--swp_query_dim", type=int, default=0, help="E30: scoring dim (0 = max(head_dim, 4*token_emb)).")
    p.add_argument(
        "--swp_auto_fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="E30: shrink K so n_windows≥2 on short sequences (default on).",
    )
    p.add_argument(
        "--experiment_id",
        default=None,
        help="W&B / JSON identity (E27, E26, …). Used when --wandb is on.",
    )
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log the probe JSON summary to W&B (WANDB_PROJECT from env). Default off.",
    )
    p.add_argument("--seq_len", type=int, default=None, help="override scale seq_len (S0 hunts)")
    p.add_argument("--min_gap", type=int, default=None, help="override scale min_gap (S0 hunts)")
    p.add_argument("--local_window", type=int, default=None, help="override scale local_window")
    p.add_argument(
        "--evidence_align",
        default=None,
        choices=("spread", "right"),
        help="spread (default) or pack evidence against min_gap (near-copy S0 at long seq)",
    )
    p.add_argument(
        "--sdpa_math",
        action="store_true",
        help="Force PyTorch MATH SDPA (disable flash/mem-efficient kernels).",
    )
    p.add_argument(
        "--value_embed_layers",
        default="0,1",
        help="layer indices with value embeddings; 0,1 puts VE on E18's global read",
    )
    p.add_argument("--steps", type=int, default=800, help="advertised step budget per arch")
    p.add_argument(
        "--k1_mult",
        type=int,
        default=4,
        help="dense may train up to steps*k1_mult (K1: 4× before declaring a rung ill-posed)",
    )
    p.add_argument(
        "--dense_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train dense first; skip other arches if it misses 75%% (default on)",
    )
    p.add_argument(
        "--skip_uncalibrated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do not score E18/e21/e30/encdec on a rung whose dense control missed 75%%",
    )
    p.add_argument("--n_distractors", type=int, default=None)
    p.add_argument("--n_decoys", type=int, default=None)
    p.add_argument("--key_len", type=int, default=None)
    p.add_argument("--value_len", type=int, default=None)
    p.add_argument("--hops", type=int, default=None)
    p.add_argument("--span_len", type=int, default=None)
    p.add_argument("--width", type=int, default=None, help="Glyph vocab width 16 or 32 (ignored for DNA)")
    p.add_argument("--noise", default=None, help="Glyph noise: markov|dyck|arith|mixed|iid")
    p.add_argument("--every_k", type=int, default=None, help="Glyph every_k stride")
    p.add_argument("--modulus", type=int, default=None, help="Glyph filter_mod modulus")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_rows", type=int, default=64)
    p.add_argument("--early_stop_acc", type=float, default=0.99)
    p.add_argument("--max_params", type=int, default=100_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--amp",
        default="auto",
        choices=("auto", "bf16", "fp16", "off"),
        help="CUDA autocast. auto=bf16 when supported, off on CPU. Use off to match the CPU tiny numbers bit-for-bit.",
    )
    p.add_argument("--out", default=None, help="directory for JSON bundles (one file per task)")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    names = list(args.recipe) if args.recipe else list(args.task)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skip --wandb", flush=True)
        else:
            run_name = args.experiment_id or (out_dir.name if out_dir else "bapo_probe")
            try:
                wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT"),
                    entity=os.environ.get("WANDB_ENTITY", "ksopyla"),
                    name=run_name,
                    group=args.experiment_id or run_name,
                    job_type="bapo_probe",
                    config={
                        "experiment_id": args.experiment_id,
                        "scale": args.scale,
                        "recipes": names,
                        "arch": args.arch,
                        "hidden": args.hidden,
                        "message_global_anchors": args.message_global_anchors,
                        "message_prefix_ae": args.message_prefix_ae,
                        "message_ratio": args.message_ratio,
                        "message_identity_slots": args.message_identity_slots,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — probe must still train
                print(f"wandb.init failed ({exc}); continue without W&B", flush=True)
                wandb_run = None
            else:
                print(f"W&B run: {wandb_run.url} id={wandb_run.id}", flush=True)
                if out_dir:
                    (out_dir / "wandb_run.txt").write_text(
                        f"id={wandb_run.id}\nurl={wandb_run.url}\n"
                    )

    bundles = []
    for name in names:
        bundle = run_rung(name, args, recipe_name=name)
        bundles.append(bundle)
        if out_dir:
            path = out_dir / f"{args.scale}_{bundle['task']}.json"
            path.write_text(json.dumps(bundle, indent=2))
            print(f"wrote {path}")

    if out_dir:
        summary = {
            "scale": args.scale,
            "recipes": names,
            "tasks": [b["task"] for b in bundles],
            "arches": args.arch,
            "solvable_acc": SOLVABLE_ACC,
            "calibrated_recipes": list(CALIBRATED_RECIPES),
            "glyph_core_recipes": list(GLYPH_CORE_RECIPES),
            "amp": args.amp,
            "rungs": [
                {
                    "task": b["task"],
                    "generator_task": b.get("generator_task"),
                    "calibrated": b["calibrated"],
                    "acc": {a: r["final"]["acc"] for a, r in b["results"].items()},
                    "information_flow": {a: r["info"]["information_flow"] for a, r in b["results"].items()},
                    "recovered_bits": {a: r["info"]["recovered_bits"] for a, r in b["results"].items()},
                    "bytes_per_input_token": {a: r["info"]["bytes_per_input_token"] for a, r in b["results"].items()},
                    "params": {a: r["params"] for a, r in b["results"].items()},
                }
                for b in bundles
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_dir / 'summary.json'}")
    if wandb_run is not None:
        import wandb

        payload = {}
        for b in bundles:
            for arch, r in b["results"].items():
                prefix = f"{b['task']}/{arch}"
                payload[f"{prefix}/acc"] = r["final"]["acc"]
                payload[f"{prefix}/recovered_bits"] = r["info"]["recovered_bits"]
                payload[f"{prefix}/information_flow"] = r["info"]["information_flow"]
                if "slot_geometry" in r:
                    payload[f"{prefix}/slot_rankme"] = r["slot_geometry"].get("slot_rankme")
                if "prefix_ae" in r:
                    payload[f"{prefix}/ae_key_acc"] = r["prefix_ae"].get("key_acc")
                for mode, ev in (r.get("channel_ablations") or {}).items():
                    if isinstance(ev, dict) and "acc" in ev:
                        payload[f"{prefix}/ablation_{mode}_acc"] = ev["acc"]
        wandb.log(payload)
        wandb.finish()
    return 0 if all(b["calibrated"] for b in bundles) else 2


if __name__ == "__main__":
    raise SystemExit(main())

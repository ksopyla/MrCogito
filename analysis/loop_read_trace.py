#!/usr/bin/env python
"""Where does the memory read look? Per loop round, at the position that predicts the first answer letter.

For a probe checkpoint (`<arch>.pt` saved by `verification/bapo_capability_probe.py --save_ckpt`) of an E31 /
E33a model on a chain (`chain`, `chain_parallel`) or keyed recall (`recall`) exam, the script regenerates rows of
the same exam, runs the model with R = 1 … R rounds, and splits the global read's slot attention (mean over query
heads) by what each memory entry's latent read:

  hop1 … hop4   the asked chain's edges in path order (recall: hop1 = the asked fact)
  other_chain   any other edge / fact
  filler        latents with < 30 % of their read weight on an edge block

A model that follows the chain puts round r's mass on hop r; a model that answers from a shortcut spreads it
evenly over edges. `raw` is the share on the question side's own tokens. Reference: E33a diagnosis,
docs/4_Research_Notes/e33a_loop_diagnosis_20261004.md (§3).

  uv run python analysis/loop_read_trace.py --ckpt Cache/study/.../e33a_pchain3_loop_s2/e31_li_m1.pt --rows 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nn.perceiver_ar_lm as par  # noqa: E402
from data.bapo_ladder import config_for, generate_row_for  # noqa: E402
from evaluation.bapo_models import ArchSpec, build_model  # noqa: E402

CATS = ("hop1", "hop2", "hop3", "hop4", "other_chain", "filler")


def load(ckpt: Path, overrides: dict):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = st["data"]
    cfg = config_for(meta["scale"], meta["task"], **{**meta["over"], **overrides})
    spec = ArchSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in st["spec"].items()})
    model = build_model(st["arch"], vocab_size=meta["vocab_size"], seq_len=cfg.seq_len, answer_start=cfg.answer_start,
                        pad_id=meta["pad_id"], bos_id=meta["bos_id"], eos_id=meta["eos_id"], spec=spec, seed=0)
    _, unexpected = model.load_state_dict(st["state_dict"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in {ckpt}: {unexpected[:4]}")
    return model.eval(), cfg


def blocks(cfg, toks: np.ndarray, nodes: list[tuple]) -> tuple[list[tuple[int, str]], int]:
    """[(block start, class)], block length — classes for chain edges or recall facts."""
    v, L = cfg.vocab, cfg.key_len
    out = []
    if cfg.task == "recall":
        qkey = tuple(toks[cfg.answer_start - 1 - L:cfg.answer_start - 1])
        for p in np.nonzero(toks == v.control("keymark"))[0]:
            out.append((int(p), "hop1" if tuple(toks[p + 1:p + 1 + L]) == qkey else "other_chain"))
        return out, 1 + L + cfg.value_len
    path = {(nodes[i], nodes[i + 1]): f"hop{i + 1}" for i in range(len(nodes) - 1)}
    for p in np.nonzero(toks == v.control("hop"))[0]:
        edge = (tuple(toks[p + 1:p + 1 + L]), tuple(toks[p + 1 + L:p + 1 + 2 * L]))
        out.append((int(p), path.get(edge, "other_chain")))
    return out, 1 + 2 * L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="<arch>.pt from the probe (or its directory)")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hops", type=int, default=None, help="override the exam's hops")
    ap.add_argument("--chain_overhang", type=int, default=None, help="override the exam's overhang")
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    if ckpt.is_dir():
        ckpt = next(ckpt.glob("*.pt"))
    over = {k: v for k, v in (("hops", args.hops), ("chain_overhang", args.chain_overhang)) if v is not None}
    model, cfg = load(ckpt, over)
    R = int(getattr(model, "loop_rounds", 1) or 1)
    geo = model.memory_writer.geometry(cfg.seq_len)

    calls: list[dict] = []
    orig_attend = par.attend_message

    def spy(q, k, v, k_bar, v_bar, *, ctx, key_valid, backend, block_masks=None, q_slot=None):
        calls.append(dict(q=q.detach(), k=k.detach(), k_bar=k_bar.detach(), ctx=ctx, key_valid=key_valid))
        return orig_attend(q, k, v, k_bar, v_bar, ctx=ctx, key_valid=key_valid, backend=backend,
                           block_masks=block_masks, q_slot=q_slot)

    w_store: dict = {}
    orig_read = model.memory_writer._latent_read

    def read_spy(z, x, mask, W):
        w, out = orig_read(z, x, mask, W)
        w_store["w"] = w.detach()
        return w, out

    par.attend_message = spy
    model.memory_writer._latent_read = read_spy
    agg = {r: dict.fromkeys(CATS, 0.0) for r in range(1, R + 1)}
    top = {r: dict.fromkeys(CATS, 0) for r in range(1, R + 1)}
    raw = dict.fromkeys(range(1, R + 1), 0.0)
    acc = dict.fromkeys(range(1, R + 1), 0)
    rng = np.random.default_rng(args.seed)
    t = cfg.answer_start - 1
    try:
        for _ in range(args.rows):
            row = generate_row_for(cfg, rng)
            toks = row.input_ids
            nodes = [tuple(n) for n in (row.meta or {}).get("nodes", [])]
            blk, blen = blocks(cfg, toks, nodes)
            ids = torch.from_numpy(toks[None]).long()
            for r in range(1, R + 1):
                calls.clear()
                model._loop_rounds_override = r if R > 1 else None
                with torch.no_grad():
                    logits = model(ids, return_logits=True).logits
                acc[r] += int(logits[0, t].argmax() == int(toks[t + 1]))
                c = calls[-1]  # the last global read of an R = r forward is round r's read
                S = c["q"].shape[1]
                mask = par.dense_message_mask(S, c["ctx"], c["key_valid"], c["q"].device)[0, 0, t]
                K = torch.cat([c["k"], c["k_bar"].to(c["k"].dtype)], dim=1)[0, :, 0].float()
                lg = torch.einsum("hd,kd->hk", c["q"][0, t].float(), K) / c["q"].shape[-1] ** 0.5
                p = torch.softmax(lg.masked_fill(~mask[None], float("-inf")), -1).mean(0)
                raw[r] += float(p[:S].sum())
                slots = p[S:]
                w = w_store["w"][:, :, : geo.latents].float().mean(1)  # [n_windows, K, W] (one row)
                cls = []
                for wi, s0 in enumerate(geo.starts):
                    for kk in range(geo.latents):
                        best, best_m = "filler", 0.0
                        for b0, name in blk:
                            lo, hi = max(b0 - s0, 0), min(b0 + blen - s0, w.shape[-1])
                            if hi > lo:
                                mm = float(w[wi, kk, lo:hi].sum())
                                if mm > best_m:
                                    best, best_m = name, mm
                        cls.append(best if best_m >= 0.3 else "filler")
                for j, name in enumerate(cls[: slots.shape[0]]):
                    agg[r][name] += float(slots[j])
                top[r][cls[int(slots.argmax())]] += 1
    finally:
        par.attend_message = orig_attend
        model._loop_rounds_override = None
    n = args.rows
    print(f"{ckpt.parent.name}: task={cfg.task} hops={getattr(cfg, 'hops', '-')} rounds={R} rows={n} "
          f"memory entries={geo.n_slots}")
    for r in range(1, R + 1):
        print(f"  round {r}: raw {raw[r] / n:.2f} | " + "  ".join(f"{c_} {agg[r][c_] / n:.3f}" for c_ in CATS)
              + " | top entry " + " ".join(f"{c_}:{top[r][c_]}" for c_ in CATS if top[r][c_])
              + f" | first letter {acc[r] / n:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

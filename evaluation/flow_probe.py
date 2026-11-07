"""E33p instrumentation, part B (plan §8 B): the same held-out rows through every arm, compared after training.

Per arm and checkpoint, on 16 LongDocs-lite rows (round-robin over the sources, the long-document eval's rows,
seed 7), at 512 query positions per row ≥ 2k (half of them far-copy positions, `copy_split.copy_sources`, k = 3):
  B1 same choices — the dense ceiling (run first, `--save_ref`) stores each query's and loop round's top-256 raw keys
     beyond the 512-token window with their attention (head mean); every other arm reports the recall of the dense
     top-32 in what it opened (notebook arms: the picked tokens of the 2 pages with the most note mass), and the dense
     mass that falls in that set (within the dense top-256) next to the dense mass of the whole top-256;
  B2 copy-source hit rate — on far-copy queries, whether the copy's source (the earlier token, or the one before it)
     is in that set, and every arm's own attention on the source;
  B3 CKA inputs — each block's (and loop round's) output at 2,048 fixed tokens per row (fp16, `--acts`); CKA between
     arms and across rounds: scripts/compare_flow.py;
  B4 logit lens (loss read from every block, last 1,024 positions) and the reading layer's attention by distance
     (log₂ buckets), plus the training-time read / select / writer scalars on these rows.

Usage (the dense ceiling first):
  uv run python evaluation/flow_probe.py --checkpoint DENSE/final --data DATA/longdocs_lite_gpt2 --length 16384 \
      --save_ref ref.pt --acts dense_acts.pt --out dense.json
  uv run python evaluation/flow_probe.py --checkpoint ARM/final --data ... --dense_ref ref.pt --acts arm_acts.pt \
      --out arm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn.flow_stats import FlowRecorder  # noqa: E402

N_DIST = 18  # log₂ distance buckets (up to 128k)


class ProbeRecorder(FlowRecorder):
    """FlowRecorder on fixed query rows, which also keeps per-query sets and the layer outputs at fixed tokens."""

    def __init__(self, q_rows: torch.Tensor, tok_rows: torch.Tensor, *, ref: dict | None, save_ref: bool,
                 far_src: torch.Tensor, window: int = 512):
        super().__init__(n_sample=len(q_rows), lens_tokens=1024)
        self.q_rows, self.tok_rows, self.ref, self.save_ref = q_rows, tok_rows, ref, save_ref
        self.far_src, self.window = far_src, window      # far_src [Q]: the copy source of the query's target, −1 none
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.dist = defaultdict(lambda: torch.zeros(N_DIST, dtype=torch.float64))
        self.new_ref: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.acts: dict[str, torch.Tensor] = {}

    def _sample_rows(self, S, device):
        rows = self.q_rows.to(device)
        return rows, [("probe", slice(0, rows.numel()))]

    def residual(self, layer_idx, x):
        super().residual(layer_idx, x)
        self.acts[self._tag(f"L{layer_idx}")] = x[0, self.tok_rows.to(x.device)].detach().half().cpu()

    @torch.no_grad()
    def _capture(self, *, kind, rows, p, S, opened, ctx, index):
        r = int(self.round or 0)
        pr = p.mean(0)[:, :S].float()                                                     # [Q, S] head mean
        j = torch.arange(S, device=pr.device)
        dist = (rows[:, None] - j[None]).clamp(min=0)
        b = torch.log2(dist.float() + 1).floor().long().clamp(max=N_DIST - 1)
        self.dist[r] += torch.zeros(N_DIST, device=pr.device).scatter_add_(0, b.flatten(), pr.flatten()).double().cpu()
        far_ok = dist >= self.window
        if self.save_ref:
            top = pr.masked_fill(~far_ok, -1.0).topk(min(256, S), dim=-1)
            self.new_ref[r] = (top.indices.to(torch.int32).cpu(), top.values.clamp(min=0).half().cpu())
        # the set this arm reads beyond its window
        if kind == "notebook" and index is None and ctx is not None and ctx.override not in ("none", "raw"):
            sel = self._top_pages(p, S, ctx, rows)
        elif kind == "notebook" and (index is None or ctx is None or ctx.override in ("none", "raw")):
            sel = None                                                                     # window control
        else:
            sel = opened & far_ok
        src = self.far_src.to(pr.device)
        has = src >= 0
        if has.any():
            s0 = src.clamp(min=0)
            sm = pr.gather(1, s0[:, None]).squeeze(1) + pr.gather(1, (s0 - 1).clamp(min=0)[:, None]).squeeze(1)
            self.metrics[f"r{r}/src_mass"].append(float(sm[has].mean()))
            if sel is not None:
                hit = sel.gather(1, s0[:, None]).squeeze(1) | sel.gather(1, (s0 - 1).clamp(min=0)[:, None]).squeeze(1)
                self.metrics[f"r{r}/src_hit"].append(float(hit[has].float().mean()))
        if self.ref is not None and r in self.ref and sel is not None:
            pos, prob = (t.to(pr.device) for t in self.ref[r])
            inside = sel.gather(1, pos.long())                                            # [Q, 256]
            pf = prob.float()
            ok = (pf[:, :32] > 0).float()                                                  # keys beyond the window only
            self.metrics[f"r{r}/recall32"].append(float((inside[:, :32] * ok).sum() / ok.sum().clamp_min(1)))
            self.metrics[f"r{r}/dense_mass_in_set"].append(float((pf * inside).sum(-1).mean()))
            self.metrics[f"r{r}/dense_mass_top256"].append(float(pf.sum(-1).mean()))

    def _top_pages(self, p, S, ctx, rows):
        """Notebook arms without opening: the picked tokens of the 2 pages whose notes take the most attention."""
        C = ctx.n_slots
        n_w, W = ctx.window_valid.shape[1], ctx.window_valid.shape[2]
        K = C // n_w
        nm = p.mean(0)[:, S:S + C].float().view(len(rows), n_w, K).sum(-1)                # [Q, n_w]
        top = nm.topk(min(2, n_w), dim=-1).indices
        pick = ctx.window_valid[0]                                                         # [n_w, W]
        sel = torch.zeros(len(rows), S, dtype=torch.bool, device=p.device)
        for t in range(top.shape[1]):
            w = top[:, t]
            sel.scatter_(1, self._page_tok[w], pick[w])                                   # [Q, W] positions
        return sel


def pick_queries(ids: np.ndarray, S: int, n: int, seed: int, min_pos: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """n query positions in [min_pos, S − 1): half far-copy queries (their target is a far copy), half random;
    and each query's copy source (−1 none)."""
    from evaluation.copy_split import copy_sources

    cat, src = copy_sources(ids[:S], 3, 512)
    rng = np.random.default_rng(seed)
    cand = np.arange(min_pos, S - 1)
    far = cand[cat[cand + 1] == 2]
    nf = min(n // 2, len(far))
    q_far = rng.choice(far, nf, replace=False) if nf else np.zeros(0, dtype=np.int64)
    rest = np.setdiff1d(cand, q_far)
    q = np.sort(np.concatenate([q_far, rng.choice(rest, n - nf, replace=False)]))
    s = np.where(cat[q + 1] == 2, src[q + 1], -1)
    return q, s


def passkey_row(tokenizer, filler: list[int], S: int, depth: float, rng) -> tuple[list[int], np.ndarray, np.ndarray,
                                                                                   list[int]]:
    """A planted-password row (long_context_probes.build_passkey: needle at `depth`, question and answer at the end)
    and its probe queries: the position before each answer token, and as each query's source the same token inside
    the needle. Returns (ids [S], queries [n], sources [n], answer [n])."""
    from evaluation.long_context_probes import build_passkey

    ids, answer = build_passkey(tokenizer, filler, S, depth, rng)
    n = len(answer)
    ans_at = len(ids) - n
    found = [i for i in range(ans_at - n) if ids[i:i + n] == answer]
    if not found:
        raise ValueError("the answer's tokens do not appear inside the needle")
    q = np.arange(ans_at - 1, ans_at - 1 + n)
    return ids, q, np.arange(found[0], found[0] + n), answer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--queries", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=2048, help="fixed tokens per row for the CKA inputs")
    ap.add_argument("--dense_ref", help="the dense ceiling's --save_ref file (B1)")
    ap.add_argument("--save_ref", help="write this model's top-256 choices (run it on the dense ceiling)")
    ap.add_argument("--acts", help="write the layer outputs at the fixed tokens (B3)")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--row_tokens", type=int, default=16384,
                    help="document rows are drawn among rows of this length (the long-document eval's rows), then cut")
    ap.add_argument("--passkey", action="store_true",
                    help="planted-password rows instead of plain documents: per row a 5-digit pass key at an even "
                         "spread of depths (needle ≥ 640 tokens before the question), the queries are the answer "
                         "tokens and their source the key inside the needle; adds greedy correctness per row")
    ap.add_argument("--open_eval", default="on", choices=["on", "reuse", "off", "gist_off"],
                    help="opening arms: the read's eval switch (model.open_override); E33t 'reuse' lets a loop pass "
                         "re-open the notes it opened in earlier passes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from data.bin_token_dataset import BinRowDataset
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    S = a.length
    srcs = sorted(json.loads((Path(a.data) / "manifest.json").read_text())["sources"])
    dss = {s: BinRowDataset([{"name": s, "files": [str(Path(a.data) / f"{s}_val_*.bin")]}], a.row_tokens, "doc",
                            seed=7)
           for s in srcs}
    order = [(srcs[i % len(srcs)], i // len(srcs)) for i in range(a.rows)]
    ref_in = torch.load(a.dense_ref, weights_only=True) if a.dense_ref else None
    ref_out, acts_out = [], defaultdict(list)
    per_row, scal = [], defaultdict(list)
    dist_tot = defaultdict(lambda: torch.zeros(N_DIST, dtype=torch.float64))
    metrics = defaultdict(list)
    if a.passkey:
        import random

        from evaluation.long_context_probes import argmax_tokens, resolve_tokenizer_name
        from transformers import AutoTokenizer

        tok_pk = AutoTokenizer.from_pretrained(resolve_tokenizer_name(a.checkpoint))
        rng_pk = random.Random(7)
        top_depth = 1.0 - 640 / S                                       # the needle stays beyond the 512 window
    import contextlib

    switch = contextlib.ExitStack()
    if a.open_eval != "on":
        switch.enter_context(model.open_override(a.open_eval))
    for r_i, (src, i) in enumerate(order):
        item = dss[src][i]
        ids, doc = item["input_ids"][:S], item["doc_ids"][:S]
        correct = None
        if a.passkey:
            depth = 0.05 + (top_depth - 0.05) * (r_i + 0.5) / a.rows
            pk, q, fs, answer = passkey_row(tok_pk, item["input_ids"].tolist(), S, depth, rng_pk)
            assert len(pk) == S, f"passkey row has {len(pk)} tokens, not {S} (filler too short)"
            ids, doc = torch.tensor(pk), torch.zeros(S, dtype=torch.long)
        else:
            q, fs = pick_queries(ids.numpy(), S, a.queries, seed=1000 + r_i)
        tok = torch.linspace(0, S - 1, min(a.tokens, S)).round().long()
        rec = ProbeRecorder(torch.from_numpy(q), tok, ref=(ref_in["rows"][r_i] if ref_in else None),
                            save_ref=bool(a.save_ref), far_src=torch.from_numpy(fs))
        if getattr(model, "memory_writer", None) is not None:
            geo = model.memory_writer.geometry(S)
            st = torch.tensor(geo.starts, device=a.device)
            rec._page_tok = (st[:, None] + torch.arange(geo.window, device=a.device)[None]).clamp(max=S - 1)
        rec.begin(ids[None].to(a.device))
        model._set_flow(rec)
        try:
            with torch.no_grad():
                model.hidden_states(ids[None].to(a.device), doc_ids=doc[None].to(a.device))
        finally:
            model._set_flow(None)
        stats = rec.finish(model)
        if a.passkey:
            pred = argmax_tokens(model, ids[None].to(a.device), int(q[0]), int(q[-1]) + 1).tolist()
            correct = {"exact": int(pred == answer), "first": int(pred[0] == answer[0]), "depth": round(depth, 3),
                       "distance": int(q[0] - fs[0])}
            for k in ("exact", "first"):
                metrics[f"passkey/{k}"].append(float(correct[k]))
        for k, v in stats.items():
            scal[k].append(v)
        for k, v in rec.metrics.items():
            metrics[k].append(float(np.mean(v)))
        for r, d in rec.dist.items():
            dist_tot[r] += d
        if a.save_ref:
            ref_out.append(rec.new_ref)
        if a.acts:
            for k, v in rec.acts.items():
                acts_out[k].append(v)
        per_row.append({"source": src, "row": i, "far_queries": int((fs >= 0).sum()), **(correct or {}),
                        **{k: float(np.mean(v)) for k, v in rec.metrics.items()}})
        print(r_i, src, i, json.dumps({**(correct or {}), **{k: round(float(np.mean(v)), 4)
                                                               for k, v in rec.metrics.items()}}), flush=True)
    switch.close()
    res = {"checkpoint": a.checkpoint, "length": a.length, "rows": a.rows, "queries": a.queries,
           "mode": "passkey" if a.passkey else "documents", "open_eval": a.open_eval,
           "metrics": {k: float(np.mean(v)) for k, v in metrics.items()},
           "scalars": {k: float(np.mean(v)) for k, v in scal.items()},
           "dist_mass": {f"r{r}": (d / d.sum().clamp_min(1e-12)).tolist() for r, d in sorted(dist_tot.items())},
           "per_row": per_row}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.save_ref:
        torch.save({"rows": ref_out, "length": S, "queries": a.queries}, a.save_ref)
    if a.acts:
        torch.save({k: torch.cat(v, 0) for k, v in acts_out.items()}, a.acts)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

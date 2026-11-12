"""E37 stage B: our eval protocol on the ORIGINAL LSA model (scripts/lsa_orig.py, their torchtitan checkpoint), in their
venv. The model is loaded with their own evaluation builder (lsa.training.native.benchmark.run.load_backbone: their
`build_model`, their checkpoint loader, the hidden-state forward that sets each sparse layer's length to the batch's)
and wrapped so OUR scoring code runs on it unchanged:

  longdoc  evaluation/long_doc_loss.py's `score` (position buckets 0-512, 512-2k, ..., per source, rows of one held-out
           LongDocs-lite document, seed 7) at 4k-128k, 8 rows per source; also `open_off` (their LSA layers read sinks
           + window only: top_k 0), as our port's lsa arm is scored. JSON as long_doc_loss.py writes it.
  ruler    evaluation/run_lsa_suite.py's protocol: lm-eval's 13 RULER tasks, `--limit` documents each, prompts sized by
           our GPT-2 tokenizer files, string-match grading, greedy answers of at most 32 tokens, one lm-eval call per
           length, max_length = longest length + 64; through lm-eval's HFLM request path as our PerceiverARLMEval.
  core     the same 8 lm-eval tasks (log-likelihood; HFLM's batching), max_length 2048, batch 32.
  extra    run_lsa_suite.py's `extra` (BLiMP, SWDE, BABILong tasks 1-3 at --extra_lengths), the harness's own generation
           settings, BABILong batched by length as RULER.
  val      held-out loss: the trunk manifest's FineWeb-Edu val rows at 4k (256 rows, the m100 trainer's eval set,
           document-start labels ignored as our collator does) and the decay manifest's LongDocs-lite val rows at 16k.

lm-eval: our runs use lm-eval 0.4.11 (uv.lock); their venv carries 0.4.13, whose RULER generators were refactored. The
node puts 0.4.11 first on PYTHONPATH (pip --target, no deps); the version used is written into every JSON.

Model-side differences from our models' eval path (forced by their kernels, not changing what is scored):
  * rows are right-padded with <|endoftext|> to a multiple of 512 (their LSA layer reads whole 512-key blocks; causal
    layers never read the pad back, as in their own lm-eval wrapper);
  * generation strips lm-eval's LEFT padding and writes each row's next token into its own slot (their NativeLM's
    scheme): their fused sparse read has no padding mask, so a left pad would be read as the row's sinks.

Usage (node; their venv's python; this repo's root as the working directory):
  python evaluation/lsa_orig_eval.py longdoc --checkpoint R/decay/checkpoint/step-763 --data D --out R/eval_longdoc.json
  python evaluation/lsa_orig_eval.py ruler --checkpoint ... --data D --lengths 4096,8192 --out R/ruler_a.json
  python evaluation/lsa_orig_eval.py core --checkpoint ... --data D --out R/eval_core.json
  python evaluation/lsa_orig_eval.py extra --checkpoint ... --data D --out R/eval_extra.json
  python evaluation/lsa_orig_eval.py val --checkpoint ... --name decay_final|trunk_final --data D --out R/eval_val.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
PAD_MULTIPLE = 512   # their LSA block
EVAL_MAX = 131072


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # @dataclass under `from __future__ import annotations` looks its module up here
    spec.loader.exec_module(mod)
    return mod


LO = _load("mc_lsa_orig", ROOT / "scripts" / "lsa_orig.py")


def _single_process_env() -> None:
    """Their build_model initialises torch.distributed; one process, one GPU."""
    if "RANK" in os.environ:
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))


def load_backbone(checkpoint: str, work: Path):
    """Their evaluation builder on the decay phase's config (the arm, the ~160M body), then the checkpoint's weights."""
    _single_process_env()
    LO.patch_body()
    # built for the longest eval row (the context limit torchtitan's input preparation checks); every forward then
    # sets the sparse layers' length to its own batch's, as their benchmark does
    exp, _ = LO.experiment("decay", work, extra=[f"data.seq_len={EVAL_MAX}", f"run.optim.tokens_per_rank={EVAL_MAX}"])
    from lsa.training.native.benchmark.run import load_backbone as their_load_backbone

    print(f"[lsa_orig_eval] kernels: {LO.fit_kernels()}", flush=True)
    bb = their_load_backbone(exp, checkpoint)
    from lsa.training.native.layers import MultiresInnerAttention

    attn = [m for m in _model_of(bb).modules() if isinstance(m, MultiresInnerAttention)] if _model_of(bb) else []
    return bb, exp, attn


def _model_of(bb):
    """The torchtitan model inside their Backbone closure (for the attention layers' settings)."""
    for cell in (bb.hidden.__closure__ or ()):
        v = cell.cell_contents
        if isinstance(v, torch.nn.Module):
            return v
    return None


def _pad_to(x: torch.Tensor, multiple: int, value: int) -> torch.Tensor:
    S = x.shape[1]
    Sp = -(-S // multiple) * multiple
    if Sp == S:
        return x
    return torch.cat([x, x.new_full((x.shape[0], Sp - S), value)], 1)


class TokenLossModel:
    """long_doc_loss.score's model contract: model(input_ids, labels, doc_ids, return_per_token_loss=True) ->
    (out, per [B, S-1], valid [B, S-1]); per[t] is the loss of predicting token t + 1."""

    CHUNK = 4096

    def __init__(self, bb, eot: int = 50256):
        self.bb, self.eot = bb, eot

    @torch.no_grad()
    def __call__(self, input_ids, labels=None, doc_ids=None, return_per_token_loss=True):
        labels = input_ids if labels is None else labels
        B, S = input_ids.shape
        h = self.bb.hidden(_pad_to(input_ids, PAD_MULTIPLE, self.eot))[:, :S]
        hid = h[:, :-1].reshape(-1, h.shape[-1])
        tgt = labels[:, 1:].reshape(-1)
        per = torch.empty(hid.shape[0], dtype=torch.float32, device=hid.device)
        for i in range(0, hid.shape[0], self.CHUNK):
            logits = self.bb.project(hid[i:i + self.CHUNK].float()).float()
            per[i:i + self.CHUNK] = F.cross_entropy(logits, tgt[i:i + self.CHUNK].clamp_min(0), reduction="none")
        valid = tgt != -100
        per = per.masked_fill(~valid, 0.0)
        return None, per.view(B, S - 1), valid.view(B, S - 1)


def cmd_longdoc(a) -> None:
    ld = _load("mc_long_doc_loss", ROOT / "evaluation" / "long_doc_loss.py")
    bins = LO.bin_dataset()
    bb, exp, attn = load_backbone(a.checkpoint, Path(a.out).parent / "eval_work")
    model = TokenLossModel(bb)
    srcs = sorted(json.loads((Path(a.data) / "longdocs_lite_gpt2" / "manifest.json").read_text())["sources"])
    res = {"checkpoint": a.checkpoint, "has_notebook": False, "lengths": {}, "model": "lsa_orig (their trainer)",
           "open_off": "their LSA layers with top_k 0 (sinks + window); the Gated DeltaNet layers unchanged"}
    for S in (int(x) for x in a.lengths.split(",")):
        per_len = {}
        for src in srcs:
            try:
                ds = bins.BinRowDataset([{"name": src, "files": [str(Path(a.data) / "longdocs_lite_gpt2" /
                                                                       f"{src}_val_*.bin")]}], S, "doc", seed=7)
            except ValueError:
                per_len[src] = {"rows": 0}
                continue
            try:
                per_len[src] = {"notes_on": ld.score(model, ds, S, a.rows, "cuda")}
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:   # a length past what fits: record, go on
                per_len[src] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                torch.cuda.empty_cache()
            if a.open_off and attn and "notes_on" in per_len[src]:
                saved = [m.attention for m in attn]
                try:
                    for m in attn:
                        m.attention = dataclasses.replace(m.attention, top_k=0)
                    per_len[src]["open_off"] = ld.score(model, ds, S, a.rows, "cuda")
                except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
                    per_len[src]["open_off_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    torch.cuda.empty_cache()
                finally:
                    for m, sv in zip(attn, saved):
                        m.attention = sv
            print(S, src, json.dumps({k: v.get("loss") if isinstance(v, dict) else v
                                      for k, v in per_len[src].items()}), flush=True)
        res["lengths"][S] = per_len
        Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


def make_lm(bb, tokenizer: str, max_length: int, batch: int):
    import transformers
    from lm_eval.models.huggingface import HFLM

    class _Shim(torch.nn.Module):
        """What HFLM reads off a pre-built model: .config, .device, eval(), tie_weights()."""

        def __init__(self):
            super().__init__()
            cfg = getattr(transformers, "PretrainedConfig", None) or transformers.PreTrainedConfig
            self.config = cfg(vocab_size=50304)

        @property
        def device(self):
            return bb.device

        def tie_weights(self):
            pass

    class LsaOrigLMEval(HFLM):
        """HFLM over their backbone: _model_call returns log-probabilities as PerceiverARLMEval does; greedy uncached
        generation without left padding (see the module docstring)."""

        LOGITS_CHUNK_BYTES = 2 * 1024**3

        def _model_call(self, inps, attn_mask=None, labels=None):
            assert attn_mask is None and labels is None
            B, S = inps.shape
            eot = self.tokenizer.eos_token_id
            with torch.no_grad():
                h = bb.hidden(_pad_to(inps, PAD_MULTIPLE, eot))[:, :S]
                V = 50304
                rows = max(1, self.LOGITS_CHUNK_BYTES // (S * V * 4))
                out = torch.empty(B, S, V, dtype=torch.float32, device=inps.device)
                for i in range(0, B, rows):
                    out[i:i + rows] = F.log_softmax(bb.project(h[i:i + rows].float()).float(), dim=-1)
                return out

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            attn = generation_kwargs.get("attention_mask")
            eot = self.tokenizer.eos_token_id
            B, L0 = context.shape
            new = max(0, int(max_length) - L0)
            lens = (attn.sum(1) if attn is not None else torch.full((B,), L0, device=context.device)).long()
            width = -(-(int(lens.max()) + new) // PAD_MULTIPLE) * PAD_MULTIPLE
            buf = torch.full((B, width), eot, dtype=context.dtype, device=context.device)
            for b in range(B):   # lm-eval pads on the left: the real prompt is the row's last lens[b] tokens
                buf[b, :lens[b]] = context[b, L0 - int(lens[b]):]
            cur = lens.clone()
            rows = torch.arange(B, device=context.device)
            done = torch.zeros(B, dtype=torch.bool, device=context.device)
            stops = [s for s in (stop or []) if s]
            t = 0
            with torch.no_grad():
                for _ in range(new):
                    h = bb.hidden(buf)
                    nxt = bb.project(h[rows, cur - 1].float()).float().argmax(-1)
                    nxt = torch.where(done, torch.full_like(nxt, eot), nxt)
                    buf[rows, cur] = nxt.to(buf.dtype)
                    cur += 1
                    t += 1
                    done |= nxt == eot
                    if stops:
                        for b in range(B):
                            if not done[b]:
                                text = self.tokenizer.decode(buf[b, int(lens[b]):int(cur[b])].tolist())
                                done[b] = any(s in text for s in stops)
                    if bool(done.all()):
                        break
            out = torch.full((B, L0 + t), eot, dtype=context.dtype, device=context.device)
            out[:, :L0] = context
            for b in range(B):
                out[b, L0:] = buf[b, int(lens[b]):int(lens[b]) + t]
            return out

    return LsaOrigLMEval(pretrained=_Shim(), backend="causal", tokenizer=tokenizer, max_length=max_length,
                         batch_size=batch)


def _version(mod: str) -> str:
    m = __import__(mod)
    return f"{getattr(m, '__version__', '?')} ({Path(m.__file__).parent})"


def _lm_eval_suite(a, suite: str) -> None:
    import lm_eval
    from lm_eval.tasks import TaskManager

    rs = _load("mc_run_lsa_suite", ROOT / "evaluation" / "run_lsa_suite.py")
    tok = str(Path(a.data) / "tokenizer_gpt2")
    bb, exp, _ = load_backbone(a.checkpoint, Path(a.out).parent / "eval_work")
    if suite == "extra":  # run_lsa_suite.py's `extra` (BLiMP, SWDE, BABILong 1-3), same row lengths as our models'
        lengths = [int(x) for x in a.extra_lengths.split(",")]
        max_len = max(lengths) + max(lengths) // 8 + 512
        out = {"checkpoint": a.checkpoint, "suite": "extra", "lengths": lengths, "limit": a.limit, "notes_off": False,
               "concept_sum_len": -1, "lm_eval": _version("lm_eval"), "model": "lsa_orig (their trainer)",
               "max_length": max_len, "extra_cap": a.extra_cap}
        save = lambda: Path(a.out).write_text(json.dumps(out, indent=2, default=str))

        def evaluate(tasks_, limit_, md):
            L = int(md["max_seq_lengths"][:-1]) * 1024 if md else 0  # BABILong rows: batch by length, as RULER's
            batch = a.batch or (max(1, min(8, a.batch_tokens // L)) if L else 8)
            return lm_eval.simple_evaluate(
                model=make_lm(bb, tok, max_len, batch), tasks=tasks_, limit=limit_ if limit_ and limit_ > 0 else None,
                log_samples=False, task_manager=TaskManager(metadata=md) if md else None, bootstrap_iters=0)["results"]

        t0 = time.time()
        rs.run_extra(evaluate, lengths, a.limit, out, save, cap=a.extra_cap or None,
                     parts=tuple(x for x in a.extra_parts.split(",") if x))
        out["seconds"] = round(time.time() - t0, 1)
        save()
        return
    lengths = [int(x) for x in a.lengths.split(",")] if suite == "ruler" else None
    max_len = max(lengths) + 2 * a.max_gen_toks if suite == "ruler" else 2048
    tasks = [t for t in a.tasks.split(",") if t] or (rs.RULER if suite == "ruler" else rs.CORE)
    if "ruler_qa_hotpot" in tasks:
        rs.redirect_hotpot_download()
    out = {"checkpoint": a.checkpoint, "suite": suite, "lengths": lengths, "limit": a.limit, "notes_off": False,
           "concept_sum_len": -1, "lm_eval": _version("lm_eval"), "model": "lsa_orig (their trainer)"}
    t0 = time.time()

    def run(lm, md):
        return lm_eval.simple_evaluate(
            model=lm, tasks=tasks, limit=a.limit if a.limit > 0 else None, log_samples=False,
            task_manager=TaskManager(metadata=md) if md else None,
            gen_kwargs=f"max_gen_toks={a.max_gen_toks}" if suite == "ruler" else None, bootstrap_iters=0,
        )["results"]

    if suite == "ruler":
        out["results"], out["mean"], out["tasks"], out["seconds_per_length"] = {}, {}, {}, {}
        for L in lengths:
            t1 = time.time()
            # batch by length (rows never see each other: right padding, causal layers); max_length fixed as ours
            lm = make_lm(bb, tok, max_len, a.batch or max(1, min(8, a.batch_tokens // L)))
            r = run(lm, {"tokenizer": tok, "max_seq_lengths": [L]})
            out["results"][L] = r
            vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
            vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
            out["mean"][L] = sum(vals) / len(vals) if vals else None
            out["tasks"][L] = len(vals)
            out["seconds_per_length"][L] = round(time.time() - t1, 1)
            print(f"RULER {L}: {len(vals)} tasks, mean {out['mean'][L]} ({time.time() - t1:.0f} s)", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    else:
        out["results"] = run(make_lm(bb, tok, max_len, a.batch or 32), None)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out.get("mean") or out["results"], indent=1, default=str)[:3000])


@torch.no_grad()
def _val_loss(model: TokenLossModel, ds, batch: int) -> dict:
    """Token-mean loss with our collator's labels (document starts ignored), and the HF trainer's eval_loss (mean of
    per-batch means over batches of `batch` rows)."""
    tot, cnt, batch_means = 0.0, 0, []
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(len(ds), i + batch))]
        ids = torch.stack([x["input_ids"] for x in items]).cuda()
        doc = torch.stack([x["doc_ids"] for x in items]).cuda()
        # rows of an answer-masked manifest (E37 SFT) carry their own labels: -100 off the answers
        labels = torch.stack([x["labels"] for x in items]).cuda() if "labels" in items[0] else ids.clone()
        labels[:, 1:] = labels[:, 1:].masked_fill(doc[:, 1:] != doc[:, :-1], -100)
        _, per, valid = model(ids, labels)
        s, c = float(per[valid].sum()), int(valid.sum())
        tot, cnt = tot + s, cnt + c
        batch_means.append(s / max(1, c))
    return {"rows": len(ds), "tokens": cnt, "token_mean": tot / max(1, cnt),
            "hf_eval_loss": sum(batch_means) / max(1, len(batch_means)), "eval_batch": batch}


def cmd_val(a) -> None:
    """One checkpoint per process (their builder initialises torch.distributed once); results merge into --out."""
    bins = LO.bin_dataset()
    man = LO.manifests(Path(a.data), Path(a.out).parent / "eval_work")
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {
        "model": "lsa_orig (their trainer)", "note": "fineweb: the trunk manifest's eval rows (256 at 4k, seed 43); "
        "longdocs: the decay manifest's eval rows (16 per source at 16k, seed 1043); hf_eval_loss as our trainer's"}
    bb, _, _ = load_backbone(a.checkpoint, out.parent / "eval_work")
    model = TokenLossModel(bb)
    r = {"checkpoint": a.checkpoint}
    if not a.sft_only:
        _, fw_val, _ = bins.load_bin_manifest(str(man["trunk"]))
        r["fineweb_val_4k"] = _val_loss(model, fw_val, 4)
    if a.name != "trunk_final" and not a.sft_only:
        _, ld_val, _ = bins.load_bin_manifest(str(man["decay"]))
        r["longdocs_val_16k"] = _val_loss(model, ld_val, 1)
    if a.sft_manifest:  # E37 SFT held-out instruction loss (answer tokens only)
        _, sft_val, m = bins.load_bin_manifest(a.sft_manifest)
        r[f"sft_test_{m['seq_len'] // 1024}k"] = _val_loss(model, sft_val, 1)
    res[a.name] = r
    print(a.name, json.dumps(r), flush=True)
    out.write_text(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("longdoc", "ruler", "core", "val", "extra"))
    ap.add_argument("--checkpoint", required=True, help="torchtitan step folder of the decay (.../checkpoint/step-N)")
    ap.add_argument("--name", default="decay_final", help="val: the key the checkpoint's losses go under")
    ap.add_argument("--data", required=True, help="dir with fineweb_edu_gpt2/, longdocs_lite_gpt2/, tokenizer_gpt2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="4096,16384,32768,65536,131072")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--open_off", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--batch", type=int, default=0, help="0: by length (RULER) / 32 (core)")
    ap.add_argument("--batch_tokens", type=int, default=131072, help="RULER: rows of a batch = min(8, this / length)")
    ap.add_argument("--max_gen_toks", type=int, default=32)
    ap.add_argument("--extra_lengths", default="2048,4096,8192,16384", help="extra: BABILong context lengths")
    ap.add_argument("--sft_manifest", default="", help="val, E37 SFT: also score this manifest's eval rows (answers)")
    ap.add_argument("--sft_only", action="store_true", help="val with --sft_manifest: skip the FineWeb / LongDocs sets")
    ap.add_argument("--extra_cap", type=int, default=0, help="extra, smoke tests only: documents per BLiMP set / SWDE")
    ap.add_argument("--extra_parts", default="blimp,swde,babilong", help="extra: which parts to run")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.what == "longdoc":
        cmd_longdoc(a)
    elif a.what == "val":
        cmd_val(a)
    else:
        _lm_eval_suite(a, a.what)


if __name__ == "__main__":
    main()

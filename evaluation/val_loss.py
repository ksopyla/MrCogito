"""E37, LSA's validation loss for our models: held-out loss of one perceiver_ar checkpoint on the m100 trainer's eval
sets — FineWeb-Edu val at 4k (the trunk manifest's 256 rows) and LongDocs-lite val at 16k (the decay manifest's 16
rows per source) — exactly as `evaluation/lsa_orig_eval.py val` scores the original LSA, so the three models'
numbers compare. Labels as our collator's (a document's first token is not predicted from the previous document);
`token_mean` is the token-weighted mean, `hf_eval_loss` the HF trainer's (mean of per-batch means, `eval_batch` rows
a batch). Weights in bf16 (the trainer evaluated fp32 weights under bf16 autocast: expect a few thousandths of a nat).
Results for several checkpoints merge into --out under --name.
Usage: uv run python evaluation/val_loss.py --checkpoint RUN/decay/.../final --name decay_final \
         --data DATA (fineweb_edu_gpt2/, longdocs_lite_gpt2/) --out R/eval_val.json [--no_longdocs]
       E37 SFT: add --sft_manifest MIX/manifest_sft.json (its eval rows, loss on answer tokens only) [--sft_only]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def val_loss(model, ds, batch: int, device: str = "cuda") -> dict:
    """lsa_orig_eval._val_loss with our model's per-token loss (long_doc_loss.score's contract)."""
    tot, cnt, batch_means = 0.0, 0, []
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(len(ds), i + batch))]
        ids = torch.stack([x["input_ids"] for x in items]).to(device)
        doc = torch.stack([x["doc_ids"] for x in items]).to(device)
        # rows of an answer-masked manifest (E37 SFT) carry their own labels: -100 off the answers
        labels = torch.stack([x["labels"] for x in items]).to(device) if "labels" in items[0] else ids.clone()
        labels[:, 1:] = labels[:, 1:].masked_fill(doc[:, 1:] != doc[:, :-1], -100)
        _, per, valid = model(input_ids=ids, labels=labels, doc_ids=doc, return_per_token_loss=True)
        s, c = float(per.float()[valid].sum()), int(valid.sum())
        tot, cnt = tot + s, cnt + c
        batch_means.append(s / max(1, c))
    return {"rows": len(ds), "tokens": cnt, "token_mean": tot / max(1, cnt),
            "hf_eval_loss": sum(batch_means) / max(1, len(batch_means)), "eval_batch": batch}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="decay_final", help="the key this checkpoint's losses go under")
    ap.add_argument("--data", required=True, help="dir with fineweb_edu_gpt2/ and longdocs_lite_gpt2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_longdocs", action="store_true", help="FineWeb-Edu only (a trunk checkpoint)")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--sft_manifest", default="", help="E37 SFT: also score this manifest's eval rows (answer tokens)")
    ap.add_argument("--sft_only", action="store_true", help="with --sft_manifest: skip the FineWeb / LongDocs sets")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    bins = _load("mc_bin_token_dataset", ROOT / "data" / "bin_token_dataset.py")
    run = _load("mc_run_e33n", ROOT / "scripts" / "run_e33n.py")
    out = Path(a.out)
    man = run.manifests(Path(a.data), out.parent / "eval_work", "m100")
    model = load_perceiver_ar_for_eval(a.checkpoint, attn_backend=a.backend)
    r = {"checkpoint": a.checkpoint}
    if not a.sft_only:
        _, fw_val, _ = bins.load_bin_manifest(str(man["trunk"]))
        r["fineweb_val_4k"] = val_loss(model, fw_val, 4)
    if not a.no_longdocs and not a.sft_only:
        _, ld_val, _ = bins.load_bin_manifest(str(man["decay"]))
        r["longdocs_val_16k"] = val_loss(model, ld_val, 1)
    if a.sft_manifest:  # held-out instruction loss: smol-smoltalk test conversations, answer tokens only
        _, sft_val, m = bins.load_bin_manifest(a.sft_manifest)
        r[f"sft_test_{m['seq_len'] // 1024}k"] = val_loss(model, sft_val, 1)
    res = json.loads(out.read_text()) if out.exists() else {
        "note": "fineweb: the trunk manifest's eval rows (256 at 4k, seed 43); longdocs: the decay manifest's eval "
                "rows (16 per source at 16k, seed 1043); hf_eval_loss as our trainer's; evaluation/val_loss.py"}
    res[a.name] = r
    print(a.name, json.dumps(r), flush=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

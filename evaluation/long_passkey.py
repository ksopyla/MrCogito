"""E33t stage C (plan §10.8): the planted password in a real held-out document, at 128k–1M.

A password ("The pass key is 12345. Remember it.") is planted at a depth of a held-out LongDocs-lite document (val
shards, one document per row, so the filler is ordinary prose, not repeated text), the question ("What is the pass key?
The pass key is") ends the row, and the answer is scored teacher-forced: the model's greedy prediction at each answer
position, given the true prefix (first token right; all five digits right). 20 depths (2.5 % … 97.5 %) per length;
the filler document cycles through the documents long enough (one at 1M: an illustration of depth, not of documents).
The window control reads only its last 512 tokens, so its score is the guessing floor; `--notes_off` scores the same
rows with the notebook switched off.

Usage: uv run python evaluation/long_passkey.py --checkpoint RUN/long/.../final --data DATA/longdocs_lite_gpt2 \
         --tokenizer DATA/tokenizer_gpt2 --lengths 131072,1048576 --open_impl list --out passkey.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def filler_rows(data: str, S: int) -> list[torch.Tensor]:
    """One row of S tokens from each held-out document at least S long (the first span of each), all sources."""
    from data.bin_token_dataset import BinRowDataset

    rows, seen = [], set()
    for src in sorted(json.loads((Path(data) / "manifest.json").read_text())["sources"]):
        try:
            ds = BinRowDataset([{"name": src, "files": [str(Path(data) / f"{src}_val_*.bin")]}], S, "doc", seed=7)
        except ValueError:
            continue
        from evaluation.long_doc_loss import doc_key

        for i in range(len(ds)):
            k = (src, doc_key(ds, i))
            if k not in seen:
                seen.add(k)
                rows.append(ds[i]["input_ids"])
    return rows


@torch.no_grad()
def answer_preds(model, ids: list[int], n: int) -> list[int]:
    x = torch.tensor(ids, device="cuda")[None]
    h = model.hidden_states(x)[0]
    S = h.shape[0]
    return torch.nn.functional.linear(h[S - n - 1:S - 1], model.lm_head.weight).argmax(-1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lengths", default="131072")
    ap.add_argument("--depths", type=int, default=20)
    ap.add_argument("--two_level", type=int, default=0,
                    help="E33t stage C eval switch: a catalogue model keeps this many pages first, then chooses "
                         "among their notes (lm_open_levels=2; the page key is the mean of its tokens' keys)")
    ap.add_argument("--open_impl", default="table", choices=("table", "list"))
    ap.add_argument("--notes_off", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = load_perceiver_ar_for_eval(a.checkpoint, device="cuda", attn_backend="flex")
    if a.two_level:
        if str(getattr(model.config, "lm_note_read", "gist")) != "index":
            raise SystemExit("--two_level is for catalogue models")
        model.config.lm_open_levels, model.config.lm_open_pages = 2, int(a.two_level)
    has_notes = getattr(model, "memory_writer", None) is not None and \
        str(getattr(model.config, "message_override", "real")) == "real"
    modes = ["notes_on"] + (["notes_off"] if a.notes_off and has_notes else [])
    res = {"checkpoint": a.checkpoint, "open_impl": a.open_impl, "two_level": a.two_level, "lengths": {}}
    question = tok.encode(" What is the pass key? The pass key is", add_special_tokens=False)
    for L in (int(x) for x in a.lengths.split(",")):
        rows = filler_rows(a.data, L)
        if not rows:
            res["lengths"][L] = {"error": "no held-out document this long"}
            continue
        rng = random.Random(a.seed + L)
        cases = []
        for di in range(a.depths):
            depth = (di + 0.5) / a.depths
            key = f"{rng.randint(0, 99999):05d}"
            needle = tok.encode(f" The pass key is {key}. Remember it. ", add_special_tokens=False)
            answer = tok.encode(f" {key}", add_special_tokens=False)
            filler = rows[di % len(rows)].tolist()[:L - len(needle) - len(question) - len(answer)]
            cut = int(len(filler) * depth)
            cases.append((depth, filler[:cut] + needle + filler[cut:] + question + answer, answer))
        per_mode = {}
        for mode in modes:
            t0 = time.time()
            first = exact = 0
            detail = []
            off = model.message_override("none") if mode == "notes_off" else nullcontext()
            with model.open_impl(a.open_impl), off:
                for depth, ids, answer in cases:
                    pred = answer_preds(model, ids, len(answer))
                    f, e = int(pred[0] == answer[0]), int(pred == answer)
                    first, exact = first + f, exact + e
                    detail.append({"depth": round(depth, 4), "first": f, "exact": e})
            per_mode[mode] = {"first_token": first / len(cases), "exact": exact / len(cases), "rows": len(cases),
                              "documents": min(len(rows), len(cases)), "seconds": round(time.time() - t0, 1),
                              "per_depth": detail}
            print(L, mode, json.dumps({k: v for k, v in per_mode[mode].items() if k != "per_depth"}), flush=True)
        res["lengths"][L] = per_mode
        Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

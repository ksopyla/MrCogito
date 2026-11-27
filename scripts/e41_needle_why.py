"""E41: why a planted number is missed at long range - was its place opened, or opened and not used?

The planted-number test (scripts/e41_long_needle.py, the same texts by the same seed) scores a number as read when
every one of its tokens is predicted. At 128k and beyond the first token is right in nearly every text and a later
one is wrong in a third of them. Each answer token is predicted by its own query, and each query chooses for
itself which tokens of the row it opens. This script records, for every answer token and each of the reading
layer's passes, whether the read opened the two source positions that carry it: the planted token itself and the
one before it (whose context matches the query's). Then:

  place opened, token right   the read works
  place opened, token wrong   the reader has the token and does not use it
  place not opened            the choice of what to open missed (routing)

  uv run python scripts/e41_needle_why.py --checkpoint CK --mix MIX --lengths 131072,262144 --depths 32 --out X.json

One forward per text (a 256k row needs about 9 GB). The model's own read; `--scan` / `--top_down` as in the test.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mix", required=True, type=Path)
    ap.add_argument("--sources", default="books,arxiv,doab,usgpo,hansard")
    ap.add_argument("--lengths", default="131072,262144")
    ap.add_argument("--depths", type=int, default=32)
    ap.add_argument("--list_from", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--top_down", action="store_true")
    ap.add_argument("--top_k", type=int, default=0, help="tokens a query may open (0: the checkpoint's own, 512)")
    ap.add_argument("--width", type=int, default=0, help="concepts a query ranks for opening (0: the checkpoint's own)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    import nn.concept_tree as ct
    from e41_long_needle import WORDS, filler
    from nn.perceiver_ar_lm import PerceiverARLM

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = PerceiverARLM.from_pretrained(a.checkpoint).cuda().eval()
    eot = int(model.config.eos_token_id if model.config.eos_token_id is not None else 0)
    for mod in model.modules():
        c_ = getattr(mod, "concept", None)
        if isinstance(c_, dict) and "top_k" in c_:
            if a.top_k:
                c_["top_k"] = a.top_k
            if a.width:
                c_["width"] = a.width
        if isinstance(getattr(mod, "tree", None), dict):
            if a.top_down:
                mod.tree["sparse"] = True
            if a.scan:
                mod.tree["scan"] = True
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    text = filler(a.mix, a.sources.split(","), max(lengths) + 4096, eot)

    def enc(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    out = {"checkpoint": a.checkpoint, "depths": a.depths, "lengths": {}}
    for S in lengths:
        rows = []
        for d in range(a.depths):
            word = rng.choice(WORDS)
            number = "".join(rng.choice("0123456789") for _ in range(7))
            needle = enc(f" One of the special magic numbers for {word} is: {number}.")
            ask = enc(f"\nWhat is the special magic number for {word} mentioned in the provided text? "
                      f"The special magic number for {word} mentioned in the provided text is")
            ans = enc(f" {number}")
            body = S - 1 - len(needle) - len(ask) - len(ans)
            off = rng.randrange(0, max(len(text) - body, 1))
            fill = text[off:off + body].tolist()
            at = int((d + 0.5) / a.depths * body)
            ids = [eot] + fill[:at] + needle + fill[at:] + ask + ans
            n = len(ans)
            k = next((i for i in range(len(needle) - n + 1) if needle[i:i + n] == ans), None)
            if k is None:  # the number splits differently inside the sentence: no token-by-token source
                continue
            p0 = 1 + at + k                                     # the row position of the number's first token
            x = torch.tensor(ids, device="cuda")[None]
            ct.WATCH = {"q0": S - n - 1, "pos": torch.arange(p0 - 1, p0 + n), "log": [], "mass": [], "rank": []}
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    with model.open_impl("list" if S >= a.list_from else None):
                        h = model.hidden_states(x)[0][S - n - 1:S - 1]
                    pred = torch.nn.functional.linear(h.float(), model.lm_head.weight.float()).argmax(-1).tolist()
                log, mass, ranks = ct.WATCH["log"], ct.WATCH["mass"], ct.WATCH["rank"]
            finally:
                ct.WATCH = None
            # per pass of the reading layer: [n answer queries, n + 1 source positions]
            passes, opened_n = [], []
            for qp, hit, cnt in log:
                m = torch.zeros(n, n + 1, dtype=torch.bool)
                c = torch.zeros(n, dtype=torch.long)
                for i, q in enumerate(qp.tolist()):
                    j = q - (S - n - 1)
                    if 0 <= j < n:
                        m[j] = hit[0, i]
                        c[j] = cnt[0, i]
                passes.append(m)
                opened_n.append(c)
            # where each answer query's attention went, per pass: {part: [n, h]}, "tgt": [n, h, n + 1]
            att = []
            for qp, rec in mass:
                m = {k: torch.zeros(n, *v.shape[2:]) for k, v in rec.items()}
                for i, q in enumerate(qp.tolist()):
                    j = q - (S - n - 1)
                    if 0 <= j < n:
                        for k, v in rec.items():
                            m[k][j] = v[0, i]
                att.append(m)
            # the scan rank of the concept that holds each answer token's source, per pass (-1: not scored)
            rk = []
            for qp, r_ in ranks:
                m = torch.full((n,), -1, dtype=torch.long)
                for i, q in enumerate(qp.tolist()):
                    j = q - (S - n - 1)
                    if 0 <= j < n:
                        m[j] = r_[0, i, j + 1]
                rk.append(m)
            toks = []
            for j in range(n):
                # the source of answer token j: positions p0 + j - 1 (context) and p0 + j (the token) = columns j, j + 1
                toks.append({"right": pred[j] == ans[j],
                             "token_opened": [bool(m[j, j + 1]) for m in passes],
                             "before_opened": [bool(m[j, j]) for m in passes],
                             "any_of_number_opened": [bool(m[j].any()) for m in passes],
                             "opened": [int(c[j]) for c in opened_n],
                             "scan_rank": [int(m[j]) for m in rk],
                             # per pass: the best head's share on the token to copy, on the one before it, on
                             # the whole number; and the heads' mean share by part
                             "on_token": [round(float(m["tgt"][j, :, j + 1].max()), 4) for m in att],
                             "on_before": [round(float(m["tgt"][j, :, j].max()), 4) for m in att],
                             "on_number": [round(float(m["tgt"][j].sum(-1).max()), 4) for m in att],
                             "on_other_digit": [round(float((m["tgt"][j].sum(-1) - m["tgt"][j, :, j + 1]
                                                             - m["tgt"][j, :, j]).max()), 4) for m in att],
                             "parts": [{k: round(float(m[k][j].mean()), 4) for k in ("win", "sink", "open", "sum")}
                                       for m in att]})
            rows.append({"depth": d, "at": at, "n": n, "tokens": toks, "all": pred == ans})
            print(f"L={S} depth {d + 1}/{a.depths}: " + " ".join(
                ("+" if t["right"] else "-") + ("o" if any(t["token_opened"]) else ".") for t in toks), flush=True)

        def frac(xs):
            xs = list(xs)
            return [sum(xs), len(xs)]

        summ = {"texts": len(rows), "all_digits": sum(r["all"] for r in rows), "passes": len(rows[0]["tokens"][0]["opened"]) if rows else 0}
        every = [(j, t) for r in rows for j, t in enumerate(r["tokens"])]
        for name, sel in (("first", [t for j, t in every if j == 0]), ("later", [t for j, t in every if j > 0])):
            op = [t for t in sel if any(t["token_opened"])]
            no = [t for t in sel if not any(t["token_opened"])]
            summ[name] = {"right": frac(t["right"] for t in sel),
                          "token_opened_any_pass": frac(any(t["token_opened"]) for t in sel),
                          "token_opened_last_pass": frac(t["token_opened"][-1] for t in sel if t["token_opened"]),
                          "before_opened_any_pass": frac(any(t["before_opened"]) for t in sel),
                          "right_when_opened": frac(t["right"] for t in op),
                          "right_when_not_opened": frac(t["right"] for t in no),
                          "mean_tokens_opened": round(sum(sum(t["opened"]) / max(len(t["opened"]), 1) for t in sel)
                                                      / max(len(sel), 1), 1)}
            for lab, grp in (("right", [t for t in sel if t["right"]]), ("wrong", [t for t in sel if not t["right"]])):
                if grp and grp[0].get("on_token"):
                    mean = lambda f: round(sum(f(t) for t in grp) / len(grp), 4)
                    summ[name][f"attention_when_{lab}"] = {
                        "n": len(grp),
                        "best_head_on_token_max_over_passes": mean(lambda t: max(t["on_token"])),
                        "best_head_on_before_max_over_passes": mean(lambda t: max(t["on_before"])),
                        "best_head_on_other_digit_max_over_passes": mean(lambda t: max(t["on_other_digit"])),
                        "last_pass_parts": {k: mean(lambda t, k=k: t["parts"][-1][k]) for k in ("win", "sink", "open", "sum")}}
        out["lengths"][str(S)] = {"summary": summ, "rows": rows}
        print(f"L={S}: {json.dumps(summ)}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

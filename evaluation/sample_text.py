"""Text samples for a model card: nucleus sampling and beam search, the way current base-model cards show text.

Greedy decoding loops on small base models ("The capital of France is Paris. The capital of France is ..."), so
greedy samples undersell a model and are not how anyone would run it. This script writes, for each prompt:

  sample  temperature 0.7, nucleus (top-p) 0.9, repetition penalty 1.1 over prompt + reply (as in 🤗 Transformers'
          `repetition_penalty`), one reply per seed (seeds 0, 1, 2 by default), so a card can show the seed-0 reply
          and say it was not picked;
  beam    4 beams, no repeated 3-gram, the highest-scoring finished beam (length-normalised log-probability).

Works for our perceiver_ar checkpoints (loaded with `load_perceiver_ar_for_eval`, SDPA attention: prompts here are
short, and flex attention would recompile for every new length) and for any 🤗 causal LM (`--hf`). Every step re-runs
the whole row (no cache): fine for prompts and replies of a few hundred tokens.

    uv run python evaluation/sample_text.py --model <our checkpoint folder> --out samples.json
    uv run python evaluation/sample_text.py --model HuggingFaceTB/SmolLM2-135M --hf --out smollm2_samples.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROMPTS = [
    "The capital of France is",
    "In 1905, Albert Einstein published",
    "def fibonacci(n):",
]


def penalise(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """🤗 Transformers' repetition penalty: divide positive logits of seen tokens by `penalty`, multiply negative."""
    if penalty == 1.0:
        return logits
    g = logits.gather(-1, seen)
    logits = logits.scatter(-1, seen, torch.where(g > 0, g / penalty, g * penalty))
    return logits


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    srt, idx = logits.sort(-1, descending=True)
    cum = F.softmax(srt, -1).cumsum(-1)
    drop = cum - F.softmax(srt, -1) > p  # keep the smallest set whose mass reaches p (the top token always stays)
    srt = srt.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)


@torch.no_grad()
def next_logits(model, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids).logits[:, -1].float()


@torch.no_grad()
def sample(model, ids, new, eos, seed, temperature=0.7, top_p=0.9, penalty=1.1) -> list[int]:
    g = torch.Generator(device=ids.device).manual_seed(seed)
    out = ids
    for _ in range(new):
        logits = penalise(next_logits(model, out), out, penalty) / temperature
        nxt = torch.multinomial(F.softmax(top_p_filter(logits, top_p), -1), 1, generator=g)
        out = torch.cat([out, nxt], 1)
        if eos is not None and int(nxt) == eos:
            break
    return out[0, ids.shape[1]:].tolist()


def _repeats(seq: list[int], tok: int, n: int) -> bool:
    if n <= 0 or len(seq) < n - 1:
        return False
    tail = tuple(seq[len(seq) - (n - 1):] + [tok]) if n > 1 else (tok,)
    return any(tuple(seq[i:i + n]) == tail for i in range(len(seq) - n + 1))


@torch.no_grad()
def beam(model, ids, new, eos, beams=4, no_repeat=3) -> list[int]:
    """Plain beam search: `beams` hypotheses, no repeated `no_repeat`-gram, best finished by mean log-probability."""
    L0 = ids.shape[1]
    hyps = [(ids[0].tolist(), 0.0)]
    done: list[tuple[list[int], float]] = []
    for _ in range(new):
        batch = torch.tensor([h for h, _ in hyps], device=ids.device)
        logp = F.log_softmax(next_logits(model, batch), -1)
        cand = []
        for b, (h, s) in enumerate(hyps):
            vals, toks = logp[b].topk(4 * beams)
            for v, t in zip(vals.tolist(), toks.tolist()):
                if _repeats(h, t, no_repeat):
                    continue
                cand.append((h + [t], s + v))
        cand.sort(key=lambda x: x[1] / (len(x[0]) - L0), reverse=True)
        hyps = []
        for h, s in cand:
            (done if eos is not None and h[-1] == eos else hyps).append((h, s))
            if len(hyps) == beams:
                break
        if not hyps:
            break
    done += hyps
    best = max(done, key=lambda x: x[1] / max(1, len(x[0]) - L0))
    return best[0][L0:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", action="store_true", help="a 🤗 causal LM instead of a perceiver_ar checkpoint")
    ap.add_argument("--prompts", default="", help="JSON list of prompts (default: the card's three)")
    ap.add_argument("--new", type=int, default=60, help="new tokens per reply")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--start", default="auto", choices=("auto", "none"),
                    help="auto: our checkpoints start with <|endoftext|> as in training; HF models as their tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.hf:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    else:
        from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

        model = load_perceiver_ar_for_eval(a.model, device=a.device, attn_backend="sdpa")
    prompts = json.loads(a.prompts) if a.prompts else PROMPTS
    eos = tok.eos_token_id
    lead = [eos] if (not a.hf and a.start == "auto") else []
    seeds = [int(s) for s in a.seeds.split(",") if s]
    res = {"model": a.model, "new_tokens": a.new, "start_tokens": lead,
           "sample": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, "seeds": seeds},
           "beam": {"beams": 4, "no_repeat_ngram": 3}, "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "prompts": []}
    for p in prompts:
        ids = torch.tensor([lead + tok(p, add_special_tokens=a.hf).input_ids], device=a.device)
        row = {"prompt": p, "sample": {}, "beam": tok.decode(beam(model, ids, a.new, eos))}
        for s in seeds:
            row["sample"][s] = tok.decode(sample(model, ids, a.new, eos, s))
        res["prompts"].append(row)
        print(json.dumps(row, indent=1, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

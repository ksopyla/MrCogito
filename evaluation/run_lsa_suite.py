"""E33n evaluation 3–4: LSA's benchmark suite (arXiv 2610.42147, repo configs/training/native/qwen35_0_8b_gpt2/README.md)
on a perceiver_ar checkpoint, through lm-eval-harness.

  ruler  the 13 RULER tasks (needle-in-a-haystack single 1–3, multikey 1–3, multiquery, multivalue, variable tracking,
         common / frequent word extraction, SQuAD and HotpotQA QA), `--limit` documents each (LSA: 100), prompts sized by
         the model's own GPT-2 tokenizer, RULER's string-match grading (fraction of references found in the reply);
         LSA's headline column is the mean of the 13 at 4,096 tokens. Greedy answers are capped at `--max_gen_toks`
         tokens (32 holds every RULER answer; the harness default of 128 only lengthens uncached decoding).
  core   HellaSwag, ARC-easy, ARC-challenge, PIQA, WinoGrande, LAMBADA (OpenAI), WikiText, SciQ (log-likelihood).

Usage: uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint RUN/final \
         --tokenizer DATA/tokenizer_gpt2 --suite ruler --lengths 4096,8192,16384 --out ruler.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RULER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multiquery", "niah_multivalue", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"]
CORE = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada_openai", "wikitext", "sciq"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--suite", choices=("ruler", "core"), required=True)
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tasks", default="", help="comma-separated subset (default: the whole suite)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_gen_toks", type=int, default=32)
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--notes_off", action="store_true", help="notebook models: message_override none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import lm_eval
    from lm_eval.tasks import TaskManager

    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval

    lengths = [int(x) for x in a.lengths.split(",")]
    max_len = max(lengths) + 2 * a.max_gen_toks if a.suite == "ruler" else 2048
    lm = PerceiverARLMEval(pretrained=a.checkpoint, tokenizer=a.tokenizer, attn_backend=a.backend,
                           max_length=max_len, batch_size=a.batch, device=a.device)
    if a.notes_off:
        lm.model._message_override = "none"
    tasks = [t for t in a.tasks.split(",") if t] or (RULER if a.suite == "ruler" else CORE)
    metadata = {"tokenizer": a.tokenizer, "max_seq_lengths": lengths} if a.suite == "ruler" else None
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, limit=a.limit if a.limit > 0 else None, log_samples=False,
        task_manager=TaskManager(metadata=metadata) if metadata else None,
        gen_kwargs=f"max_gen_toks={a.max_gen_toks}" if a.suite == "ruler" else None,
        bootstrap_iters=0,
    )
    out = {"checkpoint": a.checkpoint, "suite": a.suite, "lengths": lengths if a.suite == "ruler" else None,
           "limit": a.limit, "notes_off": a.notes_off, "seconds": round(time.time() - t0, 1),
           "results": res["results"]}
    if a.suite == "ruler":  # the LSA column: mean of the 13 tasks at each length
        out["mean"] = {}
        for L in lengths:
            vals = [r.get(f"{L},none") for r in res["results"].values() if isinstance(r, dict)]
            vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
            out["mean"][L] = sum(vals) / len(vals) if vals else None
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out.get("mean") or {k: v for k, v in res["results"].items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

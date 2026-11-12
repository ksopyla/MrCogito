"""E33n evaluation 3–4: LSA's benchmark suite (arXiv 2610.42147, repo configs/training/native/qwen35_0_8b_gpt2/README.md)
on a perceiver_ar checkpoint, through lm-eval-harness.

  ruler  the 13 RULER tasks (needle-in-a-haystack single 1–3, multikey 1–3, multiquery, multivalue, variable tracking,
         common / frequent word extraction, SQuAD and HotpotQA QA), `--limit` documents each (LSA: 100), prompts sized by
         the model's own GPT-2 tokenizer, RULER's string-match grading (fraction of references found in the reply);
         LSA's headline column is the mean of the 13 at 4,096 tokens. Greedy answers are capped at `--max_gen_toks`
         tokens (32 holds every RULER answer; the harness default of 128 only lengthens uncached decoding).
  core   HellaSwag, ARC-easy, ARC-challenge, PIQA, WinoGrande, LAMBADA (OpenAI), WikiText, SciQ (log-likelihood).
  extra  three tests added 2026-11-12 for a wider view (docs/experiments_specs/ahead/E37_sft_sensitivity.md):
         BLiMP — grammar, all 67 minimal-pair sets (log-likelihood, accuracy, mean over the sets);
         SWDE — read a real web page and copy one field out of it, all validation pages (greedy, ≤ 48 tokens,
                "contains" grading, the Based paper's recall task);
         BABILong tasks 1–3 — one, two or three facts scattered through book text, `--limit` documents per task at
                each of `--extra_lengths` (its configs 2k … 128k; 2-shot, greedy ≤ 16 tokens, string match).
         Each part keeps the harness's own generation settings; one lm-eval call per BABILong length.

Usage: uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint RUN/final \
         --tokenizer DATA/tokenizer_gpt2 --suite ruler --lengths 4096,8192,16384 --out ruler.json
       uv run python evaluation/run_lsa_suite.py --checkpoint RUN/final --tokenizer DATA/tokenizer_gpt2 \
         --suite extra --extra_lengths 2048,4096,8192,16384 --out extra.json
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
BABILONG = ["babilong_qa1", "babilong_qa2", "babilong_qa3"]


def babilong_config(L: int) -> str:
    """BABILong's dataset config for a context length in tokens (its qa1–qa5 configs: 0k, 1k, 2k, 4k, …, 128k)."""
    if L % 1024 or L // 1024 not in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError(f"BABILong has no {L}-token config")
    return f"{L // 1024}k"


def _metric(r: dict, task: str, metric: str) -> float | None:
    v = r.get(task, {}).get(metric) if isinstance(r.get(task), dict) else None
    return v if isinstance(v, (int, float)) else None


def run_extra(evaluate, lengths: list[int], limit: int, out: dict, save, cap: int | None = None) -> None:
    """The `extra` suite. `evaluate(tasks, limit, metadata)` runs lm-eval and returns its `results`; `save()` writes
    `out` after every part so a long run keeps what it has. BLiMP and SWDE run on every document unless `cap` (smoke
    tests only) limits them."""
    out["results"], out["summary"] = {}, {}
    r = evaluate(["blimp"], cap, None)
    sets = [v for k, v in ((k, _metric(r, k, "acc,none")) for k in r if k.startswith("blimp_")) if v is not None]
    out["results"]["blimp"] = r
    out["summary"]["blimp"] = _metric(r, "blimp", "acc,none") or (sum(sets) / len(sets) if sets else None)
    out["summary"]["blimp_sets"] = len(sets)
    save()
    r = evaluate(["swde"], cap, None)
    out["results"]["swde"] = r
    out["summary"]["swde"] = _metric(r, "swde", "contains,none")
    save()
    out["summary"]["babilong"] = {}
    for L in lengths:
        r = evaluate(BABILONG, limit, {"max_seq_lengths": babilong_config(L)})
        out["results"][f"babilong_{L}"] = r
        per = {t: _metric(r, t, "acc,none") for t in BABILONG}
        vals = [v for v in per.values() if v is not None]
        out["summary"]["babilong"][L] = {**per, "mean": sum(vals) / len(vals) if vals else None}
        save()

# lm-eval's RULER HotpotQA task downloads the dev set (distractor) from this host, which no longer resolves
# (2026-11-05, from Odra and Polonez). The same 7,405 questions are on the Hub as parquet.
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def hotpot_dev_from_hub() -> list[dict]:
    """HotpotQA dev (distractor) in the original JSON layout: question, answer, context = [[title, sentences], ...]."""
    import datasets
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [{"question": r["question"], "answer": r["answer"],
             "context": [[t, s] for t, s in zip(r["context"]["title"], r["context"]["sentences"])]} for r in ds]


def redirect_hotpot_download() -> None:
    """Serve HOTPOT_URL from the Hub mirror. lm-eval loads task code as a fresh module (spec_from_file_location), so
    the task's own functions cannot be patched; its download goes through the shared `requests.get`, which can."""
    import requests
    real_get = requests.get

    class _Json:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self) -> None:
            pass

        def json(self):
            return self._data

    def get(url, *args, **kwargs):
        return _Json(hotpot_dev_from_hub()) if url == HOTPOT_URL else real_get(url, *args, **kwargs)

    requests.get = get


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--suite", choices=("ruler", "core", "extra"), required=True)
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--extra_lengths", default="2048,4096,8192,16384", help="extra: BABILong context lengths")
    ap.add_argument("--extra_cap", type=int, default=0, help="extra, smoke tests only: documents per BLiMP set / SWDE")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tasks", default="", help="comma-separated subset (default: the whole suite)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_gen_toks", type=int, default=32)
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--notes_off", action="store_true", help="notebook models: message_override none")
    ap.add_argument("--concept_sum_len", type=int, default=-1,
                    help="E37 concept read: length scale of the summaries' weight past this row length (the training "
                         "length; 0 = off; default: the checkpoint's own setting)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import lm_eval
    from lm_eval.tasks import TaskManager

    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval

    lengths = [int(x) for x in (a.extra_lengths if a.suite == "extra" else a.lengths).split(",")]
    # extra: a BABILong row is its config's length plus the 2-shot examples and the instruction; SWDE pages are short
    max_len = (max(lengths) + 2 * a.max_gen_toks if a.suite == "ruler"
               else max(lengths) + max(lengths) // 8 + 512 if a.suite == "extra" else 2048)
    lm = PerceiverARLMEval(pretrained=a.checkpoint, tokenizer=a.tokenizer, attn_backend=a.backend,
                           max_length=max_len, batch_size=a.batch, device=a.device)
    if a.notes_off:
        lm.model._message_override = "none"
    if a.concept_sum_len >= 0:
        from nn.concept_memory import set_concept_sum_len

        if not set_concept_sum_len(lm.model, a.concept_sum_len):
            raise SystemExit("--concept_sum_len: the checkpoint has no concept read")
    tasks = [t for t in a.tasks.split(",") if t] or (RULER if a.suite == "ruler" else CORE)
    if "ruler_qa_hotpot" in tasks:
        redirect_hotpot_download()
    t0 = time.time()
    if a.suite == "extra":
        out = {"checkpoint": a.checkpoint, "suite": "extra", "lengths": lengths, "limit": a.limit,
               "notes_off": a.notes_off, "concept_sum_len": a.concept_sum_len, "max_length": max_len}
        save = lambda: Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        evaluate = lambda tasks_, limit_, md: lm_eval.simple_evaluate(
            model=lm, tasks=tasks_, limit=limit_ if limit_ and limit_ > 0 else None, log_samples=False,
            task_manager=TaskManager(metadata=md) if md else None, bootstrap_iters=0)["results"]
        out["extra_cap"] = a.extra_cap
        run_extra(evaluate, lengths, a.limit, out, save, cap=a.extra_cap or None)
        out["seconds"] = round(time.time() - t0, 1)
        save()
        print(json.dumps(out["summary"], indent=1, default=str))
        return
    run = lambda md: lm_eval.simple_evaluate(
        model=lm, tasks=tasks, limit=a.limit if a.limit > 0 else None, log_samples=False,
        task_manager=TaskManager(metadata=md) if md else None,
        gen_kwargs=f"max_gen_toks={a.max_gen_toks}" if a.suite == "ruler" else None,
        bootstrap_iters=0,
    )["results"]
    out = {"checkpoint": a.checkpoint, "suite": a.suite, "lengths": lengths if a.suite == "ruler" else None,
           "limit": a.limit, "notes_off": a.notes_off, "concept_sum_len": a.concept_sum_len}
    if a.suite == "ruler":
        # One call per length: lm-eval concatenates the lengths into one dataset per task and `limit` takes its first
        # documents, so a single call with several lengths scored only the first length (found 2026-11-05).
        out["results"], out["mean"] = {}, {}
        for L in lengths:
            r = run({"tokenizer": a.tokenizer, "max_seq_lengths": [L]})
            out["results"][L] = r
            vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
            vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
            out["mean"][L] = sum(vals) / len(vals) if vals else None  # the LSA column: mean of the 13 tasks
            Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    else:
        out["results"] = run(None)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out.get("mean") or out["results"], indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

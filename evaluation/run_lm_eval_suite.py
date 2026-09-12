#!/usr/bin/env python
"""Run the small-LM reasoning suite (lm-evaluation-harness) on a Perceiver AR checkpoint or on a
Hugging Face reference model, and write one compact summary row next to the full harness JSON.

The suite is the zero-shot, loglikelihood-scored set that SmolLM2 / Qwen / Pythia model cards
report, so our numbers are directly comparable at equal harness version:

    core : hellaswag arc_easy arc_challenge piqa winogrande openbookqa boolq social_iqa
           commonsense_qa lambada_openai wikitext          (~10 min for a 125M model on one 3090)
    full : core + mmlu (0-shot) + sciq + copa                (+ ~30 min)

Usage (GPU server, from the repo root):
    uv run python evaluation/run_lm_eval_suite.py --checkpoint Cache/Training/<run>/final --tag e18_stageA
    uv run python evaluation/run_lm_eval_suite.py --hf_model HuggingFaceTB/SmolLM2-135M --tag ref_smollm2_135m
    uv run python evaluation/run_lm_eval_suite.py --checkpoint <ck> --tag t --tier full --limit 200   # smoke

Outputs:
    <out_dir>/<tag>.json          full `lm_eval.simple_evaluate` result (results, versions, configs)
    <out_dir>/summary.csv         one row per (tag): main metric per task + suite average (appended)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

CORE_TASKS = [
    "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "boolq",
    "social_iqa", "commonsense_qa", "lambada_openai", "wikitext",
]
FULL_EXTRA = ["mmlu", "sciq", "copa"]

# Task yamls that override built-in harness tasks of the same name (later include paths win in
# `TaskManager`). Currently: `social_iqa` read from the Hub's parquet branch because datasets>=4
# refuses the dataset's loading script.
TASK_OVERRIDES_DIR = Path(__file__).resolve().parent / "lm_eval_tasks"

# Metric reported per task in the summary row (the one the model cards quote).
MAIN_METRIC = {
    "hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm",
    "winogrande": "acc", "openbookqa": "acc_norm", "boolq": "acc", "social_iqa": "acc",
    "commonsense_qa": "acc", "lambada_openai": "acc", "wikitext": "word_perplexity",
    "mmlu": "acc", "sciq": "acc_norm", "copa": "acc",
}
# Tasks averaged into `avg_acc` (accuracy-like only; perplexities are reported but not averaged).
ACC_TASKS = [t for t in CORE_TASKS + FULL_EXTRA if MAIN_METRIC[t] != "word_perplexity"]


def _tasks_for(tier: str, override: str | None) -> list[str]:
    if override:
        return [t.strip() for t in override.split(",") if t.strip()]
    return CORE_TASKS + (FULL_EXTRA if tier == "full" else [])


def _metric(results: dict, task: str, name: str):
    r = results.get(task, {})
    for k, v in r.items():
        if k.split(",")[0] == name and isinstance(v, (int, float)):
            return float(v)
    return None


def _stderr(results: dict, task: str, name: str):
    r = results.get(task, {})
    for k, v in r.items():
        if k.split(",")[0] == f"{name}_stderr" and isinstance(v, (int, float)):
            return float(v)
    return None


def summarize(results: dict, tasks: list[str]) -> dict:
    row: dict = {}
    accs = []
    for t in tasks:
        m = MAIN_METRIC.get(t, "acc")
        v = _metric(results, t, m)
        row[f"{t}/{m}"] = v
        se = _stderr(results, t, m)
        if se is not None:
            row[f"{t}/{m}_stderr"] = se
        if v is not None and t in ACC_TASKS:
            accs.append(v)
    row["avg_acc"] = sum(accs) / len(accs) if accs else None
    row["n_acc_tasks"] = len(accs)
    return row


def build_model(args):
    import lm_eval

    if args.checkpoint:
        import evaluation.lm_eval_perceiver_ar as adapter  # registers "perceiver_ar"

        return adapter.PerceiverARLMEval(
            pretrained=args.checkpoint,
            tokenizer=args.tokenizer,
            attn_backend=args.attn_backend,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        ), "perceiver_ar", args.checkpoint
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=args.hf_model,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        dtype="bfloat16" if args.device.startswith("cuda") else "float32",
        trust_remote_code=True,
    ), "hf", args.hf_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Perceiver AR checkpoint dir (config.json + weights)")
    src.add_argument("--hf_model", help="Hugging Face causal LM id/path for a reference row")
    p.add_argument("--tag", required=True, help="row label / output stem, e.g. e18_stageA_ck9030")
    p.add_argument("--tier", choices=["core", "full"], default="core")
    p.add_argument("--tasks", default=None, help="comma list overriding the tier")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="examples per task (smoke tests)")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", default="auto",
                   help="int, or 'auto' (harness probes the largest batch that fits per request "
                        "type: short multiple-choice rows vs 2048-token rolling windows)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--attn_backend", default="sdpa", help="perceiver_ar only: sdpa (default) | flex")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="Cache/Evaluation_reports/lm_eval")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tasks = _tasks_for(args.tier, args.tasks)
    try:
        args.batch_size = int(args.batch_size)
    except ValueError:
        pass  # "auto" / "auto:N"

    import lm_eval
    from lm_eval.tasks import TaskManager

    t0 = time.time()
    lm, model_kind, model_ref = build_model(args)
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=TaskManager(include_path=str(TASK_OVERRIDES_DIR)),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=False,
    )
    elapsed = time.time() - t0
    if res is None:  # non-zero rank in a distributed harness run
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag, "model_kind": model_kind, "model": model_ref, "tasks": tasks,
        "num_fewshot": args.num_fewshot, "limit": args.limit, "max_length": args.max_length,
        "elapsed_s": elapsed, "results": res["results"], "versions": res.get("versions"),
        "n_samples": res.get("n-samples"), "lm_eval_config": res.get("config"),
    }
    (out_dir / f"{args.tag}.json").write_text(json.dumps(payload, indent=2, default=str))

    row = {"tag": args.tag, "model": model_ref, "num_fewshot": args.num_fewshot,
           "limit": args.limit if args.limit else "", **summarize(res["results"], tasks)}
    csv_path = out_dir / "summary.csv"
    existing: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("tag") != args.tag] + [row]
    fields: list[str] = []
    for r in existing:
        for k in r:
            if k not in fields:
                fields.append(k)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing:
            w.writerow(r)

    print(f"\n=== lm-eval suite: {args.tag} ({model_ref}) — {elapsed/60:.1f} min ===")
    for t in tasks:
        m = MAIN_METRIC.get(t, "acc")
        v = row.get(f"{t}/{m}")
        se = row.get(f"{t}/{m}_stderr")
        if v is None:
            print(f"  {t:18s} {m:16s}   n/a")
        elif m == "word_perplexity":
            print(f"  {t:18s} {m:16s} {v:8.2f}")
        else:
            print(f"  {t:18s} {m:16s} {100*v:6.2f}" + (f" ± {100*se:.2f}" if se else ""))
    if row["avg_acc"] is not None:
        print(f"  {'avg_acc':18s} ({row['n_acc_tasks']} tasks)      {100*row['avg_acc']:6.2f}")
    print(f"  json: {out_dir / (args.tag + '.json')}\n  csv : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

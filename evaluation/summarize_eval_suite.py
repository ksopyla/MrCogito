"""Aggregate the perceiver_ar evaluation suite into one comparison table.

Reads, per tag:
  Cache/eval/<tag>/longctx_suite.json   (evaluation/long_context_probes.py --probe suite)
  Cache/eval/<tag>/reach.json           (--probe reach, optional)
  Cache/Evaluation_reports/lm_eval/summary.csv   (evaluation/run_lm_eval_suite.py rows)

and prints a markdown table (tags as columns) plus writes it to --out (markdown) so the numbers
can be pasted into a run report. Missing pieces are shown as "-" rather than failing, so the
table is useful while the suite is still running.

Usage:
    uv run python evaluation/summarize_eval_suite.py --tags e18_stageA_ck9030,e18b_R_final,smollm2_135m
    uv run python evaluation/summarize_eval_suite.py --all --out Cache/eval/summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LM_TASK_ORDER = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "boolq",
                 "social_iqa", "commonsense_qa", "lambada_openai", "wikitext", "mmlu", "sciq", "copa"]


def _fmt(v, nd=3):
    if v is None or v == "":
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "-"
    return f"{f:.{nd}f}"


def load_longctx(eval_root: Path, tag: str) -> dict[str, float]:
    """Flatten the suite JSON into `<probe>@<len>` (exact), `<probe>_tok@<len>`, `ce[lo,hi)`."""
    p = eval_root / tag / "longctx_suite.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    flat: dict[str, float] = {}
    for probe, res in (r.get("results") or {}).items():
        if not isinstance(res, dict):
            continue
        for k, v in res.items():
            if probe == "buckets":
                if k.startswith("ce["):
                    flat[k] = v
                continue
            if "@" not in k:
                continue
            base, ln = k.split("@", 1)
            if base.endswith("_n") or base.endswith("_first_token_acc"):
                continue
            if base.endswith("_token_acc"):
                flat[f"{base[:-len('_token_acc')]}_tok@{ln}"] = v
            else:
                flat[f"{base}@{ln}"] = v
    for probe, err in (r.get("errors") or {}).items():
        flat[f"error:{probe}"] = err.splitlines()[0] if isinstance(err, str) else str(err)
    return flat


def load_reach(eval_root: Path, tag: str) -> dict[str, float]:
    p = eval_root / tag / "reach.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    return {f"reach:{k}": v for k, v in r.items()
            if isinstance(v, (int, float)) and k not in ("seed",)}


def load_lm_eval(csv_path: Path) -> dict[str, dict[str, str]]:
    """summary.csv rows keyed by tag; `task/metric` columns collapse to `task` (stderr dropped)."""
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            norm: dict[str, str] = {}
            for k, v in row.items():
                if k is None:
                    continue
                if "/" in k:
                    task, metric = k.split("/", 1)
                    if metric.endswith("_stderr"):
                        continue
                    norm[task] = v
                else:
                    norm[k] = v
            out[row["tag"]] = norm
    return out


def _order_longctx_keys(keys: set[str]) -> list[str]:
    def sort_key(k: str):
        if k.startswith("ce["):
            lo = int(k[3:].split(",")[0])
            return (0, lo, k)
        if "@" in k:
            base, ln = k.split("@", 1)
            probe = base.replace("_tok", "")
            return (1, ["passkey", "multikey", "vt", "fwe"].index(probe) if probe in
                    ["passkey", "multikey", "vt", "fwe"] else 9, int(ln) if ln.isdigit() else 0, base.endswith("_tok"))
        if k.startswith("reach:"):
            return (2, 0, k)
        return (3, 0, k)
    return sorted(keys, key=sort_key)


def build_table(tags: list[str], eval_root: Path, lm_csv: Path) -> str:
    lm = load_lm_eval(lm_csv)
    longctx = {t: load_longctx(eval_root, t) for t in tags}
    reach = {t: load_reach(eval_root, t) for t in tags}
    lines = ["| metric | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]

    lines.append("| **reasoning (lm-eval, 0-shot)** |" + " |" * len(tags))
    lm_keys = [k for k in LM_TASK_ORDER if any(k in lm.get(t, {}) for t in tags)]
    for k in ["avg_acc", "n_acc_tasks"] + lm_keys:
        if not any(k in lm.get(t, {}) for t in tags):
            continue
        nd = 0 if k == "n_acc_tasks" else (2 if k == "wikitext" else 3)
        lines.append(f"| {k} | " + " | ".join(_fmt(lm.get(t, {}).get(k), nd) for t in tags) + " |")

    lc_keys = set().union(*(set(v) for v in longctx.values())) | set().union(*(set(v) for v in reach.values()))
    lc_keys = {k for k in lc_keys if not k.startswith("error:")}
    if lc_keys:
        lines.append("| **long context (teacher-forced)** |" + " |" * len(tags))
        for k in _order_longctx_keys(lc_keys):
            vals = [{**longctx[t], **reach[t]}.get(k) for t in tags]
            lines.append(f"| {k} | " + " | ".join(_fmt(v) for v in vals) + " |")
    errs = [(t, k, v) for t in tags for k, v in longctx[t].items() if k.startswith("error:")]
    if errs:
        lines.append("")
        lines.append("Probe errors:")
        for t, k, v in errs:
            lines.append(f"- {t} {k[len('error:'):]}: {v}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default=None, help="comma list; column order")
    p.add_argument("--all", action="store_true", help="every Cache/eval/<tag> dir plus every lm-eval row")
    p.add_argument("--eval_root", default="Cache/eval")
    p.add_argument("--lm_csv", default="Cache/Evaluation_reports/lm_eval/summary.csv")
    p.add_argument("--out", default=None, help="write the markdown table here as well")
    args = p.parse_args(argv)
    eval_root, lm_csv = Path(args.eval_root), Path(args.lm_csv)
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    elif args.all:
        tags = sorted({d.name for d in eval_root.iterdir() if d.is_dir()} | set(load_lm_eval(lm_csv))
                      if eval_root.exists() else set(load_lm_eval(lm_csv)))
    else:
        p.error("give --tags or --all")
    table = build_table(tags, eval_root, lm_csv)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()

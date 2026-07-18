#!/usr/bin/env python
"""Build deductive-stories dataset stages (graph → expand → filter → split → export).

Azure OpenAI env (see .env.example):
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_VERSION
  AZURE_OPENAI_WRITER_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT)
  AZURE_OPENAI_JUDGE_DEPLOYMENT (optional; defaults to writer)

Examples:
  uv run python scripts/build_deductive_stories.py graph --domain detective --n 20
  uv run python scripts/build_deductive_stories.py expand --in ... --mock-llm
  uv run python scripts/build_deductive_stories.py filter --in ... --mock-llm
  uv run python scripts/build_deductive_stories.py split --in ...
  uv run python scripts/build_deductive_stories.py export --in ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.deductive_stories.expand.fidelity_judge import run_fidelity_judge
from data.deductive_stories.expand.outline import build_outline
from data.deductive_stories.expand.writer import AzureChatClient, expand_story
from data.deductive_stories.filters.dedup import dedup_examples
from data.deductive_stories.filters.distractor_robustness import mark_distractor_view
from data.deductive_stories.filters.parametric_leak import run_parametric_leak_filter
from data.deductive_stories.filters.split import (
    assign_splits_by_template,
    summarize_splits,
)
from data.deductive_stories.graph.base import get_adapter, make_example_from_graph
from data.deductive_stories.io import load_examples_jsonl, save_examples_jsonl
from data.deductive_stories.materialize.hf_export import export_hf_dataset
from data.deductive_stories.materialize.length_views import materialize_length_views
from data.deductive_stories.noise.inject import apply_split_noise_policy


def _default_work_dir() -> Path:
    return Path("Cache/datasets_raw/deductive_stories")


def cmd_graph(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.domain)
    examples = []
    # Generate separate seed ranges per intended split so template families match.
    plans: list[tuple[str, int]] = []
    if args.split == "all":
        n_test = max(1, args.n // 5)
        n_val = max(1, args.n // 10)
        n_train = max(1, args.n - n_test - n_val)
        plans = [("train", n_train), ("validation", n_val), ("test", n_test)]
    else:
        plans = [(args.split, args.n)]

    seed = args.seed
    for split, count in plans:
        for i in range(count):
            graph = adapter.generate(
                seed=seed + i,
                split=split,
                distractor_ratio=args.distractor_ratio,
            )
            ex = make_example_from_graph(graph, split=split)
            ex.outline = build_outline(graph, chapters_target=args.chapters)
            examples.append(ex)
        seed += count + 1000

    out = Path(args.out)
    save_examples_jsonl(examples, out)
    print(json.dumps({"wrote": str(out), "n": len(examples), "splits": summarize_splits(examples)}))


def cmd_expand(args: argparse.Namespace) -> None:
    examples = load_examples_jsonl(Path(args.in_path))
    client = None if args.mock_llm else AzureChatClient()
    for ex in examples:
        if not ex.outline:
            ex.outline = build_outline(ex.graph, chapters_target=args.chapters)
        story, writer_id = expand_story(
            ex.graph,
            ex.outline,
            client=client,
            mock_llm=args.mock_llm,
            words_per_chapter=args.words_per_chapter,
        )
        ex.story_text = story
        ex.writer_model = writer_id
        apply_split_noise_policy(ex)
        materialize_length_views(ex)
        ok, _ = run_fidelity_judge(ex, client=client, mock_llm=args.mock_llm)
        if not ok and args.require_fidelity:
            continue
    accepted = [ex for ex in examples if ex.accepted]
    out = Path(args.out or args.in_path)
    save_examples_jsonl(accepted if args.require_fidelity else examples, out)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "input": len(examples),
                "accepted": len(accepted),
                "mock_llm": bool(args.mock_llm),
            }
        )
    )


def cmd_filter(args: argparse.Namespace) -> None:
    examples = load_examples_jsonl(Path(args.in_path))
    client = None if args.mock_llm else AzureChatClient()
    kept = []
    for ex in examples:
        mark_distractor_view(ex)
        ok_leak, _ = run_parametric_leak_filter(
            ex, client=client, mock_llm=args.mock_llm
        )
        if ok_leak:
            kept.append(ex)
    kept, dropped = dedup_examples(kept, threshold=args.dedup_threshold)
    out = Path(args.out or args.in_path)
    save_examples_jsonl(kept, out)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "kept": len(kept),
                "dedup_dropped": len(dropped),
            }
        )
    )


def cmd_split(args: argparse.Namespace) -> None:
    examples = load_examples_jsonl(Path(args.in_path))
    examples = assign_splits_by_template(examples)
    out = Path(args.out or args.in_path)
    save_examples_jsonl(examples, out)
    print(json.dumps({"wrote": str(out), "splits": summarize_splits(examples)}))


def cmd_export(args: argparse.Namespace) -> None:
    examples = load_examples_jsonl(Path(args.in_path))
    examples = [ex for ex in examples if ex.accepted]
    for ex in examples:
        if not ex.story_8k and ex.story_text:
            materialize_length_views(ex)
    out_dir = Path(args.out)
    export_hf_dataset(
        examples,
        output_dir=out_dir,
        push_repo=args.push_repo,
        private=not args.public,
    )
    print(json.dumps({"saved": str(out_dir), "n": len(examples), "pushed": args.push_repo}))


def cmd_pilot(args: argparse.Namespace) -> None:
    """One-shot small end-to-end run (graph→expand→filter→split→export)."""
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    graph_path = work / f"graphs_{args.domain}.jsonl"
    expanded_path = work / f"expanded_{args.domain}.jsonl"
    filtered_path = work / f"filtered_{args.domain}.jsonl"
    export_dir = work / f"hf_{args.domain}"

    args.out = str(graph_path)
    args.split = "all"
    args.n = args.n
    cmd_graph(args)

    expand_args = argparse.Namespace(
        in_path=str(graph_path),
        out=str(expanded_path),
        mock_llm=args.mock_llm,
        chapters=args.chapters,
        words_per_chapter=args.words_per_chapter,
        require_fidelity=True,
    )
    cmd_expand(expand_args)

    filter_args = argparse.Namespace(
        in_path=str(expanded_path),
        out=str(filtered_path),
        mock_llm=args.mock_llm,
        dedup_threshold=0.85,
    )
    cmd_filter(filter_args)

    split_args = argparse.Namespace(in_path=str(filtered_path), out=str(filtered_path))
    cmd_split(split_args)

    export_args = argparse.Namespace(
        in_path=str(filtered_path),
        out=str(export_dir),
        push_repo=None,
        public=False,
    )
    cmd_export(export_args)
    print(json.dumps({"pilot_done": True, "export": str(export_dir)}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("graph", help="Generate solver-verified graphs + outlines")
    g.add_argument("--domain", required=True, choices=["detective", "se_debug"])
    g.add_argument("--n", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    g.add_argument("--distractor-ratio", type=float, default=0.35)
    g.add_argument("--chapters", type=int, default=8)
    g.add_argument("--out", type=str, default=str(_default_work_dir() / "graphs.jsonl"))
    g.set_defaults(func=cmd_graph)

    e = sub.add_parser("expand", help="LLM (or mock) narrative expansion + fidelity")
    e.add_argument("--in", dest="in_path", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--mock-llm", action="store_true")
    e.add_argument("--chapters", type=int, default=8)
    e.add_argument("--words-per-chapter", type=int, default=200)
    e.add_argument("--require-fidelity", action="store_true", default=True)
    e.set_defaults(func=cmd_expand)

    f = sub.add_parser("filter", help="Parametric-leak + dedup filters")
    f.add_argument("--in", dest="in_path", required=True)
    f.add_argument("--out", default=None)
    f.add_argument("--mock-llm", action="store_true")
    f.add_argument("--dedup-threshold", type=float, default=0.85)
    f.set_defaults(func=cmd_filter)

    s = sub.add_parser("split", help="Enforce template-id train/test firewall")
    s.add_argument("--in", dest="in_path", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_split)

    x = sub.add_parser("export", help="Write HF DatasetDict (+ optional push)")
    x.add_argument("--in", dest="in_path", required=True)
    x.add_argument("--out", type=str, default=str(_default_work_dir() / "hf_export"))
    x.add_argument("--push-repo", default=None, help="e.g. ksopyla/deductive-stories")
    x.add_argument("--public", action="store_true")
    x.set_defaults(func=cmd_export)

    pilot = sub.add_parser("pilot", help="Small end-to-end local pilot")
    pilot.add_argument("--domain", required=True, choices=["detective", "se_debug"])
    pilot.add_argument("--n", type=int, default=12)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--distractor-ratio", type=float, default=0.35)
    pilot.add_argument("--chapters", type=int, default=6)
    pilot.add_argument("--words-per-chapter", type=int, default=120)
    pilot.add_argument("--mock-llm", action="store_true", default=True)
    pilot.add_argument("--no-mock-llm", action="store_false", dest="mock_llm")
    pilot.add_argument("--work-dir", type=str, default=str(_default_work_dir() / "pilot"))
    pilot.set_defaults(func=cmd_pilot)

    return p


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

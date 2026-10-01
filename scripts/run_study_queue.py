#!/usr/bin/env python
"""Generate per-GPU queue scripts for a study plan (probe jobs + optional length ladder).

A *plan* is a Python module under `scripts/study_plans/` exposing
`jobs(phase: str) -> list[dict]`. Each job dict:

    name      unique id; results go to <out>/<name>/
    args      probe flags (list of str), appended after the plan's BASE flags
    cost      estimated GPU-hours (balancing only)
    init      name of the job whose checkpoint this one starts from (curriculum), or None
    ladder    lengths for verification/length_ladder.py after training, or None
    rows      ladder rows per length (default 64)
    save      keep the checkpoint (default: True when a ladder runs or a child needs it)

Jobs linked by `init` form a chain: a child waits for its parent (queued on the same host) to
write `DONE` or `FAILED`, so a chain may span GPUs; parents are queued first. A job writes
`DONE` only when training (exit 0 or 2) and its ladder both succeed, otherwise `FAILED`;
a child whose parent has no `DONE` is skipped. Re-running a script skips `DONE` jobs.

    uv run python scripts/run_study_queue.py --plan e30_vs_e31 --phase ratio \\
        --out Cache/study/e30_vs_e31 --host odra --gpus 0 1 2 --mode scripts
"""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBE = "verification/bapo_capability_probe.py"
LADDER = "verification/length_ladder.py"


def load_plan(name: str):
    return importlib.import_module(f"scripts.study_plans.{name}")


def depth(j: dict, by: dict[str, dict]) -> int:
    d, seen = 0, set()
    while j.get("init") in by:
        if j["name"] in seen:
            raise SystemExit(f"init cycle at {j['name']}")
        seen.add(j["name"])
        j, d = by[j["init"]], d + 1
    return d


def assign(jobs: list[dict], gpus: list[str]) -> dict[str, list[dict]]:
    """Parents before children (by depth), longest-first within a depth, least-loaded GPU.
    A child waits for its parent's DONE/FAILED marker, so a chain may span GPUs on one host."""
    by = {j["name"]: j for j in jobs}
    if len(by) != len(jobs):
        raise SystemExit("duplicate job names in plan")
    load = {g: 0.0 for g in gpus}
    out: dict[str, list[dict]] = {g: [] for g in gpus}
    for j in sorted(jobs, key=lambda j: (depth(j, by), -float(j.get("cost", 1.0)))):
        g = min(load, key=load.get)
        out[g].append(j)
        load[g] += float(j.get("cost", 1.0))
    return out


def job_lines(j: dict, *, out: Path, base: list[str], parents: set[str], local: set[str]) -> list[str]:
    d = out / j["name"]
    q = shlex.quote
    save = j.get("save", bool(j.get("ladder")) or j["name"] in parents)
    cmd = ["uv", "run", "python", PROBE, *base, *j["args"], "--out", str(d)]
    if save:
        cmd += ["--save_ckpt", "@out"]
    if j.get("init"):
        cmd += ["--init_ckpt", str(out / j["init"])]
    lines = []
    if j.get("init") in local:  # parent queued on this host: wait for it to finish (either way)
        pd = out / j["init"]
        lines += [f'[ -f {q(str(d / "DONE"))} ] || until [ -f {q(str(pd / "DONE"))} ] || [ -f {q(str(pd / "FAILED"))} ]; '
                  f'do sleep 60; done']
    lines += [f'if [ -f {q(str(d / "DONE"))} ]; then echo "skip {j["name"]} (done)"']
    if j.get("init"):
        lines += [f'elif [ ! -f {q(str(out / j["init"] / "DONE"))} ]; then echo "SKIP {j["name"]} (parent {j["init"]} not done)"']
    lines += [
        "else",
        f'  echo "JOB {j["name"]} $(date +%Y-%m-%dT%H:%M:%S)"; mkdir -p {q(str(d))}; rm -f {q(str(d / "FAILED"))}',
        f"  {shlex.join(cmd)} > {q(str(d / 'probe.log'))} 2>&1",
        f'  rc=$?; echo "EXIT-TRAIN {j["name"]} $rc"',
    ]
    if j.get("ladder"):
        lad = ["uv", "run", "python", LADDER, "--ckpt", str(d), "--lengths", *map(str, j["ladder"]),
               "--rows", str(j.get("rows", 64))]
        lines += [
            "  lrc=0",
            f"  if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then {shlex.join(lad)} > {q(str(d / 'ladder.log'))} 2>&1; lrc=$?; "
            f'echo "EXIT-LADDER {j["name"]} $lrc"; fi',
        ]
    else:
        lines += ["  lrc=0"]
    lines += [
        f'  if {{ [ $rc -eq 0 ] || [ $rc -eq 2 ]; }} && [ $lrc -eq 0 ]; then touch {q(str(d / "DONE"))}; '
        f'else touch {q(str(d / "FAILED"))}; fi',
        "fi",
    ]
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="odra")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--only", nargs="*", default=None, help="job-name substrings to keep")
    p.add_argument("--mode", default="print", choices=("print", "scripts"))
    p.add_argument("--repo_dir", default=str(ROOT))
    p.add_argument("--wait_for", nargs="*", default=[],
                   help="queue logs that must contain QUEUE-END before this queue starts (no GPU sharing); "
                   "'{g}' is replaced by the GPU id")
    args = p.parse_args()

    plan = load_plan(args.plan)
    jobs = plan.jobs(args.phase)
    if args.only:
        keep = {j["name"] for j in jobs if any(s in j["name"] for s in args.only)}
        # keep parents of kept jobs so chains stay whole
        by = {j["name"]: j for j in jobs}
        for n in list(keep):
            while by[n].get("init") in by:
                n = by[n]["init"]
                keep.add(n)
        jobs = [j for j in jobs if j["name"] in keep]
    out = Path(args.out)
    base = list(plan.BASE)
    parents = {j["init"] for j in jobs if j.get("init")}
    by_gpu = assign(jobs, args.gpus)
    local = {j["name"] for j in jobs}
    total = sum(float(j.get("cost", 1.0)) for j in jobs)
    print(f"plan {args.plan}/{args.phase}: {len(jobs)} jobs, ~{total:.0f} GPU-h on {len(args.gpus)} GPUs "
          f"(~{total / max(len(args.gpus), 1):.1f} h wall)")
    for g, js in by_gpu.items():
        print(f"  GPU {g}: {len(js)} jobs, ~{sum(float(j.get('cost', 1.0)) for j in js):.1f} h")
    if args.mode == "print":
        for j in jobs:
            print(f"[{j['name']}] init={j.get('init')} ladder={j.get('ladder')} :: {' '.join(j['args'])}")
        return 0
    launch = out / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    (out / f"plan_{args.phase}_{args.host}.json").write_text(json.dumps(
        {"plan": args.plan, "phase": args.phase, "base": base, "jobs": jobs,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=ROOT).stdout.strip()}, indent=2))
    starter = ["#!/usr/bin/env bash", "byobu has-session -t study 2>/dev/null || byobu new-session -d -s study"]
    for g, js in by_gpu.items():
        if not js:
            continue
        path = launch / f"{args.host}_{args.phase}_gpu{g}.sh"
        log = launch / f"{args.host}_{args.phase}_gpu{g}.log"
        lines = [
            "#!/usr/bin/env bash",
            f"# study {args.plan}/{args.phase} · host {args.host} · GPU {g} · {len(js)} jobs",
            "set -uo pipefail",
            f"cd {shlex.quote(args.repo_dir)}",
            'export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2',
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"export CUDA_VISIBLE_DEVICES={g}",
            f"exec > >(tee -a {shlex.quote(str(log))}) 2>&1",
            *[f"until grep -q QUEUE-END {shlex.quote(w.replace('{g}', str(g)))} 2>/dev/null; do sleep 120; done"
              for w in args.wait_for],
            # never share a GPU: wait until it is idle
            f"while nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done",
        ]
        for j in js:
            lines += [f"while nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done"]
            lines += job_lines(j, out=out, base=base, parents=parents, local=local)
        lines += ['echo "QUEUE-END $(date +%Y-%m-%dT%H:%M:%S)"']
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)
        starter.append(f"byobu new-window -t study -n {args.phase}_g{g} {shlex.quote('bash ' + str(path))}")
        print(f"  wrote {path}")
    st = launch / f"{args.host}_{args.phase}_start.sh"
    st.write_text("\n".join(starter) + "\n")
    st.chmod(0o755)
    print(f"  start: bash {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

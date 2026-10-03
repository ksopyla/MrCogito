# Capability ledger

Compact, committed copies of every capability-suite and length-battery result, one JSON file per
result folder and server: `suite/<run>.<host>.json`, `study/<study>.<host>.json`. Every past result
can be compared, scored and drawn from here with the servers off. The raw folders (logs, checkpoints,
every ladder) are archived on the NAS at the `archive_path` inside each file.

- Process and rules: [capability checks](../../engineering_specs/capability_checks.md) · skill `capability-checks`
- Add or refresh a file: `bash scripts/pull_capability_results.sh <host> <remote folder>` (also archives to the NAS), then commit
- List: `uv run python analysis/capability_ledger.py summary`
- Score a suite run: `uv run python analysis/capability_scorecard.py --in_dir suite/<file>.json --out_dir Cache/capability/<run>_scored`
- Draw everything: `uv run python analysis/capability_board.py` → `docs/3_Evaluations_and_Baselines/capability_board.html`
- Schema: the docstring of `analysis/capability_ledger.py` (schema 1). One job per line, so diffs show what changed.

Like the rest of `2_Experiments_Registry/`, results here are append-only: a refresh rewrites a file from
its folder (new jobs appear, finished jobs keep their numbers); never edit numbers by hand.

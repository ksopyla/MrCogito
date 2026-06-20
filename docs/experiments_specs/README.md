# Experiments

Two files per experiment, joined by the `<ID>`:
- `<ID>.md` — the **frozen spec** (*intent*): one hypothesis, the foundation it builds on, the one variable it changes, success/kill criteria decided **before** running. From [TEMPLATE.md](TEMPLATE.md).
- `<ID>_plan.md` — the **implementation plan** (*design*): which modules to reuse/extend, the forward pass with shapes, inputs/data, loss, config, launch, tests, risks. From [PLAN_TEMPLATE.md](PLAN_TEMPLATE.md).

The *implementation* is never a new script — it is args/config over the shared foundation
(env-var overrides on the existing bash launcher). See `.cursor/rules/experiment-discipline.mdc`.

This folder is **self-indexing** (filenames = IDs) — no manual index table (it would drift,
same reason `literature_review/` has none).

## Where things live (no duplicate logs)
- **Intent / criteria** → `docs/experiments_specs/<ID>.md` (spec) and `<ID>_plan.md` (design).
- **Results** → `docs/2_Experiments_Registry/master_experiment_log.md` (canonical, append-only) + run reports. The specs are *not* a results log.
- **Live one-line memory** → `docs/1_Strategy_and_Plans/agenda.md` (pointers into the ledger).

## Lifecycle
1. `experiment-design` → `<ID>.md` spec (the frame).
2. `implementation-plan` → `<ID>_plan.md` (repo-rooted design), pulling from `research-explain` / `research-synthesis` when a paper is involved.
3. `research-implement` implements the plan; the spec is **frozen** when the run starts (`Status: active`).
4. `experiment-track` records results to `master_experiment_log.md` + a run report, sets the spec `Status: done|killed`, fills its **Result** link, and adds a one-line `agenda.md` learning.

## ID scheme
`E0NN_short_slug` — zero-padded, stable, never reused. The ID joins
spec ↔ plan ↔ launch/run ↔ run report ↔ checkpoint ↔ WandB run.

The live driver is [`../1_Strategy_and_Plans/agenda.md`](../1_Strategy_and_Plans/agenda.md).

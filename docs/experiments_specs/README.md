# Experiments

Two files per experiment, joined by the `<ID>`:
- `<lifecycle>/<ID>.md` — the **frozen spec** (*intent*): one hypothesis, the foundation it builds on, the one variable it changes, success/kill criteria decided **before** running. From [TEMPLATE.md](TEMPLATE.md).
- `<lifecycle>/<ID>_plan.md` — the **implementation plan** (*design*): which modules to reuse/extend, the forward pass with shapes, inputs/data, loss, config, launch, tests, risks. From [PLAN_TEMPLATE.md](PLAN_TEMPLATE.md).

The *implementation* is never a new script — it is args/config over the shared foundation
(env-var overrides on the existing bash launcher). See `.cursor/rules/project-overview.mdc`.

The lifecycle folders provide a quick view of the research history:

- [`ahead/`](ahead/) — draft, approved, on-hold, active, or otherwise not closed
- [`done_success/`](done_success/) — completed and passed its decisive criterion, including a
  control that delivered the pre-registered decisive answer
- [`done_failed/`](done_failed/) — completed or killed without establishing its proposed
  mechanism; mixed partial positives belong here when the decisive criterion failed
- [`canceled/`](canceled/) — deliberately rejected, superseded, or stopped before a run

Each folder remains **self-indexing** (filenames = IDs); there is no manual table that can
drift. Keep the spec and plan together when both exist.

To resolve an experiment by ID, search all four lifecycle folders. Never assume a spec is at
the root, and never reuse an ID found in any lifecycle folder.

## Where things live (no duplicate logs)
- **Intent / criteria** → `docs/experiments_specs/<lifecycle>/<ID>.md` (spec) and
  `<lifecycle>/<ID>_plan.md` (design).
- **Results** → `docs/2_Experiments_Registry/master_experiment_log.md` (canonical, append-only) + run reports. The specs are *not* a results log.
- **Live one-line memory** → `docs/1_Strategy_and_Plans/agenda.md` (pointers into the ledger).

## Lifecycle
1. `experiment-design` creates the new spec in `ahead/`.
2. `implementation-plan` creates `ahead/<ID>_plan.md`, pulling from `research-explain` /
   `research-synthesis` when a paper is involved.
3. `research-implement` implements the plan; the spec is **frozen** when the run starts
   (`Status: active`).
4. `experiment-track` records results to `master_experiment_log.md` + a run report, updates
   the spec `Status` and `Result`, then moves the spec/plan pair to `done_success/` or
   `done_failed/`. A deliberately abandoned design moves to `canceled/`.

## ID scheme
`E0NN_short_slug` — zero-padded, stable, never reused. The ID joins
spec ↔ plan ↔ launch/run ↔ run report ↔ checkpoint ↔ WandB run.

The live driver is [`../1_Strategy_and_Plans/agenda.md`](../1_Strategy_and_Plans/agenda.md).

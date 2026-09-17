# Experiments

Two files per experiment, joined by the `<ID>`:
- `<lifecycle>/<ID>.md` — the **frozen spec** (*intent*): one coherent architectural
  hypothesis, the foundation it builds on, what is materially new, success/kill criteria
  decided **before** running. From [TEMPLATE.md](TEMPLATE.md). Prefer bold bets over
  micro-A/Bs; see `project-overview.mdc` → Research Stance.
- `<lifecycle>/<ID>_plan.md` — the **implementation plan** (*design*): which modules to
  reuse/extend, the forward pass with shapes, inputs/data, loss, config, launch, tests,
  risks. From [PLAN_TEMPLATE.md](PLAN_TEMPLATE.md). Plans must not water down a bold spec.

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
- **Results index** → `docs/2_Experiments_Registry/master_experiment_log.md`
  (Experiment Index + lean Training Runs — append-only, scannable; not a lab notebook).
  Deep metrics → `run_reports/`. Specs are *not* a results log.
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
The ID joins spec ↔ plan ↔ launch/run ↔ run report ↔ checkpoint ↔ WandB run.
Never reuse an ID found in any lifecycle folder.

**When to design at all.** Do not create a spec in `ahead/` unless the user asked
to design or run an experiment.

**Family `E0NN`.** The family is the integer ID (E18, E21, …). Assign a new
family number only if the user asked for a new experiment family, or approved a
bet that is a different encode/reason/decode object. Zero-padded, globally
unique, never reused. Do not take the next free integer just because it is
free. E26–E29 were flavours/runs of E21 that were mis-numbered; do not repeat
that.

**Flavours (default for a small detail).** A small architecture change inside
the current family — extra loss head, identity keys, a loop, a mask — is a
flavour of **that** family: `E{NN}a`, `E{NN}b`, `E{NN}c`, … Sequential
lowercase letters on the current family's number. Never reuse a letter inside
a family. The unlettered ID is the parent; the first small change is `a`
(E21 → `E21a`, not a new E26; E18 → `E18a`). Same order of compute / memory /
parameters as the parent unless the user changes scale. Spec path:
`docs/experiments_specs/ahead/E{NN}a_….md`.

**Not a flavour / not a new family.** A new dataset, probe, eval protocol,
length ladder, dense control, or optimizer/LR knob is a *run* of the current
family or flavour ID, logged under that ID — not a new spec ID.

**When in doubt.** Small delta on the current family → next unused flavour
letter of that family. User did not ask → write nothing in `ahead/`.

The live driver is [`../1_Strategy_and_Plans/agenda.md`](../1_Strategy_and_Plans/agenda.md).

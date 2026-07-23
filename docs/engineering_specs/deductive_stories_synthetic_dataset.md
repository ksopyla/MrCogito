# Deductive Stories — synthetic long-context dataset (engineering spec)

> **SEPARATE PROJECT (2026-07-19).** Intent and design below stay as the research
> trace. The in-repo v0 prototype (shallow graph → blunt LLM expansion) produced
> quality far below a hard long-context deduction corpus and was **removed from
> MrCogito**. Implement and iterate this dataset in its **own repository**; only
> consume a finished HF dataset here via mix recipes / eval. Do not re-land the
> generation pipeline under `data/` until that external project meets the quality
> bar in this spec.

- **Type:** engineering foundation (dataset generation + HF publish + mix integration). **Not** an `E0NN` experiment.
- **Status:** **deferred to a separate repo** (2026-07-19). Spec retained as the design contract. In-MrCogito code/prototype removed after Azure pilot showed insufficient narrative / hardness quality.
- **Owner:** Krzysztof Sopyla
- **Serves:** hard train+eval signal for windowed-AR (E05/E10 family), Gemma concept memory (E16), Concept-Flow (E08), and revived concept-conditioned masked diffusion — i.e. “can concepts carry multi-hop deduction over long narrative?”
- **Primary goal:** ship a **graph-first, solver-verified**, multi-domain long narrative corpus with **1–5 exact-match questions**, published on Hugging Face, and wired into existing recipe→pretok mixes without contaminating the frozen eval split.

## Problem (why this is needed)

Current long mixes (`smollm3_inspired_2k_e05`, `e16b_long_4k_v1`) teach language and some CoT traces, but do **not** provide:

1. a **train-disjoint, hard, verifiable** long-horizon deduction gate (8K–16K),
2. a **controllable** difficulty / distractor / hop-depth axis,
3. a training ingredient that forces the model to bind scattered evidence into a small answerable set.

Existing in-repo synthetic data (`data/delayed_recall.py`) only tests sparse key→value recall — not narrative deduction. E14/E15 already showed that a weak protocol can fail for non-architectural reasons; this dataset must not repeat that.

## Locked decisions (grill session 2026-07-18)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Primary job | **C** — both train + eval; hard firewall; **eval frozen & published first** |
| 2 | Gold answer authority | **B** — solver/world-model writes the answer; LLM judge may only veto *narrative fidelity* |
| 3 | Intermediate gold structure | **D** — shared **typed event graph** + per-domain compilers/adapters |
| 4 | Answer form (eval gate) | **A** — short **exact-match** (entity / id / number / enum / yes-no); optional MC is secondary view only |
| 5 | Length regime | **Dual views from one graph** — full **8K / 16K** stories for eval (+ OOD length); **2K / 4K** condensed or chapter-sliced views for training (matches live launchers) |
| 6 | Noise policy | **Staged** — train: mostly in-domain red herrings; eval: higher distractor density + optional OOD haystack inserts |
| 7 | Domain MVP | **v0:** detective + SE debugging; **v1:** logistics + riddles (same schema, new adapters) |
| 8 | Scale targets | **Eval:** ~800 hard stories (v0: 400 detective + 400 SE); **Train:** ~10k accepted MVP → ~20k when v1 lands |
| 9 | Mix weight | Start at **5%** of a long recipe (peer to `big_reasoning_traces`); raise to 10% only if CE / geometry stay healthy |
| 10 | Pipeline family | **MuSR-style chaptered expansion + ProverGen-style solver gold + QwenLong-style robustness filters** |

Treat user domain ideas (Sherlock-like detective, long debug sessions, logistics, riddles) as **flavor clues**, not literary requirements. Hardness and verifiability beat prose quality.

---

## What this dataset must prove (and what would kill it)

**Claim (dataset quality, not model):** Accepted rows are (i) answerable from the published context, (ii) **not** answerable without it, (iii) stable under controlled distractors, (iv) exact-match scorable offline.

**Kill criteria for a generation batch (reject / regenerate, do not publish):**

| Check | Kill if |
|-------|---------|
| Solver consistency | Re-solving the stored graph yields a different answer |
| Parametric leak | Strong teacher answers correctly **without** the story (pass@k > 0 on no-context) |
| Distractor fragility | Adding approved distractors flips the gold or makes pass@k → 0 for a capable teacher that previously solved it |
| Narrative fidelity | Judge finds missing gold facts / contradictions vs graph (threshold: any critical fact missing) |
| Length | Tokenized story outside target bucket (±10%) after expansion |
| Dedup | Near-duplicate of another row (MinHash / embedding) or template-id collision across train↔eval |

**Downstream model gates** (used by experiment-evaluate later; not this eng ship gate): exact-match accuracy on frozen `test` at 8K; kill/interpret via concept ablations (`no-concept`, `shuffle-concepts`) — defined in experiment specs, not here.

---

## Pipeline (canonical)

```text
idea seed / domain template
    → typed event graph (programmatic)
    → query set (1–5) + solver gold answers
    → chapter plan (outline tied to graph nodes)
    → LLM narrative expansion (chaptered; no answer leakage in prose constraints)
    → fidelity judge (LLM; reject-only)
    → length + noise injection (in-domain / OOD per split policy)
    → robustness filters (no-context fail; distractor survive)
    → dedup + template-holdout split
    → dual length materialization (2K/4K train views; 8K/16K eval views)
    → HF publish (eval first) + local pretok / mix recipe hook
```

**Do not** prompt an LLM for a 16K story and then ask it for QAs. Graph and answers exist first.

### Stage contracts

| Stage | Input | Output | Owner |
|-------|-------|--------|-------|
| 1 Idea / template | domain, difficulty knobs | `template_id`, parameter draw | domain adapter |
| 2 Graph | template draw | `event_graph` JSON (nodes/edges/attrs) | domain adapter |
| 3 Queries | graph | `questions[]` + `answers[]` + `support_node_ids[]` | shared query engine |
| 4 Solve | graph + queries | gold answers + optional proof path | **solver** (hard) |
| 5 Outline | graph | chapter outline with fact coverage checklist | shared |
| 6 Expand | outline | `story_text` (chapters) | LLM writer |
| 7 Fidelity | story + graph | pass/fail + missing facts | LLM judge (reject-only) |
| 8 Noise | story + policy | `story_text_noisy` | shared injector |
| 9 Robustness | story + QAs | accept/reject | teacher probes |
| 10 Materialize | accepted row | length views + HF rows | shared exporter |

---

## Shared typed event graph (schema v0)

Minimal shared schema (domain adapters may add typed attrs; core fields are mandatory):

```json
{
  "schema_version": "deductive_graph_v0",
  "domain": "detective|se_debug|logistics|riddle",
  "template_id": "detective.alibi_v2",
  "seed": 42,
  "entities": [{"id": "E1", "type": "person|service|sku|...", "name": "...", "attrs": {}}],
  "events": [{"id": "V1", "type": "...", "time": 0, "actors": ["E1"], "attrs": {}, "text_seed": "..."}],
  "relations": [{"src": "V1", "dst": "V2", "type": "causes|precedes|contradicts|depends_on|..."}],
  "hidden_state": {},
  "queries": [
    {
      "qid": "Q1",
      "type": "who|what|when|which|yesno|enum|number",
      "prompt_template": "...",
      "answer_type": "entity_id|enum|number|bool|string_norm",
      "support_node_ids": ["V3", "V7"],
      "hop_depth": 3
    }
  ],
  "gold": [{"qid": "Q1", "value": "E4", "normalized": "e4", "solver": "reachability_v0"}]
}
```

**Invariant:** every `gold.value` must be reproducible by a pure function `solve(graph, query) → value` with **no LLM**.

### Domain adapters (what each compiles to)

| Domain | World model (v0/v1) | Typical queries | Plausible in-domain noise |
|--------|---------------------|-----------------|---------------------------|
| **Detective** (v0) | Timeline + alibis + clue→culprit DAG (MuSR-like) | culprit id, weapon, location, true/false claim | false leads, irrelevant witnesses, red-herring motives |
| **SE debugging** (v0) | Service dependency graph + failing symptom → root cause | root cause service/id, failing config key, blast radius | unrelated log lines, red-herring stack traces, concurrent incidents |
| **Logistics** (v1) | Constraint / routing state (capacities, ETAs, hubs) | feasible/infeasible, bottleneck hub, delay cause | decoy shipments, irrelevant weather notes |
| **Riddles** (v1) | Small closed world with unique satisfying assignment | single entity/property that fits all constraints | near-miss entities, distractor constraints that look relevant |

---

## Narrative, noise, and length

### Narrative expansion

- Chaptered generation (MuSR): each chapter must cover a checklist of graph nodes; uncovered → regenerate chapter.
- Writer **must not** emit the gold answer string in an isolated “solution” paragraph; answers appear only as distributed facts.
- Providers: pluggable (Azure OpenAI / GLM / local teacher). Keys only via `.env`; never committed.

### Noise (makes sense = domain-plausible)

| Split | Policy |
|-------|--------|
| `train` | In-domain red herrings only; distractor ratio moderate; optional mild chapter padding |
| `validation` | Same family as train; slightly higher distractor ratio |
| `test` (frozen) | Higher distractor density; **plus** optional OOD haystack inserts (PG19-style paragraphs) for a labeled `test_ood_noise` subset |
| Length OOD | Train views ≤4K; `test_16k` never seen in training |

Noise must be **tagged** in metadata (`noise_kind`, `noise_ratio`) so ablations can strip it.

### Length materialization

From one accepted graph+story:

| View | Target tokens | Use |
|------|---------------|-----|
| `story_2k` | ~2048 | train mix for E05/E10 |
| `story_4k` | ~4096 | train mix for E16b |
| `story_8k` | ~8192 | primary eval |
| `story_16k` | ~16384 | length OOD eval |

Condensation rules (pick one implementation; document in code):

1. **Chapter subset** — keep a minimal chapter cover of `support_node_ids` + controlled distractor chapters, or
2. **Outline densification** — regenerate shorter prose from the same outline with a max-token budget.

Same `example_id` across views; eval always uses the long view.

---

## Questions, scoring, hardness

- **Count:** 3 questions default per story (allowed range 1–5); eval prefers 3–5 with mixed hop depths.
- **Answer type:** short exact-match after normalization (lower/strip; entity ids canonicalized; numbers as canonical float/int).
- **Primary metric:** mean exact-match accuracy over questions (micro and macro per story).
- **Secondary (optional, not gate):** MC view generated from gold + hard negatives drawn from sibling entities — for humans only.
- **Hardness knobs (must be in metadata):** `hop_depth`, `n_entities`, `n_events`, `distractor_ratio`, `noise_kind`, `length_bucket`.

“Hard” means: high hop depth + high distractor ratio + parametric-leak filtered + teacher can still solve *with* context. Not “long and vague.”

---

## Splits, size, contamination firewall

### Splits

| Split | Role | Size (targets) |
|-------|------|----------------|
| `train` | Mix ingredient | v0 ~10k accepted; v1 ~20k |
| `validation` | Tuning / early stop probes | ~500 |
| `test` | **Frozen public eval** | v0 ~800 (400+400); expand with v1 domains later **without rewriting** old `test` rows |
| `test_16k` | Length OOD | subset or parallel materialization of `test` |
| `test_ood_noise` | Noise OOD | labeled subset of `test` |

**Firewall rules (non-negotiable):**

1. Disjoint `template_id` families across train vs test (GSM-Symbolic-style holdout).
2. Disjoint entity-name banks / ontology vocab where applicable.
3. Embedding / MinHash near-dup removal across the whole corpus before split finalize.
4. No regeneration of `test` after publish; append new configs instead.
5. Scan overlap vs public benchmarks we care about (DetectiveQA, MuSR, BABILong prompts) — record in dataset card.

### Mix integration (existing foundation)

- Publish HF dataset → add source to `data/mix_recipes/*.json` with weight **0.05**, `text_columns: ["story_text"]` or a packed `text` field that includes story + question + answer for causal LM (see Training views below).
- Pretokenize via `scripts/pretokenize_mix.py` into a **new cache tree** if seqlen/tokenizer differs.
- Prefer `story_2k` / `story_4k` columns for training; never silently truncate `story_8k` in a 2K run without a dedicated condensed view.

**Training text packing (v0 recommendation):**

```text
{story}

Question: {q}
Answer: {a}
```

Only answer tokens (or answer+short suffix) need to be supervised if we add a custom collator later; v0 can use full causal LM on the packed text (weaker but zero new trainer code). Custom answer-mask collator is a **follow-up eng item**, not a blocker for dataset publish.

---

## Hugging Face publish structure

**Proposed repo:** `ksopyla/deductive-stories` (name TBD at upload).

### Configs

- `detective`, `se_debug`, (`logistics`, `riddle` in v1)
- optional: `all` concatenated

### Splits

`train`, `validation`, `test`, plus length views as columns (not separate repos).

### Row schema (published columns)

| Column | Type | Notes |
|--------|------|-------|
| `example_id` | string | stable UUID |
| `domain` | string | |
| `template_id` | string | holdout key |
| `split` | string | |
| `story_text` | string | canonical long (8K) |
| `story_2k` / `story_4k` / `story_16k` | string | optional materializations |
| `questions` | list[string] | 1–5 |
| `answers` | list[string] | normalized gold |
| `answers_raw` | list[string] | pre-normalization |
| `support_node_ids` | list[list[string]] | per question |
| `event_graph` | string (JSON) | full graph |
| `hop_depths` | list[int] | |
| `distractor_ratio` | float | |
| `noise_kind` | string | |
| `length_tokens_smollm` / `_gemma` | int | measured |
| `writer_model` / `judge_model` | string | provenance |
| `solver_id` | string | |
| `generation_version` | string | e.g. `deductive_stories_v0.1` |

### Dataset card must include

- Pipeline stages + verification stack
- Teacher/judge model names and dates
- Contamination / holdout statement
- Cost and acceptance rate summary
- License (synthetic; note any seed corpora for OOD noise)
- Link to generation code in this repo

### Sharding

Parquet; ~5k–10k rows/file; versioned regenerations get a new `generation_version`, never silent overwrite of `test`.

---

## Cost & compute budget (order of magnitude)

Assumptions: ~3–5× rejection sampling; ~12K narrative tokens; API frontier writer+judge.

| Tier | $/accepted story | 10k train + 0.8k eval |
|------|------------------|------------------------|
| Local/open teacher | ~$0.05–0.15 | ~$500–1.6k |
| API frontier | ~$0.20–0.65 | ~$2k–7k |

**Policy:** prefer **local/open** writer for train volume; reserve frontier models for eval-quality narrative and judge veto. Cap monthly spend in `.env` / run config; log accept/reject rates per stage.

Pilot before full spend: **100 stories/domain** through the full filter stack; measure accept rate; only then scale.

---

## Repo layout (to implement)

```text
data/deductive_stories/
  schema.py              # graph + row pydantic/dataclasses
  graph/
    base.py              # typed event graph + solve() interface
    detective.py         # v0 adapter
    se_debug.py          # v0 adapter
    logistics.py         # v1
    riddle.py            # v1
  expand/
    outline.py
    writer.py            # provider-agnostic LLM client
    fidelity_judge.py
  noise/
    inject.py            # in-domain + OOD haystack
  filters/
    parametric_leak.py
    distractor_robustness.py
    dedup.py
  materialize/
    length_views.py
    hf_export.py
  prompts/               # versioned prompt templates
scripts/
  build_deductive_stories.py   # CLI stages: graph|expand|filter|split|export|pretok-hook
tests/
  deductive_stories/     # solver round-trip, split firewall, schema, normalization
```

Pattern after: `data/delayed_recall.py` + `scripts/build_delayed_recall_dataset.py` (deterministic synth → disk → manifest).

**Config, not fork:** dataset becomes an HF source in mix recipes; no new training entrypoint.

---

## CLI stages (minimum)

```bash
uv run python scripts/build_deductive_stories.py graph   --domain detective --n 500 --seed 42
uv run python scripts/build_deductive_stories.py expand  --in ... --writer $WRITER
uv run python scripts/build_deductive_stories.py filter  --in ... --judge $JUDGE
uv run python scripts/build_deductive_stories.py split   --in ... --holdout-templates ...
uv run python scripts/build_deductive_stories.py export  --in ... --hf ksopyla/deductive-stories --push
```

Each stage is resumable; writes JSONL/Parquet intermediates under `Cache/` or `$DATASETS_RAW_DIR` (gitignored).

---

## Verification & tests (eng acceptance)

Ship is “done” when:

1. Solver round-trip tests pass for each v0 adapter (graph → gold → re-solve).
2. Split firewall test fails CI if train/test share `template_id`.
3. Normalization tests cover entity/number/bool.
4. Pilot 100+100 rows pass parametric-leak + fidelity filters at documented accept rates.
5. HF dataset card + `test` split published; SHA / revision pinned in this spec’s changelog section.
6. A mix recipe snippet documented (not necessarily merged into production train until an experiment asks).

---

## Out of scope (explicit)

- Training a model on this data (experiment specs / experiment-run).
- Replacing delayed-recall diagnostics.
- Human literary editing of stories.
- Claiming DetectiveQA-level quality on day one.
- 8K/16K **training** runs (eval/materialization only until launchers and GPUs support it).

---

## Implementation order

1. Schema + detective + se_debug solvers + unit tests (no LLM).
2. Outline + writer + fidelity judge (provider-agnostic) + pilot 100/domain.
3. Noise + robustness filters + dedup + split firewall.
4. Length materialization (2K/4K/8K/16K).
5. HF export + dataset card; **publish `test` first**.
6. Scale train to ~10k; add mix recipe weight 0.05; pretok hook.
7. v1 adapters: logistics, riddles.

---

## Open items to define at implementation time (defaults set)

| Item | Default if unspecified |
|------|------------------------|
| Writer model | Local/open for train; frontier optional for eval narrative |
| Judge model | Different family from writer when possible |
| Tokenizer for length buckets | Measure with both SmolLM and Gemma tokenizers; gate on max of the two |
| MC secondary view | Generate but do not score as primary |
| Answer-mask collator | Deferred; full causal LM packing for v0 mix |
| HF repo name | `ksopyla/deductive-stories` |
| License | Apache-2.0 for code; dataset card states synthetic terms |

---

## Changelog

- **2026-07-18** — Spec opened from grill session: C/B/D/A + dual-length + staged noise + v0 detective/SE + size/mix/cost/HF contracts locked.
- **2026-07-18** — v0 prototyped under `data/deductive_stories/` + `scripts/build_deductive_stories.py` (Azure OpenAI); Azure smoke passed fidelity checks but narrative quality was too low (fact dumps, not hard deduction stories).
- **2026-07-19** — **Code removed from MrCogito**; this eng spec kept as the design trace. Next implementation belongs in a **separate project/repo**; MrCogito will only ingest a published HF dataset when quality is adequate.

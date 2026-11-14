# Architecture catalogue

One NN Builder design for every architecture and experiment arm MrCogito has built or is building, with parameter
counts that match the trainer exactly. Open any of them in the builder (**Open…** lists this folder), or read them
through the MCP server (`list_designs`, `get_design_spec`, `diff_design`).

Kept by the architecture-tracker session. It checks the active branches about every 4 hours and adds or updates a
design when a spec, the arm table (`ARMS` in `scripts/run_e33n.py`), `nn/` or a trainer changes.

## How to read the numbers

- **total** = every weight of the model as the trainer builds it. Every design's name carries this number:
  `<experiment> · <plain name> · <total> (<core> core)`.
- **core** = the total without the word tables (the input embedding with its input MLP, and the output layer).
  This is the Kaplan et al. convention. The testbed is *called* "30M" after the notebook arms' 30.2M core. Their
  total is 72.7M, because the GPT-2 vocabulary puts 42.4M weights into the two word tables.
- **Exact against the trainer.** For the testbed arms: `evaluation/text_checks.count_params` on
  `scripts/run_e33n.py model_args(arm, size, ffn)`, with the FFN width the trainer really uses (`fit()`: 1472 at
  s30, 2624 at m100; not the `model_args()` default of 1536). For the champion: `evaluation/bapo_models.build_model`
  as the exam runner calls it. The LSA session checked the testbed arms against the stored weights of the trained
  checkpoints on 7 Nov: LSA port seed 42, notes_open and catalogue seed 1, all equal.
- **⚑ in the table** = the arm has a mechanism with no builder field, usually because it has no weights: a top-k
  cut, a per-query choice of what to open, pass exclusion, balanced filing, an eval switch. Each design's
  description names it under "Not a builder field". Weightless steps that cost compute are drawn as 0-parameter
  `custom` cells carrying their FLOPs. Exact reads of chosen tokens are drawn as a widened attention window.
- Tests: `tests/designs_catalogue.test.ts` pins every design's total and core count to the trainer. The other
  checks are the schema, validation, a current `derived` block, the true total in the name, and a row in this
  table. `tests/e33n_designs.test.ts` and `tests/lsa_designs.test.ts` come from the testbed branch and pin the same
  numbers for their files.

## Regenerating

```bash
cd tools/nn-builder && node scripts/designs.mjs          # rebuild every design and the table below
node scripts/designs.mjs --check                          # print counts and hashes, write nothing
npm test && npm run typecheck
```

`scripts/designs.mjs` starts from the as-trained E33n designs (`e33n_s30`, `e33n_m100`). It applies one change per
arm and builds the E37 and champion designs in code. Then it writes each file's name, description and `derived`
block, and this table between the markers. Edit the table through the script, not by hand.

## Designs

<!-- designs:start -->
| design | experiment · plain name | new against its parent | status | source (branch @ commit) · spec | total | core | date |
|---|---|---|---|---|---:|---:|---|
| `e33p_window_s30` | **E33p** · window-only floor (`window`) | the reading layer reads only its last 512 tokens; the page writer is still built (its weights are counted) but its notes never reach the read (message_override none) | done | e33n-lsa-testbed @ `8701b17` · E33p_note_indexed_exact_read.md | 72,687,252 | 30,246,164 | 2026-11-06 |
| `e33p_dense_s30` | **E33p** · dense ceiling (`dense`) | no page writer and no notebook; the reading layer reads every earlier token (full causal attention) | done | e33n-lsa-testbed @ `8701b17` · E33p_note_indexed_exact_read.md | 68,701,396 | 26,260,308 | 2026-11-06 |
| `e33n_full_s30` | **E33n** · full note read (`e33`) | every query reads every note of every closed page (no top-k cut) beside its last 512 tokens | failed | e33n-lsa-testbed @ `8701b17` · E33n_lsa_testbed_topk_notes.md | 72,687,252 | 30,246,164 | 2026-11-05 |
| `e33n_s30` | **E33n** · top-64 note read (`e33n`) | E33's notebook on the LSA testbed (GPT-2 vocabulary, natural long text); each query and head reads only its 64 best notes ⚑ | failed | e33n-lsa-testbed @ `27577bc` · E33n_lsa_testbed_topk_notes.md | 72,687,252 | 30,246,164 | 2026-11-05 |
| `e33p_notes_open_30m` | **E33p** · notes open their tokens (`notes_open`) | tied note keys; each query opens the raw tokens of its 64 best notes (≤ 512 tokens) and still reads every note as gist ⚑ | done | e33n-lsa-testbed @ `7f64a09` · E33p_note_indexed_exact_read.md | 72,687,252 | 30,246,164 | 2026-11-06 |
| `e33p_pages_open_s30` | **E33p** · notes open whole pages (`pages_open`) | the same index, coarser: a page's score is the sum of its notes' scores and each query opens its 2 best pages whole (≤ 512 tokens) ⚑ | done | e33n-lsa-testbed @ `7f64a09` · E33p_note_indexed_exact_read.md | 72,687,252 | 30,246,164 | 2026-11-06 |
| `e33p_lsa_port_30m` | **E33p** · LSA port (`lsa`) | LSA's released read ported to the dense trunk: buckets of ~8 similar keys per closed 512-key block, a query opens the best buckets (512 keys) + 4 sinks + its last 512 tokens ⚑ | done | e33n-lsa-testbed @ `7f64a09` · E33p_note_indexed_exact_read.md | 68,701,396 | 26,260,308 | 2026-11-06 |
| `e33p_lsa_s30` | **E33p** · LSA port (alias) (`lsa`) | the same arm as e33p_lsa_port_30m, drawn first by the 10M-context session; kept so that branch merges cleanly ⚑ | done | arch-10m @ `3cae268` · E33p_note_indexed_exact_read.md | 68,701,396 | 26,260,308 | 2026-11-07 |
| `e33t_catalogue_30m` | **E33t** · catalogue read (`catalogue`) | pages without overlap; each token filed under one note (≤ 16 per note); the notes are only an index (their values are not read); up to 128 notes' tokens (≤ 512) open per query; a new place each loop pass; 4 learned empty slots; an index loss ⚑ | done | e33n-lsa-testbed @ `47339bf` · E33t_catalogue_read.md | 72,687,765 | 30,246,677 | 2026-11-07 |
| `e33t_catalogue_2l_s30` | **E33t** · two-level catalogue (`catalogue_2l`) | the catalogue searched in two levels: a query keeps its 16 best pages first, then scores only their 512 notes ⚑ | done | e33n-lsa-testbed @ `47339bf` · E33t_catalogue_read.md | 72,687,765 | 30,246,677 | 2026-11-07 |
| `e33t_catalogue_reuse_s30` | **E33t** · catalogue, re-opening allowed (`catalogue_reuse`) | the pass exclusion off in training (lm_open_new_per_pass False): a later loop pass may re-open notes an earlier pass opened ⚑ | done | e33n-lsa-testbed @ `26d18f4` · E33t_catalogue_read.md | 72,687,765 | 30,246,677 | 2026-11-07 |
| `e33t_catalogue_reuse_n4_s30` | **E33t** · catalogue, 4 tokens per note (`catalogue_reuse_n4`) | 64 notes per 256-token page (4 tokens per note on average) instead of 32; the read's budget stays 512 tokens (up to 128 notes) ⚑ | queued | e33n-lsa-testbed @ `34eff60` · E33t_catalogue_read.md | 72,769,685 | 30,328,597 | 2026-11-08 |
| `e33t_catalogue_reuse_n16_s30` | **E33t** · catalogue, 16 tokens per note (`catalogue_reuse_n16`) | 16 notes per 256-token page (16 tokens per note, exactly the filing cap) instead of 32 ⚑ | queued | e33n-lsa-testbed @ `34eff60` · E33t_catalogue_read.md | 72,646,805 | 30,205,717 | 2026-11-08 |
| `e33t_catalogue_reuse_p512_s30` | **E33t** · catalogue, 512-token pages (`catalogue_reuse_p512`) | 512-token pages with 64 notes (still 8 tokens per note); the writer reads twice as long a page ⚑ | queued | e33n-lsa-testbed @ `34eff60` · E33t_catalogue_read.md | 72,900,757 | 30,459,669 | 2026-11-08 |
| `e37a_s30` | **E37** · concept memory (`e37a`) | each closed 512-key block of the reading layer's own keys is grouped by content into 64 balanced concepts of 8 keys; a query reads exactly the tokens of its strongest concepts (512) and one summary entry for every other concept ⚑ | done | arch-10m @ `95ee75d` · E37_concept_memory.md | 68,701,396 | 26,260,308 | 2026-11-07 |
| `e37a_c32_s30` | **E37** · concept memory, 32 keys per concept (`e37a_c32`) | compression ladder: 16 concepts of 32 keys per block (same 512 opened tokens; only the summaries get coarser) ⚑ | training | arch-10m @ `6e49738` · E37_concept_memory.md | 68,701,396 | 26,260,308 | 2026-11-07 |
| `e37a_c128_s30` | **E37** · concept memory, 128 keys per concept (`e37a_c128`) | compression ladder: 4 concepts of 128 keys per block (same 512 opened tokens) ⚑ | done | arch-10m @ `6e49738` · E37_concept_memory.md | 68,701,396 | 26,260,308 | 2026-11-07 |
| `e37b_s30` | **E37** · concept memory + reasoner (`e37b`) | a 2-layer, width-512 reasoner over the concepts, looped twice; its update is added to each concept's mean value and to its member tokens' values ⚑ | done | arch-10m @ `95ee75d` · E37_concept_memory.md | 75,323,092 | 32,882,004 | 2026-11-07 |
| `e33n_m100` | **E33n** · top-64 note read (`e33n`) | the 30M arm at 100M (width 896, 3 pre layers, 7 stack layers, FFN 2624) ⚑ | parked | e33n-lsa-testbed @ `27577bc` · E33n_lsa_testbed_topk_notes.md | 166,375,003 | 100,554,203 | 2026-11-05 |
| `e37_window_m100` | **E37** · window-only floor (`window`) | the 30M floor at 100M | queued | arch-10m @ `8701b17` · E37_concept_memory.md | 166,375,003 | 100,554,203 | 2026-11-07 |
| `e37_dense_m100` | **E37** · dense ceiling (`dense`) | the 30M ceiling at 100M | queued | arch-10m @ `8701b17` · E37_concept_memory.md | 162,356,379 | 96,535,579 | 2026-11-07 |
| `e37_lsa_m100` | **E37** · LSA port (`lsa`) | the LSA port at 100M ⚑ | queued | arch-10m @ `7f64a09` · E37_concept_memory.md | 162,356,379 | 96,535,579 | 2026-11-07 |
| `e37_notes_open_m100` | **E37** · notes open their tokens (`notes_open`) | E33p's notes_open at 100M (the best E33-family read on natural text; the catalogue may replace it) ⚑ | queued | arch-10m @ `7f64a09` · E37_concept_memory.md | 166,375,003 | 100,554,203 | 2026-11-07 |
| `e37a_m100` | **E37** · concept memory (`e37a`) | e37a at 100M ⚑ | queued | arch-10m @ `95ee75d` · E37_concept_memory.md | 162,356,379 | 96,535,579 | 2026-11-07 |
| `e37b_m100` | **E37** · concept memory + reasoner (`e37b`) | e37b at 100M (the reasoner's input map is 896 → 512) ⚑ | queued | arch-10m @ `95ee75d` · E37_concept_memory.md | 169,141,915 | 103,321,115 | 2026-11-07 |
| `e31_li_m1_cap30m` | **E31** · latent-memory champion (capability checks) (`e31_li_m1`) | the capability-check champion: E31's sliding-window latent memory on the DNA tasks — width 960, 4 layers (pre-encoder, global read, 2 stack layers), the E31 page writer (256-token pages every 192, 32 notes per page, one key and value per note), reads after QUERY go to the notes only ⚑ | done | dev @ `9253d7a` · E31_sliding_window_latent_memory.md | 35,129,900 | 33,947,436 | 2026-09-27 |
| `lsa_paper_native_0p8b` | **LSA** · LSA as published (Qwen3.5-0.8B) | the outside reference: Gated DeltaNet hybrid, 24 layers of width 1024, the LSA read on every fourth layer ⚑ | done | e33n-lsa-testbed @ `1769cf9` · 1_Strategy_and_Plans (architectures review); arXiv 2610.42147 | 752,393,024 | 498,113,344 | 2026-10-30 |
| `e33m_plain_30m` | **E33** · notebook (text checks) | E33's notebook on the text checks (4,096-token vocabulary): the E31c writer plus a 4-round loop over the read layer and the next, the order term and count heads | done | dev @ `edb1c00` · E33_iterative_concept_reads.md | 29,946,402 | 25,911,266 | 2026-10-24 |
<!-- designs:end -->

## Built-in presets (no file; open them from the preset menu)

Exact against the trainer (`tests/params.test.ts`), counted on 24 Oct 2026. These are the models of the text
checks, with a 4,096-token vocabulary.

| preset | what | total | status |
|---|---|---:|---|
| `dense_plain_30m` / `_50m` / `_100m` | dense text model, the text checks' ceiling: width 576 / 576 / 896, 8 / 12 / 12 layers | 30,293,140 / 50,057,374 / 100,966,686 | done (text checks) |
| `e33m_plain_30m` / `_100m` | E33 notebook on the text checks (saved copy: `e33m_plain_30m.json`) | 29,946,402 / 100,965,425 | done |
| `e31c_plain_30m` / `_100m` | E31c: the E31 writer on text, one closed-window read, no loop | 29,855,316 / 100,823,774 (checked against `text_checks` `e31c_plain` on 8 Nov) | done |
| `e36_kv` | E36: a trained writer + key/value expander for a frozen SmolLM2-360M reader | 412,570,560 (22,437,888 trained) | done (S0 + S1, 18 Oct) |

E33's text-check variants E33d (note–page distance), E33e (note rank), E33f (notes on the wire) and E35 (latent
messages) are knobs on the E33 preset. They are not drawn separately.

## Older families in the master log (not drawn)

These families ran for real, but they are built on a different foundation (an encoder–decoder, a Gemma backbone,
or the older capability-probe arches). The first pass did not draw them; any one can be added on request.

| family | what (master log) | status |
|---|---|---|
| E01–E05b | concept encoder + AR / parallel decoder (concept_ar) | E02–E04, E05b done; E01, E05 failed |
| E10–E10e, E14, E15 | Gemma-3-1B backbone with a concept memory | failed |
| E16, E16a, E16b, E17–E17e | shared-depth recurrent concepts / per-layer banks on Gemma | E16b, E17 done; the rest failed or mixed |
| E18, E18b, E22 | Perceiver AR v2 (one global read) and the Perceiver concept LM | failed |
| E21, E24–E29 | exclusive compressed reads on the BAPO DNA ladder (capability probe arches e18 / e21) | E25 done (mapped); E26–E29 failed |
| E30 | sliding-window Perceiver exclusive write (`e30_li`) | parked 2026-10-01 |
| E33a | `e33a_loop` = the champion + a 4-round read–think–reread loop (capability checks) | failed (inconclusive) |

## Change log

- **2026-10-24 · dev `edb1c00`:** `e33m_plain_30m` saved, the builder's first design (29,946,402).
- **2026-11-05 · e33n-lsa-testbed `27577bc`:** `e33n_s30` and `e33n_m100`, the testbed as trained (72,687,252 /
  166,375,003). The top-k cut is not a field.
- **2026-11-07 · arch-10m `3cae268`:** `e33p_dense_s30`, `e33p_lsa_s30`, `e37a_s30`, `e37a_c32_s30`,
  `e37a_c128_s30`, `e37b_s30`. Two defects:
  - the `derived` blocks were copied from `e33n_s30`, so they claimed 72,687,252 and its hash for every file;
  - E37's weightless summaries were drawn as a frozen 64 → 64 notes cell, which adds 8,192 to every E37 total.
- **2026-11-08 · e33n-lsa-testbed `1769cf9`:** `e33p_lsa_port_30m`, `e33p_notes_open_30m`, `e33t_catalogue_30m`
  and `lsa_paper_native_0p8b`. These were the first names to carry the true total.
- **2026-11-08 · architecture tracker (this branch):**
  - Every design is now regenerated by `scripts/designs.mjs`, named `<ID> · <plain> · <total> (<core> core)`, and
    has a current `derived` block. A test pins its count to the trainer.
  - E37: the summaries are now a notes cell with **no projections**, a small engine addition: `kv_out.projections:
    0`, the slot states are the keys/values. E37 totals now equal the trainer's: 68,701,396 (e37a and the ladder)
    and 75,323,092 (e37b).
  - Arms drawn for the first time: the window floor, E33n's full note read, pages_open, the two-level catalogue,
    the catalogue with re-opening allowed, the six 100M arms of E37 stage A, and the E31 champion.
  - `e33p_lsa_s30` is now an exact alias of `e33p_lsa_port_30m`, which has the richer drawing with the bucket
    index's FLOPs. The catalogue's read label now names the new-place-per-pass rule.
- **2026-11-08 · arch-10m `e9b37d2`:** E37's length scale of the summaries (`concept_sum_len`) is an eval switch
  with no weights, off in training. It is noted in the E37 descriptions; no design change.
- **2026-11-08 · arch-10m `7c7158c`:** the concept arms (e37a, the ladder, e37b) now train with the length scale
  on, set to the decay's row length (16,384), where it is inert. Recounted every arm at 30M and 100M on this
  commit with the trainer: totals, core counts and every weight shape unchanged. Text only: the E37 descriptions
  say so, and the six 100M arms are now `queued` (stage A has the author's go-ahead, one 4-GPU AWS spot node, in
  the order e37a, lsa, dense, window, notes_open, e37b). `e37b_m100` moves from `parked` to `queued`: the 30M
  screen shelved the reasoner, but the launch list still includes it, last.
- **2026-11-08 · e33n-lsa-testbed `34eff60` (read at `0227838`):** E33t stage B, three new designs around the
  stage-A winner (the catalogue with re-opening): `e33t_catalogue_reuse_n4_s30` (64 notes per 256-token page,
  72,769,685 / 30,328,597 core), `_n16_s30` (16 notes, 72,646,805 / 30,205,717) and `_p512_s30` (512-token pages,
  64 notes, 72,900,757 / 30,459,669). Counted with the trainer on `0227838`; the builder matches all three exactly
  (only the writer's learned slots and its position prior change). `ARM.catalogue` gains `notes` and `page`; the
  three stage-A catalogue designs are unchanged (same hashes). Stage A's scores (`a372da3`) go into their notes:
  catalogue 71 % of the gap at 16k, two-level 68 %, re-opening 91 % (now `done`, the winner). Stage C (`0227838`)
  is a proposal awaiting approval with no new arm yet. arch-10m `3833ca1`/`07ca3a5` change only eval scripts.

## Branches last seen

The next check reads only commits after these.

| branch | last commit seen | its time | read at |
|---|---|---|---|
| `origin/dev` | `644b770` (this catalogue) | 2026-11-08 16:28 +01:00 | 2026-11-08 20:26 +01:00 |
| `origin/e33n-lsa-testbed` | `0227838` | 2026-11-08 19:13 +01:00 | 2026-11-08 20:26 +01:00 |
| `origin/arch-10m` | `07ca3a5` | 2026-11-08 18:32 +01:00 | 2026-11-08 20:26 +01:00 |
| `origin/e33-multihop` | `070c56f` (fully merged into dev) | 2026-10-22 20:52 +02:00 | 2026-11-08 20:26 +01:00 |
| `origin/process/capability-eval-tracking` | `7422a11` (fully merged into dev) | 2026-10-19 10:04 +02:00 | 2026-11-08 20:26 +01:00 |

The check:

```bash
git fetch origin
git log --format='%h %ci %s' <seen>..origin/<branch> -- docs/experiments_specs scripts/run_e33n.py nn training
```

## Merging other branches into dev

`e33n-lsa-testbed` and `arch-10m` added some of these files themselves, so a merge into dev after this branch
gives add/add conflicts in `tools/nn-builder/designs/`. Resolve them with dev's version, then run
`node scripts/designs.mjs` and `npm test`.

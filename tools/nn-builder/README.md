# NN Builder

Compose a neural network from layer slots and see what it costs while you build it: parameters, forward and
backward compute, GPU memory, how many GPUs a run needs and how long it takes, and how all of that changes with
the row length. Save the design as a JSON file that a later step turns into PyTorch code.

Self-contained (its own `package.json` and dependencies) so it can move to its own repository.

## Run

```bash
cd tools/nn-builder && npm start
```

The first run installs the dependencies; then the app opens at http://localhost:5178. Needs Node ≥ 20.19.

```bash
npm test          # engine tests: exact parameter counts, FLOPs, memory, schema, edits
npm run typecheck
```

Browser end-to-end test (39 checks: presets, drag and drop, wires, loops, undo / redo, inspector, estimate tabs,
save / open / export). With the app running, open it and paste into the DevTools console:

```js
const { run } = await import("/tests/e2e/browser-e2e.ts"); console.table((await run()).steps);
```

It saves `designs/e2e_test.json`; delete that file afterwards.

## Use

- **Frame of slots.** Each stack is a column; each row is a layer; each row holds sublayer slots that run left to
  right. Drag a layer from the palette into a **+** slot (same row) or between rows (a new row). Drag a slot or a
  row handle **⋮⋮** to move it; drop it on the palette to remove it. Click a palette item to add it to the
  selected row.
- **Wires.** Drag from a dot: to a later row = a skip (U-net) wire; to a cross-attention or notebook read = its
  keys/values; onto another stack's header = that stack's input. The embedding's lower-right dot is its token table.
- **Loops.** Shift-click row handles to select neighbouring rows, then **Loop** (shared weights, R rounds) or
  **Repeat** (R independent copies) in the inspector. The −/+ on a loop's tag changes R.
- **Stacks.** `+ Stack` adds a decoder, an encoder, a page writer (pages over a stack), a slots-per-page notes
  stack, or a fixed latent set. The stack inspector sets its width and sequence.
- **Everything is configurable**: click any cell, row, stack, wire or loop.
- **Estimates**: the strip at the top is always live; the side tabs hold compute & memory (with a per-cell
  breakdown), GPUs & time (presets for Odra, Polonez and an AWS p5), and cost against row length (1k → 1M).
- **Save** writes `designs/<name>.json` (this folder); **Open…** lists them; **Export / Import** use files.
  Undo/redo: ⌘Z / ⇧⌘Z. Delete removes the selection. The current design is also kept in the browser.

Presets: MrCogito dense 30M / 50M / 100M and E33 notebook 30M / 100M (the trainer's exact models), GPT-2 small,
a T5-small-shaped encoder–decoder, and a blank one-layer model.

## How the numbers are made

- **Parameters** are exact formulas per cell (`src/engine/cells.ts`). Tests check them against the MrCogito
  trainer's own counter (`evaluation/text_checks.count_params`, run on 24 Oct 2026): dense 30,293,140 /
  50,057,374 / 100,966,686 and E33 29,946,402 / 100,965,425; and GPT-2 small = 124,439,808.
- **Compute** counts the multiply-adds (2 FLOP each) of every projection, attention score and value read,
  averaged over the row (causal masks, sliding windows, pages, notes that become visible page by page). Backward
  = 2 × forward for trained weights, 0 for no-grad parts; checkpointing adds one forward. Norms and activations
  are not counted. Agrees with the single-page E33 builder to within 1 %.
- **Memory** per GPU = weights + gradients + optimizer state (by number format, optimizer and sharding) + the
  activations each cell saves for backward (counted tensor by tensor; loops keep every round) + a loss-chunk
  workspace + CUDA context, with a fitted overhead factor. Fitted so that, at 4k rows on a 24 GB RTX 3090, dense
  30M fits 8 rows, E33 30M does not fit 8 but fits 4, and dense 100M fits 4 — what the real runs did.
- **Speed** is calibrated on measured RTX 3090 throughput: dense 30M ≈ 60k tokens/s and E33 30M ≈ 20k (dense 50M,
  41.5k measured, is predicted within 15 % without being fitted). Plain matmul work runs at ~24 % of the card's
  nominal peak; masked, gathered and looped work (notebook reads, page writers, the loop's feedback) at about a
  sixth of that speed per FLOP. Other GPUs scale by their peak with an editable efficiency (the A100/H100 value is
  a guess for a ~500M model). Treat new numbers as ±30 %. `node scripts/calibrate.mjs` prints the fit.

## Files

| path | what |
|---|---|
| `src/engine/` | the cost engine (no UI): `types.ts` (the graph format), `cells.ts` (per-cell costs), `analyze.ts` (whole graph, memory, GPU plan), `edit.ts` (pure edits), `presets.ts`, `serialize.ts`, `hardware.ts` (GPUs, number formats, calibration) |
| `src/ui/` | React interface |
| `server/designs-api.ts` | dev-server endpoint that reads and writes `designs/*.json` |
| `schema/graph.schema.json`, `docs/schema.md` | the graph file format, machine-readable and in words |
| `designs/` | saved designs (an E33 30M example is there) |
| `tests/` | vitest suites |

## From a design to PyTorch

Save the design, then ask Claude to generate PyTorch code from `tools/nn-builder/designs/<name>.json`.
[`docs/schema.md`](docs/schema.md) gives the forward pass of every cell, wire and loop; the file's `derived` block
holds the parameter count of every cell, so the generated model can be checked against it.

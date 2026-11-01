# NN Builder

Compose a neural network from layer slots and see what it costs while you build it: parameters, forward and
backward compute, GPU memory, how many GPUs a run needs and how long it takes, and how all of that changes with
the row length. Save the design as a JSON file; coding agents (Claude Code, Cursor, Codex) read it through a local
MCP server and implement it.

Self-contained (its own `package.json` and dependencies) so it can move to its own repository.

## Run

```bash
cd tools/nn-builder && npm start
```

The first run installs the dependencies; then the app opens at http://localhost:5178. Needs Node ≥ 20.19.

```bash
npm test          # engine tests: exact parameter counts, FLOPs, memory, shapes, tokens, schema, edits, spec, MCP server
npm run typecheck
npm run mcp       # the MCP server on stdio (agents start it themselves; see docs/agent-spec.md)
```

Browser end-to-end test (66 checks: presets, drag and drop, wires, loops, undo / redo, inspector, estimate tabs,
GPU and token presets, resizing, wire details, tabs and comparison, save / open / export, the unsaved-changes
mark). With the app running, open it and paste into the DevTools console:

```js
const { run } = await import("/tests/e2e/browser-e2e.ts"); console.table((await run()).steps);
```

It presses where a mouse would land (a covered or clipped handle fails), reaches far targets by holding at the
frame's edge, and passes at 1440 × 900 and 1920 × 1080. It saves `designs/e2e_test.json`; delete that file
(and `designs/.history/e2e_test/`) afterwards.

## Use

- **Stacks.** Add a main block with **+ Stack** in the top bar: a decoder, an encoder, a page writer (it starts from
  the decoder's token table by itself), notes (slots per page; they read a page writer's pages, and attach to one
  added later) or fixed latents. For a notebook model from "Blank: one layer": + Stack → Pages, + Stack → Notes, drag
  the Pages output dot onto the Notes cross-attention, drop a Notebook read between two decoder rows, then drag the
  Notes output dot onto it (the read takes on the notes' key/value shape).
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

Presets (each opens in its own tab):
- MrCogito, exact against the trainer: dense 30M / 50M / 100M, E33 notebook 30M / 100M, E31c latent notebook
  30M / 100M, and E36 (a 22.4M trained writer + key/value expander for a frozen SmolLM2-360M reader).
- Outside references, exact against their Hugging Face configs: GPT-2 small (124,439,808), T5-small (60,506,624),
  SmolLM3-3B (3,075,098,624), DeepSeek-V2-Lite (15,706,484,224; MLA + shared-expert MoE, 2.45B used per token).
- DeepSeek-V4.1-Flash, **approximate**: its text backbone (≈552B + a 196B Engram memory, 16B used per token); the
  compressed sparse attention, hyper-connections and drafting heads are approximated or left out.
- A blank one-layer model.

**Compare.** Every open design is a tab (switch with a click, × to close, **+ copy** to try a variant). The
**Compare** tab lines them up — parameters (total, trained, used per token), compute, memory, speed, tokens and time
— under one training setup (the current design's) or each its own.

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
| `server/designs-api.ts` | dev-server endpoint that reads and writes `designs/*.json` (each save also keeps `designs/.history/<name>/<hash>.json`) |
| `schema/graph.schema.json`, `docs/schema.md` | the graph file format, machine-readable and in words |
| `src/engine/spec.ts`, `schema/spec.schema.json`, `docs/agent-spec.md` | the spec agents read (generated from a design), its hash, the diff between versions, validation; the agent guide |
| `mcp/server.ts`, `scripts/mcp.mjs` | the read-only MCP server for agents (`npm run mcp`; built into `mcp/dist/`) |
| `designs/` | saved designs (an E33 30M example is there) |
| `tests/` | vitest suites |

## From a design to PyTorch

Save the design (a design with unsaved changes has an amber dot on its tab and on Save). Then ask a coding agent
to implement it. The agent reads it through the `nn-builder` MCP server: `list_designs`, `get_design_spec` (every
block with resolved settings and shapes, the execution order with loops, wires, exact parameter counts,
validation, and a hash of the architecture), `validate_design`, and `diff_design` (what changed since the hash
it implemented). Setup for Claude Code, Cursor and Codex, and how to read a spec: [`docs/agent-spec.md`](docs/agent-spec.md).
The design file itself is documented in [`docs/schema.md`](docs/schema.md).

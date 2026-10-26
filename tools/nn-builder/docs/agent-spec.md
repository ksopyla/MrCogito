# Reading a design: the agent guide

NN Builder is for people: they draw a network and see its shapes and costs. Coding agents (Claude Code, Cursor,
Codex, …) implement it. This page is for those agents: how to read a design, what each field means, and how to
follow the design as it changes.

There are two files per design:

- **the design** — `designs/<name>.json` (format `nn-builder/graph`, [schema.md](schema.md)): what the person drew.
  The app writes it on **Save**.
- **the spec** — format `nn-builder/spec`, generated from the design and never edited: every block with its
  settings resolved, input and output shapes, what it does and its tensor steps, the execution order with loops
  spelled out, what each wire carries, exact parameter counts, validation, and a hash of the architecture.
  Machine-readable schema: [`schema/spec.schema.json`](../schema/spec.schema.json).

Agents read the spec through a small local MCP server. It reads the designs folder directly, so the app need not
be running. **An agent sees the last saved version**: the app marks a design with unsaved changes with an amber
dot (on its tab and on Save).

## Setup

Once, in `tools/nn-builder`: `npm install`. The server builds itself on first start (and again after the engine
changes); it prints nothing on stdout but the protocol.

Claude Code (from the repository root, or put the same entry in the project's `.mcp.json`):

```bash
claude mcp add nn-builder -- node /absolute/path/to/tools/nn-builder/scripts/mcp.mjs
```

Cursor — `.cursor/mcp.json` (project) or `~/.cursor/mcp.json`:

```json
{ "mcpServers": { "nn-builder": { "command": "node", "args": ["/absolute/path/to/tools/nn-builder/scripts/mcp.mjs"] } } }
```

Codex — `~/.codex/config.toml`:

```toml
[mcp_servers.nn-builder]
command = "node"
args = ["/absolute/path/to/tools/nn-builder/scripts/mcp.mjs"]
```

Another designs folder: add `--designs <folder>` to the arguments (or set `NN_BUILDER_DESIGNS`).

## What the server offers

All read-only.

| tool | arguments | returns |
|---|---|---|
| `list_designs` | — | each saved design: name, title, last save, hash, parameters, stack names, error and warning counts |
| `get_design_spec` | `name` | the spec (below), plus `saved` (time) and `file` |
| `validate_design` | `name` | `ok`, `errors`, `warnings`, `hash` |
| `diff_design` | `name`, `since_hash` | what changed from that version to the saved one, and the parameter count before and after |

Resources: `nnb://designs/{name}/spec` (the same spec) and `nnb://guide` (this page). Prompt: `implement_design`
(`name`, optional `since_hash`) — the workflow below as a ready instruction.

`name` is the file name without `.json`, as `list_designs` shows it.

## Workflow

1. **Implement.** `get_design_spec`. Read `how_to_read` and `block_types`. If `validation.errors` is not empty,
   stop and report them: the design cannot be built as drawn. Warnings are worth reading (an unused stack, a
   learned position table shorter than the row, …).
2. **Build** one module per stack and one submodule per block, run in `execution_order`. Fill in what the spec
   leaves open (initialisation, kernels, file layout, naming) the way the surrounding code does; the design is a
   picture of the architecture, not a line-by-line program.
3. **Check** against the spec: trainable parameters per block (`blocks[].parameters.count`) and in total
   (`parameters.trained`), and a forward pass on random token ids whose output shapes match `outputs`.
4. **Keep `design.hash`** next to the model (a comment or a config field).
5. **When the design changes** (the person edits and saves it): `diff_design` with the hash you kept. It lists what
   was added, removed, moved or changed — stacks, rows, blocks, wires, loops — field by field, so you change only
   that. Then `get_design_spec` again for the new hash.

Every version an agent reads, and every save from the app, is kept in `designs/.history/<name>/<hash>.json`; that
is what `diff_design` compares against. A hash with no copy there gets an error listing the known versions.

## The spec, field by field

| field | meaning |
|---|---|
| `format`, `version` | `"nn-builder/spec"`, `1` |
| `design` | `name`, `description` (the person's notes; often say what the model is for), `hash` |
| `how_to_read` | the conventions below, in short — read them first |
| `block_types` | what each block type computes, with the meaning of its settings |
| `dims` | named sizes used in shapes: `B` (batch, any size), `S` (tokens in a row), `S_src` (encoder source), `N` (latents), `pages`, `W` (tokens per page), `slots`, `notes` — with their values for this design |
| `inputs`, `outputs` | the model's tensors: e.g. `main.input_ids` `[B, S]` int64 → `main.logits` `[B, S, vocab]` |
| `stacks[]` | one per stream: `width`, `sequence` (its shape and kind), `starts_from`, `rows`, `blocks`, `output` (shape and where it goes) |
| `stacks[].rows[]` | `blocks` in order, and `before`: what is added to the row's input before it runs (`x0_mix`, `skips_in`) |
| `stacks[].blocks[]` | `id`, `type`, `title`, `what_it_does`, `settings` (all resolved), `residual`, `input` / `output` (tensor name and shape), `reads` (the wired tensor a cross-attention or notebook read uses), `runs_per_row` (> 1 inside a loop), `frozen`, `parameters` (count and parts), `tensor_steps` |
| `wires[]` | `kind` (`skip`, `source` = keys/values for an attention, `input` = starts another stack), `from`, `to`, `carries` (shape), `meaning` |
| `execution_order[]` | steps: `row`, `loop` (rows run `count` times with shared weights; `round_embedding`, `feedback`) or `repeat` (`count` copies) |
| `parameters` | `total`, `trained` (frozen excluded), `used_per_token` (unused experts and lookup tables excluded), `by_stack` |
| `training` | row length, rows per step, token budget, number format, optimizer, weight decay — the person's intended setup |
| `validation` | `errors` (must be empty to build) and `warnings`, each `{where, message}` |

Shapes are lists: a string is a named dim from `dims`, a number is a fixed size. Tensor names: `<stack>.input_ids`,
`<stack>.x` (the stream), `<row>.output`, `<block>.output`, `<block>.table` (an embedding's token table).

### Conventions

- **Residual.** `pre`: `x = x + f(norm(x))`. `post`: `x = norm(x + f(x))`. `none`: `x = f(norm(x))`.
  `settings.norm` names the norm.
- **Before a row.** `x0_mix`: `x = α·x + β·x0`, two learned scalars (α starts at 1, β at 0); `x0` is the stream
  after the stack's first row. Each skip adds `w · output(from_row)`; `w` is one learned scalar starting at 0
  (`scalar`), one per channel (`vector`), or 1 (`none`).
- **Loops.** A `loop` step runs its rows `count` times with the same weights. `round_embedding`: one learned
  vector per round, added at the start of the round. `feedback: "soft"`: between rounds, the state goes through
  the stack's final norm and LM head, the softmax is re-embedded with the token table and input projection, and
  added to the next round's input through a per-channel gate that starts at 0. A `repeat` step is `count`
  independent copies, one after another.
- **Reads.** A cross-attention projects the wired tensor into its keys and values. A notebook read uses the wired
  notes directly as extra keys and values beside its own recent-token window; a note becomes readable once the page
  it summarises is complete.
- **Order.** `execution_order` runs everything a row reads before that row: in the E33 example the page writer and
  the notes run between the decoder's L1 and its looped read layers.
- **Hash.** `design.hash` covers the architecture: stacks, rows, blocks, wires and loops, with their names and
  labels. The design's own name and description, the layout and the training setup are not in it, so widening a
  stack on screen or changing the token budget does not change it.

### An example block

The E33 notebook read (`main.L2.attn`, trimmed):

```json
{
  "id": "main.L2.attn", "type": "attention", "title": "Notebook read",
  "what_it_does": "notebook read: the last 256 tokens plus the notes from Notes, each note readable once its page is complete",
  "settings": { "mode": "self", "heads": 9, "head_dim": 64, "kv_heads": 1, "causal": true, "window": 256,
                "qk_norm": true, "norm": "rmsnorm", "memory": true, "dual_key": true, "...": "..." },
  "residual": "pre",
  "input":  { "tensor": "main.x", "shape": ["B", "S", 576] },
  "output": { "tensor": "main.x", "shape": ["B", "S", 576] },
  "reads": { "wire": "wire.notes", "from": "notes.output", "as": "notes", "shape": ["B", "notes", 1, 64] },
  "runs_per_row": 4,
  "parameters": { "count": 826190, "parts": [{ "name": "query", "count": 331776 }, "..."] },
  "tensor_steps": [
    { "step": "queries q", "shape": ["B", "S", 9, 64] },
    { "step": "notes: ready keys / values", "shape": ["B", "notes", 1, 64] },
    { "step": "scores q·k", "shape": ["B", 9, "S", 928], "note": "a band of 256 recent keys per query; computed in tiles" },
    "..."
  ]
}
```

and the step that runs it:

```json
{ "kind": "loop", "stack": "main", "rows": ["main.L2", "main.L3"], "count": 4,
  "shared_weights": true, "round_embedding": true, "feedback": "soft" }
```

## A diff

After the person widened one FFN and saved:

```json
{
  "from_hash": "nnb1-1bd9d7ec43a7620523251415c770", "to_hash": "nnb1-049098962cb62410bff287edff66",
  "changes": [{ "change": "changed", "what": "block", "id": "main.L0.ffn", "where": "Decoder · L0",
                "field": "hidden", "from": 1152, "to": 1600 }],
  "parameters": { "from": 29946402, "to": 30720546 },
  "summary": "1 change: 0 added, 0 removed, 1 settings changed, 0 moved; parameters 29,946,402 → 30,720,546"
}
```

## Without MCP

The same functions are in `src/engine/spec.ts` (`buildSpec`, `validateDesign`, `designHash`, `diffDesigns`); the
tests in `tests/spec.test.ts` and `tests/mcp.test.ts` show them in use. The design file itself is documented in
[schema.md](schema.md).

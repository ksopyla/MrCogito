// The spec an agent reads to implement a design: everything spelled out (blocks with resolved settings, tensor shapes,
// what each block does, execution order with loops and repeats as steps, wires with what they carry, exact parameter
// counts, validation), a content hash of the architecture, and the diff between two versions. Read-only: generated
// from the design file, never edited. Documented in docs/agent-spec.md; schema in schema/spec.schema.json.
import { cellTitle, mixing, stackIO } from "../ui/describe";
import { analyze, type Analysis, type Issue } from "./analyze";
import { cellTensorSteps, edgeTensors, positions, rowInputShape, stackOutputShape, type Dim } from "./shapes";
import type { Cell, Edge, Endpoint, Graph, Group, Stack } from "./types";

export const SPEC_FORMAT = "nn-builder/spec";
export const SPEC_VERSION = 1;

export type ShapeEntry = string | number;
export interface SpecTensor {
  tensor: string;
  shape: ShapeEntry[];
}
export interface SpecBlock {
  id: string;
  type: Cell["type"];
  title: string;
  label?: string;
  what_it_does: string;
  settings: Record<string, unknown>;
  residual?: "pre" | "post" | "none";
  input: SpecTensor;
  output: SpecTensor;
  reads?: { wire: string; from: string; as: "keys/values" | "notes"; shape: ShapeEntry[] };
  runs_per_row: number;
  frozen: boolean;
  parameters: { count: number; parts: { name: string; count: number }[] };
  tensor_steps: { step: string; shape: ShapeEntry[]; note?: string }[];
}
export interface SpecRow {
  id: string;
  label: string;
  blocks: string[];
  before: { x0_mix: boolean; skips_in: { from_row: string; weight: "scalar" | "vector" | "none" }[] };
}
export interface SpecStack {
  id: string;
  name: string;
  role: Stack["role"];
  width: number;
  frozen: boolean;
  sequence: { kind: Stack["seq"]["kind"]; shape: ShapeEntry[]; description: string };
  starts_from: string;
  rows: SpecRow[];
  blocks: SpecBlock[];
  output: SpecTensor & { goes_to: string[] };
}
export interface SpecStep {
  kind: "row" | "loop" | "repeat";
  stack: string;
  rows: string[];
  count?: number;
  shared_weights?: boolean;
  round_embedding?: boolean;
  feedback?: "none" | "soft";
  note: string;
}
export interface SpecProblem {
  where: string;
  message: string;
}
export interface Spec {
  format: typeof SPEC_FORMAT;
  version: typeof SPEC_VERSION;
  design: { name: string; description: string; hash: string };
  how_to_read: string[];
  block_types: Record<string, string>;
  dims: { name: string; value: number | null; meaning: string }[];
  inputs: { name: string; shape: ShapeEntry[]; dtype: string; meaning: string }[];
  outputs: { name: string; shape: ShapeEntry[]; meaning: string }[];
  stacks: SpecStack[];
  wires: { id: string; kind: Edge["kind"]; from: string; to: string; carries: ShapeEntry[]; meaning: string }[];
  execution_order: SpecStep[];
  parameters: { total: number; trained: number; used_per_token: number; by_stack: Record<string, number> };
  training: { row_length: number; rows_per_step: number; tokens: number; precision: string; optimizer: string; weight_decay: number };
  validation: { errors: SpecProblem[]; warnings: SpecProblem[] };
}

const HOW_TO_READ = [
  "The model is a set of stacks. Each stack carries one stream of vectors of shape [B, positions…, width]; `sequence.shape` gives it.",
  "Run the rows in `execution_order`. Inside a row, run its blocks in the listed order; each block takes the stream and returns the new stream.",
  "A block's `residual` says how its result joins the stream: 'pre' → x = x + f(norm(x)); 'post' → x = norm(x + f(x)); 'none' → x = f(norm(x)). `settings.norm` names the norm (rmsnorm, layernorm with bias, none).",
  "Before a row runs, `before` adds to its input: if x0_mix, x = α·x + β·x0 (two learned scalars, α starts at 1 and β at 0; x0 is the stack's stream after its first row); each skip adds w · (that earlier row's output), with w one learned scalar starting at 0 ('scalar'), one per channel ('vector'), or 1 ('none').",
  "A 'loop' step runs its rows `count` times with the same weights. round_embedding: a learned vector per round, added at the start of each round. feedback 'soft': between rounds, the round's state goes through the stack's final norm and LM head, the softmax is re-embedded with the stack's token table and its input projection, and added to the next round's input through a per-channel gate that starts at 0. A 'repeat' step is `count` independent copies of its rows, one after another.",
  "`reads` on a block: a cross-attention projects the wired tensor into its keys and values; a notebook read (memory) uses the wired notes directly as extra keys/values beside its own window, each note readable once the page it summarises is complete.",
  "Shapes: B is the batch (any size); other names are listed under `dims` with their sizes; numbers are fixed sizes. `tensor_steps` walks each block's forward pass.",
  "`parameters` gives exact counts per block and part, from the builder's reference formulas: count your module's trainable parameters to check the build. Frozen blocks keep pretrained weights and are not trained.",
  "`validation.errors` must be empty before building; `design.hash` identifies this version of the architecture (the design's name and description, its layout and the training setup are not part of it): keep it with your code, and ask for the diff since that hash when the design changes.",
];

const BLOCK_TYPES: Record<string, string> = {
  embedding: "token table [vocab, dim] (+ hashed n-gram tables of n-gram_buckets rows), then proj: none (stream = table), linear (dim → width) or gated_mlp (up0(e) + up2(silu(up1(e))), dim → width), then out_norm. share_with: reuse another embedding's table.",
  positional: "rope: rotary positions (base theta) applied to q and k of the attention blocks after it in the stack (unless the block sets nope); learned: a [max_len, width] table added; t5_bias: T5 relative-position bias with buckets × heads learned values; alibi / sinusoidal / none.",
  attention: "mode self (keys/values from the stream) or cross (from the wired tensor). heads × head_dim queries, kv_heads key/value heads (grouped-query when fewer). causal; window > 0 = only the last `window` keys. qk_norm: RMSNorm over head_dim on q and k. bias on q/k/v/o. norm: pre-norm. kv_norm: norm on the cross source. memory: also read wired notes. q_rank / o_rank: low-rank query (with a norm in between) / output. mla: DeepSeek multi-head latent attention (q_rank, kv_rank cached latent, rope_dim, nope_dim, v_dim). extras: E33 order term, count heads, page position prior, length-aware logit scale.",
  ffn: "swiglu: down(silu(gate(x)) · up(x)); geglu: gelu instead of silu; mlp: down(activation(up(x))). hidden = inner width. experts > 1: a mixture of experts (router width → experts, top_k per token, shared_experts always on, router_bias per expert).",
  norm: "normalise the stream: rmsnorm (weight) or layernorm (weight and bias).",
  linear: "project the stream to `out` (0 = keep the width), optional bias, then out_norm.",
  latent_init: "replace the stream with learned queries (one per slot, the same on every page; or one per latent), plus a linear map of address features, then norm.",
  kv_out: "turn each slot into ready keys/values (notes) of entries × kv_heads × head_dim, for `layers` reader layers; projections 2 (k, v) or 4 (k, k-read, v, v-state); in_norms norms on the input; key_norm RMSNorm on keys.",
  pool: "the first block of a slots stack wired from a pages stack: each page of tokens becomes one vector (mean, or attention with one learned query per head).",
  lm_head: "final_norm, then logits over vocab (tied = reuse the stack's token table; needs dim = width); loss = cross-entropy against the next token (softcap: cap·tanh(logits/cap); z_loss weight on logsumexp²); chunked = computed in chunks.",
  custom: "a block the builder only counts (params, forward FLOPs); its meaning is in `note` and the design's description. sparse = a lookup table.",
};

const SYMBOLIC = new Set(["B", "S", "S_src", "N", "pages", "W", "slots", "notes"]);
const shapeOf = (dims: Dim[]): ShapeEntry[] => dims.map((d) => (SYMBOLIC.has(d.n) ? d.n : d.v));

function endpointName(g: Graph, e: Endpoint): string {
  const s = g.stacks.find((x) => x.id === e.stack);
  if (e.port === "table" && e.cell) return `${e.cell}.table`;
  if (e.row) return `${e.row}.output`;
  if (e.cell) return `${e.cell}.output`;
  return `${s?.id ?? e.stack}.output`;
}

function settingsOf(c: Cell): Record<string, unknown> {
  const { id: _id, label: _label, type: _type, frozen: _frozen, ...rest } = c as Cell & Record<string, unknown>;
  void _id, _label, _type, _frozen;
  return JSON.parse(JSON.stringify(rest));
}

// ------------------------------------------------------------------ execution order

/** Row-level dependencies → an order that runs everything a row reads before it (stack order breaks ties). */
function executionOrder(g: Graph): SpecStep[] {
  const rowHome = new Map<string, { stack: Stack; index: number }>();
  g.stacks.forEach((s) => s.rows.forEach((r, index) => rowHome.set(r.id, { stack: s, index })));
  const cellRow = new Map<string, string>();
  for (const s of g.stacks) for (const r of s.rows) for (const c of r.cells) cellRow.set(c.id, r.id);
  const lastRow = (stackId: string) => g.stacks.find((s) => s.id === stackId)?.rows.at(-1)?.id;
  const endRow = (e: Endpoint) => e.row ?? (e.cell ? cellRow.get(e.cell) : undefined) ?? lastRow(e.stack);

  // a group runs as one unit
  const unitOf = new Map<string, string>();
  for (const grp of g.groups) for (const r of grp.rows) unitOf.set(r, grp.id);
  const unit = (row: string) => unitOf.get(row) ?? row;

  const deps = new Map<string, Set<string>>();
  const add = (row: string, dep: string | undefined) => {
    if (!dep) return;
    const a = unit(row), b = unit(dep);
    if (a === b) return;
    if (!deps.has(a)) deps.set(a, new Set());
    deps.get(a)!.add(b);
  };
  for (const s of g.stacks)
    s.rows.forEach((r, i) => {
      deps.set(unit(r.id), deps.get(unit(r.id)) ?? new Set());
      if (i > 0) add(r.id, s.rows[i - 1].id);
      else for (const e of g.edges) if (e.kind === "input" && e.to.stack === s.id) add(r.id, endRow(e.from));
      for (const c of r.cells) for (const e of g.edges) if (e.kind === "source" && e.to.cell === c.id) add(r.id, endRow(e.from));
      for (const e of g.edges) if (e.kind === "skip" && e.to.row === r.id) add(r.id, e.from.row);
    });

  // priority: the stack's position, then the row's
  const key = (u: string) => {
    const grp = g.groups.find((x) => x.id === u);
    const row = grp ? grp.rows[0] : u;
    const h = rowHome.get(row);
    return h ? g.stacks.indexOf(h.stack) * 1e6 + h.index : 1e12;
  };
  const done = new Set<string>();
  const order: string[] = [];
  const pending = [...deps.keys()];
  while (pending.length) {
    const ready = pending.filter((u) => [...(deps.get(u) ?? [])].every((d) => done.has(d) || !deps.has(d)));
    const next = (ready.length ? ready : pending).sort((a, b) => key(a) - key(b))[0]; // a cycle: take the earliest
    order.push(next);
    done.add(next);
    pending.splice(pending.indexOf(next), 1);
  }

  return order.map((u) => {
    const grp = g.groups.find((x) => x.id === u);
    if (grp) {
      const s = g.stacks.find((x) => x.id === grp.stack)!;
      const rows = s.rows.filter((r) => grp.rows.includes(r.id)).map((r) => r.id);
      return grp.kind === "loop"
        ? { kind: "loop" as const, stack: grp.stack, rows, count: grp.count, shared_weights: true, round_embedding: grp.round_embedding, feedback: grp.feedback,
            note: `${s.name}: run ${rows.join(", ")} ${grp.count} times with the same weights` }
        : { kind: "repeat" as const, stack: grp.stack, rows, count: grp.count, shared_weights: false,
            note: `${s.name}: ${grp.count} independent copies of ${rows.join(", ")}, one after another` };
    }
    const h = rowHome.get(u)!;
    return { kind: "row" as const, stack: h.stack.id, rows: [u], note: `${h.stack.name} · ${h.stack.rows[h.index].label ?? `row ${h.index}`}` };
  });
}

// ------------------------------------------------------------------ the spec

export function buildSpec(g: Graph, given?: Analysis): Spec {
  const a = given ?? analyze(g);
  const b = a.memory.microBatch;
  const errors: SpecProblem[] = a.issues.filter((i) => i.level !== "warning").map(toProblem);
  const warnings: SpecProblem[] = a.issues.filter((i) => i.level === "warning").map(toProblem);

  // dims: B, then the position dims of every stack (S, S_src, pages, W, slots, N, notes)
  const dims: Spec["dims"] = [{ name: "B", value: null, meaning: "rows (sequences) in a batch: any size" }];
  const addDim = (name: string, value: number, meaning: string) => {
    if (!dims.some((d) => d.name === name)) dims.push({ name, value, meaning });
  };
  for (const s of g.stacks) {
    const res = a.stacks.find((x) => x.id === s.id)!;
    for (const p of positions(res.seq)) {
      const meaning =
        p.n === "S" ? "tokens in a training row" :
        p.n === "S_src" ? `tokens in ${s.name}'s source text` :
        p.n === "N" ? `learned latents in ${s.name}` :
        p.n === "pages" ? `pages over ${g.stacks.find((x) => x.id === (s.seq as { of?: string }).of)?.name ?? "the row"}${res.seq.pages ? ` (${res.seq.pages.window} tokens, a new page every ${res.seq.pages.stride})` : ""}` :
        p.n === "W" ? "tokens in a page" :
        p.n === "slots" ? `slots per page in ${s.name}` : p.n;
      addDim(p.n, p.v, meaning);
    }
    const out = stackOutputShape(g, a, s.id, b).find((d) => d.n === "notes");
    if (out) addDim("notes", out.v, `notes ${s.name} hands to its readers (pages × slots × entries)`);
  }

  const inputs: Spec["inputs"] = [];
  const outputs: Spec["outputs"] = [];
  const stacks: SpecStack[] = g.stacks.map((s) => {
    const res = a.stacks.find((x) => x.id === s.id)!;
    const io = stackIO(s, g);
    const blocks: SpecBlock[] = [];
    const rows: SpecRow[] = s.rows.map((row, ri) => {
      for (const c of row.cells) {
        const r = a.cells.get(c.id);
        if (!r) continue;
        const steps = cellTensorSteps(g, a, c.id, b).filter((t) => t.dims.length || t.label === "loss");
        const first = steps[0]?.dims ?? [];
        const last = [...steps].reverse().find((t) => t.dims.length)?.dims ?? [];
        const wire = g.edges.find((e) => e.kind === "source" && e.to.cell === c.id);
        const block: SpecBlock = {
          id: c.id,
          type: c.type,
          title: cellTitle(c),
          ...(c.label ? { label: c.label } : {}),
          what_it_does: mixing(c, s, g).text,
          settings: settingsOf(c),
          ...(c.type === "attention" || c.type === "ffn" ? { residual: c.residual } : {}),
          input: { tensor: c.type === "embedding" ? `${s.id}.input_ids` : `${s.id}.x`, shape: shapeOf(first) },
          output: { tensor: `${s.id}.x`, shape: shapeOf(last) },
          runs_per_row: r.execs,
          frozen: Boolean(c.frozen || s.frozen),
          parameters: { count: r.params, parts: r.parts.map((p) => ({ name: p.name, count: p.params })) },
          tensor_steps: steps.map((t) => ({ step: t.label, shape: shapeOf(t.dims), ...(t.note ? { note: t.note } : {}) })),
        };
        if (wire) {
          const w = edgeTensors(g, a, wire.id, b);
          const memory = c.type === "attention" && c.mode === "self" && c.memory;
          block.reads = { wire: wire.id, from: endpointName(g, wire.from), as: memory ? "notes" : "keys/values", shape: shapeOf(w.from.dims) };
        }
        if (c.type === "embedding") {
          inputs.push({ name: `${s.id}.input_ids`, shape: shapeOf(first), dtype: "int64", meaning: `token ids for ${s.name} (vocabulary ${c.vocab})` });
        }
        if (c.type === "lm_head") {
          const logits = steps.find((t) => t.label === "logits");
          if (logits) outputs.push({ name: `${s.id}.logits`, shape: shapeOf(logits.dims), meaning: `next-token logits of ${s.name}; the loss is cross-entropy against the next token` });
        }
        blocks.push(block);
      }
      return {
        id: row.id,
        label: row.label ?? `row ${ri}`,
        blocks: row.cells.map((c) => c.id),
        before: {
          x0_mix: row.x0_mix,
          skips_in: g.edges.filter((e) => e.kind === "skip" && e.to.row === row.id).map((e) => ({ from_row: e.from.row ?? "", weight: e.kind === "skip" ? e.weight : "none" })),
        },
      };
    });
    const firstRow = s.rows[0];
    const inShape = firstRow ? rowInputShape(g, a, firstRow.id, b) : [];
    return {
      id: s.id,
      name: s.name,
      role: s.role,
      width: s.width,
      frozen: Boolean(s.frozen),
      sequence: { kind: s.seq.kind, shape: shapeOf([{ n: "B", v: b }, ...positions(res.seq), { n: "d", v: res.widthOut || s.width }]), description: io.input },
      starts_from: inShape.kind === "ids" ? "token ids" : io.input,
      rows,
      blocks,
      output: { tensor: `${s.id}.output`, shape: shapeOf(stackOutputShape(g, a, s.id, b)), goes_to: io.outputs },
    };
  });

  const wires: Spec["wires"] = g.edges.map((e) => {
    const t = edgeTensors(g, a, e.id, b);
    return {
      id: e.id,
      kind: e.kind,
      from: endpointName(g, e.from),
      to: e.kind === "input" ? `${e.to.stack} (its stream starts here)` : e.kind === "skip" ? `${e.to.row} (added before it runs)` : `${e.to.cell}`,
      carries: shapeOf(t.from.dims),
      meaning: t.summary,
    };
  });

  return {
    format: SPEC_FORMAT,
    version: SPEC_VERSION,
    design: { name: g.name, description: g.description, hash: designHash(g) },
    how_to_read: HOW_TO_READ,
    block_types: Object.fromEntries(Object.entries(BLOCK_TYPES).filter(([k]) => g.stacks.some((s) => s.rows.some((r) => r.cells.some((c) => c.type === k))))),
    dims,
    inputs,
    outputs,
    stacks,
    wires,
    execution_order: executionOrder(g),
    parameters: { total: a.params.total, trained: a.params.trainable, used_per_token: a.params.active, by_stack: a.params.byStack },
    training: { row_length: g.run.row_length, rows_per_step: g.run.rows_per_step, tokens: a.tokens.budget, precision: g.run.precision, optimizer: g.run.optimizer, weight_decay: g.run.weight_decay },
    validation: { errors, warnings },
  };
}

const toProblem = (i: Issue): SpecProblem => ({ where: i.where, message: i.msg });

export function validateDesign(g: Graph): { errors: SpecProblem[]; warnings: SpecProblem[] } {
  const a = analyze(g);
  return {
    errors: a.issues.filter((i) => i.level !== "warning").map(toProblem),
    warnings: a.issues.filter((i) => i.level === "warning").map(toProblem),
  };
}

// ------------------------------------------------------------------ hash and diff

/** JSON with sorted keys, so equal designs give equal text. */
function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object")
    return `{${Object.keys(v as object)
      .filter((k) => (v as Record<string, unknown>)[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`)
      .join(",")}}`;
  return JSON.stringify(v);
}

/** cyrb53: a small, fast 53-bit string hash (two seeds give 106 bits). */
function cyrb53(str: string, seed: number): string {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, "0");
}

/** The architecture's identity: stacks, wires and groups (the design's name, notes, layout and run setup excluded). */
export function designHash(g: Graph): string {
  const text = canonical({ stacks: g.stacks, edges: g.edges, groups: g.groups });
  return `nnb1-${cyrb53(text, 1)}${cyrb53(text, 2)}`;
}

/** A stable text for change detection in the app: the whole saved file (layout included), not just the architecture. */
export const contentKey = (g: Graph): string => cyrb53(canonical({ ...g, derived: undefined }), 7);

export interface Change {
  change: "added" | "removed" | "changed" | "moved";
  what: "stack" | "row" | "block" | "wire" | "group";
  id: string;
  where: string;
  field?: string;
  from?: unknown;
  to?: unknown;
}
export interface Diff {
  from_hash: string;
  to_hash: string;
  changes: Change[];
  parameters: { from: number; to: number };
  summary: string;
}

const eq = (x: unknown, y: unknown) => canonical(x) === canonical(y);

function fieldChanges(what: Change["what"], id: string, where: string, a: Record<string, unknown>, b: Record<string, unknown>, skip: string[]): Change[] {
  const out: Change[] = [];
  for (const k of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort()) {
    if (skip.includes(k) || eq(a[k], b[k])) continue;
    out.push({ change: "changed", what, id, where, field: k, from: a[k], to: b[k] });
  }
  return out;
}

export function diffDesigns(before: Graph, after: Graph): Diff {
  const changes: Change[] = [];
  const stackName = (g: Graph, id: string) => g.stacks.find((s) => s.id === id)?.name ?? id;

  const sa = new Map(before.stacks.map((s) => [s.id, s]));
  const sb = new Map(after.stacks.map((s) => [s.id, s]));
  for (const [id, s] of sa) if (!sb.has(id)) changes.push({ change: "removed", what: "stack", id, where: s.name });
  for (const [id, s] of sb) {
    const old = sa.get(id);
    if (!old) {
      changes.push({ change: "added", what: "stack", id, where: s.name });
      continue;
    }
    changes.push(...fieldChanges("stack", id, s.name, old as unknown as Record<string, unknown>, s as unknown as Record<string, unknown>, ["id", "rows"]));
  }

  const rows = (g: Graph) => new Map(g.stacks.flatMap((s) => s.rows.map((r, i) => [r.id, { s, r, i }] as const)));
  const ra = rows(before), rb = rows(after);
  for (const [id, x] of ra) if (!rb.has(id)) changes.push({ change: "removed", what: "row", id, where: `${x.s.name} · ${x.r.label ?? `row ${x.i}`}` });
  for (const [id, x] of rb) {
    const old = ra.get(id);
    const where = `${x.s.name} · ${x.r.label ?? `row ${x.i}`}`;
    if (!old) {
      changes.push({ change: "added", what: "row", id, where, to: `position ${x.i} of ${x.s.name}` });
      continue;
    }
    if (old.s.id !== x.s.id || relativeOrderChanged(before, after, old.s.id, x.s.id, id))
      changes.push({ change: "moved", what: "row", id, where, from: `position ${old.i} of ${old.s.name}`, to: `position ${x.i} of ${x.s.name}` });
    changes.push(...fieldChanges("row", id, where, old.r as unknown as Record<string, unknown>, x.r as unknown as Record<string, unknown>, ["id", "cells"]));
  }

  const cells = (g: Graph) => new Map(g.stacks.flatMap((s) => s.rows.flatMap((r) => r.cells.map((c, i) => [c.id, { s, r, c, i }] as const))));
  const ca = cells(before), cb = cells(after);
  for (const [id, x] of ca) if (!cb.has(id)) changes.push({ change: "removed", what: "block", id, where: `${x.s.name} · ${x.r.label ?? x.r.id}` });
  for (const [id, x] of cb) {
    const old = ca.get(id);
    const where = `${x.s.name} · ${x.r.label ?? x.r.id}`;
    if (!old) {
      changes.push({ change: "added", what: "block", id, where, to: x.c });
      continue;
    }
    if (old.r.id !== x.r.id || old.i !== x.i) changes.push({ change: "moved", what: "block", id, where, from: `${old.r.id} slot ${old.i}`, to: `${x.r.id} slot ${x.i}` });
    changes.push(...fieldChanges("block", id, where, old.c as unknown as Record<string, unknown>, x.c as unknown as Record<string, unknown>, ["id"]));
  }

  const ea = new Map(before.edges.map((e) => [e.id, e]));
  const ebm = new Map(after.edges.map((e) => [e.id, e]));
  const wireWhere = (g: Graph, e: Edge) => `${e.kind}: ${stackName(g, e.from.stack)} → ${stackName(g, e.to.stack)}`;
  for (const [id, e] of ea) if (!ebm.has(id)) changes.push({ change: "removed", what: "wire", id, where: wireWhere(before, e) });
  for (const [id, e] of ebm) {
    const old = ea.get(id);
    if (!old) changes.push({ change: "added", what: "wire", id, where: wireWhere(after, e), to: { from: e.from, to: e.to } });
    else changes.push(...fieldChanges("wire", id, wireWhere(after, e), old as unknown as Record<string, unknown>, e as unknown as Record<string, unknown>, ["id"]));
  }

  const ga = new Map(before.groups.map((x) => [x.id, x]));
  const gb = new Map(after.groups.map((x) => [x.id, x]));
  const groupWhere = (g: Graph, x: Group) => `${stackName(g, x.stack)} · ${x.kind} over ${x.rows.join(", ")}`;
  for (const [id, x] of ga) if (!gb.has(id)) changes.push({ change: "removed", what: "group", id, where: groupWhere(before, x) });
  for (const [id, x] of gb) {
    const old = ga.get(id);
    if (!old) changes.push({ change: "added", what: "group", id, where: groupWhere(after, x), to: { kind: x.kind, rows: x.rows, count: x.count } });
    else changes.push(...fieldChanges("group", id, groupWhere(after, x), old as unknown as Record<string, unknown>, x as unknown as Record<string, unknown>, ["id"]));
  }

  const pa = analyze(before).params.total, pb = analyze(after).params.total;
  const count = (c: Change["change"]) => changes.filter((x) => x.change === c).length;
  const summary = changes.length
    ? `${changes.length} change${changes.length > 1 ? "s" : ""}: ${count("added")} added, ${count("removed")} removed, ${count("changed")} settings changed, ${count("moved")} moved; parameters ${pa.toLocaleString("en-US")} → ${pb.toLocaleString("en-US")}`
    : "no changes";
  return { from_hash: designHash(before), to_hash: designHash(after), changes, parameters: { from: pa, to: pb }, summary };
}

/** A row moved if the rows that both versions share are in a different order around it. */
function relativeOrderChanged(a: Graph, b: Graph, sa: string, sb: string, row: string): boolean {
  const ra = a.stacks.find((s) => s.id === sa)?.rows.map((r) => r.id) ?? [];
  const rb = b.stacks.find((s) => s.id === sb)?.rows.map((r) => r.id) ?? [];
  const shared = new Set(ra.filter((x) => rb.includes(x)));
  const fa = ra.filter((x) => shared.has(x)), fb = rb.filter((x) => shared.has(x));
  return fa.indexOf(row) !== fb.indexOf(row);
}


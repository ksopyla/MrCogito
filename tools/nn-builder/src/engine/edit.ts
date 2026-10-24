// Pure edits of a graph: each returns a new graph and never changes the one it was given.
import { attention, defaultBody, newId, round64 } from "./defaults";
import type { Cell, CellBody, Edge, Endpoint, Graph, Group, Row, Stack } from "./types";

const clone = (g: Graph): Graph => structuredClone(g);

export function findRow(g: Graph, rowId: string) {
  for (const stack of g.stacks) {
    const index = stack.rows.findIndex((r) => r.id === rowId);
    if (index >= 0) return { stack, row: stack.rows[index], index };
  }
  return undefined;
}

export function findCell(g: Graph, cellId: string) {
  for (const stack of g.stacks)
    for (const row of stack.rows) {
      const index = row.cells.findIndex((c) => c.id === cellId);
      if (index >= 0) return { stack, row, cell: row.cells[index], index };
    }
  return undefined;
}

const stackOf = (g: Graph, id: string) => {
  const s = g.stacks.find((x) => x.id === id);
  if (!s) throw new Error(`unknown stack ${id}`);
  return s;
};

const isLayerRow = (r: Row) => r.cells.some((c) => c.type === "attention" || c.type === "ffn");
const touches = (e: Endpoint, rowIds: Set<string>, cellIds: Set<string>) =>
  (e.row !== undefined && rowIds.has(e.row)) || (e.cell !== undefined && cellIds.has(e.cell));

function dropWires(g: Graph, rowIds: Set<string>, cellIds: Set<string>) {
  g.edges = g.edges.filter((e) => !touches(e.from, rowIds, cellIds) && !touches(e.to, rowIds, cellIds));
  // an embedding that shared a removed table gets its own again
  for (const s of g.stacks) for (const r of s.rows) for (const c of r.cells) if (c.type === "embedding" && c.share_with && cellIds.has(c.share_with)) delete c.share_with;
}

/** A row that lands between two rows of a loop or repeat joins it, so groups always cover neighbouring rows. */
function healGroups(g: Graph, stackId: string) {
  const s = stackOf(g, stackId);
  const member = new Set(g.groups.flatMap((x) => x.rows));
  for (const grp of g.groups.filter((x) => x.stack === stackId)) {
    const idx = grp.rows.map((r) => s.rows.findIndex((x) => x.id === r)).filter((i) => i >= 0);
    if (!idx.length) continue;
    const lo = Math.min(...idx), hi = Math.max(...idx);
    grp.rows = s.rows.slice(lo, hi + 1).filter((r) => grp.rows.includes(r.id) || !member.has(r.id)).map((r) => r.id);
    for (const r of grp.rows) member.add(r);
  }
}

function dropRowsFromGroups(g: Graph, rowIds: Set<string>) {
  for (const grp of g.groups) grp.rows = grp.rows.filter((r) => !rowIds.has(r));
  g.groups = g.groups.filter((grp) => grp.rows.length > 0);
}

export const makeCell = (body: CellBody): Cell => ({ ...structuredClone(body), id: newId(body.type.slice(0, 4)) }) as Cell;

// ------------------------------------------------------------------ rows and cells

export function insertRow(g0: Graph, stackId: string, index: number, bodies: CellBody[], label?: string): Graph {
  const g = clone(g0);
  const s = stackOf(g, stackId);
  const at = Math.max(0, Math.min(index, s.rows.length));
  const above = s.rows[at - 1];
  const row: Row = { id: newId("row"), x0_mix: Boolean(above && isLayerRow(above) && above.x0_mix), cells: bodies.map(makeCell) };
  if (label) row.label = label;
  s.rows.splice(at, 0, row);
  // a new row inside a group joins it
  for (const grp of g.groups) {
    if (grp.stack !== stackId) continue;
    const idx = grp.rows.map((r) => s.rows.findIndex((x) => x.id === r));
    if (Math.min(...idx) < at && Math.max(...idx) > at) grp.rows.splice(grp.rows.indexOf(s.rows[at - 1].id) + 1, 0, row.id);
  }
  return g;
}

export function insertCell(g0: Graph, rowId: string, index: number, body: CellBody): Graph {
  const g = clone(g0);
  const home = findRow(g, rowId);
  if (!home) throw new Error(`unknown row ${rowId}`);
  home.row.cells.splice(Math.max(0, Math.min(index, home.row.cells.length)), 0, makeCell(body));
  return g;
}

export function moveCell(g0: Graph, cellId: string, toRowId: string, index: number): Graph {
  const g = clone(g0);
  const from = findCell(g, cellId);
  const to = findRow(g, toRowId);
  if (!from || !to) return g0;
  from.row.cells.splice(from.index, 1);
  let at = index;
  if (from.row.id === to.row.id && from.index < index) at -= 1;
  to.row.cells.splice(Math.max(0, Math.min(at, to.row.cells.length)), 0, from.cell);
  for (const e of g.edges) {
    if (e.from.cell === cellId) e.from.stack = to.stack.id;
    if (e.to.cell === cellId) e.to.stack = to.stack.id;
  }
  return g;
}

export function moveRow(g0: Graph, rowId: string, index: number): Graph {
  const g = clone(g0);
  const home = findRow(g, rowId);
  if (!home) return g0;
  const rows = home.stack.rows;
  rows.splice(home.index, 1);
  const at = home.index < index ? index - 1 : index;
  rows.splice(Math.max(0, Math.min(at, rows.length)), 0, home.row);
  // a group whose rows are no longer neighbours lets the moved row go; a group it lands inside takes it in
  for (const grp of g.groups.filter((x) => x.stack === home.stack.id)) {
    const idx = grp.rows.map((r) => rows.findIndex((x) => x.id === r)).sort((a, b) => a - b);
    if (idx.some((v, i) => i > 0 && v !== idx[i - 1] + 1)) grp.rows = grp.rows.filter((r) => r !== rowId);
  }
  g.groups = g.groups.filter((x) => x.rows.length > 0);
  healGroups(g, home.stack.id);
  return g;
}

export function removeCell(g0: Graph, cellId: string): Graph {
  const g = clone(g0);
  const home = findCell(g, cellId);
  if (!home) return g0;
  home.row.cells.splice(home.index, 1);
  dropWires(g, new Set(), new Set([cellId]));
  return g;
}

export function removeRow(g0: Graph, rowId: string): Graph {
  const g = clone(g0);
  const home = findRow(g, rowId);
  if (!home) return g0;
  home.stack.rows.splice(home.index, 1);
  dropWires(g, new Set([rowId]), new Set(home.row.cells.map((c) => c.id)));
  dropRowsFromGroups(g, new Set([rowId]));
  return g;
}

export function duplicateRow(g0: Graph, rowId: string, times = 1): Graph {
  const g = clone(g0);
  const home = findRow(g, rowId);
  if (!home) return g0;
  const copies: Row[] = [];
  for (let k = 0; k < times; k++) {
    const map = new Map<string, string>();
    const row: Row = {
      ...structuredClone(home.row),
      id: newId("row"),
      label: home.row.label ? `${home.row.label}′` : undefined,
      cells: home.row.cells.map((c) => {
        const id = newId(c.type.slice(0, 4));
        map.set(c.id, id);
        return { ...structuredClone(c), id };
      }),
    };
    // copies of a cross-attention or memory read keep reading the same source
    for (const e of g0.edges)
      if (e.kind === "source" && e.to.cell && map.has(e.to.cell)) g.edges.push({ ...structuredClone(e), id: newId("wire"), to: { ...e.to, cell: map.get(e.to.cell)! } });
    copies.push(row);
  }
  home.stack.rows.splice(home.index + 1, 0, ...copies);
  healGroups(g, home.stack.id);
  return g;
}

export function updateCell(g0: Graph, cellId: string, patch: Partial<CellBody> & { label?: string }): Graph {
  const g = clone(g0);
  const home = findCell(g, cellId);
  if (!home) return g0;
  const c = home.cell as Cell & { extras?: object };
  const old = c.extras ? { ...c.extras } : undefined;
  const extras = (patch as { extras?: object }).extras;
  Object.assign(c, patch);
  if (extras && old) c.extras = { ...old, ...extras };
  return g;
}

export function updateRow(g0: Graph, rowId: string, patch: Partial<Row>): Graph {
  const g = clone(g0);
  const home = findRow(g, rowId);
  if (home) Object.assign(home.row, patch);
  return g;
}

export function updateStack(g0: Graph, stackId: string, patch: Partial<Stack>): Graph {
  const g = clone(g0);
  Object.assign(stackOf(g, stackId), patch);
  return g;
}

export function updateGroup(g0: Graph, groupId: string, patch: Partial<Group>): Graph {
  const g = clone(g0);
  const grp = g.groups.find((x) => x.id === groupId);
  if (grp) Object.assign(grp, patch);
  return g;
}

export function updateRun(g0: Graph, patch: Partial<Graph["run"]>): Graph {
  const g = clone(g0);
  Object.assign(g.run, patch);
  return g;
}

/** Change a stack's width, keeping full-width attention full-width and each FFN's width ratio. */
export function setStackWidth(g0: Graph, stackId: string, width: number): Graph {
  const g = clone(g0);
  const s = stackOf(g, stackId);
  const old = s.width;
  s.width = width;
  for (const r of s.rows)
    for (const c of r.cells) {
      if (c.type === "attention" && c.heads * c.head_dim === old) c.heads = Math.max(1, Math.round(width / c.head_dim));
      if (c.type === "attention" && c.kv_heads > c.heads) c.kv_heads = c.heads;
      if (c.type === "attention" && c.heads % c.kv_heads) c.kv_heads = 1;
      if (c.type === "ffn") c.hidden = round64((c.hidden * width) / old);
      if (c.type === "embedding" && c.proj === "none" && c.dim === old) c.dim = width;
      if (c.type === "linear" && c.out === old) c.out = width;
    }
  return g;
}

// ------------------------------------------------------------------ wires

/** Wire `from` to `to` by what they are: row → later row = skip; anything → attention cell = source; → stack = input. */
export function connect(g0: Graph, from: Endpoint, to: Endpoint): { graph: Graph; error?: string } {
  const g = clone(g0);
  if (to.cell) {
    const target = findCell(g, to.cell);
    if (!target || target.cell.type !== "attention") return { graph: g0, error: "wires into a cell feed attention only (cross-attention or a notebook read)" };
    if (from.stack === target.stack.id && !from.cell && !from.row) return { graph: g0, error: "a cell cannot read its own stack's output" };
    const a = target.cell;
    if (a.mode === "self" && !a.memory) a.memory = true; // a wire into self-attention makes it a notebook read
    g.edges = g.edges.filter((e) => !(e.kind === "source" && e.to.cell === to.cell));
    g.edges.push({ id: newId("wire"), kind: "source", from, to: { stack: target.stack.id, cell: to.cell } });
    return { graph: g };
  }
  if (to.row) {
    const a = from.row ? findRow(g, from.row) : undefined;
    const b = findRow(g, to.row);
    if (!a || !b) return { graph: g0, error: "drag a skip wire from a row's output dot" };
    if (a.stack.id !== b.stack.id) return { graph: g0, error: "skip wires join rows of one stack; to feed another stack, drop on its header" };
    if (b.index <= a.index) return { graph: g0, error: "a skip wire must go to a later row" };
    if (g.edges.some((e) => e.kind === "skip" && e.from.row === a.row.id && e.to.row === b.row.id)) return { graph: g0, error: "these rows are already joined" };
    g.edges.push({ id: newId("wire"), kind: "skip", from: { stack: a.stack.id, row: a.row.id }, to: { stack: b.stack.id, row: b.row.id }, weight: "scalar" });
    return { graph: g };
  }
  if (to.stack === from.stack) return { graph: g0, error: "a stack cannot start from itself" };
  stackOf(g, to.stack);
  g.edges = g.edges.filter((e) => !(e.kind === "input" && e.to.stack === to.stack));
  g.edges.push({ id: newId("wire"), kind: "input", from, to: { stack: to.stack } });
  return { graph: g };
}

export function removeEdge(g0: Graph, edgeId: string): Graph {
  const g = clone(g0);
  g.edges = g.edges.filter((e) => e.id !== edgeId);
  return g;
}

export function updateEdge(g0: Graph, edgeId: string, patch: Partial<Edge>): Graph {
  const g = clone(g0);
  const e = g.edges.find((x) => x.id === edgeId);
  if (e) Object.assign(e, patch);
  return g;
}

/** Replace the stack's skip wires with the U-net pattern over its layer rows. */
export function unetSkips(g0: Graph, stackId: string): Graph {
  const g = clone(g0);
  const s = stackOf(g, stackId);
  const layers = s.rows.filter(isLayerRow);
  const k = Math.floor(layers.length / 2);
  g.edges = g.edges.filter((e) => !(e.kind === "skip" && e.to.stack === stackId));
  for (let j = 0; j < k; j++)
    g.edges.push({ id: newId("wire"), kind: "skip", from: { stack: stackId, row: layers[k - 1 - j].id }, to: { stack: stackId, row: layers[layers.length - k + j].id }, weight: "scalar" });
  return g;
}

// ------------------------------------------------------------------ groups

export function addGroup(g0: Graph, kind: Group["kind"], stackId: string, rowIds: string[], count: number): Graph {
  const g = clone(g0);
  const s = stackOf(g, stackId);
  const idx = rowIds.map((r) => s.rows.findIndex((x) => x.id === r)).sort((a, b) => a - b);
  if (idx.length === 0 || idx.some((i) => i < 0) || idx.some((v, i) => i > 0 && v !== idx[i - 1] + 1)) throw new Error("rows must be neighbouring rows of one stack");
  if (g.groups.some((grp) => grp.rows.some((r) => rowIds.includes(r)))) throw new Error("a row is already in a loop or repeat");
  g.groups.push({ id: newId(kind), kind, stack: stackId, rows: idx.map((i) => s.rows[i].id), count: Math.max(1, count), round_embedding: kind === "loop", feedback: "none" });
  return g;
}

export function removeGroup(g0: Graph, groupId: string): Graph {
  const g = clone(g0);
  g.groups = g.groups.filter((x) => x.id !== groupId);
  return g;
}

// ------------------------------------------------------------------ stacks

export type StackTemplate = "decoder" | "encoder" | "pages" | "slots" | "latents";

export function addStack(g0: Graph, template: StackTemplate): Graph {
  const g = clone(g0);
  const main = g.stacks.find((s) => s.seq.kind === "tokens") ?? g.stacks[0];
  const w = template === "pages" ? 256 : template === "slots" || template === "latents" ? 512 : main?.width ?? 512;
  const id = newId("stack");
  const r = (cells: CellBody[], label: string, x0 = false): Row => ({ id: newId("row"), label, x0_mix: x0, cells: cells.map(makeCell) });
  const layer = (o: Parameters<typeof attention>[1] = {}) => r([attention(w, o), defaultBody("ffn", w)], "layer");
  let s: Stack;
  switch (template) {
    case "decoder":
      s = { id, name: "Decoder", role: "decoder", width: w, seq: { kind: "tokens" }, rows: [r([defaultBody("embedding", w), defaultBody("positional", w)], "input"), layer(), r([defaultBody("lm_head", w)], "head")] };
      break;
    case "encoder":
      s = { id, name: "Encoder", role: "encoder", width: w, seq: { kind: "source", length: 0 }, rows: [r([defaultBody("embedding", w), defaultBody("positional", w)], "input"), layer({ causal: false }), r([defaultBody("norm", w)], "out")] };
      break;
    case "pages":
      s = { id, name: "Pages", role: "writer", width: w, seq: { kind: "pages", of: main.id, window: 256, stride: 192 }, rows: [r([{ ...(defaultBody("linear", w) as object), out_norm: "rmsnorm" } as CellBody], "in"), layer({ causal: false })] };
      break;
    case "slots": {
      const pages = g.stacks.find((x) => x.seq.kind === "pages");
      s = {
        id, name: "Notes", role: "latent", width: w, seq: { kind: "slots", of: pages?.id ?? "", per_page: 32 },
        rows: [r([defaultBody("latent_init", w)], "slots"), r([attention(w, { mode: "cross", causal: false })], "read page"), r([defaultBody("ffn", w)], "think"), r([defaultBody("kv_out", w)], "notes")],
      };
      break;
    }
    case "latents":
      s = { id, name: "Latents", role: "latent", width: w, seq: { kind: "latents", count: 256 }, rows: [r([defaultBody("latent_init", w)], "latents"), r([attention(w, { mode: "cross", causal: false })], "read"), layer({ causal: false })] };
      break;
  }
  g.stacks.push(s);
  return g;
}

export function removeStack(g0: Graph, stackId: string): Graph {
  const g = clone(g0);
  const gone = g.stacks.find((s) => s.id === stackId);
  g.stacks = g.stacks.filter((s) => s.id !== stackId);
  g.edges = g.edges.filter((e) => e.from.stack !== stackId && e.to.stack !== stackId);
  g.groups = g.groups.filter((x) => x.stack !== stackId);
  if (gone) dropWires(g, new Set(gone.rows.map((r) => r.id)), new Set(gone.rows.flatMap((r) => r.cells.map((c) => c.id))));
  return g;
}

export function moveStack(g0: Graph, stackId: string, dir: -1 | 1): Graph {
  const g = clone(g0);
  const i = g.stacks.findIndex((s) => s.id === stackId);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= g.stacks.length) return g0;
  [g.stacks[i], g.stacks[j]] = [g.stacks[j], g.stacks[i]];
  return g;
}

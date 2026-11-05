// Tensor shapes along the forward pass: what each cell receives, makes inside, and hands on, for a batch of B rows.
import type { Analysis } from "./analyze";
import type { SeqInfo } from "./cells";
import type { Cell, Edge, Graph } from "./types";

export interface Dim {
  n: string; // name: B, S, pages, W, slots, h, dh, d, F, V, ...
  v: number;
}
export type Shape = Dim[] & { kind?: "ids" | "start" };
export interface TensorStep {
  label: string;
  dims: Dim[];
  note?: string;
}

const dim = (n: string, v: number): Dim => ({ n, v });
const shape = (dims: Dim[], kind?: Shape["kind"]): Shape => Object.assign([...dims], kind ? { kind } : {});

/** The position dimensions of a stack's stream (without the batch and the width). */
export function positions(seq: SeqInfo): Dim[] {
  switch (seq.kind) {
    case "tokens":
      return [dim("S", seq.P)];
    case "source":
      return [dim("S_src", seq.P)];
    case "latents":
      return [dim("N", seq.P)];
    case "pages":
      return [dim("pages", seq.pages?.count ?? 1), dim("W", seq.pages?.window ?? seq.P)];
    case "slots":
      return [dim("pages", seq.pages?.count ?? 1), dim("slots", seq.perPage ?? seq.P)];
  }
}

function locate(g: Graph, a: Analysis, cellId: string) {
  for (const s of g.stacks)
    for (const row of s.rows) {
      const cell = row.cells.find((c) => c.id === cellId);
      if (cell) return { stack: s, row, cell, seq: a.stacks.find((x) => x.id === s.id)!.seq, res: a.cells.get(cellId)! };
    }
  return undefined;
}

/** Step by step: the tensors a cell reads, makes and returns. */
export function cellTensorSteps(g: Graph, a: Analysis, cellId: string, b: number): TensorStep[] {
  const home = locate(g, a, cellId);
  if (!home || !home.res) return [];
  const { cell: c, seq, res } = home;
  const B = dim("B", b);
  const pos = positions(seq);
  const X = (w: number): Dim[] => [B, ...pos, dim("d", w)];
  const d = res.widthIn, out = res.widthOut;
  const steps: TensorStep[] = [];
  const push = (label: string, dims: Dim[], note?: string) => steps.push({ label, dims, note });

  switch (c.type) {
    case "embedding":
      push("token ids", [B, ...pos], `integers 0 … ${c.vocab - 1}`);
      push("table lookup", [B, ...pos, dim("e", c.dim)], c.ngram_tables ? `plus ${c.ngram_tables} hashed n-gram tables` : c.share_with ? "shared table" : undefined);
      if (c.proj === "gated_mlp") push("input MLP: up0(e) + up2(silu(up1(e)))", X(out));
      if (c.proj === "linear") push("projection to the width", X(out));
      if (c.out_norm !== "none") push(c.out_norm === "rmsnorm" ? "RMSNorm" : "LayerNorm", X(out));
      push("vectors", X(out));
      break;
    case "positional":
      push("input", X(d));
      if (c.kind === "learned") push("+ position table", [...pos, dim("d", d)]);
      push(c.kind === "rope" ? "unchanged: rotary angles turn q and k inside the attention that follows" : `positions: ${c.kind}`, X(d));
      break;
    case "attention":
      attentionSteps(g, a, c, home, b, push);
      break;
    case "ffn": {
      push("input", X(d));
      if (c.norm !== "none") push("pre-norm", X(d));
      const F = dim("F", c.hidden);
      if (c.experts > 1) push("router scores", [B, ...pos, dim("experts", c.experts)], `each token goes to ${c.top_k}`);
      if (c.kind === "mlp") {
        push("hidden: up", [B, ...pos, F]);
        push(`${c.activation.toUpperCase()}`, [B, ...pos, F]);
      } else {
        push("hidden: gate and up (each)", [B, ...pos, F]);
        push(c.kind === "swiglu" ? "silu(gate) · up" : "gelu(gate) · up", [B, ...pos, F]);
      }
      push("down", X(d));
      if (c.residual !== "none") push("⊕ added to the stream", X(d));
      break;
    }
    case "norm":
      push("input", X(d));
      push(c.kind === "rmsnorm" ? "RMSNorm" : c.kind === "layernorm" ? "LayerNorm" : "unchanged", X(d));
      break;
    case "linear":
      push("input", X(d));
      push("projection", X(out));
      if (c.out_norm !== "none") push("norm", X(out));
      break;
    case "latent_init": {
      const n = seq.kind === "slots" ? seq.perPage ?? 0 : seq.P;
      push("learned queries", [dim(seq.kind === "slots" ? "slots" : "N", n), dim("d", out)], "the same queries for every row" + (seq.kind === "slots" ? " and every page" : ""));
      if (c.address_feats) push("+ address map of the page position", [B, ...pos.slice(0, 1), dim("d", out)]);
      push("stream starts", X(out));
      break;
    }
    case "kv_out": {
      push("input", X(d));
      const notes = (seq.pages?.count ?? 1) * (seq.perPage ?? seq.P) * c.entries;
      const L = Math.max(1, c.layers ?? 1);
      const layer = L > 1 ? [dim("layers", L)] : [];
      push(`${c.projections === 4 ? "4" : "2"} projections → keys / values`, [B, ...pos, dim("entries", c.entries), ...layer, dim("g", c.kv_heads), dim("dh", c.head_dim)]);
      push(L > 1 ? `notes: ready keys / values for each of ${L} reader layers` : "notes: ready keys / values for a reader", [B, dim("notes", notes), ...layer, dim("g", c.kv_heads), dim("dh", c.head_dim)]);
      break;
    }
    case "pool": {
      const p = seq.pages;
      push("pages from the wired stack", [B, dim("pages", p?.count ?? 1), dim("W", p?.window ?? seq.P), dim("d", d)]);
      push(c.mode === "mean" ? "mean over each page" : `attention pooling: ${c.heads} learned queries per page`, [B, dim("pages", p?.count ?? 1), dim("d", d)]);
      push("one vector per page", X(out));
      break;
    }
    case "lm_head":
      push("input", X(d));
      if (c.final_norm !== "none") push("final norm", X(d));
      push("logits", [B, ...pos, dim("V", c.vocab)], c.chunked ? "computed in chunks; never kept whole" : undefined);
      push("loss", [], "one number: mean cross-entropy against the next token");
      break;
    case "custom":
      push("input", X(d));
      push(c.note || "custom", X(out));
      break;
  }
  return steps;
}

function attentionSteps(
  g: Graph,
  a: Analysis,
  c: Extract<Cell, { type: "attention" }>,
  home: NonNullable<ReturnType<typeof locate>>,
  b: number,
  push: (label: string, dims: Dim[], note?: string) => void,
) {
  const { seq, res } = home;
  const B = dim("B", b);
  const pos = positions(seq);
  const d = res.widthIn;
  const h = dim("h", c.heads), g_ = dim("g", c.kv_heads), dh = dim("dh", c.head_dim);
  const wire = g.edges.find((e) => e.kind === "source" && e.to.cell === c.id);
  const src = wire ? a.stacks.find((s) => s.id === wire.from.stack) : undefined;
  push("input x", [B, ...pos, dim("d", d)]);
  if (c.norm !== "none") push("pre-norm", [B, ...pos, dim("d", d)]);
  if (c.mla) {
    const m = c.mla;
    if (m.q_rank > 0) push("query latent (low-rank + norm)", [B, ...pos, dim("q_rank", m.q_rank)]);
    push("queries q (no-position part + rotary part)", [B, ...pos, h, dim("nope+rope", m.nope_dim + m.rope_dim)]);
    push("keys/values latent: the only thing cached", [B, ...pos, dim("kv_rank+rope", m.kv_rank + m.rope_dim)]);
    push("keys k, values v, rebuilt from the latent", [B, ...pos, h, dim("nope+rope | v", m.nope_dim + m.rope_dim)]);
    const qPos = pos.at(-1)!;
    push("scores q·k", [B, ...pos.slice(0, -1), h, qPos, dim(qPos.n, c.window > 0 ? Math.min(c.window, qPos.v) : qPos.v)], "masked: each token sees itself and earlier tokens");
    push("softmax · v", [B, ...pos, h, dim("v", m.v_dim)]);
    push("heads joined", [B, ...pos, dim("h·v", c.heads * m.v_dim)]);
    push("output projection", [B, ...pos, dim("d", d)]);
    if (c.residual !== "none") push("⊕ added to the stream", [B, ...pos, dim("d", d)]);
    return;
  }
  push("queries q", [B, ...pos, h, dh]);

  // keys: own stream, another stack, or ready notes
  const qPos = pos.at(-1)!;
  let keys: Dim;
  if (c.mode === "cross" && src) {
    const srcPos = positions(src.seq);
    const pageRead = seq.kind === "slots" && src.seq.kind === "pages"; // a slot reads its own page: [pages, W]
    push("keys k, values v (each)", [B, ...srcPos, g_, dh], `from ${src.name}`);
    keys = pageRead ? srcPos[1] : dim(srcPos.map((p) => p.n).join("·"), srcPos.reduce((n, p) => n * p.v, 1));
    const lead = pos.slice(0, -1);
    push("scores q·k", [B, ...lead, h, qPos, keys], [pageRead ? "each slot reads only its own page" : "", c.kernel === "flash" ? "computed in tiles, never stored whole (flash)" : "stored for backward (eager)"].filter(Boolean).join("; "));
  } else {
    push("keys k, values v (each)", [B, ...pos, g_, dh]);
    const own = c.window > 0 ? Math.min(c.window, qPos.v) : qPos.v;
    let notes = 0;
    if (c.memory && src && src.seq.pages) {
      const last = g.stacks.find((s) => s.id === src.id)!.rows.flatMap((r) => r.cells).at(-1);
      const entries = last?.type === "kv_out" ? last.entries : 1;
      notes = src.seq.pages.count * (src.seq.perPage ?? 0) * entries;
      push("notes: ready keys / values", [B, dim("notes", notes), g_, dh], `from ${src.name}; each readable once its page is complete`);
    }
    keys = dim(notes ? "w+notes" : c.window > 0 ? "w" : qPos.n, own + notes);
    const lead = pos.slice(0, -1);
    push(
      "scores q·k",
      [B, ...lead, h, qPos, keys],
      [c.window > 0 ? `a band of ${c.window} recent keys per query` : c.causal ? "masked: each token sees itself and earlier tokens" : "every token sees every token",
        c.kernel === "flash" ? "computed in tiles, never stored whole (flash)" : "stored for backward (eager)"].join("; "),
    );
  }
  push("softmax · v", [B, ...pos, h, dh]);
  push("heads joined", [B, ...pos, dim("h·dh", c.heads * c.head_dim)]);
  push("output projection", [B, ...pos, dim("d", d)]);
  if (c.residual !== "none") push("⊕ added to the stream", [B, ...pos, dim("d", d)]);
}

/** The stream entering a row (token ids for the input row). */
export function rowInputShape(g: Graph, a: Analysis, rowId: string, b: number): Shape {
  for (const s of g.stacks) {
    const i = s.rows.findIndex((r) => r.id === rowId);
    if (i < 0) continue;
    const res = a.stacks.find((x) => x.id === s.id)!;
    const first = s.rows[i].cells[0];
    const pos = positions(res.seq);
    if (first?.type === "embedding") return shape([dim("B", b), ...pos], "ids");
    if (first?.type === "latent_init") return shape([], "start");
    const rr = res.rows[i];
    const prev = res.rows.slice(0, i).reverse().find((r) => r.cells.length);
    const w = rr.cells[0]?.widthIn || prev?.cells.at(-1)?.widthOut || s.width;
    return shape([dim("B", b), ...pos, dim("d", w)]);
  }
  return shape([]);
}

/** What a stack hands on: logits for a head, notes for a notes stack, else its stream. */
export function stackOutputShape(g: Graph, a: Analysis, stackId: string, b: number): Shape {
  const s = g.stacks.find((x) => x.id === stackId);
  const last = s?.rows.flatMap((r) => r.cells).at(-1);
  if (!s || !last) return shape([]);
  const steps = cellTensorSteps(g, a, last.id, b).filter((x) => x.dims.length);
  return shape(steps.at(-1)?.dims ?? []);
}

const short = (v: number) => (v >= 1 << 20 && v % (1 << 18) === 0 ? `${v / (1 << 20)}M` : v >= 1024 && v % 512 === 0 ? `${v / 1024}k` : String(v));

/** compact: "B×4k×576" · numbers: "4 × 4,096 × 576" · named: "[B 4, S 4,096, d 576]" */
export function fmtShape(s: Dim[] & { kind?: string }, mode: "compact" | "numbers" | "named"): string {
  if (s.kind === "start") return "learned queries";
  let body: string;
  if (mode === "compact") body = s.map((x) => (x.n === "B" ? "B" : short(x.v))).join("×");
  else if (mode === "numbers") body = s.map((x) => x.v.toLocaleString("en-US")).join(" × ");
  else body = `[${s.map((x) => `${x.n} ${x.v.toLocaleString("en-US")}`).join(", ")}]`;
  return s.kind === "ids" ? `ids ${body}` : body;
}

/** Elements in a tensor. */
export const elements = (s: Dim[]) => s.reduce((n, x) => n * x.v, 1);

// ------------------------------------------------------------------ wires: what they carry and what happens at the other end

export interface EdgeTensors {
  from: { label: string; dims: Dim[] };
  to: { label: string; dims: Dim[] };
  steps: TensorStep[];
  params: number; // parameters the wire itself adds (a skip's weight)
  summary: string;
}

const stackName = (g: Graph, id: string) => g.stacks.find((s) => s.id === id)?.name ?? id;
const rowLabel = (g: Graph, id: string) => {
  for (const s of g.stacks) {
    const i = s.rows.findIndex((r) => r.id === id);
    if (i >= 0) return `${s.name} · ${s.rows[i].label ?? `row ${i}`}`;
  }
  return id;
};
const cellName = (c: Cell) => c.label ?? c.type.replace("_", " ");

/** The tensor at the start of a wire. */
function endpointDims(g: Graph, a: Analysis, e: Edge["from"], b: number): Dim[] {
  const res = a.stacks.find((s) => s.id === e.stack);
  if (!res) return [];
  const B = dim("B", b);
  if (e.port === "table" && e.cell) {
    const cell = g.stacks.find((s) => s.id === e.stack)?.rows.flatMap((r) => r.cells).find((c) => c.id === e.cell);
    return cell?.type === "embedding" ? [B, ...positions(res.seq), dim("e", cell.dim)] : [];
  }
  if (e.row) {
    const rr = res.rows.find((r) => r.id === e.row);
    const w = rr?.cells.at(-1)?.widthOut;
    return w ? [B, ...positions(res.seq), dim("d", w)] : [];
  }
  return stackOutputShape(g, a, e.stack, b);
}

export function edgeTensors(g: Graph, a: Analysis, edgeId: string, b: number): EdgeTensors {
  const e = g.edges.find((x) => x.id === edgeId);
  const empty: EdgeTensors = { from: { label: "", dims: [] }, to: { label: "", dims: [] }, steps: [], params: 0, summary: "" };
  if (!e) return empty;
  const fromDims = endpointDims(g, a, e.from, b);
  const fromLabel = e.from.port === "table" ? `${stackName(g, e.from.stack)} token table` : e.from.row ? `${rowLabel(g, e.from.row)} output` : `${stackName(g, e.from.stack)} output`;
  const steps: TensorStep[] = [{ label: fromLabel, dims: fromDims }];

  if (e.kind === "skip") {
    const to = e.to.row ? rowInputShape(g, a, e.to.row, b) : shape([]);
    const width = fromDims.at(-1)?.v ?? 0;
    const params = e.weight === "scalar" ? 1 : e.weight === "vector" ? width : 0;
    steps.push({ label: e.weight === "scalar" ? "× σ, one learned number (starts at 0)" : e.weight === "vector" ? "× w, one learned weight per channel" : "as is", dims: fromDims });
    steps.push({ label: `added to the input of ${rowLabel(g, e.to.row ?? "")}`, dims: [...to] });
    return { from: { label: fromLabel, dims: fromDims }, to: { label: `${rowLabel(g, e.to.row ?? "")} input`, dims: [...to] }, steps, params,
      summary: "A U-net skip: an earlier row's output, scaled, is added to a later row's input; both are the same shape." };
  }

  if (e.kind === "input") {
    const target = g.stacks.find((s) => s.id === e.to.stack);
    const tres = a.stacks.find((s) => s.id === e.to.stack);
    if (!target || !tres) return empty;
    const last = fromDims.at(-1) ?? dim("d", 0);
    if (tres.seq.kind === "pages" && tres.seq.pages) {
      const p = tres.seq.pages;
      steps.push({ label: `cut into pages: ${p.count} pages of ${p.window} tokens, a new page every ${p.stride}`, dims: [dim("B", b), dim("pages", p.count), dim("W", p.window), last],
        note: p.window > p.stride ? `pages overlap: ${p.count * p.window} page tokens for ${tres.seq.tokensOf ? a.stacks.find((s) => s.id === tres.seq.tokensOf)?.P : "the"} row tokens` : undefined });
    } else steps.push({ label: `starts ${target.name}`, dims: fromDims });
    const first = target.rows.flatMap((r) => r.cells)[0];
    let toDims: Dim[] = steps.at(-1)!.dims;
    if (first) {
      const inner = cellTensorSteps(g, a, first.id, b).filter((s) => s.dims.length);
      for (const s of inner.slice(1)) steps.push({ ...s, label: `${cellName(first)}: ${s.label}` });
      toDims = inner.at(-1)?.dims ?? toDims;
    }
    return { from: { label: fromLabel, dims: fromDims }, to: { label: `${target.name} (its stream starts here)`, dims: toDims }, steps, params: 0,
      summary: `${target.name} starts from this tensor${tres.seq.kind === "pages" ? ", cut into overlapping pages" : ""}.` };
  }

  // source: keys/values for a cross-attention, or ready notes for a notebook read
  const home = g.stacks.find((s) => s.rows.some((r) => r.cells.some((c) => c.id === e.to.cell)));
  const cell = home?.rows.flatMap((r) => r.cells).find((c) => c.id === e.to.cell);
  if (!home || !cell) return empty;
  const inner = cellTensorSteps(g, a, cell.id, b);
  for (const s of inner) if (/^(queries|keys|notes|scores|softmax)/.test(s.label)) steps.push(s);
  const notes = cell.type === "attention" && cell.mode === "self" && cell.memory;
  const toDims = inner.filter((s) => s.dims.length).at(-1)?.dims ?? [];
  return {
    from: { label: fromLabel, dims: fromDims },
    to: { label: `${home.name} · ${cellName(cell)}`, dims: toDims },
    steps,
    params: 0,
    summary: notes
      ? "Ready notes: the read uses them directly as extra keys/values (no projection), next to its recent-token window; each note becomes readable once its page is complete."
      : "Cross-attention: this cell's queries come from its own stream; the wired tensor is projected into keys and values.",
  };
}

// Plain-word names for cells, stacks and wires.
import type { SeqInfo } from "../engine/cells";
import type { Cell, CellType, Edge, Graph, Stack } from "../engine/types";
import { len } from "./format";

export type Category = "input" | "attn" | "ffn" | "mem" | "out" | "other";

export const CATEGORY: Record<CellType, Category> = {
  embedding: "input",
  positional: "input",
  attention: "attn",
  ffn: "ffn",
  linear: "mem",
  latent_init: "mem",
  kv_out: "mem",
  norm: "out",
  lm_head: "out",
  custom: "other",
};

export const CATEGORY_LABEL: Record<Category, string> = {
  input: "Input",
  attn: "Attention",
  ffn: "Feed-forward",
  mem: "Memory & projections",
  out: "Norm & output",
  other: "Other",
};

const k = (n: number) => (n >= 1000 && n % 1000 === 0 ? `${n / 1000}k` : n >= 1024 && n % 1024 === 0 ? `${n / 1024}Ki` : String(n));

export function cellTitle(c: Cell): string {
  switch (c.type) {
    case "embedding":
      return c.share_with ? "Embedding (shared)" : "Embedding";
    case "positional":
      return { rope: "Rotary positions", learned: "Learned positions", sinusoidal: "Sinusoidal positions", alibi: "ALiBi", none: "No positions" }[c.kind];
    case "attention":
      if (c.mode === "cross") return "Cross-attention";
      if (c.memory) return "Notebook read";
      if (c.window > 0) return `Local attention`;
      return c.causal ? "Full attention" : "Full attention ↔";
    case "ffn":
      return c.experts > 1 ? "Mixture of experts" : c.kind === "swiglu" ? "SwiGLU" : c.kind === "geglu" ? "GeGLU" : `MLP (${c.activation.toUpperCase()})`;
    case "norm":
      return c.kind === "rmsnorm" ? "RMSNorm" : c.kind === "layernorm" ? "LayerNorm" : "Identity";
    case "linear":
      return "Linear";
    case "latent_init":
      return "Learned queries";
    case "kv_out":
      return "Notes out";
    case "lm_head":
      return "LM head + loss";
    case "custom":
      return c.note || "Custom";
  }
}

export function cellSub(c: Cell, widthIn: number): string {
  switch (c.type) {
    case "embedding":
      return `${k(c.vocab)} × ${c.dim}${c.ngram_tables ? ` + ${c.ngram_tables} n-gram` : ""}${c.proj === "gated_mlp" ? " → MLP" : c.proj === "linear" ? " → proj" : ""}`;
    case "positional":
      return c.kind === "rope" ? `θ ${k(c.theta)}` : c.kind === "learned" ? `${len(c.max_len)} positions` : "";
    case "attention": {
      const kv = c.kv_heads !== c.heads ? ` · kv ${c.kv_heads}` : "";
      const win = c.window > 0 ? ` · w ${len(c.window)}` : "";
      return `${c.heads} × ${c.head_dim}${kv}${win}`;
    }
    case "ffn":
      return c.experts > 1 ? `${c.experts} × ${c.hidden} · top ${c.top_k}` : `${widthIn} → ${c.hidden}`;
    case "norm":
      return String(widthIn);
    case "linear":
      return `${widthIn} → ${c.out || widthIn}`;
    case "latent_init":
      return c.address_feats ? `+ address ${c.address_feats}` : "";
    case "kv_out":
      return `${c.entries} × ${c.kv_heads}×${c.head_dim} · ${c.projections} maps`;
    case "lm_head":
      return `${k(c.vocab)}${c.tied ? " · tied" : ""}${c.z_loss ? " · z-loss" : ""}`;
    case "custom":
      return "";
  }
}

export function seqSummary(s: Stack, seq: SeqInfo | undefined, g: Graph): string {
  const name = (id: string) => g.stacks.find((x) => x.id === id)?.name ?? id;
  switch (s.seq.kind) {
    case "tokens":
      return `${len(seq?.P ?? 0)} tokens`;
    case "source":
      return `${len(seq?.P ?? 0)} source tokens`;
    case "latents":
      return `${s.seq.count} latents`;
    case "pages":
      return `${seq?.pages?.count ?? "?"} pages × ${seq?.pages?.window ?? s.seq.window} of ${name(s.seq.of)}`;
    case "slots":
      return `${seq?.pages?.count ?? "?"} pages × ${s.seq.per_page} slots`;
  }
}

export function edgeLabel(e: Edge, g: Graph): string {
  const stack = (id: string) => g.stacks.find((s) => s.id === id)?.name ?? id;
  if (e.kind === "skip") return "skip (U-net) wire";
  if (e.kind === "input") return `${stack(e.from.stack)} → starts ${stack(e.to.stack)}`;
  return `${stack(e.from.stack)} → keys/values`;
}

// ------------------------------------------------------------------ connections: how a cell mixes tokens, what goes in and out

export type MixKind = "embed" | "causal" | "local" | "twoway" | "page" | "cross" | "notes" | "point" | "start" | "head" | "custom";

const stackName = (g: Graph, id: string) => g.stacks.find((s) => s.id === id)?.name ?? id;
const rowName = (g: Graph, id: string) => {
  for (const s of g.stacks) {
    const i = s.rows.findIndex((r) => r.id === id);
    if (i >= 0) return s.rows[i].label ?? `row ${i}`;
  }
  return id;
};
const endName = (g: Graph, e: Edge["from"]) =>
  e.port === "table" ? `${stackName(g, e.stack)} token table` : e.row ? `${stackName(g, e.stack)} · ${rowName(g, e.row)}` : `${stackName(g, e.stack)} output`;

/** How a cell connects the positions of its stream: which tokens each token reads. */
export function mixing(c: Cell, s: Stack, g: Graph): { kind: MixKind; text: string } {
  const src = g.edges.find((e) => e.kind === "source" && e.to.cell === c.id);
  const inPages = s.seq.kind === "pages" || s.seq.kind === "slots";
  switch (c.type) {
    case "embedding":
      return { kind: "embed", text: `token ids → vectors: one table lookup per token${c.share_with ? " (shared table)" : ""}` };
    case "positional":
      return { kind: "point", text: c.kind === "rope" ? "positions: rotary angles applied inside the attention that follows" : c.kind === "learned" ? "adds a learned position vector to each token" : `positions: ${c.kind}` };
    case "attention":
      if (c.mode === "cross")
        return { kind: "cross", text: `cross-attention: queries from this stream, keys/values from ${src ? endName(g, src.from) : "a wired stack (none yet)"}` };
      if (c.memory)
        return { kind: "notes", text: `notebook read: the last ${c.window || "all"} tokens plus the notes from ${src ? stackName(g, src.from.stack) : "a notes stack (none yet)"}, each note readable once its page is complete` };
      if (inPages) return { kind: "page", text: `self-attention inside each page${c.causal ? " (causal)" : " (two-way)"}: pages do not see each other` };
      if (c.window > 0) return { kind: "local", text: `local self-attention: each token reads the ${c.window} most recent tokens${c.causal ? "" : " around it"}` };
      return c.causal ? { kind: "causal", text: "causal self-attention: each token reads itself and every earlier token" } : { kind: "twoway", text: "two-way self-attention: each token reads every token" };
    case "ffn":
      return { kind: "point", text: c.experts > 1 ? `per token, no mixing: each token goes to ${c.top_k} of ${c.experts} experts` : "per token, no mixing: the same MLP on every token on its own" };
    case "norm":
    case "linear":
      return { kind: "point", text: `per token, no mixing (${c.type === "norm" ? "normalises" : "projects"} each token on its own)` };
    case "latent_init":
      return { kind: "start", text: "starts the stream with learned query vectors (one per slot / latent)" };
    case "kv_out":
      return { kind: "point", text: "per slot: turns each slot into ready keys/values (notes) for a reader" };
    case "lm_head":
      return { kind: "head", text: `per token → logits over ${c.vocab} tokens; the loss compares them with the next token` };
    case "custom":
      return { kind: "custom", text: "custom block: its mixing is not modelled" };
  }
}

export const MIX_LEGEND: [MixKind, string][] = [
  ["causal", "self-attention, every earlier token"],
  ["local", "self-attention, a recent window"],
  ["twoway", "self-attention, every token (two-way)"],
  ["page", "attention inside each page"],
  ["notes", "notebook read: window + notes"],
  ["cross", "cross-attention: keys/values from another stack"],
  ["point", "per token, no mixing (feed-forward, norm, projection)"],
  ["embed", "token ids → vectors"],
  ["head", "vectors → logits → loss"],
];

/** What starts a stack's stream and where its output goes. */
export function stackIO(s: Stack, g: Graph): { input: string; outputs: string[] } {
  const first = s.rows.flatMap((r) => r.cells)[0];
  const wire = g.edges.find((e) => e.kind === "input" && e.to.stack === s.id);
  let input = "nothing yet: add an embedding or wire an input";
  if (wire) input = endName(g, wire.from);
  else if (first?.type === "embedding") input = `token ids (vocabulary ${first.vocab})`;
  else if (first?.type === "latent_init") input = "learned queries";
  const outputs: string[] = [];
  const head = s.rows.flatMap((r) => r.cells).find((c) => c.type === "lm_head");
  if (head && head.type === "lm_head") outputs.push(`logits (${head.vocab}) → next-token loss`);
  const notes = s.rows.flatMap((r) => r.cells).at(-1)?.type === "kv_out";
  const counts = new Map<string, number>();
  for (const e of g.edges) {
    if (e.from.stack !== s.id || e.kind === "skip") continue;
    let text: string;
    if (e.kind === "input") text = `${e.from.port === "table" ? "token table" : "stream"} → starts ${stackName(g, e.to.stack)}`;
    else text = `${notes ? "notes" : "keys/values"} → ${stackName(g, e.to.stack)} · ${notes ? "notebook read" : "cross-attention"}`;
    counts.set(text, (counts.get(text) ?? 0) + 1);
  }
  for (const [t, n] of counts) outputs.push(n > 1 ? `${t} ×${n}` : t);
  return { input, outputs };
}

/** In words: what feeds a cell, how it mixes tokens, and where its output goes. */
export function cellConnections(g: Graph, cellId: string, widthIn: number): [string, string][] {
  for (const s of g.stacks)
    for (let ri = 0; ri < s.rows.length; ri++) {
      const row = s.rows[ri];
      const ci = row.cells.findIndex((c) => c.id === cellId);
      if (ci < 0) continue;
      const c = row.cells[ci];
      const prevRow = s.rows.slice(0, ri).reverse().find((r) => r.cells.length > 0);
      const nextRow = s.rows.slice(ri + 1).find((r) => r.cells.length > 0);
      let input: string;
      if (ci > 0) input = `${cellTitle(row.cells[ci - 1])}, same row (width ${widthIn})`;
      else if (prevRow) input = `the stream after ${prevRow.label ?? "the row above"} (width ${widthIn})`;
      else input = stackIO(s, g).input;
      if (c.type === "embedding" || c.type === "latent_init") input = c.type === "embedding" ? "token ids" : "nothing: it starts the stream";
      const out: [string, string][] = [["Input", input], ["Mixes tokens", mixing(c, s, g).text]];
      const src = g.edges.find((e) => e.kind === "source" && e.to.cell === c.id);
      if (src) out.push([c.type === "attention" && c.memory ? "Notes from" : "Keys/values from", endName(g, src.from)]);
      const resid = (c.type === "attention" || c.type === "ffn") && c.residual !== "none";
      const next = ci < row.cells.length - 1 ? `${cellTitle(row.cells[ci + 1])}, same row` : nextRow ? `the next row (${nextRow.label ?? "below"})` : stackIO(s, g).outputs.join("; ") || `${s.name} output`;
      out.push(["Output", `${resid ? "added back to the stream (residual ⊕)" : c.type === "lm_head" ? "logits" : "replaces the stream"} → ${next}`]);
      const feeds = g.edges.filter((e) => e.from.cell === c.id && e.kind !== "skip");
      for (const e of feeds) out.push(["Also feeds", `${e.from.port === "table" ? "token table" : "output"} → ${stackName(g, e.to.stack)}`]);
      return out;
    }
  return [];
}

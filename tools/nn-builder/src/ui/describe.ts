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

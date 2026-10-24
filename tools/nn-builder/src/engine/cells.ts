// Per-cell costs: exact parameters, forward FLOPs per row, and the bytes each cell saves for backward.
// FLOPs count the multiply-adds (2 FLOP each) of every projection, attention score and value read;
// norms, activations and RoPE are not counted. A row is one training sample.
import { avgCausalKeys, avgVisibleNotes, type PageGeom } from "./geometry";
import type { AttentionCell, Cell, EmbeddingCell, FfnCell, NormKind, SeqSpec } from "./types";

export interface SeqInfo {
  kind: SeqSpec["kind"];
  P: number; // positions per row
  pages?: PageGeom; // pages stacks: their pages; slots stacks: the pages the slots sit on
  pageWindowCfg?: number; // the configured page width (parameter shapes never depend on the row length)
  perPage?: number; // slots stacks: slots per page
  tokensOf?: string; // pages / slots stacks: the token stack the pages cover
  pagesStack?: string; // slots stacks: their pages stack
}

export interface NotesInfo {
  total: number; // notes in one row
  perPage: number; // notes per page (slots × entries)
  pages: PageGeom;
  tokensOf?: string;
  kvHeads: number;
  headDim: number;
}

export interface SourceInfo {
  stackId: string;
  width: number;
  P: number;
  seq: SeqInfo;
  notes?: NotesInfo; // the source ends in a kv_out cell: it hands over ready keys/values
}

export interface Bytes {
  act: number; // a saved activation element
  resid: number; // a residual-stream / norm-input element
}

export interface CellCtx {
  stackId: string;
  width: number; // the stream width entering the cell
  stackWidth: number;
  S: number;
  seq: SeqInfo;
  source?: SourceInfo; // cross-attention keys/values or the notes of a memory read
  embedding?: { cell: EmbeddingCell; shared: boolean }; // the stack's embedding, for a tied head
  bytes: Bytes;
}

export interface Part {
  name: string;
  params: number;
}
export interface FlopTerm {
  name: string;
  fwd: number; // forward FLOPs per row, one execution
  bwd: number; // backward multiplier: 2 = trainable weights, 1 = frozen weights (input grads only), 0 = no grad
  keys?: boolean; // grows with the number of keys each query sees (attention scores and reads)
}
export interface CellCost {
  parts: Part[];
  params: number;
  flops: FlopTerm[];
  act: number; // bytes per row saved for backward, one execution
  workspace: number; // transient bytes per row
  widthOut: number;
  irregular: boolean; // masked / gathered work that runs slower than plain matmuls
  issues: string[];
}

export const normParams = (kind: NormKind, d: number) => (kind === "rmsnorm" ? d : kind === "layernorm" ? 2 * d : 0);

const sum = (parts: Part[]) => parts.reduce((n, p) => n + p.params, 0);
const nonzero = (parts: Part[]) => parts.filter((p) => p.params !== 0);

function cost(c: Partial<CellCost> & { widthOut: number }): CellCost {
  const parts = nonzero(c.parts ?? []);
  return {
    parts,
    params: sum(parts),
    flops: (c.flops ?? []).filter((f) => f.fwd !== 0),
    act: c.act ?? 0,
    workspace: c.workspace ?? 0,
    widthOut: c.widthOut,
    irregular: c.irregular ?? false,
    issues: c.issues ?? [],
  };
}

/** Bytes a pre-norm (or post-norm) saves per position: the norm's input and the low-precision copy a linear reads. */
const normActs = (kind: NormKind, d: number, b: Bytes) => (kind === "none" ? b.act * d : (b.resid + b.act) * d);

export function cellCost(cell: Cell, ctx: CellCtx): CellCost {
  switch (cell.type) {
    case "embedding":
      return embeddingCost(cell, ctx);
    case "positional": {
      const learned = cell.kind === "learned";
      return cost({
        widthOut: ctx.width,
        parts: [{ name: "position table", params: learned ? cell.max_len * ctx.width : 0 }],
        issues: learned && ctx.seq.P > cell.max_len ? [`learned positions cover ${cell.max_len} tokens but the row has ${ctx.seq.P}`] : [],
      });
    }
    case "attention":
      return attentionCost(cell, ctx);
    case "ffn":
      return ffnCost(cell, ctx);
    case "norm":
      return cost({ widthOut: ctx.width, parts: [{ name: cell.kind, params: normParams(cell.kind, ctx.width) }], act: ctx.bytes.resid * ctx.width * ctx.seq.P });
    case "linear": {
      const out = cell.out || ctx.width;
      return cost({
        widthOut: out,
        parts: [
          { name: "weight", params: ctx.width * out },
          { name: "bias", params: cell.bias ? out : 0 },
          { name: "out norm", params: normParams(cell.out_norm, out) },
        ],
        flops: [{ name: "projection", fwd: 2 * ctx.width * out * ctx.seq.P, bwd: 2 }],
        act: (ctx.bytes.act * ctx.width + (cell.out_norm === "none" ? 0 : ctx.bytes.resid * out)) * ctx.seq.P,
      });
    }
    case "latent_init": {
      const d = ctx.stackWidth;
      const n = ctx.seq.kind === "slots" ? ctx.seq.perPage ?? 0 : ctx.seq.kind === "latents" ? ctx.seq.P : 0;
      return cost({
        widthOut: d,
        parts: [
          { name: "learned queries", params: n * d },
          { name: "address map", params: cell.address_feats * d },
          { name: "norm", params: normParams(cell.norm, d) },
        ],
        flops: [{ name: "address map", fwd: 2 * cell.address_feats * d * ctx.seq.P, bwd: 2 }],
        act: ctx.bytes.resid * d * ctx.seq.P,
        issues: n === 0 ? ["learned queries need a slots or latents stack"] : [],
      });
    }
    case "kv_out": {
      const d = ctx.width;
      const out = cell.entries * cell.kv_heads * cell.head_dim;
      return cost({
        widthOut: d,
        parts: [
          { name: "norms", params: cell.in_norms * d },
          { name: `${cell.projections} projections`, params: cell.projections * d * out },
          { name: "key norm", params: cell.key_norm ? cell.head_dim : 0 },
        ],
        flops: [{ name: "projections", fwd: 2 * cell.projections * d * out * ctx.seq.P, bwd: 2 }],
        act: cell.in_norms * (ctx.bytes.resid + ctx.bytes.act) * d * ctx.seq.P,
        irregular: true,
      });
    }
    case "lm_head": {
      const d = ctx.width;
      const P = ctx.seq.P;
      const emb = ctx.embedding?.cell;
      const issues: string[] = [];
      if (cell.tied && !emb) issues.push("tied head without an embedding in this stack");
      if (cell.tied && emb && (emb.dim !== d || emb.vocab !== cell.vocab)) issues.push("tied head needs the embedding table to be vocab × width");
      return cost({
        widthOut: d,
        parts: [
          { name: "final norm", params: normParams(cell.final_norm, d) },
          { name: "output matrix", params: cell.tied ? 0 : cell.vocab * d },
          { name: "bias", params: cell.bias ? cell.vocab : 0 },
        ],
        flops: [{ name: "logits", fwd: 2 * d * cell.vocab * P, bwd: 2 }],
        act: (normActs(cell.final_norm, d, ctx.bytes) + (cell.chunked ? 0 : 8 * cell.vocab)) * P,
        workspace: (cell.chunked ? Math.min(P, 1024) * 3 : P) * cell.vocab * 4,
        issues,
      });
    }
    case "custom":
      return cost({
        widthOut: cell.out || ctx.width,
        parts: [{ name: cell.note || "custom", params: cell.params }],
        flops: [{ name: cell.note || "custom", fwd: cell.flops_per_position * ctx.seq.P, bwd: 2 }],
        act: cell.act_bytes_per_position * ctx.seq.P,
      });
  }
}

function embeddingCost(cell: EmbeddingCell, ctx: CellCtx): CellCost {
  const e = cell.dim;
  const d = cell.proj === "none" ? e : ctx.stackWidth;
  const P = ctx.seq.P;
  const b = ctx.bytes;
  const proj = cell.proj === "linear" ? e * d : cell.proj === "gated_mlp" ? 2 * e * d + d * d : 0;
  const projFlops = cell.proj === "linear" ? 2 * e * d : cell.proj === "gated_mlp" ? 2 * (2 * e * d + d * d) : 0;
  const projActs = cell.proj === "linear" ? b.act * e : cell.proj === "gated_mlp" ? b.act * (e + 2 * d) : 0;
  return cost({
    widthOut: d,
    parts: [
      { name: "token table", params: cell.share_with ? 0 : cell.vocab * e },
      { name: "n-gram tables", params: cell.ngram_tables * cell.ngram_buckets * e },
      { name: cell.proj === "gated_mlp" ? "input MLP" : "projection", params: proj },
      { name: "norm", params: normParams(cell.out_norm, d) },
    ],
    flops: [{ name: "input MLP", fwd: projFlops * P, bwd: 2 }],
    act: (b.resid * e + projActs + (cell.out_norm === "none" ? 0 : b.resid * d)) * P,
  });
}

/** Embedding projection FLOPs per position, reused by the loop's soft feedback (re-embedding the head's reading). */
export function embeddingProjFlops(cell: EmbeddingCell, d: number): number {
  const e = cell.dim;
  return cell.proj === "linear" ? 2 * e * d : cell.proj === "gated_mlp" ? 2 * (2 * e * d + d * d) : 0;
}

function selfKeys(cell: AttentionCell, seq: SeqInfo): number {
  const span = (n: number) => (cell.causal ? avgCausalKeys(n, cell.window) : cell.window > 0 ? Math.min(cell.window, n) : n);
  if (seq.kind === "pages" && seq.pages) return seq.pages.count * seq.pages.window * span(seq.pages.window);
  if (seq.kind === "slots" && seq.pages && seq.perPage) return seq.pages.count * seq.perPage * span(seq.perPage);
  return seq.P * span(seq.P);
}

function attentionCost(cell: AttentionCell, ctx: CellCtx): CellCost {
  const d = ctx.width;
  const { heads: h, head_dim: dh, kv_heads: g } = cell;
  const P = ctx.seq.P;
  const b = ctx.bytes;
  const src = ctx.source;
  const issues: string[] = [];
  if (g < 1 || h % g !== 0) issues.push(`heads (${h}) must be a multiple of key/value heads (${g})`);

  const cross = cell.mode === "cross";
  const notes = !cross && cell.memory ? src?.notes : undefined;
  if (cross && !src) issues.push("cross-attention has no source: drag a wire from the stack it reads");
  if (cross && src?.notes) issues.push("the source hands over ready notes: read them with a memory read on self-attention");
  if (!cross && cell.memory && !notes) issues.push("memory read has no notes: drag a wire from a notes stack (one that ends in a kv_out cell)");
  if (notes && (notes.kvHeads !== g || notes.headDim !== dh)) issues.push(`notes are ${notes.kvHeads} × ${notes.headDim} but this read uses ${g} × ${dh} key/value heads`);

  const dSrc = cross ? src?.width ?? d : d;
  const Psrc = cross ? src?.P ?? 0 : P;

  // keys seen, summed over all queries of the row
  let keys = 0;
  let pageRead = false;
  if (!cross) keys = selfKeys(cell, ctx.seq);
  else if (src) {
    pageRead = ctx.seq.kind === "slots" && src.seq.kind === "pages" && ctx.seq.pagesStack === src.stackId;
    keys = pageRead && ctx.seq.pages ? ctx.seq.pages.count * (ctx.seq.perPage ?? 0) * ctx.seq.pages.window : P * Psrc;
  }
  let visibleNotes = 0;
  if (notes) {
    const sameRow = notes.tokensOf === ctx.stackId && ctx.seq.kind === "tokens";
    if (!sameRow) issues.push("notes were written over another stack's tokens: every note counts as visible");
    visibleNotes = sameRow ? avgVisibleNotes(P, notes.pages, notes.perPage) : notes.total;
  }
  const scoreCoef = notes && cell.dual_key ? 6 : 4;
  const keysPerQuery = P ? (keys + P * visibleNotes) / P : 0;

  const parts: Part[] = [
    { name: "pre-norm", params: normParams(cell.norm, d) },
    { name: "query", params: d * h * dh + (cell.bias ? h * dh : 0) },
    { name: "key/value", params: 2 * dSrc * g * dh + (cell.bias ? 2 * g * dh : 0) },
    { name: "output", params: h * dh * d + (cell.bias ? d : 0) },
    { name: "q/k norm", params: cell.qk_norm ? 2 * dh : 0 },
    { name: "source norm", params: cross ? normParams(cell.kv_norm, dSrc) : 0 },
    { name: "position prior", params: cell.extras.pos_prior ? (ctx.seq.perPage ?? 0) * h * (ctx.seq.pageWindowCfg ?? 0) : 0 },
    { name: "logit scale", params: cell.extras.logit_scale ? 1 : 0 },
  ];
  if (cell.extras.pos_prior && !pageRead) issues.push("a position prior needs a slots stack reading its own pages");

  const flops: FlopTerm[] = [
    { name: "q / out projections", fwd: 2 * 2 * d * h * dh * P, bwd: 2 },
    { name: "k / v projections", fwd: 2 * 2 * dSrc * g * dh * Psrc, bwd: 2 },
    { name: "scores + value reads", fwd: scoreCoef * h * dh * (keys + P * visibleNotes), bwd: 2, keys: true },
  ];

  const nc = cell.extras.count_heads;
  if (notes && cell.extras.order_term) {
    parts.push({ name: "order term", params: h + d + 1 });
    flops.push({ name: "order term (rank count, no grad)", fwd: 2 * h * dh * notes.total * P, bwd: 0, keys: true });
  }
  if (notes && nc > 0) {
    const km = cell.extras.count_max;
    parts.push({ name: "count heads", params: 2 * nc + d * nc * dh + dh + nc * (km + 2) * d });
    flops.push({ name: "count heads", fwd: (2 * nc * dh * (notes.total + Math.max(cell.window, 0)) + 2 * d * nc * dh + 2 * nc * (km + 2) * d) * P, bwd: 2, keys: true });
  }
  if (!notes && (cell.extras.order_term || nc > 0)) issues.push("order term and count heads only act on a memory read");

  // saved for backward
  const kvPos = cross ? Psrc : P;
  let act =
    (normActs(cell.norm, d, b) + b.act * (2 * h * dh + (cell.qk_norm ? h * dh : 0)) + 4 * h) * P +
    b.act * g * dh * (2 + (cell.qk_norm ? 1 : 0)) * kvPos +
    (cross && cell.kv_norm !== "none" ? (b.resid + b.act) * dSrc * Psrc : 0);
  if (cell.kernel === "eager") act += 4 * h * keysPerQuery * P;
  if (notes && nc > 0) act += 8 * nc * (notes.total + Math.max(cell.window, 0)) * P;

  return cost({
    widthOut: d,
    parts,
    flops,
    act,
    irregular: Boolean(notes) || cell.kernel === "eager" || ctx.seq.kind === "pages" || ctx.seq.kind === "slots",
    issues,
  });
}

function ffnCost(cell: FfnCell, ctx: CellCtx): CellCost {
  const d = ctx.width;
  const F = cell.hidden;
  const P = ctx.seq.P;
  const b = ctx.bytes;
  const gated = cell.kind !== "mlp";
  const E = Math.max(1, cell.experts);
  const k = E > 1 ? Math.min(Math.max(1, cell.top_k), E) : 1;
  const perExpert = (gated ? 3 * d * F : 2 * d * F) + (cell.bias ? (gated ? 2 * F : F) + d : 0);
  return cost({
    widthOut: d,
    parts: [
      { name: "pre-norm", params: normParams(cell.norm, d) },
      { name: E > 1 ? `${E} experts` : gated ? `${cell.kind} (3 matrices)` : "MLP (2 matrices)", params: E * perExpert },
      { name: "router", params: E > 1 ? d * E : 0 },
    ],
    flops: [
      { name: E > 1 ? `${k} active experts` : "feed-forward", fwd: k * (gated ? 6 : 4) * d * F * P, bwd: 2 },
      { name: "router", fwd: E > 1 ? 2 * d * E * P : 0, bwd: 2 },
    ],
    act: (normActs(cell.norm, d, b) + k * b.act * (gated ? 4 : 2) * F) * P,
    irregular: E > 1,
  });
}

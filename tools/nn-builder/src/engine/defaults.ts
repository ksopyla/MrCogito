// Default settings for every cell type, the palette, and the default run setup.
import type { AttentionCell, AttentionExtras, Cell, CellBody, CellType, RunSetup } from "./types";

export const DEFAULT_RUN: RunSetup = {
  row_length: 4096,
  rows_per_step: 96,
  micro_batch: 0,
  precision: "amp_bf16",
  optimizer: "adamw",
  weight_decay: 0.1,
  zero_stage: 0,
  checkpointing: false,
  gpu: "rtx3090",
  gpu_count: 3,
  token_budget: 1.5e9,
  deadline_hours: 0,
  mfu: 0,
  irregular_ratio: 0,
  dp_efficiency: 0.95,
};

export const NO_EXTRAS: AttentionExtras = { order_term: false, count_heads: 0, count_max: 10, pos_prior: false, logit_scale: false };

let counter = 0;
export function newId(prefix: string): string {
  counter += 1;
  return `${prefix}_${Date.now().toString(36).slice(-4)}${counter.toString(36)}${Math.random().toString(36).slice(2, 5)}`;
}

export const round64 = (x: number) => Math.max(64, Math.round(x / 64) * 64);

export function attention(width: number, o: Partial<AttentionCell> = {}): AttentionCell {
  const head_dim = o.head_dim ?? 64;
  const heads = o.heads ?? Math.max(1, Math.round(width / head_dim));
  return {
    type: "attention",
    mode: "self",
    heads,
    head_dim,
    kv_heads: heads,
    causal: true,
    window: 0,
    qk_norm: true,
    bias: false,
    norm: "rmsnorm",
    kv_norm: "rmsnorm",
    residual: "pre",
    memory: false,
    dual_key: false,
    nope: false,
    kernel: "flash",
    ...o,
    extras: { ...NO_EXTRAS, ...(o.extras ?? {}) },
  };
}

export function defaultBody(type: CellType, width: number): CellBody {
  switch (type) {
    case "embedding":
      return { type, vocab: 4096, dim: width, ngram_tables: 0, ngram_buckets: 8192, proj: "none", out_norm: "none" };
    case "positional":
      return { type, kind: "rope", theta: 500000, max_len: 4096 };
    case "attention":
      return attention(width);
    case "ffn":
      return { type, kind: "swiglu", hidden: round64((width * 8) / 3), activation: "gelu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" };
    case "norm":
      return { type, kind: "rmsnorm" };
    case "linear":
      return { type, out: width, bias: false, out_norm: "none" };
    case "latent_init":
      return { type, address_feats: 0, norm: "rmsnorm" };
    case "kv_out":
      return { type, entries: 1, kv_heads: 1, head_dim: 64, projections: 2, in_norms: 1, key_norm: true };
    case "lm_head":
      return { type, vocab: 4096, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 1e-4, softcap: 30, chunked: true };
    case "custom":
      return { type, params: 0, flops_per_position: 0, act_bytes_per_position: 0, out: 0, note: "custom block" };
  }
}

export interface PaletteItem {
  key: string;
  label: string;
  group: string;
  hint: string;
  make: (width: number) => CellBody;
}

export const PALETTE: PaletteItem[] = [
  { key: "embedding", group: "Input", label: "Token embedding", hint: "token table (+ n-gram tables, input MLP)", make: (w) => defaultBody("embedding", w) },
  { key: "rope", group: "Input", label: "Rotary positions", hint: "RoPE for the attention after it (0 params)", make: (w) => defaultBody("positional", w) },
  { key: "learned_pos", group: "Input", label: "Learned positions", hint: "a position table added to the stream", make: (w) => ({ ...(defaultBody("positional", w) as object), kind: "learned" }) as CellBody },
  { key: "full_attn", group: "Attention", label: "Full attention", hint: "every earlier token", make: (w) => attention(w) },
  { key: "local_attn", group: "Attention", label: "Local attention", hint: "a sliding window of recent tokens", make: (w) => attention(w, { window: 256 }) },
  { key: "cross_attn", group: "Attention", label: "Cross-attention", hint: "keys/values from another stack (wire it)", make: (w) => attention(w, { mode: "cross", causal: false }) },
  { key: "memory_read", group: "Attention", label: "Notebook read", hint: "recent tokens + notes wired in from a notes stack", make: (w) => attention(w, { window: 256, memory: true }) },
  { key: "swiglu", group: "Feed-forward", label: "SwiGLU", hint: "gate · up → down", make: (w) => defaultBody("ffn", w) },
  { key: "mlp", group: "Feed-forward", label: "MLP (GELU)", hint: "up → GELU → down", make: (w) => ({ ...(defaultBody("ffn", w) as object), kind: "mlp", hidden: 4 * w }) as CellBody },
  { key: "moe", group: "Feed-forward", label: "Mixture of experts", hint: "8 experts, top-2 active", make: (w) => ({ ...(defaultBody("ffn", w) as object), experts: 8, top_k: 2 }) as CellBody },
  { key: "latent_init", group: "Memory", label: "Learned queries", hint: "starts a slots / latents stack", make: (w) => defaultBody("latent_init", w) },
  { key: "kv_out", group: "Memory", label: "Notes out", hint: "slot states → ready keys/values for a reader", make: (w) => defaultBody("kv_out", w) },
  { key: "linear", group: "Memory", label: "Linear projection", hint: "change the width", make: (w) => defaultBody("linear", w) },
  { key: "norm", group: "Output", label: "Norm", hint: "RMSNorm or LayerNorm", make: (w) => defaultBody("norm", w) },
  { key: "lm_head", group: "Output", label: "LM head + loss", hint: "final norm, logits, cross-entropy", make: (w) => defaultBody("lm_head", w) },
  { key: "custom", group: "Other", label: "Custom block", hint: "your own parameter / FLOP numbers", make: (w) => defaultBody("custom", w) },
];

export const withId = (body: CellBody, prefix = "c"): Cell => ({ ...body, id: newId(prefix) }) as Cell;

// Every setting of every cell type, as form fields. `extras` fields live under attention.extras.
import type { CellType } from "../engine/types";

export interface Field {
  key: string;
  label: string;
  kind: "int" | "num" | "bool" | "select" | "text";
  min?: number;
  max?: number;
  step?: number;
  log?: boolean;
  options?: [string | number, string][];
  help?: string;
  extras?: boolean;
  show?: (c: Record<string, unknown>) => boolean;
}

const NORM: [string, string][] = [["rmsnorm", "RMSNorm"], ["layernorm", "LayerNorm"], ["none", "none"]];
const RESID: [string, string][] = [["pre", "pre-norm residual"], ["post", "post-norm residual"], ["none", "no residual (replace)"]];
const ex = (c: Record<string, unknown>) => (c.extras ?? {}) as Record<string, unknown>;

export const CELL_FIELDS: Record<CellType, Field[]> = {
  embedding: [
    { key: "vocab", label: "Vocabulary", kind: "int", min: 2, max: 262144, log: true },
    { key: "dim", label: "Table width (e)", kind: "int", min: 8, max: 16384, log: true, help: "with no projection the stream width is e" },
    { key: "proj", label: "Projection to width", kind: "select", options: [["none", "none (stream = table)"], ["linear", "linear e → width"], ["gated_mlp", "gated MLP (MrCogito input layer)"]] },
    { key: "out_norm", label: "Output norm", kind: "select", options: NORM },
    { key: "ngram_tables", label: "Hashed n-gram tables", kind: "int", min: 0, max: 4 },
    { key: "ngram_buckets", label: "Buckets per n-gram table", kind: "int", min: 256, max: 1048576, log: true, show: (c) => Number(c.ngram_tables) > 0 },
  ],
  positional: [
    { key: "kind", label: "Kind", kind: "select", options: [["rope", "rotary (RoPE)"], ["learned", "learned table"], ["sinusoidal", "sinusoidal"], ["alibi", "ALiBi"], ["none", "none (NoPE)"]] },
    { key: "theta", label: "RoPE base θ", kind: "num", min: 100, max: 10000000, log: true, show: (c) => c.kind === "rope" },
    { key: "max_len", label: "Table length", kind: "int", min: 16, max: 1048576, log: true, show: (c) => c.kind === "learned" },
  ],
  attention: [
    { key: "mode", label: "Reads", kind: "select", options: [["self", "its own stream (self-attention)"], ["cross", "another stack (cross-attention)"]] },
    { key: "heads", label: "Query heads", kind: "int", min: 1, max: 256 },
    { key: "head_dim", label: "Head size", kind: "int", min: 8, max: 512, step: 8 },
    { key: "kv_heads", label: "Key/value heads", kind: "int", min: 1, max: 256, help: "fewer than query heads = grouped-query / multi-query" },
    { key: "causal", label: "Causal (no peeking ahead)", kind: "bool" },
    { key: "window", label: "Window (0 = whole row)", kind: "int", min: 0, max: 1048576, log: true, show: (c) => c.mode === "self" },
    { key: "memory", label: "Also read wired-in notes", kind: "bool", show: (c) => c.mode === "self", help: "a notebook read: notes from a notes stack, each visible once its page is complete" },
    { key: "dual_key", label: "Score every key twice (E33 read + rank keys)", kind: "bool", show: (c) => Boolean(c.memory) },
    { key: "qk_norm", label: "q/k norm", kind: "bool" },
    { key: "bias", label: "Biases", kind: "bool" },
    { key: "norm", label: "Pre-norm", kind: "select", options: NORM },
    { key: "kv_norm", label: "Source norm", kind: "select", options: NORM, show: (c) => c.mode === "cross" },
    { key: "residual", label: "Residual", kind: "select", options: RESID },
    { key: "nope", label: "Skip rotary positions here", kind: "bool" },
    { key: "kernel", label: "Kernel", kind: "select", options: [["flash", "flash / fused (no score matrix kept)"], ["eager", "eager (keeps the score matrix)"]] },
    { key: "order_term", label: "Order term (E33)", kind: "bool", extras: true, show: (c) => Boolean(c.memory) },
    { key: "count_heads", label: "Count heads (E33)", kind: "int", min: 0, max: 16, extras: true, show: (c) => Boolean(c.memory) },
    { key: "count_max", label: "Count up to", kind: "int", min: 1, max: 64, extras: true, show: (c) => Boolean(c.memory) && Number(ex(c).count_heads) > 0 },
    { key: "pos_prior", label: "Position prior over the page", kind: "bool", extras: true, show: (c) => c.mode === "cross" },
    { key: "logit_scale", label: "Length-aware logit scale", kind: "bool", extras: true },
  ],
  ffn: [
    { key: "kind", label: "Kind", kind: "select", options: [["swiglu", "SwiGLU"], ["geglu", "GeGLU"], ["mlp", "plain MLP"]] },
    { key: "hidden", label: "Hidden width", kind: "int", min: 16, max: 131072, step: 64, log: true },
    { key: "activation", label: "Activation", kind: "select", options: [["gelu", "GELU"], ["relu", "ReLU"], ["silu", "SiLU"]], show: (c) => c.kind === "mlp" },
    { key: "experts", label: "Experts (1 = dense)", kind: "int", min: 1, max: 512, log: true },
    { key: "top_k", label: "Active experts per token", kind: "int", min: 1, max: 64, show: (c) => Number(c.experts) > 1 },
    { key: "bias", label: "Biases", kind: "bool" },
    { key: "norm", label: "Pre-norm", kind: "select", options: NORM },
    { key: "residual", label: "Residual", kind: "select", options: RESID },
  ],
  norm: [{ key: "kind", label: "Kind", kind: "select", options: NORM }],
  linear: [
    { key: "out", label: "Output width (0 = same)", kind: "int", min: 0, max: 65536, log: true },
    { key: "bias", label: "Bias", kind: "bool" },
    { key: "out_norm", label: "Output norm", kind: "select", options: NORM },
  ],
  latent_init: [
    { key: "address_feats", label: "Address features (0 = off)", kind: "int", min: 0, max: 1024 },
    { key: "norm", label: "Norm", kind: "select", options: NORM },
  ],
  kv_out: [
    { key: "entries", label: "Keys/values per slot", kind: "int", min: 1, max: 64 },
    { key: "kv_heads", label: "Key/value heads (= the reader's)", kind: "int", min: 1, max: 64 },
    { key: "head_dim", label: "Head size (= the reader's)", kind: "int", min: 8, max: 512, step: 8 },
    { key: "projections", label: "Projections", kind: "select", options: [[2, "2 (k, v)"], [4, "4 (k, k-read, v, v-state)"]] },
    { key: "in_norms", label: "Input norms", kind: "int", min: 0, max: 2 },
    { key: "key_norm", label: "Key norm", kind: "bool" },
  ],
  lm_head: [
    { key: "vocab", label: "Vocabulary", kind: "int", min: 2, max: 262144, log: true },
    { key: "tied", label: "Tie to the token table", kind: "bool" },
    { key: "bias", label: "Bias", kind: "bool" },
    { key: "final_norm", label: "Final norm", kind: "select", options: NORM },
    { key: "chunked", label: "Chunked loss (no full logits kept)", kind: "bool" },
    { key: "z_loss", label: "z-loss weight", kind: "num", min: 0, max: 0.01, step: 0.00001 },
    { key: "softcap", label: "Logit soft-cap (0 = off)", kind: "num", min: 0, max: 100, step: 1 },
  ],
  custom: [
    { key: "note", label: "Name", kind: "text" },
    { key: "params", label: "Parameters", kind: "int", min: 0, max: 1e11, log: true },
    { key: "flops_per_position", label: "Forward FLOPs per position", kind: "num", min: 0, max: 1e12, log: true },
    { key: "act_bytes_per_position", label: "Saved bytes per position", kind: "num", min: 0, max: 1e8, log: true },
    { key: "out", label: "Output width (0 = same)", kind: "int", min: 0, max: 65536, log: true },
  ],
};

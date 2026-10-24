// The saved graph: a stable, documented JSON format (docs/schema.md, schema/graph.schema.json).
// Everything a later code generator needs is here; `ui` is layout only and never changes the model.

export const SCHEMA_ID = "nn-builder/graph";
export const SCHEMA_VERSION = 1;

export type NormKind = "rmsnorm" | "layernorm" | "none";
export type Residual = "pre" | "post" | "none";

/** How many positions a stack's stream has, as a function of the training row length S. */
export type SeqSpec =
  | { kind: "tokens" } // the training row: S positions, the loss is on these
  | { kind: "source"; length: number } // an encoder input of its own length (0 = same as the row)
  | { kind: "pages"; of: string; window: number; stride: number } // overlapping pages over stack `of`
  | { kind: "slots"; of: string; per_page: number } // per_page learned slots for every page of pages-stack `of`
  | { kind: "latents"; count: number }; // a fixed set of learned latents (Perceiver)

export interface EmbeddingCell {
  type: "embedding";
  vocab: number;
  dim: number; // table width e (the stream width comes from `proj`)
  ngram_tables: number; // hashed n-gram tables added to the token table
  ngram_buckets: number;
  proj: "none" | "linear" | "gated_mlp"; // gated_mlp: up0(x) + up2(silu(up1(x))), the MrCogito input layer
  out_norm: NormKind;
  share_with?: string; // id of another embedding cell whose table this one reuses (0 own params)
}

export interface PositionalCell {
  type: "positional";
  kind: "rope" | "learned" | "sinusoidal" | "alibi" | "none";
  theta: number; // rope base
  max_len: number; // learned table length
}

export interface AttentionExtras {
  order_term: boolean; // E33: per-head order strength + a d→1 order gate on the note read
  count_heads: number; // E33: count heads beside the read (0 = off)
  count_max: number;
  pos_prior: boolean; // per-slot, per-head position prior over the page (slots → page reads)
  logit_scale: boolean; // one learnable length-aware query scale
}

export interface AttentionCell {
  type: "attention";
  mode: "self" | "cross"; // cross: keys/values come from the stack wired into this cell
  heads: number;
  head_dim: number;
  kv_heads: number; // grouped-query: heads must be a multiple of kv_heads
  causal: boolean;
  window: number; // 0 = the whole row; w = sliding window of w keys (self only)
  qk_norm: boolean;
  bias: boolean;
  norm: NormKind; // pre-norm (or post-norm) of this sublayer
  kv_norm: NormKind; // cross only: norm on the source side
  residual: Residual;
  memory: boolean; // self only: also reads the notes wired into this cell, each visible once its page is complete
  dual_key: boolean; // memory read scores every key twice (E33's read and rank keys)
  nope: boolean; // skip the rotary position for this layer
  kernel: "flash" | "eager"; // eager stores the score matrix for backward
  extras: AttentionExtras;
}

export interface FfnCell {
  type: "ffn";
  kind: "swiglu" | "geglu" | "mlp";
  hidden: number;
  activation: "gelu" | "relu" | "silu"; // mlp only
  bias: boolean;
  experts: number; // 1 = dense
  top_k: number;
  norm: NormKind;
  residual: Residual;
}

export interface NormCell {
  type: "norm";
  kind: NormKind;
}

export interface LinearCell {
  type: "linear";
  out: number; // output width; 0 = keep the width
  bias: boolean;
  out_norm: NormKind;
}

export interface LatentInitCell {
  type: "latent_init"; // learned queries that start a slots/latents stream
  address_feats: number; // a linear map from page-address features (0 = off)
  norm: NormKind;
}

export interface KvOutCell {
  type: "kv_out"; // projects slot states into keys/values a reader uses directly (the notes)
  entries: number; // keys/values per slot
  kv_heads: number;
  head_dim: number;
  projections: number; // 2 = k, v; 4 = k, k_read, v, v_state (E33)
  in_norms: number;
  key_norm: boolean;
}

export interface LmHeadCell {
  type: "lm_head";
  vocab: number;
  tied: boolean; // reuse the stack's embedding table (needs dim = width)
  bias: boolean;
  final_norm: NormKind;
  z_loss: number;
  softcap: number;
  chunked: boolean; // chunked cross-entropy, recomputed in backward (no [rows, S, V] logits kept)
}

export interface CustomCell {
  type: "custom";
  params: number;
  flops_per_position: number; // forward FLOPs per position
  act_bytes_per_position: number;
  out: number; // output width; 0 = keep
  note: string;
}

export type CellBody =
  | EmbeddingCell
  | PositionalCell
  | AttentionCell
  | FfnCell
  | NormCell
  | LinearCell
  | LatentInitCell
  | KvOutCell
  | LmHeadCell
  | CustomCell;

export type CellType = CellBody["type"];
export type Cell = CellBody & { id: string; label?: string };

export interface Row {
  id: string;
  label?: string;
  x0_mix: boolean; // x ← α·x + β·x0 before the row (x0 = the stack's stream after its input layer), 2 scalars
  cells: Cell[];
}

export interface Stack {
  id: string;
  name: string;
  role: "decoder" | "encoder" | "writer" | "latent" | "other";
  width: number;
  seq: SeqSpec;
  rows: Row[];
}

/** One end of a wire. A row means its output; a cell plus port means that cell's port; a stack alone means its output. */
export interface Endpoint {
  stack: string;
  row?: string;
  cell?: string;
  port?: "out" | "table";
}

export type Edge =
  | { id: string; kind: "skip"; from: Endpoint; to: Endpoint; weight: "scalar" | "vector" | "none" }
  | { id: string; kind: "source"; from: Endpoint; to: Endpoint } // keys/values (cross-attention) or notes (memory)
  | { id: string; kind: "input"; from: Endpoint; to: Endpoint }; // starts a stack's stream from another stack

export interface Group {
  id: string;
  kind: "loop" | "repeat"; // loop: shared weights run `count` times; repeat: `count` independent copies
  stack: string;
  rows: string[]; // contiguous row ids
  count: number;
  round_embedding: boolean; // loop: a learned vector per round (count × width)
  feedback: "none" | "soft"; // loop: the round's LM-head reading, re-embedded, joins the next round (gate: width)
}

export type Precision = "amp_bf16" | "bf16_master" | "pure_bf16" | "fp32";
export type OptimizerKind = "adamw" | "adamw_8bit" | "sgd_momentum" | "muon" | "adafactor";

export interface RunSetup {
  row_length: number;
  rows_per_step: number;
  micro_batch: number; // 0 = the largest that fits
  precision: Precision;
  optimizer: OptimizerKind;
  weight_decay: number;
  zero_stage: 0 | 1 | 2 | 3;
  checkpointing: boolean;
  gpu: string;
  gpu_count: number;
  token_budget: number;
  deadline_hours: number;
  mfu: number; // 0 = the GPU's default
  irregular_ratio: number; // 0 = calibrated default; speed of masked / gathered / looped work relative to plain matmuls
  dp_efficiency: number;
}

export interface Graph {
  schema: typeof SCHEMA_ID;
  version: typeof SCHEMA_VERSION;
  name: string;
  description: string;
  stacks: Stack[];
  edges: Edge[];
  groups: Group[];
  run: RunSetup;
  ui?: Record<string, unknown>;
}

// Ready-made graphs. The MrCogito presets reproduce the trainer's models exactly (tests/params.test.ts).
import { DEFAULT_RUN, attention, defaultBody } from "./defaults";
import { SCHEMA_ID, SCHEMA_VERSION, type Cell, type CellBody, type Edge, type Graph, type Group, type Row, type Stack } from "./types";

const cell = (id: string, body: CellBody, label?: string): Cell => ({ ...body, id, ...(label ? { label } : {}) }) as Cell;
const row = (id: string, cells: Cell[], o: Partial<Row> = {}): Row => ({ id, x0_mix: false, cells, ...o });

function graph(name: string, description: string, stacks: Stack[], edges: Edge[] = [], groups: Group[] = [], run = {}): Graph {
  return { schema: SCHEMA_ID, version: SCHEMA_VERSION, name, description, stacks, edges, groups, run: { ...DEFAULT_RUN, ...run } };
}

// ------------------------------------------------------------------ MrCogito: the trainer's dense and E33 text models

const V = 4096;
const mrInput = (s: string): Row =>
  row(`${s}.input`, [
    cell(`${s}.input.emb`, { type: "embedding", vocab: V, dim: 256, ngram_tables: 0, ngram_buckets: 8192, proj: "gated_mlp", out_norm: "rmsnorm" }, "token table + input MLP"),
    cell(`${s}.input.pos`, { type: "positional", kind: "rope", theta: 500000, max_len: 4096 }),
  ], { label: "input" });
const mrHead = (s: string): Row =>
  row(`${s}.head`, [cell(`${s}.head.lm`, { type: "lm_head", vocab: V, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 1e-4, softcap: 30, chunked: true })], { label: "head" });
const mrLayer = (s: string, i: number, d: number, F: number, window: number, label?: string, extra: Partial<Parameters<typeof attention>[1]> = {}): Row =>
  row(`${s}.L${i}`, [
    cell(`${s}.L${i}.attn`, attention(d, { kv_heads: 1, window, ...extra })),
    cell(`${s}.L${i}.ffn`, { ...(defaultBody("ffn", d) as object), hidden: F } as CellBody),
  ], { label: label ?? `L${i}`, x0_mix: true });

/** U-net skips: layers [0, n/2) push their output, layers [n − n/2, n) take them back in reverse, each with one weight. */
function unetSkips(s: string, n: number): Edge[] {
  const k = Math.floor(n / 2);
  return Array.from({ length: k }, (_, j) => ({
    id: `${s}.skip${j}`,
    kind: "skip" as const,
    from: { stack: s, row: `${s}.L${k - 1 - j}` },
    to: { stack: s, row: `${s}.L${n - k + j}` },
    weight: "scalar" as const,
  }));
}

function dense(d: number, layers: number, F: number, tokens: number) {
  const s = "main";
  const main: Stack = {
    id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [mrInput(s), ...Array.from({ length: layers }, (_, i) => mrLayer(s, i, d, F, 0)), mrHead(s)],
  };
  return graph(`dense_plain ${Math.round(d === 896 ? 100 : layers === 12 ? 50 : 30)}M`,
    `MrCogito dense text model, plain input (no n-gram tables): width ${d}, ${layers} layers, FFN ${F}.`,
    [main], unetSkips(s, layers), [], { token_budget: tokens, gpu_count: 3 });
}

function e33(d: number, pre: number, post: number, F: number, tokens: number) {
  const s = "main";
  const n = pre + 1 + post;
  const read = pre;
  const layers = Array.from({ length: n }, (_, i) =>
    i === read
      ? mrLayer(s, i, d, F, 256, "read", { memory: true, dual_key: true, extras: { order_term: true, count_heads: 2, count_max: 10, pos_prior: false, logit_scale: false } })
      : mrLayer(s, i, d, F, 16));
  const main: Stack = { id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" }, rows: [mrInput(s), ...layers, mrHead(s)] };

  const dw = 256, D = 512, K = 32, hl = 8;
  const pages: Stack = {
    id: "pages", name: "Pages", role: "writer", width: dw, seq: { kind: "pages", of: s, window: 256, stride: 192 },
    rows: [
      row("pages.in", [
        cell("pages.in.proj", { type: "linear", out: dw, bias: false, out_norm: "rmsnorm" }, "table → writer width"),
        cell("pages.in.pos", { type: "positional", kind: "rope", theta: 10000, max_len: 256 }),
      ], { label: "in" }),
      ...[0, 1].map((i) =>
        row(`pages.enc${i}`, [
          cell(`pages.enc${i}.attn`, attention(dw, { causal: false, window: 0 })),
          cell(`pages.enc${i}.ffn`, { ...(defaultBody("ffn", dw) as object), hidden: 2 * dw } as CellBody),
        ], { label: `page layer ${i}` })),
    ],
  };
  const notes: Stack = {
    id: "notes", name: "Notes", role: "latent", width: D, seq: { kind: "slots", of: "pages", per_page: K },
    rows: [
      row("notes.init", [cell("notes.init.q", { type: "latent_init", address_feats: 64, norm: "rmsnorm" }, "32 slots per page")], { label: "slots" }),
      row("notes.read", [cell("notes.read.attn", attention(D, { mode: "cross", heads: hl, head_dim: D / hl, kv_heads: hl, causal: false, kernel: "eager",
        extras: { order_term: false, count_heads: 0, count_max: 10, pos_prior: true, logit_scale: false } }), "read the page")], { label: "read page" }),
      row("notes.ffn", [cell("notes.ffn.ffn", { ...(defaultBody("ffn", D) as object), hidden: 2 * D } as CellBody)], { label: "think" }),
      row("notes.out", [cell("notes.out.kv", { type: "kv_out", entries: 1, kv_heads: 1, head_dim: 64, projections: 4, in_norms: 2, key_norm: true }, "notes (keys, values)")], { label: "notes" }),
    ],
  };
  const edges: Edge[] = [
    ...unetSkips(s, n),
    { id: "wire.table", kind: "input", from: { stack: s, cell: `${s}.input.emb`, port: "table" }, to: { stack: "pages" } },
    { id: "wire.pages", kind: "source", from: { stack: "pages" }, to: { stack: "notes", cell: "notes.read.attn" } },
    { id: "wire.notes", kind: "source", from: { stack: "notes" }, to: { stack: s, cell: `${s}.L${read}.attn` } },
  ];
  const groups: Group[] = [
    { id: "loop.read", kind: "loop", stack: s, rows: [`${s}.L${read}`, `${s}.L${read + 1}`], count: 4, round_embedding: true, feedback: "soft" },
    { id: "loop.writer", kind: "loop", stack: "notes", rows: ["notes.read", "notes.ffn"], count: 2, round_embedding: false, feedback: "none" },
  ];
  return graph(`e33m_plain ${d === 896 ? 100 : 30}M`,
    `MrCogito E33 notebook text model: a page writer makes 32 notes per page; one read layer (+ the next) loops 4 rounds with shared weights. Width ${d}, ${n} layers, FFN ${F}.`,
    [main, pages, notes], edges, groups, { token_budget: tokens, gpu_count: 3 });
}

// ------------------------------------------------------------------ outside references and examples

function gpt2Small() {
  const s = "main", d = 768;
  const layers = Array.from({ length: 12 }, (_, i) =>
    row(`${s}.L${i}`, [
      cell(`${s}.L${i}.attn`, attention(d, { qk_norm: false, bias: true, norm: "layernorm" })),
      cell(`${s}.L${i}.ffn`, { type: "ffn", kind: "mlp", hidden: 3072, activation: "gelu", bias: true, experts: 1, top_k: 1, norm: "layernorm", residual: "pre" }),
    ], { label: `L${i}` }));
  const main: Stack = {
    id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row(`${s}.input`, [
        cell(`${s}.input.emb`, { type: "embedding", vocab: 50257, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell(`${s}.input.pos`, { type: "positional", kind: "learned", theta: 10000, max_len: 1024 }),
      ], { label: "input" }),
      ...layers,
      row(`${s}.head`, [cell(`${s}.head.lm`, { type: "lm_head", vocab: 50257, tied: true, bias: false, final_norm: "layernorm", z_loss: 0, softcap: 0, chunked: false })], { label: "head" }),
    ],
  };
  return graph("GPT-2 small", "OpenAI GPT-2 small (124M): learned positions, LayerNorm, GELU MLP, head tied to the token table.", [main], [], [], { row_length: 1024 });
}

function encoderDecoder() {
  const d = 512, F = 2048, n = 6, Vt = 32128;
  const layer = (s: string, i: number, causal: boolean, cross: boolean): Row =>
    row(`${s}.L${i}`, [
      cell(`${s}.L${i}.attn`, attention(d, { causal, qk_norm: false })),
      ...(cross ? [cell(`${s}.L${i}.cross`, attention(d, { mode: "cross", causal: false, qk_norm: false, kv_norm: "none" }))] : []),
      cell(`${s}.L${i}.ffn`, { type: "ffn", kind: "mlp", hidden: F, activation: "relu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }),
    ], { label: `L${i}` });
  const enc: Stack = {
    id: "enc", name: "Encoder", role: "encoder", width: d, seq: { kind: "source", length: 0 },
    rows: [
      row("enc.input", [
        cell("enc.input.emb", { type: "embedding", vocab: Vt, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell("enc.input.pos", { type: "positional", kind: "rope", theta: 10000, max_len: 4096 }),
      ], { label: "input" }),
      ...Array.from({ length: n }, (_, i) => layer("enc", i, false, false)),
      row("enc.out", [cell("enc.out.norm", { type: "norm", kind: "rmsnorm" })], { label: "out" }),
    ],
  };
  const dec: Stack = {
    id: "dec", name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row("dec.input", [
        cell("dec.input.emb", { type: "embedding", vocab: Vt, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none", share_with: "enc.input.emb" }),
        cell("dec.input.pos", { type: "positional", kind: "rope", theta: 10000, max_len: 4096 }),
      ], { label: "input" }),
      ...Array.from({ length: n }, (_, i) => layer("dec", i, true, true)),
      row("dec.head", [cell("dec.head.lm", { type: "lm_head", vocab: Vt, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const edges: Edge[] = Array.from({ length: n }, (_, i) => ({ id: `wire.cross${i}`, kind: "source" as const, from: { stack: "enc" }, to: { stack: "dec", cell: `dec.L${i}.cross` } }));
  return graph("Encoder–decoder", "A T5-small-shaped encoder–decoder: 6 bidirectional encoder layers, 6 decoder layers that cross-attend to the encoder, one shared token table.", [enc, dec], edges, [], { row_length: 1024 });
}

function blank() {
  const s = "main", d = 512;
  const main: Stack = {
    id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row(`${s}.input`, [cell(`${s}.input.emb`, defaultBody("embedding", d)), cell(`${s}.input.pos`, defaultBody("positional", d))], { label: "input" }),
      row(`${s}.L0`, [cell(`${s}.L0.attn`, attention(d)), cell(`${s}.L0.ffn`, defaultBody("ffn", d))], { label: "L0" }),
      row(`${s}.head`, [cell(`${s}.head.lm`, defaultBody("lm_head", d))], { label: "head" }),
    ],
  };
  return graph("New model", "", [main]);
}

export const PRESETS: { id: string; label: string; build: () => Graph }[] = [
  { id: "dense_plain_30m", label: "MrCogito dense · 30M", build: () => dense(576, 8, 1472, 1.5e9) },
  { id: "dense_plain_50m", label: "MrCogito dense · 50M", build: () => dense(576, 12, 1792, 1.5e9) },
  { id: "dense_plain_100m", label: "MrCogito dense · 100M", build: () => dense(896, 12, 2304, 2e9) },
  { id: "e33m_plain_30m", label: "MrCogito E33 notebook · 30M", build: () => e33(576, 2, 5, 1152, 1.5e9) },
  { id: "e33m_plain_100m", label: "MrCogito E33 notebook · 100M", build: () => e33(896, 3, 8, 2176, 2e9) },
  { id: "gpt2_small", label: "GPT-2 small · 124M", build: gpt2Small },
  { id: "encdec", label: "Encoder–decoder (T5-small shape)", build: encoderDecoder },
  { id: "blank", label: "Blank: one layer", build: blank },
];

export function buildPreset(id: string): Graph {
  const p = PRESETS.find((x) => x.id === id);
  if (!p) throw new Error(`unknown preset ${id}`);
  return p.build();
}

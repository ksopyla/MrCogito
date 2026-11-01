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
  return notebook(d, pre, post, F, tokens, true);
}

/** E31c: the same page writer and notes, one closed-window read, no loop / order term / count heads. */
function e31c(d: number, pre: number, post: number, F: number, tokens: number) {
  return notebook(d, pre, post, F, tokens, false);
}

function notebook(d: number, pre: number, post: number, F: number, tokens: number, loop: boolean) {
  const s = "main";
  const n = pre + 1 + post;
  const read = pre;
  const layers = Array.from({ length: n }, (_, i) =>
    i === read
      ? mrLayer(s, i, d, F, 256, "read", { memory: true, dual_key: true, extras: { order_term: loop, count_heads: loop ? 2 : 0, count_max: 10, pos_prior: false, logit_scale: false } })
      : mrLayer(s, i, d, F, 16));
  const main: Stack = { id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" }, rows: [mrInput(s), ...layers, mrHead(s)] };

  const dw = 256, D = 512, K = 32, hl = 8;
  const pages: Stack = {
    id: "pages", name: "Pages", role: "writer", width: dw, seq: { kind: "pages", of: s, window: 256, stride: 192 },
    rows: [
      row("pages.in", [
        cell("pages.in.proj", { type: "linear", out: dw, bias: false, out_norm: "rmsnorm" }, "Project to writer"),
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
    ...(loop ? [{ id: "loop.read", kind: "loop" as const, stack: s, rows: [`${s}.L${read}`, `${s}.L${read + 1}`], count: 4, round_embedding: true, feedback: "soft" as const }] : []),
    { id: "loop.writer", kind: "loop", stack: "notes", rows: ["notes.read", "notes.ffn"], count: 2, round_embedding: false, feedback: "none" },
  ];
  const size = d === 896 ? 100 : 30;
  return loop
    ? graph(`e33m_plain ${size}M`,
      `MrCogito E33 notebook text model: a page writer makes 32 notes per page; one read layer (+ the next) loops 4 rounds with shared weights. Width ${d}, ${n} layers, FFN ${F}.`,
      [main, pages, notes], edges, groups, { token_budget: tokens, gpu_count: 3 })
    : graph(`e31c_plain ${size}M`,
      `MrCogito E31c latent notebook: a page writer makes 32 notes per page; one read layer reads the last 256 tokens plus the finished pages' notes (closed read). No loop. Width ${d}, ${n} layers, FFN ${F}.`,
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
  // T5-small (Raffel et al. 2020, t5-small on Hugging Face): relative-position bias in each stack, RMS-style norms
  // without bias, ReLU MLP, one table shared by both stacks and tied to the head: 60,506,624 parameters.
  const d = 512, F = 2048, n = 6, Vt = 32128;
  const layer = (s: string, i: number, causal: boolean, cross: boolean): Row =>
    row(`${s}.L${i}`, [
      cell(`${s}.L${i}.attn`, attention(d, { causal, qk_norm: false })),
      ...(cross ? [cell(`${s}.L${i}.cross`, attention(d, { mode: "cross", causal: false, qk_norm: false, kv_norm: "none" }))] : []),
      cell(`${s}.L${i}.ffn`, { type: "ffn", kind: "mlp", hidden: F, activation: "relu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }),
    ], { label: `L${i}` });
  const bias = (id: string) => cell(id, { type: "positional", kind: "t5_bias", theta: 10000, max_len: 512, buckets: 32, heads: 8 }, "relative-position bias");
  const enc: Stack = {
    id: "enc", name: "Encoder", role: "encoder", width: d, seq: { kind: "source", length: 0 },
    rows: [
      row("enc.input", [
        cell("enc.input.emb", { type: "embedding", vocab: Vt, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }, "shared table"),
        bias("enc.input.pos"),
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
        bias("dec.input.pos"),
      ], { label: "input" }),
      ...Array.from({ length: n }, (_, i) => layer("dec", i, true, true)),
      row("dec.head", [cell("dec.head.lm", { type: "lm_head", vocab: Vt, tied: true, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const edges: Edge[] = Array.from({ length: n }, (_, i) => ({ id: `wire.cross${i}`, kind: "source" as const, from: { stack: "enc" }, to: { stack: "dec", cell: `dec.L${i}.cross` } }));
  return graph("T5-small", "Google T5-small (60.5M): 6 bidirectional encoder layers, 6 decoder layers that cross-attend to the encoder's output, relative-position bias, one table shared by both stacks and tied to the head.", [enc, dec], edges, [], { row_length: 512 });
}

function smollm3() {
  // HuggingFaceTB/SmolLM3-3B config.json: 36 layers, width 2048, 16 query / 4 key-value heads of 128, FFN 11008,
  // no RoPE on every 4th layer, tied 128,256-token table: 3,075,098,624 parameters.
  const s = "main", d = 2048;
  const block = (i: number): Row =>
    row(`${s}.L${i}`, [
      cell(`${s}.L${i}.attn`, attention(d, { heads: 16, head_dim: 128, kv_heads: 4, qk_norm: false, nope: i === 3 })),
      cell(`${s}.L${i}.ffn`, { type: "ffn", kind: "swiglu", hidden: 11008, activation: "silu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }),
    ], { label: i === 3 ? "L3 (no RoPE)" : `L${i}` });
  const main: Stack = {
    id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row(`${s}.input`, [
        cell(`${s}.input.emb`, { type: "embedding", vocab: 128256, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell(`${s}.input.pos`, { type: "positional", kind: "rope", theta: 5e6, max_len: 65536 }),
      ], { label: "input" }),
      ...[0, 1, 2, 3].map(block),
      row(`${s}.head`, [cell(`${s}.head.lm`, { type: "lm_head", vocab: 128256, tied: true, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const groups: Group[] = [{ id: "rep.blocks", kind: "repeat", stack: s, rows: [0, 1, 2, 3].map((i) => `${s}.L${i}`), count: 9, round_embedding: false, feedback: "none" }];
  return graph("SmolLM3-3B", "Hugging Face SmolLM3-3B (Jul 2025): Llama-style decoder, 36 layers as 9 × (3 RoPE layers + 1 NoPE layer), GQA 16 / 4, tied embeddings.", [main], [], groups,
    { row_length: 4096, gpu: "h100_80", gpu_count: 8, token_rule: "custom", token_budget: 11.2e12, zero_stage: 3 });
}

function deepseekV2Lite() {
  // deepseek-ai/DeepSeek-V2-Lite config.json: 27 layers, width 2048, MLA (kv latent 512, rotary 64, no-position 128,
  // value 128, 16 heads), layer 0 dense (FFN 10944), layers 1–26 MoE (64 routed experts of 1408, 6 per token, 2 shared),
  // untied 102,400-token head: 15,706,484,224 parameters.
  const s = "main", d = 2048;
  const mla = { q_rank: 0, kv_rank: 512, rope_dim: 64, nope_dim: 128, v_dim: 128 };
  const attn = (id: string) => cell(id, attention(d, { heads: 16, head_dim: 128, kv_heads: 16, qk_norm: false, mla }), "MLA");
  const main: Stack = {
    id: s, name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row(`${s}.input`, [
        cell(`${s}.input.emb`, { type: "embedding", vocab: 102400, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell(`${s}.input.pos`, { type: "positional", kind: "rope", theta: 10000, max_len: 163840 }),
      ], { label: "input" }),
      row(`${s}.L0`, [attn(`${s}.L0.attn`), cell(`${s}.L0.ffn`, { type: "ffn", kind: "swiglu", hidden: 10944, activation: "silu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }, "dense")], { label: "L0" }),
      row(`${s}.L1`, [attn(`${s}.L1.attn`), cell(`${s}.L1.ffn`, { type: "ffn", kind: "swiglu", hidden: 1408, activation: "silu", bias: false, experts: 64, top_k: 6, shared_experts: 2, norm: "rmsnorm", residual: "pre" }, "MoE 64 + 2 shared")], { label: "L1–26" }),
      row(`${s}.head`, [cell(`${s}.head.lm`, { type: "lm_head", vocab: 102400, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const groups: Group[] = [{ id: "rep.moe", kind: "repeat", stack: s, rows: [`${s}.L1`], count: 26, round_embedding: false, feedback: "none" }];
  return graph("DeepSeek-V2-Lite", "DeepSeek-V2-Lite (May 2024), the smallest official DeepSeek with MLA attention and shared-expert MoE: 15.7B parameters, about 2.4B used per token.", [main], [], groups,
    { row_length: 4096, gpu: "h100_80", gpu_count: 8, token_rule: "custom", token_budget: 5.7e12, zero_stage: 3 });
}

function deepseekV41Flash() {
  // APPROXIMATE. deepseek-ai/DeepSeek-V4.1-Flash config.json + README (16 Oct 2026): a 40-layer causal encoder–decoder
  // (20 + 20), width 5120, 64 query heads of 512 sharing one key/value head, low-rank query (1280) and output (1024),
  // MoE in every layer (384 routed experts of 2304, 6 per token, 1 shared), Engram n-gram memory at layers 1 and 14
  // (2 × ~384M rows of 256), untied 129,280-token head. Approximated: the compressed sparse attention (CSA2) as a
  // window of 128 + 512 sparse keys; the decoder's global keys/values from the encoder as a causal cross-attention;
  // left out: hyper-connections (mHC), the DSpark / next-token-prediction heads, the vision encoder.
  const d = 5120, V = 129280, F = 2304;
  const moe = (id: string) => cell(id, { type: "ffn", kind: "swiglu", hidden: F, activation: "silu", bias: false, experts: 384, top_k: 6, shared_experts: 1, router_bias: true, norm: "rmsnorm", residual: "pre" }, "MoE 384 + 1 shared");
  const attn = (id: string, window: number, label: string) => cell(id, attention(d, { heads: 64, head_dim: 512, kv_heads: 1, window, qk_norm: false, q_rank: 1280, o_rank: 1024 }), label);
  const engram = (id: string, rows: number) => cell(id, { type: "custom", params: rows * 256, flops_per_position: 2 * 8 * 256 * d, act_bytes_per_position: 2 * 8 * 256, out: 0, note: "Engram n-gram memory", sparse: true });
  const encRow = (i: number, label: string, extra?: Cell): Row =>
    row(`enc.L${i}`, [attn(`enc.L${i}.attn`, 640, "sparse attention ≈ 128 + 512 keys"), ...(extra ? [extra] : []), moe(`enc.L${i}.ffn`)], { label });
  const enc: Stack = {
    id: "enc", name: "Causal encoder", role: "encoder", width: d, seq: { kind: "tokens" },
    rows: [
      row("enc.input", [
        cell("enc.input.emb", { type: "embedding", vocab: V, dim: d, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell("enc.input.pos", { type: "positional", kind: "rope", theta: 10000, max_len: 1048576 }),
      ], { label: "input" }),
      encRow(0, "E0"),
      encRow(1, "E1 + Engram", engram("enc.L1.engram", 384006168)),
      encRow(2, "E2–13"),
      encRow(14, "E14 + Engram", engram("enc.L14.engram", 384016682)),
      encRow(15, "E15–19"),
    ],
  };
  const dec: Stack = {
    id: "dec", name: "Decoder", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      row("dec.L0", [
        attn("dec.L0.attn", 128, "local window 128"),
        cell("dec.L0.cross", attention(d, { mode: "cross", heads: 64, head_dim: 512, kv_heads: 1, causal: true, window: 512, qk_norm: false, kv_norm: "none", q_rank: 1280, o_rank: 1024 }), "global keys/values from the encoder"),
        moe("dec.L0.ffn"),
      ], { label: "D0–19" }),
      row("dec.head", [cell("dec.head.lm", { type: "lm_head", vocab: V, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const edges: Edge[] = [
    { id: "wire.enc_in", kind: "input", from: { stack: "enc", row: "enc.input" }, to: { stack: "dec" } },
    { id: "wire.global_kv", kind: "source", from: { stack: "enc" }, to: { stack: "dec", cell: "dec.L0.cross" } },
  ];
  const groups: Group[] = [
    { id: "rep.enc", kind: "repeat", stack: "enc", rows: ["enc.L2"], count: 12, round_embedding: false, feedback: "none" },
    { id: "rep.enc2", kind: "repeat", stack: "enc", rows: ["enc.L15"], count: 5, round_embedding: false, feedback: "none" },
    { id: "rep.dec", kind: "repeat", stack: "dec", rows: ["dec.L0"], count: 20, round_embedding: false, feedback: "none" },
  ];
  return graph("DeepSeek-V4.1-Flash (approximate)", "APPROXIMATE text backbone of DeepSeek-V4.1-Flash (16 Oct 2026): 552B backbone + 196B Engram memory, 6 of 384 experts per token. Sparse attention, hyper-connections and the drafting heads are approximated or left out; see the preset's source comment.", [enc, dec], edges, groups,
    { row_length: 4096, gpu: "b300", gpu_count: 64, token_rule: "custom", token_budget: 45e12, zero_stage: 3, precision: "bf16_master" });
}

function e36() {
  // MrCogito E36 (key/value way): the first 3 layers of SmolLM2-135M read each 512-token span of a memory text
  // (two-way inside the span) and pool it to one entry; a linear expander turns each entry into keys and values for
  // all 32 layers of a frozen SmolLM2-360M, which reads them before its own tokens. Trainable: 22,437,888.
  const span = 512, spans = 8;
  const text: Stack = {
    id: "text", name: "Memory text", role: "encoder", width: 576, seq: { kind: "source", length: span * spans },
    rows: [row("text.input", [{ ...cell("text.input.emb", { type: "embedding", vocab: 49152, dim: 576, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }, "SmolLM2-135M table (borrowed)"), frozen: true }], { label: "input" })],
  };
  const writer: Stack = {
    id: "writer", name: "Writer", role: "writer", width: 576, seq: { kind: "pages", of: "text", window: span, stride: span },
    rows: [0, 1, 2].map((i) =>
      row(`writer.L${i}`, [
        ...(i === 0 ? [cell("writer.pos", { type: "positional", kind: "rope", theta: 10000, max_len: span })] : []),
        cell(`writer.L${i}.attn`, attention(576, { heads: 9, head_dim: 64, kv_heads: 3, causal: false, qk_norm: false })),
        cell(`writer.L${i}.ffn`, { type: "ffn", kind: "swiglu", hidden: 1536, activation: "silu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }),
      ], { label: `135M layer ${i}` })),
  };
  const entries: Stack = {
    id: "entries", name: "Entries", role: "latent", width: 576, seq: { kind: "slots", of: "writer", per_page: 1 },
    rows: [
      row("entries.pool", [cell("entries.pool.mean", { type: "pool", mode: "mean", heads: 9 }, "one entry per span"), cell("entries.pool.norm", { type: "norm", kind: "rmsnorm" })], { label: "pool" }),
      row("entries.kv", [cell("entries.kv.expand", { type: "kv_out", entries: 1, kv_heads: 5, head_dim: 64, projections: 2, in_norms: 0, key_norm: false, layers: 32, bias: true, gains: true }, "keys/values for 32 layers")], { label: "expand" }),
    ],
  };
  const reader: Stack = {
    id: "reader", name: "Reader (SmolLM2-360M, frozen)", role: "decoder", width: 960, seq: { kind: "tokens" }, frozen: true,
    rows: [
      row("reader.input", [
        cell("reader.input.emb", { type: "embedding", vocab: 49152, dim: 960, ngram_tables: 0, ngram_buckets: 0, proj: "none", out_norm: "none" }),
        cell("reader.input.pos", { type: "positional", kind: "rope", theta: 100000, max_len: 8192 }),
      ], { label: "input" }),
      ...Array.from({ length: 32 }, (_, i) =>
        row(`reader.L${i}`, [
          cell(`reader.L${i}.attn`, attention(960, { heads: 15, head_dim: 64, kv_heads: 5, qk_norm: false, memory: true })),
          cell(`reader.L${i}.ffn`, { type: "ffn", kind: "swiglu", hidden: 2560, activation: "silu", bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre" }),
        ], { label: `L${i}` })),
      row("reader.head", [cell("reader.head.lm", { type: "lm_head", vocab: 49152, tied: true, bias: false, final_norm: "rmsnorm", z_loss: 0, softcap: 0, chunked: true })], { label: "head" }),
    ],
  };
  const edges: Edge[] = [
    { id: "wire.text", kind: "input", from: { stack: "text" }, to: { stack: "writer" } },
    { id: "wire.writer", kind: "input", from: { stack: "writer" }, to: { stack: "entries" } },
    ...Array.from({ length: 32 }, (_, i) => ({ id: `wire.entries${i}`, kind: "source" as const, from: { stack: "entries" }, to: { stack: "reader", cell: `reader.L${i}.attn` } })),
  ];
  return graph("E36 frozen-reader writer", "MrCogito E36, key/value way: 3 SmolLM2-135M layers write each 512-token span of a memory text into one entry, expanded into keys and values for all 32 layers of a frozen SmolLM2-360M that then reads its own 512 tokens.", [text, writer, entries, reader], edges, [],
    { row_length: 512, rows_per_step: 32, gpu_count: 1, token_budget: 0.59e9 });
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
  { id: "e31c_plain_30m", label: "MrCogito E31c latent notebook · 30M", build: () => e31c(576, 2, 5, 1152, 1.5e9) },
  { id: "e31c_plain_100m", label: "MrCogito E31c latent notebook · 100M", build: () => e31c(896, 3, 8, 2176, 2e9) },
  { id: "e36_kv", label: "MrCogito E36 memory writer · frozen 360M reader", build: e36 },
  { id: "gpt2_small", label: "GPT-2 small · 124M", build: gpt2Small },
  { id: "encdec", label: "T5-small · encoder–decoder · 60.5M", build: encoderDecoder },
  { id: "smollm3_3b", label: "SmolLM3-3B", build: smollm3 },
  { id: "deepseek_v2_lite", label: "DeepSeek-V2-Lite · 15.7B MoE (MLA)", build: deepseekV2Lite },
  { id: "deepseek_v41_flash", label: "DeepSeek-V4.1-Flash · approximate", build: deepseekV41Flash },
  { id: "blank", label: "Blank: one layer", build: blank },
];

export function buildPreset(id: string): Graph {
  const p = PRESETS.find((x) => x.id === id);
  if (!p) throw new Error(`unknown preset ${id}`);
  return p.build();
}

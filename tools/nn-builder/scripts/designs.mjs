// The architecture tracker's generator: every saved MrCogito design in designs/, rebuilt from the as-trained E33n
// testbed designs (one change per arm), plus the E37 concept-memory arms, the E31 capability-check champion and the
// static references. It refreshes each file's name, description and `derived` block (keeping `exported_at` when
// nothing changed) and rewrites the table in designs/CATALOGUE.md between its markers.
//
// Run from tools/nn-builder:  node scripts/designs.mjs          (write)
//                             node scripts/designs.mjs --check  (print the table, write nothing)
//
// Truth for every count: the trainer's own factory (scripts/run_e33n.py model_args(arm, size, ffn) through
// training/concept_pretraining_factories, = evaluation/text_checks.count_params; FFN from fit(): 1472 at s30, 2624
// at m100) or, for the champion, evaluation/bapo_models.build_model as the exam runner calls it. The numbers are
// pinned in tests/designs_catalogue.test.ts; this script only draws.
import { createServer } from "vite";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const CHECK = process.argv.includes("--check");
const server = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { analyze } = await server.ssrLoadModule("/src/engine/analyze.ts");
const { parseGraph, toExport } = await server.ssrLoadModule("/src/engine/serialize.ts");
const { validateDesign, designHash } = await server.ssrLoadModule("/src/engine/spec.ts");

const DIR = "designs";
const load = (f) => {
  const { graph, error } = parseGraph(readFileSync(`${DIR}/${f}.json`, "utf8"));
  if (error) throw new Error(`${f}: ${error}`);
  return graph;
};
const clone = (x) => structuredClone(x);
const rowOf = (g, id) => g.stacks.flatMap((s) => s.rows).find((r) => r.id === id);
const custom = (id, label, params, flops, act, note, out = 0) =>
  ({ type: "custom", id, label, params, flops_per_position: flops, act_bytes_per_position: act, out, note });
const ffnCell = (id, hidden, kind = "swiglu", activation = "gelu") =>
  ({ type: "ffn", kind, hidden, activation, bias: false, experts: 1, top_k: 1, norm: "rmsnorm", residual: "pre", id,
     label: `${kind === "swiglu" ? "SwiGLU" : "MLP"} ${hidden}${kind === "mlp" ? ` (${activation === "silu" ? "SiLU" : activation})` : ""}` });
const attnCell = (id, o) => ({
  type: "attention", mode: "self", heads: 8, head_dim: 64, kv_heads: 8, causal: true, window: 0, qk_norm: true, bias: false,
  norm: "rmsnorm", kv_norm: "rmsnorm", residual: "pre", memory: false, dual_key: false, nope: false, kernel: "flash",
  extras: { order_term: false, count_heads: 0, count_max: 10, pos_prior: false, logit_scale: false }, id, ...o,
});

// ------------------------------------------------------------------ the testbed (scripts/run_e33n.py)
const SIZE = {
  s30: { base: "e33n_s30", pre: 2, width: 576, heads: 9, ffn: 1472, emb: 13_493_056, out: 28_948_032 },
  m100: { base: "e33n_m100", pre: 3, width: 896, heads: 14, ffn: 2624, emb: 20_790_528, out: 45_030_272 },
};
const readRow = (z) => `main.L${SIZE[z].pre}`;
const base = (z) => load(SIZE[z].base);

/** The read layer of a size and its attention cell. */
function readLayer(g, z) {
  const r = rowOf(g, readRow(z));
  return { r, a: r.cells.find((c) => c.type === "attention") };
}

/** No page writer, no notebook: the trunk alone (the dense, LSA and E37 arms). */
function trunkOnly(g) {
  g.stacks = g.stacks.filter((s) => s.id === "main");
  g.edges = g.edges.filter((e) => e.kind === "skip");
  g.groups = g.groups.filter((x) => x.id === "loop.read");
  return g;
}

const ARM = {
  window(z) {
    const g = base(z);
    g.edges = g.edges.filter((e) => e.id !== "wire.notes");
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 512, memory: false, dual_key: false, nope: true, label: "window read: the last 512 tokens only, no notes · no RoPE" });
    r.label = "reading layer · window only (notes off)";
    rowOf(g, "notes.out").label = "notes (built, never read)";
    return g;
  },
  dense(z) {
    const g = trunkOnly(base(z));
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 0, memory: false, dual_key: false, nope: true, label: "full causal read of every earlier token, no RoPE" });
    r.label = "reading layer · full read";
    return g;
  },
  full(z) {
    const g = base(z);
    const { r, a } = readLayer(g, z);
    a.label = "notebook read: last 512 raw tokens + every note of every closed page · no RoPE";
    r.label = "reading layer · every note";
    return g;
  },
  notes_open(z) {
    const g = base(z);
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 1024, label: "Notebook read: score notes → reopen the 64 best notes' tokens (≤512) + the notes + last 512 tokens · no RoPE" });
    r.label = "reading layer · notes choose and are read";
    return g;
  },
  pages_open(z) {
    const g = base(z);
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 1024, label: "Page read: score pages (sum of their notes' scores) → reopen the 2 best pages whole (≤512 tokens) + the notes + last 512 tokens · no RoPE" });
    r.label = "reading layer · notes choose pages, notes are read";
    return g;
  },
  lsa(z) {
    const g = trunkOnly(base(z));
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 1028, memory: false, dual_key: false, nope: true, label: "LSA read: last 512 tokens + 4 sinks + 512 tokens from the best buckets · no RoPE" });
    r.cells.unshift(custom(`${r.id}.index`, "bucket index (LSA): ~8 similar keys per bucket, best head scores centres",
      0, SIZE[z].heads * 256 * 64 * 2, 0,
      "No weights. Groups each closed 512-key block by similarity (Ward merging), stores a centre per bucket; a query " +
      "scores the centres (best head), picks buckets until 512 keys. FLOPs shown at 4k (about 256 visible centres on " +
      "average), no gradient."));
    r.label = "reading layer · LSA read";
    return g;
  },
  catalogue(z, { twoLevel = false, reuse = false } = {}) {
    const g = base(z);
    g.stacks.find((s) => s.id === "pages").seq.stride = 256;
    g.stacks.find((s) => s.id === "pages").rows[0].label = "in: 256-token pages, no overlap";
    const { r, a } = readLayer(g, z);
    Object.assign(a, { window: 1028, label: "Catalogue read: notes only choose → open ≤128 notes' tokens (≤512) + 4 empty slots + last 512 tokens · no RoPE" +
      (reuse ? " · a later loop pass may re-open notes an earlier pass opened" : " · each loop pass opens notes not opened before") });
    r.cells.splice(1, 0, custom(`${r.id}.sinks`, "4 learned empty slots + index temperature", 4 * 64 * 2 + 1, 0, 0,
      "sink_k and sink_v [4, 64] and the index loss's log temperature (nn/perceiver_ar_lm.py). The index loss is " +
      "KL(read mass of opened notes || softmax(temperature × note score)) on sampled queries with sampled negative " +
      "notes; training only."));
    if (twoLevel)
      r.cells.unshift(custom(`${r.id}.pages`, "page-first search: keep the 16 best pages, then score only their notes", 0, 0, 0,
        "No weights, no gradient. Page key = mean of the page's token keys; a query keeps its 16 best pages (max-head " +
        "cosine) and scores only their 16 × 32 = 512 notes instead of all of them (lm_open_levels 2, lm_open_pages 16). " +
        "FLOPs not counted."));
    r.label = `reading layer · catalogue (index only${twoLevel ? ", pages first" : ""}${reuse ? ", re-opening allowed" : ""})`;
    rowOf(g, "notes.out").cells.unshift(custom("notes.out.filing", "balanced filing: each token under one note, ≤16 per note", 0, 0, 0,
      "Sinkhorn-balanced assignment of the page's tokens to its 32 notes, capped at 16 tokens per note; decides which " +
      "tokens a note opens. No weights."));
    return g;
  },
  /** E37 concept memory on the dense trunk: per closed 512-key block, `per` balanced content concepts of the reading
   *  layer's own keys; exact reads of the largest estimated contributions plus one summary entry per other concept. */
  e37(z, per, reasoner = false) {
    const g = trunkOnly(base(z));
    const d = SIZE[z].width;
    const { r, a } = readLayer(g, z);
    const keys = 512 / per;
    Object.assign(a, { window: 1280, memory: true, dual_key: false, nope: true,
      label: "concept read: 4 sinks + the raw tokens of its own and the previous 512-block + the members of the concepts " +
        "with the largest estimated mass (512 tokens, exact) + one summary entry per other concept; modelled as a " +
        "1,280-key window plus the summaries; no RoPE" });
    r.label = `reading layer · concept read (${per} concepts per block)`;
    const blocks = {
      id: "blocks", name: "Concept builder (512-key blocks)", role: "encoder", width: d,
      seq: { kind: "pages", of: "main", window: 512, stride: 512 },
      rows: [{ id: "blocks.group", x0_mix: false, label: `group each closed block into ${per} concepts`,
        cells: [custom("blocks.group.kmeans",
          `balanced soft k-means on the reading layer's own keys: ${per} concepts × ${keys} keys, farthest-point seeds, ` +
          "Sinkhorn, no gradient, no weights (rebuilt every loop round)", 0, 32768, 0, "balanced soft k-means on the unit keys")] }],
    };
    const stats = custom("concepts.stats.sum",
      "per concept: count n, mean key, key variance, mean value, mean state z, links to its member tokens (sums, no " +
      `weights)${reasoner ? "; z goes on to the reasoner" : "; the summary is (mean key, mean value)"}`, 0, 0, 0, "concept statistics", reasoner ? 0 : 64);
    const summary = { type: "kv_out", entries: 1, kv_heads: 1, head_dim: 64, projections: 0, in_norms: 0, key_norm: false, id: "concepts.out.kv",
      label: "one summary key + value per concept (1 KV head × 64): the mean key and mean value as they are, no weights" };
    const rows = [{ id: "concepts.stats", x0_mix: false, label: "statistics + member links", cells: [stats] }];
    if (reasoner) {
      rows.push(
        { id: "concepts.in", x0_mix: false, label: "reasoner input",
          cells: [{ type: "linear", out: 512, bias: false, out_norm: "none", id: "concepts.in.lin", label: `z ${d} → 512 (added back every loop: input injection)` }] },
        ...[0, 1].map((i) => ({ id: `concepts.R${i}`, x0_mix: false, label: `reasoner layer ${i}`,
          cells: [attnCell(`concepts.R${i}.attn`, { causal: false, qk_norm: false, kv_norm: "none",
            label: "concepts attend to the concepts of their own and earlier blocks of the document (the builder counts only their own block's)" }),
          ffnCell(`concepts.R${i}.ffn`, 2048, "mlp", "silu")] })),
        { id: "concepts.upd", x0_mix: false, label: "value update",
          cells: [{ type: "norm", kind: "rmsnorm", id: "concepts.upd.norm" },
            { type: "linear", out: 64, bias: false, out_norm: "none", id: "concepts.upd.lin",
              label: "512 → 64: added to each concept's mean value and spread to its member tokens; zero at the start" }] });
    }
    rows.push({ id: "concepts.out", x0_mix: false, label: "summary entries", cells: [summary] });
    g.stacks.push(blocks, { id: "concepts", name: `Concepts (${per} per block)${reasoner ? " + reasoner" : ""}`, role: "latent", width: d,
      seq: { kind: "slots", of: "blocks", per_page: per }, rows });
    g.edges.push(
      { id: "w.blocks.in", kind: "input", from: { stack: "main", row: `main.L${SIZE[z].pre - 1}` }, to: { stack: "blocks" } },
      { id: "w.concepts.in", kind: "input", from: { stack: "blocks" }, to: { stack: "concepts" } },
      { id: "w.read", kind: "source", from: { stack: "concepts" }, to: { stack: "main", cell: a.id } });
    if (reasoner) g.groups.push({ id: "loop.reasoner", kind: "loop", stack: "concepts", rows: ["concepts.R0", "concepts.R1"], count: 2, round_embedding: false, feedback: "none" });
    return g;
  },
};

// ------------------------------------------------------------------ the E31 capability-check champion (e31_li_m1)
function champion() {
  const d = 960, V = 13, F = 1920;
  const layer = (i, label, o = {}, extra = []) => ({
    id: `main.L${i}`, x0_mix: true, label,
    cells: [...extra, attnCell(`main.L${i}.attn`, { heads: 15, kv_heads: 1, window: 16, kv_norm: "rmsnorm", label: "local window 16", ...o }), ffnCell(`main.L${i}.ffn`, F)],
  });
  const ve = (i) => custom(`main.L${i}.ve`, "value embedding: a 13 × 16 table → 64, added to the values", 13 * 16 + 64 * 16 + 1, 0, 0,
    "value_embed [13, 16] + value_proj [16 → 64] + value_lambda (one scalar); E18's value embeddings on the pre-encoder " +
    "and the global read (ArchSpec.value_embed_layers (0, 1)). Not a builder field.");
  const main = {
    id: "main", name: "Reader", role: "decoder", width: d, seq: { kind: "tokens" },
    rows: [
      { id: "main.input", x0_mix: false, label: "input", cells: [
        { type: "embedding", vocab: V, dim: 128, ngram_tables: 0, ngram_buckets: 8192, proj: "gated_mlp", out_norm: "rmsnorm", id: "main.input.emb", label: "DNA-task alphabet 13 × 128 + input MLP" },
        { type: "positional", kind: "rope", theta: 10000, max_len: 1024, id: "main.input.pos", label: "RoPE" }] },
      layer(0, "pre-encoder · local 16", {}, [ve(0)]),
      layer(1, "global read · last 256 tokens + notes", { window: 256, memory: true, dual_key: true,
        label: "global read: the last 256 tokens (message_raw_window) + the notes of closed pages; answers read the notes only (exclusive after QUERY)" }, [ve(1)]),
      layer(2, "stack 0 · local 16"),
      layer(3, "stack 1 · local 16"),
      { id: "main.head", x0_mix: false, label: "head", cells: [{ type: "lm_head", vocab: V, tied: false, bias: false, final_norm: "rmsnorm", z_loss: 1e-4, softcap: 0, chunked: false, id: "main.head.lm", label: "untied output layer 13" }] },
    ],
  };
  const t = base("s30");
  const pages = clone(t.stacks.find((s) => s.id === "pages"));
  pages.rows[0].cells[0].label = "token table (128) → writer width 256";
  const notes = clone(t.stacks.find((s) => s.id === "notes"));
  notes.rows.find((r) => r.id === "notes.out").cells[0].label = "notes: one key + one value each (1 KV head × 64, m = 1)";
  return {
    schema: "nn-builder/graph", version: 1, name: "", description: "",
    stacks: [main, pages, notes],
    edges: [
      { id: "main.skip0", kind: "skip", from: { stack: "main", row: "main.L1" }, to: { stack: "main", row: "main.L2" }, weight: "scalar" },
      { id: "main.skip1", kind: "skip", from: { stack: "main", row: "main.L0" }, to: { stack: "main", row: "main.L3" }, weight: "scalar" },
      { id: "wire.table", kind: "input", from: { stack: "main", cell: "main.input.emb", port: "table" }, to: { stack: "pages" } },
      { id: "wire.pages", kind: "source", from: { stack: "pages" }, to: { stack: "notes", cell: "notes.read.attn" } },
      { id: "wire.notes", kind: "source", from: { stack: "notes" }, to: { stack: "main", cell: "main.L1.attn" } },
    ],
    groups: [{ id: "loop.writer", kind: "loop", stack: "notes", rows: ["notes.read", "notes.ffn"], count: 2, round_embedding: false, feedback: "none" }],
    run: { ...t.run, row_length: 1024, rows_per_step: 128, micro_batch: 32, token_budget: 0, gpu: "rtx3090", gpu_count: 1 },
  };
}

// ------------------------------------------------------------------ the catalogue: one entry per design file
const TB = "e33n-lsa-testbed", A10 = "arch-10m";
const TRAIN30 = "Protocol: 1.5B FineWeb-Edu tokens in rows of 4,096, then a 0.2B LongDocs-lite decay in rows of 16,384, 64 rows per step (E33p stage 1).";
const TRAIN100 = "Protocol (E37 stage A screen): 1.0B FineWeb-Edu tokens at 4k, then 0.2B LongDocs-lite at 16k, 128 rows per step.";
const s30 = (arm) => `scripts/run_e33n.py --arm ${arm} --size s30 (FFN 1472 from fit())`;
const m100 = (arm) => `scripts/run_e33n.py --arm ${arm} --size m100 (FFN 2624 from fit())`;
const SPEC = {
  E33n: "docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md",
  E33p: "docs/experiments_specs/ahead/E33p_note_indexed_exact_read.md",
  E33t: "docs/experiments_specs/ahead/E33t_catalogue_read.md",
  E37: "docs/experiments_specs/ahead/E37_concept_memory.md",
  E31: "docs/experiments_specs/ahead/E31_sliding_window_latent_memory.md",
  LSA: "docs/1_Strategy_and_Plans (architectures review); arXiv 2610.42147",
  E33: "docs/experiments_specs/ahead/E33_iterative_concept_reads.md",
};

const UX = {
  topk: "The top-k cut (lm_read_topk 64) is chosen without gradient and has no builder field; compute is drawn as the full read's, which the trained kernel also spends.",
  notes: "The per-query choice of notes has no weights and no builder field; the opened tokens are drawn as a 1,024-key window (512 window + 512 opened).",
  pages: "The page choice has no weights and no builder field; drawn like notes_open as a 1,024-key window.",
  lsa: "The bucket index and the choice have no weights and no gradient: a 0-parameter custom cell carries their FLOPs; the read is drawn as a 1,028-key window.",
  e37: "Grouping, statistics and the choice have no weights (0-parameter custom cells); the summaries are the concepts' mean key and value (a notes cell with no projections); the block window and opened tokens are drawn as a 1,280-key window. The summaries' length scale (concept_sum_len: past that length their total weight is held at its level) has no weights and no builder field. The 30M runs trained without it and use it as an eval switch (arch-10m e9b37d2); from stage A the concept arms train with it set to the decay's row length, 16,384, where it is inert (arch-10m 7c7158c, 8 Nov).",
};
const STAGE_A = "Stage A has the author's go-ahead (8 Nov, ≈ $360): one 4-GPU AWS spot node trains the arms one after another, in the order e37a, lsa, dense, window, notes_open, e37b (arch-10m 7c7158c); not scored yet.";
UX.e37b = `${UX.e37} The reasoner's attention over earlier blocks' concepts is counted for its own block only.`;

const ENTRIES = [
  // ---- the 30M LSA testbed: what is training or being decided now
  { file: "e33p_window_s30", id: "E33p", plain: "window-only floor", size: "s30", arm: "window", build: () => ARM.window("s30"),
    parent: "e33n_s30", what: "the reading layer reads only its last 512 tokens; the page writer is still built (its weights are counted) but its notes never reach the read (message_override none)",
    status: "done", branch: TB, commit: "8701b17", spec: "E33p", date: "2026-11-06",
    extra: "The floor of the window-to-dense gap. First trained in E33n (5 Nov); retrained in E33p stage 1 as a same-protocol control (seed 42 at home, seed 1 on AWS)." },
  { file: "e33p_dense_s30", id: "E33p", plain: "dense ceiling", size: "s30", arm: "dense", build: () => ARM.dense("s30"),
    parent: "e33n_s30", what: "no page writer and no notebook; the reading layer reads every earlier token (full causal attention)",
    status: "done", branch: TB, commit: "8701b17", spec: "E33p", date: "2026-11-06",
    extra: "The ceiling (100 % of the gap). E33n's dense run (5 Nov) trained with one loop round (factory bug); E33p retrained it with 4." },
  { file: "e33n_full_s30", id: "E33n", plain: "full note read", size: "s30", arm: "e33", build: () => ARM.full("s30"),
    parent: "e33n_s30", what: "every query reads every note of every closed page (no top-k cut) beside its last 512 tokens",
    status: "failed", branch: TB, commit: "8701b17", spec: "E33n", date: "2026-11-05",
    extra: "Result (E33n stage 1): notes recover 5.5 % of the window-to-dense gap at 16k (kill line 15 %); E33p's retrained copy 4 %." },
  { file: "e33n_s30", id: "E33n", plain: "top-64 note read", size: "s30", arm: "e33n", static: true,
    parent: "e33m_plain_30m", what: "E33's notebook on the LSA testbed (GPT-2 vocabulary, natural long text); each query and head reads only its 64 best notes",
    status: "failed", branch: TB, commit: "27577bc", spec: "E33n", date: "2026-11-05",
    extra: "Result: 6.1 % of the gap at 16k, the same as the full read; 97 % of the gap is far-copy tokens. Closed at stage 1 (K1)." ,
    unexpressed: UX.topk },
  { file: "e33p_notes_open_30m", id: "E33p", plain: "notes open their tokens", size: "s30", arm: "notes_open", build: () => ARM.notes_open("s30"),
    parent: "e33n_s30", what: "tied note keys; each query opens the raw tokens of its 64 best notes (≤ 512 tokens) and still reads every note as gist",
    status: "done", branch: TB, commit: "7f64a09", spec: "E33p", date: "2026-11-06",
    extra: "Result (stage 1, seed 1, AWS): 39 % of the gap at 16k, 21 % at 64k (LSA port: 99 %). Measured causes: 30 % of notes empty, overlapping pages reopened, summaries take a third of the attention.",
    unexpressed: UX.notes },
  { file: "e33p_pages_open_s30", id: "E33p", plain: "notes open whole pages", size: "s30", arm: "pages_open", build: () => ARM.pages_open("s30"),
    parent: "e33p_notes_open_30m", what: "the same index, coarser: a page's score is the sum of its notes' scores and each query opens its 2 best pages whole (≤ 512 tokens)",
    status: "done", branch: TB, commit: "7f64a09", spec: "E33p", date: "2026-11-06",
    extra: "Result (stage 1): 34 % of the gap at 16k; RULER 4k mean 0.105 (LSA port 0.139).",
    unexpressed: UX.pages },
  { file: "e33p_lsa_port_30m", id: "E33p", plain: "LSA port", size: "s30", arm: "lsa", build: () => ARM.lsa("s30"),
    parent: "e33p_dense_s30", what: "LSA's released read ported to the dense trunk: buckets of ~8 similar keys per closed 512-key block, a query opens the best buckets (512 keys) + 4 sinks + its last 512 tokens",
    status: "done", branch: TB, commit: "7f64a09", spec: "E33p", date: "2026-11-06",
    extra: "Result (stage 1): 99 % of the gap at 16k, beats dense at 128k. The reference every new read is judged against.",
    unexpressed: UX.lsa },
  { file: "e33p_lsa_s30", id: "E33p", plain: "LSA port (alias)", size: "s30", arm: "lsa", build: () => ARM.lsa("s30"), alias: "e33p_lsa_port_30m",
    parent: "e33p_lsa_port_30m", what: "the same arm as e33p_lsa_port_30m, drawn first by the 10M-context session; kept so that branch merges cleanly",
    status: "done", branch: A10, commit: "3cae268", spec: "E33p", date: "2026-11-07", unexpressed: UX.lsa },
  { file: "e33t_catalogue_30m", id: "E33t", plain: "catalogue read", size: "s30", arm: "catalogue", build: () => ARM.catalogue("s30"),
    parent: "e33p_notes_open_30m", what: "pages without overlap; each token filed under one note (≤ 16 per note); the notes are only an index (their values are not read); up to 128 notes' tokens (≤ 512) open per query; a new place each loop pass; 4 learned empty slots; an index loss",
    status: "done", branch: TB, commit: "47339bf", spec: "E33t", date: "2026-11-07",
    extra: "Trained 7 Nov 11:24–16:05 UTC (AWS, seed 1). Preliminary result: 75 % of the gap at 16k, 99 % at 64k. Probe: opens a planted password on pass 1 (94 % at 4k) but rarely returns to it on passes 2–4.",
    unexpressed: "Balanced filing, the per-pass exclusion and the index loss have no weights (0-parameter custom cells); the read is drawn as a 1,028-key window." },
  { file: "e33t_catalogue_2l_s30", id: "E33t", plain: "two-level catalogue", size: "s30", arm: "catalogue_2l", build: () => ARM.catalogue("s30", { twoLevel: true }),
    parent: "e33t_catalogue_30m", what: "the catalogue searched in two levels: a query keeps its 16 best pages first, then scores only their 512 notes",
    status: "training", branch: TB, commit: "47339bf", spec: "E33t", date: "2026-11-07",
    extra: "Queued after the catalogue on the AWS node (stage A); not scored yet.",
    unexpressed: "The page-first search has no weights and no builder field (a 0-parameter custom cell, FLOPs not counted)." },
  { file: "e33t_catalogue_reuse_s30", id: "E33t", plain: "catalogue, re-opening allowed", size: "s30", arm: "catalogue_reuse", build: () => ARM.catalogue("s30", { reuse: true }),
    parent: "e33t_catalogue_30m", what: "the pass exclusion off in training (lm_open_new_per_pass False): a later loop pass may re-open notes an earlier pass opened",
    status: "training", branch: TB, commit: "26d18f4", spec: "E33t", date: "2026-11-07",
    extra: "Approved 7 Nov ~21:00 UTC after the planted-password probe; chained on the AWS node after the seed-1 window and LSA controls (infra/aws/jobs/e33t_reuse.sh).",
    unexpressed: "Same weights and drawing as the catalogue; the difference is a training switch with no builder field (named in the read's label)." },
  // ---- E37 concept memory (the 10M-context branch)
  { file: "e37a_s30", id: "E37", plain: "concept memory", size: "s30", arm: "e37a", build: () => ARM.e37("s30", 64),
    parent: "e33p_dense_s30", what: "each closed 512-key block of the reading layer's own keys is grouped by content into 64 balanced concepts of 8 keys; a query reads exactly the tokens of its strongest concepts (512) and one summary entry for every other concept",
    status: "done", branch: A10, commit: "95ee75d", spec: "E37", date: "2026-11-07",
    extra: "30M home screen (seed 42, done 7 Nov 23:25 UTC): 100 % of the gap at 16k, 86 % at 64k (LSA port 99 / 127 %); the summaries cause the fade past the training length.",
    unexpressed: UX.e37 },
  { file: "e37a_c32_s30", id: "E37", plain: "concept memory, 32 keys per concept", size: "s30", arm: "e37a_c32", build: () => ARM.e37("s30", 16),
    parent: "e37a_s30", what: "compression ladder: 16 concepts of 32 keys per block (same 512 opened tokens; only the summaries get coarser)",
    status: "training", branch: A10, commit: "6e49738", spec: "E37", date: "2026-11-07",
    extra: "Training on Polonez GPUs 0+1; main phase due about 13:10 UTC 8 Nov; its evals were not written yet at arch-10m 7c7158c (13:40 UTC).", unexpressed: UX.e37 },
  { file: "e37a_c128_s30", id: "E37", plain: "concept memory, 128 keys per concept", size: "s30", arm: "e37a_c128", build: () => ARM.e37("s30", 4),
    parent: "e37a_s30", what: "compression ladder: 4 concepts of 128 keys per block (same 512 opened tokens)",
    status: "done", branch: A10, commit: "6e49738", spec: "E37", date: "2026-11-07",
    extra: "Done 8 Nov 09:10 UTC: 81 % of the gap at 16k, 65 % at 64k (e37a 100 / 86 %): coarse concepts cost ~19 points.", unexpressed: UX.e37 },
  { file: "e37b_s30", id: "E37", plain: "concept memory + reasoner", size: "s30", arm: "e37b", build: () => ARM.e37("s30", 64, true),
    parent: "e37a_s30", what: "a 2-layer, width-512 reasoner over the concepts, looped twice; its update is added to each concept's mean value and to its member tokens' values",
    status: "done", branch: A10, commit: "95ee75d", spec: "E37", date: "2026-11-07",
    extra: "Done 8 Nov 01:11 UTC: 99.8 % at 16k, 89 % at 64k; −0.2 points vs e37a at 16k, so the reasoner is shelved for stage B (K2).",
    unexpressed: UX.e37b },
  // ---- 100M: E33n's size and the E37 stage A arms (queued on the AWS node, 8 Nov)
  { file: "e33n_m100", id: "E33n", plain: "top-64 note read", size: "m100", arm: "e33n", static: true,
    parent: "e33n_s30", what: "the 30M arm at 100M (width 896, 3 pre layers, 7 stack layers, FFN 2624)",
    status: "parked", branch: TB, commit: "27577bc", spec: "E33n", date: "2026-11-05",
    extra: "Never trained: E33n closed at stage 1 (30M).",
    unexpressed: UX.topk },
  { file: "e37_window_m100", id: "E37", plain: "window-only floor", size: "m100", arm: "window", build: () => ARM.window("m100"),
    parent: "e33p_window_s30", what: "the 30M floor at 100M", status: "queued", branch: A10, commit: "8701b17", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: STAGE_A },
  { file: "e37_dense_m100", id: "E37", plain: "dense ceiling", size: "m100", arm: "dense", build: () => ARM.dense("m100"),
    parent: "e33p_dense_s30", what: "the 30M ceiling at 100M", status: "queued", branch: A10, commit: "8701b17", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: STAGE_A },
  { file: "e37_lsa_m100", id: "E37", plain: "LSA port", size: "m100", arm: "lsa", build: () => ARM.lsa("m100"),
    parent: "e33p_lsa_port_30m", what: "the LSA port at 100M", status: "queued", branch: A10, commit: "7f64a09", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: STAGE_A, unexpressed: UX.lsa },
  { file: "e37_notes_open_m100", id: "E37", plain: "notes open their tokens", size: "m100", arm: "notes_open", build: () => ARM.notes_open("m100"),
    parent: "e33p_notes_open_30m", what: "E33p's notes_open at 100M (the best E33-family read on natural text; the catalogue may replace it)", status: "queued", branch: A10, commit: "7f64a09", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: STAGE_A, unexpressed: UX.notes },
  { file: "e37a_m100", id: "E37", plain: "concept memory", size: "m100", arm: "e37a", build: () => ARM.e37("m100", 64),
    parent: "e37a_s30", what: "e37a at 100M", status: "queued", branch: A10, commit: "95ee75d", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: `${STAGE_A} Trains with the length scale on and is also scored with it off.`, unexpressed: UX.e37 },
  { file: "e37b_m100", id: "E37", plain: "concept memory + reasoner", size: "m100", arm: "e37b", build: () => ARM.e37("m100", 64, true),
    parent: "e37b_s30", what: "e37b at 100M (the reasoner's input map is 896 → 512)", status: "queued", branch: A10, commit: "95ee75d", spec: "E37", date: "2026-11-07", train: TRAIN100,
    extra: `${STAGE_A} Last in the order: the 30M screen shelved the reasoner for stage B (S3 missed, K2 met), but the launch list still includes it. Trains with the length scale on.`, unexpressed: UX.e37b },
  // ---- the champion and the references
  { file: "e31_li_m1_cap30m", id: "E31", plain: "latent-memory champion (capability checks)", size: "cap30m", arm: "e31_li_m1", build: champion,
    parent: "e31c_plain_30m (preset)", what: "the capability-check champion: E31's sliding-window latent memory on the DNA tasks — width 960, 4 layers (pre-encoder, global read, 2 stack layers), the E31 page writer (256-token pages every 192, 32 notes per page, one key and value per note), reads after QUERY go to the notes only",
    status: "done", branch: "dev", commit: "9253d7a", spec: "E31", date: "2026-09-27",
    extra: "Champion of the capability checks (no-harm reference for every new variant, analysis/capability_board.py --champion). Built by verification/bapo_capability_probe.py with the E31 platform args (scripts/study_plans/e30_vs_e31.py BASE: hidden 960, token embedding 128, no n-grams, raw window 256) on the 13-symbol DNA alphabet; the results ledger stores 35,129,900 for its runs.",
    unexpressed: "The value embeddings on layers 0 and 1 (1,233 parameters each) are custom cells; the exclusive-after-QUERY mask has no builder field.",
    train: "Protocol: the capability checks (C0–C5 core package, seeds 0–2, trained from scratch per task; rows of 1k–16k)." },
  { file: "lsa_paper_native_0p8b", id: "LSA", plain: "LSA as published (Qwen3.5-0.8B)", size: "0p8b", arm: "—", static: true,
    parent: "—", what: "the outside reference: Gated DeltaNet hybrid, 24 layers of width 1024, the LSA read on every fourth layer",
    status: "done", branch: TB, commit: "1769cf9", spec: "LSA", date: "2026-10-30",
    extra: "Not ours: never trained here.",
    unexpressed: "Gated DeltaNet layers and the attention output gate are custom cells (exact parameters, approximate FLOPs)." },
  { file: "e33m_plain_30m", id: "E33", plain: "notebook (text checks)", size: "30m", arm: "—", static: true,
    parent: "e31c_plain_30m (preset)", what: "E33's notebook on the text checks (4,096-token vocabulary): the E31c writer plus a 4-round loop over the read layer and the next, the order term and count heads",
    status: "done", branch: "dev", commit: "edb1c00", spec: "E33", date: "2026-10-24",
    extra: "A saved copy of the built-in preset e33m_plain_30m (trainer-exact, tests/params.test.ts)." },
];

// ------------------------------------------------------------------ build, check, write
const M = (n) => `${(n / 1e6).toFixed(1)}M`;
function counts(g) {
  const a = analyze(g);
  const main = a.stacks.find((s) => s.id === "main");
  const cells = main.rows.flatMap((r) => r.cells);
  const p = (t) => cells.find((c) => c.type === t)?.params ?? 0;
  const width = g.stacks.find((s) => s.id === "main").width;
  return { a, total: a.params.total, core: a.params.total - p("embedding") - (p("lm_head") - width), emb: p("embedding"), out: p("lm_head") - width };
}

const results = [];
for (const e of ENTRIES) {
  const g = e.static ? load(e.file) : e.build();
  const c = counts(g);
  const sizeName = { s30: "30M testbed", m100: "100M testbed", cap30m: "capability-check 30M", "0p8b": "0.8B", "30m": "30M text checks" }[e.size] ?? e.size;
  g.name = `${e.id} · ${e.plain} · ${M(c.total)} (${M(c.core)} core)`;
  if (!e.static || e.file.startsWith("e33n_")) {
    const params = `Parameters: ${c.total.toLocaleString("en-US")} in total, ${c.core.toLocaleString("en-US")} without the word tables ` +
      `(input embedding ${c.emb.toLocaleString("en-US")} + output layer ${c.out.toLocaleString("en-US")}); exact against the trainer's own count` +
      (e.size === "cap30m" ? " (evaluation/bapo_models.build_model as the exam runner calls it, = the ledger's stored count)." : " (evaluation/text_checks.count_params on " + (e.size === "m100" ? m100(e.arm) : s30(e.arm)) + ").") +
      (e.size.startsWith("s30") || e.size === "s30" ? " The testbed is called “30M” after the notebook arms' 30.2M without word tables (Kaplan et al.'s convention)." : "");
    g.description = [
      `${e.id} ${e.arm !== "—" ? `arm ${e.arm}` : ""} — ${e.plain}, ${sizeName}.`.replace("  ", " "),
      e.alias ? `Alias of ${e.alias}: identical architecture.` : "",
      `New against ${e.parent}: ${e.what}.`,
      e.train ?? (e.size === "s30" ? TRAIN30 : ""),
      `Built ${e.date}; status ${e.status} (as of 2026-11-08). ${e.extra ?? ""}`.trim(),
      `Spec: ${SPEC[e.spec] ?? e.spec}. Source: origin/${e.branch} @ ${e.commit}.`,
      e.unexpressed ? `Not a builder field: ${e.unexpressed}` : "",
      params,
    ].filter(Boolean).join(" ");
  }
  else if (!g.description.includes("without the word table"))
    g.description = `${g.description.trim()} Parameters: ${c.total.toLocaleString("en-US")} in total, ${c.core.toLocaleString("en-US")} without the word tables ` +
      `(input embedding ${c.emb.toLocaleString("en-US")} + output layer ${c.out.toLocaleString("en-US")}). ${e.extra ?? ""}`.trim();
  const v = validateDesign(g);
  if (v.errors.length) throw new Error(`${e.file}: ${JSON.stringify(v.errors)}`);
  const out = toExport(g);
  const path = `${DIR}/${e.file}.json`;
  if (existsSync(path)) {
    const old = JSON.parse(readFileSync(path, "utf8"));
    if (old.derived?.hash === out.derived.hash && old.derived?.params_total === out.derived.params_total && old.name === out.name && old.description === out.description)
      out.derived.exported_at = old.derived.exported_at;
  }
  if (!CHECK) writeFileSync(path, JSON.stringify(out, null, 2) + "\n");
  results.push({ e, c, hash: out.derived.hash, warnings: v.warnings.length });
}

// the catalogue table
const BR = { [TB]: "e33n-lsa-testbed", [A10]: "arch-10m", dev: "dev" };
const table = [
  "| design | experiment · plain name | new against its parent | status | source (branch @ commit) · spec | total | core | date |",
  "|---|---|---|---|---|---:|---:|---|",
  ...results.map(({ e, c }) => `| \`${e.file}\` | **${e.id}** · ${e.plain}${e.arm !== "—" ? ` (\`${e.arm}\`)` : ""} | ${e.what}${e.unexpressed ? " ⚑" : ""} | ${e.status} | ${BR[e.branch]} @ \`${e.commit}\` · ${(SPEC[e.spec] ?? e.spec).split("/").at(-1)} | ${c.total.toLocaleString("en-US")} | ${c.core.toLocaleString("en-US")} | ${e.date} |`),
].join("\n");
const CAT = `${DIR}/CATALOGUE.md`;
if (!CHECK && existsSync(CAT)) {
  const text = readFileSync(CAT, "utf8");
  const a = "<!-- designs:start -->", b = "<!-- designs:end -->";
  if (text.includes(a) && text.includes(b))
    writeFileSync(CAT, text.slice(0, text.indexOf(a) + a.length) + "\n" + table + "\n" + text.slice(text.indexOf(b)));
}
for (const { e, c, hash, warnings } of results)
  console.log(e.file.padEnd(26), c.total.toLocaleString("en-US").padStart(12), c.core.toLocaleString("en-US").padStart(12), hash, warnings ? `${warnings} warnings` : "");
await server.close();

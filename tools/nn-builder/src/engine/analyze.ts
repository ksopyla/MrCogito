// Whole-graph estimates: parameters, forward/backward FLOPs, GPU memory, GPU count and wall-clock time.
import { cellCost, embeddingProjFlops, type Bytes, type CellCost, type NotesInfo, type Part, type SeqInfo, type SourceInfo } from "./cells";
import { pageGeometry } from "./geometry";
import { CAL, GiB, MEMORY_CAL, OPTIMIZERS, PRECISIONS, gpuById } from "./hardware";
import type { Cell, EmbeddingCell, Endpoint, Graph, Group, KvOutCell, LmHeadCell, Row, Stack } from "./types";

export interface Issue {
  where: string; // a cell, row, stack, edge or group id
  msg: string;
}

export interface CellResult {
  id: string;
  type: Cell["type"];
  stackId: string;
  rowId: string;
  params: number; // counted once for loops, × copies for repeats
  parts: Part[];
  execs: number; // times it runs per row
  fwdPerToken: number;
  bwdPerToken: number;
  keysFwdPerToken: number; // the part that grows with the keys each query sees
  actBytesPerRow: number; // all executions
  widthIn: number;
  widthOut: number;
  irregular: boolean;
  issues: string[];
}

export interface RowResult {
  id: string;
  index: number;
  params: number;
  rowExtras: number; // x0 mix and skip weights
  execs: number;
  copies: number;
  fwdPerToken: number;
  actBytesPerRow: number;
  checkpointed: boolean;
  cells: CellResult[];
}

export interface GroupResult {
  id: string;
  kind: Group["kind"];
  count: number;
  params: number; // round embedding + feedback gate (loops)
  fwdPerToken: number; // the feedback lens
  actBytesPerRow: number;
}

export interface StackResult {
  id: string;
  name: string;
  role: Stack["role"];
  P: number;
  seq: SeqInfo;
  widthOut: number;
  params: number;
  fwdPerToken: number;
  rows: RowResult[];
  groups: GroupResult[];
}

export interface MemoryResult {
  microBatch: number;
  maxMicroBatch: number;
  weights: number;
  grads: number;
  master: number;
  optimizer: number;
  castCopies: number;
  activations: number;
  workspace: number;
  context: number;
  total: number;
  capacity: number;
  fits: boolean;
  actBytesPerRow: number; // as configured (with or without checkpointing), before the overhead factor
  actBytesPerRowNoCkpt: number;
  actBytesPerRowCkpt: number;
}

export interface Analysis {
  S: number;
  issues: Issue[];
  params: { total: number; byType: Record<string, number>; byStack: Record<string, number> };
  stacks: StackResult[];
  cells: Map<string, CellResult>;
  flops: {
    fwdPerToken: number;
    bwdPerToken: number;
    recomputePerToken: number;
    trainPerToken: number;
    keysFwdPerToken: number;
    perStep: number;
    perRun: number;
    irregularTrainPerToken: number;
  };
  memory: MemoryResult;
  time: {
    tokensPerSecondPerGpu: number;
    tokensPerSecond: number;
    hours: number;
    gpuHours: number;
    steps: number;
    accumulation: number;
    mfu: number;
  };
  gpus: { minForMemory: number; forDeadline: number };
}

interface Index {
  stacks: Map<string, Stack>;
  cellHome: Map<string, { stack: Stack; row: Row; cell: Cell }>;
  rowHome: Map<string, { stack: Stack; row: Row; index: number }>;
}

function buildIndex(g: Graph): Index {
  const stacks = new Map(g.stacks.map((s) => [s.id, s]));
  const cellHome = new Map<string, { stack: Stack; row: Row; cell: Cell }>();
  const rowHome = new Map<string, { stack: Stack; row: Row; index: number }>();
  for (const s of g.stacks)
    s.rows.forEach((row, index) => {
      rowHome.set(row.id, { stack: s, row, index });
      for (const cell of row.cells) cellHome.set(cell.id, { stack: s, row, cell });
    });
  return { stacks, cellHome, rowHome };
}

// ------------------------------------------------------------------ geometry and widths (no costs needed)

function seqInfos(g: Graph, S: number, issues: Issue[]): Map<string, SeqInfo> {
  const out = new Map<string, SeqInfo>();
  const byId = new Map(g.stacks.map((s) => [s.id, s]));
  const visiting = new Set<string>();
  const visit = (id: string): SeqInfo => {
    const done = out.get(id);
    if (done) return done;
    const s = byId.get(id);
    if (!s || visiting.has(id)) {
      issues.push({ where: id, msg: s ? "stack sequences form a cycle" : `unknown stack ${id}` });
      return { kind: "tokens", P: S };
    }
    visiting.add(id);
    let info: SeqInfo;
    const q = s.seq;
    if (q.kind === "tokens") info = { kind: "tokens", P: S };
    else if (q.kind === "source") info = { kind: "source", P: q.length > 0 ? q.length : S };
    else if (q.kind === "latents") info = { kind: "latents", P: q.count };
    else if (q.kind === "pages") {
      const base = visit(q.of);
      const pages = pageGeometry(base.P, q.window, q.stride);
      info = { kind: "pages", P: pages.count * pages.window, pages, pageWindowCfg: q.window, tokensOf: q.of };
    } else {
      const base = visit(q.of);
      if (base.kind !== "pages" || !base.pages) {
        issues.push({ where: id, msg: "slots must sit on a pages stack" });
        info = { kind: "slots", P: q.per_page, perPage: q.per_page, pageWindowCfg: 0 };
      } else
        info = {
          kind: "slots",
          P: base.pages.count * q.per_page,
          pages: base.pages,
          perPage: q.per_page,
          pageWindowCfg: base.pageWindowCfg,
          tokensOf: base.tokensOf,
          pagesStack: q.of,
        };
    }
    visiting.delete(id);
    out.set(id, info);
    return info;
  };
  for (const s of g.stacks) visit(s.id);
  return out;
}

function cellWidthOut(cell: Cell, w: number, stackWidth: number): number {
  switch (cell.type) {
    case "embedding":
      return cell.proj === "none" ? cell.dim : stackWidth;
    case "linear":
      return cell.out || w;
    case "custom":
      return cell.out || w;
    case "latent_init":
      return stackWidth;
    default:
      return w;
  }
}

/** Stream width entering every row and leaving every row, following input wires across stacks. */
function widths(g: Graph, ix: Index, issues: Issue[]) {
  const rowIn = new Map<string, number>();
  const rowOut = new Map<string, number>();
  const stackOut = new Map<string, number>();
  const visiting = new Set<string>();
  const tapWidth = (e: Endpoint): number => {
    if (e.cell) {
      const home = ix.cellHome.get(e.cell);
      if (!home) return 0;
      if (e.port === "table" && home.cell.type === "embedding") return home.cell.dim;
      visit(home.stack.id);
      let w = rowIn.get(home.row.id) ?? 0;
      for (const c of home.row.cells) {
        w = cellWidthOut(c, w, home.stack.width);
        if (c.id === e.cell) break;
      }
      return w;
    }
    visit(e.stack);
    return e.row ? rowOut.get(e.row) ?? 0 : stackOut.get(e.stack) ?? 0;
  };
  const visit = (id: string) => {
    if (stackOut.has(id)) return;
    const s = ix.stacks.get(id);
    if (!s) return;
    if (visiting.has(id)) {
      issues.push({ where: id, msg: "input wires form a cycle" });
      stackOut.set(id, s.width);
      return;
    }
    visiting.add(id);
    const input = g.edges.find((e) => e.kind === "input" && e.to.stack === id);
    let w = input ? tapWidth(input.from) : 0;
    for (const row of s.rows) {
      rowIn.set(row.id, w);
      for (const c of row.cells) w = cellWidthOut(c, w, s.width);
      rowOut.set(row.id, w);
    }
    visiting.delete(id);
    stackOut.set(id, w);
  };
  for (const s of g.stacks) visit(s.id);
  return { rowIn, rowOut, stackOut, tapWidth };
}

function notesOf(s: Stack, seq: SeqInfo): NotesInfo | undefined {
  const last = s.rows.flatMap((r) => r.cells).at(-1);
  if (!last || last.type !== "kv_out" || !seq.pages) return undefined;
  const kv = last as KvOutCell;
  const perPage = (seq.perPage ?? 0) * kv.entries;
  return { total: seq.pages.count * perPage, perPage, pages: seq.pages, tokensOf: seq.tokensOf, kvHeads: kv.kv_heads, headDim: kv.head_dim };
}

// ------------------------------------------------------------------ the analysis

export function analyze(g: Graph, opts: { S?: number } = {}): Analysis {
  const run = g.run;
  const S = Math.max(1, Math.round(opts.S ?? run.row_length));
  const prec = PRECISIONS[run.precision] ?? PRECISIONS.amp_bf16;
  const bytes: Bytes = { act: prec.act, resid: prec.resid };
  const issues: Issue[] = [];
  const ix = buildIndex(g);
  const seqs = seqInfos(g, S, issues);
  const W = widths(g, ix, issues);
  const cells = new Map<string, CellResult>();

  const groupOfRow = new Map<string, Group>();
  for (const grp of g.groups) {
    const s = ix.stacks.get(grp.stack);
    if (!s) {
      issues.push({ where: grp.id, msg: "group on an unknown stack" });
      continue;
    }
    const idx = grp.rows.map((r) => s.rows.findIndex((x) => x.id === r)).sort((a, b) => a - b);
    if (idx.some((i) => i < 0) || idx.some((v, i) => i > 0 && v !== idx[i - 1] + 1))
      issues.push({ where: grp.id, msg: "a group must cover neighbouring rows of one stack" });
    for (const r of grp.rows) {
      if (groupOfRow.has(r)) issues.push({ where: grp.id, msg: "a row sits in two groups" });
      else groupOfRow.set(r, grp);
    }
  }

  const sourceFor = (cellId: string): SourceInfo | undefined => {
    const e = g.edges.find((x) => x.kind === "source" && x.to.cell === cellId);
    if (!e) return undefined;
    const home = ix.stacks.get(e.from.stack);
    const seq = seqs.get(e.from.stack);
    if (!home || !seq) return undefined;
    return { stackId: home.id, width: W.tapWidth(e.from), P: seq.P, seq, notes: notesOf(home, seq) };
  };

  const stackResults: StackResult[] = [];
  let fwd = 0, bwd = 0, recompute = 0, keysFwd = 0, irregularTrain = 0, irregularRecompute = 0;
  let actNoCkpt = 0, actCkptSaved = 0, actCkptPeakRow = 0, workspace = 0;
  const byType: Record<string, number> = {};
  const byStack: Record<string, number> = {};

  for (const s of g.stacks) {
    const seq = seqs.get(s.id)!;
    const embCell = s.rows.flatMap((r) => r.cells).find((c) => c.type === "embedding") as (EmbeddingCell & { id: string }) | undefined;
    const headCell = s.rows.flatMap((r) => r.cells).find((c) => c.type === "lm_head") as (LmHeadCell & { id: string }) | undefined;
    const rows: RowResult[] = [];
    let stackParams = 0, stackFwd = 0;

    s.rows.forEach((row, index) => {
      const grp = groupOfRow.get(row.id);
      const execs = grp ? grp.count : 1; // a loop runs its rows R times, a repeat has R copies
      const copies = grp?.kind === "repeat" ? grp.count : 1;
      const wIn = W.rowIn.get(row.id) ?? 0;
      let w = wIn;
      const rowCells: CellResult[] = [];
      let rowParams = 0, rowFwd = 0, rowAct = 0;
      const checkpointed = row.cells.some((c) => !["embedding", "positional", "lm_head"].includes(c.type));

      // row extras: x0 re-injection (every copy) and incoming skip weights (a repeat's first copy only)
      const skipRuns = grp?.kind === "repeat" ? 1 : execs;
      let skipParams = 0;
      let extraAct = row.x0_mix ? bytes.resid * w * seq.P * execs : 0;
      for (const e of g.edges)
        if (e.kind === "skip" && e.to.row === row.id) {
          skipParams += e.weight === "scalar" ? 1 : e.weight === "vector" ? w : 0;
          extraAct += bytes.resid * w * seq.P * skipRuns;
          const from = ix.rowHome.get(e.from.row ?? "");
          if (!from) issues.push({ where: e.id, msg: "skip wire from an unknown row" });
          else if (from.stack.id !== s.id) issues.push({ where: e.id, msg: "skip wires join rows of one stack" });
          else if (from.index >= index) issues.push({ where: e.id, msg: "a skip wire must go to a later row" });
          else if ((W.rowOut.get(from.row.id) ?? 0) !== w) issues.push({ where: e.id, msg: "a skip wire joins rows of different widths" });
        }
      const extras = (row.x0_mix ? 2 : 0) * copies + skipParams;
      rowParams += extras;
      rowAct += extraAct;

      for (const cell0 of row.cells) {
        let cell = cell0;
        if (cell.type === "embedding" && cell.share_with && ix.cellHome.get(cell.share_with)?.cell.type !== "embedding") {
          issues.push({ where: cell.id, msg: `shares a table with ${cell.share_with}, which is not an embedding here: counted as its own table` });
          cell = { ...cell, share_with: undefined };
        }
        const c: CellCost = cellCost(cell, {
          stackId: s.id,
          width: w,
          stackWidth: s.width,
          S,
          seq,
          source: sourceFor(cell.id),
          embedding: embCell ? { cell: embCell, shared: Boolean(embCell.share_with) } : undefined,
          bytes,
        });
        const f = c.flops.reduce((n, t) => n + t.fwd, 0) * execs;
        const b = c.flops.reduce((n, t) => n + t.fwd * t.bwd, 0) * execs;
        const k = c.flops.filter((t) => t.keys).reduce((n, t) => n + t.fwd, 0) * execs;
        const res: CellResult = {
          id: cell.id,
          type: cell.type,
          stackId: s.id,
          rowId: row.id,
          params: c.params * copies,
          parts: c.parts.map((p) => ({ ...p, params: p.params * copies })),
          execs,
          fwdPerToken: f / S,
          bwdPerToken: b / S,
          keysFwdPerToken: k / S,
          actBytesPerRow: c.act * execs,
          widthIn: w,
          widthOut: c.widthOut,
          irregular: c.irregular,
          issues: c.issues,
        };
        if (cell.type === "lm_head" && cell.tied && embCell?.share_with) res.issues.push("the head is tied to a shared table");
        for (const m of c.issues) issues.push({ where: cell.id, msg: m });
        cells.set(cell.id, res);
        rowCells.push(res);
        rowParams += res.params;
        rowFwd += f;
        rowAct += res.actBytesPerRow;
        byType[cell.type] = (byType[cell.type] ?? 0) + res.params;
        fwd += f;
        bwd += b;
        keysFwd += k;
        if (c.irregular) irregularTrain += f + b;
        if (checkpointed) recompute += f;
        if (checkpointed && c.irregular) irregularRecompute += f;
        workspace = Math.max(workspace, c.workspace);
        w = c.widthOut;
      }
      rows.push({ id: row.id, index, params: rowParams, rowExtras: extras, execs, copies, fwdPerToken: rowFwd / S, actBytesPerRow: rowAct, checkpointed, cells: rowCells });
      stackParams += rowParams;
      stackFwd += rowFwd;
      actNoCkpt += rowAct;
      if (checkpointed) {
        actCkptSaved += bytes.resid * wIn * seq.P * execs;
        actCkptPeakRow = Math.max(actCkptPeakRow, execs ? rowAct / execs : 0);
      } else actCkptSaved += rowAct;
    });

    // loop extras: a learned vector per round, and the soft feedback through the stack's own head and table
    const groups: GroupResult[] = [];
    for (const grp of g.groups.filter((x) => x.stack === s.id)) {
      let params = 0, lensFwd = 0, lensAct = 0;
      if (grp.kind === "loop") {
        const w = W.rowIn.get(grp.rows[0]) ?? s.width;
        if (grp.round_embedding) params += grp.count * w;
        if (grp.feedback === "soft") {
          params += w;
          if (!embCell || !headCell) issues.push({ where: grp.id, msg: "soft feedback needs an embedding and an LM head in the same stack" });
          else {
            const V = headCell.vocab, e = embCell.dim;
            lensFwd = (2 * w * V + 2 * V * e + embeddingProjFlops(embCell, w)) * seq.P * (grp.count - 1);
            lensAct = (8 * V + bytes.act * (e + 2 * w) + bytes.resid * w) * seq.P * (grp.count - 1);
          }
        }
      }
      groups.push({ id: grp.id, kind: grp.kind, count: grp.count, params, fwdPerToken: lensFwd / S, actBytesPerRow: lensAct });
      stackParams += params;
      stackFwd += lensFwd;
      fwd += lensFwd;
      bwd += lensFwd; // frozen-style: input gradients only
      irregularTrain += 2 * lensFwd;
      recompute += lensFwd;
      irregularRecompute += lensFwd;
      actNoCkpt += lensAct;
      actCkptSaved += lensAct;
      byType.loop = (byType.loop ?? 0) + params;
    }

    byStack[s.id] = stackParams;
    stackResults.push({ id: s.id, name: s.name, role: s.role, P: seq.P, seq, widthOut: W.stackOut.get(s.id) ?? s.width, params: stackParams, fwdPerToken: stackFwd / S, rows, groups });
  }

  for (const e of g.edges) {
    if (!ix.stacks.has(e.from.stack) || !ix.stacks.has(e.to.stack)) issues.push({ where: e.id, msg: "a wire end points at a missing stack" });
    if (e.kind === "source" && (!e.to.cell || !ix.cellHome.has(e.to.cell))) issues.push({ where: e.id, msg: "a source wire must end on a cell" });
  }

  const total = Object.values(byStack).reduce((a, b) => a + b, 0);
  const ckpt = run.checkpointing;
  const fwdT = fwd / S, bwdT = bwd / S, recT = ckpt ? recompute / S : 0;
  const trainT = fwdT + bwdT + recT;
  const irrT = (irregularTrain + (ckpt ? irregularRecompute : 0)) / S;

  // ---------------------------------------------------------------- memory
  const gpu = gpuById(run.gpu);
  const opt = OPTIMIZERS[run.optimizer] ?? OPTIMIZERS.adamw;
  const ckptRow = actCkptSaved + actCkptPeakRow;
  const actPerRow = ckpt ? ckptRow : actNoCkpt;
  const memFor = (b: number, N: number): MemoryResult => {
    const z = run.zero_stage;
    const weights = (total * prec.weight) / (z >= 3 ? N : 1);
    const grads = (total * prec.grad) / (z >= 2 ? N : 1);
    const master = (total * prec.master) / (z >= 1 ? N : 1);
    const optimizer = (total * opt.bytes * prec.optScale) / (z >= 1 ? N : 1);
    const castCopies = (total * prec.cast) / (z >= 3 ? N : 1);
    const activations = b * actPerRow * MEMORY_CAL.activationOverhead;
    const ws = b * workspace;
    const context = gpu.contextGiB * GiB;
    const totalBytes = (weights + grads + master + optimizer + castCopies + activations + ws) * MEMORY_CAL.fragmentation + context;
    const capacity = gpu.memGiB * GiB;
    return {
      microBatch: b, maxMicroBatch: 0, weights, grads, master, optimizer, castCopies, activations, workspace: ws, context,
      total: totalBytes, capacity, fits: totalBytes <= capacity,
      actBytesPerRow: actPerRow, actBytesPerRowNoCkpt: actNoCkpt, actBytesPerRowCkpt: ckptRow,
    };
  };
  const N = Math.max(1, Math.round(run.gpu_count));
  const rowsPerGpu = Math.max(1, Math.ceil(run.rows_per_step / N));
  let maxB = 0;
  for (let b = 1; b <= rowsPerGpu; b *= 2) if (memFor(b, N).fits) maxB = b; else break;
  if (maxB > 0) for (let b = maxB + 1; b <= Math.min(rowsPerGpu, 2 * maxB - 1); b++) if (memFor(b, N).fits) maxB = b; else break;
  // automatic micro-batch: the largest that fits and splits each GPU's share of the step evenly
  const even = run.rows_per_step % N === 0 ? Array.from({ length: maxB }, (_, i) => maxB - i).find((b) => rowsPerGpu % b === 0) : undefined;
  const micro = run.micro_batch > 0 ? run.micro_batch : Math.max(1, even ?? maxB);
  const memory = { ...memFor(micro, N), maxMicroBatch: maxB };

  // ---------------------------------------------------------------- speed, time and GPU count
  const mfu = run.mfu > 0 ? run.mfu : gpu.mfu;
  const rho = run.irregular_ratio > 0 ? run.irregular_ratio : CAL.irregularRatio;
  const peak = gpu.peakTflops * 1e12;
  const secPerToken = (trainT - irrT) / (peak * mfu) + irrT / (peak * mfu * rho);
  const tpsGpu = secPerToken > 0 ? 1 / secPerToken : 0;
  const dp = N > 1 ? run.dp_efficiency : 1;
  const tps = N * tpsGpu * dp;
  const hours = tps > 0 ? run.token_budget / tps / 3600 : Infinity;
  let minForMemory = Infinity;
  for (const n of [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]) {
    if (n > 1 && run.zero_stage === 0) break;
    if (memFor(1, n).fits) {
      minForMemory = n;
      break;
    }
  }
  const gpuSeconds = run.deadline_hours > 0 && tpsGpu > 0 ? run.token_budget / (run.deadline_hours * 3600 * tpsGpu) : 0;
  const forDeadline = gpuSeconds === 0 ? 0 : gpuSeconds <= 1 ? 1 : Math.ceil(gpuSeconds / run.dp_efficiency);

  return {
    S,
    issues,
    params: { total, byType, byStack },
    stacks: stackResults,
    cells,
    flops: {
      fwdPerToken: fwdT,
      bwdPerToken: bwdT,
      recomputePerToken: recT,
      trainPerToken: trainT,
      keysFwdPerToken: keysFwd / S,
      perStep: trainT * run.rows_per_step * S,
      perRun: trainT * run.token_budget,
      irregularTrainPerToken: irrT,
    },
    memory,
    time: {
      tokensPerSecondPerGpu: tpsGpu,
      tokensPerSecond: tps,
      hours,
      gpuHours: hours * N,
      steps: run.token_budget / (run.rows_per_step * S),
      accumulation: Math.ceil(run.rows_per_step / (N * micro)),
      mfu,
    },
    gpus: { minForMemory, forDeadline },
  };
}

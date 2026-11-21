// The frame: one column per stack, rows of slots inside, loop / repeat brackets, and the wires drawn over it.
import { Fragment, useLayoutEffect, useRef, useState } from "react";
import type { Analysis, CellResult, RowResult } from "../engine/analyze";
import type { Cell, Edge, Endpoint, Graph, Group, Row, Stack } from "../engine/types";
import { CATEGORY, MIX_LEGEND, cellSub, cellTitle, embeddingParts, mixing, seqSummary, sharedTable, stackIO } from "./describe";
import { useDrag, useIsOver, type Target } from "./drag";
import * as F from "./format";
import { MixGlyph } from "./Glyph";
import { fmtShape, positions as positionsOf, rowInputShape, stackOutputShape } from "../engine/shapes";

export type Sel =
  | { kind: "cell"; id: string }
  | { kind: "row"; ids: string[]; stack: string }
  | { kind: "stack"; id: string }
  | { kind: "edge"; id: string }
  | { kind: "group"; id: string }
  | null;

interface Props {
  graph: Graph;
  analysis: Analysis;
  sel: Sel;
  setSel: (s: Sel) => void;
  onAddRow: (stackId: string) => void;
  onGroupCount: (groupId: string, count: number) => void;
  onStackWidth: (stackId: string, width: number | null) => void;
}

/** Stack widths the user set by dragging (layout only, kept in graph.ui). */
export const stackWidths = (g: Graph) => ((g.ui?.stackWidths ?? {}) as Record<string, number>);

const SHAPES_ON = "nn-builder.shapes";
const readFlag = (key: string) => {
  try {
    return localStorage.getItem(key) !== "0";
  } catch {
    return true;
  }
};
const writeFlag = (key: string, v: boolean) => {
  try {
    localStorage.setItem(key, v ? "1" : "0");
  } catch {
    /* storage blocked */
  }
};

export function Frame(p: Props) {
  const [shapes, setShapes] = useState(() => readFlag(SHAPES_ON));
  const toggleShapes = (v: boolean) => (setShapes(v), writeFlag(SHAPES_ON, v));
  return (
    <div className={`frame-wrap ${shapes ? "" : "shapes-off"}`}>
      <div className="frame-scroll" onPointerDown={(e) => e.target === e.currentTarget && p.setSel(null)}>
        <div className="frame" onPointerDown={(e) => e.target === e.currentTarget && p.setSel(null)}>
          {p.graph.stacks.map((s) => (
            <StackColumn key={s.id} stack={s} {...p} />
          ))}
          <Wires graph={p.graph} analysis={p.analysis} sel={p.sel} setSel={p.setSel} />
        </div>
      </div>
      <Legend shapes={shapes} setShapes={toggleShapes} batch={p.analysis.memory.microBatch} />
    </div>
  );
}

const LEGEND_OPEN = "nn-builder.legendOpen";

function Legend({ shapes, setShapes, batch }: { shapes: boolean; setShapes: (v: boolean) => void; batch: number }) {
  const [open, setOpenRaw] = useState(() => {
    try {
      return localStorage.getItem(LEGEND_OPEN) !== "0";
    } catch {
      return true;
    }
  });
  const setOpen = (v: boolean) => {
    setOpenRaw(v);
    try {
      localStorage.setItem(LEGEND_OPEN, v ? "1" : "0");
    } catch {
      /* storage blocked */
    }
  };
  return (
    <div className={`legend-box ${open ? "open" : ""}`}>
      <button className="legend-toggle" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} Connections
      </button>
      {open && (
        <div className="legend-body">
          <div className="legend-sec">How each slot connects tokens (row = reading token, column = what it reads)</div>
          {MIX_LEGEND.map(([k, t]) => (
            <div key={k} className="legend-row">
              <MixGlyph kind={k} title={t} size={14} />
              <span>{t}</span>
            </div>
          ))}
          <div className="legend-sec">Tensors</div>
          <label className="legend-row">
            <input type="checkbox" checked={shapes} onChange={(e) => setShapes(e.target.checked)} />
            show tensor shapes: <span className="shape-chip static">B×4k×576</span> = batch × positions × width (B = {batch} rows per GPU now)
          </label>
          <div className="legend-sec">Flow</div>
          <div className="legend-row"><span className="lg-rail" /> the stream runs down the rows; each row reads it left to right (→)</div>
          <div className="legend-row"><span className="resid static">+</span> residual: the slot's output is added back to the stream</div>
          <div className="legend-row"><span className="lg-wire skip" /> skip wire: a row's output added to a later row</div>
          <div className="legend-row"><span className="lg-wire source" /> keys/values or notes into an attention slot</div>
          <div className="legend-row"><span className="lg-wire input" /> a stack's input (e.g. the token table feeding a writer)</div>
        </div>
      )}
    </div>
  );
}

function StackColumn(p: Props & { stack: Stack }) {
  const { stack: s, analysis: a, graph: g } = p;
  const res = a.stacks.find((x) => x.id === s.id);
  const { start } = useDrag();
  const over = useIsOver((t) => t.drop === "stack" && t.stack === s.id);
  const selected = p.sel?.kind === "stack" && p.sel.id === s.id;
  const issues = a.issues.filter((i) => i.where === s.id);
  const io = stackIO(s, g);
  const [dragW, setDragW] = useState<number | null>(null);
  const width = dragW ?? stackWidths(g)[s.id];
  const startResize = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const box = e.currentTarget.parentElement!.getBoundingClientRect();
    const x0 = e.clientX, w0 = box.width;
    let w = w0;
    document.body.dataset.resizing = "1";
    const move = (ev: PointerEvent) => setDragW((w = Math.max(320, Math.round(w0 + ev.clientX - x0))));
    const up = () => {
      delete document.body.dataset.resizing;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setDragW(null);
      if (Math.abs(w - w0) > 2) p.onStackWidth(s.id, w);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // consecutive rows of one group render inside one bracket
  const segments: { group?: Group; rows: Row[]; first: number }[] = [];
  s.rows.forEach((row, i) => {
    const grp = g.groups.find((x) => x.stack === s.id && x.rows.includes(row.id));
    const last = segments[segments.length - 1];
    if (grp && last?.group?.id === grp.id) last.rows.push(row);
    else segments.push({ group: grp, rows: [row], first: i });
  });

  const shared = sharedTable(s, g);

  return (
    <section className={`stack role-${s.role} ${width ? "sized" : ""}`} style={width ? { width } : undefined} data-stack-box={s.id}>
      <span className="stack-resize" onPointerDown={startResize} onDoubleClick={() => p.onStackWidth(s.id, null)} title="drag to resize this stack; double-click for the automatic width" />
      <header
        className={`stack-head ${selected ? "selected" : ""} ${over}`}
        data-drop="stack"
        data-stack={s.id}
        onClick={() => p.setSel({ kind: "stack", id: s.id })}
      >
        <span className="dot in" data-anchor={`stackin:${s.id}`} title="input: drop a wire here to start this stack from another one" />
        <div className="stack-title">
          <strong>{s.name}</strong>
          <span className="chip">{s.role}</span>
          {issues.length > 0 && <span className="warn" title={issues.map((i) => i.msg).join("\n")}>⚠</span>}
        </div>
        <div className="stack-meta">
          <span>{seqSummary(s, res?.seq, g)}</span>
          <span>width {s.width}</span>
          <span className="num">{F.params(res?.params ?? 0)}</span>
        </div>
        <div className="stack-io in" title="what starts this stack's stream">
          <span className="io-k">in</span> {io.input}
        </div>
      </header>
      <div className="rows">
        {shared && (
          <button
            type="button"
            className="shared-table cat-input"
            title={`The same table as ${shared.from.name}'s embedding: no weights of its own. Click to see it.`}
            onClick={() => p.setSel({ kind: "cell", id: shared.cell.id })}
          >
            <span className="dot in" data-anchor={`shared:${s.id}`} />
            <strong>Token table</strong> <span className="muted">{shared.cell.vocab.toLocaleString("en-US")} × {shared.cell.dim}</span>
            <span className="shared-note">same as {shared.from.name} · 0 new weights</span>
          </button>
        )}
        {segments.map((seg) => {
          const body = seg.rows.map((row, k) => {
            const i = seg.first + k;
            const r = res?.rows.find((x) => x.id === row.id);
            return (
              <Fragment key={row.id}>
                {(k > 0 || !seg.group) && <RowGap stack={s.id} index={i} />}
                <RowView {...p} row={row} res={r} index={i} />
              </Fragment>
            );
          });
          if (!seg.group) return body;
          const grp = seg.group;
          const gres = res?.groups.find((x) => x.id === grp.id);
          const gsel = p.sel?.kind === "group" && p.sel.id === grp.id;
          return (
            <Fragment key={grp.id}>
              <RowGap stack={s.id} index={seg.first} />
              <div className={`group ${grp.kind} ${gsel ? "selected" : ""}`}>
                <div className="group-tag" onClick={() => p.setSel({ kind: "group", id: grp.id })}>
                  <span>{grp.kind === "loop" ? "↻ loop" : "⧉ repeat"}</span>
                  <button className="mini" onClick={(e) => (e.stopPropagation(), p.onGroupCount(grp.id, Math.max(1, grp.count - 1)))} aria-label="fewer">−</button>
                  <strong>×{grp.count}</strong>
                  <button className="mini" onClick={(e) => (e.stopPropagation(), p.onGroupCount(grp.id, grp.count + 1))} aria-label="more">+</button>
                  <span className="muted">{grp.kind === "loop" ? "shared weights" : "independent copies"}</span>
                  {gres && gres.params > 0 && <span className="muted">+{F.params(gres.params)}</span>}
                </div>
                {body}
              </div>
            </Fragment>
          );
        })}
        <RowGap stack={s.id} index={s.rows.length} last onAdd={() => p.onAddRow(s.id)} />
      </div>
      <footer className="stack-io out" title="where this stack's output goes">
        <span className="io-k">out</span> <span className="shape-chip static out">{fmtShape(stackOutputShape(g, a, s.id, a.memory.microBatch), "compact")}</span>{" "}
        {io.outputs.length ? io.outputs.join(" · ") : `${s.name} output (not wired)`}
        <span
          className="dot out"
          data-anchor={`stackout:${s.id}`}
          title="output (after the last row): drag to a cross-attention or notebook read, or onto another stack's header"
          onPointerDown={(e) => start(e, { kind: "wire", from: { stack: s.id }, label: `${s.name} output` })}
        />
      </footer>
    </section>
  );
}

function RowGap({ stack, index, last, onAdd }: { stack: string; index: number; last?: boolean; onAdd?: () => void }) {
  const over = useIsOver((t) => t.drop === "rowgap" && t.stack === stack && t.index === index);
  return (
    <div className={`rowgap ${last ? "last" : ""} ${over}`} data-drop="rowgap" data-stack={stack} data-index={index} onClick={onAdd}>
      {last && <span>+ add a row (or drop a layer here)</span>}
    </div>
  );
}

function RowView(p: Props & { stack: Stack; row: Row; res?: RowResult; index: number }) {
  const { row, res, stack: s } = p;
  const { start } = useDrag();
  const over = useIsOver((t) => t.drop === "row" && t.row === row.id);
  const selected = p.sel?.kind === "row" && p.sel.ids.includes(row.id);
  const extras: string[] = [];
  if (row.x0_mix) extras.push("x0");
  const skips = p.graph.edges.filter((e) => e.kind === "skip" && e.to.row === row.id).length;
  if (skips) extras.push(`+${skips} skip`);
  const label = row.label ?? `row ${p.index}`;
  const b = p.analysis.memory.microBatch;
  const inShape = fmtShape(rowInputShape(p.graph, p.analysis, row.id, b), "compact");
  const prevRow = p.index > 0 ? s.rows[p.index - 1] : undefined;
  const changed = !prevRow || fmtShape(rowInputShape(p.graph, p.analysis, prevRow.id, b), "compact") !== inShape;

  const clickRow = (e: PointerEvent) => {
    if (e.shiftKey && p.sel?.kind === "row" && p.sel.stack === s.id) {
      const idx = [...p.sel.ids, row.id].map((id) => s.rows.findIndex((r) => r.id === id));
      const lo = Math.min(...idx), hi = Math.max(...idx);
      p.setSel({ kind: "row", stack: s.id, ids: s.rows.slice(lo, hi + 1).map((r) => r.id) });
    } else p.setSel({ kind: "row", stack: s.id, ids: [row.id] });
  };

  return (
    <div className={`row ${selected ? "selected" : ""} ${over}`} data-drop="row" data-row={row.id} data-stack={s.id}>
      <span className="dot in row-in" data-anchor={`rowin:${row.id}`} />
      <div
        className="row-handle"
        title="drag to move the row; click to select (shift-click to select several, then loop them)"
        onPointerDown={(e) => start(e, { kind: "row", id: row.id, stack: s.id, label }, clickRow)}
      >
        <span className="grip">⋮⋮</span>
        <span className="row-label">{label}</span>
      </div>
      <span className={`rail ${p.index === 0 ? "first" : ""} ${p.index === s.rows.length - 1 ? "last" : ""}`} title="the stream flows down the rows; this row reads it here" />
      {row.cells.length > 0 && <span className={`shape-chip ${changed ? "changed" : ""}`} title={`tensor entering ${label}: ${inShape} (B = rows per GPU, then positions, then width)`}>{inShape}</span>}
      <div className="cells">
        {row.cells.map((c, i) => (
          <Fragment key={c.id}>
            {i > 0 && <span className="link" aria-hidden>→</span>}
            <CellCard {...p} cell={c} index={i} res={p.analysis.cells.get(c.id)} />
          </Fragment>
        ))}
        <Slot row={row.id} index={row.cells.length} />
      </div>
      <div className="row-meta">
        <span className="num">{F.params(res?.params ?? 0)}</span>
        {extras.length > 0 && <span className="muted">{extras.join(" ")}</span>}
      </div>
      <span
        className="dot out"
        data-anchor={`rowout:${row.id}`}
        title="drag to a later row for a skip wire, or to a cross-attention / notebook read"
        onPointerDown={(e) => start(e, { kind: "wire", from: { stack: s.id, row: row.id }, label: `${label} output` })}
      />
    </div>
  );
}

function Slot({ row, index }: { row: string; index: number }) {
  const over = useIsOver((t) => t.drop === "slot" && t.row === row);
  return (
    <div className={`slot ${over}`} data-drop="slot" data-row={row} data-index={index} title="drop a layer here">
      +
    </div>
  );
}

function CellCard(p: Props & { stack: Stack; row: Row; cell: Cell; index: number; res?: CellResult }) {
  const { cell: c, res } = p;
  const { live, start } = useDrag();
  const t: Target | null = live?.target ?? null;
  const hereCell = t?.drop === "cell" && t.cell === c.id;
  const ins = hereCell && live?.payload.kind !== "wire" && live?.ok ? (t!.index === p.index ? "ins-before" : "ins-after") : "";
  const wireOver = hereCell && live?.payload.kind === "wire" ? (live.ok ? "over" : "over-bad") : "";
  const selected = p.sel?.kind === "cell" && p.sel.id === c.id;
  const title = c.label ?? cellTitle(c);
  const sub = cellSub(c, res?.widthIn ?? p.stack.width);
  const takesWire = c.type === "attention";
  const fed = takesWire && p.graph.edges.some((e) => e.kind === "source" && e.to.cell === c.id);
  const mix = mixing(c, p.stack, p.graph);
  const resid = (c.type === "attention" || c.type === "ffn") && c.residual !== "none";
  const frozen = Boolean(c.frozen || p.stack.frozen);
  const parts = c.type === "embedding" ? embeddingParts(c, res?.widthOut ?? p.stack.width, res) : null;
  return (
    <div
      className={`cell cat-${CATEGORY[c.type]} ${parts ? "split" : ""} ${selected ? "selected" : ""} ${frozen ? "frozen" : ""} ${ins} ${wireOver}`}
      title={parts ? title : undefined}
      data-drop="cell"
      data-row={p.row.id}
      data-cell={c.id}
      data-index={p.index}
      data-wire={takesWire ? "1" : "0"}
      onPointerDown={(e) => start(e, { kind: "cell", id: c.id, label: title }, () => p.setSel({ kind: "cell", id: c.id }))}
    >
      {c.type === "attention" && (c.mode === "cross" || c.memory || fed) && <span className={`dot cell-in ${fed ? "fed" : ""}`} data-anchor={`cellin:${c.id}`} title="keys/values come in here" />}
      {parts ? (
        <div className="cell-parts">
          {parts.map((part, i) => (
            <div key={part.name} className="cell-part">
              <div className="cell-title">
                {i === 0 && frozen && <span className="frost" title="frozen: weights kept, not trained">❄</span>}
                <span className="cell-name" title={part.name}>{part.name}</span>
                {i === 0 && res && res.issues.length > 0 && <span className="warn" title={res.issues.join("\n")}>⚠</span>}
                {i === 1 && <MixGlyph kind={mix.kind} title={mix.text} />}
              </div>
              <div className="cell-sub" title={part.sub}>{part.sub}</div>
              <div className="cell-foot">
                <span />
                <span className="cell-num">{part.params > 0 ? F.params(part.params) : "0"}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="cell-title">
            {frozen && <span className="frost" title="frozen: weights kept, not trained">❄</span>}
            <span className="cell-name" title={title}>{title}</span>
            {res && res.issues.length > 0 && <span className="warn" title={res.issues.join("\n")}>⚠</span>}
            <MixGlyph kind={mix.kind} title={mix.text} />
          </div>
          {resid && <span className="resid" title={`residual: the output is added back to the stream${c.residual === "post" ? " (norm after the add)" : ""}`}>+</span>}
          <div className="cell-sub">{c.label ? `${cellTitle(c)} · ${sub}` : sub}</div>
          <div className="cell-foot">
            {res && res.execs > 1 ? <span className="badge" title={`runs ${res.execs} times per row`}>×{res.execs}</span> : <span />}
            <span className="cell-num">{res && res.params > 0 ? F.params(res.params) : "0"}</span>
          </div>
        </>
      )}
      {c.type === "embedding" && (
        <span
          className="dot table"
          data-anchor={`table:${c.id}`}
          title="token table output: drag onto another stack's header to feed it"
          onPointerDown={(e) => start(e, { kind: "wire", from: { stack: p.stack.id, cell: c.id, port: "table" }, label: "token table" })}
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------------ wires

interface WirePath {
  id: string;
  kind: Edge["kind"];
  d: string;
  label: string;
  tensor: string;
  mid?: boolean;
  lx: number;
  ly: number;
}

/** The wires that touch the selection: they are drawn strong, the others faint. */
function relatedWires(g: Graph, sel: Sel): Set<string> | null {
  if (!sel || sel.kind === "edge" || sel.kind === "group") return null;
  const touches = (e: Edge, cells: Set<string>, rows: Set<string>, stack?: string) =>
    (e.to.cell && cells.has(e.to.cell)) || (e.from.cell && cells.has(e.from.cell)) || (e.to.row && rows.has(e.to.row)) || (e.from.row && rows.has(e.from.row)) || (stack && (e.from.stack === stack || e.to.stack === stack));
  let cells = new Set<string>(), rows = new Set<string>(), stack: string | undefined;
  if (sel.kind === "cell") cells = new Set([sel.id]);
  if (sel.kind === "row") {
    rows = new Set(sel.ids);
    cells = new Set(g.stacks.flatMap((s) => s.rows.filter((r) => rows.has(r.id)).flatMap((r) => r.cells.map((c) => c.id))));
  }
  if (sel.kind === "stack") stack = sel.id;
  return new Set(g.edges.filter((e) => touches(e, cells, rows, stack)).map((e) => e.id));
}

function wireShape(e: Edge, g: Graph, a: Analysis): string {
  const b = a.memory.microBatch;
  if (e.from.port === "table" && e.from.cell) {
    const c = a.cells.get(e.from.cell);
    const home = g.stacks.find((s) => s.id === e.from.stack);
    const seq = a.stacks.find((s) => s.id === home?.id)?.seq;
    const cell = home?.rows.flatMap((r) => r.cells).find((x) => x.id === e.from.cell);
    if (!c || !seq || cell?.type !== "embedding") return "";
    return fmtShape([{ n: "B", v: b }, ...positionsOf(seq), { n: "e", v: cell.dim }], "compact");
  }
  if (e.from.row) {
    const s = g.stacks.find((x) => x.id === e.from.stack);
    const i = s?.rows.findIndex((r) => r.id === e.from.row) ?? -1;
    const res = a.stacks.find((x) => x.id === e.from.stack);
    const w = res?.rows[i]?.cells.at(-1)?.widthOut;
    return res && w ? fmtShape([{ n: "B", v: b }, ...positionsOf(res.seq), { n: "d", v: w }], "compact") : "";
  }
  return fmtShape(stackOutputShape(g, a, e.from.stack, b), "compact");
}

function wireLabel(e: Edge, g: Graph): string {
  if (e.kind === "skip") return e.weight === "scalar" ? "skip ×σ" : e.weight === "vector" ? "skip ×w" : "skip";
  if (e.kind === "input") return e.from.port === "table" ? "same table" : "in";
  const src = g.stacks.find((s) => s.id === e.from.stack);
  return src?.rows.flatMap((r) => r.cells).at(-1)?.type === "kv_out" ? "notes" : "keys/values";
}

const fromAnchor = (e: Endpoint) => (e.cell && e.port === "table" ? `table:${e.cell}` : e.row ? `rowout:${e.row}` : `stackout:${e.stack}`);
const toAnchor = (e: Edge) => (e.kind === "input" && e.from.port === "table" ? `shared:${e.to.stack}` : e.to.cell ? `cellin:${e.to.cell}` : e.to.row ? `rowout:${e.to.row}` : `stackin:${e.to.stack}`);

function Wires({ graph, analysis, sel, setSel }: { graph: Graph; analysis: Analysis; sel: Sel; setSel: (s: Sel) => void }) {
  const [paths, setPaths] = useState<WirePath[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const svg = useRef<SVGSVGElement>(null);
  const related = relatedWires(graph, sel);
  useLayoutEffect(() => {
    // the frame is the svg's parent (a parent's ref is not attached yet when a child's layout effect runs)
    const root = svg.current?.parentElement;
    if (!root) return;
    const measure = () => {
      const base = root.getBoundingClientRect();
      const at = (key: string) => {
        const el = root.querySelector<HTMLElement>(`[data-anchor="${CSS.escape(key)}"]`);
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return { x: r.left + r.width / 2 - base.left, y: r.top + r.height / 2 - base.top };
      };
      const out: WirePath[] = [];
      for (const e of graph.edges) {
        const a = at(fromAnchor(e.from));
        const b = at(toAnchor(e));
        if (!a || !b) continue;
        const label = wireLabel(e, graph);
        const tensor = wireShape(e, graph, analysis);
        if (e.kind === "skip") {
          const off = 18 + Math.min(60, Math.abs(b.y - a.y) / 12);
          out.push({ id: e.id, kind: e.kind, label, tensor, lx: b.x + off * 0.75 + 4, ly: b.y - 2, d: `M ${a.x} ${a.y} C ${a.x + off} ${a.y}, ${b.x + off} ${b.y + 5}, ${b.x + 3} ${b.y + 5}` });
        } else {
          const dx = Math.max(50, Math.abs(b.x - a.x) / 2);
          const long = Math.abs(b.x - a.x) > 260;
          out.push({ id: e.id, kind: e.kind, label, tensor, mid: long, lx: long ? (a.x + b.x) / 2 : b.x - 8, ly: long ? (a.y + b.y) / 2 - 6 : b.y - 8, d: `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}` });
        }
      }
      setPaths(out);
      setSize({ w: root.scrollWidth, h: root.scrollHeight });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [graph, analysis]);

  return (
    <svg className="wires" ref={svg} width={size.w} height={size.h}>
      <defs>
        {["skip", "source", "input"].map((k) => (
          <marker key={k} id={`arrow-${k}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" className={`arrow ${k}`} />
          </marker>
        ))}
      </defs>
      {paths.map((p) => {
        const selected = sel?.kind === "edge" && sel.id === p.id;
        const rel = related ? (related.has(p.id) ? "related" : "faint") : "";
        return (
          <g key={p.id} className={`wire ${p.kind} ${selected ? "selected" : ""} ${rel}`} onPointerDown={(e) => (e.stopPropagation(), setSel({ kind: "edge", id: p.id }))}>
            <title>{p.tensor ? `${p.label}: ${p.tensor}` : p.label}</title>
            <path d={p.d} className="hit" />
            <path d={p.d} className="line" markerEnd={`url(#arrow-${p.kind})`} />
            {(p.kind !== "skip" || selected || rel === "related") && (
              <text x={p.lx} y={p.ly} className="wire-label" textAnchor={p.kind === "skip" ? "start" : p.mid ? "middle" : "end"}>
                {p.label}
                {p.tensor && <tspan className="wire-shape"> {p.tensor}</tspan>}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// The frame: one column per stack, rows of slots inside, loop / repeat brackets, and the wires drawn over it.
import { Fragment, useLayoutEffect, useRef, useState } from "react";
import type { Analysis, CellResult, RowResult } from "../engine/analyze";
import type { Cell, Edge, Endpoint, Graph, Group, Row, Stack } from "../engine/types";
import { CATEGORY, cellSub, cellTitle, seqSummary } from "./describe";
import { useDrag, useIsOver, type Target } from "./drag";
import * as F from "./format";

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
}

export function Frame(p: Props) {
  return (
    <div className="frame-scroll" onPointerDown={(e) => e.target === e.currentTarget && p.setSel(null)}>
      <div className="frame" onPointerDown={(e) => e.target === e.currentTarget && p.setSel(null)}>
        {p.graph.stacks.map((s) => (
          <StackColumn key={s.id} stack={s} {...p} />
        ))}
        <Wires graph={p.graph} analysis={p.analysis} sel={p.sel} setSel={p.setSel} />
      </div>
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

  // consecutive rows of one group render inside one bracket
  const segments: { group?: Group; rows: Row[]; first: number }[] = [];
  s.rows.forEach((row, i) => {
    const grp = g.groups.find((x) => x.stack === s.id && x.rows.includes(row.id));
    const last = segments[segments.length - 1];
    if (grp && last?.group?.id === grp.id) last.rows.push(row);
    else segments.push({ group: grp, rows: [row], first: i });
  });

  return (
    <section className={`stack role-${s.role}`}>
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
        <span
          className="dot out"
          data-anchor={`stackout:${s.id}`}
          title="output: drag to a cross-attention or notebook read, or onto another stack's header"
          onPointerDown={(e) => start(e, { kind: "wire", from: { stack: s.id }, label: `${s.name} output` })}
        />
      </header>
      <div className="rows">
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
      <div className="cells">
        {row.cells.map((c, i) => (
          <CellCard key={c.id} {...p} cell={c} index={i} res={p.analysis.cells.get(c.id)} />
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
  return (
    <div
      className={`cell cat-${CATEGORY[c.type]} ${selected ? "selected" : ""} ${ins} ${wireOver}`}
      data-drop="cell"
      data-row={p.row.id}
      data-cell={c.id}
      data-index={p.index}
      data-wire={takesWire ? "1" : "0"}
      onPointerDown={(e) => start(e, { kind: "cell", id: c.id, label: title }, () => p.setSel({ kind: "cell", id: c.id }))}
    >
      {c.type === "attention" && (c.mode === "cross" || c.memory || fed) && <span className={`dot cell-in ${fed ? "fed" : ""}`} data-anchor={`cellin:${c.id}`} title="keys/values come in here" />}
      <div className="cell-title">
        {title}
        {res && res.execs > 1 && <span className="badge">×{res.execs}</span>}
        {res && res.issues.length > 0 && <span className="warn" title={res.issues.join("\n")}>⚠</span>}
      </div>
      <div className="cell-sub">{c.label ? `${cellTitle(c)} · ${sub}` : sub}</div>
      <div className="cell-num">{res && res.params > 0 ? F.params(res.params) : "0"}</div>
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
}

const fromAnchor = (e: Endpoint) => (e.cell && e.port === "table" ? `table:${e.cell}` : e.row ? `rowout:${e.row}` : `stackout:${e.stack}`);
const toAnchor = (e: Endpoint) => (e.cell ? `cellin:${e.cell}` : e.row ? `rowout:${e.row}` : `stackin:${e.stack}`);

function Wires({ graph, analysis, sel, setSel }: { graph: Graph; analysis: Analysis; sel: Sel; setSel: (s: Sel) => void }) {
  const [paths, setPaths] = useState<WirePath[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const svg = useRef<SVGSVGElement>(null);
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
        const b = at(toAnchor(e.to));
        if (!a || !b) continue;
        if (e.kind === "skip") {
          const off = 18 + Math.min(60, Math.abs(b.y - a.y) / 12);
          out.push({ id: e.id, kind: e.kind, d: `M ${a.x} ${a.y} C ${a.x + off} ${a.y}, ${b.x + off} ${b.y + 5}, ${b.x + 3} ${b.y + 5}` });
        } else {
          const dx = Math.max(50, Math.abs(b.x - a.x) / 2);
          const end = e.kind === "input" ? b : { x: b.x, y: b.y };
          out.push({ id: e.id, kind: e.kind, d: `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}` });
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
        return (
          <g key={p.id} className={`wire ${p.kind} ${selected ? "selected" : ""}`} onPointerDown={(e) => (e.stopPropagation(), setSel({ kind: "edge", id: p.id }))}>
            <path d={p.d} className="hit" />
            <path d={p.d} className="line" markerEnd={`url(#arrow-${p.kind})`} />
          </g>
        );
      })}
    </svg>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analyze } from "../engine/analyze";
import { PALETTE } from "../engine/defaults";
import * as E from "../engine/edit";
import { buildPreset } from "../engine/presets";
import { parseGraph } from "../engine/serialize";
import type { Graph } from "../engine/types";
import { DragProvider, type Payload, type Target } from "./drag";
import { CostPanel, LengthPanel, SummaryStrip, TrainingPanel } from "./Estimates";
import { Frame, type Sel } from "./Frame";
import { useHistory } from "./history";
import { Inspector } from "./Inspector";
import { Palette } from "./Palette";
import { TopBar } from "./TopBar";

const AUTOSAVE = "nn-builder.current";

function initial(): Graph {
  try {
    const raw = localStorage.getItem(AUTOSAVE);
    const res = raw ? parseGraph(raw) : null;
    if (res?.graph) return res.graph;
  } catch {
    /* storage blocked: start fresh */
  }
  return buildPreset("e33m_plain_30m");
}

const TABS: [string, string][] = [
  ["inspect", "Inspect"],
  ["cost", "Compute & memory"],
  ["train", "GPUs & time"],
  ["length", "Row length"],
];

export function App() {
  const hist = useHistory(initial);
  const { graph, apply, undo, redo } = hist;
  const analysis = useMemo(() => analyze(graph), [graph]);
  const [sel, setSelRaw] = useState<Sel>(null);
  const [tab, setTab] = useState("inspect");
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const toast = useCallback((m: string) => {
    setToastMsg(m);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setToastMsg(null), 5000);
  }, []);
  const setSel = useCallback((s: Sel) => {
    setSelRaw(s);
    if (s) setTab("inspect");
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(AUTOSAVE, JSON.stringify(graph));
    } catch {
      /* storage blocked */
    }
  }, [graph]);

  // a selection whose target was removed (undo, delete) is dropped
  useEffect(() => {
    if (!sel) return;
    const gone =
      (sel.kind === "cell" && !E.findCell(graph, sel.id)) ||
      (sel.kind === "row" && sel.ids.some((id) => !E.findRow(graph, id))) ||
      (sel.kind === "stack" && !graph.stacks.some((s) => s.id === sel.id)) ||
      (sel.kind === "edge" && !graph.edges.some((e) => e.id === sel.id)) ||
      (sel.kind === "group" && !graph.groups.some((x) => x.id === sel.id));
    if (gone) setSelRaw(null);
  }, [graph, sel]);

  const removeSelected = useCallback(() => {
    if (!sel) return;
    if (sel.kind === "cell") apply((g) => E.removeCell(g, sel.id));
    if (sel.kind === "row") apply((g) => sel.ids.reduce((x, id) => E.removeRow(x, id), g));
    if (sel.kind === "edge") apply((g) => E.removeEdge(g, sel.id));
    if (sel.kind === "group") apply((g) => E.removeGroup(g, sel.id));
    setSelRaw(null);
  }, [sel, apply]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        if (typing) return;
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (!typing && (e.key === "Delete" || e.key === "Backspace")) {
        e.preventDefault();
        removeSelected();
      } else if (!typing && e.key === "Escape") setSelRaw(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, removeSelected]);

  const widthAt = (t: Target): number => {
    if (t.row) {
      const r = analysis.stacks.flatMap((s) => s.rows).find((x) => x.id === t.row);
      const c = r?.cells[t.index ?? 0];
      return c?.widthIn || r?.cells.at(-1)?.widthOut || graph.stacks.find((s) => s.id === t.stack)?.width || 512;
    }
    const s = analysis.stacks.find((x) => x.id === t.stack);
    const row = s?.rows[t.index ?? 0] ?? s?.rows.at(-1);
    return row?.cells[0]?.widthIn || graph.stacks.find((x) => x.id === t.stack)?.width || 512;
  };

  const accepts = useCallback(
    (p: Payload, t: Target) => {
      switch (p.kind) {
        case "new":
          return t.drop === "slot" || t.drop === "rowgap" || t.drop === "cell";
        case "cell":
          return t.drop === "slot" || t.drop === "rowgap" || t.drop === "trash" || (t.drop === "cell" && t.cell !== p.id) || t.drop === "row";
        case "row":
          return t.drop === "trash" || (t.drop === "rowgap" && t.stack === p.stack);
        case "wire":
          if (t.drop === "cell") return Boolean(t.wire);
          if (t.drop === "row") return Boolean(p.from.row) && t.stack === p.from.stack && t.row !== p.from.row;
          if (t.drop === "stack") return t.stack !== p.from.stack;
          return false;
      }
    },
    [],
  );

  const onDrop = (p: Payload, t: Target) => {
    const slotRow = t.drop === "row" ? t.row : t.row;
    const slotIndex = t.drop === "row" ? E.findRow(graph, t.row!)?.row.cells.length ?? 0 : t.index ?? 0;
    if (p.kind === "new") {
      const item = PALETTE.find((x) => x.key === p.key)!;
      const body = item.make(widthAt(t));
      if (t.drop === "rowgap") apply((g) => E.insertRow(g, t.stack!, t.index!, [body]));
      else apply((g) => E.insertCell(g, slotRow!, slotIndex, body));
    } else if (p.kind === "cell") {
      if (t.drop === "trash") apply((g) => E.removeCell(g, p.id));
      else if (t.drop === "rowgap")
        apply((g) => {
          const withRow = E.insertRow(g, t.stack!, t.index!, []);
          const newRow = withRow.stacks.find((s) => s.id === t.stack)!.rows[t.index!];
          return E.moveCell(withRow, p.id, newRow.id, 0);
        });
      else apply((g) => E.moveCell(g, p.id, slotRow!, slotIndex));
    } else if (p.kind === "row") {
      if (t.drop === "trash") apply((g) => E.removeRow(g, p.id));
      else apply((g) => E.moveRow(g, p.id, t.index!));
    } else {
      const to = t.drop === "cell" ? { stack: graph.stacks.find((s) => s.rows.some((r) => r.cells.some((c) => c.id === t.cell)))!.id, cell: t.cell } : t.drop === "row" ? { stack: t.stack!, row: t.row } : { stack: t.stack! };
      const res = E.connect(graph, p.from, to);
      if (res.error) toast(res.error);
      else apply(() => res.graph);
    }
  };

  const quickAdd = (key: string) => {
    const item = PALETTE.find((x) => x.key === key)!;
    if (sel?.kind === "row" && sel.ids.length === 1) {
      const t: Target = { drop: "slot", row: sel.ids[0], stack: sel.stack, index: E.findRow(graph, sel.ids[0])!.row.cells.length };
      apply((g) => E.insertCell(g, t.row!, t.index!, item.make(widthAt(t))));
    } else if (sel?.kind === "cell") {
      const h = E.findCell(graph, sel.id)!;
      apply((g) => E.insertCell(g, h.row.id, h.index + 1, item.make(widthAt({ drop: "slot", row: h.row.id, stack: h.stack.id, index: h.index + 1 }))));
    } else {
      const s = graph.stacks.find((x) => x.seq.kind === "tokens") ?? graph.stacks[0];
      const head = s.rows.findIndex((r) => r.cells.some((c) => c.type === "lm_head"));
      const index = head >= 0 ? head : s.rows.length;
      apply((g) => E.insertRow(g, s.id, index, [item.make(widthAt({ drop: "rowgap", stack: s.id, index }))]));
      toast(`Added ${item.label} as a new row in ${s.name}. Select a row first to add into it.`);
    }
  };

  const load = (g: Graph) => {
    apply(() => g);
    setSelRaw(null);
  };

  return (
    <DragProvider accepts={accepts} onDrop={onDrop}>
      <div className="app">
        <TopBar graph={graph} load={load} apply={apply} undo={undo} redo={redo} canUndo={hist.canUndo} canRedo={hist.canRedo} toast={toast} />
        <SummaryStrip analysis={analysis} graph={graph} onOpen={setTab} />
        <div className="main">
          <Palette onQuickAdd={quickAdd} />
          <Frame
            graph={graph}
            analysis={analysis}
            sel={sel}
            setSel={setSel}
            onAddRow={(stackId) => {
              const s = graph.stacks.find((x) => x.id === stackId)!;
              apply((g) => E.insertRow(g, stackId, s.rows.length, []));
            }}
            onGroupCount={(id, count) => apply((g) => E.updateGroup(g, id, { count }), `group:${id}`)}
          />
          <aside className="side">
            <nav className="tabs">
              {TABS.map(([k, l]) => (
                <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
                  {l}
                </button>
              ))}
            </nav>
            <div className="side-body">
              {tab === "inspect" && <Inspector graph={graph} analysis={analysis} sel={sel} setSel={setSel} apply={apply} toast={toast} />}
              {tab === "cost" && <CostPanel graph={graph} analysis={analysis} apply={apply} setSel={setSel} />}
              {tab === "train" && <TrainingPanel graph={graph} analysis={analysis} apply={apply} setSel={setSel} />}
              {tab === "length" && <LengthPanel graph={graph} analysis={analysis} apply={apply} setSel={setSel} />}
            </div>
          </aside>
        </div>
        {toastMsg && (
          <div className="toast" role="status" onClick={() => setToastMsg(null)}>
            {toastMsg}
          </div>
        )}
      </div>
    </DragProvider>
  );
}

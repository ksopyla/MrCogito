// The graph with undo / redo. Quick runs of the same edit (dragging a slider) fold into one step.
import { useCallback, useMemo, useState } from "react";
import type { Graph } from "../engine/types";

export interface HistoryState {
  past: Graph[];
  present: Graph;
  future: Graph[];
  key: string | null;
  at: number;
}

const LIMIT = 200;
const FOLD_MS = 1200;

/** One edit. A keyed edit repeated within FOLD_MS replaces the last one: it is applied to the state before the
 *  run began, so a relative edit (rescaling a stack's width) ends where one jump to the final value would. */
export function historyStep(st: HistoryState, fn: (g: Graph) => Graph, key: string | undefined, now: number): HistoryState {
  const fold = key !== undefined && key === st.key && now - st.at < FOLD_MS && st.past.length > 0;
  const base = fold ? st.past[st.past.length - 1] : st.present;
  const next = fn(base);
  if (!fold && next === st.present) return st;
  return {
    past: fold ? st.past : [...st.past, st.present].slice(-LIMIT),
    present: next,
    future: [],
    key: key ?? null,
    at: now,
  };
}

export function useHistory(initial: () => Graph) {
  const [s, setS] = useState<HistoryState>(() => ({ past: [], present: initial(), future: [], key: null, at: 0 }));

  const apply = useCallback((fn: (g: Graph) => Graph, key?: string) => setS((st) => historyStep(st, fn, key, Date.now())), []);
  const undo = useCallback(() => setS((st) => (st.past.length ? { past: st.past.slice(0, -1), present: st.past[st.past.length - 1], future: [st.present, ...st.future], key: null, at: 0 } : st)), []);
  const redo = useCallback(() => setS((st) => (st.future.length ? { past: [...st.past, st.present], present: st.future[0], future: st.future.slice(1), key: null, at: 0 } : st)), []);

  return { graph: s.present, apply, undo, redo, canUndo: s.past.length > 0, canRedo: s.future.length > 0 };
}

// ------------------------------------------------------------------ several open designs, each with its own undo history

export interface Tab {
  id: string;
  hist: HistoryState;
}
interface Workspace {
  tabs: Tab[];
  active: string;
}

const fresh = (g: Graph): HistoryState => ({ past: [], present: g, future: [], key: null, at: 0 });
let tabCounter = 0;
const tabId = () => `tab${Date.now().toString(36)}${(tabCounter++).toString(36)}`;

export function useWorkspace(initial: () => Graph[], activeIndex = 0) {
  const [ws, setWs] = useState<Workspace>(() => {
    const tabs = initial().map((g) => ({ id: tabId(), hist: fresh(g) }));
    return { tabs, active: tabs[Math.min(activeIndex, tabs.length - 1)].id };
  });
  const current = ws.tabs.find((t) => t.id === ws.active) ?? ws.tabs[0];
  const onActive = useCallback((f: (h: HistoryState) => HistoryState) => setWs((w) => ({ ...w, tabs: w.tabs.map((t) => (t.id === w.active ? { ...t, hist: f(t.hist) } : t)) })), []);

  const apply = useCallback((fn: (g: Graph) => Graph, key?: string) => onActive((h) => historyStep(h, fn, key, Date.now())), [onActive]);
  const undo = useCallback(() => onActive((st) => (st.past.length ? { past: st.past.slice(0, -1), present: st.past[st.past.length - 1], future: [st.present, ...st.future], key: null, at: 0 } : st)), [onActive]);
  const redo = useCallback(() => onActive((st) => (st.future.length ? { past: [...st.past, st.present], present: st.future[0], future: st.future.slice(1), key: null, at: 0 } : st)), [onActive]);
  const open = useCallback((g: Graph) => setWs((w) => {
    const t = { id: tabId(), hist: fresh(g) };
    return { tabs: [...w.tabs, t], active: t.id };
  }), []);
  const switchTo = useCallback((id: string) => setWs((w) => (w.tabs.some((t) => t.id === id) ? { ...w, active: id } : w)), []);
  const close = useCallback((id: string, fallback: () => Graph) => setWs((w) => {
    const i = w.tabs.findIndex((t) => t.id === id);
    if (i < 0) return w;
    const tabs = w.tabs.filter((t) => t.id !== id);
    if (!tabs.length) {
      const t = { id: tabId(), hist: fresh(fallback()) };
      return { tabs: [t], active: t.id };
    }
    return { tabs, active: w.active === id ? tabs[Math.min(i, tabs.length - 1)].id : w.active };
  }), []);

  // a design's entry changes only when that design does, so per-tab work (analyses, saving) is not redone needlessly
  const tabs = useMemo(() => ws.tabs.map((t) => ({ id: t.id, graph: t.hist.present })), [ws.tabs]);
  return {
    graph: current.hist.present,
    apply,
    undo,
    redo,
    canUndo: current.hist.past.length > 0,
    canRedo: current.hist.future.length > 0,
    tabs,
    active: current.id,
    open,
    switchTo,
    close,
  };
}

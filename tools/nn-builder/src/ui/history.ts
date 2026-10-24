// The graph with undo / redo. Quick runs of the same edit (dragging a slider) fold into one step.
import { useCallback, useState } from "react";
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

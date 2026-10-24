// Pointer-based drag and drop: palette → slot, slot → slot, row → row, and dot → target for wires.
// One gesture system for everything, so it works the same with a mouse, a pen or a touch screen.
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { Endpoint } from "../engine/types";

export type Payload =
  | { kind: "new"; key: string; label: string }
  | { kind: "cell"; id: string; label: string }
  | { kind: "row"; id: string; stack: string; label: string }
  | { kind: "wire"; from: Endpoint; label: string };

export interface Target {
  drop: "slot" | "rowgap" | "row" | "cell" | "stack" | "trash";
  row?: string;
  stack?: string;
  cell?: string;
  index?: number;
  wire?: boolean; // a cell that takes a wire (attention)
}

interface Live {
  payload: Payload;
  x: number;
  y: number;
  sx: number;
  sy: number;
  target: Target | null;
  ok: boolean;
}

interface Ctx {
  live: Live | null;
  start: (e: React.PointerEvent, payload: Payload, onClick?: (e: PointerEvent) => void) => void;
}

const DragCtx = createContext<Ctx>({ live: null, start: () => {} });
export const useDrag = () => useContext(DragCtx);

function readTarget(x: number, y: number): Target | null {
  const el = document.elementFromPoint(x, y)?.closest<HTMLElement>("[data-drop]");
  if (!el) return null;
  const d = el.dataset;
  const t: Target = { drop: d.drop as Target["drop"], row: d.row, stack: d.stack, cell: d.cell, wire: d.wire === "1" };
  if (d.index !== undefined) t.index = Number(d.index);
  if (t.drop === "cell" && t.index !== undefined) {
    const r = el.getBoundingClientRect();
    if (x > r.left + r.width / 2) t.index += 1;
  }
  return t;
}

export function DragProvider(props: { children: ReactNode; accepts: (p: Payload, t: Target) => boolean; onDrop: (p: Payload, t: Target) => void }) {
  const [live, setLive] = useState<Live | null>(null);
  const ref = useRef(props);
  ref.current = props;

  const start = useCallback((e: React.PointerEvent, payload: Payload, onClick?: (e: PointerEvent) => void) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault(); // no text selection or native drag while the pointer is down
    const sx = e.clientX, sy = e.clientY;
    let dragging = false;
    let cur: Live = { payload, x: sx, y: sy, sx, sy, target: null, ok: false };
    // near an edge of the frame, scroll it so the other end of a wire or a far row comes into view
    const pane = document.querySelector<HTMLElement>(".frame-scroll");
    let raf = 0;
    const edgeScroll = () => {
      raf = 0;
      if (!pane || !dragging || Math.hypot(cur.x - sx, cur.y - sy) < 40) return;
      const r = pane.getBoundingClientRect();
      const edge = 48, speed = 14;
      const dx = cur.x < r.left + edge && cur.x > r.left - 200 ? -speed : cur.x > r.right - edge && cur.x < r.right + 200 ? speed : 0;
      const dy = cur.y < r.top + edge ? -speed : cur.y > r.bottom - edge ? speed : 0;
      if (!dx && !dy) return;
      pane.scrollBy(dx, dy);
      const target = readTarget(cur.x, cur.y);
      cur = { ...cur, target, ok: target ? ref.current.accepts(payload, target) : false };
      setLive(cur);
      raf = requestAnimationFrame(edgeScroll);
    };
    const move = (ev: PointerEvent) => {
      if (!dragging && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 5) return;
      if (!dragging) {
        dragging = true;
        document.body.dataset.drag = payload.kind;
        window.getSelection()?.removeAllRanges();
      }
      const target = readTarget(ev.clientX, ev.clientY);
      cur = { ...cur, x: ev.clientX, y: ev.clientY, target, ok: target ? ref.current.accepts(payload, target) : false };
      setLive(cur);
      if (!raf) raf = requestAnimationFrame(edgeScroll);
    };
    const end = (ev: PointerEvent) => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("keydown", esc);
      delete document.body.dataset.drag;
      setLive(null);
      if (!dragging) return onClick?.(ev);
      if (cur.target && cur.ok) ref.current.onDrop(payload, cur.target);
    };
    const esc = (ev: KeyboardEvent) => {
      if (ev.key !== "Escape") return;
      cur = { ...cur, target: null, ok: false };
      end(new PointerEvent("pointerup"));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("keydown", esc);
  }, []);

  useEffect(() => () => void delete document.body.dataset.drag, []);

  return (
    <DragCtx.Provider value={{ live, start }}>
      {props.children}
      {live && <Ghost live={live} />}
    </DragCtx.Provider>
  );
}

function Ghost({ live }: { live: Live }) {
  if (live.payload.kind === "wire")
    return (
      <svg className="drag-wire" width="100%" height="100%">
        <path d={`M ${live.sx} ${live.sy} C ${live.sx + 60} ${live.sy}, ${live.x - 60} ${live.y}, ${live.x} ${live.y}`} className={live.ok ? "ok" : ""} />
        <circle cx={live.x} cy={live.y} r={5} className={live.ok ? "ok" : ""} />
      </svg>
    );
  return (
    <div className={`drag-ghost ${live.ok ? "ok" : ""}`} style={{ left: live.x + 12, top: live.y + 10 }}>
      {live.payload.label}
      {live.target?.drop === "trash" && live.ok ? " → remove" : ""}
    </div>
  );
}

/** True while this drop target is the one under the pointer and accepts the drag. */
export function useIsOver(match: (t: Target) => boolean): "" | "over" | "over-bad" {
  const { live } = useDrag();
  if (!live?.target || !match(live.target)) return "";
  return live.ok ? "over" : "over-bad";
}

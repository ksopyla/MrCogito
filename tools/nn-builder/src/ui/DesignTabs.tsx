// The open designs as tabs: click to switch, × to close, + to copy the current one and try a variant.
// A dot after the name marks a design that differs from its last save (or was never saved).
import { useMemo } from "react";
import { analyze } from "../engine/analyze";
import type { Graph } from "../engine/types";
import * as F from "./format";

export function DesignTabs(p: { tabs: { id: string; graph: Graph; saved: string | null; unsaved: boolean }[]; active: string; onSwitch: (id: string) => void; onClose: (id: string) => void; onCopy: () => void }) {
  // the parameter count of every open design (cheap: one analysis each, only when a design changes)
  const counts = useMemo(() => new Map(p.tabs.map((t) => [t.id, analyze(t.graph).params.total])), [p.tabs]);
  return (
    <nav className="design-tabs" aria-label="Open designs">
      {p.tabs.map((t) => (
        <div
          key={t.id}
          className={`design-tab ${t.id === p.active ? "on" : ""} ${t.unsaved ? "unsaved" : ""}`}
          onClick={() => p.onSwitch(t.id)}
          title={`${t.graph.description || t.graph.name}${t.unsaved ? (t.saved ? "\n· changed since the last save" : "\n· not saved yet") : "\n· saved"}`}
        >
          <span className="dt-name">{t.graph.name}</span>
          <span className="dt-num">{F.params(counts.get(t.id) ?? 0)}</span>
          {p.tabs.length > 1 && (
            <button className="dt-close" aria-label={`close ${t.graph.name}`} onClick={(e) => (e.stopPropagation(), p.onClose(t.id))}>
              ×
            </button>
          )}
        </div>
      ))}
      <button className="dt-add" onClick={p.onCopy} title="copy this design into a new tab to try a variant">
        + copy
      </button>
    </nav>
  );
}

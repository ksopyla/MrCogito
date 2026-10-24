// The top bar: start from a preset, add stacks, undo / redo, and save / open / export / import graph files.
import { useRef, useState } from "react";
import * as E from "../engine/edit";
import { PRESETS } from "../engine/presets";
import { parseGraph, toExport } from "../engine/serialize";
import type { Graph } from "../engine/types";

interface Props {
  graph: Graph;
  load: (g: Graph) => void;
  apply: (fn: (g: Graph) => Graph) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  toast: (msg: string) => void;
}

export const slug = (name: string) => name.trim().toLowerCase().replace(/[^a-z0-9_.-]+/g, "_").replace(/^[_.-]+|[_.-]+$/g, "") || "design";

function download(name: string, text: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function TopBar(p: Props) {
  const [menu, setMenu] = useState<"open" | "stack" | null>(null);
  const [designs, setDesigns] = useState<{ name: string; modified: string }[] | null>(null);
  const file = useRef<HTMLInputElement>(null);

  const save = async () => {
    const name = slug(p.graph.name);
    try {
      const r = await fetch(`/api/designs/${encodeURIComponent(name)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(toExport(p.graph)) });
      const body = await r.json();
      if (!r.ok) throw new Error(body.error ?? r.statusText);
      p.toast(`Saved to tools/nn-builder/${body.saved}. Ask Claude to turn it into PyTorch code.`);
    } catch (e) {
      download(`${name}.json`, JSON.stringify(toExport(p.graph), null, 2));
      p.toast(`Could not save on disk (${(e as Error).message}); downloaded ${name}.json instead.`);
    }
  };

  const openMenu = async () => {
    setMenu(menu === "open" ? null : "open");
    try {
      const r = await fetch("/api/designs");
      setDesigns(r.ok ? await r.json() : []);
    } catch {
      setDesigns([]);
    }
  };

  const openDesign = async (name: string) => {
    setMenu(null);
    const r = await fetch(`/api/designs/${encodeURIComponent(name)}`);
    const res = parseGraph(await r.text());
    if (res.error || !res.graph) return p.toast(res.error ?? "could not open");
    p.load(res.graph);
    p.toast(`Opened designs/${name}.json`);
  };

  const importFile = async (f: File) => {
    const res = parseGraph(await f.text());
    if (res.error || !res.graph) return p.toast(`${f.name}: ${res.error}`);
    p.load(res.graph);
    p.toast(`Imported ${f.name}`);
  };

  return (
    <header className="topbar">
      <div className="brand">NN Builder</div>
      <select
        className="preset"
        value=""
        onChange={(e) => {
          const pr = PRESETS.find((x) => x.id === e.target.value);
          if (pr) p.load(pr.build());
        }}
        aria-label="Start from a preset"
      >
        <option value="">Start from…</option>
        {PRESETS.map((x) => (
          <option key={x.id} value={x.id}>
            {x.label}
          </option>
        ))}
      </select>
      <div className="menu-wrap">
        <button onClick={() => setMenu(menu === "stack" ? null : "stack")}>+ Stack</button>
        {menu === "stack" && (
          <div className="menu" onMouseLeave={() => setMenu(null)}>
            {([
              ["decoder", "Decoder (tokens, causal)"],
              ["encoder", "Encoder (its own source, two-way)"],
              ["pages", "Pages over a stack (page writer)"],
              ["slots", "Slots per page → notes"],
              ["latents", "Fixed latents (Perceiver)"],
            ] as [E.StackTemplate, string][]).map(([k, l]) => (
              <button key={k} onClick={() => (p.apply((g) => E.addStack(g, k)), setMenu(null))}>
                {l}
              </button>
            ))}
          </div>
        )}
      </div>
      <button onClick={p.undo} disabled={!p.canUndo} title="Undo (⌘Z)">↶</button>
      <button onClick={p.redo} disabled={!p.canRedo} title="Redo (⇧⌘Z)">↷</button>
      <span className="grow" />
      <button className="primary" onClick={save} title="Save to tools/nn-builder/designs/<name>.json">Save</button>
      <div className="menu-wrap">
        <button onClick={openMenu}>Open…</button>
        {menu === "open" && (
          <div className="menu right" onMouseLeave={() => setMenu(null)}>
            {designs === null && <span className="muted pad">Loading…</span>}
            {designs?.length === 0 && <span className="muted pad">No saved designs yet.</span>}
            {designs?.map((d) => (
              <button key={d.name} onClick={() => openDesign(d.name)}>
                {d.name} <span className="muted small">{d.modified.slice(0, 16).replace("T", " ")}</span>
              </button>
            ))}
            <hr />
            <button onClick={() => (setMenu(null), file.current?.click())}>Import a file…</button>
          </div>
        )}
      </div>
      <button onClick={() => download(`${slug(p.graph.name)}.json`, JSON.stringify(toExport(p.graph), null, 2))} title="Download the graph file (with the estimates)">Export</button>
      <input ref={file} type="file" accept=".json,application/json" hidden onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])} />
    </header>
  );
}

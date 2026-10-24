// The inspector: every setting of the selected cell, row, stack, wire or loop, with its own costs on top.
import { useState } from "react";
import type { Analysis, CellResult } from "../engine/analyze";
import * as E from "../engine/edit";
import type { Cell, CellBody, Edge, Graph, SeqSpec, Stack } from "../engine/types";
import { NumberField, Select, TextField, Toggle } from "./controls";
import { cellTitle, edgeLabel } from "./describe";
import { CELL_FIELDS, type Field } from "./fields";
import type { Sel } from "./Frame";
import * as F from "./format";

interface Props {
  graph: Graph;
  analysis: Analysis;
  sel: Sel;
  setSel: (s: Sel) => void;
  apply: (fn: (g: Graph) => Graph, key?: string) => void;
  toast: (msg: string) => void;
}

export function Inspector(p: Props) {
  const { sel } = p;
  if (!sel) return <GraphInfo {...p} />;
  if (sel.kind === "cell") return <CellInspector {...p} id={sel.id} />;
  if (sel.kind === "row") return <RowInspector {...p} ids={sel.ids} stack={sel.stack} />;
  if (sel.kind === "stack") return <StackInspector {...p} id={sel.id} />;
  if (sel.kind === "edge") return <EdgeInspector {...p} id={sel.id} />;
  return <GroupInspector {...p} id={sel.id} />;
}

const rowName = (g: Graph, id?: string) => {
  const h = id ? E.findRow(g, id) : undefined;
  return h ? `${h.stack.name} · ${h.row.label ?? `row ${h.index}`}` : "—";
};

function Costs({ items }: { items: [string, string, string?][] }) {
  return (
    <dl className="costs">
      {items.map(([k, v, hint]) => (
        <div key={k} title={hint}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function cellCostItems(r: CellResult, a: Analysis): [string, string, string?][] {
  const b = a.memory.microBatch;
  return [
    ["Parameters", F.exact(r.params)],
    ["Runs per row", r.execs > 1 ? `${r.execs} × (loop or repeat)` : "1"],
    ["Width in → out", `${r.widthIn} → ${r.widthOut}`],
    ["Forward / token", F.flops(r.fwdPerToken)],
    ["Backward / token", F.flops(r.bwdPerToken), "weight + input gradients; no-grad parts count 0"],
    ["Training share", `${((100 * (r.fwdPerToken + r.bwdPerToken)) / Math.max(1, a.flops.fwdPerToken + a.flops.bwdPerToken)).toFixed(1)} %`],
    ["Saved for backward", `${F.bytes(r.actBytesPerRow)} per row · ${F.bytes(r.actBytesPerRow * b)} per micro-batch of ${b}`],
  ];
}

function FieldView({ f, cell, onChange }: { f: Field; cell: Record<string, unknown>; onChange: (key: string, v: unknown, extras?: boolean) => void }) {
  const raw = f.extras ? (cell.extras as Record<string, unknown>)[f.key] : cell[f.key];
  if (f.kind === "bool") return <Toggle label={f.label} value={Boolean(raw)} help={f.help} onChange={(v) => onChange(f.key, v, f.extras)} />;
  if (f.kind === "select") return <Select label={f.label} value={raw as string} options={f.options as [string, string][]} help={f.help} onChange={(v) => onChange(f.key, v, f.extras)} />;
  if (f.kind === "text") return <TextField label={f.label} value={String(raw ?? "")} onChange={(v) => onChange(f.key, v, f.extras)} />;
  return <NumberField label={f.label} value={Number(raw)} min={f.min} max={f.max} step={f.step} log={f.log} int={f.kind === "int"} help={f.help} onChange={(v) => onChange(f.key, v, f.extras)} />;
}

function CellInspector(p: Props & { id: string }) {
  const home = E.findCell(p.graph, p.id);
  const r = p.analysis.cells.get(p.id);
  if (!home || !r) return <p className="muted pad">Nothing selected.</p>;
  const c = home.cell;
  const set = (key: string, v: unknown, extras?: boolean) =>
    p.apply((g) => E.updateCell(g, c.id, (extras ? { extras: { [key]: v } } : { [key]: v }) as Partial<CellBody>), `cell:${c.id}:${key}`);
  const fields = CELL_FIELDS[c.type].filter((f) => !f.show || f.show(c as unknown as Record<string, unknown>));
  const wire = p.graph.edges.find((e) => e.kind === "source" && e.to.cell === c.id);
  const embeddings = p.graph.stacks.flatMap((s) => s.rows.flatMap((row) => row.cells.filter((x) => x.type === "embedding" && x.id !== c.id).map((x) => [x.id, `${s.name} · ${x.label ?? "embedding"}`] as [string, string])));

  return (
    <div className="inspector">
      <h2>
        {cellTitle(c)} <span className="muted small">in {rowName(p.graph, home.row.id)}</span>
      </h2>
      <Costs items={cellCostItems(r, p.analysis)} />
      {r.parts.length > 1 && (
        <details className="parts">
          <summary>Where the {F.params(r.params)} parameters are</summary>
          <ul>
            {r.parts.map((x) => (
              <li key={x.name}>
                <span>{x.name}</span>
                <span className="num">{F.exact(x.params)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {r.issues.length > 0 && (
        <ul className="issues">
          {r.issues.map((m) => (
            <li key={m}>⚠ {m}</li>
          ))}
        </ul>
      )}
      <h3>Settings</h3>
      <TextField label="Label" value={c.label ?? ""} onChange={(v) => p.apply((g) => E.updateCell(g, c.id, { label: v || undefined } as Partial<CellBody>))} />
      {fields.map((f) => (
        <FieldView key={f.key} f={f} cell={c as unknown as Record<string, unknown>} onChange={set} />
      ))}
      {c.type === "embedding" && (
        <Select label="Share the token table with" value={c.share_with ?? ""} options={[["", "nobody (own table)"], ...embeddings]} onChange={(v) => set("share_with", v || undefined)} />
      )}
      {c.type === "attention" && (c.mode === "cross" || c.memory) && (
        <p className="muted small">
          {wire ? <>Reads: <strong>{p.graph.stacks.find((s) => s.id === wire.from.stack)?.name}</strong> {wire.from.row ? `(${rowName(p.graph, wire.from.row)})` : "(its output)"}. </> : "No wire yet: drag from a stack's output dot onto this cell. "}
          {wire && <button className="link" onClick={() => p.apply((g) => E.removeEdge(g, wire.id))}>remove wire</button>}
        </p>
      )}
      <div className="actions">
        <button onClick={() => p.apply((g) => E.insertCell(g, home.row.id, home.index + 1, structuredClone(c) as CellBody))}>Duplicate</button>
        <button className="danger" onClick={() => (p.apply((g) => E.removeCell(g, c.id)), p.setSel(null))}>Remove</button>
      </div>
    </div>
  );
}

function RowInspector(p: Props & { ids: string[]; stack: string }) {
  const [times, setTimes] = useState(1);
  const [count, setCount] = useState(4);
  const rows = p.ids.map((id) => E.findRow(p.graph, id)).filter(Boolean) as NonNullable<ReturnType<typeof E.findRow>>[];
  if (!rows.length) return <p className="muted pad">Nothing selected.</p>;
  const one = rows.length === 1 ? rows[0] : null;
  const res = p.analysis.stacks.find((s) => s.id === p.stack)?.rows.filter((r) => p.ids.includes(r.id)) ?? [];
  const sum = (f: (r: (typeof res)[number]) => number) => res.reduce((n, r) => n + f(r), 0);
  const inGroup = p.graph.groups.find((g) => g.rows.some((r) => p.ids.includes(r)));
  const group = (kind: "loop" | "repeat") => {
    try {
      p.apply((g) => E.addGroup(g, kind, p.stack, p.ids, count));
    } catch (e) {
      p.toast((e as Error).message);
    }
  };
  const skips = one ? p.graph.edges.filter((e) => e.kind === "skip" && e.to.row === one.row.id) : [];

  return (
    <div className="inspector">
      <h2>{one ? rowName(p.graph, one.row.id) : `${rows.length} rows of ${rows[0].stack.name}`}</h2>
      <Costs
        items={[
          ["Parameters", F.exact(sum((r) => r.params))],
          ["Forward / token", F.flops(sum((r) => r.fwdPerToken))],
          ["Saved for backward", `${F.bytes(sum((r) => r.actBytesPerRow))} per row`],
        ]}
      />
      {one && (
        <>
          <h3>Row</h3>
          <TextField label="Label" value={one.row.label ?? ""} onChange={(v) => p.apply((g) => E.updateRow(g, one.row.id, { label: v || undefined }))} />
          <Toggle label="Mix in x0 (α·x + β·x0, 2 weights)" value={one.row.x0_mix} help="x0 = the stack's stream after its input row" onChange={(v) => p.apply((g) => E.updateRow(g, one.row.id, { x0_mix: v }))} />
          {skips.length > 0 && <h3>Skip wires into this row</h3>}
          {skips.map((e) => (
            <div key={e.id} className="field-row">
              <span className="small">from {rowName(p.graph, e.from.row)}</span>
              <button className="link" onClick={() => p.setSel({ kind: "edge", id: e.id })}>edit</button>
            </div>
          ))}
          <div className="actions">
            <button onClick={() => p.apply((g) => E.moveRow(g, one.row.id, Math.max(0, one.index - 1)))}>↑ Up</button>
            <button onClick={() => p.apply((g) => E.moveRow(g, one.row.id, one.index + 2))}>↓ Down</button>
          </div>
          <div className="actions">
            <NumberField label="Copies" value={times} min={1} max={64} int onChange={setTimes} />
            <button onClick={() => p.apply((g) => E.duplicateRow(g, one.row.id, times))}>Duplicate ×{times}</button>
          </div>
        </>
      )}
      <h3>Loop or repeat {rows.length > 1 ? "these rows" : "this row"}</h3>
      {inGroup ? (
        <p className="muted small">
          Already in a {inGroup.kind}. <button className="link" onClick={() => p.setSel({ kind: "group", id: inGroup.id })}>edit it</button>
        </p>
      ) : (
        <>
          <NumberField label="Rounds / copies" value={count} min={1} max={64} int onChange={setCount} />
          <div className="actions">
            <button onClick={() => group("loop")} title="the same weights run again and again">↻ Loop ×{count} (shared weights)</button>
            <button onClick={() => group("repeat")} title="independent copies of the rows">⧉ Repeat ×{count} (copies)</button>
          </div>
          <p className="muted small">Shift-click another row handle to select a range of rows.</p>
        </>
      )}
      <div className="actions">
        <button className="danger" onClick={() => (p.apply((g) => p.ids.reduce((x, id) => E.removeRow(x, id), g)), p.setSel(null))}>
          Remove {rows.length > 1 ? `${rows.length} rows` : "row"}
        </button>
      </div>
    </div>
  );
}

function seqOptions(s: Stack, g: Graph): [SeqSpec["kind"], string][] {
  const opts: [SeqSpec["kind"], string][] = [["tokens", "the training row (tokens)"], ["source", "its own source sequence (encoder)"], ["latents", "a fixed set of latents"]];
  if (g.stacks.some((x) => x.id !== s.id)) opts.push(["pages", "pages over another stack"]);
  if (g.stacks.some((x) => x.seq.kind === "pages" && x.id !== s.id)) opts.push(["slots", "slots on each page of a pages stack"]);
  return opts;
}

function StackInspector(p: Props & { id: string }) {
  const s = p.graph.stacks.find((x) => x.id === p.id);
  const res = p.analysis.stacks.find((x) => x.id === p.id);
  if (!s || !res) return <p className="muted pad">Nothing selected.</p>;
  const setSeq = (seq: SeqSpec) => p.apply((g) => E.updateStack(g, s.id, { seq }), `seq:${s.id}`);
  const others = p.graph.stacks.filter((x) => x.id !== s.id).map((x) => [x.id, x.name] as [string, string]);
  const pagesStacks = p.graph.stacks.filter((x) => x.seq.kind === "pages" && x.id !== s.id).map((x) => [x.id, x.name] as [string, string]);
  const q = s.seq;
  const changeKind = (k: SeqSpec["kind"]) => {
    if (k === "tokens") setSeq({ kind: "tokens" });
    if (k === "source") setSeq({ kind: "source", length: 0 });
    if (k === "latents") setSeq({ kind: "latents", count: 256 });
    if (k === "pages") setSeq({ kind: "pages", of: others[0][0], window: 256, stride: 192 });
    if (k === "slots") setSeq({ kind: "slots", of: pagesStacks[0][0], per_page: 32 });
  };
  const issues = p.analysis.issues.filter((i) => i.where === s.id);
  return (
    <div className="inspector">
      <h2>{s.name}</h2>
      <Costs
        items={[
          ["Parameters", F.exact(res.params)],
          ["Positions per row", F.exact(res.P)],
          ["Forward / token", F.flops(res.fwdPerToken)],
          ["Output width", String(res.widthOut)],
        ]}
      />
      {issues.length > 0 && <ul className="issues">{issues.map((i) => <li key={i.msg}>⚠ {i.msg}</li>)}</ul>}
      <h3>Stack</h3>
      <TextField label="Name" value={s.name} onChange={(v) => p.apply((g) => E.updateStack(g, s.id, { name: v }))} />
      <Select label="Role" value={s.role} options={[["decoder", "decoder"], ["encoder", "encoder"], ["writer", "writer"], ["latent", "latent / notes"], ["other", "other"]]} onChange={(v) => p.apply((g) => E.updateStack(g, s.id, { role: v }))} />
      <NumberField label="Width (d_model)" value={s.width} min={8} max={32768} step={8} log int help="full-width attention and FFN ratios follow the width" onChange={(v) => p.apply((g) => E.setStackWidth(g, s.id, v), `width:${s.id}`)} />
      <Select label="Sequence" value={q.kind} options={seqOptions(s, p.graph)} onChange={changeKind} />
      {q.kind === "source" && <NumberField label="Source length (0 = the row length)" value={q.length} min={0} max={1048576} log int onChange={(v) => setSeq({ ...q, length: v })} />}
      {q.kind === "latents" && <NumberField label="Latents" value={q.count} min={1} max={65536} log int onChange={(v) => setSeq({ ...q, count: v })} />}
      {q.kind === "pages" && (
        <>
          <Select label="Pages over" value={q.of} options={others} onChange={(v) => setSeq({ ...q, of: v })} />
          <NumberField label="Page width" value={q.window} min={1} max={65536} log int onChange={(v) => setSeq({ ...q, window: v })} />
          <NumberField label="Page stride" value={q.stride} min={1} max={65536} log int help="smaller than the width = overlapping pages" onChange={(v) => setSeq({ ...q, stride: v })} />
        </>
      )}
      {q.kind === "slots" && (
        <>
          <Select label="Slots on the pages of" value={q.of} options={pagesStacks} onChange={(v) => setSeq({ ...q, of: v })} />
          <NumberField label="Slots per page" value={q.per_page} min={1} max={4096} log int onChange={(v) => setSeq({ ...q, per_page: v })} />
        </>
      )}
      <h3>Wires</h3>
      <div className="actions">
        <button onClick={() => p.apply((g) => E.unetSkips(g, s.id))} title="layers in the first half feed the mirrored layers of the second half">U-net skips</button>
        <button onClick={() => p.apply((g) => ({ ...g, edges: g.edges.filter((e) => !(e.kind === "skip" && e.to.stack === s.id)) }))}>Remove skips</button>
      </div>
      <div className="actions">
        <button onClick={() => p.apply((g) => E.moveStack(g, s.id, -1))}>← Left</button>
        <button onClick={() => p.apply((g) => E.moveStack(g, s.id, 1))}>Right →</button>
        <button className="danger" disabled={p.graph.stacks.length < 2} onClick={() => (p.apply((g) => E.removeStack(g, s.id)), p.setSel(null))}>Remove stack</button>
      </div>
    </div>
  );
}

function EdgeInspector(p: Props & { id: string }) {
  const e = p.graph.edges.find((x) => x.id === p.id) as Edge | undefined;
  if (!e) return <p className="muted pad">Nothing selected.</p>;
  const issues = p.analysis.issues.filter((i) => i.where === e.id);
  const end = (x: Edge["from"]) => (x.cell ? `${p.graph.stacks.find((s) => s.id === x.stack)?.name} · ${x.port === "table" ? "token table" : cellTitle(E.findCell(p.graph, x.cell)?.cell as Cell)}` : x.row ? rowName(p.graph, x.row) : `${p.graph.stacks.find((s) => s.id === x.stack)?.name} output`);
  return (
    <div className="inspector">
      <h2>{edgeLabel(e, p.graph)}</h2>
      <Costs items={[["From", end(e.from)], ["To", end(e.to)]]} />
      {issues.length > 0 && <ul className="issues">{issues.map((i) => <li key={i.msg}>⚠ {i.msg}</li>)}</ul>}
      {e.kind === "skip" && (
        <Select
          label="Skip weight"
          value={e.weight}
          options={[["scalar", "one learned number (U-net σ)"], ["vector", "one per channel"], ["none", "plain add"]]}
          onChange={(v) => p.apply((g) => E.updateEdge(g, e.id, { weight: v } as Partial<Edge>))}
        />
      )}
      <p className="muted small">
        {e.kind === "skip" && "The source row's output is added (times the weight) to this row's input."}
        {e.kind === "source" && "The target cell reads its keys and values (or ready notes) from here."}
        {e.kind === "input" && "The target stack's stream starts from this output."}
      </p>
      <div className="actions">
        <button className="danger" onClick={() => (p.apply((g) => E.removeEdge(g, e.id)), p.setSel(null))}>Remove wire</button>
      </div>
    </div>
  );
}

function GroupInspector(p: Props & { id: string }) {
  const grp = p.graph.groups.find((x) => x.id === p.id);
  const res = p.analysis.stacks.flatMap((s) => s.groups).find((x) => x.id === p.id);
  if (!grp || !res) return <p className="muted pad">Nothing selected.</p>;
  const set = (patch: Parameters<typeof E.updateGroup>[2]) => p.apply((g) => E.updateGroup(g, grp.id, patch), `group:${grp.id}`);
  return (
    <div className="inspector">
      <h2>{grp.kind === "loop" ? "Loop" : "Repeat"} over {grp.rows.length} row{grp.rows.length > 1 ? "s" : ""}</h2>
      <Costs
        items={[
          ["Rows", grp.rows.map((r) => rowName(p.graph, r)).join(", ")],
          ["Extra parameters", F.exact(res.params)],
          ["Feedback forward / token", F.flops(res.fwdPerToken)],
        ]}
      />
      <Select label="Kind" value={grp.kind} options={[["loop", "loop: the same weights run again"], ["repeat", "repeat: independent copies"]]} onChange={(v) => set({ kind: v })} />
      <NumberField label={grp.kind === "loop" ? "Rounds" : "Copies"} value={grp.count} min={1} max={64} int onChange={(v) => set({ count: v })} />
      {grp.kind === "loop" && (
        <>
          <Toggle label="A learned vector per round (rounds × width)" value={grp.round_embedding} onChange={(v) => set({ round_embedding: v })} />
          <Select label="Feedback between rounds" value={grp.feedback} options={[["none", "none"], ["soft", "soft: the round's LM-head reading, re-embedded (E33q)"]]} onChange={(v) => set({ feedback: v })} />
        </>
      )}
      <div className="actions">
        <button className="danger" onClick={() => (p.apply((g) => E.removeGroup(g, grp.id)), p.setSel(null))}>Ungroup</button>
      </div>
    </div>
  );
}

function GraphInfo(p: Props) {
  const issues = p.analysis.issues;
  return (
    <div className="inspector">
      <h2>{p.graph.name}</h2>
      <TextField label="Name" value={p.graph.name} onChange={(v) => p.apply((g) => ({ ...g, name: v }))} />
      <label className="field">
        <span className="field-label">Notes</span>
        <textarea rows={4} defaultValue={p.graph.description} key={p.graph.description} onBlur={(e) => p.apply((g) => ({ ...g, description: e.target.value }))} />
      </label>
      <h3>How to build</h3>
      <ul className="howto">
        <li>Drag a layer from the left into a <b>+</b> slot (same row) or between rows (a new row).</li>
        <li>Drag a slot or a row handle <b>⋮⋮</b> to move it; drop it on the palette to remove it.</li>
        <li>Drag from a dot <span className="dot-inline" />: to a later row (skip), to a cross-attention or notebook read (keys/values), or onto a stack header (its input).</li>
        <li>Shift-click row handles to select several rows, then loop or repeat them.</li>
        <li>Click anything to see and change its settings and costs here.</li>
      </ul>
      {issues.length > 0 && (
        <>
          <h3>Warnings</h3>
          <ul className="issues">
            {issues.map((i, k) => (
              <li key={k}>⚠ {i.msg}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

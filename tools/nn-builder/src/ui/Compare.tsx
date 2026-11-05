// Open designs side by side: one column per design, the same training setup for all (or each its own).
import { useMemo, useState } from "react";
import { analyze, type Analysis } from "../engine/analyze";
import { gpuById } from "../engine/hardware";
import type { Graph } from "../engine/types";
import { Toggle } from "./controls";
import * as F from "./format";

interface Props {
  tabs: { id: string; graph: Graph }[];
  active: string;
  onSwitch: (id: string) => void;
}

interface Metric {
  name: string;
  value: (a: Analysis) => number;
  fmt: (v: number, a: Analysis) => string;
  lower?: boolean; // lower is better: the best column is marked
}

const METRICS: Metric[] = [
  { name: "Parameters", value: (a) => a.params.total, fmt: F.params },
  { name: "Trained", value: (a) => a.params.trainable, fmt: F.params },
  { name: "Used per token", value: (a) => a.params.active, fmt: F.params },
  { name: "Training compute / token", value: (a) => a.flops.trainPerToken, fmt: F.flops, lower: true },
  { name: "Attention reads (share of forward)", value: (a) => a.flops.keysFwdPerToken / Math.max(1, a.flops.fwdPerToken), fmt: (v) => `${(v * 100).toFixed(0)} %` },
  { name: "Memory per GPU", value: (a) => a.memory.total, fmt: (v, a) => `${F.bytes(v)} ${a.memory.fits ? "✓" : "✗"}`, lower: true },
  { name: "Rows that fit per GPU", value: (a) => a.memory.maxMicroBatch, fmt: (v) => String(v) },
  { name: "Speed per GPU (tokens/s)", value: (a) => a.time.tokensPerSecondPerGpu, fmt: F.rate },
  { name: "Training tokens", value: (a) => a.tokens.budget, fmt: F.tokens },
  { name: "Time", value: (a) => a.time.hours, fmt: F.hours, lower: true },
  { name: "Whole-run compute", value: (a) => a.flops.perRun, fmt: F.flops, lower: true },
];

export function ComparePanel({ tabs, active, onSwitch }: Props) {
  const [same, setSame] = useState(true);
  const base = tabs.find((t) => t.id === active)?.graph ?? tabs[0].graph;
  const runs = useMemo(
    () => tabs.map((t) => ({ ...t, a: analyze(same ? { ...t.graph, run: { ...base.run } } : t.graph) })),
    [tabs, same, base.run],
  );
  const gpu = gpuById(base.run.gpu);
  return (
    <div className="panel">
      <p className="small muted">
        Every open design, side by side. Open more from <b>Start from…</b> or <b>Open…</b>, or copy this one with <b>+</b> on the tab bar and change it.
      </p>
      <Toggle
        label={`Same training setup for all: this design's (${base.run.gpu_count} × ${gpu.label.split(" ·")[0]}, ${F.len(base.run.row_length)} rows, ${base.run.token_rule && base.run.token_rule !== "custom" ? base.run.token_rule : F.tokens(base.run.token_budget)} tokens)`}
        value={same}
        onChange={setSame}
      />
      {runs.length < 2 && <p className="small">Only one design is open: open another to compare.</p>}
      <div className="compare-scroll">
        <table className="compare">
          <thead>
            <tr>
              <th />
              {runs.map((r) => (
                <th key={r.id} className={r.id === active ? "current" : ""}>
                  <button className="link" onClick={() => onSwitch(r.id)} title="switch to this design">
                    {r.graph.name}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => {
              const vals = runs.map((r) => m.value(r.a));
              const max = Math.max(...vals.map((v) => (Number.isFinite(v) ? v : 0)), 1e-12);
              const best = m.lower ? Math.min(...vals.filter(Number.isFinite)) : NaN;
              return (
                <tr key={m.name}>
                  <th scope="row">{m.name}</th>
                  {runs.map((r, i) => (
                    <td key={r.id} className={`${r.id === active ? "current" : ""} ${runs.length > 1 && vals[i] === best ? "best" : ""}`}>
                      <span className="cmp-val">{m.fmt(vals[i], r.a)}</span>
                      <span className="cmp-bar" style={{ width: `${Number.isFinite(vals[i]) ? Math.max(2, (vals[i] / max) * 100) : 0}%` }} />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">Bars are relative to the largest value in the row; bold marks the lowest cost.</p>
    </div>
  );
}

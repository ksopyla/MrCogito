// Small form controls: a number with a slider (log-scaled for wide ranges), a switch, a select.
import { useEffect, useState } from "react";

const toSlider = (v: number, min: number, max: number, log?: boolean) => {
  if (!log) return v;
  const lo = Math.log(Math.max(min, 1)), hi = Math.log(Math.max(max, 2));
  return Math.round(((Math.log(Math.max(v, 1)) - lo) / (hi - lo)) * 1000);
};
const fromSlider = (s: number, min: number, max: number, log?: boolean, step = 1) => {
  if (!log) return s;
  const lo = Math.log(Math.max(min, 1)), hi = Math.log(Math.max(max, 2));
  const v = Math.exp(lo + (s / 1000) * (hi - lo));
  return Math.max(min, Math.round(v / step) * step);
};

export function NumberField(p: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  log?: boolean;
  int?: boolean;
  help?: string;
  suffix?: string;
}) {
  const min = p.min ?? 0, max = p.max ?? 1e6, step = p.step ?? (p.int ? 1 : 0.01);
  const [text, setText] = useState(String(p.value));
  useEffect(() => setText(String(p.value)), [p.value]);
  const commit = (raw: string) => {
    const n = Number(raw.replace(/[_,\s]/g, "").replace(/k$/i, "e3").replace(/m$/i, "e6").replace(/b$/i, "e9"));
    if (!Number.isFinite(n)) return setText(String(p.value));
    const v = Math.min(max, Math.max(min, p.int ? Math.round(n) : n));
    if (v !== p.value) p.onChange(v);
    else setText(String(p.value));
  };
  return (
    <label className="field">
      <span className="field-label" title={p.help}>
        {p.label}
        {p.help && <span className="help"> ⓘ</span>}
      </span>
      <div className="field-row">
        <input
          type="range"
          min={p.log ? 0 : min}
          max={p.log ? 1000 : max}
          step={p.log ? 1 : step}
          value={toSlider(p.value, min, max, p.log)}
          onChange={(e) => {
            const v = fromSlider(Number(e.target.value), min, max, p.log, p.log ? step : 1);
            p.onChange(p.int ? Math.round(v) : v);
          }}
        />
        <input
          className="num-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={(e) => commit(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && commit((e.target as HTMLInputElement).value)}
        />
        {p.suffix && <span className="muted">{p.suffix}</span>}
      </div>
    </label>
  );
}

export function Toggle(p: { label: string; value: boolean; onChange: (v: boolean) => void; help?: string }) {
  return (
    <label className="field toggle" title={p.help}>
      <input type="checkbox" checked={p.value} onChange={(e) => p.onChange(e.target.checked)} />
      <span>{p.label}</span>
      {p.help && <span className="help"> ⓘ</span>}
    </label>
  );
}

export function Select<T extends string | number>(p: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void; help?: string }) {
  return (
    <label className="field">
      <span className="field-label" title={p.help}>
        {p.label}
        {p.help && <span className="help"> ⓘ</span>}
      </span>
      <select
        value={String(p.value)}
        onChange={(e) => {
          const hit = p.options.find(([v]) => String(v) === e.target.value);
          if (hit) p.onChange(hit[0]);
        }}
      >
        {p.options.map(([v, l]) => (
          <option key={String(v)} value={String(v)}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}

export function TextField(p: { label: string; value: string; onChange: (v: string) => void }) {
  const [text, setText] = useState(p.value);
  useEffect(() => setText(p.value), [p.value]);
  return (
    <label className="field">
      <span className="field-label">{p.label}</span>
      <input className="text-input" value={text} onChange={(e) => setText(e.target.value)} onBlur={() => text !== p.value && p.onChange(text)} onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />
    </label>
  );
}

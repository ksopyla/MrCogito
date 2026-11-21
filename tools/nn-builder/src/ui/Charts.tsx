// Small SVG charts: a line chart over row length (log axes) with a hover crosshair, and a stacked memory bar.
import { useState } from "react";
import * as F from "./format";

export interface Series {
  name: string;
  color: string; // a CSS variable name, e.g. "--series-1"
  values: (number | null)[];
  dashed?: boolean;
}

const W = 380, H = 190, L = 50, R = 12, T = 12, B = 26;

export function LineChart(p: { xs: number[]; series: Series[]; yFormat: (v: number) => string; yTick?: (v: number) => string; xFormat?: (v: number) => string; title: string; note?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const all = p.series.flatMap((s) => s.values.filter((v): v is number => v !== null && v > 0));
  if (!all.length) return null;
  const lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
  const y0 = Math.floor(lo), y1 = Math.max(y0 + 1, Math.ceil(hi === lo ? hi + 0.5 : hi));
  const xl = Math.log2(p.xs[0]), xh = Math.log2(p.xs[p.xs.length - 1]);
  const X = (x: number) => L + ((Math.log2(x) - xl) / (xh - xl || 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - T - B);
  const ticks = [];
  for (let e = y0; e <= y1; e++) ticks.push(10 ** e);
  const xf = p.xFormat ?? F.len;
  const xTicks = p.xs.filter((_, i) => i % 2 === 0);
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    p.xs.forEach((x, i) => Math.abs(X(x) - px) < Math.abs(X(p.xs[best]) - px) && (best = i));
    setHover(best);
  };
  return (
    <figure className="chart">
      <figcaption>
        <strong>{p.title}</strong>
        {p.series.length > 1 && (
          <span className="legend">
            {p.series.map((s) => (
              <span key={s.name}>
                <i style={{ background: `var(${s.color})` }} className={s.dashed ? "dashed" : ""} />
                {s.name}
              </span>
            ))}
          </span>
        )}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} onPointerMove={onMove} onPointerLeave={() => setHover(null)} role="img" aria-label={p.title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} className="grid" />
            <text x={L - 6} y={Y(t) + 3} className="tick" textAnchor="end">
              {(p.yTick ?? p.yFormat)(t)}
            </text>
          </g>
        ))}
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="axis" />
        {xTicks.map((x) => (
          <text key={x} x={X(x)} y={H - B + 14} className="tick" textAnchor="middle">
            {xf(x)}
          </text>
        ))}
        {p.series.map((s) => {
          const pts = p.xs.map((x, i) => [x, s.values[i]] as const).filter(([, v]) => v !== null && v > 0);
          const d = pts.map(([x, v], i) => `${i ? "L" : "M"} ${X(x).toFixed(1)} ${Y(v as number).toFixed(1)}`).join(" ");
          return <path key={s.name} d={d} className={`series ${s.dashed ? "dashed" : ""}`} style={{ stroke: `var(${s.color})` }} />;
        })}
        {hover !== null && (
          <g>
            <line x1={X(p.xs[hover])} x2={X(p.xs[hover])} y1={T} y2={H - B} className="crosshair" />
            {p.series.map((s) => {
              const v = s.values[hover];
              return v !== null && v > 0 ? <circle key={s.name} cx={X(p.xs[hover])} cy={Y(v)} r={4} className="marker" style={{ fill: `var(${s.color})` }} /> : null;
            })}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${(X(p.xs[hover]) / W) * 100}%` }}>
          <div className="tt-head">{xf(p.xs[hover])} tokens per row</div>
          {p.series.map((s) => (
            <div key={s.name} className="tt-row">
              <i style={{ background: `var(${s.color})` }} />
              <span>{s.name}</span>
              <strong>{s.values[hover] === null ? "—" : p.yFormat(s.values[hover] as number)}</strong>
            </div>
          ))}
        </div>
      )}
      {p.note && <p className="chart-note">{p.note}</p>}
    </figure>
  );
}

export interface Segment {
  name: string;
  value: number;
  color: string;
}

export function StackBar({ segments, capacity, format }: { segments: Segment[]; capacity: number; format: (v: number) => string }) {
  const total = segments.reduce((n, s) => n + s.value, 0);
  const scale = Math.max(total, capacity) * 1.02;
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div className="stackbar">
      <div className="bar">
        {segments.map((s, i) =>
          s.value > 0 ? (
            <div key={s.name} className="seg" style={{ width: `${(s.value / scale) * 100}%`, background: `var(${s.color})` }} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} />
          ) : null,
        )}
        <div className="cap" style={{ left: `${(capacity / scale) * 100}%` }} title={`GPU memory ${format(capacity)}`} />
      </div>
      <div className="legend wrap">
        {segments.map((s, i) => (
          <span key={s.name} className={hover === i ? "hl" : ""}>
            <i style={{ background: `var(${s.color})` }} />
            {s.name} <strong>{format(s.value)}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

/** Compact tick labels for powers of ten: 100M, 1G, 10G. */
export function compact(unit: string) {
  return (v: number) => {
    for (const [u, n] of [[1e15, "P"], [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]] as const) if (v >= u) return `${+(v / u).toPrecision(3)}${n}${unit}`;
    return `${+v.toPrecision(3)}${unit}`;
  };
}

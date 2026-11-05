// A small picture of how a cell connects tokens: rows = the token that reads, columns = what it reads.
import type { MixKind } from "./describe";

const N = 5;

function filled(kind: MixKind, i: number, j: number): boolean {
  switch (kind) {
    case "causal":
      return j <= i;
    case "local":
      return j <= i && i - j < 2;
    case "twoway":
    case "cross":
      return true;
    case "page":
      return (i < 3 && j < 3) || (i >= 2 && j >= 2);
    case "notes":
      return j <= i && i - j < 2;
    case "point":
    case "embed":
    case "head":
      return i === j;
    case "start":
      return j === 0;
    case "custom":
      return false;
  }
}

export function MixGlyph({ kind, title, size = 18 }: { kind: MixKind; title: string; size?: number }) {
  const cols = kind === "notes" ? N + 1 : N;
  const c = 3, gap = 0.6;
  const w = cols * (c + gap) + (kind === "notes" ? 1.2 : 0);
  const h = N * (c + gap);
  const cells = [];
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      cells.push(<rect key={`${i}-${j}`} x={j * (c + gap)} y={i * (c + gap)} width={c} height={c} rx={0.6} className={filled(kind, i, j) ? "on" : "off"} />);
  if (kind === "notes")
    for (let i = 0; i < N; i++)
      cells.push(<rect key={`n${i}`} x={N * (c + gap) + 1.2} y={i * (c + gap)} width={c} height={c} rx={0.6} className={i >= 2 ? "on notes" : "off"} />);
  return (
    <svg className={`glyph g-${kind}`} viewBox={`-0.5 -0.5 ${w + 1} ${h + 1}`} width={(size * (w + 1)) / (h + 1)} height={size} role="img" aria-label={title}>
      <title>{title}</title>
      {cells}
      {kind === "custom" && (
        <text x={w / 2} y={h / 2 + 2.5} textAnchor="middle" fontSize={8}>
          ?
        </text>
      )}
    </svg>
  );
}

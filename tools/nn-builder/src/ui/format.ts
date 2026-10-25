// Number formatting for the UI. "GB" means 2^30 bytes, as GPU memory is reported.

export const exact = (n: number) => Math.round(n).toLocaleString("en-US");

export function params(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return `${(n / 1e9).toFixed(a >= 1e10 ? 1 : 2)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e8 ? 1 : 2)}M`;
  if (a >= 1e4) return `${(n / 1e3).toFixed(a >= 1e5 ? 0 : 1)}k`;
  return exact(n);
}

export function flops(f: number): string {
  const units: [number, string][] = [[1e18, "EFLOP"], [1e15, "PFLOP"], [1e12, "TFLOP"], [1e9, "GFLOP"], [1e6, "MFLOP"], [1e3, "kFLOP"]];
  for (const [u, name] of units) if (f >= u) return `${(f / u).toFixed(f / u >= 100 ? 0 : 1)} ${name}`;
  return `${Math.round(f)} FLOP`;
}

export function bytes(b: number): string {
  if (b >= 2 ** 30) return `${(b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 1 : 2)} GB`;
  if (b >= 2 ** 20) return `${(b / 2 ** 20).toFixed(b >= 100 * 2 ** 20 ? 0 : 1)} MB`;
  if (b >= 2 ** 10) return `${(b / 2 ** 10).toFixed(0)} KB`;
  return `${Math.round(b)} B`;
}

export function hours(h: number): string {
  if (!Number.isFinite(h)) return "—";
  if (h < 1 / 60) return "< 1 min";
  if (h < 1) return `${Math.round(h * 60)} min`;
  if (h < 48) return `${h.toFixed(h < 10 ? 1 : 0)} h`;
  return `${(h / 24).toFixed(h < 240 ? 1 : 0)} days`;
}

const trimmed = (x: number, digits: number) => String(+x.toFixed(digits));

/** Token counts in billions (trillions from 1T): 0.6B, 1.5B, 56.1B, 444B, 2T. */
export function tokens(n: number): string {
  if (n >= 1e12) {
    const t = n / 1e12;
    return `${trimmed(t, t < 10 ? 2 : t < 100 ? 1 : 0)}T`;
  }
  const b = n / 1e9;
  return `${trimmed(b, b < 0.1 ? 3 : b < 10 ? 2 : b < 100 ? 1 : 0)}B`;
}

/** A typed token count: "1.5" or "1.5B" = 1.5 billion, "500M", "2T", "1,5" (decimal comma), or a raw count. */
export function parseTokens(raw: string): number | null {
  let s = raw.trim().toLowerCase().replace(/[_\s]/g, "");
  if (!s) return null;
  const unit = /[kmbt]$/.exec(s)?.[0];
  if (unit) s = s.slice(0, -1);
  const commas = (s.match(/,/g) ?? []).length;
  s = commas === 1 && !s.includes(".") && !/,\d{3}$/.test(s) ? s.replace(",", ".") : s.replace(/,/g, "");
  const v = Number(s);
  if (!s || !Number.isFinite(v) || v <= 0) return null;
  const mult = unit === "k" ? 1e3 : unit === "m" ? 1e6 : unit === "b" ? 1e9 : unit === "t" ? 1e12 : v >= 1e5 ? 1 : 1e9;
  return Math.round(v * mult);
}

/** Round token budgets for the slider: 10M … 20T in 1-1.2-1.5-2-2.5-3-4-5-6-8 steps. */
export const TOKEN_STEPS: number[] = [];
for (let e = 7; e <= 13; e++)
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]) {
    const v = Math.round(m * 10 ** e);
    if (v <= 2e13) TOKEN_STEPS.push(v);
  }

/** The slider step closest to a budget (on a log scale). */
export function nearestTokenStep(n: number): number {
  let best = 0;
  for (let i = 1; i < TOKEN_STEPS.length; i++)
    if (Math.abs(Math.log(TOKEN_STEPS[i] / n)) < Math.abs(Math.log(TOKEN_STEPS[best] / n))) best = i;
  return best;
}

export function rate(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}k`;
  return n.toFixed(0);
}

export function len(S: number): string {
  if (S >= 1 << 20 && S % (1 << 18) === 0) return `${S / (1 << 20)}M`;
  if (S >= 1024 && S % 512 === 0) return `${S / 1024}k`;
  return exact(S);
}

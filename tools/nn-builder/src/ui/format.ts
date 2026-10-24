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

export function tokens(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n % 1e9 ? 1 : 0)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return exact(n);
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

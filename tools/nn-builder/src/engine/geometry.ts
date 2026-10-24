// Sequence geometry: how many positions each stack has and how many keys each query sees.

export interface PageGeom {
  window: number; // effective page width (≤ the row)
  stride: number;
  starts: number[];
  count: number;
}

/** Pages of `window` tokens every `stride` tokens; the last page is pulled back to end at the row's end. */
export function pageGeometry(S: number, window: number, stride: number): PageGeom {
  const w = Math.min(Math.max(window, 1), S);
  const st = Math.max(1, Math.min(stride, w));
  if (S <= w) return { window: w, stride: st, starts: [0], count: 1 };
  const starts: number[] = [];
  for (let x = 0; x <= S - w; x += st) starts.push(x);
  if (starts[starts.length - 1] !== S - w) starts.push(S - w);
  return { window: w, stride: st, starts, count: starts.length };
}

/** Average keys a causal query sees in a row of S with a sliding window of w keys (0 = the whole row). */
export function avgCausalKeys(S: number, w: number): number {
  if (w <= 0 || w >= S) return (S + 1) / 2;
  return ((w * (w + 1)) / 2 + (S - w) * w) / S;
}

/** Average notes a token sees when a page's notes become readable only after the page's last token ("closed" read). */
export function avgVisibleNotes(S: number, pages: PageGeom, perPage: number): number {
  let sum = 0;
  for (const st of pages.starts) {
    const done = st + pages.window - 1;
    if (done <= S - 1) sum += S - done;
  }
  return (sum * perPage) / S;
}

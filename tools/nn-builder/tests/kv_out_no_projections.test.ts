import { describe, expect, test } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema/graph.schema.json";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";
import type { Graph, KvOutCell } from "../src/engine/types";

// kv_out with projections 0 (added 8 Nov 2026 for E37): the slot states already are the keys/values, so the notes
// cell has no weights and no projection FLOPs. Used for E37's concept summaries (mean key, mean value).
function withProjections(p: number, width?: number): Graph {
  const g = buildPreset("e33m_plain_30m");
  const notes = g.stacks.find((s) => s.id === "notes")!;
  if (width !== undefined) notes.width = width;
  const kv = notes.rows.at(-1)!.cells[0] as KvOutCell & { id: string };
  Object.assign(kv, { projections: p, in_norms: 0, key_norm: false });
  return g;
}

describe("kv_out with no projections", () => {
  const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

  test("the schema accepts projections 0", () => {
    expect(validate(withProjections(0, 64)), JSON.stringify(validate.errors)).toBe(true);
  });

  test("it has no parameters and no projection FLOPs", () => {
    const two = analyze(withProjections(2, 64)), zero = analyze(withProjections(0, 64));
    // two projections of a 64-wide state into one 64-wide key and value: 2 · 64 · 64
    expect(two.params.total - zero.params.total).toBe(2 * 64 * 64);
    expect(zero.cells.get("notes.out.kv")!.params).toBe(0);
    expect(zero.flops.fwdPerToken).toBeLessThan(two.flops.fwdPerToken);
  });

  test("the state width must equal the notes' key/value width", () => {
    const wrong = analyze(withProjections(0));
    expect(wrong.issues.some((i) => i.where === "notes.out.kv" && /must equal/.test(i.msg))).toBe(true);
    expect(analyze(withProjections(0, 64)).issues.filter((i) => i.where === "notes.out.kv")).toEqual([]);
  });
});

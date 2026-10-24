import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { defaultBody } from "../src/engine/defaults";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";

const ids = (g: ReturnType<typeof buildPreset>) => [
  ...g.stacks.flatMap((s) => s.rows.map((r) => r.id)),
  ...g.stacks.flatMap((s) => s.rows.flatMap((r) => r.cells.map((c) => c.id))),
];

describe("slots and rows", () => {
  test("dropping a feed-forward into a new row adds its parameters", () => {
    const g = buildPreset("blank");
    const before = analyze(g).params.total;
    const next = E.insertRow(g, "main", 2, [defaultBody("ffn", 512)]);
    expect(next.stacks[0].rows).toHaveLength(4);
    expect(analyze(next).params.total - before).toBe(512 + 3 * 512 * 1344);
  });

  test("edits never change the graph they were given", () => {
    const g = buildPreset("blank");
    const copy = JSON.stringify(g);
    E.insertRow(g, "main", 1, [defaultBody("ffn", 512)]);
    E.removeRow(g, "main.L0");
    expect(JSON.stringify(g)).toBe(copy);
  });

  test("moving a cell to another row keeps the total", () => {
    const g = buildPreset("blank");
    const next = E.moveCell(g, "main.L0.ffn", "main.input", 2);
    expect(next.stacks[0].rows[0].cells.map((c) => c.id)).toEqual(["main.input.emb", "main.input.pos", "main.L0.ffn"]);
    expect(analyze(next).params.total).toBe(analyze(g).params.total);
  });

  test("duplicating a row ×3 gives three new rows with fresh ids", () => {
    const g = buildPreset("dense_plain_30m");
    const next = E.duplicateRow(g, "main.L3", 3);
    expect(next.stacks[0].rows).toHaveLength(g.stacks[0].rows.length + 3);
    expect(new Set(ids(next)).size).toBe(ids(next).length);
  });

  test("removing a row drops its wires and empties its loop", () => {
    const g = buildPreset("e33m_plain_30m");
    const next = E.removeRow(E.removeRow(g, "main.L2"), "main.L3");
    expect(next.edges.some((e) => e.to.cell === "main.L2.attn")).toBe(false);
    expect(next.groups.some((x) => x.id === "loop.read")).toBe(false);
    expect(analyze(next).issues.filter((i) => /unknown|missing/.test(i.msg))).toEqual([]);
  });
});

describe("wires by drag and drop", () => {
  test("row → later row of the same stack makes a skip wire with one weight", () => {
    const g = buildPreset("blank");
    const two = E.insertRow(g, "main", 2, [defaultBody("ffn", 512)]);
    const later = two.stacks[0].rows[2].id;
    const { graph, error } = E.connect(two, { stack: "main", row: "main.L0" }, { stack: "main", row: later });
    expect(error).toBeUndefined();
    expect(graph.edges[0]).toMatchObject({ kind: "skip", weight: "scalar" });
    expect(analyze(graph).params.total - analyze(two).params.total).toBe(1);
  });

  test("a stack → a cross-attention cell feeds its keys and values, replacing an older wire", () => {
    const g = buildPreset("encdec");
    const { graph } = E.connect(g, { stack: "dec", row: "dec.L1" }, { stack: "dec", cell: "dec.L0.cross" });
    const wires = graph.edges.filter((e) => e.to.cell === "dec.L0.cross");
    expect(wires).toHaveLength(1);
    expect(wires[0].from.row).toBe("dec.L1");
  });

  test("a wire into another stack's header starts that stack's stream", () => {
    const g = buildPreset("e33m_plain_30m");
    const cut = { ...g, edges: g.edges.filter((e) => e.kind !== "input") };
    const { graph } = E.connect(cut, { stack: "main", cell: "main.input.emb", port: "table" }, { stack: "pages" });
    expect(analyze(graph).params.total).toBe(29_946_402);
  });

  test("a wire backwards inside one stack is refused", () => {
    const g = buildPreset("dense_plain_30m");
    expect(E.connect(g, { stack: "main", row: "main.L5" }, { stack: "main", row: "main.L2" }).error).toMatch(/later row/);
  });
});

describe("loops and width", () => {
  test("looping two rows 3 times keeps the parameters and triples their compute", () => {
    const g = buildPreset("dense_plain_30m");
    const looped = E.addGroup(g, "loop", "main", ["main.L2", "main.L3"], 3);
    const a = analyze(g), b = analyze(looped);
    expect(b.params.total).toBe(a.params.total + 3 * 576); // only the per-round embedding
    const rowF = (x: typeof a) => x.stacks[0].rows.filter((r) => ["main.L2", "main.L3"].includes(r.id)).reduce((n, r) => n + r.fwdPerToken, 0);
    expect(rowF(b)).toBeCloseTo(3 * rowF(a), 0);
  });

  test("rows that are not neighbours cannot be grouped", () => {
    const g = buildPreset("dense_plain_30m");
    expect(() => E.addGroup(g, "loop", "main", ["main.L1", "main.L4"], 2)).toThrow(/neighbouring/);
  });

  test("widening a stack keeps full-width attention full-width", () => {
    const g = E.setStackWidth(buildPreset("dense_plain_30m"), "main", 768);
    const attn = g.stacks[0].rows[1].cells[0];
    expect(attn.type === "attention" && attn.heads * attn.head_dim).toBe(768);
  });
});

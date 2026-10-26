// The spec an agent reads to implement a design, its content hash, the diff between versions, and validation.
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, test } from "vitest";
import specSchema from "../schema/spec.schema.json";
import { analyze } from "../src/engine/analyze";
import * as E from "../src/engine/edit";
import { PRESETS, buildPreset } from "../src/engine/presets";
import { buildSpec, designHash, diffDesigns, validateDesign } from "../src/engine/spec";
import type { Graph } from "../src/engine/types";

const blocks = (s: ReturnType<typeof buildSpec>) => s.stacks.flatMap((x) => x.blocks);
const block = (s: ReturnType<typeof buildSpec>, id: string) => blocks(s).find((b) => b.id === id)!;

describe("the spec an agent reads", () => {
  const dense = buildSpec(buildPreset("dense_plain_30m"));
  const e33 = buildSpec(buildPreset("e33m_plain_30m"));

  test("names its format, the design and how to read it", () => {
    expect(dense.format).toBe("nn-builder/spec");
    expect(dense.version).toBe(1);
    expect(dense.design.name).toBe("dense_plain 30M");
    expect(dense.how_to_read.length).toBeGreaterThan(4);
  });

  test("lists every block with settings, input and output shapes, and parameters", () => {
    expect(blocks(dense)).toHaveLength(1 + 1 + 8 * 2 + 1);
    const attn = block(dense, "main.L0.attn");
    expect(attn.settings).toMatchObject({ heads: 9, head_dim: 64, kv_heads: 1, causal: true, window: 0 });
    expect(attn.input.shape).toEqual(["B", "S", 576]);
    expect(attn.output.shape).toEqual(["B", "S", 576]);
    expect(attn.residual).toBe("pre");
    expect(attn.parameters.count).toBe(analyze(buildPreset("dense_plain_30m")).cells.get("main.L0.attn")!.params);
    expect(attn.what_it_does).toMatch(/earlier token/);
    expect(attn.tensor_steps.find((t) => /^scores/.test(t.step))?.shape).toEqual(["B", 9, "S", "S"]);
  });

  test("parameter totals match the analysis", () => {
    expect(dense.parameters.total).toBe(30_293_140);
    expect(e33.parameters.total).toBe(29_946_402);
    expect(Object.values(e33.parameters.by_stack).reduce((a, b) => a + b, 0)).toBe(29_946_402);
  });

  test("names the dims with their sizes", () => {
    const dim = (n: string) => e33.dims.find((d) => d.name === n);
    expect(dim("B")?.value).toBeNull();
    expect(dim("S")?.value).toBe(4096);
    expect(dim("pages")?.value).toBe(21);
    expect(dim("W")?.value).toBe(256);
    expect(dim("slots")?.value).toBe(32);
  });

  test("execution order runs the page writer and the notes before the read that needs them", () => {
    const at = (row: string) => e33.execution_order.findIndex((s) => s.rows.includes(row));
    expect(at("main.input")).toBeLessThan(at("pages.in"));
    expect(at("pages.enc1")).toBeLessThan(at("notes.init"));
    expect(at("notes.out")).toBeLessThan(at("main.L2"));
    expect(at("main.L1")).toBeLessThan(at("main.L2"));
    expect(new Set(e33.execution_order.flatMap((s) => s.rows)).size).toBe(buildPreset("e33m_plain_30m").stacks.flatMap((s) => s.rows).length);
  });

  test("loops and repeats are explicit steps", () => {
    const loop = e33.execution_order.find((s) => s.kind === "loop" && s.stack === "main")!;
    expect(loop).toMatchObject({ rows: ["main.L2", "main.L3"], count: 4, shared_weights: true, round_embedding: true, feedback: "soft" });
    const smol = buildSpec(buildPreset("smollm3_3b"));
    expect(smol.execution_order.find((s) => s.kind === "repeat")).toMatchObject({ count: 9, shared_weights: false });
  });

  test("rows say what is added before they run (x0 mix and skips)", () => {
    const row = dense.stacks[0].rows.find((r) => r.id === "main.L4")!;
    expect(row.before).toMatchObject({ x0_mix: true, skips_in: [{ from_row: "main.L3", weight: "scalar" }] });
  });

  test("wires say what they carry", () => {
    const notes = e33.wires.find((w) => w.id === "wire.notes")!;
    expect(notes.kind).toBe("source");
    expect(notes.carries).toEqual(["B", "notes", 1, 64]);
    expect(notes.meaning).toMatch(/notes/);
  });

  test("the model's inputs and outputs are listed", () => {
    expect(dense.inputs).toEqual([expect.objectContaining({ name: "main.input_ids", shape: ["B", "S"] })]);
    expect(dense.outputs).toEqual([expect.objectContaining({ name: "main.logits", shape: ["B", "S", 4096] })]);
  });

  test("the same design always gives the same spec", () => {
    expect(JSON.stringify(buildSpec(buildPreset("e33m_plain_30m")))).toBe(JSON.stringify(e33));
  });

  test("every preset's spec is valid against the published schema", () => {
    const validate = new Ajv2020({ allErrors: true, strict: false }).compile(specSchema);
    for (const p of PRESETS) {
      const ok = validate(buildSpec(p.build()));
      expect(validate.errors ?? [], p.id).toEqual([]);
      expect(ok).toBe(true);
    }
  });
});

describe("the design hash", () => {
  const g = buildPreset("dense_plain_30m");
  const h = designHash(g);

  test("ignores layout, the design's name and notes, and the training setup", () => {
    const g2: Graph = { ...structuredClone(g), name: "renamed", description: "other", ui: { stackWidths: { main: 900 } } };
    g2.run.gpu = "h100_80";
    g2.run.token_budget = 9e9;
    expect(designHash(g2)).toBe(h);
  });

  test("changes when the architecture changes", () => {
    expect(designHash(E.updateCell(g, "main.L0.ffn", { hidden: 1600 } as never))).not.toBe(h);
    expect(designHash(E.removeEdge(g, "main.skip0"))).not.toBe(h);
  });
});

describe("what changed between two versions", () => {
  const before = buildPreset("dense_plain_30m");

  test("nothing changed", () => {
    expect(diffDesigns(before, structuredClone(before)).changes).toEqual([]);
  });

  test("a changed setting names the block, the field and both values", () => {
    const d = diffDesigns(before, E.updateCell(before, "main.L0.ffn", { hidden: 1600 } as never));
    expect(d.changes).toEqual([expect.objectContaining({ change: "changed", what: "block", id: "main.L0.ffn", field: "hidden", from: 1472, to: 1600 })]);
    expect(d.parameters.to - d.parameters.from).toBe(3 * 576 * 128);
  });

  test("an added row and a removed wire", () => {
    const after = E.removeEdge(E.insertRow(before, "main", 3, [{ type: "norm", kind: "rmsnorm" }]), "main.skip0");
    const d = diffDesigns(before, after);
    expect(d.changes.some((c) => c.change === "added" && c.what === "row")).toBe(true);
    expect(d.changes.some((c) => c.change === "added" && c.what === "block")).toBe(true);
    expect(d.changes.some((c) => c.change === "removed" && c.what === "wire" && c.id === "main.skip0")).toBe(true);
    expect(d.from_hash).toBe(designHash(before));
    expect(d.to_hash).toBe(designHash(after));
  });
});

describe("validation an agent can trust", () => {
  test("no preset has an error", () => {
    for (const p of PRESETS) expect(validateDesign(p.build()).errors, p.id).toEqual([]);
  });

  test("a cross-attention without a source is an error", () => {
    const g = E.insertCell(buildPreset("blank"), "main.L0", 1, { ...(buildPreset("encdec").stacks[1].rows[1].cells[1] as object) } as never);
    expect(validateDesign(g).errors.some((e) => /no source/.test(e.message))).toBe(true);
  });

  test("a stack that nothing starts is an error", () => {
    const g = E.addStack(buildPreset("blank"), "pages");
    expect(validateDesign(g).errors.some((e) => /nothing starts/.test(e.message))).toBe(true);
  });

  test("a stack whose output goes nowhere is a warning", () => {
    const g = E.addStack(buildPreset("blank"), "encoder");
    const v = validateDesign(g);
    expect(v.errors).toEqual([]);
    expect(v.warnings.some((w) => /not used/.test(w.message))).toBe(true);
  });
});

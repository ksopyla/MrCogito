import { describe, expect, test } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema/graph.schema.json";
import { analyze } from "../src/engine/analyze";
import { PRESETS, buildPreset } from "../src/engine/presets";
import { parseGraph, toExport } from "../src/engine/serialize";

const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

describe("every preset is a valid graph file", () => {
  for (const p of PRESETS) {
    test(p.id, () => {
      const ok = validate(toExport(p.build()));
      expect(validate.errors ?? []).toEqual([]);
      expect(ok).toBe(true);
    });
  }
});

describe("export and import", () => {
  test("an exported graph loads back with the same parameter count", () => {
    const g = buildPreset("e33m_plain_30m");
    const back = parseGraph(JSON.stringify(toExport(g)));
    expect(back.error).toBeUndefined();
    expect(analyze(back.graph!).params.total).toBe(29_946_402);
  });

  test("the export carries each cell's parameter count and widths for code generation", () => {
    const out = toExport(buildPreset("dense_plain_30m"));
    expect(out.derived.params_total).toBe(30_293_140);
    expect(out.derived.cells["main.L0.attn"]).toMatchObject({ width_in: 576, width_out: 576 });
  });

  test("a file that is not a graph is refused with a reason", () => {
    expect(parseGraph('{"hello": 1}').error).toMatch(/not an nn-builder graph/);
    expect(parseGraph("not json").error).toMatch(/JSON/);
  });

  test("the schema refuses a cell with a misspelled setting", () => {
    const g = toExport(buildPreset("blank")) as unknown as { stacks: { rows: { cells: Record<string, unknown>[] }[] }[] };
    g.stacks[0].rows[1].cells[0].haeds = 8;
    expect(validate(g)).toBe(false);
  });
});

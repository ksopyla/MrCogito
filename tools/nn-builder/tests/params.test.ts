import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";

// Truth source: evaluation/text_checks.count_params in MrCogito (the trainer's own factory on the meta device),
// run on 24 Oct 2026 with `uv run python`. These must stay exact.
const TRAINER_COUNTS: Record<string, number> = {
  dense_plain_30m: 30_293_140,
  dense_plain_50m: 50_057_374, // the calibration ladder's 50M step (fit_params("dense_plain", "mid"))
  dense_plain_100m: 100_966_686,
  e33m_plain_30m: 29_946_402,
  e33m_plain_100m: 100_965_425,
};

describe("parameter counts are exact for the trainer's reference models", () => {
  for (const [id, expected] of Object.entries(TRAINER_COUNTS)) {
    test(id, () => {
      expect(analyze(buildPreset(id)).params.total).toBe(expected);
    });
  }
});

describe("parameter counts are exact for an outside reference", () => {
  test("GPT-2 small (Hugging Face count, tied head)", () => {
    expect(analyze(buildPreset("gpt2_small")).params.total).toBe(124_439_808);
  });
});

describe("parameter breakdown of E33 at 30M matches the trainer's modules", () => {
  const a = analyze(buildPreset("e33m_plain_30m"));
  const byStack = (name: string) => a.stacks.find((s) => s.name === name)!.params;

  test("the notebook writer (pages + notes stacks) is 3,985,856", () => {
    expect(byStack("Pages") + byStack("Notes")).toBe(3_985_856);
  });

  test("the input layer is 1,675,840 and the LM head 2,359,872", () => {
    const main = a.stacks.find((s) => s.name === "Decoder")!;
    const cellParams = (type: string) =>
      main.rows.flatMap((r) => r.cells).filter((c) => c.type === type).reduce((n, c) => n + c.params, 0);
    expect(cellParams("embedding")).toBe(1_675_840);
    expect(cellParams("lm_head")).toBe(2_359_872);
  });
});

describe("loops share weights, repeats copy them", () => {
  test("more loop rounds add only the per-round embedding", () => {
    const g = buildPreset("e33m_plain_30m");
    const base = analyze(g).params.total;
    const loop = g.groups.find((x) => x.kind === "loop" && x.stack === "main")!;
    loop.count = 6;
    expect(analyze(g).params.total - base).toBe(2 * 576);
  });

  test("a repeat ×3 of one dense layer triples that layer", () => {
    const g = buildPreset("dense_plain_30m");
    const before = analyze(g);
    const row = g.stacks[0].rows[3];
    const rowParams = before.stacks[0].rows.find((r) => r.id === row.id)!.params;
    g.groups.push({ id: "rep", kind: "repeat", stack: g.stacks[0].id, rows: [row.id], count: 3, round_embedding: false, feedback: "none" });
    expect(analyze(g).params.total - before.params.total).toBe(2 * rowParams);
  });
});

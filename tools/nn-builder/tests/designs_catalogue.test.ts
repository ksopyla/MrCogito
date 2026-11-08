import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema/graph.schema.json";
import { analyze } from "../src/engine/analyze";
import { parseGraph } from "../src/engine/serialize";
import { designHash, validateDesign } from "../src/engine/spec";

// Every saved design against the trainer (the architecture tracker's pins; designs/CATALOGUE.md).
// Truth sources, run on 8 Nov 2026 with `uv run python` on each branch's own code:
// - testbed arms: evaluation/text_checks.count_params on scripts/run_e33n.py model_args(arm, size, ffn), with the FFN
//   from fit() (1472 at s30, 2624 at m100); equal to the runner's own count() and to the trained checkpoints' stored
//   weights where they exist (LSA port seed 42, notes_open and catalogue seed 1; checked 7 Nov);
// - the champion: evaluation/bapo_models.build_model as verification/bapo_capability_probe.py calls it with the E31
//   platform args (13-symbol DNA alphabet) = 35,129,900, the count the capability ledger stores for its runs;
// - lsa_paper_native_0p8b: Qwen/Qwen3.5-0.8B config.json (text model); e33m_plain_30m: tests/params.test.ts.
// `core` = total − the input embedding (table + input MLP) − the output matrix (Kaplan et al.'s non-embedding count).
const EXPECTED: Record<string, { total: number; core: number }> = {
  // 30M testbed (s30)
  e33p_window_s30: { total: 72_687_252, core: 30_246_164 },
  e33p_dense_s30: { total: 68_701_396, core: 26_260_308 },
  e33n_full_s30: { total: 72_687_252, core: 30_246_164 },
  e33n_s30: { total: 72_687_252, core: 30_246_164 },
  e33p_notes_open_30m: { total: 72_687_252, core: 30_246_164 },
  e33p_pages_open_s30: { total: 72_687_252, core: 30_246_164 },
  e33p_lsa_port_30m: { total: 68_701_396, core: 26_260_308 },
  e33p_lsa_s30: { total: 68_701_396, core: 26_260_308 },
  e33t_catalogue_30m: { total: 72_687_765, core: 30_246_677 },
  e33t_catalogue_2l_s30: { total: 72_687_765, core: 30_246_677 },
  e33t_catalogue_reuse_s30: { total: 72_687_765, core: 30_246_677 },
  e37a_s30: { total: 68_701_396, core: 26_260_308 },
  e37a_c32_s30: { total: 68_701_396, core: 26_260_308 },
  e37a_c128_s30: { total: 68_701_396, core: 26_260_308 },
  e37b_s30: { total: 75_323_092, core: 32_882_004 },
  // 100M testbed (m100)
  e33n_m100: { total: 166_375_003, core: 100_554_203 },
  e37_window_m100: { total: 166_375_003, core: 100_554_203 },
  e37_dense_m100: { total: 162_356_379, core: 96_535_579 },
  e37_lsa_m100: { total: 162_356_379, core: 96_535_579 },
  e37_notes_open_m100: { total: 166_375_003, core: 100_554_203 },
  e37a_m100: { total: 162_356_379, core: 96_535_579 },
  e37b_m100: { total: 169_141_915, core: 103_321_115 },
  // the champion and the references
  e31_li_m1_cap30m: { total: 35_129_900, core: 33_947_436 },
  lsa_paper_native_0p8b: { total: 752_393_024, core: 498_113_344 },
  e33m_plain_30m: { total: 29_946_402, core: 25_911_266 },
};

const DIR = path.join(import.meta.dirname, "..", "designs");
const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);
const files = readdirSync(DIR).filter((f) => f.endsWith(".json")).map((f) => f.replace(/\.json$/, "")).sort();
const catalogue = readFileSync(path.join(DIR, "CATALOGUE.md"), "utf8");

describe("every saved design is pinned and catalogued", () => {
  test("no design without a pin, no pin without a design", () => {
    expect(files).toEqual(Object.keys(EXPECTED).sort());
  });

  for (const name of files) {
    test(name, () => {
      const want = EXPECTED[name];
      expect(want, `${name}.json has no pin in tests/designs_catalogue.test.ts`).toBeDefined();
      const text = readFileSync(path.join(DIR, `${name}.json`), "utf8");
      const raw = JSON.parse(text);
      expect(validate(raw), JSON.stringify(validate.errors)).toBe(true);
      const parsed = parseGraph(text);
      expect(parsed.error).toBeUndefined();
      const g = parsed.graph!;
      expect(validateDesign(g).errors).toEqual([]);

      const a = analyze(g);
      expect(a.params.total).toBe(want.total);
      const cells = a.stacks.find((s) => s.id === "main")!.rows.flatMap((r) => r.cells);
      const cell = (type: string) => cells.find((c) => c.type === type)!.params;
      const width = g.stacks.find((s) => s.id === "main")!.width;
      expect(a.params.total - cell("embedding") - (cell("lm_head") - width)).toBe(want.core);

      // the name states the true total and core, never a size label alone
      expect(raw.name).toContain(`${(want.total / 1e6).toFixed(1)}M`);
      expect(raw.name).toContain(`${(want.core / 1e6).toFixed(1)}M core`);
      // the exported block is current (scripts/designs.mjs rewrites it)
      expect(raw.derived?.params_total).toBe(want.total);
      expect(raw.derived?.hash).toBe(designHash(g));
      // and the catalogue has its row
      expect(catalogue).toContain(`| \`${name}\` |`);
    });
  }
});

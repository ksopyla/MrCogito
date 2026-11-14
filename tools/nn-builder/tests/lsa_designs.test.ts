import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema/graph.schema.json";
import { analyze } from "../src/engine/analyze";
import { parseGraph } from "../src/engine/serialize";
import { validateDesign } from "../src/engine/spec";

// The "which LSA did we lose to" designs against the trainer and the trained weights. Truth source for the testbed
// arms: scripts/run_e33n.py (model_args(arm, "s30", 1472) through the trainer's factory), equal to the parameters
// stored in the trained checkpoints (LSA port seed 42, notes_open and catalogue seed 1), checked 7 Nov 2026.
// The paper's native model: Qwen/Qwen3.5-0.8B config.json, text model only; the Hub's 873.4M adds the vision tower
// (100,592,896) and the multi-token-prediction layer (20,452,864).
const EXPECTED: Record<string, { total: number; nonEmbedding: number }> = {
  e33p_lsa_port_30m: { total: 68_701_396, nonEmbedding: 26_260_308 },
  e33p_notes_open_30m: { total: 72_687_252, nonEmbedding: 30_246_164 },
  e33t_catalogue_30m: { total: 72_687_765, nonEmbedding: 30_246_677 },
  // the E33t winner (64 notes per page, every pass may reopen notes): model_args("catalogue_reuse_n4", "s30", 1472)
  e33t_catalogue_reuse_n4_30m: { total: 72_769_685, nonEmbedding: 30_328_597 },
  lsa_paper_native_0p8b: { total: 752_393_024, nonEmbedding: 498_113_344 },
};

const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

describe("the LSA comparison designs match the trainer and the trained weights", () => {
  for (const [name, want] of Object.entries(EXPECTED)) {
    test(name, () => {
      const text = readFileSync(path.join(import.meta.dirname, "..", "designs", `${name}.json`), "utf8");
      const raw = JSON.parse(text);
      expect(validate(raw), JSON.stringify(validate.errors)).toBe(true);
      const parsed = parseGraph(text);
      expect(parsed.error).toBeUndefined();
      const g = parsed.graph!;
      expect(validateDesign(g).errors).toEqual([]);
      const a = analyze(g);
      expect(a.params.total).toBe(want.total);
      // non-embedding = total − the input table − the output weight (an untied head's matrix; a tied head only its
      // final norm, which the trainer keeps), as in e33n_designs.test.ts
      const cells = a.stacks.find((s) => s.id === "main")!.rows.flatMap((r) => r.cells);
      const cell = (type: string) => cells.find((c) => c.type === type)!.params;
      const width = g.stacks.find((s) => s.id === "main")!.width;
      expect(a.params.total - cell("embedding") - (cell("lm_head") - width)).toBe(want.nonEmbedding);
      // the tab name states the real total, never the testbed's "30M" alone
      const millions = (want.total / 1e6).toFixed(1);
      expect(raw.name.includes(`${millions}M`) || raw.name.includes(`${Math.round(want.total / 1e6)}M`)).toBe(true);
    });
  }
});

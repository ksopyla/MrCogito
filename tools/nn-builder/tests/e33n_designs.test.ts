import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, test } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import schema from "../schema/graph.schema.json";
import { analyze } from "../src/engine/analyze";
import { parseGraph } from "../src/engine/serialize";
import { designHash, validateDesign } from "../src/engine/spec";

// The saved E33n designs (designs/e33n_*.json) against the trainer. Truth source: evaluation/text_checks.count_params
// on scripts/run_e33n.py model_args("e33n", size, ffn) with the GPT-2 vocabulary (50,257), run on 5 Nov 2026.
// The top-k note read (lm_read_topk 64) has no builder field and no parameters; it is recorded in the description.
const TRAINER: Record<string, { total: number; nonEmbedding: number }> = {
  e33n_s30: { total: 72_687_252, nonEmbedding: 30_246_164 },
  e33n_m100: { total: 166_375_003, nonEmbedding: 100_554_203 },
};

const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);
const load = (name: string) => {
  const text = readFileSync(path.join(import.meta.dirname, "..", "designs", `${name}.json`), "utf8");
  return { raw: JSON.parse(text), parsed: parseGraph(text) };
};

describe("the E33n designs match the trainer exactly", () => {
  for (const [name, want] of Object.entries(TRAINER)) {
    test(name, () => {
      const { raw, parsed } = load(name);
      expect(validate(raw), JSON.stringify(validate.errors)).toBe(true);
      expect(parsed.error).toBeUndefined();
      const g = parsed.graph!;
      expect(validateDesign(g).errors).toEqual([]);
      const a = analyze(g);
      expect(a.params.total).toBe(want.total);
      // the trainer's non-embedding count leaves out the input layer and the output weight, but keeps the final norm
      const main = a.stacks.find((s) => s.id === "main")!;
      const cell = (type: string) => main.rows.flatMap((r) => r.cells).find((c) => c.type === type)!.params;
      const width = g.stacks.find((s) => s.id === "main")!.width;
      expect(a.params.total - cell("embedding") - (cell("lm_head") - width)).toBe(want.nonEmbedding);
      expect(raw.derived.params_total).toBe(want.total);
      expect(raw.derived.hash).toBe(designHash(g));
    });
  }
});

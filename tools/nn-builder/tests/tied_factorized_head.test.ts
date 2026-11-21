// The tied factorised head (lm_head `tied` + `factorized`): the output reuses the input token table through one
// projection, logits = (P·final_norm(x))·Eᵀ, P [table width × stream width] the only new weights (MrCogito's
// nn/perceiver_ar_lm.TiedFactorizedHead, the E38 leaderboard models). The table is counted once, with the embedding.
import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";
import { cellTensorSteps } from "../src/engine/shapes";
import { validateDesign } from "../src/engine/spec";

// e33m_plain_30m: width 576, a 4,096 × 256 table, an untied 4,096-token head with an RMSNorm
const HEAD = "main.head.lm";
const d = 576, e = 256, V = 4096;
const factorised = () => E.updateCell(buildPreset("e33m_plain_30m"), HEAD, { tied: true, factorized: true } as never);
const headMsgs = (g: ReturnType<typeof factorised>) => analyze(g).issues.filter((i) => i.where === HEAD && i.level !== "warning").map((i) => i.msg);

describe("a tied head through a projection to the table's width", () => {
  test("holds the final norm and the projection only; the table stays with the embedding", () => {
    const before = analyze(buildPreset("e33m_plain_30m"));
    const g = factorised();
    const a = analyze(g);
    const head = a.cells.get(HEAD)!;
    expect(head.params).toBe(d + d * e);
    expect(head.parts.map((p) => p.name)).toEqual(["final norm", "projection to the table (width → table width)"]);
    expect(a.params.total).toBe(before.params.total - V * d + d * e);
    expect(a.cells.get("main.input.emb")!.params).toBe(before.cells.get("main.input.emb")!.params);
    expect(headMsgs(g)).toEqual([]);
    expect(validateDesign(g).errors).toEqual([]);
  });

  test("its forward FLOPs are the projection plus logits against the e-wide table", () => {
    const g = factorised();
    const a = analyze(g);
    const plain = analyze(buildPreset("e33m_plain_30m"));
    const perToken = (x: typeof a) => x.cells.get(HEAD)!.fwdPerToken;
    expect(perToken(plain)).toBe(2 * d * V);
    expect(perToken(a)).toBe(2 * d * e + 2 * e * V);
  });

  test("the shapes show the projection to the table's width before the logits", () => {
    const g = factorised();
    const steps = cellTensorSteps(g, analyze(g), HEAD, 2).map((s) => [s.label, s.dims.map((x) => `${x.n}=${x.v}`).join(",")]);
    const i = steps.findIndex(([l]) => l === "projection to the token table's width");
    expect(i).toBeGreaterThan(0);
    expect(steps[i][1]).toMatch(/e=256$/);
    expect(steps[i + 1][0]).toBe("logits");
  });

  test("a plain tied head still needs a vocab × width table", () => {
    const g = E.updateCell(buildPreset("e33m_plain_30m"), HEAD, { tied: true } as never);
    expect(headMsgs(g).join(" ")).toMatch(/vocab × width/);
  });

  test("factorized without tied, or with another vocabulary, is an error", () => {
    const untied = E.updateCell(buildPreset("e33m_plain_30m"), HEAD, { factorized: true } as never);
    expect(headMsgs(untied).join(" ")).toMatch(/turn on tied/);
    expect(analyze(untied).cells.get(HEAD)!.params).toBe(d + V * d); // still counted as an untied head
    const other = E.updateCell(factorised(), HEAD, { vocab: 5000 } as never);
    expect(headMsgs(other).join(" ")).toMatch(/embedding's vocabulary/);
  });
});

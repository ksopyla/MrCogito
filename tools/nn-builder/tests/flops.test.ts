import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";
import { pageGeometry } from "../src/engine/geometry";

// Training FLOPs per token from the hand-checked single-page builder (docs/3_Evaluations_and_Baselines/
// e33_dense_builder.html, 24 Oct): multiply-adds of projections, scores and value reads, training = 3 × forward.
// The general engine counts a few extra small terms (the writer's address map, the count heads' projections),
// so it must agree to within 1 %.
const OLD_BUILDER: Record<string, Record<number, number>> = {
  dense_plain_30m: { 4096: 288_672_768, 16384: 628_411_392 },
  dense_plain_100m: { 4096: 863_665_152, 16384: 1_656_388_608 },
  e33m_plain_30m: { 4096: 347_567_547, 16384: 406_256_047 },
  e33m_plain_100m: { 4096: 986_934_270, 16384: 1_074_592_384 },
};

describe("training FLOPs per token agree with the single-page builder", () => {
  for (const [id, byLen] of Object.entries(OLD_BUILDER)) {
    for (const [S, expected] of Object.entries(byLen)) {
      test(`${id} at ${S} tokens`, () => {
        const g = buildPreset(id);
        g.run.row_length = Number(S);
        const got = analyze(g).flops.trainPerToken;
        expect(Math.abs(got - expected) / expected).toBeLessThan(0.01);
      });
    }
  }
});

describe("forward and backward split", () => {
  test("a plain dense model trains at 3 × its forward pass", () => {
    const f = analyze(buildPreset("dense_plain_30m")).flops;
    expect(f.bwdPerToken / f.fwdPerToken).toBeCloseTo(2, 6);
  });

  test("gradient checkpointing adds one forward pass", () => {
    const g = buildPreset("dense_plain_30m");
    const off = analyze(g).flops;
    g.run.checkpointing = true;
    const on = analyze(g).flops;
    expect(on.trainPerToken - off.trainPerToken).toBeGreaterThan(0.9 * off.fwdPerToken);
    expect(on.trainPerToken - off.trainPerToken).toBeLessThanOrEqual(off.fwdPerToken + 1);
  });
});

describe("page geometry", () => {
  test("4,096 tokens in pages of 256 every 192 make 21 pages", () => {
    const p = pageGeometry(4096, 256, 192);
    expect(p.count).toBe(21);
    expect(p.starts.at(-1)).toBe(4096 - 256);
  });

  test("a row shorter than a page is one page", () => {
    expect(pageGeometry(100, 256, 192).count).toBe(1);
  });
});

import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";
import { TOKEN_RULES, ruleTokens } from "../src/engine/tokens";

const N = 29_946_402; // E33 30M

describe("training-token presets", () => {
  test("Chinchilla-optimal is 20 tokens per parameter", () => {
    expect(ruleTokens("chinchilla", N)).toBe(20 * N);
  });

  test("the over-trained small-model ratios come from their own token counts and sizes", () => {
    expect(ruleTokens("llama3_8b", N)).toBeCloseTo((15e12 / 8e9) * N, -3);
    expect(ruleTokens("tinyllama", N)).toBeCloseTo((3e12 / 1.1e9) * N, -3);
    expect(ruleTokens("smollm2_135m", N)).toBeCloseTo((2e12 / 135e6) * N, -3);
  });

  test("Pythia is a fixed 300B tokens at every size", () => {
    expect(ruleTokens("pythia", N)).toBe(300e9);
    expect(ruleTokens("pythia", 10 * N)).toBe(300e9);
  });

  test("every preset has a label and a source", () => {
    for (const r of TOKEN_RULES) expect(r.label.length > 0 && r.source.length > 0).toBe(true);
  });
});

describe("the estimates use the chosen budget", () => {
  test("a custom budget is used as typed", () => {
    const g = buildPreset("e33m_plain_30m");
    Object.assign(g.run, { token_rule: "custom", token_budget: 2.5e9 });
    const a = analyze(g);
    expect(a.tokens.budget).toBe(2.5e9);
    expect(a.flops.perRun).toBeCloseTo(a.flops.trainPerToken * 2.5e9, -6);
  });

  test("Chinchilla follows the model: widening it raises the budget", () => {
    const g = buildPreset("e33m_plain_30m");
    g.run.token_rule = "chinchilla";
    const a = analyze(g);
    expect(a.tokens.budget).toBe(20 * N);
    expect(a.tokens.perParam).toBeCloseTo(20, 9);
    expect(a.tokens.chinchillaMultiple).toBeCloseTo(1, 9);
    expect(a.time.hours).toBeCloseTo(a.tokens.budget / a.time.tokensPerSecond / 3600, 9);
    const wider = analyze(E.setStackWidth(g, "main", 768));
    expect(wider.tokens.budget).toBe(20 * wider.params.total);
    expect(wider.tokens.budget).toBeGreaterThan(a.tokens.budget);
  });

  test("tokens per parameter and the Chinchilla multiple for a custom budget", () => {
    const g = buildPreset("e33m_plain_30m");
    Object.assign(g.run, { token_rule: "custom", token_budget: 1.5e9 });
    const t = analyze(g).tokens;
    expect(t.perParam).toBeCloseTo(1.5e9 / N, 6);
    expect(t.chinchillaMultiple).toBeCloseTo(1.5e9 / N / 20, 6);
  });
});

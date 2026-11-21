import { describe, expect, test } from "vitest";
import { TOKEN_STEPS, nearestTokenStep, parseTokens, tokens } from "../src/ui/format";

describe("token counts read in billions", () => {
  test.each([
    [1.5e9, "1.5B"],
    [598_928_040, "0.6B"],
    [0.5e9, "0.5B"],
    [56.14e9, "56.1B"],
    [443.7e9, "444B"],
    [300e9, "300B"],
    [2e12, "2T"],
    [15e12, "15T"],
    [20e6, "0.02B"],
  ])("%d tokens → %s", (n, text) => {
    expect(tokens(n)).toBe(text);
  });
});

describe("typing a token count", () => {
  test.each([
    ["1.5", 1.5e9],
    ["1.5B", 1.5e9],
    ["1.5 b", 1.5e9],
    ["500M", 5e8],
    ["2T", 2e12],
    ["300", 300e9],
    ["1,5", 1.5e9],
    ["1500000000", 1.5e9],
    ["1,500,000,000", 1.5e9],
    ["3e9", 3e9],
  ])("%s → %d", (text, n) => {
    expect(parseTokens(text)).toBe(n);
  });

  test("nonsense is refused", () => {
    expect(parseTokens("abc")).toBeNull();
    expect(parseTokens("")).toBeNull();
    expect(parseTokens("-2")).toBeNull();
  });
});

describe("the slider's round steps", () => {
  test("they include the round budgets people use", () => {
    for (const n of [0.5e9, 1e9, 1.5e9, 2e9, 300e9, 1e12]) expect(TOKEN_STEPS).toContain(n);
    expect([...TOKEN_STEPS].sort((a, b) => a - b)).toEqual(TOKEN_STEPS);
  });

  test("a value between steps sits at the nearest step", () => {
    expect(TOKEN_STEPS[nearestTokenStep(598_928_040)]).toBe(0.6e9);
    expect(TOKEN_STEPS[nearestTokenStep(1.5e9)]).toBe(1.5e9);
  });
});

import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";
import type { Graph } from "../src/engine/types";

const on3090 = (id: string, micro: number, mutate?: (g: Graph) => void) => {
  const g = buildPreset(id);
  g.run.gpu = "rtx3090";
  g.run.micro_batch = micro;
  g.run.row_length = 4096;
  mutate?.(g);
  return analyze(g);
};

// Measured on the project's RTX 3090s, 4,096-token rows, bf16 autocast, AdamW, no gradient checkpointing.
describe("GPU memory matches what the 24 GB cards did", () => {
  test("dense 30M fits 8 rows", () => {
    expect(on3090("dense_plain_30m", 8).memory.fits).toBe(true);
  });
  test("E33 30M runs out of memory at 8 rows", () => {
    expect(on3090("e33m_plain_30m", 8).memory.fits).toBe(false);
  });
  test("E33 30M fits 4 rows", () => {
    expect(on3090("e33m_plain_30m", 4).memory.fits).toBe(true);
  });
  test("dense 100M fits 4 rows", () => {
    expect(on3090("dense_plain_100m", 4).memory.fits).toBe(true);
  });
  test("the largest micro-batch the planner picks for E33 30M is 4 to 7 rows", () => {
    const b = on3090("e33m_plain_30m", 0).memory.microBatch;
    expect(b).toBeGreaterThanOrEqual(4);
    expect(b).toBeLessThan(8);
  });
  test("gradient checkpointing cuts activation memory", () => {
    const off = on3090("e33m_plain_30m", 8).memory;
    const on = on3090("e33m_plain_30m", 8, (g) => (g.run.checkpointing = true)).memory;
    expect(on.activations).toBeLessThan(0.5 * off.activations);
    expect(on.fits).toBe(true);
  });
});

describe("speed is calibrated on measured RTX 3090 throughput", () => {
  const perGpu = (id: string) => on3090(id, 0).time.tokensPerSecondPerGpu;
  test("dense 30M ≈ 60k tokens/s", () => {
    expect(Math.abs(perGpu("dense_plain_30m") / 60_000 - 1)).toBeLessThan(0.03);
  });
  test("E33 30M ≈ 20k tokens/s", () => {
    expect(Math.abs(perGpu("e33m_plain_30m") / 20_000 - 1)).toBeLessThan(0.03);
  });
  test("dense 50M ≈ 41.5k tokens/s (a held-out check, not used to calibrate)", () => {
    expect(Math.abs(perGpu("dense_plain_50m") / 41_500 - 1)).toBeLessThan(0.15);
  });
});

describe("the GPU planner", () => {
  test("wall-clock time is the token budget over the cluster's throughput", () => {
    const g = buildPreset("dense_plain_30m");
    Object.assign(g.run, { gpu: "rtx3090", gpu_count: 3, token_budget: 1.5e9, dp_efficiency: 0.95 });
    const t = analyze(g).time;
    expect(t.hours).toBeCloseTo(1.5e9 / (3 * 0.95 * t.tokensPerSecondPerGpu) / 3600, 6);
  });
  test("a model too big for one card asks for sharded GPUs", () => {
    const g = buildPreset("dense_plain_100m");
    g.stacks[0].width = 4096; // blow the model up to ~2B parameters by widening every layer
    for (const row of g.stacks[0].rows)
      for (const c of row.cells) {
        if (c.type === "attention") c.heads = 64;
        if (c.type === "ffn") c.hidden = 11008;
      }
    Object.assign(g.run, { gpu: "rtx3090", zero_stage: 3, micro_batch: 1 });
    const a = analyze(g);
    expect(a.params.total).toBeGreaterThan(1.5e9);
    expect(a.gpus.minForMemory).toBeGreaterThan(1);
  });
});

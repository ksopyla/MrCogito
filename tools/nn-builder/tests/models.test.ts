// Exact parameter counts for the reference architectures beyond MrCogito's own text models.
import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";

const total = (id: string) => analyze(buildPreset(id)).params.total;

describe("MrCogito E31c, the latent notebook without the loop (trainer counts, 25 Oct)", () => {
  test("30M", () => expect(total("e31c_plain_30m")).toBe(29_855_316));
  test("100M", () => expect(total("e31c_plain_100m")).toBe(100_823_774));
});

describe("MrCogito E36, a trained writer for a frozen SmolLM2-360M reader", () => {
  const a = analyze(buildPreset("e36_kv"));
  test("trainable: 3 SmolLM2-135M layers + norm (10,620,864) + the key/value expander (11,817,024)", () => {
    expect(a.params.trainable).toBe(22_437_888);
  });
  test("with the frozen reader (361,821,120) and the borrowed token table (28,311,552)", () => {
    expect(a.params.total).toBe(22_437_888 + 361_821_120 + 28_311_552);
  });
  test("frozen weights need no gradients or optimizer state", () => {
    const g = buildPreset("e36_kv");
    g.run.precision = "amp_bf16";
    const m = analyze(g).memory;
    expect(m.optimizer).toBe(22_437_888 * 8);
    expect(m.grads).toBe(22_437_888 * 4);
  });
  test("every reader layer reads the 8 entries", () => {
    expect(a.issues.filter((i) => /notes|memory/.test(i.msg))).toEqual([]);
    expect(buildPreset("e36_kv").edges.filter((e) => e.kind === "source")).toHaveLength(32);
  });
});

describe("outside references (Hugging Face configs, 25 Oct)", () => {
  test("SmolLM3-3B: 36 layers, GQA 16/4, NoPE every 4th layer, tied 128k table", () => {
    expect(total("smollm3_3b")).toBe(3_075_098_624);
  });

  test("DeepSeek-V2-Lite: MLA + 64 routed / 2 shared experts", () => {
    expect(total("deepseek_v2_lite")).toBe(15_706_484_224);
  });

  test("DeepSeek-V2-Lite uses about 2.4B parameters per token", () => {
    const active = analyze(buildPreset("deepseek_v2_lite")).params.active;
    expect(active).toBeGreaterThan(2.3e9);
    expect(active).toBeLessThan(2.5e9);
  });

  test("T5-small: shared table tied to the head, relative position bias", () => {
    expect(total("encdec")).toBe(60_506_624);
    expect(analyze(buildPreset("encdec")).issues).toEqual([]);
  });

  test("DeepSeek-V4.1-Flash, approximate: backbone near 552B plus a 196B Engram memory", () => {
    const a = analyze(buildPreset("deepseek_v41_flash"));
    const engram = a.params.byType.custom ?? 0;
    expect(Math.abs(a.params.total - engram - 552e9) / 552e9).toBeLessThan(0.03);
    expect(Math.abs(engram - 196e9) / 196e9).toBeLessThan(0.01);
  });
});

describe("dense models: every parameter is used per token except the embedding lookup", () => {
  test("dense 30M", () => {
    const a = analyze(buildPreset("dense_plain_30m"));
    expect(a.params.active).toBe(a.params.total - 4096 * 256);
    expect(a.params.trainable).toBe(a.params.total);
  });
});

import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { buildPreset } from "../src/engine/presets";
import { cellTensorSteps, edgeTensors, fmtShape, rowInputShape, stackOutputShape } from "../src/engine/shapes";

const sizes = (dims: { v: number }[]) => dims.map((d) => d.v);
const setup = (id: string) => {
  const g = buildPreset(id);
  return { g, a: analyze(g) };
};
const step = (steps: ReturnType<typeof cellTensorSteps>, label: RegExp) => {
  const s = steps.find((x) => label.test(x.label));
  if (!s) throw new Error(`no step ${label} in ${steps.map((x) => x.label).join(" | ")}`);
  return sizes(s.dims);
};

describe("tensor shapes through the dense model (B = 8 rows)", () => {
  const { g, a } = setup("dense_plain_30m");

  test("token ids become vectors in the input layer", () => {
    const steps = cellTensorSteps(g, a, "main.input.emb", 8);
    expect(sizes(steps[0].dims)).toEqual([8, 4096]);
    expect(step(steps, /table/)).toEqual([8, 4096, 256]);
    expect(sizes(steps.at(-1)!.dims)).toEqual([8, 4096, 576]);
  });

  test("attention: queries, shared key/value head, scores over the row", () => {
    const steps = cellTensorSteps(g, a, "main.L0.attn", 8);
    expect(step(steps, /^queries/)).toEqual([8, 4096, 9, 64]);
    expect(step(steps, /^keys/)).toEqual([8, 4096, 1, 64]);
    expect(step(steps, /^scores/)).toEqual([8, 9, 4096, 4096]);
    expect(sizes(steps.at(-1)!.dims)).toEqual([8, 4096, 576]);
  });

  test("the feed-forward widens to its hidden size and back", () => {
    const steps = cellTensorSteps(g, a, "main.L0.ffn", 8);
    expect(step(steps, /hidden/)).toEqual([8, 4096, 1472]);
    expect(sizes(steps.at(-1)!.dims)).toEqual([8, 4096, 576]);
  });

  test("the head gives logits over the vocabulary", () => {
    expect(step(cellTensorSteps(g, a, "main.head.lm", 8), /^logits/)).toEqual([8, 4096, 4096]);
    expect(sizes(stackOutputShape(g, a, "main", 8))).toEqual([8, 4096, 4096]);
  });

  test("each layer row takes the stream as [B, S, d]", () => {
    expect(fmtShape(rowInputShape(g, a, "main.L3", 8), "compact")).toBe("B×4k×576");
    expect(fmtShape(rowInputShape(g, a, "main.input", 8), "compact")).toBe("ids B×4k");
  });
});

describe("tensor shapes through E33's page writer and notes", () => {
  const { g, a } = setup("e33m_plain_30m");

  test("pages are [B, pages, page tokens, width]", () => {
    expect(sizes(rowInputShape(g, a, "pages.enc0", 4))).toEqual([4, 21, 256, 256]);
  });

  test("each page's 32 slots read their own page", () => {
    const steps = cellTensorSteps(g, a, "notes.read.attn", 4);
    expect(step(steps, /^queries/)).toEqual([4, 21, 32, 8, 64]);
    expect(step(steps, /^scores/)).toEqual([4, 21, 8, 32, 256]);
  });

  test("the notes are 21 × 32 ready keys/values", () => {
    expect(sizes(stackOutputShape(g, a, "notes", 4))).toEqual([4, 672, 1, 64]);
  });

  test("the read scores the last 256 tokens plus every note", () => {
    expect(step(cellTensorSteps(g, a, "main.L2.attn", 4), /^scores/)).toEqual([4, 9, 4096, 256 + 672]);
  });
});

describe("cross-attention reads the encoder", () => {
  test("keys/values come from the encoder's 512 positions (T5-small)", () => {
    const { g, a } = setup("encdec");
    const steps = cellTensorSteps(g, a, "dec.L0.cross", 2);
    expect(step(steps, /^keys/)).toEqual([2, 512, 8, 64]);
    expect(step(steps, /^scores/)).toEqual([2, 8, 512, 512]);
  });
});

describe("what a wire carries and what the other end does with it", () => {
  test("the token table feeding the page writer is cut into pages", () => {
    const { g, a } = setup("e33m_plain_30m");
    const w = edgeTensors(g, a, "wire.table", 4);
    expect(sizes(w.from.dims)).toEqual([4, 4096, 256]);
    expect(step(w.steps, /^cut into pages/)).toEqual([4, 21, 256, 256]);
    expect(sizes(w.to.dims)).toEqual([4, 21, 256, 256]);
  });

  test("the notes become the read's extra keys/values", () => {
    const { g, a } = setup("e33m_plain_30m");
    const w = edgeTensors(g, a, "wire.notes", 4);
    expect(sizes(w.from.dims)).toEqual([4, 672, 1, 64]);
    expect(step(w.steps, /^scores/)).toEqual([4, 9, 4096, 256 + 672]);
  });

  test("the encoder output is projected into keys/values by each cross-attention", () => {
    const { g, a } = setup("encdec");
    const w = edgeTensors(g, a, "wire.cross0", 2);
    expect(sizes(w.from.dims)).toEqual([2, 512, 512]);
    expect(step(w.steps, /^keys/)).toEqual([2, 512, 8, 64]);
  });

  test("a skip adds one row's output to a later row's input, times one weight", () => {
    const { g, a } = setup("dense_plain_30m");
    const w = edgeTensors(g, a, "main.skip0", 8);
    expect(sizes(w.from.dims)).toEqual([8, 4096, 576]);
    expect(sizes(w.to.dims)).toEqual([8, 4096, 576]);
    expect(w.params).toBe(1);
  });
});

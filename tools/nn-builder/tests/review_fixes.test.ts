// Regression tests for bugs found in review (24 Oct 2026).
import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";
import { historyStep, type HistoryState } from "../src/ui/history";

describe("changing one attention extra keeps the others", () => {
  test("toggling the logit scale on the E33 read keeps its count heads and order term", () => {
    const g = E.updateCell(buildPreset("e33m_plain_30m"), "main.L2.attn", { extras: { logit_scale: true } } as never);
    const c = E.findCell(g, "main.L2.attn")!.cell;
    expect(c.type === "attention" && c.extras).toMatchObject({ logit_scale: true, count_heads: 2, count_max: 10, order_term: true });
    expect(analyze(g).params.total).toBe(29_946_402 + 1);
  });
});

describe("loops stay whole when rows move into them", () => {
  test("duplicating a looped row puts the copy inside the loop", () => {
    const g = E.duplicateRow(buildPreset("e33m_plain_30m"), "main.L2");
    const loop = g.groups.find((x) => x.id === "loop.read")!;
    expect(loop.rows).toHaveLength(3);
    expect(analyze(g).issues.filter((i) => /neighbouring/.test(i.msg))).toEqual([]);
  });

  test("moving a row between two looped rows makes it part of the loop", () => {
    const g0 = buildPreset("e33m_plain_30m");
    const at = g0.stacks[0].rows.findIndex((r) => r.id === "main.L3");
    const g = E.moveRow(g0, "main.L1", at);
    expect(g.groups.find((x) => x.id === "loop.read")!.rows).toEqual(["main.L2", "main.L1", "main.L3"]);
    expect(analyze(g).issues.filter((i) => /neighbouring/.test(i.msg))).toEqual([]);
  });
});

describe("a slider drag folds into one undo step and ends where typing would", () => {
  test("dragging the width 576 → 1152 in steps of 8 rescales heads and FFN like one jump", () => {
    let st: HistoryState = { past: [], present: buildPreset("dense_plain_30m"), future: [], key: null, at: 0 };
    for (let w = 584; w <= 1152; w += 8) st = historyStep(st, (g) => E.setStackWidth(g, "main", w), "width:main", 1000 + w);
    const jump = E.setStackWidth(buildPreset("dense_plain_30m"), "main", 1152);
    expect(st.past).toHaveLength(1);
    expect(analyze(st.present).params.total).toBe(analyze(jump).params.total);
  });
});

describe("a shared token table that is gone", () => {
  test("removing the encoder gives the decoder its own table back", () => {
    const g = E.removeStack(buildPreset("encdec"), "enc");
    const emb = E.findCell(g, "dec.input.emb")!.cell;
    expect(emb.type === "embedding" && emb.share_with).toBeFalsy();
    expect(analyze(g).stacks[0].rows[0].cells[0].params).toBe(32128 * 512);
  });

  test("a share pointing at nothing counts the table and warns", () => {
    const g = buildPreset("encdec");
    const emb = E.findCell(g, "dec.input.emb")!.cell;
    if (emb.type === "embedding") emb.share_with = "no.such.cell";
    const a = analyze(g);
    expect(a.cells.get("dec.input.emb")!.params).toBe(32128 * 512);
    expect(a.issues.some((i) => i.where === "dec.input.emb" && /shares a table/.test(i.msg))).toBe(true);
  });
});

describe("repeats copy rows, not the skip wires into them", () => {
  test("a scalar skip into a ×4 repeated row adds one weight; its x0 mix is copied", () => {
    const g0 = buildPreset("dense_plain_30m"); // L4 takes a skip from L3 and mixes x0
    const g = E.addGroup(g0, "repeat", "main", ["main.L4"], 4);
    const row = (x: typeof g) => analyze(x).stacks[0].rows.find((r) => r.id === "main.L4")!;
    const layer = row(g0).params - row(g0).rowExtras; // one copy without x0 and skip
    expect(row(g).params).toBe(4 * layer + 4 * 2 + 1);
  });
});

describe("GPUs for a deadline", () => {
  test("a run one GPU finishes in time asks for one GPU", () => {
    const g = buildPreset("dense_plain_30m");
    const tps = analyze(g).time.tokensPerSecondPerGpu;
    Object.assign(g.run, { token_budget: 0.98 * tps * 3600, deadline_hours: 1, dp_efficiency: 0.9 });
    expect(analyze(g).gpus.forDeadline).toBe(1);
  });
});

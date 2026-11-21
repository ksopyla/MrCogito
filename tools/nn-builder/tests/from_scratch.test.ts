// Building a network from "Blank: one layer" with the moves the interface offers: what each new stack starts with,
// what the warnings say when something is missing, and a notebook read taking on its notes' shape.
import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import { PALETTE } from "../src/engine/defaults";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";
import type { AttentionCell, Graph, Stack } from "../src/engine/types";

const messages = (g: Graph) => analyze(g).issues.map((i) => i.msg);
const byKind = (g: Graph, kind: Stack["seq"]["kind"]) => g.stacks.find((s) => s.seq.kind === kind)!;
const cellOf = (s: Stack, type: string) => s.rows.flatMap((r) => r.cells).find((c) => c.type === type)!;
const read = () => PALETTE.find((x) => x.key === "memory_read")!.make(512);

/** Blank + page writer + notes, the pages wired into the notes. */
function withNotes(): Graph {
  const g = E.addStack(E.addStack(buildPreset("blank"), "pages"), "slots");
  return E.connect(g, { stack: byKind(g, "pages").id }, { stack: byKind(g, "slots").id, cell: cellOf(byKind(g, "slots"), "attention").id }).graph;
}
/** …plus a notebook read in the decoder, not wired yet. */
function withRead(): { g: Graph; id: string } {
  const g = E.insertRow(withNotes(), "main", 2, [read()]);
  return { g, id: g.stacks[0].rows[2].cells[0].id };
}
const readOf = (g: Graph, id: string) => E.findCell(g, id)!.cell as AttentionCell;

describe("a new page writer", () => {
  test("starts from the token table of the stack it pages", () => {
    const g = E.addStack(buildPreset("blank"), "pages");
    const pages = byKind(g, "pages");
    expect(g.edges).toEqual([expect.objectContaining({ kind: "input", from: { stack: "main", cell: "main.input.emb", port: "table" }, to: { stack: pages.id } })]);
    expect(analyze(g).issues.filter((i) => i.level !== "warning")).toEqual([]);
  });

  test("unwired again, the warning names the dot to drag and where", () => {
    const g = E.addStack(buildPreset("blank"), "pages");
    const cut = { ...g, edges: [] };
    expect(messages(cut).join()).toMatch(/nothing starts.*token-table dot.*embedding.*onto this stack's header/);
  });

  test("over a stack with no embedding gets no wire, and the warning names its output dot", () => {
    const base = E.removeCell(buildPreset("blank"), "main.input.emb");
    const g = E.addStack(base, "pages");
    expect(g.edges).toEqual([]);
    expect(messages(g).join()).toMatch(/nothing starts.*output dot onto this stack's header/);
  });
});

describe("notes with no page writer", () => {
  const g = E.addStack(buildPreset("blank"), "slots");

  test("say what to add, without naming an empty stack", () => {
    const m = messages(g);
    expect(m.join()).toMatch(/there is none yet: add one with \+ Stack → Pages/);
    expect(m.some((x) => /unknown stack/.test(x))).toBe(false);
  });

  test("attach to the page writer added after them", () => {
    const next = E.addStack(g, "pages");
    expect(byKind(next, "slots").seq).toMatchObject({ of: byKind(next, "pages").id });
    expect(messages(next).some((x) => /none yet|sit on no page writer/.test(x))).toBe(false);
  });

  test("lose it and say so when the page writer is removed, then attach to a new one", () => {
    const both = withNotes();
    const gone = E.removeStack(both, byKind(both, "pages").id);
    expect(messages(gone).join()).toMatch(/there is none yet/);
    const again = E.addStack(gone, "pages");
    expect(byKind(again, "slots").seq).toMatchObject({ of: byKind(again, "pages").id });
  });

  test("a page writer added beside attached notes leaves them where they are", () => {
    const both = withNotes();
    const first = byKind(both, "pages").id;
    expect(byKind(E.addStack(both, "pages"), "slots").seq).toMatchObject({ of: first });
  });

  test("on a stack that is no page writer, the warning points at the setting", () => {
    const both = withNotes();
    const stray = E.updateStack(both, byKind(both, "slots").id, { seq: { kind: "slots", of: "main", per_page: 32 } });
    expect(messages(stray).join()).toMatch(/sit on no page writer: choose one under "Slots on the pages of"/);
  });
});

describe("wiring notes into a notebook read", () => {
  test("the read takes on the notes' key/value shape, and the design is clean", () => {
    const { g, id } = withRead();
    expect(readOf(g, id)).toMatchObject({ kv_heads: 8, head_dim: 64 });
    expect(messages(g).join()).toMatch(/memory read has no notes/);
    const res = E.connect(g, { stack: byKind(g, "slots").id }, { stack: "main", cell: id });
    expect(readOf(res.graph, id)).toMatchObject({ kv_heads: 1, head_dim: 64, heads: 8 });
    expect(res.note).toMatch(/1 × 64 key\/value heads \(was 8 × 64\).*match the notes/);
    expect(analyze(res.graph).issues).toEqual([]);
  });

  test("heads stay a multiple of the key/value heads, and the note says so", () => {
    const { g, id } = withRead();
    const notes = byKind(g, "slots");
    const odd = E.updateCell(g, cellOf(notes, "kv_out").id, { kv_heads: 3, head_dim: 32 } as never);
    const res = E.connect(odd, { stack: notes.id }, { stack: "main", cell: id });
    expect(readOf(res.graph, id)).toMatchObject({ kv_heads: 3, head_dim: 32, heads: 9 });
    expect(res.note).toMatch(/9 heads \(was 8\)/);
    expect(messages(res.graph).some((m) => /notes are|multiple of/.test(m))).toBe(false);
  });

  test("notes changed later warn, and the warning says how to match them", () => {
    const { g, id } = withRead();
    const notes = byKind(g, "slots");
    const wired = E.connect(g, { stack: notes.id }, { stack: "main", cell: id }).graph;
    const wider = E.updateCell(wired, cellOf(notes, "kv_out").id, { kv_heads: 2 } as never);
    expect(messages(wider).join()).toMatch(/notes are 2 × 64 but this read uses 1 × 64 key\/value heads: drag the wire onto it again/);
    expect(readOf(E.connect(wider, { stack: notes.id }, { stack: "main", cell: id }).graph, id).kv_heads).toBe(2);
  });

  test("a matching read changes nothing: E33 wired again keeps its exact count", () => {
    const g = buildPreset("e33m_plain_30m");
    const cut = E.removeEdge(g, "wire.notes");
    const res = E.connect(cut, { stack: "notes" }, { stack: "main", cell: "main.L2.attn" });
    expect(res.note).toBeUndefined();
    expect(analyze(res.graph).params.total).toBe(29_946_402);
  });

  test("a wire from an ordinary stack leaves the attention's heads alone", () => {
    const withR = E.insertRow(buildPreset("blank"), "main", 2, [read()]);
    const id = withR.stacks[0].rows[2].cells[0].id;
    const res = E.connect(withR, { stack: "main", row: "main.L0" }, { stack: "main", cell: id });
    expect(res.note).toBeUndefined();
    expect(readOf(res.graph, id).kv_heads).toBe(8);
  });
});

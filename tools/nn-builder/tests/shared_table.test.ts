// The token table on screen: an embedding drawn as table + input MLP, and the dashed block on a stack a table wire
// feeds. Display only: neither changes a count, the file or the agent's spec.
import { describe, expect, test } from "vitest";
import { analyze } from "../src/engine/analyze";
import * as E from "../src/engine/edit";
import { PRESETS, buildPreset } from "../src/engine/presets";
import { buildSpec } from "../src/engine/spec";
import type { EmbeddingCell, Graph } from "../src/engine/types";
import { embeddingParts, sharedTable } from "../src/ui/describe";

const stack = (g: Graph, id: string) => g.stacks.find((s) => s.id === id)!;
const emb = (g: Graph, id: string) => E.findCell(g, id)!.cell as EmbeddingCell;
const partsOf = (g: Graph, id: string) => {
  const res = analyze(g).cells.get(id)!;
  return embeddingParts(emb(g, id), res.widthOut, res);
};

describe("an embedding with an input MLP is drawn in two parts", () => {
  test("E33 30M: the table and the MLP add up to the cell", () => {
    const g = buildPreset("e33m_plain_30m");
    expect(partsOf(g, "main.input.emb")).toEqual([
      { name: "Token table", sub: "4,096 × 256", params: 1_048_576 },
      { name: "Input MLP", sub: "256 → 576", params: 627_264 },
    ]);
    expect(analyze(g).cells.get("main.input.emb")!.params).toBe(1_048_576 + 627_264);
  });

  test("a linear projection is named as one", () => {
    const g = E.updateCell(buildPreset("e33m_plain_30m"), "main.input.emb", { proj: "linear" } as never);
    expect(partsOf(g, "main.input.emb")![1].name).toBe("Input projection");
  });

  test("n-gram tables count with the table, a shared table holds nothing", () => {
    const g = E.updateCell(buildPreset("e33m_plain_30m"), "main.input.emb", { ngram_tables: 2, ngram_buckets: 100 } as never);
    expect(partsOf(g, "main.input.emb")![0]).toMatchObject({ sub: "4,096 × 256 + 2 n-gram", params: 1_048_576 + 2 * 100 * 256 });
    const two = E.addStack(buildPreset("e33m_plain_30m"), "encoder");
    const second = two.stacks.at(-1)!.rows[0].cells[0].id;
    const shared = E.updateCell(two, second, { proj: "gated_mlp", share_with: "main.input.emb" } as never);
    expect(partsOf(shared, second)![0]).toMatchObject({ name: "Token table (shared)", params: 0 });
  });

  test("with no MLP it stays one part", () => {
    expect(partsOf(buildPreset("blank"), "main.input.emb")).toBeNull();
    expect(partsOf(buildPreset("gpt2_small"), "main.input.emb")).toBeNull();
  });
});

describe("a stack fed by a token-table wire shows the table it shares", () => {
  test("E33's Pages does; its other stacks do not", () => {
    const g = buildPreset("e33m_plain_30m");
    const s = sharedTable(stack(g, "pages"), g)!;
    expect(s.from.name).toBe("Decoder");
    expect(s.cell.id).toBe("main.input.emb");
    expect(sharedTable(stack(g, "main"), g)).toBeNull();
    expect(sharedTable(stack(g, "notes"), g)).toBeNull();
  });

  test("it follows the wire, on any design: a new page writer has it, removing the wire removes it", () => {
    const g = E.addStack(buildPreset("blank"), "pages");
    const pages = g.stacks.at(-1)!;
    expect(sharedTable(pages, g)?.from.id).toBe("main");
    expect(sharedTable(pages, E.removeEdge(g, g.edges[0].id))).toBeNull();
  });

  test("a wire from a stack's output is not a shared table", () => {
    const g = buildPreset("e33m_plain_30m");
    const other = E.connect(g, { stack: "main", row: "main.L0" }, { stack: "pages" }).graph;
    expect(sharedTable(stack(other, "pages"), other)).toBeNull();
  });

  test("the only presets with a table wire are the four notebook models, and they read clearly", () => {
    const fed = PRESETS.filter((p) => p.build().edges.some((e) => e.from.port === "table")).map((p) => p.id);
    expect(fed).toEqual(["e33m_plain_30m", "e33m_plain_100m", "e31c_plain_30m", "e31c_plain_100m"]);
    for (const id of fed) expect(E.findCell(buildPreset(id), "pages.in.proj")!.cell.label).toBe("Project to writer");
  });

  test("the block is no cell: Pages keeps its blocks and its count in the numbers and the spec", () => {
    const g = buildPreset("e33m_plain_30m");
    const pages = buildSpec(g).stacks.find((s) => s.id === "pages")!;
    expect(pages.blocks[0]).toMatchObject({ id: "pages.in.proj", label: "Project to writer", parameters: { count: 65_792 } });
    expect(pages.blocks.some((b) => b.type === "embedding")).toBe(false);
    expect(analyze(g).params.total).toBe(29_946_402);
  });
});

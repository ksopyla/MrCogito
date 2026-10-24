// Export / import of the graph file (docs/schema.md). The export adds a `derived` block: the estimates at
// export time, so a code generator can check the model it writes against the counts the builder showed.
import { analyze } from "./analyze";
import { DEFAULT_RUN } from "./defaults";
import { SCHEMA_ID, SCHEMA_VERSION, type Graph } from "./types";

export interface Derived {
  exported_at: string;
  row_length: number;
  params_total: number;
  params_by_stack: Record<string, number>;
  train_flops_per_token: number;
  cells: Record<string, { params: number; width_in: number; width_out: number; runs_per_row: number }>;
}

export type ExportedGraph = Graph & { derived: Derived };

export function toExport(g: Graph): ExportedGraph {
  const a = analyze(g);
  const cells: Derived["cells"] = {};
  for (const [id, c] of a.cells) cells[id] = { params: c.params, width_in: c.widthIn, width_out: c.widthOut, runs_per_row: c.execs };
  const clean = JSON.parse(JSON.stringify(g)) as Graph & { derived?: unknown };
  delete clean.derived;
  return {
    ...clean,
    derived: {
      exported_at: new Date().toISOString(),
      row_length: a.S,
      params_total: a.params.total,
      params_by_stack: a.params.byStack,
      train_flops_per_token: Math.round(a.flops.trainPerToken),
      cells,
    },
  };
}

export function parseGraph(text: string): { graph?: Graph; error?: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { error: `not valid JSON: ${(e as Error).message}` };
  }
  const o = raw as Partial<Graph> & { derived?: unknown };
  if (!o || typeof o !== "object" || o.schema !== SCHEMA_ID) return { error: "not an nn-builder graph (the file has no \"schema\": \"nn-builder/graph\")" };
  if (o.version !== SCHEMA_VERSION) return { error: `graph format version ${String(o.version)} is not supported (this app reads ${SCHEMA_VERSION})` };
  if (!Array.isArray(o.stacks) || o.stacks.length === 0) return { error: "the graph has no stacks" };
  const { derived: _derived, ...rest } = o;
  void _derived;
  const graph: Graph = {
    ...(rest as Graph),
    description: o.description ?? "",
    edges: o.edges ?? [],
    groups: o.groups ?? [],
    run: { ...DEFAULT_RUN, ...(o.run ?? {}) },
  };
  return { graph };
}

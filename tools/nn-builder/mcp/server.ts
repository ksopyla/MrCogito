// nn-builder MCP server: read-only access to saved designs for coding agents (Claude Code, Cursor, Codex, …).
// Agents list designs, read a design's spec (everything needed to implement it), validate it, and ask what changed
// since the version they implemented. Runs over stdio; reads the designs folder, so the app need not be running.
// Start: node scripts/mcp.mjs [--designs <folder>]. Guide: docs/agent-spec.md.
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { analyze } from "../src/engine/analyze";
import { parseGraph } from "../src/engine/serialize";
import { buildSpec, designHash, diffDesigns, validateDesign } from "../src/engine/spec";
import type { Graph } from "../src/engine/types";

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, here.endsWith(path.join("mcp", "dist")) ? "../.." : "..");
const flag = (name: string) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
};
const designsDir = path.resolve(flag("--designs") ?? process.env.NN_BUILDER_DESIGNS ?? path.join(appRoot, "designs"));
const historyDir = path.join(designsDir, ".history");
const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;

// ------------------------------------------------------------------ the designs folder

async function designNames(): Promise<string[]> {
  try {
    return (await fs.readdir(designsDir)).filter((f) => f.endsWith(".json")).map((f) => f.replace(/\.json$/, "")).filter((n) => NAME.test(n)).sort();
  } catch {
    return [];
  }
}

async function loadDesign(name: string): Promise<{ graph: Graph; file: string; modified: string }> {
  if (!NAME.test(name)) throw new Error(`bad design name: ${name}`);
  const file = path.join(designsDir, `${name}.json`);
  let text: string;
  try {
    text = await fs.readFile(file, "utf8");
  } catch {
    const known = await designNames();
    throw new Error(`no design "${name}" in ${designsDir}. Saved designs: ${known.join(", ") || "none"}. Save it in the app first.`);
  }
  const res = parseGraph(text);
  if (!res.graph) throw new Error(`${file}: ${res.error}`);
  return { graph: res.graph, file, modified: (await fs.stat(file)).mtime.toISOString() };
}

/** Keep a copy of every version an agent has seen, by hash, so "what changed since" has an answer later. */
async function remember(name: string, g: Graph): Promise<string> {
  const hash = designHash(g);
  const dir = path.join(historyDir, name);
  const file = path.join(dir, `${hash}.json`);
  try {
    await fs.access(file);
  } catch {
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(file, JSON.stringify(g, null, 2) + "\n");
  }
  return hash;
}

async function recalled(name: string, hash: string): Promise<Graph | undefined> {
  if (!/^nnb1-[0-9a-f]{28}$/.test(hash)) return undefined;
  try {
    const g = parseGraph(await fs.readFile(path.join(historyDir, name, `${hash}.json`), "utf8")).graph;
    // a copy is trusted only if it still hashes to its name (the app names save copies by the hash it computed)
    return g && designHash(g) === hash ? g : undefined;
  } catch {
    return undefined;
  }
}

async function knownVersions(name: string): Promise<string[]> {
  try {
    return (await fs.readdir(path.join(historyDir, name))).filter((f) => f.endsWith(".json")).map((f) => f.replace(/\.json$/, ""));
  } catch {
    return [];
  }
}

const result = (data: Record<string, unknown>) => ({ content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }], structuredContent: data });
const failure = (e: unknown) => ({ content: [{ type: "text" as const, text: (e as Error).message }], isError: true });

// ------------------------------------------------------------------ the server

const server = new McpServer(
  { name: "nn-builder", version: "1.0.0" },
  {
    instructions:
      "Read-only access to neural-network designs drawn in NN Builder. To implement a design: list_designs, then get_design_spec (its how_to_read explains the format); build only when validation.errors is empty; keep design.hash with your code. When the user changes the design, call diff_design with that hash to see what changed.",
  },
);

server.registerTool(
  "list_designs",
  {
    title: "List saved designs",
    description: "The designs saved in NN Builder: name, last save, architecture hash, parameter count, and how many validation errors and warnings each has.",
    inputSchema: {},
    annotations: { readOnlyHint: true },
  },
  async () => {
    try {
      const designs = [];
      for (const name of await designNames()) {
        try {
          const d = await loadDesign(name);
          const a = analyze(d.graph);
          const v = validateDesign(d.graph);
          designs.push({ name, title: d.graph.name, saved: d.modified, hash: designHash(d.graph), parameters: a.params.total, stacks: d.graph.stacks.map((s) => s.name), errors: v.errors.length, warnings: v.warnings.length });
        } catch (e) {
          designs.push({ name, error: (e as Error).message });
        }
      }
      return result({ folder: designsDir, designs });
    } catch (e) {
      return failure(e);
    }
  },
);

server.registerTool(
  "get_design_spec",
  {
    title: "Get a design's spec",
    description:
      "Everything needed to implement a design: blocks with resolved settings, input/output shapes, what each block does and its tensor steps; rows and what is added before them; execution order with loops and repeats; wires and what they carry; named dims; exact parameter counts; validation; and the architecture hash. Its how_to_read field explains the conventions.",
    inputSchema: { name: z.string().describe("the design's file name without .json, as list_designs shows it") },
    annotations: { readOnlyHint: true },
  },
  async ({ name }) => {
    try {
      const d = await loadDesign(name);
      await remember(name, d.graph);
      return result({ ...buildSpec(d.graph), saved: d.modified, file: d.file } as unknown as Record<string, unknown>);
    } catch (e) {
      return failure(e);
    }
  },
);

server.registerTool(
  "validate_design",
  {
    title: "Validate a design",
    description: "Check that the design can be built: every stack's stream starts somewhere, dims fit between blocks, every attention that reads another stack is wired, loops cover neighbouring rows. Errors must be empty before building; warnings are worth a look.",
    inputSchema: { name: z.string().describe("the design's file name without .json") },
    annotations: { readOnlyHint: true },
  },
  async ({ name }) => {
    try {
      const d = await loadDesign(name);
      const v = validateDesign(d.graph);
      return result({ name, hash: designHash(d.graph), ok: v.errors.length === 0, ...v });
    } catch (e) {
      return failure(e);
    }
  },
);

server.registerTool(
  "diff_design",
  {
    title: "What changed since a version",
    description:
      "Compare the saved design with an earlier version, given the hash you implemented (design.hash from get_design_spec). Lists added, removed, moved and changed stacks, rows, blocks, wires and loops, field by field, with the parameter count before and after.",
    inputSchema: {
      name: z.string().describe("the design's file name without .json"),
      since_hash: z.string().describe("the design.hash of the version you implemented"),
    },
    annotations: { readOnlyHint: true },
  },
  async ({ name, since_hash }) => {
    try {
      const d = await loadDesign(name);
      await remember(name, d.graph);
      const before = await recalled(name, since_hash);
      if (!before) {
        const known = await knownVersions(name);
        throw new Error(`no record of version ${since_hash} of "${name}". Known versions: ${known.join(", ") || "none"}. Versions are recorded when a spec is read or the design is saved.`);
      }
      return result({ name, ...diffDesigns(before, d.graph) } as unknown as Record<string, unknown>);
    } catch (e) {
      return failure(e);
    }
  },
);

server.registerResource(
  "design-spec",
  new ResourceTemplate("nnb://designs/{name}/spec", {
    list: async () => ({ resources: (await designNames()).map((name) => ({ uri: `nnb://designs/${name}/spec`, name: `${name} spec`, mimeType: "application/json" })) }),
  }),
  { title: "Design spec", description: "The spec of a saved design (same as get_design_spec)", mimeType: "application/json" },
  async (uri, { name }) => {
    const d = await loadDesign(String(name));
    await remember(String(name), d.graph);
    return { contents: [{ uri: uri.href, mimeType: "application/json", text: JSON.stringify(buildSpec(d.graph), null, 2) }] };
  },
);

server.registerResource(
  "guide",
  "nnb://guide",
  { title: "How to read a design spec", description: "The agent guide: the spec format, conventions and the implement / re-implement workflow", mimeType: "text/markdown" },
  async (uri) => ({ contents: [{ uri: uri.href, mimeType: "text/markdown", text: await fs.readFile(path.join(appRoot, "docs", "agent-spec.md"), "utf8") }] }),
);

server.registerPrompt(
  "implement_design",
  {
    title: "Implement a design in PyTorch",
    description: "Instructions for implementing (or updating) a saved design",
    argsSchema: {
      name: z.string().describe("the design's file name without .json"),
      since_hash: z.string().optional().describe("the hash you implemented before, to update an existing implementation"),
    },
  },
  ({ name, since_hash }) => ({
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: [
            `Implement the NN Builder design "${name}" in PyTorch.`,
            `1. Call get_design_spec with name "${name}". Read how_to_read and block_types first. Stop and report if validation.errors is not empty.`,
            since_hash ? `2. This updates an existing implementation of version ${since_hash}: call diff_design with since_hash "${since_hash}" and change only what the diff lists.` : "2. Build one nn.Module per stack and one submodule per block, in the order of execution_order (loops reuse their modules count times; repeats make count copies).",
            "3. Match each block's settings, input and output shapes, residual rule and the row's `before` additions. Wires say which tensor each attention reads.",
            "4. Fill in what the spec leaves open (initialisation, kernels, naming) the way the surrounding code does.",
            "5. Check: count trainable parameters per block and in total against `parameters`, and run a forward pass on random token ids to check the output shapes. Report any difference.",
            "6. Keep design.hash in a comment next to the model, so the next change can be diffed against it.",
          ].join("\n"),
        },
      },
    ],
  }),
);

await server.connect(new StdioServerTransport());

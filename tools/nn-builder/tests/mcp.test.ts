// The MCP server as an agent sees it: started over stdio on a folder of saved designs, then asked to list, read,
// validate and diff them, the way Claude Code, Cursor or Codex would.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import * as E from "../src/engine/edit";
import { buildPreset } from "../src/engine/presets";
import { toExport } from "../src/engine/serialize";
import { designHash } from "../src/engine/spec";
import type { Graph } from "../src/engine/types";

const root = path.resolve(import.meta.dirname, "..");
let dir: string;
let client: Client;
const save = (name: string, g: Graph) => fs.writeFile(path.join(dir, `${name}.json`), JSON.stringify(toExport(g), null, 2));
const call = async (name: string, args: Record<string, unknown> = {}) => {
  const r = (await client.callTool({ name, arguments: args })) as CallToolResult;
  return { ...r, data: r.structuredContent as Record<string, any> };
};

beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "nnb-mcp-"));
  await save("e33", buildPreset("e33m_plain_30m"));
  const dense = buildPreset("dense_plain_30m");
  await save("headless", E.removeRow(dense, dense.stacks[0].rows[0].id));
  await fs.writeFile(path.join(dir, "broken.json"), "{ not json");
  client = new Client({ name: "nnb-test-agent", version: "0.0.0" });
  await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(root, "scripts", "mcp.mjs"), "--designs", dir], cwd: root, stderr: "pipe" }));
}, 120_000);

afterAll(async () => {
  await client?.close();
  await fs.rm(dir, { recursive: true, force: true });
});

describe("an agent reading designs over MCP", () => {
  test("offers four read-only tools, the guide and one spec per design", async () => {
    const tools = (await client.listTools()).tools;
    expect(tools.map((t) => t.name).sort()).toEqual(["diff_design", "get_design_spec", "list_designs", "validate_design"]);
    expect(tools.every((t) => t.annotations?.readOnlyHint)).toBe(true);
    const uris = (await client.listResources()).resources.map((r) => r.uri);
    expect(uris).toContain("nnb://guide");
    expect(uris).toContain("nnb://designs/e33/spec");
    expect(client.getInstructions()).toMatch(/get_design_spec/);
  });

  test("lists the saved designs with their hash, size and problems", async () => {
    const { data } = await call("list_designs");
    const byName = Object.fromEntries(data.designs.map((d: { name: string }) => [d.name, d]));
    expect(Object.keys(byName).sort()).toEqual(["broken", "e33", "headless"]);
    expect(byName.e33).toMatchObject({ title: "e33m_plain 30M", parameters: 29_946_402, errors: 0, hash: designHash(buildPreset("e33m_plain_30m")) });
    expect(byName.headless.errors).toBeGreaterThan(0);
    expect(byName.broken.error).toBeTruthy();
  });

  test("reads a design's spec: blocks, order, wires, parameters, hash", async () => {
    const { data, isError } = await call("get_design_spec", { name: "e33" });
    expect(isError).toBeFalsy();
    expect(data.format).toBe("nn-builder/spec");
    expect(data.design.hash).toMatch(/^nnb1-[0-9a-f]{28}$/);
    expect(data.parameters.total).toBe(29_946_402);
    expect(data.validation.errors).toEqual([]);
    expect(data.execution_order.some((s: { kind: string }) => s.kind === "loop")).toBe(true);
    expect(data.wires.length).toBeGreaterThan(0);
    expect(data.how_to_read.length).toBeGreaterThan(4);
  });

  test("the spec resource is the same spec", async () => {
    const r = await client.readResource({ uri: "nnb://designs/e33/spec" });
    const spec = JSON.parse((r.contents[0] as { text: string }).text);
    expect(spec.design.hash).toBe(designHash(buildPreset("e33m_plain_30m")));
  });

  test("validation names what stops a build", async () => {
    const ok = await call("validate_design", { name: "e33" });
    expect(ok.data.ok).toBe(true);
    const bad = await call("validate_design", { name: "headless" });
    expect(bad.data.ok).toBe(false);
    expect(bad.data.errors.map((e: { message: string }) => e.message).join(" ")).toMatch(/nothing starts this stack's stream/);
  });

  test("after the user changes and saves the design, the diff lists what to change", async () => {
    const before = (await call("get_design_spec", { name: "e33" })).data.design.hash;
    await save("e33", E.updateCell(buildPreset("e33m_plain_30m"), "main.L0.ffn", { hidden: 1600 } as never));
    const { data, isError } = await call("diff_design", { name: "e33", since_hash: before });
    expect(isError).toBeFalsy();
    expect(data.from_hash).toBe(before);
    expect(data.to_hash).not.toBe(before);
    expect(data.changes).toEqual([expect.objectContaining({ change: "changed", what: "block", id: "main.L0.ffn", field: "hidden", from: 1152, to: 1600 })]);
    expect(data.parameters.to - data.parameters.from).toBe(3 * 576 * (1600 - 1152));
  });

  test("an unknown design or version is a clear error, not a crash", async () => {
    const noDesign = await call("get_design_spec", { name: "nope" });
    expect(noDesign.isError).toBe(true);
    expect(JSON.stringify(noDesign.content)).toMatch(/Saved designs: .*e33/);
    const badName = await call("validate_design", { name: "../etc/passwd" });
    expect(badName.isError).toBe(true);
    const noVersion = await call("diff_design", { name: "e33", since_hash: "nnb1-0000000000000000000000000000" });
    expect(noVersion.isError).toBe(true);
    expect(JSON.stringify(noVersion.content)).toMatch(/Known versions: nnb1-/);
  });

  test("the implement_design prompt walks through spec, checks and keeping the hash", async () => {
    const fresh = await client.getPrompt({ name: "implement_design", arguments: { name: "e33" } });
    const text = (fresh.messages[0].content as { text: string }).text;
    expect(text).toMatch(/get_design_spec/);
    expect(text).toMatch(/design\.hash/);
    const update = await client.getPrompt({ name: "implement_design", arguments: { name: "e33", since_hash: "nnb1-abc" } });
    expect((update.messages[0].content as { text: string }).text).toMatch(/diff_design/);
  });

  test("the guide resource is the agent guide", async () => {
    const r = await client.readResource({ uri: "nnb://guide" });
    expect((r.contents[0] as { text: string }).text).toMatch(/get_design_spec/);
  });
});

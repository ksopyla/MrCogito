// Starts the NN Builder MCP server over stdio (for Claude Code, Cursor, Codex, …).
// Builds it first when the sources are newer than the build. Prints nothing on stdout: stdio carries the protocol.
//   node scripts/mcp.mjs [--designs <folder>]
import { existsSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const out = path.join(root, "mcp", "dist", "server.mjs");

function newest(dir) {
  let t = 0;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) t = Math.max(t, newest(p));
    else if (/\.(ts|tsx|json)$/.test(e.name)) t = Math.max(t, statSync(p).mtimeMs);
  }
  return t;
}

const sources = Math.max(newest(path.join(root, "src", "engine")), newest(path.join(root, "src", "ui")), statSync(path.join(root, "mcp", "server.ts")).mtimeMs);
if (!existsSync(out) || statSync(out).mtimeMs < sources) {
  const { build } = await import("vite");
  await build({ configFile: path.join(root, "vite.mcp.config.ts"), root, logLevel: "silent" });
}
await import(pathToFileURL(out).href);

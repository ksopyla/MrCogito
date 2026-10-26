// Save / load designs as JSON files in ./designs, so a saved graph is a plain file in the repo that a later
// code-generation request can read. Dev-server middleware only; nothing here runs in the browser.
// Each save also keeps a copy under designs/.history/<name>/<hash>.json (the hash the app put in `derived`), so the
// MCP server can say what changed. No engine imports here: this file loads with the Vite config.
import { promises as fs } from "node:fs";
import path from "node:path";
import type { Plugin } from "vite";

const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;

export function designsApi(dir = path.resolve(import.meta.dirname, "..", "designs")): Plugin {
  return {
    name: "nn-builder-designs-api",
    configureServer(server) {
      server.middlewares.use("/api/designs", async (req, res) => {
        const send = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(body));
        };
        try {
          await fs.mkdir(dir, { recursive: true });
          let name: string;
          try {
            name = decodeURIComponent((req.url ?? "/").replace(/^\/+|\?.*$/g, "")).replace(/\.json$/, "");
          } catch {
            return send(400, { error: "bad design name" });
          }
          if (!name) {
            if (req.method !== "GET") return send(405, { error: "method not allowed" });
            const files = (await fs.readdir(dir)).filter((f) => f.endsWith(".json") && NAME.test(f.replace(/\.json$/, "")));
            const list = await Promise.all(
              files.map(async (f) => ({ name: f.replace(/\.json$/, ""), modified: (await fs.stat(path.join(dir, f))).mtime.toISOString() })),
            );
            return send(200, list.sort((a, b) => b.modified.localeCompare(a.modified)));
          }
          if (!NAME.test(name)) return send(400, { error: `bad design name: ${name}` });
          const file = path.join(dir, `${name}.json`);
          if (req.method === "GET") return send(200, JSON.parse(await fs.readFile(file, "utf8")));
          if (req.method === "PUT") {
            const chunks: Buffer[] = [];
            for await (const c of req) chunks.push(c as Buffer);
            let graph: { schema?: string; derived?: { hash?: unknown } };
            try {
              graph = JSON.parse(Buffer.concat(chunks).toString("utf8"));
            } catch {
              return send(400, { error: "the body is not JSON" });
            }
            if (graph?.schema !== "nn-builder/graph") return send(400, { error: "not an nn-builder graph" });
            const text = JSON.stringify(graph, null, 2) + "\n";
            await fs.writeFile(file, text);
            const hash = typeof graph.derived?.hash === "string" && /^nnb1-[0-9a-f]{28}$/.test(graph.derived.hash) ? graph.derived.hash : undefined;
            if (hash) {
              await fs.mkdir(path.join(dir, ".history", name), { recursive: true });
              await fs.writeFile(path.join(dir, ".history", name, `${hash}.json`), text);
            }
            return send(200, { saved: path.relative(path.resolve(dir, ".."), file), hash });
          }
          return send(405, { error: "method not allowed" });
        } catch (e) {
          const err = e as NodeJS.ErrnoException;
          send(err.code === "ENOENT" ? 404 : 500, { error: err.message });
        }
      });
    },
  };
}

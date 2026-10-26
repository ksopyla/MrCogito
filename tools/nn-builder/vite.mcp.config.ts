// Builds the MCP server (mcp/server.ts + the engine) into one Node module: mcp/dist/server.mjs.
// Dependencies (the MCP SDK, zod) stay external and load from node_modules.
import { defineConfig } from "vite";

export default defineConfig({
  logLevel: "warn",
  build: {
    ssr: "mcp/server.ts",
    outDir: "mcp/dist",
    emptyOutDir: true,
    target: "node20",
    minify: false,
    rollupOptions: { output: { format: "es", entryFileNames: "server.mjs" } },
  },
});

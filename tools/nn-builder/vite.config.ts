/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { designsApi } from "./server/designs-api";

export default defineConfig({
  plugins: [react(), designsApi()],
  server: { port: 5178, strictPort: false, open: false },
  test: { include: ["tests/**/*.test.ts"] },
});

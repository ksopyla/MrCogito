// `npm start`: install the dependencies the first time, then run the app and open it in the browser.
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const npm = process.platform === "win32" ? "npm.cmd" : "npm";

if (!existsSync(path.join(root, "node_modules", ".bin", "vite"))) {
  console.log("First run: installing dependencies…");
  const r = spawnSync(npm, ["install", "--no-audit", "--no-fund"], { cwd: root, stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const vite = spawn(path.join(root, "node_modules", ".bin", "vite"), ["--open", ...process.argv.slice(2)], { cwd: root, stdio: "inherit" });
vite.on("exit", (code) => process.exit(code ?? 0));

// Prints the numbers the calibration constants in src/engine/hardware.ts are fitted from.
// Run from tools/nn-builder: node scripts/calibrate.mjs
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { analyze } = await server.ssrLoadModule("/src/engine/analyze.ts");
const { buildPreset } = await server.ssrLoadModule("/src/engine/presets.ts");
const { GiB } = await server.ssrLoadModule("/src/engine/hardware.ts");

const peak = 71e12;
const at = (id, micro = 0) => {
  const g = buildPreset(id);
  Object.assign(g.run, { gpu: "rtx3090", row_length: 4096, micro_batch: micro, irregular_ratio: 0, mfu: 0 });
  return analyze(g);
};
const d = at("dense_plain_30m").flops;
const mfu = (60_000 * d.trainPerToken) / peak;
const e = at("e33m_plain_30m").flops;
const reg = e.trainPerToken - e.irregularTrainPerToken;
const rho = e.irregularTrainPerToken / ((peak * mfu) / 20_000 - reg);
console.log({ denseTrain: d.trainPerToken, regularMfu3090: mfu.toFixed(5), e33Train: e.trainPerToken, e33Irregular: e.irregularTrainPerToken, irregularRatio: rho.toFixed(5) });
for (const [id, b] of [["dense_plain_30m", 8], ["dense_plain_30m", 16], ["e33m_plain_30m", 8], ["e33m_plain_30m", 4], ["dense_plain_100m", 4], ["dense_plain_100m", 8], ["e33m_plain_100m", 4], ["e33m_plain_100m", 2]]) {
  const m = at(id, b).memory;
  const f = (x) => (x / GiB).toFixed(2);
  console.log(id.padEnd(17), "rows", String(b).padStart(2), "total", f(m.total), "GiB | activations", f(m.activations), "| states", f(m.weights + m.grads + m.optimizer + m.castCopies), "| per row", (m.actBytesPerRow / 2 ** 20).toFixed(0), "MiB | max rows", at(id).memory.maxMicroBatch);
}
await server.close();

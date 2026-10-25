// The estimate panels: compute & memory, the training plan (GPUs and time), and cost against row length.
import { useMemo } from "react";
import { analyze, type Analysis } from "../engine/analyze";
import * as E from "../engine/edit";
import { CAL, GPUS, GiB, MEMORY_CAL, OPTIMIZERS, PRECISIONS, gpuById } from "../engine/hardware";
import { TOKEN_RULES, ruleSpec, ruleTokens } from "../engine/tokens";
import type { Graph, RunSetup } from "../engine/types";
import { LineChart, StackBar, compact } from "./Charts";
import { NumberField, Select, Toggle } from "./controls";
import { cellTitle } from "./describe";
import type { Sel } from "./Frame";
import * as F from "./format";

interface Props {
  graph: Graph;
  analysis: Analysis;
  apply: (fn: (g: Graph) => Graph, key?: string) => void;
  setSel: (s: Sel) => void;
}

const memSegments = (m: Analysis["memory"]) => {
  const fr = (m.total - m.context) / Math.max(1, m.weights + m.master + m.grads + m.optimizer + m.activations + m.castCopies + m.workspace);
  return [
    { name: "Weights", value: (m.weights + m.master) * fr, color: "--series-1" },
    { name: "Gradients", value: m.grads * fr, color: "--series-2" },
    { name: "Optimizer", value: m.optimizer * fr, color: "--series-3" },
    { name: "Activations", value: m.activations * fr, color: "--series-4" },
    { name: "Other", value: (m.castCopies + m.workspace) * fr + m.context, color: "--series-5" },
  ];
};

export function SummaryStrip({ analysis: a, graph: g, onOpen }: { analysis: Analysis; graph: Graph; onOpen: (tab: string) => void }) {
  const gpu = gpuById(g.run.gpu);
  const m = a.memory;
  return (
    <div className="summary">
      <button className="stat" onClick={() => onOpen("cost")} title={`${F.exact(a.params.total)} parameters`}>
        <span className="k">Parameters</span>
        <span className="v">{F.params(a.params.total)}</span>
      </button>
      <button className="stat" onClick={() => onOpen("cost")} title="forward + backward per token">
        <span className="k">Training compute</span>
        <span className="v">{F.flops(a.flops.trainPerToken)}<small> / token</small></span>
      </button>
      <button className="stat" onClick={() => onOpen("cost")}>
        <span className="k">Memory per GPU · {m.microBatch} row{m.microBatch > 1 ? "s" : ""} × {F.len(a.S)}</span>
        <span className={`v ${m.fits ? "" : "bad"}`}>
          {F.bytes(m.total)} <small>of {gpu.memGiB} GB {m.fits ? "✓" : "✗ too big"}</small>
        </span>
      </button>
      <button className="stat" onClick={() => onOpen("train")}>
        <span className="k">Time for {F.tokens(a.tokens.budget)} tokens</span>
        <span className="v">{F.hours(a.time.hours)} <small>on {g.run.gpu_count} × {gpu.label.split(" ·")[0]}</small></span>
      </button>
      <button className="stat" onClick={() => onOpen("train")}>
        <span className="k">Speed</span>
        <span className="v">{F.rate(a.time.tokensPerSecondPerGpu)}<small> tokens/s per GPU</small></span>
      </button>
      {a.issues.length > 0 && (
        <button className="stat warnstat" onClick={() => onOpen("inspect")} title={a.issues.map((i) => i.msg).join("\n")}>
          <span className="k">Warnings</span>
          <span className="v">⚠ {a.issues.length}</span>
        </button>
      )}
    </div>
  );
}

export function CostPanel({ graph: g, analysis: a, setSel }: Props) {
  const f = a.flops;
  const m = a.memory;
  const train = Math.max(1, f.fwdPerToken + f.bwdPerToken);
  return (
    <div className="panel">
      <h3>Compute</h3>
      <dl className="costs">
        <div><dt>Forward / token</dt><dd>{F.flops(f.fwdPerToken)}</dd></div>
        <div><dt>Backward / token</dt><dd>{F.flops(f.bwdPerToken)} <span className="muted">({(f.bwdPerToken / Math.max(1, f.fwdPerToken)).toFixed(2)} × forward)</span></dd></div>
        {f.recomputePerToken > 0 && <div><dt>Recompute (checkpointing)</dt><dd>{F.flops(f.recomputePerToken)}</dd></div>}
        <div><dt>Training / token</dt><dd><strong>{F.flops(f.trainPerToken)}</strong></dd></div>
        <div><dt>Attention reads (share of forward)</dt><dd>{((100 * f.keysFwdPerToken) / Math.max(1, f.fwdPerToken)).toFixed(0)} %</dd></div>
        <div><dt>Per optimizer step ({g.run.rows_per_step} × {F.len(a.S)})</dt><dd>{F.flops(f.perStep)}</dd></div>
        <div><dt>Whole run ({F.tokens(a.tokens.budget)} tokens)</dt><dd>{F.flops(f.perRun)}</dd></div>
      </dl>
      <h3>GPU memory, per GPU</h3>
      <StackBar segments={memSegments(m)} capacity={m.capacity} format={F.bytes} />
      <p className="small">
        <strong className={m.fits ? "good" : "bad"}>{F.bytes(m.total)} of {F.bytes(m.capacity)} {m.fits ? "fits" : "does not fit"}</strong> at {m.microBatch} row{m.microBatch > 1 ? "s" : ""} per GPU.
        {" "}The largest micro-batch that fits: <strong>{m.maxMicroBatch || "none"}</strong>.
      </p>
      <p className="small muted">
        Saved activations per row: {F.bytes(m.actBytesPerRowNoCkpt)} without checkpointing, {F.bytes(m.actBytesPerRowCkpt)} with it ({g.run.checkpointing ? "on" : "off"} now).
      </p>
      <h3>Where it goes</h3>
      <table className="breakdown">
        <thead>
          <tr><th>Part</th><th className="num">Params</th><th>Training compute</th><th className="num">Saved / row</th></tr>
        </thead>
        <tbody>
          {a.stacks.map((s) => [
            <tr key={s.id} className="stack-row"><td colSpan={4}>{s.name} <span className="muted">· {F.params(s.params)} · {F.flops(s.fwdPerToken)} forward/token</span></td></tr>,
            ...s.rows.flatMap((r) => r.cells.map((c) => {
              const cell = E.findCell(g, c.id)?.cell;
              const share = (c.fwdPerToken + c.bwdPerToken) / train;
              return (
                <tr key={c.id} onClick={() => setSel({ kind: "cell", id: c.id })}>
                  <td>{cell ? cell.label ?? cellTitle(cell) : c.type}{c.execs > 1 && <span className="badge">×{c.execs}</span>}</td>
                  <td className="num">{F.params(c.params)}</td>
                  <td><span className="sharebar" style={{ width: `${Math.max(1, share * 100)}%` }} /> <span className="muted">{(share * 100).toFixed(share < 0.1 ? 1 : 0)} %</span></td>
                  <td className="num">{F.bytes(c.actBytesPerRow)}</td>
                </tr>
              );
            })),
            ...s.groups.filter((x) => x.params > 0 || x.fwdPerToken > 0).map((x) => (
              <tr key={x.id} onClick={() => setSel({ kind: "group", id: x.id })}>
                <td>{x.kind} extras</td>
                <td className="num">{F.params(x.params)}</td>
                <td><span className="sharebar" style={{ width: `${Math.max(1, (2 * x.fwdPerToken * 100) / train)}%` }} /></td>
                <td className="num">{F.bytes(x.actBytesPerRow)}</td>
              </tr>
            )),
          ])}
        </tbody>
      </table>
      <p className="small muted">
        FLOPs count every projection, attention score and value read (2 per multiply-add); norms and activations are not counted. Training = forward + backward (2 × forward for trained weights, 0 for no-grad parts).
      </p>
    </div>
  );
}

function TokensSection({ graph: g, analysis: a, set }: { graph: Graph; analysis: Analysis; set: (patch: Partial<RunSetup>, key?: string) => void }) {
  const rule = g.run.token_rule ?? "custom";
  const n = a.params.total;
  const spec = ruleSpec(rule);
  const t = a.tokens;
  return (
    <>
      <h3>Training tokens</h3>
      <div className="tok-picker" role="radiogroup" aria-label="Training-token budget">
        {TOKEN_RULES.map((r) => (
          <button key={r.id} role="radio" aria-checked={rule === r.id} className={`tok-btn ${rule === r.id ? "on" : ""}`} title={`${r.label}. ${r.source}.`} onClick={() => set({ token_rule: r.id })}>
            <span className="tok-name">{r.short}</span>
            <span className="tok-val">{F.tokens(ruleTokens(r.id, n))}</span>
          </button>
        ))}
        <button role="radio" aria-checked={rule === "custom"} className={`tok-btn ${rule === "custom" ? "on" : ""}`} title="type your own number below" onClick={() => set({ token_rule: "custom", token_budget: t.budget })}>
          <span className="tok-name">Custom</span>
          <span className="tok-val">{rule === "custom" ? F.tokens(t.budget) : "your number"}</span>
        </button>
      </div>
      <NumberField label="Training tokens" value={t.budget} min={1e6} max={1e14} log step={1e6} int help="typing a number switches to Custom" onChange={(v) => set({ token_rule: "custom", token_budget: v }, "run:token_budget")} />
      <p className="small">
        <strong>{F.tokens(t.budget)}</strong> tokens = {t.perParam >= 100 ? F.exact(t.perParam) : t.perParam.toFixed(1)} per parameter ·{" "}
        {t.chinchillaMultiple >= 10 ? Math.round(t.chinchillaMultiple) : t.chinchillaMultiple.toFixed(2)}× Chinchilla-optimal · {F.flops(a.flops.perRun)} of training compute
      </p>
      <p className="small muted">{spec ? `${spec.label}; it follows the model's size (${spec.source}).` : "Custom: your number, kept when the model changes."}</p>
    </>
  );
}

export function TrainingPanel({ graph: g, analysis: a, apply }: Props) {
  const run = g.run;
  const set = (patch: Partial<RunSetup>, key?: string) => apply((x) => E.updateRun(x, patch), key ?? `run:${Object.keys(patch).join(",")}`);
  const gpu = gpuById(run.gpu);
  const t = a.time;
  return (
    <div className="panel">
      <h3>Hardware</h3>
      <div className="gpu-picker" role="radiogroup" aria-label="GPU model">
        {GPUS.filter((x) => !x.hidden || x.id === run.gpu).map((x) => (
          <button key={x.id} role="radio" aria-checked={run.gpu === x.id} className={`gpu-btn ${run.gpu === x.id ? "on" : ""}`} title={`${x.label} · ${x.peakTflops} TFLOPS bf16 dense`} onClick={() => set({ gpu: x.id })}>
            <span className="gpu-name">{x.short}</span>
            <span className="gpu-mem">{x.memGiB} GB</span>
          </button>
        ))}
      </div>
      <NumberField label="Number of GPUs" value={run.gpu_count} min={1} max={1024} log int onChange={(v) => set({ gpu_count: v })} />
      <p className="small muted">
        {gpu.label} · {gpu.peakTflops} TFLOPS bf16 peak · {gpu.note}.
      </p>
      <TokensSection graph={g} analysis={a} set={set} />
      <h3>Run</h3>
      <NumberField label="Row length (tokens)" value={run.row_length} min={64} max={1048576} log step={64} int onChange={(v) => set({ row_length: v })} />
      <NumberField label="Rows per optimizer step" value={run.rows_per_step} min={1} max={4096} log int onChange={(v) => set({ rows_per_step: v })} />
      <NumberField label="Micro-batch per GPU (0 = largest that fits)" value={run.micro_batch} min={0} max={512} int onChange={(v) => set({ micro_batch: v })} />
      <Select label="Number format" value={run.precision} options={Object.entries(PRECISIONS).map(([k, v]) => [k, v.label]) as [RunSetup["precision"], string][]} onChange={(v) => set({ precision: v })} />
      <Select label="Optimizer" value={run.optimizer} options={Object.entries(OPTIMIZERS).map(([k, v]) => [k, v.label]) as [RunSetup["optimizer"], string][]} onChange={(v) => set({ optimizer: v })} />
      <NumberField label="Weight decay" value={run.weight_decay} min={0} max={1} step={0.01} help="carried for code generation; does not change the estimates" onChange={(v) => set({ weight_decay: v })} />
      <Select label="Shard states across GPUs" value={run.zero_stage} options={[[0, "no (plain data parallel)"], [1, "optimizer (ZeRO-1)"], [2, "+ gradients (ZeRO-2)"], [3, "+ weights (ZeRO-3 / FSDP)"]]} onChange={(v) => set({ zero_stage: v })} />
      <Toggle label="Gradient checkpointing (recompute each row in backward)" value={run.checkpointing} onChange={(v) => set({ checkpointing: v })} />
      <NumberField label="Finish within (hours, 0 = no deadline)" value={run.deadline_hours} min={0} max={10000} log int onChange={(v) => set({ deadline_hours: v })} />
      <details>
        <summary>Speed calibration</summary>
        <NumberField label={`Share of peak for plain matmuls (0 = default ${(gpu.mfu * 100).toFixed(1)} %)`} value={run.mfu} min={0} max={1} step={0.005} onChange={(v) => set({ mfu: v })} />
        <NumberField label={`Speed of masked / looped work vs plain (0 = default ${CAL.irregularRatio.toFixed(3)})`} value={run.irregular_ratio} min={0} max={1} step={0.005} onChange={(v) => set({ irregular_ratio: v })} />
        <NumberField label="Multi-GPU scaling efficiency" value={run.dp_efficiency} min={0.1} max={1} step={0.01} onChange={(v) => set({ dp_efficiency: v })} />
        <p className="small muted">
          Calibrated on your RTX 3090 runs at 4k rows: dense 30M ≈ 60k tokens/s, E33 30M ≈ 20k. Masked reads, page writers and loop feedback run at about a sixth of the speed of plain matmuls per FLOP. {gpu.note}. Treat new numbers as ±30 %.
        </p>
      </details>
      <h3>Plan</h3>
      <dl className="costs">
        <div><dt>Speed</dt><dd>{F.rate(t.tokensPerSecondPerGpu)} tokens/s per GPU · {F.rate(t.tokensPerSecond)} total</dd></div>
        <div><dt>Time for {F.tokens(a.tokens.budget)} tokens</dt><dd><strong>{F.hours(t.hours)}</strong> on {run.gpu_count} GPU{run.gpu_count > 1 ? "s" : ""} · {F.hours(t.gpuHours)} GPU-time</dd></div>
        <div><dt>Optimizer steps</dt><dd>{F.exact(t.steps)} of {run.rows_per_step} rows · {t.accumulation} accumulation step{t.accumulation > 1 ? "s" : ""} of {a.memory.microBatch} rows per GPU</dd></div>
        <div><dt>GPUs needed to fit</dt><dd>{Number.isFinite(a.gpus.minForMemory) ? a.gpus.minForMemory : run.zero_stage === 0 ? "does not fit one GPU: shard states or checkpoint" : "does not fit even sharded"}</dd></div>
        {run.deadline_hours > 0 && <div><dt>GPUs to finish in {run.deadline_hours} h</dt><dd><strong>{a.gpus.forDeadline}</strong></dd></div>}
        <div><dt>Plain matmul efficiency</dt><dd>{(t.mfu * 100).toFixed(1)} % of {gpu.peakTflops} TFLOPS</dd></div>
      </dl>
    </div>
  );
}

const LENGTHS = Array.from({ length: 11 }, (_, i) => 1024 * 2 ** i); // 1k → 1M

export function LengthPanel({ graph: g, analysis: a }: Props) {
  const rows = useMemo(() => LENGTHS.map((S) => ({ S, a: analyze(g, { S }) })), [g]);
  const gpu = gpuById(g.run.gpu);
  const states = a.memory.weights + a.memory.master + a.memory.grads + a.memory.optimizer + a.memory.castCopies;
  const room = Math.max(1, (a.memory.capacity - a.memory.context) / MEMORY_CAL.fragmentation - states);
  return (
    <div className="panel">
      <p className="small muted">How the same network behaves when each training row is longer. Pages, notes and attention windows grow with the row; weights do not.</p>
      <LineChart
        title="Forward compute per token"
        xs={LENGTHS}
        series={[
          { name: "weights (projections)", color: "--series-1", values: rows.map((r) => r.a.flops.fwdPerToken - r.a.flops.keysFwdPerToken) },
          { name: "attention reads", color: "--series-2", values: rows.map((r) => r.a.flops.keysFwdPerToken) },
        ]}
        yFormat={F.flops}
        yTick={compact("")}
        note="Training ≈ 3 × forward."
      />
      <LineChart
        title="Saved activations per row"
        xs={LENGTHS}
        series={[
          { name: "no checkpointing", color: "--series-1", values: rows.map((r) => (r.a.memory.actBytesPerRowNoCkpt * MEMORY_CAL.activationOverhead) / GiB) },
          { name: "checkpointing", color: "--series-2", values: rows.map((r) => (r.a.memory.actBytesPerRowCkpt * MEMORY_CAL.activationOverhead) / GiB) },
          { name: `room on one ${gpu.label.split(" ·")[0]}`, color: "--muted", values: rows.map(() => room / GiB), dashed: true },
        ]}
        yFormat={(v) => F.bytes(v * GiB)}
        yTick={(v) => (v >= 1000 ? `${+(v / 1024).toPrecision(3)} TB` : `${+v.toPrecision(3)} GB`)}
        note="Above the dashed line not even one row fits on one GPU."
      />
      <LineChart
        title={`Time for ${F.tokens(a.tokens.budget)} tokens on ${g.run.gpu_count} × ${gpu.label.split(" ·")[0]}`}
        xs={LENGTHS}
        series={[{ name: "hours", color: "--series-1", values: rows.map((r) => r.a.time.hours) }]}
        yFormat={F.hours}
      />
      <table className="breakdown">
        <thead>
          <tr><th>Row</th><th className="num">Train / token</th><th className="num">Saved / row</th><th className="num">Rows fit</th><th className="num">Tokens/s/GPU</th><th className="num">Time</th></tr>
        </thead>
        <tbody>
          {rows.map(({ S, a: r }) => (
            <tr key={S} className={S === a.S ? "current" : ""}>
              <td>{F.len(S)}</td>
              <td className="num">{F.flops(r.flops.trainPerToken)}</td>
              <td className="num">{F.bytes(r.memory.actBytesPerRow)}</td>
              <td className="num">{r.memory.maxMicroBatch || "0"}</td>
              <td className="num">{F.rate(r.time.tokensPerSecondPerGpu)}</td>
              <td className="num">{F.hours(r.time.hours)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

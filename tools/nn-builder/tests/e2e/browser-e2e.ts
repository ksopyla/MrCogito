// End-to-end test of the running app, driven in a real browser page.
// Run it in the page served by `npm start` (DevTools console, or a browser-automation tool):
//   const { run } = await import("/tests/e2e/browser-e2e.ts"); await run();
// It drives the interface with pointer events (drag and drop, wires, loops), clicks and keys, checks the saved
// graph after every step against the engine, saves / opens / exports a design, and returns a pass/fail report.
// It writes designs/e2e_test.json (delete it afterwards) and leaves the blank preset loaded.
import { analyze } from "../../src/engine/analyze";
import { parseGraph } from "../../src/engine/serialize";
import type { Graph } from "../../src/engine/types";

interface Step {
  step: string;
  ok: boolean;
  detail: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const $ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector<T>(sel);
const $$ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => [...root.querySelectorAll<T>(sel)];

function graph(): Graph {
  const res = parseGraph(localStorage.getItem("nn-builder.current") ?? "");
  if (!res.graph) throw new Error(`no saved graph: ${res.error}`);
  return res.graph;
}
const params = () => analyze(graph()).params.total;

function center(el: Element) {
  const b = el.getBoundingClientRect();
  return { x: b.left + b.width / 2, y: b.top + b.height / 2 };
}

function visible(el: Element) {
  el.scrollIntoView({ block: "center", inline: "nearest" });
}

function fire(type: string, x: number, y: number, init: PointerEventInit = {}) {
  const target = document.elementFromPoint(x, y) ?? document.body;
  target.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, isPrimary: true, pointerType: "mouse", ...init }));
}

/** Press on `from`, move in steps to `to` (an element's centre or a point), release. */
async function drag(from: Element, to: Element | { x: number; y: number }, init: PointerEventInit = {}) {
  const a = center(from);
  const b = to instanceof Element ? center(to) : to;
  from.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, cancelable: true, clientX: a.x, clientY: a.y, button: 0, buttons: 1, pointerId: 1, isPrimary: true, pointerType: "mouse", ...init }));
  for (let i = 1; i <= 10; i++) {
    fire("pointermove", a.x + ((b.x - a.x) * i) / 10, a.y + ((b.y - a.y) * i) / 10, init);
    await sleep(16);
  }
  fire("pointerup", b.x, b.y, init);
  await sleep(120);
}

async function click(el: Element, init: PointerEventInit = {}) {
  await drag(el, el, init); // press and release in place = a click for the slot/row/cell handlers
}

async function selectPreset(id: string) {
  const sel = $<HTMLSelectElement>("select.preset")!;
  Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!.call(sel, id);
  sel.dispatchEvent(new Event("change", { bubbles: true }));
  await sleep(250);
}

const row = (label: string) => $$(".stack")[0] && $$(".row").find((r) => r.querySelector(".row-label")?.textContent === label);
const cellIn = (r: Element, title: string) => $$(".cell", r).find((c) => c.querySelector(".cell-title")?.textContent?.startsWith(title));
const paletteItem = (label: string) => $$(".pal-item").find((p) => p.querySelector(".pal-label")?.textContent === label)!;
const tab = async (label: string) => {
  $$<HTMLButtonElement>(".tabs button").find((b) => b.textContent?.startsWith(label))!.click();
  await sleep(150);
};

async function setInput(input: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new FocusEvent("blur", { bubbles: false }));
  input.dispatchEvent(new FocusEvent("focusout", { bubbles: true }));
  await sleep(150);
}

export async function run(): Promise<{ passed: number; failed: number; steps: Step[] }> {
  const steps: Step[] = [];
  const check = (step: string, ok: boolean, detail = "") => steps.push({ step, ok, detail });
  const attempt = async (step: string, fn: () => Promise<void>) => {
    try {
      await fn();
    } catch (e) {
      check(step, false, `threw: ${(e as Error).message}`);
    }
  };
  const pane = $(".frame-scroll")!;

  await attempt("presets reproduce the trainer's counts", async () => {
    const want: [string, number][] = [["dense_plain_30m", 30_293_140], ["e33m_plain_30m", 29_946_402], ["dense_plain_100m", 100_966_686], ["gpt2_small", 124_439_808]];
    for (const [id, n] of want) {
      await selectPreset(id);
      check(`preset ${id} = ${n.toLocaleString("en-US")}`, params() === n, `got ${params()}`);
    }
    await selectPreset("e33m_plain_30m");
    check("E33 draws its wires (skips, table → pages, pages → notes, notes → read)", $$(".wire").length === 7, `${$$(".wire").length} wires`);
    check("summary strip shows the E33 count", $(".stat .v")?.textContent === "29.95M", $(".stat .v")?.textContent ?? "");
    await selectPreset("encdec");
    check("encoder–decoder draws six cross-attention wires", $$(".wire.source").length === 6, `${$$(".wire.source").length}`);
  });

  await selectPreset("blank");
  pane.scrollLeft = 0;
  const base = params();
  let p = base;

  await attempt("drag a layer from the palette between rows", async () => {
    visible(row("head")!);
    const gap = $$(".rowgap").find((g) => g.dataset.stack === "main" && g.dataset.index === "2")!;
    await drag(paletteItem("SwiGLU"), gap);
    const rows = $$(".row").map((r) => r.querySelector(".row-label")?.textContent);
    check("a SwiGLU row appears between L0 and head", rows.length === 4 && rows[3] === "head", rows.join(", "));
    check("its parameters are added exactly (512 + 3·512·1344)", params() - p === 512 + 3 * 512 * 1344, `Δ ${params() - p}`);
    p = params();
  });

  await attempt("drag a layer into a row's slot", async () => {
    const r = $$(".row")[2];
    await drag(paletteItem("Full attention"), $(".slot", r)!);
    check("the row now holds SwiGLU + Full attention", $$(".cell-title", $$(".row")[2]).length === 2);
    check("a 512-wide attention adds 4·512² + 512 + 128", params() - p === 4 * 512 * 512 + 512 + 128, `Δ ${params() - p}`);
    p = params();
  });

  await attempt("move a slot to another row", async () => {
    const from = cellIn($$(".row")[2], "Full attention")!;
    await drag(from, $(".slot", $$(".row")[1])!);
    check("L0 holds two attention slots, the new row one", $$(".cell", $$(".row")[1]).length === 3 && $$(".cell", $$(".row")[2]).length === 1);
    check("moving changes no parameter", params() === p);
  });

  await attempt("move a row by its handle", async () => {
    const handle = $(".row-handle", $$(".row")[2])!;
    const gap = $$(".rowgap").find((g) => g.dataset.stack === "main" && g.dataset.index === "1")!;
    await drag(handle, gap);
    const labels = $$(".row").map((r) => r.querySelector(".row-label")?.textContent);
    check("the moved row is now second", labels[1] !== "L0" && labels[2] === "L0", labels.join(", "));
  });

  await attempt("wire a skip from one row to a later row", async () => {
    const before = graph().edges.length;
    await drag($(".dot.out", $$(".row")[1])!, $(".row-meta", $$(".row")[3])!);
    const g = graph();
    check("a skip wire is added and drawn", g.edges.length === before + 1 && g.edges.at(-1)!.kind === "skip" && $$(".wire.skip").length === 1);
    check("its weight is one parameter", params() - p === 1, `Δ ${params() - p}`);
    p = params();
  });

  await attempt("drop a slot on the palette to remove it", async () => {
    const victim = cellIn($$(".row")[2], "Full attention")!;
    await drag(victim, $(".palette")!);
    check("the attention slot is gone", params() - p === -(4 * 512 * 512 + 512 + 128), `Δ ${params() - p}`);
    p = params();
  });

  await attempt("select a cell and edit it in the inspector", async () => {
    await click(cellIn($$(".row")[2], "Full attention")!);
    await tab("Inspect");
    const heads = $$<HTMLLabelElement>(".inspector .field").find((f) => f.textContent?.startsWith("Query heads"))!;
    await setInput($<HTMLInputElement>("input.num-input", heads)!, "4");
    const a = graph().stacks[0].rows[2].cells.find((c) => c.type === "attention");
    check("query heads set to 4", a?.type === "attention" && a.heads === 4, JSON.stringify(a && "heads" in a ? a.heads : null));
    check("q and output shrink to 4 × 64 (2·512·256 fewer each … net)", params() - p === -(2 * 512 * 512 - 2 * 512 * 256), `Δ ${params() - p}`);
    p = params();
  });

  await attempt("shift-select two rows and loop them", async () => {
    await click($(".row-handle", $$(".row")[1])!);
    await click($(".row-handle", $$(".row")[2])!, { shiftKey: true });
    const loopBtn = $$<HTMLButtonElement>(".inspector button").find((b) => b.textContent?.startsWith("↻ Loop"))!;
    const fwd = analyze(graph()).flops.fwdPerToken;
    loopBtn.click();
    await sleep(150);
    check("a loop bracket appears", $$(".group.loop").length === 1);
    check("a loop adds only its per-round vectors (4 × 512)", params() - p === 4 * 512, `Δ ${params() - p}`);
    check("looped rows run 4 times: more compute", analyze(graph()).flops.fwdPerToken > 2 * fwd);
    p = params();
    $$<HTMLButtonElement>(".group-tag button").find((b) => b.textContent === "+")!.click();
    await sleep(150);
    check("the + on the loop tag makes it 5 rounds", graph().groups[0].count === 5 && params() - p === 512);
    p = params();
  });

  await attempt("undo and redo", async () => {
    for (let i = 0; i < 9; i++) { // the nine edits since the blank preset was loaded
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "z", metaKey: true, bubbles: true }));
      await sleep(40);
    }
    check("undo returns to the blank model", params() === base, `${params()} vs ${base}`);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "z", metaKey: true, shiftKey: true, bubbles: true }));
    await sleep(100);
    check("redo replays the first step (the SwiGLU row)", params() - base === 512 + 3 * 512 * 1344, `Δ ${params() - base}`);
  });

  await attempt("delete key removes the selected slot", async () => {
    const before = params();
    await click(cellIn($$(".row")[2], "SwiGLU")!);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Delete", bubbles: true }));
    await sleep(120);
    check("the SwiGLU is removed", before - params() === 512 + 3 * 512 * 1344, `Δ ${params() - before}`);
  });

  await attempt("encoder–decoder by hand: add a stack and wire it into a cross-attention", async () => {
    $$<HTMLButtonElement>(".topbar button").find((b) => b.textContent === "+ Stack")!.click();
    await sleep(80);
    $$<HTMLButtonElement>(".menu button").find((b) => b.textContent?.startsWith("Encoder"))!.click();
    await sleep(150);
    check("an encoder stack is added", $$(".stack").length === 2);
    // put the encoder on the left so its output dot and the decoder sit side by side
    $(".stack-head", $$(".stack")[1])!.click(); // the header selects on a plain click
    await sleep(120);
    $$<HTMLButtonElement>(".inspector button").find((b) => b.textContent === "← Left")!.click();
    await sleep(150);
    const dec = $$(".stack")[1];
    await drag(paletteItem("Cross-attention"), $(".slot", $$(".row", dec)[1])!);
    pane.scrollLeft = 0;
    await sleep(80);
    const cross = $$(".cell", dec).find((c) => c.querySelector(".cell-title")?.textContent?.startsWith("Cross"))!;
    check("a cross-attention without a source warns", Boolean(cross.querySelector(".warn")));
    await drag($(".stack-head .dot.out", $$(".stack")[0])!, cross);
    const g = graph();
    check("the wire feeds the cross-attention", g.edges.some((e) => e.kind === "source" && e.from.stack === g.stacks[0].id) && $$(".wire.source").length === 1);
    check("its warning is gone", !$$(".cell", $$(".stack")[1]).find((c) => c.querySelector(".cell-title")?.textContent?.startsWith("Cross"))!.querySelector(".warn"));
  });

  await attempt("estimate tabs", async () => {
    await tab("Compute");
    check("compute & memory shows the memory bar and the per-cell table", Boolean($(".stackbar")) && $$(".breakdown tbody tr").length > 5);
    await tab("GPUs");
    $$<HTMLButtonElement>(".chips button").find((b) => b.textContent?.startsWith("Polonez"))!.click();
    await sleep(150);
    check("the Polonez preset sets 4 × RTX 3090", graph().run.gpu_count === 4 && graph().run.gpu === "rtx3090");
    await tab("Row length");
    const charts = $$("figure.chart");
    check("row length shows three charts and a table of 11 lengths", charts.length === 3 && $$(".breakdown tbody tr").length === 11);
    const svg = $("svg", charts[0])!;
    const b = svg.getBoundingClientRect();
    svg.dispatchEvent(new PointerEvent("pointermove", { bubbles: true, clientX: b.left + b.width * 0.7, clientY: b.top + b.height / 2 }));
    await sleep(100);
    check("hovering a chart shows its tooltip", Boolean($(".tooltip", charts[0])));
  });

  await attempt("save, open and export", async () => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await tab("Inspect");
    await setInput($$<HTMLInputElement>(".inspector .text-input")[0], "e2e test");
    check("the design is renamed", graph().name === "e2e test");
    $$<HTMLButtonElement>(".topbar button").find((b) => b.textContent === "Save")!.click();
    await sleep(500);
    const list = (await (await fetch("/api/designs")).json()) as { name: string }[];
    check("Save writes designs/e2e_test.json", list.some((d) => d.name === "e2e_test"));
    const saved = parseGraph(await (await fetch("/api/designs/e2e_test")).text());
    check("the saved file reloads with the same count", Boolean(saved.graph) && analyze(saved.graph!).params.total === params());
    await selectPreset("dense_plain_30m");
    $$<HTMLButtonElement>(".topbar button").find((b) => b.textContent === "Open…")!.click();
    await sleep(400);
    $$<HTMLButtonElement>(".menu button").find((b) => b.textContent?.startsWith("e2e_test"))!.click();
    await sleep(400);
    check("Open… loads it back", graph().name === "e2e test");
    // capture the export instead of downloading it
    let captured: { name: string; text: string } | null = null;
    const orig = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function (this: HTMLAnchorElement) {
      const name = this.download;
      void fetch(this.href).then((r) => r.text()).then((text) => (captured = { name, text }));
    };
    try {
      $$<HTMLButtonElement>(".topbar button").find((b) => b.textContent === "Export")!.click();
      await sleep(300);
    } finally {
      HTMLAnchorElement.prototype.click = orig;
    }
    const exp = captured as { name: string; text: string } | null;
    const parsed = exp ? JSON.parse(exp.text) : null;
    check("Export gives e2e_test.json with the derived counts", exp?.name === "e2e_test.json" && parsed?.derived?.params_total === params(), exp?.name ?? "nothing captured");
  });

  await selectPreset("blank");
  const passed = steps.filter((s) => s.ok).length;
  return { passed, failed: steps.length - passed, steps };
}

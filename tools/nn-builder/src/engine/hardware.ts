// GPUs, number formats and optimizers: the bytes and FLOP rates the estimates are built from.
import type { OptimizerKind, Precision } from "./types";

export const GiB = 2 ** 30;

// Speed calibration on the project's own RTX 3090 runs (4,096-token rows, bf16, AdamW, 19–24 Oct 2026):
// dense_plain 30M ran at ≈ 60k tokens/s, e33m_plain 30M at ≈ 20k. Plain matmul work runs at REGULAR_MFU of the
// card's nominal peak; masked, gathered or looped work (notebook reads, page writers, the loop's feedback) runs
// IRREGULAR_RATIO times as fast per FLOP. Both are fitted by scripts/calibrate.mjs and checked by tests/memory_speed.test.ts.
export const CAL = {
  regularMfu3090: 0.24395,
  irregularRatio: 0.17502,
};

export interface GpuSpec {
  id: string;
  label: string;
  memGiB: number;
  peakTflops: number; // nominal dense bf16 tensor-core peak
  contextGiB: number; // CUDA context + cuBLAS/allocator reserve
  mfu: number; // default share of the peak that plain matmul work reaches
  note: string;
}

export const GPUS: GpuSpec[] = [
  { id: "rtx3090", label: "RTX 3090 · 24 GB", memGiB: 24, peakTflops: 71, contextGiB: 0.6, mfu: CAL.regularMfu3090,
    note: "speed calibrated on the project's measured runs" },
  { id: "rtx4090", label: "RTX 4090 · 24 GB", memGiB: 24, peakTflops: 165, contextGiB: 0.6, mfu: CAL.regularMfu3090,
    note: "same efficiency as the 3090 assumed" },
  { id: "a100_80", label: "A100 · 80 GB", memGiB: 80, peakTflops: 312, contextGiB: 0.8, mfu: 0.3,
    note: "efficiency is a guess for a ~500M model; small models reach less" },
  { id: "h100_80", label: "H100 SXM · 80 GB", memGiB: 80, peakTflops: 989, contextGiB: 0.8, mfu: 0.3,
    note: "efficiency is a guess for a ~500M model; small models reach less (p5.48xlarge = 8 of these)" },
];

export const gpuById = (id: string): GpuSpec => GPUS.find((g) => g.id === id) ?? GPUS[0];

export interface PrecisionSpec {
  label: string;
  weight: number; // bytes per parameter, kept for the whole run
  grad: number;
  master: number; // fp32 master copy when weights are low precision
  cast: number; // bf16 copies autocast keeps of the weights during forward/backward
  act: number; // bytes per saved activation element
  resid: number; // bytes per element of the residual stream and of norm inputs
  optScale: number; // optimizer state width relative to fp32
}

export const PRECISIONS: Record<Precision, PrecisionSpec> = {
  amp_bf16: { label: "bf16 autocast, fp32 weights (PyTorch AMP)", weight: 4, grad: 4, master: 0, cast: 2, act: 2, resid: 4, optScale: 1 },
  bf16_master: { label: "bf16 weights + fp32 master copy", weight: 2, grad: 2, master: 4, cast: 0, act: 2, resid: 2, optScale: 1 },
  pure_bf16: { label: "pure bf16 (weights, grads, optimizer)", weight: 2, grad: 2, master: 0, cast: 0, act: 2, resid: 2, optScale: 0.5 },
  fp32: { label: "fp32 everywhere", weight: 4, grad: 4, master: 0, cast: 0, act: 4, resid: 4, optScale: 1 },
};

export const OPTIMIZERS: Record<OptimizerKind, { label: string; bytes: number }> = {
  adamw: { label: "AdamW (2 fp32 states)", bytes: 8 },
  adamw_8bit: { label: "AdamW 8-bit states", bytes: 2 },
  sgd_momentum: { label: "SGD + momentum", bytes: 4 },
  muon: { label: "Muon (1 momentum state)", bytes: 4 },
  adafactor: { label: "Adafactor (factored, ≈ 0)", bytes: 0 },
};

// Saved activations are counted tensor by tensor; real PyTorch keeps some extra copies (dtype casts, views,
// allocator rounding). These two factors are fitted to what the 24 GB cards did (tests/memory_speed.test.ts).
export const MEMORY_CAL = {
  fragmentation: 1.1,
  activationOverhead: 1.25,
};

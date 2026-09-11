# Small-LM (100M–1B) training recipes and reference numbers

What a from-scratch ~600M model should reach, and the free wins the 2025–26 recipes agree on.
Collected 2026-09-11 (research-scout sweep; figures not re-read directly are the scout's and
marked as such where uncertain). Used to set credible targets for the E18 main run and its
successors. Related: [`datasets_review_for_training.md`](datasets_review_for_training.md),
[`frontier_open_models_architecture.md`](frontier_open_models_architecture.md),
[`long_context_architectures_training.md`](long_context_architectures_training.md).

---

## Reference points (base models unless noted)

| Model | Params | Tokens | Ctx | HellaSwag | ARC-C | MMLU | GSM8K | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| modded-nanogpt (GPT-2 speedrun) | 124M | ~10B | 1K | — (target: FineWeb val CE 3.28) | — | — | — | https://github.com/KellerJordan/modded-nanogpt |
| nanochat d26 | ~124M | ~10B | 2K | — | — | — | — | https://github.com/karpathy/nanochat |
| Gemma 3 PT 270M | 270M | 6T | 128k | 40.9 | 29.0 | — | — | https://ai.google.dev/gemma/docs/core/model_card_3 |
| SmolLM2-360M | 360M | 4T | 2k→8k | 54.5 | ~47 (ARC avg) | 35.8 (cloze) | 3.2 | https://huggingface.co/HuggingFaceTB/SmolLM2-360M |
| Qwen3-0.6B-Base | 600M | 36T | 32k | 41.0 | — | 52.3 | 48.3 | https://huggingface.co/Qwen/Qwen3-0.6B-Base (Swallow harness) |
| Llama 3.2 1B | 1.23B | ≤9T | 128k | 41.2 | 32.8 | 32.2 | — | https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md |
| Falcon-H1-0.5B (instruct) | 500M | — | — | 51.9 | 37.8 | 53.4 | 68.4 | https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct |
| TinyLlama 1.1B | 1.1B | 2T | 2k | 61.5 | 32.7 | 26.6 | — | https://arxiv.org/abs/2401.02385 |
| SmolLM2-1.7B | 1.7B | 11T | 2k→8k | 68.7 | ~52 | 19.4 (MMLU-Pro) | 31.1 | https://arxiv.org/abs/2502.02737 |
| LFM2-1.2B (instruct) | 1.2B | 10–12T | 32k | — | — | 55.2 | 58.3 | https://arxiv.org/abs/2511.23404 |
| MobileLLM-R1-950M | 950M | <5T | — | — | — | — | 67.5 (post-trained) | https://arxiv.org/abs/2509.24945 |
| SmolLM3-3B | 3B | 11.2T | 64k→128k | — | — | — | — | https://huggingface.co/blog/smollm3 |
| Nemotron 3 Nano | 31.6B (3.2B active) | 25T | 1M | — | — | 78.3 (MMLU-Pro) | — | https://arxiv.org/abs/2512.20848 |

**Credible band for ~600M at 300B–1T tokens (interpolated, not published):** HellaSwag 58–65,
ARC-C 35–48, MMLU 28–40, GSM8K 5–30; math/code share of the mix dominates GSM8K (SmolLM2-360M
at 4T: 3.2; Qwen3-0.6B at 36T: ~48). E18's M1 target ("within 2 points of SmolLM2-360M") is
conservative and fine for a 300B-token stage-1.

## Recipes

### Speedrun / efficiency (modded-nanogpt, nanochat)
- **modded-nanogpt** record stack (8×H100, GPT-2 124M → 3.28 val CE): RoPE, QK-norm, ReLU²,
  **value embeddings**, **U-net skips**, FP8 head, logit softcap, sparse attention gates,
  **short→long SWA warmup**, FlexAttention, **Muon → NorMuon / Polar Express** orthogonalisation,
  cautious weight decay. Wall-clock ~2.17 min (Dec 2025); step record ~2.7–2.9k (2026).
  https://github.com/KellerJordan/modded-nanogpt · https://arxiv.org/abs/2510.05491 · https://arxiv.org/pdf/2505.16932
- **nanochat**: one `--depth` dial; ~10.5 tokens/param; batch ∝ D^0.383; LR ∝ √(B/B_ref);
  GPT-2 grade for ~$50–100 on 8×H100. https://github.com/karpathy/nanochat · https://karpathy-nanochat.mintlify.app/training/scaling-laws
- Our E18 stack already carries value embeddings, U-net skips, QK-norm, softcap, Muon; missing:
  NorMuon/Polar Express, FP8 head, short→long SWA warmup.

### Published small-model recipes
- **SmolLM3 (3B / 11.2T):** GQA-4, **NoPE every 4th layer**, WSD (2k warmup, 10% linear decay),
  AdamW 2e-4 / wd 0.1, 2.36M-token batch, 4k ctx → +100B long stage (4k→32k→64k, RoPE θ 1.5M→5M;
  YaRN to 128k). https://huggingface.co/blog/smollm3
- **SmolLM2 (1.7B / 11T):** 4-stage mix (FineWeb-Edu + DCLM → code → Stack-Edu/math → 14% math
  in decay). https://arxiv.org/abs/2502.02737
- **OLMo-3:** Dolma 3 (76% web, 14% science PDFs, 7% code, 3% math) → 100B mid-training → 50B
  long-context stage. https://arxiv.org/pdf/2512.13961
- **Qwen3 small:** 36T in 3 stages (30T general, ~5T reasoning, hundreds of B at 32k). https://arxiv.org/abs/2505.09388
- **MobileLLM-R1-950M:** ~2T high-quality tokens + heavy math/code post-training. https://arxiv.org/abs/2509.24945

### Hashed n-gram / lookup embeddings
- **Over-Tokenized Transformer (OTT):** 400M with 128× input vocab ≈ 1B-equivalent loss;
  log-linear vocab→loss; sum of 1–3-gram hashes; bucket count coprime to V. https://arxiv.org/abs/2501.16975
- **Engram (DeepSeek, Jan 2026):** 27B iso-FLOP MoE, MMLU +3.4, ARC-C +3.7, NIAH 84→97; shipped at
  196B in V4.1-Flash. https://arxiv.org/abs/2601.07372
- **SCONE:** 1B + offloaded f-gram table beats a 1.9B baseline at ~48% lower inference FLOPs. https://arxiv.org/abs/2502.01637
- **Qwen3.8-Flash-Next:** loss falls monotonically with n-gram vocab while downstream accuracy
  *saturates* — size tables on benchmarks, not CE. https://arxiv.org/html/2608.30320

### Optimiser, schedule, auxiliary losses
- **Muon at ~600M:** Moonlight — ~2× compute-efficiency vs AdamW; weight decay + RMS-matched update
  scale 0.2·√max(A,B) lets AdamW-tuned LRs be reused. https://arxiv.org/abs/2502.16982
  **μP for Muon:** LR-μP + wd ∝ 1/width gives ~1.4× over AdamW at 190M–1.4B; naive 1/width Muon
  scaling fails. https://papers.nips.cc/paper_files/paper/2025/hash/bdcd9f6327db5877dee502cdec183159-Abstract-Conference.html
- **MTP below 1B:** original MTP *hurts* on standard NLP < 1–3B; a forward-MTP curriculum helps
  130M–1.3B; Token-Order Prediction is the safer variant. https://arxiv.org/html/2508.19228 ·
  https://arxiv.org/html/2505.22757 · https://aclanthology.org/2025.babylm-main.41/ → keep MTP out of the 600M run.
- **z-loss 1e-4:** cheap logit-stability insurance in cooldown (Marin, OLMo); neutral in 124M–720M
  mid-train ablations. https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md
- **WSD vs cosine:** WSD wins on flexibility (mix swaps, checkpoint reuse); needs ~20% decay to
  match cosine. https://arxiv.org/abs/2410.05192 · https://arxiv.org/pdf/2602.02522
- **Batch ramp:** B_opt ∝ D^0.383 (Power Lines / nanochat); Qwen3.8-Flash-Next found ramping
  *unnecessary* under Muon + gated residual (+18.8% steps). https://karpathy-nanochat.mintlify.app/training/scaling-laws

### Long-context stage at small scale
- **ProLong:** 20B @64k + 20B @512k from an 8B; mix 30% code repos / 30% books / 3% textbooks /
  37% ShortMix; RoPE θ 8e6 → 1.28e8; cross-document masking; LR reset per stage. https://arxiv.org/pdf/2410.02660
- **SmolLM3:** +100B only; NoPE + higher θ sufficient; extra long-doc upsampling did not help
  RULER/HELMET. https://huggingface.co/blog/smollm3
- Rule of thumb: train ≥ target eval length; books/code/repos > needle-only synthetic in pretraining;
  document masking standard for packed sequences.

### Data (licences)
| Corpus | Licence / caveat | URL |
|---|---|---|
| DCLM-Baseline (4T) | CC-BY-4.0 | https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 |
| Dolma 3 (5.93T) | ODC-BY | https://huggingface.co/datasets/allenai/dolma3_mix-6T |
| FineWeb / FineWeb-Edu | ODC-BY | https://huggingface.co/datasets/HuggingFaceFW/fineweb |
| Nemotron-CC v2 / v2.1 | NVIDIA agreement — training use, gated, non-redistributable | https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1 |
| Common Corpus (~2T, Pleias) | public-domain / permissive | https://arxiv.org/abs/2504.18225 |
| ClimbMix | CC-BY-NC — ablations only | https://github.com/karpathy/nanochat |
| Long docs | PG-19, arXiv (Dolma), ProLong ShortMix, repo-level code | https://arxiv.org/pdf/2410.02660 |

### Compute
- **SLMTrainBench** (150M–8B, ctx 512–32k, TPS + modelled MFU) — nearest to our 600M is 700M. https://huggingface.co/datasets/FAIRC/SLMTrainBench
- H100 bf16 dense peak 989 TFLOP/s; typical small-LM MFU 25–45%. https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
- No published FA3 numbers for 600M @256k; expect ≪ 8k TPS and sequence parallelism.

---

## Free wins for the E18 main run (ranked, with what we already have)

1. **NorMuon / Polar Express + wd + RMS-matched scale** (have Muon; upgrade the orthogonaliser).
2. **Data mix and stage decay** (math/code 10–15% in the decay) — bigger than any knob.
3. **Hashed n-gram tables sized by benchmark, not CE** (have tables; re-size at the pilot).
4. **WSD with ~20% decay** and mix swaps at stage boundaries.
5. Value embeddings + QK-norm + U-net skips (have).
6. **z-loss 1e-4 in cooldown** (have `z_coef`; enable in decay).
7. **NoPE on the global read only after a copy check** (our tiny study: not free); θ bump + YaRN.
8. **Skip MTP** at 600M; skip batch ramp under Muon.

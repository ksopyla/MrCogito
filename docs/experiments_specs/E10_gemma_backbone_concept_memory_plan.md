# E10 — Implementation Plan (Gemma-3-1B backbone concept memory, Design C)

- **Spec:** [E10_gemma_backbone_concept_memory.md](E10_gemma_backbone_concept_memory.md) ·
  **Status:** implemented; readiness-audited 2026-07-11; awaiting Odra calibration/launch
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's platform change: frozen `google/gemma-3-1b-pt` + LoRA, global layers
> windowed, concepts read via a gated parallel cross-attention branch on the 4 global layers,
> concepts written per 512-token block via the BiXT write op. Repo-rooted: every class/path below
> was read (incl. the installed `transformers 4.57.6` Gemma3 modeling source).

## 1. Source & fit
- **Origin:** 2026-07-07/08 design discussion (Design C) + `docs/literature_review/recurrent_memory_transformers.md`
  (RMT §B pretrained retrofit, Block-Recurrent §B gated write, Infini-attention §D collapse warning).
- **Synthesis verdict:** **Adapt** — take the pretrained-backbone retrofit (RMT/ICAE/LongLoRA
  precedent) + gated zero-init read (Flamingo/LLaMA-Adapter) + gated recurrent write (E09's
  `ConceptWriteHead` design); drop the from-scratch BiXT encoder (recurrent encode = recurrent
  decode through the same write op) and drop in-softmax concept tokens (breaks zero-init).
- **Architecture mapping (ONE):** a new **backbone family** beside `concept_ar`/`perceiver_denoise` —
  touches model + data collator + entrypoint wiring; encoder/decoder modules of the prior line are
  untouched.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `Gemma3ForCausalLM` / `Gemma3TextModel` | reuse as-is (frozen; **mask-dict input** at `modeling_gemma3.py:533` `if not isinstance(causal_mask_mapping := attention_mask, dict)` is the surgery hook) | site-packages `transformers/models/gemma3/modeling_gemma3.py` |
| `Gemma3Attention.q_proj/k_proj/v_proj/o_proj`, `q_norm/k_norm`, `scaling` | **reuse by reference** inside the concept-read branch (no new attention weights; LoRA-adapted automatically) | same, `:263-341` |
| `BiXTCrossAttention` (`update_tokens=False`) | reuse as-is = the concept **write op** (dim_lat=dim_tok=1152) | `nn/concept_encoder.py:335` |
| `PerceiverDenoiseTrainer` eval hooks (`concept_ablation_ce`, `encode_concepts` duck-typing) | reuse + align E10 live gates: within-sample RankMe, read/write gates, Δshuffle at positions ≥2K (≥1024) | `training/train_perceiver_denoise.py` |
| `DataCollatorForPrefixGeneration` | NOT used (E10 is plain next-token CE) | `data/data_collators.py:163` |
| **`DataCollatorForCausalLM`** | **NEW** — pad-to-batch-max `input_ids`/`attention_mask`/`labels(-100 at pad)`; seedless (no corruption) | `data/data_collators.py` |
| **`BackboneConceptLM` + `BackboneConceptConfig` + `ConceptReadBranch` + `GlobalLayerWithConceptRead` + `ConceptWriteHead`** | **NEW — reusable, config-selectable** (`concept_io_mode="global_kv"`; E11 `mem_tokens` / E12 `kv_prefix` land here later) | `nn/backbone_concept_lm.py` |
| `train_perceiver_denoise.py` | extend — `BACKBONE_MODEL` branch (model build, `causal_lm` objective, collator, W&B identity) | `training/train_perceiver_denoise.py` |
| `train_perceiver_denoise_multigpu.sh` + `launch_e05.sh` pattern | extend / new thin wrapper `launch_e10.sh` | `scripts/` |
| `pretokenize_mix.py` spine | reuse as-is with `--tokenizer google/gemma-3-1b-pt`; **manifest name must not collide** with the SmolLM2 manifest → `MANIFEST` env override in the launcher | `scripts/pretokenize_mix.py`, launcher `:130-165` |
| `peft` (LoRA r=16, targets `q_proj,k_proj,v_proj,o_proj`) | **new dependency** (`uv add peft`) | `pyproject.toml` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `S`=2048 seq, `K`=512 (block = write cadence = Gemma sliding window), `C`=128,
`H`=1152, `V`=262144, `L`=26 (4 global at idx 5,11,17,23).

```
z = z0.expand(B, C, H)                                  # learned init state (the "encoder" is the loop)
for b in 0..ceil(S/K)-1:                                 # 4 blocks at S=2048; 16 at eval S=8192
    s, e   = b*K, min((b+1)*K, S)
    carry  = input_ids[:, s-K:s]  (b>0)                  # (B, K)  gold prev block (teacher forcing)
    dec_in = cat([carry, input_ids[:, s:e]])             # (B, ≤2K)
    mask4d = sliding_causal_window_mask(dec_in, pads)    # (B,1,Q,Q) float −inf; window=K, causal
    out    = gemma.model(inputs_embeds=embed(dec_in),
                         attention_mask={"full_attention": mask4d,     # ← global layers WINDOWED
                                         "sliding_attention": mask4d},
                         use_cache=False)                # positions reset per block: 0..len-1
      # inside the 4 wrapped global layers:
      #   h_out += tanh(g_ℓ) · ConceptRead_ℓ(input_layernorm(h_in), z)   # g_ℓ zero-init
      #   ConceptRead: q=q_norm(q_proj(x)) [no RoPE], k=k_norm(k_proj(z)), v=v_proj(z), GQA, o_proj
    logits_b = lm_head(out.last_hidden_state[:, -(e-s):])                 # (B, K, V) — per block,
    loss_b   = CE(logits_b, labels[:, s:e])                               # NEVER cat full (B,S,V)!
    h_blk    = out.last_hidden_state[:, -(e-s):]                          # (B, K, H)
    z = z + tanh(α) · sandwich_norm(BiXT_lat←tok(norm(z), norm(h_blk), pad_mask))   # gated write
loss = Σ loss_b·n_b / Σ n_b                              # token-weighted; BPTT across blocks
```
**Invariants:** O(N·(K+C)) total (per-block (2K)² windowed + C·K write + C reads); positions never
exceed 2K → unbounded length extrapolation; `(B,S,V)` logits never materialized (V=262K ⇒ that
tensor would be ~8.6 GB at B=8).

**Zero-init property (the key regression test — nuance found during implementation):** with all
`g_ℓ=0` and `α=0` the graft is inert, and the block loop equals a single all-windowed forward of
plain Gemma *exactly for the first two blocks* (RoPE is relative, so the per-block position reset
changes nothing; verified to atol 1e-4 in `tests/test_backbone_concept_lm.py`). From block 2 on
they deliberately diverge: in a full-sequence windowed forward the receptive field grows by
~(W−1) per stacked layer (≈13K tokens over 26 layers), while the block loop hard-truncates
history at carry+block (~2K at K=512). That truncated context is exactly the channel the concepts
must supply — and the control arm shares the identical block protocol, so A/B attribution stays
clean. Consequently Stage 0 measures the gap with the **blockwise** scorer (the E10 protocol,
concepts off), not a single windowed forward. `CONCEPT_NUM=0` skips reads/writes entirely (the
trained control arm).

## 4. Inputs & data
- **Dataset:** `smollm3_inspired_2k_e05` recipe, re-pretokenized with `google/gemma-3-1b-pt`
  tokenizer at seq 2048. Launcher sets `MANIFEST=${DATASETS_TOK_DIR}/smollm3_inspired_2k_e05_gemma_manifest.json`
  (and a Gemma-suffixed `--cache_dir` subtree) so the SmolLM2 cache is untouched.
- **Collator:** new `DataCollatorForCausalLM` → `input_ids [B,S]`, `attention_mask`, `labels`
  (=input_ids, −100 at pad). Blocking is internal to the model forward (same principle as E09's plan).
- **Gemma tokenizer:** has real `pad_token` (`<pad>`, id 0) + `bos` — collator uses them; BOS
  prepended by the tokenizer during pretokenize (`add_bos_token=True` default for Gemma).
- **8K extrapolation eval:** small fineweb-edu long-doc slice pretokenized at 8192 (eval-only
  manifest; used by Stage 0 + final eval, not by training).

## 5. Loss & training objective
- **Loss:** plain next-token CE (shifted inside the model per block), token-weighted across blocks.
  No `loss_manager` component (keeps the platform introduction to one signal); concept regularizers
  are follow-up levers.
- **Objective:** new `objective_variant="causal_lm"` in the entrypoint (added to `VALID_OBJECTIVES`
  and to the trainer's fast path — `compute_loss` already routes plain objectives through
  `model(**inputs)`, `training/train_perceiver_denoise.py:560-570`).
- **Trainable params:** LoRA adapters (backbone, all 26 layers' q/k/v/o) + read gates `g_ℓ` (4
  scalars) + write head (BiXT + norms + α) + `z0` [C,H]. Backbone weights + embeddings + lm_head
  frozen. The final write remains zero-connected to the loss so short single-block batches keep
  every write parameter in the DDP graph (`find_unused_parameters=False`); it contributes no
  optimization signal unless a later block reads the state.

## 6. Config & launch
- **`BackboneConceptConfig`** (new `PretrainedConfig`, `model_type="backbone_concept"`):
  `backbone_model="google/gemma-3-1b-pt"`, `concept_num=128` (0 ⇒ control), `concept_block=512`,
  `concept_io_mode="global_kv"`, `write_num_heads=4`, `read_gate_init=0.0`, `write_gate_init=0.0`,
  `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `lora_targets="q_proj,k_proj,v_proj,o_proj"`,
  `global_attention_mode="windowed"` (`"full"` = intact Gemma, for Stage-0 upper baseline),
  `checkpoint_family="backbone_concept"`.
- **Entrypoint wiring:** `ModelArguments.backbone_model` (default None ⇒ every existing family
  byte-identical) + `concept_block`, `concept_io_mode`, `lora_*`; when set, `main()` builds
  `BackboneConceptLM`, selects `DataCollatorForCausalLM`, builds a `backbone_concept` W&B identity.
  `concept_num` reuses the existing arg.
- **Launcher:** `scripts/launch_e10.sh` (pattern of `launch_e05.sh`): pins `BACKBONE_MODEL`,
  `OBJECTIVE_VARIANT=causal_lm`, `TOKENIZER_NAME=google/gemma-3-1b-pt`, seq 2048, the Gemma manifest
  name, LR 1e-4 / warmup 500 / clip 0.5, bf16 + gradient checkpointing, batch 4×accum 6.
  Arms: default = concept arm; `CONCEPT_NUM=0 bash scripts/launch_e10.sh` = control.
- **Stage 0:** `analysis/run_e10_stage0.py` — loads the untrained wrapper twice
  (`global_attention_mode=full` vs `windowed`, `concept_num=0`) and reports per-position-bucket CE on
  the held-out slice at 2048 + 8192 → the gap **G** (spec gate ≥ 0.05 nats).

## 7. Tests & smoke
`tests/test_backbone_concept_lm.py`, all on a **tiny random `Gemma3TextConfig`** (H=64, L=6 with
5:1 layer pattern, heads 2/kv 1, head_dim 32, V=256, window=8, no hub access):
1. **Zero-init equivalence (the load-bearing test):** gates at 0 ⇒ block-loop logits == single
   plain-Gemma forward with the same windowed mask (atol 1e-5); and `concept_num=0` == gates-at-0.
2. **Shapes + finite loss** for S not divisible by K (ragged last block) and batches with padding.
3. **Gradient reach:** after `loss.backward()`, LoRA params, `g_ℓ`, write-head params, and `z0` all
   have nonzero grads (write head via blocks < last).
4. **No-leak (block causality):** perturbing block b's tokens leaves blocks < b logits unchanged.
5. **Read-gate effect:** with `g_ℓ` forced >0 and shuffled z, beyond-carry logits change (read is live).
6. **Ablation contract:** `concept_ablation_ce(input_ids, mask, labels, window_k=None)` returns the
   dict the trainer averages, with the decisive "beyond" region beginning at `2K`;
   `encode_concepts(...).last_hidden_state` is `[B,C,H]`.
7. **Production checkpointing path:** open read/write gates retain nonzero concept and BiXT gradients
   with `gradient_checkpointing=True`.
- **Tier-1 checkpoint analysis:** `analysis/run_concept_analysis.py --model_type backbone_concept`
  runs held-out geometry + recurrent concept-ablation on saved E10 checkpoints (generation-only
  extras remain disabled because E10 has no separate concept-to-text decoder).
- **Local MPS smoke:** tiny config, 3 optimizer steps on random batches — loss finite and decreasing.

## 8. Risks & tradeoffs
- **Risk — mask-dict API drift** (`transformers` internals). *Cheapest signal:* unit test 1 breaks on
  version bump. *Mitigation:* pin the behavior in the test; masks are built by us, not by HF utils.
- **Risk — gated repo access on the servers.** `gemma-3-1b-pt` license is accepted for `ksopyla`
  (verified 2026-07-08 locally); Odra/Polonez use the same token via `.env`. *Fallback:* none needed.
- **Risk — hooks/wrappers vs gradient checkpointing.** Wrapper modules (not hooks) hold the read
  branch, so `GradientCheckpointingLayer` recompute stays consistent. *Cheapest signal:* grad-reach
  test with `gradient_checkpointing=True` variant.
- **Risk — read branch never opens (g stays ~0).** *Cheapest signal:* log `tanh(g_ℓ)`, `tanh(α)` +
  Δshuffle each eval; spec kill-gate at 50% budget. *Fallback:* init `g` at small positive; per-head
  gates.
- **Risk — recurrent-state collapse** (Infini-attention). *Cheapest signal:* within-sample RankMe of
  final z each eval (spec guard ≥0.3·C, kill <0.15·C). *Fallback:* stronger sandwich norms, lower
  write LR.
- **Risk — checkpoint size** (frozen 1B saved per checkpoint, bf16 ~2 GB × save_total_limit 5).
  Acceptable on Odra; a lora-only save is a follow-up optimization, not this spec.
- **Risk — 262K-vocab logits memory.** Handled by per-block CE (never cat `(B,S,V)`); eval uses the
  same path. *Cheapest signal:* peak-VRAM print in smoke.

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch — nn/backbone_concept_lm.py
class ConceptReadBranch(nn.Module):            # reuses the wrapped layer's own projections
    def forward(self, x_normed, z, attn):      # (B,Q,H),(B,C,H), attn=Gemma3Attention
        q = attn.q_norm(reshape(attn.q_proj(x_normed)))          # no RoPE — position-free memory
        k = attn.k_norm(reshape(attn.k_proj(z))); v = reshape(attn.v_proj(z))
        return attn.o_proj(sdpa(q, repeat_kv(k), repeat_kv(v), scale=attn.scaling))

class GlobalLayerWithConceptRead(nn.Module):   # replaces model.layers[i] for the 4 global layers
    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        if self.concept_state is not None:     # set by BackboneConceptLM before each block forward
            read = self.read_branch(self.layer.input_layernorm(hidden_states), self.concept_state,
                                    self.layer.self_attn)
            out = (out[0] + torch.tanh(self.gate) * read, *out[1:])
        return out

class ConceptWriteHead(nn.Module):             # E09 design, backbone-dim
    def forward(self, z, h_blk, pad_mask):     # (B,C,H),(B,K,H) -> (B,C,H)
        lat, _ = self.bixt(self.norm_lat(z), self.norm_tok(h_blk), key_padding_mask=pad_mask)
        return z + torch.tanh(self.alpha) * self.sandwich(lat)

class BackboneConceptLM(PreTrainedModel):      # config_class = BackboneConceptConfig
    def forward(self, input_ids, attention_mask=None, labels=None, **kw): ...  # block loop, loss
    def concept_ablation_ce(self, input_ids, attention_mask, labels, window_k=None) -> dict: ...
    def encode_concepts(self, input_ids, attention_mask=None, return_dict=True): ...  # final z
```

*Handoff: spec + this plan → `research-implement`. Implementation follows in this session (user
pre-approved).*

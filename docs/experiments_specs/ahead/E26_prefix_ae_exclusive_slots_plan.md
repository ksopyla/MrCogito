# E26 — Implementation Plan

- **Spec:** [E26_prefix_ae_exclusive_slots.md](E26_prefix_ae_exclusive_slots.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`

> Weak prefix-block AE is the **write** objective for exclusive r=16 slots.
> Do not derisk into “train `u` on answer CE longer.” Do not add extra hops.

## 1. Source & fit
- **Origin:** E25 learned-pool wipe · Deng Fine-AE · ICAE AE+LM · CT attn-recon ·
  literature lever 1.
- **Synthesis verdict:** Adapt weak local recon; freeze pooler at mean until AE exists;
  Reject answer-only compressor training.
- **Architecture mapping:** bottleneck write + aux loss (decode path unchanged exclusive).
- **Boldness check:** AE gradients on `u`/`delta` only; answer CE does not pool.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `KVCompressor` | reuse; allow `identity_slots=False` under AE | `nn/perceiver_ar_lm.py` |
| exclusive inplace attend | reuse | `nn/perceiver_ar_lm.py` |
| `PrefixAEHead` | **new** — linear slot → `r` token logits | `nn/perceiver_ar_lm.py` (or `nn/concept_losses.py` if registered) |
| probe loss | extend: add `λ L_AE` | `verification/bapo_capability_probe.py` |
| `ArchSpec` | add `message_prefix_ae`, `message_prefix_ae_weight` | `evaluation/bapo_models.py` |

## 3. Forward pass (tensor shapes)
`B`, `S=512`, `r=16`, `nb=⌈S/r⌉`, `g`, `dh`, `V`=DNA vocab (~20).
```
h [B,S,H] → compressor → k̄,v̄ [B,nb,g,dh]
for valid complete sender blocks j:
    logits_j = PrefixAEHead(k̄_j or concat(k̄_j,v̄_j))   # [B, r, V]
    L_AE += CE(logits_j, ids[j*r:(j+1)*r])  # pad/invalid ignored
L_answer = packed CE on y
# stopgrad: compressor params see L_AE only (not L_answer)
```
Weak head: one `Linear` on flattened slot K/V (or mean of K,V). No transformer decoder.

## 4. Inputs & data
DNA `recall_single` seq=512 as E25. Boundary = DNA `query` control id (already wired).

## 5. Loss & training objective
`L = L_answer + λ L_AE` with `λ=1.0` default. Report `ae_key_acc` on tokens that fall
in key spans (same definition as E27, even if anchors are off).

## 6. Config & launch
- `PerceiverARConfig.message_prefix_ae: bool = False`
- `message_prefix_ae_weight: float = 0.0`
- `message_prefix_ae_stopgrad_answer: bool = True` (the claim)
- Probe flags `--message_prefix_ae --message_prefix_ae_weight 1.0`
- Launch: see spec. Do **not** pass `--message_identity_slots` (that bypasses `u`/`delta`).
  Mean-at-init remains because `u`/`delta` start at 0.

## 7. Tests & smoke
- AE logits shape `[B, nb, r, V]`; ignored positions do not contribute.
- With `u=0,delta=0`, AE still trains the head on frozen means (S1b can pass without
  moving the compressor — that is allowed; MATCH may still fail → K3).
- `message_prefix_ae=False` → byte-identical to current E21 forward.
- Probe 20 steps, both losses finite.

## 8. Risks & tradeoffs
- **Risk:** DNA vocab is tiny so AE is easy and MATCH still 0 (K3). That is a
  successful kill of “AE ⇒ addressable read.”
- **Risk:** leaking AE logits into the LM head. Keep a separate module.
- **Fallback after K3/K4:** E27 hybrid / r=1 for `b`; do not extra-hop.

## 9. Code sketches
```python
# sketch
class PrefixAEHead(nn.Module):
    def __init__(self, g, dh, r, vocab):
        self.proj = nn.Linear(g * dh, r * vocab, bias=False)
    def forward(self, k_bar):  # [B,nb,g,dh]
        B, nb, g, dh = k_bar.shape
        return self.proj(k_bar.flatten(-2)).view(B, nb, r, vocab)
```

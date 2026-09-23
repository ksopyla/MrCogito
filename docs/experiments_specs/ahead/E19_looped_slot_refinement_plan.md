# E19 — Implementation Plan

- **Spec:** [E19_looped_slot_refinement.md](E19_looped_slot_refinement.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`

> Tied loops over exclusive slots with per-slot targets. **Not**
> `message_extra_slot_attends`. Arith is structure control: primary `subexpr`,
> stack `match`, shortcut `eval`. Injected 1-token atoms; never glued BPE.

## 1. Source & fit
- **Origin:** literature lever 4 (LOTUS, Saunshi, Slot Attention T=3) · E25 extra-hop
  hops-only · CogitoProbe arith card.
- **Synthesis verdict:** Adapt tied set refinement **after** a load-bearing gate;
  Reject pause tokens / one extra attend as the reasoner; Reject arith as semantics.
- **Architecture mapping:** reason over the bottleneck (slot set) + per-slot loss.
- **Boldness check:** R≥4 tied, `extra_slot_attends=0`, eval is not S1.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| exclusive E21 + Wave A compressor | reuse from gate ckpt | `nn/perceiver_ar_lm.py` |
| `write_back_hook` | reuse as optional KV write; **not** the loop | `nn/perceiver_ar_lm.py` |
| `SlotSetLoop` | **new** tied SelfAttn+FFN over `nb` slots | `nn/perceiver_ar_lm.py` |
| PrefixAEHead | reuse if E26 landed | E26 module |
| CogitoProbe arith | reuse | `data/concept_probes/` |
| `message_extra_slot_attends` | **do not set** | stays 0 |

## 3. Forward pass
`C = nb` exclusive slots `[B,C,g,dh]` (or concatenated H).
```
z = slots
for r in range(R):          # R = message_slot_loops, weights tied
    z = z + TiedBlock(z)    # causal or bidirectional *over slots only*
k̄,v̄ = project(z)
exclusive read as E21
L = L_answer + L_slot       # L_slot = AE and/or CE on packed subexpr node ids
```
Do not attend extra times over *frozen* K/V (`extra_slot_attends`). Queries may
update because z updates.

## 4. Inputs & data
- Warm-start weights from the gate run.
- Train/eval: `Cache/concept_probes/full_1k4k/arith`, `seq_len=1024` first.
- Break out `task ∈ {subexpr, match, eval}`. Hub name `ksopyla/cogito-probe-arith`.
- Atoms: `ARITH_ATOMS` injected; `input_ids` already composed.

## 5. Loss & training objective
Packed answer CE (all three tasks in the mix) **plus** per-slot targets:
- If E26 AE exists, keep `L_AE` on prefix blocks.
- Optional: supervise slots from `meta` node values on `subexpr` rows only.
Answer-only `eval` rows must not dominate the average (subexpr is 50% of the mix).
Report metrics **by task**, never a micro-average that hides eval-only wins.

## 6. Config & launch
- `message_slot_loops: int = 0` (default; E18-loadable)
- `message_slot_loop_tied: bool = True`
- `PAR_MESSAGE_SLOT_LOOPS=4`, `PAR_MESSAGE_EXTRA_SLOT_ATTENDS=0`
- R=1 probe: same ckpt, `loops=1` at eval (test-time R), plus a trained R=1 if S4
  needs a matched train — prefer eval-time R first (cheaper).

## 7. Tests & smoke
- `R=0` byte-identical to gate architecture with loops module skipped.
- `R=2` shares block weights (`is` the same `nn.Module`).
- Arith `eval` rows still train but S1 uses `subexpr` mask only.
- RankMe of slots logged each eval (K4).

## 8. Risks & tradeoffs
- **Risk:** loops over slots that still don’t hold the AST → K2 calculator.
  **Cheapest signal:** `eval` vs `subexpr` split after 1k steps.
- **Risk:** calling extra hop “R=1 loops.” Spec forbids setting
  `message_extra_slot_attends`.
- **Gate:** do not launch without a load-bearing Wave A/B pass.

## 9. Code sketches
```python
# sketch
class SlotSetLoop(nn.Module):
    def __init__(self, d, n_heads):
        self.block = SlotBlock(d, n_heads)  # one shared block
    def forward(self, z, R):
        for _ in range(R):
            z = z + self.block(z)
        return z
```

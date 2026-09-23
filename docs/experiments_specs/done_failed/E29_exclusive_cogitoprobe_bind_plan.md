# E29 — Implementation Plan

- **Spec:** [E29_exclusive_cogitoprobe_bind.md](E29_exclusive_cogitoprobe_bind.md) · **Status:** done_failed (S1 miss)
- **Authored by:** `implementation-plan` · for → `research-implement`

> Bind is the train exam. Props is a gist **control** on the same checkpoint.
> Do not turn this into an arith run. Do not Hub upload.

## 1. Source & fit
- **Origin:** diagnosis document embedding · CogitoProbe bind/props cards (PR 39) ·
  literature MATCH-class on language atoms.
- **Synthesis verdict:** Adapt entity-centric slots; Reject bag-of-tokens success on
  `attr_color` alone.
- **Architecture mapping:** exclusive memory + binding objective.
- **Boldness check:** hop_friend_place is the gate, not attr_color.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| E28 loader / `PAR_MESSAGE_*` | reuse | launcher + CogitoProbe disk |
| bind/props generators | reuse | `data/concept_probes/generate.py` |
| eval by `task` field | new helper | `evaluation/` CogitoProbe breakout |
| props shuffle controls | new, config-eval | eval helper (filler vs `meta.propositions`) |

## 3. Forward pass
Identical exclusive E21 to E28. Train only on bind `labels`. At eval, break out
`attr_color` / `who_place` / `hop_friend_place`. Props: no grad; counterfactual
token shuffles using `meta` JSON.

## 4. Inputs & data
- Train: `Cache/concept_probes/full_1k4k/bind/train` (`seq_len=1024` first).
- Eval: bind `test` by task; props `test` for S3.
- Hub names: `ksopyla/cogito-probe-bind`, `ksopyla/cogito-probe-props`.
- Same tokenizer, boundary `Q`, packing-off as E28.

## 5. Loss & training objective
Packed bind-answer CE. Props not in the train mix (control must not be fitted).

## 6. Config & launch
`EXPERIMENT_ID=E29`. Same `PAR_MESSAGE_*` as E28 / Wave A winner.
Eval command (specified, not a new trainer):
`uv run python evaluation/eval_cogito_probe.py --root Cache/concept_probes/full_1k4k --families bind props --by_task`

(If that helper does not exist yet, it is the reusable foundation piece; do not fork
`train_*.py`.)

## 7. Tests & smoke
- Bind rows have `task ∈ {attr_color, who_place, hop_friend_place}`.
- Props `meta.propositions` shuffle changes gold colours; filler shuffle does not
  change `answer` (generator invariant — assert in `tests/test_concept_probes.py` if
  missing).
- Dense S0 smoke on pilot bind @1024.

## 8. Risks & tradeoffs
- **Risk:** `hop_friend_place` dense S0 fails at H=256 — K1, shrink hops or grow
  budget, do not score E21.
- **Risk:** training bind + eval props domain shift. S3 is a control, not a second
  S1. Mixed S1-pass / S3-fail is a gist warning (K3).
- **Fallback:** do not add arith `eval` to the mix.

## 9. Code sketches
```python
# sketch
tasks = {"attr_color": [], "who_place": [], "hop_friend_place": []}
for row, metrics in batch:
    tasks[row["task"]].append(metrics.recovered_bits)
```

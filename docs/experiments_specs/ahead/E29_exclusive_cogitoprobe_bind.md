# E29 — Exclusive slots bind (entity, attribute, value), not a bag or gist

- **Status:** draft 2026-09-16 (queue rank 4, Wave B; awaiting KS approval) · not launched
- **Serves:** diagnosis “the long-range object is a document embedding” on
  language-shaped atoms. Queue:
  [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md).
- **Implementation plan:** [E29_exclusive_cogitoprobe_bind_plan.md](E29_exclusive_cogitoprobe_bind_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-16 · closed —

> One claim, the bind family’s claim: concepts bind tuples. `attr_color` can be a
> label list; `who_place` / `hop_friend_place` cannot if entities share the attribute
> vocabulary. **`ksopyla/cogito-probe-props`** is the registered *gist control* on the
> same checkpoint (filler shuffle holds; proposition shuffle kills), not a second
> architecture. Arith is not this spec.

## Hypothesis
If exclusive r=16 (Wave A winner) is trained on packed CE from
**`ksopyla/cogito-probe-bind`** at seq=1024 (then 4096 if 1k passes), then
**`hop_friend_place` recovered bits ≥ 0.75 × dense** whenever **`attr_color`** also
clears that bar **because** one-hop friend→place requires entity-centric slots (or an
addressable tuple table), which a bag-of-tokens or a single document embedding cannot
implement when colours/places are shared — and on **`ksopyla/cogito-probe-props`**
held-out, shuffling filler n-grams must **not** move answers while shuffling
proposition colours must.

## Builds-on
- **Foundation:** same exclusive `perceiver_ar` + CogitoProbe disk shards as E28.
  Cards: `docs/3_Evaluations_and_Baselines/dataset_cards/cogito-probe-bind/` ·
  `.../cogito-probe-props/`.
- **Init / checkpoint:** random small H=256, or warm-start from E28 only if E28 S1
  passed (same tokenizer/mask). No SVD init.
- **Baseline to beat:** dense S0 ≥ 75% on seq=1024 bind tasks. E22 far marginal 0.05
  nats / document gist. E18b T 4.49% vs dense 99.33% (content addressing).
- **Materially new:** exclusive compressed read scored on *binding* vs *gist*, not DNA
  MATCH and not FineWeb CE.

## The architectural bet
Exclusive slots must store `(entity, attr, value)` enough to answer `hop_friend_place`.
Break-out tasks: `attr_color` / `who_place` / `hop_friend_place` (card mix ~151/159/154
on pilot). Props `prop_color` is the gist control. **Out of scope:** bits capacity
(E28); arith (E19); extra hop; 32k train.

## Why this is not a safe retread
DNA MATCH identity at 1280 already showed exclusive *can* bind a 4-symbol key when
slots are r=1. Bind asks whether compressed slots bind *shared-vocabulary entities*
— the document-embedding failure mode E22/E18 actually learned. Surprising if r=16
slots do it without identity-on-everything.

## Success criteria (set BEFORE running)
- **S0:** dense ≥ 75% on seq=1024 `attr_color` **and** `hop_friend_place` (if hop dense
  misses, the rung is ill-posed — fix generator, do not score E21).
- **S1 binding:** E21 `hop_friend_place` recovered bits ≥ **0.75 × dense** at 1024.
- **S2 not labels-only:** `attr_color` also ≥ 0.75 × dense (otherwise hop is uninterpretable).
- **S3 gist control (props, same ckpt, no extra train):** filler-token shuffle in
  `context` changes acc by ≤ 5 points; shuffling colours in `meta.propositions` drops
  acc to within 10 points of chance.
- **S4:** `message_override` swapped ≤ none on bind answers.

## Kill criteria (set BEFORE running)
- **K1:** dense hop < 75% @1024 — do not score E21.
- **K2:** `attr_color` ≥ 0.75 × dense **and** `hop_friend_place` at chance after budget —
  latents are labels/gist, not bindings. Kill the “semantic slots” claim for this compressor.
- **K3:** props filler shuffle *does* drop answers (tracks n-grams) — gist, even if hop
  moved. Record as mixed; do not grow length.
- **K4:** S1 miss at 1024 while dense saturates — exclusive r=16 cannot bind at this
  scale. Do not 4k/8k/32k.

## Plan
- **Data:** local `Cache/concept_probes/full_1k4k/{bind,props}` from the same
  `--scale full --lengths 1024 4096` build as E28. Hub ids as names only. **No upload.**
- **Compute:** Odra. **~8 GPU-h** (3 arms at 1k; 4k only if S1).
- **Launch:** same `PAR_MESSAGE_*` / `PRESERVE_PRECOMPUTED_LABELS` pattern as E28,
  `EXPERIMENT_ID=E29`, train split = bind. Eval bind test by `task`; one props
  counterfactual pass on the best ckpt.
- **New foundation code:** task-breakout metrics + props shuffle eval on disk shards
  (reusable CogitoProbe eval helper). Compressor flags from Wave A / E28.

## Result
- Run id: `<run_id>`
- WandB: —
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: —

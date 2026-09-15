# Before / after — real messages from our sessions, rewritten

Every "before" below is a verbatim excerpt from a real session (2026-09-11 → 2026-09-12). Every
"after" says the same thing inside the budget from `SKILL.md`. Read these before writing a first pass.

---

## 1. Status check

**He asked:** *"What is the current status?"* (5 words)

**Before — 521 words, opened with gate codes he had never seen defined:**

> **E22 is mid-eval.** Arm A and arm C have both finished 0.5B-token training with nearly identical
> eval loss. Arm A already fails passkey (same as C). The dense control and the proper concept-ablation
> probe are still running. […]
> - Concept probe (`--probe concept`) is the process on GPU 1 — this is the S1 measurement that was
>   missing from the mid-run checkpoint-160 peek (Δ_none ~0.17–0.21, below the 0.30 gate)
>
> So S4 is already a miss on the final A checkpoint. S2 looks like a miss on buckets. S1 is still in flight.

**What happened next:** four hours later he asked *"Please report current status"* — the same question,
shorter. The report told him nothing he could hold.

**After — 58 words:**

> Bad news so far. The concept model and the control that has **no** long-range memory are scoring the
> same, and neither can find a password hidden 30k words earlier (0 out of 100, both).
>
> Two checks still running (about 2 h): how much the model actually leans on the concepts, and the
> full-attention reference model for comparison.
>
> If those land where I expect, this design is dead and I will say so.

---

## 2. Result summary

**Before — 537 words, headline was a table of gate codes:**

> **Both machines are idle.** Arm A and arm C evals are finished. Dense and the long-doc pretok died
> when Polonez `/home` hit 100%.
>
> ## Headline vs E22 gates
>
> | **S1** concept ablation ≥ 0.30 nats beyond 4k | **Miss** | Δ_none: 0.27 / 0.26 / 0.25 / **0.23** nats (4k→32k). Shuffled Δ 0.32 → **0.19** on the far bucket. |
> | **S2** A better than C by ≥3% on far tokens | **Miss** | Far CE A **4.162** vs C **4.133**. A is slightly worse. |
> | **S3** A within 3% of dense | **Unknown** | Dense died at ~step 390 / 637. |
> | **S4** passkey ≥ 0.5 | **Miss** | A and C both **0.0** at 8k and 32k. |

**What happened next:** he replied *"Remove old checkpoints if needed"* — he took the disk-cleanup
sentence and ignored four failed criteria. The result never landed.

**After — 108 words:**

> **E22 (concepts as a notebook) failed.** I am stopping it.
>
> - The model does read its notebook — taking it away costs it something real. But the pages it uses
>   are the ones summarising text it can already see; the genuinely distant pages are worth almost nothing.
> - The control with no notebook at all scores the same. So the notebook earns nothing.
> - Password-in-a-long-document: 0 out of 100. It cannot retrieve.
> - The notebook itself is healthy — pages differ from each other. So the failure is in *reading*, not writing.
>
> This is the third time the same pattern has killed a design: plain next-word prediction never forces
> the model to look far away. Details in the run report; I'll write the root cause next.
>
> Ask me about: why reading fails and writing doesn't, or how to force the model to need distance.

---

## 3. Pitching an experiment

**Before — 665 words, the message that produced *"I don't get what we need to do next, there are to
much information"*:**

> **1. The frontier shipped our skeleton, under text.** DeepSeek-V4.1-Flash (released 2026-09-10) is a
> Causal Encoder-Decoder: 20 encoder layers, 20 decoder layers, decoder global K/V projected once from
> encoder final states at ~890 B/token, 1M context. That is structurally E18's "one full-causal global
> read, K/V computed once, ~1 KB/token". Qwen3.8 (3 GDN : 1 gated attention, hashed n-gram host-memory
> tables in Flash-Next) and GLM-5.3-Flash (34 KDA + 11 sparse MLA, top-2048 of 1M) fill in the rest […]
> Arms R (r=16) and U (r=1), probes `none/swapped/raw` … Success and kill criteria S1–S6 / K1–K4,
> launch command over the existing `scripts/launch_e18.sh` with four new env knobs.

**After — 143 words:**

> **The bet: make the model *need* its notebook, by taking the book away mid-exam.**
>
> Everything we have tried has failed the same way — the model can always reread the raw text, so it
> never learns to use compressed memory. So: halfway through a document, cut the raw text off. Everything
> after that point can only reach the first half through a small set of compressed summaries. Now the
> compression is on the critical path; if it is bad, the model's score is bad.
>
> Two versions run side by side: one compresses 16× (the real bet), one compresses not at all (the
> reference showing the ceiling).
>
> Cost: about a day on one machine. It reuses the model we already have — no new training script.
>
> **Decision I need:** run this now, or first try moving the long-range read deeper into the stack
> (cheaper, 1.4 GPU-hours, answers a smaller question). I recommend this one.
>
> Ask me about: how the cut works, what the frontier labs shipped that looks like this, or how I'll know it failed.

---

## 4. "Explain it simply" — the second pass

This one **worked**. It arrived on his sixteenth message, after he asked twice, byte-identical. Copy
this shape, and use it the *first* time.

> The short version: we built a tiny DNA-like exam where the answer is **fully determined** by something
> written far back, then asked whether the concept array can carry that answer. It can, but weakly.
> There is **one shared concept array**, not a separate concept stack per decoder layer.
>
> **Recall** is a labelled filing cabinet. Early in the strip you bury several tagged boxes:
> `KEY=cat VALUE=blue`, `KEY=dog VALUE=red`. At the end you ask "what is the value of `cat`?"
> **Far copy** is a fax. A 32-letter string is written early. At the end you must reproduce it letter for letter.
> **Chain** is a scavenger hunt. You bury `A→B`, `B→C`, `C→D` in random order, mixed with fake arrows.
> **Count** is a tally. "How many `G`s were there, modulo 4?"
>
> **Cross-entropy** is "how surprised is the model by the true next letter?" 0 means it is sure.
> 1.386 means it is as unsure as a random guess among 4 letters.

Why it works: an image per idea, the metric explained instead of printed, and the structural question
("one array or one per layer?") answered in the first three lines. Its one flaw — at 1,530 words it was
still the longest message in the thread. **Language was fixed, length was not.** Do both.

---

## 5. Asking for a decision

**Before — a 7-item numbered menu of code changes, ending:**

> Say which of these you want built and I will implement them as config-selectable changes with tests.

**What happened next:** *"Do all your best to improve it and iterate on this architecture"* — he could
not choose, because nothing in the list told him what he was choosing between.

**After:**

> **One decision.** The concept summaries are currently blurry: each one secretly covers ~3,000 tokens
> of context, not the 16 we designed for — so neighbouring summaries are near-duplicates.
>
> - **Sharpen them** (limit what each summary can see). Clean test of the original idea. Half a day.
> - **Keep them blurry, force the model to use them anyway** (change the objective). Tests a different
>   thing — whether blur was ever the problem.
>
> I recommend sharpening first: if it fails, blur was never the issue and we save the second run.
>
> The other five ideas I found are in the plan document; none of them change what we learn next.

---

## 6. Closing a long autonomous stretch

**Before — after 786 messages and hundreds of tool calls, the turn ended with:**

> Now I wait for R2 to finish training on Polonez (estimated ~21:20 UTC).

**What happened next:** he eventually had to ask *"could you give me a status report where we are,
what was implemented?"* — after an entire evaluation layer, an experiment implementation and 527 tests
had shipped without one summary.

**After — 96 words:**

> **Done and pushed: you can now measure a checkpoint properly.** Three things exist that did not before.
>
> - A standard benchmark runner, so our models get scored the same way public models are — we can finally
>   say "worse than SmolLM2 by this much" instead of guessing.
> - Long-context probes: password-in-a-document, multi-key lookup, and a fair "cut the model's reach and
>   measure the damage" test.
> - The new experiment itself is implemented and smoke-tested on one GPU; every old checkpoint still loads.
>
> One run is training now, finishing around 21:20. I'll report when it lands.
>
> Ask me about: how the reach test works, or what the first benchmark numbers say.

---

## Quick reference — swap table

| he sees this and stalls | say this |
|---|---|
| "arm A vs arm C" | "the concept model vs the control with no memory" |
| "S2 miss, K1 met" | "it missed the bar we set, and hit the condition we agreed would kill it" |
| "Δ_none 0.25 nats" | "taking the concepts away costs it 0.25 — real, but small" |
| "eval 4.170 vs 4.164" | "a tie (lower is better)" |
| "passkey 0.0 @32k" | "it never finds a password hidden 30k words back — 0 out of 100" |
| "RankMe 210/2048" | "the summaries are genuinely different from each other, not collapsed" |
| "`PAR_GLOBAL_POSITIONS=7`" | "move the long-range lookup to the middle of the stack" |
| "BPB 1.467 vs pythia-160m 1.265" | "worse than a public model of our size — but that one saw 680× more text, which explains the whole gap" |
| "K1 met and the run was killed" | "it hit the failure condition we wrote down in advance, so I stopped it" |
| "E21 S1 PASS 63.95" when the JSON is E18 | that number is E18 (raw read); E21 was skipped or is a different row |
| `53.82 vs 47.34` with no arms | "compressed read 53.8 bits vs the 47.3-bit bar (75% of dense); prize 64 bits" |
| a 12-rung Status dump in chat | this cell only, plus "the map is in the run report" |
| "next: try 1100, 1200, keep-SWA, extra hop, and 3-item" | `Next: STOP MATCH2 2-item length extra-steps.` **or** one concrete cell |

---

## 7. Identity — do not relabel a control

**He asked:** a result for the compressed exclusive read (E21).

**Before — E18's live score written as E21:**

> E21 S1 PASS 63.95 bits @400. Flow 1.0 vs dense.

**What is wrong:** 63.95 @400 on that recipe was the **raw-read** arm (E18). The exclusive
slot arm was either skipped (dense K1) or a different row. Relabeling ships a false pass.

**After — chat Brief, 72 words:**

> **The compressed read was not scored.** The full-attention student never cleared 75% on
> this exam (24%, zero bits after four times the budget), so we never trained the compressed
> one — the test would have been noise.
>
> The earlier 64-bit pass at this length was the *raw* long-range read, a different student.
> Details in the run report.
>
> Next: stop length extra-steps on this exam. Ask me about: why dense died here, or the last
> length that *was* solvable.

**After — ledger/PR gate block:**

```
S0 FAIL / K1. Dense 24.1% / 0 bits @3200 (need ≥75%).
S1 not scored (e18 / e21 skipped). Do not relabel 1024 MATCH2 61.81 bits as this cell.
Next: STOP MATCH2 2-item length extra-steps. Do not 1100/1200/1408.
```

---

## 8. Next-ONE vs STOP extra-steps

**Before — five next experiments in the close:**

> Next: seq=1100, seq=1200, `--message_keep_local_swa`, extra hop, and 3-item MATCH2 at 1024.

**After — one line:**

> Next: **STOP MATCH2 2-item length extra-steps.** Do not 1100/1200. Do not 3-item at 1024.

The 8k extra-step (same recipe, longer budget) is allowed only when the spec says so — typically
when the 800 run was *climbing* and already past S0. Chance at the 800 floor → no extra-step.

---

## 9. PR of one calibrated rung (archive; density OK)

Keep this order. Capability map goes *under* the cell, never instead of it.

```
Title: E25 seq=1152 MATCH2 — dense K1, E21 not scored

This rung: dense 24.1% / 0 bits @3200. e18/e21 skipped.
S0 FAIL / K1. S1 not scored. 8k not run. Plot: learning_curves.png
Next: STOP MATCH2 2-item length extra-steps.

## Capability map
(INDEX / MATCH / SELECT / HOPS walls only — numbers that already live in a report)
```

"""Text capability checks — the definitions the runner, scorer and scorecard share (draft).

Spec: docs/engineering_specs/text_capability_checks.md. Skill: `text-checks`.

- TIERS: parameter target, token budget, training length, global batch, compute cap, layer shape.
- ARCHES: each round-1 model as trainer arguments (the same names as the training CLI) on top of
  the tier shape. FFN width is fitted per architecture so every model lands in the tier's
  parameter band (±5 %) — the one knob the protocol lets differ for parameter parity.
- Recipe cards: each architecture's own initialization / optimizer / schedule, from
  `evaluation/text_checks_recipes/<arch>.json` (priors until tuned; see §7 of the spec).
- Scoring rules (pass mark, controls, gating levels) for the scorecard.

Everything here is a draft to be calibrated on the first server runs (budgets, caps, shapes).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

TEXT_CHECKS_VERSION = "text-v1-draft"   # v1: exams on text-world-v2 (every guessing rate < 10 %), 2026-10-10
RECIPE_DIR = Path(__file__).resolve().parent / "text_checks_recipes"
PASS = 0.75
SHORTCUT_MARGIN = 0.10            # evidence-removed score must stay within floor + 10 points (or 2 SE)
GATING_TASKS = ("quote", "lookup", "keyed", "latest", "compose")   # text-core T1–T5
FRONTIER_TASKS = ("count", "deduce")                                  # T6, T7
CURVE_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
VOCAB = 4096

# Reading rules added by the protocol review of 2026-10-13 (spec §10.5). They change how a score is read, not how
# it is measured, so they apply to every result already in the ledgers.
# - Far evidence: single-fact exams place the fact early / middle / late (5 %, 50 %, 95 % of the document before
#   the question; balanced, item i gets bin i % 3). At 1k-4k the late fact sits 50-200 tokens before the question,
#   inside any local window, so a model with no long-range read answers those items. The memory score is the
#   score on early + middle facts ("far"); the late facts are reported as "near".
# - Effective floor: the best score of a model-free reader that does not do the task. Guessing one candidate
#   gives `floor`; for latest, tracking every move but ignoring whose it is gives 1 in 5 (audit text-world-v2).
# - Learned: above the effective floor by 10 points or 3 standard errors, whichever is larger. Pass (75 %) is
#   unchanged: "learned" says a skill switched on, "pass" says it is mastered.
DEPTH_FAR = ("early", "middle")
DEPTH_NEAR = ("late",)
PARTIAL_FLOOR = {"latest": 0.20}
LEARNED_MARGIN = 0.10
LEARNED_SE = 3.0
# Models with no length method on their recipe card ("none") are scored past the training length, but that score
# measures position extrapolation, not memory (dense: 0 % at 16k on every task, 2026-10-11).


def depth_counts(n: int) -> dict:
    """Items per depth bin in a cell of `n` items (the builder assigns bin i % 3: early, middle, late)."""
    return {"early": (n + 2) // 3, "middle": (n + 1) // 3, "late": n // 3}


def far_near(cell: dict | None) -> dict | None:
    """Far (early + middle facts) and near (late facts) exact scores of a single-fact cell, with item counts.

    Uses the scorer's own fields when present (scorer from 2026-10-13); for older results derives them from
    `by_depth` and the builder's bin assignment, and checks that every bin share is a whole number of items.
    None for exams without depth bins (latest, compose, count, deduce place their facts throughout)."""
    if not cell:
        return None
    if cell.get("far") is not None:
        return {k: cell.get(k) for k in ("far", "far_n", "near", "near_n")}
    bd = cell.get("by_depth") or {}
    if not all(d in bd for d in DEPTH_FAR + DEPTH_NEAR) or not cell.get("n"):
        return None
    cnt = cell.get("by_depth_n") or depth_counts(int(cell["n"]))
    if any(abs(bd[d] * cnt[d] - round(bd[d] * cnt[d])) > 1e-6 for d in cnt):
        return None  # bins not assigned as assumed: do not guess
    far_n = sum(cnt[d] for d in DEPTH_FAR)
    near_n = sum(cnt[d] for d in DEPTH_NEAR)
    return {"far": sum(bd[d] * cnt[d] for d in DEPTH_FAR) / far_n, "far_n": far_n,
            "near": sum(bd[d] * cnt[d] for d in DEPTH_NEAR) / near_n, "near_n": near_n}


def effective_floor(task: str, floor: float | None) -> float | None:
    """The guessing floor, raised to the best known partial reader for the task."""
    if floor is None:
        return PARTIAL_FLOOR.get(task)
    return max(floor, PARTIAL_FLOOR.get(task, 0.0))


def learned(score: float | None, n: int | None, task: str, floor: float | None) -> bool:
    """True when `score` (over `n` items) is clearly above the task's effective floor."""
    if score is None or not n:
        return False
    eff = effective_floor(task, floor) or 0.0
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    return score - eff > max(LEARNED_MARGIN, LEARNED_SE * se)


def headline(task: str, cell: dict | None) -> dict | None:
    """The score a cell is read by: far facts for single-fact exams, every item otherwise."""
    if not cell or cell.get("exact") is None:
        return None
    fn = far_near(cell)
    if fn:
        return {"score": fn["far"], "n": fn["far_n"], "near": fn["near"], "all": cell["exact"], "kind": "far"}
    return {"score": cell["exact"], "n": cell.get("n"), "near": None, "all": cell["exact"], "kind": "all"}


@dataclass(frozen=True)
class Tier:
    name: str
    target_params: float          # all trainable params incl. embeddings, head, memory
    tokens: float                 # training tokens (data budget)
    seq_len: int
    global_rows: int              # rows per optimizer step (fixed for every model and host)
    per_device_batch: int
    cap_gpu_hours: float          # compute cap: active training GPU-hours
    hidden: int
    pre_layers: int
    global_layers: int
    stack_layers: int
    eval_lengths: tuple = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    extra_lengths: tuple = (4096, 16384)    # harder / paraphrase splits
    tune_runs: int = 4
    tune_tokens: float = 60e6
    band: float = 0.05


TIERS = {
    "screen": Tier("screen", 30e6, 0.6e9, 4096, 96, 8, 24.0, hidden=576, pre_layers=2, global_layers=1,
                   stack_layers=5),
    # calibration ladder step 2 (spec §16.1, 2026-10-10): 50M, deeper rather than wider than the screen shape
    # (12 layers at width 576 vs 8), the step-1 budget
    "mid": Tier("mid", 50e6, 1.5e9, 4096, 96, 8, 40.0, hidden=576, pre_layers=2, global_layers=1,
                stack_layers=9, tune_runs=3, tune_tokens=60e6),
    "main": Tier("main", 100e6, 2.0e9, 4096, 96, 4, 120.0, hidden=896, pre_layers=3, global_layers=1,
                 stack_layers=8, tune_runs=3, tune_tokens=100e6),
    # local pipeline-check tier (Apple M5 Max laptop, ~1 h per model): confirms the pipeline works and
    # catches obvious errors only; no conclusions on learnability or budgets (those runs go to the servers).
    "lab": Tier("lab", 16e6, 100e6, 1024, 32, 16, 3.0, hidden=256, pre_layers=2, global_layers=1,
                stack_layers=4, eval_lengths=(1024, 2048, 4096), extra_lengths=(1024,), tune_runs=3,
                tune_tokens=12e6, band=0.10),
    # local plumbing check only (minutes on a laptop); never a result
    "smoke": Tier("smoke", 2e6, 0.25e6, 1024, 8, 4, 0.5, hidden=128, pre_layers=1, global_layers=1,
                  stack_layers=2, eval_lengths=(512, 1024, 2048), extra_lengths=(1024,), tune_runs=2,
                  tune_tokens=0.05e6, band=0.25),
}


def _base(t: Tier) -> dict:
    """Arguments every model shares at a tier (the E31 'as tested' input layer and windows)."""
    return {
        "model_family": "perceiver_ar", "objective_variant": "causal_lm", "decoder_type": "causal_ar",
        "hidden_size": t.hidden, "token_embedding_dim": 256 if t.hidden >= 256 else t.hidden,
        "head_dim": 64 if t.hidden >= 256 else 32, "num_kv_heads": 1,
        "par_pre_layers": t.pre_layers, "par_pre_window": 16, "par_global_layers": t.global_layers,
        "num_hidden_layers": t.stack_layers, "par_block": 16,
        "par_ngram_orders": "2,3", "par_ngram_buckets": 8192 if t.hidden >= 256 else 1024,
        "par_value_embed_layers": f"0,{t.pre_layers}", "par_value_embed_dim": 64 if t.hidden >= 256 else 16,
        "logit_softcap": 30.0, "z_loss": 1e-4,
    }


_E31C = {
    "par_mode": "perceiver", "message_write": "latent_memory", "message_raw_window": 256,
    "lm_read": "closed", "lm_context": "page_bidir", "lm_window": 256, "lm_stride": 192,
    "lm_latents": 32, "lm_latent_dim": 512, "lm_heads": 8, "lm_writer_dim": 256, "lm_enc_layers": 2,
    "lm_rounds": 2, "lm_reader_tokens": 1, "lm_addr": "none", "lm_slot_pos": "reader",
}
_E31C_SMOKE = {"lm_latents": 8, "lm_latent_dim": 64, "lm_heads": 2, "lm_writer_dim": 64, "lm_enc_layers": 1,
               "lm_rounds": 1}


@dataclass(frozen=True)
class Arch:
    name: str
    role: str
    args: dict = field(default_factory=dict)
    has_notebook: bool = False


ARCHES = {
    "dense": Arch("dense", "ceiling at the training length; language reference", {"par_mode": "dense"}),
    # diagnostic control (lab tier, 2026-10-09): the dense model without the E31 input layer (no hashed
    # n-gram features, no value embeddings) — does that layer delay learning to copy from context?
    "dense_plain": Arch("dense_plain", "diagnostic: dense without the n-gram / value-embedding input layer",
                        {"par_mode": "dense", "par_ngram_orders": "", "par_value_embed_layers": ""}),
    "local": Arch("local", "no-long-memory control: E31c trained with its notebook removed (recent 256 tokens only)",
                  {**_E31C, "message_override": "none"}),
    "e31c": Arch("e31c", "E31 latent notebook, closed-window text read", dict(_E31C), has_notebook=True),
    "e31c_loop": Arch("e31c_loop", "E31c + E33a read-think-reread loop (4 rounds)",
                      {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.3,
                       "message_loop_exit_targets": "answer"}, has_notebook=True),
    # E33d (2026-10-14): E33 + a page distance on the note read — one signed slope per head, starting at 0,
    # score −= s_h · log(1 + pages between the note and the reader); spec E33d_note_page_distance.md
    "e33d": Arch("e33d", "E33 + page distance on the notes (per-head log age bias, starts at 0)",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.3,
                  "message_loop_exit_targets": "answer", "lm_note_age": "log"}, has_notebook=True),
    # E33e (2026-10-16): E33 + the notes' order among matching notes — the same per-head strength, starting at 0,
    # score −= s_h · log(1 + soft count of newer notes that match about as well as the best note); a lone fact
    # pays nothing at any length. Spec E33e_note_rank.md
    "e33e": Arch("e33e", "E33 + rank among matching notes (per-head strength on the newer-match count, starts at 0)",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.3,
                  "message_loop_exit_targets": "answer", "lm_note_age": "rank"}, has_notebook=True),
    # E33f (2026-10-16): E33 trained as two agents — every "Question" starts a new side, so a question reads its
    # document ONLY through the notes (no raw tokens, no local window across it), and the page under the question
    # closes at it (its notes are readable at once). Warm-started from E33; spec E33f_notes_on_the_wire.md.
    # `message_boundary_token` names the token by its text; the runner looks its id up in the data's tokenizer.
    "e33f": Arch("e33f", "E33 in split mode: each question reads its document only through the notes",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.3,
                  "message_loop_exit_targets": "answer", "message_boundary_token": "Question"}, has_notebook=True),
    # Text multi-hop fixes (2026-10-17; note e33_text_multihop_diagnosis_20261017.md).
    # E33x: E33 with the loop's exit losses off. Every round's exit asked for the final answer (on every token), which
    # makes the rounds interchangeable and, on a two-step question, asks round 1 for an answer it cannot have read yet.
    "e33x": Arch("e33x", "E33 with the loop's exit losses off",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0}, has_notebook=True),
    # E33r: E33x + a per-round read target (needs data built with --train_read_targets): round r of a compose / deduce
    # answer is pushed to read fact r (its raw tokens or the notes of its pages); single-step questions (bridges,
    # lookup, keyed, quote) on round 1.
    "e33r": Arch("e33r", "E33 with exits off and a per-round read target",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5},
                 has_notebook=True),
    # E33o: E33r aimed at the hop's output word (data built with --train_read_targets output): round r is pushed onto
    # the notes whose latent read that word (the teacher's name, the place, the class word, the property), not any
    # note of its page. E33r reached the right page of "Every kavi is shiny." 95 % of the time and answered 12 %.
    "e33o": Arch("e33o", "E33 with exits off and a per-round read target on the hop's output word",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "message_loop_read_target": "note"}, has_notebook=True),
    # E33b: E33o + note read-back — each note's value must name the tokens it read (training-only head, zero-init).
    # E33r's notes on a property word did not encode which property it was (probe 11 %; on a place name 68 %).
    # Run 1 (17 Oct) used a random-init head at 0.2 and scrambled the warm-started values; from run 2: zero-init, 0.1.
    # E33g (2026-10-17, roadmap text_t4_t7_roadmap_20261017.md §4.1): E33r + order heads — E33e's rank among matching
    # notes at FIXED per-head strengths (five heads 0 for lookup / keyed / quote, four at 1, 2, 4, 8 for order) with the
    # tighter match gate. E33e's strengths, learned from 0, rose to 0.15 and fell back: story text never pays for order,
    # so one learned scalar per head stays near 0. Warm-started from E33r on the reasoning data.
    "e33g": Arch("e33g", "E33r + four fixed order heads (rank among matching notes at 1, 2, 4, 8; the other heads 0)",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "lm_note_age": "rank", "lm_note_rank_margin": 0.5, "lm_note_rank_sharp": 4.0,
                  "lm_note_age_init": "1,2,4,8", "lm_note_age_learn": "false"}, has_notebook=True),
    # E33h (2026-10-17, roadmap §4.5): E33r + two count heads — a summing read beside the global read. A softmax read
    # averages, so count shows only as a saturating share; here each head adds σ(a·q·k + b) over the notes (each token
    # of the document once: a page owns its first `stride` tokens) and the raw tokens no closed page owns yet, codes the
    # sum as a soft one-hot over 0..10, and adds it to the read through a zero-initialised projection.
    # E33gk (2026-10-17): E33g's fixed order heads + E33ok (note-level read target with the writer's key gradient
    # stopped). E33g's order term gave compose +8 / +6 and deduce +7 over its matched control (E33rb); E33ok's stop
    # kept the note content under the note-level target (place probe 80 %) and wrote the property word (33 %).
    "e33gk": Arch("e33gk", "E33g's order heads + E33ok's note-level read target with key-side losses off the writer",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_read_target": "note", "lm_key_grad": "detach",
                   "lm_note_age": "rank", "lm_note_rank_margin": 0.5, "lm_note_rank_sharp": 4.0,
                   "lm_note_age_init": "1,2,4,8", "lm_note_age_learn": "false"}, has_notebook=True),
    # E33ak / E33gak (2026-10-18): E33ok / E33gk with the key's own read of the page (lm_key_grad="address"). With
    # the plain stop, keys could only reuse the content read: E33ok wrote the property (deduce 13 → 36 % at 4k) but lost
    # compose (61 → 42 %) and long lookup (89 → 52 % at 16k), where a note's key must carry a name read elsewhere.
    # E33qn (2026-10-18): E33q with the round-to-round feedback off — the naming lens alone. Removing the feedback at
    # exam time left E33q's scores unchanged (compose 93 / 99 %, deduce 99 / 99 % at 4k / 16k; its gate stayed at
    # |mean| 0.009), so the simpler recipe is trained on its own.
    "e33qn": Arch("e33qn", "E33r + rounds name their hop through the shared LM head (no feedback)",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_name_lens": 0.3, "message_loop_feedback": "none"}, has_notebook=True),
    "e33ak": Arch("e33ak", "E33ok with the key's own read of the page",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_read_target": "note", "lm_key_grad": "address"}, has_notebook=True),
    "e33gak": Arch("e33gak", "E33gk with the key's own read of the page",
                   {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                    "message_loop_read_target": "note", "lm_key_grad": "address",
                    "lm_note_age": "rank", "lm_note_rank_margin": 0.5, "lm_note_rank_sharp": 4.0,
                    "lm_note_age_init": "1,2,4,8", "lm_note_age_learn": "false"}, has_notebook=True),
    "e33h": Arch("e33h", "E33r + two count heads (a summing read over notes and recent tokens)",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "lm_count_heads": 2}, has_notebook=True),
    # E33q (2026-10-17, roadmap §4.3; the decode-and-append oracle): E33r + the loop asks its next question with the name
    # it found. Round r names hop r's output word through the shared LM head at the decision position (weight 0.3; data
    # built with --train_read_targets output, whose spans also serve the page-level read target), and its LM-head
    # distribution, re-embedded with the shared token table and input layer, joins round r+1 through a gate from 0.
    # Oracle: E33r composes 58 % at 4k, 81 % with the gold bridge written into the question.
    "e33q": Arch("e33q", "E33r + rounds name their hop and pass the name to the next round's question",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "message_loop_name_lens": 0.3, "message_loop_feedback": "soft"}, has_notebook=True),
    # E33i (2026-10-18, roadmap §4.1): order on demand. E33g's fixed order heads left latest at 22 % at 4k; probes on its
    # weights: ordering every head with a wider match (margin 2) lifts the newest pick 29 → 50 % at 4k but costs lookup
    # 80 → 54 % and compose 68 → 44 %. So every head gets strength 4 on the rank term, the match margin is 2, and a
    # learned per-position switch σ(w·x + b) (w = 0, b = −2) scales it: a "now" question can turn order up, story text
    # and other questions keep it off. Warm-started from E33g with its order strengths reset (WARM_START_SKIP).
    "e33i": Arch("e33i", "E33g with order on demand: every head ranked, scaled by a learned per-question switch",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "lm_note_age": "rank", "lm_note_rank_margin": 2.0, "lm_note_rank_sharp": 2.0,
                  "lm_note_age_init": "4", "lm_note_age_learn": "false", "lm_note_order_gate": "true"},
                 has_notebook=True),
    # E33j (2026-10-18): E33q (rounds name their hop through the LM head) + E33i's order on demand, trained on data whose
    # latest questions carry a one-step chain: the asked person's LAST move (built with the 18 Oct generator, read
    # targets "output"). Round 1 must read that move's page and name its place, so the read target and the naming
    # supervise exactly the order choice that the rank term + per-question switch can make. E33q's naming took compose
    # 60 → 94 % and deduce 13 → 99 % at 4k (id split) with no order term.
    "e33j": Arch("e33j", "E33q + order on demand, trained with the latest question's last move as its read target",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "message_loop_name_lens": 0.3, "message_loop_feedback": "soft",
                  "lm_note_age": "rank", "lm_note_rank_margin": 2.0, "lm_note_rank_sharp": 2.0,
                  "lm_note_age_init": "4", "lm_note_age_learn": "false", "lm_note_order_gate": "true"},
                 has_notebook=True),
    # E33k (2026-10-18): E33h's count heads told the true number. E33h (answer loss only) stayed at 23 % at 4k. The data
    # now marks each count question's visits (read-target quads, round base 100), and each count head's sum at the
    # decision position must equal the number of visits (Huber, weight 0.1). Warm-started from E33h.
    "e33k": Arch("e33k", "E33h + the count heads' sum supervised with the true number of visits",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "lm_count_heads": 2, "lm_count_aux": 0.1}, has_notebook=True),
    # E33jf (2026-10-18): E33j with the order term always on (no per-question switch). E33i's switch never opened
    # (bias −2.03 after 0.3B tokens); here every question reads with order, and the latest supervision (read target +
    # naming on the last move) plus the other tasks' supervision teach the reader to live with it.
    "e33jf": Arch("e33jf", "E33j with the order term always on (no switch)",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_name_lens": 0.3, "message_loop_feedback": "soft",
                   "lm_note_age": "rank", "lm_note_rank_margin": 2.0, "lm_note_rank_sharp": 2.0,
                   "lm_note_age_init": "4", "lm_note_age_learn": "false"}, has_notebook=True),
    "e33b": Arch("e33b", "E33o + note read-back (each note's value names its tokens)",
                 {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                  "message_loop_read_target": "note", "lm_readback_aux": 0.1}, has_notebook=True),
    # E33rb: E33r (page-level read target) + note read-back. The note-level target of E33o / E33b emptied the notes
    # (a probe recovers the place from a note at the place name 14 % / 12 %, E33r 68 %, guessing 8 %; lookup at 16k
    # 87 → 0 / 18 %), so read-back is tested again on E33r's page target alone.
    "e33rb": Arch("e33rb", "E33r + note read-back (each note's value names its tokens)",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "lm_readback_aux": 0.1}, has_notebook=True),
    # E33ok / E33bk: E33o / E33b with the writer's key gradient stopped (lm_key_grad="detach"). A note's key and
    # value came from one latent read, so the note-level target pulled the latent at a place name toward the person
    # asked about and the place dropped out of its value; with the stop, read targets train the key projections only.
    "e33ok": Arch("e33ok", "E33o with key-side losses kept off the writer's read",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_read_target": "note", "lm_key_grad": "detach"}, has_notebook=True),
    "e33bk": Arch("e33bk", "E33b with key-side losses kept off the writer's read",
                  {**_E31C, "message_loop_rounds": 4, "message_loop_exit_aux": 0.0, "message_loop_read_aux": 0.5,
                   "message_loop_read_target": "note", "lm_readback_aux": 0.1, "lm_key_grad": "detach"},
                  has_notebook=True),
}
# Plain-input variants (2026-10-10): the same models without the word-pair/triple hash tables and value
# embeddings. Dense with plain input learned quote, lookup, keyed and latest at 30M / 1.5B tokens on v2, where
# dense with the tables learned at most one (spec §16.1), so the architecture round runs on plain input. FFN width
# is refitted, so each stays in the parameter band.
_PLAIN = {"par_ngram_orders": "", "par_value_embed_layers": ""}
for _k in ("local", "e31c", "e31c_loop", "e33d", "e33e", "e33f", "e33x", "e33r", "e33o", "e33b", "e33rb", "e33ok", "e33bk",
           "e33g", "e33h", "e33q", "e33i", "e33j", "e33jf", "e33k", "e33gk", "e33ak", "e33gak", "e33qn"):
    _a = ARCHES[_k]
    ARCHES[f"{_k}_plain"] = Arch(f"{_k}_plain", _a.role + "; plain input (no n-gram tables, no value embeddings)",
                                 {**_a.args, **_PLAIN}, has_notebook=_a.has_notebook)

# trainer argument → launcher env var (scripts/train_concept_pretraining_multigpu.sh)
ENV_OF = {
    "hidden_size": "HIDDEN_SIZE", "intermediate_size": "INTERMEDIATE_SIZE", "token_embedding_dim": "TOKEN_EMBEDDING_DIM",
    "num_hidden_layers": "NUM_LAYERS", "objective_variant": "OBJECTIVE_VARIANT", "decoder_type": "DECODER_TYPE",
    "model_family": "MODEL_FAMILY", "par_mode": "PAR_MODE", "par_pre_layers": "PAR_PRE_LAYERS",
    "par_pre_window": "PAR_PRE_WINDOW", "par_global_layers": "PAR_GLOBAL_LAYERS", "par_block": "PAR_BLOCK",
    "num_attention_heads": "NUM_ATTENTION_HEADS", "num_kv_heads": "NUM_KV_HEADS", "head_dim": "HEAD_DIM",
    "par_ngram_orders": "PAR_NGRAM_ORDERS", "par_ngram_buckets": "PAR_NGRAM_BUCKETS",
    "par_value_embed_layers": "PAR_VALUE_EMBED_LAYERS", "par_value_embed_dim": "PAR_VALUE_EMBED_DIM",
    "logit_softcap": "LOGIT_SOFTCAP", "z_loss": "Z_LOSS", "attn_backend": "ATTN_BACKEND",
    "message_write": "PAR_MESSAGE_WRITE", "message_raw_window": "PAR_MESSAGE_RAW_WINDOW",
    "message_override": "PAR_MESSAGE_OVERRIDE", "lm_read": "PAR_LM_READ", "lm_context": "PAR_LM_CONTEXT",
    "lm_window": "PAR_LM_WINDOW", "lm_stride": "PAR_LM_STRIDE", "lm_latents": "PAR_LM_LATENTS",
    "lm_latent_dim": "PAR_LM_LATENT_DIM", "lm_heads": "PAR_LM_HEADS", "lm_writer_dim": "PAR_LM_WRITER_DIM",
    "lm_enc_layers": "PAR_LM_ENC_LAYERS", "lm_rounds": "PAR_LM_ROUNDS", "lm_reader_tokens": "PAR_LM_READER_TOKENS",
    "lm_addr": "PAR_LM_ADDR", "lm_slot_pos": "PAR_LM_SLOT_POS", "lm_note_age": "PAR_LM_NOTE_AGE", "lm_note_rank_margin": "PAR_LM_NOTE_RANK_MARGIN",
    "lm_note_rank_sharp": "PAR_LM_NOTE_RANK_SHARP", "lm_note_age_init": "PAR_LM_NOTE_AGE_INIT",
    "lm_note_age_learn": "PAR_LM_NOTE_AGE_LEARN", "lm_count_heads": "PAR_LM_COUNT_HEADS",
    "lm_count_max": "PAR_LM_COUNT_MAX", "lm_count_aux": "PAR_LM_COUNT_AUX", "message_loop_name_lens": "PAR_LOOP_NAME_LENS",
    "message_loop_feedback": "PAR_LOOP_FEEDBACK", "message_loop_feedback_temp": "PAR_LOOP_FEEDBACK_TEMP",
    "lm_writer_input": "PAR_LM_WRITER_INPUT", "lm_note_rank_gate": "PAR_LM_NOTE_RANK_GATE",
    "lm_note_order_gate": "PAR_LM_NOTE_ORDER_GATE",
    "message_loop_rounds": "PAR_LOOP_ROUNDS",
    "message_loop_exit_aux": "PAR_LOOP_EXIT_AUX", "message_loop_exit_targets": "PAR_LOOP_EXIT_TARGETS",
    "message_loop_exit_mask": "PAR_LOOP_EXIT_MASK", "message_loop_read_aux": "PAR_LOOP_READ_AUX",
    "message_loop_read_target": "PAR_LOOP_READ_TARGET", "lm_readback_aux": "PAR_LM_READBACK_AUX",
    "lm_key_grad": "PAR_LM_KEY_GRAD",
    "message_boundary_token_id": "PAR_MESSAGE_BOUNDARY_TOKEN_ID",
}


def resolve_token_args(args: dict, tokenizer_dir) -> dict:
    """Arch args may name a token by its text (`message_boundary_token`: "Question"); its id comes from the data's
    own tokenizer, so one arch card serves every data version."""
    if "message_boundary_token" not in args:
        return args
    from transformers import AutoTokenizer

    a = dict(args)
    word = a.pop("message_boundary_token")
    ids = AutoTokenizer.from_pretrained(str(tokenizer_dir)).encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"message_boundary_token {word!r} is {len(ids)} tokens in {tokenizer_dir}; it must be one")
    a["message_boundary_token_id"] = ids[0]
    return a


def model_args(arch: str, tier: str, intermediate_size: int | None = None) -> dict:
    t = TIERS[tier]
    a = ARCHES[arch]
    args = {**_base(t), **a.args}
    if tier == "smoke" and "lm_latents" in args:
        args.update(_E31C_SMOKE)
    args["num_attention_heads"] = t.hidden // args["head_dim"]
    args["intermediate_size"] = intermediate_size or _round64(t.hidden * 8 / 3)
    return args


def _round64(x: float) -> int:
    return max(64, int(round(x / 64.0)) * 64)


def count_params(args: dict, vocab: int = VOCAB) -> int:
    """Exact trainable parameter count, built through the trainer's own factory on the meta device."""
    import torch
    from types import SimpleNamespace

    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "sdpa", False
    tok = type("T", (), {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "__len__": lambda self: vocab})()
    with torch.device("meta"):
        model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="text-checks"))
    # training-only heads that the trained model never runs (the note read-back) are not part of its size
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith(TRAIN_ONLY))


TRAIN_ONLY = ("note_readback.",)


def fit_params(arch: str, tier: str, vocab: int = VOCAB) -> tuple[dict, int]:
    """Pick the FFN width (multiple of 64) that brings this model closest to the tier's target."""
    t = TIERS[tier]
    lo, hi = 64, _round64(t.hidden * 12)
    best = None
    while lo <= hi:  # params grow monotonically with the FFN width
        mid = _round64((lo + hi) / 2)
        n = count_params(model_args(arch, tier, mid), vocab)
        if best is None or abs(n - t.target_params) < abs(best[1] - t.target_params):
            best = (mid, n)
        if n < t.target_params:
            lo = mid + 64
        else:
            hi = mid - 64
    width, n = best
    return model_args(arch, tier, width), n


def in_band(n: int, tier: str) -> bool:
    t = TIERS[tier]
    return abs(n - t.target_params) <= t.band * t.target_params


DEFAULT_RECIPE = {
    "status": "prior (untuned)",
    "init": "trainer default: normal(0.02) projections, zero-init residual outputs (perceiver_ar)",
    "optimizer": "adam", "weight_decay": 0.1, "max_grad_norm": 1.0,
    "lr": {"screen": 1e-3, "mid": 1e-3, "main": 6e-4, "smoke": 1e-3, "lab": 1e-3},
    "warmup_frac": 0.02, "scheduler": "cosine",
    "length_method": "none",
    "tuning": {},
}


def load_recipe(arch: str) -> dict:
    path = RECIPE_DIR / f"{arch}.json"
    card = json.loads(path.read_text()) if path.exists() else {}
    out = json.loads(json.dumps(DEFAULT_RECIPE))
    for k, v in card.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def steps_for(tokens: float, global_rows: int, mean_row_tokens: float) -> int:
    """Optimizer steps that consume `tokens`. Every model reads the same rows in the same order,
    so the same step count is the same data for all."""
    return max(1, int(math.ceil(tokens / (global_rows * mean_row_tokens))))


def per_device_batch(recipe: dict, tier: str) -> int:
    """Rows per GPU micro-batch: the tier default unless the model's recipe card lowers it to fit memory
    (the global batch, hence the data and the optimizer steps, stay the same through accumulation)."""
    return int((recipe.get("per_device_batch") or {}).get(tier, TIERS[tier].per_device_batch))


def grad_accum(tier: str, n_gpus: int, pdb: int | None = None) -> int:
    t = TIERS[tier]
    pdb = pdb or t.per_device_batch
    per_step = pdb * n_gpus
    if t.global_rows % per_step:
        raise ValueError(f"global batch {t.global_rows} rows is not a multiple of {pdb} × {n_gpus} GPUs")
    return t.global_rows // per_step

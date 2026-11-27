"""E33n runner: the four arms × sizes × phases of the LSA-shaped protocol, as trainer commands (one GPU per run).

Spec: docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md · plan: E33n_plan.md.

  arms   window  — reading layer sees only its last 512 raw tokens (E31c writer built, notes switched off)
         dense   — reading layer: full causal attention over every earlier raw token (E18-style raw global read)
         dense_rope — dense with RoPE on the reading layer (every other arm's read has no position code)
         e33     — E33: last 512 raw tokens + every note of every closed 256-token page (full note read)
         e33n    — E33n: last 512 raw tokens + the 64 best notes per query and head (top-k note read)
         notes_open / pages_open — E33p: tied note keys; each query opens the raw tokens of its 64 best notes'
                   members (its 2 best pages), ≤ 512 tokens, plus every note as gist and the 512-token window
         lsa     — E33p: LSA's released flat read on the dense arm (sinks + window + 512 keys from its best buckets)
         catalogue / catalogue_2l — E33t: the notes only choose what opens (4 learned empty slots, no summaries in the
                   read), balanced filing (≤ 16 per note), max-head cosine, up to 128 notes until 512 tokens, a new
                   place per loop pass, an index loss; pages of 256 without overlap; _2l searches pages first
         catalogue_reuse — E33t: the catalogue with the pass exclusion off (a loop pass may re-open notes earlier
                   passes opened). The planted-password probe (2026-11-07) found the catalogue opens the key on
                   pass 1 (94 % at 4k) but may not return to it on passes 2–4, where the LSA port and dense copy it
         catalogue_reuse_n4 / _n16 / _p512 — E33t stage B: note and page sizes around the stage-A winner
                   (catalogue_reuse: 256-token pages, 32 notes = 8 tokens per note): 4 and 16 tokens per note, and
                   512-token pages with 64 notes (8 per note); everything else unchanged
         Note (2026-11-06): before then the trainer passed the loop only to notebook models, so E33n's dense arm
         trained with one round; it now loops 4 times like every other arm.
         All: E33's loop (prelude → [reading layer + 1 local layer] × 4 tied → coda), exits off, local windows 128,
         plain input (no n-gram tables / value embeddings), GPT-2 vocabulary (50,257), tied factorised embedding.
  sizes  s30 (30M) · m100 (100M) · l300 (300M) — FFN width fitted to the target with the trainer's own factory.
  phases trunk — FineWeb-Edu GPT-2 shards, rows of 4,096 (documents packed, no cross-document attention),
                 warmup then constant learning rate (the WSD "stable" part; LSA branches its decays from it);
         decay — from the trunk's final weights, fresh optimizer, LongDocs-lite rows of 16,384 (one document per
                 row), warmup 3 % then 1−sqrt decay over 97 % to 10 % of the peak (LSA's 32k decay shape).
         long  — E33t stage C (plan §10.6): from the decay's final weights, fresh optimizer, LongDocs-lite rows of
                 65,536 (one document per row), 0.1B tokens at s30, the same tokens per step as the decay, picking up
                 at the rate the decay ended on (10 % of the peak) with a 3 % warmup, then 1−sqrt to 1 % of the peak.
                 Its own row order (manifest seed 2042) and, by default, trainer seed 1 for every arm, so all arms
                 see the same rows. --grad_ckpt recomputes each layer in the backward (24 GB cards at 64k).
         sft   — E37 instruction tuning (docs/experiments_specs/ahead/E37_sft_sensitivity.md): from a decay's final
                 weights, fresh optimizer, the mix of scripts/build_sft_mix.py (--sft_manifest: smol-smoltalk chats
                 packed into 16,384-token rows, loss on the answers, plus 10 % LongDocs-lite replay), 16 rows per step,
                 200M tokens, the decay's schedule shape at a peak LR given by --lr (the pre-flight's pick); evaluates
                 the held-out answer loss (the manifest's eval rows) before the first step and at the last.
         split — E37c (docs/experiments_specs/done_failed/E37c_two_agent_split_training.md): from the decay's final weights,
                 fresh optimizer, the decay's rows of 16,384 in another order (manifest seed 3042), every row cut at
                 random points (gaps log-uniform in [--cut_min, --cut_max]): a local layer reads its own side of a cut
                 only, the reading layer the whole document. The long phase's rate (10 % of the peak → 1 %), 0.2B
                 tokens at m100. Arms without a notebook only (dense, lsa, e37a, e37b).

Usage (on a node, from the repo root):
  uv run python scripts/run_e33n.py cmd --arm e33n --size s30 --phase trunk --data /opt/dlami/nvme/data --out OUT
  uv run python scripts/run_e33n.py params            # parameter counts of every arm × size
  uv run python scripts/run_e33n.py cmd --arm lsa --size m100 --phase sft --data D --out OUT \
      --init OUT_DECAY/decay/final --sft_manifest D/sft_smoltalk_16k/manifest_sft.json --lr 1.5e-4
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOCAB = 50257
EOT = 50256

SIZES = {
    #        hidden  e    pre  stack  target   trunk tok  decay tok  long tok  trunk rows/step (×4k)  lr peak
    "s30":  dict(hidden=576,  e=256, pre=2, stack=5, target=30e6,  trunk=1.5e9, decay=0.2e9, long=0.1e9, rows=64,
                 lr=1.0e-3, split=0.1e9),
    "m100": dict(hidden=896,  e=384, pre=3, stack=7, target=100e6, trunk=3.0e9, decay=0.4e9, long=0.2e9, rows=128,
                 lr=6.4e-4, split=0.2e9),
    "l300": dict(hidden=1280, e=512, pre=4, stack=9, target=300e6, trunk=5.0e9, decay=0.6e9, long=0.3e9, rows=256,
                 lr=4.5e-4, split=0.3e9),
    "tiny": dict(hidden=128,  e=64,  pre=1, stack=2, target=8e6,   trunk=2e6,   decay=1e6,   long=0.5e6, rows=4,
                 lr=1.0e-3, split=0.5e6),
}
# Tiny-ML leaderboard sizes (2026-11-14, author: "SmolLM2, bigger body"): the output reuses the input token table
# (lm_head_tied), the SmolLM2 vocabulary is counted, and the body grows to fill the board's 150M total-parameter cap
# with a margin. Sized by size_report() below; data comes from a built mix (--mix), so the token budgets are the mix's.
SIZES["m120t"] = dict(hidden=896, e=384, pre=3, stack=9, target=None, ffn=2816, trunk=8.0e9, decay=3.0e9, long=0.5e9,
                      rows=128, lr=6.0e-4, tied=True, vocab=49152)

_LOOP = {"message_loop_rounds": 4, "message_loop_exit_aux": 0.0}  # loop span 1 is the default
_WRITER = {
    "par_mode": "perceiver", "message_write": "latent_memory", "message_boundary_token_id": -1,
    "message_raw_window": 512, "lm_read": "closed", "lm_context": "page_bidir", "lm_window": 256, "lm_stride": 192,
    "lm_latents": 32, "lm_latent_dim": 512, "lm_heads": 8, "lm_writer_dim": 256, "lm_enc_layers": 2, "lm_rounds": 2,
    "lm_reader_tokens": 1, "lm_addr": "none", "lm_slot_pos": "reader",
}
ARMS = {
    "window": {**_WRITER, **_LOOP, "message_override": "none"},
    "dense": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP},
    # E37 fair full-attention control (author 2026-11-13: "Yes, start checks"): the dense arm with RoPE on the reading
    # layer too (rope_theta 500k, as the local layers); "dense" reads without a position code (par_global_nope, as
    # every read here), which spreads thin past twice its 16k training length
    "dense_rope": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_nope": "false"},
    "e33":{**_WRITER, **_LOOP},
    "e33n": {**_WRITER, **_LOOP, "lm_read_topk": 64},
    # E33p (docs/experiments_specs/ahead/E33p_note_indexed_exact_read.md): the notes as an index, per-query choice
    "notes_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 64, "lm_open_cap": 512},
    "pages_open": {**_WRITER, **_LOOP, "lm_note_key": "tied", "lm_open": "pages", "lm_open_k": 2, "lm_open_cap": 512},
    # LSA's released flat read on the dense arm's reading layer (nn/lsa_buckets.py), budget matched at 512 keys
    "lsa": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "lsa",
            "lsa_window": 512, "lsa_sinks": 4, "lsa_block": 512, "lsa_bucket": 8, "lsa_pool": 4, "lsa_bucket_cap": 16,
            "lsa_top_k": 512},
    # E33t (docs/experiments_specs/ahead/E33t_catalogue_read.md): the notes only choose what opens; balanced filing,
    # max-head cosine, budget filled by tokens (up to 128 notes), a new place per pass, learned empty slots, index loss
    "catalogue": {**_WRITER, **_LOOP, "lm_stride": 256, "lm_note_key": "tied", "lm_open": "notes", "lm_open_k": 128,
                  "lm_open_cap": 512, "lm_note_read": "index", "lm_filing": "balanced", "lm_open_score": "max_cos",
                  "lm_open_new_per_pass": True, "lm_sinks": 4, "lm_index_loss": 0.1},
    # E37 (docs/experiments_specs/ahead/E37_concept_memory.md): concept memory on the dense arm's reading layer —
    # balanced content concepts of 8 keys per 512-key block, exact reads of the largest estimated contributions
    # (512 tokens), one summary entry per other concept; e37b adds the looped concept reasoner
    "e37a": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "concept",
             "concept_block": 512, "concept_per_block": 64, "concept_sinks": 4, "concept_top_k": 512},
    "e37b": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "concept",
             "concept_block": 512, "concept_per_block": 64, "concept_sinks": 4, "concept_top_k": 512,
             "concept_reasoner_layers": 2, "concept_reasoner_dim": 512, "concept_reasoner_heads": 8,
             "concept_reasoner_loops": 2},
    # E37 compression ladder: e37a with coarser concepts (32 and 128 keys per concept); the same 512 opened tokens,
    # so only the far field (the summaries) gets coarser — a 10M context needs few, rich summaries
    "e37a_c32": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "concept",
                 "concept_block": 512, "concept_per_block": 16, "concept_sinks": 4, "concept_top_k": 512},
    "e37a_c128": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "concept",
                  "concept_block": 512, "concept_per_block": 4, "concept_sinks": 4, "concept_top_k": 512},
    # E39 (docs/experiments_specs/ahead/E39_living_concepts.md): living concepts on the dense arm's reading layer —
    # contiguous spans cut where adjacent keys stop resembling each other (mean 8 tokens), document-wide concepts that
    # each span joins or founds (1 span in 8 founds), concepts first then their spans by match, 512 opened tokens, one
    # summary per concept for what was not opened; living_spans: every span its own concept, searched flat
    "living": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "living",
               "living_stride": 512, "living_span_max": 16, "living_span_cap": 128, "living_keep": 32,
               "living_levels": 2, "concept_sinks": 4, "concept_top_k": 512},
    "living_spans": {"par_mode": "perceiver", "message_boundary_token_id": -1, **_LOOP, "par_global_read": "living",
                     "living_stride": 512, "living_span_max": 16, "living_span_cap": 128, "living_levels": 1,
                     "concept_sinks": 4, "concept_top_k": 512},
}
# E39a (docs/experiments_specs/ahead/E39a_living_flat.md, author 2026-11-26): living concepts with every visible span
# competing for the 512 opened tokens in training too (the test switch of E39's R2); same weights shape, so stages 2-3
# start from E39's stage-1 weights
ARMS["living_flat"] = {**ARMS["living"], "living_search": "flat"}
# E41 (docs/experiments_specs/ahead/E41_tree_concept_model.md): the tree concept model. Concept memory's concepts as
# the leaves of a tree of exact summaries (8 nodes per container, containers of 512 → 4k → 32k … tokens), a reasoner
# over the tree's nodes (key and value corrections, a plan per block, the next-concept prediction), a top-down read,
# and closed-book training rows (40 % as normal, 20 % bounded to 16 nodes per level, 40 % cut at a drawn level).
# Every new part starts as an identity: a warm start from e37a's weights is e37a at step 0.
# tree_read: the read and the closed-book rows only (no new weights); tree_noaux: + the reasoner, no extra losses.
ARMS["tree_read"] = {**ARMS["e37a"], "concept_tree": "true", "tree_row_mix": "0.4,0.2,0.4"}
# The reasoner: two layers of width 384 looped twice (with the plan and the two prediction heads about 7.3M weights;
# width 512 would put m120t at 153M, over the board's 150M cap).
ARMS["tree_noaux"] = {**ARMS["tree_read"], "tree_reasoner_layers": 2, "tree_reasoner_dim": 384,
                      "tree_reasoner_heads": 6, "tree_reasoner_loops": 2, "tree_plan": "true"}
ARMS["tree"] = {**ARMS["tree_noaux"], "tree_next_concept": 0.1, "tree_route_loss": 0.1}
# recipe variants (home retrofits pick run 2's): fewer closed-book rows; the read drawn per step (several cards)
ARMS["tree_soft"] = {**ARMS["tree"], "tree_row_mix": "0.7,0.15,0.15"}
ARMS["tree_step"] = {**ARMS["tree"], "tree_row_draw": "step"}
# every row read as normal (no closed-book or bounded rows): the reasoner, the plan and the two losses on the
# recall recipe's own read; with `tree` it separates what the closed-book rows do from what the reasoner does
ARMS["tree_flat"] = {**ARMS["tree"], "tree_row_mix": ""}
# run 2 (2026-11-27): the tree read and the closed-book rows without the reasoner (home retrofits: closed-book rows
# alone cost the normal read +0.002, the reasoner with its losses +0.021), the read drawn per step
ARMS["tree_read_step"] = {**ARMS["tree_read"], "tree_row_draw": "step"}
# the reasoner taken apart (home retrofits): values only (no key corrections), no plan, no extra losses …
ARMS["tree_vals"] = {**ARMS["tree_read"], "tree_reasoner_layers": 2, "tree_reasoner_dim": 384,
                     "tree_reasoner_heads": 6, "tree_reasoner_loops": 2, "tree_reasoner_keys": "false"}
# … and the full reasoner with small extra losses that do not reach the token layers
ARMS["tree_gentle"] = {**ARMS["tree_noaux"], "tree_next_concept": 0.02, "tree_route_loss": 0.02,
                       "tree_aux_detach": "true"}
# values and the plan from the reasoned state (no key corrections, no losses): is the plan or the keys the cost?
ARMS["tree_vp"] = {**ARMS["tree_vals"], "tree_plan": "true"}
# values, and a plan read from the next-concept predictor, which predicts the block that reads it (two ahead); the
# prediction loss trains the predictor's own layers only, the next-word loss decides how the plan is used
ARMS["tree_pred"] = {**ARMS["tree_vals"], "tree_plan": "true", "tree_plan_from": "pred", "tree_predict_ahead": 2,
                     "tree_next_concept": 0.1, "tree_aux_detach": "true"}
# 2026-11-27 02:16 UTC: the plan is what costs (values + plan +0.009, plan from the predictor +0.023, values only
# +0.002). Without any plan: key and value corrections (tree_kv), and the same with both losses shaping the
# reasoned states the corrections are read from (tree_np: the next-concept loss as the author meant it)
ARMS["tree_kv"] = {**ARMS["tree_vals"], "tree_reasoner_keys": "true"}
ARMS["tree_np"] = {**ARMS["tree_kv"], "tree_next_concept": 0.1, "tree_route_loss": 0.1}
# wide summaries (2026-11-27): a summary hands a reader 64 numbers shared by all 14 heads; here every head reads its
# own slice of a learned linear read of the node's mean state (896 numbers), nothing else new
ARMS["tree_wide"] = {**ARMS["tree_read"], "tree_wide": "true"}
# run 3: tree_np with the read drawn per step (several cards), next to run 2's tree_read_step
ARMS["tree_np_step"] = {**ARMS["tree_np"], "tree_row_draw": "step"}
ARMS["catalogue_2l"] = {**ARMS["catalogue"], "lm_open_levels": 2, "lm_open_pages": 16}  # pages first, then notes
# E40 (docs/experiments_specs/done_failed/E40_depth_mixed_read.md): concept memory with DeepCrossAttention's GRN-v3 depth
# mixes for the query, key and value inputs (nn/depth_mix.py): the reading layer's runs only, or 2-DCA on every run.
# Identity at initialisation: a warm start from e37a's weights is e37a at step 0.
ARMS["e40_read"] = {**ARMS["e37a"], "par_depth_mix": "read"}
ARMS["e40_dca2"] = {**ARMS["e37a"], "par_depth_mix": "dca2"}
ARMS["catalogue_reuse"] = {**ARMS["catalogue"], "lm_open_new_per_pass": False}  # every pass may re-open any note
# E33t stage B (approved by the author 2026-11-08): note and page sizes around catalogue_reuse; the read's budget stays
# 512 tokens (up to 128 notes still fills it at 4 tokens per note); the filing cap 16 is exact at 16 tokens per note
ARMS["catalogue_reuse_n4"] = {**ARMS["catalogue_reuse"], "lm_latents": 64}                     # 4 tokens per note
ARMS["catalogue_reuse_n16"] = {**ARMS["catalogue_reuse"], "lm_latents": 16}                    # 16 tokens per note
ARMS["catalogue_reuse_p512"] = {**ARMS["catalogue_reuse"], "lm_window": 512, "lm_stride": 512, "lm_latents": 64}
# E37 SFT: rows per optimizer step (16 × 16,384 = 262k tokens) and tokens of the stage, at every size
SFT = dict(rows=16, tokens=200e6)

TINY_OPEN = {"notes_open": {"lm_open_k": 4, "lm_open_cap": 64}, "pages_open": {"lm_open_k": 1, "lm_open_cap": 64},
             "lsa": {"lsa_window": 64, "lsa_block": 64, "lsa_top_k": 64},
             "catalogue": {"lm_open_k": 16, "lm_open_cap": 64, "lm_index_queries": 32, "lm_index_negatives": 4,
                           "lm_stride": 64},
             "catalogue_2l": {"lm_open_k": 16, "lm_open_cap": 64, "lm_index_queries": 32, "lm_index_negatives": 4,
                              "lm_stride": 64, "lm_open_pages": 2},
             "e37a": {"concept_block": 64, "concept_per_block": 8, "concept_top_k": 64},
             "e37b": {"concept_block": 64, "concept_per_block": 8, "concept_top_k": 64, "concept_reasoner_dim": 64,
                      "concept_reasoner_heads": 2},
             # E39 tiny (laptop pipeline checks, sdpa): 64-token strides of spans up to 8 tokens
             "living": {"living_stride": 64, "living_span_max": 8, "living_span_cap": 16, "living_keep": 8,
                        "living_width": 16, "concept_top_k": 64},
             "living_spans": {"living_stride": 64, "living_span_max": 8, "living_span_cap": 16, "living_width": 16,
                              "concept_top_k": 64}}
TINY_OPEN["living_flat"] = TINY_OPEN["living"]
# E41 tiny: 64-token blocks of 8 concepts, 2 nodes per container, containers of 2
TINY_OPEN["tree_read"] = {**TINY_OPEN["e37a"], "tree_probes": 2, "tree_fan": 2, "tree_train_follow": 2}
TINY_OPEN["tree_noaux"] = {**TINY_OPEN["tree_read"], "tree_reasoner_dim": 64, "tree_reasoner_heads": 2}
TINY_OPEN["tree"] = {**TINY_OPEN["tree_noaux"], "tree_codes_coarse": 64, "tree_codes_fine": 128, "tree_code_dim": 16,
                     "tree_route_queries": 32}
TINY_OPEN["tree_soft"] = TINY_OPEN["tree_step"] = TINY_OPEN["tree_flat"] = TINY_OPEN["tree"]
TINY_OPEN["tree_read_step"] = TINY_OPEN["tree_read"]
TINY_OPEN["tree_vals"] = TINY_OPEN["tree_noaux"]
TINY_OPEN["tree_gentle"] = TINY_OPEN["tree_pred"] = TINY_OPEN["tree"]
TINY_OPEN["tree_vp"] = TINY_OPEN["tree_kv"] = TINY_OPEN["tree_noaux"]
TINY_OPEN["tree_np"] = TINY_OPEN["tree_np_step"] = TINY_OPEN["tree"]
TINY_OPEN["tree_wide"] = TINY_OPEN["tree_read"]
TINY_OPEN["catalogue_reuse"] = TINY_OPEN["catalogue"]
TINY_OPEN["catalogue_reuse_n4"] = {**TINY_OPEN["catalogue"], "lm_latents": 16}               # tiny pages are 64
TINY_OPEN["catalogue_reuse_n16"] = {**TINY_OPEN["catalogue"], "lm_latents": 4}
TINY_OPEN["catalogue_reuse_p512"] = {**TINY_OPEN["catalogue"], "lm_window": 128, "lm_stride": 128, "lm_latents": 16}
TINY_WRITER = {"lm_latents": 8, "lm_latent_dim": 64, "lm_heads": 2, "lm_writer_dim": 64, "lm_enc_layers": 1,
               "lm_rounds": 1, "message_raw_window": 64, "lm_window": 64, "lm_stride": 48}


def model_args(arm: str, size: str, ffn: int | None = None) -> dict:
    z = SIZES[size]
    a = {
        "model_family": "perceiver_ar", "objective_variant": "causal_lm", "decoder_type": "causal_ar",
        "hidden_size": z["hidden"], "token_embedding_dim": z["e"], "head_dim": 64 if z["hidden"] >= 256 else 32,
        "num_kv_heads": 1, "par_pre_layers": z["pre"], "par_pre_window": 128 if size != "tiny" else 32,
        "par_global_layers": 1, "num_hidden_layers": z["stack"], "par_block": 128 if size != "tiny" else 32,
        "par_global_nope": "true", "par_ngram_orders": "", "par_value_embed_layers": "",
        "logit_softcap": 30.0, "z_loss": 1e-4,
    }
    a.update(ARMS[arm])
    if z.get("tied"):
        a["lm_head_tied"] = "true"
    if size == "tiny" and "lm_latents" in a:
        a.update(TINY_WRITER)
        if "lm_read_topk" in a:
            a["lm_read_topk"] = 4
    if size == "tiny" and arm in TINY_OPEN:
        a.update(TINY_OPEN[arm])
    if a.get("par_global_read") in ("concept", "living"):
        # E37 length scale (2026-11-08): past the long decay's row length the summaries' total weight is held at
        # that length's level, so the trained model reads longer documents without the fade. Inert in training.
        # E39 keeps it unchanged (the spec: the length scale is one of the parts kept on purpose).
        a.setdefault("concept_sum_len", decay_len(size))
    a["num_attention_heads"] = z["hidden"] // a["head_dim"]
    a["intermediate_size"] = ffn or max(64, round(z["hidden"] * 8 / 3 / 64) * 64)
    return a


def count(args: dict, non_embedding: bool = True, vocab: int = VOCAB) -> int:
    """Trainable parameters through the trainer's own factory (meta device). Sizes are named by the NON-embedding
    count (Kaplan et al. 2020's convention): the GPT-2 vocabulary puts ~V·(e + H) parameters in the input table and
    the untied output layer (29M + 13M at width 576), which would otherwise be most of a "30M" model."""
    import torch
    from types import SimpleNamespace

    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in args.items() if k in fields})
    ma.attn_backend, ma.use_liger = "sdpa", False
    tok = type("T", (), {"pad_token_id": EOT, "bos_token_id": EOT, "eos_token_id": EOT, "__len__": lambda self: vocab})()
    with torch.device("meta"):
        model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=4096, tokenizer_name="gpt2"))
    skip = ("embed.", "lm_head.") if non_embedding else ()
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith(skip))


def fit(arm: str, size: str) -> tuple[dict, int]:
    """FFN width (multiple of 64) bringing the WINDOW arm's non-embedding count closest to the target; every arm of a
    size uses that width, so arms differ only in the reading layer (the window arm carries the writer too). A size
    with a fixed "ffn" (the leaderboard sizes) uses it as given."""
    z = SIZES[size]
    if z.get("ffn"):
        args = model_args(arm, size, int(z["ffn"]))
        return args, count(args, vocab=int(z.get("vocab", VOCAB)))
    lo, hi, best = 64, z["hidden"] * 12, None
    while lo <= hi:
        mid = max(64, round((lo + hi) / 2 / 64) * 64)
        n = count(model_args("window", size, mid))
        if best is None or abs(n - z["target"]) < abs(best[1] - z["target"]):
            best = (mid, n)
        lo, hi = (mid + 64, hi) if n < z["target"] else (lo, mid - 64)
    args = model_args(arm, size, best[0])
    return args, count(args)


def decay_len(size: str) -> int:
    """Row length of the long decay (LongDocs-lite)."""
    return 1024 if size == "tiny" else 16384


def manifests(data: Path, out: Path, size: str) -> dict:
    """Write the token-shard manifests (trunk / decay / long / split) for a run under `out`, return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    fw, ld = data / "fineweb_edu_gpt2", data / "longdocs_lite_gpt2"
    tiny = size == "tiny"
    trunk = {"seq_len": 512 if tiny else 4096, "rule": "stream", "eot": EOT, "seed": 42,
             "train": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_train_*.bin")], "weight": 1.0}],
             "eval": [{"name": "fineweb_edu", "files": [str(fw / "finewebedu_val_*.bin")], "rows": 256}]}
    srcs = json.loads((ld / "manifest.json").read_text())["sources"] if (ld / "manifest.json").exists() else {}
    decay = {"seq_len": decay_len(size), "rule": "doc", "eot": EOT, "seed": 1042,
             "train": [{"name": s, "files": [str(ld / f"{s}_train_*.bin")], "weight": 1.0} for s in srcs],
             "eval": [{"name": s, "files": [str(ld / f"{s}_val_*.bin")], "rows": 16} for s in srcs]}
    long = {**decay, "seq_len": 2048 if tiny else 65536, "seed": 2042,
            "eval": [{**e, "rows": 4} for e in decay["eval"]]}
    split = {**decay, "seed": 3042}  # E37c: the decay's rows, another order
    p = {k: out / f"manifest_{k}.json" for k in ("trunk", "decay", "long", "split")}
    for k, m in (("trunk", trunk), ("decay", decay), ("long", long), ("split", split)):
        p[k].write_text(json.dumps(m, indent=2))
    return p


def mix_supply(mix: dict) -> dict:
    """What a built mix can feed each phase, from mix.json alone (the builder's supply report). Per phase: the rows
    one pass over the data holds — data.bin_token_dataset.BinRowDataset keeps the sources in the proportion of
    their weights, so the scarcest source relative to its weight sets the size of a pass — and how many passes the
    phase's token budget makes. A source with no row at the phase's length is listed under "empty"."""
    out = {}
    for name, ph in mix["phases"].items():
        seq, tokens = int(ph["seq_len"]), float(ph["tokens"])
        avail = {s["name"]: s.get("rows_available") for s in ph.get("supply", [])}
        w = {s["name"]: float(s["weight"]) for s in ph["train"]}
        # no report (a hand-written mix.json): nothing is known to be empty, and the size of a pass is unknown
        empty = [n for n in w if n in avail and not avail[n]]
        live = {n: w[n] for n in w if avail.get(n)}
        tot_w = sum(live.values()) or 1.0
        per = {n: avail[n] / (live[n] / tot_w) for n in live}
        limit = min(per, key=per.get) if per else None
        rows = int(per[limit]) if limit else 0
        ev = {s["name"]: s.get("rows_available") for s in ph.get("eval_supply", [])}
        out[name] = {"seq_len": seq, "tokens": tokens, "rows_per_pass": rows, "tokens_per_pass": rows * seq,
                     "passes": round(tokens / (rows * seq), 2) if rows else None, "limited_by": limit,
                     "empty": empty, "eval_empty": [n for n, r in ev.items() if not r],
                     "rows_available": avail}
    return out


def mix_manifests(mix_dir: Path, out: Path, lenient: bool = False) -> tuple[dict, dict]:
    """Phase manifests of a built data mix (scripts/build_m21_mix.py: DIR/mix.json, DIR/tokenizer/,
    DIR/<source>_train_*.bin and <source>_val_*.bin). Returns (manifest paths per phase, the mix).

    A "stream" source that several phases read (web text in the trunk and again, as replay, in the long phases) is
    split by shards: the last phase takes shards from the end until it has its share of that phase's tokens (+3 %),
    the one before takes the next ones, the trunk keeps the rest — no token is read in two phases. "doc" sources
    (long documents) are shared: the 64k phase reads the documents long enough for it, as E37's 64k stage did.

    An evaluation source with no row at a phase's length (the builder's eval_supply) is left out of that phase's
    evaluation. A training source with no row or no shard is an error — unless `lenient` (small check builds): then
    it is left out with a note, and a shared stream source with too few shards to split is read by every phase."""
    import glob
    import os

    mix_dir = Path(mix_dir)
    mix = json.loads((mix_dir / "mix.json").read_text())
    phases, eot = mix["phases"], int(mix["eot"])
    out.mkdir(parents=True, exist_ok=True)
    shards = {}
    for ph in phases.values():
        for s in ph["train"]:
            shards.setdefault(s["name"], sorted(glob.glob(str(mix_dir / f"{s['name']}_train_*.bin"))))

    def rule_of(ph: dict, s: dict) -> str:
        return s.get("rule", ph.get("rule", "stream"))

    def n_tok(f: str) -> int:
        return (os.path.getsize(f) - 256 * 4) // 2

    stream_users = {}
    for name_ph, ph in phases.items():
        for s in ph["train"]:
            if rule_of(ph, s) == "stream":
                stream_users.setdefault(s["name"], []).append(name_ph)
    order = [k for k in ("long", "decay", "trunk") if k in phases] + [k for k in phases if k not in ("long", "decay", "trunk")]
    supply = mix_supply(mix)

    def note(msg: str) -> None:
        print(f"mix note: {msg}", file=sys.stderr)

    # a shared stream source needs at least one shard per phase that reads it, or it cannot be split
    unsplit = {n for n, users in stream_users.items() if len(users) > 1 and len(shards.get(n, [])) < len(users)}
    if unsplit and not lenient:
        raise ValueError(f"mix {mix_dir}: too few shards to split between phases: {sorted(unsplit)}")
    for n in sorted(unsplit):
        note(f"{n} has {len(shards[n])} shard(s) for {len(stream_users[n])} phases: every phase reads all of it")
    taken = {name: 0 for name in shards}  # shards already given away, from the end
    paths = {}
    for name_ph in order:
        ph = phases[name_ph]
        seq = int(ph["seq_len"])
        train = []
        for s in ph["train"]:
            name, rule = s["name"], rule_of(ph, s)
            files = shards[name]
            if not files or name in supply[name_ph]["empty"]:
                why = "no shard" if not files else f"no row of {seq} tokens"
                if not lenient:
                    raise FileNotFoundError(f"mix {mix_dir}: {why} of source {name} (phase {name_ph})")
                note(f"phase {name_ph}: {name} left out ({why})")
                continue
            if rule == "stream" and len(stream_users.get(name, [])) > 1 and name not in unsplit:
                avail = files[: len(files) - taken[name]]
                first_user = [k for k in ("trunk", "decay", "long") if k in stream_users[name]][0]
                if name_ph == first_user:
                    use = avail
                else:
                    need, got, use = float(s["weight"]) * float(ph["tokens"]) * 1.03, 0, []
                    for f in reversed(avail):
                        if got >= need:
                            break
                        use.insert(0, f)
                        got += n_tok(f)
                    taken[name] += len(use)
                if not use:
                    raise ValueError(f"mix {mix_dir}: source {name} has no shard left for phase {name_ph}")
                files = use
            e = {"name": name, "files": files, "weight": float(s["weight"])}
            if rule != ph.get("rule", "stream"):
                e["rule"] = rule
            train.append(e)
        rules = {s["name"]: rule_of(ph, s) for s in ph["train"]}
        ev = []
        for s in ph.get("eval", []):
            if s["name"] in supply[name_ph]["eval_empty"]:
                note(f"phase {name_ph}: no evaluation row of {seq} tokens in {s['name']}, left out of the evaluation")
                continue
            e = {"name": s["name"], "files": [str(mix_dir / f"{s['name']}_val_*.bin")], "rows": int(s["rows"])}
            r = s.get("rule", rules.get(s["name"], ph.get("rule", "stream")))
            if r != ph.get("rule", "stream"):
                e["rule"] = r
            ev.append(e)
        m = {"seq_len": seq, "rule": ph.get("rule", "stream"), "eot": eot, "seed": int(ph.get("seed", 42)),
             "train": train, "eval": ev}
        paths[name_ph] = out / f"manifest_{name_ph}.json"
        paths[name_ph].write_text(json.dumps(m, indent=2))
    return paths, mix


def _final_eval_every(steps: int) -> int:
    """The smallest divisor of steps that is at least steps / 10: about ten evaluations, the last at the last step
    (the HF Trainer evaluates only at multiples of eval_steps)."""
    return min(d for d in range(1, steps + 1) if steps % d == 0 and d * 10 >= steps)


def train_cmd(arm: str, size: str, phase: str, *, data: Path, out: Path, init: str | None = None,
              lr: float | None = None, tokens: float | None = None, micro: int | None = None, seed: int = 42,
              wandb: bool = False, eval_every: int = 0, backend: str = "flex", local: bool = False,
              flow: bool = False, gpus: int = 1, sft_manifest: str | None = None,
              grad_ckpt: bool = False, compile: str = "off", liger: bool = False, rows: int | None = None,
              workers: int = 6, fast: bool = False, mix: str | None = None,
              save_every: int | None = None, mix_lenient: bool = False, optimizer: str = "adam",
              muon_adamw_lr: float | None = None, warmup: int | None = None,
              muon_peak: float | None = None, cut: tuple[int, int] = (32, 4096), constant_lr: bool = False,
              depth_mix_lr: float | None = None, tree_reasoner_lr: float | None = None) -> list[str]:
    liger = liger or fast
    z = SIZES[size]
    args, n = fit(arm, size)
    if phase == "split" and cut[1] > 0:  # E37c: cut rows (the model refuses arms with a notebook or a message boundary)
        args = {**args, "par_cut_gap_min": int(cut[0]), "par_cut_gap_max": int(cut[1])}
    tokenizer = str(data / "tokenizer_gpt2")
    if phase == "sft":
        if not (sft_manifest and init and lr):
            raise ValueError("phase sft needs --sft_manifest, --init (a decay's final folder) and --lr")
        man = {"sft": Path(sft_manifest).resolve()}
        seq = json.loads(man["sft"].read_text())["seq_len"]
        if seq != decay_len(size):
            raise ValueError(f"sft manifest seq_len {seq} != the decay's row length {decay_len(size)} at {size}")
        tokens = tokens or SFT["tokens"]
        rows_per_step = SFT["rows"]
        # one row per pass at 16k beyond s30 (24 GB home cards); gradient accumulation makes up the 16 rows
        micro = micro or max(1, min(rows_per_step, 2 if size == "s30" else 1))
    else:
        if mix:  # a built data mix: its tokenizer, sources, row lengths and tokens per phase
            man, mx = mix_manifests(Path(mix), out, lenient=mix_lenient)
            if phase not in man:
                raise ValueError(f"mix {mix} has no phase {phase!r}")
            tokenizer = str(Path(mix) / mx.get("tokenizer", "tokenizer"))
            tokens = tokens or float(mx["phases"][phase]["tokens"])
            if z.get("vocab") and int(mx.get("vocab_size", 0)) > int(z["vocab"]):
                # a leaderboard size is sized for its vocabulary; a larger one would cross the 150M cap
                raise ValueError(f"size {size} is sized for a {z['vocab']}-token vocabulary; mix {mix} has "
                                 f"{mx['vocab_size']}")
        elif z.get("tied"):
            raise ValueError(f"size {size} trains on a built data mix (--mix): it is sized for its tokenizer")
        else:
            man = manifests(data, out, size)
        seq = json.loads(man[phase].read_text())["seq_len"]
        tokens = tokens or z[phase]
        rows_per_step = z["rows"] * 4096 // seq  # the same tokens per step in every phase
        rows_per_step = max(1, rows_per_step)
        micro = micro or (1 if phase == "long" else max(1, min(rows_per_step, (16 if phase == "trunk" else 4)
                                                                 if size == "s30" else (8 if phase == "trunk" else 2))))
    if rows:  # speed checks only: fewer rows per optimizer step, so a short run logs enough steps
        rows_per_step = max(1, int(rows))
        micro = min(micro, rows_per_step)
    # gpus > 1: one model on several GPUs (data parallel, torchrun); each optimizer step still sees rows_per_step
    # rows: micro × accum × gpus. When gpus does not divide the step (64 rows on 3 GPUs) the step is rounded down
    # and the step count rises to keep the same total tokens.
    gpus = max(1, int(gpus))
    accum = max(1, rows_per_step // (micro * gpus))
    if micro * accum * gpus != rows_per_step:
        import sys
        print(f"note: {micro}×{accum}×{gpus} = {micro * accum * gpus} rows per step instead of {rows_per_step}",
              file=sys.stderr)
    steps = max(1, int(round(tokens / (micro * accum * gpus * seq))))
    if muon_peak and optimizer == "muon" and phase != "sft":
        # a whole Muon run from one peak, each phase as the AdamW recipe: trunk and decay start at the peak (the decay
        # ends at 10 % of it), the 64k phase starts where the decay ended; the AdamW fallback scales the same way
        if lr is not None:
            raise ValueError("give --muon_peak or --lr, not both")
        lr = 0.1 * muon_peak if phase in ("long", "split") else muon_peak
        muon_adamw_lr = muon_adamw_lr or (0.1 * z["lr"] if phase in ("long", "split") else z["lr"])
    lr = lr or (0.1 * z["lr"] if phase in ("long", "split") else z["lr"])  # long, split: the rate the decay ended on
    launch = (["torchrun", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else ["python"])
    cmd = ["uv", "run", *launch, "training/train_concept_pretraining.py"]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    if constant_lr:  # E40 retrofit: the rate held constant from the first step (no warm-up, no decay)
        sched = ["--lr_scheduler_type", "constant"]
    elif phase == "trunk":
        sched = ["--lr_scheduler_type", "constant_with_warmup",
                 "--warmup_steps", str(warmup if warmup is not None else max(1, int(0.01 * steps)))]
    else:  # decay: peak → 10 % of it; long: 10 % → 1 % of the peak (min_lr_ratio is relative to its own start)
        sched = ["--lr_scheduler_type", "warmup_stable_decay",
                 "--warmup_steps", str(warmup if warmup is not None else max(1, int(0.03 * steps))),
                 "--lr_scheduler_kwargs", json.dumps({"num_decay_steps": int(0.97 * steps), "decay_type": "1-sqrt",
                                                      "min_lr_ratio": 0.1})]
    # --save_every N: at most N steps between checkpoints, and never fewer than ten checkpoints in a phase
    save_every = min(int(save_every), max(50, steps // 10)) if save_every else max(50, steps // 10)
    if phase == "sft" and not eval_every:
        eval_every = _final_eval_every(steps)
    cmd += [
        "--attn_backend", backend, "--attn_pad_multiple", "1", "--use_liger", "True" if liger else "False",
        "--bin_manifest", str(man[phase]), "--tokenizer_name", tokenizer,
        "--max_seq_length", str(seq), "--batch_packing_mode", "none",
        "--per_device_train_batch_size", str(micro), "--per_device_eval_batch_size", str(max(1, micro // 2)),
        "--gradient_accumulation_steps", str(accum), "--learning_rate", str(lr), "--max_steps", str(steps),
        *sched, "--weight_decay", "0.1", "--adam_beta1", "0.9", "--adam_beta2", "0.95", "--max_grad_norm", "1.0",
        "--optim", "adamw_torch_fused", "--bf16", "True", "--logging_steps", "10",
        "--eval_strategy", "steps" if eval_every else "no", "--eval_steps", str(eval_every or save_every),
        "--save_strategy", "steps", "--save_steps", str(save_every), "--save_total_limit", "2",
        "--output_dir", str(out / phase), "--seed", str(seed), "--report_to", "wandb" if wandb else "none",
        "--overwrite_output_dir", "True", "--disable_tqdm", "True", "--dataloader_num_workers", str(workers),
        "--prediction_loss_only", "True", "--chunked_ce_block_size", "2048",
    ]
    if optimizer == "muon":
        # Muon on the matrices at --lr (no default: the size's AdamW rate is not a Muon rate); the AdamW fallback
        # (word tables, output layer, norms) at muon_adamw_lr, default the size's AdamW peak, following the schedule;
        # no weight decay on 1-D params, as HF AdamW groups them
        if not muon_peak and (lr is None or lr == z["lr"]):  # (a peak's 64k rate may equal the size's AdamW rate)
            raise ValueError("--optimizer muon needs its own --lr or --muon_peak (e.g. 0.01-0.02)")
        cmd += ["--optimizer", "muon", "--muon_adamw_lr", str(muon_adamw_lr or z["lr"]),
                "--muon_follow_schedule", "True", "--muon_decay_1d", "False"]
        if depth_mix_lr is not None:  # E40: the depth mixes' own AdamW rate (constant, no weight decay)
            cmd += ["--depth_mix_lr", str(depth_mix_lr)]
        if tree_reasoner_lr is not None:  # E41 retrofits: the tree reasoner's own AdamW rate (constant)
            cmd += ["--tree_reasoner_lr", str(tree_reasoner_lr)]
    elif depth_mix_lr is not None:
        raise ValueError("--depth_mix_lr is a Muon-run option (its own AdamW group)")
    if compile != "off":  # the whole model through torch.compile (the attention kernel is compiled either way)
        cmd += ["--model_compile", compile]
    if fast:  # the 2026-11-14 speed settings: Liger's fused loss (above) and the compiled concept grouping
        if args.get("par_global_read") == "concept":
            cmd += ["--concept_compile", "True"]
    if init:
        cmd += ["--model_name_or_path", init]
    if phase == "sft":
        # the base weights' held-out answer loss (the "before"); the CE as the token mean over each optimizer step
        # (answer counts differ by row), as the original LSA's torchtitan trainer computes it
        cmd += ["--eval_on_start", "True", "--loss_step_token_mean", "True"]
    if grad_ckpt:  # activations recomputed in backward (16k rows on a 24 GB card, 64k rows), never shorter rows
        if flow:  # the recomputed layers would record their instrumentation a second time
            raise ValueError("--flow and --grad_ckpt are not combined")
        cmd += ["--gradient_checkpointing", "True"]
    if flow:  # E33p instrumentation: dynamics every 200 steps (decay: 50 of ~763) and weight snapshots
        cmd += ["--flow_every", "200" if phase == "trunk" else "50",
                "--snapshot_every", "1000" if phase == "trunk" else "500"]
    if local:  # laptop pipeline check (Apple GPU / CPU): reference attention, plain AdamW, no worker processes
        import torch

        swap = {"--attn_backend": "sdpa", "--optim": "adamw_torch", "--dataloader_num_workers": "0",
                "--bf16": "True" if torch.backends.mps.is_available() else "False"}
        cmd = [swap.get(cmd[i - 1], x) if i else x for i, x in enumerate(cmd)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("params")
    sub.add_parser("sizes", help="total parameters per arm for the Tiny-ML leaderboard sizes")
    mi = sub.add_parser("mixinfo", help="what a built mix can feed each phase (reads DIR/mix.json only)")
    mi.add_argument("--mix", required=True)
    c = sub.add_parser("cmd")
    c.add_argument("--arm", required=True, choices=sorted(ARMS))
    c.add_argument("--size", required=True, choices=sorted(SIZES))
    c.add_argument("--phase", required=True, choices=("trunk", "decay", "long", "split", "sft"))
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--init")
    c.add_argument("--lr", type=float)
    c.add_argument("--tokens", type=float)
    c.add_argument("--micro", type=int)
    c.add_argument("--eval_every", type=int, default=0)
    c.add_argument("--backend", default="flex")
    c.add_argument("--wandb", action="store_true")
    c.add_argument("--local", action="store_true", help="laptop pipeline check (sdpa, adamw_torch, no workers)")
    c.add_argument("--flow", action="store_true", help="E33p instrumentation (flow/grad/upd/read/select/lens keys)")
    c.add_argument("--seed", type=int, default=42, help="trainer seed: initial weights and the order rows are drawn in")
    c.add_argument("--gpus", type=int, default=1, help="GPUs for one model (data parallel via torchrun)")
    c.add_argument("--sft_manifest", help="sft: manifest_sft.json of scripts/build_sft_mix.py")
    c.add_argument("--compile", default="off", choices=("off", "static", "dynamic"),
                   help="torch.compile the whole model (static: fixed row shapes)")
    c.add_argument("--liger", action="store_true", help="Liger's fused linear + cross-entropy for the loss")
    c.add_argument("--optimizer", default="adam", choices=("adam", "muon"),
                   help="muon: nn.muon.Muon on the matrices at --lr, AdamW on the rest at --muon_adamw_lr")
    c.add_argument("--muon_adamw_lr", type=float, help="Muon's AdamW fallback LR (default: the size's AdamW peak)")
    c.add_argument("--muon_peak", type=float, help="with --optimizer muon, a whole run from one Muon peak: trunk and "
                   "decay at it, the 64k phase at 10 %% of it (the AdamW fallback likewise); instead of --lr")
    c.add_argument("--warmup", type=int, help="warm-up steps instead of 1 %% (trunk) / 3 %% (later phases) of the phase")
    c.add_argument("--mix_lenient", action="store_true",
                   help="small check builds: leave out a source with no row at a phase's length and share a stream "
                        "source that has too few shards to split, instead of stopping")
    c.add_argument("--save_every", type=int, help="checkpoint at least every N optimizer steps (default and upper "
                                                  "bound: a tenth of the phase; long spot runs lose less)")
    c.add_argument("--mix", help="a built data mix (scripts/build_m21_mix.py output dir): its tokenizer, sources, "
                                 "row lengths and tokens per phase replace the GPT-2 FineWeb-Edu / LongDocs-lite data")
    c.add_argument("--fast", action="store_true",
                   help="the speed settings measured 2026-11-14: --liger plus the compiled concept grouping (E37 "
                        "arms); same loss function and same concepts, pair it with a larger --micro")
    c.add_argument("--rows", type=int, help="rows per optimizer step instead of the size's (speed checks; E40's "
                                             "retrofit: 65,536 tokens per step)")
    c.add_argument("--constant_lr", action="store_true", help="E40 retrofit: the rate held constant, no warm-up or decay")
    c.add_argument("--depth_mix_lr", type=float, help="E40, Muon: the depth mixes' own AdamW rate (constant)")
    c.add_argument("--tree_reasoner_lr", type=float, help="E41, Muon: the tree reasoner's own AdamW rate (constant)")
    c.add_argument("--workers", type=int, default=6, help="dataloader worker processes per GPU")
    c.add_argument("--grad_ckpt", action="store_true", help="recompute each layer in the backward (less memory)")
    c.add_argument("--cut_min", type=int, default=32, help="E37c split phase: shortest gap between cuts, tokens")
    c.add_argument("--cut_max", type=int, default=4096, help="E37c split phase: longest gap between cuts, tokens (0 = no cuts: the "
                   "control, the same rows, rate and seed without a cut)")
    a = ap.parse_args()
    if a.what == "mixinfo":
        f = Path(a.mix)
        mix = json.loads((f / "mix.json" if f.is_dir() else f).read_text())
        srcs = mix.get("sources", {})
        print(f"mix {mix.get('name')}: {sum(float(v.get('tokens', 0)) for v in srcs.values()) / 1e9:.2f}B train tokens in "
              f"{len(srcs)} sources; missing: {mix.get('missing') or 'none'}; external missing: "
              f"{mix.get('external_missing') or 'none'}")
        for name, r in mix_supply(mix).items():
            passes = f"{r['passes']:.2f}" if r["passes"] else "n/a"
            print(f"{name:6} rows of {r['seq_len']:>6}: budget {r['tokens'] / 1e9:.2f}B tokens, one pass "
                  f"{r['tokens_per_pass'] / 1e9:.3f}B ({r['rows_per_pass']:,} rows, limited by {r['limited_by']}) "
                  f"-> {passes} passes; empty: {r['empty'] or 'none'}; no eval rows: {r['eval_empty'] or 'none'}")
            ph = mix["phases"][name]
            for sup in ph.get("supply", []):
                av, need = sup.get("rows_available"), sup.get("rows_needed")
                flag = "  SHORT" if (av is not None and need and av < need) or not av else ""
                print(f"         {sup['name']:15} {sup.get('rule', ''):6} rows {av if av is not None else 0:>9,} of {need:>9,} needed{flag}")
        return
    if a.what == "cmd" and a.phase == "sft" and not (a.lr and a.init and a.sft_manifest):
        ap.error("--phase sft needs --lr (the pre-flight's pick), --init (a decay's final folder) and --sft_manifest")
    if a.what == "sizes":   # total parameters per arm for the leaderboard sizes (the board counts everything, once)
        for size in [k for k, z in SIZES.items() if z.get("tied")]:
            V = int(SIZES[size].get("vocab", VOCAB))
            for arm in ("e37a", "lsa", "catalogue_reuse_n4", "dense_rope", "window", "tree"):
                args, n = fit(arm, size)
                tot = count(args, non_embedding=False, vocab=V)
                print(f"{size} {arm:20} body {n / 1e6:7.2f}M  total {tot / 1e6:7.2f}M  ffn {args['intermediate_size']}"
                      f"  {'OK' if tot < 150e6 else 'OVER 150M'}")
        return
    if a.what == "params":
        for size in ("s30", "m100", "l300"):
            for arm in ARMS:
                args, n = fit(arm, size)
                print(f"{size:5} {arm:7} non-embedding {n/1e6:7.1f}M  total {count(args, False)/1e6:7.1f}M  "
                      f"ffn {args['intermediate_size']}")
        return
    cmd = train_cmd(a.arm, a.size, a.phase, data=Path(a.data), out=Path(a.out), init=a.init, lr=a.lr,
                    tokens=a.tokens, micro=a.micro, wandb=a.wandb, eval_every=a.eval_every, backend=a.backend,
                    local=a.local, flow=a.flow, seed=a.seed, gpus=a.gpus, sft_manifest=a.sft_manifest,
                    grad_ckpt=a.grad_ckpt, compile=a.compile, liger=a.liger, rows=a.rows, workers=a.workers,
                    fast=a.fast, mix=a.mix, save_every=a.save_every, mix_lenient=a.mix_lenient,
                    optimizer=a.optimizer, muon_adamw_lr=a.muon_adamw_lr, warmup=a.warmup, muon_peak=a.muon_peak,
                    cut=(a.cut_min, a.cut_max), constant_lr=a.constant_lr, depth_mix_lr=a.depth_mix_lr,
                    tree_reasoner_lr=a.tree_reasoner_lr)
    print(shlex.join(cmd))


if __name__ == "__main__":
    main()

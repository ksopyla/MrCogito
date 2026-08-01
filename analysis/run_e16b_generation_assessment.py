#!/usr/bin/env python3
"""E16b free-run generation assessment vs base Gemma-3-1B-pt.

Answers three questions with numbers + snippets:
  1. How bad is repetition across generation length cutoffs?
  2. Does longer *prompt* context help or hurt free-run quality?
  3. Did E16b LoRA+concepts degrade open-ended generation relative to
     the frozen Gemma-3-1B-pt backbone?

Writes JSON under Cache/Evaluation_reports/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nn.backbone_concept_lm import BackboneConceptLM

SHORT_PROMPTS = [
    "The future of renewable energy depends on",
    "Once upon a time, in a quiet village",
    "The most important skill in software engineering is",
    "Scientists have long wondered whether",
    "In machine learning, a gradient is",
    "The meaning of life is",
    "Long ago, the kingdom of",
    "The most surprising discovery of the last decade was",
]

CHAT_MSGS = [
    "Continue this sentence in fluent English: The future of renewable energy depends on",
    "Write a short fairy-tale continuation: Once upon a time, in a quiet village",
    "Explain briefly: The most important skill in software engineering is",
    "Continue scientifically: Scientists have long wondered whether",
    "Explain: In machine learning, a gradient is",
    "Reflect briefly: The meaning of life is",
    "Continue the story: Long ago, the kingdom of",
    "Describe: The most surprising discovery of the last decade was",
]

LENGTH_CUTOFFS = [32, 64, 128, 256, 512, 1024]
CONTEXT_LENS = [128, 512, 1024, 2048]


def distinct_n(ids: Sequence[int], n: int) -> float:
    if n <= 0 or len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def rep_3(ids: Sequence[int]) -> float:
    n = 3
    if len(ids) < n + 1:
        return 0.0
    seen: Counter = Counter()
    repeated = total = 0
    for i in range(len(ids) - n):
        full = tuple(ids[i : i + n + 1])
        total += 1
        if seen[full] > 0:
            repeated += 1
        seen[full] += 1
    return repeated / max(total, 1)


def summarize(ids: Sequence[int]) -> Dict[str, float]:
    return {
        "n_tokens": float(len(ids)),
        "distinct_1": distinct_n(ids, 1),
        "distinct_2": distinct_n(ids, 2),
        "rep_3": rep_3(ids),
    }


def by_length(ids: Sequence[int], cutoffs: Sequence[int]) -> Dict[str, Dict[str, float]]:
    out = {}
    for L in cutoffs:
        if 0 < L <= len(ids):
            out[str(L)] = summarize(ids[:L])
    return out


def format_gemma_chat(user_message: str) -> str:
    return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def preview(text: str, n: int = 160) -> str:
    return " ".join(text[:n].split())


@torch.no_grad()
def _e16b_next_token_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            concept_mode: str) -> torch.Tensor:
    """Last-position LM logits for remote checkpoints that lack ``generate``.

    Prefers ``next_token_logits`` / ``generate`` when present (newer local code).
    Falls back to a one-step teacher-forced CE path: ``per_position`` predictions
    on a shifted labels view is unavailable without ``return_last_hidden``, so we
    reconstruct the last hidden via a patched block forward when needed.
    """
    if hasattr(model, "next_token_logits"):
        return model.next_token_logits(input_ids, attention_mask, concept_mode=concept_mode)

    # Fallback: run block loop and project the final position. Older Odra code has
    # no return_last_hidden; monkeypatch a tiny last-hidden capture.
    B, S = input_ids.shape
    K = model.config.concept_block
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    dtype = model.backbone.model.embed_tokens.weight.dtype
    use_concepts = model.has_concepts and concept_mode != "zero"
    z = model.concept_init.unsqueeze(0).expand(B, -1, -1) if use_concepts else None
    model._concept_state["shuffle"] = concept_mode == "shuffle"
    model._concept_state["permutation"] = None

    import math
    n_blocks = math.ceil(S / K)
    last_h = None
    for b in range(n_blocks):
        s, e = b * K, min((b + 1) * K, S)
        blk_len = e - s
        lo = s - K if b > 0 else 0
        dec_ids = input_ids[:, lo:e]
        dec_mask = attention_mask[:, lo:e]
        mask4d = model._windowed_causal_mask(dec_mask, dtype)
        model._concept_state["z"] = z
        attention_masks = {"full_attention": mask4d, "sliding_attention": mask4d}
        inputs_embeds = model.backbone.model.embed_tokens(dec_ids)
        if model.config.concept_io_mode == "shared_depth_recurrent" and model.has_concepts:
            h, z = model._forward_shared_depth_block(
                inputs_embeds, attention_masks, z,
                block_len=blk_len,
                block_pad_mask=attention_mask[:, s:e] == 0,
                concept_mode=concept_mode,
            )
        else:
            out = model.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                use_cache=False,
            )
            h = out.last_hidden_state
            if use_concepts and concept_mode != "static":
                h_blk = h[:, -blk_len:]
                blk_pad = attention_mask[:, s:e] == 0
                write_base = (
                    model.concept_init.unsqueeze(0).expand(B, -1, -1)
                    if concept_mode == "one_block"
                    else z
                )
                z = model.write_head(write_base, h_blk, blk_pad)
        last_h = h[:, -1, :]  # last decoded position of this block

    model._concept_state["z"] = None
    model._concept_state["shuffle"] = False
    model._concept_state["permutation"] = None
    weight = model.backbone.lm_head.weight.float()
    return torch.nn.functional.linear(last_h.float(), weight)


@torch.no_grad()
def _e16b_generate_loop(model, input_ids, attention_mask, *, max_new: int, greedy: bool,
                        temperature: float, top_p: float, top_k: int, concept_mode: str):
    if hasattr(model, "generate"):
        return model.generate(
            input_ids, attention_mask,
            max_new_tokens=max_new,
            do_sample=not greedy,
            temperature=temperature if not greedy else 1.0,
            top_k=top_k if not greedy else 0,
            top_p=top_p if not greedy else 1.0,
            concept_mode=concept_mode,
        )
    cur = input_ids
    mask = attention_mask
    eos = model.config.eos_token_id
    finished = torch.zeros(cur.shape[0], dtype=torch.bool, device=cur.device)
    for _ in range(max_new):
        logits = _e16b_next_token_logits(model, cur, mask, concept_mode)
        if greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits.float() / max(temperature, 1e-5)
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1]
                logits = logits.masked_fill(logits < kth.unsqueeze(-1), float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        if finished.any():
            pad_id = model.config.pad_token_id or eos or 0
            next_id = torch.where(finished.view(-1, 1), torch.full_like(next_id, pad_id), next_id)
        cur = torch.cat([cur, next_id], dim=1)
        mask = torch.cat([mask, (~finished).long().view(-1, 1)], dim=1)
        if eos is not None:
            finished = finished | (next_id.squeeze(-1) == eos)
            if bool(finished.all()):
                break
    return cur


@torch.no_grad()
def gen_e16b(model, tokenizer, prompt: str, device, *, max_new: int, max_prompt: int,
             greedy: bool, temperature: float, top_p: float, top_k: int,
             concept_mode: str, seed: int) -> Dict:
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt,
                    add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    t0 = time.time()
    out = _e16b_generate_loop(
        model, input_ids, attention_mask,
        max_new=max_new, greedy=greedy, temperature=temperature,
        top_p=top_p, top_k=top_k, concept_mode=concept_mode,
    )
    dt = time.time() - t0
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "ids": gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        "prompt_n_tokens": int(input_ids.shape[1]),
        "seconds": dt,
        "summary": summarize(gen_ids),
        "by_length": by_length(gen_ids, LENGTH_CUTOFFS),
    }


@torch.no_grad()
def gen_base(model, tokenizer, prompt: str, device, *, max_new: int, max_prompt: int,
             greedy: bool, temperature: float, top_p: float, top_k: int,
             seed: int) -> Dict:
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt,
                    add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    t0 = time.time()
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=not greedy,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if not greedy:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k or 50)
    out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    dt = time.time() - t0
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "ids": gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        "prompt_n_tokens": int(input_ids.shape[1]),
        "seconds": dt,
        "summary": summarize(gen_ids),
        "by_length": by_length(gen_ids, LENGTH_CUTOFFS),
    }


def mean_by_length(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict]] = {}
    for r in rows:
        for L, m in (r.get("by_length") or {}).items():
            buckets.setdefault(L, []).append(m)
    out = {}
    for L, ms in buckets.items():
        keys = ("distinct_1", "distinct_2", "rep_3")
        out[L] = {k: sum(m[k] for m in ms) / len(ms) for k in keys}
        out[L]["n_samples"] = float(len(ms))
    return out


def load_long_docs(tokenizer, n_docs: int, min_tokens: int = 4200) -> List[str]:
    """Pull long-ish documents from a local/streamable corpus."""
    from datasets import load_dataset

    docs: List[str] = []
    # Prefer FineWeb-edu if cached; else minipile.
    for name, config in (
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("JeanKaddour/minipile", None),
    ):
        try:
            kwargs = dict(split="train", streaming=True)
            if config:
                ds = load_dataset(name, config, **kwargs)
            else:
                ds = load_dataset(name, **kwargs)
            for sample in ds:
                text = (sample.get("text") or "").strip()
                if len(text) < 2000:
                    continue
                n_tok = len(tokenizer(text, truncation=True, max_length=min_tokens + 64)["input_ids"])
                if n_tok < min_tokens:
                    continue
                docs.append(text)
                if len(docs) >= n_docs:
                    return docs
        except Exception as e:
            print(f"  corpus {name} unavailable ({type(e).__name__}: {e})")
            continue
    return docs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--e16b_path", required=True)
    p.add_argument("--base_model", default="google/gemma-3-1b-pt")
    p.add_argument("--output_json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--ctx_max_new_tokens", type=int, default=256)
    p.add_argument("--max_prompts", type=int, default=8)
    p.add_argument("--n_ctx_docs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_base", action="store_true")
    p.add_argument("--skip_context_sweep", action="store_true")
    p.add_argument("--skip_chat", action="store_true")
    p.add_argument("--sample", action="store_true",
                   help="Also run nucleus sampling (T=0.8, top_p=0.95) on short prompts.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    print("Loading E16b…")
    tok_e = AutoTokenizer.from_pretrained(args.e16b_path, local_files_only=True)
    e16b = BackboneConceptLM.from_pretrained(
        args.e16b_path, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    print(f"  concept_io={e16b.config.concept_io_mode} C={e16b.config.concept_num} K={e16b.config.concept_block}")

    base = None
    tok_b = None
    if not args.skip_base:
        print(f"Loading base {args.base_model}…")
        tok_b = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map=None,
        ).to(device).eval()
        if tok_b.pad_token_id is None:
            tok_b.pad_token = tok_b.eos_token

    short_rows: List[Dict] = []
    prompts = SHORT_PROMPTS[: args.max_prompts]
    chat_msgs = CHAT_MSGS[: args.max_prompts]

    def run_short(style: str, prompt: str, decode: str, concept_mode: str = "real"):
        greedy = decode == "greedy"
        temperature, top_p, top_k = (0.8, 0.95, 50) if not greedy else (1.0, 1.0, 0)
        # E16b
        r = gen_e16b(
            e16b, tok_e, prompt, device,
            max_new=args.max_new_tokens, max_prompt=1024,
            greedy=greedy, temperature=temperature, top_p=top_p, top_k=top_k,
            concept_mode=concept_mode, seed=args.seed,
        )
        row = {
            "model": "e16b",
            "concept_mode": concept_mode,
            "prompt_style": style,
            "decode": decode,
            "prompt": prompt[:200],
            **{k: r[k] for k in ("text", "n_tokens", "prompt_n_tokens", "seconds", "summary", "by_length")},
            "text_preview": preview(r["text"]),
            "text_tail": preview(r["text"][-200:]) if r["text"] else "",
        }
        short_rows.append(row)
        s = r["summary"]
        print(f"  [e16b|{concept_mode}|{style}|{decode}] "
              f"d1={s['distinct_1']:.3f} r3={s['rep_3']:.3f} n={r['n_tokens']} "
              f"{r['seconds']:.1f}s | {row['text_preview']!r}")

        if base is not None and concept_mode == "real":
            rb = gen_base(
                base, tok_b, prompt, device,
                max_new=args.max_new_tokens, max_prompt=1024,
                greedy=greedy, temperature=temperature, top_p=top_p, top_k=top_k,
                seed=args.seed,
            )
            brow = {
                "model": "gemma3_1b_pt",
                "concept_mode": "n/a",
                "prompt_style": style,
                "decode": decode,
                "prompt": prompt[:200],
                **{k: rb[k] for k in ("text", "n_tokens", "prompt_n_tokens", "seconds", "summary", "by_length")},
                "text_preview": preview(rb["text"]),
                "text_tail": preview(rb["text"][-200:]) if rb["text"] else "",
            }
            short_rows.append(brow)
            sb = rb["summary"]
            print(f"  [base|n/a|{style}|{decode}] "
                  f"d1={sb['distinct_1']:.3f} r3={sb['rep_3']:.3f} n={rb['n_tokens']} "
                  f"{rb['seconds']:.1f}s | {brow['text_preview']!r}")

    print("\n=== Part A: short-prompt continuation / chat ===")
    for p in prompts:
        run_short("continuation", p, "greedy", "real")
        run_short("continuation", p, "greedy", "zero")
    if not args.skip_chat:
        for msg in chat_msgs:
            run_short("chat", format_gemma_chat(msg), "greedy", "real")
    if args.sample:
        for p in prompts[:4]:
            run_short("continuation", p, "sample", "real")
            if base is not None:
                pass  # base already paired inside run_short when concept_mode=real

    # Aggregates
    def filt(**kw):
        rows = short_rows
        for k, v in kw.items():
            rows = [r for r in rows if r.get(k) == v]
        return rows

    aggregates = {
        "e16b_real_cont_greedy": mean_by_length(filt(model="e16b", concept_mode="real",
                                                     prompt_style="continuation", decode="greedy")),
        "e16b_zero_cont_greedy": mean_by_length(filt(model="e16b", concept_mode="zero",
                                                     prompt_style="continuation", decode="greedy")),
        "e16b_real_chat_greedy": mean_by_length(filt(model="e16b", concept_mode="real",
                                                     prompt_style="chat", decode="greedy")),
        "base_cont_greedy": mean_by_length(filt(model="gemma3_1b_pt", prompt_style="continuation",
                                                decode="greedy")),
        "base_chat_greedy": mean_by_length(filt(model="gemma3_1b_pt", prompt_style="chat",
                                                decode="greedy")),
    }
    if args.sample:
        aggregates["e16b_real_cont_sample"] = mean_by_length(
            filt(model="e16b", concept_mode="real", prompt_style="continuation", decode="sample"))
        aggregates["base_cont_sample"] = mean_by_length(
            filt(model="gemma3_1b_pt", prompt_style="continuation", decode="sample"))

    print("\n=== Aggregates (short prompts) ===")
    for name, agg in aggregates.items():
        if not agg:
            print(f"  {name}: (empty)")
            continue
        parts = []
        for L in sorted(agg, key=int):
            m = agg[L]
            parts.append(f"{L}:d1={m['distinct_1']:.2f}/r3={m['rep_3']:.2f}")
        print(f"  {name}: " + " | ".join(parts))

    # Part B: context-length sweep
    ctx_rows: List[Dict] = []
    if not args.skip_context_sweep:
        print("\n=== Part B: prompt-context length sweep ===")
        docs = load_long_docs(tok_e, n_docs=args.n_ctx_docs, min_tokens=max(CONTEXT_LENS) + 64)
        print(f"  loaded {len(docs)} long docs")
        for di, doc in enumerate(docs):
            # tokenize once
            all_ids = tok_e(doc, return_tensors="pt", truncation=True,
                            max_length=max(CONTEXT_LENS))["input_ids"][0]
            for L in CONTEXT_LENS:
                if all_ids.numel() < L:
                    continue
                prefix_ids = all_ids[:L]
                prompt = tok_e.decode(prefix_ids, skip_special_tokens=True)
                # E16b real
                r = gen_e16b(
                    e16b, tok_e, prompt, device,
                    max_new=args.ctx_max_new_tokens, max_prompt=L,
                    greedy=True, temperature=1.0, top_p=1.0, top_k=0,
                    concept_mode="real", seed=args.seed,
                )
                row = {
                    "model": "e16b",
                    "concept_mode": "real",
                    "doc_index": di,
                    "prompt_n_tokens": L,
                    "n_tokens": r["n_tokens"],
                    "seconds": r["seconds"],
                    "summary": r["summary"],
                    "by_length": r["by_length"],
                    "text_preview": preview(r["text"]),
                    "text_tail": preview(r["text"][-200:]) if r["text"] else "",
                }
                ctx_rows.append(row)
                s = r["summary"]
                print(f"  [e16b doc{di} L={L}] d1={s['distinct_1']:.3f} r3={s['rep_3']:.3f} "
                      f"n={r['n_tokens']} {r['seconds']:.1f}s | {row['text_preview']!r}")

                if base is not None:
                    rb = gen_base(
                        base, tok_b, prompt, device,
                        max_new=args.ctx_max_new_tokens, max_prompt=L,
                        greedy=True, temperature=1.0, top_p=1.0, top_k=0,
                        seed=args.seed,
                    )
                    brow = {
                        "model": "gemma3_1b_pt",
                        "concept_mode": "n/a",
                        "doc_index": di,
                        "prompt_n_tokens": L,
                        "n_tokens": rb["n_tokens"],
                        "seconds": rb["seconds"],
                        "summary": rb["summary"],
                        "by_length": rb["by_length"],
                        "text_preview": preview(rb["text"]),
                        "text_tail": preview(rb["text"][-200:]) if rb["text"] else "",
                    }
                    ctx_rows.append(brow)
                    sb = rb["summary"]
                    print(f"  [base doc{di} L={L}] d1={sb['distinct_1']:.3f} r3={sb['rep_3']:.3f} "
                          f"n={rb['n_tokens']} {rb['seconds']:.1f}s | {brow['text_preview']!r}")

    ctx_agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name in ("e16b", "gemma3_1b_pt"):
        by_L: Dict[int, List[Dict]] = {}
        for r in ctx_rows:
            if r["model"] != model_name:
                continue
            by_L.setdefault(int(r["prompt_n_tokens"]), []).append(r["summary"])
        ctx_agg[model_name] = {}
        for L, ms in sorted(by_L.items()):
            ctx_agg[model_name][str(L)] = {
                "distinct_1": sum(m["distinct_1"] for m in ms) / len(ms),
                "distinct_2": sum(m["distinct_2"] for m in ms) / len(ms),
                "rep_3": sum(m["rep_3"] for m in ms) / len(ms),
                "n_samples": float(len(ms)),
            }

    print("\n=== Context-sweep aggregates ===")
    for model_name, agg in ctx_agg.items():
        for L, m in agg.items():
            print(f"  {model_name} @prompt={L}: d1={m['distinct_1']:.3f} r3={m['rep_3']:.3f} n={m['n_samples']:.0f}")

    report = {
        "e16b_path": args.e16b_path,
        "base_model": None if args.skip_base else args.base_model,
        "device": str(device),
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "ctx_max_new_tokens": args.ctx_max_new_tokens,
        "short_prompt_rows": [
            {k: v for k, v in r.items() if k != "ids"} for r in short_rows
        ],
        "short_aggregates": aggregates,
        "context_sweep_rows": ctx_rows,
        "context_sweep_aggregates": ctx_agg,
        "interpretation_hints": {
            "healthy": "flat-or-rising distinct_1 across length cutoffs; rep_3 << 0.5",
            "repetition_loop": "distinct_1 collapses with length; rep_3 → 0.8+",
            "broke_gemma": "e16b distinct_1 << base at matched cutoffs on continuation",
            "chat_sft_needed": "only if chat << continuation AND base chat is fine; "
                              "chat format alone does not fix continuation loops",
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop huge raw ids if any slipped in
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

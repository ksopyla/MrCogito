#!/usr/bin/env python
"""E36 S2 (teacher matching) rows: memory text + the target the frozen reader continues, one format for all sources.

Each row: input_ids = memory tokens + target tokens, mem_len = number of memory tokens. The writer turns the
memory into entries (512 tokens each); the student reader sees the entries + the target; the teacher reader sees the
raw memory + the target (evaluation/frozen_reader_eval.py and nn/frozen_reader_memory.py).

Sources (no hand-made labels; public text only):
  squad  : SQuAD v1.1 train — memory = the passage; target = "Question: q\nAnswer: a\n";
  synth  : PleIAs SYNTH, English rows with a seed passage — memory = the Wikipedia seed passage;
           target = "Question: query\nAnswer: synthetic answer\n" (truncated);
  text   : the E36 long-document rows (PG-19 + FinePDFs) — memory = 4,096 tokens, target = the next 512.

  uv run python scripts/build_e36_s2_data.py --text_manifest $DATASETS_TOK_DIR/e36_longdoc_4k6_v1_manifest.json \
      --out_dir $DATASETS_TOK_DIR/e36_s2_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _qa_rows(ds, tok, passage_col, q_col, a_fn, max_mem, max_tgt, num_proc):
    def enc(batch):
        mem = tok([p.strip() for p in batch[passage_col]], add_special_tokens=False)["input_ids"]
        tgt_text = [f"Question: {q.strip()}\nAnswer: {a_fn(batch, i).strip()}\n" for i, q in enumerate(batch[q_col])]
        tgt = tok(tgt_text, add_special_tokens=False)["input_ids"]
        ids, ml = [], []
        for m, t in zip(mem, tgt):
            m, t = m[:max_mem], t[:max_tgt]
            ids.append(m + t)
            ml.append(len(m))
        return {"input_ids": ids, "mem_len": ml}

    return ds.map(enc, batched=True, num_proc=num_proc, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--text_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--text_rows", type=int, default=60000)
    ap.add_argument("--synth_files", type=int, default=2)
    ap.add_argument("--max_mem", type=int, default=1024)
    ap.add_argument("--max_tgt", type=int, default=512)
    ap.add_argument("--eval_rows", type=int, default=500)
    ap.add_argument("--num_proc", type=int, default=32)
    ap.add_argument("--weights", default="text=0.34,squad=0.33,synth=0.33")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    weights = dict((k, float(v)) for k, v in (kv.split("=") for kv in a.weights.split(",")))
    built = {}

    squad = load_dataset("rajpurkar/squad", split="train")
    built["squad"] = _qa_rows(squad, tok, "context", "question", lambda b, i: b["answers"][i]["text"][0],
                              a.max_mem, a.max_tgt, a.num_proc)

    files = [f"synth_{i:03d}.parquet" for i in range(1, a.synth_files + 1)]
    synth = load_dataset("PleIAs/SYNTH", data_files=files, split="train")
    synth = synth.filter(lambda b: [lang == "en" and bool(t) and len(t) > 200 and bool(q) and bool(ans)
                                    for lang, t, q, ans in zip(b["language"], b["query_seed_text"], b["query"],
                                                               b["synthetic_answer"])],
                         batched=True, num_proc=a.num_proc)
    built["synth"] = _qa_rows(synth, tok, "query_seed_text", "query", lambda b, i: b["synthetic_answer"][i],
                              a.max_mem, a.max_tgt, a.num_proc)

    man = json.load(open(a.text_manifest))
    parts = []
    per = a.text_rows // len(man["sources"])
    for src in man["sources"]:
        ds = load_from_disk(src["train_path"]).shuffle(seed=0).select(range(per))
        parts.append(ds.map(lambda b: {"input_ids": [r[1:4609] for r in b["input_ids"]],
                                       "mem_len": [4096] * len(b["input_ids"])},
                            batched=True, num_proc=a.num_proc, remove_columns=ds.column_names))
    built["text"] = concatenate_datasets(parts)

    sources = []
    for name, ds in built.items():
        ds = ds.filter(lambda b: [m > 0 and len(x) > m + 2 for m, x in zip(b["mem_len"], b["input_ids"])],
                       batched=True, num_proc=a.num_proc).shuffle(seed=0)
        ev = ds.select(range(a.eval_rows))
        tr = ds.select(range(a.eval_rows, len(ds)))
        tp, ep = os.path.join(a.out_dir, name, "train"), os.path.join(a.out_dir, name, "eval")
        tr.save_to_disk(tp)
        ev.save_to_disk(ep)
        sources.append({"name": name, "train_path": tp, "eval_path": ep, "weight": weights[name]})
        print(f"[e36-s2] {name}: train={len(tr):,} eval={len(ev):,}", flush=True)
    manifest = {"mix_id": os.path.basename(a.out_dir.rstrip("/")), "max_seq_length": 4610, "objective": "causal_lm",
                "format": "memory+target rows with mem_len (E36 S2)", "sources": sources}
    path = a.out_dir.rstrip("/") + "_manifest.json"
    json.dump(manifest, open(path, "w"), indent=2)
    print(f"[e36-s2] manifest {path}", flush=True)


if __name__ == "__main__":
    main()

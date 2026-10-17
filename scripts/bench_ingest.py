#!/usr/bin/env python
"""Ingest-speed benchmark: our latent-memory page writer (E33f) vs an open embedding model (EmbeddingGemma 2) on the
same 1M-token document, plus the query-side costs of the "memory tape" design (one question embedded on GPU and CPU,
exact vector search over a store of chunk vectors).

Two parts, two environments, one text file between them:
  writer    project env (the E33f checkpoint, our tokenizer). Builds the demo's 1M-token world document
            (`scripts/a2a_demo.build_document`, the same builder as `notes_a2a_demo.py`), saves its text, times
            `write_notes` (bf16, inference mode) on GPU, and on a 64k-token slice on CPU (fp32).
  embedder  an isolated env with a recent sentence-transformers (EmbeddingGemma 2 needs a newer transformers than
            the project pins). Splits the same text into chunks of N Gemma tokens, times encode_document end to end
            (tokenize + forward + pooling) and the forward alone, per batch size; then query latency and search.

Speeds are reported per character as well as per token: the two tokenizers cut the same text differently.

  uv run python scripts/bench_ingest.py --part writer --checkpoint <E33f final> --data <text_checks data dir> \\
      --text_out Cache/e35/bench_doc.txt --out Cache/e35/bench_writer.json
  uv run --isolated --no-project --with torch --with sentence-transformers --with transformers \\
      python scripts/bench_ingest.py --part embedder --text_in Cache/e35/bench_doc.txt --out Cache/e35/bench_embedder.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats: int) -> list:
    out = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return out


def env_info() -> dict:
    info = {"torch": torch.__version__, "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
            "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def bench_writer(a) -> dict:
    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm
    from scripts.a2a_demo import build_document
    from scripts.build_text_checks_data import load_stories

    tok = AutoTokenizer.from_pretrained(os.path.join(a.data, "tokenizer"))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    _, held = load_stories(a.stories, 1, 0)
    body, _ = build_document("lookup", a.length, a.seed, held, ntok)
    os.makedirs(os.path.dirname(os.path.abspath(a.text_out)), exist_ok=True)
    open(a.text_out, "w").write(body)
    t = time.perf_counter()
    ids_list = tok.encode(body, add_special_tokens=False)
    tokenize_s = time.perf_counter() - t
    res = {"part": "writer", "checkpoint": a.checkpoint, "chars": len(body), "bytes": len(body.encode()),
           "tokens": len(ids_list), "tokenize_s": tokenize_s, "env": env_info()}

    model = load_perceiver_lm(a.checkpoint, "cuda", "sdpa").eval()  # bf16 on CUDA, as in the demo
    for prm in model.parameters():
        prm.requires_grad_(False)
    w = model.memory_writer
    res["writer_params"] = sum(p.numel() for p in w.parameters())
    res["token_table_params"] = model.embed.tok.weight.numel()
    ids = torch.tensor([ids_list], device="cuda")
    with torch.inference_mode():
        model.write_notes(ids[:, :65536], chunk_pages=a.chunk_pages)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        notes = model.write_notes(ids, chunk_pages=a.chunk_pages)[0].shape[1]
        times = timed(lambda: model.write_notes(ids, chunk_pages=a.chunk_pages), a.repeats)
    res["gpu"] = {"dtype": "bfloat16", "notes": notes, "times_s": times, "median_s": statistics.median(times),
                  "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["gpu"]["tokens_per_s"] = len(ids_list) / res["gpu"]["median_s"]
    res["gpu"]["chars_per_s"] = len(body) / res["gpu"]["median_s"]
    print(f"writer GPU: {len(ids_list):,} tokens ({len(body):,} chars) → {notes:,} notes in "
          f"{res['gpu']['median_s']:.3f} s median of {a.repeats} ({res['gpu']['tokens_per_s'] / 1e6:.2f}M tokens/s)", flush=True)

    del model
    torch.cuda.empty_cache()
    torch.set_num_threads(a.cpu_threads)
    model = load_perceiver_lm(a.checkpoint, "cpu", "sdpa", dtype=torch.float32).eval()
    sl = torch.tensor([ids_list[: a.cpu_tokens]])
    n_chars = len(tok.decode(ids_list[: a.cpu_tokens]))
    with torch.inference_mode():
        model.write_notes(sl[:, :8192], chunk_pages=a.chunk_pages)
        times = timed(lambda: model.write_notes(sl, chunk_pages=a.chunk_pages), max(1, a.repeats // 2))
    med = statistics.median(times)
    res["cpu"] = {"dtype": "float32", "threads": a.cpu_threads, "tokens": sl.shape[1], "chars": n_chars, "times_s": times,
                  "median_s": med, "tokens_per_s": sl.shape[1] / med, "chars_per_s": n_chars / med}
    print(f"writer CPU ({a.cpu_threads} threads): {sl.shape[1]:,} tokens in {med:.2f} s "
          f"({res['cpu']['tokens_per_s'] / 1e3:.1f}k tokens/s)", flush=True)
    return res


def _load_st(name: str, device: str, dtype, text_only: bool):
    from sentence_transformers import SentenceTransformer

    kw = {"device": device, "model_kwargs": {"torch_dtype": dtype}}
    if text_only:
        kw["config_kwargs"] = {"vision_config": None, "audio_config": None}
    return SentenceTransformer(name, **kw)


def _features(m, texts):
    f = m.tokenize(texts) if hasattr(m, "tokenize") else m.preprocess(texts)
    return {k: (v.to(m.device) if torch.is_tensor(v) else v) for k, v in f.items()}


def bench_embedder(a) -> dict:
    import sentence_transformers
    import transformers

    body = open(a.text_in).read()
    res = {"part": "embedder", "model": a.embedder, "chars": len(body), "bytes": len(body.encode()), "env": env_info(),
           "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__}
    m = _load_st(a.embedder, "cuda", torch.bfloat16, a.text_only).eval()
    params = sum(p.numel() for p in m.parameters())
    table = max((mod.weight.numel() for name, mod in m.named_modules()
                 if isinstance(mod, torch.nn.Embedding)), default=0)
    res["params"], res["token_table_params"] = params, table
    doc_prompt = (getattr(m, "prompts", {}) or {}).get("document", "")
    res["document_prompt"] = doc_prompt
    tk = m.tokenizer
    t = time.perf_counter()
    ids = tk(body, add_special_tokens=False)["input_ids"]
    res["tokens"], res["tokenize_whole_s"] = len(ids), time.perf_counter() - t
    print(f"embedder: {params / 1e6:.0f}M params ({table / 1e6:.0f}M in the token table); the text is {len(ids):,} of its "
          f"tokens, {len(body):,} chars; document prompt {doc_prompt!r}", flush=True)
    res["runs"] = []
    for n in a.chunk_tokens:
        room = n - len(tk(doc_prompt, add_special_tokens=False)["input_ids"]) - 2
        chunks = [tk.decode(ids[i:i + room]) for i in range(0, len(ids), room)]
        texts = [doc_prompt + c for c in chunks]
        for bs in a.batch_sizes:
            with torch.inference_mode():
                m.encode(texts[: 2 * bs], batch_size=bs, convert_to_tensor=True)  # warm-up
                torch.cuda.reset_peak_memory_stats()
                e2e = timed(lambda: m.encode(texts, batch_size=bs, convert_to_tensor=True), a.repeats)
                feats = [_features(m, texts[i:i + bs]) for i in range(0, len(texts), bs)]
                fwd = timed(lambda: [m(f) for f in feats], a.repeats)
            r = {"chunk_tokens": n, "batch_size": bs, "chunks": len(chunks), "e2e_s": e2e, "e2e_median_s": statistics.median(e2e),
                 "forward_s": fwd, "forward_median_s": statistics.median(fwd), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            r["e2e_chars_per_s"] = len(body) / r["e2e_median_s"]
            r["forward_chars_per_s"] = len(body) / r["forward_median_s"]
            res["runs"].append(r)
            print(f"  chunks of {n} tokens, batch {bs}: end to end {r['e2e_median_s']:.2f} s, forward only "
                  f"{r['forward_median_s']:.2f} s ({len(chunks):,} chunks, peak {r['peak_mem_gb']:.1f} GB)", flush=True)
            del feats

    q = "When is the Q4 launch, and who owns the launch checklist?"
    enc_q = getattr(m, "encode_query", None) or (lambda x, **k: m.encode(x, prompt_name="query", **k))
    with torch.inference_mode():
        for _ in range(5):
            enc_q([q])
        res["query_gpu_ms"] = statistics.median(timed(lambda: enc_q([q]), 50)) * 1e3

    del m
    torch.cuda.empty_cache()
    torch.set_num_threads(a.cpu_threads)
    mc = _load_st(a.embedder, "cpu", torch.float32, a.text_only).eval()
    enc_qc = getattr(mc, "encode_query", None) or (lambda x, **k: mc.encode(x, prompt_name="query", **k))
    n = a.chunk_tokens[0]
    room = n - 12
    chunks = [doc_prompt + tk.decode(ids[i:i + room]) for i in range(0, room * a.cpu_chunks, room)]
    with torch.inference_mode():
        for _ in range(3):
            enc_qc([q])
        q_cpu = statistics.median(timed(lambda: enc_qc([q]), 30)) * 1e3
        mc.encode(chunks[:8], batch_size=8)
        thr = timed(lambda: mc.encode(chunks, batch_size=8), 1)[0]
    n_chars = sum(len(c) - len(doc_prompt) for c in chunks)
    res["cpu"] = {"threads": a.cpu_threads, "dtype": "float32", "query_ms": q_cpu, "chunks": len(chunks), "chunk_tokens": n,
                  "ingest_s": thr, "ingest_chars_per_s": n_chars / thr}
    print(f"query embedding: GPU {res['query_gpu_ms']:.1f} ms, CPU ({a.cpu_threads} threads) {q_cpu:.1f} ms; CPU ingest "
          f"{n_chars / thr / 1e3:.1f}k chars/s", flush=True)

    # exact search over a store of chunk vectors (random unit vectors: speed only)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(a.search_rows, a.search_dim, generator=g)
    M = M / M.norm(dim=1, keepdim=True)
    qv = M[123] + 0.01 * torch.randn(a.search_dim, generator=g)

    def search(mat, vec):
        s = mat @ vec
        return torch.topk(s, 100).indices

    cpu_ms = statistics.median(timed(lambda: search(M, qv), 20)) * 1e3
    Mg, qg = M.to("cuda", torch.bfloat16), qv.to("cuda", torch.bfloat16)
    search(Mg, qg)
    gpu_ms = statistics.median(timed(lambda: search(Mg, qg), 50)) * 1e3
    res["search"] = {"rows": a.search_rows, "dim": a.search_dim, "cpu_ms": cpu_ms, "cpu_dtype": "float32",
                     "gpu_ms": gpu_ms, "gpu_dtype": "bfloat16", "top": 100}
    print(f"exact search over {a.search_rows:,} × {a.search_dim}: CPU {cpu_ms:.1f} ms, GPU {gpu_ms:.2f} ms", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=("writer", "embedder"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu_threads", type=int, default=8)
    # writer
    p.add_argument("--checkpoint")
    p.add_argument("--data", help="text-checks data dir (tokenizer)")
    p.add_argument("--length", type=int, default=1_048_576)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--stories", choices=("full", "smoke"), default="full")
    p.add_argument("--chunk_pages", type=int, default=512)
    p.add_argument("--cpu_tokens", type=int, default=65_536)
    p.add_argument("--text_out", default="Cache/e35/bench_doc.txt")
    # embedder
    p.add_argument("--text_in", default="Cache/e35/bench_doc.txt")
    p.add_argument("--embedder", default="google/embeddinggemma-2")
    p.add_argument("--text_only", type=int, default=1, help="load the text tower only (EmbeddingGemma 2)")
    p.add_argument("--chunk_tokens", type=int, nargs="+", default=[256, 512])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[64, 256])
    p.add_argument("--cpu_chunks", type=int, default=64)
    p.add_argument("--search_rows", type=int, default=625_000)
    p.add_argument("--search_dim", type=int, default=256)
    a = p.parse_args()
    res = bench_writer(a) if a.part == "writer" else bench_embedder(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

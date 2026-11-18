"""Hub smoke test for a release: load it as a reader would, score one sentence, sample 20 tokens (2026-11-18).

The release checklist (`.cursor/skills/model-release`, `.cursor/skills/huggingface-project`) runs this after the
private upload and before the repository goes public; the card's "How to use" shows the `from_pretrained` snippet only
when it passed. It imports nothing from this repository, so run it in an environment with only `torch` and
`transformers` (and in isolated mode, so the repo's `nn/` cannot be picked up):

    python -I scripts/hub_smoke_test.py Meridian21Lab/<repo>            # from the Hub (a private repo needs a login)
    python -I scripts/hub_smoke_test.py /path/to/export --local         # an export folder, local files only

Checks: the model class comes from the repository's own code (`transformers_modules`), the sentence's loss is finite
and below `--max_loss`, 20 sampled tokens (seed 0, temperature 0.7, top-p 0.9) decode to text. Prints one JSON line;
exits 1 when a check fails. Engineering spec: docs/engineering_specs/hub_from_pretrained_loading.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

SENTENCE = "The capital of France is Paris, and the capital of Italy is Rome."
PROMPT = "The capital of France is"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="Hub id (or an export folder with --local)")
    ap.add_argument("--local", action="store_true", help="local files only")
    ap.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    ap.add_argument("--max_loss", type=float, default=6.0, help="fail above this loss (nats per token)")
    a = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, a.dtype) if device.startswith("cuda") else torch.float32
    kw = dict(local_files_only=a.local, revision=a.revision)
    t0 = time.time()
    # trust_remote_code on the tokenizer too: Transformers 5 reads the model config (custom code) to pick the
    # tokenizer, and without it asks interactively (4.57: no effect)
    tok = AutoTokenizer.from_pretrained(a.repo, trust_remote_code=True, **kw)
    model = AutoModelForCausalLM.from_pretrained(a.repo, trust_remote_code=True, dtype=dtype, **kw).to(device).eval()
    t_load = time.time() - t0

    enc = tok(SENTENCE, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = float(model(**enc, labels=enc["input_ids"]).loss)
    p = tok(PROMPT, return_tensors="pt").to(device)
    torch.manual_seed(0)
    t0 = time.time()
    out = model.generate(**p, max_new_tokens=20, min_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9)
    t_gen = time.time() - t0
    sample = tok.decode(out[0, p["input_ids"].shape[1]:], skip_special_tokens=True)

    own_code = type(model).__module__.startswith("transformers_modules.")
    repo_code = sorted({m.split(".")[0] for m in sys.modules} & {"nn", "evaluation", "training"})
    checks = {
        "class_from_repo_code": own_code,
        "no_mrcogito_modules_imported": not repo_code,
        "loss_finite_and_below_max": math.isfinite(loss) and loss < a.max_loss,
        "sampled_20_tokens": int(out.shape[1] - p["input_ids"].shape[1]) == 20 and bool(sample.strip()),
    }
    report = {
        "repo": a.repo, "revision": a.revision, "ok": all(checks.values()), "checks": checks,
        "class": f"{type(model).__module__}.{type(model).__name__}", "transformers": transformers.__version__,
        "torch": torch.__version__, "device": device, "dtype": str(dtype).replace("torch.", ""),
        "sentence_loss": round(loss, 4), "prompt": PROMPT, "sample": sample,
        "seconds": {"load": round(t_load, 1), "generate_20": round(t_gen, 2)},
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

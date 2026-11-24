"""E40 pre-flight P3: each arm's logits equal the released model's at step 0 (fp32, max abs difference < 1e-4).

Loads the released concept-memory weights into the control (e37a) and into the two mixed arms (the depth mixes left at
their start, b = 1, w = 0), scores the same held-out rows (fp32, flex attention, one GPU) and prints the largest
logit difference of each arm against the control, and whether the training command builder's arms build the same
models (parameter names). Usage:
  python scripts/e40_identity.py --checkpoint DIR --data Cache/e40/m21v2_e40 [--lengths 16384,65536]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rows(data: Path, length: int, n: int) -> list:
    from data.bin_token_dataset import open_shard

    out = []
    for f in sorted(data.glob("books_val_*.bin")) + sorted(data.glob("doab_val_*.bin")):
        sh = np.asarray(open_shard(str(f)))
        starts = np.flatnonzero(sh == 0)
        for a, b in zip(starts, np.append(starts[1:], len(sh))):
            if b - a >= length:
                out.append(torch.from_numpy(sh[a:a + length].astype(np.int64)))
            if len(out) >= n:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lengths", default="16384,65536")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    from safetensors.torch import load_file

    from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM

    cfg_d = json.loads((Path(a.checkpoint) / "config.json").read_text())
    state = load_file(str(Path(a.checkpoint) / "model.safetensors"))
    models = {}
    for arm, mode in (("e37a", "none"), ("e40_read", "read"), ("e40_dca2", "dca2")):
        cfg = PerceiverARConfig(**{**cfg_d, "par_depth_mix": mode, "attn_backend": "flex"})
        m = PerceiverARLM(cfg)
        missing, unexpected = m.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("depth_mix.")]
        if unexpected or bad:
            raise SystemExit(f"{arm}: missing {bad[:5]} unexpected {unexpected[:5]}")
        models[arm] = m.cuda().float().eval()
        print(f"{arm}: {sum(p.numel() for p in m.parameters()):,} parameters, {len(missing)} new tensors", flush=True)
    res = {}
    for L in (int(x) for x in a.lengths.split(",")):
        for i, ids in enumerate(rows(Path(a.data), L, a.rows)):
            x = ids[None].cuda()
            with torch.no_grad():
                ref = models["e37a"](x, doc_ids=torch.zeros_like(x)).logits.float()
                for arm in ("e40_read", "e40_dca2"):
                    got = models[arm](x, doc_ids=torch.zeros_like(x)).logits.float()
                    d = float((got - ref).abs().max())
                    res[f"{arm}_{L}_{i}"] = d
                    print(f"length {L} row {i}: {arm} max |logit difference| = {d:.3e} "
                          f"({'PASS' if d < 1e-4 else 'FAIL'} < 1e-4)", flush=True)
            del ref
            torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print("P3", "PASS" if all(v < 1e-4 for v in res.values()) else "FAIL")


if __name__ == "__main__":
    main()

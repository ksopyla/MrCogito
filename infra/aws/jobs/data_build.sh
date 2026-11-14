#!/bin/bash
# E33n data, built from public Hugging Face sources into DATA (no credentials, no S3):
#   fineweb_edu_gpt2/   kjj0/finewebedu10B-gpt2: the first N_SHARDS train shards (100M tokens each) + the validation shard
#   tokenizer_gpt2/     GPT-2 tokenizer with <|endoftext|> as the pad token (the trainer needs a pad id)
#   longdocs_lite_gpt2/ LSA's LongDocs recipe from the Common Pile, documents ≥ 16k tokens, TOKENS_PER_SOURCE each
#                       (0 skips it)
# Used by data_prep.sh on the AWS CPU node (60 shards, 300e6) and directly on Odra / Polonez for GPU checks.
# Usage: bash infra/aws/jobs/data_build.sh DATA [N_SHARDS] [TOKENS_PER_SOURCE] [THREADS]
set -euo pipefail
D="$1"; NSH="${2:-60}"; TPS="${3:-300e6}"; TH="${4:-16}"
mkdir -p "$D"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY=(uv run --no-project --python 3.12 --with tiktoken==0.12.0 --with "huggingface_hub>=0.24" --with numpy
    --with "transformers>=4.57,<5")

echo "== FineWeb-Edu GPT-2 shards ($NSH) $(date -u)"
"${PY[@]}" python - "$D" "$NSH" <<'EOF'
import sys
from huggingface_hub import snapshot_download
d, n = sys.argv[1], int(sys.argv[2])
pats = ["finewebedu_val_000000.bin"] + [f"finewebedu_train_{i:06d}.bin" for i in range(1, n + 1)]
snapshot_download("kjj0/finewebedu10B-gpt2", repo_type="dataset", local_dir=f"{d}/fineweb_edu_gpt2",
                  allow_patterns=pats, max_workers=16)
EOF

echo "== tokenizer"
"${PY[@]}" python - "$D" <<'EOF'
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("openai-community/gpt2")
t.pad_token = t.eos_token
t.save_pretrained(f"{sys.argv[1]}/tokenizer_gpt2")
print(len(t), t.pad_token_id, t.eos_token_id)
EOF

if [ "$TPS" != 0 ]; then   # TOKENS_PER_SOURCE 0: trunk data only
  echo "== LongDocs-lite ($TPS tokens per source) $(date -u)"
  "${PY[@]}" python scripts/build_longdocs_lite.py --out "$D/longdocs_lite_gpt2" --tokens_per_source "$TPS" --threads "$TH"
fi
echo "data done $(date -u)"

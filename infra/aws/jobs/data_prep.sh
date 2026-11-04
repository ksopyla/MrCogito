#!/bin/bash
# mc: no-sync
# E33n data job (CPU node, c7i.4xlarge): FineWeb-Edu GPT-2 shards and LongDocs-lite → s3://BUCKET/data/.
#   fineweb_edu_gpt2/   kjj0/finewebedu10B-gpt2: first 60 train shards (6B tokens) + the validation shard
#   longdocs_lite_gpt2/ LSA's LongDocs recipe from the Common Pile, documents ≥ 16k tokens, 300M train tokens per source
#   tokenizer_gpt2/     GPT-2 tokenizer with <|endoftext|> as the pad token (the trainer needs a pad id)
set -euo pipefail
source /etc/mrcogito.env
D="$MC_DISK/data"
mkdir -p "$D"
PY="uv run --no-project --python 3.12 --with tiktoken==0.12.0 --with huggingface_hub>=0.24 --with numpy --with transformers>=4.57,<5"

echo "== FineWeb-Edu GPT-2 shards $(date -u)"
$PY python - <<EOF
from huggingface_hub import snapshot_download
pats = ["finewebedu_val_000000.bin"] + [f"finewebedu_train_{i:06d}.bin" for i in range(1, 61)]
snapshot_download("kjj0/finewebedu10B-gpt2", repo_type="dataset", local_dir="$D/fineweb_edu_gpt2",
                  allow_patterns=pats, max_workers=16)
EOF
aws s3 sync --only-show-errors "$D/fineweb_edu_gpt2/" "s3://$MC_BUCKET/data/fineweb_edu_gpt2/" --exclude ".cache/*"
echo "fineweb done $(date -u)"

echo "== tokenizer"
$PY python - <<EOF
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("openai-community/gpt2")
t.pad_token = t.eos_token
t.save_pretrained("$D/tokenizer_gpt2")
print(len(t), t.pad_token_id, t.eos_token_id)
EOF
aws s3 sync --only-show-errors "$D/tokenizer_gpt2/" "s3://$MC_BUCKET/data/tokenizer_gpt2/"

echo "== LongDocs-lite $(date -u)"
$PY python scripts/build_longdocs_lite.py --out "$D/longdocs_lite_gpt2" --tokens_per_source 300e6 --threads 16
aws s3 sync --only-show-errors "$D/longdocs_lite_gpt2/" "s3://$MC_BUCKET/data/longdocs_lite_gpt2/"
echo "data job done $(date -u)"

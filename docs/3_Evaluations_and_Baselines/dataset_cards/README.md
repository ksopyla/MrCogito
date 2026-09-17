# CogitoProbe dataset cards

Four synthetic Hugging Face datasets for **long-haystack memory tests**.
Each card is written for an external reader: what the task is, why it exists,
and a copy-paste load snippet. Internal experiment nicknames stay out of the
Hub README.

Spec (research-internal):
[`docs/engineering_specs/concept_compression_probe_suite.md`](../../engineering_specs/concept_compression_probe_suite.md).
Computed statistics: [`cogito-probe-stats.json`](cogito-probe-stats.json)
(public v0, `--scale full`, seed `20260916`).

| Card | Hub | One-line job |
|---|---|---|
| [cogito-probe-bits](cogito-probe-bits/README.md) | [ksopyla/cogito-probe-bits](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | Recall values for keys buried in a haystack |
| [cogito-probe-bind](cogito-probe-bind/README.md) | [ksopyla/cogito-probe-bind](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | Who has which colour / who lives where / friend's city |
| [cogito-probe-arith](cogito-probe-arith/README.md) | [ksopyla/cogito-probe-arith](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | Nested arithmetic + bracket matching |
| [cogito-probe-props](cogito-probe-props/README.md) | [ksopyla/cogito-probe-props](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | Object colours amid fluent filler |

Each family: 10,240 rows (train 8448 / validation 896 / test 896). Author: Krzysztof Sopyła. License: Apache-2.0.

Rebuild the four README files from the stats JSON (does not touch parquet):

```bash
uv run python scripts/build_concept_probe_datasets.py \
  --from_stats docs/3_Evaluations_and_Baselines/dataset_cards/cogito-probe-stats.json \
  --cards_out docs/3_Evaluations_and_Baselines/dataset_cards
```

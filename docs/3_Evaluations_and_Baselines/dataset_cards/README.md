# CogitoProbe dataset cards

Four synthetic Hugging Face cards for the concept-compression series. Spec:
[`docs/engineering_specs/concept_compression_probe_suite.md`](../../engineering_specs/concept_compression_probe_suite.md).
Computed statistics: [`cogito-probe-stats.json`](cogito-probe-stats.json)
(public v0, `--scale full`, seed `20260916`).

| Card | Hub | Claim |
|---|---|---|
| [cogito-probe-bits](cogito-probe-bits/README.md) | [ksopyla/cogito-probe-bits](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | Unique-bit capacity vs haystack length |
| [cogito-probe-bind](cogito-probe-bind/README.md) | [ksopyla/cogito-probe-bind](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | Entity–attribute binding vs bag-of-tokens |
| [cogito-probe-arith](cogito-probe-arith/README.md) | [ksopyla/cogito-probe-arith](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | AST / Dyck-3 vs eval-only calculator shortcut |
| [cogito-probe-props](cogito-probe-props/README.md) | [ksopyla/cogito-probe-props](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | Proposition set vs fluent filler n-grams |

Each family: 10,240 rows (train 8448 / validation 896 / test 896). Author: Krzysztof Sopyła. License: Apache-2.0.

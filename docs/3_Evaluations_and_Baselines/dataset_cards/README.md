# CogitoProbe dataset cards

Four synthetic Hugging Face cards for the concept-compression series. Spec:
[`docs/engineering_specs/concept_compression_probe_suite.md`](../../engineering_specs/concept_compression_probe_suite.md).
Computed statistics: [`cogito-probe-stats.json`](cogito-probe-stats.json).

**Not uploaded.** Hub ids below are proposed; regenerate with
`scripts/build_concept_probe_datasets.py` then `hf upload` only after approval.

| Card | Claim |
|---|---|
| [cogito-probe-bits](cogito-probe-bits/README.md) | Unique-bit capacity vs haystack length |
| [cogito-probe-bind](cogito-probe-bind/README.md) | Entity–attribute binding vs bag-of-tokens |
| [cogito-probe-arith](cogito-probe-arith/README.md) | AST / Dyck-3 vs eval-only calculator shortcut |
| [cogito-probe-props](cogito-probe-props/README.md) | Proposition set vs fluent filler n-grams |

# Cogito Capability Checks — Hugging Face dataset cards

The `README.md` in each folder is the exact card on the Hub (source of truth for the card text).
Collection: https://huggingface.co/collections/ksopyla/cogito-capability-checks-6aca2db88ba9d2714727489f
One goal for the family: *does your model actually use what it read?* Project page: https://ai.ksopyla.com/projects/cogito-capability-datasets/

| Card | Hub | Question it answers |
|---|---|---|
| [cogito-probe-bits](cogito-probe-bits/README.md) | [ksopyla/cogito-probe-bits](https://huggingface.co/datasets/ksopyla/cogito-probe-bits) | How many key→value facts did it keep? (bits vs length) |
| [cogito-probe-bind](cogito-probe-bind/README.md) | [ksopyla/cogito-probe-bind](https://huggingface.co/datasets/ksopyla/cogito-probe-bind) | Who-has-what binding vs bag-of-tokens |
| [cogito-probe-arith](cogito-probe-arith/README.md) | [ksopyla/cogito-probe-arith](https://huggingface.co/datasets/ksopyla/cogito-probe-arith) | Expression tree vs eval-only calculator shortcut |
| [cogito-probe-props](cogito-probe-props/README.md) | [ksopyla/cogito-probe-props](https://huggingface.co/datasets/ksopyla/cogito-probe-props) | Facts vs fluent filler |
| [cogito-text-world](cogito-text-world/README.md) | [ksopyla/cogito-text-world](https://huggingface.co/datasets/ksopyla/cogito-text-world) | Text capability checks T1–T7 at 1k–128k (`text-world-v2`; v0 at tag `text-world-v0`) |

Probes: public v0, `--scale full`, seed `20260916`, 10,240 rows each (8448/896/896), Apache-2.0;
statistics in [`cogito-probe-stats.json`](cogito-probe-stats.json), spec
[`concept_compression_probe_suite.md`](../../engineering_specs/concept_compression_probe_suite.md).
Text World: CDLA-Sharing-1.0, spec `docs/engineering_specs/text_capability_checks.md` (branch
`e31-text-memory`).

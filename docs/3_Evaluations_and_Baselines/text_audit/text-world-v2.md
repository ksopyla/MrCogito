# text-world-v2 model-free audit (Odra build 2026-10-10 07:54 UTC, published tokenizer, all 29,400 frozen items, 1k–128k; analysis/text_checks_audit.py)

| split | task | n | floor | rule reader | twin: rule reader / answer in text | best shortcut | items with filler about the asked fact | candidates sharing 1st token |
|---|---|---|---|---|---|---|---|---|
| harder | compose | 800 | 0.06 | 100.0% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.2% | 0% |
| harder | count | 800 | 0.09 | 100.0% | 8% / — | all visits by the person (ignores the place): 0% (on 86%) | 16.0% | 0% |
| harder | deduce | 800 | 0.08 | 100.0% | 0% / — | property rule nearest the person's sentence: 11% (on 100%) | 0.0% | 0% |
| harder | keyed | 800 | 0.02 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 1.2% | 82% |
| harder | latest | 800 | 0.06 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 14% (on 100%) | 9.2% | 0% |
| harder | lookup | 800 | 0.06 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 4.6% | 0% |
| harder | quote | 800 | 0.04 | 100.0% | 0% / 0% | first sign: 0% (on 100%) | 4.2% | 0% |
| id | compose | 2600 | 0.08 | 100.0% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.4% | 0% |
| id | count | 2600 | 0.09 | 100.0% | 9% / — | all visits to the place (ignores the name): 1% (on 6%) | 33.4% | 0% |
| id | deduce | 2600 | 0.08 | 100.0% | 0% / — | last property rule: 9% (on 100%) | 0.0% | 0% |
| id | keyed | 2600 | 0.06 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 11.0% | 0% |
| id | latest | 2600 | 0.09 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 20% (on 100%) | 23.3% | 0% |
| id | lookup | 2600 | 0.08 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 11.5% | 0% |
| id | quote | 2600 | 0.08 | 100.0% | 0% / 0% | first sign: 0% (on 100%) | 13.2% | 0% |
| paraphrase | compose | 800 | 0.08 | 100.0% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.4% | 0% |
| paraphrase | count | 800 | 0.09 | 100.0% | 9% / — | all visits to the place (ignores the name): 2% (on 8%) | 26.5% | 0% |
| paraphrase | deduce | 800 | 0.08 | 100.0% | 0% / — | property rule nearest the person's sentence: 8% (on 100%) | 0.0% | 0% |
| paraphrase | keyed | 800 | 0.06 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 4.9% | 0% |
| paraphrase | latest | 800 | 0.09 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 20% (on 100%) | 16.1% | 0% |
| paraphrase | lookup | 800 | 0.08 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 6.8% | 0% |
| paraphrase | quote | 800 | 0.08 | 100.0% | 0% / 0% | first sign: 0% (on 100%) | 6.2% | 0% |

All shortcuts (accuracy on the items where the shortcut gives an answer, and that coverage):
- harder/compose · own home (0 hops): 0% on 100% of items
- harder/compose · one hop short: 0% on 100% of items
- harder/compose · reversed first link's home: 0% on 100% of items
- harder/count · all visits by the person (ignores the place): 0% on 86% of items
- harder/deduce · first property rule: 6% on 100% of items
- harder/deduce · last property rule: 8% on 100% of items
- harder/deduce · property rule nearest the person's sentence: 11% on 100% of items
- harder/keyed · first place: 0% on 100% of items
- harder/keyed · last place: 0% on 100% of items
- harder/keyed · look-alike's home: 0% on 92% of items
- harder/latest · last place in the document: 0% on 100% of items
- harder/latest · busiest mover's last place (ignores the name): 14% on 100% of items
- harder/latest · first home (no update): 0% on 100% of items
- harder/lookup · first place: 0% on 100% of items
- harder/lookup · last place: 0% on 100% of items
- harder/quote · first sign: 0% on 100% of items
- harder/quote · last sign: 0% on 100% of items
- id/compose · own home (0 hops): 0% on 100% of items
- id/compose · one hop short: 0% on 100% of items
- id/compose · reversed first link's home: 0% on 100% of items
- id/count · all visits by the person (ignores the place): 0% on 86% of items
- id/count · all visits to the place (ignores the name): 1% on 6% of items
- id/deduce · first property rule: 9% on 100% of items
- id/deduce · last property rule: 9% on 100% of items
- id/deduce · property rule nearest the person's sentence: 9% on 100% of items
- id/keyed · first place: 0% on 100% of items
- id/keyed · last place: 0% on 100% of items
- id/keyed · look-alike's home: 0% on 76% of items
- id/latest · last place in the document: 0% on 100% of items
- id/latest · busiest mover's last place (ignores the name): 20% on 100% of items
- id/latest · first home (no update): 0% on 100% of items
- id/lookup · first place: 0% on 100% of items
- id/lookup · last place: 0% on 100% of items
- id/quote · first sign: 0% on 100% of items
- id/quote · last sign: 0% on 100% of items
- paraphrase/compose · own home (0 hops): 0% on 100% of items
- paraphrase/compose · one hop short: 0% on 100% of items
- paraphrase/compose · reversed first link's home: 0% on 100% of items
- paraphrase/count · all visits by the person (ignores the place): 0% on 89% of items
- paraphrase/count · all visits to the place (ignores the name): 2% on 8% of items
- paraphrase/deduce · first property rule: 7% on 100% of items
- paraphrase/deduce · last property rule: 7% on 100% of items
- paraphrase/deduce · property rule nearest the person's sentence: 8% on 100% of items
- paraphrase/keyed · first place: 0% on 100% of items
- paraphrase/keyed · last place: 0% on 100% of items
- paraphrase/keyed · look-alike's home: 0% on 78% of items
- paraphrase/latest · last place in the document: 0% on 100% of items
- paraphrase/latest · busiest mover's last place (ignores the name): 20% on 100% of items
- paraphrase/latest · first home (no update): 0% on 100% of items
- paraphrase/lookup · first place: 0% on 100% of items
- paraphrase/lookup · last place: 0% on 100% of items
- paraphrase/quote · first sign: 0% on 100% of items
- paraphrase/quote · last sign: 0% on 100% of items

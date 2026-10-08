| split | task | n | floor | rule reader | twin: rule reader / answer in text | best shortcut | items with filler about the asked fact | candidates sharing 1st token |
|---|---|---|---|---|---|---|---|---|
| harder | compose | 800 | 0.12 | 99.8% | 0% / 0% | symmetric sister's home (1st hop reversed): 25% (on 100%) | 0.0% | 26% |
| harder | count | 800 | 0.14 | 100.0% | 12% / — | all visits to the place (ignores the name): 27% (on 73%) | 16.1% | 0% |
| harder | deduce | 800 | 0.50 | 100.0% | 0% / — | first property rule: 51% (on 100%) | 10.8% | 0% |
| harder | keyed | 800 | 0.02 | 100.0% | 0% / 0% | first place: 1% (on 100%) | 1.6% | 89% |
| harder | latest | 800 | 0.11 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 100% (on 100%) | 10.2% | 26% |
| harder | lookup | 800 | 0.00 | 100.0% | 0% / 0% | last place: 99% (on 100%) | 20.0% | — |
| harder | quote | 800 | 0.06 | 100.0% | 0% / 0% | first sign: 17% (on 100%) | 5.9% | 40% |
| id | compose | 2600 | 0.12 | 99.9% | 0% / 0% | own home (0 hops): 14% (on 100%) | 0.1% | 25% |
| id | count | 2600 | 0.14 | 99.9% | 14% / — | all visits to the place (ignores the name): 51% (on 89%) | 32.9% | 0% |
| id | deduce | 2600 | 0.50 | 100.0% | 0% / — | first property rule: 50% (on 100%) | 19.0% | 0% |
| id | keyed | 2600 | 0.06 | 100.0% | 0% / 0% | first place: 16% (on 100%) | 10.3% | 48% |
| id | latest | 2600 | 0.20 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 100% (on 100%) | 24.0% | 18% |
| id | lookup | 2600 | 0.00 | 100.0% | 0% / 0% | first place: 98% (on 100%) | 27.8% | — |
| id | quote | 2600 | 0.25 | 100.0% | 0% / 0% | last sign: 33% (on 100%) | 27.6% | 9% |
| paraphrase | compose | 800 | 0.12 | 99.9% | 0% / 0% | symmetric sister's home (1st hop reversed): 14% (on 100%) | 0.1% | 31% |
| paraphrase | count | 800 | 0.14 | 99.9% | 15% / — | all visits to the place (ignores the name): 49% (on 92%) | 29.5% | 0% |
| paraphrase | deduce | 800 | 0.50 | 100.0% | 0% / — | first property rule: 50% (on 100%) | 13.2% | 0% |
| paraphrase | keyed | 800 | 0.06 | 100.0% | 0% / 0% | first place: 18% (on 100%) | 4.8% | 48% |
| paraphrase | latest | 800 | 0.20 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 100% (on 100%) | 16.2% | 16% |
| paraphrase | lookup | 800 | 0.00 | 99.9% | 0% / 0% | first place: 100% (on 100%) | 22.2% | — |
| paraphrase | quote | 800 | 0.25 | 100.0% | 0% / 0% | first sign: 33% (on 100%) | 20.8% | 11% |

All shortcuts (accuracy on the items where the shortcut gives an answer, and that coverage):
- harder/compose · own home (0 hops): 13% on 100% of items
- harder/compose · one hop short: 0% on 100% of items
- harder/compose · symmetric sister's home (1st hop reversed): 25% on 100% of items
- harder/count · all visits to the place (ignores the name): 27% on 73% of items
- harder/count · all visits by the person (ignores the place): 0% on 68% of items
- harder/deduce · first property rule: 51% on 100% of items
- harder/deduce · last property rule: 49% on 100% of items
- harder/deduce · filler says it directly: 49% on 15% of items
- harder/keyed · first place: 1% on 100% of items
- harder/keyed · last place: 1% on 100% of items
- harder/keyed · look-alike's home: 0% on 92% of items
- harder/latest · last place in the document: 0% on 100% of items
- harder/latest · busiest mover's last place (ignores the name): 100% on 100% of items
- harder/latest · first home (no update): 0% on 100% of items
- harder/lookup · the only place word: 100% on 98% of items
- harder/lookup · first place: 98% on 100% of items
- harder/lookup · last place: 99% on 100% of items
- harder/quote · first sign: 17% on 100% of items
- harder/quote · last sign: 14% on 100% of items
- id/compose · own home (0 hops): 14% on 100% of items
- id/compose · one hop short: 0% on 100% of items
- id/compose · symmetric sister's home (1st hop reversed): 13% on 100% of items
- id/count · all visits to the place (ignores the name): 51% on 89% of items
- id/count · all visits by the person (ignores the place): 0% on 71% of items
- id/deduce · first property rule: 50% on 100% of items
- id/deduce · last property rule: 50% on 100% of items
- id/deduce · filler says it directly: 49% on 22% of items
- id/keyed · first place: 16% on 100% of items
- id/keyed · last place: 15% on 100% of items
- id/keyed · look-alike's home: 0% on 76% of items
- id/latest · last place in the document: 0% on 100% of items
- id/latest · busiest mover's last place (ignores the name): 100% on 100% of items
- id/latest · first home (no update): 0% on 100% of items
- id/lookup · the only place word: 100% on 95% of items
- id/lookup · first place: 98% on 100% of items
- id/lookup · last place: 98% on 100% of items
- id/quote · first sign: 33% on 100% of items
- id/quote · last sign: 33% on 100% of items
- paraphrase/compose · own home (0 hops): 13% on 100% of items
- paraphrase/compose · one hop short: 0% on 100% of items
- paraphrase/compose · symmetric sister's home (1st hop reversed): 14% on 100% of items
- paraphrase/count · all visits to the place (ignores the name): 49% on 92% of items
- paraphrase/count · all visits by the person (ignores the place): 0% on 75% of items
- paraphrase/deduce · first property rule: 50% on 100% of items
- paraphrase/deduce · last property rule: 50% on 100% of items
- paraphrase/deduce · filler says it directly: 45% on 18% of items
- paraphrase/keyed · first place: 18% on 100% of items
- paraphrase/keyed · last place: 16% on 100% of items
- paraphrase/keyed · look-alike's home: 0% on 78% of items
- paraphrase/latest · last place in the document: 0% on 100% of items
- paraphrase/latest · busiest mover's last place (ignores the name): 100% on 100% of items
- paraphrase/latest · first home (no update): 0% on 100% of items
- paraphrase/lookup · the only place word: 100% on 98% of items
- paraphrase/lookup · first place: 100% on 100% of items
- paraphrase/lookup · last place: 99% on 100% of items
- paraphrase/quote · first sign: 33% on 100% of items
- paraphrase/quote · last sign: 32% on 100% of items

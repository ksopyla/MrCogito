| split | task | n | floor | rule reader | twin: rule reader / answer in text | best shortcut | items with filler about the asked fact | candidates sharing 1st token |
|---|---|---|---|---|---|---|---|---|
| harder | compose | 200 | 0.12 | 99.5% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.5% | 16% |
| harder | count | 200 | 0.14 | 100.0% | 13% / — | all visits by the person (ignores the place): 0% (on 76%) | 1.0% | 0% |
| harder | deduce | 200 | 0.50 | 100.0% | 0% / — | first property rule: 50% (on 100%) | 0.0% | 0% |
| harder | keyed | 200 | 0.02 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 0.0% | 82% |
| harder | latest | 200 | 0.11 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 10% (on 100%) | 0.0% | 16% |
| harder | lookup | 200 | 0.25 | 100.0% | 0% / 0% | first place: 8% (on 100%) | 2.5% | 10% |
| harder | quote | 200 | 0.06 | 100.0% | 0% / 0% | first sign: 0% (on 100%) | 0.0% | 0% |
| id | compose | 600 | 0.12 | 100.0% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.0% | 15% |
| id | count | 600 | 0.14 | 99.8% | 14% / — | all visits to the place (ignores the name): 8% (on 9%) | 9.7% | 0% |
| id | deduce | 600 | 0.50 | 100.0% | 0% / — | last property rule: 53% (on 100%) | 0.0% | 0% |
| id | keyed | 600 | 0.06 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 1.7% | 29% |
| id | latest | 600 | 0.20 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 21% (on 100%) | 5.2% | 11% |
| id | lookup | 600 | 0.25 | 100.0% | 0% / 0% | first place: 9% (on 100%) | 7.7% | 6% |
| id | quote | 600 | 0.25 | 100.0% | 0% / 0% | first sign: 7% (on 100%) | 6.7% | 0% |
| paraphrase | compose | 200 | 0.12 | 100.0% | 0% / 0% | own home (0 hops): 0% (on 100%) | 0.0% | 18% |
| paraphrase | count | 200 | 0.14 | 100.0% | 12% / — | all visits to the place (ignores the name): 8% (on 6%) | 3.5% | 0% |
| paraphrase | deduce | 200 | 0.50 | 100.0% | 0% / — | first property rule: 50% (on 100%) | 0.0% | 0% |
| paraphrase | keyed | 200 | 0.06 | 100.0% | 0% / 0% | first place: 0% (on 100%) | 0.5% | 36% |
| paraphrase | latest | 200 | 0.20 | 100.0% | 0% / 0% | busiest mover's last place (ignores the name): 19% (on 100%) | 1.5% | 8% |
| paraphrase | lookup | 200 | 0.25 | 100.0% | 0% / 0% | first place: 10% (on 100%) | 2.5% | 8% |
| paraphrase | quote | 200 | 0.25 | 100.0% | 0% / 0% | first sign: 10% (on 100%) | 2.0% | 0% |

All shortcuts (accuracy on the items where the shortcut gives an answer, and that coverage):
- harder/compose · own home (0 hops): 0% on 100% of items
- harder/compose · one hop short: 0% on 100% of items
- harder/compose · reversed first link's home: 0% on 100% of items
- harder/count · all visits by the person (ignores the place): 0% on 76% of items
- harder/count · all visits to the place (ignores the name): 0% on 0% of items
- harder/deduce · first property rule: 50% on 100% of items
- harder/deduce · last property rule: 50% on 100% of items
- harder/deduce · filler says it directly: 100% on 1% of items
- harder/keyed · first place: 0% on 100% of items
- harder/keyed · last place: 0% on 100% of items
- harder/keyed · look-alike's home: 0% on 87% of items
- harder/latest · last place in the document: 0% on 100% of items
- harder/latest · busiest mover's last place (ignores the name): 10% on 100% of items
- harder/latest · first home (no update): 0% on 100% of items
- harder/lookup · first place: 8% on 100% of items
- harder/lookup · last place: 0% on 100% of items
- harder/quote · first sign: 0% on 100% of items
- harder/quote · last sign: 0% on 100% of items
- id/compose · own home (0 hops): 0% on 100% of items
- id/compose · one hop short: 0% on 100% of items
- id/compose · reversed first link's home: 0% on 100% of items
- id/count · all visits by the person (ignores the place): 0% on 78% of items
- id/count · all visits to the place (ignores the name): 8% on 9% of items
- id/deduce · first property rule: 47% on 100% of items
- id/deduce · last property rule: 53% on 100% of items
- id/deduce · filler says it directly: 25% on 1% of items
- id/keyed · first place: 0% on 100% of items
- id/keyed · last place: 0% on 100% of items
- id/keyed · look-alike's home: 0% on 77% of items
- id/latest · last place in the document: 0% on 100% of items
- id/latest · busiest mover's last place (ignores the name): 21% on 100% of items
- id/latest · first home (no update): 0% on 100% of items
- id/lookup · first place: 9% on 100% of items
- id/lookup · last place: 0% on 100% of items
- id/quote · first sign: 7% on 100% of items
- id/quote · last sign: 0% on 100% of items
- paraphrase/compose · own home (0 hops): 0% on 100% of items
- paraphrase/compose · one hop short: 0% on 100% of items
- paraphrase/compose · reversed first link's home: 0% on 100% of items
- paraphrase/count · all visits by the person (ignores the place): 0% on 80% of items
- paraphrase/count · all visits to the place (ignores the name): 8% on 6% of items
- paraphrase/deduce · first property rule: 50% on 100% of items
- paraphrase/deduce · last property rule: 50% on 100% of items
- paraphrase/keyed · first place: 0% on 100% of items
- paraphrase/keyed · last place: 0% on 100% of items
- paraphrase/keyed · look-alike's home: 0% on 74% of items
- paraphrase/latest · last place in the document: 0% on 100% of items
- paraphrase/latest · busiest mover's last place (ignores the name): 19% on 100% of items
- paraphrase/latest · first home (no update): 0% on 100% of items
- paraphrase/lookup · first place: 10% on 100% of items
- paraphrase/lookup · last place: 0% on 100% of items
- paraphrase/quote · first sign: 10% on 100% of items
- paraphrase/quote · last sign: 0% on 100% of items

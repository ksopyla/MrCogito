# E25 wall plots

One comparison panel per measured wall (dense vs E18 vs E21 vs window-only).
PNG is the GitHub-facing figure; SVG is the vector original. Per-hunt learning
curves stay with the probe JSON under `/opt/cursor/artifacts/e25_*/` (not in git).

| group | what | files |
|---|---|---|
| [`index/`](index/) | copy / bandwidth | `tiny_far_copy`, `r16_1024_vs_1536` |
| [`match/`](match/) | key→value lookup | `identity_1280_vs_1536`, `r8_pool_1024_vs_1280`, `ratio_r1_vs_r4_at_1280`, `match2_nd1_1024_vs_1280` |
| [`select/`](select/) | fact vs decoy | `length_692_vs_696`, `pooling_r8_r12_r16` |
| [`hops/`](hops/) | multi-edge chains | `ordered_264_272_288`, `shuffled_nd0_nd1_extrahop` |
| [`transfer/`](transfer/) | extra exclusive hop | `extrahop_hops_vs_match` |
| [`overview/`](overview/) | all scored E21 bits | `all_scored_rungs_e21_bits` |

Numbers for those panels: [`metrics.md`](metrics.md). Reconstruction metadata: [`_meta/fill_meta.json`](_meta/fill_meta.json).

Canonical write-up: [`../e25_e21_dna_capability_report_20260916.md`](../e25_e21_dna_capability_report_20260916.md).
Hunt catalogue: [`../e25_README.md`](../e25_README.md).

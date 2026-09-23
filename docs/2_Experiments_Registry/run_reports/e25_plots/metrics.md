# E25 wall metrics (bits / information flow / bytes-per-token)

Companion to the grouped figures in this folder. Canonical narrative:
[`../e25_e21_dna_capability_report_20260916.md`](../e25_e21_dna_capability_report_20260916.md).

cell | dense bits / flow / B/tok | e18 bits / flow / B/tok | e21 bits / flow / B/tok | e18_local bits / flow / B/tok | json
---|---|---|---|---|---
**First rung: tiny packed far_copy (seq=128, r=16 complete-block)** | | | | |
tiny INDEX @2400 | 63.31 / 0.989 / 0.06182 | 63.05 / 0.985 / 0.06157 | 17.70 / 0.277 / 0.01729 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_tiny_far_copy/tiny_far_copy.json`
tiny INDEX @8k near-pass | — (not scored) | — (not scored) | 47.01 / 0.734 / 0.0459 | — (not scored) | `/opt/cursor/artifacts/e25_tiny_far_copy_e21_steps/tiny_far_copy.json`
**INDEX r=16 frozen mean wall (1024 S1 PASS, 1536 FAIL]** | | | | |
1024 S1 PASS | 63.95 / 0.999 / 0.007806 | 63.12 / 0.986 / 0.007705 | 53.82 / 0.841 / 0.00657 | 0.00 / 0.000 / 6.615e-08 | `/opt/cursor/artifacts/e25_bridge1k_ip_r16_mean/bridge_1k_far_copy.json`
1536 S1 FAIL | 62.41 / 0.975 / 0.005079 | 0.00 / 0.000 / 0 | 0.00 / 0.000 / 0 | 0.02 / 0.000 / 1.919e-06 | `/opt/cursor/artifacts/e25_bridge1k_1536_ip_r16_mean/bridge_1k_far_copy.json`
**MATCH identity wall (1280 S1 PASS, 1536 FAIL]** | | | | |
1280 identity PASS | 63.97 / 0.999 / 0.006247 | 63.95 / 0.999 / 0.006245 | 63.94 / 0.999 / 0.006244 | 0.00 / 0.000 / 2.257e-07 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_h256_recall/bridge_1k_recall_single.json`
1536 identity FAIL | 63.95 / 0.999 / 0.005204 | 63.95 / 0.999 / 0.005204 | 1.50 / 0.023 / 0.0001219 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall/bridge_1k_recall_single.json`
**MATCH r=8 frozen-mean wall (1024 S1 PASS, 1280 FAIL]** | | | | |
1024 r=8 PASS | 62.90 / 0.983 / 0.007679 | 62.61 / 0.978 / 0.007643 | 60.35 / 0.943 / 0.007367 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_ip_r8_h256_recall/bridge_1k_recall_single.json`
1280 r=8 FAIL | 63.85 / 0.998 / 0.006235 | 63.95 / 0.999 / 0.006245 | 0.00 / 0.000 / 0 | 0.00 / 0.000 / 4.377e-07 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r8_h256_recall/bridge_1k_recall_single.json`
**MATCH pooling-ratio wall at 1280 (r=1 PASS, r=4 FAIL]** | | | | |
1280 r=1 identity PASS | 63.97 / 0.999 / 0.006247 | 63.95 / 0.999 / 0.006245 | 63.94 / 0.999 / 0.006244 | 0.00 / 0.000 / 2.257e-07 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_h256_recall/bridge_1k_recall_single.json`
1280 r=4 rem-off FAIL | — (not scored) | 63.05 / 0.985 / 0.006157 | 0.00 / 0.000 / 0 | 0.00 / 0.000 / 4.219e-07 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r4_h256_recall_s8k/bridge_1k_recall_single.json`
**MATCH2 n_dist=1 identity wall (1024 S1 PASS, 1280 FAIL]** | | | | |
1024 MATCH2 nd1 PASS | 58.81 / 0.919 / 0.007179 | 0.00 / 0.000 / 0 | 61.81 / 0.966 / 0.007545 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_ip_id_match2_nd1/bridge_1k_recall.json`
1280 MATCH2 nd1 FAIL | 62.94 / 0.983 / 0.006147 | 0.00 / 0.000 / 3.028e-07 | 0.01 / 0.000 / 1.124e-06 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1/bridge_1k_recall.json`
**SELECT identity length wall (692 S1 PASS, 696 FAIL]** | | | | |
692 PASS | 63.97 / 1.000 / 0.01156 | 63.89 / 0.998 / 0.01154 | 62.76 / 0.981 / 0.01134 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_692_ip_id_h256_select/bridge_1k_select_1decoy.json`
696 FAIL | 63.95 / 0.999 / 0.01148 | 63.93 / 0.999 / 0.01148 | 0.00 / 0.000 / 1.463e-07 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_696_ip_id_h256_select/bridge_1k_select_1decoy.json`
**SELECT pooling @512 (r=8 rem-off PASS, r=12 rem-on PASS, r=16 rem-on S1 edge)** | | | | |
r=8 rem-off PASS | 47.91 / 0.998 / 0.0117 | 47.90 / 0.998 / 0.0117 | 45.53 / 0.948 / 0.01111 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge512_ip_r8_select/bridge_select_1decoy.json`
r=12 rem-on PASS | 47.93 / 0.999 / 0.0117 | 47.90 / 0.998 / 0.01169 | 43.29 / 0.902 / 0.01057 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge512_ip_r12_rem_select/bridge_select_1decoy.json`
r=16 rem-on edge | 47.71 / 0.994 / 0.01165 | 47.94 / 0.999 / 0.0117 | 35.94 / 0.749 / 0.008774 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge512_ip_r16_rem_select/bridge_select_1decoy.json`
**Ordered hops glob=2 (264 S1 PASS, 272 FAIL severed, 288 FAIL extra-hop)** | | | | |
264 glob2 S1 PASS | — (not scored) | 9.87 / 0.380 / 0.004672 | 25.61 / 0.985 / 0.01212 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge264_chain_k13_glob2_s8k/bridge_chain_ordered.json`
272 glob2 FAIL | — (not scored) | 23.64 / 0.909 / 0.01087 | 0.02 / 0.001 / 7.987e-06 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_s8k/bridge_chain_ordered.json`
288 extra-hop FAIL | 25.77 / 0.991 / 0.01118 | 0.00 / 0.000 / 1.361e-06 | 0.00 / 0.000 / 0 | 0.00 / 0.000 / 8.109e-07 | `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2_extrahop/bridge_chain_ordered.json`
**Shuffled hops (n_dist=0 S1 PASS, n_dist=1 FAIL, extra-hop n_dist=1/2 PASS)** | | | | |
nd0 S1 PASS | 25.66 / 0.987 / 0.01253 | 16.37 / 0.630 / 0.007993 | 16.60 / 0.638 / 0.008104 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge256_chain_shuf0_k13_glob2/bridge_chain_shuffled.json`
nd1 S1 FAIL | 21.54 / 0.829 / 0.01052 | 25.44 / 0.979 / 0.01242 | 6.28 / 0.242 / 0.003069 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2/bridge_chain.json`
nd1 extra-hop PASS | 23.92 / 0.920 / 0.01168 | 9.75 / 0.375 / 0.004762 | 24.08 / 0.926 / 0.01176 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_extrahop/bridge_chain.json`
nd2 extra-hop PASS | 22.94 / 0.882 / 0.0112 | 0.00 / 0.000 / 0 | 22.60 / 0.869 / 0.01103 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge256_chain_shuf_k13_glob2_extrahop/bridge_chain.json`
**Extra-hop transfer: hops-composition PASS, MATCH/MATCH2 length FAIL** | | | | |
hops 272 extra-hop PASS | — (not scored) | 25.49 / 0.980 / 0.01171 | 25.48 / 0.980 / 0.01171 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_extrahop_s8k/bridge_chain_ordered.json`
MATCH2 1280 extra-hop FAIL | 63.20 / 0.987 / 0.006171 | 0.00 / 0.000 / 0 | 0.01 / 0.000 / 5.159e-07 | 0.01 / 0.000 / 8.564e-07 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1_extrahop/bridge_1k_recall.json`
MATCH r=8 1280 extra-hop FAIL | 58.94 / 0.921 / 0.005756 | 63.99 / 1.000 / 0.006249 | 0.01 / 0.000 / 9.515e-07 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r8_h256_recall_extrahop/bridge_1k_recall_single.json`
MATCH id 1536 extra-hop FAIL | 63.92 / 0.999 / 0.005202 | 63.94 / 0.999 / 0.005204 | 0.00 / 0.000 / 0 | 0.00 / 0.000 / 0 | `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall_extrahop/bridge_1k_recall_single.json`

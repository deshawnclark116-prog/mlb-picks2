# Phase 1A component tables (DEVELOPMENT: 2025 + 2026 wk1-3, burned; never holdout results)

## Team environment (negative-binomial distributions; baseline = shrunk current/prior team average)

| target | selected | base MAE | sel MAE | base CRPS | sel CRPS | 2025 base->sel MAE | 2026 wk1-3 base->sel MAE | p(MAE) | p(CRPS) |
|---|---|---|---|---|---|---|---|---|---|
| plays | C2_poisson_glm | 6.842 | 6.746 | 4.863 | 4.787 | 6.808 -> 6.754 | 7.039 -> 6.697 | 0.071 | 0.062 |
| dropbacks | B0_blend | 6.517 | 6.517 | 4.607 | 4.607 | 6.520 -> 6.520 | 6.503 -> 6.503 | 1.0 | 1.0 |
| rushes | C2_poisson_glm | 5.633 | 5.497 | 3.945 | 3.860 | 5.725 -> 5.558 | 5.101 -> 5.150 | 0.0015 | 0.0015 |
| rushes_designed | C2_poisson_glm | 5.546 | 5.337 | 3.899 | 3.797 | 5.619 -> 5.372 | 5.122 -> 5.140 | 0.0005 | 0.001 |
| targets | C1_opp_adjust | 5.924 | 5.850 | 4.155 | 4.118 | 5.989 -> 5.873 | 5.545 -> 5.716 | 0.0335 | 0.0795 |
| rz_plays | C2_poisson_glm | 3.997 | 3.906 | 2.809 | 2.757 | 3.998 -> 3.922 | 3.989 -> 3.818 | 0.038 | 0.0265 |
| rz_rushes | C2_poisson_glm | 2.480 | 2.429 | 1.741 | 1.705 | 2.474 -> 2.435 | 2.515 -> 2.398 | 0.0615 | 0.0285 |
| rz_targets | C2_poisson_glm | 2.280 | 2.220 | 1.597 | 1.567 | 2.299 -> 2.232 | 2.174 -> 2.148 | 0.0085 | 0.018 |
| gl_plays | B0_blend | 1.792 | 1.792 | 1.219 | 1.219 | 1.801 -> 1.801 | 1.739 -> 1.739 | 1.0 | 1.0 |
| gl_rushes | B0_blend | 1.224 | 1.224 | 0.824 | 0.824 | 1.207 -> 1.207 | 1.318 -> 1.318 | 1.0 | 1.0 |
| def_snaps | C2_poisson_glm | 11.934 | 11.745 | 8.539 | 8.342 | 12.038 -> 11.915 | 11.333 -> 10.762 | 0.0795 | 0.0105 |

## Availability P(active) (baseline = status lookup)

| side | time | selected | base logloss / Brier / ECE | selected logloss / Brier / ECE | 2025 ll base->sel | 2026 ll base->sel |
|---|---|---|---|---|---|---|
| offense | T24 | C2_xgb | 0.5749 / 0.1957 / 0.0308 | 0.2941 / 0.0898 / 0.0140 | 0.5453 -> 0.2885 | 0.7274 -> 0.3229 |
| offense | T90 | C2_xgb | 0.3082 / 0.0898 / 0.0103 | 0.1715 / 0.0521 / 0.0122 | 0.3036 -> 0.1675 | 0.3316 -> 0.1923 |
| defense | T24 | C2_xgb | 0.5529 / 0.1854 / 0.0358 | 0.3490 / 0.1072 / 0.0277 | 0.5152 -> 0.3201 | 0.7472 -> 0.4974 |
| defense | T90 | C2_xgb | 0.2912 / 0.0823 / 0.0101 | 0.2103 / 0.0603 / 0.0137 | 0.2801 -> 0.1928 | 0.3485 -> 0.3009 |

## Role-state share models (next-game share given playing; MAE; baseline R0 = last-8 mean)

| share type | n | R0 last8 | R1 ewma | R2 kalman | R3 bocpd | R4 hmm | R5 learned | selected | R5 vs R0 p | 3-class role-change logloss: freq / R5 |
|---|---|---|---|---|---|---|---|---|---|---|
| carry | 2491 | 0.0927 | 0.0876 | 0.0874 | 0.0919 | 0.1047 | 0.0856 | R5 | 0.0 | 1.0787 / 1.0897 |
| rb_carry | 1764 | 0.1055 | 0.0989 | 0.0984 | 0.1080 | 0.1152 | 0.0964 | R5 | 0.0 | 1.0908 / 1.0913 |
| qb_rush | 727 | 0.0617 | 0.0610 | 0.0605 | 0.0620 | 0.0656 | 0.0585 | R5 | 0.0135 | 1.0305 / 1.0052 |
| target | 6402 | 0.0479 | 0.0471 | 0.0478 | 0.0497 | 0.0551 | 0.0456 | R5 | 0.0 | 0.9122 / 0.9042 |
| route | 5534 | 0.1349 | 0.1225 | 0.1249 | 0.1346 | 0.1391 | 0.1190 | R5 | 0.0 | 1.0239 / 0.9629 |
| qb_att | 727 | 0.1677 | 0.1540 | 0.1554 | 0.1691 | 0.1686 | 0.1237 | R5 | 0.0 | 0.9793 / 1.0174 |
| rz_carry | 2316 | 0.1722 | 0.1698 | 0.1757 | 0.1895 | 0.2048 | 0.1618 | R5 | 0.0 | 1.0885 / 1.1479 |
| rz_target | 6008 | 0.1112 | 0.1107 | 0.1119 | 0.1183 | 0.1290 | 0.0904 | R5 | 0.0 | 1.0568 / 1.1408 |
| def_snap | 11031 | 0.1453 | 0.1299 | 0.1306 | 0.1442 | 0.1469 | 0.1243 | R5 | 0.0 | 1.0355 / 1.0092 |

## Opportunity allocation (player counts, pregame universe, non-participants = 0; selected chain vs dumb baseline)

| type | n | base MAE | sel MAE | base CRPS | sel CRPS | base cov80 | sel cov80 | 2025 CRPS base->sel | 2026 wk1-3 CRPS base->sel | p(MAE) | p(CRPS) | selected components |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry | 3951 | 2.607 | 1.895 | 1.819 | 1.378 | 0.8466 | 0.9099 | 1.8285 -> 1.377 | 1.7721 -> 1.3829 | 0.0 | 0.0 | coherent,role,avail,dispersion |
| target | 8887 | 1.356 | 1.155 | 0.963 | 0.828 | 0.8888 | 0.9275 | 0.9661 -> 0.8259 | 0.9441 -> 0.8358 | 0.0 | 0.0 | coherent,role,avail,team,dispersion |
| qb_att | 1397 | 11.135 | 5.189 | 7.702 | 3.902 | 0.5247 | 0.8912 | 7.7178 -> 3.9436 | 7.6147 -> 3.6756 | 0.0 | 0.0 | coherent,role,avail,dispersion |
| rz_carry | 3934 | 0.622 | 0.571 | 0.459 | 0.418 | 0.9212 | 0.941 | 0.4616 -> 0.4199 | 0.4438 -> 0.4071 | 0.0 | 0.0 | coherent,role,avail,dispersion |
| rz_target | 8861 | 0.288 | 0.285 | 0.223 | 0.213 | 0.928 | 0.9541 | 0.2261 -> 0.2159 | 0.2088 -> 0.196 | 0.133 | 0.0 | coherent,role,avail |
| route | 7447 | 6.299 | 4.911 | 4.753 | 3.586 | 0.8053 | 0.8565 | 4.7529 -> 3.5863 | n/a -> n/a | 0.0 | 0.0 | role,avail,dispersion |
| def_snap | 15495 | 13.829 | 9.958 | 9.999 | 7.412 | 0.8747 | 0.9069 | 9.7303 -> 7.2512 | 11.3849 -> 8.2401 | 0.0 | 0.0 | role,avail,team,dispersion |

## Leave-one-component-out (improvement in CRPS of the full chain over the chain without the component; + = component helps)

| type | coherent | role | avail | team | dispersion |
|---|---|---|---|---|---|
| carry | +0.1035 (p=0.0) | +0.0262 (p=0.008) | +0.3147 (p=0.0) | +0.0001 (p=0.5015) | +0.0181 (p=0.0) |
| target | +0.0254 (p=0.0) | +0.0251 (p=0.0) | +0.0924 (p=0.0) | +0.0019 (p=0.044) | +0.0048 (p=0.0) |
| qb_att | +1.0843 (p=0.0) | +0.1137 (p=0.0) | +2.0443 (p=0.0) | +0.0000 (p=1.0) | +0.0704 (p=0.0) |
| rz_carry | +0.0088 (p=0.0) | +0.0057 (p=0.005) | +0.0268 (p=0.0) | +0.0014 (p=0.272) | +0.0021 (p=0.036) |
| rz_target | +0.0015 (p=0.022) | +0.0069 (p=0.0) | +0.0092 (p=0.0) | +0.0001 (p=0.43) | +0.0001 (p=0.3665) |
| route | n/a | +0.2536 (p=0.0) | +0.6557 (p=0.0) | +0.0000 (p=1.0) | +0.2120 (p=0.0) |
| def_snap | n/a | +0.2645 (p=0.021) | +1.4938 (p=0.0) | +0.0491 (p=0.006) | +0.3179 (p=0.0) |

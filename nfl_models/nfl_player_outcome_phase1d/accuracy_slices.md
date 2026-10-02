# Accuracy slices (Phase 1D) - every within-tolerance figure names its universe

DEVELOPMENT (2025 + 2026 wk1-3; burned, not holdout). Simulated draws per row: 1000. Row A is the expanded pregame universe: most of its rows have a true outcome of 0 and are forecast near 0, so a large within-tolerance share there does NOT mean the forecasts of active players are that accurate; compare with rows D / E / H / F. B and C select on the game outcome and are diagnostics only.

## rush_yds, T24, combined 2025 + 2026 wk1-3

| universe | n | share of rows with true outcome 0 | MAE(median) | within +/-5 | within +/-10 | within +/-15 | within +/-20 | within +/-25 | within +/-30 | within +/-35 | within +/-40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 4615 | 0.5387 | 9.6179 | 0.6349 | 0.7244 | 0.7902 | 0.8418 | 0.8776 | 0.8988 | 0.9224 | 0.9404 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 2588 | 0.194 | 16.1433 | 0.3941 | 0.5379 | 0.6461 | 0.7322 | 0.7937 | 0.8296 | 0.8679 | 0.8972 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 2102 | 0.0195 | 19.7672 | 0.2712 | 0.4329 | 0.5618 | 0.6656 | 0.7402 | 0.7854 | 0.8344 | 0.8711 |
| D_protocol_eligible_stable_role | 694 | 0.2089 | 20.7258 | 0.3026 | 0.4164 | 0.5447 | 0.6412 | 0.7147 | 0.7594 | 0.8213 | 0.8473 |
| E_protocol_eligible_volatile_role | 347 | 0.4611 | 15.8991 | 0.4928 | 0.5677 | 0.6455 | 0.7147 | 0.7723 | 0.8098 | 0.8559 | 0.8876 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 1387 | 0.2978 | 19.5828 | 0.3655 | 0.4614 | 0.5703 | 0.6619 | 0.7304 | 0.77 | 0.8234 | 0.8587 |
| F_forecast_P(active)>=0.90 | 1705 | 0.1589 | 17.9564 | 0.3232 | 0.4815 | 0.5988 | 0.7009 | 0.7707 | 0.8123 | 0.8551 | 0.8874 |
| G_expected_rush_att_tier[0,1) | 2131 | 0.8855 | 1.3895 | 0.9451 | 0.9639 | 0.9704 | 0.977 | 0.9831 | 0.9855 | 0.9873 | 0.992 |
| G_expected_rush_att_tier[1,5) | 1483 | 0.3763 | 10.4405 | 0.5125 | 0.6864 | 0.7937 | 0.8503 | 0.8881 | 0.913 | 0.9326 | 0.9467 |
| G_expected_rush_att_tier[5,10) | 472 | 0.0763 | 21.43 | 0.1907 | 0.3263 | 0.4788 | 0.6377 | 0.7246 | 0.7754 | 0.8347 | 0.8771 |
| G_expected_rush_att_tier[10,1e+09) | 529 | 0.0095 | 29.9195 | 0.1248 | 0.2212 | 0.3327 | 0.4556 | 0.5595 | 0.62 | 0.7108 | 0.7713 |

## rec_yds, T24, combined 2025 + 2026 wk1-3

| universe | n | share of rows with true outcome 0 | MAE(median) | within +/-5 | within +/-10 | within +/-15 | within +/-20 | within +/-25 | within +/-30 | within +/-35 | within +/-40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 10060 | 0.5596 | 9.8884 | 0.5995 | 0.7036 | 0.7786 | 0.8352 | 0.8722 | 0.9017 | 0.9241 | 0.9409 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 6649 | 0.3453 | 14.3157 | 0.4198 | 0.57 | 0.6787 | 0.7615 | 0.8158 | 0.8589 | 0.8908 | 0.915 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 4984 | 0.1116 | 18.7659 | 0.263 | 0.443 | 0.5758 | 0.68 | 0.7506 | 0.808 | 0.8515 | 0.8842 |
| D_protocol_eligible_stable_role | 1710 | 0.2591 | 18.4846 | 0.2942 | 0.4292 | 0.5585 | 0.6655 | 0.7497 | 0.8053 | 0.8468 | 0.8801 |
| E_protocol_eligible_volatile_role | 855 | 0.3591 | 19.9861 | 0.3906 | 0.4772 | 0.5556 | 0.6433 | 0.6994 | 0.7673 | 0.8035 | 0.8433 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 3420 | 0.2921 | 18.6958 | 0.3281 | 0.4523 | 0.5658 | 0.6655 | 0.7392 | 0.7965 | 0.8348 | 0.8708 |
| F_forecast_P(active)>=0.90 | 4820 | 0.2734 | 16.332 | 0.328 | 0.5031 | 0.6313 | 0.7263 | 0.7905 | 0.8402 | 0.8772 | 0.9048 |
| G_expected_targets_tier[0,1) | 5043 | 0.8687 | 2.1507 | 0.9012 | 0.9322 | 0.952 | 0.9685 | 0.9756 | 0.9808 | 0.9873 | 0.9909 |
| G_expected_targets_tier[1,3) | 2553 | 0.396 | 12.1021 | 0.4313 | 0.6416 | 0.7638 | 0.8273 | 0.8637 | 0.8974 | 0.9221 | 0.9369 |
| G_expected_targets_tier[3,6) | 1721 | 0.122 | 20.7533 | 0.1761 | 0.3347 | 0.4823 | 0.627 | 0.7281 | 0.8019 | 0.8495 | 0.8838 |
| G_expected_targets_tier[6,1e+09) | 743 | 0.0377 | 29.6347 | 0.1104 | 0.2194 | 0.3392 | 0.4401 | 0.533 | 0.611 | 0.6743 | 0.747 |

## rec, T24, combined 2025 + 2026 wk1-3

| universe | n | share of rows with true outcome 0 | MAE(median) | within +/-0 | within +/-1 | within +/-2 | within +/-3 |
|---|---|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 10060 | 0.5552 | 0.8021 | 0.5663 | 0.8021 | 0.908 | 0.9583 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 6649 | 0.3387 | 1.146 | 0.3787 | 0.7173 | 0.8693 | 0.9404 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 4984 | 0.1021 | 1.46 | 0.237 | 0.6334 | 0.8236 | 0.9185 |
| D_protocol_eligible_stable_role | 1710 | 0.2532 | 1.3988 | 0.3146 | 0.6351 | 0.8158 | 0.914 |
| E_protocol_eligible_volatile_role | 855 | 0.3579 | 1.4754 | 0.3789 | 0.6082 | 0.7719 | 0.8877 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 3420 | 0.2883 | 1.4032 | 0.3415 | 0.6249 | 0.8061 | 0.9088 |
| F_forecast_P(active)>=0.90 | 4820 | 0.2664 | 1.281 | 0.3 | 0.6797 | 0.8539 | 0.9369 |
| G_expected_targets_tier[0,1) | 5043 | 0.8667 | 0.2245 | 0.8667 | 0.9498 | 0.9792 | 0.9905 |
| G_expected_targets_tier[1,3) | 2553 | 0.3858 | 1.0656 | 0.3392 | 0.7775 | 0.904 | 0.9522 |
| G_expected_targets_tier[3,6) | 1721 | 0.1162 | 1.5845 | 0.201 | 0.5584 | 0.8059 | 0.9233 |
| G_expected_targets_tier[6,1e+09) | 743 | 0.039 | 2.0047 | 0.1534 | 0.4482 | 0.6743 | 0.8412 |

## pass_yds, T24, combined 2025 + 2026 wk1-3

| universe | n | share of rows with true outcome 0 | MAE(median) | within +/-10 | within +/-20 | within +/-30 | within +/-40 | within +/-50 | within +/-60 | within +/-75 | within +/-100 | within +/-125 | within +/-150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 1688 | 0.5776 | 37.2687 | 0.5776 | 0.6143 | 0.657 | 0.6872 | 0.7198 | 0.763 | 0.8004 | 0.8596 | 0.9028 | 0.9342 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 763 | 0.0852 | 71.548 | 0.1625 | 0.2372 | 0.3263 | 0.3893 | 0.4548 | 0.5452 | 0.6252 | 0.7405 | 0.8191 | 0.8847 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 731 | 0.0246 | 77.0676 | 0.1108 | 0.1902 | 0.2832 | 0.3502 | 0.42 | 0.5144 | 0.5978 | 0.7196 | 0.803 | 0.87 |
| D_protocol_eligible_stable_role | 527 | 0.1025 | 67.652 | 0.1556 | 0.2239 | 0.3207 | 0.3852 | 0.4573 | 0.5522 | 0.63 | 0.7647 | 0.8425 | 0.907 |
| E_protocol_eligible_volatile_role | 264 | 0.8106 | 24.1469 | 0.7955 | 0.8182 | 0.8295 | 0.8447 | 0.8485 | 0.8598 | 0.8788 | 0.8902 | 0.9205 | 0.9394 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 1053 | 0.4103 | 51.0595 | 0.4198 | 0.4663 | 0.5214 | 0.5651 | 0.6087 | 0.6686 | 0.7217 | 0.8082 | 0.8689 | 0.9117 |
| F_forecast_P(active)>=0.90 | 492 | 0.0366 | 68.0478 | 0.0976 | 0.1809 | 0.2785 | 0.3496 | 0.4268 | 0.5325 | 0.622 | 0.7663 | 0.8516 | 0.9187 |
| G_expected_pass_att_tier[0,15) | 1088 | 0.864 | 18.6227 | 0.8438 | 0.8603 | 0.8741 | 0.8824 | 0.8915 | 0.9035 | 0.9118 | 0.9256 | 0.9338 | 0.9458 |
| G_expected_pass_att_tier[15,25) | 200 | 0.145 | 82.4849 | 0.075 | 0.11 | 0.195 | 0.26 | 0.315 | 0.41 | 0.5 | 0.645 | 0.815 | 0.89 |
| G_expected_pass_att_tier[25,1e+09) | 400 | 0.015 | 65.3779 | 0.105 | 0.1975 | 0.2975 | 0.37 | 0.455 | 0.5575 | 0.6475 | 0.7875 | 0.8625 | 0.925 |


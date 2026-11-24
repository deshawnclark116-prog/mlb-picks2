# NFL yardage model challenger — independent research summary

**Evaluation:** 2025 Week 6 through 2026 Week 11, 1,119 eligible rushing and 2,956 eligible receiving player-games. Weekly walk-forward fitting; every challenger input is pregame or strictly earlier-game data. No sportsbook features in challengers. The historical live v2 comparator retains its historical sportsbook features, to keep the comparison honest.

**STATUS: rushing surpassed the old best; receiving has not met the old best.**

| Metric | Rushing | Receiving |
|---|---:|---:|
| Live v2 historical refit MAE | 24.9902 | 22.5954 |
| Previous published best MAE | 24.86 | 22.44 |
| Best evaluated candidate | ens_full_median | tenure_oline_lgb |
| Best MAE | 24.5497 | 22.4491 |

## Best candidate detail

### Rushing — ens_full_median (Heterogeneous ensemble (rushing))
- MAE 24.5497 yards; gain 0.4405 vs live v2 and 0.3103 vs prior best (positive is improvement)
- Paired P(better): row 0.9920; week-block 0.9793; 95% week-block improvement CI [0.0171, 0.8324]
- CRPS 17.7541; Brier(reach 70% of own projection) 0.22247; Brier(fixed-v2-event) 0.21796

### Receiving — tenure_oline_lgb (Position + missed-week tenure)
- MAE 22.4491 yards; gain 0.1463 vs live v2 and -0.0091 vs prior best (positive is improvement)
- Paired P(better): row 0.9367; week-block 0.9707; 95% week-block improvement CI [-0.0042, 0.2864]
- CRPS 16.2686; Brier(reach 70% of own projection) 0.23760; Brier(fixed-v2-event) 0.22972

## All model-family experiments

Every model attempted is retained in [all_experiment_results.json](all_experiment_results.json), including failures. Here are the top ten for each market.

### Rushing

| Version | Model | MAE | CRPS | P(week-block better than v2) |
|---|---|---:|---:|---:|
| Heterogeneous ensemble (rushing) | ens_full_median | 24.5497 | 17.7541 | 0.9793 |
| Heterogeneous ensemble (rushing) | ens_full_equal | 24.5776 | 17.7606 | 0.9760 |
| Heterogeneous ensemble (rushing) | ens_full_adapt_cov | 24.5776 | 17.7606 | 0.9760 |
| Position + missed-week tenure | tenure_full_xgb | 24.5897 | 17.7702 | 0.9713 |
| Heterogeneous ensemble (rushing) | xgb_full | 24.5897 | 17.7702 | 0.9713 |
| Position + missed-week tenure | tenure_cb_full | 24.5992 | 17.8034 | 0.9687 |
| Heterogeneous ensemble (rushing) | cat_full | 24.5992 | 17.8034 | 0.9687 |
| LightGBM + CatBoost | lgb_role_recent | 24.6067 | 17.8004 | 0.9740 |
| Heterogeneous ensemble (rushing) | ens_all_mean | 24.6173 | 17.7851 | 0.9620 |
| Heterogeneous ensemble (rushing) | ens_all_adapt_cov | 24.6173 | 17.7851 | 0.9620 |

### Receiving

| Version | Model | MAE | CRPS | P(week-block better than v2) |
|---|---|---:|---:|---:|
| Position + missed-week tenure | tenure_oline_lgb | 22.4491 | 16.2686 | 0.9707 |
| Heterogeneous ensemble (receiving) | lgb_ol | 22.4491 | 16.2686 | 0.9707 |
| Heterogeneous ensemble (receiving) | ens_ol_equal | 22.4568 | 16.2596 | 0.9813 |
| Heterogeneous ensemble (receiving) | ens_all_mean | 22.4677 | 16.2628 | 0.9820 |
| Heterogeneous ensemble (receiving) | ens_all_adapt_cov | 22.4677 | 16.2628 | 0.9820 |
| Timestamped depth chart (receiving) | depth_full_cat | 22.4884 | 16.3221 | 0.9400 |
| Timestamped depth chart (receiving) | depth_oline_xgb | 22.4923 | 16.3160 | 0.8867 |
| Timestamped depth chart (receiving) | depth_oline_lgb | 22.4928 | 16.3213 | 0.9400 |
| Heterogeneous ensemble (receiving) | ens_full_median | 22.5103 | 16.2947 | 0.8653 |
| Heterogeneous ensemble (receiving) | ens_full_equal | 22.5224 | 16.2983 | 0.8407 |

## Audit constraints and limitations

- The 2025–26 historical period is heavily reused development data, NOT a pristine untouched test. The reported bootstrap values do not correct for the many candidate models we evaluated, so they are exploratory.
- Data used: nflverse play-by-play 2023–26, FTN charting 2023–26 (FTN Data via nflverse, CC BY-SA 4.0), weekly injuries and snap counts, and time-stamped depth charts only when their snapshot date strictly predates the game. Playing time/team lineup proxies are computed strictly from preceding weeks.
- The actual pregame availability of retrospective injury statuses is not independently timestamp-verifiable in the supplied weekly injuries CSV. Formal prospective validation should archive those designations at forecast time.
- Exact evaluation count assertions matched the earlier benchmark population in every completed test. The same-game played-player population is a retrospective evaluation filter; deploying to every candidate on a real pregame slate requires a separate eligibility model.
- Rush ensemble uses median of three model-family predictions and was not allowed to blend with the sportsbook-input live v2 baseline.
- A statistically convincing promotion requires prospective immutable forecasts scored against new later games, not another tuning sweep on these reused games.
- Production remains unchanged: all tests were in the draft research branch. No model is promoted automatically.

## Evidence and execution

- [FTN/role + injury result](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45090804241)
- [Offensive-line test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45091109777)
- [LightGBM/CatBoost test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45091168201)
- [Position/tenure test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45091400679)
- [Timestamped depth test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45091784542)
- [Historical-weighted ensemble test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45091850063)
- [Receiver specialist test](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45092108228)

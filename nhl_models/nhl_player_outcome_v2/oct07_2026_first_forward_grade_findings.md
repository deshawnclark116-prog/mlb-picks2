# NHL V2 Oct 7 2026 — first genuine forward outcome audit

**Evidence grade: CLEAN_FORWARD_DESCRIPTIVE_ONLY, NOT CONFIRMATORY; NO PROMOTION.**
Source: frozen B2 v1.1 forecasts in original hash-chained ledger, official-final game-center/stats-REST NHL SOG fetched after the games. Original model, engine lock and 926-row forecast ledger remain unchanged. This report was created after the games; the original predictions were not.

**Independent execution:** [GitHub Actions full grading run](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37841727071), [immutable summary artifact](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37841727071/artifacts/11578460802), Oct 8 2026. Separate CI verifies frozen engine lock and adversarial grade-key tests. Research-only isolated grader `nhl_v2_forward_integrity_grade.py` is explicitly not the frozen v1.1 original grader.

## Source and identity integrity

- Games 2026020053 Pittsburgh–Washington, 2026020054 Colorado–Winnipeg, 2026020055 Edmonton–Anaheim; Oct 7 2026.
- 176 T90 + 176 T30 immutable FORECAST rows. 4 identity-ambiguous rows per horizon (same game/horizon/start/player ID forecast under WSH *and* PIT), 8 total, explicitly UNGRADED before reading outcomes. No silent winner selection.
- Valid graded rows: 172 per horizon, 344 total. Missing official grades: zero; no deliberate target exclusions beyond pre-identified identity conflicts.
- Full population: 104 of 172 played, 68 of 172 did not play and correctly received zero SOG under this **unconditional** target. A full-universe MAE therefore benefits from many easy zero labels and cannot stand alone as a skill claim.
- Meaningful expected-participant population: 71 of 172; 68/71 actually played, 3/71 did not. These are the primary diagnostic counts.
- 47 of 172 full-population rows lacked a verified player-name display; 0 of 71 meaningful rows did. All 176 original rows per horizon recorded `NO_ROSTER_DATA_PUBLISHED_YET` and had NO certified pregame roster state.
- `human_frozen_mean`: missing for every original Oct 7 forecast. The same-season MIN_PRIOR=5 comparator rule cannot be satisfied this early. No claim against a professional analyst is possible from these data.

## Frozen projections vs actual shot counts

| Horizon | Population | n | MAE(expected SOG) | MAE(median forecast) | CRPS (game macro) | Bias | >2-shot misses | >3-shot misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T90 | FULL | 172 | 0.683074 | 0.610465 | 0.434301 | +0.106935 | 11 | 3 |
| T90 | MEANINGFUL | 71 | 0.994117 | 0.985915 | 0.671983 | +0.220912 | 6 | 1 |
| T30 | FULL | 172 | 0.683075 | 0.610465 | 0.434301 | +0.106936 | 11 | 3 |
| T30 | MEANINGFUL | 71 | 0.994121 | 0.985915 | 0.671984 | +0.220918 | 6 | 1 |

**Same-row simple baseline (not a professional analyst):** Among the 71 meaningful players with a pregame prior-10 average, B2 T90 MAE 0.9941 vs simple mean MAE 1.1549; descriptive improvement ~13.9%. No tournament/game bootstrap confidence interval and far below confirmation minimum. On full population 135 comparable rows, model MAE 0.8111 vs simple 1.1216. This is not a published predictive performance claim.

**T90 versus T30:** Central projections are effectively identical. No certified late-lineup/role evidence was used; short-horizon availability value has not been demonstrated.

## Selected large misses (T90)

| Player | Forecast mean | Actual SOG | Absolute error | Population |
|---|---:|---:|---:|---|
| Beckett Sennecke | 2.066 | 7 | 4.934 | MEANINGFUL |
| Alex Tuch | 1.824 | 6 | 4.176 | FULL nonmeaningful |
| Cutter Gauthier | 3.246 | 7 | 3.754 | FULL nonmeaningful |
| Connor McDavid | 3.992 | 1 | 2.992 | MEANINGFUL |
| Leo Carlsson | 2.940 | 0 | 2.940 | MEANINGFUL |
| Nathan MacKinnon | 3.791 | 1 | 2.791 | MEANINGFUL |

Postgame fixed-ratio large-miss classifications: 3 FULL and 1 MEANINGFUL >3-shot misses all classified `MULTIPLE`. These labels are descriptive flags from outcomes/actual ice time, not preregistered causal proof or forecasting inputs. Do not select one explanation post hoc.

## Required next architecture, not yet certified

1. **Roster/ownership:** before issuing player/team forecasts, qualify reliable pre-cutoff roster/transaction/line confirmations, resolve cross-team membership, and abstain when ambiguous. Certify recall on >=40 games across >=3 dates and >=99.5% final dressed-player recall at each horizon before deploying availability features.
2. **Opportunity and deployment:** model probability of dressing and conditional total TOI, EV/PP/PK deployments, line matching and recent role shifts; compare predicted TOI directly with realized TOI. All pregame features need timestamped sources; postgame TOI is an outcome/diagnostic only.
3. **Shot generation:** team shots/pace, opponent allowed and style, player shot-attempt share per 60, shot-on-goal conversion. Validate shot attempts and SOG separately. Preserve injury/matchup evidence as genuinely timed observations, not narrative bonuses.
4. **Human comparator:** start saving independent timestamped expert-level expected SOG/uncertainty (and a strong transparent statistical analyst baseline) before future games, especially early season using certified preceding-season evidence. No retrofitting historical "human" projections or using betting lines as model input.
5. **Validation:** a new v1.2+ engine/version/lock only after source and component gates. Head-to-head same rows and horizons, meaningful-player MAE, CRPS, dispersion, calibration, large-miss forensics, and source coverage. Freeze before its first forecast. Existing B2 forward confirmation rule still requires 28 calendar days and >=15,000 graded meaningful player-games at T90; no one-night promotion.

**Verdict:** Forward data acquisition, freeze, and official grading now work in an isolated verified path. Player/team identity and availability remain genuine forecast blockers. B2 has not established superiority over professional pregame research. Keep draft, unmerged, research-only.

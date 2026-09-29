# NFL player-outcome error budget (Phase 0)

Research only. No production serving, champion models, model artifacts
or API behaviour were changed. Script: `nfl_player_outcome_error_budget_a.py`.
Machine-readable results: `report.json` (same folder).

```
python -u nfl_player_outcome_error_budget_a.py --data-dir /tmp/nfl_data
```
Runtime ~60 s on cached nflverse files (2022-2026). Deterministic (seeded).

## Protocol

| | |
|---|---|
| Train | 2023 + 2024 weeks 1-12 |
| Validation | 2024 weeks 13-18 (early stopping, interval calibration) |
| Evaluation | 2025 (n = 854 RB rushing, 2,294 WR/TE/RB receiving, 539 QB, 2,631 TD rows) and 2026 weeks 1-3 (n = 148 / 390 / 96 / 427) |
| Population | players who actually played: a **conditional production forecast** ("assuming he is active"). Availability measured separately (below). |
| Incumbent | `nfl_yardage_v2.FEATURES` recipe, `reg:absoluteerror` XGBoost, **re-fit on the train split**. The shipped v2 artifact was trained on 2023-2025, so it cannot be scored honestly on 2025 (see leakage). |
| Features | strictly as-of: only games before the target week |

**Holdout status (honest):** 2025 has been evaluated by the v2 gate, v3 gate, v4
evaluation, five v5 experiments and this audit: it is a development set now, not
a holdout. 2026 weeks 1-3 have also been looked at repeatedly. The next clean
forward window is **2026 weeks 5-18**: freeze Phase 1 component models and their
gates before week 5 kickoffs, log every forecast before kickoff, score after.

Oracle rungs: *plug-in* = the decomposed product with that component replaced by
its actual value; *learned* = a model re-fit with the actual component as an input.
MAE is not additive, so budgets below are **sequential** (order: team -> player
share -> efficiency) and are measured from the decomposed all-predicted row.

## Oracle error budget: results

### Rushing yards (RB), 2025, n = 854

| Rung | MAE | medAE | RMSE | bias | ±10 | ±20 | ±30 |
|---|---|---|---|---|---|---|---|
| 0 incumbent (v2 recipe, re-fit) | 25.11 | 18.50 | 35.05 | -7.06 | 27.8% | 53.0% | 70.5% |
| 0b incumbent without spread/total | 25.15 | 19.10 | 35.00 | -7.46 | 27.0% | 52.2% | 72.8% |
| 1 decomposed, all predicted (team carries x share x YPC) | 25.44 | 19.60 | 33.99 | -0.14 | 24.6% | 50.7% | 70.6% |
| 2 oracle team carries (plug-in / learned) | 22.32 / 21.90 | 16.87 / 15.41 | | | 29.7 / 36.1% | 57.1 / 60.1% | 76.1 / 76.5% |
| 3 oracle player carries (plug-in / learned) | 16.67 / 16.32 | 11.55 / 11.12 | | | 44.5 / 47.2% | 71.1 / 73.0% | 84.7 / 85.5% |
| 4 oracle carries + non-explosive yards (tail predicted) | 10.93 | 4.29 | 20.98 | | 71.3% | 83.8% | 91.0% |
| 4b oracle carries + explosive (20+) yards (base predicted) | 11.09 | 8.59 | 14.63 | | 55.3% | 82.6% | 95.2% |
| exact | 0 | | | | | | |

2026 wk1-3 (n = 148): incumbent 24.89, decomposed 25.48, oracle team 21.38,
oracle carries 13.10, tail-predicted 7.73.

**Sequential budget (from 25.44):** team carry volume 3.1 (12%) -> player carry
share 5.7 (22%) -> per-carry efficiency 16.7 (66%). Within efficiency, knowing the
explosive-run yards alone or the ordinary-carry yards alone each removes ~5.6, so
roughly half of efficiency error is the 20+ yard tail.

Component forecast errors (2025): team carries MAE 5.84 (RMSE 7.34); carry share
MAE 0.126; player carries MAE 4.15; YPC MAE 1.48 (players with 5+ carries).

### Receiving yards (WR/TE/RB), 2025, n = 2,294

| Rung | MAE | medAE | bias | ±10 | ±20 | ±30 |
|---|---|---|---|---|---|---|
| 0 incumbent (re-fit) | 22.14 | 16.54 | -4.44 | 30.2% | 58.6% | 76.2% |
| 0b without spread/total | 22.29 | 16.86 | -4.50 | 30% | 57% | |
| 1 decomposed (team targets x share x yards/target) | 22.81 | 18.61 | +1.32 | 28% | 54% | |
| 2 oracle team targets (plug-in / learned) | 21.72 / 21.51 | | | 30 / 32% | 57 / 59% | |
| 3 oracle routes (plug-in / learned) | 21.04 / 20.78 | | | 32 / 35% | 60 / 61% | |
| 4 oracle targets (plug-in / learned) | 16.49 / 16.33 | 12.19 | | 44% | 70% | |
| 5 oracle receptions (plug-in / learned) | 13.08 / 13.18 | 9.16 | | 53% | 78% | |
| 6 oracle receptions + completed air yards (YAC predicted) | 8.72 | 5.96 | | 69% | 90% | |

2026 wk1-3 (n = 390): incumbent 22.34, oracle targets 15.40, oracle receptions 12.28,
air known 8.06. Routes (participation data) exist only through 2025.

**Sequential budget (from 22.81):** team pass volume 1.1 (5%) -> route participation
0.7 (3%) -> targets per route 4.6 (20%) -> catch result 3.4 (15%) -> air yards on
catches 4.4 (19%) -> YAC 8.7 (38%). Opportunity total 28%, per-target outcome 72%.

Component errors: team targets MAE 5.83; target share MAE 0.062; player targets MAE 2.11.

### Receptions, 2025, n = 2,294

| Rung | MAE | within 1 |
|---|---|---|
| incumbent recipe (direct) | 1.645 | 37% |
| decomposed | 1.644 | 37% |
| oracle team targets | 1.514 | 40% |
| oracle routes | 1.434 | 44% |
| oracle targets | 0.832 | 69% |

Budget: opportunity (team + routes + targets/route) 0.81 of 1.64 (49%), catch result 0.83 (51%).

### Passing yards (QB), 2025, n = 539 (no production incumbent exists; direct model with v2 recipe used)

| Rung | MAE | medAE | ±20 |
|---|---|---|---|
| direct (v2 recipe) | 63.78 | 54.80 | 22% |
| decomposed (plays x pass rate x QB share x comp% x yds/comp) | 68.57 | 53.80 | 17% |
| oracle team plays | 62.99 | | 20% |
| oracle team dropbacks | 58.58 | | 21% |
| oracle QB attempts (plug-in / learned) | 40.13 / 40.52 | | 31% |
| oracle completions | 32.88 / 33.73 | | 40% |
| oracle completions + completed air yards | 24.40 | 19.98 | 50% |

Budget (from 68.6): plays 5.6 (8%), pass rate 4.4 (6%), QB attempts given dropbacks
18.5 (27%, inflated by QB injuries/benchings mid-game and backups in the population),
completions 7.3 (11%), air yards 8.5 (12%), YAC 24.4 (36%). Team plays MAE 6.8;
pass-rate MAE 0.080; QB attempts MAE 8.1.

### Anytime TD (RB/WR/TE), binary. 2025 n = 2,631, base rate 29.2%

| Rung | logloss | Brier | AUC (ranking, not accuracy) | ECE |
|---|---|---|---|---|
| base rate | 0.6043 | 0.2069 | 0.49 | 0.014 |
| incumbent-style (history + team + usage) | 0.5791 | 0.1965 | 0.640 | 0.017 |
| + oracle team red-zone plays | 0.5749 | 0.1949 | 0.652 | 0.012 |
| + oracle player touches | 0.5452 | 0.1838 | 0.713 | 0.019 |
| + oracle team offensive TDs | 0.5316 | 0.1796 | 0.731 | 0.022 |
| + oracle player red-zone / goal-line / end-zone opportunities | 0.4665 | 0.1535 | 0.815 | 0.023 |
| + oracle team TDs and player RZ opportunities | 0.4310 | 0.1395 | 0.850 | 0.029 |

2026 wk1-3 (n = 427) same ordering (incumbent 0.5447 -> player RZ opps 0.4220).
Knowing team red-zone trips adds almost nothing once player usage is known;
**who gets the red-zone and goal-line touches** is the dominant TD uncertainty.

### Interval / distribution check (incumbent, 2025)

| | CRPS (approx) | 80% coverage | 80% width | 50% coverage | 50% width |
|---|---|---|---|---|---|
| rushing | 19.04 | 81.3% | 83.8 yds | 50.5% | 42.8 yds |
| receiving | 16.54 | 78.6% | 68.3 yds | 50.3% | 37.9 yds |

Calibrated, but wide: the honest 80% band for an RB is ~84 yards.

## Dominant source of error, by family

| Family | Dominant source | Share of decomposed MAE |
|---|---|---|
| Rushing yards | per-carry efficiency (half of it the 20+ yd tail) | 66% (player carry share 22%, team volume 12%) |
| Receiving yards | per-target outcome: catch + air + YAC | 72% (targets per route 20%, team volume 5%, routes 3%) |
| Receptions | split: targets vs catch result | 49% / 51% |
| Passing yards | per-attempt outcome (completion, air, YAC) | ~59%; QB attempts 27% |
| Anytime TD | player red-zone / goal-line opportunity allocation | largest single logloss drop (0.579 -> 0.467) |

The **forecastable** part is opportunity: carries, targets per route, QB
attempts, red-zone touches. Today we miss player carries by 4.15/game and targets
by 2.11/game. Knowing them exactly would take rushing MAE from 25.1 to 16.3-16.7
(within-20: 53% -> 71-73%) and receiving from 22.1 to 16.3-16.5 (59% -> 70%).
That is the practical ceiling for any pre-game model; everything past it is
per-play outcome noise that no pre-game input removes.

## Regime segmentation (incumbent, 2025)

Rushing (MAE / bias):
- game script (post-game, diagnostic only): team won by 5+ -> 30.1 / **-15.2**; lost by 4+ -> 19.7 / +3.3
- draft round 1-2 RBs 29.7 / **-10.7**; round 3+ 21.9 / -4.8; undrafted 24.0 / -4.8
- snap share top third 29.4 / -8.6; bottom third 21.0 / -4.8
- new head coach this season 27.1 / **-11.3**; same HC 24.6 / -6.0
- usage tier (carries last 3) high 27.9, mid 25.2, low 21.9
- different starting QB 25.5 vs same 25.0 (no effect); rookies 25.6 / -9.4
- injury-report Questionable (and played) 23.0 / +0.8 (n = 36)

Receiving:
- WR 25.1 / -5.3; TE 19.7 / -4.0; RB 15.2 / -2.0
- targets tier high 26.5, mid 21.3, low 17.7
- draft round 1-2 24.4 / -5.5; round 3+ 20.1; undrafted 19.6
- snap share top third 26.2 / -6.9; bottom 18.8 / -1.1
- role trend: shrinking role (last 3 below last 8) 24.2 vs stable 20.5
- Questionable and played: 26.2 vs not listed 21.9 (n = 114)

**Repeated player residuals:** none worth modelling. Odd-vs-even-week
correlation of a player's mean residual: rushing -0.08 (57 players), receiving
0.14 (162). Cross-fitted per-player bias correction makes MAE worse (rushing
26.0 -> 28.8, receiving 23.2 -> 25.2). Errors are game-level, not "the model
always misses on player X".

**Systematic negative bias:** the incumbent under-projects on average (-7.1 rush,
-4.4 rec) because it predicts the **median** of a right-skewed outcome
(`reg:absoluteerror`). That is MAE-optimal but reads as "low on featured players":
the bias is largest for round 1-2 RBs (-10.7), high-snap players (-8.6) and
winning game scripts (-15.2). The mean-based decomposed forecast has bias ~0 at
nearly the same MAE. The objective, not missing information, explains most of it.

## Availability layer (not in the conditional numbers above)

P(played | final injury-report status), QB/RB/WR/TE with a real role (>= 30% snaps
in one of the prior 3 games), 2023-2025:

| Status | n | P(played) | snap share change if played |
|---|---|---|---|
| listed, no game status | 1,998 | 96.5% | -1.1 pts |
| Questionable | 728 | **69.4%** | -3.9 pts |
| Doubtful | 95 | 1.1% | -16.3 pts |
| Out | 567 | 0.0% | |

A true pre-game forecast must mix "plays" and "does not play" for Questionable
players (~31% do not play). The current board treats every non-Out/Doubtful
player as fully active.

## Architecture map (what exists)

| Layer | Existing code | Status |
|---|---|---|
| Data: player-week stats, schedules, lines | `nfl_player_games_foundation_a.py` (SQLite), `nfl_yardage_v3.Data` (CSV) | usable |
| Data: per-play pools | `nfl_pbp_foundation_a.py` (carry/target pools only) | usable, narrow |
| Defense profile, scheme, red zone, xTD | `nfl_context_v4.PlayData` (pbp + participation + FTN) | usable; participation ends 2025 |
| Availability | ESPN roster status at serving (`nfl_serving_builder_a`), realized participation in research training | skewed (see leakage) |
| Team volume | `nfl_rushing_volume_premise_check_a.py` (R^2 0.0665 with spread+total), `nfl_rush_team_volume_model_a.py` (never run; no report in repo) | stub |
| Share / role | last-3 share, vacated usage (v3), availability-aware usage (v5) | features only, no role-state model |
| Efficiency | shrunk rates inside XGBoost features | implicit |
| Distribution | `nfl_sim.py` two-stage bootstrap of the player's own games (count from one past game, yards resampled) | no game context, no teammate coherence; beat naive by 0.5 yd (baseline_audit) |
| Point models | v2 (served), v3/v4/v5 (research, failed ship rules) | direct final-stat regressions |
| Binary props | receptions/rushing/receiving classifiers, TD, sacks, tackles, INTs gates | classifier-at-a-line designs |
| Early season | prior-season / preseason builders | separate path, not audited here |

Missing entirely: team play-volume and pass-rate (PROE) model, route
participation / targets-per-route model, role-state or changepoint model,
depth-chart ingestion, count/distributional component models, joint team-level
simulation with shares that sum coherently, QB passing outcomes, P(active) layer.

## Leakage and evaluation flaws found

1. **Shipped v2 artifact is in-sample on 2025.** Trained on 2023-2025. On 2025 it
   scores MAE 21.46 rush / 20.92 rec vs the honest re-fit 25.11 / 22.14: 3.65 and 1.22 yards
   of in-sample flattery. Any comparison against that artifact on 2025 (v3 gate's `v2_mae`,
   first v5 run) was invalid.
2. **Served ranges are in-sample.** v2's `actual_over_projection_quantiles` (the p10/p90 on the
   board and `safe_line_90`) come from residuals on the 2025-2026 evaluation games of the
   pre-refit model, then are applied to the refit model.
3. **Training-serving availability skew in v3/v4/v5.** Research training builds
   "teammate out", vacated usage, returning-starter and defenders-out from **realized same-game
   participation**. Pre-game, only 30.2% of regular-player absences (528 of 1,750 in 2024-2025)
   were announced Out/Doubtful on the final report; the rest were IR/PUP/suspension/benching/late.
   Part of that is knowable from rosters, but the training features see more than serving can.
4. **Conditional vs pre-game conflated.** Every gate scores only players who played. Nothing
   scores the 31% of Questionable players who sit.
5. **Sportsbook inputs in the incumbent.** v2 uses spread and implied team total (closing lines).
   Under the new standard they are out. Cost of removing them: +0.04 MAE rushing, +0.15 receiving.
6. **Observed weather.** v3 uses game-time weather from `games.csv` (recorded, not forecast).
   Minor, but it is not pre-game information.
7. **Repeated use of 2025 and 2026 wk1-3** (see protocol). Reported p-values on 2025 after many
   looks overstate confidence.
8. **Objective mismatch.** The gates reward MAE (median), while the board is read as "expected
   yards" and complaints were about under-projection. Median objective => -4 to -7 yd average bias.
9. **TD calibration fit on the evaluation season.** `TdProjector` Platt-calibrates on 2025; any 2025
   calibration figure for it is in-sample (serving 2026 is fine).
10. **Week-3 FanDuel comparison is n = 98, one week.** Not evidence either way on its own.
11. Classifier gates report AUC; AUC is ranking quality, not accuracy.

## Recommended Phase 1 architecture

Each layer is its own model, scored on its own target, then combined by a game-level simulation.

| Layer | Target | Model family | Why |
|---|---|---|---|
| A. Availability | P(active), P(limited) | logistic on report status, practice trend, position, prior snaps | Q players play 69%; binary with small n per cell |
| B. Team environment | team plays; dropback rate | negative binomial / Gaussian for plays; beta-logit for pass rate; hierarchical team effects, shrunk to prior season | plays MAE 6.8, pass-rate MAE 0.080 today |
| C. Role state | RB carry share, WR/TE route rate, QB attempt share | beta / Dirichlet-multinomial across the team's players (shares sum to 1) with a changepoint/HMM role state (backup / committee / lead / bell-cow), priors from draft capital and depth chart, overridden by current deployment | share error is the biggest forecastable piece of rushing (22% of MAE) |
| D. Opportunity | targets per route; red-zone / goal-line share | hierarchical binomial for TPRR (player + team + coverage matchup); multinomial for RZ touches | TPRR is 20% of receiving MAE; RZ allocation drives TD logloss 0.579 -> 0.467 |
| E. Efficiency | yards per carry; catch prob; air yards; YAC | heavy-tailed (mixture: base gamma/lognormal + explosive-play Bernoulli x tail), strongly shrunk; defense scheme (box rate, coverage, pressure) enters here | 66-72% of error lives here but is mostly noise; heavy shrinkage, not more features |
| F. TDs | player TDs | Poisson / binomial on RZ + GL opportunities x conversion | TDs are events, not a yardage multiple |
| G. Joint simulation | final stat distributions | simulate the game: plays -> pass/rush -> multinomial allocation to active players -> per-play outcomes; preserves teammate negative correlation and QB/receiver dependence | replaces `nfl_sim.py`'s resampling of old games |

Point forecast shown = distribution median (or mean, stated explicitly), plus 50%
and 80% intervals, scored by MAE, within-k, CRPS and coverage.

## Prioritised build sequence

1. **Availability layer (A)** and switch every research feature from realized to pre-game
   availability (fix leakage #3). Small, removes a known skew.
2. **Team environment (B)** without sportsbook inputs. Gate: beat team rolling averages on
   plays and pass rate (MAE, CRPS).
3. **Role/share model (C)**: highest forecastable payoff. Gate: carry-share MAE < 0.126 and
   target-share MAE < 0.062 on 2025, and player carries MAE < 4.15, targets MAE < 2.11.
4. **Opportunity (D)** incl. red-zone allocation; feeds TD model (F).
5. **Efficiency (E)** with scheme matchups at this stage only; gate on per-carry / per-target CRPS.
6. **Simulation (G)**, then compare end-to-end against the incumbent on 2026 weeks 5-18,
   frozen in advance.

Expected ceiling if C and D were perfect: rushing MAE ~16.5 / 72% within 20; receiving
~16.4 / 70%. Realistic target for Phase 1: close a third of that gap
(rushing ~22.5, receiving ~20.2).

## Frozen / failed ideas: do not repeat

- Adding more columns to a direct final-stat XGBoost: v3 (usage/injuries/weather), v4 (full
  defense + scheme + red zone: 36-39% of splits, no accuracy gain), v5 (availability-aware
  workload, two-stage volume, injury-report status). All landed at rushing 24.7-25.2 / receiving
  22.0-22.4 on the same rows.
- Ratio targets (actual / blend): negligible.
- Post-hoc size map (tree number -> median actual): hurt (rush 24.75 -> 25.30).
- Squared-error / pseudo-Huber objectives on the direct model: no MAE gain.
- Per-player residual correction: makes MAE worse (no persistent player bias).
- Sportsbook spread/total as inputs: worth 0.04-0.15 MAE; excluded going forward.
- Resampling a player's old games (`nfl_sim.py`) as the distribution: ignores this game.

## What could not be established from current data

- **True routes run.** Participation gives "on the field for a dropback", a route proxy (RBs
  who stay in to block count as routes), and only through 2025.
- **Depth charts, snap-by-personnel packages, third-down/two-minute roles**: no source ingested.
- **Offensive coordinator / play-caller changes**: only head coach is in `games.csv`.
- **Contracts, transactions, practice-squad elevations**: not ingested.
- **Coverage/man-zone for 2026**: participation not published for 2026; 2025 tendencies used.
- **Tackles/sacks/INTs for defenders** are in the stats file, but no oracle ladder was built here
  (no incumbent point forecast exists; existing defensive work is classifiers at a line).
- **Returning-from-absence regime**: the replay only sees games with a stat row, so absences
  mostly don't appear; this segment is empty (n = 3) and needs roster-level data.

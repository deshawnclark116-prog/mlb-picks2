# Phase1O-R: role change / usage inflection engine

**Verdict: REJECTED_ROLE_CHANGE_REPLACEMENT** (receiving: REJECTED_ROLE_CHANGE_REPLACEMENT; rushing: REJECTED_ROLE_CHANGE_REPLACEMENT). Frozen Phase1D allocators remain the role comparator and are not reopened. Workload only: target share / targets and carry share / carries. 2025 was never opened (the development gate failed); 2026 W1-4 is `NOT_RUN_PINNED_BYTES_UNAVAILABLE`.

## Question

Can pregame role-change information (recency acceleration, concentration, teammate vacancy, rookie development, change-point detection, teammate competition) improve on the frozen Phase1D allocation, measured on actual player opportunity, with shares that always sum to one? The comparators were Phase1D (C1), the transparent prior-3 share (C2) and a frozen competent-human role baseline (C3). The candidate form was `max(0, Phase1D share + ridge delta)` renormalised over the active eligible roster; ridge lambda in {10, 100, 1000} chosen on 2024 W9-18.

## Source audit first

Accepted pinned sources only: nflverse weekly stats, weekly rosters (status, years_exp, draft_number) and the nflverse snap-count release (digest-pinned 2023-2025). The snap file carries only a PFR id, joined to GSIS through the pinned roster `pfr_id` (exact join, no names). Skill-position snap rows with offense snaps mapped to GSIS: **93.8%**, below the preregistered 95% minimum, so **family B (snap change) is `BLOCKED_DATA`** and was not tested as a signal family. (The human baseline still uses snap trend where the exact id exists, which only strengthens that comparator.) Depth charts, injury reports, participation, routes, target-game snaps and every 2026 file are not used as features.

## Receiving head: 2024 W9-18 selection (1303 Phase1D fixed meaningful rows)

Oracle player-opportunity MAE (predicted share x actual team opportunity): Phase1D **1.942**, prior-3 share 2.088, competent human 2.117. Share MAE: Phase1D 0.0616, human 0.0667.

| Family | lambda | Oracle MAE | Gain vs Phase1D | Required | Bootstrap upper-95 (cand - Phase1D) | Gain vs human | Survives |
|---|---:|---:|---:|---:|---:|---:|---|
| A_recency_acceleration | 1000 | 1.949 | -0.007 | 0.100 | 0.010 | 0.168 | False |
| C_usage_concentration | 1000 | 1.943 | -0.001 | 0.100 | 0.004 | 0.174 | False |
| D_teammate_vacancy | 10 | 1.941 | 0.001 | 0.100 | 0.007 | 0.176 | False |
| E_rookie_development | 1000 | 1.934 | 0.008 | 0.100 | -0.001 | 0.183 | False |
| F_change_point | 1000 | 1.946 | -0.004 | 0.100 | 0.006 | 0.171 | False |
| G_role_competition | 10 | 1.939 | 0.003 | 0.100 | 0.000 | 0.178 | False |
| B_snap_change | - | - | - | - | - | - | BLOCKED_DATA |

No family met the practical threshold against Phase1D (the best, `E_rookie_development`, is within noise of Phase1D), so no survivor combination was formed. All six families beat the human baseline comfortably, which only shows that Phase1D already contains what a competent human reads from recent usage.

Role-change detection (tau 0.05; predicted flag at tau/2, reference = prior-3 share): Phase1D precision 0.451, recall 0.380, false-promotion rate 0.549; candidate 0.450 / 0.380 / 0.550; human 0.354 / 0.215 / 0.646. Actual role changes in the period: 707. The candidate does not detect role changes any better than Phase1D, and roughly half of the flagged promotions are false for both.

Postgame-only forensics (never used for fitting or selection): role error (actual team volume x predicted share) MAE 1.934; team-volume error (frozen Phase1B team projection x actual share) MAE 0.923. Errors by actual target-game snap bucket: 25_60 n=320 MAE 1.640; ge60 n=903 MAE 2.012; lt25 n=50 MAE 2.731; missing n=30 MAE 1.423.

## Rushing head: 2024 W9-18 selection (533 Phase1D fixed meaningful rows)

Oracle player-opportunity MAE (predicted share x actual team opportunity): Phase1D **2.986**, prior-3 share 3.243, competent human 3.197. Share MAE: Phase1D 0.1145, human 0.1214.

| Family | lambda | Oracle MAE | Gain vs Phase1D | Required | Bootstrap upper-95 (cand - Phase1D) | Gain vs human | Survives |
|---|---:|---:|---:|---:|---:|---:|---|
| A_recency_acceleration | 100 | 2.973 | 0.013 | 0.200 | 0.036 | 0.224 | False |
| C_usage_concentration | 1000 | 3.022 | -0.036 | 0.200 | 0.067 | 0.176 | False |
| D_teammate_vacancy | 1000 | 2.978 | 0.008 | 0.200 | 0.009 | 0.219 | False |
| E_rookie_development | 1000 | 3.005 | -0.019 | 0.200 | 0.045 | 0.193 | False |
| F_change_point | 1000 | 3.007 | -0.021 | 0.200 | 0.033 | 0.190 | False |
| G_role_competition | 1000 | 2.994 | -0.008 | 0.200 | 0.027 | 0.203 | False |
| B_snap_change | - | - | - | - | - | - | BLOCKED_DATA |

No family met the practical threshold against Phase1D (the best, `A_recency_acceleration`, is within noise of Phase1D), so no survivor combination was formed. All six families beat the human baseline comfortably, which only shows that Phase1D already contains what a competent human reads from recent usage.

Role-change detection (tau 0.1; predicted flag at tau/2, reference = prior-3 share): Phase1D precision 0.416, recall 0.407, false-promotion rate 0.584; candidate 0.426 / 0.433 / 0.574; human 0.394 / 0.340 / 0.606. Actual role changes in the period: 268. The candidate does not detect role changes any better than Phase1D, and roughly half of the flagged promotions are false for both.

Postgame-only forensics (never used for fitting or selection): role error (actual team volume x predicted share) MAE 2.973; team-volume error (frozen Phase1B team projection x actual share) MAE 2.349. Errors by actual target-game snap bucket: 25_60 n=193 MAE 3.102; ge60 n=246 MAE 2.855; lt25 n=90 MAE 2.996; missing n=4 MAE 3.503.

## Decision

Neither head has a surviving family, so the **development gate failed** and, by the preregistered chronology, **2025 stayed unopened** and the Phase1E downstream check was `NOT_RUN_NO_HEAD_QUALIFIED`. No hyperparameter, family, threshold or feature was changed after seeing results. Coherence held everywhere (maximum |share sum - 1| = 0.0e+00).

## Largest remaining role-allocation failure source

The role error that remains is realized availability and game-script variation that no pregame usage-history signal removes: the oracle role error is about 1.9 targets and 3.0 carries per player-game, and a better description of recent usage does not reduce it. Information that could (verified starter and health state, Phase1M) is not accessible with the accepted sources.

## Honest caveats

- The audit result for family B follows the preregistered rule; improving the id map after seeing it would have been rescue tuning.
- Selection and lambda choice used the same 2024 W9-18 rows, which favors the candidates; they still failed.
- Fit rows were 2024 W1-8 only (one half season), and the selected lambda sat at the grid edge (1000) for most families, which is a strong sign of no learnable signal beyond Phase1D rather than an under-tuned grid.
- Historical rosters are final-week snapshots, an inherited Phase1D contract.

## Reproduction

```bash
python nfl_v2_phase1o_role_sources.py --fetch --data-dir /tmp/p1o-data --out /tmp/p1o-audit.json
python nfl_v2_phase1o_role_change.py --stage develop --data-dir /tmp/p1o-data --out-dir /tmp/p1o-out
python nfl_v2_phase1o_role_change.py --stage validate --data-dir /tmp/p1o-data --out-dir /tmp/p1o-out   # refuses: gate failed
python nfl_v2_phase1o_role_change.py --stage burned2026 --out-dir /tmp/p1o-out
python -m pytest -q tests/test_nfl_v2_*.py
```

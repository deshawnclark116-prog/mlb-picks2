# NHL skater GOALS head (phase G1) — result: `GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED`

Protocol `phase_goals_g1_protocol.json` (commit `4e33445`) was committed before any goals-model performance. SOG B2 (`B2_HISTORICAL_CHAMPION_RETAINED`) and the frozen attempt extension were not reopened.

**Target / population.** Skater goal count per candidate-game, exact Phase 1A candidate universe (non-participants = 0), labels from the frozen S0 scoring table joined 1:1 (397,778 rows; shots == sog everywhere). Goalie goals and shootout goals are never skater goals. 30 historical rows have goals = sog + 1: labels are untouched; `effective_sog = max(sog, goals)` is used only as a conversion denominator (27 of them are in the 2018-2025 candidate table).

**Exposure.** 2018-2023 development. 2024 / 2025 were `DESCRIPTIVELY_EXPOSED_NOT_MODEL_SCORED` (and the B2 SOG input had been confirmed on 2024 / scored on 2025): late-period confirmation only, never pristine.

**Architectures.** G0 position-conversion compound; G1 hierarchical Beta (ML, training-only, per position) player-conversion compound; G2 direct Poisson/NB2 comparator (37 features). G0/G1 integrate exactly over the full frozen-B2 NB2 SOG PMF (binomial / beta-binomial, goals <= sampled SOG, tail mass ~1e-10).

| mean over D1-D4 | CRPS | NLL | PIT-KS |
|---|---|---|---|
| G0 | 0.106837 | 0.355478 | 0.0039 |
| **G1** | **0.106439** | 0.354187 | 0.0035 |
| G2 | 0.106530 | 0.354228 | 0.0059 |

G1 beat G0 by 0.37% CRPS (below the 0.5% practical bar) but with a pooled blocked-bootstrap upper-95 bound < 0 and all guards passing; G2 did not beat G1. Rejected architectures are frozen.

**2024** (refit 2018-2023, scored once): passed C1-C4 vs G2 (CRPS 0.107903 vs 0.107927).
**2025** (refit 2018-2024, scored once): C1 CRPS, C2 NLL, C4 slices, PIT-KS/coverage/ECE/zero-goal all passed; **C3 mean-bias failed** (|bias| 6.0% > 5%: predicted mean 0.1246 vs observed 0.1326; implied conversion 10.2% vs 11.1%). Status therefore `GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED`. No rescue tuning.

**Reading.** The model is well calibrated in distribution shape but its mean is persistently low (negative bias in every fold, -1% to -6%): conversion estimated on past years lags a rising league rate. A drift-aware conversion construct would be a NEW protocol, not a re-cut of this one. 2026 forward evidence is the cleanest confirmation.

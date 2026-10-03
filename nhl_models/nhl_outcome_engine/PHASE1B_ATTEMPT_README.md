# NHL Phase 1B-A — bounded shot-attempt information-value screen (FINAL)

**Final SOG status: `B2_HISTORICAL_CHAMPION_RETAINED`.** The attempt extension is `FROZEN_REJECTED` (`ATTEMPT_SIGNAL_NOT_JUSTIFIED`): no rescue tuning, no variants, no full crawl.

## Path
1. Protocol committed first (`ed877d2`); acquisition of the registered 7,923 source games; the PBP quality gate FAILED (113 / 285,146 PBP-vs-official SOG differences, all +1) so no model was scored.
2. Source adjudication (protocol `1ad2a97`): all 113 official SOG values equal the CURRENT NHL boxscore; official Stats REST individual attempts (missedShots / shotAttemptsBlocked) exist only for 2022+; the official HTML play-by-play equals the API event record on SOG (both +1 vs the stat line) with one genuine event-level conflict. **The literal protocol-1 rule returned ATTEMPT_SOURCE_NOT_TRUSTWORTHY** (kept verbatim in `phase1b_attempt_source_adjudication.json`).
3. That rule asked an event record to "confirm" an official stat line it is known to differ from (a design flaw), so `PRE_SCORING_AMENDMENT_1` (committed `ed7a08e`, after seeing adjudication data, before any scoring) replaced it with player-game-level certification over ALL 7,923 games (API PBP == official HTML, and == official realtime where it exists; any conflict removes the player-game; official SOG is the only SOG source). Result: 285,069 / 285,146 certified, 77 (0.027%) removed, zero conflicts in 2017-2021 -> `HYBRID_ATTEMPT_SOURCE_JUSTIFIED` (reviewer note: this status rests on a pre-scoring amendment). v2 data contract committed (`5852b56`) before scoring.
4. The ORIGINAL preregistered experiment ran unchanged on the v2 rows (800 validation games, folds D1-D4, 2018-2023 targets only; 2024/2025 never read).

## Result (A1 = B2 + four attempt features vs A0 = frozen B2 refit)
| fold | A0 CRPS | A1 CRPS | delta | NLL delta |
|---|---|---|---|---|
| D1 | 0.58270 | 0.58123 | -0.001469 | -0.002194 |
| D2 | 0.62568 | 0.62330 | -0.002380 | -0.002764 |
| D3 | 0.63105 | 0.62787 | -0.003184 | -0.003220 |
| D4 | 0.60743 | 0.60617 | -0.001258 | -0.001616 |

Mean relative macro-game CRPS improvement **0.339%**; relative NLL change -0.181%; pooled bootstrap (seed 20261002) one-sided 95% upper bound -0.001205.

| gate | pass | value |
|---|---|---|
| S1 mean relative CRPS gain >= 0.5% | **False** | 0.339% |
| S2 one-sided 95% bootstrap upper bound < 0 | True | -0.001205 |
| S3 improves in >= 3 of 4 folds | True | 4/4 |
| S4 NLL not worse by > 0.5% | True | -0.181% |
| S5 F/D slices | True | |
| S6 early/established slices | True | |
| S7 calibration (PIT KS, 80/90 PIT coverage) | True | |

**Reading:** the attempt features carry a real but small signal (every fold improves, upper bound < 0) that is below the preregistered 0.5% materiality bar; S1 alone fails. The registered rule is not rescued. Reopen conditions: a new preregistered protocol (e.g. a materially different attempt construct) - never a re-cut of this one.

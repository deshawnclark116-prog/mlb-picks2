# NHL SOG Phase 1A — historical player SOG distribution engine

**FINAL STATUS: HISTORICAL_CHAMPION** (selected architecture **B2**, promoted in 2024 over B0). Historical only; forward deployment remains blocked by the Phase 0B live-timing blocker. No production NHL change; no sportsbook data; no model beyond B0-B3; no Monte Carlo.

Preregistration: `phase1a_protocol.json` (committed before any 2024/2025 performance). 2024 selection committed before 2025 was scored. Phase 0A's tiny 2025 feasibility sample was data-integrity exposure only.

## Data
11052 games, 397778 skater-game rows (2017-2025, M2 acquisition: 324 weekly windows, 972 requests, 439 MB, 105.4 s, 0 guard failures). Target rows 2018-2025: 9781 games, 453076 candidate rows, 345318 played, 6704 unobservable players; candidate player coverage 0.98096, SOG coverage 0.98586. All data-quality gates PASS (`phase1a_data_quality.json`).

| season | candidate rows | played | unobservable | player coverage | SOG coverage |
|---|---|---|---|---|---|
| 2018 | 58324 | 44939 | 816 | 0.98217 | 0.98669 |
| 2019 | 49716 | 38248 | 702 | 0.98198 | 0.98739 |
| 2020 | 41727 | 30477 | 749 | 0.97601 | 0.98281 |
| 2021 | 63415 | 46211 | 997 | 0.97888 | 0.98404 |
| 2022 | 60102 | 46335 | 873 | 0.98151 | 0.98609 |
| 2023 | 59869 | 46377 | 844 | 0.98213 | 0.98677 |
| 2024 | 59644 | 46374 | 850 | 0.982 | 0.98624 |
| 2025 | 60279 | 46357 | 873 | 0.98152 | 0.98609 |

## Development folds (CRPS / NLL, macro-game; targets 2018-2023)
| fold | train | validate | B0 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| D1 | 2018-2019 | 2020 | 0.64468 / 1.44968 | 0.58686 / 1.32500 | 0.58531 / 1.31578 | 0.58366 / 1.29247 |
| D2 | 2018-2020 | 2021 | 0.67512 / 1.48680 | 0.61805 / 1.37007 | 0.61581 / 1.35715 | 0.61424 / 1.32682 |
| D3 | 2018-2021 | 2022 | 0.68261 / 1.51892 | 0.62406 / 1.38380 | 0.62266 / 1.37503 | 0.62083 / 1.35403 |
| D4 | 2018-2022 | 2023 | 0.66535 / 1.49110 | 0.61168 / 1.36618 | 0.61032 / 1.35802 | 0.60845 / 1.33671 |

Selected (registered tie rule): B1 alpha=0.001, B3 C=0.01, B3 Poisson alpha=0.1.

## 2024 confirmation (fit 2018-2023) — gates G1..G7 (columns in alphabetical gate order)
| comparison | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|
| B1 vs B0 | PASS | PASS | PASS | PASS | FAIL | PASS | PASS |
| B2 vs B0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| B3 vs B2 | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |

Frozen 2024-selected architecture: **B2**.

## 2025 holdout (fit 2018-2024, scored once) — gates H1..H6 vs B0
| comparison | H1 | H2 | H3 | H4 | H5 | H6 slices | H6 season slices |
|---|---|---|---|---|---|---|---|
| B2 vs B0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

Promotion-reference guardrail: {"promotion_reference": "B0", "selected_crps": 0.5797770707159464, "reference_crps": 0.6308409297465154, "pass": true}

| model | CRPS | NLL |
|---|---|---|
| B0 | 0.63084 | 1.43524 |
| B1 | 0.58167 | 1.31741 |
| B2 | 0.57978 | 1.30815 |
| B3 | 0.57894 | 1.29139 |

## Remaining blockers
Forward snapshot timing (Phase 0B) is still BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE; no point-in-time roster source for players new to a team; this result is HISTORICAL_CHAMPION only. Phase 1B was not started.

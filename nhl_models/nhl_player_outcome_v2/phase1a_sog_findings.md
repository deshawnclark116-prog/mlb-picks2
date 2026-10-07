# NHL V2 Phase1A-SOG - count-distribution certification and forward lock

Evidence grade of every historical number: **BURNED_REPRODUCTION_ONLY**. 2024 and 2025 are burned; V1 is a burned prior-research corpus. The first clean confirmation is forward 2026 data created after the engine lock. Nothing here validates or promotes anything.

## V1 migration

- **BLOCKED_TIMING**: 2 component(s)
- **BURNED_RESULT_REFERENCE_ONLY**: 3 component(s)
- **DO_NOT_PORT**: 4 component(s)
- **PORT_AS_FIXED_ARCHITECTURE**: 6 component(s)
- **PORT_DATA_ENGINEERING_ONLY**: 4 component(s)
- **PORT_TEST_GUARD**: 2 component(s)
- **REJECTED_DO_NOT_REOPEN**: 1 component(s)

Phase1B shot-attempt extension: REJECTED_DO_NOT_REOPEN (mean relative CRPS gain 0.00339 vs 0.5% materiality). B3 is not ported; participation/availability stays a separate uncertified layer.

## Data source

Official NHL source (api-web schedule + stats-REST skater summary/time-on-ice), 324 windows, retrieved 2026-10-02, vendored with sha256 equal to the V1 manifest (True). Source re-check: 8820 / 8820 sampled rows byte-identical across 6 windows; both hosts reachable. No published terms found: research-only, low-rate use. Gates all pass: True. The pinned V1 builder and the V2 builder produce a bit-identical table; V1's recorded table hash is not reproducible and is not used.

| season | candidate player coverage | SOG coverage | full-universe played fraction | meaningful rows | meaningful played fraction |
|---|---|---|---|---|---|
| 2018 | 0.9822 | 0.9867 | 0.771 | 41897 | 0.928 |
| 2019 | 0.9820 | 0.9874 | 0.769 | 35542 | 0.929 |
| 2020 | 0.9760 | 0.9828 | 0.730 | 28192 | 0.913 |
| 2021 | 0.9789 | 0.9840 | 0.729 | 42619 | 0.911 |
| 2022 | 0.9815 | 0.9861 | 0.771 | 43235 | 0.931 |
| 2023 | 0.9821 | 0.9868 | 0.775 | 43304 | 0.929 |
| 2024 | 0.9820 | 0.9862 | 0.778 | 43354 | 0.932 |
| 2025 | 0.9815 | 0.9861 | 0.769 | 43225 | 0.927 |

The full candidate universe is ~77% participants; ~23% of rows are non-participant zeros that make any unconditional score look easier. The MEANINGFUL expected-participant universe (prior information only) is ~93% participants and is the headline population.

## Burned reproduction of the fixed B2 (fit 2018-2024, score 2025, T90)

Status: **B2_REPRODUCED** (n equal: True; dCRPS 8.16e-08, dNLL 6.98e-08, dMAE 1.15e-07). The V2 pipeline reproduces V1 to ~1e-7.

| population | n | MAE(mean) | bias | median AE | MAE(median fcst) | CRPS | NLL | 50/80/90 coverage | P(3+) Brier | Brier skill | ECE | slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 FULL | 60279 | 0.8775 | 0.0272 | 0.713 | 0.8186 | 0.57978 | 1.30815 | 0.804/0.941/0.970 | 0.11701 | 0.1604 | 0.0061 | 0.845 |
| 2025 MEANINGFUL | 43225 | 1.0377 | 0.0290 | 0.888 | 0.9934 | 0.69417 | 1.52950 | 0.770/0.936/0.971 | 0.14666 | 0.1278 | 0.0057 | 0.980 |
| 2025 PLAYED | 46357 | 1.0338 | -0.0719 | 0.875 | 1.0083 | 0.71071 | 1.59134 | 0.754/0.924/0.961 | 0.15059 | 0.1103 | 0.0135 | 0.775 |
| 2024 FULL | 59644 | 0.8884 | 0.0294 | 0.729 | 0.8274 | 0.58868 | 1.32598 | 0.804/0.940/0.971 | 0.12011 | 0.1586 | 0.0083 | 0.857 |
| 2024 MEANINGFUL | 43354 | 1.0462 | 0.0297 | 0.902 | 0.9997 | 0.70129 | 1.54319 | 0.770/0.936/0.971 | 0.14995 | 0.1233 | 0.0085 | 0.957 |
| 2024 PLAYED | 46374 | 1.0396 | -0.0652 | 0.887 | 1.0101 | 0.71453 | 1.59756 | 0.756/0.924/0.962 | 0.15298 | 0.1121 | 0.0136 | 0.790 |

Central 50/80/90% intervals over-cover because the V1 interval definition on discrete counts is conservative; they are reported as defined, not tuned.

## Comparators on identical rows (burned; not validation)

| season | population | model | n | MAE | bias | CRPS | NLL | P(3+) Brier |
|---|---|---|---|---|---|---|---|---|
| 2024 | FULL | V2_B2 | 50522 | 0.9436 | 0.0281 | 0.62125 | 1.39021 | 0.13057 |
| 2024 | FULL | human_frozen | 50522 | 1.1049 | 0.3033 | 0.71343 | 1.53378 | 0.13464 |
| 2024 | FULL | simple_prior10_mean | 50522 | 1.0993 | 0.2087 | 0.72094 | 1.54506 | 0.14045 |
| 2024 | FULL | simple_season_rate_x_prior3_toi | 50522 | 1.0773 | 0.2265 | 0.70164 | 1.51606 | 0.13578 |
| 2024 | MEANINGFUL | V2_B2 | 41097 | 1.0467 | 0.0320 | 0.70171 | 1.54632 | 0.15002 |
| 2024 | MEANINGFUL | human_frozen | 41097 | 1.0870 | 0.1721 | 0.71790 | 1.57127 | 0.15016 |
| 2024 | MEANINGFUL | simple_prior10_mean | 41097 | 1.0953 | 0.0831 | 0.73595 | 1.60021 | 0.15579 |
| 2024 | MEANINGFUL | simple_season_rate_x_prior3_toi | 41097 | 1.0725 | 0.1071 | 0.71591 | 1.56921 | 0.15118 |
| 2025 | FULL | V2_B2 | 50844 | 0.9333 | 0.0268 | 0.61646 | 1.38068 | 0.12747 |
| 2025 | FULL | human_frozen | 50844 | 1.0941 | 0.3129 | 0.70372 | 1.51804 | 0.13086 |
| 2025 | FULL | simple_prior10_mean | 50844 | 1.0846 | 0.2135 | 0.71027 | 1.52804 | 0.13647 |
| 2025 | FULL | simple_season_rate_x_prior3_toi | 50844 | 1.0619 | 0.2175 | 0.69044 | 1.49886 | 0.13161 |
| 2025 | MEANINGFUL | V2_B2 | 40891 | 1.0397 | 0.0307 | 0.69640 | 1.53464 | 0.14734 |
| 2025 | MEANINGFUL | human_frozen | 40891 | 1.0836 | 0.1852 | 0.71489 | 1.56420 | 0.14749 |
| 2025 | MEANINGFUL | simple_prior10_mean | 40891 | 1.0894 | 0.0924 | 0.73250 | 1.59233 | 0.15338 |
| 2025 | MEANINGFUL | simple_season_rate_x_prior3_toi | 40891 | 1.0670 | 0.1020 | 0.71273 | 1.56237 | 0.14849 |

On the MEANINGFUL expected-participant rows B2 improves game-macro CRPS over the Phase0 human baseline by about 2.3-2.6% and over prior-10 by about 4.7-4.9% (blocked-bootstrap intervals exclude zero). On the full universe the gap (~12%) is mostly participation modelling and should not be read as skill at shot volume.

Production SOG>=3 classifier (played rows only; biased toward the classifier because B2 mixes participation): Brier 0.15674 vs V2 B2 0.15877 vs human 0.15716 on 33980 identical rows. The classifier is slightly better on this restricted task; B2 has no advantage there and none is claimed.

## Engine lock and forward

- Locked engine `nhl-v2-sog-b2-1.1` (fixed B2, per horizon T24H/T90/T30, targets 2018-2025; supersedes `nhl-v2-sog-b2-1.0`, which produced 0 real forecasts - reason PRE_FIRST_FORECAST_INFRASTRUCTURE_HARDENING; model files byte-identical); lock eligible_from_cutoff_utc 2026-10-07T19:32:37Z; availability layer recorded but NOT used (OBSERVED_NOT_CERTIFIED); no refit during the forward window.
- Forward ledger (v1.1): decision key = game, horizon, scheduled start (revisions create new keys; MISSED_REVISED_CUTOFF); per-run content-addressed source manifests with retrieval provenance; hard source-completeness gate (SOURCE_INCOMPLETE / SOURCE_FETCH_FAILED are never turned into predictions); append-only hash chain; grader built before the first forecast (RAW and CLEAN ledgers, fixed-ratio large-miss forensics); MISSED_CUTOFF and INVALID_LATE are recorded, never backfilled.
- Scheduling limitation: scheduled workflows fire only from the default branch and this branch must not be merged, so forecasts exist only for windows in which a run actually happened (manual/dispatch or a session). Every other window is recorded MISSED_CUTOFF.
- No betting-market input anywhere; no simulation.


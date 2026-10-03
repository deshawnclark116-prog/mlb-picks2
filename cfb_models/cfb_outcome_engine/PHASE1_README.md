# CFB Outcome Engine v1 — first execution milestone (RESEARCH / SHADOW ONLY)

Production (today's Week 5 board, legacy models, workflows, ledger, grading) is untouched; no Week >= 5 row exists anywhere in this rebuild (`cfb_phase1_common.assert_research_allowed`).

Sequence (all committed in this order): registry + parity map -> source audit + identity audit -> frozen 2018-2025 tables + candidate universe -> contracts -> **preregistered protocol** -> amendment 1 (play-coverage validity, derived from the audit) -> code + tests -> development (D1-D4, validate 2021-2024) -> this result.

| component | baseline | challenger | result (mean over folds) | decision |
|---|---|---|---|---|
| T1 team rush attempts (CRPS) | B0 4.909 | C1 4.742 | -3.4%, bootstrap upper95 < 0, all guards pass | **C1 retained** |
| T2 team pass attempts (CRPS) | B0 4.838 | C1 4.827 | -0.23%, upper95 > 0, PIT-KS +0.027 | **B0 retained**, C1 frozen rejected |
| P1 participation (log loss) | B0 0.4931 | C1 0.4475 | -9.2%, ECE 0.031 -> 0.016, all slices better | **C1 retained** |
| O1 carry-share allocation (CRPS, given realized team carries) | B0 0.7991 | C1 0.6537 | -18.2% overall, but TE +20.4% / WR +8.8% worse (G4) | **B0 retained** (rule), C1 frozen rejected |

All are historical DEVELOPMENT results; 2025 (late confirmation, previously exposed) and 2026 Weeks 1-4 (burned diagnostic) were not touched. No head is certified. Blocked sources: depth charts, injuries, snaps, routes, true targets, recruiting, portal, coaching.

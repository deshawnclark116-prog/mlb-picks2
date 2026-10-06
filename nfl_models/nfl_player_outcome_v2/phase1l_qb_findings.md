# Phase1L-QB — dedicated QB opportunity / passing volume

**BLOCKED_STARTER_STATE_DATA. Stop before fitting, not a predictive rejection.**

The starter audit cannot establish a defensible primary historical expected-starter population for 2024 development. No Phase1L candidate, comparator, oracle or passing-yard performance was computed. The protocol, arithmetic/receipt implementation and source-stop development lock were committed in that order before the results. There is no selected architecture or refit. All A–G families are **BLOCKED_STARTER_STATE_DATA**, not failed football signals.

Receiving stays **FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION**; receiver depth stays **SURVIVED_SIGNAL_NOT_PROMOTED**. Phase1K stays **REJECTED_EFFICIENCY_REPLACEMENT**. No receiving reopening, rushing rescue, new efficiency fit or Week5+ access.

## Source/starter-state audit

This phase opens only the pinned **2023/2024** PBP and weekly stat files. The 2025/2026 state coverage below is explicitly inherited from the previously frozen H/J source audits; it is **not a fresh current-season check**. No later-season data payload, paid API, undocumented ESPN endpoint, current depth chart or gamebook was opened. Source inventory remains unchanged.

| Archive | 2024 | 2025 | Pregame limitation |
|---|---:|---:|---|
| Weekly roster rows | 46,579 | 46,849 | Final weekly status, no original per-status cutoff receipts. Prior identity is usable; current membership/health is unresolved. |
| Injury rows | 6,215 | 6,068 | Original publication/version rows: **0 / 0**. Modification timestamps: 6,215 / 0; modification is not first publication. |
| QB depth rows | 1,539 | 21,429 | 2024 has **zero `dt` timestamps**. 2025 has 221 retained load times overall, not starter announcements or health/revision certification; upstream systematic reuse rights unresolved. |
| Accepted expected-starter/health version archives | **0** | **0** | No accepted source can certify expectation at the historical cutoff. |

Frozen 2023 and 2026 state counts/provenance also remain in the audit. 2026 depth metadata is not an inspected starter-state payload. Original transactions/coach announcement archives and documented abnormal in-game exit labels are absent. Weekly roster deltas do not establish publication time. Official game starters would be **POSTGAME_EVALUATION ONLY** and were not loaded.

Preserved classifications: weekly rosters for current state, injuries, 2024 depth and transactions **BLOCKED_TIMING**; 2025+ depth and coach archive **UNKNOWN_NEEDS_ACCESS**; systematic official report/undocumented ESPN acquisition **BLOCKED_LICENSE**; gamebook starter identity **NOT_USEFUL as a pregame predictor**. Sportradar depth/injuries and SportsDataIO state remain **PAID_BUT_VIABLE**, with no authorized data access or historical original-version acceptance. Documented API existence does not certify old as-of snapshots.

The existing V1 QB-out flag chooses the most-used recent passer and reads an assumed T24 injury report. Its Phase1C simulation allocates dropbacks from rescaled prior attempt propensities and subtracts sacks/scrambles. Neither constitutes a timestamped expected-starter archive. Phase1B/E use prior role share × team attempts; the old human baseline uses prior QB attempts plus opponent/league volume. The frozen generic team/QB allocator is not reopened or relabeled as a certified opportunity model. None of that code was executed by this phase.

### Opportunity-count semantics

Both pinned seasons contain **272 regular-season games**. Audited opportunity plays / opportunity dropbacks / all-passer attempts:

- 2023: **33,903 / 20,760 / 18,315**.
- 2024: **33,406 / 20,187 / 17,811**.

Plays count dropback OR non-kneel rush OR spike; a scramble counts once. Two-point/no-play rows are excluded. nflfastR has `qb_spike=1`, `pass_attempt=1`, but `qb_dropback=0`, so the documented opportunity denominator adds spikes. Official **QB** attempt labels reconcile exactly in both seasons. All eligible dropbacks/attempts have owner identity after assigning scramble ownership from rusher GSIS. Owners without official QB position labels remain separately counted (54 / 41 player-games); they are not silently treated as QBs.

The initial prefit audit comparison incorrectly included those non-QB/unclassified owners in the QB mismatch count. The implementation commit corrected the comparator before the development lock; no fit or performance existed. This is a source semantic correction, not rescue tuning or a new predictive feature.

History requires earlier season/week **and source game date +48h <= target game date −24h at UTC00:00**. This conservative completion/cutoff bound is the existing historical doctrine; corrected releases are not proof of original historical publication vintages. Target-game opportunity/context/starter labels never enter a feature. Forbidden sportsbook columns are removed by an allowlist.

## Raw population/state receipts

The deterministic ledger has **1,202 source-state audit receipts across all 544 2024 team-games**. Candidate identities come from QB-labeled *prior* completed current-team games in the last-eight legal history window; they never come from target-game participants. Departed/inactive prior QBs may remain in this unresolved pool. New target-game passers are not introduced by hindsight. This is an audit pool, **not a meaningful-starter promotion cohort**, and its easy zeros are never scored.

**394 prior-continuity hints** meet the fixed two-game dominance rule (>=80% of dropbacks and >=20 official attempts each). Those hints do not establish current health, membership, replacement or expected starter status. All 1,202 candidate rows are **UNCERTAIN_STARTER**, with confidence **null** and zero certified primary rows. The full six-state contract is preregistered; stable/change/multi states are deliberately not fabricated from unavailable evidence. No change/replacement performance can be claimed.

Raw actual attempts/yards/team plays/dropbacks/share are isolated under `evaluation_truth`. Projection/error fields are null, not zero estimates. Undocumented abnormal exits stay `UNKNOWN_NOT_CENSORED_NO_ACCEPTED_EXIT_LABEL`; no low-attempt or poor-performance censoring. The ledger does not claim to enumerate all unknowable pregame emergency/new-QB candidates.

## Frozen competent-human volume benchmark

The new mandatory hand-calculable benchmark is a **frozen definition, unscored**:

`expected plays × expected dropback rate × expected starter share × attempts/dropback`.

- Plays: last-three current-team mean with two prior-league games of shrinkage.
- Dropback rate: last-three ratio of sums, with 100 prior-league plays of shrinkage.
- Share: accepted single expected starter only; last-three current-team QB/team dropback ratio with 20 prior dropbacks at share 1. If the expected starter is unknown, **abstain**.
- Conversion: last-five QB attempts/dropbacks across teams, with 50 prior-league dropbacks of shrinkage.
- League summaries: most recent 128 strictly legal prior team-games. No optimization, Monte Carlo or odds.

The executable arithmetic is tested on synthetic data only. It was never applied to real candidate rows without accepted starter evidence. The baseline must not inherit the old starter's history when a replacement is expected. Current-team role and across-team QB skill remain distinct.

## Candidate decisions and performance gates

| Family | Football question | Decision | 2024 attempt MAE |
|---|---|---|---|
| A recent QB workload | Recent attempts/dropbacks/share | BLOCKED_STARTER_STATE_DATA | Not run |
| B team volume | Plays × dropback tendency | BLOCKED_STARTER_STATE_DATA | Not run |
| C opponent environment | Opportunity/possession allowance | BLOCKED_STARTER_STATE_DATA | Not run |
| D QB conversion | Sack/scramble/attempt generation | BLOCKED_STARTER_STATE_DATA | Not run |
| E starter/role | Changes, fallback hierarchy, uncertainty | BLOCKED_STARTER_STATE_DATA | Not run |
| F script scenarios | Competitive/lead/trail/possession mixtures | BLOCKED_STARTER_STATE_DATA | Not run |
| G coherent chain | Only independently surviving parents | BLOCKED_STARTER_STATE_DATA | Not run |

The protocol fixes 2024 W1–8 estimation / W9–18 selection. A future accepted experiment must beat the **strongest mandatory comparator** by **max(1.0 attempt, 5%)** MAE on development and selection-free 2025, with upper 95% paired game-block delta <0 and component/catastrophic-miss guards. No 0.1-attempt gains qualify. Bootstrap is 2,000 seeded 14-calendar-day moving blocks of complete games, never rows. No bootstrap was run on this blocked phase.

Mandatory comparators are genuine V1/Phase1C analytic or frozen opportunity on the same cohort, Phase1B, recent-three/five QB attempts, the transparent team chain, the new human chain and the old workload-three opportunity component. An unavailable same-cohort V1 artifact must remain explicitly unavailable; the generic V2 team model is not a substitute, and no simulator is run.

**2024 development, 2025 confirmation, 2026 W1–4 diagnostics and every stable/change/uncertain slice: NOT_RUN_SOURCE_GATE_FAILED.** These have no numerical metrics. Previously read E/F/Week4 results are locked history, not newly untouched Phase1L evidence. No new selection or performance access occurred in 2025/2026.

**Passing yards: NOT_RUN_INDEPENDENT_ATTEMPT_GATE.** No YPA replacement or workload/efficiency cancellation test. **Oracle decomposition: NOT_RUN_NO_NORMAL_WORKLOAD_EVALUATION.** No new claim ranking team-volume, pass-rate, share, conversion or efficiency error is possible. Old F's passing forensic showed workload was a major bottleneck on its different historical population; its 39.95-yard perfect-workload and 50.40-yard perfect-efficiency errors are historical motivation, not Phase1L scores or a same-cohort decomposition.

The largest demonstrated blocker here is **unobserved historical expected QB assignment/health**. Dense prior opportunity history cannot answer who was expected to play now. To reopen, first accept a legally authorized depth/expected-starter plus health version archive covering 2024 and 2025, with original published/observed/revision receipts and GSIS/team/game joins. Independently documented abnormal-exit labels are needed for clean censor slices. A new data-acceptance protocol is required; changing a source-gate flag cannot start scoring this frozen phase.

## Reproduction and scope

```bash
python nfl_v2_phase1l_qb_sources.py --fetch --data-dir /tmp/phase1l-data --out /tmp/phase1l-audit.json
python nfl_v2_phase1l_qb_opportunity.py --data-dir /tmp/phase1l-data --stage develop --lock /tmp/phase1l-lock.json --benchmark /tmp/phase1l-human.json
python nfl_v2_phase1l_qb_opportunity.py --data-dir /tmp/phase1l-data --stage confirm --lock /tmp/phase1l-lock.json --out /tmp/phase1l-results.json --receipts /tmp/phase1l-receipts.jsonl.gz
python -m pytest -q tests/test_nfl_v2_*.py
```

`confirm` reproduces the **blocked stop**, not confirmation scores. It opens only pinned 2023/2024, checks exact audit/lock identity, produces the raw 2024 ledger and refuses automatic reopening. CI compares all generated audit/benchmark/lock/results/gzip bytes, uploads artifacts and updates a compact comment on existing PR #64. Reproduction refuses revised source bytes or a latest-data fallback.

All **1,174 protected pre-existing files** remain byte-identical to starting HEAD. Existing-file exceptions are additive registry metadata and a **test-only** K scope endpoint frozen to K's own published result. A new L guard checks the current tree independently. K scientific code/protocol/results/receipts/snapshot, receiving history, source inventory, production/frontend/scheduler/grading/calibration/frozen forecasts and other sports are unchanged.

Local checks: **23 Phase1L tests; 158 total V2 pytest tests; three standalone audit/baseline/direct-chain scripts; Python compilation; actionlint; diff whitespace check**, all passed. Exact artifact reproduction is checked separately and in CI. No new PR, no merge; PR #64 remains draft.

**Final: BLOCKED_STARTER_STATE_DATA. Preserve all locked incumbents; do not substitute a retrospective starter oracle or promote a conditional continuity-only population.**

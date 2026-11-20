# NFL V7 numerical accuracy diagnostic — 2026 Week 10 (run Nov 20, 2026)

## Verdict: NO UPGRADE — RETROSPECTIVE EXPERIMENT, NOT PROSPECTIVE PROOF

The actual V7 roster-returner formula was executed against independent
nflverse final yardage results for 13 games that began November 15–17,
using **actual saved pregame** roster-status evidence and original
production predictions from the immutable GitHub commit
03d5032be7abfb7900d608401d05b96948a0a7cf
(pre-kickoff roster snapshot captured 2026-11-15T09:15:53Z).
The exact original V6B research forecasts were also reused.

The snapshot was generated from players ON THE ORIGINAL PRODUCTION BOARD,
not a complete league roster. V7 has never been properly frozen before
these games. This is an **exploratory, selection-biased counterfactual**;
it cannot be passed off as an honest prospective model-promotion test.
Later PFR source release publication timestamps are not verified as-of.
The input rosters lack full teammate and QB coverage. Do not use the
results to tune against this week or declare a live betting edge.

## Results: real yards, exact player/game/team/stat identity

| Population | n | V7 median MAE | Original V6B median MAE | Original production median MAE |
|---|---:|---:|---:|---:|
| V7/V6B same-player, same-stat resolved actuals | 176 | 19.879 yards | **19.166 yards** | N/A |
| V7/V6B/production all matching exact player IDs | 38 | 35.413 yards | **33.974 yards** | 34.516 yards |
| Triple-match receiving yards | 23 | 36.578 yards | 36.226 yards | 36.778 yards |
| Triple-match rushing yards | 15 | 33.627 yards | **30.520 yards** | 31.047 yards |

On the 176 same-player V7-versus-V6B rows, V7 was within 10 yards
on 47.7% and V6B on 50.0%. The diagnostic therefore does not support
an improvement in V7 even relative to the already-rejected V6B shadow.
It DOES NOT establish a general future-performance ordering because
the selected roster is incomplete and the source lead times differ.

- Game count with usable archived Sunday/Monday original data: 13.
- Resolved original V7 row count: 193.
- Resolved V7+V6B exact-ID comparisons: 176.
- Resolved V7+V6B+production exact-ID comparisons: 38.
- Missing/ambiguous actual row: 37; NOT silently scored as 0.
- Production missing same exact player ID: 138; NOT name-matched.
- V7 players missing from the V6B original pool: 17.
- One historical game excluded because the archived roster source
  postdated kickoff. No backfilling.
- Test status: successful original historical run with the output
  label EXPLORATORY_BIASED_ROSTER_DIAGNOSTIC_NOT_PROSPECTIVE.

## Next scientific decisions

1. Keep production incumbent; V7 is NOT promoted. Do not optimize on
   these outcomes or reinterpret conditional projections as probabilities
   of active participation.
2. Complete and freeze a **full-team, independently captured roster**
   and QB starter scenario per real upcoming game. Timestamp provider
   arrival and source SHA before kickoff; do not treat any listed player
   as certain to play. Preserve injury statuses.
3. Evaluate genuine 2026 Weeks 11+ V7 original-firstseen receipts
   against matching original published V2 and V6B on actual player
   yards, with source lead-time differences and coverage explicitly
   reported. Four untouched complete weeks and >=250 fully comparable
   observations required before even considering promotion.
4. Next new hypothesis must solve uncertainty of actual participation
   and how returned teammates change the carry/target allocation,
   rather than adding more roster-listed players as if guaranteed
   fully participating. Validate P(active) separately, then the
   whole distribution, not a cosmetic threshold.

Research files:
- nfl_v7_archived_week10_numeric_diagnostic.py
- tests/test_nfl_v7_archived_week10_numeric_diagnostic.py
- .github/workflows/nfl_v7_week10_retrospective.yml
- Historical GitHub Actions output includes full report and
  every resolved V7 row / same-identity triple comparison.

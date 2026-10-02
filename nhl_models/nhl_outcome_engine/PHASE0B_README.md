# NHL outcome engine — Phase 0B (historical contract, acquisition, forward snapshots)

**HISTORICAL FOUNDATION: PASS** (contract, acquisition and probes validated by `tests/test_nhl_phase0b.py`).
**FORWARD SNAPSHOT TIMING: BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE** (`recommended_horizons: []`; no horizon is validated).
**OVERALL:** historical Phase 1A research may proceed. Forward horizon selection is NOT solved.

No model was fitted in Phase 0B. No sportsbook data was used. No production NHL file changed. A full play-by-play crawl is deferred.

## Key rule
Historical player candidacy must be known before the target game. Candidate universe for target game G, team TEAM, cutoff T = players who appeared **for that team** in TEAM's last 10 completed games before T (games != G with start + 210 min <= T), `nhl_outcome_contract.candidate_universe`. Nothing else adds a candidate; a target participant outside it is `unobservable_at_T` (reported, never inserted). Target-game boxscore / roster / position / TOI / PP / shifts / SOG / events / scratches are grading-only. After membership, prior NHL history from all teams may feed player *skill*; role / deployment features use current-team appearances only.

## Evidence (derived from the committed evidence files)
- Historical acquisition: M2 (schedule + stats-REST summary + time-on-ice by date window, no team filter, `total < 10000` and `rows == total` guard) reproduced 151 sample games / 5435 skater rows / 867 traded-player skater-games against the boxscore with no missing rows: **True**; projected 84 requests / ~21 s per season vs ~834 s per-game. The repo's `skater_games` table stays unused (unsorted pagination duplicates and the team filter drop players).
- Candidate coverage on the sample (graded after candidate construction):
  - 2023-11-13..2023-11-19: 1594/1620 skaters observable at T (0.9839), SOG coverage 0.9895, unobservable 26 {'STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW': 6, 'NO_LOADED_HISTORY': 20}
  - 2024-03-04..2024-03-10: 1810/1871 skaters observable at T (0.9674), SOG coverage 0.9746, unobservable 61 {'NO_LOADED_HISTORY': 29, 'STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW': 6, 'PRIOR_NHL_HISTORY_ELSEWHERE': 26}
- Shot attempts: the small probe reconstructs attempts / SOG / missed / blocked and excludes shootout events; attempts are a later challenger.
- Goalie: historical confirmed pregame goalie is NOT reconstructable from the tested endpoints; pregame `starter`-like fields are observed, unvalidated signals.

## Remaining blockers
1. Forward snapshot timing: compliant live evidence (schedule hash, retrieval started/completed, hard stop at puck drop) has not been collected; no horizon is recommended.
2. No historical point-in-time roster / transaction source: players new to a team are unobservable historically.
3. A durable (non-interactive) scheduler is needed for the forward capture.

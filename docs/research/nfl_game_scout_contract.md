# NFL game-by-game scouting evidence — execution contract

## Why this exists

The NFL live yardage board has historically called some receiving/rushing projections HIGH confidence while publishing `projection_probability_source=UNAVAILABLE_V2_CONTEXT_NO_MATCHED_DISTRIBUTION`. A game-specific evidence report must flag such contradictions instead of treating the display label as a statistically calibrated chance. The report must also show **which historical play-level football mechanisms are observable**, rather than reciting only team defensive rankings.

This branch creates an operational **research-only, source-backed scouting evidence layer**, NOT a new validated predicting model. It must never be used to advertise better hit rates or alter a prediction before a proper prospective benchmark. In particular, it **does not satisfy** the owner's complete coach/scout-grade intelligence objective yet.

## What the executable does

`nfl_game_scout.py` matches the current board to the exact season/week, kickoff and player ID in the independent role snapshot. It combines:

- Same-player recent *completed* receiving target depths and running-play gaps, with observed opportunity counts and historical efficiency by play type.
- Opponent's prior **six** completed defensive games: yards and opportunity count allowed on short/intermediate/deep targets and inside/edge runs, with explicit sample sizes.
- Recent role-share direction from the current three-game role-evidence receipt, plus teammate target/carry competition in that same team and market.
- Prior three games of named defender and offensive-line snap exposure, plus PFR weekly defender-attributed historical targets, yards and completions when the PFR player ID matches the snap source; pretip Sleeper injury status where name/team match. **Defender-attributed charting does not identify which receiver was assigned to the defender. Previous snap exposure is not game-day active verification.**
- A mismatch and missing-evidence register, including HIGH-confidence flags with no matching player-specific probability distribution.

Crucially, player-versus-coverage *assignment* is not present in these free sources. Neither are individual offensive lineman vs defensive lineman blocks, route-by-route release or speed/separation, and a verified weekly coaching game plan. The scout report spells those out as **unverified**, never fabricates them, and does not infer a defender being assigned to a particular receiver merely because he played cornerback in recent games.

Play rows are filtered to `(season,week) < (forecast season, forecast week)`, no exceptions. Pregame board and role receipt must carry identical generated timestamps, season and week; game kickoff must also match. Historical release retrieval time is not an immutable original-game-day data availability receipt, so the historical source is **not** misrepresented as a first-seen pregame capture.

## How this addresses the requested deep analysis — and what is still missing

This is the **evidence foundation**: for every upcoming player, identify role movement, known competition, prior actual target depth/run gaps, defensive vulnerabilities, real defenders and OL who played recently, injury cautions, and sample sufficiency. It separates **observed football evidence** from **inferences about future usage**.

Still required before calling the system finished:

1. Verified game-day personnel and lineup/inactive feeds, with source arrival times and player-team identifiers.
2. Player-versus-player assignments *only when available from genuinely sourced route/alignment/coverage data*, with coverage and uncertainty reported. Past snap counts are insufficient.
3. Coaching tendencies, motion/route concepts and QB preference conditional on coverage and pressure, plus blocking front matchup where real assignments are available.
4. An explicit opportunity mechanism (team plays, playcall split, competing routes/targets/carries) and a separately estimated yards-per-opportunity mechanism; fit their effects, **do not hand-tune matchup multipliers**.
5. Out-of-time source-frozen trials with exact same player/game cohorts, missed-starts, calibrated distributions, top-four forecasts assessed at a **clearly defined same event**, and strong baselines (current production, recent-five, bookmaker where legitimately comparable).
6. Make live changes **only** after a challenger passes the preregistered gate. No claim that adding a descriptive feature proves a predictive edge.

## Run / test

```bash
python -m unittest discover -s tests -p test_nfl_game_scout.py -v
python nfl_game_scout.py \
  --board docs/nfl_predictions.json \
  --roles docs/nfl_role_snapshot.json \
  --injuries docs/nfl_sleeper_injury_snapshot.json \
  --pbp /path/to/play_by_play_2025.csv.gz \
  --pbp /path/to/play_by_play_2026.csv.gz \
  --snaps /path/to/snap_counts_2025.csv \
  --snaps /path/to/snap_counts_2026.csv \
  --pfr-def /path/to/advstats_week_def_2025.csv \
  --pfr-def /path/to/advstats_week_def_2026.csv \
  --out /tmp/nfl_game_scout_evidence.json
```

Run the PR workflow **NFL pregame player-specific scout evidence (research only)** to fetch the public nflverse sources, run unit tests and write an artifact containing real upcoming games and player-specific factual scouting. The artifact is inspectable; it cannot secretly edit production `docs/nfl_predictions.json`, `nfl_serving_builder_a.py`, the top-four selection code, or any current betting odds.

### Quality bars

- Zero current/future week plays or snaps can affect a given report.
- No undocumented factor is multiplied into an existing forecast.
- Nonplayed roster status remains unknown, not fabricated as a certain starter or a zero opportunity.
- Every football matchup comparison carries its observed event counts and an explicit noncausal description.
- HIGH-confidence metadata missing a matched distribution is exposed as a data warning.
- A failed download or invalid receipt is a visible failure, never a quietly substituted fake sample.

The next milestone is a pre-registered **mechanism-to-outcome** experiment on the live captured evidence: does the named defensive/pregame lineup context predict **future targets/carries** and **efficiency separately**, better than previous-week role and player-only baselines? This scout evidence layer is the source-integrity prerequisite, not the answer to that experimental question.

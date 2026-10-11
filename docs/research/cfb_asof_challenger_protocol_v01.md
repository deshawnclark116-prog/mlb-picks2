# CFB strictly pregame context challenger — isolated research protocol v0.1

**Status: RESEARCH ONLY. No production model update, official prediction, or betting edge certification.**

## Why this challenger exists

The existing `cfb_context_v2.build_context` determines historical vacated
workload and QB change by inspecting the *target game's final player boxscore*.
That information cannot be used before kickoff. The Oct 11 integrity audit
proved that the original context model's historical feature definitions do
not match its own serving-time feature availability.

`cfb_asof_challenger.StrictAsOf.features` supplies the same interface for
historical replay and for future shadow scoring. It derives participation
shares and team pace exclusively from weeks preceding the target week, sets
`vacated_car`, `vacated_rec`, and `qb_change` to **unknown** when no
timestamped, certified earlier personnel source exists, and derives ratings
using only results preceding the target week. It never interprets the actual
target boxscore as a pregame roster or starter report.

## Research-only training architecture

The candidate reuses the existing CFB context XGBoost experiment engine with
only the strict context function substituted **inside an isolated process**.
Existing champion models, the normal prediction publisher and archived
original first-pregame picks are not modified. The original engine functions
and output directory are restored even when training fails. An explicit
model-output safety guard rejects production locations.

- Train: 2019–2023 historical games.
- Early stopping: 2024.
- Inspect retrospective 2025 comparison; 2025 has already been examined.
- Calibrate the frozen research challenger on 2025, making 2025 a
  **development/calibration** season, not an unbiased evaluation.
- Write models + `research_manifest.json` ONLY into the shadow folder.
- Use `ShadowScorer.probability` to score a candidate with the exact same
  `StrictAsOf.features` API, without making it an official pick.

Training CLI after a real 2019–2025 database is assembled:

```bash
python -u cfb_asof_challenger.py --db /path/to/complete-history.sqlite \
  --out-dir /tmp/cfb_asof_challenger
```

The CI workflow only certifies synthetic future-result poisoning and output
isolation: it does **not** claim that expensive real-data retraining or
deployment has already occurred.

## Important remaining integrity gates

This challenger removes the specific known contamination from contextual
player-workload features, but **does not automatically validate** the base
feature-engine replay population, the presumed eligibility of historical
players, or roster availability before kickoff. Those are separate sources
of train/serve shift. Neither `player_games` data from an already-finished
game nor a historical active-player label is a certified pregame roster.

Upcoming-week ratings require special treatment because the old
`build_ratings` only fills weeks with completed games. When a target
week is missing, the challenger reconstructs the ratings from completed
games STRICTLY before that target week. It must not fall back to
`build_ratings.latest` calculated on the entire database.

The context training engine's 69.5 rushing-yard and TD thresholds are
research labels, **not sportsbook-verified current lines**. Future market
price, active starters, injuries, snap/role uncertainties and source
coverage must be independently certified before any official betting card.

## Promotion requirements

For new 2026 games whose outcomes were genuinely unknown at forecast time,
archive candidate forecasts *before kickoff* and compare against canonical
unchanged first-seen incumbent forecasts on identical games and populations.
Require paired Brier and log-loss gains (and MAE or proper count metrics
for continuous and count player outcomes), calibration quality, cohort
coverage and uncertainty. The Oct 10 results and already-viewed 2025 data
cannot be reused to claim a blind victory. No automatic model promotion.

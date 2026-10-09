# NFL Outcome Engine V2

Status: **RESEARCH ONLY / PROTOCOL LOCKED / UNBUILT**

V2 exists because the prior engine proved that a coherent, leak-safe simulator can still have central player projections that are not accurate enough for the product we want.

## The V2 rule in plain English

We predict the football first:

1. Who is actually likely to play?
2. How many team plays, rushes, dropbacks and targets should this matchup create?
3. Who is actually going to get those opportunities?
4. What should that player do with those opportunities against this opponent?
5. That creates the model's **point projection**.
6. Only after that do we simulate uncertainty and calculate ranges/probabilities.
7. Only after the forecast is frozen may sportsbook data be compared downstream.

Monte Carlo is not allowed to manufacture the model's opinion.

## What counts as success

The headline population is meaningful active players, not a universe padded with backups and nonparticipants.

Every market is judged independently on:
- point-projection MAE and bias,
- percentage within useful error tolerances,
- opportunity error,
- role/share error,
- efficiency error,
- full-distribution CRPS and interval calibration.

A model does not pass just because it is statistically better than a weak baseline. Its absolute error must be useful.

## Every forecast must explain itself

A V2 forecast must emit a machine-readable receipt showing the chain from:
team environment -> player opportunity -> role/share -> matchup adjustment -> efficiency -> point projection -> calibrated uncertainty.

That receipt is also used after the game to identify where a miss entered the system.

## Forward evidence

All data through 2026 Week 4 is burned for development/diagnostics. Week 5 onward can count as clean forward evidence only when a V2 candidate was genuinely frozen and forecast before its cutoff. No backfills.

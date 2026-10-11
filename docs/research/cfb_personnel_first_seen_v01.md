# CFB immutable pregame evidence corpus · v0.1

**Status: Live data-capture infrastructure, NOT a calibrated predictive model.**
This feature does not change any historical or current CFB pick, threshold,
probability or market. It is a necessary prerequisite to training a genuine
pregame analyst model without using the future.

## Why this exists

The current CFB context research `cfb_context_v2.build_context`
computes absences from **the actual target game's completed boxscore**,
which was unknown when that game began. Its production ServingContext has
no corresponding pregame absence data and substitutes `0.0`. A model
trained on hindsight can appear stronger than its game-day predictions.

The existing `cfb_pregame_analyst_evidence.py` captures timestamped ESPN
roster source-status fields and observed player opportunity shares. But
`docs/cfb_pregame_evidence.json` is *rewritten every daily rebuild*.
Without a frozen first-sighting ledger, future research could
accidentally use later information and would be unable to distinguish
injury news genuinely available before kickoff from a postgame outcome.

## The new contract

`cfb_personnel_first_seen.py` appends an immutable receipt for each
season/week/game/market/team/player ID **strictly before kickoff**.

Every receipt records:

* Exact generated forecast, game ID, player ID, matchup, market, original
  probability, original directional research-threshold pick and cutoff;
* Original ESPN roster/injury source status and actual roster retrieval
  time (or explicitly unknown if absent; NOT "confirmed healthy");
* Strictly prior completed-team-game touches/attempts and observed
  role share, preserving missing stat rows as unknown, not DNP/zero;
* Original evidence warnings and a hash of the source evidence;
* Immutable disclaimers: no starting lineup certification, no sportsbook
  market quote, no calibrated probability of beating a market.

The writer refuses board/snapshot generation mismatches, broken or
after-game timestamps, post-capture roster updates, duplicate model
identities, or a claim of verified lineup/books that its source cannot
support. Existing source receipts are never overwritten, even if the
same team/player is later downgraded, scratched or reports a different
role. Rechecking the same board is idempotent.

**No backfills:** only capture evidence that can be verified *at the
actual run time*, not by pretending today's downloaded ESPN pages
were what they said yesterday or in Weeks 1–6.

The combined workflow now appends to
`docs/cfb_personnel_first_seen.jsonl` after the live
`docs/cfb_pregame_evidence.json` capture and saves a coverage/status
report `docs/cfb_personnel_ledger_status.json`. These persist through
the existing append-only GitHub main artifact publication.

## What this still CANNOT certify

An ESPN roster item saying "Active" does not prove game-day available,
starter, full participation, snap percentage, routes, or injury-free.
Passing targets are not in the existing CFB roster/boxscore fields;
receptions cannot be rebranded as targets. The model's fixed 69.5
rushing-yards and 1.5 passing-TD research thresholds are *not verified
sportsbook lines*. Moneyline has no sportsbook moneyline odds attached.
None of this is a betting recommendation.

## Next promotion gates (none bypassable)

1. Pair every future pregame receipt to the actual completed-game
   starter/playing-time outcomes using an independent source, explicitly
   including rostered-but-DNP cases instead of selection-biased row drops.
2. Train a brand-new context candidate on truly prior data, with
   opponent/team/coach, offensive personnel, QB and injury features
   sourced only from original immutable cutoff receipts.
3. Score the EXACT same forward first-seen games as the existing champion,
   compare log loss/Brier/calibration and subgroup harms on a future
   as-yet-unseen window, with game-cluster uncertainty.
4. Line-specific bets require confirmed sportsbook player/book/odds,
   real quoted event identity, timely availability and separately
   calibrated P(win), net vig and actionable market.
5. If the new model cannot demonstrate improvement, retain the incumbent
   and preserve the failed experiment. Do not tune on Saturday, October
   10's already-observed failures to claim a forward win.

This contract establishes the audit trail needed to prevent repeating
hindsight feature leakage. It does not promise a new model's superiority.

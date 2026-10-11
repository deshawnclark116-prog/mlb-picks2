# CFB Saturday October 10, 2026: immutable-first-pregame forensic audit

**Status: observation only. No outcome-driven model, probability-calibration,
feature, eligibility threshold, or market activation changes.** Do not mix
this already-consulted day into a falsely untouched validation set.

## Scope, source of truth and coverage

- Kickoff local date: Saturday **2026-10-10** in `America/New_York`
  (UTC window 2026-10-10T04:00:00Z through 2026-10-11T04:00:00Z).
- Fixed model generation: CFB Week 6, 2026; first valid pregame
  predictions from `docs/cfb_picks_log.jsonl`, matched by canonical game
  ID, entity, stat market and line to `docs/cfb_record.json`.
- **213** unique predictions actually logged before kickoff. **197** with
  gradeable outcomes, **16 without a matched player stat line**:
  4 rushing, 11 passing TD, 1 anytime TD. Those 16 are **NOT
  wins, losses, confirmed DNPs or automatic zeroes**; must reconcile
  identity, starter/roster and true outcome coverage to remove bias.
- **197** scored across **42** finished games: **131 correct, 66 incorrect
  (66.5%)**. This is not the forecast's continuous-yardage accuracy.

| Market | Won / graded | Hit rate | Mean stated confidence | Brier score |
|---|---:|---:|---:|---:|
| Moneyline | 25/42 | 59.5% | 74.0% | 0.2652 |
| Passing TD over/under 1.5 | 46/65 | 70.8% | 59.7% | 0.2141 |
| Rushing yards over/under 69.5 | 37/54 | 68.5% | 65.3% | 0.2166 |
| Anytime TD over 0.5 | 23/36 | 63.9% | 62.4% | 0.1985 |

These are **model-side choices versus research thresholds**, not genuine
FanDuel player betting markets. No real pregame odds, vig or closing
lines, so **hit rate is not proof of positive betting expected value**.

### Concerning moneyline calibration

- 42 selections averaged **74.03%** win probability yet just **25/42
  won (59.52%)**, 6.09 fewer wins than the confidence values collectively
  imply.
- Of **22 selections at 75% or greater**, only **13 (59.1%)** won;
  predicted expected winners **18.59** (average probability 84.5%).
- **75–85% tier:** just **7/14 (50%)**, average predicted win
  probability **81.0%**.
- Several very-confident errors: Toledo 87.59% LOST to Buffalo;
  Duke 85.53% lost to Georgia Tech; Pittsburgh 84.79% lost to
  North Carolina; Texas 84.26% lost to Oklahoma; Florida 84.08%
  lost to South Carolina. Other misses: Alabama 75.93% against
  Georgia; LSU 74.62% against Kentucky.
- **Repeated warning**, not a one-week assertion:
  - CFB Week 4: **26/42** moneyline winners (61.9%) versus **77.9%**
    mean quoted win probability.
  - CFB Week 5: **35/49** winners (71.4%) versus 69.9% stated.
  - CFB Week 6 full window: **32/52** winners (61.5%) versus
    72.2% stated.
  All come from real first-seen ledger rows; the Saturday-only result
  is a distinct slice of Week 6, not a substitute for it.

### Rushing and touchdown numbers should not be mistaken for projected stats

- All 54 Saturday rushing market entries used the same artificial
  **69.5** yard threshold (41 unders: 25 winners; 13 overs: 12 winners).
  That is not each player's expected yards or a verified sportsbook line.
- Some misses were enormous relative to this arbitrary threshold:
  Fame Ijeboi UNDER 69.5, actual **195**; CJ Baxter UNDER 69.5,
  actual **123**; Tylik Hill UNDER 69.5, actual **118**; J'ai'Den
  Thomas UNDER 69.5, actual **112**.
- All 65 Saturday passing-TD entries used **1.5** TDs: 38 unders
  (29 wins) and 27 overs (17 wins).
- **ZERO** scored Saturday rush selections had a stored pregame
  `projected`, `projected_median`, or `projected_mean`.
  The sim overlay had **0** projected rows on the archived Week 6
  output. Therefore there is no honest way to report the exact model
  rush-yard forecast MAE for yesterday. A probability above/below
  69.5 is NOT a projected numerical yard value.
- Passing and receiving **yard** markets remain suspended, so there
  are no validated continuous projections for those markets either.

## Proven training/serving defect (not a hypothetical explanation)

In `cfb_context_v2.build_context()`, prior-game regulars marked
`absent` are determined from the **target game's actual player stat
rows**, and used to fill `vacated_car`, `vacated_rec`, and
`qb_change` for *that same target game*. This violates the model's
as-of contract. Its training and historical validation therefore use
some information unavailable before that game.

In `ServingContext.__init__()` these three team fields are instead
**hardcoded to 0.0**; no actual pregame injury/starter/depth chart is
verified. Moreover `add_context(..., pid=None,...)` used in moneyline
historical training has null context fields, whereas live
`ServingContext.features(...,pid=None,...)` injects team pace
and zero-filled availability fields. This is a reproducible
train-vs-serve schema/distribution mismatch.

Unit regression tests freeze the bad behavior as a FAILING SCIENCE
INVARIANT: changing only the target game's lineup alters historical
`vacated_car` for a different player in that game, even when all
previous games are unchanged. This test is an alarm, **not
permission to promote the current model**.

One further presentation discrepancy: the serving builder calculates a
weekly growing-pool Platt map and includes it in the board metadata,
**but context-v2 overrides those probabilities** with an older context
model's 2025-only Platt map when loaded. Do not describe those
published numbers as fitted by the growing pool.

## Correct next work, without hindsight fitting

1. Label Week 6 moneyline probability evidence **overconfident on this
   slice**; add ongoing by-week/side calibration drift monitoring.
   Do not invent a post-hoc 2026 probability fix.
2. Reconstruct the **exact training and 2025 test context strictly as-of**:
   no current-game participants; source expected starting roles/injury
   reports with timestamps if available, otherwise missing-value,
   consistent in training and live.
3. Replay the locked incumbent and leakage-free challenger using
   identical players/games from genuine first-seen forecasts; report
   full expected-stat MAE, signed bias, Brier, ECE, role/injury bins,
   paired game-cluster bootstrap, coverage, and failed features.
4. For numeric rushing/passing/receiving projections, **log the actual
   first-seen regression median with distribution and timestamps**,
   without deriving fabricated medians from fixed threshold picks.
5. Do not issue official CFB bets without a verified sportsbook market,
   matchup/participant availability, and independently calibrated
   P(over/under actual offered line) and positive net value.

**Verdict: current grader is real but insufficient for the user-facing
human-analyst goal.** Severe ML overconfidence, missing role-status
coverage, research fixed-line props, absent numerical yardage medians,
and confirmed historical look-ahead mean the system needs a
leak-safe structural rebuild and new *future* evidence, not cosmetic
card filtering or tuning yesterday's outcomes.

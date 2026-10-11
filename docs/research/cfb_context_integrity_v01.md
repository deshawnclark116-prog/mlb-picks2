# CFB context production-fidelity investigation · protocol v1

**Status:** Research-only, no production model or betting-pick change. Written
before consulting this investigation's real-data outputs (2026-10-11).
**Starting point:** Saturday 2026-10-10 first-pregame CFB moneylines were
25/42 correct (59.5%), with roughly 74% mean published winning confidence.
Those outcomes have already been viewed; they are **not a blind holdout** for
any model changed in response.

## Code-level defect uncovered, independently reproducible

Existing `cfb_context_v2.build_context`, used for historical model training,
calculates `absent = [p for p in regulars if p not in cur or sum(cur[p]) == 0]`.
Here `cur` is **the actual player-game boxscore from the very game being
predicted**. It then passes current-game-derived `vacated_car`,
`vacated_rec`, and `qb_change` to its historical model training.

But `cfb_context_v2.ServingContext` must operate before kickoff. It
currently populates those same columns with `0.0`, which does not
mean the staff has verified an intact lineup. Thus both hindsight
contamination of training and a distribution shift in production are present.

A second technical risk: `build_ratings.latest[season]` is **post-all
completed weeks**, not a historical as-of team rating; it cannot be
reused to retrospectively certify training/serving equivalence.

**Specific acceptance:** no current-week player identity, carries, targets,
stat row absence, final score, depth chart, injury or starter label may
enter an as-of feature unless a timestamped capture proves it existed
strictly before the forecast. Unknown is **null**, not fabricated active,
inactive, zero missing opportunity, or a sportsbook bet.

The candidate as-of feature function in
`cfb_context_asof_parity_audit.PriorWeekUsage` deliberately holds out
whole target weeks, limits role data to three preceding completed **week**
games, computes player/team carries/reception/pass attempts and expected
pace from those observations, and returns **None** for all unidentified
pregame teammate absences. Historical and prospective candidates must call
this same function. Team ratings must use `build_ratings[(season,team,week)]`
rather than `build_ratings.latest`. The existing production model is left
untouched because it was trained on the contaminated feature definition.

## Falsification and validation (not post-holdout cosmetic tuning)

1. Unit tests create an established backfield player present in Weeks
   1–3 but absent in Week 4, and prove the original trainer uses actual
   Week-4 personnel to infer vacated work.
2. Tests mutate the target and all future weeks, including extreme final
   scores, then prove the new as-of feature output and target-week
   pregame team rating do not change.
3. GitHub CI rebuilds real 2025 cfbfastR + 2026 ESPN player games, prints
   target-game leakage rates, and archives immutable JSON evidence.
4. A separate leak-free candidate would require a genuinely distinct
   training cohort, pre-specified features, strict prospective publication
   dates and a **future blind test** against the actual first-pregame
   champion (same games, same players, same markets). Saturday's results
   cannot be recycled to choose weights, thresholds or calibration.
5. Availability and market prices remain independently unverified:
   `cfb_pregame_analyst_evidence.py` is a roster/usage research receipt,
   not a confirmed active lineup, and 69.5 rushing / 1.5 pass-TD
   values are **research targets**, not sportsbook quotes.
6. No candidate can be promoted until it demonstrates better
   forward log loss/Brier, calibration and side-specific behavior on
   identical first-seen games and proves no unacceptable subgroup harm.

This audit is evidence of a concrete structural defect, **not** an
explanation that has already been causally proved for the six most
confident losing Saturday moneylines. It does not establish that any
additional feature, injury scrape, simulation, or reweighting improves
prediction accuracy.

## Measured historical contamination (after hypotheses were frozen)

Verified GitHub Actions run `38163331076`, using real cfbfastR
2025 and ESPN 2026 player-game source tables:

| Dataset | Distinct audited player/weeks | Legacy vacated carry nonzero | Legacy vacated receptions nonzero | Legacy QB-change nonzero |
|---|---:|---:|---:|---:|
| 2025 | 15,503 | 8,858 | 11,407 | 1,519 |
| 2026 | 5,718 | 2,568 | 2,932 | 300 |

The same player-week can contribute to multiple columns; do NOT sum
column counts as unique players or unique historical forecasts. All
listed nonzero numbers were produced by considering the *actual target
game's boxscore*, so none is a verified pregame availability feature.

Strictly prior-week team opportunity/usage values matched legacy
features in 2025. In 2026 there were 22 pace discrepancies, 16
receptions-share discrepancies, 10 carry-share discrepancies and 2
passing-attempt-share discrepancies; those require investigation of
multi-game same-week ingestion/order before further use.

**Scope limit:** The specific vacated-player variables are not fed to
moneyline training via `add_context(..., pid=None, ...)`, so this
leakage proves player-market context contamination, NOT the cause of
Saturday's 42 moneyline result. Moneyline overconfidence and its 2025
Platt plus in-game paired normalization still need separate,
falsifiable tests on archived pregame forecasts. No causal attribution
to missing injuries or the market is yet established.

The current archived CFB model is left live for continuity of research
but should NOT be described as having a clean, independently verified
prospective probability advantage. New context champion promotion
remains blocked until training/serving parity is repaired and the
replacement earns a truly future forward outcome evaluation.

# CFB moneyline probability fidelity · read-only research v0.1

**Status: observability only; no betting promotion, model training, or
changes to published moneyline predictions.** October 10, 2026 outcomes
are already known. This cannot serve as an untouched calibration holdout.

## What serving currently does

In `cfb_serving_builder_a.py`, the side-specific Platt probabilities
from the base model are replaced, when the context model is present, by
the per-team probabilities from `cfb_context_v2.ServingContext.prob`.
For each game with both team candidates, those two probabilities are
then rescaled to sum to one:

`P(pick) = P_context(pick) / [P_context(home)+P_context(away)]`

This nonlinear operation does not preserve a previously demonstrated
single-team calibration guarantee automatically. It must be measured
against actual unseen games. It should not be confused with the separate
player-market issue: the context player's historical absences were
constructed from completed target-game participation.

The new `cfb_moneyline_pair_provenance.describe_pair` is an exact
post-processing RECEIPT (not a new projection). Every NEW, genuine
pregame moneyline pick carries the untouched original and opposing
context probabilities, their sum, normalization shift and final model
probability. This permits future evaluation of whether normalization
changed calibration without recovering it from postgame information.
The existing `model_prob`, `pick`, existing frozen XGBoost artifacts,
decision thresholds and book-absence contracts are UNCHANGED.

## Retrospective falsification

`cfb_moneyline_firstseen_diagnostic.py` joins the **canonical**
first-pregame grading results to their exact first pregame append-only
ledger entries. It evaluates the published context-v2 probability
against the `raw_prob` from the older XGBoost model on the **same side
and same game**, with game-resampled paired Brier differences, and
reports cases where `raw_prob < 0.5` yet context selected that team.

The old raw scores are **not** a valid replacement betting strategy:
they can select the opposite side, are not guaranteed calibrated for the
served selected side, and would require a fresh forward selection policy.
All 2026 outcomes examined here are consulted. No fixed threshold,
temperature adjustment or confidence-haircut can be selected from
Saturday's outcome as proof of improved prospective performance.

## What it cannot establish yet

We do not have historical pre-pair context Platt output in the first
2026 ledger; before the new metadata is published there is NO honest
causal attribution of moneyline overconfidence to paired normalization.
The first newly captured receipts can be evaluated only after genuinely
future games finish. No sportsbook moneyline prices, lines, actual
starting lineups or offensive personnel are currently certified by
these moneyline model artifacts.

Promote a future challenger ONLY if original first-seen, same-matchup
forward log loss/Brier, reliability bins, credible game-cluster
uncertainty and subgroup safety beat the incumbent. Do not hide observed
Week 6 failures with retroactive corrections or unsupported odds.

# NFL analyst-intent research registry: opportunity-first shadow v0.1

**Status:** Research-only, frozen before examining its 2025/2026 results.
**Purpose:** Repair a concrete failure in football reasoning. A small-sample backup
can look HIGH workload-stable without having a plausible starting role; a median
4 yards away from a book line is not a calibrated edge.

## What the production yardage model really knows

* `nfl_yardage_v2.Projector` serves rushing/receiving yards using the
  last three games, prior/current usage, player share, opponent **aggregate**
  yards allowed, game spread and implied team total. It does not certify
  active status, snap-count expectations or future depth-chart role.
* The repository already contains `nfl_yardage_v3` and `nfl_context_v4`,
  with snaps, teammates missing, QB changes, opposing scheme and defensive
  personnel. More features are **not proof of better results**.
* The locked V4 comparison on 2025 evaluated rush 24.89 yards MAE versus
  simpler V3 inputs at 24.83; receiving 22.20 versus 22.05. V4 is not
  promoted by this work.
* The independent first-seen 2026 Weeks 1-4 audit reported 417 matched
  rushing/receiving projections: MAE 22.55 vs same-player last-three
  23.69; paired **game-cluster** 95% CI for model uplift [-0.47,2.75].
  Thus the live model's superiority is not established.
* Source HIGH/MEDIUM/LOW tags reflect simulation distribution tightness,
  not exact-yard accuracy or P(over the bookmaker line). A market being
  offered does not confirm the player will start.

## Research question and precommitted structure

Can expected **carries or targets first**, multiplied by shrunk
yard-per-opportunity efficiency and a leak-safe defensive correction,
outperform naive recent-yard averages on historical game results?

No weights are to be tuned after consulting 2025 and 2026 results.
The candidate is written before evaluating those held-out years.

Candidate inputs, all computed with `week < target_week`:

1. Player recent-three carries or targets (65%) and season average (35%).
2. Recent player/team attempt share times recent team attempt volume (50/50
   blend with #1). Sample size and role trend retained in the receipt.
3. Player last-eight yards per attempt shrunk toward prior-as-of league
   efficiency with 30 rushing / 20 receiving pseudo-attempts.
4. Opponent prior-as-of allowed yards per attempt, shrunk by
   `opportunities/(opportunities+50)` and bounded to a ±10% effect.
5. Refuse unsupported new-team and low-volume players instead of declaring
   them predictable. This is a *coverage limitation*, not an out/injury label.

All scheduled games for the same week are held out simultaneously.
Game-level bootstrap, not independent player bootstrap.

## What this does NOT do yet

* It does **not** verify next-game active/inactive, starters, snap
  participation or injury-news recency. Those need timestamped source
  captures before the cutoff; using the actual game's snaps in training
  to "predict" participation would be hindsight leakage.
* It does **not** treat missing nflverse player-stat rows as played zeroes.
  Retrospective backtesting only includes observed player-game rows.
  Missing-player selection bias must be solved before claiming deployed
  forecast quality.
* It does **not** generate betting probabilities, confidence labels,
  bettable cards or recommendations.
* 2026 outcomes have already been examined during earlier research, so
  2026 is a **consulted evaluation**, not an untouched blind holdout.
* It does **not** change existing production models, forecast timing,
  selection, standings or website components.

## Required promotion gates (none are satisfied by these tests alone)

- Archived as-of pregame predictions compared to real outcome for **every**
  roster-eligible player, including zeros for genuine confirmed DNP and
  excluding ambiguous status with explicit coverage counts.
- Player-role and availability layer sourced from timestamped
  pre-kickoff roster, depth chart, snap/route and injury data.
- Same-game, same-player, same-snapshot comparisons with frozen V2 serving
  and naive; paired game-cluster bootstrap intervals **and** subgroup harm
  for backs, receivers, new teams, workload expansion and rookies.
- A truly future untouched season or forward paper period.
- For betting: side probabilities separately calibrated to the **actual**
  market, market freshness and bookmaker, expected value net of vig.
  Six highest raw stability grades cannot constitute picks.

**Decision until proven:** No promotion. The current system still produces
research projections; book availability is an execution gate, not evidence
of model accuracy.

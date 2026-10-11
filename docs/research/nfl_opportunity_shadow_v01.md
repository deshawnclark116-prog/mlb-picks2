# NFL analyst-intent research registry: opportunity-first shadow v0.1

**Status: FROZEN / REJECTED for production.** Code and parameters were committed
before the real 2025/2026 evaluation. Do not rescue-tune on these outcomes.
The candidate failed its first-seen-champion superiority gate.
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

## Frozen evaluation outcomes (2026-10-11; test run 38110852058)

Real nflverse data 2024–2026, with entire weeks scored before updating
features. The **observed-only** research cohort compared against same-cohort
last-three-game average, which is too weak a baseline for promotion:

| Cohort | Shadow MAE | Last-3 MAE | Paired game-cluster lift CI |
|---|---:|---:|---|
| 2025 rush (n=772) | 26.426 | 28.161 | [+1.022,+2.522] |
| 2025 receive (n=2044) | 23.889 | 25.284 | [+0.965,+1.838] |
| 2026 rush (n=125) | 27.960 | 28.573 | [-1.070,+2.447] |
| 2026 receive (n=333) | 26.982 | 27.988 | [-0.258,+2.167] |

Observed-row coverage as share of player-stat rows:
2025 rush 47.7%, receive 38.5%; 2026 rush 32.9%, receive 25.8%.
Those figures MUST NOT be framed as prospective roster eligibility coverage.

### Correct test: frozen production's FIRST-SEEN pregame predictions

The test used `nfl_numeric_accuracy_audit.evaluate` to select genuine
logged-before-kickoff historical model medians, and matched those to the
opportunity hypothesis on **the same season/week/player/team/market/game and
official actual yardage**. It further withheld shadow comparisons where
the entire prior week was not completed before the incumbent was published.

- Canonical eligible first-seen numeric forecasts: **430**.
- Paired as-of comparable forecasts: **108**, across **22** NFL games.
- Missing unique shadow projection: **191**; preceding team/defense week's
  information not yet available at incumbent publication: **131**.
- Matched cohort original champion MAE: **26.318 yards**.
- Matched cohort opportunity hypothesis MAE: **30.414 yards**.
- Shadow gain vs incumbent: **-4.096 yards** (i.e. shadow WORSE).
- Paired NFL game-cluster bootstrap 95% CI for gain **[-6.499,-1.127]**.
- Receiving subset: 63 forecasts; shadow worse **1.166 yards**,
  CI [-4.473,+2.248] — not established separately.
- Rushing subset: 45 forecasts; shadow worse **8.199 yards**,
  CI [-13.914,-3.039].
- Historical role-expanding, role-declining and no-change subsets all had
  negative point-estimate gains. No favorable subgroup promotion.

**Decision: FROZEN NOT PROMOTED.** This hypothesis is not competitive with
what production actually predicted before kickoff on its matched observed
cohort. Moreover, the matched coverage of **25.1%** of the incumbent's
first-seen predictions makes any global champion generalization invalid.

No after-the-fact weight changes, thresholds, feature rescue or 2025/2026
repeated optimization. What can reopen the research: genuinely new
pre-kickoff evidence (participation/snap/route/injury reports) and an
independent, forward-timestamped study. The retained role-trend features
remain a research *diagnostic*, never a production pick confidence.

# NFL intelligence program — audit and results (2026-11-24)

Objective (from the owner): an NFL engine that explains *why* a player should perform a certain way — coverage, pressure, blocking schemes, fronts, coaching,
target competition, game script — and that keeps numerical projections separate from threshold probabilities, calibrated from one coherent distribution.
This document is the audit of what exists, what was newly tested, and what the evidence says. Every number is from a script in this folder.

Protocol for the new experiments (`run_*.py`): walk-forward by week, each week predicted from a model trained only on strictly earlier weeks (2016 onward);
**development** window = 2024 weeks 6-18; **confirmation** = 2025 week 6 – 2026 week 11. A feature group is credited only if it helps on BOTH windows.
Caveat stated plainly: the 2025-26 window had been used for earlier model-selection decisions (not for these features), so it is a confirmation, not a pristine
holdout. The untouched test is the live 2026 weeks 12+ (see "Forward holdout").

---

## 1. What information does the system genuinely have?

Live serving (`nfl_yardage_v2x`, safe-number table): player-game box scores 2016+ (carries, targets, yards), recent/season/prior-season averages and volume,
share of team volume, spread / implied team total / home-away, opponent total yards allowed, injury and practice status (nflverse), teammates listed
out/questionable, offensive snap share.

Research-only (not live): the Phase 1 event simulators (role state, opportunity, efficiency), V3 joint game-script, V5, V6A/B joint-opportunity shadow, V7
roster-returner, V8 passing, independent architecture tournament, challenger runs (role/quality/line/FTN/specialist).

Newly verified as **available for free and now joined play-by-play** (`build_plays.py`, 346,620 run/pass plays 2016-2026):

| Source | What it gives | Coverage | Live (2026)? |
|---|---|---|---|
| nflverse play-by-play | air yards, pass location/length, run location/gap, shotgun, xpass / pass-over-expected, EPA | 2016-2026 | yes |
| nflverse participation | on-field offense/defense players per play, offense & defense personnel, formation, defenders in box, pass rushers, pressure, time to throw, targeted route, man/zone, coverage shell | personnel/box/players 100% 2016-25; pressure 100% 2023-25; man/zone + shell ~55-61% of plays 2018-25 | **no** (2026 file not published) |
| FTN charting | play action, motion, screen, RPO, QB location, box count, blitzers, QB out of pocket, drops, catchable, contested | 100% of plays 2022-2026 | yes |
| PFR advanced (weekly) | per-defender targets, yards allowed, completions allowed, pressures, missed tackles; per-rusher yards before/after contact, broken tackles; passer pressure | 2018-2026 | yes |
| nflverse snap counts / injuries / depth charts / players | snap shares by position, weekly injury+practice status, depth chart (weekly to 2024; ESPN daily snapshots 2025+), id map | 2012+ | yes |
| Next Gen Stats, ff_opportunity | not found at the tried release paths | — | — |

## 2. What is only approximated by broad proxies?
- *Who covers a receiver*: opponent total yards allowed (team level). Lineup-aware defender quality was built in this program (below).
- *How the defense plays the run*: opponent rushing yards per game. Gap/box/personnel versions were built (below).
- *Pass protection / pressure*: nothing live before; team pressure and time-to-throw now exist historically.
- *Target competition / role*: share of team volume over recent games, snap share; no route participation (proxy possible from participation lists, 2016-25).
- *Coaching tendencies*: team pace and pass rate only.

## 3. Important football relationships not represented anywhere (and why)
- **Who actually covers whom, route alignment (slot/outside), release/separation per route, blocking assignments, coverage per receiver per snap.** Not in any free
  dataset located. `route` in participation is the targeted receiver's route only; no alignment. I did not infer individual matchups.
- Offensive-line individual pass-block/run-block win rates (not free).
- Coordinator-level tendencies and in-week coaching news (not a data feed).
- Participation-derived features for the *current* season (file not yet published).

## 4. Free sources that could provide deeper information
See the table in section 1; all of those that exist were joined. The remaining gap is alignment/tracking (who-covers-whom, separation, speed), which is licensed data.

## 5. Tested properly vs genuinely untested

**Tested in this program, with results (receiving yards; average miss in yards; lower is better):**

| Information added to the live model's inputs | 2024 dev | 2025-26 confirm | Verdict |
|---|---|---|---|
| (none: current v2x inputs) | 22.67 | 22.33 | baseline |
| **Receiver's own play-level profile** (shrunk yards/target, target depth, air yards, catch rate) | **22.41** | **21.91** | **replicates: ~1.5-1.8%** |
| Receiver's results vs man / vs zone | 22.40 | 21.96 | same information as the profile (efficiency) |
| Defense yards allowed by target depth | 22.64 | 22.34 | no replicated gain |
| Defense scheme: pressure, blitz, man rate, coverage shells | 22.60 | 22.36 | no replicated gain |
| Offense environment: protection, time to throw, play action, motion, personnel | 22.69 | 22.33 | no gain |
| Lineup-aware defender quality (per-defender charted coverage, who is expected in the secondary, starters out) | 22.42 vs 22.36 | 21.91 vs 21.95 | inconsistent; not signal |
| Explicit interactions (depth mix × defense depth, man/zone mix, pressure × deep share, shell × deep share, blitz × short) added to the profile | 22.37 | 21.99 | adds nothing beyond the profile |

Rushing yards:

| Information | 2024 dev | 2025-26 confirm | Verdict |
|---|---|---|---|
| (none: current inputs) | 24.61 | 24.51 | baseline |
| Rusher profile (ypc, explosive, stuffed, box splits, gap splits, yards before/after contact) | 24.62 | 24.33 | weak / confirmation only |
| Defense run profile, offense run profile, DL/LB starters out, OL starters out/questionable | 24.63-24.69 | 24.44-24.51 | no gain |
| Explicit matchup interactions (gap tendency × defense by gap, box matchup, ypc matchup, blocking × front) | 24.44 | 24.11 | looked like a gain — **but see control** |
| **Control:** same features with a *random* opponent instead of the real one | 24.41 | 24.19 | **same gain with a random opponent → not a matchup effect; it is the rusher's own efficiency** |

**Live-season check using only features computable live** (`run_*_live.py`): receiving profile 2026-only 22.80 → 22.36 (1.9%, P=1.00); 2025 21.95 → 21.54.
Rushing profile 2025 25.14 → 24.87 (P=0.98), 2026-only 23.71 → 23.63 (P=0.68; n=492).

**Game environment** (`team_env.py`; team-level volume, average miss in plays): pass attempts 6.26 (last-8 average) → 5.99 (script) → 5.85 (+ pass-rate-over-expected,
neutral pass rate, opponent-faced pace) in development; 6.56 → 6.27 on confirmation. Rush attempts 5.77 → 5.37; 5.93 → 5.65. Total plays: no real gain. Even the
team-level environment has ~6 attempts of irreducible game-flow noise.

**Earlier tests in the repo/session (not repeated):** who is out / injury status (≈0.1 yd even with perfect knowledge of who sat), snap share (+ ~0.6%), training on 2016+
(+1.2% rushing, +0.6% receiving), weather, rest, defense by position group, depth-chart rank, workload features, full play-by-play game simulation and 9 tuning variants,
blend of simulation with the model (≤0.6%), individual-player floors, challenger role/quality/line/FTN/specialist models, V3/V5/V6/V7 candidates.

**Genuinely untested:** alignment-aware matchups (no data); route-participation proxy from participation lists; QB-specific target preference (pairing) features;
coordinator/coaching-change effects; red-zone/goal-line usage for touchdowns; QB passing yards / attempts / completions as markets in this framework; receptions as a market;
sacks and anytime-TD markets under this protocol.

## 6. Why did earlier work fail — information, features, data, architecture, or weak signal?
- **Weak signal, with a measured ceiling.** If a player's real carries/targets were known, the average miss would still be 15.7 (rushing) / 16.8 (receiving) yards
  (`research/nfl_model_tests`, `volt.py`): ~16 yards is per-play luck. The live model misses by 22-25; the gap (6-9 yards) is uncertainty in workload, and
  workload itself is predicted only ~11% better than a recent average. Nothing tested so far shrinks that gap by more than a couple of percent.
- **Feature construction was a real, fixable problem in one place:** the box-score efficiency proxy. Replacing it with play-level, shrunk per-target/per-carry
  skill gave the only replicated gains in this program (receiving ~1.5-1.9%, rushing ~1%).
- **Data limits, not model architecture:** many architectures (boosted trees, opportunity-first, game simulation, ensembles, specialists) landed within ~1% of each other.
  The matchup features that were *buildable* (team-level scheme, defender lineup quality, gap/box matchups) showed no replicated signal, and the one rushing result
  that looked like a matchup gain failed its random-opponent control.
- **What would change this:** alignment/tracking data (who covers whom, separation, route and block assignments) — the information a scouting department has. It is
  not in any free source I could find. I did not fabricate it.

## Forward holdout and promotion rule
Nothing here was promoted on the basis of these experiments alone. The receiving profile is the first candidate that meets the bar (replicates on both windows and in the
live season using live-computable features). Plan: freeze the model, features and thresholds, and grade it only on 2026 weeks 12+ as they happen, alongside the current
live model, before it replaces anything. Rushing profile: insufficient live-season evidence; stays research.

---

## 7. Distribution, calibration and the top-four selector (`dist_fit.py`, `dist_eval.py`; 2026-11-24)
One coherent model per market: walk-forward XGBoost quantile models (5%..95%, 11 levels, trained only on earlier weeks) on the best candidate inputs
(receiving: live inputs + play-level receiver profile; rushing: live inputs). The numerical projection is the median; every threshold probability P(Y >= T) is read
off the same predicted distribution (piecewise-linear CDF through the predicted quantiles). Players are conditional on having played (DNP is a separate probability).

**Numerical accuracy (median projection, average miss, yards):** receiving 22.40 (2024) / 21.90 (2025-26); rushing 24.52 / 24.55.
**Predicted range:** the 80% range held 77-78% (receiving) and 76-80% (rushing); the 90% range held 88-89%.

**Threshold probabilities (all 5-yard thresholds, n = 35k-63k per cell):** Brier 0.135-0.138 versus 0.21-0.23 for a constant; calibration error 0.5-1.4 points.
Isotonic calibration fitted on earlier weeks changed nothing (raw 0.1358 vs calibrated 0.1359): the quantile distribution is already calibrated.
Reliability, all buckets: stated 4.9% -> actual 5.3%; 15% -> 15%; 25% -> 25%; 35% -> 35%; 45% -> 46%; 55% -> 55%; 65% -> 66%; 75% -> 75%; 85% -> 85%; 94% -> 94%.

**Top-four selector** (per player, the highest 5-yard threshold with calibrated chance >= 70%; four best per game):
| Ranking | Window | Picks | Stated | Actual | Avg threshold |
|---|---|---|---|---|---|
| Calibrated chance (design) | 2025-26 | 1293 | 73.7% | 73.5% | 34 yds |
| Raw chance | 2025-26 | 1278 | 73.6% | 73.7% | 35 |
| Random four of eligible | 2025-26 | 1293 | 73.0% | 72.2% | 34 |
| Biggest projection | 2025-26 | 1293 | 73.0% | 73.4% | 38 |
| Calibrated chance | 2024 | 642 | 74.6% | 73.5% | 36 |
| Random four of eligible | 2024 | 642 | 74.0% | 74.8% | 36 |

Reading: the stated chances are trustworthy (stated = actual). But ranking by chance among players who already clear 70% does not pick *better* than choosing at random
among them; the useful lever is the size of the threshold at a given reliability (ranking by biggest eligible threshold gave 38 yds at the same 73% hit rate vs 34).
Caveat: all figures condition on the player playing; real-world hit rates are lower by the did-not-play rate, which must be modelled separately.

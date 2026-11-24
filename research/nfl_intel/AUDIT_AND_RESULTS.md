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

# NFL analyst role preflight: source-backed, research-only

This is the missing **pregame personnel and opportunity evidence**, not
another numerical model fitted on known outcomes and not a product badge.

Source types and truthful semantics:
- `docs/nfl_predictions.json`: immutable model's recorded yard median,
  original stability grade and historical game count (no bookmaker market here).
- `nfl_model.sqlite` recreated from nflverse: completed **prior-week**
  team games and actual observed carries/targets, player shares and
  explicit gaps when no player-stat row is present.
- ESPN NFL team rosters: live roster names and injury-status strings,
  with UTC **retrieval** timestamp; ESPN roster presence **cannot**
  certify starter/active/inactives at kickoff, which often change later.
- The model's 5 carries / 3 targets recent-usage floors trigger a
  **source-quality warning**, not a proven sportsbook success gate.

The audit never reads stats from this week's game for prior-role features.
No missing row is silently interpreted as an observed zero or confirmed DNP.
No unsupported name match is declared a verified active starter.
No betting probabilities or new forecasts are produced.

First live evidence check for 2026-10-11 PHI at JAX, CI run
`38111114489` (15 yard projections): **both rosters retrieved**.
The original rule flagged **5** for inadequate observed opportunity/row
coverage. A later transparent, diagnostic-only low-volume flag using
the existing V2 5-rush/3-target eligibility floors brought that to
**7 flagged out of 15**, on live CI run `38111178382`.
Those flags include HIGH-stability Dameon Pierce (recent team rows
not established) and HIGH-stability Josh Cameron (1.33 recent targets,
5.37% recorded target share); also Ameer Abdullah (2 recent rushes),
Will Shipley (4.33 recent rushes), LeQuint Allen Jr. and Travis Hunter
(no measured recent opportunity) and Elijah Moore (role rows missing).
Bhayshul Tuten had 15 observed recent rushes per game and 55.71%
recorded team rushing share in the specified comparison.
This is evidence-risk triage, **NOT** a validated bet exclusion rule;
further blind validation is required before automatic deployment.

**DO NOT** confuse ESPN roster presence with inactives/starting lineup
proof, recent attempts with snaps/routes, or an offered FanDuel prop with
a high-probability wager. Book offerings and injuries require rechecking
before kickoff.

## Required next model research

- Archive timestamped starting/depth chart, eligible/active reports,
  routes, offensive snaps and teammate availability for every NFL game.
- Model opportunity distribution before efficiency, conditioned on
  roster/scoring/game script and opponent defense, with player-game
  missingness explicitly represented; **do not tune the frozen failed**
  opportunity-only hypothesis on 2025/2026.
- Build a new model on an earlier training window, then forward test on
  future, untouched outcomes, exact same games against first-seen locked
  champion and strong naive baselines. Cluster bootstrap paired errors
  and subgroup harm, not a single training AUC.
- Calibrate P(over a **verified sportsbook line**) separately;
  require evidence-backed value after vig, line age and participant
  confirmation. The original "HIGH workload stability" grade is
  prohibited as a proxy for hit probability.

This component is **research-only** and cannot be used as a source of
official picks until those steps pass.

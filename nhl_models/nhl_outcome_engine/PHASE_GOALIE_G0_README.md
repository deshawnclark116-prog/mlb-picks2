# NHL Goalie G0 - feasibility audit (no model fit)

Protocol `phase_goalie_g0_protocol.json` (committed first); results `phase_goalie_g0_feasibility.json`. The legacy goalie-saves model (AUC 0.5673, calibration p 0.001) is a historical comparator only.

| question | result |
|---|---|
| Q1 appearances / starts | IDENTIFIABLE: 22,104 of 22,104 team-games have exactly one starter |
| Q2 starts vs relief | DISTINGUISHABLE: relief = 6.2% of appearances; 6.5% of starters pulled (<3300 s) |
| Q3 prior-only goalie skill | CONSTRUCTIBLE_WITH_COVERAGE_LIMITS (no leakage; 2018+ starts have >=5 prior appearances for ~97%, >=10 for ~94-97%) |
| Q4 team shot environment | CONSTRUCTIBLE_WITH_COVERAGE_LIMITS (every 2018+ team-game has >=5 prior team games) |
| Q5 label reconciliation | **UNRECONCILED** (strict): boxscore == stats-REST on 425/425 sampled goalies; 72 rows (0.31%, 2022+, all +1) violate saves + goals against = shots against; team-level differences vs opponent SOG are the opponent's empty-net goals in 99.58% of team-games |
| Q6 conditional-on-start saves | RESEARCHABLE_HISTORICALLY: saves | start mean 26.6, dispersion index 2.22; shots against vs opponent prior SOG r = 0.202; game save% vs goalie prior save% r = 0.058 (descriptive, no model) |
| Q7 pregame start | BLOCKED_PENDING_LIVE_CONFIRMATION_EVIDENCE; prior-only descriptive: same starter as previous game 44.8% (back-to-back 11.6%, rested 51.0%) |

**Split:** `GOALIE_SAVES_CONDITIONAL_ON_START_COMPONENT` = RESEARCH (researchable with limits; needs a source-adjudication step for the Q5 +1 class before scoring); `PREGAME_START_PROBABILITY_COMPONENT` = BLOCKED (confirmation); the unconditional pregame goalie-saves head = BLOCKED. The starter fact is a label / conditioning event only, never a pregame feature. No challenger was fit.

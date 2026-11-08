# NFL meaningful yardage targets — research contract V1

**Created 2026-11-08. This is a selection-layer research candidate, not a validated betting model.**

The weekly player-yardage simulator still produces independent, deterministic
10,000-draw, historical-usage conditional distributions. The modeled mean,
median, P10/P90, and original book-line-specific calculations are unchanged.

Instead of pretending that P(reaching a mean) is an overall confidence score,
the candidate selector enumerates **predeclared** meaningful integer outcomes
from each player's own original draws:

- Receiving yards: 25+, 30+, 35+, ... (5-yard grid).
- Rushing yards: 30+, 35+, 40+, ... (5-yard grid).
- Each offered target must additionally be at least **55% of that player's
  published model mean**, rounding up to the next 5-yard threshold.
- At least **10,000 actual source-matched draws** are mandatory. No refitted
  normal curve, arbitrary bookmaker line, synthetic odds or default percent.
- The target is the **hardest** (largest yardage number) with a one-sided
  95% Wilson Monte Carlo sampling lower bound of **at least 65%**.
- When none qualifies, publish the complete valid numeric projection but **do
  not promote any target**. Never force 10 featured outcomes per game.
- Independently verified recent role evidence (3 observed rows; no warnings)
  and a matching pre-kickoff forecast/role receipt are required for featuring.
- A recent roster listing is **not** certified active status. The simulation
  omits DNP uncertainty and full-game joint correlation.
- P(hit) is therefore **conditional on simulated usage and uncalibrated**.
  The Wilson bound is MC *sampling precision*, not real-world reliability.
  Never describe these rates as proven 65% win rates.
- The frontend displays one best target per qualified player+market and ranks
  by its original simulation probability, descending. The raw model mean,
  median and the other grid thresholds stay accessible.
- Model-derived targets and bookmaker markets remain separate.
- Never rerank using postgame observations. The first **genuine pre-kickoff**
  top ten per game must be frozen in the append-only
  `docs/nfl_model_target_firstseen.jsonl`. Later refreshes cannot replace it.
- The independent postgame grader queries actual NFL player-game box scores;
  missing/ambiguous athletes remain unresolved, never assumed zero.
  Grade at least six hours after kickoff. Measure actual hit rate and
  Brier score across ALL frozen predictions, including failures. 
  `docs/nfl_model_target_forward_grade.json` is owned only by the
  postgame grading workflow to prevent same-file git add/add conflicts.

## Promotion / falsification gate

V1's 25/30-yard floors, 55% relative threshold, 5-yard grid, 65% lower bound
and role checks are **precommitted research decisions**. Do **not** tune them
after seeing this week's results just to increase apparent accuracy or board
size. Compare V1 to the previous mean-target baseline and real forward outcomes,
by market and role bucket. Evaluate score calibration, leakage, coverage,
stability, and missed/missing records. Promotion to official betting or
"validated 65%+" labels requires independent forward evidence and preferably
a full-game coupled opportunity model with official participant availability.

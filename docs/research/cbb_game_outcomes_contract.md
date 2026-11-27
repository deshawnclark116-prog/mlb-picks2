# CBB Game Intelligence V0 — moneylines, spreads and totals first

**Research branch, not live betting predictions.** The user prioritizes game markets over player props. All three markets MUST be functions of one home-score/away-score distribution. A prediction like "home wins 75%" is a probability of the binary home-win event, *not* "75% confidence the predicted score is exact."

## Real source and role

This research downloads the [sportsdataverse/hoopR NCAA men's basketball boxscore and schedule releases](https://github.com/sportsdataverse/sportsdataverse-data/releases) for ESPN seasons 2023–2027. The November 25, 2026, release files include actual current-season results and future fixtures. Player-box data describes **previous completed game** playing time and rotation; it is not a verified starting-five, injury or availability report.

## Model inputs, without sportsbook data

- Opponent- and season-aware offensive/defensive points per estimated possession, 3-pointers / 2-pointers through effective FG%, possession/turnover cost, offensive rebounding versus opponent defensive rebounding, free-throw attempts per FG attempt.
- Historical team pace, home vs. neutral, role concentration, rotation change, and recent form. Each profile uses completed **earlier dates only**. Last year's same-team history is discounted to 3 pseudo-games, and all features shrink toward a fixed national prior.
- A physical baseline predicts pace × offense-vs-defense efficiency. A regularized multi-output model jointly estimates corrected winning margin and total points, implying one home and one away projected score.
- Player boxscores contribute prior rotation exposure. In this v0 they do **not** provide enough real evidence to claim defensive assignments, exact coach play calls, injuries or game-day lineups.

## Evaluation

Training: 2023–2024 historical ESPN seasons.
Model-strength selection: 2025.
One out-of-sample historical test: 2026.
Current 2026–27 season (ESPN season=2027): **research-only upcoming game projection**, not a backtest. Do not select based on the 2026 holdout.

Reports: winner accuracy, moneyline Brier score, margin MAE, total MAE, plus explicit comparison to possession/efficiency mechanical baselines. No fake against-spread or over/under wins: historic line prices and first-seen forecasts haven't been collected.

- Home moneyline P(win): normal CDF at margin zero with margin residual uncertainty estimated from development season.
- Home spread P(cover): P(home margin + *actual sourced home handicap* > 0).
- Total P(over): P(combined points > *actual sourced closing or pretip market line*).
- Unless sportsbook line is provided, **P(cover)** and **P(over)** are null. A projected margin/total is not itself a known available sportsbook line.
- Margin and total are modeled jointly with observed development residual covariance documented. V0 probabilities assume a normal marginal residual distribution; they are not yet forward calibrated or verified betting value signals.

**Ship rule:** Withhold from production unless out-of-time Brier and score errors beat defensible baseline on the same actual games and confidence calibration is verified. Only then add forward first-seen snapshot logging, out-of-sample calibration and real sportsbook line capture. Model should abstain when roster/lineup status or market coverage is insufficient. No under/over publication rule has yet been decided.

## How to run

Use GitHub Actions workflow "CBB game outcomes research (moneylines spreads totals)" on the research PR. It downloads source data and runs unit tests; a single workflow artifact stores all holdout metrics and next seven days' research projections, and **does not publish or alter** the current NBA, NFL or other sports systems.

Before moving beyond v0, assess better conditioning: coaching style and rotations, actual player availability, conference and travel differences, transfers and season resets, lineup continuity, matchup fit, offensive rebound / turnover interaction, and genuine tempo-conditioned possession simulation. Add only when actual available pregame sources and a clean post-holdout evaluation justify it. Avoid a giant feature list with no validated improvement.

**Integer-line push handling:** First-seen market evaluation approximates the integer scoreboard using rounded normal residuals. Whole-point spreads and totals include explicit modeled P(push), withheld from realized ROI; half-point lines have zero push mass. This has not been forward-calibrated and is not a recommendation to bet.


## November 25–26, 2026 audit: stronger baselines and uncertainty falsification

The earlier 2026 ESPN season has already been inspected and cannot be reused as a fresh promotion holdout. Two *diagnostic-only* extensions were executed in the research workflow; detailed, reproducible JSON + readable Markdown are saved as GitHub Actions artifacts.

- **Fixed independent Elo check** (80 rating-point home prior, 20 K-factor, 25% seasonal regression to 1500, same-day outcomes never visible to other games): 2026 model moneyline Brier **0.18099**, Elo **0.19350**, while winner accuracy was **71.87% versus 71.73%**. Model's nominal 80% interval covers **81.2%** of margins and **78.2%** of totals. The 2025 development season shows similarly better Brier than Elo, but these historic records are not first-seen odds-graded outcomes.
- **Neutral venue problem**: 708 neutral-site 2026 games, model Brier **0.23752**, Elo **0.22403**; model winner accuracy **59.5%**. Game-by-game evidence analysis is incomplete on sites/coaching and should not assume stronger neutral-court accuracy. With both teams having at least five current games (4,912 matchups), the model's Brier **0.20294** effectively ties Elo **0.20278**.
- **High-confidence calibration**: for model-selected favorites predicted 80–90% chances, observed win rate was **83.4%** vs mean predicted **85.0%** (n=1,059); at 90–100%, observed **96.0%** vs mean **96.1%** (n=1,013). Predictions are not equivalent to profitable moneyline prices.
- **Score dispersion failure in first game(s)**: among 516 games where either team had **zero previous games that season**, the nominal 80% model margin interval covered only **62.4%** of 2026 finals. Treat early-season exact score projections as especially uncertain.
- **Separate uncertainty research branch candidate, NOT promoted**: pregame history groups fixed as (any team zero), (both teams 1–4), (at least one team 5+). Normal residual scales estimated **from 2025 only** with a fixed 100-game shrinkage prior. Diagnostic 2026 zero-history margin coverage improved **62.4%→77.9%**, overall margin coverage **81.2%→80.6%**, and combined score negative log-likelihood **8.331→8.310** (lower better). However overall moneyline Brier **worsened 0.18099→0.18122**. Therefore **no automatic moneyline variance change**; any separate score-interval adjustment needs original 2027 pretip result validation and a preregistered gate.

**Promotion status: no automatic promotion**. No sportsbook lines were configured in the real runner (zero verified odds matches); no graded live ATS/O-U profitability. Do not fit new parameters to the already-inspected 2026 subgroup results. No deployment to production in this PR.


## Champion registry — 2026-11-27

**CBB_HYBRID_V1** is the sole primary CBB forecasting model and baseline for all subsequent challenger tests, authorized by the owner. Non-neutral margin/win: fixed nonlinear opponent-residual model; non-neutral total: fixed residual Ridge; neutral: frozen swap+Elo constituent. Historical original V0 is retired from active competition; its receipts and code remain for audit. No candidate may benchmark against V0 instead of the hybrid. Compare on paired forward first-seen cohorts, probability calibration/Brier, margin and total MAE, disagreement and competitive games, and genuine market lines when available. Historical 2023–27 results are inspected, not fresh promotion holdouts. This selection is a forecasting champion, not certified betting profitability. Current champion receipts are isolated in cbb-forward-state/cbb_forward/champion_v1. Missing market prices remain null.

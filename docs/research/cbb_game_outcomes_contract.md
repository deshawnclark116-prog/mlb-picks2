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

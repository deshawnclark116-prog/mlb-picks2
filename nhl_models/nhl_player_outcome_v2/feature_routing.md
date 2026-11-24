# NHL V2 feature routing (Phase0 plan, nothing built)

Causal chain: AVAILABILITY/LINEUP -> TEAM GAME ENVIRONMENT -> ICE TIME / LINE / PP ROLE -> OPPORTUNITY -> EFFICIENCY -> CENTRAL PROJECTION -> (later) UNCERTAINTY.

| layer | candidate inputs | pregame availability | status |
|---|---|---|---|
| availability / lineup | confirmed starter, backup, scratches, injuries | forward capture only (T24H..T2); historical BLOCKED_TIMING | ACCEPTED_FORWARD_ONLY |
| team game environment | opponent prior shots-for / shots-against / goals, home flag, days of rest, venue | strictly earlier games | ACCEPTED_HISTORICAL |
| ice time / role | prior-3/10 total TOI, prior EV and PP TOI, games started | prior-game box scores (V1 acquisition) | ACCEPTED_HISTORICAL for history; target-game PP unit BLOCKED_TIMING |
| opportunity | TOI x shot rate by game state (EV/PP) | derived | to build in Phase1 |
| efficiency | shrunk shooting pct, save pct (regressed) | derived | to build in Phase1 |
| central projection | count mean/median per stat | derived | Phase1 |
| derived thresholds | P(X>=k) from the count distribution | derived | Phase1 |

Forbidden routes: target-game score state, any betting-market line, price or derived probability, target-game TOI, postgame lines/PP relabelled as pregame, shootout events as skater stats.
Opponent features route mechanically through the TEAM GAME ENVIRONMENT layer only (damped, season-to-date league normalised). A feature enters only with mechanism, causal layer, pregame availability and out-of-time value.

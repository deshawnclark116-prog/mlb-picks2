# CFB Outcome Engine v1 — dependency graph (RESEARCH / SHADOW ONLY)

```
 sources (schedule, box scores, PBP, rosters, identity)  --Stage 1 audit-->  source registry
        |
 identity contract (provider ID only; unresolved => history does not cross schools)  --Stage 1B-->
        |
 point-in-time candidate universe (no target-game participation oracle)  --Stage 2-->
        |
 SHARED STATE (built once)  --Stage 3-->
   player state | team state | opponent state | role state | organizational-intent prior
        |
 team environment (strength, pace, pass/rush mix, scoring, FBS/FCS regime)  --Stage 4-->
        |
 availability / participation (historical participation prob; forward game-day info is forward-only)  --Stage 8-->
        |
 opportunity (team plays -> rush attempts / dropbacks -> shares)  --Stage 5/7-->
        |
 efficiency (per carry / per attempt / per target; TD & INT rates; hierarchical shrinkage)  --Stage 9-->
        |
 coherent joint simulator (availability -> plays -> volume -> QB -> shares -> opportunities -> events -> score)  --Stage 13-->
        |
 outcome heads: QB (att, comp, yds, TD, INT, rush) | RB (carries, yds, TD) | receivers (tgt, rec, yds, TD) | derived (anytime TD, fantasy) | team (plays, points, win prob)
        |
 append-only shadow forecast store + scheduler (T24 / T90 if timing supports)  --Stage 15-->  clean forward window (earliest 2026 Week 6)
```

Legacy models (rushing_yards / passing_touchdowns / anytime_touchdowns / moneyline = LEGACY_ACTIVE_COMPARATOR; passing_yards / receiving_yards = LEGACY_SUSPENDED) are **comparators only**; they feed nothing.

Head dependencies: QB passing yards/TD/INT <- team dropbacks, QB attempt share, completion & yards-per-completion efficiency; RB rushing <- team rush attempts, carry share, yards-per-carry; receiving <- team pass attempts, target share, catch / yards efficiency; anytime TD <- joint TD process; team points <- team plays + scoring rate; win probability <- team points distribution (derived, not a classifier).

Build order (evidence-driven, revisable after the source audit): identity -> universe -> shared state -> team volume -> shares / opportunity -> participation -> efficiency -> joint simulator.

## Status after the first execution milestone (development only)
| component | result |
|---|---|
| identity / candidate universe / shared state | COMPONENT_CANDIDATE |
| T1 team rush attempts | C1 (NB2 GLM on shared team state) retained over B0 |
| T2 team pass attempts | B0 retained; C1 FROZEN_REJECTED |
| P1 participation | C1 (logistic) retained over B0 |
| O1 carry share | B0 retained; C1 FROZEN_REJECTED (slice guard) |
| efficiency, TD / INT, QB / receiver shares, simulator, store | UNBUILT |

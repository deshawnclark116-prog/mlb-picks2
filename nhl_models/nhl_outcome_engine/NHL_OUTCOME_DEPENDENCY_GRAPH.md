# NHL outcome engine dependency graph

One shared NHL state engine feeding independently validated outcome heads. No head is a disconnected classifier; each consumes the shared upstream state where it is scientifically valid. Sportsbook lines / odds are never inputs.

```
pregame information (schedule, rosters evidence, prior completed games)  [information cutoff = scheduled start - horizon]
  -> player / team identity
  -> target-game membership evidence ------------------+
  -> availability / expected participation -------------+
  -> current-team role / deployment (EV/PP, TOI state) --+--> SHARED SKATER STATE
  -> all-team SKILL history (legal prior games only) ----+
  -> team pace / shot environment, opponent environment -+--> SHARED TEAM STATE
  -> goalie candidate / start state (BLOCKED historically; forward collector) --> GOALIE STATE
  -> schedule / venue / home-away

SHARED SKATER STATE --> SOG head (B2: HISTORICAL_CHAMPION; attempts extension in research)
SOG + shooting conversion + TOI/PP role --> GOALS head (UNBUILT)
team goal environment + role + teammate opportunity --> ASSISTS head (UNBUILT)
GOALS + ASSISTS (joint process, identity-checked) --> POINTS head (legacy FAILED; new head UNBUILT)
team shot environment + opponent environment + GOALIE STATE --> GOALIE WORKLOAD / SAVES head (legacy FAILED; new head UNBUILT, Phase G0)
team offence + opponent defence + goalie + home context --> TEAM GOAL distributions (UNBUILT)
home goal dist  x  away goal dist --> joint score --> regulation / OT / shootout --> WIN probability (moneyline; legacy comparator only)
```

## Status of every component / head (see NHL_OUTCOME_ENGINE_REGISTRY.json for evidence)
| item | status |
|---|---|
| player identity, team identity, schedule / venue, role / deployment, team + opponent shot environment | COMPONENT_CANDIDATE |
| target-game membership, availability / participation, TOI distribution state | RESEARCH |
| goalie candidate / start state | BLOCKED (historical PIT starter not reconstructable; forward collector measuring) |
| skater SOG | HISTORICAL_CHAMPION (Phase 1A B2 retained: B2_HISTORICAL_CHAMPION_RETAINED); attempt extension FROZEN_REJECTED |
| skater goals | RESEARCH — G1 hierarchical conversion selected; GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED (2025 mean-bias guard failed) |
| skater assists / team goals | UNBUILT |
| skater points | LEGACY_FAILED (new head UNBUILT) |
| goalie saves | LEGACY_FAILED (new head UNBUILT) |
| moneyline / win probability | LEGACY_STABLE_COMPARATOR (new head UNBUILT) |

## Build order (re-ranked by the feasibility evidence; causal / data dependency only)
Default was: shared state, SOG, goalie conditional, goals, assists, points, team goals, moneyline. **Changed:** goals / assists / points move ahead of the goalie conditional head.
1. shared skater state (DONE) 2. SOG (DONE: B2 retained) 3. goals (DONE, not established) 4. assists 5. points (derived from the joint goals / assists process) 6. goalie conditional-on-start workload / saves (after a G0 Q5 source adjudication; can run in parallel) 7. team goal distributions 8. moneyline (derived from team goals)

Why: goals / assists / points labels are fully reconciled and need only shared state + SOG; the opposing goalie is unknown pregame (start BLOCKED) so skater goals cannot consume goalie state; the goalie conditional head has an unresolved label adjudication and feeds team goals, not skater goals.

## Feasibility status (read the phase READMEs)
Goalie G0: conditional component RESEARCH (with limits), start component BLOCKED. Scoring S0: labels reconciled, POINTS == GOALS + ASSISTS exact. Team-game T0: derivation identifiable; independent Poisson understates regulation ties.

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
| skater goals / assists / team goals | UNBUILT |
| skater points | LEGACY_FAILED (new head UNBUILT) |
| goalie saves | LEGACY_FAILED (new head UNBUILT) |
| moneyline / win probability | LEGACY_STABLE_COMPARATOR (new head UNBUILT) |

## Default build order (to be re-ranked by feasibility evidence, by causal / data dependency only)
1. shared skater state 2. SOG 3. goalie conditional workload / saves 4. goals 5. assists 6. points 7. team goal distributions 8. moneyline

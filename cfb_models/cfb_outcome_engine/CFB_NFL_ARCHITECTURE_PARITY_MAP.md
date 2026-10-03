# NFL Phase 1 -> CFB Outcome Engine parity map

Architecture is ported, parameters and source assumptions are NOT. CFB modules never import NFL modules; shared football infrastructure is extracted only after both architectures are stable.

| NFL Phase 1 | CFB v1 | CFB-specific change |
|---|---|---|
| Protocol (nfl_player_outcome_phase1_protocol.json) | cfb_outcome_engine/phase_protocol (Stage 10, registered before performance) | same discipline: burned data labelled, clean-forward window, component gates |
| nfl_phase1_common.py (splits, metrics, distribution scoring) | cfb_phase1_common.py | research cutoff enforcement (Week 5 forbidden), splits, CRPS / NLL / PIT, week-block bootstrap |
| nfl_phase1_data.py (as-of contract, chronological replay) | cfb_phase1_data.py | as-of contract with CFB weekly granularity; target week excluded; source assumptions logged (no row-level timestamps in most CFB sources) |
| nfl_phase1_availability.py (P(active) T24/T90 + conditional snap share) | cfb_phase1_availability.py | participation probability from prior participation / roster / depth competition; NO snaps source -> participation = a box-score line; injuries BLOCKED_HISTORICAL_PIT (forward-only) |
| nfl_phase1_team_environment.py (plays, dropbacks, rush att, RZ; NB / GLM; B0 blend) | cfb_phase1_team_environment.py | team volume from rush attempts + pass attempts (plays proxy; no PBP in the frozen DB); strength via internal ratings; FBS/FCS regime; B0 = prior-season-shrunk team mean |
| nfl_phase1_role_state.py (carry / target / QB-att shares) | cfb_phase1_role_state.py | shares from player box lines; route / snap shares NOT available; returning / newcomer priors |
| nfl_phase1_opportunity.py (Dirichlet-multinomial allocation) | cfb_phase1_opportunity.py | same allocation coherence: team total ~ NB, active_i ~ Bernoulli, shares ~ Dirichlet, 'other' mass for non-candidates |
| nfl_phase1_efficiency.py / rushing / receiving / passing_efficiency.py | cfb_phase1_efficiency.py + rushing / receiving / passing | hierarchical shrinkage league -> position -> team -> player; NFL air-yard / YAC chains need PBP: CFB frozen data only has box totals -> yardage per opportunity from box + ESPN PBP only if the audit proves coverage |
| nfl_phase1_event_models.py (generic pmf / hazard runners) | cfb_phase1_event_models.py | TD / INT / catch hazards; explosive plays only if PBP coverage is proven |
| nfl_phase1c_sim.py (coherent play-level simulator) | cfb_phase1_sim.py | Stage 13, only after components validate; CDF / Wasserstein stability for lattice outcomes (not the NFL quantile R11 rule) |
| nfl_phase1_forecast.py (cutoff, snapshot binding, simulate, store) | cfb_phase1_forecast.py | Stage 15 |
| nfl_phase1_store.py (append-only batch store, hard error on same ID / different bytes) | cfb_phase1_store.py | Stage 15; independent re-implementation, no NFL import |
| nfl_phase1_snapshots.py (content-addressed snapshots at each horizon) | cfb_phase1_snapshots.py | Stage 15; forward-only roster / depth / injury evidence |
| nfl_phase1e_scheduler.py (tick per (game, horizon), dispatch ledger) | cfb_phase1e_scheduler.py (later) | Stage 15-16; weekly Saturday slate with T24 / T90; no deploy before the architecture is frozen |
| nfl_phase1_evaluate.py (development evaluation, component tables) | cfb_phase1_evaluate.py | Stage 12 |

**Structural differences that change design:** (1) CFB has weekly Saturday slates and huge roster churn (transfers, freshmen) -> identity audit and organizational-intent priors; (2) no snap / route / target source in the frozen data (player_games has no targets column) -> participation and share definitions use box-score lines; (3) FBS vs FCS regime and weak opponent data; (4) no point-in-time injury or depth chart history; (5) one season ~ 12-15 games -> aggressive hierarchical shrinkage.

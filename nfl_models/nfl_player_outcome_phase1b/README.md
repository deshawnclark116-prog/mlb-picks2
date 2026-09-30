# NFL player-outcome engine — Phase 1B (efficiency, matchup, explosive-play and event layer) + Phase 1A hardening

**Status: SHADOW / DEVELOPMENT. Nothing is frozen, production is untouched, no API changed, no champion replaced.**
All scores below are on **burned development data (2025 + 2026 wk1-3)** — never a holdout. Training = 2023 + 2024 wk1-12, validation = 2024 wk13-18
(hyper-parameters and feature-level selection use validation only; development is reported for every candidate). No clean-forward data was used.
No `nfl_player_outcome_phase1_freeze.json` exists or was written.

Files: `report.json` (everything), `selected_architecture.json`, `rejected_candidates.json`, `scheme_ablation.json`, `interaction_ablation.json`,
`tail_calibration.json`, `persistence_diagnostics.json`, `archetype_tests.json`, `data_source_audit.json`, `end_to_end_development.json`,
`phase1a_hardening_report.json`, `phase1a_hardening_selected.json`, `hardening_raw/`.
Code: `nfl_phase1b_data.py` (as-of records; reuses `nfl_context_v4.PlayData.def_profile` — no scheme parser was duplicated), `nfl_phase1_efficiency.py`
(engine), `nfl_phase1_rushing_efficiency.py`, `nfl_phase1_receiving_efficiency.py`, `nfl_phase1_passing_efficiency.py`, `nfl_phase1_event_models.py`
(hazards / pmfs / TD), `nfl_phase1_defense_events.py`, `nfl_phase1b_evaluate.py`, `nfl_phase1_hardening.py`, `nfl_phase1_hardening_report.py`,
`nfl_phase1_snapshots.py`; tests: `tests/test_nfl_phase1b_invariants.py`.

## 0. Phase 1A hardening

**A/B. Injury timing and forward snapshots.** Historical injury features are now described everywhere as a *historical final-weekly-report proxy evaluated under the
documented nflverse timing assumption (A2)*, not as proven T-24h data. For clean-forward, `nfl_phase1_snapshots.py` downloads the source at T-24h and, independently, at T-90m,
sha256-hashes it, records the retrieval timestamp, stores an immutable read-only file (`open(..., "xb")`, never overwritten), appends to an append-only manifest, refuses any
retrieval after the forecast time, and serves forecasts only from the saved, hash-re-verified snapshot. Protocol amendments E (availability model = gradient boosting, matching the
selected Phase 1A architecture; a test asserts consistency) and F (snapshot procedure) were added. Tests: 5 hardening tests.

**C. Opportunity-interval calibration** (validation-fit, development-scored; rule fixed in advance: p<0.10, CRPS gain >= 0.5%, PIT coverage within 0.03, MAE not worse).
The "80% carry interval covers 91%" is a **discreteness artifact, not miscalibration**: the forecast's own implied 10-90 coverage is 90.5%; the randomized PIT covers
80.6% of the 0.1-0.9 band and 50.4% of the 0.25-0.75 band. Candidates compared: as-is NB/Dirichlet-multinomial, calibration-aware alpha, PIT recalibration, empirical residual pmf,
temperature, zero-mass map (beta-binomial / zero-inflated forms are subsumed by the zero-mass and empirical-residual maps; not fitted separately). Carry: nothing beats as-is
(best alternative CRPS 1.3797 vs 1.3779). qb_att, rz_carry, rz_target, target, route: no candidate passes. **def_snap: PIT recalibration adopted for that count distribution**
(CRPS 7.412 -> 7.240, p=0.000, PIT coverage 0.845 -> 0.789 @80%, MAE 9.947 -> 9.885) — but it made downstream tackle CRPS worse end-to-end (0.9737 -> 0.9767, p 0.99), so the Phase 1B
assembly keeps the raw samples (unresolved conflict, see U).

**D. Named-player rushing mass** (forecast 0.909 vs actual 0.940; 638 dev team-games): sampled availability is not the cause (oracle availability changes it by 0.0004).
Decomposition of the 0.0204 gap: the fixed outside bucket (train mean 0.0793 vs dev actual outside 0.070; 2026 wk1-3 RB/FB outside carriers 11.9%), normalization behaviour when
starters are out, propensity under-sum (-0.0131), and universe omission (RB/FB outside carriers 4.4% of carries; 3.3% by players with >=3 prior games not in the candidate window,
1.4% debuts, 0.3% eligibility filter; WR/TE/other 3.4% is outside by design). Fix tests: outside-bucket policies (proportional, rolling 64/128/256/512) on the accepted universe:
**none passes** (best carry variant is still 0.0013 CRPS worse than the fixed bucket; qb_att rolling-512 is 0.0045 better but p=0.118; target none); on a depth-chart-extended universe rolling buckets pass for carry/target/qb_att
(carry CRPS 0.0068 better, p=0.000) but that universe is not comparable and did not close the forecast-share gap (actual share 0.940 -> 0.967, forecast 0.909 -> 0.913).
**Decision: no change to Phase 1A** (see `phase1a_hardening_report.json`).

## 1. Architecture (all components)
Per opportunity: hierarchical shrunk pmf/rate (league -> position -> player; recency decay and kappa chosen on validation) x exponential-tilt / offset-logistic / offset-Poisson adjustment
over nested as-of families **B0 hierarchical only, B1 +team offense, B2 +opponent allowed rates, B3 +defensive scheme (v4 profile: box, stacked, blitz, man/zone, C0-C6, pressure),
B4 +personnel absences (injury-report OL/DL/LB/DB counts), B5 +player x matchup interactions**. Level = highest level whose train-fit validation log score improves the previous kept level by
>= 1e-4 nats/opportunity. No XGBoost importance is used as evidence. Bins: rush -10..99 (negative lump + singles to 29 + tail bins), air, YAC. Phase 1A samples are consumed as-is (asserted by test).

## 2. Results (development; log score in nats per opportunity, lower is better)
**Rushing (per carry).** league pmf 2.8179, position pmf 2.8081, hierarchical B0 2.8080 (tuned kappa_player = infinity: **player history received zero weight**).
Two-process (ordinary < T + hazard + tail) vs single-process: validation log scores single 2.83938, T15 2.83902, T20 2.83915, T25 2.83931 — differences <= 0.0004 nats, **no threshold or structure is preferred**;
selected by rule: two-process T=15 at B3. Development: B0 2.80801 -> B2 2.80621 (p=0.001, CRPS 2.7029 -> 2.6999, 0.11%); B3 2.80666, B4 2.80678, B5 2.80681. Scheme family (B3 vs B2) does not help on development
(leave-scheme-out p=0.67). Tail bins (dev, B0): negative pred 0.0880 obs 0.0842; 1-4 0.464/0.471; 10-19 0.091/0.088; 20+ 0.0243/0.0229.
**Receiving chain (per target).** air yards: hierarchical (kappa 160, decay 0.95) 3.2420 vs position 3.2572 (player persistence +0.0152 nats, p=0); matchup levels add nothing (B1-B4 p 0.43-0.77; B5 interaction +0.0013, p=0.0145).
catch | air bucket: B0 0.58923 (position 0.5897, league 0.5916; player weight kappa 80; persistence p=0.128); opponent completion allowed is the only family that helps (leave-opp-out p=0.0145); scheme no.
YAC | catch: hierarchical B0 2.6125 (kappa_player = infinity: no player YAC skill detected); every added family <= 0.0001 nats. Composed per-target outcome (incomplete / <10 / 10-19 / 20+): league lookup 1.2770 -> selected 1.2475 (p=0.000).
**Passing chain.** sack B0 0.24917 (B5 0.24883, p=0.0005 but 0.0003 nats); interception B0 0.10477 (nothing significant after B0; B5 p=0.07); pass TD B0 0.19058; completion B0 0.61724 (QB persistence +0.00096, p=0.000; opponent completion allowed helps, p=0.0005);
air yards: matchup B2 3.40789 vs B0 3.4089 (p=0.000, +0.001); YAC: no player weight, no family helps. Composed attempt outcome 1.2705 -> 1.2674.
**TD conversion (red zone / outside strata).** naive player-specific (kappa 5) is WORSE than the position average: rush TD 0.1148 vs 0.1124 (tuned shrinkage 0.1115, persistence p=0.001); receiving TD 0.1543 vs 0.1489 (tuned 0.1489). No reputation priors. Matchup families: rush TD B2 p=0.0435 (0.0001 nats), receiving TD B4 p=0.076.
**Defensive events (per snap; Poisson given snaps).** tackles 2.0224 (league) -> 1.9562 (position) -> 1.8812 (player, kappa 60 snaps); families add nothing on development (B4 1.88142, B5 1.88175, both worse than B0); sacks 0.3771 -> 0.3404 -> 0.3229 (B3 0.32191, best, p=0.000 vs league); interceptions 0.1451 -> 0.1356 -> 0.1345 (B2 0.1343).
Tail: tackles P(>=6) pred 0.1799 obs 0.1810; sacks P(>=1) pred 0.1073 obs 0.0941 (over-predicts); INT P(>=1) 0.0344 obs 0.0346.

## 3. Defensive-scheme ablation, interaction, archetypes, persistence
* **Scheme (v4 profile, placed at the causal component: rush ypc/explosive, catch, air, YAC, sack, INT, TD, defender events):** leave-scheme-out from B4 never improves a component on development beyond noise except the QB sack hazard
  (leave-scheme-out p=0.0145) and rec TD (p=0.055); for rush, catch, YAC, air the estimate is *negative* (scheme hurts by 0.0002-0.0004 nats). Coverage-type labels do not exist for 2026, so those rates are stale from 2025.
* **Scheme x player interaction (B5 vs B4):** helps only rec air yards (+0.0013, p=0.0145) and sacks (+0.0004, p=0.0005 vs B0); neutral or worse elsewhere. Not enough to justify a general interaction layer.
* **Archetypes** (k-means k=3,5 on as-of traits, TRAIN-fit; rule: validation log score +>=1e-4 nats AND development p<0.10 with >=1e-3 nats): rushing — none (k3 valid 2.83936 vs reference 2.83938; k5 worse); receiving air yards k5 **passes the component rule**
  (validation +0.00136 nats; development +0.0039 nats, p=0.000, CRPS 0.45% better; k3 fails validation). **Downstream check fails:** with the k5 air-yard pmf the end-to-end receiving-yards CRPS is 7.8191 vs 7.8100 without (p=0.99 not better); receptions 0.6217 vs 0.6221 (p=0.047).
  The headline assembly therefore does NOT use archetypes; the conflict (better air pmf, worse yards CRPS) is listed under U.
* **Player-skill persistence** (hierarchical B0 vs position-only): detected for receiving air yards, QB completion, rush TD, defender tackles/sacks/INT; **not detected** for rush yards per carry, YAC (receiving and passing), QB air yards, sack avoidance (p=0.31), interception (p=0.39), pass TD (p=0.11), receiving TD (p=0.66), catch (p=0.128). Where tuned kappa_player = infinity the B0-vs-position difference reflects position->league shrinkage only.

## 4. End-to-end DEVELOPMENT assembly (Phase 1A opportunity samples x Phase 1B per-opportunity distributions; 200 CRN draws)
Universe = Phase 1A scored (pregame) candidates, non-participants score 0. CRPS (lower better); baseline = last-8-team-game empirical distribution of the player's stat.
| stat | n | CRPS selected | CRPS last-8 | gain vs last-8 | p | simple eff. CRPS | hier B0 CRPS |
|---|---|---|---|---|---|---|---|
| rushing yards (RB, QB) | 3951 | 7.934 | 9.596 | 1.663 | 0.000 | 7.953 | 7.953 |
| QB rushing yards | 736 | 8.203 | 8.801 | 0.598 | 0.005 | 8.223 | 8.224 |
| rush TD | 3951 | 0.1088 | 0.1205 | 0.0117 | 0.000 | 0.1093 | 0.1088 |
| receiving yards | 8872 | 7.810 | 9.523 | 1.713 | 0.000 | 7.865 | 7.827 |
| receptions | 8872 | 0.6221 | 0.7703 | 0.1483 | 0.000 | 0.6252 | 0.6220 |
| receiving TD | 8872 | 0.0865 | 0.0983 | 0.0118 | 0.000 | 0.0868 | 0.0868 |
| passing yards | 1397 | 30.25 | 41.99 | 11.74 | 0.000 | 30.88 | 30.56 |
| pass TD | 1397 | 0.3339 | 0.4002 | 0.0664 | 0.000 | 0.3383 | 0.3339 |
| interceptions | 1397 | 0.2009 | 0.2486 | 0.0477 | 0.000 | 0.1995 | 0.2009 |
| tackles | 15495 | 0.9737 | 1.2282 | 0.2545 | 0.000 | 1.0198 | 0.9734 |
| sacks | 15495 | 0.0763 | 0.0847 | 0.0085 | 0.000 | 0.0801 | 0.0766 |
| defensive INT | 15495 | 0.0258 | 0.0288 | 0.0030 | 0.000 | 0.0260 | 0.0258 |
| anytime TD (rush+rec) | 10241 | 0.0917 | 0.1032 | 0.0115 | 0.000 | 0.0921 | 0.0920 |

The selected Phase 1B chain beats the crude last-8 baseline everywhere, but that baseline is weak (no availability, no opportunity model); most of the gain comes from Phase 1A opportunities. The
increment of Phase 1B efficiency over a *position-average* efficiency (simple) is small but significant for rushing yards (-0.019, p=0.012), receiving yards (-0.055), passing yards (-0.63), tackles (-0.046); for interceptions the simple
model is better (0.1995 vs 0.2009, p=0.73). QB rush yards 80% interval coverage is 0.757 (under-covered); rushing 0.869, receiving 0.906, passing 0.904. **Comparators not reproduced:** an incumbent distributional forecast and the
Phase 0B decomposition are not pregame distribution forecasters in this repository (see `end_to_end_development.json`).

## 5. Approximations and limits
Within-bin yard values use bin means/midpoints (coarse bins above 25-30 yards). Per-opportunity draws are independent across players and independent of the TD draw; red-zone counts are `min(rz sample, total sample)` because Phase 1A samples types independently.
Kneels and non-player plays are excluded from tallies. Sacks are evaluated as a hazard only (not part of the assembly outputs). The injury-report personnel family measures reported absences, not OL quality (see `data_source_audit.json`: OL depth-chart starters exist only for 2025+, nothing out-of-time to fit).
Week-level as-of rule (a Sunday game cannot see a same-week Thursday game). 2026 coverage/pressure labels are stale.
15 receiving rows lacked a record (targets by players outside RB/FB/WR/TE); the extended depth-chart universe was tested and not adopted.

## 6. Selected Phase 1B architecture (development-selected; not frozen)
See `selected_architecture.json`. Rushing: two-process T=15 at B3 (structures statistically tied; player weight 0). Receiving: air yards hierarchical B3 (player kappa 160, decay 0.95); catch B4 (kappa 80); YAC B0 (position). Passing: sack B0, INT B0, TD B0, completion B4, air B3, YAC B1.
TD: rush B0 (kappa 160), receiving B1 (kappa 160). Defence: tackles B5 by validation (development prefers B0/B1), sacks B4, INT B2; per-snap Poisson.

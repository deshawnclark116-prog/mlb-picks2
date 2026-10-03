# NFL Player Outcome Engine, Phase 1A: opportunity forecasting (SHADOW, development results)

Shadow research only. Production serving, champions, artifacts and the API are untouched. No sportsbook
number is read anywhere (enforced by tests). **Every score below is DEVELOPMENT** (train 2023 + 2024 wk1-12,
validation 2024 wk13-18, development 2025 + 2026 wk1-3). Those weeks are burned; none of this is a holdout result.
The clean-forward window has not started, and the final Phase 1 freeze record does not exist.

Files: `report.json` (all numbers), `component_tables.md` (tables generated from the report),
`selected_architecture.json`, `rejected_candidates.json`; code: `nfl_phase1_{data,common,availability,team_environment,
role_state,opportunity,evaluate}.py`; tests: `tests/test_nfl_phase1_invariants.py`, `tests/_determinism_probe.py`;
protocol: `nfl_models/nfl_player_outcome_phase1_protocol.json` (v1.1, with a `phase1a_decisions` block).

```
python -u nfl_phase1_evaluate.py --data-dir /tmp/nfl_data                       # ~15 min (role-state is the slow part)
python -u tests/test_nfl_phase1_invariants.py --data-dir /tmp/nfl_data --with-run
```

## Data-time contract (`nfl_phase1_data.py`)

Every input carries source, season/week/game, information-available timestamp, retrieval time and the forecast
cutoff (T-24h = kickoff - 24h; T-90m = kickoff - 90 minutes). Assumptions (`AS_OF_ASSUMPTIONS`):

| Source | Timestamp treatment |
|---|---|
| stats, snaps, play-by-play | available from kickoff + 24h (A1) |
| injuries | **no timestamp in nflverse**; assumed published 16:00 ET two days before kickoff day (A2). The historical injury features are a **final-weekly-report proxy evaluated under this documented nflverse timing assumption**, not proven T-24h information; rows may absorb later supplemental changes. Clean-forward uses hashed immutable snapshots taken at T-24h and, independently, T-90m |
| weekly roster status | game-day snapshot (Phase 0B: 0.0% of 5,395 INA players had snaps); T-90m only; T-24h uses the previous week's status (A3) |
| depth charts 2025-26 | `dt` snapshot; latest with `dt <= cutoff` (A4) |
| depth charts 2024 | week-labelled, publication semantics unknown: **never used** (A5) |
| draft / rookie year, schedule | known in advance (A6, A7); no sportsbook column read |

## Results (development). Full tables in `component_tables.md`

### E. Availability (P(active); baseline = status lookup)

| | base logloss / Brier / ECE | selected (gradient boosting) | 2025 ll base -> sel | 2026 wk1-3 ll base -> sel |
|---|---|---|---|---|
| offense T-24h | 0.5749 / 0.1957 / 0.0308 | 0.2941 / 0.0898 / 0.0140 | 0.5453 -> 0.2885 | 0.7274 -> 0.3229 |
| offense T-90m | 0.3082 / 0.0898 / 0.0103 | 0.1715 / 0.0521 / 0.0122 | 0.3036 -> 0.1675 | 0.3316 -> 0.1923 |
| defense T-24h | 0.5529 / 0.1854 / 0.0358 | 0.3490 / 0.1072 / 0.0277 | 0.5152 -> 0.3201 | 0.7472 -> 0.4974 |
| defense T-90m | 0.2912 / 0.0823 / 0.0101 | 0.2103 / 0.0603 / 0.0137 | 0.2801 -> 0.1928 | 0.3485 -> 0.3009 |

Logistic regression was also compared and rejected (offense T-24h logloss 0.3314 vs 0.2941 for boosting). Week-block bootstrap p < 0.001 for all four.
Limits:
- **ECE is slightly worse than the baseline at T-90m** (0.0122 vs 0.0103 offense; 0.0137 vs 0.0101 defense), although logloss and Brier are much better.
- **The offense T-24h calibration curve overpredicts in the middle bins** (predicted 0.35 / 0.45 / 0.55 / 0.65 vs observed 0.28 / 0.40 / 0.50 / 0.61).
- **Small samples:** offense T-24h Doubtful (n=35) has mean forecast 0.137 vs observed 0.0; Questionable (n=319) 0.634 vs 0.608.
- **By position (offense T-24h):**

| Position | mean forecast vs observed |
|---|---|
| QB | 0.542 vs 0.513 |
| RB | 0.696 vs 0.687 |
| TE | 0.765 vs 0.765 |
| WR | 0.724 vs 0.710 |

Depth-chart rank, trained inside 2025 only (train 2025 wk1-6, early stop wk7-9, evaluate the rest; n=5,975): logloss 0.2918 -> 0.2587, p=0.0 (offense).

**Three-state label not built.** No source labels "limited", and any snap-share cutoff would be arbitrary. Modelled as P(active) plus a
separate conditional snap-share distribution: offense median MAE 0.1349 -> 0.1177 (quantile score 0.04942 -> 0.04218),
defense 0.1441 -> 0.1244 (0.05295 -> 0.04752), p=0.0; 80% band coverage offense 0.772, defense 0.888 (too wide).

### F. Team environment (negative-binomial distributions; baseline = shrunk current/prior average)

Selected where the candidate beat the baseline on MAE and CRPS at p < 0.10 (development, pooled 2025 + 2026 wk1-3):

| target | selected | base -> sel MAE | base -> sel CRPS | p (MAE / CRPS) |
|---|---|---|---|---|
| plays | Poisson GLM | 6.842 -> 6.746 | 4.863 -> 4.787 | 0.071 / 0.062 |
| rushes | Poisson GLM | 5.633 -> 5.497 | 3.945 -> 3.860 | 0.0015 / 0.0015 |
| designed rushes | Poisson GLM | 5.546 -> 5.337 | 3.899 -> 3.797 | 0.0005 / 0.001 |
| targets | opponent-adjusted blend | 5.924 -> 5.850 | 4.155 -> 4.118 | 0.034 / 0.080 |
| red-zone plays | Poisson GLM | 3.997 -> 3.906 | 2.809 -> 2.757 | 0.038 / 0.027 |
| red-zone rushes | Poisson GLM | 2.480 -> 2.429 | 1.741 -> 1.705 | 0.062 / 0.029 |
| red-zone targets | Poisson GLM | 2.280 -> 2.220 | 1.597 -> 1.567 | 0.009 / 0.018 |
| defensive snaps | Poisson GLM | 11.934 -> 11.745 | 8.539 -> 8.342 | 0.080 / 0.011 |
| **dropbacks, goal-line plays, goal-line rushes** | **blend kept** (nothing beat it) | | | |

Gradient boosting (Poisson objective) lost to the GLM or the blend everywhere it was tried. **The gains are small
(1-4%) and come mainly from 2025:** in the 2026 wk1-3 subsample (about 94 team-games) rushes (5.101 -> 5.150), designed
rushes (5.122 -> 5.140) and targets (5.545 -> 5.716) were slightly *worse* than the blend. The pooled p-values are
dominated by 2025. Treat these selections as weakly supported.

### G. Role-state share models (next-game share given playing; MAE; baseline R0 = last-8 mean)

| share | n | R0 last-8 | R1 EWMA | R2 Kalman | R3 changepoint | R4 HMM | R5 learned | R5 vs R0 (p) |
|---|---|---|---|---|---|---|---|---|
| carry (RB+QB) | 2,491 | 0.0927 | 0.0876 | 0.0874 | 0.0919 | 0.1047 | **0.0856** | 0.0 |
| RB carry | 1,764 | 0.1055 | 0.0989 | 0.0984 | 0.1080 | 0.1152 | **0.0964** | 0.0 |
| QB rush | 727 | 0.0617 | 0.0610 | 0.0605 | 0.0620 | 0.0656 | **0.0585** | 0.0135 |
| target | 6,402 | 0.0479 | 0.0471 | 0.0478 | 0.0497 | 0.0551 | **0.0456** | 0.0 |
| route (2025 truth only) | 5,534 | 0.1349 | 0.1225 | 0.1249 | 0.1346 | 0.1391 | **0.1190** | 0.0 |
| QB attempt share | 727 | 0.1677 | 0.1540 | 0.1554 | 0.1691 | 0.1686 | **0.1237** | 0.0 |
| RZ carry | 2,316 | 0.1722 | 0.1698 | 0.1757 | 0.1895 | 0.2048 | **0.1618** | 0.0 |
| RZ target | 6,008 | 0.1112 | 0.1107 | 0.1119 | 0.1183 | 0.1290 | **0.0904** | 0.0 |
| defensive snap share | 11,031 | 0.1453 | 0.1299 | 0.1306 | 0.1442 | 0.1469 | **0.1243** | 0.0 |

- **Rejected formulations:** the changepoint (R3) and HMM (R4) models are worse than last-8 on nearly every type; they are not built into Phase 1A.
- **Filters:** the simple Kalman/EWMA filters beat last-8 by a modest 3-9%.
- **The learned model wins on every type.** For RZ carry / RZ target, part of the MAE gain comes from a median objective predicting near zero on sparse shares, which is why the opportunity layer uses its own mean-objective propensity models.
- **P(expand / stable / contract).** The learned model beats the class-frequency baseline in 3-class logloss only for target (0.9122 -> 0.9042), route (1.0239 -> 0.9629), QB rush (1.0305 -> 1.0052) and defensive snaps (1.0355 -> 1.0092); the Kalman filter is used for RB carry (1.0874 vs 1.0908 class frequency, marginal; the learned model is 1.0913). **For carry, QB attempts, RZ carry and RZ target no model beats the class frequency, so the engine ships no role-change probability there.** (Point estimates; not bootstrapped.)
- **Regimes (share MAE, last-8 -> learned, RB carry):**

| Regime | n | last-8 -> learned |
|---|---|---|
| teammate injury (vacated >= 0.15) | 96 | 0.1556 -> 0.1217 |
| depth demotion | 69 | 0.1312 -> 0.0943 |
| abrupt contraction | 291 | 0.1433 -> 0.1172 |
| abrupt expansion | 283 | 0.1459 -> 0.1304 |
| depth promotion | 118 | 0.1223 -> 0.1147 (Kalman 0.1101 is best there) |
| player returning | 131 | 0.1000 -> 0.0969 |
| rookie | 287 | 0.1143 -> 0.1101 |

  The learned model is best or near best in every change regime for targets, routes, QB attempts and defensive snaps
  (tables in `report.json`); the smallest regimes (team change n=17-27, QB depth promotion n=15) are too small to interpret.

### H. Organizational-intent and other family ablations (leave-one-family-out, week-block bootstrap)

| Family | Survives (p < 0.10, contribution > 0) | Size of the effect |
|---|---|---|
| own availability | carry, QB rush, target, route, QB att, RZ carry, def snap (not RB carry p=0.155, not RZ target p=0.48) | 0.0136 on QB attempts; 0.0006-0.0022 elsewhere |
| teammate absence | carry, RB carry, target, route, RZ target | 0.0002-0.0015 |
| organizational intent (draft round/pick, experience, team change) | **target only** (0.00018, p=0.0) | fails for carry (p=0.131), RB carry, QB rush, route, QB att, RZ carry/target, def snap |
| opponent and team pass-rate style | **RZ target only** (0.00039, p=0.049) | fails everywhere else |
| role-state filter features (Kalman/changepoint/HMM outputs) | carry, RB carry, target, def snap (marginal, p=0.059); not route (p=0.19), RZ target (p=0.13), QB att, QB rush, RZ carry | 0.0002-0.0010 |
| **depth-chart rank** (2025+ split, own ablation) | carry (p=0.019), RB carry (0.035), QB att (0.0095), RZ carry (0.026); marginal for QB rush (0.054) and target (0.093); **not** route (0.30) or RZ target (0.21) | 0.0001-0.0049 MAE gain |

**Organizational priors barely help.** They carry a real but tiny signal for target share (0.0002 MAE) and nothing for the other types; current usage plus
availability dominates. Depth-chart rank helps most where roles are set by the coaches (carries, QB attempts). Depth snapshots exist only from 2025, so the family
cannot be trained inside the 2023-24 development split; it enters the walk-forward fit on 2025+ rows. Families with p in 0.05-0.15 flipped
between two runs that differed only in row order, so those decisions are weak evidence.

### I-M. Opportunity allocation (player counts, **pregame universe, non-participants = 0**; chain vs dumb baseline)

Baseline = NB(team blend x last-8 share x status-lookup P(active)) with a per-stat dispersion fit on train. Chain = P(active) x role propensity x
team-coherent multinomial (Dirichlet concentration tuned on validation).

| type | n | base -> sel MAE | base -> sel CRPS | base -> sel 80% coverage | p(MAE) / p(CRPS) | components kept |
|---|---|---|---|---|---|---|
| carries | 3,951 | 2.607 -> 1.895 | 1.819 -> 1.378 | 0.847 -> 0.910 | 0.0 / 0.0 | coherence, role, availability, dispersion |
| targets | 8,887 | 1.356 -> 1.155 | 0.963 -> 0.828 | 0.889 -> 0.928 | 0.0 / 0.0 | + team model |
| QB attempts | 1,397 | 11.135 -> 5.189 | 7.702 -> 3.902 | 0.525 -> 0.891 | 0.0 / 0.0 | coherence, role, availability, dispersion |
| RZ carries | 3,934 | 0.622 -> 0.571 | 0.459 -> 0.418 | 0.921 -> 0.941 | 0.0 / 0.0 | coherence, role, availability, dispersion |
| RZ targets | 8,861 | 0.288 -> 0.285 | 0.223 -> 0.213 | 0.928 -> 0.954 | **0.133** / 0.0 | coherence, role, availability |
| routes (2025 only) | 7,447 | 6.299 -> 4.911 | 4.753 -> 3.586 | 0.805 -> 0.857 | 0.0 / 0.0 | role, availability, dispersion |
| defensive snaps | 15,495 | 13.829 -> 9.958 | 9.999 -> 7.412 | 0.875 -> 0.907 | 0.0 / 0.0 | role, availability, team model, dispersion |

Per-period CRPS (2025 / 2026 wk1-3, base -> sel): carries 1.829 -> 1.377 / 1.772 -> 1.383; targets 0.966 -> 0.826 / 0.944 -> 0.836; QB attempts 7.718 -> 3.944 / 7.615 -> 3.676.
Both periods move the same way.

**Leave-one-component-out (CRPS gain of the full chain over the chain without the component):**

| type | coherence | role model | availability | team model | dispersion |
|---|---|---|---|---|---|
| carries | +0.104 | +0.026 | **+0.315** | +0.0001 (p=0.50) | +0.018 |
| targets | +0.025 | +0.025 | **+0.092** | +0.002 (p=0.044) | +0.005 |
| QB attempts | **+1.084** | +0.114 | **+2.044** | 0.000 (p=1.0) | +0.070 |
| RZ carries | +0.009 | +0.006 | +0.027 | +0.001 (p=0.27) | +0.002 |
| RZ targets | +0.0015 | +0.007 | +0.009 | +0.0001 | +0.0001 (p=0.37) |
| routes | n/a | +0.254 | **+0.656** | 0.000 | +0.212 |
| defensive snaps | n/a | +0.264 | **+1.494** | +0.049 | +0.318 |

- **Availability is the dominant component everywhere.** Modelling P(active) is worth more than every other component combined for carries, QB attempts, routes and defensive snaps.
- **Team coherence matters most for QB attempts:** without it, one team's QBs can jointly overshoot the dropback total. That is the "137%" problem, quantified.
- **The team-volume model earns its place only for targets (p=0.044) and defensive snaps (p=0.006).** For everything else the blend stays inside the allocation.
- **RZ targets:** the interval improves (CRPS 0.223 -> 0.213) but the median forecast is not significantly better than baseline (MAE p=0.133). Treat as marginal.

**Robustness check on the protocol's eligible universe** (players meeting the prior-usage rules; removes the easy zeros of bench candidates):
carries n=1,184 MAE 5.213 -> 3.820, CRPS 3.579 -> 2.710; targets n=2,998 MAE 2.368 -> 1.998, CRPS 1.641 -> 1.414; QB attempts n=873 MAE 12.952 -> 6.828, CRPS 8.752 -> 4.984. All p=0.0.
17-31% of eligible players still had zero actual opportunity, so availability is not a tail issue.

**Coherence / calibration caveats:**
- Named + outside-bucket = team total exactly in every simulated sample (asserted).
- The *forecast* share of the team total credited to named candidates is lower than reality for carries (0.909 vs 0.940 actual), QB attempts (0.790 vs 0.844) and RZ carries (0.902 vs 0.973): a fixed outside-candidate weight absorbs too much when named players are out.
- The 10-90% bands are conservative for discrete counts (carries 0.910 vs the 0.80 nominal). They are not calibrated to 80%.

**Regimes (baseline -> selected, MAE / CRPS):**

| Regime | Carries | Targets | QB attempts |
|---|---|---|---|
| top-share teammate Out/Doubtful | 3.597 -> 2.597 / 2.638 -> 1.907 (n=124) | 1.633 -> 1.248 / 1.168 -> 0.889 (n=494) | 12.671 -> 8.148 / 9.410 -> 6.361 (n=44) |
| teammate injury (vacated >= 0.15) | 3.389 -> 2.491 (n=351) | 1.583 -> 1.281 (n=1,310) | 12.797 -> 7.949 (n=59) |
| depth promotion / demotion | 2.895 -> 2.432 / 2.353 -> 1.929 | 1.406 -> 1.307 / 1.205 -> 1.059 | n/a (n=15 / 37) |
| abrupt expansion / contraction | 4.477 -> 2.876 / 3.955 -> 2.792 | 2.257 -> 1.830 / 2.361 -> 1.971 | 12.462 -> 5.875 / 10.743 -> 3.133 |
| rookie | 3.226 -> 2.435 | 1.240 -> 1.048 | 10.547 -> 6.313 |

Caveat: **"player returning" and "team change" rows show very large gains** (carries 1.655 -> 0.560; QB attempts 9.877 -> 2.438). Those rows are mostly players who stay out,
so the gain is the availability model correctly predicting zero, not better share estimation. For RZ targets the starter-out and teammate-injury regimes show
*no* MAE improvement (0.268 -> 0.274; 0.300 -> 0.302), only a CRPS improvement.

### J. Route and target paths

Two receiving paths are forecast: **structural** (dropbacks -> route share -> targets per route -> targets) and **fallback** (team targets -> target allocation).
On 2025 development (n=7,447; 2026 target-game routes unavailable): target CRPS structural 0.8749 vs allocation 0.8259; MAE 1.174 vs 1.146; p(structural not better) = 0.986 (MAE), 1.0 (CRPS).
**The official forward receiving forecast is the allocation path.** The structural path is logged in parallel and never swaps in. Route forecasts (MAE 6.299 -> 4.911 on 2025) are logged, but
route-proxy truth for 2026 does not exist, so routes are not a promotion gate.

### K. QB attempts

The coherent chain cuts QB attempts MAE from 11.135 to 5.189 (CRPS 7.702 -> 3.902); on protocol-eligible QBs 12.952 -> 6.828. Availability (+2.044 CRPS) and coherence (+1.084) do almost all of it.
QB attempt share alone (learned model) improved MAE 0.1677 -> 0.1237. No role-change probability ships for QB attempts (no model beats class frequency).

### L. Red-zone allocation

RZ carries improve on MAE and CRPS (0.622 -> 0.571; 0.459 -> 0.418). RZ targets improve CRPS only. Role-state RZ share MAE 0.1112 -> 0.0904 (target), 0.1722 -> 0.1618 (carry), with the median-objective caveat above.

### M. Defensive snaps

Baseline 13.829 -> 9.958 MAE (CRPS 9.999 -> 7.412), p=0.0, both periods (2025 CRPS 9.730 -> 7.251; 2026 wk1-3 11.385 -> 8.240). The forecast mean is biased high (+3.2 snaps).
Defensive snap shares are non-compositional (about 11 players on the field), so reconciliation is reported, not a sum-to-one constraint: forecast 10.36 named snap-share units per team vs 9.26 actual.

## Tests (38/38 pass)

20 unit / synthetic checks plus 18 data invariants (`tests/test_nfl_phase1_invariants.py --with-run`):
- **As-of contract:** no post-cutoff source timestamp on 35,597 player rows (T-24h and T-90m); every forecast cutoff precedes kickoff; depth snapshot `dt <= cutoff`; every history entry completed >= 24h before the cutoff; 2024 depth charts unused; forecast records timestamped before kickoff (0 of 5,737 bad).
- **Leakage:** perturbing the target week's realized stats leaves that unit's pre-game inputs unchanged; the T-24h availability prediction is invariant to game-day roster status; role state uses only prior games; no game past 2026 wk3 in the replay.
- **Fit audit:** 144 fits audited, 0 forbidden (no 2025-26 evaluation row in any architecture-selection fit except the explicitly labelled depth ablations); no sportsbook variable in any feature set and no sportsbook column referenced in any source file.
- **Allocation:** shares nonnegative and sum to 1 on real team-games; inactive branch receives exactly 0; named counts never exceed the team total.
- **Determinism:** repeated simulation is bit-identical; replay fingerprint and team-environment scores are identical across processes with different hash seeds.
- **Synthetic stress cases:** RB1 out, WR1 out, starting QB out, rookie promoted, two-RB committee, returning player. Each redistributes coherently (for example with the QB out, the backup takes the entire dropback volume).

## N. Failed / rejected architectures
- Role state: changepoint (BOCPD) and 5-level HMM (worse than last-8 on nearly every type; HMM up to 19% worse); they are not built into Phase 1A. Kalman and EWMA give modest gains but lose to the learned model.
- Team environment: gradient boosting (Poisson) and the opponent-adjusted blend lost to the GLM or the blend for most targets; blend kept for dropbacks and goal-line plays.
- Availability: logistic regression (rejected for boosting); a three-state (inactive/limited/normal) label (unsupported by data).
- Allocation: independent (non-coherent) shares, the last-8 propensity (replaced by the learned mean model), the structural receiving path (worse CRPS than allocation).
- Feature families that did not earn a place: organizational intent for 8 of 9 share types, opponent/team-style for 8 of 9, role-state filter features for 5 of 9, own availability for RB carry and RZ target.

## O. Selected Phase 1A architecture (frozen after the final freeze record, not before)
- **P(active):** gradient boosting on T-24h information (injury designation, practice status, previous-week roster status, games missed, recent snap participation, position, experience, depth rank on 2025+); T-90m adds game-day roster status. Conditional snap-share quantiles (boosting).
- **Team environment:** Poisson GLM (plays, rushes, designed rushes, red-zone plays/rushes/targets, defensive snaps); opponent-adjusted blend (targets); blend for dropbacks and goal-line. Negative-binomial dispersion fit on train.
- **Role state:** learned mean-share models over the families that earned their place (per type, in `selected_architecture.json`); Kalman filter for RB-carry role-change probabilities; role-change probabilities only for target, route, QB rush, defensive snaps, RB carry.
- **Allocation:** P(active) x role propensity, team-coherent (Dirichlet-)multinomial with an outside-candidate bucket for carries, targets, QB attempts and red zone; total x conditional share for routes and defensive snaps; Dirichlet concentration per type; team-volume model only for targets and defensive snaps.
- **Official receiving path:** target allocation.

## P. Unresolved data limitations
- nflverse injury rows carry no timestamp (assumption A2 is documented, not verified). If they absorb Saturday changes, the final-weekly-report-proxy results are slightly optimistic; the forward snapshot procedure (nfl_phase1_snapshots.py) removes this for clean-forward.
- Route-proxy truth is unavailable for 2026 games: routes are graded only on 2025 and cannot be a forward gate.
- The development evaluation is burned; every p-value here is optimistic.
- 2026 weeks 1-3 is a small sample (about 94 team-games): the team-environment selections for rushes and targets were not confirmed there.
- The candidate pool (all active-roster skill players) is broader than the protocol eligibility universe, so the headline gains include easy zeros; the eligible-universe check above is the fairer number.
- Depth-chart snapshots exist only from 2025; their value cannot be tested in the 2023-24 training split.
- No offensive line, opponent defensive personnel or coordinator/play-caller data is available with timestamps; those inputs are not used.
- The named-share underestimation and over-wide count intervals above are unfixed.


> **Amendment (Phase 1 hardening):** wherever this document says 'T-24h' for historical injury-derived features, read: historical final-weekly-report proxy evaluated under the documented nflverse timing assumption (A2). The availability model is gradient boosting (logistic was compared and rejected).

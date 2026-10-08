# Tennis Phase1H — human research and causal point mechanics

**Moneyline decision: REJECTED on ATP and WTA. Component I survives the retrospective point gate only; no promotion. 2025 remains sealed.**

The source-qualified specification was frozen at fd19246 before the permitted 2023–2024 diagnostics. All evidence is retrospective/burned. No new current physical-state source was qualified, no live forecast was generated, and nothing changes production.

## Audit and limitations

Phase1A scalar scoring and guards were retained. Its initial CI failure was a forbidden word in a docstring; executable solver arithmetic is byte-identical. Phase0 numerical outputs reproduce exactly to CI tolerance. Phase1H uses tournament-cluster intervals instead of Phase0 row resampling.

Important source defects: numeric tour levels were missing from the initial implementation whitelist; that run is INVALIDATED and retained. Category correction does not alter features, constants, model methods or gates. WTA reuses event IDs across editions and some event dates vary; these events now fail closed. A missing output folder and an observed point-rate boundary caused aborted runs, both fixed without changing forecast values or populations. No failed run or diagnostic drove parameter tuning.

The old claim that WTA I means ITF is not reliable: 2015 I includes Hobart and Auckland International. The locked I exclusion is preserved, restricting WTA external validity. Indoor/hand/height raw columns exist, although omitted by the old parser. ATP indoor is populated and WTA indoor empty; pre-match roof/court-vintage safety is not proven. These discoveries are recorded, not retrofitted into candidates.

TML tourney_date remains a tournament-start proxy, with the full 28-day embargo and no same-event updates. Exact last-week form, fatigue, duration timing, travel, health, venue/weather and shot-style intelligence remain BLOCKED_DATA. No current information replaces historical vintages. Source feasibility covers ten source/product categories; no vendor payload was authenticated, bought or scraped. Licensing limitations remain explicit; no raw CSV is committed.

## Frozen evidence-human baseline

ATP/WTA separate: tour -> player overall -> player surface pooled serve/conceded point counts (300 pseudo-points each), explicit receiver adjustment around legal league mean, then old opponent-adjusted residual history (last10 legal matches, 1000 pseudo-points). It is hand-calculable and reproducible but incomplete. The recent residual is >=28 days old and is not current-week form. Unsupported dimensions produce no fabricated observations or final-win penalties.

## Paired moneyline results

All model rows are identical within each tour/period. RAW includes decided retirements, excludes W/O. Every eligible match has a model projection (100% model-specific coverage); source-format, identity, event-date and prior-history exclusions are reported separately. Rank missingness uses the declared neutral .5 fallback and coverage is explicit.

| Tour / period | Matches | Overall Elo | Surface Elo | Rank | Phase0 Elo/rank | Evidence-human | Candidate I |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATP 2020–2022 selection | 5778 | 0.223738 | 0.222685 | 0.222467 | 0.217162 | 0.225765 | 0.220791 |
| ATP 2023–2024 burned | 4739 | 0.226994 | 0.226813 | 0.224312 | 0.218634 | 0.227942 | 0.223342 |
| WTA 2020–2022 selection | 4555 | 0.225908 | 0.227386 | 0.226413 | 0.220634 | 0.233098 | 0.227727 |
| WTA 2023–2024 burned | 4413 | 0.225437 | 0.224875 | 0.228605 | 0.220552 | 0.229511 | 0.224552 |

Scores are Brier (lower better). Full JSONs include log loss, ECE, intercept/slope, buckets, high-confidence misses, RAW/CLEAN, surface and matchup slices. Calibration regression is descriptive only, never refitted into predictions.

## Component results and decisions

| Tour | Family | Selection point MAE | Decision |
|---|---|---:|---|
| ATP | S | 0.064295 | WEAK (retrospective component gate) |
| ATP | I | 0.061577 | SURVIVES (retrospective component gate) |
| ATP | C | 0.061704 | REJECTED (retrospective component gate) |
| ATP | D | 0.061439 | REJECTED (retrospective component gate) |
| ATP | R | 0.061760 | REJECTED (retrospective component gate) |
| WTA | S | 0.068616 | REJECTED (retrospective component gate) |
| WTA | I | 0.066635 | SURVIVES (retrospective component gate) |
| WTA | C | 0.067160 | REJECTED (retrospective component gate) |
| WTA | D | 0.066812 | REJECTED (retrospective component gate) |
| WTA | R | 0.067406 | REJECTED (retrospective component gate) |

S = surface serving; I = overall serving plus explicit receiver interaction; C = surface + interaction; D = first/second-serve mixture; R = old opponent-adjusted residual. U is the no-receiver overall serve baseline. HUMAN is a fixed comparison, not an earned descendant. Failed families were not combined or fit into a final-win rescue stack. D/R match tests and STACK were not run because their parent gate failed. D has a small directional component gain but below the locked practical threshold.

Return MAE is serve MAE under the complement identity, not independent corroboration. First/second metrics printed for aggregate families are shared D-channel diagnostics, not independently fitted channel forecasts. Explained receipts explicitly identify which contributions enter each family.

## Tournament-cluster evidence

ATP: I improves point MAE 4.74% vs U; paired change -0.003063, 95% CI [-0.003681, -0.002408] across 160 tournaments. Moneyline I minus strongest Phase0 blend is +0.003629, CI [0.001124, 0.006055]. No 2% Brier improvement; no confirmation authorized.

WTA: I improves point MAE 2.56% vs U; paired change -0.001751, 95% CI [-0.002204, -0.001305] across 116 tournaments. Moneyline I minus strongest Phase0 blend is +0.007093, CI [0.003856, 0.010185]. No 2% Brier improvement; no confirmation authorized.

## Burned error diagnostics

ATP: I point MAE 0.060724, hold MAE 0.114897. Supplying actual match point rates to the stationary solver yields diagnostic Brier 0.069168 on 4604 CLEAN point-complete matches; actual-point hold oracle MAE 0.044129. Literal incumbent paired Brier 0.220226 on 4739 rows, labeled chronology-unsafe and not a fair causal promotion comparator.

WTA: I point MAE 0.066782, hold MAE 0.153878. Supplying actual match point rates to the stationary solver yields diagnostic Brier 0.053145 on 4266 CLEAN point-complete matches; actual-point hold oracle MAE 0.045570. Literal incumbent paired Brier 0.221109 on 4413 rows, labeled chronology-unsafe and not a fair causal promotion comparator.

The oracle contains outcome information and is not an achievable forecast or a measured ceiling on purchasable information. It shows that estimating match-day serving/returning performance is the larger remaining gap; the scoring transformation has substantially smaller descriptive hold error when given realized point performance. Historical marginal rates cannot explain all matchup/day variation. This does not prove health or style would close that gap. Missing health/current form and uncertain point dependence remain hypotheses requiring new timestamp-safe evidence, not excuses to rescue tuning.

## Receipts and reproducibility

Committed explained ledgers contain 4,739 ATP and 4,413 WTA burned diagnostic receipts. Each includes legal-history cutoff/hash, player identities and names, overall/surface strengths and sample counts, routed receiver adjustment, old trajectory where used, hold/break and exact set-score/tiebreak-count distribution, all benchmark probabilities, differences, uncertainty and separately labeled evaluation outcomes. Identical selection rerun produced byte-identical JSON objects and lock. Tests verify channel accounting, zero/missing data handling, orientation symmetry, exact solver agreement, source firewall and chronology attacks.

The exact original Phase0 protocol/manifest/frozen artifacts remain unchanged. Only research additions, registry/inventory references and solver docstring wording changed. Source downloads stay capped at 2024. Native production/frontend/API/scheduler/other sports and NFL/NHL branches remain untouched.

## Next evidence gate

Do not reopen this candidate with recalibration, weight fishing or a new smoother on burned diagnostics. First qualify a licensed timestamped match schedule/finalization source and player-ID crosswalk, then independently validate any embargo replacement. Prioritize trustworthy current serve/return observations and physical workload timing; injury/shot-style/environment must earn source and component gates. ATP indoor and handedness are possible raw evidence for a new preregistered hypothesis, not surviving signals. A truly complete competent-human benchmark is still BLOCKED_DATA.

**Final: I = SURVIVES_COMPONENT_SIGNAL_NOT_PROMOTED; S = WEAK ATP / REJECTED WTA; C/D/R = REJECTED as eligible combinations; physical/current/style families = BLOCKED_DATA; Phase1H moneyline = REJECTED. No 2025 confirmation, no promotion, no merge.**

## Commit trail

- eed2aeae2bece2164dcc950df1c588ccee625402 Tennis Phase1H: commit independent evidence feasibility and causal design gates before research
- f8f9f625eab64673ce8840c69954f2569c833e14 Tennis Phase1H: deterministic routed point models, partial human benchmark, receipts and adversarial checks; no results
- 4d3d7fd6cc50932d2c0d8f8a9230dfee8e213b7c Tennis Phase1H: freeze 2020-2022 component decisions and code lock before burned diagnostics; 2025 sealed
- 6146803c5fac5cbc679812ed06e038247d628e09 Tennis Phase1H: preserve invalid population run and independently document source-code correction; no tuning
- bdc66db1e3fbd9ab08781b26949da33ef89f1dd9 Tennis Phase1H: correct ordinary-tour category routing and diagnostic output creation without changing models or gates
- 761f4b0cfb4657fed25c36d2249a66161a3c4fb9 Tennis Phase1H: handle boundary observed point rates in descriptive oracles; forecast rows unchanged
- 8160f50ab040482c2306802e46f36280c14be758 Tennis Phase1H: preregister fail-closed handling of reused WTA event IDs and mixed event dates
- 4677c58510c35065f402331153d948185bdcd826 Tennis Phase1H: enforce event identity/date qualification before input replay
- fd192467cee289e959d7768f74f28e77769e8e52 Tennis Phase1H: freeze source-qualified selection and component failures before diagnostics; sealed confirmation blocked
- 1c5b1b386b06c80a004118fa8a38895d97314889 Tennis Phase1H: make receipt contributions explicit without changing frozen predictions or selection

This findings/diagnostic freeze commit follows the trail above; its SHA is supplied in the final report.

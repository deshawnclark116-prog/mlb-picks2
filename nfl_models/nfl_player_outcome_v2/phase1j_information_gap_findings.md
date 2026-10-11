# Phase1J-DATA — receiving information gap audit

**Receiving remains frozen at Phase1F pending accepted new information. Rushing is ready for a separate PBP-only research protocol. No model was fitted, selected, scored or forecast in Phase1J.**

Audit base: `411d6f8ca12f24e1f192ab2d7768589cf77d1bc0`, `codex/nfl-outcome-engine-v2`, existing draft PR #64. Documentation/schema/release observations end **October 5, 2026, 8:01 PM ET**. These are dated observations, not promises of continued availability. The JSON audit records each URL, HTTP status, receipt time and content hash. A 404/403 is an access finding, not proof a provider has no data.

Phase1H and Phase1I remain **REJECTED_EFFICIENCY_REPLACEMENT**. Their code, protocols, results, snapshots, receipts and findings are unchanged. Phase1I receiver depth is preserved separately as **SURVIVED_SIGNAL_NOT_PROMOTED**; its historical registry entry is not rewritten. Completed-air production remains the locked research problem. No old confirmation metric was used to choose a feature, architecture or threshold here.

## What is actually new, and what is still missing

There are real new information products, rather than another way to weight PBP air yards:

1. **Public FTN target quality and intent labels** already exist in the frozen corpus but were not routed into H/I: catchable, contested, drop, created reception, read thrown, designed/checkdown/scramble read, motion and screen/play-action labels.
2. **Licensed FTN full participation/route/matchup data** is a verifiable product. Its public catalog and OpenAPI schema identify player route roles/positions, associated defenders, separation details, coverage shell, pressure and time-to-pressure. We do not have its API entitlement or data rows.
3. **Licensed pregame state feeds** exist from Sportradar and SportsDataIO. Their schemas support expected depth hierarchy, injury/practice state and timestamps; a verified starter announcement and original earlier versions remain separate requirements.

None is currently certified as a complete, authorized **historical original-vintage plus live** source for the required receiving experiment. A provider's historical feed, latest `last_updated`, or retrospective CSV is not an archive of what was public at a particular T24/T90 cutoff. This is the reason for freezing the modeling track, **not** a claim that new football data does not exist. Paid access is viable in principle; no purchase, vendor contact or collection was performed.

## Source audit and classifications

The machine-readable catalog has **29 source/use cases** with provider/product, URLs, access cost, seasons, granularity, IDs/joins, historical/live coverage, publication and as-of status, T24/T90, cadence/latency, license, acquisition, reliability, repo presence, receiving layer, leakage risk and difficulty. Classifications apply to the specified use, not a blanket endorsement or rejection of a provider. Unknown costs, sample coverage and rights are explicitly unknown.

| Classification | Count | Main examples |
|---|---:|---|
| USABLE_NOW | 1 | Existing prior completed-game PBP history; not new receiving information |
| USABLE_WITH_API_OR_CONNECTOR | 1 | Forward, immutable receipts of the licensed public FTN subset; historical replay not certified |
| PAID_BUT_VIABLE | 8 | FTN full API; Sportradar depth and injuries; SportsDataIO state; SIS feeds; TruMedia API; Stats Perform feeds; Genius NFL API |
| BLOCKED_TIMING | 8 | Retained FTN history, final weekly injuries/rosters, 2024 depth, NGS aggregates, transaction deltas, weather/coach archives |
| BLOCKED_LICENSE | 3 | Undocumented ESPN acquisition, systematic official report collection, PFR bulk acquisition |
| UNKNOWN_NEEDS_ACCESS | 3 | PFF enterprise export, original starter/news archive, ESPN-derived 2025+ depth reuse rights |
| NOT_LIVE | 2 | Public participation; Big Data Bowl competition samples |
| NOT_USEFUL | 3 | Reweighting existing pair PBP; postgame starters as features; provider expected-stat/model outputs |

**PAID_BUT_VIABLE means an actual structured data product was verified, not that the required fields, history, timestamps or license are already accepted.** PFF stays UNKNOWN_NEEDS_ACCESS: paid consumer Premium Stats/articles are not proof of a legitimate export API. SIS raw feeds and TruMedia NFL API are documented, but exact route fields and versioned histories require samples. Stats Perform's market-oriented Dynamic Stats API is not evidence of a raw NFL route feed; soccer Opta Vision is not NFL tracking. Genius offers an official NFL API, but that does not establish our entitlement to raw NGS tracking. None supplies permission to use sportsbook inputs or provider projections.

Primary technical evidence: [FTN full field catalog](https://ftnfantasy.com/ftn-data-nfl-catalog), [FTN FAQ/product](https://ftnfantasy.com/data), [FTN OpenAPI](https://charting.ftntools.com/api/openapi.json), [Sportradar weekly depth](https://developer.sportradar.com/football/reference/nfl-weekly-depth-charts), [Sportradar weekly injuries](https://developer.sportradar.com/football/reference/nfl-weekly-injuries), [SportsDataIO dictionary](https://sportsdata.io/developers/data-dictionary/nfl).

## A. Verified pregame QB state

**Current model missing:** Phase1I chooses a QB from strictly prior roster identities and earlier current-team passing volume. It does not verify starter changes, practice participation, explicit restrictions or a coach's pregame starter announcement. It cannot safely turn a postgame passer or gamebook starter into a pregame fact.

Actual historical depth inspection:

- **2024:** 37,312 rows, 32 teams, **no `dt`**. Position/depth/week helps describe history but cannot reconstruct the original pregame chart.
- **2025:** 554,215 rows, 32 teams, **221 retained loaded timestamps**, August 3, 2025 through March 14, 2026 (the file is labeled the 2025 season, not 2026 regular-season evidence). 21,429 QB rows, 21,342 with GSIS IDs. LT/LG/C/RG/RT exist with some missing GSIS mappings. No duplicate rank-1 position slots in the audited daily format. `dt` explicitly means **when loaded**, not when a coach verified a starter.
- **2026 depth:** release metadata exists/updated October 5; no payload downloaded or player state inspected.

Sportradar documents depth rank `1` as the starter in a **depth chart** and explicitly allows updates during/postgame. Weekly historical endpoint shape does not recover overwritten versions. Injury `status_date` is a status-update timestamp, not necessarily original publication. Its workflow documents Sunday injury reports Friday 9 PM ET and inactives at 90 minutes before kickoff; polling delay means an inactive record cannot be assumed received at the exact T90 boundary. SportsDataIO has `DepthOrder`/`Updated`, but some old player practice fields are deprecated. Neither a depth rank nor inactive absence confirms the starting QB or a medical restriction.

Proposed state schema is in the audit: GSIS candidates, game/team/cutoff, **VERIFIED_STARTER / EXPECTED_STARTER / UNCERTAIN_STARTER / MULTI_QB_PACKAGE / EMERGENCY_REPLACEMENT**, separate practice/health/restriction evidence, original `published_at`, observed receipt, source SHA, version/retraction chain and missingness. No probabilities are fabricated from ranks. A post-cutoff emergency change remains an outcome, not retroactive pregame knowledge. Official announcements are potentially useful, but no complete licensed historical announcement/revision archive was established.

**Decision:** verified historical QB state **BLOCKED_DATA** until original reports/announcement vintages and rights pass acceptance. Forward state capture is feasible with a licensed feed; no collector or scheduler changes here.

## B. Routes, alignment, separation and target quality

The [public participation documentation](https://nflreadr.nflverse.com/reference/load_participation.html) explicitly says **2023 onward is supplied after all postseason games have completed**. It is not a contemporaneous public weekly source. The targeted receiver `route` label is not every receiver's route-tree ledger. Lists of players on the field are not routes run.

A repository parsing issue must be recorded without changing frozen code: `nfl_phase1_data.py` increments a field called `routes` for `offense_players` on passing plays. That includes blockers/QBs/other on-field players. **Those counts are NOT routes-run or route participation.** Phase1J adds an inventory interpretation override; it does not modify that code or any old result.

What full FTN actually documents:

- `skp_pos`: WR/slot/inline-TE/H-TE/back alignment categories.
- `skp_role`: route roles vs pass-protection/run-block roles; API on-field players also expose `position` and `role`.
- `rte`: route type; route action/detail schema. Exact per-target vs every-route type coverage needs a sample.
- Associated defender IDs and matchup structures. This is **not automatically verified shadow coverage or future primary assignment**.
- `sep`/`trg_sep`, contested/catchable/drop-related details, pressure `qbp`, `ttp` and `ttpr`, shell/blitz/read types. Categorical separation is not necessarily continuous tracking distance.

FAQ: charting/all-22 participation **2019+**, expanded participation **2021+**; roughly 24-hour charting with Sunday releases Monday night/Tuesday; commercial pricing starts **$5,000/year**, private **$3,000/year**. Agreement/field package applies. OpenAPI status exposes `is_charted`, `needs_reimport`, `last_updated`, **not a first-publication history or old approved payload versions**. Participant/event/play matching endpoints exist; coverage and GSIS mapping need a licensed sample.

Public NGS separation/cushion is **qualified player-week, target-selected aggregate**: nearest defender at catch/incompletion, not all-route separation or defender assignment. Minimum-attempt qualification creates missing players. QB tight-window and time-to-throw are distinct from pressure responsibility; time-to-throw excludes sacks. Week-0 season aggregates are unsafe if they contain later games. No aggregate is mislabeled route-level data; no NGS payload from current 2026 was fetched.

**Decision:** actual all-player routes/route rate/alignment/route-level separation/defender assignment are **BLOCKED_DATA in current access**. Full FTN is the clearest verified acquisition path, subject to license, semantics, IDs and original vintages. Big Data Bowl is a limited, terms-bound research sample, **NOT_LIVE**, not a production tracking feed.

## C. Offensive line and protection

`offensive_line_quality = BLOCKED_NO_TIMESTAMP_SAFE_SOURCE_IN_CURRENT_REPO` stays unchanged.

2025 daily expected OL depth exists, but it does not supply verified health, the historical 2024 pregame five, replacement timing or blocking responsibility. Postgame starters/snap counts can evaluate an assignment but cannot provide that assignment pregame. NFL/club injury pages require original dated revisions and an authorized acquisition basis. A Friday injury-time assumption is not proof.

Licensed depth/injury APIs can support forward expected-starter state. FTN full schema, SIS and TruMedia offer plausible raw blocking/pressure paths; no player-level pass-block win/loss/responsibility/history sample or original-publication archive is accepted. QB time-to-throw, sack rate, rushers/blitz counts and generic opponent EPA are **not OL quality**. Exact pressure responsibility, time-to-pressure and win/loss meanings require provider definitions rather than inference.

**Decision:** OL starter/replacement historical timing and pass-protection data remain blocked. A future PBP rushing experiment does not require pretending otherwise.

## D. Current 2026 pressure, coverage and FTN state

The only 2026 records processed are SHA-verified frozen **Weeks 1–4**. Current release metadata was checked without downloading season payloads.

- **Participation/man-zone:** no 2026 asset; public 2023+ participation is postseason-only. Previous-season tendencies are **STALE_ONLY**, not current coverage.
- **True pressure:** no current public participation labels. FTN public `n_pass_rushers`/`n_blitzers` is not a true-pressure label. Full paid FTN pressure schema exists; current data coverage is unknown without entitlement.
- **FTN blitz:** October 5 release digest exactly matches the frozen H/I file. All Weeks 1–3, **one Week 4 game**, **49 games versus 63 PBP games**. No new complete Week 4 snapshot is claimed.
- **New public FTN target-quality/read joins:** 3,029/3,936 frozen 2026 PBP targets. Missing labels are missing, not negative observations. On run plays/non-targets a `FALSE` catchable value is not an uncatchable-target observation.
- **NGS:** global passing/receiving/rushing assets updated October 5; metadata freshness does not establish current-season row coverage. No current aggregate records inspected.

Retained FTN source coverage (not model performance):

| Season | REG PBP targets | FTN target joins | Retained `date_pulled` range |
|---|---:|---:|---|
| 2023 | 17,483 | 17,483 | September 6, 2024 only |
| 2024 | 17,013 | 17,013 | November 13, 2024 – September 1, 2025 |
| 2025 | 16,609 | 16,609 | September 22–23, 2026 |
| 2026 W1–4 | 3,936 | 3,029 | October 5, 2026 only |

These are retrieval times of the **retained version**, not proof the underlying charting was first created that late. Conversely they cannot prove that version was available earlier. Retained 2024 charting cannot certify the W1–8 fit-period vintage; the 2025 retained version is after the season. Prior-season files that actually existed before a cutoff can be used as explicitly stale history in a future protocol, not reconstructed current-season snapshots. We do not retroactively certify or rescore H/I.

Historical injuries have modification dates in 2023–24 but **no original publication timestamps**; 2025–26 frozen modification dates are empty. A hash verifies the snapshot now, not its historical availability.

## Human gap matrix and acquisition priorities

The JSON matrix covers **18 items**. “HUMAN_CAN_KNOW” means a person can read pregame reports or obtain prior film/charting; it does not mean they know next game's routes, assignments, pressures or medical severity with certainty. Qualitative football value, not ease of coding or old validation scores, orders the gaps:

1. Starting QB
2. QB health
3. Actual route participation
4. Separation/target quality
5. Alignment
6. First-read/designed involvement
7. Defender injuries
8. Likely primary CB matchup
9. Current coverage tendency
10. Pass protection
11. Expected OL starters
12. Recent role changes
13. Teammate injuries
14. Target competition
15. Coaching/play-calling changes
16. Receiver role
17. Issued weather/roof information
18. Existing depth profile (already known; preserve signal, do not reweight)

The **five acquisition packages**, with full component, chronology, cost and recommendation records in `phase1j_source_priority.json`, are:

| Rank | New information | Specific receiving mechanism / next action |
|---|---|---|
| 1 | Timestamped expected/verified QB + health | Attach receiver depth/catchability to the right available thrower. Obtain licensed original depth/practice/announcement versions; build state receipts only after acceptance. |
| 2 | All-player routes + alignment/intent | Distinguish release from protection and slot/outside/TE role. Request FTN route-role sample, IDs and historical vintages; not an on-field proxy. |
| 3 | Depth-specific catchability/separation/contest/drop | Distinguish throw/window/drop mechanisms limiting completed air. Public FTN content is real, but historical vintages need acceptance; archive legally eligible forward receipts under a separate protocol. |
| 4 | Defensive personnel + defender/coverage association | Replace coarse opponent average with actual prior leverage/personnel context. Require original injury versions and associated-defender semantics; do not assume future shadow coverage. |
| 5 | First-read/designed/motion role intent | Separate primary/design involvement from checkdowns/scramble responses. Require read/route denominator definitions and receipt history; motion alone does not prove a schemed target. |

These are **source acquisition priorities, not selected models or proven incremental lift**. No new information earns a model family merely by sounding plausible. Per-field coverage, license, original versions, ID joins and latency must pass first. A subsequent protocol must isolate a claimed component and preserve the oracle-target/no-cancellation rule. Previously exposed 2025/2026 periods cannot be relabeled a new untouched holdout. No Phase1K receiving protocol is created now.

## Rushing recommendation

**RUSHING_READY_FOR_NEXT_RESEARCH_PHASE**, restricted to a new prior-PBP mechanistic protocol without OL inputs:

- Keep the frozen improved carry-share allocator as workload; do not modify it here.
- Prior defensive run EPA, explosive-run allowance, RB-specific allowance, run location/gap and red-zone/goal-line carry role exist in PBP. Exclude QB kneels/scrambles and use stable identities under a preregistered population rule.
- Box count/stacked-light history exists in FTN posthoc. It remains gated by original-vintage/forward receipts and adequate prior-carry joins. Zero box on an irrelevant play is not proof of a light box. Box count is not a named defensive front.
- DL/LB absence remains timing-blocked; exact front assignment and player-level OL quality remain blocked.

The entry experiment can examine prior PBP run-environment mechanics separately, first with **actual carries** to test efficiency, then predicted carries only if efficiency passes. No final-yard cancellation promotion. Use 2024-only development under a separate preregistration; do not tune on 2025/2026, claim those previously exposed periods newly untouched, or access Week 5+ before a separate frozen forward protocol. Readiness is permission to design a test, not evidence of improvement. No rushing model is built in J.

## Reproduction, scope and checks

`nfl_v2_phase1j_information_audit.py` has no model/network imports. It validates the catalog and reproduces source-only joins/timestamp coverage from the exact H corpus. It rejects revised bytes, a newly published unaudited participation source and any 2026 Week 5+ row before processing its labels. Depth audits accept only 2024/2025 files.

```bash
python nfl_v2_phase1j_information_audit.py --check
python nfl_v2_phase1j_information_audit.py --check \
  --data-dir /workspace/scratch/phase1h-data \
  --depth-2024 /workspace/scratch/phase1j-evidence/depth2024.response \
  --depth-2025 /workspace/scratch/phase1j-evidence/depth2025.response
python -m pytest -q tests/test_nfl_v2_phase1j_information_audit.py
```

The CI job validates the dated catalog and protected-file hashes **without fetching live records or executing models**. A test-only Phase1I scope guard now compares its already-published final tree to its own base, so future independent audit files do not make the frozen I audit fail. Its protected-byte proof remains; no Phase1I scientific code or frozen artifact changes. New Phase1J tests independently compare every protected file against the requested starting HEAD.

Scope disclosure: one undated ESPN endpoint was probed for HTTP reachability. The downloaded response was **never parsed, inspected for player state or used**, and was removed because its week could not be bounded. Thus no Week 5+ records were extracted or used, but this audit does **not** claim no potentially future-state response bytes were received. All substantive 2026 record processing used the frozen W1–4 corpus. No other current-season data payload was downloaded.

No production/frontend/scheduler/grading/forecast/scientific-store changes, sportsbook inputs, Monte Carlo, new fit/evaluation, vendor contact or other sport changes. `source_inventory.json` preserves old entries as locked history and adds explicit future-acceptance overrides; `research_registry.json` preserves every earlier phase entry and adds the decisions below. Existing PR #64 remains draft and unmerged.

Local validation: **22 Phase1J tests + 3 frozen H/I history tests passed**. Full source/depth coverage reproduced exactly, Python compilation and actionlint passed, and `git diff --check` is clean. **1,155 pre-existing protected files are byte-identical** to the requested HEAD; the only old-file changes are additive inventory/registry metadata and the test-only I scope endpoint.

Receiving: **FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION**

Rushing: **RUSHING_READY_FOR_NEXT_RESEARCH_PHASE**

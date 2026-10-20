# Phase1M-DATA — QB starter / health state acquisition and acceptance gate

**Decision: `PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS`.** No source passed the 2024 historical gate. No source passed forward/live acceptance. Phase1L stays frozen at `BLOCKED_STARTER_STATE_DATA` and is **not** reopened. **NO_PROVIDER_PAYLOAD_TESTED** for every paid vendor.

This phase is source qualification only. No QB attempt, passing-yard or any other model was fitted; no 2024 fit, 2025 validation, 2026 diagnostic, Monte Carlo or sportsbook field was used. Receiving remains `FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION`, receiver depth `SURVIVED_SIGNAL_NOT_PROMOTED`, rushing efficiency `REJECTED_EFFICIENCY_REPLACEMENT`, QB opportunity `BLOCKED_STARTER_STATE_DATA`. No frozen artifact, production file, frontend, scheduler, grading, calibration, forecast or other sport was touched.

Order of work (git history): protocol committed first (`30dd857`), then implementation/tests/workflow (`a146386`), then results, findings and registry updates.

## What the model needs to know, and what we found

The only legitimate question is *what was knowable before kickoff at T24 (kickoff − 24h) and T90 (kickoff − 90 min)*, never who eventually started. The temporal rule is implemented once, as `state_at(team, game, cutoff)`: a record is visible only if its publication time (or, when the provider gives none, **our** retrieval time) and its retrieval time are both at or before the cutoff, its validity window covers the cutoff, it is not postgame-only, it is not a backfill unless original-vintage, and it is eligible for that cutoff. Later revisions never leak backward; the highest *visible* revision wins. A provider that cannot supply as-of semantics cannot certify Phase1L.

## Canonical schema — status: design complete, archive empty

`phase1m_qb_state_schema.json` defines all 40 fields requested (identity and crosswalk, 16 state types, ranks, health/practice/injury fields, source/publication/retrieval/effective/valid timestamps, revision id and sequence, evidence type, cutoff eligibility, `as_of_certified`, confidence, stale/postgame/backfilled/original-vintage flags, raw and normalized hashes). Rows are append-only, uniquely keyed by `(source_provider, source_record_id, revision_sequence)`, and certification requires a publication timestamp, original vintage and no backfill/postgame flag. `confidence` is null unless certified; no probability is derived from a depth rank. The QUESTIONABLE → EXPECTED_STARTER → VERIFIED_STARTER sequence is tested for exact visibility at T24, T90 and kickoff. **There are no rows**: no accepted source exists.

## Provider scorecard (14 candidate records, evidence dated 2026-10-06)

| Provider / product | What the documentation actually shows | Classification |
|---|---|---|
| **Sportradar** NFL Weekly Depth Charts, Weekly Injuries, Game Roster, Daily Change Log | Depth rank 1 = starter; injury status, practice status, body part, `status_date` (last update); inactives entered about 90 min before kickoff; Daily Change Log of modified entities; season parameter 2000–2026 / 2009–2026. Depth charts "may be updated during or post game". Original versions are **not** documented as retained. Trial key exists (terms unread). | **PROMISING_NEEDS_SAMPLE** |
| **SportsDataIO** DepthCharts / Injuries | `DepthOrder` (1 = starter), per-change timestamped depth charts (2022 announcement), injury `Status`, `BodyPart`, `Updated`; `Practice` fields **deprecated**. Historical content "requires contacting". The documentation states the free trial covers only UEFA Champions League, so **no NFL sample**. | **PROMISING_NEEDS_CONTRACT** |
| **Sports Info Solutions** Data Hub | NFL injury database from 2016 with diagnoses, prognoses, return dates. No expected-starter/depth product evidenced; publication times unknown. | PROMISING_NEEDS_SAMPLE (injury layer only) |
| **Stats Perform / Opta** | Live player stats and betting feeds; no NFL pregame state product documented. | BLOCKED_ACCESS |
| **Genius Sports** | Official NFL data distributor to media/operators; play-by-play/tracking oriented; no pregame QB state product evidenced; operator terms may conflict with the no-sportsbook rule. | BLOCKED_ACCESS |
| **TruMedia** API | Player/team/play statistics only; no depth, injury or roster-status endpoint. | NOT_USEFUL |
| **FTN full API** | Postgame charting/participation. Site returned HTTP 403 in this phase, so Phase1J evidence is cited, not refreshed. | POSTGAME_ONLY |
| **PFF** Pro API | Grades/charted detail; nothing about depth, expected starter or injury versions; API price not stated. | BLOCKED_ACCESS |
| **NFL.com official injury report** | Authoritative, practice status by day and game status, but a mutable page with no visible timestamp, names only, no API, no systematic-collection license. Not scraped. | BLOCKED_LICENSE |
| **nflverse depth charts 2001–2024** | 2024 has zero `dt` timestamps. | BLOCKED_TIMING |
| **nflverse depth charts 2025+** (ESPN-derived, `dt`) | See the timing probe below. `dt` is a load time; upstream reuse rights for ESPN-derived rows unresolved. | **PROMISING_NEEDS_CONTRACT** (forward-only) |
| **nflverse injuries** | 2024 modification timestamps only, zero original publication rows; no starter identity. | BLOCKED_TIMING |
| **Internet Archive Wayback** (archived NFL.com / ESPN / Ourlads pages) | Availability API probed; see below. Rights unresolved, names only, content not opened. | BLOCKED_LICENSE |
| **Repository sources** (rosters, transactions, gamebook starters, V1 `qb_out`) | No certified pregame starter rows; gamebook starters are postgame evaluation only. | BLOCKED_TIMING |

Classification counts: PROMISING_NEEDS_SAMPLE 2, PROMISING_NEEDS_CONTRACT 2, BLOCKED_TIMING 3, BLOCKED_ACCESS 3, BLOCKED_LICENSE 2, NOT_USEFUL 1, POSTGAME_ONLY 1, accepted 0.

A documented endpoint, a `last_updated` field, or a historical week parameter is **not** evidence that the original pre-kickoff state can be replayed. That is exactly the gap the Phase1L audit found, and no vendor documentation closes it.

## Sample / trial acceptance results

- **Vendor trials/samples: none run.** Sportradar offers a trial key (terms not read); SportsDataIO's trial excludes NFL; others require sales. Every vendor row reads `NO_PROVIDER_PAYLOAD_TESTED`. The ten acceptance questions are `NOT_TESTED` for them. Nothing is fabricated as a pass.
- **Preregistered archetype sample.** Fixed before any result: for 2024 weeks 3–18, labels from realized attempts stratify the sample only (never a feature, never scored); 4 team-games per archetype by the smallest `sha256(game_id|team|phase1m-sample-v1)`. Eligible: 376 stable veteran, 35 change-to-prior-backup-or-return, 22 multi-QB, 10 new-or-surprise. 16 selected. Injury replacement, returning starter, rookie promotion, benching and recent acquisition are **not separable from realized data** and are not claimed. The historical archetype test is `NOT_RUN_NO_SOURCE_ACCESS`.
- **Web archive (Internet Archive Availability API, metadata only).** 96 requests (16 team-games × T24/T90 × 3 page families) at one request per 6 seconds. Attempt 1 hit HTTP 429 on the first call (0 completed; the stop rule recorded it) and attempt 2 completed all 96. Only **5 of 96 (5.2%)** returned a capture at or before the cutoff and within 72 hours; 34 were older; 48 returned a capture only *after* the cutoff (the API does not enumerate versions, so a pre-cutoff capture may exist but is unproven); 9 returned none. The CDX enumeration endpoint is blocked by this environment's egress policy and was not used. No archived page body was opened. Conclusion: the archive is crawl-scheduled, not game-scheduled, and cannot support T24/T90 reconstruction at the required coverage even before rights and identity (names, not ids) are considered.
- **Repository depth timing probe (2025 only, counts only).** `depth_charts_2025.csv` (sha256 `f5a4aa3f…`, 221 QB-bearing load times, 2025-08-03 to 2026-03-14). For all **544** 2025 regular-season team-games, a QB-bearing snapshot exists before **both** T24 and T90, within 72 h, with a single rank-1 QB (median age 12.9 h at T24 and 9.3 h at T90; p90 18.0 h and 16.4 h). Post-registration descriptive addition (recorded as amendment 1): the rank-1 QB changes 39 times across 17 teams, and 2 team-games differ between their T24 and T90 snapshots, so the log does move. Caveats that keep this from being an acceptance: `dt` is when the record was **loaded**, not when a starter was announced; the rows are ESPN-derived with unresolved reuse rights; rank 1 on a depth chart is not an announced starter; it contains **no 2024** and no health state. It does show that the existing repository forward capture is timing-capable.
- **Live payloads:** no Week 5+ (2026) player-state payload was opened; live capability is judged from documentation, release metadata and the forward design.

## Prior-continuity hint audit

`NOT_RUN_NO_CERTIFIED_STARTER_STATE`. The 1,202 receipts and 394 hints stay unpromoted. The function that will report continuity accuracy, false-continuity rate and the six slices exists and is tested only on synthetic certified rows. False-change rate additionally needs certified rows for teams without a hint.

## Purchase / access decision

| Slot | Candidate | Why, and what is still unproven |
|---|---|---|
| Best technical fit | **Sportradar** | Closest documented match: depth rank, injury + practice status, inactives at about T-90, and a Daily Change Log. Unproven: retained original 2024 versions. |
| Best practical fit | **Sportradar trial key as an acceptance test before any spend** | Cheapest way to answer the retained-versions question. |
| Best low-cost fit | **nflverse depth 2025+ and injuries via the existing hash-captured forward snapshots** | Free, GSIS ids, timing-capable (above). Forward-only; ESPN-derived rights open. |
| Best enterprise fit | **Sportradar production package**, or **SportsDataIO** if written answers confirm archived per-change history | Contract and license terms unknown. |
| Best forward-only option | **Repository forward capture (nflverse) plus a vendor trial in parallel** | Cannot validate 2024. |
| Not worth pursuing | TruMedia; FTN and PFF for this gate; Stats Perform and Genius for this gate; Wayback scraping | No documented pregame state, postgame data, or unresolved rights. |

For the two leading vendors:

1. **Solves the 2024 gate?** Neither is shown to. Both need written proof of retained original pre-kickoff versions with first-publication times.
2. **2026 live?** Both plausibly yes by polling (Sportradar: hourly or faster; SportsDataIO: per-change timestamps). Untested.
3. **Starter identity?** Depth order 1 only; no announced-starter field seen. Sportradar's Game Roster is "game-day truth" and is a postgame-adjacent record, not a pregame input.
4. **Health/practice?** Sportradar yes (injury and practice status, body part). SportsDataIO injury status yes; practice fields deprecated.
5. **Revision timestamps?** Sportradar `status_date` and a change log; SportsDataIO `Updated`. First-publication time is not established for either.
6. **Integration:** a polling collector writing the raw layer and the normalized schema (MEDIUM effort, designed but not built or started).
7. **Documented price:** none for either. Both read **CONTACT SALES / UNKNOWN**. The only dollar figures anywhere in the scorecard are the FTN FAQ figures ($3,000–$5,000 per year) quoted from the Phase1J audit; that page returned HTTP 403 this phase and is not re-verified.
8. **Rights questions:** internal analytical storage, derived-model use, retention after subscription ends, and whether historical archives can be used for model validation.
9. **Proof before spending:** (a) a trial or sample call for three 2024 games from the preregistered sample returning pre-kickoff depth/injury rows with timestamps; (b) the change log reconstructing at least one QUESTIONABLE → EXPECTED_STARTER sequence; (c) GSIS crosswalk on the sample QBs; (d) written license terms. Do **not** purchase because a vendor advertises depth charts.

## Phase1L reopen decision

**`PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS`.** Reason: no accepted historical or forward source, but two vendors (Sportradar, SIS) are `PROMISING_NEEDS_SAMPLE` and two (SportsDataIO, nflverse 2025+) are `PROMISING_NEEDS_CONTRACT`, so only a sample, a contract answer or a rights clearance stands between us and an acceptance test. Forward-only data would not validate Phase1L on 2024 in any case.

Historical reopening would require, on a tested payload: certified pre-cutoff starter state for at least 95% of the 544 2024 team-games at each cutoff, at least 99% GSIS joins, a publication timestamp on 100% of rows, and original-vintage proof.

## Clean-forward collection plan (design only; nothing started)

`phase1m_forward_capture_plan.json` specifies a raw layer (provider, retrieved_at, published_at, payload_hash, game, team, player, raw_payload_reference) and the normalized schema layer; T24 capture completing before kickoff − 24h and T90 capture before kickoff − 90 min (T90 inactives need explicit lag receipts); append-only, read-only-after-write storage with a hash verifier; revisions as new rows with `valid_to` set on the superseded interval; confidence null unless certified; no outcomes, postgame starters or box scores as inputs; no unauthorized recurring scraping. It activates only after the repository owner authorizes a named source and terms and a vendor trial passes the harness.

## Limits and honest caveats

- The scorecard is documentation evidence at a point in time, not a hands-on test. Vendors may retain versions their public documentation does not mention; a sample request is how that gets answered.
- The Wayback probe measures existence and capture time only. It does not show that any capture displays a certified starter state.
- Archetype labels use realized passer attempts solely to choose the sample; no archetype outcome was evaluated.
- The 2025 timing probe is 2025 only; 2025 remains Phase1L's selection-free confirmation season and was not used for any selection.
- Per the evidence date, FTN pages were inaccessible (HTTP 403) and Genius documentation was seen only through search results, which is why those rows are weak.

## Reproduction

```bash
python nfl_v2_phase1m_qb_state_sources.py --check            # schema, scorecard and forward plan, byte for byte, no network
python -m pytest -q tests/test_nfl_v2_phase1m_qb_state.py    # archive, harness, decision rule, frozen-scope guards
# stages that read pinned inputs (not run in CI):
python nfl_v2_phase1m_qb_state_acceptance.py --data-dir DIR --select-sample --out sample.json
python nfl_v2_phase1m_qb_state_acceptance.py --data-dir DIR --timing-probe --out timing.json
python nfl_v2_phase1m_qb_state_acceptance.py --probe-wayback --sample sample.json --out wayback.json   # dated, throttled, network
```

CI never calls an external provider or archive.

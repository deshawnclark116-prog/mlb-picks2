# NHL Phase1C — evidence admission & frozen B2 probability integrity

**Status:** research/integrity-only; no model promotion, no fresh NHL provider capture, and no live forecasts. Stacked on rejected Phase1B PR #69, itself stacked on integrity-repair PR #68 and frozen forward-engine PR #65. The Phase1C branch does **not** rewrite old Phase1B decisions, forecast rows, source bytes or model lock.

## What is proven by code

The original B2 `distribution_summary` stores a shifted cumulative survival:
`P1=1`, `P2=P(SOG>=1)`, ..., `P5=P(SOG>=4)`. This is a defect in
**threshold labeling**; existing expected SOG and original NB2 parameters
remain the locked estimates. The read-only `nhl_v2_phase1c_probabilities.py`
calculates `P(SOG>=k)=nbinom.sf(k-1,1/alpha,(1/alpha)/(1/alpha+mu))`
for k=1..5. These are **posthoc corrected interpretations**, never claims
that corrected probabilities were originally published ahead of a game.
Original forecast bytes, artifacts, evaluation locks and threshold fields
are preserved. Original P1..P5 may not be shown under their old labels.

The separate CLI `nhl_v2_phase1c_probabilities.py --output <file>` verifies
the actual frozen ledger before producing **a derived audit report only**.
It does not backfill or edit any past forecast record. No new model outcome
or historical accuracy claim is produced.

The offline intake `nhl_v2_phase1c_ingest.py` takes an explicit source-normalized
JSON packet and a separate operator-attested grant. The packet is the raw
**normalized operator submission**, not a retrieved original vendor payload.
The exact packet bytes are hashed and stored immutably, and the received-at
timestamp comes from the importing process clock, never a packet claim.
The permission grant has to state access, automated collection, internal
storage, modeling and raw-evidence retention are authorized for this scope.
The grant must match the provider and evidence type.

A free-text permission assertion alone does **not** legally prove the grant:
all ingested records are classified `OPERATOR_ATTESTED_NOT_INDEPENDENTLY_LEGAL_VERIFIED`.
Do not use them to train, promote or issue official NHL predictions until
license evidence, provider identity mapping, original raw provenance,
coverage and timestamps are independently certified.

The importer forbids retrospectively backdated "pregame" packets: importing
after the cutoff is rejected even if a file claims an earlier retrieval.
It fails closed on missing rights, future publication/revisions, ambiguous
identities, absent observations, unsupported role ordinals and confirmations
made by roster-only sources. A roster membership observation does **not**
imply a player dressed, and provider source projections do not imply a
verified deployed line or power-play group.

## External source decisions (documentation research, not source collection)

| Source | Candidate observations | What has been verified | Current gate |
|---|---|---|---|
| NHL.com / public NHL endpoints | schedule, roster hypotheses, official games | Previous Phase1B actually stored pre-cutoff endpoint responses, but found zero nonempty roster or scratch observations; NHL Terms restrict unauthorized automated harvesting | **BLOCKED** until collection/use authorization and source content qualification |
| SportsDataIO NHL | season line combinations, PP combinations, injury status and transactions | Public NHL API docs/workflow guide describe these feeds; data rights FAQ says ML/model-use rights can be licensed on a use-case basis | **CONTRACT NEEDED**: feeds, snapshot timestamps, archive/replay and permitted model use must be specific |
| Sportradar NHL v7 | depth charts, injuries, player game context | Public documentation advertises feeds; authenticated access required, injury feeds have documented update cadence | **CONTRACT NEEDED**, NHL-ID mapping and timestamped qualified snapshots |
| Existing frozen historical corpus | skater TOI by EV/PP/PK and SOG; pre-2024 adjudicated total attempts | Research history (exposed/in-sample, historical original publication vintage absent); no current strength-specific attempts | **EXPOSED RESEARCH ONLY**, no right to claim a newly certified pregame informational edge |

Documentation inspected October 9, 2026:
- https://www.nhl.com/info/terms-of-service
- https://sportsdata.io/developers/workflow-guide/nhl
- https://sportsdata.io/help/data-rights-and-licensing-questions
- https://developer.sportradar.com/ice-hockey/reference/nhl-overview
- https://developer.sportradar.com/ice-hockey/reference/nhl-injuries

These are capability/terms claims **from public documentation**, not successful
licensed login, provider API access, complete sample, or actual-lineup certification.
No provider API key was requested or consumed and no prohibited scrape occurred.

## First actual Phase1C immutable-ledger audit (CI, no refitting)

At research branch head `f3b22d04f8ed10078b368465ddec03edf59b8725`, the new CI audited
the verified 1,563-row Phase1B snapshot and reported:

- **1,556** immutable `FORECAST` records; seven nonforecast/missed records untouched
- **1,556 mismatches at each of P1, P2, P3, P4 and P5** when checked against correctly labeled NB2 thresholds
- **1,547 distinct `forecast_id` values**, with **nine collision groups comprising 18 forecast rows**
- October 7 PIT/WSH: four collision groups/eight rows (known from the Oct 7 safe grader)
- October 8 BOS/UTA: four additional collision groups/eight rows
- October 9 VGK/TOR: one additional collision group/two rows
- The forecast ledger, model, prefit data and source hashes were not modified.

**New severity clarification:** cross-team duplicate forecast IDs are not a
one-night Oct 7 phenomenon; they continue in later frozen forecasts. The
Phase1C adapter preserves both rows and flags their original row hashes.
It does *not* resolve or rewrite the original frozen generator, and no
duplicate forecast ID is eligible for a unique-player join. Any future new
generator must reject contradictory pregame player-team membership before
emission, with an independently timestamped ownership source. Do not
pretend the frozen B2 code is already fixed.

CI: https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/38016229968

## Real admission sequence

1. Acquire written access rights for the exact feeds, permitted automation, storage, history and model training, and retain the reference in the restricted grant manifest outside Git.
2. Create a real adapter that imports provider source bytes with retrieval clock and vendor identity crosswalk; only derive an operator-normalized JSON packet after hashing original bytes and recording deterministic transformation code/version.
3. Admit packets prospectively *as they arrive*; do not redate or retrofit previously exposed games.
4. Compare complete skater availability for both teams against actual final dressed rosters, independently for T24H/T90/T30.
5. Require each horizon's locked **40 complete games, >=3 distinct dates, >=99.5% dressed-player recall**. This qualifies candidate coverage; it still does not magically certify an active roster as a confirmed game lineup.
6. Measure line/PP change detection and TOI/shot-attempt predictive utility **without fitting on burned periods** before considering a candidate model. Retain rejected Phase1B features and no new forward engine until data passes gates.

## Reproduction / blockers

The Phase1B branch froze 1,563 ledger rows at its last synchronization, including
1,556 forecasts with shifted fields and 7 missed-window records; PR #65's
append-only real forward branch has subsequently continued. This Phase1C
stack deliberately does not mutate or squash any later PR #65 ledger states.
Before an eventual separate research promotion, reconcile the advancing
PR #65 prefix, rerun hash/immutability audits and independently prove no
forecast row was lost.

**Blockers:** certified T24H/T90/T30 lineups all zero, current verified deployment
and strength-specific attempts absent, independent human comparator absent,
qualified contract samples absent. Phase1B feature experiments are rejected.
No production/API/frontend or other sport is changed.

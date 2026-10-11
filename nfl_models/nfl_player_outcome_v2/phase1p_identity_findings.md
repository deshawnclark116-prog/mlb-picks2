# Phase1P-DATA: player identity / snap mapping hardening

**Decision: PHASE1Q_SNAP_CAN_OPEN.** Combined 2023-2024 primary-position (WR, TE, RB, FB, HB) snap-row mapping share is **99.78%** (11,896 of 11,922 rows), against the unchanged 95% gate. The Phase1O family B blocker was an identity-coverage problem, not a predictive result: the snap-change hypothesis has still never been tested.

## What was wrong

Phase1O joined the nflverse snap file (PFR ids only) to GSIS ids through the pinned weekly-roster `pfr_id` column, which is blank for about 39% of roster rows. That exact roster-only rule covered 93.80% of skill rows including QBs (93.29% for the primary positions) and was correctly blocked at the 95% minimum.

## Sources inspected

- **nflverse players dataset (`players.csv`)**: the canonical identity table with `gsis_id`, `pfr_id`, name, position and other metadata. 24,844 rows, 22,679 with both ids. **New pinned source.** The upstream release file is rolling (it was modified on the retrieval day), so its exact bytes (sha256 `d531dcff...5d4cc`, 7,298,556 bytes, retrieved 2026-10-06T16:06Z) are frozen as an identity-only extract `phase1p_identity_extract.csv.gz` (sha256 `22873d96...766a`, columns gsis_id, pfr_id, display_name, position). Reproduction reads only that committed extract; it never downloads a latest file.

- **Pinned weekly rosters 2023-2024** (already accepted): identity columns only, used as the second exact source.
- **Pinned snap counts 2023-2024** (already accepted): metadata columns only.
- Repo player crosswalks: none beyond the above.

## Rule (preregistered before coverage was finalized)

1. exact canonical pfr_id -> gsis_id; 2. exact roster pfr_id -> gsis_id; 3. when both exist they must agree, otherwise the row is `SOURCE_CONFLICT` and stays unmapped; ambiguous ids stay unmapped; two distinct PFR ids resolving to one gsis within one team-week are a conflict. The matcher has no access to names, outcomes, target-game information or future weeks (the code reads only the allowlisted identity columns, enforced by a test). No player was patched.

## Coverage

| Season | Rows | Mapped | Unmapped | Share | Unique PFR ids | Unique GSIS ids |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 5,980 | 5,968 | 12 | 99.80% | 530 | 529 |
| 2024 | 5,942 | 5,928 | 14 | 99.76% | 542 | 539 |
| **Combined** | 11,922 | 11,896 | 26 | **99.78%** | 671 | 668 |

Rule usage: {'AGREE_CANONICAL_ROSTER': 11122, 'CANONICAL_ONLY': 774}. Ambiguous rows: 0. One-to-many PFR->GSIS conflicts among snap ids: 0. Many-to-one conflicts: 0. Canonical vs roster crosswalk: 2016 agreeing pfr ids, 3 disagreeing. Every mapped GSIS id appears on a pinned roster of its season (11,896 of 11,896), so Phase1Q can join them.

## By position (combined 2023-2024)

| Position | Rows | Mapped | Share |
|---|---:|---:|---:|
| WR | 5,386 | 5,382 | 99.93% |
| TE | 3,284 | 3,262 | 99.33% |
| RB | 2,905 | 2,905 | 100.00% |
| FB | 347 | 347 | 100.00% |
| HB | 0 | 0 | n/a (no rows) |
| QB (diagnostic) | 1,379 | 1,379 | 100.00% |

## Unmapped rows

Reason counts (primary positions): {'AMBIGUOUS_PFR_TO_GSIS': 0, 'NO_CANONICAL_ID_RECORD': 26, 'OTHER': 0, 'PFR_ID_MISSING': 0, 'ROSTER_CROSSWALK_MISSING': 0, 'SOURCE_CONFLICT': 0}. All 26 unmapped rows are `NO_CANONICAL_ID_RECORD`: three players (two TEs and a WR) whose PFR ids exist in neither the canonical table nor any pinned roster. They stay unmapped and are listed in `phase1p_unmapped_snap_rows.jsonl.gz` (identity fields only). No rule was changed after looking at them.

## Gate

- combined mapping >= 95%: **yes**
- no fuzzy / name-derived mapping: yes (matcher never reads names; tested)
- deterministic exact rules and pinned bytes: yes (extract digest-verified; roster and snap digests verified)
- reproducible audit: CI regenerates and byte-compares every artifact
- no outcome leakage: no target, carry, yard, score or target-game column is read

The 2025 and 2026 files were not opened, and no model was fitted.

## Honest caveats

- The canonical table is a current snapshot; it contains only identity pairs for the matching, and a pair does not change with outcomes, but it was retrieved after the 2023-2024 seasons. Only the id pair is used.
- Before the protocol was committed I had seen one preliminary aggregate coverage figure (above 99%); the rules are the task brief's hierarchy and nothing was tuned on unmapped rows.
- Mapping coverage is not predictive evidence. It only makes the snap-change test possible.

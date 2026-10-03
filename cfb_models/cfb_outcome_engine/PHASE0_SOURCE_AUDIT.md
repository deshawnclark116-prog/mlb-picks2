# CFB Outcome Engine v1 — Phase 0 source audit (research / shadow only)

Reproduce: `python cfb_phase0_source_audit.py --raw <dir of cfbfastR-data season files>` (2017-2025; no 2026 file is read). Probe responses are in `phase0_source_registry.json`.

**What exists (usable):** schedules / kickoff / final points (2017-2025, 100% FBS), play-level player-attributed plays with period / clock / down / distance / yards-to-goal / score state (0% missing) → team plays proxy, rush / pass-attempt counts, red-zone plays; player box lines; ESPN athlete IDs that are stable across transfers; season roster snapshots (position, height, weight; class `year` with mixed encoding).

**What is blocked (probed, not fabricated):** depth charts (ESPN endpoint empty), injuries (league endpoint: 3 players, one dated 2020), snaps, routes, true targets (`target_player_id` on only 7-29% of pass attempts), recruiting pedigree (`recruit_ids` 0% populated), transfer portal, coaching / scheme.

**Consequences for the design:** participation is a box-score line (a player with no recorded stat is invisible → availability is a prior-participation model, never an injury model); receiving opportunity = receptions; team volume = attributed-play counts; organizational-intent priors are limited to returning / newcomer status inferred from appearance history; FCS opponents form a separate low-information regime; historical injuries / depth charts are `BLOCKED_HISTORICAL_PIT` and can only become forward-only features if coverage appears.

**Identity:** 2,918 box-score producer transfers keep the same positive ESPN athlete_id with 0 name conflicts (2,840 confirmed by both season rosters); placeholder negative roster ids (2017-2019) and 704 same-name successors with a new id are unresolved and never stitched (`cfb_identity_contract.json`).

**Research cutoff:** this audit read no 2026 file; the research loader rejects 2026 Week >= 5 (and treats Weeks 1-4 as burned diagnostic only).

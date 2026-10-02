# PIT@CLE T90 live rehearsal: NOT EXECUTED

The container rebooted (uptime 1 min at 22:56Z); the dispatcher daemon (started 19:29Z) died after its 19:33:04Z tick and was not running at the T90 cutoff
(2026-10-01T22:45:00Z). No snapshot was retrieved, no Phase 1 forecast and no frozen-v2 forecast exist for T90. The first tick after the reboot (22:56Z)
recorded T90 as MISSED_REAL_CUTOFF (see dispatch_ledger.jsonl). T24 was MISSED_REAL_CUTOFF (cutoff passed before the dispatcher existed). Nothing was backfilled.
The only real-data Phase 1 / v2 runs are dry runs (probe roots, not live evidence).

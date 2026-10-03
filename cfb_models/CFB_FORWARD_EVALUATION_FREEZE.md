# CFB forward-evaluation freeze (Week 5, 2026)

**From the 2026-10-03 pre-slate infrastructure patch until the Week 5 slate is fully graded: NO CFB predictive model, calibration, feature, eligibility threshold or market status may be changed based on Week 5 outcomes.** Operational bug fixes are allowed only if they do not use outcome knowledge. This keeps Week 5 clean forward evidence.

- `passing_yards` / `receiving_yards` stay suspended; nothing is un-suspended or rescued from Week 5 results.
- Forward evaluation uses ONLY the immutable first pregame ledger entry (original `logged_at`, `model_prob`, side, `model_source`); the refreshable live-board probability is never used.
- Canonical key: `season | week | game_id | market | entity | line` (entity = player_id for props, `GAME` for moneyline). Duplicates: the earliest valid pregame row wins and duplicates are reported in `docs/cfb_forward_status.json`.
- Rows logged at/after kickoff, or whose kickoff cannot be verified, are not forward evidence (counted, never deleted).
- Moneyline is graded at game level from completed ESPN final points; ties and cancelled / postponed / forfeited games are excluded, an unfinished game stays ungraded; no sportsbook line is used.
- 2026 forward results are observational: they produce no promotion / rejection decision and cannot trigger same-week rescue tuning.

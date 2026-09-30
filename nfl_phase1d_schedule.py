"""
NFL_PHASE1D_SCHEDULE  (Phase 1D)  -- kickoff / cutoff safety

The forecast cutoff is a function of the kickoff, and the kickoff comes from the schedule source, which can change (flex scheduling, postponement,
cancellation, neutral-site time changes). Rules (pre-registered in protocol Amendment G, implemented and tested here):

  1. Before a game is forecast the CURRENT schedule is retrieved and stored as an immutable snapshot (games.csv, market columns stripped). The kickoff
     is read from THAT snapshot, never from a cached or hard-coded value; cutoff = kickoff - 24h (T24) or - 90m (T90).
  2. The snapshot must have been retrieved at or before that cutoff. A retrieval after the cutoff means no forecast: SAFE_EXPLICIT_FAILURE("late_retrieval").
  3. The schedule ledger is append-only. Every distinct kickoff ever observed for a game is a REVISION (1, 2, ...). Nothing is edited.
  4. Forecast ids contain the cutoff, so a forecast made under a different kickoff has different ids; an earlier forecast is NEVER overwritten.
  5. Which forecast is the SCORED one for (game, horizon): the latest forecast whose cutoff equals (final kickoff - horizon) AND whose retrieval
     was <= that cutoff. If the kickoff moved EARLIER after a forecast, the old forecast (made at the old cutoff, i.e. later than the new cutoff)
     is invalid for primary scoring unless it was retrieved before the new cutoff; if the kickoff moved LATER, a forecast at the new cutoff is
     produced when the new cutoff arrives and supersedes the older one; if no valid forecast exists the game-horizon is reported MISSING (explicit
     failure), never filled.
  6. Postponed to a date outside the window / cancelled / absent from the current schedule: no forecast (SAFE_EXPLICIT_FAILURE with the reason); earlier
     forecasts stay in the log, flagged GAME_NOT_PLAYED_AS_SCHEDULED and excluded from primary scoring. The final kickoff is the kickoff of the last
     schedule revision before the game was played.
  7. Neutral-site time changes are ordinary kickoff changes (rules 3-5).
"""
import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1_store_lock as LK

UTC = timezone.utc
HORIZON_DELTA = {"T24": timedelta(hours=24), "T90": timedelta(minutes=90)}


def iso(ts):
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def parse_schedule(games_bytes, seasons=None):
    """{game_id: {season, week, home, away, kick}} for regular-season games of the given seasons (from schedule snapshot bytes)."""
    out = {}
    for r in csv.DictReader(io.StringIO(games_bytes.decode("utf-8"))):
        if r["game_type"] != "REG" or not r["season"].isdigit():
            continue
        if seasons and int(r["season"]) not in seasons:
            continue
        if not r.get("gameday"):
            continue
        out[r["game_id"]] = {"season": int(r["season"]), "week": int(r["week"]), "home": r["home_team"], "away": r["away_team"],
                             "kick": P1.kickoff_utc(r["gameday"], r["gametime"])}
    return out


def forecast_cutoff(kick, horizon):
    return kick - HORIZON_DELTA[horizon]


class ScheduleLedger:
    """Append-only revisions of every game's kickoff (and its status)."""

    def __init__(self, root, lock_timeout=60.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "schedule_ledger.jsonl"
        self.lock = LK.StoreLock(self.root / "schedule_ledger.lock", timeout=lock_timeout, bind_dir=self.root)

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def latest(self, game_id):
        rs = [r for r in self.rows() if r["game_id"] == game_id]
        return rs[-1] if rs else None

    def observe(self, game_id, kick, set_id, retrieval_ts, status="scheduled"):
        """Record an observation. Returns (row, changed): a new revision row is appended only when the kickoff or status differs from the latest."""
        with self.lock:
            cur = self.latest(game_id)
            kiso = iso(kick) if kick else None
            if cur and cur["kickoff"] == kiso and cur["status"] == status:
                return cur, False
            rev = (cur["revision"] + 1) if cur else 1
            change = None
            if cur and cur["kickoff"] and kiso:
                d = parse_iso(kiso) - parse_iso(cur["kickoff"])
                change = "moved_later" if d > timedelta(0) else "moved_earlier"
            elif cur and status != cur["status"]:
                change = f"status:{cur['status']}->{status}"
            row = {"game_id": game_id, "revision": rev, "kickoff": kiso, "status": status, "change": change, "set_id": set_id, "retrieval_ts": iso(retrieval_ts)}
            with open(self.path, "ab") as fh:
                fh.write(json.dumps(row, sort_keys=True).encode() + b"\n"); fh.flush(); os.fsync(fh.fileno())
            return row, True


def scored_forecast_ok(forecast, final_kick, horizon):
    """Rule 5: is this forecast valid for primary scoring given the FINAL kickoff?"""
    want = forecast_cutoff(final_kick, horizon)
    return parse_iso(forecast["cutoff"]) == want and parse_iso(forecast["retrieval_ts"]) <= want


def select_primary(forecasts, final_kick, horizon):
    """Among the stored forecasts of one (game, horizon): the latest valid for the final kickoff, or None (=> reported MISSING)."""
    ok = [f for f in forecasts if scored_forecast_ok(f, final_kick, horizon)]
    return max(ok, key=lambda f: f["retrieval_ts"]) if ok else None


def game_status_for_scoring(ledger_rows, played):
    """GAME_NOT_PLAYED_AS_SCHEDULED for postponed/cancelled games; else PLAYED."""
    last = ledger_rows[-1] if ledger_rows else None
    if last is not None and last["status"] in ("cancelled", "postponed", "absent_from_schedule") and not played:
        return "GAME_NOT_PLAYED_AS_SCHEDULED"
    return "PLAYED" if played else "NOT_YET_PLAYED"

"""2026 NFL verified kickoff revision from league/current team primary sources.

The 2026 Week 9 LV@SF kickoff in the older nfldata games.csv/foundation
snapshot is 16:05 Eastern, but NFL.com's CURRENT Week 9 schedule and the
Raiders' November 5 broadcast announcement both give 16:25 Eastern.
Do not silently accept a stale provider or rewrite an earlier T24 receipt.

The revision is exact team/season/week/date/old-clock constrained. Every
other kickoff passes through unchanged; any third contradictory time fails
closed, requiring a fresh verification. All changes are prospective and
restricted to as-of schedule SOURCE presentation, not numerical player
projections.
"""
from __future__ import annotations

from datetime import datetime,timezone
from zoneinfo import ZoneInfo

VERSION="NFL_WEEK9_OFFICIAL_KICKOFF_REVISION_V1"
EASTERN=ZoneInfo("America/New_York")
REVISION={
    "season":2026,"week":9,"home_team":"SF","away_team":"LV",
    "date":"2026-11-08",
    "stale_local_time":"16:05","verified_local_time":"16:25",
    "stale_kickoff_utc":"2026-11-08T21:05:00Z",
    "verified_kickoff_utc":"2026-11-08T21:25:00Z",
    "primary_sources":[
        "https://www.nfl.com/schedules/2026/by-week/week-9",
        "https://www.raiders.com/news/how-to-watch-listen-to-and-livestream-raiders-at-49ers"
    ],
    "verification_note":"NFL current Week 9 schedule and Raiders Nov 5 team broadcast guide agree on 4:25 ET; the prior season fixture CSV carried 4:05 ET.",
}


def normalize_utc(value):
    ts=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if ts.tzinfo is None:
        raise ValueError("NFL_OFFICIAL_KICKOFF_NAIVE_SOURCE_TIME")
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00","Z")


def _is_target(row):
    return (str(row.get("season"))=="2026" and str(row.get("week"))=="9"
            and row.get("home_team")=="SF" and row.get("away_team")=="LV")


def fix_fixture_clock(row):
    """Read-only upstream row -> corrected copy, evidence, or reject conflict."""
    row=dict(row)
    if not _is_target(row):
        return row,None
    kickoff=row.get("kickoff_utc")
    if kickoff is not None:
        old=normalize_utc(kickoff)
        if old not in (REVISION["stale_kickoff_utc"],REVISION["verified_kickoff_utc"]):
            raise ValueError("NFL_OFFICIAL_KICKOFF_THIRD_CONFLICTING_SOURCE_TIME")
        row["kickoff_utc"]=REVISION["verified_kickoff_utc"]
    else:
        day=str(row.get("gameday") or "")
        if day!=REVISION["date"]:
            raise ValueError("NFL_OFFICIAL_KICKOFF_CONFLICTING_GAME_DATE")
        original=str(row.get("gametime") or "")
        if original not in (REVISION["stale_local_time"],REVISION["verified_local_time"]):
            raise ValueError("NFL_OFFICIAL_KICKOFF_THIRD_CONFLICTING_SOURCE_TIME")
        row["gametime"]=REVISION["verified_local_time"]
    return row,{**REVISION,"original_source_kickoff":kickoff or original,
                "already_correct":(old==REVISION["verified_kickoff_utc"]
                                   if kickoff is not None else original==REVISION["verified_local_time"])}


def reconcile_db_schedule_rows(rows,season,week):
    """[(home,away,ISO kickoff)] preserving order; no fictional games added."""
    adjusted=[];proof=[]
    for home,away,kickoff in rows:
        rec,revision=fix_fixture_clock({
            "season":season,"week":week,"home_team":home,"away_team":away,
            "kickoff_utc":kickoff})
        adjusted.append((home,away,rec["kickoff_utc"]))
        if revision:proof.append(revision)
    return adjusted,proof


def reconcile_shadow_source_row(row):
    """A future live games.csv snapshot only. Original raw provider bytes live
    separately with their SHA256 and must never be rewritten or backdated."""
    return fix_fixture_clock(row)

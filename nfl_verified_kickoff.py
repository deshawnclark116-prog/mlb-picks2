"""Primary-source NFL flex corrections for stale nflverse/foundation schedules.

These are strictly fixture-specific metadata overrides, NOT changes to player
projections, probabilities, original pregame capture times, or frozen history.
Only an exact source-stale or league-verified time is accepted; a third value
fails closed.  The same function is used by the production schedule manifest,
its independent validator and the Phase1 shadow scheduler sanitizer.

A kickoff revision changes the *future* T90 snapshot key. It never converts
a previous T24 snapshot captured for a different kickoff into a valid forecast.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

VERSION="NFL_PRIMARY_SOURCE_VERIFIED_FLEX_REVISIONS_V2"
EASTERN=ZoneInfo("America/New_York")

# Historic Week 9 entry retained for backwards compatibility / original tests.
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
    "verification_note":"Current NFL Week 9 schedule and Raiders guide agreed on 4:25 ET; earlier foundation CSV carried 4:05 ET."
}

# NFL November 9, 2026 flex notice. Both clubs and NFL Week 11 schedule
# independently confirm that these two games EXCHANGED kickoff windows.
WEEK11_REVISIONS=(
    {
        "season":2026,"week":11,"home_team":"PHI","away_team":"PIT",
        "date":"2026-11-22",
        "stale_local_time":"16:25","verified_local_time":"13:00",
        "stale_kickoff_utc":"2026-11-22T21:25:00Z",
        "verified_kickoff_utc":"2026-11-22T18:00:00Z",
        "primary_sources":[
            "https://media.nfl.com/news-and-releases/week-11-schedule-change--jaguars-giants-to-4-25-p-m--et--steeler",
            "https://www.nfl.com/schedules/2026/by-week/week-11",
            "https://www.steelers.com/news/steelers-eagles-game-flexed"
        ],
        "verification_note":"NFL Week 11 flex moved Steelers at Eagles to 1:00 PM Eastern November 22."
    },
    {
        "season":2026,"week":11,"home_team":"NYG","away_team":"JAX",
        "date":"2026-11-22",
        "stale_local_time":"13:00","verified_local_time":"16:25",
        "stale_kickoff_utc":"2026-11-22T18:00:00Z",
        "verified_kickoff_utc":"2026-11-22T21:25:00Z",
        "primary_sources":[
            "https://media.nfl.com/news-and-releases/week-11-schedule-change--jaguars-giants-to-4-25-p-m--et--steeler",
            "https://www.nfl.com/schedules/2026/by-week/week-11",
            "https://www.giants.com/news/week-11-flex-scheduling-giants-vs-jaguars-moves-to-4-25-p-m-et-on-nov-22"
        ],
        "verification_note":"NFL Week 11 flex moved Jaguars at Giants to 4:25 PM Eastern November 22."
    }
)

REVISIONS=(REVISION,*WEEK11_REVISIONS)

def normalize_utc(value):
    ts=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if ts.tzinfo is None:
        raise ValueError("NFL_OFFICIAL_KICKOFF_NAIVE_SOURCE_TIME")
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def _match(row,revision):
    return (
        str(row.get("season"))==str(revision["season"])
        and str(row.get("week"))==str(revision["week"])
        and row.get("home_team")==revision["home_team"]
        and row.get("away_team")==revision["away_team"]
    )

def _is_target(row):
    """Legacy Week9 helper retained for downstream compatibility."""
    return _match(row,REVISION)

def _revision_for(row):
    matches=[revision for revision in REVISIONS if _match(row,revision)]
    if len(matches)>1:
        raise ValueError("NFL_OFFICIAL_KICKOFF_DUPLICATED_REVISION")
    return matches[0] if matches else None

def fix_fixture_clock(row):
    """Read-only provider row -> verified fixture metadata, proof, or strict error.

    The target includes full team pairing, season and week. A matched source
    must have the correct date and one of exactly two known kickoff instants.
    Never infer a revision from the clock alone.
    """
    row=dict(row)
    revision=_revision_for(row)
    if revision is None:
        return row,None
    kickoff=row.get("kickoff_utc")
    if kickoff is not None:
        old=normalize_utc(kickoff)
        if old not in (revision["stale_kickoff_utc"],revision["verified_kickoff_utc"]):
            raise ValueError("NFL_OFFICIAL_KICKOFF_THIRD_CONFLICTING_SOURCE_TIME")
        row["kickoff_utc"]=revision["verified_kickoff_utc"]
        original=kickoff
        already=old==revision["verified_kickoff_utc"]
    else:
        day=str(row.get("gameday") or "")
        if day!=revision["date"]:
            raise ValueError("NFL_OFFICIAL_KICKOFF_CONFLICTING_GAME_DATE")
        original=str(row.get("gametime") or "")
        # nfldata gametime is HH:MM 24h EASTERN and already uses this form.
        if original not in (revision["stale_local_time"],revision["verified_local_time"]):
            raise ValueError("NFL_OFFICIAL_KICKOFF_THIRD_CONFLICTING_SOURCE_TIME")
        row["gametime"]=revision["verified_local_time"]
        already=original==revision["verified_local_time"]
    return row,{**revision,"original_source_kickoff":original,"already_correct":already}

def reconcile_db_schedule_rows(rows,season,week):
    """[(home,away,UTC)] -> new metadata rows and source-verification receipts.

    Order preserved. Repeated use is idempotent and records already_correct,
    so a newer provider fixture cannot be mistakenly treated as a novel flex.
    """
    adjusted=[];proof=[]
    for home,away,kickoff in rows:
        rec,revision=fix_fixture_clock({
            "season":season,"week":week,"home_team":home,"away_team":away,
            "kickoff_utc":kickoff})
        adjusted.append((home,away,rec["kickoff_utc"]))
        if revision:
            proof.append(revision)
    return adjusted,proof

def reconcile_shadow_source_row(row):
    """Current schedule only: retain original downloaded bytes and timestamp
    as separate proof; never backdate or edit CAS snapshots already frozen."""
    return fix_fixture_clock(row)

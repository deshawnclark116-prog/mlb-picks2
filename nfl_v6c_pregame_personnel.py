"""NFL V6C pre-kickoff personnel assertions (RESEARCH ONLY).

These are independently captured source assertions, not an automated article
reader or certification that the source text supports the assertion. Source URLs
and timestamps are validated, and only future games with an immutable capture
clock may use the evidence. An absent QB assertion is UNKNOWN, never "the most
recent quarterback must start". Every adjustment is scoped to an exact game,
team and player ID. No historical after-the-fact starting QB feed is allowed.

Schema:
{
 "schema": "NFL_V6C_PREGAME_PERSONNEL_EVIDENCE_A",
 "captured_at_utc": "2026-11-12T20:00:00Z",
 "game_id": "2026_10_WAS_NYG",
 "kickoff_utc": "2026-11-13T01:15:00Z",
 "teams": [
  {"team": "NYG",
   "starter_qb": {"player_id": "...", "player": "...",
     "sources": [{"url": "https://www.nygiants.com/news/...",
                  "published_at_utc": "2026-11-12T18:00:00Z"}]},
   "confirmed_out": [
     {"player_id": "...", "player": "...",
      "sources": [{"url": "https://www.giants.com/news/...",
                   "published_at_utc": "2026-11-12T17:00:00Z"}]}
   ]}
 ]
}
No role change is made on a team lacking source evidence. An assertion is
NOT proof of the source content, a game-day active list, or future health.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nfl_v2_role_evidence_registry import official_url
from nfl_v6_joint_opportunity_shadow import utc

SCHEMA = "NFL_V6C_PREGAME_PERSONNEL_EVIDENCE_A"


def _identity(row):
    if not isinstance(row, dict) or not row.get("player_id") or not row.get("player"):
        raise ValueError("PERSONNEL_ASSERTION_MISSING_PLAYER_ID_OR_NAME")
    return str(row["player_id"])


def _sources(row, *, captured, kickoff):
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("PERSONNEL_ASSERTION_MISSING_PRIMARY_SOURCE")
    for source in sources:
        if not isinstance(source, dict) or not official_url(source.get("url")):
            raise ValueError("PERSONNEL_SOURCE_NOT_OFFICIAL_TEAM_OR_NFL_NEWS")
        published = utc(source["published_at_utc"])
        if published > captured or published >= kickoff:
            raise ValueError("PERSONNEL_SOURCE_LATE_OR_AFTER_KICKOFF")


def read_personnel(path, game, cutoff, *, captured_at_utc):
    """Validate a pregame-only source receipt and return per-team adjustments.

    This reader never fetches newer sources or infers a starter from usage.
    The snapshot must be captured no later than the model's information cutoff.
    """
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise ValueError("PERSONNEL_WRONG_SCHEMA")
    captured = utc(doc["captured_at_utc"])
    model_capture = utc(captured_at_utc)
    cut = utc(cutoff)
    kickoff = utc(game["kickoff_utc"])
    if not (captured <= model_capture <= cut < kickoff):
        raise ValueError("PERSONNEL_CAPTURE_OUTSIDE_PREGAME_CUTOFF")
    if doc.get("game_id") != game["game_id"] or utc(doc["kickoff_utc"]) != kickoff:
        raise ValueError("PERSONNEL_GAME_ID_OR_KICKOFF_MISMATCH")
    required = {game["home_team"], game["away_team"]}
    teams = doc.get("teams", [])
    if not isinstance(teams, list):
        raise ValueError("PERSONNEL_TEAMS_NOT_LIST")
    by_team = {}
    for item in teams:
        team = item.get("team")
        if team not in required or team in by_team:
            raise ValueError("PERSONNEL_DUPLICATE_OR_FOREIGN_TEAM")
        qb = item.get("starter_qb")
        starter = None
        if qb is not None:
            starter = _identity(qb)
            _sources(qb, captured=captured, kickoff=kickoff)
        inactive = set()
        for out in item.get("confirmed_out", []):
            player_id = _identity(out)
            _sources(out, captured=captured, kickoff=kickoff)
            if player_id in inactive:
                raise ValueError("PERSONNEL_DUPLICATE_OUT_ID")
            inactive.add(player_id)
        if starter and starter in inactive:
            raise ValueError("PERSONNEL_STARTER_MARKED_OUT")
        by_team[team] = {
            "starter_id": starter,
            "inactive_ids": inactive,
            "source_asserted_starter": starter is not None,
            "source_asserted_out_count": len(inactive),
        }
    for team in required - set(by_team):
        by_team[team] = {
            "starter_id": None,
            "inactive_ids": set(),
            "source_asserted_starter": False,
            "source_asserted_out_count": 0,
        }
    return {
        "teams": by_team,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "captured_at_utc": captured.isoformat(),
        "role_assertions_are_not_automatically_source_verified": True,
        "game_id": game["game_id"],
    }

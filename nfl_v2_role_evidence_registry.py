"""League-wide pregame NFL starter-role assertions for an UNPROMOTED challenger.

No inferred starters from historical usage, no attribution of a new scenario
to an old T24/T90 forecast. Every role transfer requires a source assertion
with exact teams, kickoff, player identities, and timestamped official news.
The supporting news text is human-reviewed source evidence, not a machine-
verified article download. Leave unknown teams uncorrected and unbettable.
"""
from __future__ import annotations
import re
from urllib.parse import urlparse
from datetime import datetime,timezone

SCHEMA="NFL_V2_LEAGUEWIDE_VERIFIED_STARTER_ASSERTIONS_V1"
LEGACY="nfl-confirmed-starting-quarterback-source-evidence-v1"

OFFICIAL_HOSTS=set("""
nfl.com 49ers.com chicagobears.com bengals.com buffalobills.com
denverbroncos.com clevelandbrowns.com buccaneers.com azcardinals.com
chargers.com chiefs.com colts.com commanders.com dallascowboys.com
miamidolphins.com philadelphiaeagles.com atlantafalcons.com
nygiants.com jaguars.com newyorkjets.com detroitlions.com
packers.com panthers.com patriots.com raiders.com therams.com
baltimoreravens.com neworleanssaints.com seahawks.com steelers.com
houstontexans.com titansonline.com vikings.com
""".split())


def time(s):
    t=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if t.tzinfo is None:raise ValueError("STARTER_EVIDENCE_TIMESTAMP_NOT_UTC")
    return t.astimezone(timezone.utc)


def norm(s):
    return " ".join(re.sub(r"[^a-z ]"," ",str(s or "").casefold()).split())


def official_url(link):
    if not isinstance(link,str):return False
    u=urlparse(link)
    host=(u.hostname or "").lower()
    if host.startswith("www."):host=host[4:]
    return u.scheme=="https" and u.username is None and u.password is None and (
        host in OFFICIAL_HOSTS) and u.path.startswith(("/news/","/articles/"))


def targets(board,evidence,*,now,horizons=("T24","T90")):
    if now.tzinfo is None:raise ValueError("NAIVE_SCENARIO_CLOCK")
    now=now.astimezone(timezone.utc)
    if board.get("schema")!="nfl-new-engine-shadow-v1" or board.get(
            "status",{}).get("publication_state")!="PUBLISHED":
        raise ValueError("NO_PUBLISHED_V2_ORIGINAL")
    if evidence.get("schema") not in (SCHEMA,LEGACY):
        raise ValueError("UNKNOWN_STARTER_SOURCE_SCHEMA")
    games={g["game_id"]:g for g in board["games"]}
    fields=board["row_fields"]
    required={"game_id","team","player","pos","outcome","p_active"}
    if len(fields)!=len(set(fields)) or not required.issubset(fields):
        raise ValueError("INVALID_PUBLISHED_PLAYER_SCHEMA")
    idx={v:i for i,v in enumerate(fields)}
    output=[];seen=set()
    for a in evidence.get("assertions",[]):
        gid=a["game_id"];team=a["team"];opp=a["opponent"]
        if gid not in games:raise ValueError("EVIDENCE_GAME_NOT_ON_BOARD")
        game=games[gid]
        if {team,opp}!={game["home"],game["away"]} or team==opp:
            raise ValueError("EVIDENCE_TEAM_OPPONENT_MISMATCH")
        kickoff=time(game["kickoff_utc"])
        if time(a["kickoff_utc"])!=kickoff or now>=kickoff:
            raise ValueError("EVIDENCE_KICKOFF_CHANGED_OR_PASSED")
        starter=a["verified_starter"]
        out=a.get("out_players",[])
        if not starter or not out or len(set(map(norm,out)))!=len(out):
            raise ValueError("MISSING_OR_DUPLICATE_QB_ROLE_EVIDENCE")
        if norm(starter) in set(map(norm,out)):
            raise ValueError("STARTER_AND_OUT_QB_CONFLICT")
        sources=a.get("official_sources",[])
        if not sources:raise ValueError("MISSING_OFFICIAL_PRIMARY_SOURCE")
        support=set()
        newest=datetime(1970,1,1,tzinfo=timezone.utc)
        for s in sources:
            if not official_url(s.get("url")):
                raise ValueError("UNTRUSTED_QB_STARTER_NEWS_DOMAIN")
            posted=time(s["published_at_utc"])
            if posted>now or posted>=kickoff:
                raise ValueError("POST_GAME_OR_FUTURE_QB_STARTER_NEWS")
            newest=max(newest,posted)
            support.update(map(str,s.get("supports") or []))
        if not any(norm(starter)+" will start" in norm(x) for x in support):
            raise ValueError("STARTER_NOT_AFFIRMED_BY_SOURCE_ASSERTION")
        for o in out:
            if not any(norm(o) in norm(x) and ("out" in norm(x) or "injured reserve" in norm(x))
                       for x in support):
                raise ValueError("MISSING_OFFICIAL_OUT_QB_ASSERTION")
        for h in horizons:
            meta=game.get("horizons",{}).get(h)
            if not meta:continue
            cutoff=time(meta["cutoff"])
            k=(gid,team,h)
            if k in seen:raise ValueError("CONFLICTING_VERIFIED_QB_STARTERS_SAME_GAME")
            seen.add(k)
            rows=[r for r in board["forecasts"].get(h,[])
                  if r[idx["game_id"]]==gid and r[idx["team"]]==team
                  and r[idx["pos"]]=="QB" and r[idx["outcome"]]=="pass_yds"]
            named=[r for r in rows if norm(r[idx["player"]])==norm(starter)]
            if len(named)!=1:
                raise ValueError("MISSING_OR_AMBIGUOUS_NAMED_QB_SNAPSHOT")
            if any(len([r for r in rows if norm(r[idx["player"]])==norm(o)])!=1
                   for o in out):
                raise ValueError("MISSING_OR_AMBIGUOUS_OUT_QB_SNAPSHOT")
            output.append({
                "game_id":gid,"team":team,"opponent":opp,
                "season":int(gid.split("_")[0]),"week":int(gid.split("_")[1]),
                "horizon":h,"official_verified_starter":starter,
                "original_model_p_active":named[0][idx["p_active"]],
                "original_forecast_cutoff_utc":meta["cutoff"],
                "kickoff_utc":game["kickoff_utc"],
                "official_sources":sources,
                "latest_official_source_utc":newest.isoformat(),
                "source_was_known_at_original_cutoff":newest<=cutoff,
                "new_pregame_scenario_generated_after_original_cutoff":now>cutoff,
                "must_not_relabel_as_original_T24_T90":True,
                "identity_verified_by":"exact player/team/game/horizon in published shadow"
            })
    return output

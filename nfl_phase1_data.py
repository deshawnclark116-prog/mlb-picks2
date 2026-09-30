"""
NFL_PHASE1_DATA  (Phase 1A, shadow research)

The as-of data contract every Phase 1A feature goes through, and the
chronological replay that turns nflverse files into pre-game forecast rows.

AS-OF CONTRACT
--------------
Every input record carries:
  source            nflverse release name
  season/week/game  game it belongs to
  info_ts           information-available timestamp (UTC). If the source has
                    no timestamp, a documented assumption (AS_OF_ASSUMPTIONS)
                    sets it and the assumption id is logged
  retrieval_ts      when the file was downloaded (file mtime)
A forecast has a cutoff (T-24h = kickoff - 24h, T-90m = kickoff - 90m). A record
with info_ts > cutoff is unavailable. Every forecast row stores the maximum
info_ts it used; invariants assert it is <= cutoff < kickoff.

No sportsbook numbers are read. (games.csv spread/total/moneyline columns are
never touched.)
"""
import csv
import gzip
import math
import os
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

SEASONS = [2022, 2023, 2024, 2025, 2026]
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
RZ, GL = 20, 5
OFF_POS = ("QB", "RB", "FB", "WR", "TE")
DEF_GRP = {"DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
           "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB"}
T24 = timedelta(hours=24)
T90 = timedelta(minutes=90)
MARKET_COLUMNS = ("spread_line", "total_line", "away_moneyline", "home_moneyline", "away_spread_odds",
                  "home_spread_odds", "under_odds", "over_odds")

AS_OF_ASSUMPTIONS = {
    "A1_completed_game": "stats, snap counts, play-by-play and participation of a game are treated as available "
                         "from kickoff + 24h (conservative; nflverse updates nightly).",
    "A2_injury_report": "nflverse injuries rows carry NO timestamp. Each row is treated as the club's final weekly "
                        "report (game status + last practice status). Assumed info_ts = 16:00 ET two calendar days "
                        "before kickoff day (Friday for Sunday games, Saturday for Monday, Tuesday for Thursday, "
                        "Thursday for Saturday). Real NFL final reports are published no later than this for all "
                        "regular slots, so the row precedes the T-24h cutoff. RISK: if nflverse rows absorb later "
                        "supplemental changes (e.g. a Saturday downgrade), T-24h features contain slightly late information.",
    "A3_weekly_roster": "weekly_rosters status (ACT/INA/RES/DEV/CUT) is a game-day snapshot (Phase 0B: 0.0% of 5,395 "
                        "INA players had snaps). Assumed info_ts = kickoff - 90 minutes (inactives deadline). Usable "
                        "at T-90m only; T-24h uses the previous week's status (info_ts = previous game's kickoff - 90m).",
    "A4_depth_chart_2025_2026": "depth_charts rows carry a snapshot timestamp 'dt'; info_ts = dt. The latest snapshot "
                                "per team with dt <= cutoff is used.",
    "A5_depth_chart_2024": "depth_charts_2024 is week-labelled with no publication time. NOT used by any Phase 1A "
                           "model (publication semantics not established).",
    "A6_players_file": "draft round/pick and rookie season (players.csv) are known before the season.",
    "A7_schedule": "home/away, kickoff time and rest days (games.csv) are known in advance. No sportsbook column is read.",
}


def kickoff_utc(gameday, gametime):
    hh, mm = (gametime or "13:00").split(":")[:2]
    d = datetime.strptime(gameday, "%Y-%m-%d").replace(hour=int(hh), minute=int(mm), tzinfo=ET)
    return d.astimezone(UTC)


def injury_info_ts(kick):
    k_et = kick.astimezone(ET)
    d = (k_et - timedelta(days=2)).replace(hour=16, minute=0, second=0, microsecond=0)
    return d.astimezone(UTC)


def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def norm(n):
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", n or "")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", n)
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class Data:
    """Raw per-game facts, keyed by (season, week, team)."""

    def __init__(self, data_dir):
        d = Path(data_dir)
        self.dir = d
        self.retrieval_ts = {}
        for f in d.iterdir():
            self.retrieval_ts[f.name] = datetime.fromtimestamp(os.path.getmtime(os.path.realpath(f)), UTC).isoformat()
        # schedule (no sportsbook columns read)
        self.game = {}                 # (s, w, team) -> ctx
        self.team_games = defaultdict(list)
        for r in csv.DictReader(open(d / "games.csv", newline="", encoding="utf-8")):
            if r["game_type"] != "REG" or int(r["season"]) not in SEASONS:
                continue
            s, w = int(r["season"]), int(r["week"])
            kick = kickoff_utc(r["gameday"], r["gametime"])
            final = r["result"] not in ("", "NA")
            for team, opp, home, rest, orest in ((r["home_team"], r["away_team"], 1, r["home_rest"], r["away_rest"]),
                                                  (r["away_team"], r["home_team"], 0, r["away_rest"], r["home_rest"])):
                self.game[(s, w, team)] = {"opp": opp, "home": home, "kick": kick, "game_id": r["game_id"], "final": final,
                                           "rest": fnum(rest), "opp_rest": fnum(orest),
                                           "coach": r["home_coach"] if home else r["away_coach"]}
                self.team_games[team].append((kick, s, w))
        for t in self.team_games:
            self.team_games[t].sort()
        # rosters (game-day snapshot) and pfr -> gsis map
        self.roster = {}               # (s, w, gsis) -> dict
        self.pfr2gsis = {}
        self.roster_by_team = defaultdict(list)   # (s, w, team) -> [gsis]
        for s in SEASONS:
            p = d / f"roster_weekly_{s}.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r.get("game_type", "REG") != "REG" or not r.get("week"):
                    continue
                w = int(r["week"])
                self.roster[(s, w, r["gsis_id"])] = {"team": r["team"], "status": r["status"], "pos": r["position"],
                                                    "name": norm(r["full_name"])}
                self.roster_by_team[(s, w, r["team"])].append(r["gsis_id"])
                if r.get("pfr_id"):
                    self.pfr2gsis[r["pfr_id"]] = r["gsis_id"]
        self.players = {}
        for r in csv.DictReader(open(d / "players.csv", newline="", encoding="utf-8")):
            self.players[r["gsis_id"]] = {"rookie": fnum(r.get("rookie_season")), "round": fnum(r.get("draft_round")),
                                          "pick": fnum(r.get("draft_pick")), "pos": r.get("position")}
            if r.get("pfr_id"):
                self.pfr2gsis.setdefault(r["pfr_id"], r["gsis_id"])
        # player stats rows (offense)
        self.stat = defaultdict(dict)  # (s, w, team) -> gsis -> row
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"stats_player_week_{s}.csv", newline="", encoding="utf-8")):
                if r.get("season_type", "REG") != "REG":
                    continue
                g = lambda k: fnum(r.get(k)) or 0.0
                self.stat[(s, int(r["week"]), r["team"])][r["player_id"]] = {
                    "name": norm(r["player_display_name"]), "pos": r["position"], "car": g("carries"), "tgt": g("targets"),
                    "rec": g("receptions"), "att": g("attempts")}
        # snap counts
        self.osnap = defaultdict(dict)   # (s, w, team) -> gsis -> (snaps, pct)
        self.dsnap = defaultdict(dict)   # (s, w, team) -> pfr -> {gsis, name, grp, snaps, pct}
        self.team_def_snaps = {}
        self.team_off_snaps = {}
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"snap_counts_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") != "REG":
                    continue
                k = (s, int(r["week"]), r["team"])
                gid = self.pfr2gsis.get(r["pfr_player_id"])
                os_, op = fnum(r["offense_snaps"]) or 0.0, fnum(r["offense_pct"]) or 0.0
                ds, dp = fnum(r["defense_snaps"]) or 0.0, fnum(r["defense_pct"]) or 0.0
                if os_ > 0:
                    if gid is None:
                        gid = next((g for g, x in self.stat.get(k, {}).items() if x["name"] == norm(r["player"])), None)
                    if gid:
                        self.osnap[k][gid] = (os_, op)
                    if op > 0:
                        self.team_off_snaps[k] = max(self.team_off_snaps.get(k, 0), round(os_ / op))
                if ds > 0 and r.get("position") in DEF_GRP:
                    self.dsnap[k][r["pfr_player_id"]] = {"gsis": gid, "name": norm(r["player"]), "grp": DEF_GRP[r["position"]],
                                                         "snaps": ds, "pct": dp}
                    if dp > 0:
                        self.team_def_snaps[k] = max(self.team_def_snaps.get(k, 0), round(ds / dp))
        # play-by-play components
        self.tg = defaultdict(lambda: defaultdict(float))
        self.pp = defaultdict(lambda: defaultdict(float))
        dropbacks = {}
        for s in SEASONS:
            with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                        continue
                    if r.get("two_point_attempt") == "1" or r.get("qb_kneel") == "1" or r.get("qb_spike") == "1":
                        continue
                    w = int(r["week"]); k = (s, w, r["posteam"])
                    yl = fnum(r["yardline_100"])
                    rz, gl = yl is not None and yl <= RZ, yl is not None and yl <= GL
                    t = self.tg[k]
                    t["plays"] += 1
                    db = r["qb_dropback"] == "1"
                    if db:
                        t["dropbacks"] += 1
                        dropbacks[(r["game_id"], r["play_id"])] = k
                    if rz:
                        t["rz_plays"] += 1
                    if gl:
                        t["gl_plays"] += 1
                    rid, qid, pid = r["rusher_player_id"], r["receiver_player_id"], r["passer_player_id"]
                    is_pass = r["pass_attempt"] == "1" and r["sack"] != "1"
                    if r["rush_attempt"] == "1" and rid:
                        t["rushes"] += 1
                        if r.get("qb_scramble") != "1":
                            t["rushes_designed"] += 1
                        p = self.pp[(s, w, rid)]
                        p["car"] += 1
                        if rz:
                            t["rz_rushes"] += 1; p["rz_car"] += 1
                        if gl:
                            t["gl_rushes"] += 1; p["gl_car"] += 1
                    if is_pass and pid:
                        self.pp[(s, w, pid)]["att"] += 1
                    if is_pass and qid:
                        t["targets"] += 1
                        p = self.pp[(s, w, qid)]
                        p["tgt"] += 1
                        if rz:
                            t["rz_targets"] += 1; p["rz_tgt"] += 1
                    if r["pass_touchdown"] == "1" or r["rush_touchdown"] == "1":
                        t["off_td"] += 1
        self.has_routes = set()
        self.part_games = set()
        for s in SEASONS:
            p = d / f"participation_{s}.csv"
            if not p.exists():
                continue
            self.has_routes.add(s)
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                k = dropbacks.get((r["nflverse_game_id"], r["play_id"]))
                if k is None:
                    continue
                self.part_games.add(k)
                for g in (r["offense_players"] or "").split(";"):
                    if g:
                        self.pp[(k[0], k[1], g)]["routes"] += 1
        # injuries (week-labelled; assumption A2)
        self.inj = {}
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"injuries_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") == "REG":
                    self.inj[(s, int(r["week"]), r["gsis_id"])] = (r["report_status"] or "", r["practice_status"] or "", r["team"])
        # depth charts 2025+ (timestamped)
        self.depth = defaultdict(list)   # team -> sorted [(dt, {gsis: best_rank_by_posabb})]
        for s in (2025, 2026):
            p = d / f"depth_charts_{s}.csv"
            if not p.exists():
                continue
            snaps = defaultdict(dict)
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if not r.get("dt") or not r.get("gsis_id"):
                    continue
                dt = datetime.strptime(r["dt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                key = (r["team"], dt)
                rk = fnum(r["pos_rank"]) or 9.0
                ab = r["pos_abb"]
                cur = snaps[key].get(r["gsis_id"])
                if cur is None or rk < cur[1]:
                    snaps[key][r["gsis_id"]] = (ab, rk)
            for (team, dt), m in snaps.items():
                self.depth[team].append((dt, m))
        for t in self.depth:
            self.depth[t].sort(key=lambda x: x[0])
        self.depth_times = {t: [x[0] for x in v] for t, v in self.depth.items()}

    def depth_at(self, team, cutoff):
        ts = self.depth_times.get(team)
        if not ts:
            return None, None
        i = bisect_right(ts, cutoff) - 1
        if i < 0:
            return None, None
        return self.depth[team][i][0], self.depth[team][i][1]

    def played_off(self, key, gid):
        return gid in self.osnap.get(key, {}) or any(self.stat.get(key, {}).get(gid, {}).get(c, 0) > 0 for c in ("car", "tgt", "att"))


# ------------------------------------------------------------------ replay
SHARE_KEYS = ("car_sh", "tgt_sh", "route_rt", "att_sh", "rz_car_sh", "rz_tgt_sh", "snap")
TEAM_KEYS = ("plays", "dropbacks", "rushes", "rushes_designed", "targets", "rz_plays", "gl_plays", "rz_rushes",
             "rz_targets", "gl_rushes", "off_td")


def team_actual(D, key):
    t = D.tg.get(key, {})
    out = {k: float(t.get(k, 0.0)) for k in TEAM_KEYS}
    out["def_snaps"] = float(D.team_def_snaps.get(key, 0.0))
    return out


def player_game_record(D, key, gid, T):
    s, w, team = key
    p = D.pp.get((s, w, gid), {})
    st = D.stat.get(key, {}).get(gid)
    osn = D.osnap.get(key, {}).get(gid)
    routes = p.get("routes", 0.0) if key in D.part_games else None
    rec = {"s": s, "w": w, "team": team, "kick": D.game[key]["kick"], "played": D.played_off(key, gid),
           "pos": (st or {}).get("pos") or (D.roster.get((s, w, gid)) or {}).get("pos"),
           "car": p.get("car", 0.0), "tgt": p.get("tgt", 0.0), "att": p.get("att", 0.0),
           "rz_car": p.get("rz_car", 0.0), "rz_tgt": p.get("rz_tgt", 0.0), "gl_car": p.get("gl_car", 0.0),
           "routes": routes, "snaps": osn[0] if osn else 0.0, "snap_pct": osn[1] if osn else 0.0}
    rec["car_sh"] = rec["car"] / T["rushes"] if T["rushes"] else None
    rec["tgt_sh"] = rec["tgt"] / T["targets"] if T["targets"] else None
    rec["att_sh"] = rec["att"] / T["dropbacks"] if T["dropbacks"] else None
    rec["route_rt"] = routes / T["dropbacks"] if routes is not None and T["dropbacks"] else None
    rec["rz_car_sh"] = rec["rz_car"] / T["rz_rushes"] if T["rz_rushes"] else None
    rec["rz_tgt_sh"] = rec["rz_tgt"] / T["rz_targets"] if T["rz_targets"] else None
    rec["snap"] = rec["snap_pct"]
    return rec


def build(D, depth_universe=False):
    """Chronological replay. For each team-game (in kickoff order) returns a
    unit with pre-game candidate rows (T-24 and T-90 information) and actuals.
    Histories only contain games whose A1 info_ts <= the T-24 cutoff."""
    games = sorted({(c["kick"], s, w, t) for (s, w, t), c in D.game.items() if (s, w, t) in D.stat})
    ph = defaultdict(list)      # gsis -> game records (only games with a row for him)
    dh = defaultdict(list)      # pfr -> defensive records
    th = defaultdict(list)      # team -> team offense records
    ta = defaultdict(list)      # team -> what opponents did vs this defense
    pending = list(games)       # games not yet absorbed, in kickoff order
    ptr = 0
    units = []
    for kick, s, w, team in games:
        key = (s, w, team); ctx = D.game[key]
        c24 = kick - T24; c90 = kick - T90
        # absorb every game whose info_ts (kickoff + 24h) <= T-24 cutoff
        while ptr < len(pending) and pending[ptr][0] + timedelta(hours=24) <= c24:
            _, ps, pw, pt = pending[ptr]
            absorb(D, (ps, pw, pt), ph, dh, th, ta)
            ptr += 1
        last_info = pending[ptr - 1][0] + timedelta(hours=24) if ptr else None
        prev_games = [g for g in D.team_games[team] if g[0] < kick]
        recent = {(g[1], g[2]) for g in prev_games[-3:]}
        prev_wk = (prev_games[-1][1], prev_games[-1][2]) if prev_games else None
        prev_kick = prev_games[-1][0] if prev_games else None
        # candidates: history with this team in its last 3 games, or on last week's roster (ACT) at an offensive position
        cands = set()
        for gid, h in ph.items():
            if h and h[-1]["team"] == team and (h[-1]["s"], h[-1]["w"]) in recent:
                cands.add(gid)
        if prev_wk:
            for gid in D.roster_by_team.get((prev_wk[0], prev_wk[1], team), []):
                r = D.roster.get((prev_wk[0], prev_wk[1], gid))
                if r and r["status"] == "ACT" and r["pos"] in OFF_POS:
                    cands.add(gid)
        dts, dmap = D.depth_at(team, c24)
        _, dmap_prev = D.depth_at(team, prev_kick - T24) if prev_kick else (None, None)
        if depth_universe and dmap:
            # roster-universe extension: skill players listed on the team's latest timestamped depth chart (snapshot <= T-24h)
            # who already have >= 3 prior game rows (arrivals by trade / signing / promotion). 2025+ only (no snapshots before).
            for gid_, (ab_, _rk) in dmap.items():
                if ab_ in ("QB", "RB", "FB", "WR", "TE") and len(ph.get(gid_, ())) >= 3:
                    cands.add(gid_)
        dts90, dmap90 = D.depth_at(team, c90)
        T = team_actual(D, key)
        rows = []
        for gid in sorted(cands):
            full = ph.get(gid, [])
            h = full[-16:]
            pos = h[-1]["pos"] if h and h[-1]["pos"] else None
            if pos is None and prev_wk:
                pos = (D.roster.get((prev_wk[0], prev_wk[1], gid)) or {}).get("pos")
            if pos == "FB":
                pos = "RB"
            if pos not in ("QB", "RB", "WR", "TE"):
                continue
            inj = D.inj.get((s, w, gid))
            inj_ts = injury_info_ts(kick) if inj else None
            prev_roster = D.roster.get((prev_wk[0], prev_wk[1], gid)) if prev_wk else None
            game_roster = D.roster.get((s, w, gid))
            rec = player_game_record(D, key, gid, T)
            info = [x for x in (last_info, inj_ts, (prev_kick - T90) if prev_roster else None, dts) if x is not None]
            rows.append({"gid": gid, "pos": pos, "hist": h, "n_hist": len(full),
                         "n_cur": sum(1 for g in full if g["s"] == s and g["played"]),
                         "inj": inj, "inj_info_ts": inj_ts, "prev_roster": prev_roster["status"] if prev_roster else "missing",
                         "game_roster_T90": game_roster["status"] if game_roster else "missing",
                         "depth": dmap.get(gid) if dmap else None, "depth_ts": dts, "depth_listed_any": dmap is not None,
                         "depth_prev": dmap_prev.get(gid) if dmap_prev else None,
                         "draft": D.players.get(gid, {}), "actual": rec,
                         "max_info_ts_T24": max(info) if info else None,
                         "max_info_ts_T90": max(info + [kick - T90]) if info else kick - T90})
        # defensive candidates
        drows = []
        for pfr, h in dh.items():
            if h and h[-1]["team"] == team and (h[-1]["s"], h[-1]["w"]) in recent:
                cur = D.dsnap.get(key, {}).get(pfr)
                gid = h[-1]["gsis"]
                inj = D.inj.get((s, w, gid)) if gid else None
                prev_roster = D.roster.get((prev_wk[0], prev_wk[1], gid)) if gid and prev_wk else None
                game_roster = D.roster.get((s, w, gid)) if gid else None
                drows.append({"pfr": pfr, "gid": gid, "grp": h[-1]["grp"], "hist": h[-16:], "inj": inj,
                              "inj_info_ts": injury_info_ts(kick) if inj else None,
                              "prev_roster": prev_roster["status"] if prev_roster else "missing",
                              "game_roster_T90": game_roster["status"] if game_roster else "missing",
                              "actual": {"played": cur is not None, "pct": cur["pct"] if cur else 0.0, "snaps": cur["snaps"] if cur else 0.0}})
        # team-level history (as of cutoff)
        units.append({"key": key, "kick": kick, "c24": c24, "c90": c90, "opp": ctx["opp"], "home": ctx["home"],
                      "rest": ctx["rest"], "opp_rest": ctx["opp_rest"], "coach": ctx["coach"],
                      "team_hist": list(th[team]), "opp_off_hist": list(th[ctx["opp"]]), "def_allowed_hist": list(ta[team]),
                      "opp_allowed_hist": list(ta[ctx["opp"]]),
                      "team_actual": T, "players": rows, "defenders": drows, "last_info_ts": last_info,
                      "n_prev_team_games": len(prev_games)})
    return units


def absorb(D, key, ph, dh, th, ta):
    s, w, team = key
    ctx = D.game[key]
    T = team_actual(D, key)
    atts = [(D.pp.get((s, w, g), {}).get("att", 0.0), g) for g in D.stat.get(key, {})]
    tr = {"s": s, "w": w, "kick": ctx["kick"], **T, "starting_qb": max(atts)[1] if atts and max(atts)[0] > 0 else None}
    th[team].append(tr); ta[ctx["opp"]].append(tr)
    ids = sorted(set(D.stat.get(key, {})) | set(D.osnap.get(key, {})))
    for gid in ids:
        rec = player_game_record(D, key, gid, T)
        if rec["pos"] == "FB":
            rec["pos"] = "RB"
        ph[gid].append(rec)
    for pfr, x in D.dsnap.get(key, {}).items():
        dh[pfr].append({"s": s, "w": w, "team": team, "kick": ctx["kick"], "grp": x["grp"], "gsis": x["gsis"],
                        "pct": x["pct"], "snaps": x["snaps"]})

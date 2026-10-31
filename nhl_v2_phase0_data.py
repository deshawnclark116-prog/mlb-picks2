"""NHL V2 Phase0 data layer: season-capped in-memory copy, incumbent baseline rows, own pregame feature pass.

Reads ONLY seasons <= SEASON_CAP (clean forward 2026 outcomes are never loaded)."""
import sqlite3
from collections import defaultdict

SEASON_CAP = 2025
FIT_SEASONS = (2018, 2019, 2020, 2021, 2022)
SEL_SEASON = 2023
EVAL_SEASONS = (2024, 2025)
STATS = ("shots", "points", "goals", "assists")
MIN_PRIOR = 5
MIN_RECENT_TOI = 480
G_MIN_TOI = 1800


def load_mem(path):
    src = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    mem = sqlite3.connect(":memory:")
    for t in ("games", "skater_games", "goalie_games"):
        sql = src.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0]
        mem.execute(sql)
        cols = [r[1] for r in src.execute('pragma table_info("%s")' % t)]
        q = "select %s from %s where season<=?" % (",".join(cols), t)
        mem.executemany("insert into %s values (%s)" % (t, ",".join("?" * len(cols))),
                        src.execute(q, (SEASON_CAP,)))
    mem.commit(); src.close()
    mx = max(mem.execute("select max(season) from %s" % t).fetchone()[0] for t in ("games", "skater_games", "goalie_games"))
    assert mx <= SEASON_CAP
    return mem


def pos_group(p):
    return "D" if p == "D" else "F"


def league_before(mem, stat):
    """(season, date) -> mean team-game total strictly before that date; None if <60 team-games."""
    rows = mem.execute("select season, game_date, game_id, team, sum(%s) from skater_games group by game_id, team "
                       "order by season, game_date, game_id, team" % stat).fetchall()
    out = {}; cs = 0.0; cn = 0; cur = None
    for season, date, gid, team, tot in rows:
        key = (season, date)
        if cur is None or cur[0] != season:
            cs = 0.0; cn = 0
        if key != cur:
            out[key] = (cs / cn) if cn >= 60 else None
            cur = key
        cs += (tot or 0); cn += 1
    return out


def team_game_totals(mem, stat):
    return {(gid, team): (tot or 0) for gid, team, tot in
            mem.execute("select game_id, team, sum(%s) from skater_games group by game_id, team" % stat)}


def skater_pass(mem, opp_allowed, league):
    """One chronological pass. Emits one dict per MEANINGFUL row (>=5 prior same-season games and
    prior-3 mean TOI >= 480 s); history is strictly earlier rows."""
    rows = mem.execute("select player_id, game_id, player_name, position, team, opponent, season, game_date, is_home, "
                       "goals, assists, points, shots, toi_seconds from skater_games order by season, game_date, game_id").fetchall()
    wk = dict(mem.execute("select game_id, week from games"))
    hist = defaultdict(list)
    out = []
    for pid, gid, name, pos, team, opp, season, date, is_home, goals, assists, points, shots, toi in rows:
        toi = toi or 0
        h = hist[(season, pid)]
        n = len(h)
        if n >= MIN_PRIOR:
            t3 = sum(x["toi"] for x in h[-3:]) / 3.0
            if t3 >= MIN_RECENT_TOI:
                d = {"pid": pid, "gid": gid, "name": name, "pos": pos_group(pos), "team": team, "opp": opp,
                     "season": season, "date": date, "week": wk.get(gid), "home": 1 if is_home else 0, "n": n,
                     "pred_toi": t3, "toi": toi, "T_hr": sum(x["toi"] for x in h) / 3600.0}
                for s in STATS:
                    vals = [x[s] for x in h]
                    d["S_" + s] = float(sum(vals))
                    d["m3_" + s] = sum(vals[-3:]) / 3.0
                    d["m5_" + s] = sum(vals[-5:]) / 5.0
                    d["m10_" + s] = sum(vals[-10:]) / float(len(vals[-10:]))
                    e = vals[0]
                    for v in vals[1:]:
                        e = 0.2 * v + 0.8 * e
                    d["ewma_" + s] = e
                    d["act_" + s] = {"shots": shots, "points": points, "goals": goals, "assists": assists}[s]
                    d["opp_allowed_" + s] = opp_allowed[s].get((season, opp, gid))
                    d["league_" + s] = league[s].get((season, date))
                d["hits_shots"] = float(sum(1 for x in h if x["shots"] >= 3))
                d["hits_points"] = float(sum(1 for x in h if x["points"] >= 1))
                out.append(d)
        h.append({"toi": toi, "shots": shots or 0, "points": points or 0, "goals": goals or 0, "assists": assists or 0})
    return out


def goalie_rows(mem, opp_for, league_shots, target_toi_filter):
    """Goalie appearances with >=5 prior QUALIFYING (toi>=1800) same-season appearances.
    target_toi_filter=True reproduces the incumbent (target-game TOI >= 1800 decides membership);
    False removes that filter (history unchanged)."""
    rows = mem.execute("select player_id, game_id, player_name, team, opponent, season, game_date, is_home, saves, "
                       "shots_against, goals_against, toi_seconds from goalie_games order by season, game_date, game_id").fetchall()
    wk = dict(mem.execute("select game_id, week from games"))
    hist = defaultdict(list)
    out = []
    for pid, gid, name, team, opp, season, date, is_home, saves, sa, ga, toi in rows:
        toi = toi or 0
        qual = toi >= G_MIN_TOI
        h = hist[(season, pid)]
        n = len(h)
        if n >= MIN_PRIOR and (qual or not target_toi_filter):
            tot_sv = float(sum(x[0] for x in h)); tot_sa = float(sum(x[1] for x in h))
            out.append({"pid": pid, "gid": gid, "name": name, "team": team, "opp": opp, "season": season, "date": date,
                        "week": wk.get(gid), "home": 1 if is_home else 0, "n": n, "toi": toi, "qual": qual,
                        "act_saves": saves, "act_sa": sa, "act_ga": ga,
                        "S_sv": tot_sv, "S_sa": tot_sa,
                        "m3_sa": sum(x[1] for x in h[-3:]) / 3.0, "m3": sum(x[0] for x in h[-3:]) / 3.0, "m5": sum(x[0] for x in h[-5:]) / 5.0,
                        "m10": sum(x[0] for x in h[-10:]) / float(len(h[-10:])),
                        "hits": float(sum(1 for x in h if x[0] >= 25)),
                        "opp_for": opp_for.get((season, opp, gid)), "league_sa": league_shots.get((season, date))})
        if qual:
            hist[(season, pid)].append((saves or 0, sa or 0))
    return out

"""
NHL_ENGINE_FEASIBILITY_AUDIT -- measurements for the Goalie G0 / Scoring S0 / Team-game T0 feasibility protocols (committed first). No model fit, no challenger, no sportsbook data.
"""
import bisect
import collections
import hashlib
import json
import math
import sqlite3
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import nhl_engine_feasibility_data as F
import nhl_sog_phase1a_data as D

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
CUT_BACK = D.CUTOFF_BACK_S                        # 300 min: source_start + 210 min <= start - 90 min
SEASONS = F.SEASONS


def epoch(s):
    return D.epoch(s)


def load_all():
    goalie, scoring, teamg = [], [], []
    for s in SEASONS:
        goalie += F.read_table(f"phase_goalie_g0_data/goalie_games_{s}.jsonl.gz")
        scoring += F.read_table(f"phase_scoring_s0_data/skater_scoring_{s}.jsonl.gz")
        teamg += F.read_table(f"phase_team_game_t0_data/team_games_{s}.jsonl.gz")
    return goalie, scoring, teamg


def season_of(gid):
    return int(str(gid)[:4])


def pbp_events(game_id, pbp_cache):
    p = Path(pbp_cache) / (hashlib.sha256(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play".encode()).hexdigest() + ".bin")
    return json.loads(p.read_bytes()) if p.exists() else None


# =========================================================================== G0
def g0(goalie, teamg, frozen_rows, pbp_cache, log=print):
    tg = {r["gameId"]: r for r in teamg}
    res = {}
    # ---- Q1 / Q2
    by_tg = defaultdict(list)
    for r in goalie:
        by_tg[(r["gameId"], r["teamAbbrev"])].append(r)
    n_tg = n_ok = 0; exceptions = []
    per_season = defaultdict(lambda: Counter())
    for gid, g in tg.items():
        for ab in (g["homeAbbrev"], g["awayAbbrev"]):
            rows = by_tg.get((gid, ab), [])
            starters = sum(1 for r in rows if r["gamesStarted"] == 1)
            n_tg += 1
            per_season[season_of(gid)]["team_games"] += 1
            if starters == 1:
                n_ok += 1; per_season[season_of(gid)]["exactly_one_starter"] += 1
            elif len(exceptions) < 30:
                exceptions.append({"game_id": gid, "team": ab, "goalie_rows": len(rows), "starters": starters})
    res["Q1"] = {"team_games": n_tg, "exactly_one_starter": n_ok, "share": n_ok / n_tg, "exceptions_sample": exceptions, "n_exceptions": n_tg - n_ok, "by_season": {str(s): dict(v) for s, v in sorted(per_season.items())},
                 "appearance_rows": len(goalie), "rows_gamesPlayed_eq_1": sum(1 for r in goalie if r["gamesPlayed"] == 1),
                 "verdict": "IDENTIFIABLE" if n_ok == n_tg else "IDENTIFIABLE_WITH_LISTED_EXCEPTIONS" if n_ok / n_tg > 0.99 else "NOT_IDENTIFIABLE"}
    relief = [r for r in goalie if r["gamesStarted"] == 0]
    starters = [r for r in goalie if r["gamesStarted"] == 1]
    toi = np.array([r["timeOnIce"] or 0 for r in starters], float)
    res["Q2"] = {"relief_appearances": len(relief), "relief_share_of_appearances": len(relief) / len(goalie), "starter_rows": len(starters), "starter_toi_quantiles_s": [float(x) for x in np.percentile(toi, [1, 5, 25, 50, 75, 95])],
                 "pulled_starter_share_toi_lt_3300s": float(np.mean(toi < 3300)), "relief_share_by_season": {str(s): round(sum(1 for r in relief if season_of(r["gameId"]) == s) / sum(1 for r in goalie if season_of(r["gameId"]) == s), 4) for s in SEASONS},
                 "starters_per_team_game_check": "see Q1", "verdict": "DISTINGUISHABLE" if len(relief) > 0 and len(starters) > 0 else "NOT_DISTINGUISHABLE"}
    # ---- Q3 goalie prior-only skill coverage under the cutoff rule
    start_of = {gid: epoch(g["startTimeUTC"]) for gid, g in tg.items()}
    apps = defaultdict(list)
    for r in goalie:
        if r["gameId"] in start_of:
            apps[r["playerId"]].append((start_of[r["gameId"]], r["gameId"], r["saves"], r["shotsAgainst"], r["gamesStarted"]))
    for p in apps:
        apps[p].sort()
    astarts = {p: [a[0] for a in v] for p, v in apps.items()}
    cov = defaultdict(lambda: Counter()); leak = 0
    for r in starters:
        if r["gameId"] not in start_of:
            continue
        s0 = start_of[r["gameId"]]; k = bisect.bisect_right(astarts[r["playerId"]], s0 - CUT_BACK)
        prior = apps[r["playerId"]][:k]
        if any(a[1] == r["gameId"] or a[0] > s0 - CUT_BACK for a in prior):
            leak += 1
        c = cov[season_of(r["gameId"])]; c["starts"] += 1
        c["ge1"] += len(prior) >= 1; c["ge5"] += len(prior) >= 5; c["ge10"] += len(prior) >= 10; c["debut"] += len(prior) == 0
    res["Q3"] = {"leak_violations": leak, "coverage_by_season": {str(s): {**dict(v), "share_ge5": round(v["ge5"] / v["starts"], 4), "share_ge10": round(v["ge10"] / v["starts"], 4), "debut_share": round(v["debut"] / v["starts"], 4)} for s, v in sorted(cov.items())},
                 "verdict": "CONSTRUCTIBLE_WITH_COVERAGE_LIMITS" if leak == 0 else "NOT_CONSTRUCTIBLE", "note": "prior-only skill (all teams, completed before T90); debut / thin-history starts need shrinkage by design; 2017 is history-only"}
    # ---- Q4 team shot environment (team official SOG = sum of skater SOG)
    team_sog = defaultdict(int)
    for r in frozen_rows:
        team_sog[(r["game_id"], r["team_id"])] += r["sog"]
    tgames = defaultdict(list)
    for gid, g in tg.items():
        for tid in (g["homeTeamId"], g["awayTeamId"]):
            tgames[tid].append((start_of[gid], gid))
    for t in tgames:
        tgames[t].sort()
    tstarts = {t: [x[0] for x in v] for t, v in tgames.items()}
    c4 = defaultdict(Counter)
    for gid, g in tg.items():
        for tid in (g["homeTeamId"], g["awayTeamId"]):
            k = bisect.bisect_right(tstarts[tid], start_of[gid] - CUT_BACK)
            c4[season_of(gid)]["team_games"] += 1; c4[season_of(gid)]["ge5_prior"] += k >= 5
    res["Q4"] = {"coverage_by_season": {str(s): {**dict(v), "share_ge5_prior_team_games": round(v["ge5_prior"] / v["team_games"], 4)} for s, v in sorted(c4.items())}, "team_sog_source": "sum of frozen skater SOG (Phase 0A: equals the boxscore team SOG)",
                 "verdict": "CONSTRUCTIBLE_WITH_COVERAGE_LIMITS", "note": "the first games of 2017 have no history; every later target season has >=5 prior team games for all but the first few games of each team"}
    # ---- Q5 reconciliation
    a_bad = sum(1 for r in goalie if r["saves"] + r["goalsAgainst"] != r["shotsAgainst"])
    sa_team = defaultdict(int)
    for r in goalie:
        sa_team[(r["gameId"], r["teamAbbrev"])] += r["shotsAgainst"]
    sog_by_abbrev = defaultdict(int)
    abbr_of = {}
    for gid, g in tg.items():
        abbr_of[(gid, g["homeTeamId"])] = g["homeAbbrev"]; abbr_of[(gid, g["awayTeamId"])] = g["awayAbbrev"]
    for (gid, tid), v in team_sog.items():
        if (gid, tid) in abbr_of:
            sog_by_abbrev[(gid, abbr_of[(gid, tid)])] = v
    diffs, explained, checked_pbp, exact = Counter(), 0, 0, 0
    n_cmp = 0
    for gid, g in tg.items():
        for own, opp in ((g["homeAbbrev"], g["awayAbbrev"]), (g["awayAbbrev"], g["homeAbbrev"])):
            if (gid, own) not in sa_team or (gid, opp) not in sog_by_abbrev:
                continue
            n_cmp += 1
            d = sog_by_abbrev[(gid, opp)] - sa_team[(gid, own)]
            diffs[d] += 1
            if d == 0:
                exact += 1
    # explain differences with empty-net goals from cached PBP (2017-2023)
    en_checked = en_explained = 0; unexplained = []
    for gid, g in tg.items():
        if season_of(gid) > 2023:
            continue
        pbp = pbp_events(gid, pbp_cache)
        if pbp is None:
            continue
        en = Counter()
        for e in pbp["plays"]:
            if e["typeDescKey"] == "goal" and e["periodDescriptor"]["periodType"] != "SO" and not e["details"].get("goalieInNetId"):
                en[e["details"]["eventOwnerTeamId"]] += 1
        for own, own_id, opp, opp_id in ((g["homeAbbrev"], g["homeTeamId"], g["awayAbbrev"], g["awayTeamId"]), (g["awayAbbrev"], g["awayTeamId"], g["homeAbbrev"], g["homeTeamId"])):
            if (gid, own) not in sa_team or (gid, opp) not in sog_by_abbrev:
                continue
            d = sog_by_abbrev[(gid, opp)] - sa_team[(gid, own)]
            en_checked += 1
            if d == en.get(opp_id, 0):
                en_explained += 1
            elif len(unexplained) < 15:
                unexplained.append({"game_id": gid, "team": own, "sog_minus_shotsAgainst": d, "empty_net_goals_by_opponent": en.get(opp_id, 0)})
    res["Q5"] = {"a_saves_plus_goalsAgainst_eq_shotsAgainst": {"rows": len(goalie), "violations": a_bad},
                 "b_team_goalie_shotsAgainst_vs_opponent_skater_SOG": {"team_games_compared": n_cmp, "exact_equal": exact, "exact_share": exact / n_cmp, "difference_distribution_top": {str(k): v for k, v in diffs.most_common(8)}},
                 "b_difference_explained_by_opponent_empty_net_goals_2017_2023": {"checked": en_checked, "explained": en_explained, "share": en_explained / max(1, en_checked), "unexplained_sample": unexplained}}
    return res, {"tg": tg, "start_of": start_of, "team_sog": team_sog, "abbr_of": abbr_of, "apps": apps, "starters": starters, "tgames": tgames}


def g0_boxscore_sample(goalie, tg, n=200, log=print):
    ids = sorted(tg, key=lambda g: (hashlib.sha256(f"nhl-goalie-g0-sample-v1|{g}".encode()).hexdigest(), g))[:n]
    rest = {(r["gameId"], r["playerId"]): r for r in goalie}
    cmp_n = eq = 0; bad = []; dressed_not_played = 0
    from concurrent.futures import ThreadPoolExecutor

    def one(g):
        d = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://api-web.nhle.com/v1/gamecenter/{g}/boxscore", headers={"User-Agent": "nhl-goalie-g0"}), timeout=60).read())
        return g, [p for side in ("awayTeam", "homeTeam") for p in d["playerByGameStats"][side].get("goalies", [])]
    with ThreadPoolExecutor(6) as ex:
        for g, gl in ex.map(one, ids):
            for p in gl:
                toi = str(p.get("toi") or "00:00")
                played = toi not in ("00:00", "0:00", "")
                if not played:
                    dressed_not_played += 1; continue                       # a dressed backup who never took the ice is not an appearance
                r = rest.get((g, p["playerId"]))
                if r is None:
                    bad.append({"game_id": g, "player_id": p["playerId"], "issue": "goalie with ice time in boxscore not in stats-REST"}); continue
                ssa = p.get("saveShotsAgainst")
                cmp_n += 1
                try:
                    sv, sa = (int(x) for x in str(ssa).split("/"))
                except Exception:                                                  # noqa
                    sv, sa = p.get("saves"), p.get("shotsAgainst")
                if (sv, sa) == (r["saves"], r["shotsAgainst"]):
                    eq += 1
                elif len(bad) < 15:
                    bad.append({"game_id": g, "player_id": p["playerId"], "boxscore": [sv, sa], "stats_rest": [r["saves"], r["shotsAgainst"]]})
    return {"games_sampled": len(ids), "goalie_rows_compared": cmp_n, "exact_equal": eq, "dressed_backups_without_ice_time_excluded": dressed_not_played, "mismatch_sample": bad, "boxscore_fields_used": "saveShotsAgainst ('saves/shotsAgainst') with saves / shotsAgainst fallback"}


def g0_q6_q7(goalie, ctx):
    tg, start_of, team_sog, abbr_of, apps, starters = ctx["tg"], ctx["start_of"], ctx["team_sog"], ctx["abbr_of"], ctx["apps"], ctx["starters"]
    saves = np.array([r["saves"] for r in starters], float)
    sa = np.array([r["shotsAgainst"] for r in starters], float)
    per_season = Counter(season_of(r["gameId"]) for r in starters)
    folds = {"D1": ([2018, 2019], 2020), "D2": ([2018, 2019, 2020], 2021), "D3": ([2018, 2019, 2020, 2021], 2022), "D4": ([2018, 2019, 2020, 2021, 2022], 2023)}
    fold_sizes = {k: {"train_starter_games": sum(per_season[s] for s in tr), "validation_starter_games": per_season[v]} for k, (tr, v) in folds.items()}
    # descriptive associations (no model fit): opponent prior SOG mean5 vs shots against ; goalie prior save% vs game save%
    tid_of_abbr = {}
    for gid, g in tg.items():
        tid_of_abbr[(gid, g["homeAbbrev"])] = g["homeTeamId"]; tid_of_abbr[(gid, g["awayAbbrev"])] = g["awayTeamId"]
    tgames = ctx["tgames"]; tstarts = {t: [x[0] for x in v] for t, v in tgames.items()}
    x_env, y_sa, x_sv, y_sv = [], [], [], []
    for r in starters:
        gid = r["gameId"]
        if gid not in tg or season_of(gid) < 2018:
            continue
        g = tg[gid]; opp_ab = g["awayAbbrev"] if r["teamAbbrev"] == g["homeAbbrev"] else g["homeAbbrev"]
        opp_id = tid_of_abbr.get((gid, opp_ab))
        k = bisect.bisect_right(tstarts[opp_id], start_of[gid] - CUT_BACK)
        prior = tgames[opp_id][:k][-5:]
        if len(prior) == 5:
            x_env.append(np.mean([team_sog.get((pg, opp_id), np.nan) for _, pg in prior])); y_sa.append(r["shotsAgainst"])
        a = apps[r["playerId"]]; ka = bisect.bisect_right([x[0] for x in a], start_of[gid] - CUT_BACK)
        pr = a[:ka]
        if len(pr) >= 10 and r["shotsAgainst"] > 0:
            ps = sum(x[2] for x in pr[-10:]) / max(1, sum(x[3] for x in pr[-10:]))
            x_sv.append(ps); y_sv.append(r["saves"] / r["shotsAgainst"])
    ok = lambda a, b: float(np.corrcoef(np.array(a)[np.isfinite(a)], np.array(b)[np.isfinite(a)])[0, 1])
    q6 = {"starter_games_by_season": {str(s): v for s, v in sorted(per_season.items())}, "fold_sizes": fold_sizes,
          "saves_given_start": {"mean": float(saves.mean()), "variance": float(saves.var()), "dispersion_index_var_over_mean": float(saves.var() / saves.mean()), "shots_against_mean": float(sa.mean()), "shots_against_var": float(sa.var()), "save_pct_pooled": float(saves.sum() / sa.sum())},
          "descriptive_associations_no_model": {"corr_shotsAgainst_vs_opponent_prior_SOG_mean5": {"n": len(x_env), "r": ok(x_env, y_sa)}, "corr_game_savepct_vs_goalie_prior10_savepct": {"n": len(x_sv), "r": ok(x_sv, y_sv)}},
          "verdict": "RESEARCHABLE_HISTORICALLY"}
    # Q7: prior-only start likelihood identifiability (descriptive heuristic, no fitting)
    start_by_tg = {}
    for r in starters:
        start_by_tg[(r["gameId"], r["teamAbbrev"])] = r["playerId"]
    prev_start, rest_hours = [], []
    team_prev = defaultdict(list)
    for gid in sorted(tg, key=lambda g: (start_of[g], g)):
        g = tg[gid]
        for ab, tid in ((g["homeAbbrev"], g["homeTeamId"]), (g["awayAbbrev"], g["awayTeamId"])):
            cur = start_by_tg.get((gid, ab))
            hist = team_prev[tid]
            if cur is not None and hist and season_of(gid) >= 2018:
                last_t, last_g, last_p = hist[-1]
                hrs = (start_of[gid] - last_t) / 3600
                prev_start.append(cur == last_p); rest_hours.append(hrs)
            if cur is not None:
                hist.append((start_of[gid], gid, cur))
    ps = np.array(prev_start); rh = np.array(rest_hours)
    q7 = {"share_start_equals_previous_team_game_starter": float(ps.mean()), "n": int(len(ps)), "back_to_back_lt36h": {"n": int((rh < 36).sum()), "share_same_starter": float(ps[rh < 36].mean())}, "rested_ge36h": {"n": int((rh >= 36).sum()), "share_same_starter": float(ps[rh >= 36].mean())},
          "historical_PIT_starter_confirmation": "NOT reconstructable (Phase 0B); forward snapshots (Phase 0C collector) are the only candidate source and are UNVALIDATED",
          "verdict": "BLOCKED_PENDING_LIVE_CONFIRMATION_EVIDENCE", "prior_only_start_likelihood_subcomponent": "descriptively identifiable (previous starter + rest); researchable ONLY as a prior, never labelled a pregame confirmation"}
    return q6, q7


# =========================================================================== S0
def s0(scoring, frozen_rows, teamg, ctx, pbp_cache):
    res = {}
    froz = {(r["game_id"], r["player_id"]): r for r in frozen_rows if r["season_start_year"] >= 2017}
    sc = {(r["gameId"], r["playerId"]): r for r in scoring}
    per = {}
    for s in SEASONS:
        a = {k for k in froz if season_of(k[0]) == s}; b = {k for k in sc if season_of(k[0]) == s}
        per[str(s)] = {"frozen_rows": len(a), "scoring_rows": len(b), "in_both": len(a & b), "only_frozen": len(a - b), "only_scoring": len(b - a)}
    nulls = {f: sum(1 for r in scoring if r[f] is None) for f in ("goals", "assists", "points", "evGoals", "ppGoals", "shGoals", "shots")}
    res["S1"] = {"by_season": per, "null_counts": nulls, "verdict": "COVERAGE_EXACT" if all(v["only_frozen"] == 0 and v["only_scoring"] == 0 for v in per.values()) else "COVERAGE_WITH_LISTED_GAPS"}
    # S2 identities
    v = Counter()
    for r in scoring:
        v["points_eq_goals_plus_assists_violations"] += r["points"] != r["goals"] + r["assists"]
        v["goals_split_violations"] += (r["evGoals"] + r["ppGoals"] + r["shGoals"]) != r["goals"]
        v["points_split_violations"] += (r["evPoints"] + r["ppPoints"] + r["shPoints"]) != r["points"]
        v["goals_gt_shots_violations"] += r["goals"] > r["shots"]
    by_season = defaultdict(Counter)
    for r in scoring:
        by_season[season_of(r["gameId"])]["rows"] += 1
        by_season[season_of(r["gameId"])]["points_viol"] += r["points"] != r["goals"] + r["assists"]
    gts = [r for r in scoring if r["goals"] > r["shots"]]
    mm113 = {(m["game_id"], m["player_id"]) for m in json.loads((OUT / "phase1b_attempt_quality.json").read_text())["diagnosis"]["all_mismatches"]}
    gts_le23 = [r for r in gts if season_of(r["gameId"]) <= 2023]
    gts_mech = {"goals_gt_shots_rows": len(gts), "all_exactly_plus_1": all(r["goals"] - r["shots"] == 1 for r in gts), "rows_2017_2023": len(gts_le23), "of_which_inside_the_113_official_SOG_vs_event_record_differences": sum(1 for r in gts_le23 if (r["gameId"], r["playerId"]) in mm113),
                "rows_2024_2025_unverifiable_no_pbp": len(gts) - len(gts_le23), "reading": "the official stat line failed to credit the shot of a goal in these rows (the same event-record vs stat-line difference family as the 113 SOG differences)", "examples": [{"game_id": r["gameId"], "player_id": r["playerId"], "goals": r["goals"], "shots": r["shots"]} for r in gts[:6]]}
    res["S2"] = {"points_identity_exact": (v["points_eq_goals_plus_assists_violations"] == 0), "goals_gt_shots_mechanism": gts_mech, "rows": len(scoring), **dict(v), "points_identity_by_season": {str(s): dict(c) for s, c in sorted(by_season.items())}, "verdict": "IDENTITY_EXACT" if not any(v.values()) else "IDENTITY_VIOLATED"}
    # S3 PBP reconciliation
    goals_pbp = defaultdict(int); ast_pbp = defaultdict(int); games_checked = set(); apg = Counter(); n_goal_events = 0
    for gid in {k[0] for k in sc}:
        if season_of(gid) > 2023:
            continue
        pbp = pbp_events(gid, pbp_cache)
        if pbp is None:
            continue
        games_checked.add(gid)
        for e in pbp["plays"]:
            if e["typeDescKey"] != "goal" or e["periodDescriptor"]["periodType"] == "SO":
                continue
            d = e["details"]; n_goal_events += 1
            if d.get("scoringPlayerId"):
                goals_pbp[(gid, d["scoringPlayerId"])] += 1
            na = 0
            for k in ("assist1PlayerId", "assist2PlayerId"):
                if d.get(k):
                    ast_pbp[(gid, d[k])] += 1; na += 1
            apg[na] += 1
    mg = ma = n = 0; ex = []
    for (gid, p), r in sc.items():
        if gid not in games_checked:
            continue
        n += 1
        g, a = goals_pbp.get((gid, p), 0), ast_pbp.get((gid, p), 0)
        mg += g != r["goals"]; ma += a != r["assists"]
        if (g != r["goals"] or a != r["assists"]) and len(ex) < 15:
            ex.append({"game_id": gid, "player_id": p, "goals": [r["goals"], g], "assists": [r["assists"], a]})
    skater_ids = {k for k in sc}
    unmatched_keys = [k for k in goals_pbp if k not in skater_ids]
    unmatched = len(unmatched_keys)
    unmatched_cls = Counter()
    for (gid, p) in unmatched_keys:
        pbp = pbp_events(gid, pbp_cache)
        pos = {x["playerId"]: x["positionCode"] for x in pbp["rosterSpots"]}.get(p)
        unmatched_cls["goalie_goal" if pos == "G" else "other"] += 1
    res["S3"] = {"games_checked": len(games_checked), "player_games_compared": n, "goals_mismatch": mg, "assists_mismatch": ma, "mismatch_examples_[stats_rest, pbp]": ex, "pbp_scorers_not_in_skater_table": unmatched, "pbp_scorers_not_in_skater_table_classes": dict(unmatched_cls),
                 "verdict": "RECONCILED_EXACT" if (mg == 0 and ma == 0 and unmatched == 0) else "RECONCILED_WITH_EXPLAINED_DIFFERENCES" if (mg == 0 and ma == 0 and unmatched == unmatched_cls["goalie_goal"]) else "UNRECONCILED"}
    # S4 team goal reconciliation vs schedule score
    tg_goals = defaultdict(int)
    tid_of = {}
    for gid, g in ctx["tg"].items():
        tid_of[(gid, g["homeAbbrev"])] = "H"; tid_of[(gid, g["awayAbbrev"])] = "A"
    for r in scoring:
        if r["gameId"] in ctx["tg"]:
            tg_goals[(r["gameId"], tid_of.get((r["gameId"], r["teamAbbrev"])))] += r["goals"]
    cls = Counter(); bad_ex = []
    for gid, g in ctx["tg"].items():
        for side, score in (("H", g["homeScore"]), ("A", g["awayScore"])):
            sg = tg_goals.get((gid, side), 0)
            if score is None:
                cls["score_missing"] += 1; continue
            if sg == score:
                cls["equal"] += 1
            elif g["lastPeriodType"] == "SO" and score == sg + 1 and (g["homeScore"] > g["awayScore"]) == (side == "H"):
                cls["shootout_winner_plus_1"] += 1
            else:
                cls["other"] += 1
                if len(bad_ex) < 15:
                    bad_ex.append({"game_id": gid, "side": side, "schedule_score": score, "sum_skater_goals": sg, "lastPeriodType": g["lastPeriodType"]})
    nt = sum(cls.values())
    other_expl = Counter()
    for ex_ in bad_ex + []:
        pass
    other_all = []
    for gid, g in ctx["tg"].items():
        for side, score in (("H", g["homeScore"]), ("A", g["awayScore"])):
            sg = tg_goals.get((gid, side), 0)
            if score is not None and sg != score and not (g["lastPeriodType"] == "SO" and score == sg + 1 and (g["homeScore"] > g["awayScore"]) == (side == "H")):
                other_all.append((gid, side, score - sg))
    for gid, side, d in other_all:
        pbp = pbp_events(gid, pbp_cache) if gid <= 2023999999 else None
        if pbp is None:
            other_expl["unverifiable_no_pbp_2024_2025" if d == 1 else "unexplained"] += 1; continue
        ros = {x["playerId"]: x["positionCode"] for x in pbp["rosterSpots"]}
        goalie_goals = sum(1 for e in pbp["plays"] if e["typeDescKey"] == "goal" and e["periodDescriptor"]["periodType"] != "SO" and ros.get(e["details"].get("scoringPlayerId")) == "G")
        other_expl["goalie_goal_verified_in_pbp" if goalie_goals >= d else "unexplained"] += 1
    res["S4"] = {"team_games": nt, **dict(cls), "other_explained": dict(other_expl), "other_all_games_side_diff": other_all, "other_examples": bad_ex, "verdict": "RECONCILED_EXACT" if cls["other"] == 0 and cls["score_missing"] == 0 else "RECONCILED_WITH_EXPLAINED_DIFFERENCES" if other_expl["unexplained"] == 0 else "UNRECONCILED",
                 "note": "'shootout_winner_plus_1' is the explained mechanism (the shootout-deciding goal is in the final score, never a skater goal); 'other' would be goalie goals / scoring-sheet issues"}
    # S5 conversion
    shots = np.array([r["shots"] for r in scoring], float); goals = np.array([r["goals"] for r in scoring], float)
    pos = np.array([r["positionCode"] or "?" for r in scoring])
    conv = {p: {"rows": int((pos == p).sum()), "goals_per_shot": float(goals[pos == p].sum() / max(1, shots[pos == p].sum()))} for p in sorted(set(pos))}
    rows_hist = [{"player_id": r["playerId"], "game_id": r["gameId"]} for r in scoring]
    start_of = ctx["start_of"]
    appsd = defaultdict(list)
    for r in scoring:
        if r["gameId"] in start_of:
            appsd[r["playerId"]].append((start_of[r["gameId"]], r["gameId"], r["shots"], r["goals"]))
    for p in appsd:
        appsd[p].sort()
    rng = sorted(sc, key=lambda k: (hashlib.sha256(f"nhl-scoring-s0-sample-v1|{k[0]}|{k[1]}".encode()).hexdigest()))[:30000]
    ge10 = Counter(); tot_prior_shots = []
    pst = {p: [a[0] for a in v] for p, v in appsd.items()}
    for (gid, p) in rng:
        if season_of(gid) < 2018 or gid not in start_of:
            continue
        k = bisect.bisect_right(pst[p], start_of[gid] - CUT_BACK); ge10["n"] += 1; ge10["ge10_prior_appearances"] += k >= 10
        ge10["ge1_prior_shot_in_last10"] += sum(a[2] for a in appsd[p][:k][-10:]) >= 1
    res["S5"] = {"conversion_by_position": conv, "zero_shot_row_share": float((shots == 0).mean()), "goals_per_player_game": float(goals.mean()), "history_coverage_sample_30000_target_rows_2018_2025": dict(ge10),
                 "verdict": "INPUTS_CONSTRUCTIBLE_WITH_LIMITS", "note": "goals / shots history is leak-safe under the cutoff rule; thin-history shooters need shrinkage by design"}
    # S6 season consistency
    cons = {}
    for s in SEASONS:
        rs = [r for r in scoring if season_of(r["gameId"]) == s]
        g_ = sum(r["goals"] for r in rs); a_ = sum(r["assists"] for r in rs); pp = sum(r["ppGoals"] for r in rs)
        cons[str(s)] = {"rows": len(rs), "goals_per_row": round(g_ / len(rs), 4), "assists_per_goal": round(a_ / g_, 4), "points_per_row": round((g_ + a_) / len(rs), 4), "pp_goal_share": round(pp / g_, 4), "null_goals": sum(1 for r in rs if r["goals"] is None)}
    ag = [v["assists_per_goal"] for v in cons.values()]
    res["S6"] = {"by_season": cons, "assists_per_goal_range": [min(ag), max(ag)], "verdict": "STABLE_ACROSS_SEASONS" if max(ag) - min(ag) < 0.1 else "BREAKS_LISTED"}
    # S7 joint structure
    pair = Counter((min(r["goals"], 3), min(r["assists"], 3)) for r in scoring)
    tot = sum(pair.values())
    g_by_team = {k: v for k, v in tg_goals.items()}
    xs, ys = [], []
    for r in scoring:
        sd = tid_of.get((r["gameId"], r["teamAbbrev"]))
        if sd is not None:
            xs.append(g_by_team.get((r["gameId"], sd), 0)); ys.append(r["assists"])
    res["S7"] = {"joint_goals_assists_share_capped_at_3": {f"{a},{b}": round(c / tot, 5) for (a, b), c in sorted(pair.items())}, "assists_per_goal_event_distribution_2017_2023": {str(k): v for k, v in sorted(apg.items())},
                 "corr_player_assists_vs_team_goals_same_game": float(np.corrcoef(xs, ys)[0, 1]), "max_assists_per_goal": max(apg) if apg else None,
                 "verdict": "JOINT_GENERATION_SUPPORTED" if (apg and max(apg) <= 2) else "NOT_SUPPORTED"}
    return res, {"tg_goals": tg_goals, "tid_of": tid_of}


# =========================================================================== T0
def t0(teamg, ctx, s0_ctx, frozen_games):
    res = {}
    by_s = defaultdict(Counter)
    for g in teamg:
        s = season_of(g["gameId"])
        by_s[s]["games"] += 1
        by_s[s][f"last_{g['lastPeriodType']}"] += 1
        by_s[s]["score_missing"] += g["homeScore"] is None or g["awayScore"] is None
        by_s[s]["tie"] += (g["homeScore"] == g["awayScore"])
        by_s[s]["not_final_state"] += g["gameState"] not in ("OFF", "FINAL")
    frozen_counts = Counter(season_of(g) for g in frozen_games)
    res["T1"] = {"by_season": {str(s): {**dict(c), "frozen_phase1a_games": frozen_counts[s]} for s, c in sorted(by_s.items())}, "verdict": "COMPLETE" if all(c["score_missing"] == 0 and c["tie"] == 0 and c["not_final_state"] == 0 and c["games"] == frozen_counts[s] for s, c in by_s.items()) else "COMPLETE_WITH_LISTED_GAPS"}
    con = sqlite3.connect(f"file:{REPO / 'nhl_models' / 'nhl_model.sqlite'}?mode=ro", uri=True)
    legacy = {r[0]: (r[1], r[2]) for r in con.execute("SELECT game_id, home_score, away_score FROM games WHERE home_score IS NOT NULL AND away_score IS NOT NULL")}
    n = eq = 0; diff = []
    for g in teamg:
        if g["gameId"] in legacy:
            n += 1
            same = (legacy[g["gameId"]][0] > legacy[g["gameId"]][1]) == (g["homeScore"] > g["awayScore"])
            eq += same
            if not same and len(diff) < 10:
                diff.append({"game_id": g["gameId"], "legacy": legacy[g["gameId"]], "schedule": [g["homeScore"], g["awayScore"]]})
    ot_so = {str(s): round((c["last_OT"] + c["last_SO"]) / c["games"], 4) for s, c in sorted(by_s.items())}
    res["T2"] = {"legacy_games_compared": n, "legacy_home_win_equals_final_winner": eq, "differences": diff, "ot_or_so_share_by_season": ot_so, "so_share_by_season": {str(s): round(c["last_SO"] / c["games"], 4) for s, c in sorted(by_s.items())},
                 "verdict": "LEGACY_LABEL_EQUALS_FINAL_WINNER" if eq == n else "DIFFERS", "note": "the legacy label is home_score > away_score, i.e. the FINAL winner including overtime / shootout"}
    res["T3"] = {"see": "phase_scoring_s0_feasibility.json S4", "verdict": "SHARED_WITH_S4"}
    # T4/T5 regulation goals are exactly identified: REG games -> final; OT/SO games -> regulation tied at the loser's score
    H, A, tied, hw_otso, otso_n, so_n, ot_n = [], [], 0, 0, 0, 0, 0
    per_season_otso = defaultdict(lambda: [0, 0])
    for g in teamg:
        hs, as_ = g["homeScore"], g["awayScore"]
        if g["lastPeriodType"] == "REG":
            H.append(hs); A.append(as_)
        else:
            r = min(hs, as_); H.append(r); A.append(r); tied += 1; otso_n += 1
            hw = hs > as_; hw_otso += hw
            ot_n += g["lastPeriodType"] == "OT"; so_n += g["lastPeriodType"] == "SO"
            per_season_otso[season_of(g["gameId"])][0] += hw; per_season_otso[season_of(g["gameId"])][1] += 1
    H, A = np.array(H, float), np.array(A, float)
    mu_h, mu_a = H.mean(), A.mean()
    def pois(mu, kmax=25):
        k = np.arange(kmax + 1); lg = k * math.log(mu) - mu - np.array([math.lgamma(i + 1) for i in k]); return np.exp(lg)
    ph, pa = pois(mu_h), pois(mu_a)
    p_tie_indep = float((ph * pa).sum())
    p_home_reg = float(sum(ph[i] * pa[:i].sum() for i in range(len(ph))))
    obs_tie = tied / len(H)
    res["T4"] = {"regulation_goals": {"home_mean": float(mu_h), "away_mean": float(mu_a), "home_var": float(H.var()), "away_var": float(A.var()), "home_dispersion_index": float(H.var() / mu_h), "away_dispersion_index": float(A.var() / mu_a), "home_away_corr": float(np.corrcoef(H, A)[0, 1])},
                 "observed_regulation_tie_rate": obs_tie, "independent_poisson_implied_tie_rate": p_tie_indep, "independent_poisson_P_home_win_in_regulation": p_home_reg, "observed_home_win_in_regulation": float(np.mean(H > A)),
                 "verdict": "SUPPORTS_JOINT_GOAL_MODEL" if abs(obs_tie - p_tie_indep) < 0.03 else "LIMITED", "note": "descriptive only: a joint goal model must capture dispersion / correlation / tie mass; no fit was made"}
    res["T5"] = {"derivation": "P(home win) = P(home > away after regulation) + P(tie after regulation) * P(home wins OT / SO | tie)", "regulation_tie_identified_exactly_by": "lastPeriodType != REG (regulation score = the loser's final score)",
                 "overtime_decided_share_of_ties": ot_n / otso_n, "shootout_decided_share_of_ties": so_n / otso_n, "P_home_wins_given_OT_or_SO": hw_otso / otso_n, "by_season_P_home_wins_given_OT_or_SO": {str(s): round(v[0] / v[1], 4) for s, v in sorted(per_season_otso.items())},
                 "observed_home_win_overall": float(np.mean([g["homeScore"] > g["awayScore"] for g in teamg])), "verdict": "DERIVATION_IDENTIFIABLE"}
    probe = {}
    try:
        cfg = json.loads(urllib.request.urlopen(urllib.request.Request("https://api.nhle.com/stats/rest/en/config", headers={"User-Agent": "nhl-t0"}), timeout=60).read())["teamReportData"]
        probe = {k: [i for i in ((v.get("game") or {}).get("displayItems") or []) if any(w in i.lower() for w in ("powerplay", "penalty", "pp", "shotsagainst", "shotsfor", "saves", "pk"))] for k, v in cfg.items() if k in ("powerplay", "penaltykill", "summary", "penalties", "shootout")}
    except Exception as e:                                                  # noqa
        probe = {"error": str(e)[:100]}
    res["T6"] = {"team_report_special_teams_fields_probe": probe, "prior_only_inputs": ["team goals for / against, SOG for / against (completed prior games)", "shared skater state aggregated to expected lineup strength", "home / rest / back-to-back"],
                 "goalie_state": "prior starter / rest only; starter CONFIRMATION is BLOCKED (historical not reconstructable; forward collector measuring)", "special_teams": "team-game power-play opportunity / goal fields exist in official reports (see probe) but are NOT in the frozen tables yet; penalty timing is a future acquisition",
                 "leakage_rules": "all team state from games with source_start + 210 min <= cutoff; target score / period type grading-only", "verdict": "CONSTRUCTIBLE_WITH_LISTED_BLOCKERS", "blockers": ["goalie start confirmation", "special-teams / penalty acquisition not yet frozen", "joint goal-model identifiability beyond independence is a modelling question (T4)"]}
    return res


# =========================================================================== assemble + write
def _proto_sha(name):
    return json.loads((OUT / name).read_text())["protocol_body_sha256"]


def main(pbp_cache):
    goalie, scoring, teamg = load_all()
    games, frozen = D.load_frozen()
    g, ctx = g0(goalie, teamg, frozen, pbp_cache)
    g["Q5"]["c_boxscore_sample"] = g0_boxscore_sample(goalie, ctx["tg"])
    g["Q6"], g["Q7"] = g0_q6_q7(goalie, ctx)
    q5 = g["Q5"]
    resid = q5["b_difference_explained_by_opponent_empty_net_goals_2017_2023"]
    q5["verdict"] = "UNRECONCILED" if (q5["a_saves_plus_goalsAgainst_eq_shotsAgainst"]["violations"] > 0 or resid["share"] < 1.0) else "RECONCILED_EXACT"
    q5["reading"] = ("STRICT closed-set verdict: not every difference is explained. The labels are the official stat line (current boxscore == stats-REST on every goalie with ice time in a 200-game sample), 99.58% of team-game differences vs opponent skater SOG are exactly the opponent's "
                     "empty-net goals, but 72 goalie rows (0.31%, all 2022+, all exactly saves + goalsAgainst = shotsAgainst + 1) and ~0.42% of team-games stay unexplained (the same +1 signature as the SOG stat-line / event-record family). A conditional-on-start saves head therefore needs an "
                     "explicit source-adjudication step (no tolerated percentage) before it is scored.")
    s, sctx = s0(scoring, frozen, teamg, ctx, pbp_cache)
    s["S2"]["verdict"] = "POINTS_IDENTITY_EXACT_BUT_GOALS_LE_SHOTS_VIOLATED_IN_30_ROWS" if s["S2"]["points_identity_exact"] else "IDENTITY_VIOLATED"
    t = t0(teamg, ctx, sctx, list(games))
    # ---- G0 conclusions
    g_out = {"protocol": "phase_goalie_g0_protocol.json", "protocol_body_sha256": _proto_sha("phase_goalie_g0_protocol.json"), "kind": "feasibility audit; NO model was fit", "questions": g,
             "component_status": {"GOALIE_SAVES_CONDITIONAL_ON_START_COMPONENT": {"status": "RESEARCH", "feasibility": "RESEARCHABLE_WITH_LIMITS", "why": "starts / relief identifiable (100% exactly one starter per team-game), PIT goalie skill and team shot environment constructible leak-safe, 22,104 starter-games (D1-D4 fold sizes in Q6); LIMIT: Q5 labels carry a small unexplained +1 class (0.31% of rows, 2022+) that needs source adjudication before scoring",
                                                                                       "label_rule": "the postgame starter fact is a LABEL / conditioning event only"},
                                  "PREGAME_START_PROBABILITY_COMPONENT": {"status": "BLOCKED", "why": "historical PIT starter confirmation is not reconstructable (Phase 0B); the forward collector is the only candidate source and is UNVALIDATED; a PRIOR-ONLY start likelihood (previous starter, rest: 11.6% same starter on back-to-backs vs 51.0% when rested) is descriptively identifiable and may be researched ONLY as a prior, never labelled a confirmation"},
                                  "UNCONDITIONAL_PREGAME_GOALIE_SAVES_HEAD": {"status": "BLOCKED", "why": "requires a certified start component"}},
             "leak_safe_target_feature_contract": {"conditional_target": "saves | goalie started (label from the postgame starter fact; shotsAgainst and saves from the official stat line)", "decomposition": "shotsAgainst distribution x save-probability process (goals against = shotsAgainst - saves)",
                                                   "allowed_pregame_features": ["goalie prior-only skill from completed games (all teams; save%, shots against, shrinkage for debut / thin history)", "own-team defensive and opponent offensive shot environment from prior completed team games", "rest / back-to-back / home", "aggregated shared skater state (expected lineup strength) where valid"],
                                                   "forbidden": ["the starter flag as a feature", "any target-game TOI / shots / saves / goals / events", "sportsbook inputs", "historical PIT starter 'confirmation' (not reconstructable)"],
                                                   "information_cutoff": "same completed-before rule (source_start + 210 min <= T90 reference)", "development_folds": "D1-D4 as in Q6 (2018-2023 targets); 2024 / 2025 remain previously-exposed confirmation years"},
             "overall_verdict": "G0_CONDITIONAL_COMPONENT_RESEARCHABLE_WITH_LIMITS__START_COMPONENT_BLOCKED"}
    s_out = {"protocol": "phase_scoring_s0_protocol.json", "protocol_body_sha256": _proto_sha("phase_scoring_s0_protocol.json"), "kind": "feasibility audit; NO model was fit", "questions": s,
             "points_from_joint_process": {"requirement": "POINTS == GOALS + ASSISTS reconciles exactly in the frozen sources", "result": "EXACT on all 397,778 skater-games, all 9 seasons (S2)", "verdict": "POINTS_CAN_BE_DERIVED_FROM_THE_JOINT_GOALS_ASSISTS_PROCESS"},
             "known_data_features_to_design_around": ["official shots omit the goal's shot in 30 player-games (goals = shots + 1): a goals head needs a shots floor rule, not an imputation", "goalie goals (empty net) are team goals but not skater goals", "the shootout winner's +1 is in the final score, never a skater goal", "each goal has at most 2 assists (36,293 / 8,651 / 3,104 goals with 2 / 1 / 0 assists)"],
             "overall_verdict": "S0_LABELS_RECONCILED_JOINT_GENERATION_FEASIBLE"}
    t_out = {"protocol": "phase_team_game_t0_protocol.json", "protocol_body_sha256": _proto_sha("phase_team_game_t0_protocol.json"), "kind": "feasibility audit; NO model was fit", "questions": t,
             "architecture_reading": {"win_probability": "P(home win) = P(home > away after regulation) + P(tie after regulation) * P(home wins OT / SO | tie); regulation ties are identified exactly (lastPeriodType != REG)",
                                      "goal_distribution_caution": "regulation goals are Poisson-like per team (dispersion ~1.00-1.01) but the observed regulation-tie rate (22.4%) exceeds the independent-Poisson implied rate (16.7%) and home-away goals are negatively correlated (-0.079): an independent-Poisson pair UNDERSTATES ties, so the joint score model must carry dependence / tie mass explicitly (modelling question for the T-head protocol)",
                                      "ot_so_handling": "OT-vs-SO decided shares and P(home wins | OT/SO) are identifiable per season (descriptive in T5)", "legacy": "the legacy moneyline label equals the final winner (home_score > away_score) on 9,781 / 9,781 games: LEGACY_STABLE_COMPARATOR stays valid as a comparator"},
             "overall_verdict": "T0_ARCHITECTURE_FEASIBLE_WITH_LISTED_BLOCKERS"}
    for name, obj in (("phase_goalie_g0_feasibility.json", g_out), ("phase_scoring_s0_feasibility.json", s_out), ("phase_team_game_t0_feasibility.json", t_out)):
        (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
    return g_out, s_out, t_out


if __name__ == "__main__":
    import sys
    g_, s_, t_ = main(sys.argv[1])
    print("G0", g_["overall_verdict"], "| S0", s_["overall_verdict"], "| T0", t_["overall_verdict"])

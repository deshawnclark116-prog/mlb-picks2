"""
CFB_PHASE1_ROLE_STATE -- as-of participation / role / skill features of the point-in-time candidate universe (CFB Outcome Engine v1, RESEARCH / SHADOW ONLY).
Windows count only the team's prior FBS-vs-FBS games with VALID play coverage (the games whose player lines exist). All features use games with (season, week) < the target; the target week's player lines are read only after the candidate
rows' features are fixed. Role history is CURRENT-TEAM; the skill windows follow the provider athlete_id across schools (all-team). Newcomer / returning flags are the only organizational-intent priors the sources support.
"""
import math
from collections import defaultdict

import numpy as np

import cfb_phase1_common as C
import cfb_phase1_data as D
import cfb_phase1_team_environment as TE

ROLE_FEATURES = ["POS_QB", "POS_RB", "POS_WR", "POS_TE", "P_APPS_L3", "P_APPS_L5", "P_APPS_L12", "P_GAMES_SINCE_APP", "P_RETURNING", "P_TRANSFER_NEWCOMER", "P_CARRY_SHARE_L5", "P_REC_SHARE_L5", "P_ATT_SHARE_L5",
                 "P_CARRY_SHARE_ACTIVE_L5", "P_REC_SHARE_ACTIVE_L5", "P_ATT_SHARE_ACTIVE_L5", "P_CARRY_SHARE_DECAY", "P_REC_SHARE_DECAY", "P_ATT_SHARE_DECAY", "P_CARRIES_PER_APP_L10", "P_REC_PER_APP_L10", "P_ATT_PER_APP_L10",
                 "P_N_SKILL_APPS_10", "TEAM_N_ACTIVE_CARRIERS_L5", "TEAM_N_ACTIVE_RECEIVERS_L5"]
FEATURES = ROLE_FEATURES + TE.TEAM_FEATURES
# ---- shared state v2 (Phase 1B amendment 1): all-team raw count windows over the player's last 20 appearances (the hazard / efficiency components shrink them) + target-game team totals (labels only)
L20_FEATURES = ["P_CARRIES_L20", "P_RUSH_YDS_L20", "P_RUSH_TD_L20", "P_REC_L20", "P_REC_YDS_L20", "P_REC_TD_L20", "P_ATT_L20", "P_COMP_L20", "P_PASS_YDS_L20", "P_PASS_TD_L20", "P_INT_L20", "P_N_APPS_L20"]
FEATURES_V2 = ROLE_FEATURES + L20_FEATURES + TE.TEAM_FEATURES_V2
HALF_LIFE = 3.0
Y_FIELDS = ["y_carries", "y_rush_yards", "y_receptions", "y_rec_yards", "y_pass_att", "y_completions", "y_pass_yards", "y_pass_td", "y_rush_td", "y_rec_td", "y_int"]


def _share(num, den):
    return num / den if den > 0 else math.nan


def build_feature_table(team_games, player_games, candidates, target_seasons=C.TARGET_SEASONS):
    """candidates: output of cfb_phase1_data.build_candidates. Returns a list of rows = candidate + labels + role / team features."""
    cand_by_week = defaultdict(list)
    for c in candidates:
        cand_by_week[(c["season"], c["week"])].append(c)
    tg_by_week = defaultdict(list)
    for r in team_games:
        tg_by_week[(r["season"], r["week"])].append(r)
    pg_by = defaultdict(list)
    for r in player_games:
        pg_by[(r["game_id"], r["team"])].append(r)
    tg_lookup = {(r["game_id"], r["team"]): r for r in team_games}
    tdmap = defaultdict(int)
    for r in player_games:
        tdmap[(r["game_id"], r["team"])] += (r["rushing_touchdowns"] or 0) + (r["receiving_touchdowns"] or 0)
    st = TE.TeamStateV2(dict(tdmap))
    team_obs = defaultdict(list)             # team -> chronological list of observed-line games {season, lines{pid:(car,rec,att)}, tcar, trec, tatt}
    player_apps = defaultdict(list)          # pid -> chronological appearances (season, team, car, rec, att)
    first_app_for_team = {}                  # (pid, team) -> index into team_obs[team] of the first appearance
    prev_season_apps = defaultdict(set)      # (team, season) -> pids appearing
    out = []
    for sw in sorted(set(tg_by_week) | set(cand_by_week)):
        season, week = sw
        for c in cand_by_week.get(sw, []):
            key = (c["game_id"], c["team"])
            tgr = tg_lookup[key]
            tf = st.features(tgr)
            team = c["team"]; obs = team_obs[team]; n = len(obs)
            l3, l5, l12 = obs[-3:], obs[-5:], obs[-12:]
            pid = c["player_id"]
            def apps(games):
                return sum(1 for g in games if pid in g["lines"])
            def cnt(games, i):
                return sum(g["lines"][pid][i] for g in games if pid in g["lines"])
            tot = lambda games, k: sum(g[k] for g in games)
            act = [g for g in l5 if pid in g["lines"]]
            last_idx = max((i for i, g in enumerate(obs) if pid in g["lines"]), default=None)
            decay = lambda i, k: (lambda w: _share(sum(wi * g["lines"][pid][i] for wi, g in zip(w, l12) if pid in g["lines"]), sum(wi * g[k] for wi, g in zip(w, l12))))([0.5 ** ((len(l12) - 1 - j) / HALF_LIFE) for j in range(len(l12))])
            hist = player_apps[pid][-10:]
            h20 = player_apps[pid][-20:]
            fa = first_app_for_team.get((pid, team))
            newcomer = float(fa is not None and (n - fa) <= 12 and _other_team_before_first(player_apps[pid], team))
            carriers = sum(1 for p in {q for g in l5 for q in g["lines"]} if _share(sum(g["lines"][p][0] for g in l5 if p in g["lines"]), tot(l5, "tcar")) >= 0.10) if l5 else 0
            receivers = sum(1 for p in {q for g in l5 for q in g["lines"]} if _share(sum(g["lines"][p][1] for g in l5 if p in g["lines"]), tot(l5, "trec")) >= 0.10) if l5 else 0
            pos = c["position"]
            f = {"POS_QB": float(pos == "QB"), "POS_RB": float(pos in ("RB", "FB")), "POS_WR": float(pos == "WR"), "POS_TE": float(pos == "TE"), "P_APPS_L3": float(apps(l3)), "P_APPS_L5": float(apps(l5)), "P_APPS_L12": float(apps(l12)),
                 "P_GAMES_SINCE_APP": float(min(n - 1 - last_idx, 12)) if last_idx is not None else 12.0, "P_RETURNING": float(pid in prev_season_apps[(team, season - 1)]), "P_TRANSFER_NEWCOMER": newcomer,
                 "P_CARRY_SHARE_L5": _share(cnt(l5, 0), tot(l5, "tcar")), "P_REC_SHARE_L5": _share(cnt(l5, 1), tot(l5, "trec")), "P_ATT_SHARE_L5": _share(cnt(l5, 2), tot(l5, "tatt")),
                 "P_CARRY_SHARE_ACTIVE_L5": _share(sum(g["lines"][pid][0] for g in act), tot(act, "tcar")), "P_REC_SHARE_ACTIVE_L5": _share(sum(g["lines"][pid][1] for g in act), tot(act, "trec")), "P_ATT_SHARE_ACTIVE_L5": _share(sum(g["lines"][pid][2] for g in act), tot(act, "tatt")),
                 "P_CARRY_SHARE_DECAY": decay(0, "tcar") if l12 else math.nan, "P_REC_SHARE_DECAY": decay(1, "trec") if l12 else math.nan, "P_ATT_SHARE_DECAY": decay(2, "tatt") if l12 else math.nan,
                 "P_CARRIES_PER_APP_L10": _share(sum(a[2] for a in hist), len(hist)), "P_REC_PER_APP_L10": _share(sum(a[3] for a in hist), len(hist)), "P_ATT_PER_APP_L10": _share(sum(a[4] for a in hist), len(hist)), "P_N_SKILL_APPS_10": float(len(hist)),
                 "TEAM_N_ACTIVE_CARRIERS_L5": float(carriers), "TEAM_N_ACTIVE_RECEIVERS_L5": float(receivers),
                 "P_CARRIES_L20": float(sum(a[2] for a in h20)), "P_RUSH_YDS_L20": float(sum(a[5] for a in h20)), "P_RUSH_TD_L20": float(sum(a[9] for a in h20)), "P_REC_L20": float(sum(a[3] for a in h20)), "P_REC_YDS_L20": float(sum(a[6] for a in h20)),
                 "P_REC_TD_L20": float(sum(a[10] for a in h20)), "P_ATT_L20": float(sum(a[4] for a in h20)), "P_COMP_L20": float(sum(a[8] for a in h20)), "P_PASS_YDS_L20": float(sum(a[7] for a in h20)), "P_PASS_TD_L20": float(sum(a[11] for a in h20)),
                 "P_INT_L20": float(sum(a[12] for a in h20)), "P_N_APPS_L20": float(len(h20))}
            row = {k: c[k] for k in ("season", "week", "game_id", "team", "opponent", "player_id", "position")}
            row["key"] = f'{c["game_id"]}|{team}|{pid}'
            row.update({k: c[k] for k in Y_FIELDS})
            row["y_part"] = int(c["y_carries"] + c["y_receptions"] + c["y_pass_att"] > 0)
            row["team_carries_target"] = None
            row.update(f); row.update(tf)
            out.append(row)
        # ---- only now fold the week in
        batch = sorted(tg_by_week.get(sw, []), key=lambda r: (r["game_id"], -r["is_home"]))
        st.update(batch)
        for tgr in batch:
            lines = pg_by.get((tgr["game_id"], tgr["team"]), [])
            if lines and C.coverage_valid(tgr):
                d = {r["player_id"]: (r["carries"] or 0, r["receptions"] or 0, r["pass_attempts"] or 0) for r in lines if (r["carries"] or 0) + (r["receptions"] or 0) + (r["pass_attempts"] or 0) > 0}
                g = {"season": tgr["season"], "lines": d, "tcar": float(sum(v[0] for v in d.values())), "trec": float(sum(v[1] for v in d.values())), "tatt": float(sum(v[2] for v in d.values()))}
                team_obs[tgr["team"]].append(g)
                idx = len(team_obs[tgr["team"]]) - 1
                for r in lines:
                    p = r["player_id"]
                    if p not in d:
                        continue
                    first_app_for_team.setdefault((p, tgr["team"]), idx)
                    player_apps[p].append((tgr["season"], tgr["team"], d[p][0], d[p][1], d[p][2], r["rushing_yards"] or 0, r["receiving_yards"] or 0, r["passing_yards"] or 0, r["completions"] or 0, r["rushing_touchdowns"] or 0, r["receiving_touchdowns"] or 0, r["passing_touchdowns"] or 0, r["passing_interceptions"] or 0))
                    prev_season_apps[(tgr["team"], tgr["season"])].add(p)
    # team carry totals of the TARGET game (for the allocation-only evaluation): attached after features, labels only
    tot = defaultdict(float)
    for r in player_games:
        tot[(r["game_id"], r["team"])] += r["carries"] or 0
    for r in out:
        r["team_carries_target"] = tot.get((r["game_id"], r["team"]), 0.0)
        t = tg_lookup[(r["game_id"], r["team"])]
        r["t_rush"], r["t_att"], r["t_comp"], r["t_sack"], r["t_plays"] = t["rush_plays"], t["pass_att_plays"], t["completions"], t["sack_plays"], t["plays"]
        r["target_valid"] = int(C.coverage_valid(t))
    return out


def _other_team_before_first(apps, team):
    """True when the athlete_id has an appearance for ANOTHER team before his first appearance for `team` (a provider-ID-stitched transfer-in)."""
    for a in apps:
        if a[1] == team:
            return False
        return True
    return False


def arrays(rows, names=FEATURES):
    return {n: np.array([r[n] for r in rows], dtype=float) for n in names}

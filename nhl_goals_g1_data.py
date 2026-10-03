"""
NHL_GOALS_G1_DATA -- goal labels + legal goal-history features for the skater GOALS head (protocol: phase_goals_g1_protocol.json).
Labels come from the frozen S0 skater scoring table joined 1:1 to the frozen Phase 1A rows; goal history uses the SAME cutoff rule as the shared skater state (source start <= target start - 300 min). No sportsbook field anywhere.
effective_sog_for_conversion = max(official_sog, goals): used ONLY where a historical shot COUNT is a conversion denominator.
"""
import hashlib
import json
from collections import defaultdict

import numpy as np

import nhl_engine_feasibility_data as FD
import nhl_sog_phase1a_data as D

GOAL_HISTORY_FEATURES = ["GOALS_SUM_CUM", "EFFSOG_SUM_CUM", "N_APPS_CUM", "GOALS_MEAN_APP10"]
G2_FEATURES = list(D.FEATURES) + GOAL_HISTORY_FEATURES
GMAX = 20


def effective_sog(sog, goals):
    """Preregistered data-consistency rule (S0): a goal implies at least one shot attempt on net."""
    return np.maximum(np.asarray(sog), np.asarray(goals))


def load_scoring(verify=True):
    """-> {(game_id, player_id): (goals, official_shots)} for every frozen skater-game (2017-2025)."""
    man = json.loads((FD.OUT / "nhl_engine_feasibility_data_manifest.json").read_text())
    out = {}
    for s in FD.SEASONS:
        rel = f"phase_scoring_s0_data/skater_scoring_{s}.jsonl.gz"
        if verify:
            got = hashlib.sha256((FD.OUT / rel).read_bytes()).hexdigest()
            assert got == man["files"][rel]["sha256"], f"{rel}: frozen scoring table hash mismatch"
        for r in FD.read_table(rel):
            out[(int(r["gameId"]), int(r["playerId"]))] = (int(r["goals"]), int(r["shots"]))
    return out


def attach_goals(tab, scoring):
    """Goal labels per candidate row: participants from the frozen scoring table, non-participants 0 (exactly the SOG head's rule). Returns (goals, official_shots_check_mismatches)."""
    n = len(tab["game_id"])
    goals = np.zeros(n, dtype=np.int64)
    mism = 0
    for i in range(n):
        if tab["played"][i] == 1:
            g, sh = scoring[(int(tab["game_id"][i]), int(tab["player_id"][i]))]
            goals[i] = g
            mism += int(sh != tab["sog"][i])
    return goals, mism


def history_features(tab, rows, scoring):
    """Legal all-team prior goal history per candidate row. A prior appearance is any frozen row of the player with game start <= target start - CUTOFF_BACK_S (the target game is therefore excluded by construction).
    Returns {feature: array}; goals labels are read ONLY from appearances that satisfy the cutoff."""
    per = defaultdict(list)
    for r in rows:
        g, sh = scoring[(int(r["game_id"]), int(r["player_id"]))]
        per[int(r["player_id"])].append((D.epoch(r["game_start_utc"]), int(r["game_id"]), g, int(r["sog"])))
    pre = {}
    for p, v in per.items():
        v.sort(key=lambda a: (a[0], a[1]))
        st = np.array([a[0] for a in v], dtype=np.int64)
        gl = np.array([a[2] for a in v], dtype=np.int64)
        ef = np.array([max(a[3], a[2]) for a in v], dtype=np.int64)
        pre[p] = (st, np.concatenate([[0], np.cumsum(gl)]), np.concatenate([[0], np.cumsum(ef)]), gl)
    n = len(tab["game_id"])
    out = {k: np.full(n, np.nan) for k in GOAL_HISTORY_FEATURES}
    limit = tab["start"].astype(np.int64) - D.CUTOFF_BACK_S
    for i in range(n):
        st, cg, ce, gl = pre[int(tab["player_id"][i])]
        k = int(np.searchsorted(st, limit[i], side="right"))
        out["GOALS_SUM_CUM"][i] = cg[k]; out["EFFSOG_SUM_CUM"][i] = ce[k]; out["N_APPS_CUM"][i] = k
        out["GOALS_MEAN_APP10"][i] = gl[max(0, k - 10):k].mean() if k else np.nan
    return out


def build_goals_table(tab, rows, scoring):
    """Feature table + goals label + goal-history features. Returns the extended table (original arrays untouched) and a data-quality report."""
    goals, mism = attach_goals(tab, scoring)
    hist = history_features(tab, rows, scoring)
    ext = dict(tab)
    ext["goals"] = goals
    ext["eff_sog"] = effective_sog(tab["sog"], goals)
    ext.update(hist)
    defect = (tab["played"] == 1) & (goals > tab["sog"])
    rep = {"rows": int(len(goals)), "played_rows": int(tab["played"].sum()), "official_shots_mismatch_vs_frozen_sog": int(mism), "goals_gt_sog_rows_in_candidate_table": int(defect.sum()),
           "goals_gt_sog_by_season": {int(s): int(((tab["season"] == s) & defect).sum()) for s in sorted(set(tab["season"].tolist()))}, "goals_total_by_season": {int(s): int(goals[tab["season"] == s].sum()) for s in sorted(set(tab["season"].tolist()))},
           "nonparticipant_rows_with_goals": int(((tab["played"] == 0) & (goals > 0)).sum()), "max_goals": int(goals.max())}
    return ext, rep

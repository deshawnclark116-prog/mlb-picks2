"""
NFL_PHASE1C_CONSTANTS  (Phase 1C, shadow research)

League-level structural constants used by the joint game simulator, estimated ONLY on TRAIN + VALID (through 2024 week 18):
  no-target rate of non-sack attempts, scramble share of QB carries, league sack rate per dropback, half-sack probability,
  tackle credits per tackle-ending play (composition chosen by fit), red-zone dispersion (gamma latent concentration), goal-line share of red-zone plays.
No development or forward data enters. Nothing here is a sportsbook quantity.
"""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1

LAST = (2024, 18)


def fnum(v):
    return P1.fnum(v)


def compute(data_dir):
    d = Path(data_dir)
    players = {}
    for r in csv.DictReader(open(d / "players.csv", newline="", encoding="utf-8")):
        players[r["gsis_id"]] = r.get("position")
    tg = defaultdict(lambda: defaultdict(float))          # (s,w,team) offense tallies
    nt_n = nt_d = scr = qbcar = 0.0
    for s in (2022, 2023, 2024):
        with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                    continue
                if r.get("two_point_attempt") == "1" or r.get("qb_kneel") == "1" or r.get("qb_spike") == "1":
                    continue
                w = int(r["week"])
                if (s, w) > LAST:
                    continue
                k = (s, w, r["posteam"]); t = tg[k]
                yl = fnum(r["yardline_100"]); rz = yl is not None and yl <= 20; gl = yl is not None and yl <= 5
                t["plays"] += 1
                if r["qb_dropback"] == "1":
                    t["dropbacks"] += 1
                if r["sack"] == "1":
                    t["sacks"] += 1
                if r["rush_attempt"] == "1" and r["rusher_player_id"]:
                    t["rushes"] += 1
                    if r["rush_touchdown"] == "1":
                        t["rush_td"] += 1
                    if rz:
                        t["rz_rushes"] += 1
                    if gl:
                        t["gl_rushes"] += 1
                    if players.get(r["rusher_player_id"]) == "QB":
                        qbcar += 1
                        if r.get("qb_scramble") == "1":
                            scr += 1
                if r["pass_attempt"] == "1" and r["sack"] != "1":
                    t["attempts"] += 1
                    has = bool(r["receiver_player_id"])
                    nt_d += 1; nt_n += (not has)
                    if has:
                        t["targets"] += 1
                        if rz:
                            t["rz_targets"] += 1
                        if gl:
                            t["gl_targets"] += 1
                    if r["complete_pass"] == "1":
                        t["completions"] += 1
                        if r["pass_touchdown"] == "1":
                            t["pass_td"] += 1
    # defensive credits by team-week from player stats (defender's team)
    tk = defaultdict(float); sacks_credit = defaultdict(float); half = 0.0; full = 0.0
    for s in (2022, 2023, 2024):
        for r in csv.DictReader(open(d / f"stats_player_week_{s}.csv", newline="", encoding="utf-8")):
            if r.get("season_type", "REG") != "REG" or (s, int(r["week"])) > LAST:
                continue
            key = (s, int(r["week"]), r["opponent_team"])                 # keyed by the OFFENSE that was defended against
            tk[key] += (fnum(r.get("def_tackles_solo")) or 0.0) + (fnum(r.get("def_tackle_assists")) or 0.0)
            sk = fnum(r.get("def_sacks")) or 0.0
            sacks_credit[key] += sk
            if sk > 0:
                if abs(sk - round(sk)) > 1e-9:
                    half += 1
                else:
                    full += sk
    keys = [k for k in tg if k in tk and tg[k]["plays"] > 0]
    A = np.array([[tg[k]["rushes"] - tg[k]["rush_td"], tg[k]["completions"] - tg[k]["pass_td"], tg[k]["sacks"]] for k in keys])
    y = np.array([tk[k] for k in keys])
    fits = {}
    for nm, cols in (("E1_rush+completion", [0, 1]), ("E2_rush+completion+sack", [0, 1, 2])):
        X = A[:, cols]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        fits[nm] = {"coef": [float(x) for x in b], "resid_var": float(res.var()), "mean_credits_per_play": float(y.sum() / X.sum())}
    best = min(fits, key=lambda k: fits[k]["resid_var"])
    cols = [0, 1] if best.startswith("E1") else [0, 1, 2]
    mu_c = float(y.sum() / A[:, cols].sum())
    # half sacks: half-credits come in pairs of 0.5 -> events = full + half_credits/2 ; here `half` counts credit rows with a fractional value
    events = full + half / 2.0
    p_half = float((half / 2.0) / events) if events else 0.0
    tot = lambda name: float(sum(tg[k][name] for k in tg))
    consts = {
        "estimated_on": "TRAIN + VALID only (2022 through 2024 wk18)",
        "no_target_rate_of_nonsack_attempts": nt_n / nt_d,
        "scramble_share_of_qb_carries": scr / qbcar,
        "league_sack_rate_per_dropback": tot("sacks") / tot("dropbacks"),
        "half_sack_probability": p_half,
        "tackle_credit_fits": fits, "tackle_eligible_composition": best, "tackle_credits_per_eligible_play": mu_c,
        "gl_share_of_rz_rushes": tot("gl_rushes") / tot("rz_rushes"), "gl_share_of_rz_targets": tot("gl_targets") / tot("rz_targets"),
        "rz_share_of_rushes": tot("rz_rushes") / tot("rushes"), "rz_share_of_targets": tot("rz_targets") / tot("targets"),
        "n_team_games": len(keys)}
    # red-zone dispersion: latent gamma multiplier psi (mean 1, shape c) on the as-of expected red-zone probability
    by_team = defaultdict(list)
    for k in sorted(tg):
        by_team[k[2]].append(k)
    resid, expect = [], []
    for team, ks in by_team.items():
        ew = None
        for k in ks:
            t = tg[k]
            if ew is not None and t["rushes"] >= 5:
                q = ew; R = t["rushes"]; rr = t["rz_rushes"]
                resid.append((rr - R * q) ** 2 - R * q * (1 - q)); expect.append((R * q) ** 2)
            if t["rushes"] > 0:
                x = t["rz_rushes"] / t["rushes"]
                ew = x if ew is None else 0.8 * ew + 0.2 * x
    c_inv = max(float(np.sum(resid) / np.sum(expect)), 1e-3)                # Var excess = R^2 q^2 / c
    consts["rz_latent_gamma_shape"] = 1.0 / c_inv
    consts["rz_latent_note"] = "moment fit of excess-binomial variance of red-zone rushes given as-of team share; one shared latent per team-game scales rush and pass red-zone rates"
    return consts


if __name__ == "__main__":
    import sys
    out = compute(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

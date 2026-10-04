"""
CFB_PHASE1_CONVERGENCE -- simulator convergence check registered in phase1b_simulator_registration.json: two independent N-draw runs per game, Wasserstein-1 / KS between their empirical CDFs of every output series (CDF stability for lattice outcomes).
"""
import numpy as np

import cfb_phase1_sim as SIM


def w1_ks(a, b):
    sa, sb = np.sort(a).astype(float), np.sort(b).astype(float)
    w1 = float(np.abs(sa - sb).mean())
    grid = np.union1d(np.unique(sa), np.unique(sb))
    ks = float(np.max(np.abs(np.searchsorted(sa, grid, side="right") / len(sa) - np.searchsorted(sb, grid, side="right") / len(sb))))
    return w1, ks


def series_of(o):
    out = {"team_plays": o["N"], "team_rush": o["R"], "team_att": o["A"], "team_points": o["points"]}
    for k, d in o["players"].items():
        for i, v in d.items():
            out[f"{k}:{i}"] = v
    return out


def converge(spec, N, seed_a, seed_b):
    a, b = series_of(SIM.simulate_team(spec, N, seed_a)), series_of(SIM.simulate_team(spec, N, seed_b))
    rows = []
    for k in a:
        sd = float(np.concatenate([a[k], b[k]]).std())
        if sd == 0:
            continue
        w1, ks = w1_ks(a[k], b[k]); rows.append((w1 / sd, ks))
    return rows


def criterion(all_rows):
    w = np.array([r[0] for r in all_rows]); k = np.array([r[1] for r in all_rows])
    res = {"n_series": int(len(w)), "median_norm_w1": float(np.median(w)), "p95_norm_w1": float(np.percentile(w, 95)), "p95_ks": float(np.percentile(k, 95))}
    res["pass"] = bool(res["median_norm_w1"] <= 0.03 and res["p95_norm_w1"] <= 0.10 and res["p95_ks"] <= 0.06)
    return res

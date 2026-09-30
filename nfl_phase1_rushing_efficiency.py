"""
NFL_PHASE1_RUSHING_EFFICIENCY  (Phase 1B, shadow research)

Rush yards per carry, TWO processes:
  ordinary   yards < T   : hierarchical pmf + tilt (negative-yard mass, linear yards) with matchup features
  explosive  yards >= T  : hazard h(x) (offset logistic) + tail pmf shrunk to the position level (no player tilt)
compared with a single-process pmf tilt (threshold-free) at T in {15, 20, 25}. Negative yards are supported (bin 0 = -10..-4).
"""
import numpy as np

import nfl_phase1_efficiency as F
import nfl_phase1b_data as B

BIN_VALS = np.array([(lo + hi) / 2.0 for lo, hi in B.RUSH_BINS])
BIN_VALS[0] = -5.0
PHI_SINGLE = np.column_stack([(BIN_VALS < 0).astype(float), (BIN_VALS >= 15).astype(float), np.clip(BIN_VALS, -10, 30) / 10.0])
GROUPS = {"negative": np.where(BIN_VALS < 0)[0], "zero": np.where(BIN_VALS == 0)[0],
          "1-4": np.where((BIN_VALS >= 1) & (BIN_VALS <= 4))[0], "5-9": np.where((BIN_VALS >= 5) & (BIN_VALS <= 9))[0],
          "10-19": np.where((BIN_VALS >= 10) & (BIN_VALS <= 19))[0], "20+": np.where(BIN_VALS >= 20)[0]}


def interactions(P0, R):
    """player tendency (from the baseline pmf) x matching opponent / scheme feature."""
    t_expl = P0[:, BIN_VALS >= 15].sum(1); t_neg = P0[:, BIN_VALS < 0].sum(1)
    tr = F.masks(R)[0]
    def col(a, i):
        x = a[:, i].astype(float); m = np.nanmean(x[tr]); return np.where(np.isnan(x), m, x)
    opp_expl = col(R.fam["opp"], 0); box = col(R.fam["scheme"], 1); stacked = col(R.fam["scheme"], 0)
    ce, cn = t_expl - t_expl[tr].mean(), t_neg - t_neg[tr].mean()
    return np.column_stack([ce * opp_expl, ce * stacked, cn * box])


def two_process_fn(P0, cnt, tr, va, dv, T, tail_tilt=False):
    logP0 = np.log(P0)
    body = BIN_VALS < T; tail = ~body
    phi_body = PHI_SINGLE[body][:, [0, 2]]
    def fn(X, lvl):
        h0 = P0[:, tail].sum(1)
        cnt_h = np.column_stack([cnt[:, body].sum(1), cnt[:, tail].sum(1)])
        logB_h = np.log(np.column_stack([1 - h0, h0]))
        Ph, ih = F.fit_family(logB_h, cnt_h, X, np.array([[0.0], [1.0]]), tr, va, dv)
        Pb0 = P0[:, body] / P0[:, body].sum(1, keepdims=True)
        Pb, ib = F.fit_family(np.log(Pb0), cnt[:, body], X, phi_body, tr, va, dv)
        Pt = P0[:, tail] / P0[:, tail].sum(1, keepdims=True)
        P = np.zeros_like(P0)
        P[:, body] = Ph[:, [0]] * Pb; P[:, tail] = Ph[:, [1]] * Pt
        Pv = np.zeros((int(va.sum()), P0.shape[1]))
        Pv[:, body] = ih["Pv"][:, [0]] * ib["Pv"]; Pv[:, tail] = ih["Pv"][:, [1]] * Pt[va]
        vls = float(F.logscore(Pv, cnt[va]).sum() / cnt[va].sum())
        return P, {"hazard": {k: v for k, v in ih.items() if k != "Pv"}, "body": {k: v for k, v in ib.items() if k != "Pv"}, "valid_logscore_train_only_fit": vls}
    return fn


def run(R, seed_note=""):
    tr, va, dv = F.masks(R)
    cnt = R.cnt["y"]
    F.C.audit_fit("phase1b_rush_hier", [{"s": s, "w": w} for s, w in zip(R.s[tr | va], R.w[tr | va])])
    tuned, table = F.tune_hier(R, cnt, "pl", "pp", "lg", tr, va)
    P0 = F.hier_base(R, "pl", "pp", "lg", tuned["gamma_idx"], tuned["kappa_player"], tuned["kappa_pos"])
    logP0 = np.log(P0)
    out = {"hier_tuning": tuned, "hier_tuning_grid_top5": sorted(table, key=lambda r: r["valid_logscore"])[:5]}
    # reference baselines (pure lookups): league pmf and position pmf, as-of
    P_lg = F.hier_base(R, "pl", "pp", "lg", 0, 1e12, 1e12)
    P_pos = F.hier_base(R, "pl", "pp", "lg", tuned["gamma_idx"], 1e12, 1e-6)
    sc_lg = F.score_rows(P_lg, cnt, BIN_VALS)
    sc0 = F.score_rows(P0, cnt, BIN_VALS)
    out["lookup_baselines"] = {"league_pmf": F.period_summary(sc_lg, R, dv), "position_pmf": F.period_summary(F.score_rows(P_pos, cnt, BIN_VALS), R, dv),
                               "hierarchical_player_pmf(B0)": F.period_summary(sc0, R, dv, sc_lg)}
    out["player_persistence_B0_vs_position"] = F.period_summary(sc0, R, dv, F.score_rows(P_pos, cnt, BIN_VALS))
    ex = interactions(P0, R)
    # single process
    def single(X, lvl):
        return F.fit_family(logP0, cnt, X, PHI_SINGLE, tr, va, dv)
    single_rep, single_preds = F.nested_study(R, cnt, BIN_VALS, single, ex, GROUPS)
    out["single_process"] = single_rep
    out["two_process"] = {}
    preds2 = {}
    for T in (15, 20, 25):
        rep, pr = F.nested_study(R, cnt, BIN_VALS, two_process_fn(P0, cnt, tr, va, dv, T), ex, GROUPS)
        out["two_process"][f"T{T}"] = rep
        preds2[T] = pr
    # choose the structure by VALID log score at each structure's own selected level
    cands = {"single": (single_rep["valid_logscore_by_level"][single_rep["selected_level"]], single_rep["selected_level"], single_preds)}
    for T in (15, 20, 25):
        r = out["two_process"][f"T{T}"]
        cands[f"two_process_T{T}"] = (r["valid_logscore_by_level"][r["selected_level"]], r["selected_level"], preds2[T])
    best = min(cands, key=lambda k: cands[k][0])
    out["selected"] = {"structure": best, "level": cands[best][1], "valid_logscores": {k: v[0] for k, v in cands.items()},
                       "rule": "lowest TRAIN-fit VALID log score across structures at each structure's selected nested level"}
    return out, {"P0": P0, "P_sel": cands[best][2][cands[best][1]], "P_lg": P_lg, "P_pos": P_pos, "cnt": cnt, "vals": BIN_VALS, "single": single_preds, "two": preds2}

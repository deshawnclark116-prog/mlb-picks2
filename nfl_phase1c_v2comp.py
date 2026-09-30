"""
NFL_PHASE1C_V2COMP  (Phase 1C, DEVELOPMENT comparators; nothing here feeds the engine)

Honest comparators for rushing / receiving yards on rows where the v2 recipe is defined:

  * v2 recipe REFIT (nfl_yardage_v2 feature set, XGBoost reg:absoluteerror, v2.PARAMS) on rows through 2024 wk12, early-stopped on 2024 wk13-18,
    scored on 2025 and 2026 wk1-3 -- never scored on its own training rows.  NOTE: the v2 feature set contains bookmaker inputs (spread, implied total);
    they are used ONLY inside this comparator, never in the engine.
  * production v2 artifact (nfl_models/nfl_yardage_v2_work/*.json): valid ONLY on 2026 wk1-3 (future to its training period).
  * historical blend = v2's own `blend_yds` feature.
  * last-8 empirical (from the Phase 1B study).

v2 rows are conditional on the player having PLAYED (v3.Replayer keeps realized participants), so every comparison is conditional on realized participation:
the engine's forecast is its distribution restricted to the draws in which the player is active.  Residual distribution for the point comparators: empirical
residuals of the refit model on 2024 wk13-18 only (before every scored row), added to the point forecast and floored at 0.
"""
import json
from pathlib import Path

import numpy as np

import nfl_phase1_common as C
import nfl_phase1c_evaluate as EV
import nfl_phase1c_metrics as MT
import nfl_yardage_v2 as v2
import nfl_yardage_v3 as v3

SEASONS = [2022, 2023, 2024, 2025, 2026]
MARK = {"rush_yds": ("rushing_yards", "carries", ("RB", "FB"), 5.0, "carry"), "rec_yds": ("receiving_yards", "targets", ("WR", "TE", "RB"), 3.0, "target")}


def v2_rows(data_dir, market):
    d3 = v3.Data(data_dir, SEASONS)
    yds, vol, pos, mn, _ = market
    v3.MARKETS["_c"] = {"vol": vol, "yds": yds, "positions": pos, "min_last3_vol": mn}
    rows = v3.Replayer(d3, "_c").replay()
    del v3.MARKETS["_c"]
    return rows


def fit_refit(rows):
    import xgboost as xgb
    feat = v2.FEATURES
    X = np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in feat] for r in rows], dtype=np.float32)
    y = np.array([r[2] for r in rows], float)
    sw = np.array([(r[0]["s"], r[0]["w"]) for r in rows])
    tr = np.array([C.TRAIN(s, w) for s, w in sw]); va = np.array([C.VALID(s, w) for s, w in sw]); dv = np.array([C.DEV(s, w) for s, w in sw])
    C.audit_fit("v2_refit_comparator", [{"s": s, "w": w} for s, w in sw[tr | va]])
    params = {k: v for k, v in v2.PARAMS.items()}
    b = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr], feature_names=feat), 2000, evals=[(xgb.DMatrix(X[va], label=y[va], feature_names=feat), "va")], early_stopping_rounds=60, verbose_eval=False)
    pred = b.predict(xgb.DMatrix(X, feature_names=feat), iteration_range=(0, b.best_iteration + 1))
    resid = (y - pred)[va]
    return pred, resid, X, y, sw, tr, va, dv


def production_v2(rows, market_name):
    import xgboost as xgb
    path = v2.MODEL_DIR / f"nfl_{market_name}_v2.json"
    if not path.exists():
        return None
    b = xgb.Booster(); b.load_model(str(path))
    X = np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in v2.FEATURES] for r in rows], dtype=np.float32)
    return b.predict(xgb.DMatrix(X, feature_names=v2.FEATURES))


def crps_var(S, y):
    """CRPS for draws with NaN = absent (variable number of draws per row)."""
    Ss = np.sort(S, 1)
    m = np.sum(~np.isnan(S), 1)
    t1 = np.nanmean(np.abs(S - y[:, None]), 1)
    i = np.arange(1, S.shape[1] + 1)[None, :]
    w = np.where(i <= m[:, None], 2 * i - m[:, None] - 1, 0.0)
    t2 = np.nansum(w * np.nan_to_num(Ss), 1) / np.maximum(m, 1) ** 2
    return t1 - t2


def run(ctx, data_dir, N=1000, seed=11):
    d, logs = ctx.run_joint("adjudicated", N, seed, "T24")
    res = {"label": C.DEV_LABEL, "n_draws": N, "note": __doc__, "stats": {}}
    for name, market in MARK.items():
        rows = v2_rows(data_dir, market)
        pred_refit, resid, X, y_all, sw, tr, va, dv = fit_refit(rows)
        pred_prod = production_v2(rows, market[0])
        blend = X[:, v2.FEATURES.index("blend_yds")].astype(float)
        keyrow = {(r[0]["s"], r[0]["w"], r[0]["team"], r[0]["pid"]): i for i, r in enumerate(rows)}
        keys, S = d[name]
        act = d["active_carry" if name == "rush_yds" else "active_target"][1]
        common = [(k, keyrow[k]) for k in keys if k in keyrow and C.DEV(k[0], k[1])]
        ki = {k: i for i, k in enumerate(keys)}
        kk = [k for k, _ in common]; ri = np.array([r for _, r in common]); si = np.array([ki[k] for k in kk])
        Sk = S[si].astype(np.float64); Ak = act[si] > 0.5
        Sc = np.where(Ak, Sk, np.nan)                                       # engine draws restricted to draws in which the player is active
        have = Ak.sum(1) >= 20
        y = y_all[ri]
        eng_med = np.nanmedian(np.where(Ak, Sk, np.nan), 1)
        eng_mean = np.nanmean(np.where(Ak, Sk, np.nan), 1)
        eng_med = np.where(have, eng_med, np.median(Sk, 1))
        eng_mean = np.where(have, eng_mean, Sk.mean(1))
        preds = {"engine_conditional_median": eng_med, "engine_conditional_mean": eng_mean, "v2_recipe_refit_through_2024": pred_refit[ri], "historical_blend": blend[ri]}
        if pred_prod is not None:
            preds["production_v2_artifact(2026 wk1-3 only)"] = pred_prod[ri]
        per = {}
        for tag, f in (("2025", lambda k: k[0] == 2025), ("2026_wk1_3", lambda k: k[0] == 2026), ("combined", lambda k: True)):
            m = np.array([f(k) for k in kk]) & have
            if m.sum() < 30:
                continue
            cell = {"n": int(m.sum())}
            for pn, p in preds.items():
                if pn.startswith("production") and tag != "2026_wk1_3":
                    continue                     # the production artifact was trained through 2025: only 2026 wk1-3 is future to it
                e = p[m] - y[m]
                cell[pn] = {"mae": float(np.abs(e).mean()), "median_ae": float(np.median(np.abs(e))), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                            "accuracy_curve": MT.accuracy_curve(p[m], y[m], MT.TOL[name])}
            # distributions: engine conditional draws vs v2-refit + prior-period empirical residuals
            rr = np.random.default_rng(3)
            Sv2 = np.maximum(pred_refit[ri][m][:, None] + rr.choice(resid, (int(m.sum()), 400)), 0.0)
            cell["crps_engine_conditional"] = float(crps_var(Sc[m], y[m]).mean())
            cell["crps_v2_refit_plus_2024_residuals"] = float(EV.crps(Sv2, y[m]).mean())
            blocks = np.array([f"{k[0]}-{k[1]}" for k, mm in zip(kk, m) if mm])
            e1 = np.abs(eng_med[m] - y[m]); e2 = np.abs(pred_refit[ri][m] - y[m]); e3 = np.abs(blend[ri][m] - y[m])
            cell["engine_vs_v2_refit_mae"] = dict(zip(("improvement", "p_not_better"), C.block_boot(e1, e2, blocks)))
            cell["engine_vs_blend_mae"] = dict(zip(("improvement", "p_not_better"), C.block_boot(e1, e3, blocks)))
            cell["engine_vs_v2_refit_crps"] = dict(zip(("improvement", "p_not_better"), C.block_boot(crps_var(Sc[m], y[m]), EV.crps(Sv2, y[m]), blocks)))
            per[tag] = cell
        res["stats"][name] = per
    return res

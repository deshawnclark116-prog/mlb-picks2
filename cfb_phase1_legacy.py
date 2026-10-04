"""
CFB_PHASE1_LEGACY -- comparison of the new engine with the LEGACY comparators on the SAME rows (read-only use of the legacy code and model files; production untouched).
Legacy models used: the pre-calibration context_v2 CANDIDATE models (trained <= 2023, early-stopped on 2024: no 2025 information) and the old served champion models. The production context models are NOT used: they are Platt-calibrated on completed 2025 (their calibration saw 2025 labels).
Overlap: legacy eligible rows (min-history eligibility of the legacy engines) that are also in the new candidate universe; rows outside the universe are counted and excluded from the same-universe comparison.
Events: rushing_yards > 69.5, passing_touchdowns >= 2, anytime TD >= 1, moneyline win (all derived from the new engine's full distributions).
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
LEGACY = [("rushing_yards", "player"), ("passing_touchdowns", "player"), ("anytime_touchdowns", "player"), ("moneyline", "team")]


def logloss_auc(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())) if pos.any() and (~pos).any() else None
    return {"logloss": ll, "brier": float(np.mean((p - y) ** 2)), "auc": auc, "n": int(len(y)), "base_rate": float(y.mean())}


def legacy_rows(db, season):
    """-> {market: [(key, feat_dict, label)]} using the legacy engines exactly as cfb_context_v2.run() does (read-only)."""
    import cfb_context_v2 as cx
    import cfb_serving_builder_a as sb
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ctx = cx.build_context(con); ratings = cx.build_ratings(con)
    out = {}
    for mkt, kind in LEGACY:
        if kind == "player":
            eng = sb.SeasonEngine(con, mkt, season) if mkt in sb.MARKETS else sb.AnytimeTouchdowns_engine(con, season) if False else (sb.SeasonEngine(con, mkt, season) if mkt in sb.MARKETS else sb.AnytimeTouchdownEngine(con, season))
            rows = [((r[2], r[4], r[0]), cx.add_context(r[5], season, r[0], r[4], r[2], r[3], ctx, ratings), r[5], r[6]) for r in eng.replay()]
        else:
            rows = [((r[0], r[2]), cx.add_context(r[3], season, None, r[2], r[0], r[1], ctx, ratings), r[3], r[4]) for r in sb.MoneylineEngine(con, season).replay()]
        out[mkt] = rows
    con.close()
    return out


def legacy_predictions(rows, mkt):
    """candidate context model (raw probability) and the old served champion model on the legacy rows."""
    import xgboost as xgb
    import cfb_context_v2 as cx
    import cfb_serving_builder_a as sb
    W = REPO / "cfb_models" / "cfb_context_v2_work"
    cols = json.loads((W / f"cfb_{mkt}_context_candidate_columns.json").read_text())
    b = xgb.Booster(); b.load_model(str(W / f"cfb_{mkt}_context_candidate.json"))
    X = cx.mat([r[1] for r in rows], cols)
    raw = b.predict(xgb.DMatrix(X, feature_names=cols), iteration_range=(0, b.best_iteration + 1))
    cand = cx.to_prob(mkt, raw)
    if mkt in sb.MARKETS:
        old_path = sb.MARKETS[mkt]["model_dir"] / f"cfb_{mkt}.json"; old_cols = sb.MARKETS[mkt]["features"]
    elif mkt == "anytime_touchdowns":
        old_path = sb.ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns.json"; old_cols = sb.ANYTIME_TD_FEATURES
    else:
        old_path = sb.MONEYLINE_MODEL_DIR / "cfb_moneyline.json"; old_cols = sb.MONEYLINE_FEATURES
    ob = xgb.Booster(); ob.load_model(str(old_path))
    oraw = ob.predict(xgb.DMatrix(cx.mat([r[2] for r in rows], old_cols), feature_names=old_cols), iteration_range=(0, ob.best_iteration + 1))
    return cand, cx.to_prob(mkt, oraw)


def compare(db, season, events, wins):
    """events: {(team, week, player_id): {'p_rush69','p_ptd2','p_atd'}} from the new engine; wins: {(team, week): p_win}. Returns per-market overlap-universe comparison."""
    res = {}
    try:
        rows = legacy_rows(db, season)
    except Exception as e:                                                      # noqa
        return {"status": "LEGACY_UNAVAILABLE", "reason": f"{type(e).__name__}: {e}"[:300]}
    for mkt, kind in LEGACY:
        rr = rows[mkt]
        if kind == "player":
            field = {"rushing_yards": "p_rush69", "passing_touchdowns": "p_ptd2", "anytime_touchdowns": "p_atd"}[mkt]
            keep = [i for i, r in enumerate(rr) if r[0] in events]
            y_fn = {"rushing_yards": lambda a: float(a > 69.5), "passing_touchdowns": lambda a: float(a >= 2), "anytime_touchdowns": lambda a: float(a >= 1)}[mkt]
            ours = np.array([events[rr[i][0]][field] for i in keep])
        else:
            keep = [i for i, r in enumerate(rr) if r[0] in wins]
            y_fn = lambda a: float(a)
            ours = np.array([wins[rr[i][0]] for i in keep])
        if not keep:
            res[mkt] = {"status": "NO_OVERLAP", "legacy_rows": len(rr)}; continue
        cand, old = legacy_predictions(rr, mkt)
        y = np.array([y_fn(rr[i][3]) for i in keep])
        res[mkt] = {"legacy_rows": len(rr), "overlap_rows_in_new_universe": len(keep), "outside_universe_rows": len(rr) - len(keep), "new_engine": logloss_auc(ours, y), "legacy_context_v2_candidate_raw": logloss_auc(np.asarray(cand)[keep], y),
                    "legacy_old_served_champion_raw": logloss_auc(np.asarray(old)[keep], y), "note": "same rows for all three; production Platt-calibrated models excluded (calibrated on 2025)"}
    return {"status": "OK", "markets": res}

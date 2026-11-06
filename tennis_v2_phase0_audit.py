#!/usr/bin/env python3
"""Tennis Outcome Engine V2 - Phase0 forensic audit driver (research only; nothing in production is modified or imported for writing).

Regenerates every phase0_* artifact from the hash-pinned 2015-2024 TML files. Evidence grade RETROSPECTIVE_BURNED. Seasons 2025 (sealed) and 2026 (exposed) are never read.
Usage: TENNIS_V2_RAW=/path/to/raw python tennis_v2_phase0_audit.py [out_dir]"""
import gzip
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import tennis_v2_baselines as BL
import tennis_v2_data as TD
import tennis_v2_incumbent as INC
import tennis_v2_metrics as M

REPO = Path(__file__).resolve().parent
OUT = TD.OUT
VAL, HOLD = INC.VAL, INC.HOLD
PERIODS = {"VAL": VAL, "HOLDOUT": HOLD}
TOURS = ("atp", "wta")
MARKERS = ("RET", "W/O", "WEA", "DEF", "ABD", "ABN")
ARCHITECTURE = {"moneyline": ("PARTIAL", "single Elo/surface-Elo rating, no serve/return, fatigue or rank input"),
                "set_score": ("PARTIAL", "coherent race-to-N distribution but iid sets from one rating; beaten by an empirical split on the same winner probability"),
                "total_games": ("INSUFFICIENT", "two independent per-player games-per-set smoothings times expected sets; no match distribution and no opponent interaction"),
                "games_spread": ("INSUFFICIENT", "difference of per-player margin smoothings; no distribution"),
                "aces": ("INSUFFICIENT", "per-player rate times volume with a synthetic-line hit-rate gate; no serve/return interaction or count distribution"),
                "double_faults": ("INSUFFICIENT", "per-player rate times volume with a synthetic-line hit-rate gate")}
RANK = {"SURVIVES": 3, "WEAK": 2, "REJECTED": 1, "BLOCKED_DATA": 0}


def inp(o, per):
    return per[0] <= o["match_date"] <= per[1]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda x: float(x) if hasattr(x, "item") else str(x)) + "\n")


def worse(vs):
    return min(vs, key=lambda v: RANK[v])


# ------------------------------------------------------------------ helpers
def marker_of(score):
    toks = (score or "").split()
    for t in reversed(toks):
        if t in MARKERS:
            return t
    return None


def seg_defs(o, p):
    pm = max(p, 1 - p)
    return {"best_of_3": o["best_of"] == 3, "best_of_5": o["best_of"] == 5, "hard": o["surface"] == "Hard", "clay": o["surface"] == "Clay", "grass": o["surface"] == "Grass",
            "carpet": o["surface"] == "Carpet", "large_favorite_ge_0.75": pm >= 0.75, "close_matchup_le_0.60": pm <= 0.60,
            "early_round": o["round"] in ("R128", "R64", "R32"), "late_round": o["round"] in ("QF", "SF", "F")}


def mae_by_segment(obs, preds, actual_key="actual", p_key="p_blend"):
    out = {}
    names = list(seg_defs(obs[0], 0.5)) if obs else []
    for s in names:
        idx = [i for i, o in enumerate(obs) if seg_defs(o, o[p_key])[s]]
        if idx:
            out[s] = {k: round(float(np.mean(np.abs(np.array(v)[idx] - np.array([obs[i][actual_key] for i in idx])))), 4) for k, v in preds.items()} | {"n": len(idx)}
    return out


# --------------------------------------------------------------- per-tour compute
def compute_tour(tour):
    con = TD.build_db(tour)
    rows = TD.all_matches(con)
    res = {"tour": tour, "rows": rows, "con": con}
    # moneyline
    ml_as = INC.moneyline(rows, "as_implemented"); ml_c = BL.moneyline_canonical(rows)
    res["ml_slope"] = BL.moneyline_models(ml_c)
    res["ml_as"], res["ml_c"] = ml_as, ml_c
    # total games
    tg_as = INC.total_games(rows, "as_implemented")
    tg_c, tg_consts = BL.total_games_baselines(rows)
    res["tg_as"], res["tg_c"], res["tg_consts"] = tg_as, tg_c, tg_consts
    # spread, set score
    res["sp_as"] = INC.games_spread(rows, "as_implemented"); res["sp_c"] = BL.spread_baselines(rows)
    ss_as, _ = INC.set_score(rows, "as_implemented"); res["ss_as"] = ss_as; res["ss_c"] = BL.set_score_baselines(rows)
    # serve count
    cidx = {r["match_id"]: i for i, r in enumerate(INC.order_rows([r for r in rows if r["surface"]], "canonical"))}
    pm = [dict(r) for r in con.execute("SELECT * FROM player_matches WHERE aces IS NOT NULL AND double_faults IS NOT NULL AND serve_points>0")]
    pm = [r for r in pm if r["match_id"] in cidx]
    pm.sort(key=lambda r: (cidx[r["match_id"]], r["player_id"]))
    res["pm"] = pm
    res["aces"] = BL.serve_count_obs(pm, "aces"); res["dfs"] = BL.serve_count_obs(pm, "double_faults")
    return res


def prob_block(tr, per_name, per):
    """moneyline results for one tour/period: RAW vs CLEAN, identical rows, verdict."""
    ml_as = {o["match_id"]: o for o in tr["ml_as"] if inp(o, per)}
    out = {}
    canon = {o["match_id"]: o for o in tr["ml_c"] if inp(o, per)}
    is_wo = {r["match_id"]: ("W/O" in (r["score"] or "")) for r in tr["rows"]}
    for pop in ("RAW", "CLEAN"):
        ids = [m for m in ml_as if m in canon and canon[m]["p_human"] is not None and not is_wo[m] and (pop == "RAW" or not canon[m]["is_incomplete"])]
        y = np.array([canon[m]["p1_wins"] for m in ids]); inc = np.array([ml_as[m]["p1_prob"] for m in ids])
        models = {"incumbent_as_implemented": inc, "incumbent_blend_canonical": np.array([canon[m]["p_blend"] for m in ids]), "elo_overall": np.array([canon[m]["p_overall"] for m in ids]),
                  "elo_surface_shrunk": np.array([canon[m]["p_surface_shrunk"] for m in ids]), "rank_logistic": np.array([canon[m]["p_rank"] for m in ids]), "human": np.array([canon[m]["p_human"] for m in ids])}
        met = {k: {kk: vv for kk, vv in M.prob_metrics(v, y).items() if kk != "buckets"} for k, v in models.items()}
        simple = {k: (models[k] - y) ** 2 for k in ("incumbent_blend_canonical", "elo_overall", "elo_surface_shrunk", "rank_logistic")}
        vd = M.verdict((inc - y) ** 2, simple, ece_val=met["incumbent_as_implemented"]["ece_10bin"], prob=True)
        hv = M.paired_boot((inc - y) ** 2, (models["human"] - y) ** 2)
        out[pop] = {"n": len(ids), "metrics": met, "verdict": vd, "incumbent_vs_human_paired_brier": hv, "incumbent_buckets": M.bucket_table(inc, y)}
    return out


def tg_block(tr, per):
    asi = {o["match_id"]: o for o in tr["tg_as"] if inp(o, per)}
    can = {o["match_id"]: o for o in tr["tg_c"] if inp(o, per) and o["pred"]["human"] is not None}
    ids = sorted(m for m in asi if m in can)
    act = np.array([can[m]["actual"] for m in ids], float)
    inc_as = np.array([asi[m]["predicted_mean"] for m in ids]); naive = np.array([asi[m]["naive_mean"] for m in ids])
    inc_c = np.array([can[m]["predicted_mean"] for m in ids])
    names = list(can[ids[0]]["pred"])
    preds = {k: np.array([can[m]["pred"][k] for m in ids]) for k in names}
    preds["incumbent_as_implemented"] = inc_as; preds["incumbent_canonical"] = inc_c; preds["incumbent_own_naive"] = naive
    metrics = {k: M.count_metrics(v, act) for k, v in preds.items()}
    simple_names = [k for k in names if k != "human"]
    losses = {k: np.abs(preds[k] - act) for k in simple_names}
    vd = M.verdict(np.abs(inc_as - act), losses)
    hv = M.paired_boot(np.abs(inc_as - act), np.abs(preds["human"] - act))
    obs = [can[m] for m in ids]
    seg_p = {k: preds[k] for k in ("incumbent_as_implemented", "human", vd["strongest_simple"], "bestof_fit")}
    segs = mae_by_segment(obs, seg_p)
    return {"n": len(ids), "metrics": metrics, "verdict": vd, "incumbent_vs_human_paired_mae": hv, "segments_mae": segs,
            "order_leak_share_as_implemented": float(np.mean([asi[m]["order_leak"] for m in ids])), "order_leak_share_canonical": float(np.mean([can[m]["order_leak"] for m in ids])),
            "incumbent_gate_improvement_vs_own_naive": float((np.abs(naive - act).mean() - np.abs(inc_as - act).mean()) / np.abs(naive - act).mean())}, ids


def sp_block(tr, per):
    asi = {o["match_id"]: o for o in tr["sp_as"] if inp(o, per)}
    can = {o["match_id"]: o for o in tr["sp_c"] if inp(o, per)}
    ids = sorted(m for m in asi if m in can)
    act = np.array([can[m]["actual"] for m in ids], float)
    preds = {"incumbent_as_implemented": np.array([asi[m]["predicted"] for m in ids]), "incumbent_canonical": np.array([can[m]["predicted"] for m in ids]), "incumbent_own_naive": np.array([asi[m]["naive"] for m in ids]),
             "zero_margin": np.zeros(len(ids)), "elo_bucket_fit_mean": np.array([can[m]["elo_bucket"] for m in ids])}
    metrics = {k: M.count_metrics(v, act, tiers=False) for k, v in preds.items()}
    losses = {k: np.abs(preds[k] - act) for k in ("zero_margin", "elo_bucket_fit_mean")}
    vd = M.verdict(np.abs(preds["incumbent_as_implemented"] - act), losses)
    return {"n": len(ids), "metrics": metrics, "verdict": vd}


def ss_block(tr, per):
    asi = {o["match_id"]: o for o in tr["ss_as"] if inp(o, per)}
    can = {o["match_id"]: o for o in tr["ss_c"] if inp(o, per)}
    ids = sorted(m for m in asi if m in can)
    act = [can[m]["actual"] for m in ids]
    dists = {"incumbent_as_implemented": [asi[m]["dist"] for m in ids], "incumbent_canonical": [can[m]["dist"] for m in ids], "category_frequency": [can[m]["d_freq"] for m in ids],
             "winner_prob_x_conditional_split": [can[m]["d_wp"] for m in ids], "human": [can[m]["d_human"] for m in ids]}
    metrics = {k: M.multiclass(v, act) for k, v in dists.items()}
    ll = {k: M.per_row_multiclass_ll(v, act) for k, v in dists.items()}
    simple = {k: ll[k] for k in ("category_frequency", "winner_prob_x_conditional_split")}
    vd = M.verdict(ll["incumbent_as_implemented"], simple)
    top = [(max(d.values()), max(d, key=d.get) == a) for d, a in zip(dists["incumbent_as_implemented"], act)]
    cal = []
    for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)):
        sel = [h for p, h in top if lo <= p < hi]
        if sel:
            cal.append({"top_prob_bin": "%.1f-%.1f" % (lo, min(hi, 1.0)), "n": len(sel), "mean_top_prob": float(np.mean([p for p, h in top if lo <= p < hi])), "observed": float(np.mean(sel))})
    hv = M.paired_boot(ll["incumbent_as_implemented"], ll["human"])
    return {"n": len(ids), "metrics": metrics, "verdict": vd, "calibration_by_top_probability": cal, "incumbent_vs_human_paired_logloss": hv}


def serve_block(tr, key, per):
    obs = [o for o in tr[key] if inp(o, per)]
    act = np.array([o["actual"] for o in obs], float)
    dev = [o for o in tr[key] if o["match_date"] <= BL.DEV_END]
    fit = defaultdict(list)
    for o in dev:
        fit[(o["surface"], o["best_of"])].append(o["actual"])
    fitm = {k: float(np.mean(v)) for k, v in fit.items()}
    glob = float(np.mean([o["actual"] for o in dev]))
    preds = {"incumbent": np.array([o["pred"] for o in obs]), "incumbent_own_naive": np.array([o["naive"] for o in obs]), "prior10_count_mean": np.array([o["prior10_mean"] for o in obs]),
             "surface_bestof_fit_mean": np.array([fitm.get((o["surface"], o["best_of"]), glob) for o in obs])}
    metrics = {k: M.count_metrics(v, act, tiers=False) for k, v in preds.items()}
    losses = {k: np.abs(preds[k] - act) for k in ("incumbent_own_naive", "prior10_count_mean", "surface_bestof_fit_mean")}
    vd = M.verdict(np.abs(preds["incumbent"] - act), losses)
    # oracle decomposition (descriptive; postgame): actual volume x predicted rate, predicted volume x actual rate
    sv_o = np.array([o["actual_sv"] for o in obs], float); rate = np.array([o["pred_rate"] for o in obs]); exp_sv = np.array([o["exp_sv"] for o in obs])
    orc = {"mae_actual_serve_points_x_pred_rate": float(np.abs(sv_o * rate - act).mean()), "mae_pred_serve_points_x_actual_rate": float(np.abs(exp_sv * (act / sv_o) - act).mean()),
           "mae_incumbent": float(np.abs(preds["incumbent"] - act).mean()),
           "serve_point_volume_mae": float(np.abs(exp_sv - sv_o).mean()), "serve_point_volume_mean": float(sv_o.mean())}
    # dispersion: Var(actual - pred) vs Poisson variance of the predicted mean
    pm = preds["incumbent"]
    deciles = np.quantile(pm, np.linspace(0, 1, 6)[1:-1])
    disp = []
    bins = np.searchsorted(deciles, pm)
    for b in range(5):
        s = bins == b
        if s.any():
            disp.append({"bin": b, "n": int(s.sum()), "mean_pred": float(pm[s].mean()), "mean_actual": float(act[s].mean()), "var_actual": float(act[s].var()), "poisson_var": float(pm[s].mean())})
    return {"n": len(obs), "metrics": metrics, "verdict": vd, "oracle_decomposition": orc, "dispersion_by_quintile": disp}, obs


def total_games_oracle_ceiling(tr, per):
    """DESCRIPTIVE postgame information ceilings for total games (in-sample OLS, <=6 parameters, ~5000 rows): how much error is removable by KNOWING the realised set count or the realised serve-point
    win rates. Oracles use postgame facts: they are not forecasts and cannot be shipped; they only size the headroom each information source could ever offer."""
    pmm = defaultdict(list)
    for r in tr["pm"]:
        pmm[r["match_id"]].append(r)
    rows = []
    for o in tr["tg_c"]:
        if not inp(o, per) or o["pred"]["human"] is None:
            continue
        two = pmm.get(o["match_id"], [])
        if len(two) != 2:
            continue
        spw = [(x["first_serve_won"] + x["second_serve_won"]) / x["serve_points"] for x in two]
        rows.append((o["actual"], o["best_of"] == 5, o["p_blend"], o["n_sets"], spw[0] + spw[1], abs(spw[0] - spw[1])))
    if not rows:
        return None
    a = np.array(rows, float); y = a[:, 0]; bo5 = a[:, 1]; pm = np.abs(a[:, 2] - 0.5)
    def fit(cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.abs(X @ beta - y).mean())
    out = {"n": len(y), "mae_best_of_only": fit([bo5]), "mae_best_of_plus_prematch_mismatch": fit([bo5, pm, pm * bo5]),
           "mae_best_of_plus_realised_set_count_oracle": fit([bo5, a[:, 3], a[:, 3] * bo5]),
           "mae_best_of_plus_realised_serve_win_oracle": fit([bo5, a[:, 4], a[:, 5], a[:, 4] * bo5]),
           "mae_best_of_plus_both_oracles": fit([bo5, a[:, 3], a[:, 3] * bo5, a[:, 4], a[:, 5]])}
    base = out["mae_best_of_only"]
    for k in list(out):
        if k.startswith("mae_") and k != "mae_best_of_only":
            out["rel_reduction_" + k[4:]] = 1 - out[k] / base
    return out


def run_markets(tours):
    market = {"protocol": "protocol.json native_verdict_rules", "evidence_grade": "RETROSPECTIVE_BURNED", "tours": {}}
    keep = {}
    for tour in TOURS:
        tr = tours[tour]; t = {}
        keep[tour] = {}
        for mk in ("moneyline", "total_games", "games_spread", "set_score", "aces", "double_faults"):
            t[mk] = {}
        for pn, per in PERIODS.items():
            t["moneyline"][pn] = prob_block(tr, pn, per)
            blk, ids = tg_block(tr, per); t["total_games"][pn] = blk; keep[tour][("tg", pn)] = ids
            t["total_games"][pn]["oracle_ceiling_descriptive"] = total_games_oracle_ceiling(tr, per)
            t["games_spread"][pn] = sp_block(tr, per)
            t["set_score"][pn] = ss_block(tr, per)
            t["aces"][pn], _ = serve_block(tr, "aces", per)
            t["double_faults"][pn], _ = serve_block(tr, "dfs", per)
        for mk in t:
            vs = []
            for pn in PERIODS:
                v = t[mk][pn]["RAW"]["verdict"]["verdict"] if mk == "moneyline" else t[mk][pn]["verdict"]["verdict"]
                vs.append(v)
            t[mk]["native_performance_verdict"] = worse(vs)
            t[mk]["verdict_by_period"] = dict(zip(PERIODS, vs))
            t[mk]["architecture_status"] = ARCHITECTURE[mk][0]; t[mk]["architecture_reason"] = ARCHITECTURE[mk][1]
        market["tours"][tour] = t
    return market, keep


# ------------------------------------------------------------- source / population
def source_audit(tours):
    out = {"evidence_grade": "RETROSPECTIVE_BURNED", "files": {}, "per_tour_season": {}, "findings": []}
    for name, meta in TD.MANIFEST["files"].items():
        out["files"][name] = {"sha256": meta["sha256"], "bytes": meta["bytes"], "rows": meta["rows"]}
    for tour in TOURS:
        con = tours[tour]["con"]
        for season in TD.SEASONS:
            lo, hi = "%d-01-01" % season, "%d-12-31" % season
            q = lambda sql: con.execute(sql, (lo, hi)).fetchone()[0]
            n = q("select count(*) from matches where match_date between ? and ?")
            d = {"matches": n,
                 "missing_surface": q("select count(*) from matches where match_date between ? and ? and (surface is null or surface='')"),
                 "missing_score": q("select count(*) from matches where match_date between ? and ? and (score is null or score='')"),
                 "incomplete": q("select count(*) from matches where match_date between ? and ? and is_incomplete=1"),
                 "missing_best_of": q("select count(*) from matches where match_date between ? and ? and best_of is null"),
                 "missing_winner_rank": q("select count(*) from matches where match_date between ? and ? and winner_rank is null"),
                 "missing_loser_rank": q("select count(*) from matches where match_date between ? and ? and loser_rank is null"),
                 "player_match_rows": q("select count(*) from player_matches where match_date between ? and ?"),
                 "serve_stats_rows": q("select count(*) from player_matches where match_date between ? and ? and serve_points is not null and serve_points>0"),
                 "levels": dict(con.execute("select tourney_level,count(*) from matches where match_date between ? and ? group by 1 order by 1", (lo, hi)).fetchall()),
                 "rounds": dict(con.execute("select round,count(*) from matches where match_date between ? and ? group by 1 order by 1", (lo, hi)).fetchall())}
            out["per_tour_season"]["%s_%d" % (tour, season)] = d
    out["findings"] = [
        "ATP files are a commit-pinned raw URL (immutable); WTA files come from the mutable TML website: the committed WTA gate reports cannot be reproduced bit-for-bit from the pinned WTA hashes (see phase0_current_engine_audit reproduction).",
        "ATP has no Challenger or qualifying rows; WTA has tour-level plus ITF-level rows ('I' level). Challenger coverage is ABSENT for both tours.",
        "README terms of the TML dataset forbid redistribution: raw CSVs are fetched at run time and hash-verified, never committed.",
        "ATP 2026 upstream was structurally stale at the pre-protocol probe (disclosed in protocol.json); no outcomes were read."]
    return out


def population_definition(tours):
    out = {"definitions": {"RAW": "every parsed match row with decided winner and loser, W/O excluded for evaluation ledgers", "COMPLETED": "is_incomplete == 0 and a parseable score",
                           "ELIGIBLE": "the incumbent's own per-market eligibility (see protocol.json populations)"}, "tours": {}}
    for tour in TOURS:
        tr = tours[tour]; rows = tr["rows"]; t = {"by_period": {}}
        mk = Counter(marker_of(r["score"]) or "NONE" for r in rows)
        t["marker_counts_all_2015_2024"] = dict(sorted(mk.items()))
        t["marker_counts_by_season"] = {str(s): dict(sorted(Counter(marker_of(r["score"]) or "NONE" for r in rows if r["match_date"].startswith(str(s))).items())) for s in TD.SEASONS}
        ids_with_serve = {r["match_id"] for r in tr["pm"]}
        for pn, per in PERIODS.items():
            rr = [r for r in rows if per[0] <= r["match_date"] <= per[1]]
            ret = [r for r in rr if r["is_incomplete"]]
            comp = [r for r in rr if not r["is_incomplete"]]
            d = {"raw_rows": len(rr), "walkovers": sum("W/O" in (r["score"] or "") for r in rr), "retired_or_incomplete": len(ret),
                 "retired_share": len(ret) / len(rr), "completed": len(comp), "completed_with_serve_stats": sum(r["match_id"] in ids_with_serve for r in comp),
                 "retired_with_serve_stats": sum(r["match_id"] in ids_with_serve for r in ret),
                 "eligible": {"moneyline_incumbent": sum(inp(o, per) for o in tr["ml_as"]), "total_games_incumbent": sum(inp(o, per) for o in tr["tg_as"]),
                              "games_spread_incumbent": sum(inp(o, per) for o in tr["sp_as"]), "set_score_incumbent": sum(inp(o, per) for o in tr["ss_as"]),
                              "aces_player_matches": sum(inp(o, per) for o in tr["aces"]), "double_faults_player_matches": sum(inp(o, per) for o in tr["dfs"]),
                              "total_games_identical_rows_with_baselines": sum(1 for o in tr["tg_c"] if inp(o, per) and o["pred"]["human"] is not None)}}
            # moneyline RAW vs CLEAN metric difference (incumbent, as implemented)
            ml = {o["match_id"]: o for o in tr["ml_as"] if inp(o, per)}
            for lab, sel in (("RAW_no_WO", [o for o in ml.values() if "W/O" not in (next(r for r in [rows_by(tr)[o["match_id"]]])["score"] or "")]), ("CLEAN", [o for o in ml.values() if not o["is_incomplete"]])):
                y = np.array([o["p1_wins"] for o in sel]); p = np.array([o["p1_prob"] for o in sel])
                d["moneyline_" + lab] = {"n": len(sel), "brier": M.brier(p, y), "log_loss": M.log_loss(p, y), "accuracy": float(np.mean((p >= .5) == (y == 1)))}
            t["by_period"][pn] = d
        out["tours"][tour] = t
    return out


_RB = {}


def rows_by(tr):
    k = tr["tour"]
    if k not in _RB:
        _RB[k] = {r["match_id"]: r for r in tr["rows"]}
    return _RB[k]


# ------------------------------------------------------------------ ordering
def order_audit(tours):
    out = {}
    for tour in TOURS:
        tr = tours[tour]; rows = [r for r in tr["rows"] if r["surface"] and r["winner_id"] and r["loser_id"]]
        asr = INC.order_rows(rows, "as_implemented")
        pos = {r["match_id"]: i for i, r in enumerate(asr)}
        bytour = defaultdict(list)
        for r in rows:
            if r["round"] in TD.ROUND_RANK:
                bytour[(r["match_date"], r["tourney_id"])].append(r)
        pairs = inv = 0
        for ms in bytour.values():
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = ms[i], ms[j]
                    ra, rb = TD.ROUND_RANK[a["round"]], TD.ROUND_RANK[b["round"]]
                    if ra == rb:
                        continue
                    pairs += 1
                    first_by_round = a if ra < rb else b
                    second_by_round = b if ra < rb else a
                    if pos[first_by_round["match_id"]] > pos[second_by_round["match_id"]]:
                        inv += 1
        d = {"within_tournament_cross_round_pairs": pairs, "inverted_pairs": inv, "inversion_rate": inv / pairs if pairs else None}
        for pn, per in PERIODS.items():
            ml_a = [o for o in tr["ml_as"] if inp(o, per)]; ml_c = [o for o in tr["ml_c"] if inp(o, per)]
            tg_a = [o for o in tr["tg_as"] if inp(o, per)]; tg_c = [o for o in tr["tg_c"] if inp(o, per)]
            ya = np.array([o["p1_wins"] for o in ml_a]); pa = np.array([o["p1_prob"] for o in ml_a])
            yc = np.array([o["p1_wins"] for o in ml_c]); pc = np.array([o["p_blend"] for o in ml_c])
            d[pn] = {"moneyline_exposed_share_as_implemented": float(np.mean([o["order_leak"] for o in ml_a])), "moneyline_exposed_share_canonical": float(np.mean([o["order_leak"] for o in ml_c])),
                     "moneyline_brier_as_implemented": M.brier(pa, ya), "moneyline_brier_canonical": M.brier(pc, yc),
                     "moneyline_logloss_as_implemented": M.log_loss(pa, ya), "moneyline_logloss_canonical": M.log_loss(pc, yc),
                     "total_games_exposed_share_as_implemented": float(np.mean([o["order_leak"] for o in tg_a])), "total_games_exposed_share_canonical": float(np.mean([o["order_leak"] for o in tg_c])),
                     "total_games_mae_as_implemented": float(np.mean([abs(o["predicted_mean"] - o["actual"]) for o in tg_a])), "total_games_mae_canonical": float(np.mean([abs(o["predicted_mean"] - o["actual"]) for o in tg_c]))}
            # exposed-vs-clean subset moneyline brier (as implemented): does leakage inflate performance
            ex = [o for o in ml_a if o["order_leak"]]; cl = [o for o in ml_a if not o["order_leak"]]
            d[pn]["moneyline_brier_exposed_subset"] = M.brier(np.array([o["p1_prob"] for o in ex]), np.array([o["p1_wins"] for o in ex])) if ex else None
            d[pn]["moneyline_brier_unexposed_subset"] = M.brier(np.array([o["p1_prob"] for o in cl]), np.array([o["p1_wins"] for o in cl]))
            d[pn]["moneyline_exposed_n"] = len(ex)
        out[tour] = d
    return out


# ------------------------------------------------------------------ identity
def norm(name):
    import tennis_serving_builder_a as SB
    return SB.normalize_name(name)


def identity_audit(tours):
    import tennis_serving_builder_a as SB
    out = {"scope": "historical TML identity quality plus replay of the live resolver (exact normalized match, then unique-surname fallback) on the pick log; ESPN data not retrieved", "tours": {}}
    names = defaultdict(set); ids = defaultdict(set); tour_ids = {"atp": set(), "wta": set()}
    raw_names = set(); lost_chars = Counter()
    state_rows = []
    for tour in TOURS:
        for r in tours[tour]["rows"]:
            for pid, nm in ((r["winner_id"], r["winner_name"]), (r["loser_id"], r["loser_name"])):
                if pid and nm:
                    names[(tour, norm(nm))].add(pid); ids[(tour, pid)].add(nm); tour_ids[tour].add(pid); raw_names.add(nm)
                    for ch in nm:
                        if ord(ch) > 127 and not unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode():
                            lost_chars[ch] += 1
            state_rows.append((r["match_date"], r["match_id"], r))
    for tour in TOURS:
        nm = {k[1]: v for k, v in names.items() if k[0] == tour}
        idm = {k[1]: v for k, v in ids.items() if k[0] == tour}
        sur = defaultdict(set)
        for n_, p in nm.items():
            sur[n_.split()[-1] if n_ else ""].update(p)
        out["tours"][tour] = {"players_ids": len(idm), "normalized_names": len(nm), "normalized_names_with_multiple_ids": sum(1 for v in nm.values() if len(v) > 1),
                              "ids_with_multiple_raw_names": sum(1 for v in idm.values() if len(v) > 1), "surname_groups": len(sur), "surnames_shared_by_multiple_players": sum(1 for v in sur.values() if len(v) > 1),
                              "players_in_shared_surname_groups": sum(len(v) for v in sur.values() if len(v) > 1),
                              "multi_id_name_examples": sorted([[k, sorted(v)] for k, v in nm.items() if len(v) > 1])[:10]}
    both = tour_ids["atp"] & tour_ids["wta"]
    nm_cross = {k[1] for k in names if k[0] == "atp"} & {k[1] for k in names if k[0] == "wta"}
    out["cross_tour"] = {"shared_ids": len(both), "shared_normalized_names": len(nm_cross), "shared_name_examples": sorted(nm_cross)[:10],
                         "note": "the production DB holds both tours in one sqlite and builds ONE name_index (later row wins), so a cross-tour name collision silently re-points the id"}
    out["normalization"] = {"non_ascii_chars_lost_entirely_by_normalize_name": dict(lost_chars), "raw_names_changed_by_normalization": sum(1 for n_ in raw_names if SB.normalize_name(n_) != re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n_.lower())).strip())}
    # combined production-style state (both tours, ordered match_date,match_id) and replay of the live resolver on the pick log
    state_rows.sort(key=lambda x: (x[0], x[1]))
    name_index = {}; tvotes = defaultdict(lambda: defaultdict(int)); raw_by_norm = defaultdict(set)
    for _, _, r in state_rows:
        if r["winner_id"] and r["winner_name"]:
            name_index[norm(r["winner_name"])] = r["winner_id"]; raw_by_norm[norm(r["winner_name"])].add(r["winner_name"])
        if r["loser_id"] and r["loser_name"]:
            name_index[norm(r["loser_name"])] = r["loser_id"]; raw_by_norm[norm(r["loser_name"])].add(r["loser_name"])
        if r["tourney_name"] and r["surface"]:
            tvotes[norm(r["tourney_name"])][r["surface"]] += 1
    state = {"name_index": name_index, "tourney_surface": {t: max(v, key=v.get) for t, v in tvotes.items()}}
    picks = [json.loads(l) for l in (REPO / "docs" / "tennis_picks_log.jsonl").read_text().splitlines() if l.strip()]
    cls = Counter(); examples = defaultdict(list); id_disagree = 0; players = {}; guess_detail = []
    for p in picks:
        for side in ("player1", "player2"):
            players[(p["tour"], p[side], p[side + "_id"])] = 1
    for (tour, nm_, lid), _ in sorted(players.items()):
        n_ = SB.normalize_name(nm_)
        if nm_ in raw_by_norm.get(n_, ()):
            c = "EXACT"
        elif n_ in name_index:
            c = "NORMALIZED_EXACT"
        else:
            surname = n_.split()[-1] if n_ else ""
            cand = {pid for k, pid in name_index.items() if k.split()[-1:] == [surname]}
            c = "SURNAME_FALLBACK_GUESS" if len(cand) == 1 else "AMBIGUOUS" if len(cand) > 1 else "UNMATCHED_IN_REPLAY"
        got = SB.resolve_player(state, nm_)
        if got is not None and got != lid:
            id_disagree += 1; examples["replay_id_differs_from_logged"].append([nm_, lid, got])
        cls[c] += 1
        if len(examples[c]) < 8:
            examples[c].append([tour, nm_, lid])
        if c in ("SURNAME_FALLBACK_GUESS", "AMBIGUOUS"):
            tml = sorted(ids.get((tour, lid), []))
            guess_detail.append({"tour": tour, "espn_name": nm_, "logged_id": lid, "tml_names_for_logged_id": tml, "class": c,
                                 "first_token_equal": bool(tml) and SB.normalize_name(tml[0]).split()[:1] == n_.split()[:1]})
    out["pick_log_replay"] = {"distinct_logged_players": len(players), "classes": dict(cls), "replay_id_differs_from_logged": id_disagree, "guess_and_ambiguous_detail": guess_detail, "examples": {k: v[:8] for k, v in examples.items()},
                              "note": "UNMATCHED_IN_REPLAY = player absent from 2015-2024 TML (2025-2026 debut or newer id) or name spelled differently; logged ids were produced by the live state including 2025-2026 data, so this replay is a lower bound on live resolution"}
    out["verdict"] = "identity matching is acceptable for historical TML ids; live ESPN->TML resolution contains a surname-fallback guess and a cross-tour index; V2 must forbid predictions for non-EXACT/NORMALIZED_EXACT identities"
    return out, state, picks


# ------------------------------------------------------------------- surface
def surface_audit(tours, state, picks):
    import tennis_serving_builder_a as SB
    out = {"tourney_surface_history": {}, "pick_log_replay": {}}
    for tour in TOURS:
        con = tours[tour]["con"]
        by = defaultdict(Counter)
        for t, s in con.execute("select tourney_name, surface from matches where surface is not null and surface!=''"):
            by[norm(t)][s] += 1
        mixed = {k: dict(v) for k, v in by.items() if len(v) > 1}
        n_null = con.execute("select count(*) from matches where surface is null or surface=''").fetchone()[0]
        out["tourney_surface_history"][tour] = {"tournament_names": len(by), "names_with_more_than_one_surface": len(mixed), "mixed_examples": dict(sorted(mixed.items())[:12]), "matches_without_surface": n_null}
    cls = Counter(); per_tourney = {}; conf_mismatch = Counter()
    for p in picks:
        key = (p["tour"], p["tourney"])
        if key in per_tourney:
            continue
        n_ = SB.normalize_name(p["tourney"])
        if not n_:
            c = "UNKNOWN"
        elif any(s in n_ for s in SB.GRAND_SLAM_SURFACE):
            c = "KNOWN_EXACT"
        elif n_ in state["tourney_surface"]:
            c = "HISTORICAL_TOURNEY_INFERENCE"
        else:
            c = "HARD_FALLBACK"
        per_tourney[key] = c
    for p in picks:
        c = per_tourney[(p["tour"], p["tourney"])]
        cls[c] += 1
        conf_mismatch[(c, bool(p["surface_confirmed"]))] += 1
    out["pick_log_replay"] = {"picks_by_class": dict(cls), "tournaments_by_class": dict(Counter(per_tourney.values())), "tournament_class": {"%s|%s" % k: v for k, v in sorted(per_tourney.items())},
                              "logged_surface_confirmed_by_class": {"%s|%s" % k: v for k, v in sorted(conf_mismatch.items())},
                              "share_logged_unconfirmed": float(np.mean([not p["surface_confirmed"] for p in picks])),
                              "share_logged_surface_hard": float(np.mean([p["surface"] == "Hard" for p in picks])),
                              "note": "surface_confirmed=True is emitted for both KNOWN_EXACT slams and HISTORICAL_TOURNEY_INFERENCE; there is no indoor/outdoor, altitude or court-speed field; names are matched by normalized ESPN name == normalized TML name"}
    return out


# ------------------------------------------------------- serve / return feasibility
def hold_prob(p):
    q = 1 - p
    return p ** 4 * (1 + 4 * q + 10 * q ** 2) + 20 * p ** 3 * q ** 3 * p ** 2 / (1 - 2 * p * q)


def serve_return_audit(tours):
    out = {"no_model_built": True, "tours": {}}
    for tour in TOURS:
        tr = tours[tour]; con = tr["con"]; t = {"coverage": {}, "identity": {}}
        rows = tr["rows"]; comp = [r for r in rows if not r["is_incomplete"]]
        pmr = {}
        for r in con.execute("select * from player_matches"):
            pmr.setdefault(r["match_id"], []).append(dict(r))
        for pn, per in list(PERIODS.items()) + [("DEV", ("2015-01-01", BL.DEV_END))]:
            cm = [r for r in comp if per[0] <= r["match_date"] <= per[1]]
            have = sum(1 for r in cm if len(pmr.get(r["match_id"], [])) == 2 and all(x["serve_points"] for x in pmr[r["match_id"]]))
            t["coverage"][pn] = {"completed": len(cm), "with_both_serve_rows": have, "coverage": have / len(cm)}
        bad = Counter(); tot = 0
        for ms in pmr.values():
            for x in ms:
                if not x["serve_points"]:
                    continue
                tot += 1
                f1, fw, sw, sp_, ac, df = x["first_serve_in"], x["first_serve_won"], x["second_serve_won"], x["serve_points"], x["aces"], x["double_faults"]
                if None in (f1, fw, sw, ac, df):
                    bad["missing_component"] += 1; continue
                if f1 > sp_: bad["first_in_gt_svpt"] += 1
                if fw > f1: bad["first_won_gt_first_in"] += 1
                if sw > sp_ - f1: bad["second_won_gt_second_pts"] += 1
                if ac + df > sp_: bad["ace_plus_df_gt_svpt"] += 1
                if x["break_points_saved"] is not None and x["break_points_faced"] is not None and x["break_points_saved"] > x["break_points_faced"]: bad["bpsaved_gt_bpfaced"] += 1
        t["identity"] = {"serve_rows": tot, "violations": dict(bad), "violation_rate": sum(bad.values()) / tot if tot else None}
        # out-of-time predictability of serve-point-win rate (first_won+second_won)/svpt
        seqs = []
        for r in INC.order_rows([r for r in comp if r["surface"]], "canonical"):
            ms = pmr.get(r["match_id"])
            if ms and len(ms) == 2 and all(x["serve_points"] for x in ms):
                seqs.append((r, ms))
        hist = defaultdict(list); rethist = defaultdict(list); dev_vals = []; recs = []
        for r, ms in seqs:
            spw = {x["player_id"]: (x["first_serve_won"] + x["second_serve_won"], x["serve_points"]) for x in ms}
            for x in ms:
                pid, opp = x["player_id"], x["opponent_id"]
                w, n_ = spw[pid]
                if r["match_date"] <= BL.DEV_END:
                    dev_vals.append(w / n_)
                elif len(hist[pid]) >= 10 and len(rethist[opp]) >= 10:
                    own = sum(a for a, b in hist[pid][-10:]) / sum(b for a, b in hist[pid][-10:])
                    ret = sum(a for a, b in rethist[opp][-10:]) / sum(b for a, b in rethist[opp][-10:])
                    holds = None
                    sg, bpf, bps = x["serve_games"], x["break_points_faced"], x["break_points_saved"]
                    if sg and bpf is not None and bps is not None and sg > 0:
                        holds = 1 - (bpf - bps) / sg
                    recs.append((r["match_date"], w / n_, own, ret, n_, holds))
            for x in ms:
                hist[x["player_id"]].append(spw[x["player_id"]]); rethist[x["player_id"]].append((spw[x["opponent_id"]][1] - spw[x["opponent_id"]][0], spw[x["opponent_id"]][1]))
        tour_mean = float(np.mean(dev_vals))
        hold = {}
        for pn, per in PERIODS.items():
            rr = [x for x in recs if per[0] <= x[0] <= per[1]]
            y = np.array([x[1] for x in rr]); w = np.array([x[4] for x in rr], float)
            own = np.array([x[2] for x in rr]); ret = np.array([x[3] for x in rr])
            tm = np.full(len(rr), tour_mean)
            tourret = tour_mean
            comb = np.clip(own + (1 - tour_mean) - ret , 0.3, 0.9)  # additive opponent-return adjustment around the tour mean (descriptive only)
            sse = lambda p: float(np.sum(w * (y - p) ** 2))
            base = sse(tm)
            hold[pn] = {"n": len(rr), "tour_mean_dev": tour_mean, "r2_improvement_own_history": 1 - sse(own) / base, "r2_improvement_own_plus_opponent_return": 1 - sse(comb) / base}
            hh = [x for x in rr if x[5] is not None]
            if hh:
                ph = np.array([hold_prob(min(max(x[2] + (1 - tour_mean) - x[3], 0.3), 0.9)) for x in hh]); ob = np.array([x[5] for x in hh])
                hold[pn]["hold_closed_form"] = {"n": len(hh), "mean_predicted_hold": float(ph.mean()), "mean_observed_hold": float(ob.mean()), "mae_hold_per_match": float(np.abs(ph - ob).mean())}
        t["out_of_time_serve_point_win"] = hold
        cov = t["coverage"]["HOLDOUT"]["coverage"]; vr = t["identity"]["violation_rate"]; imp = hold["HOLDOUT"]["r2_improvement_own_plus_opponent_return"]
        t["label"] = "DATA_SUFFICIENT" if (cov >= 0.90 and vr <= 0.01 and imp >= 0.02) else "PARTIAL" if cov >= 0.75 else "INSUFFICIENT"
        t["label_inputs"] = {"coverage": cov, "violation_rate": vr, "r2_improvement": imp}
        out["tours"][tour] = t
    return out


# ------------------------------------------------------------ current engine
def git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO)] + list(a), capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def reproduction(tours):
    out = {}
    for tour in TOURS:
        sfx = "" if tour == "atp" else "_wta"
        tr = tours[tour]; rep = lambda n: json.loads((REPO / ("tennis_%s_gate_report%s.json" % (n, sfx))).read_text())
        d = {}
        ml = rep("moneyline")
        for pn, per in PERIODS.items():
            o = [x for x in tr["ml_as"] if inp(x, per)]; y = np.array([x["p1_wins"] for x in o]); p = np.array([x["p1_prob"] for x in o])
            c = ml["val" if pn == "VAL" else "holdout"]
            d["moneyline_" + pn] = {"committed": {"n": c["n"], "accuracy": c["accuracy"], "brier": c["brier"], "logloss": c["logloss"]}, "replica": {"n": len(o), "accuracy": float(np.mean((p >= .5) == (y == 1))), "brier": M.brier(p, y), "logloss": M.log_loss(p, y)}}
        tg = rep("total_games")
        for pn, per in PERIODS.items():
            o = [x for x in tr["tg_as"] if inp(x, per)]
            k = "val" if pn == "VAL" else "holdout"
            d["total_games_" + pn] = {"committed": {"n": tg["n_%s_observations" % k], "model_mae": tg["model_%s_mae" % k], "naive_mae": tg["naive_%s_mae" % k]},
                                      "replica": {"n": len(o), "model_mae": float(np.mean([abs(x["predicted_mean"] - x["actual"]) for x in o])), "naive_mae": float(np.mean([abs(x["naive_mean"] - x["actual"]) for x in o]))}}
        sp = rep("games_spread")
        for pn, per in PERIODS.items():
            o = [x for x in tr["sp_as"] if inp(x, per)]; k = "val" if pn == "VAL" else "holdout"
            d["games_spread_" + pn] = {"committed": {"n": sp["n_%s_observations" % k], "model_mae": sp["model_%s_mae" % k], "naive_mae": sp["naive_%s_mae" % k]},
                                       "replica": {"n": len(o), "model_mae": float(np.mean([abs(x["predicted"] - x["actual"]) for x in o])), "naive_mae": float(np.mean([abs(x["naive"] - x["actual"]) for x in o]))}}
        sb = rep("set_betting")
        for pn, per in PERIODS.items():
            o = [x for x in tr["ss_as"] if inp(x, per)]; k = "val" if pn == "VAL" else "holdout"
            d["set_betting_" + pn] = {"committed": {"n": sb["n_%s_observations" % k], "model_accuracy": sb["%s_model_accuracy" % k]}, "replica": {"n": len(o), "model_accuracy": float(np.mean([x["predicted"] == x["actual"] for x in o]))}}
        for k_, v in d.items():
            c, r = v["committed"], v["replica"]
            v["exact_match"] = all(abs(c[f] - r[f]) < 1e-6 for f in c)
        out[tour] = d
    return out


def market_token_scan():
    proto = json.loads((OUT / "protocol.json").read_text())
    toks = next(v["banned_tokens"] for v in proto.values() if isinstance(v, dict) and "banned_tokens" in v)
    out = {}
    for f in sorted(REPO.glob("tennis_*.py")):
        if f.name.startswith("tennis_v2_"):
            continue
        hits = []
        for i, line in enumerate(f.read_text().splitlines(), 1):
            low = line.lower()
            if any(t in low for t in toks):
                hits.append(i)
        out[f.name] = {"line_count_with_banned_tokens": len(hits), "first_lines": hits[:8]}
    return out


def current_engine_audit(tours):
    return {
        "evidence_grade": "RETROSPECTIVE_BURNED",
        "base_commit": git("rev-parse", "origin/main") or "n/a",
        "reproduction_vs_committed_gate_reports": reproduction(tours),
        "market_token_scan_of_production_tennis_code": market_token_scan(),
        "firewall_classification": {
            "tennis_serving_builder_a.py": "EXECUTION: the price-feed fetch and line matching feed only the P(over) pick and the logged line; the projected total is computed before and without the line. Audited, not changed.",
            "gates": "SCIENCE-CLEAN: no gate reads a betting-market value (one docstring mentions the market in prose only).",
            "grader": "settlement only; reads ESPN results and the logged line"},
        "markets": {
            "moneyline": {"served": True, "graded_in_record": False, "formula": "Elo overall+surface (K=250/(n+5)^0.4, init 1500) blended by min surface count: w=min(m/(m+15),0.6), m>=4", "min_history": 5,
                          "timing": "state replays ALL history in (match_date, text match_id) order at serve time", "architecture": "PARTIAL"},
            "total_games": {"served": True, "formula": "mean over two players of recency-weighted games-per-set (decay 0.6, surface blend n/(n+10)<=0.7) x recency-weighted sets (best-of conditioned if >=4)", "min_history": 10, "architecture": "INSUFFICIENT",
                            "note": "projection is independent of the opponent matchup and of the other player's rating; live pick additionally normal-approximates P(over) around the projection"},
            "set_betting": {"served": True, "graded": "unagraded projection (no market); a hit = argmax exact score", "formula": "Elo match prob -> bisection -> iid per-set prob q -> race-to-N distribution -> argmax", "architecture": "PARTIAL"},
            "games_spread": {"served": False, "gate": "failed (ATP 1.17% improvement); WTA failed"},
            "aces": {"served": False, "gate": "failed"}, "double_faults": {"served": False, "gate": "failed"}},
        "pick_log": pick_log_audit(),
        "flaws": ["text match_id tie-break puts later rounds before earlier rounds (look-ahead in history); canonical order fixes it", "surface_confirmed conflates exact slam table and tournament-name history; Hard fallback otherwise",
                  "ONE sqlite / ONE name_index for both tours", "unique-surname fallback is a guess", "gate 'naive' for total games is the weakest possible comparator (see baselines)", "serving moneyline is not in graded record",
                  "WTA gate reports were produced from a mutable upstream and are not reproducible from pinned hashes"],
    }


def pick_log_audit():
    picks = [json.loads(l) for l in (REPO / "docs" / "tennis_picks_log.jsonl").read_text().splitlines() if l.strip()]
    by = Counter((p["tour"], p["market"]) for p in picks)
    keys = set().union(*[set(p) for p in picks])
    return {"rows": len(picks), "by_tour_market": {"%s|%s" % k: v for k, v in sorted(by.items())}, "fields_seen": sorted(keys), "first_logged_at": min(p["logged_at"] for p in picks), "last_logged_at": max(p["logged_at"] for p in picks),
            "unagraded_share": float(np.mean([bool(p.get("unagraded")) for p in picks])),
            "total_games_rows_with_model_projected_line": sum(1 for p in picks if p.get("model_source") == "model_projected_line"),
            "has_decision_time_snapshot_of_inputs": False, "has_hash_chain": False, "has_model_version": any("model_version" in p for p in picks),
            "science_execution_separation": "none: the same row carries model_prob, predicted_mean and the betting line and price fields"}


# ---------------------------------------------------------------- chronology
def chronology_audit():
    files = sorted(p.name for p in REPO.glob("tennis_*.py") if not p.name.startswith("tennis_v2_"))
    log = git("log", "--format=%H|%aI|%s", "--", *files)
    commits = [l.split("|", 2) for l in log.splitlines() if l]
    s2025 = {}
    for f in files:
        txt = (REPO / f).read_text()
        s2025[f] = {"mentions_2025": len(re.findall(r"2025", txt)), "mentions_2026": len(re.findall(r"2026", txt))}
    pickaxe = git("log", "-S2025", "--format=%h|%aI|%s", "--", *files).splitlines()
    proto = git("log", "--format=%H|%aI|%s", "--diff-filter=A", "--", "tennis_models/tennis_outcome_engine_v2/protocol.json").splitlines()
    results = {}
    for f in ("phase0_market_results.json", "phase0_findings.md"):
        results[f] = git("log", "--format=%H|%aI", "--diff-filter=A", "--", "tennis_models/tennis_outcome_engine_v2/" + f).splitlines()
    return {"evidence_grade": "RETROSPECTIVE_BURNED", "tennis_production_files": files, "commits_touching_tennis_production_files": [{"sha": c[0], "date": c[1], "subject": c[2]} for c in commits],
            "year_mentions_in_production_code": s2025,
            "pickaxe_commits_adding_or_removing_2025_literal": pickaxe,
            "sealed_2025_conclusion": "no production tennis script evaluates 2025: gates cap at 2024-12-31 and the only 2025 literals are a foundation docstring recording an HTTP 200 structural probe plus CLI usage examples (pickaxe commits above). 2025 rows may sit in the production database and replay into live Elo state, which is serving ingestion, not evaluation. => SEALED_UNTOUCHED for model selection/evaluation; Phase1 may use it once.",
            "exposed_2026": "forward picks since 2026-09-09; bug fixes decided on live results (8d8e6ba 2026-09-24, a3c8900 2026-09-27)",
            "protocol_added_in": proto, "results_added_in": results,
            "protocol_before_results": bool(proto) and all((not v) or (v[-1].split("|")[1] >= proto[-1].split("|")[1]) for v in results.values())}


# ------------------------------------------------------------------ receipts
def cause_flags(o, pred):
    pm = max(o["p_blend"], 1 - o["p_blend"]); fl = []
    if abs(o["n_sets"] - o["e_sets"]) >= 1: fl.append("SET_COUNT_SURPRISE")
    if o["tb_sets"] >= 2: fl.append("TIEBREAK_HEAVY")
    if o["best_of"] == 5: fl.append("BEST_OF_5")
    if pm >= 0.75 and o["actual"] < pred - 6: fl.append("HEAVY_FAVORITE_BLOWOUT")
    if pm <= 0.60 and o["actual"] > pred + 6: fl.append("CLOSE_MATCHUP_LONG")
    if o["order_leak"]: fl.append("ORDER_LEAKAGE_EXPOSED")
    return fl or ["UNATTRIBUTED"]


def r4(x):
    return None if x is None else round(float(x), 5)


def build_receipts(tours, market):
    receipts, cata = [], []
    for tour in TOURS:
        tr = tours[tour]
        asi = {o["match_id"]: o for o in tr["tg_as"]}
        simple_best = {pn: market["tours"][tour]["total_games"][pn]["verdict"]["strongest_simple"] for pn in PERIODS}
        for pn, per in PERIODS.items():
            for o in sorted([x for x in tr["tg_c"] if inp(x, per) and x["match_id"] in asi and x["pred"]["human"] is not None], key=lambda z: z["match_id"]):
                a = asi[o["match_id"]]; best = o["pred"][simple_best[pn]]
                row = {"market": "total_games", "tour": tour, "period": pn, "match_id": o["match_id"], "date": o["match_date"], "surface": o["surface"], "round": o["round"], "best_of": o["best_of"],
                       "p_blend_canonical": r4(o["p_blend"]), "incumbent_as_implemented": r4(a["predicted_mean"]), "incumbent_canonical": r4(o["predicted_mean"]), "strongest_simple_name": simple_best[pn], "strongest_simple": r4(best),
                       "human": r4(o["pred"]["human"]), "actual": o["actual"], "n_sets": o["n_sets"], "abs_err_incumbent": r4(abs(a["predicted_mean"] - o["actual"]))}
                receipts.append(row)
                if row["abs_err_incumbent"] > 10:
                    cata.append({**row, "tier": ">15" if row["abs_err_incumbent"] > 15 else ">10", "cause_flags": cause_flags({**o, "e_sets": o["e_sets"]}, a["predicted_mean"])})
            ml = {x["match_id"]: x for x in tr["ml_c"]}
            for a in sorted([x for x in tr["ml_as"] if inp(x, per)], key=lambda z: z["match_id"]):
                c = ml.get(a["match_id"])
                if c is None: continue
                row = {"market": "moneyline", "tour": tour, "period": pn, "match_id": a["match_id"], "date": a["match_date"], "p1_incumbent_as_implemented": r4(a["p1_prob"]), "p1_blend_canonical": r4(c["p_blend"]),
                       "p1_rank_logistic": r4(c["p_rank"]), "p1_human": r4(c["p_human"]), "p1_won": a["p1_wins"], "is_incomplete": a["is_incomplete"], "order_leak": a["order_leak"]}
                receipts.append(row)
                fav = max(a["p1_prob"], 1 - a["p1_prob"])
                if fav >= 0.85 and ((a["p1_prob"] >= .5) != (a["p1_wins"] == 1)):
                    cata.append({**row, "favorite_prob": r4(fav), "cause_flags": ["FAVORITE_LOST"] + (["RETIREMENT"] if a["is_incomplete"] else []) + (["ORDER_LEAKAGE_EXPOSED"] if a["order_leak"] else [])})
            for a in sorted([x for x in tr["ss_as"] if inp(x, per)], key=lambda z: z["match_id"]):
                top = max(a["dist"], key=a["dist"].get)
                row = {"market": "set_score", "tour": tour, "period": pn, "match_id": a["match_id"], "date": a["match_date"], "best_of": a["best_of"], "top_class": list(top), "top_prob": r4(a["dist"][top]),
                       "p_actual_class": r4(a["dist"].get(a["actual"], 0)), "actual": list(a["actual"])}
                receipts.append(row)
                if a["dist"][top] >= 0.5 and top != a["actual"]:
                    cata.append({**row, "cause_flags": ["TOP_CLASS_WRONG"]})
            for a in sorted([x for x in tr["sp_as"] if inp(x, per)], key=lambda z: z["match_id"]):
                receipts.append({"market": "games_spread", "tour": tour, "period": pn, "match_id": a["match_id"], "date": a["match_date"], "predicted": r4(a["predicted"]), "actual": a["actual"]})
            for key, mk, thr in (("aces", "aces", 8), ("dfs", "double_faults", 5)):
                for a in [x for x in tr[key] if inp(x, per)]:
                    if abs(a["pred"] - a["actual"]) > thr:
                        cata.append({"market": mk, "tour": tour, "period": pn, "match_id": a["match_id"], "player_id": a["player_id"], "date": a["match_date"], "pred": r4(a["pred"]), "actual": a["actual"],
                                     "serve_points_actual": a["actual_sv"], "serve_points_pred": r4(a["exp_sv"]), "cause_flags": ["VOLUME_MISS" if abs(a["actual_sv"] - a["exp_sv"]) > 0.25 * a["exp_sv"] else "RATE_MISS"]})
    return receipts, cata


def write_gz(name, rows):
    with open(OUT / name, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for r in rows:
                gz.write((json.dumps(r, sort_keys=True) + "\n").encode())


# ------------------------------------------------------------ ledger design etc
def ledger_design():
    return {"purpose": "V2 forward ledger (design only; not built, not replacing production)", "decision_key": ["tour", "tournament_id", "scheduled_start_utc", "player_a_id", "player_b_id", "horizon_label", "market"],
            "rows": {"science_receipt": ["engine_version/lock_hash", "data_manifest_hash + retrieval times", "player identity class (EXACT/NORMALIZED_EXACT only; else NO_FORECAST)", "surface class (KNOWN_EXACT/HISTORICAL_TOURNEY_INFERENCE/HARD_FALLBACK)",
                                      "canonical-order point-in-time state hash", "format (best_of), round, scheduled start", "full distribution (match win, set score classes, total-games distribution)", "created_at_utc strictly before scheduled start"],
                     "execution_record": "separate downstream record: book, line, price, retrieved_at; may reference science row id; never read by science code"},
            "integrity": ["append-only JSONL", "hash chain (prev_hash, row_hash)", "first-write-wins on decision key", "no overwrite of forecast; superseding forecast is a new row with higher horizon", "settlement rows reference decision key and carry RAW and CLEAN outcome fields"],
            "outcomes": {"RAW": "winner incl. retirement; W/O void", "CLEAN": "completed matches only", "settlement_of_retirements": "downstream book rule, never inferred in science"},
            "gaps_in_incumbent_log": "no engine version, no input snapshot, no hash chain, mutable file, science and execution fields mixed, tennis_record.json grades only total_games and set_betting"}


def registry(market):
    return {"protocol_id": "tennis_outcome_engine_v2_phase0", "evidence_grade": "RETROSPECTIVE_BURNED",
            "evaluated_model_families": {"incumbent_replicas": ["moneyline Elo blend", "total_games gps x sets", "games_spread margin x sets", "set_score iid race", "aces rate x volume", "double_faults rate x volume"],
                                         "simple_baselines": ["prior3", "prior5", "prior10", "ewma_0.6", "p1_prior10", "p2_prior10", "surface_fit", "bestof_fit", "elo_mismatch_fit", "rank_logistic", "elo_overall", "elo_surface_shrunk",
                                                              "category_frequency", "winner_prob_x_conditional_split", "zero_margin", "elo_bucket_fit_mean", "prior10_count_mean", "surface_bestof_fit_mean"],
                                         "human": ["total_games", "moneyline", "set_score"]},
            "tuned_parameters": "none (all baseline constants are DEV 2015-2019 means or a one-parameter DEV MLE; no grid, no selection on VAL/HOLDOUT)",
            "comparisons_per_tour_period": {"moneyline": 4, "total_games": 9, "games_spread": 2, "set_score": 2, "aces": 3, "double_faults": 3},
            "multiplicity_note": "6 markets x 2 tours x 2 periods; verdict is the worse of VAL and HOLDOUT against the strongest simple baseline; no multiplicity correction is applied because the verdict is conservative (best-of-many baselines)",
            "bootstrap": {"resamples": 2000, "seed": 20261008}}


def source_inventory():
    return {"used": {"TML-Database ATP (commit-pinned raw)": "tour-level main draw 2015-2024 results, ranks, rank points, ages, serve stats, scores",
                     "TML-Database WTA (website, mutable)": "tour + ITF-level results 2015-2024, same schema"},
            "consumed_by_production_not_science": {"ESPN scoreboard/schedule": "fixture list, names, competition ids (identity resolution risk)", "third-party price feed": "totals line + price for execution only"},
            "not_available": ["Challenger / qualifying rows", "point-by-point (Match Charting) data", "indoor/outdoor and court-speed fields", "injury/withdrawal timing", "scheduled start times for history", "player handedness/height in the used schema"],
            "licence": "TML README forbids redistribution: raw CSVs fetched at run time and verified by sha256, never committed",
            "sealed_exposed": {"2025": "sealed (never read)", "2026": "exposed (never read)"}}


def snapshot(market, extra):
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name.startswith("phase0_") and p.name != "phase0_snapshot.json" and p.name != "phase0_findings.md")
    h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    s = {"protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(), "source_manifest_sha256": hashlib.sha256((OUT / "source_manifest.json").read_bytes()).hexdigest(), "artifact_sha256": h,
         "verdict_table": {t: {m: {"native": market["tours"][t][m]["native_performance_verdict"], "architecture": market["tours"][t][m]["architecture_status"]} for m in market["tours"][t]} for t in TOURS}}
    s.update(extra)
    return s


def main(out_dir=None):
    tours = {t: compute_tour(t) for t in TOURS}
    market, keep = run_markets(tours)
    dump("phase0_market_results.json", market)
    dump("phase0_current_engine_audit.json", {**current_engine_audit(tours), "v2_forward_ledger_design": ledger_design()})
    dump("phase0_source_audit.json", source_audit(tours))
    dump("phase0_population_definition.json", {**population_definition(tours), "order_leakage": order_audit(tours)})
    idn, state, picks = identity_audit(tours)
    dump("phase0_identity_audit.json", idn)
    dump("phase0_surface_audit.json", surface_audit(tours, state, picks))
    dump("phase0_chronology_audit.json", chronology_audit())
    base = {}
    for tour in TOURS:
        base[tour] = {"total_games_dev_constants": tours[tour]["tg_consts"], "moneyline_rank_logistic_slope": tours[tour]["ml_slope"]}
    simple = {"constants": base, "total_games": {t: {pn: market["tours"][t]["total_games"][pn]["metrics"] for pn in PERIODS} for t in TOURS},
              "moneyline": {t: {pn: market["tours"][t]["moneyline"][pn]["RAW"]["metrics"] for pn in PERIODS} for t in TOURS},
              "games_spread": {t: {pn: market["tours"][t]["games_spread"][pn]["metrics"] for pn in PERIODS} for t in TOURS},
              "set_score": {t: {pn: market["tours"][t]["set_score"][pn]["metrics"] for pn in PERIODS} for t in TOURS},
              "aces": {t: {pn: market["tours"][t]["aces"][pn]["metrics"] for pn in PERIODS} for t in TOURS},
              "double_faults": {t: {pn: market["tours"][t]["double_faults"][pn]["metrics"] for pn in PERIODS} for t in TOURS}}
    dump("phase0_simple_baselines.json", simple)
    dump("phase0_competent_human_baseline.json", {"definitions": json.loads((OUT / "protocol.json").read_text())["baselines_frozen"], "constants": base,
                                                  "vs_incumbent": {t: {pn: {"total_games_paired_mae": market["tours"][t]["total_games"][pn]["incumbent_vs_human_paired_mae"], "moneyline_paired_brier": market["tours"][t]["moneyline"][pn]["RAW"]["incumbent_vs_human_paired_brier"],
                                                                            "set_score_paired_logloss": market["tours"][t]["set_score"][pn]["incumbent_vs_human_paired_logloss"]} for pn in PERIODS} for t in TOURS}})
    dump("phase0_serve_return_feasibility.json", serve_return_audit(tours))
    dump("research_registry.json", registry(market)); dump("source_inventory.json", source_inventory())
    receipts, cata = build_receipts(tours, market)
    write_gz("phase0_error_receipts.jsonl.gz", receipts); write_gz("phase0_catastrophic_misses.jsonl.gz", cata)
    dump("phase0_snapshot.json", snapshot(market, {"receipt_rows": len(receipts), "catastrophic_rows": len(cata), "catastrophic_by_market": dict(Counter(c["market"] for c in cata))}))
    print("phase0 complete", len(receipts), len(cata))


if __name__ == "__main__":
    main()

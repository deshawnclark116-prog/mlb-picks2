"""
NFL_PHASE1D_GATES  (Phase 1D)  -- protocol Amendment G: the pre-registered gates of the clean-forward shadow experiment, and their evaluators

`AMENDMENT_G` is the single source of truth. `write_amendment` puts it into nfl_models/nfl_player_outcome_phase1_protocol.json (version 1.2) and a test asserts that
the file, this module's constants and the evaluators agree. Nothing in this file may be edited after the first clean-forward outcome is inspected; thresholds are
never derived from forward results.

The evaluators (`evaluate_window`) take the joined forecast / baseline / score records of a window and return per-gate verdicts. They are exercised on burned
time-travel data in the Phase 1D dry run (machinery check only: burned weeks give no verdict).
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import nfl_phase1_common as C

REPO = Path(__file__).resolve().parent
PROTOCOL = REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json"
CORE = ("rush_yds", "rec_yds", "rec", "pass_yds")
ELIGIBILITY = {   # protocol v1.1 eligible_universe.prior_usage_eligibility, evaluated on the as-of prior_usage logged in every forecast record
    "rush_yds": {"pos": ["RB", "FB"], "field": "carries_l3", "min": 5.0, "min_career_rows": 3},
    "rec_yds": {"pos": ["WR", "TE", "RB", "FB"], "field": "targets_l3", "min": 3.0, "min_career_rows": 3},
    "rec": {"pos": ["WR", "TE", "RB", "FB"], "field": "targets_l3", "min": 3.0, "min_career_rows": 3},
    "pass_yds": {"pos": ["QB"], "field": "attempts_l3", "min": 15.0, "min_career_rows": 3},
    "atd": {"pos": ["RB", "FB", "WR", "TE"], "field": "carries_l3+targets_l3", "min": 4.0, "min_career_rows": 3},
}

AMENDMENT_G = {
    "id": "G",
    "protocol_version_after": "1.2",
    "registered_before_any_clean_forward_outcome": True,
    "purpose": "pre-register the numeric gates, universes, baselines and verdict timing of the clean-forward SHADOW test of the frozen Phase 1C architecture. The test decides whether the architecture "
               "is better; freezing does not assert that it is. No predictive change is permitted between the freeze and the final verdict.",
    "primary_horizon": "T24",
    "secondary_horizon": "T90 is reported separately and can never rescue a failed T24 verdict",
    "primary_universe": {
        "name": "protocol-eligible pregame universe",
        "definition": "players meeting the protocol prior-usage eligibility (ELIGIBILITY table below), evaluated on the as-of prior_usage stored in each forecast record; every eligible "
                      "player of the game is scored, players who do not participate have actual outcome = 0",
        "eligibility": ELIGIBILITY,
        "not_substituted_by": "conditional played-player or positive-opportunity results (diagnostics only); the expanded pregame candidate universe is a labelled secondary universe"},
    "outcomes": {
        "core": list(CORE) + ["(pass_yds is the QB passing-yards outcome)"],
        "primary_event": "atd (anytime offensive touchdown: rushing + receiving TD >= 1)",
        "secondary": ["rush_td", "rec_td", "pass_td", "int", "rush_yds of QBs (QB rushing yards)", "tackles", "sacks", "def_int"]},
    "baselines": {
        "historical_HB1": "nfl_phase1d_baselines.py (BASELINE_VERSION HB1): p_status x last-8 empirical distribution of qualifying games, position pool of the most recent 400 qualifying "
                          "player-games when the player has < 3, point mass at 0 with probability 1 - p_status; produced and logged before kickoff next to every forecast",
        "availability_status_only": "the Phase 1A status lookup table P(played | final report status) fitted on the training window of the weekly artifact (T-90m: game-day INA => 0)",
        "anytime_td": "HB1 anytime-TD probability: p_status x (TDs in last 8 qualifying + 2 x frozen position rate) / (n + 2); rates QB .1426, RB .2573, WR .2117, TE .1522 (2023-2024 regular season)",
        "secondary_events": "climatology (the window's own base rate of the outcome, applied to every row): a deliberately hard-to-beat reference"},
    "gates": {
        "core_distribution": {
            "per_outcome_required": list(CORE), "metric": "CRPS (19-quantile grid proxy = 2 x mean pinball)", "universe": "primary", "horizon": "T24",
            "test": "paired week-block bootstrap (2000 resamples), Phase 1C CRPS lower than HB1 CRPS over the complete clean-forward window, one-sided p < 0.05 for EACH outcome",
            "p_threshold": 0.05, "n_boot": 2000,
            "aggregate": "aggregate normalized CRPS (mean over the four outcomes of model CRPS / HB1 CRPS) is descriptive and cannot rescue a failing core outcome"},
        "core_point": {
            "metric": "MAE of the median forecast", "universe": "primary", "horizon": "T24", "material_worsening_pct": 2.0, "min_outcomes_numerically_better": 3, "of": 4,
            "rule": "no core outcome may have MAE > 1.02 x HB1 MAE, and at least 3 of 4 core outcomes must have numerically lower MAE than HB1"},
        "v2_incumbent_challenge": {
            "outcomes": ["rush_yds", "rec_yds"], "population": "common v2-eligible PLAYED-player diagnostic population (conditional on participation)",
            "comparator": "production-valid v2 point forecasts logged before kickoff by the production pipeline (external input; games without one are excluded, never imputed) and the "
                          "honest v2 residual-distribution comparator (residuals from the training window)",
            "required_for": "production replacement of v2 only; failure does not invalidate the research engine",
            "pooled_standardized_abs_error": "pooled rush+receiving absolute error, each outcome standardized by its v2 MAE; improvement with week-block p < 0.05",
            "mae_not_worse_than": 1.02, "pooled_crps_lower_than_v2_residual_distribution_p": 0.05},
        "calibration": {
            "method": "randomized PIT from the stored exact lattice cdf (discrete outcomes) or p_zero + 99-point quantile grid (yardage); seed derived from the forecast id; never naive interval coverage",
            "outcomes": list(CORE), "universe": "primary", "horizon": "T24",
            "central_80_pit_coverage": [0.75, 0.85], "central_50_pit_coverage": [0.45, 0.55], "central_90_pit_coverage": "reported",
            "no_retuning": "no calibration map may be refit from forward outcomes"},
        "availability": {
            "horizon": "T24", "universe": "all Phase 1A pregame candidates (one row per game and player)", "baseline": "availability_status_only",
            "must_beat_baseline_on": ["logloss", "Brier"], "p_threshold": 0.05, "ece_max": 0.03, "ece_bins": 10, "T90": "reported separately"},
        "anytime_td": {
            "horizon": "T24", "universe": "primary (atd eligibility)", "baseline": "anytime_td", "must_beat_baseline_on": ["logloss", "Brier"], "p_threshold": 0.05, "ece_max": 0.03,
            "auc": "diagnostic only"},
        "secondary_events": {
            "outcomes": ["rush_td", "rec_td", "pass_td", "int", "def_int", "sacks"], "min_positive_events": 50,
            "classes": ["PASS", "NO MATERIAL DIFFERENCE", "REGRESSION", "INSUFFICIENT EVENTS"],
            "rule": "INSUFFICIENT EVENTS when fewer than 50 positive events; else logloss versus climatology by paired week-block bootstrap: PASS if lower with p < 0.05, REGRESSION if higher "
                    "with p < 0.05 (reverse test), otherwise NO MATERIAL DIFFERENCE. Rare outcomes are never required to pass a significance gate on their own"},
        "uncertainty": {
            "outcomes": list(CORE), "requirement": "Spearman(predicted uncertainty score, realized normalized absolute error) > 0 over the full window (primary universe); the p-value "
                                                    "(week-block bootstrap) is reported; secondary gate. High-uncertainty rows are never excluded from primary scoring. Also reported: "
                                                    "error by uncertainty decile and a calibration curve",
            "normalized_error": "|median - y| / (|mean| + scale0)"},
        "reliability_completeness": {
            "rule": "every scheduled eligible game-horizon must have exactly one status row: FORECAST_SUCCESS or SAFE_EXPLICIT_FAILURE with a reason; no silent missing games; forecast "
                    "records are append-only and are never overwritten (same id + different bytes is a HARD ERROR)",
            "gate": "0 unexplained missing game-horizons and 0 hard errors"}},
    "verdict_timing": {
        "after_4_weeks": "descriptive interim only",
        "after_8_weeks": "inferential interim may be calculated; it cannot trigger production promotion",
        "final": "complete frozen window through 2026 week 18; no architecture change from any interim result"},
    "not_allowed": ["threshold changes", "baseline changes", "recalibration from forward outcomes", "feature/hyper-parameter/architecture changes", "dropping high-uncertainty rows from primary scoring",
                    "substituting the conditional (played-player) population for the pregame universe"],
    "refit": "weekly walk-forward refit (Amendment D) implemented as nfl_phase1d_p1a.fit_artifacts + nfl_phase1c_fit.fit_all with FROZEN hyper-parameters; every refit is a new artifact bundle "
             "hash logged in weekly_fits.jsonl",
}


def write_amendment(path=PROTOCOL):
    p = json.loads(Path(path).read_text())
    cur = p.get("amendments_v1_2", [])
    if any(a["id"] == "G" for a in cur):
        old = next(a for a in cur if a["id"] == "G")
        if {k: v for k, v in old.items() if k != "registered_utc"} != AMENDMENT_G:
            raise RuntimeError("Amendment G is already registered with different content; it cannot be edited")
        return p
    p["previous_version"] = p["version"]
    p["version"] = "1.2"
    p["amended_at_utc"] = datetime.now(timezone.utc).isoformat()
    p["amendments_v1_2"] = cur + [{**AMENDMENT_G, "registered_utc": p["amended_at_utc"]}]
    Path(path).write_text(json.dumps(p, indent=1))
    return p


# ------------------------------------------------------------------ evaluators
def eligible(rec, outcome):
    e = ELIGIBILITY[outcome]
    u = rec.get("prior_usage")
    if not u or u.get("pos") not in e["pos"] or u.get("career_rows", 0) < e["min_career_rows"]:
        return False
    v = sum(u.get(f, 0.0) for f in e["field"].split("+"))
    return v >= e["min"]


def rand_pit(rec, seed_tag=""):
    import zlib
    lo, hi = rec.get("pit_lo"), rec.get("pit_hi")
    if lo is None:
        return None
    u = np.random.default_rng(zlib.crc32((rec["forecast_id"] + seed_tag).encode())).random()
    return lo + u * (hi - lo)


def block_p(err_new, err_ref, blocks, n=2000):
    imp, p = C.block_boot(np.asarray(err_new, float), np.asarray(err_ref, float), np.asarray(blocks), n=n)
    return imp, p


def ece(p, y, bins=10):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()))


def logloss_vec(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def evaluate_window(model, base, horizon="T24"):
    """model / base: lists of joined dicts {forecast_id, outcome, horizon, block ('season-week'), player_id, game_id, y, crps_proxy, abs_error_median, mean, median, p_active,
    prior_usage, uncertainty_score, pit_lo, pit_hi, event_p, played} (baseline rows carry the same keys). Returns verdicts per gate (T24 primary)."""
    G = AMENDMENT_G["gates"]
    bi = {(r["game_id"], r["player_id"], r["outcome"], r["horizon"]): r for r in base}
    out = {"horizon": horizon, "gates": {}}
    rows = [r for r in model if r["horizon"] == horizon]
    core_res, point = {}, {}
    for o in CORE:
        rs = [r for r in rows if r["outcome"] == o and eligible(r, o) and (r["game_id"], r["player_id"], o, horizon) in bi]
        if len(rs) < 30:
            core_res[o] = {"n": len(rs), "verdict": "INSUFFICIENT DATA"}
            continue
        b = [bi[(r["game_id"], r["player_id"], o, horizon)] for r in rs]
        blocks = [r["block"] for r in rs]
        cm = np.array([r["crps_proxy"] for r in rs]); cb = np.array([x["crps_proxy"] for x in b])
        am = np.array([r["abs_error_median"] for r in rs]); ab = np.array([x["abs_error_median"] for x in b])
        imp, p = block_p(cm, cb, blocks)
        core_res[o] = {"n": len(rs), "crps_model": float(cm.mean()), "crps_baseline": float(cb.mean()), "improvement": imp, "p_not_better": p,
                       "verdict": "PASS" if imp > 0 and p < G["core_distribution"]["p_threshold"] else "FAIL"}
        point[o] = {"mae_model": float(am.mean()), "mae_baseline": float(ab.mean()), "ratio": float(am.mean() / max(ab.mean(), 1e-12)), "numerically_better": bool(am.mean() < ab.mean()),
                    "materially_worse": bool(am.mean() > (1 + G["core_point"]["material_worsening_pct"] / 100.0) * ab.mean())}
    norm = [core_res[o]["crps_model"] / max(core_res[o]["crps_baseline"], 1e-12) for o in CORE if "crps_model" in core_res[o]]
    out["gates"]["core_distribution"] = {"per_outcome": core_res, "aggregate_normalized_crps_descriptive": float(np.mean(norm)) if norm else None,
                                         "verdict": "PASS" if all(v.get("verdict") == "PASS" for v in core_res.values()) and len(core_res) == len(CORE) else "FAIL"}
    nb = sum(1 for v in point.values() if v["numerically_better"])
    out["gates"]["core_point"] = {"per_outcome": point, "n_numerically_better": nb,
                                  "verdict": "PASS" if len(point) == len(CORE) and nb >= G["core_point"]["min_outcomes_numerically_better"] and not any(v["materially_worse"] for v in point.values()) else "FAIL"}
    cal = {}
    for o in CORE:
        u = [rand_pit(r) for r in rows if r["outcome"] == o and eligible(r, o)]
        u = np.array([x for x in u if x is not None])
        if len(u) < 30:
            cal[o] = {"n": int(len(u)), "verdict": "INSUFFICIENT DATA"}
            continue
        cv = lambda lv: float(np.mean((u >= (1 - lv) / 2) & (u <= 1 - (1 - lv) / 2)))
        c80, c50, c90 = cv(0.8), cv(0.5), cv(0.9)
        lo80, hi80 = G["calibration"]["central_80_pit_coverage"]; lo50, hi50 = G["calibration"]["central_50_pit_coverage"]
        cal[o] = {"n": int(len(u)), "pit_cov80": c80, "pit_cov50": c50, "pit_cov90_reported": c90, "verdict": "PASS" if lo80 <= c80 <= hi80 and lo50 <= c50 <= hi50 else "FAIL"}
    out["gates"]["calibration"] = {"per_outcome": cal, "verdict": "PASS" if all(v.get("verdict") == "PASS" for v in cal.values()) and len(cal) == len(CORE) else "FAIL"}
    # availability: one row per (game, player) at this horizon; baseline = status-only lookup
    seen, av = set(), []
    for r in rows:
        k = (r["game_id"], r["player_id"])
        if k in seen or r.get("played") is None or r.get("p_active") is None or r.get("p_status") is None:
            continue
        seen.add(k); av.append(r)
    if len(av) >= 50:
        y = np.array([1.0 if r["played"] else 0.0 for r in av]); pm = np.array([r["p_active"] for r in av]); pb = np.array([r["p_status"] for r in av]); blocks = [r["block"] for r in av]
        ll_i, ll_p = block_p(logloss_vec(pm, y), logloss_vec(pb, y), blocks)
        br_i, br_p = block_p((pm - y) ** 2, (pb - y) ** 2, blocks)
        e = ece(pm, y, G["availability"]["ece_bins"])
        out["gates"]["availability"] = {"n": len(av), "logloss_improvement": ll_i, "logloss_p": ll_p, "brier_improvement": br_i, "brier_p": br_p, "ece": e,
                                        "verdict": "PASS" if ll_i > 0 and br_i > 0 and ll_p < 0.05 and br_p < 0.05 and e <= G["availability"]["ece_max"] else "FAIL"}
    else:
        out["gates"]["availability"] = {"n": len(av), "verdict": "INSUFFICIENT DATA"}
    ev = [r for r in rows if r["outcome"] == "atd" and eligible(r, "atd") and r.get("event_p") is not None and (r["game_id"], r["player_id"], "atd", horizon) in bi]
    if len(ev) >= 50:
        y = np.array([1.0 if r["y"] >= 1 else 0.0 for r in ev]); pm = np.array([r["event_p"] for r in ev]); pb = np.array([bi[(r["game_id"], r["player_id"], "atd", horizon)]["event_p"] for r in ev])
        blocks = [r["block"] for r in ev]
        ll_i, ll_p = block_p(logloss_vec(pm, y), logloss_vec(pb, y), blocks)
        br_i, br_p = block_p((pm - y) ** 2, (pb - y) ** 2, blocks)
        e = ece(pm, y)
        out["gates"]["anytime_td"] = {"n": len(ev), "positives": int(y.sum()), "logloss_improvement": ll_i, "logloss_p": ll_p, "brier_improvement": br_i, "brier_p": br_p, "ece": e,
                                      "verdict": "PASS" if ll_i > 0 and br_i > 0 and ll_p < 0.05 and br_p < 0.05 and e <= G["anytime_td"]["ece_max"] else "FAIL"}
    else:
        out["gates"]["anytime_td"] = {"n": len(ev), "verdict": "INSUFFICIENT DATA"}
    sec = {}
    for o in G["secondary_events"]["outcomes"]:
        rs = [r for r in rows if r["outcome"] == o and r.get("event_p") is not None]
        if not rs:
            continue
        y = np.array([1.0 if r["y"] >= 1 else 0.0 for r in rs]); npos = int(y.sum())
        if npos < G["secondary_events"]["min_positive_events"]:
            sec[o] = {"n": len(rs), "positives": npos, "class": "INSUFFICIENT EVENTS"}
            continue
        pm = np.array([r["event_p"] for r in rs]); clim = np.full(len(rs), y.mean()); blocks = [r["block"] for r in rs]
        imp, p = block_p(logloss_vec(pm, y), logloss_vec(clim, y), blocks)
        imp2, p2 = block_p(logloss_vec(clim, y), logloss_vec(pm, y), blocks)
        sec[o] = {"n": len(rs), "positives": npos, "logloss_improvement_vs_climatology": imp, "p_not_better": p, "p_not_worse": p2,
                  "class": "PASS" if imp > 0 and p < 0.05 else "REGRESSION" if imp < 0 and p2 < 0.05 else "NO MATERIAL DIFFERENCE"}
    out["gates"]["secondary_events"] = sec
    unc = {}
    for o in CORE:
        rs = [r for r in rows if r["outcome"] == o and eligible(r, o) and r.get("uncertainty_score") is not None]
        if len(rs) < 30:
            unc[o] = {"n": len(rs), "verdict": "INSUFFICIENT DATA"}
            continue
        U = np.array([r["uncertainty_score"] for r in rs]); scale0 = {"rush_yds": 5.0, "rec_yds": 5.0, "pass_yds": 20.0, "rec": 1.0}[o]
        err = np.array([r["abs_error_median"] / (abs(r["mean"]) + scale0) for r in rs])
        rho = spearman(U, err)
        q = np.quantile(U, np.linspace(0, 1, 11)); q[-1] += 1e-9
        dec = [{"decile": i + 1, "n": int(((U >= q[i]) & (U < q[i + 1])).sum()), "mean_norm_error": float(err[(U >= q[i]) & (U < q[i + 1])].mean())} for i in range(10) if ((U >= q[i]) & (U < q[i + 1])).any()]
        unc[o] = {"n": len(rs), "spearman": rho, "error_by_decile": dec, "verdict": "PASS" if rho > 0 else "FAIL"}
    out["gates"]["uncertainty"] = {"per_outcome": unc, "verdict": "PASS" if all(v.get("verdict") == "PASS" for v in unc.values()) and len(unc) == len(CORE) else "FAIL"}
    return out

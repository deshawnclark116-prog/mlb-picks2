"""
NHL_FORECAST_CONTRACT -- validator for the universal NHL outcome-head forecast object (nhl_universal_forecast_contract.json).
Every head returns a full outcome DISTRIBUTION plus provenance; no sportsbook line exists anywhere in the object, so the engine produces outcomes even when no betting market exists.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "nhl_models" / "nhl_outcome_engine"
REQUIRED = ["entity", "game_id", "team", "opponent", "prediction_timestamp", "information_cutoff", "availability_participation_state", "role_deployment_state", "outcome", "distribution", "mean", "median", "quantiles",
            "model_version", "data_snapshot_hash", "coverage_uncertainty_quality"]
ENTITY_KINDS = {"skater", "goalie", "team", "game"}
OUTCOMES = {"skater_shots_on_goal", "skater_goals", "skater_assists", "skater_points", "goalie_saves", "team_goals", "game_win_probability"}
FORBIDDEN_SUBSTRINGS = ("odds", "sportsbook", "bookmaker", "vegas", "price", "juice", "vig", "spread", "implied", "market_line", "book_line", "prop_line")
QUANTILES = ("0.05", "0.25", "0.5", "0.75", "0.95")
AVAILABILITY_STATES = {"CONFIRMED_PARTICIPANT_FORWARD_SNAPSHOT", "EXPECTED_FROM_PRIOR_APPEARANCES", "UNOBSERVABLE_AT_CUTOFF", "NOT_APPLICABLE"}


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path + "/" + str(k), k
            yield from _walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")


def cdf_from_pmf(pmf):
    out, c = [], 0.0
    for p in pmf:
        c += p; out.append(c)
    return out


def quantile(pmf, q):
    c = 0.0
    for k, p in enumerate(pmf):
        c += p
        if c >= q - 1e-12:
            return k
    return len(pmf) - 1


def validate(obj):
    """-> list of problems (empty = valid)."""
    bad = []
    for f in REQUIRED:
        if f not in obj:
            bad.append(f"missing field {f}")
    if bad:
        return bad
    for path, key in _walk(obj):
        if any(w in str(key).lower() for w in FORBIDDEN_SUBSTRINGS):
            bad.append(f"sportsbook-like field {path}")
    if obj["entity"].get("kind") not in ENTITY_KINDS:
        bad.append("entity.kind")
    if obj["outcome"] not in OUTCOMES:
        bad.append("outcome")
    if obj["availability_participation_state"].get("state") not in AVAILABILITY_STATES:
        bad.append("availability_participation_state.state")
    if obj["information_cutoff"] > obj["prediction_timestamp"]:
        bad.append("information_cutoff is after prediction_timestamp")
    d = obj["distribution"]
    pmf = d.get("pmf")
    if not pmf or abs(sum(pmf) - 1.0) > 1e-6 or any(p < -1e-12 for p in pmf):
        bad.append("distribution.pmf must be a probability vector")
    else:
        mean = sum(k * p for k, p in enumerate(pmf))
        if abs(mean - obj["mean"]) > 1e-6:
            bad.append("mean inconsistent with the pmf")
        if obj["median"] != quantile(pmf, 0.5):
            bad.append("median inconsistent with the pmf")
        for q in QUANTILES:
            if obj["quantiles"].get(q) != quantile(pmf, float(q)):
                bad.append(f"quantile {q} inconsistent with the pmf")
        for k, p in (obj.get("threshold_probabilities") or {}).items():
            thr = int(k.split(">=")[1].rstrip(")"))
            if abs(p - sum(pmf[thr:])) > 1e-6:
                bad.append(f"threshold {k} inconsistent with the pmf")
    cq = obj["coverage_uncertainty_quality"]
    if not {"state_evidence_coverage", "history_depth", "calibration_status"} <= set(cq):
        bad.append("coverage_uncertainty_quality fields")
    if not obj["data_snapshot_hash"] or len(obj["data_snapshot_hash"]) < 16:
        bad.append("data_snapshot_hash")
    return bad


def example_skater_sog():
    import math as m
    mu = 2.4
    pmf = [m.exp(-mu) * mu ** k / m.factorial(k) for k in range(30)]
    pmf[-1] += 1.0 - sum(pmf)
    return {"entity": {"kind": "skater", "player_id": 8478402, "name": "example"}, "game_id": 2026020100, "team": "EDM", "opponent": "CGY", "prediction_timestamp": "2026-10-10T21:30:00Z", "information_cutoff": "2026-10-10T21:30:00Z",
            "availability_participation_state": {"state": "EXPECTED_FROM_PRIOR_APPEARANCES", "evidence": "prior team appearances; no valid forward snapshot"}, "role_deployment_state": {"toi_mean_ct_app10": 1180.0, "pp_alloc_share_ct_app10": 0.31},
            "outcome": "skater_shots_on_goal", "distribution": {"support": "0..29", "pmf": pmf}, "mean": sum(k * p for k, p in enumerate(pmf)), "median": quantile(pmf, 0.5), "quantiles": {q: quantile(pmf, float(q)) for q in QUANTILES},
            "threshold_probabilities": {f"P(>={t})": sum(pmf[t:]) for t in (1, 2, 3, 4, 5)}, "model_version": "nhl-sog-phase1a-B2", "data_snapshot_hash": "0" * 64,
            "coverage_uncertainty_quality": {"state_evidence_coverage": "FULL", "history_depth": 10, "calibration_status": "HISTORICAL_CHAMPION_2025_BURNED"}}

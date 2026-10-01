"""Machinery check of the Amendment G evaluators on the six burned time-travel dry-run weeks (NOT a verdict: burned weeks, one model per week)."""
import json, sys
from pathlib import Path
import nfl_phase1_forecast as FC, nfl_phase1_store as ST, nfl_phase1d_gates as GT
R = Path(sys.argv[1]); out = Path(sys.argv[2])
fore = {x["id"]: x for x in FC.read_forecasts(ST.Store(R, "forecasts"))}
def join(sstore, fstore_recs):
    rows = []
    for sc in ST.Store(R, sstore).all_records():
        f = fstore_recs.get(sc["forecast_id"])
        if f is None: continue
        rows.append({"forecast_id": f["id"], "outcome": f["outcome"], "horizon": f["horizon"], "block": f"{f['season']}-{f['week']}", "player_id": f["player_id"], "game_id": f["game_id"],
                     "y": sc["outcome_value"], "crps_proxy": sc["crps_proxy"], "abs_error_median": sc["abs_error_median"], "mean": f["mean"], "median": f["median"], "p_active": f.get("p_active"),
                     "p_status": f.get("p_active_status_baseline"), "prior_usage": f.get("prior_usage"), "uncertainty_score": (f.get("uncertainty") or {}).get("score"),
                     "pit_lo": sc.get("pit_lo"), "pit_hi": sc.get("pit_hi"), "event_p": sc.get("event_p"), "played": sc.get("played")})
    return rows
model = join("scores", fore)
bfore = {x["id"]: x for x in FC.read_forecasts(ST.Store(R, "baselines"))}
base = join("baseline_scores", bfore)
res = GT.evaluate_window(model, base)
res["label"] = "MACHINERY CHECK on burned time-travel dry-run weeks; per-week models; not a verdict, not a holdout"
res["n_model_rows"], res["n_baseline_rows"] = len(model), len(base)
out.write_text(json.dumps(res, indent=1, default=float))
print({k: v.get("verdict", "n/a") if isinstance(v, dict) else v for k, v in res["gates"].items()})

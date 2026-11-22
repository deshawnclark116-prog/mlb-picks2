#!/usr/bin/env python3
"""One-time READ-ONLY trace of the exact 2026-W11 Rice receiving prediction.

Uses unchanged frozen trained V2 XGBoost model plus as-of-only 2023-2026
raw player_games data. Does not write production, retrain or publish.
"""
import json,math,sqlite3
from pathlib import Path
import numpy as np
import xgboost as xgb
import nfl_yardage_v2 as v2

DB="/tmp/rice_v2_source_20261122.sqlite"
C=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
C.row_factory=sqlite3.Row
frozen=json.loads(Path("docs/nfl_predictions.json").read_text())
rice=[r for r in frozen["picks"] if r.get("player")=="Rashee Rice" and r.get("market")=="receiving_yards" and r.get("team")=="KC"]
assert len(rice)==1,"FROZEN_RICE_RECEIVING_FORECAST_NOT_UNIQUE"
rice=rice[0]
assert rice["season"]==2026 and rice["week"]==11
assert rice["model_source"]=="v2_context"
history=v2.History(C,"receiving_yards")
prior_rows,player_h,opp_state=history.replay(until=(2026,11))
found=[(pid,h) for pid,h in player_h.items() if h and v2.norm_name(h[-1]["name"])=="rashee rice" and h[-1]["team"]=="KC"]
assert len(found)==1,"CANNOT_MAP_RICE_TO_SINGLE_HISTORICAL_PLAYER"
pid,ph=found[0]
assert v2.serving_opportunity_eligible(ph,"receiving_yards")
g=next((r for r in C.execute("""
SELECT game_id,home_team,away_team,spread_line,total_line,game_date,kickoff_utc
FROM games WHERE season=2026 AND week=11 AND
((home_team='KC' AND away_team='ARI') OR (home_team='ARI' AND away_team='KC'))""")),None)
assert g is not None, "KC_ARI_SCHEDULE_UNAVAILABLE"
is_home=g["home_team"]=="KC"
f=history.features(ph,2026,11,"KC","ARI",is_home,opp_state)
X=np.asarray([[float(f[key]) if f[key] is not None else np.nan for key in v2.FEATURES]],dtype=np.float32)
booster=xgb.Booster()
booster.load_model(str(v2.MODEL_DIR/"nfl_receiving_yards_v2.json"))
dmat=xgb.DMatrix(X,feature_names=v2.FEATURES)
pred=float(booster.predict(dmat)[0])
contr=booster.predict(dmat,pred_contribs=True)[0]
assert abs(float(np.sum(contr))-pred)<.002, "SHAP_RECOMPOSITION_FAILED"
home_spread=float(g["spread_line"] or 0)
total=float(g["total_line"] or 0)
output={
 "frozen_generated_at":frozen.get("generated_at_utc"),
 "frozen_source_forecast":{
  "player":rice["player"],"mean":rice["projected_mean"],"median":rice["projected_median"],
  "p10":rice["projected_low"],"p90":rice["projected_high"],
  "source":rice["model_source"],"percent_source":rice.get("projection_probability_source"),
  "confidence":rice.get("confidence"),"confidence_basis":rice.get("confidence_basis"),
  "fanduel_line":rice.get("fanduel_line"),"games_played":rice.get("games_played")
 },
 "live_frozen_model_replay":{
   "predicted_yards_from_raw_2023_2026_history":round(pred,5),
   "matches_original_within_point1":abs(pred-rice["projected_mean"])<=.11,
   "player_history_id":str(pid),"model_boosted_trees":booster.num_boosted_rounds(),
   "model_objective":booster.save_config()[:400],
   "model_intercept":round(float(contr[-1]),4),
   "feature_contributions":[{"feature":name,
                             "value":f[name],
                             "contribution_yards":round(float(contr[i]),4)}
                            for i,name in enumerate(v2.FEATURES)],
   "shap_sum":round(float(np.sum(contr)),5)
 },
 "model_features":f,
 "component_baseline_naive_prior_or_blend":{
    "last_season_average":v2.naive_preds(f)[0],
    "blended_average":v2.naive_preds(f)[1]
 },
 "recent_history_last10":[{ "season":g0["season"],"week":g0["week"],
      "targets":g0["vol"],"receiving_yards":g0["yds"],
      "team_targets":g0["team_vol"],"target_share":
       round(g0["vol"]/g0["team_vol"],4) if g0["team_vol"] else None,
      "team":g0["team"]
 } for g0 in ph[-10:]],
 "game_context":{
   "game_id":g["game_id"],"home":g["home_team"],"away":g["away_team"],
   "source_spread_line":g["spread_line"],"source_total_line":g["total_line"],
   "game_date":g["game_date"],"kickoff_utc":g["kickoff_utc"],
   "code_implied_team_total":f.get("implied_team_total"),
   "home_favored_is_negative_nflverse":True,
   "true_KC_implied_from_spread_if_sign_as_documented":
      total/2-home_spread/2 if is_home else total/2+home_spread/2,
   "possible_home_spread_sign_inversion":
      bool(is_home and abs(float(f.get("implied_team_total") or 0)-(total/2-home_spread/2))>0.01)
 }
}
print("RICE_V2_EXACT_SOURCE_TRACE_BEGIN",flush=True)
print(json.dumps(output,sort_keys=True,indent=2,allow_nan=False),flush=True)
print("RICE_V2_EXACT_SOURCE_TRACE_END",flush=True)
Path("/tmp/rice_v2_exact_source_trace.json").write_text(json.dumps(output,indent=2,allow_nan=False))
C.close()

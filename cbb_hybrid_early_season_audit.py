"""Historical backward-transfer audit of the frozen CBB hybrid recipe.

2023 and 2024 were used in the original development, so these are NOT fresh
holdouts. Refit each architecture using only dates prior to each test season.
No changes to current forecasts, no parameter selection on the test season.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cbb_game_engine_v0 import read_data,build_rows,fitted,causal_residual_features,assemble_hybrid,hybrid_stress
from cbb_neutral_challenger import symmetric_predict,combined_with_elo
from cbb_probability_stress import elo_predict_all

def run(rows):
    assert len({r["game_id"] for r in rows})==len(rows)
    ext=causal_residual_features(rows,[])
    elo=elo_predict_all(rows)
    def x(rs,extended=False):
        return np.stack([ext[r["game_id"]] if extended else r["x"] for r in rs])
    def y(rs):
        return np.stack([r["y"] for r in rs])
    def nonlinear():
        return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=10.,random_state=17))
    def ridge(alpha):
        return make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))
    def neutral_correct(rs,model,pred,scale):
        mask=np.array([r["neutral_site"] for r in rs])
        pred=pred.copy()
        if mask.any():
            pred[mask]=combined_with_elo(symmetric_predict(model,x(rs)[mask]),
                np.array([elo[str(r["game_id"])] for r in rs if r["neutral_site"]]),float(scale[0]))
        return pred,mask
    output={"schema":"CBB_HYBRID_BACKWARD_TRANSFER_V1","promotion_allowed":False,
            "note":"Frozen architecture, rolling earlier-season refits. 2023/24 were original development seasons, not genuinely unseen research data.",
            "cohorts":{}}
    for season in (2023,2024):
        train=[r for r in rows if 2019<=r["season"]<season-1]
        cal=[r for r in rows if r["season"]==season-1]
        test=[r for r in rows if r["season"]==season]
        assert min(len(train),len(cal),len(test))>=500
        assert max(r["tip_utc"] for r in train)<min(r["tip_utc"] for r in cal)
        assert max(r["tip_utc"] for r in cal)<min(r["tip_utc"] for r in test)
        assert len(set(r["game_id"] for r in train+cal+test))==len(train+cal+test)
        alpha=min((10,100,1000),key=lambda a:float(np.abs(fitted(train,a).predict(x(cal))-y(cal)).mean(axis=0).sum()))
        def scale(pred):
            return np.maximum(np.std(pred-y(cal),axis=0,ddof=1),[7.,9.])
        base=fitted(train,alpha)
        bs=scale(base.predict(x(cal)))
        c=nonlinear().fit(x(train,True),y(train))
        cs=scale(c.predict(x(cal,True)))
        t=ridge(alpha).fit(x(train,True),y(train))
        ts=scale(t.predict(x(cal,True)))
        train2=train+cal
        base=fitted(train2,alpha)
        original,mask=neutral_correct(test,base,base.predict(x(test)),bs)
        c=nonlinear().fit(x(train2,True),y(train2))
        t=ridge(alpha).fit(x(train2,True),y(train2))
        hybrid,hs=assemble_hybrid(original,bs,c.predict(x(test,True)),cs,t.predict(x(test,True)),ts,mask)
        assert np.array_equal(original[mask],hybrid[mask])
        results=hybrid_stress(test,original,bs,hybrid,hs,elo)
        output["cohorts"][str(season)]={"fit_seasons":sorted(set(r["season"] for r in train2)),
             "calibration_season":season-1,"training_games_before_calibration":len(train),
             "calibration_games":len(cal),"test_games":len(test),"ridge_alpha":alpha,
             "baseline_scale":bs.tolist(),"combined_scale":cs.tolist(),"totals_scale":ts.tolist(),
             "results":results}
        all_=results["cohorts"]["all"]
        compact={"season":season,"n":len(test),"all":all_,
            "home_away":results["cohorts"]["home_away"],"neutral":results["cohorts"]["neutral"],
            "disagreement":results["cohorts"].get("model_disagreement"),
            "close_elo":results["cohorts"].get("close_elo"),
            "original_top4":results["original_reliability"]["top4_each_utc_day"],
            "hybrid_top4":results["hybrid_reliability"]["top4_each_utc_day"],
            "elo_brier":results["elo_reference_brier"]}
        print("EARLY_SEASON_RESULT:",json.dumps(compact,sort_keys=True),flush=True)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("team-box","player-box","schedule"):
        p.add_argument("--"+name,action="append",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    rows,_=build_rows(read_data(a.schedule),read_data(a.team_box),read_data(a.player_box))
    result=run(rows)
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":
    main()

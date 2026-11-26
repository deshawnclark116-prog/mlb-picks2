#!/usr/bin/env python3
"""Neutral-site CBB expert falsification: invariance, fixed expert, and Elo blend.

NO 2026 result is used to choose a challenger. Candidate definitions and
eligibility gates are fixed before cohort evaluation. The 2026 season has
already been inspected; its comparison is *diagnostic* rather than clean
promotion evidence. No old predictions are overwritten.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from statistics import NormalDist

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cbb_probability_stress import elo_predict_all, normcdf

SCHEMA = "CBB_NEUTRAL_COURT_CHALLENGER_V1"
MODEL_CANDIDATES = ("incumbent", "swap_invariant", "neutral_specialist", "swap_plus_fixed_elo")
NEUTRAL_ALPHA = 300.  # fixed in source; NEVER selected using historical 2026
ELO_BLEND_WEIGHT = .50  # one fixed falsifiable challenger
MIN_VALIDATION_GAMES = 150
MIN_BRIER_GAIN_TO_WIN = .002
MAX_MARGIN_MAE_DEGRADATION = .5
# 22 features in cbb_game_engine_v0.FEATURE_NAMES; swapping team identity.
PAIR_SWAP = ((4,5), (6,7), (8,9), (10,11), (12,13), (14,15), (16,17), (20,21))
NEGATE_SWAP = (0,18,19)


def neutral_swap(x):
    """Exact reverse assignment on neutral court, no fake advantage assigned."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1,2) or arr.shape[-1] != 22:
        raise ValueError("unexpected v0 feature vector dimension")
    z = arr.copy()
    if not np.isfinite(arr).all() or not np.allclose(arr[...,2],1):
        raise ValueError("not a finite neutral-court feature vector")
    for i,j in PAIR_SWAP:
        z[...,i],z[...,j] = arr[...,j],arr[...,i]
    for i in NEGATE_SWAP:
        z[...,i] = -arr[...,i]
    return z


def symmetric_predict(model, x):
    """Game-identical neutral margins when ESPN home and away labels reverse."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None,:]
    forward = np.asarray(model.predict(a), dtype=float)
    reverse = np.asarray(model.predict(neutral_swap(a)), dtype=float)
    if forward.shape != (len(a),2) or reverse.shape != forward.shape:
        raise ValueError("unexpected model dimension")
    return np.column_stack(((forward[:,0]-reverse[:,0])/2,
                            (forward[:,1]+reverse[:,1])/2))


def neutral_specialist(train):
    """Fixed high-regularization Ridge with neutral examples and swapped pairs.

    The same record contributes both perspectives, but swapped records are
    NOT independent training games. Each original game is still one trial.
    """
    subset = [r for r in train if r["neutral_site"]]
    if len(subset) < 100:
        raise ValueError("insufficient neutral training cohort")
    x = np.stack([r["x"] for r in subset])
    y = np.stack([r["y"] for r in subset])
    xx = np.concatenate((x,neutral_swap(x)),axis=0)
    yy = np.concatenate((y,y*np.array([-1.,1.])),axis=0)
    model = make_pipeline(StandardScaler(),Ridge(alpha=NEUTRAL_ALPHA))
    model.fit(xx,yy)
    return model


def combined_with_elo(symmetric, baseline_elo, margin_sd):
    """Fixed half probability/half Elo, projected margin inverted consistently."""
    if margin_sd <= 0:
        raise ValueError("positive sigma required")
    native = np.array([normcdf(m/margin_sd) for m in symmetric[:,0]],dtype=float)
    elo = np.asarray(baseline_elo,dtype=float)
    if native.shape != elo.shape or not np.isfinite(elo).all() or np.any((elo<0)|(elo>1)):
        raise ValueError("Elo probabilities unavailable for games")
    combined = (1-ELO_BLEND_WEIGHT)*native + ELO_BLEND_WEIGHT*elo
    median_margin = np.array([
        margin_sd*NormalDist().inv_cdf(float(np.clip(p,1e-8,1-1e-8)))
        for p in combined],dtype=float)
    return np.column_stack((median_margin,symmetric[:,1]))


def eval_neutral(rows,preds,sigma):
    if not rows:
        return {"n":0}
    truth = np.stack([r["y"] for r in rows])
    p = np.array([normcdf(float(m/sigma)) for m in preds[:,0]])
    outcomes = (truth[:,0]>0).astype(float)
    return {
        "n":len(rows),
        "moneyline_brier":round(float(np.mean((p-outcomes)**2)),6),
        "winner_accuracy":round(float(np.mean((p>=.5)==outcomes)),5),
        "margin_mae":round(float(np.mean(np.abs(preds[:,0]-truth[:,0]))),4),
        "total_mae":round(float(np.mean(np.abs(preds[:,1]-truth[:,1]))),4),
        "home_win_rate":round(float(np.mean(outcomes)),5),
        "mean_model_home_win":round(float(np.mean(p)),5),
        "mean_predicted_home_margin":round(float(np.mean(preds[:,0])),4),
        "mean_actual_home_margin":round(float(np.mean(truth[:,0])),4),
    }


def compare_on_cohort(rows, original_model, specialist, elo_by_id, sigma):
    neutral = [r for r in rows if r["neutral_site"]]
    if len(neutral)<1:
        raise ValueError("no neutral games")
    x=np.stack([r["x"] for r in neutral])
    elo = np.array([elo_by_id[str(r["game_id"])] for r in neutral])
    baseline=original_model.predict(x)
    swap=symmetric_predict(original_model,x)
    special=symmetric_predict(specialist,x)
    combined=combined_with_elo(swap,elo,sigma)
    predictions={"incumbent":baseline,"swap_invariant":swap,
                 "neutral_specialist":special,"swap_plus_fixed_elo":combined}
    outputs={name:eval_neutral(neutral,pred,sigma) for name,pred in predictions.items()}
    outcome=np.array([float(r["y"][0]>0) for r in neutral])
    outputs["elo_reference"]={
        "n":len(neutral),"moneyline_brier":round(float(np.mean((elo-outcome)**2)),6),
        "winner_accuracy":round(float(np.mean((elo>=.5)==outcome)),5),
    }
    return outputs


def selection_from_2025(metrics):
    """Precommitted gate. Do not look at 2026 to select a model."""
    baseline=metrics["incumbent"]
    results=[]
    for name in MODEL_CANDIDATES[1:]:
        m=metrics[name]
        gain=baseline["moneyline_brier"]-m["moneyline_brier"]
        mae_degrade=m["margin_mae"]-baseline["margin_mae"]
        qualified=(m["n"]>=MIN_VALIDATION_GAMES and
                   gain>=MIN_BRIER_GAIN_TO_WIN and
                   mae_degrade<=MAX_MARGIN_MAE_DEGRADATION and
                   m["winner_accuracy"]+0.005>=baseline["winner_accuracy"])
        results.append({
            "name":name,"qualified":bool(qualified),
            "brier_improvement":round(gain,6),
            "margin_mae_delta":round(mae_degrade,4),
            "win_accuracy_delta":round(m["winner_accuracy"]-baseline["winner_accuracy"],5),
        })
    qualifying=sorted([r for r in results if r["qualified"]],
                      key=lambda r:(-r["brier_improvement"],r["name"]))
    return {
        "qualification_rules":{
            "minimum_2025_neutral_games":MIN_VALIDATION_GAMES,
            "minimum_brier_gain_vs_incumbent":MIN_BRIER_GAIN_TO_WIN,
            "maximum_margin_mae_increase":MAX_MARGIN_MAE_DEGRADATION,
            "maximum_winner_accuracy_drop":.005
        },
        "candidates":results,
        "chosen_using_2025_only":qualifying[0]["name"] if qualifying else "incumbent",
    }


def challenger_audit(completed, train, validation, holdout,
                     train_model, val_predictions, holdout_model,
                     holdout_predictions, sigma):
    """Produce both comparison periods; no model mutation or promotion."""
    elo=elo_predict_all(completed)
    fixed_specialist=neutral_specialist(train)
    val=compare_on_cohort(validation,train_model,fixed_specialist,elo,sigma)
    selected=selection_from_2025(val)
    holdout_specialist=neutral_specialist(train+validation)
    diag=compare_on_cohort(holdout,holdout_model,holdout_specialist,elo,sigma)
    return {
        "schema":SCHEMA,
        "hypothesis":"Neutral-site home/away label symmetry and independent Elo prior",
        "predeclared_candidates":list(MODEL_CANDIDATES),
        "validation_2025":val,
        "selection_decided_2025_only":selected,
        "previously_inspected_2026_diagnostic":diag,
        "2026_is_not_untouched_for_new_hypotheses":True,
        "training_non_neutral_unchanged":True,
        "upcoming_production_prediction_changes":False,
        "promotion_allowed":False,
        "promote_after_new_first_seen_2027_forward_evaluation_only":True,
        "note":"This tests mechanism and calibration, not travel/venue claims or sportsbook ROI",
    }


def elo_future_probabilities(completed, upcoming):
    """Future as-of Elo given ONLY completed earlier games, never future finals.

    Uses the same fixed Elo parameters and date grouping as audit Elo; after
    processing known completed seasons through current as-of, projects neutral
    upcoming fixtures without retraining Elo from their future scores.
    """
    from cbb_probability_stress import (
        ELO_HOME_ADVANTAGE, ELO_K, ELO_REGRESSION_TO_MEAN,
        _group_by_season_date,
    )
    games = sorted(completed, key=lambda r: (r["season"],r["tip_utc"],r["game_id"]))
    ratings = {}
    last_season = None
    for (_, _day), group in _group_by_season_date(games):
        season = int(group[0]["season"])
        if last_season != season:
            if last_season is not None:
                ratings = {team: 1500+(old-1500)*(1-ELO_REGRESSION_TO_MEAN)
                           for team, old in ratings.items()}
            last_season = season
        changes=defaultdict(float)
        for r in group:
            home, away=str(r["home_team_id"]),str(r["away_team_id"])
            advantage=0. if r["neutral_site"] else ELO_HOME_ADVANTAGE
            p=1/(1+10**(-(ratings.get(home,1500.) -
                            ratings.get(away,1500.)+advantage)/400))
            delta=ELO_K*(float(r["y"][0]>0)-p)
            changes[home]+=delta
            changes[away]-=delta
        for team, update in changes.items():
            ratings[team]=ratings.get(team,1500.)+update
    predictions={}
    for r in upcoming:
        if not r["neutral_site"]:
            continue
        season=int(r["season"])
        if last_season is None:
            prior={}
        elif season > last_season:
            prior={team:1500+(val-1500)*(1-ELO_REGRESSION_TO_MEAN)
                   for team,val in ratings.items()}
        elif season==last_season:
            prior=ratings
        else:
            # Do not project a game before the latest known completed season.
            continue
        home,away=str(r["home_team_id"]),str(r["away_team_id"])
        diff=prior.get(home,1500.)-prior.get(away,1500.)
        predictions[str(r["game_id"])]=1/(1+10**(-diff/400))
    return predictions


def shadow_upcoming(upcoming, incumbent, specialist, asof, model_sig, elo=None):
    """Future neutral shadow only; official original picks never replaced."""
    out=[]
    for row in upcoming:
        if not row["neutral_site"]:
            continue
        x=np.asarray(row["x"])[None,:]
        base=incumbent.predict(x)[0]
        swapped=symmetric_predict(incumbent,x)[0]
        special=symmetric_predict(specialist,x)[0]
        i={"game_id":str(row["game_id"]),"tip_utc":row["tip_utc"],
           "home":row["home"],"away":row["away"],
           "neutral_site":True,
           "incumbent_home_margin":round(float(base[0]),3),
           "incumbent_total":round(float(base[1]),3),
           "swap_home_margin":round(float(swapped[0]),3),
           "swap_total":round(float(swapped[1]),3),
           "specialist_home_margin":round(float(special[0]),3),
           "specialist_total":round(float(special[1]),3),
           "model_uncertainty_sd":round(float(model_sig),3),
           "selected_challenger_pretip_not_approved":True}
        if elo and str(row["game_id"]) in elo:
            e=elo[str(row["game_id"])]
            combine=combined_with_elo(swapped[None,:],np.array([e]),model_sig)[0]
            i["fixed_elo_home_win"]=round(float(e),5)
            i["elo_blend_home_margin"]=round(float(combine[0]),3)
            i["elo_blend_total"]=round(float(combine[1]),3)
        out.append(i)
    return out


def report_markdown(report):
    lines=[
        "# CBB neutral-site expert challenge",
        "",
        "**Do not publish challengers as official picks yet.** The 2026 cohort "
        "has already been examined; no more results from it count as fresh validation.",
        "",
        "## Development validation, 2025 (selection made using this only)", "",
        "| Method | Games | Moneyline Brier ↓ | Winner accuracy | Margin MAE ↓ | Total MAE ↓ |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for section,label in (("validation_2025","Development 2025"),
                          ("previously_inspected_2026_diagnostic","Inspected 2026 (diagnostic)")):
        if section!="validation_2025":
            lines += ["",f"## {label}","",
                      "| Method | Games | Moneyline Brier ↓ | Winner accuracy | Margin MAE ↓ | Total MAE ↓ |",
                      "|---|---:|---:|---:|---:|---:|"]
        for name, r in report[section].items():
            lines.append(
                f"| {name.replace('_',' ')} | {r['n']} | {r['moneyline_brier']:.6f} | "
                f"{100*r['winner_accuracy']:.2f}% | "
                f"{r.get('margin_mae','—')} | {r.get('total_mae','—')} |"
            )
    selection=report["selection_decided_2025_only"]
    lines+=["",f"**2025-only selected:** {selection['chosen_using_2025_only']}.",
            "This is a research shadow selection, not a production promotion.",
            "", "## Challenger gates", "",
            f"Fixed minimum improvement in 2025 Brier: {MIN_BRIER_GAIN_TO_WIN}, "
            f"max margin MAE degradation {MAX_MARGIN_MAE_DEGRADATION}, "
            "max winner accuracy decrease 0.5 percentage points.",
            "", "**Next:** new prospectively frozen 2027 neutral games and official "
            "finals must verify improvement. Do not override a failing neutral "
            "court model with unjustified '85% confidence'.", ""]
    return "\n".join(lines)


def save(report, json_path, markdown_path):
    import json
    from pathlib import Path
    p=Path(json_path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    m=Path(markdown_path);m.parent.mkdir(parents=True,exist_ok=True)
    m.write_text(report_markdown(report))

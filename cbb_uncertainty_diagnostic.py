#!/usr/bin/env python3
"""Pre-registered pregame-data-sufficiency uncertainty challenger.

The 2025 cohort estimates uncertainty; the previously *inspected* 2026 cohort
is an exploratory diagnostic. No player prop, sportsbook price or production
probability is changed. No parameter selection uses 2026 performance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from cbb_probability_stress import normcdf

SCHEMA = "CBB_UNCERTAINTY_STRATIFICATION_EXPERIMENT_V1"
GROUPS = ("one_or_more_teams_zero_games", "both_teams_1_to_4_games", "at_least_one_team_5plus_games")
SHRINK_PSEUDO_GAMES = 100.


def group(row):
    home = int(row["home_prior_games"])
    away = int(row["away_prior_games"])
    if min(home, away) < 0:
        raise ValueError("negative prior-game count")
    if min(home, away) == 0:
        return GROUPS[0]
    if max(home, away) < 5:
        return GROUPS[1]
    return GROUPS[2]


def learn_scales(validation, prediction, reference_sd):
    """No access to 2026. Fixed 100-game pooling prevents noisy group sigma."""
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(validation), 2):
        raise ValueError("validation/prediction mismatch")
    reference_sd = np.asarray(reference_sd, dtype=float)
    if reference_sd.shape != (2,) or (reference_sd <= 0).any():
        raise ValueError("invalid pooled reference variance")
    true = np.asarray([r["y"] for r in validation], dtype=float)
    residual = true - prediction
    estimates = {}
    for bucket in GROUPS:
        idx = [i for i, r in enumerate(validation) if group(r) == bucket]
        count = len(idx)
        if count:
            square_sum = np.sum(residual[idx, :] ** 2, axis=0)
        else:
            square_sum = np.zeros(2)
        group_var = ((square_sum + SHRINK_PSEUDO_GAMES*reference_sd**2) /
                     (count+SHRINK_PSEUDO_GAMES))
        estimates[bucket] = {
            "training_n_2025": count,
            "sd_margin": round(math.sqrt(float(group_var[0])), 6),
            "sd_total": round(math.sqrt(float(group_var[1])), 6),
        }
    return estimates


def gaussian_nll(residual, sd):
    if sd <= 0:
        raise ValueError("invalid normal sd")
    return 0.5 * math.log(2*math.pi*sd*sd) + residual*residual/(2*sd*sd)


def cohort_report(rows, predictions, pooled_sd, candidates):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(rows),2):
        raise ValueError("cohort/prediction mismatch")
    truth = np.asarray([r["y"] for r in rows])
    residual = truth - predictions
    baseline = np.asarray(pooled_sd,dtype=float)
    report = {}
    all_results = []
    for bucket in GROUPS:
        idx = [i for i,r in enumerate(rows) if group(r)==bucket]
        statistics = {"games": len(idx)}
        if idx:
            vals=[]
            for i in idx:
                row=rows[i]
                v=residual[i]
                candidate=np.array([candidates[group(row)]["sd_margin"],
                                    candidates[group(row)]["sd_total"]])
                original_p=normcdf(float(predictions[i,0]/baseline[0]))
                candidate_p=normcdf(float(predictions[i,0]/candidate[0]))
                win=float(row["y"][0]>0)
                vals.append({
                    "original_margin_cover": float(abs(v[0])<=1.2815515655*baseline[0]),
                    "candidate_margin_cover": float(abs(v[0])<=1.2815515655*candidate[0]),
                    "original_total_cover": float(abs(v[1])<=1.2815515655*baseline[1]),
                    "candidate_total_cover": float(abs(v[1])<=1.2815515655*candidate[1]),
                    "original_joint_nll": sum(gaussian_nll(float(v[j]),float(baseline[j])) for j in range(2)),
                    "candidate_joint_nll": sum(gaussian_nll(float(v[j]),float(candidate[j])) for j in range(2)),
                    "original_moneyline_brier": (original_p-win)**2,
                    "candidate_moneyline_brier": (candidate_p-win)**2,
                })
            for key in vals[0]:
                statistics[key+"_mean"]=round(float(np.mean([v[key] for v in vals])),6)
            all_results += vals
        report[bucket] = statistics
    overall={"games":len(rows)}
    if all_results:
        for key in all_results[0]:
            overall[key+"_mean"]=round(float(np.mean([v[key] for v in all_results])),6)
    return {"overall":overall, "groups":report}


def experiment(validation, validation_predictions, holdout, holdout_predictions, pooled_sd):
    scales=learn_scales(validation,validation_predictions,pooled_sd)
    return {
        "schema":SCHEMA,
        "hypothesis":"Pregame roster continuity / game sample size changes score uncertainty",
        "fixed_group_cutoffs":"0 prior games / both 1-4 prior games / any 5+ games",
        "shrinkage_pseudo_games":SHRINK_PSEUDO_GAMES,
        "scales_learned_using_2025_only":scales,
        "development_2025_in_sample_uncertainty":cohort_report(
            validation,validation_predictions,pooled_sd,scales),
        "previously_examined_2026_diagnostic":cohort_report(
            holdout,holdout_predictions,pooled_sd,scales),
        "promotion_allowed":False,
        "reason":"2026 historical outcomes were inspected earlier; need new prospective evaluation",
        "market_odds_used":False,
        "model_coefficients_changed":False,
        "uncertainty_changes_live_predictions":False,
    }


def to_markdown(result):
    lines=[
        "# CBB conditional score-uncertainty experiment",
        "",
        "**Research only.** This challenger was fitted on 2025 residuals and "
        "compared diagnostically to the already inspected 2026 season. "
        "**It is not a new untouched holdout and is not promoted.**",
        "",
        "| Pregame history cohort | 2025 training games | Margin SD | Total SD |",
        "|---|---:|---:|---:|",
    ]
    for k, v in result["scales_learned_using_2025_only"].items():
        lines.append(f"| {k.replace('_',' ')} | {v['training_n_2025']} | "
                     f"{v['sd_margin']:.2f} | {v['sd_total']:.2f} |")
    for key,title in (
        ("development_2025_in_sample_uncertainty","Development residuals (in-sample): 2025"),
        ("previously_examined_2026_diagnostic","Previously inspected 2026 cohort — diagnostic only"),
    ):
        d=result[key]
        lines += ["",f"## {title}","",
                  "| Sample | Games | Margin nominal 80% old → candidate | Total nominal 80% old → candidate | Moneyline Brier old → candidate | Combined normal NLL old → candidate |",
                  "|---|---:|---|---|---|---|"]
        for bucket, metrics in list(d["groups"].items())+[("overall",d["overall"])]:
            if not metrics["games"]:
                continue
            lines.append(f"| {bucket.replace('_',' ')} | {metrics['games']} | "
                         f"{100*metrics['original_margin_cover_mean']:.1f}% → {100*metrics['candidate_margin_cover_mean']:.1f}% | "
                         f"{100*metrics['original_total_cover_mean']:.1f}% → {100*metrics['candidate_total_cover_mean']:.1f}% | "
                         f"{metrics['original_moneyline_brier_mean']:.5f} → {metrics['candidate_moneyline_brier_mean']:.5f} | "
                         f"{metrics['original_joint_nll_mean']:.3f} → {metrics['candidate_joint_nll_mean']:.3f} |")
    lines += ["",
              "Lower moneyline Brier and normal negative log-likelihood are better. "
              "Coverage closer to 80% is preferable, but intervals must not be widened "
              "without penalizing uncertainty size.",
              "",
              "**Result is exploratory only:** do not treat 2026 as a new test or "
              "change the live engine without future pretip-captured outcomes.",
              ""]
    return "\n".join(lines)


def save(report,json_path,markdown_path):
    p=Path(json_path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    t=Path(markdown_path)
    t.parent.mkdir(parents=True,exist_ok=True)
    t.write_text(to_markdown(report))

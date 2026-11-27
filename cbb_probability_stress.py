#!/usr/bin/env python3
"""CBB model falsification audit: Elo, calibration, game-type slices, uncertainty.

NO new model training or selection occurs here. The 2025-26 historical cohort
was already examined, so further uses are *diagnostic*, not an untouched
promotion holdout. No bookmaker performance claims without original odds.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA = "CBB_PROBABILITY_FALSIFICATION_AUDIT_V1"
ELO_HOME_ADVANTAGE = 80.
ELO_K = 20.
ELO_REGRESSION_TO_MEAN = .25


def finite_prob(p):
    f = float(p)
    if not math.isfinite(f) or f < 0 or f > 1:
        raise ValueError("invalid probability")
    return f


def normcdf(x):
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.))


def wilson(k, n, z=1.95996398454):
    """95% Wilson interval for n independent binary games (not a blocked CI)."""
    if n < 1:
        return None
    p = k / n
    den = 1. + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return [round(max(0., ctr-half), 4), round(min(1., ctr+half), 4)]


def game_day(row):
    # Preserve the same UTC day grouping used in score feature construction.
    return datetime.fromisoformat(row["tip_utc"]).date().isoformat()


def elo_predict_all(completed):
    """Fixed, independent 80-point home-advantage, K=20 Elo benchmark.

    Preseason regress 25% toward 1500; preseason and K are fixed in code,
    not selected based on either 2025 validation or the previously examined
    2026 holdout. Update Elo only *after all games on a UTC date* were scored
    to avoid using a same-day earlier final in the next event.
    """
    games = sorted(completed, key=lambda r: (r["season"], game_day(r), r["tip_utc"], r["game_id"]))
    ratings = {}
    predictions = {}
    current_season = None
    for (_, day), group in _group_by_season_date(games):
        season = int(group[0]["season"])
        if current_season != season:
            if current_season is not None:
                ratings = {
                    team: 1500. + (value - 1500.)*(1. - ELO_REGRESSION_TO_MEAN)
                    for team, value in ratings.items()
                }
            current_season = season
        updates = defaultdict(float)
        for r in group:
            home, away = str(r["home_team_id"]), str(r["away_team_id"])
            if home == away:
                raise ValueError("same team on both sides")
            h, a = ratings.get(home, 1500.), ratings.get(away, 1500.)
            prior_difference = h - a + (0. if r["neutral_site"] else ELO_HOME_ADVANTAGE)
            p = 1 / (1 + 10 ** (-prior_difference / 400.))
            gid = str(r["game_id"])
            if gid in predictions:
                raise ValueError("duplicate game id")
            predictions[gid] = finite_prob(p)
            outcome = float(r["y"][0] > 0)
            delta = ELO_K*(outcome-p)
            updates[home] += delta
            updates[away] -= delta
        for team, update in updates.items():
            ratings[team] = ratings.get(team, 1500.) + update
    return predictions


def _group_by_season_date(games):
    groups = defaultdict(list)
    for row in games:
        groups[(int(row["season"]), game_day(row))].append(row)
    return sorted(groups.items(), key=lambda pair: pair[0])


def score_subset(rows, model_p, elo_p, margin_pred, total_pred, margin_sd, total_sd):
    """Summary with independent stronger baseline, observed calibration/coverage."""
    n = len(rows)
    if n == 0:
        return {"games": 0}
    outcomes = np.array([float(r["y"][0] > 0) for r in rows])
    margins = np.array([float(r["y"][0]) for r in rows])
    totals = np.array([float(r["y"][1]) for r in rows])
    p = np.array(model_p)
    ep = np.array(elo_p)
    mp = np.array(margin_pred)
    tp = np.array(total_pred)
    correct = ((p >= .5) == outcomes).astype(float)
    elo_correct = ((ep >= .5) == outcomes).astype(float)
    model_brier = np.mean((p-outcomes)**2)
    elo_brier = np.mean((ep-outcomes)**2)
    coverage_m = np.mean(np.abs(margins-mp) <= 1.281551565545*margin_sd)
    coverage_t = np.mean(np.abs(totals-tp) <= 1.281551565545*total_sd)
    observed_wins = float(np.mean(outcomes))
    return {
        "games": n,
        "moneyline_accuracy": round(float(np.mean(correct)), 4),
        "moneyline_accuracy_wilson95": wilson(int(sum(correct)), n),
        "elo_accuracy": round(float(np.mean(elo_correct)), 4),
        "moneyline_brier": round(float(model_brier), 6),
        "elo_brier": round(float(elo_brier), 6),
        "brier_model_minus_elo": round(float(model_brier-elo_brier), 6),
        "moneyline_logloss": round(float(np.mean(
            -outcomes*np.log(np.clip(p,1e-9,1-1e-9))
            -(1-outcomes)*np.log(np.clip(1-p,1e-9,1-1e-9))
        )), 6),
        "margin_mae": round(float(np.mean(np.abs(margins-mp))), 3),
        "total_mae": round(float(np.mean(np.abs(totals-tp))), 3),
        "margin_nominal_80pct_interval_coverage": round(float(coverage_m), 4),
        "total_nominal_80pct_interval_coverage": round(float(coverage_t), 4),
        "observed_home_win_fraction": round(observed_wins, 4),
        "mean_predicted_home_win": round(float(np.mean(p)), 4),
        "mean_absolute_win_probability_error": round(float(np.mean(np.abs(p-outcomes))), 4),
    }


def block_bootstrap_brier_difference(rows, p, elo, iterations=1500, seed=20261125):
    """Resample complete UTC game-days, not independent games.

    This CI is *descriptive*, conditional on retrospectively seen model and
    cohort. It does NOT undo the benchmark's previous selection/inspection.
    """
    perday = defaultdict(list)
    for r, m, e in zip(rows, p, elo):
        outcome = float(r["y"][0]>0)
        perday[game_day(r)].append(((m-outcome)**2-(e-outcome)**2))
    day_sum = np.array([sum(v) for _, v in sorted(perday.items())], dtype=float)
    day_n = np.array([len(v) for _, v in sorted(perday.items())], dtype=float)
    if len(day_sum) < 8:
        return None
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(day_sum), size=(iterations, len(day_sum)))
    means = np.sum(day_sum[choices], axis=1) / np.sum(day_n[choices], axis=1)
    return {
        "day_blocks": len(day_sum), "resamples": iterations,
        "95pct_interval_model_minus_elo": [round(float(x), 6)
                                           for x in np.quantile(means, [.025, .975])],
        "frequency_model_beats_elo_in_resamples": round(float(np.mean(means < 0)), 4),
        "warning": "POST_SELECTION_DESCRIPTIVE_INTERVAL_NOT_FRESH_CONFIRMATION",
    }


def audit_cohort(cohort, predictions, elo_by_id, sd):
    if len(cohort) != len(predictions):
        raise ValueError("cohort/prediction length mismatch")
    if len(cohort) < 1:
        raise ValueError("empty cohort")
    if any(str(r["game_id"]) not in elo_by_id for r in cohort):
        raise ValueError("missing independent Elo as-of baseline")
    pred = np.asarray(predictions, dtype=float)
    if pred.shape != (len(cohort), 2) or not np.isfinite(pred).all():
        raise ValueError("invalid model outputs")
    ms, ts = [float(x) for x in sd]
    if ms <= 0 or ts <= 0:
        raise ValueError("bad uncertainty")
    margin = pred[:,0]
    total = pred[:,1]
    probabilities = [finite_prob(normcdf(x/ms)) for x in margin]
    elo = [finite_prob(elo_by_id[str(r["game_id"])]) for r in cohort]
    overall = score_subset(cohort, probabilities, elo, margin, total, ms, ts)
    subsets = {}
    masks = {
        "neutral_site": [bool(r["neutral_site"]) for r in cohort],
        "listed_home_court": [not bool(r["neutral_site"]) for r in cohort],
        "both_teams_fewer_than_5_current_games": [
            max(int(r["home_prior_games"]), int(r["away_prior_games"])) < 5 for r in cohort],
        "both_teams_5plus_current_games": [
            min(int(r["home_prior_games"]), int(r["away_prior_games"])) >= 5 for r in cohort],
        "any_team_no_current_games": [
            min(int(r["home_prior_games"]), int(r["away_prior_games"])) == 0 for r in cohort],
        "both_have_prior_player_minutes": [
            bool(r.get("player_box_coverage")) for r in cohort],
        "home_is_model_favorite": [p>=.5 for p in probabilities],
        "away_is_model_favorite": [p<.5 for p in probabilities],
    }
    for name, mask in masks.items():
        idx = [i for i, b in enumerate(mask) if b]
        subsets[name] = score_subset(
            [cohort[i] for i in idx],
            [probabilities[i] for i in idx],
            [elo[i] for i in idx],
            [margin[i] for i in idx], [total[i] for i in idx], ms, ts,
        )
    buckets = []
    side_probabilities = [max(p, 1-p) for p in probabilities]
    for low, high in ((.50,.60), (.60,.70), (.70,.80), (.80,.90), (.90,1.001)):
        idx = [i for i, p in enumerate(side_probabilities) if low <= p < high]
        winners = [float((probabilities[i]>=.5) == (cohort[i]["y"][0]>0))
                   for i in idx]
        buckets.append({
            "predicted_favorite_probability_range": [low, round(min(high,1.),3)],
            "games": len(idx),
            "mean_predicted_probability": round(float(np.mean([side_probabilities[i] for i in idx])), 4) if idx else None,
            "observed_favorite_win_rate": round(float(np.mean(winners)), 4) if idx else None,
            "wilson95_for_win_rate": wilson(int(sum(winners)), len(idx)),
            "observed_minus_expected_pp": round(
                100*(float(np.mean(winners))-float(np.mean([side_probabilities[i] for i in idx]))), 2)
                if idx else None,
        })
    high_confidence = []
    for floor in (.65,.70,.75,.80,.85,.90):
        idx = [i for i, p in enumerate(side_probabilities) if p>=floor]
        winners = sum((probabilities[i]>=.5) == (cohort[i]["y"][0]>0) for i in idx)
        high_confidence.append({
            "threshold": floor, "games": len(idx),
            "hit_rate": round(winners/len(idx),4) if idx else None,
            "wilson95": wilson(int(winners), len(idx)),
            "mean_model_probability": round(float(np.mean([side_probabilities[i] for i in idx])),4) if idx else None
        })
    return {
        "overall": overall,
        "subgroups": subsets,
        "reliability": buckets,
        "model_selected_confidence_cutoffs": high_confidence,
        "elo_baseline_fixed": {"initial": 1500, "K": ELO_K, "home_advantage_elo": ELO_HOME_ADVANTAGE,
                               "offseason_regression_to_prior_fraction": ELO_REGRESSION_TO_MEAN},
        "blocked_brier_difference": block_bootstrap_brier_difference(cohort, probabilities, elo),
        "caveats": [
            "Historical scores/features reconstructed after games, not archived first-seen",
            "2025 and 2026 were viewed during earlier research; subgroup audit is diagnostic only",
            "Reference Elo chosen as simple fixed conventional benchmark, not optimized here",
            "No actual moneyline prices, spread lines, total lines or betting ROI in this audit",
            "Wilson intervals treat games independently; use daily blocked bootstrap for model-vs-Elo Brier",
            "Model probabilities assume fixed validation-season normal residual width",
        ],
    }


def make_audit(completed, validation, val_preds, holdout, holdout_preds, sd):
    elo = elo_predict_all(completed)
    return {
        "schema": SCHEMA,
        "purpose": "FALSIFICATION_AND_CALIBRATION_ONLY_NO_MODEL_SELECTION",
        "validation_2025": audit_cohort(validation, val_preds, elo, sd),
        "previously_inspected_2026_test": audit_cohort(holdout, holdout_preds, elo, sd),
        "forward_prospective_accuracy_not_yet_measured": True,
        "no_profitable_moneyline_ats_total_claim": True,
    }


def format_markdown(audit):
    out = [
        "# CBB probability falsification and stronger Elo benchmark",
        "",
        "**Diagnostics only:** 2026 has already been inspected. These analyses do "
        "not create new untouched holdout evidence, and do not justify promotion.",
        "",
        "All winning probabilities refer to an actual game winner; no odds were "
        "used. ATS and over/under profitability are NOT measured here.",
        "",
    ]
    for key, title in (("validation_2025", "Development validation: 2025"),
                       ("previously_inspected_2026_test", "Previously inspected historical cohort: 2026")):
        item = audit[key]
        t = item["overall"]
        out += [
            f"## {title}", "",
            f"Games: {t['games']}; model win accuracy: **{100*t['moneyline_accuracy']:.2f}%**; "
            f"fixed Elo win accuracy: **{100*t['elo_accuracy']:.2f}%**.",
            f"Probability Brier: model **{t['moneyline_brier']:.5f}**, "
            f"Elo **{t['elo_brier']:.5f}** "
            f"(model−Elo: {t['brier_model_minus_elo']:+.5f}; lower is better).",
            f"Nominal 80% point intervals: margin covered **"
            f"{100*t['margin_nominal_80pct_interval_coverage']:.1f}%** of finals; "
            f"total covered **{100*t['total_nominal_80pct_interval_coverage']:.1f}%**.",
            "",
            "### Reliability of the model-selected favorite", "",
            "| Predicted favorite P(win) | Games | Mean predicted | Actual win rate | 95% Wilson interval | Calibration gap |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for b in item["reliability"]:
            pretty = (f"{b['predicted_favorite_probability_range'][0]:.0%}–"
                      f"{b['predicted_favorite_probability_range'][1]:.0%}")
            m = f"{b['mean_predicted_probability']:.1%}" if b["games"] else "n/a"
            obs = f"{b['observed_favorite_win_rate']:.1%}" if b["games"] else "n/a"
            ci = str(b["wilson95_for_win_rate"])
            gap = f"{b['observed_minus_expected_pp']:+.1f} pp" if b["games"] else "n/a"
            out.append(f"| {pretty} | {b['games']} | {m} | {obs} | {ci} | {gap} |")
        out += ["", "### Model stress by game type", "",
                "| Game category | Games | Model Brier | Elo Brier | Model win % | Margin MAE | Total MAE |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for group, b in item["subgroups"].items():
            if not b["games"]:
                continue
            out.append(f"| {group.replace('_',' ')} | {b['games']} | {b['moneyline_brier']:.5f} | "
                       f"{b['elo_brier']:.5f} | {100*b['moneyline_accuracy']:.1f}% | "
                       f"{b['margin_mae']:.2f} | {b['total_mae']:.2f} |")
        boot = item["blocked_brier_difference"]
        if boot:
            out += ["", f"Daily block bootstrap Brier difference (model − Elo), "
                          f"95% diagnostic interval: {boot['95pct_interval_model_minus_elo']}. "
                          "**Post-selection, not independent confirmation.**", ""]
    out += ["---", "",
            "A better-looking high-confidence bucket alone does not prove value: "
            "the model must outperform real available market prices on genuine "
            "pre-tip archived forecasts and prices.", ""]
    return "\n".join(out)


def save(audit, path, markdown):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    Path(markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown).write_text(format_markdown(audit))

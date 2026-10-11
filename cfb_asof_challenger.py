#!/usr/bin/env python3
"""RESEARCH ONLY: train the CFB context challenger with actual pregame features.

The incumbent's historical build_context reads the TARGET game's boxscore
to derive absent teammates. This isolated runner substitutes one shared
strictly-prior-week builder in BOTH training and shadow serving. It NEVER
overwrites incumbent artifacts, changes a live pick, or makes an ROI claim.

2025 outcomes have already been examined and are a CALIBRATION/TRAINING
resource, not a fresh blind holdout. Genuine forward 2026 first-seen
forecast comparisons are required before promotion.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cfb_context_v2 as legacy
from cfb_context_asof_parity_audit import PriorWeekUsage

STATUS = "CFB_ASOF_CHALLENGER_RESEARCH_ONLY_NO_PROMOTION"
DEFAULT_OUTPUT = legacy.REPO / "cfb_models" / "cfb_asof_challenger_shadow_work"


class StrictAsOf:
    """Single historical/serving API: no target/future-week participation."""

    def __init__(self, con):
        self.con = con
        self.usage = PriorWeekUsage(con)
        self.ratings = legacy.build_ratings(con)
        self._future_ratings = {}

    def ratings_before(self, season, week):
        """Ratings fitted ONLY to seasons/weeks before the target.

        build_ratings() emits a key only for weeks containing FINAL games,
        so an upcoming week is generally absent from its returned as-of
        map. Reconstruct the rating for that week from a filtered result
        history rather than using an unbounded end-of-season rating.
        """
        key = (season, week)
        if key in self._future_ratings:
            return self._future_ratings[key]
        replay = sqlite3.connect(":memory:")
        try:
            replay.execute("""CREATE TABLE games (
                season INTEGER, week INTEGER, home_team TEXT, away_team TEXT,
                home_points REAL, away_points REAL, neutral_site INTEGER)""")
            played = self.con.execute("""
                SELECT season, week, home_team, away_team, home_points,
                       away_points, neutral_site
                FROM games
                WHERE home_points IS NOT NULL
                  AND (season < ? OR (season = ? AND week < ?))
                ORDER BY season, week
            """, (season, season, week)).fetchall()
            replay.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?)", played)
            legacy.build_ratings(replay)
            completed = legacy.build_ratings.latest
            if season in completed:
                result = dict(completed[season])
            else:
                result = {team: 0.5 * score for team, score
                          in completed.get(season - 1, {}).items()}
            self._future_ratings[key] = result
            return result
        finally:
            replay.close()

    def features(self, base, season, week, team, opponent, player_id):
        """Same function must supply training, validation and shadow features."""
        f = dict(base)
        role = self.usage.features(season, week, team, player_id)
        if player_id is None:
            # A team-level moneyline does not have a player's usage shares.
            for name in ("carry_share_last3", "rec_share_last3", "pass_share_last3"):
                role[name] = None
        f.update(role)  # Includes UNKNOWN, not fabricated zero, for absences.
        own = self.ratings.get((season, team, week))
        opp = self.ratings.get((season, opponent, week))
        if own is None or opp is None:
            prior = self.ratings_before(season, week)
            if own is None:
                own = prior.get(team, 0.0)
            if opp is None:
                opp = prior.get(opponent, 0.0)
        f.update({"team_rating": own, "opp_rating": opp,
                  "rating_diff": own - opp})
        return f


def _inject_strict_features(base, season, pid, week, team, opp, ctx, ratings):
    """Replacement for cfb_context_v2.add_context, with the SAME call ABI."""
    return ctx.features(base, season, week, team, opp, pid)


def guarded_output_dir(out_dir):
    out = Path(out_dir).resolve()
    original = legacy.MODEL_DIR.resolve()
    model_root = (legacy.REPO / "cfb_models").resolve()
    docs_root = (legacy.REPO / "docs").resolve()
    if out in (original, model_root, docs_root, legacy.REPO.resolve()):
        raise ValueError("refusing to overwrite production/model root")
    if original in out.parents or docs_root in out.parents:
        raise ValueError("research artifacts cannot target incumbent or published docs")
    return out


def train_shadow(db, out_dir=DEFAULT_OUTPUT):
    """Reuse the legacy experiment engine but strictly replace context inputs.

    Train: 2019-23; early-stop validation: 2024; 2025 retrospective
    comparison AND Platt calibration (already consulted; no holdout claim).
    Neither the normal serving builder nor its model directory is changed.
    """
    db = Path(db).resolve()
    if not db.is_file():
        raise FileNotFoundError(db)
    out = guarded_output_dir(out_dir)
    original = (legacy.MODEL_DIR, legacy.build_context, legacy.add_context)
    out.mkdir(parents=True, exist_ok=True)
    try:
        legacy.MODEL_DIR = out
        legacy.build_context = StrictAsOf
        legacy.add_context = _inject_strict_features
        # Old raw/context comparison on 2025 is merely diagnostic, not proof.
        legacy.run(str(db), eval_seasons=(2025,))
        legacy.finalize(str(db), calib_season=2025)
    finally:
        legacy.MODEL_DIR, legacy.build_context, legacy.add_context = original

    manifest = {
        "status": STATUS,
        "feature_contract": "cfb_asof_challenger.StrictAsOf.features",
        "absence_features_without_signed_pregame_source": None,
        "training_seasons": [2019, 2020, 2021, 2022, 2023],
        "early_stopping_season": 2024,
        "calibration_season_already_consulted": 2025,
        "retrospective_evaluation_season_already_consulted": 2025,
        "production_models_modified": False,
        "predictions_published": False,
        "sportsbook_lines_certified": False,
        "blind_2026_forward_evidence_required": True,
        "limitations": [
            "Replay eligible-player population may itself reflect postgame participation; as-of roster/population certification is separate.",
            "The original core feature engine has not yet been fully re-audited here.",
            "Research yardage thresholds are not verified sportsbook lines.",
            "No official picks or model promotion based on consulted 2025/2026 results."
        ],
    }
    (out / "research_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class ShadowScorer:
    """Score explicit, externally supplied PRE-game candidates, not live picks."""

    def __init__(self, con, model_dir=DEFAULT_OUTPUT):
        import xgboost as xgb
        self.strict = StrictAsOf(con)
        self.xgb = xgb
        self.root = guarded_output_dir(model_dir)
        manifest = json.loads((self.root / "research_manifest.json").read_text())
        if manifest.get("status") != STATUS:
            raise ValueError("uncertified shadow candidate artifacts")
        self.meta = json.loads((self.root / "production.json").read_text())
        self.models, self.columns = {}, {}
        for mkt in self.meta:
            booster = xgb.Booster()
            booster.load_model(str(self.root / f"cfb_{mkt}_context.json"))
            self.models[mkt] = booster
            self.columns[mkt] = json.loads(
                (self.root / f"cfb_{mkt}_context_columns.json").read_text())

    def probability(self, market, base, season, week, team, opponent, player_id):
        import numpy as np
        feat = self.strict.features(base, season, week, team, opponent, player_id)
        cols = self.columns[market]
        b = self.models[market]
        m = self.meta[market]
        raw = b.predict(self.xgb.DMatrix(legacy.mat([feat], cols),
                                         feature_names=cols),
                        iteration_range=(0, m["best_iteration"] + 1))
        x = legacy._logit(legacy.to_prob(market, raw))
        probability = 1 / (1 + np.exp(-(m["platt_a"] * x + m["platt_b"])))
        return {"market": market, "model_probability": float(probability[0]),
                "research_only": True, "production_pick": False,
                "feature_contract": "StrictAsOf.features"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    print(json.dumps(train_shadow(args.db, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()

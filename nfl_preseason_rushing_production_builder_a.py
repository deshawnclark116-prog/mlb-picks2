#!/usr/bin/env python3
"""
NFL_PRESEASON_RUSHING_PRODUCTION_BUILDER_A

Final production model for the market validated in
nfl_preseason_to_regular_season_gate_a.py (rushing_yards, AUC 0.7027 on
the 2025 holdout; receiving_yards did not clear the bar and is not built
here). Retrains on ALL available years (2022-2025 combined, train+
holdout) rather than the dev-only split used for validation -- the
holdout's job (proving this generalizes) is already done; a shipped
model should use every real data point available.

Same features, same join methodology (normalized name + position +
season, nflverse regular season joined to local ESPN preseason data) as
the validated gate script. Read that script's docstring for the full
methodology notes (data sources, the join approximation, why this is a
different question from the earlier failed preseason-predicts-preseason
attempt).

Writes nfl_models/nfl_preseason_rushing_yards.json +
nfl_models/nfl_preseason_rushing_yards_columns.json for
nfl_serving_builder_a.py to load.

Run:
    python -u nfl_preseason_rushing_production_builder_a.py
"""
import csv
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

CSV_DIR = Path("/data/nflverse_csv")
PRESEASON_DB = Path("nfl_models/nfl_preseason.sqlite")
MODEL_DIR = Path("nfl_models")
ALL_SEASONS = [2022, 2023, 2024, 2025]
LINE = 49.5
NAN = float("nan")
FEATURE_COLS = ["preseason_avg_yards", "preseason_games_played", "preseason_avg_rate"]


def norm_name(name):
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def load_regular_weeks123(seasons):
    rows = []
    for season in seasons:
        path = CSV_DIR / f"stats_player_week_{season}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("season_type") != "REG":
                    continue
                try:
                    week = int(r["week"])
                except Exception:
                    continue
                if week > 3:
                    continue
                if r.get("position") != "RB":
                    continue
                rows.append(r)
    return rows


def load_preseason_by_key(seasons):
    con = sqlite3.connect(f"file:{PRESEASON_DB}?mode=ro", uri=True)
    seasons_sql = ",".join(str(s) for s in seasons)
    rows = con.execute(f"""
        SELECT player_name, position, season, carries, rushing_yards
        FROM player_games WHERE season IN ({seasons_sql}) AND position = 'RB'
    """).fetchall()
    con.close()
    by_key = {}
    for name, pos, season, carries, ry in rows:
        key = (norm_name(name), pos, season)
        by_key.setdefault(key, []).append({"carries": carries or 0, "rushing_yards": ry or 0})
    return by_key


def build_rows(reg_rows, preseason_by_key):
    out = []
    for r in reg_rows:
        try:
            season = int(r["season"])
        except Exception:
            continue
        name = r.get("player_display_name") or r.get("player_name")
        key = (norm_name(name), "RB", season)
        games = preseason_by_key.get(key)
        if games:
            n = len(games)
            yards = [g["rushing_yards"] for g in games]
            rate = [g["carries"] for g in games]
            feat = {"preseason_avg_yards": sum(yards) / n,
                    "preseason_games_played": n,
                    "preseason_avg_rate": sum(rate) / n}
        else:
            feat = {c: NAN for c in FEATURE_COLS}
        actual = _f(r.get("rushing_yards"))
        out.append({**feat, "over_line": 1 if actual >= (LINE + 0.5) else 0})
    return out


def mat(rows, xgb):
    X = np.array([[r.get(c, NAN) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def main():
    import xgboost as xgb
    print("NFL_PRESEASON_RUSHING_PRODUCTION_BUILDER_A\n===========================================")

    reg_rows = load_regular_weeks123(ALL_SEASONS)
    preseason_by_key = load_preseason_by_key(ALL_SEASONS)
    rows = build_rows(reg_rows, preseason_by_key)
    matched = sum(1 for r in rows if not np.isnan(r["preseason_avg_yards"]))
    print(f"total rows (all {ALL_SEASONS}, weeks 1-3, RB): {len(rows)} ({matched} matched to preseason, "
          f"{matched/len(rows)*100:.1f}%)")

    n = len(rows)
    cut = int(n * 0.85)
    tr, va = rows[:cut], rows[cut:]
    print(f"train={len(tr)}  internal val={len(va)} (early stopping only, not a real holdout -- "
          f"generalization already proven in nfl_preseason_to_regular_season_gate_a.py)")

    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": 13}
    bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    print(f"best_iteration={bst.best_iteration}")

    MODEL_DIR.mkdir(exist_ok=True)
    bst.save_model(str(MODEL_DIR / "nfl_preseason_rushing_yards.json"))
    (MODEL_DIR / "nfl_preseason_rushing_yards_columns.json").write_text(json.dumps(FEATURE_COLS))
    manifest = {
        "script": "NFL_PRESEASON_RUSHING_PRODUCTION_BUILDER_A",
        "trained_on_seasons": ALL_SEASONS, "line": LINE,
        "feature_cols": FEATURE_COLS, "best_iteration": bst.best_iteration,
        "validated_holdout_auc": 0.7027,
        "validation_script": "nfl_preseason_to_regular_season_gate_a.py",
    }
    (MODEL_DIR / "nfl_preseason_rushing_yards_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nsaved: {MODEL_DIR / 'nfl_preseason_rushing_yards.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

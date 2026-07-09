#!/usr/bin/env python3
"""
HR_PA_STRUCTURAL_ORDINAL_FINAL_A

One final structural test of the expected-PA branch.

Locked doctrine
---------------
- Pitcher branch remains frozen.
- Bayesian pitch-zone overlap remains blocked.
- PA-3 HR integration remains blocked until this script passes PA-1 and PA-2.
- No sportsbook-implied totals.
- No postgame predictors.
- Strictly pregame-known structural exposure features only.

Structural upgrades
-------------------
1. Home/away flag from MLB game metadata.
2. Venue/park ID from MLB game metadata.
3. Strictly lagged team offense features built from PRIOR dates only.
4. Pregame lineup-quality features built from same-day confirmed starters using
   each batter's already-lagged pregame batter features.
5. Pregame opposing-pitcher handedness and cleaned pitcher-quality summaries.
6. Ordinal/cumulative PA model:
      P(PA >= 2)
      P(PA >= 3)
      P(PA >= 4)
      P(PA >= 5)
      P(PA >= 6)
   Then derive the full PMF:
      P(PA=1), ..., P(PA=5), P(PA>=6)

Data split
----------
Train: 2025
Forward holdout: 2026

Final gate
----------
PA-1 PASS requires:
- multiclass logloss improves by at least 0.005 vs lineup-spot baseline
- bucket accuracy improves by at least 0.005 absolute
- expected-PA MAE is not worse by more than 0.01

PA-2 PASS requires:
- P(PA>=5) mean prediction within 1 percentage point of actual
- P(PA>=5) Brier no worse than lineup-spot baseline
- P(PA>=5) logloss no worse than lineup-spot baseline
- full-distribution logloss no worse than lineup-spot baseline

If either gate fails:
    Freeze PA branch. No more PA feature cycles.
    Pivot to batter-side damage calibration / direct base-rate refinement.

Run
---
python -u hr_pa_structural_ordinal_final_a.py 2>&1 | tee /data/hr_model/hr_pa_structural_ordinal_final_a.log

Outputs
-------
/data/hr_model/hr_pa_structural_ordinal_final_a_results.json
/data/hr_model/mlb_game_metadata_cache.json
/data/hr_model/mlb_team_alias_cache.json

Paste back
----------
METADATA COVERAGE
STRUCTURAL FEATURE SET
PA-1 FINAL SUMMARY
PA-1 FINAL DELTAS
PA-2 FINAL DISTRIBUTION CALIBRATION
PA>=5 FINAL RELIABILITY
STRICT FINAL GATE READ
"""

import argparse
import gc
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"

ACTUAL_PA_CANDIDATES = [
    "actual_pa",
    "pa",
    "plate_appearances",
    "plate_appearance_count",
    "game_pa",
]

LINEUP_CANDIDATES = [
    "lineup_spot",
    "batting_order",
    "batting_order_position",
]

TEAM_CANDIDATES = [
    "team_id",
    "batting_team_id",
    "team",
]

BATTER_LINEUP_QUALITY_CANDIDATES = [
    "batter_hr_per_air_60d",
    "batter_barrel_rate_30d",
    "batter_max_ev_60d",
    "batter_fly_ball_rate_30d",
]

PITCHER_HAND_CANDIDATES = [
    "p_throws",
    "pitcher_hand",
    "opp_pitcher_hand",
]

PITCHER_QUALITY_CANDIDATES = [
    "opp_hand_k_per_pa_365d_shrunk",
    "opp_hand_gb_per_bbe_365d_shrunk",
    "opp_hand_hr_per_pa_365d_shrunk",
    "opp_hand_barrel_per_bbe_365d_shrunk",
]

TABLES = [
    "batter_games",
    "batter_game_features",
    "pitcher_game_features",
    "league_env_lag_features",
    "pitcher_hand_features_a",
]


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def require_imports():
    try:
        import numpy as np
        import pandas as pd
        import requests
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        return (
            np, pd, requests, ColumnTransformer, SimpleImputer, LogisticRegression,
            accuracy_score, brier_score_loss, log_loss, mean_absolute_error,
            mean_squared_error, Pipeline, OneHotEncoder, StandardScaler
        )
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install pandas numpy requests scikit-learn.\n"
            + repr(e)
        )


def table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def table_cols(conn, table):
    if not table_exists(conn, table):
        return []
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def alias_for_table(table):
    return {
        "batter_games": "bg",
        "batter_game_features": "bf",
        "pitcher_game_features": "pf",
        "league_env_lag_features": "le",
        "pitcher_hand_features_a": "ph",
    }[table]


def find_source(conn, candidates, preferred_tables=None):
    tables = preferred_tables or TABLES
    for table in tables:
        cols = set(table_cols(conn, table))
        for c in candidates:
            if c in cols:
                return table, c
    return None, None


def source_expr(conn, out_alias, candidates, preferred_tables=None):
    table, col = find_source(conn, candidates, preferred_tables)
    if not table:
        return f'NULL AS "{out_alias}"', None
    return (
        f'{alias_for_table(table)}."{col}" AS "{out_alias}"',
        f"{table}.{col}",
    )


def normalize_team_key(value):
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def parse_date(value):
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def daterange_chunks(start_date, end_date, chunk_days=30):
    start = parse_date(start_date)
    end = parse_date(end_date)
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def load_json(path, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_json(requests, url, params, retries=3, timeout=45):
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            print(f"request attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(min(2 * attempt, 6))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def build_team_alias_cache(requests, path):
    cache = load_json(path, {})
    if cache:
        return cache

    alias_to_id = {}
    id_to_name = {}

    for season in [2025, 2026]:
        data = fetch_json(
            requests,
            MLB_TEAMS_URL,
            {"sportId": 1, "season": season},
        )
        for team in data.get("teams", []):
            tid = int(team["id"])
            id_to_name[str(tid)] = team.get("name")

            aliases = {
                str(tid),
                team.get("name"),
                team.get("teamName"),
                team.get("clubName"),
                team.get("shortName"),
                team.get("locationName"),
                team.get("abbreviation"),
                team.get("fileCode"),
            }
            for a in aliases:
                key = normalize_team_key(a)
                if key:
                    alias_to_id[key] = tid

    cache = {
        "alias_to_id": alias_to_id,
        "id_to_name": id_to_name,
    }
    save_json(path, cache)
    return cache


def update_game_metadata_cache(
    requests,
    cache_path,
    needed_game_ids,
    start_date,
    end_date,
):
    cache = load_json(cache_path, {})
    missing = {str(g) for g in needed_game_ids if str(g) not in cache}

    if not missing:
        print("game metadata cache already complete for requested IDs", flush=True)
        return cache

    print(f"metadata missing games before fetch: {len(missing):,}", flush=True)

    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, 30):
        print(
            f"fetching MLB schedule metadata {chunk_start} to {chunk_end}...",
            flush=True,
        )
        data = fetch_json(
            requests,
            MLB_SCHEDULE_URL,
            {
                "sportId": 1,
                "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(),
                "hydrate": "venue",
            },
        )

        for d in data.get("dates", []):
            for game in d.get("games", []):
                game_pk = str(game.get("gamePk"))
                if not game_pk:
                    continue

                teams = game.get("teams", {})
                venue = game.get("venue") or {}
                cache[game_pk] = {
                    "game_date": str(game.get("officialDate") or d.get("date")),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "home_team_id": (
                        teams.get("home", {}).get("team", {}).get("id")
                    ),
                    "away_team_id": (
                        teams.get("away", {}).get("team", {}).get("id")
                    ),
                }

        save_json(cache_path, cache)
        missing = {g for g in missing if g not in cache}
        print(f"remaining metadata misses: {len(missing):,}", flush=True)

        if not missing:
            break

    return cache


def build_query(conn):
    pa_table, pa_col = find_source(conn, ACTUAL_PA_CANDIDATES)
    lineup_table, lineup_col = find_source(conn, LINEUP_CANDIDATES)
    team_table, team_col = find_source(conn, TEAM_CANDIDATES)

    if not pa_table:
        raise SystemExit("Could not find actual PA target.")
    if not lineup_table:
        raise SystemExit("Could not find lineup spot.")
    if not team_table:
        raise SystemExit("Could not find batting team field.")

    select = [
        "bg.game_date AS game_date",
        "CAST(bg.game_id AS TEXT) AS game_id",
        "bg.batter_id AS batter_id",
        f'{alias_for_table(pa_table)}."{pa_col}" AS actual_pa',
        f'{alias_for_table(lineup_table)}."{lineup_col}" AS lineup_spot',
        f'{alias_for_table(team_table)}."{team_col}" AS team_raw',
        "bg.actual_hr AS actual_hr",
    ]

    provenance = {
        "actual_pa": f"{pa_table}.{pa_col}",
        "lineup_spot": f"{lineup_table}.{lineup_col}",
        "team_raw": f"{team_table}.{team_col}",
    }

    # Pregame batter features for same-day confirmed-lineup quality.
    for c in BATTER_LINEUP_QUALITY_CANDIDATES:
        expr, src = source_expr(
            conn,
            c,
            [c],
            ["batter_game_features", "batter_games"],
        )
        select.append(expr)
        provenance[c] = src

    # Opposing pitcher hand.
    expr, src = source_expr(
        conn,
        "pitcher_hand",
        PITCHER_HAND_CANDIDATES,
        ["pitcher_hand_features_a", "batter_games", "pitcher_game_features"],
    )
    select.append(expr)
    provenance["pitcher_hand"] = src

    # Cleaned opposing pitcher quality summaries.
    for c in PITCHER_QUALITY_CANDIDATES:
        expr, src = source_expr(
            conn,
            c,
            [c],
            ["pitcher_hand_features_a", "pitcher_game_features"],
        )
        select.append(expr)
        provenance[c] = src

    joins = [
        """
        LEFT JOIN batter_game_features bf
          ON bg.game_id = bf.game_id
         AND bg.batter_id = bf.batter_id
        """,
        """
        LEFT JOIN pitcher_game_features pf
          ON bg.game_id = pf.game_id
         AND bg.batter_id = pf.batter_id
        """,
        """
        LEFT JOIN league_env_lag_features le
          ON bg.game_date = le.game_date
        """,
    ]

    if table_exists(conn, "pitcher_hand_features_a"):
        joins.append("""
        LEFT JOIN pitcher_hand_features_a ph
          ON CAST(bg.game_id AS TEXT)=ph.game_id
         AND bg.batter_id=ph.batter_id
        """)

    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    {" ".join(joins)}
    WHERE bg.game_date IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.batter_id
    """

    return sql, provenance


def resolve_team_ids(df, alias_cache, pd):
    alias_to_id = {
        str(k): int(v)
        for k, v in alias_cache.get("alias_to_id", {}).items()
    }

    def one(v):
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s.isdigit():
                return int(s)
        except Exception:
            pass
        return alias_to_id.get(normalize_team_key(v))

    df["team_id"] = df["team_raw"].map(one)
    return df


def add_metadata(df, metadata, pd):
    meta_rows = []
    for game_id, m in metadata.items():
        meta_rows.append({
            "game_id": str(game_id),
            "venue_id": m.get("venue_id"),
            "home_team_id": m.get("home_team_id"),
            "away_team_id": m.get("away_team_id"),
        })

    meta_df = pd.DataFrame(meta_rows)
    if meta_df.empty:
        raise SystemExit("No game metadata available.")

    for c in ["venue_id", "home_team_id", "away_team_id"]:
        meta_df[c] = pd.to_numeric(meta_df[c], errors="coerce")

    out = df.merge(meta_df, on="game_id", how="left")
    out["is_home"] = (
        pd.to_numeric(out["team_id"], errors="coerce")
        == pd.to_numeric(out["home_team_id"], errors="coerce")
    ).astype("float32")
    return out


def add_lineup_quality(df, pd):
    available = [
        c for c in BATTER_LINEUP_QUALITY_CANDIDATES
        if c in df.columns and df[c].notna().mean() >= 0.20
    ]

    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if not available:
        print(
            "WARNING: no batter pregame quality features available for lineup quality.",
            flush=True,
        )
        return df, []

    group_cols = ["game_id", "team_id"]
    agg = (
        df.groupby(group_cols, dropna=False)[available]
        .mean()
        .reset_index()
    )

    rename = {
        c: f"lineup_mean_{c}"
        for c in available
    }
    agg = agg.rename(columns=rename)
    df = df.merge(agg, on=group_cols, how="left")
    return df, list(rename.values())


def add_lagged_team_offense(df, pd):
    # One team-game row, then collapse to date so no same-date result can leak into
    # another same-date game (important for doubleheaders).
    game_team = (
        df.groupby(["team_id", "game_date", "game_id"], dropna=False)
        .agg(
            team_total_pa=("actual_pa", "sum"),
            team_hr=("actual_hr", "sum"),
            starter_rows=("batter_id", "count"),
        )
        .reset_index()
    )

    daily = (
        game_team.groupby(["team_id", "game_date"], dropna=False)
        .agg(
            day_total_pa=("team_total_pa", "sum"),
            day_hr=("team_hr", "sum"),
            day_games=("game_id", "nunique"),
            day_starter_rows=("starter_rows", "sum"),
        )
        .reset_index()
        .sort_values(["team_id", "game_date"])
    )

    frames = []
    for team_id, g in daily.groupby("team_id", dropna=False):
        g = g.sort_values("game_date").copy()

        # Shift first: current date never contributes to its own pregame features.
        prior_pa = g["day_total_pa"].shift(1)
        prior_hr = g["day_hr"].shift(1)
        prior_games = g["day_games"].shift(1)
        prior_starters = g["day_starter_rows"].shift(1)

        roll_pa = prior_pa.rolling(20, min_periods=5).sum()
        roll_hr = prior_hr.rolling(20, min_periods=5).sum()
        roll_games = prior_games.rolling(20, min_periods=5).sum()
        roll_starters = prior_starters.rolling(20, min_periods=5).sum()

        g["team_pa_per_game_20d_lag"] = roll_pa / roll_games
        g["team_pa_per_starter_20d_lag"] = roll_pa / roll_starters
        g["team_hr_per_pa_20d_lag"] = roll_hr / roll_pa
        frames.append(g)

    lag = pd.concat(frames, ignore_index=True)

    keep = [
        "team_id",
        "game_date",
        "team_pa_per_game_20d_lag",
        "team_pa_per_starter_20d_lag",
        "team_hr_per_pa_20d_lag",
    ]
    return df.merge(lag[keep], on=["team_id", "game_date"], how="left")


def bucket_pa(x):
    try:
        n = int(round(float(x)))
    except Exception:
        return None
    if n <= 0:
        return None
    return min(n, 6)


def empirical_baseline(train, test, np):
    global_counts = (
        train["pa_bucket"]
        .value_counts(normalize=True)
        .reindex([1, 2, 3, 4, 5, 6], fill_value=0.0)
        .to_numpy(dtype="float64")
    )

    by_spot = {}
    for spot, g in train.groupby("lineup_spot"):
        if len(g) < 50:
            continue
        by_spot[int(spot)] = (
            g["pa_bucket"]
            .value_counts(normalize=True)
            .reindex([1, 2, 3, 4, 5, 6], fill_value=0.0)
            .to_numpy(dtype="float64")
        )

    return np.vstack([
        by_spot.get(int(spot), global_counts)
        for spot in test["lineup_spot"].astype(int)
    ])


def cumulative_probs_to_pmf(np, cumulative):
    # cumulative columns correspond to P(PA >= 2), ..., P(PA >= 6)
    p = np.asarray(cumulative, dtype="float64").copy()

    # Enforce monotone non-increasing cumulative probabilities row-wise.
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j - 1])

    p = np.clip(p, 0.0, 1.0)

    pmf = np.zeros((len(p), 6), dtype="float64")
    pmf[:, 0] = 1.0 - p[:, 0]
    pmf[:, 1] = p[:, 0] - p[:, 1]
    pmf[:, 2] = p[:, 1] - p[:, 2]
    pmf[:, 3] = p[:, 2] - p[:, 3]
    pmf[:, 4] = p[:, 3] - p[:, 4]
    pmf[:, 5] = p[:, 4]

    pmf = np.clip(pmf, 0.0, 1.0)
    row_sum = pmf.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    pmf /= row_sum
    return pmf


def expected_pa_from_probs(np, probs):
    values = np.array([1, 2, 3, 4, 5, 6], dtype="float64")
    return np.asarray(probs).dot(values)


def full_metrics(
    np,
    y_true,
    probs,
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    pred_bucket = np.argmax(p, axis=1) + 1
    exp_pa = expected_pa_from_probs(np, p)

    return {
        "rows": int(len(y)),
        "multiclass_logloss": float(
            log_loss(y, p, labels=[1, 2, 3, 4, 5, 6])
        ),
        "bucket_accuracy": float(accuracy_score(y, pred_bucket)),
        "expected_pa_mae": float(mean_absolute_error(y, exp_pa)),
        "expected_pa_rmse": float(
            math.sqrt(mean_squared_error(y, exp_pa))
        ),
        "mean_actual_pa_bucket": float(y.mean()),
        "mean_predicted_expected_pa": float(exp_pa.mean()),
    }


def tail_metrics(np, y_true, probs, brier_score_loss, log_loss):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    p_ge5 = p[:, 4] + p[:, 5]
    y_ge5 = (y >= 5).astype("int8")

    return {
        "p_ge5_brier": float(brier_score_loss(y_ge5, p_ge5)),
        "p_ge5_logloss": float(
            log_loss(y_ge5, p_ge5, labels=[0, 1])
        ),
        "p_ge5_actual_rate": float(y_ge5.mean()),
        "p_ge5_mean_pred": float(p_ge5.mean()),
    }


def reliability_rows(np, y_true, probs, bins=10):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    p_ge5 = p[:, 4] + p[:, 5]
    y_ge5 = (y >= 5).astype("int8")

    # Quantile bins are more informative here because predicted tail probabilities
    # occupy a relatively narrow range.
    try:
        edges = np.unique(
            np.quantile(p_ge5, np.linspace(0, 1, bins + 1))
        )
    except Exception:
        edges = np.linspace(0, 1, bins + 1)

    rows = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (p_ge5 >= lo) & (p_ge5 <= hi)
        else:
            mask = (p_ge5 >= lo) & (p_ge5 < hi)

        if mask.sum() == 0:
            continue

        rows.append({
            "bin": i + 1,
            "n": int(mask.sum()),
            "min_pred": float(lo),
            "max_pred": float(hi),
            "mean_pred": float(p_ge5[mask].mean()),
            "actual_rate": float(y_ge5[mask].mean()),
            "gap": float(
                y_ge5[mask].mean() - p_ge5[mask].mean()
            ),
        })
    return rows


def class_calibration_rows(np, y_true, probs):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    rows = []

    for cls in [1, 2, 3, 4, 5, 6]:
        actual = float((y == cls).mean())
        pred = float(p[:, cls - 1].mean())
        rows.append({
            "pa_bucket": "6+" if cls == 6 else str(cls),
            "actual_rate": actual,
            "mean_pred": pred,
            "gap": actual - pred,
        })

    return rows


def print_summary(label, metrics):
    print(
        f"{label}: "
        f"logloss={metrics['multiclass_logloss']:.8f} "
        f"bucket_acc={metrics['bucket_accuracy']:.5f} "
        f"expected_pa_mae={metrics['expected_pa_mae']:.5f} "
        f"expected_pa_rmse={metrics['expected_pa_rmse']:.5f} "
        f"mean_actual={metrics['mean_actual_pa_bucket']:.5f} "
        f"mean_pred={metrics['mean_predicted_expected_pa']:.5f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument(
        "--metadata-cache",
        default="/data/hr_model/mlb_game_metadata_cache.json",
    )
    ap.add_argument(
        "--team-alias-cache",
        default="/data/hr_model/mlb_team_alias_cache.json",
    )
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_pa_structural_ordinal_final_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, requests, ColumnTransformer, SimpleImputer, LogisticRegression,
        accuracy_score, brier_score_loss, log_loss, mean_absolute_error,
        mean_squared_error, Pipeline, OneHotEncoder, StandardScaler
    ) = require_imports()

    print("HR_PA_STRUCTURAL_ORDINAL_FINAL_A", flush=True)
    print("================================", flush=True)
    print(f"db: {args.db}", flush=True)

    conn = sqlite3.connect(args.db)
    sql, provenance = build_query(conn)

    print("\nSCHEMA RESOLUTION", flush=True)
    print("-----------------", flush=True)
    for k, v in provenance.items():
        print(f"{k}: {v}", flush=True)

    print("\nloading batter-game rows...", flush=True)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_pa"] = pd.to_numeric(df["actual_pa"], errors="coerce")
    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce")
    df["actual_hr"] = pd.to_numeric(df["actual_hr"], errors="coerce").fillna(0)

    df = df[
        df["game_date"].notna()
        & df["actual_pa"].notna()
        & df["lineup_spot"].between(1, 9)
    ].copy()

    team_alias_cache = build_team_alias_cache(
        requests,
        args.team_alias_cache,
    )
    df = resolve_team_ids(df, team_alias_cache, pd)

    game_ids = sorted(df["game_id"].astype(str).unique())
    start_date = df["game_date"].min().date()
    end_date = df["game_date"].max().date()

    metadata = update_game_metadata_cache(
        requests,
        args.metadata_cache,
        game_ids,
        start_date,
        end_date,
    )
    df = add_metadata(df, metadata, pd)

    print("\nMETADATA COVERAGE", flush=True)
    print("-----------------", flush=True)
    print(
        f"game_rows={len(df):,} "
        f"venue_coverage={df['venue_id'].notna().mean():.4f} "
        f"home_team_coverage={df['home_team_id'].notna().mean():.4f} "
        f"team_id_coverage={df['team_id'].notna().mean():.4f}",
        flush=True,
    )
    print(
        f"is_home_resolved={df['home_team_id'].notna().mean():.4f}",
        flush=True,
    )

    if df["venue_id"].notna().mean() < 0.95:
        raise SystemExit(
            "Metadata venue coverage below 95%; stop before final gate."
        )
    if df["team_id"].notna().mean() < 0.95:
        raise SystemExit(
            "Batting-team ID resolution below 95%; stop before final gate."
        )

    df, lineup_quality_cols = add_lineup_quality(df, pd)
    df = add_lagged_team_offense(df, pd)

    df["pa_bucket"] = df["actual_pa"].map(bucket_pa)
    df = df[df["pa_bucket"].notna()].copy()
    df["pa_bucket"] = df["pa_bucket"].astype("int8")
    df["lineup_spot"] = df["lineup_spot"].astype("int8")

    train = df[df["game_date"].dt.year == 2025].copy()
    test = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    candidate_numeric = [
        "is_home",
        "team_pa_per_game_20d_lag",
        "team_pa_per_starter_20d_lag",
        "team_hr_per_pa_20d_lag",
    ] + lineup_quality_cols + [
        c for c in PITCHER_QUALITY_CANDIDATES
        if c in train.columns
    ]

    candidate_categorical = [
        "lineup_spot",
        "team_id",
        "venue_id",
        "pitcher_hand",
    ]

    numeric = []
    for c in candidate_numeric:
        if c not in train.columns:
            continue
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")
        if train[c].notna().mean() >= 0.20 and train[c].nunique(dropna=True) >= 2:
            numeric.append(c)

    categorical = []
    for c in candidate_categorical:
        if c not in train.columns:
            continue
        if train[c].notna().mean() >= 0.20 and train[c].nunique(dropna=True) >= 2:
            categorical.append(c)

    print("\nSTRUCTURAL FEATURE SET", flush=True)
    print("----------------------", flush=True)
    print("categorical:", categorical, flush=True)
    print("numeric:", numeric, flush=True)
    print(
        "internal_projected_team_runs: DEFERRED_NOT_FORCED "
        "(no honest pregame run-projection target currently available)",
        flush=True,
    )

    usable = numeric + categorical
    if not usable:
        raise SystemExit("No usable structural PA features.")

    baseline_probs = empirical_baseline(train, test, np)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    pre = ColumnTransformer(transformers, remainder="drop")

    X_train = pre.fit_transform(train[usable])
    X_test = pre.transform(test[usable])

    y_train = train["pa_bucket"].to_numpy(dtype="int8", copy=False)
    y_test = test["pa_bucket"].to_numpy(dtype="int8", copy=False)

    cumulative_test = np.zeros((len(test), 5), dtype="float64")

    print("\nFITTING ORDINAL CUMULATIVE THRESHOLDS", flush=True)
    print("-------------------------------------", flush=True)

    for j, threshold in enumerate([2, 3, 4, 5, 6]):
        y_bin = (y_train >= threshold).astype("int8")
        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=0.10,
        )
        model.fit(X_train, y_bin)
        cumulative_test[:, j] = model.predict_proba(X_test)[:, 1]
        print(
            f"P(PA>={threshold}) fitted; "
            f"train_positive_rate={y_bin.mean():.5f}",
            flush=True,
        )
        del model, y_bin
        gc.collect()

    ordinal_probs = cumulative_probs_to_pmf(np, cumulative_test)

    baseline_metrics = full_metrics(
        np, y_test, baseline_probs,
        accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    )
    model_metrics = full_metrics(
        np, y_test, ordinal_probs,
        accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    )

    baseline_tail = tail_metrics(
        np, y_test, baseline_probs,
        brier_score_loss, log_loss
    )
    model_tail = tail_metrics(
        np, y_test, ordinal_probs,
        brier_score_loss, log_loss
    )

    print("\nPA-1 FINAL SUMMARY", flush=True)
    print("------------------", flush=True)
    print_summary("LINEUP_SPOT_BASELINE", baseline_metrics)
    print_summary("STRUCTURAL_ORDINAL_MODEL", model_metrics)

    logloss_delta = (
        model_metrics["multiclass_logloss"]
        - baseline_metrics["multiclass_logloss"]
    )
    acc_delta = (
        model_metrics["bucket_accuracy"]
        - baseline_metrics["bucket_accuracy"]
    )
    mae_delta = (
        model_metrics["expected_pa_mae"]
        - baseline_metrics["expected_pa_mae"]
    )
    rmse_delta = (
        model_metrics["expected_pa_rmse"]
        - baseline_metrics["expected_pa_rmse"]
    )

    print("\nPA-1 FINAL DELTAS", flush=True)
    print("-----------------", flush=True)
    print(f"logloss_delta={logloss_delta:+.8f}", flush=True)
    print(f"bucket_accuracy_delta={acc_delta:+.8f}", flush=True)
    print(f"expected_pa_mae_delta={mae_delta:+.8f}", flush=True)
    print(f"expected_pa_rmse_delta={rmse_delta:+.8f}", flush=True)

    class_cal = class_calibration_rows(np, y_test, ordinal_probs)

    print("\nPA-2 FINAL DISTRIBUTION CALIBRATION", flush=True)
    print("-----------------------------------", flush=True)
    for r in class_cal:
        print(
            f"PA={r['pa_bucket']} "
            f"actual={r['actual_rate']:.5f} "
            f"pred={r['mean_pred']:.5f} "
            f"gap={r['gap']:+.5f}",
            flush=True,
        )

    reliability = reliability_rows(np, y_test, ordinal_probs, bins=10)

    tail_brier_delta = (
        model_tail["p_ge5_brier"]
        - baseline_tail["p_ge5_brier"]
    )
    tail_logloss_delta = (
        model_tail["p_ge5_logloss"]
        - baseline_tail["p_ge5_logloss"]
    )
    tail_mean_gap = (
        model_tail["p_ge5_mean_pred"]
        - model_tail["p_ge5_actual_rate"]
    )

    print("\nPA>=5 FINAL RELIABILITY", flush=True)
    print("-----------------------", flush=True)
    print(
        f"baseline_tail_brier={baseline_tail['p_ge5_brier']:.8f} "
        f"model_tail_brier={model_tail['p_ge5_brier']:.8f} "
        f"delta={tail_brier_delta:+.8f}",
        flush=True,
    )
    print(
        f"baseline_tail_logloss={baseline_tail['p_ge5_logloss']:.8f} "
        f"model_tail_logloss={model_tail['p_ge5_logloss']:.8f} "
        f"delta={tail_logloss_delta:+.8f}",
        flush=True,
    )
    print(
        f"actual_ge5={model_tail['p_ge5_actual_rate']:.5f} "
        f"mean_pred_ge5={model_tail['p_ge5_mean_pred']:.5f} "
        f"gap={tail_mean_gap:+.5f}",
        flush=True,
    )

    for r in reliability:
        print(
            f"bin={r['bin']:02d} n={r['n']:5d} "
            f"pred={r['mean_pred']:.5f} "
            f"actual={r['actual_rate']:.5f} "
            f"gap={r['gap']:+.5f}",
            flush=True,
        )

    pa1_pass = (
        logloss_delta <= -0.005
        and acc_delta >= 0.005
        and mae_delta <= 0.01
    )

    pa2_pass = (
        abs(tail_mean_gap) <= 0.01
        and tail_brier_delta <= 0.0
        and tail_logloss_delta <= 0.0
        and logloss_delta <= 0.0
    )

    if pa1_pass and pa2_pass:
        verdict = "PA1_PA2_PASS_READY_FOR_PA3_HR_INTEGRATION"
        next_step = "Proceed to PA-3 HR integration."
    else:
        verdict = "FREEZE_PA_BRANCH_PIVOT_BATTER_DAMAGE_CALIBRATION"
        next_step = (
            "Freeze PA branch immediately. No further PA hyperparameter tuning "
            "or feature cycles. Pivot to batter-side damage calibration / "
            "direct base-rate refinement."
        )

    results = {
        "script": "HR_PA_STRUCTURAL_ORDINAL_FINAL_A",
        "db": args.db,
        "schema_resolution": provenance,
        "metadata_coverage": {
            "venue": float(train["venue_id"].notna().mean()),
            "team_id": float(train["team_id"].notna().mean()),
        },
        "feature_set": {
            "categorical": categorical,
            "numeric": numeric,
            "internal_projected_team_runs": "DEFERRED_NOT_FORCED",
        },
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "baseline_tail": baseline_tail,
        "model_tail": model_tail,
        "class_calibration": class_cal,
        "p_ge5_reliability": reliability,
        "deltas": {
            "logloss": logloss_delta,
            "bucket_accuracy": acc_delta,
            "expected_pa_mae": mae_delta,
            "expected_pa_rmse": rmse_delta,
            "p_ge5_brier": tail_brier_delta,
            "p_ge5_logloss": tail_logloss_delta,
            "p_ge5_mean_gap": tail_mean_gap,
        },
        "gate_thresholds": {
            "pa1_logloss_delta_max": -0.005,
            "pa1_bucket_accuracy_delta_min": 0.005,
            "pa1_mae_delta_max": 0.01,
            "pa2_abs_tail_mean_gap_max": 0.01,
            "pa2_tail_brier_delta_max": 0.0,
            "pa2_tail_logloss_delta_max": 0.0,
            "pa2_full_logloss_delta_max": 0.0,
        },
        "pa1_pass": pa1_pass,
        "pa2_pass": pa2_pass,
        "verdict": verdict,
        "next_step": next_step,
    }

    save_json(args.out_json, results)

    print("\nSTRICT FINAL GATE READ", flush=True)
    print("----------------------", flush=True)
    print(f"PA1_pass: {pa1_pass}", flush=True)
    print(f"PA2_pass: {pa2_pass}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"next_step: {next_step}", flush=True)
    print(
        "Pitcher branch remains frozen. Bayesian overlap remains blocked.",
        flush=True,
    )
    print(f"final JSON: {args.out_json}", flush=True)

    del X_train, X_test, cumulative_test, ordinal_probs
    del train, test
    gc.collect()


if __name__ == "__main__":
    main()

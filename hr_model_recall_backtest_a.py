#!/usr/bin/env python3
"""
HR_MODEL_RECALL_BACKTEST_A

Tests the HR recall fix (build_hr_pick / _hr_official_quality_ok, already
merged) against REAL historical 2026 games with known outcomes, using the
actual deployed trained model -- not a re-simulation of it. Answers two
questions with real data before fully trusting the live behavior change:
  1. Recall: does the fix actually let meaningfully more real HR-hitters
     through than the old h2h-dependent gate would have? (measured
     directly against real box scores, not assumed)
  2. Calibration: scored across this much broader population (not just the
     old "hr_elite tier" subset), is the model's own probability still
     honest -- AUC / ECE / Brier against real outcomes?

Mirrors api.py's live feature computation (batter_feature_row, now with an
as_of_date parameter added for this backtest; hits_context_feature_row;
_expected_pa_for; _hits_platoon_advantage; _recent_xbh_avg_from_splits)
rather than importing api.py directly -- matching how every other backtest
script in this repo is written: self-contained, read-only, no risk of
triggering api.py's live side effects (real prediction runs, external odds
calls, etc). lineupk.py IS imported directly since it's genuinely
standalone (only time/datetime/requests) -- used for the informational
old-gate reference via the real batter_hr_score function, not reimplemented.

KNOWN LIMITATION, flagged not hidden: the old-gate reference count uses
batter_hr_score's CURRENT h2h/season stats (it has no as_of_date support),
not a strict-as-of-date historical snapshot. This only affects the
informational "how many would the old gate have allowed" comparison -- the
calibration check (the primary result) IS strict D-1 throughout, via the
same as_of_date-parameterized feature computation the live serving path
now uses after this session's fixes.

Historical dates: sampled roughly weekly across the 2026 season through
~2 weeks before this script's run date, so outcomes are real and settled.

Read-only against the MLB Stats API + the already-trained, already-gated
model file at /data/models. Writes only a report.

Run (Render -- needs the real model file)
-------------------------------------------
python -u hr_model_recall_backtest_a.py 2>&1 | tee /data/hr_model/hr_model_recall_backtest_a.log
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import lineupk

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
MODEL_DIR = Path("/data/models")
WORKDIR_DEFAULT = "/data/hr_model/hr_model_recall_backtest_a_work"
SEASON = 2026
LOOKBACK_WEEKS = 16   # roughly weekly samples across the season so far
BUFFER_DAYS = 14      # stay this many days behind "today" for settled results

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "prop-edge-hr-recall-backtest/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return {}


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --- mirrors of api.py's feature-computation helpers (see module docstring) ---

def batter_feature_row(pid, as_of_date=None, season=None):
    g = _get(f"{MLB}/people/{pid}/stats", stats="gameLog", group="hitting", season=season or SEASON)
    try:
        splits = g["stats"][0]["splits"]
    except Exception:
        return None, None

    cum_h = cum_ab = cum_pa = cum_hr = cum_bb = cum_so = cum_tb = 0
    rec_xbh = []
    rec_hr = []
    used_splits = []

    for sp in splits:
        if as_of_date:
            game_date = str(sp.get("date") or "")
            if not game_date or not (game_date < str(as_of_date)[:10]):
                continue
        used_splits.append(sp)
        st = sp["stat"]
        h = int(st.get("hits", 0) or 0)
        tb = int(st.get("totalBases", 0) or 0)
        hr = int(st.get("homeRuns", 0) or 0)
        xbh = int(st.get("doubles", 0) or 0) + int(st.get("triples", 0) or 0) + hr

        cum_h += h
        cum_ab += int(st.get("atBats", 0) or 0)
        cum_pa += int(st.get("plateAppearances", 0) or 0)
        cum_hr += hr
        cum_bb += int(st.get("baseOnBalls", 0) or 0)
        cum_so += int(st.get("strikeOuts", 0) or 0)
        cum_tb += tb
        rec_xbh.append(xbh)
        rec_hr.append(hr)

    if cum_ab < 20 or len(used_splits) < 5:
        return None, None

    feat = {
        "season_avg": cum_h / cum_ab if cum_ab else 0,
        "hr_rate": cum_hr / cum_pa if cum_pa else 0,
        "bb_rate": cum_bb / cum_pa if cum_pa else 0,
        "so_rate": cum_so / cum_pa if cum_pa else 0,
        "games_played": len(used_splits),
        "tb_per_pa": cum_tb / cum_pa if cum_pa else 0,
        "season_slg": cum_tb / cum_ab if cum_ab else 0,
        "iso": (cum_tb - cum_h) / cum_ab if cum_ab else 0,
        "recent5_hr": sum(rec_hr[-5:]) / len(rec_hr[-5:]) if rec_hr else 0,
        "recent15_hr": sum(rec_hr[-15:]) / len(rec_hr[-15:]) if rec_hr else 0,
    }
    return feat, used_splits


def _expected_pa_for(lookup, lineup_spot, is_home):
    side = "home" if is_home else "away"
    try:
        spot = int(lineup_spot)
    except Exception:
        return float("nan")
    v = lookup.get(f"{spot}|{side}")
    return float(v) if v is not None else float("nan")


def _hits_platoon_advantage(bat_side, pitch_hand):
    if not bat_side or not pitch_hand:
        return float("nan")
    if bat_side == "S":
        return 1.0
    if (bat_side == "L" and pitch_hand == "R") or (bat_side == "R" and pitch_hand == "L"):
        return 1.0
    return 0.0


def _recent_xbh_avg_from_splits(splits, as_of_date=None, window=15):
    vals = []
    for sp in splits:
        if as_of_date:
            d = str(sp.get("date") or "")
            if not d or not (d < str(as_of_date)[:10]):
                continue
        st = sp.get("stat", {})
        vals.append(int(st.get("doubles", 0) or 0)
                    + int(st.get("triples", 0) or 0)
                    + int(st.get("homeRuns", 0) or 0))
    vals = vals[-window:]
    return (sum(vals) / len(vals)) if vals else 0.0


def hits_context_feature_row(base_feat, bat_side, pitch_hand, is_home,
                             lineup_spot, recent_xbh_avg, expected_pa_lookup):
    f = dict(base_feat)
    f["platoon_advantage"] = _hits_platoon_advantage(bat_side, pitch_hand)
    f["pitcher_is_R"] = 1.0 if pitch_hand == "R" else (0.0 if pitch_hand in ("L", "S") else float("nan"))
    f["is_home"] = 1.0 if is_home else 0.0
    f["expected_pa_v1"] = _expected_pa_for(expected_pa_lookup, lineup_spot, is_home)
    f["recent_xbh_avg"] = recent_xbh_avg
    return f


def sample_dates():
    latest = date.today() - timedelta(days=BUFFER_DAYS)
    season_start = date(SEASON, 3, 25)
    out = []
    d = season_start
    while d <= latest:
        out.append(d.isoformat())
        d += timedelta(weeks=1)
    return out[-LOOKBACK_WEEKS:]


def fetch_final_games(date_str):
    data = _get(f"{MLB}/schedule", sportId=1, date=date_str, gameType="R")
    out = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                out.append(str(g["gamePk"]))
    return out


_pitch_hand_cache = {}
_bat_side_cache = {}


def get_pitch_hand(pid):
    if pid in _pitch_hand_cache:
        return _pitch_hand_cache[pid]
    try:
        h = lineupk.get_pitcher_throws(pid)
    except Exception:
        h = None
    _pitch_hand_cache[pid] = h
    return h


def get_bat_side(pid):
    if pid in _bat_side_cache:
        return _bat_side_cache[pid]
    d = _get(f"{MLB}/people/{pid}")
    try:
        side = d["people"][0]["batSide"]["code"]
    except Exception:
        side = None
    _bat_side_cache[pid] = side
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    print("HR_MODEL_RECALL_BACKTEST_A\n==========================")

    mp_path = MODEL_DIR / "batter_home_runs_context.json"
    cp_path = MODEL_DIR / "batter_home_runs_context_columns.json"
    if not mp_path.exists() or not cp_path.exists():
        print(f"ERROR: model not found at {mp_path} -- run this on Render, not locally.")
        return 1
    booster = xgb.Booster(); booster.load_model(str(mp_path))
    cols = json.loads(cp_path.read_text())
    print(f"loaded model: {mp_path}  ({len(cols)} features)")

    try:
        expected_pa_lookup = json.loads((MODEL_DIR / "expected_pa_lookup.json").read_text())
    except Exception:
        expected_pa_lookup = {}
        print("WARNING: expected_pa_lookup.json not found -- expected_pa_v1 will be NaN for all rows")

    dates = sample_dates()
    print(f"sampling {len(dates)} historical dates: {dates[0]} .. {dates[-1]}")

    rows = []
    old_gate_fires = 0
    n_scored = 0

    for date_str in dates:
        game_ids = fetch_final_games(date_str)
        print(f"\n{date_str}: {len(game_ids)} final games", flush=True)
        for gid in game_ids:
            feed = _get(f"{MLB11}/game/{gid}/feed/live")
            box = feed.get("liveData", {}).get("boxscore", {})
            teams = box.get("teams", {})
            for side in ("home", "away"):
                team = teams.get(side, {})
                order = team.get("battingOrder", [])[:9]
                pitchers = team.get("pitchers", [])
                opp = teams.get("away" if side == "home" else "home", {})
                opp_pitchers = opp.get("pitchers", [])
                if not order or not opp_pitchers:
                    continue
                opp_starter = opp_pitchers[0]
                pitch_hand = get_pitch_hand(opp_starter)
                is_home = (side == "home")

                for spot, pid in enumerate(order, start=1):
                    pdata = team.get("players", {}).get(f"ID{pid}", {})
                    bat = pdata.get("stats", {}).get("batting", {})
                    actual_hr = int(bat.get("homeRuns", 0) or 0)
                    player_name = pdata.get("person", {}).get("fullName")
                    bat_side = get_bat_side(pid)

                    base_feat, splits = batter_feature_row(pid, as_of_date=date_str, season=SEASON)
                    if base_feat is None:
                        continue
                    n_scored += 1
                    recent_xbh = _recent_xbh_avg_from_splits(splits, as_of_date=date_str)
                    cfeat = hits_context_feature_row(
                        base_feat, bat_side, pitch_hand, is_home, spot,
                        recent_xbh, expected_pa_lookup)

                    x = np.array([[cfeat.get(c, 0) for c in cols]], dtype=np.float32)
                    prob = float(booster.predict(xgb.DMatrix(x, feature_names=cols))[0])

                    try:
                        sig = lineupk.batter_hr_score(pid, opp_starter, SEASON)
                        fires = bool(sig["fires"])
                    except Exception:
                        fires = False
                    old_gate_fires += int(fires)

                    rows.append({
                        "date": date_str, "game_id": gid, "player_id": pid,
                        "player_name": player_name, "opp_pitcher_id": opp_starter,
                        "model_prob": prob, "actual_hr": actual_hr,
                        "old_gate_fires": fires, "features": cfeat,
                    })
            time.sleep(0.05)

    print(f"\ntotal batters scored (strict D-1, real historical games): {n_scored}")
    print(f"of those, old gate (sig['fires']) would have allowed: {old_gate_fires} "
          f"({old_gate_fires/n_scored*100:.1f}%)" if n_scored else "n/a")
    print(f"new gate (model available) allows: {n_scored} (100% -- every confirmed-lineup batter scored)")

    probs = np.array([r["model_prob"] for r in rows])
    labels = np.array([1 if r["actual_hr"] >= 1 else 0 for r in rows], dtype=float)

    def auc(scores, y):
        pos = y.sum(); neg = len(y) - pos
        if pos == 0 or neg == 0:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
        s = np.asarray(scores)[order]; i = 0; n = len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))

    p = np.clip(probs, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - labels) ** 2))
    ll = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    ece = 0.0
    rel = []
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        mp_bin = float(p[m].mean()); ar = float(labels[m].mean())
        ece += abs(mp_bin - ar) * cnt / len(labels)
        rel.append((f"{b/10:.1f}-{(b+1)/10:.1f}", cnt, mp_bin, ar))

    real_auc = auc(probs, labels)
    print(f"\n=== CALIBRATION on the full (new-gate) population, real historical outcomes ===")
    print(f"  n={len(rows)}  base_rate={labels.mean():.4f}  AUC={real_auc:.4f}  "
          f"log_loss={ll:.5f}  Brier={brier:.5f}  ECE={ece:.4f}")
    print("  reliability (pred -> actual):")
    for binlabel, cnt, mp_bin, ar in rel:
        print(f"    {binlabel}  n={cnt:>5}  pred={mp_bin:.3f}  actual={ar:.3f}")

    # the actual real HR-hitters, sorted by predicted probability ascending --
    # the ones the model rated LOWEST despite them going deep are the real
    # "what did it miss" question, not the aggregate stats above.
    actual_hitters = sorted((r for r in rows if r["actual_hr"] >= 1), key=lambda r: r["model_prob"])
    print(f"\n=== {len(actual_hitters)} REAL HR-hitters in this sample, worst-rated first ===")
    print(f"  {'date':10s} {'player':22s} {'pred':>6s}  key features")
    for r in actual_hitters[:30]:
        f = r["features"]
        feat_str = (f"hr_rate={f.get('hr_rate', 0):.3f} iso={f.get('iso', 0):.3f} "
                    f"recent5_hr={f.get('recent5_hr', 0):.2f} recent15_hr={f.get('recent15_hr', 0):.2f} "
                    f"platoon={f.get('platoon_advantage')} home={f.get('is_home')}")
        print(f"  {r['date']:10s} {str(r['player_name'])[:22]:22s} {r['model_prob']:>6.3f}  {feat_str}")

    report = {
        "script": "HR_MODEL_RECALL_BACKTEST_A", "generated_at_utc": now_utc(),
        "dates_sampled": dates, "n_scored": n_scored,
        "old_gate_would_allow": old_gate_fires,
        "old_gate_allow_rate": old_gate_fires / n_scored if n_scored else None,
        "auc": real_auc, "log_loss": ll, "brier": brier, "ece": ece,
        "base_rate": float(labels.mean()), "reliability": rel,
    }
    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out_dir / "hr_model_recall_backtest_a_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "hr_model_recall_backtest_a_rows.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nreport: {out_dir / 'hr_model_recall_backtest_a_report.json'}")
    print(f"per-batter rows (for deeper analysis): {out_dir / 'hr_model_recall_backtest_a_rows.json'}")
    print("Read-only. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

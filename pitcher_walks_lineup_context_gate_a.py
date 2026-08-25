#!/usr/bin/env python3
"""
PITCHER_WALKS_LINEUP_CONTEXT_GATE_A

Tests whether adding real opponent-lineup walk-proneness to the validated
pitcher_walks model (mlb_new_markets_champion_gate_a.py, AUC 0.587,
pitcher-only features) actually improves it -- flagged live 2026-08-25:
pitcher_walks was built with zero opponent awareness, unlike the mature
pitcher_strikeouts model, which blends in real per-batter lineup K-rate
(lineupk.py: 40% head-to-head + 60% vs-pitcher-hand).

Deliberately does NOT reuse lineupk.py's live API mechanism for this
backtest. lineupk.py passes `endDate` to the MLB Stats API for historical
as-of queries, and that parameter is a SILENT NO-OP (proven directly
against the live API in pitcher_k_real_lineup_gate_a.py) -- fine for live
serving (as_of_date is always "today" there, nothing to leak), but any
HISTORICAL backtest built the same way would leak each batter's full
current-season rate into games that hadn't happened yet. That exact
mistake already burned a K-side attempt in this repo once
(pitcher_k_real_lineup_gate_a.py, INVALID, disclosed).

This backtest avoids it entirely by computing lineup walk-rates from
LOCAL per-game batter data (season_{year}.jsonl, already pulled from real
MLB box scores) with a real strict-D-1 cutoff by calendar date -- no
network calls, no endDate parameter, no chance of the same leak.

Opposing lineup reconstruction: batters on the opponent team with
batting_order 1-9 for that game_pk (same approximation
total_bases_clean_baseline_a.py already uses for "is_starter" -- doesn't
perfectly exclude a mid-lineup injury replacement, but a real, disclosed,
minor approximation, not a leak).

Two arms scored on the SAME untouched 2025 holdout as the original gate:
  pitcher_only   the already-validated model (season_rate, recent5_avg,
                 recent15_avg, season_avg_bf, games_played)
  challenger     pitcher_only features + lineup_avg_bb_rate

Pre-registered pass: challenger must beat pitcher_only on AUC by a real
margin on this holdout (paired bootstrap-free direct comparison, since
both arms are scored on identical rows) -- if it doesn't, the honest
result is "lineup context doesn't help walks the way it helps
strikeouts," not a bug to force through.

Read-only on season_{year}.jsonl. Writes only its own workdir.

Run:
    python -u pitcher_walks_lineup_context_gate_a.py
"""
import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

DATA_DIR = Path("/data")
WORKDIR = Path("/data/pitcher_walks_lineup_context_gate_a_work")
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
LEAGUE_AVG_BB_RATE = 0.085  # real per-PA MLB walk rate ballpark, used as fallback
MIN_BATTER_PA = 20
NAN = float("nan")


def _n(v):
    return v if isinstance(v, (int, float)) and v is not None else 0


def load_rows(season, row_type):
    out = []
    with open(DATA_DIR / f"season_{season}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == row_type:
                out.append(r)
    return out


def filter_starters(pitcher_rows):
    by_game_team = {}
    for r in pitcher_rows:
        key = (r["game_pk"], r["team"])
        cur = by_game_team.get(key)
        if cur is None or (r.get("bf") or 0) > (cur.get("bf") or 0):
            by_game_team[key] = r
    return list(by_game_team.values())


class BatterWalkHistory:
    """Per-player prefix-sum lookup: cumulative (pa, bb) using only games
    with a strictly earlier date, no network, no endDate leak risk."""

    def __init__(self, batter_rows):
        by_player = {}
        for r in batter_rows:
            by_player.setdefault(r["player_id"], []).append(r)
        self.dates = {}
        self.cum_pa = {}
        self.cum_bb = {}
        for pid, games in by_player.items():
            games.sort(key=lambda r: r["date"])
            dates, cpa, cbb = [], [0], [0]
            pa_run = bb_run = 0
            for g in games:
                dates.append(g["date"])
                pa_run += _n(g.get("pa"))
                bb_run += _n(g.get("bb"))
                cpa.append(pa_run)
                cbb.append(bb_run)
            self.dates[pid] = dates
            self.cum_pa[pid] = cpa
            self.cum_bb[pid] = cbb

    def rate_before(self, pid, date_str):
        dates = self.dates.get(pid)
        if not dates:
            return None
        idx = bisect_left(dates, date_str)  # count of games strictly before date_str
        pa = self.cum_pa[pid][idx]
        bb = self.cum_bb[pid][idx]
        if pa < MIN_BATTER_PA:
            return None
        return bb / pa


def build_lineup_index(batter_rows):
    """(game_pk, team) -> list of batter rows with battting_order 1-9."""
    idx = {}
    for r in batter_rows:
        bo = r.get("batting_order")
        if bo is None or not (1 <= bo <= 9):
            continue
        idx.setdefault((r["game_pk"], r["team"]), []).append(r)
    return idx


def build_pitcher_rows(pitcher_starts, batter_history, lineup_index):
    """Per-pitcher chronological history -> eligible feature rows, same
    eligibility/shape as mlb_new_markets_clean_baseline_a.py's pitcher
    builder, PLUS lineup_avg_bb_rate for the opponent lineup that day."""
    by_player = {}
    for r in pitcher_starts:
        by_player.setdefault(r["player_id"], []).append(r)

    out = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_bf = cum_bb = 0
        recent_bb = []
        n = 0
        for g in games:
            bf, bb = _n(g.get("bf")), _n(g.get("bb_allowed"))
            if n >= 5 and cum_bf >= 60:
                r5 = recent_bb[-5:]
                r15 = recent_bb[-15:]

                opp_lineup = lineup_index.get((g["game_pk"], g["opponent"]), [])
                lineup_rates = []
                for b in opp_lineup[:9]:
                    rate = batter_history.rate_before(b["player_id"], g["date"])
                    lineup_rates.append(rate if rate is not None else LEAGUE_AVG_BB_RATE)
                lineup_avg_bb_rate = (sum(lineup_rates) / len(lineup_rates)
                                       if lineup_rates else LEAGUE_AVG_BB_RATE)

                out.append({
                    "player_id": pid, "date": g["date"],
                    "season_rate": cum_bb / cum_bf if cum_bf else 0.0,
                    "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                    "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                    "season_avg_bf": cum_bf / n if n else 0.0,
                    "games_played": n,
                    "lineup_avg_bb_rate": lineup_avg_bb_rate,
                    "lineup_n_batters": len(opp_lineup[:9]),
                    "actual": bb,
                })
            cum_bf += bf
            cum_bb += bb
            recent_bb.append(bb)
            n += 1
    return out


PITCHER_ONLY_COLS = ["season_rate", "recent5_avg", "recent15_avg", "season_avg_bf", "games_played"]
CHALLENGER_COLS = PITCHER_ONLY_COLS + ["lineup_avg_bb_rate"]
LINE = 1.5  # matches the already-validated pitcher_walks line


def auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels.sum(); neg = len(labels) - pos
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
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics(probs, labels):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece = 0.0
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += abs(float(p[m].mean()) - float(y[m].mean())) * cnt / n
    return {"n": n, "base_rate": round(float(y.mean()), 4), "auc": round(auc(probs, labels), 4),
            "log_loss": round(ll, 5), "brier": round(brier, 5), "ece": round(ece, 4)}


def mat(rows, cols, xgb):
    X = np.array([[r.get(c, NAN) for c in cols] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=cols)


def pick_val_cut(dates_sorted, min_rows=200, frac=0.2):
    total = len(dates_sorted)
    target = max(min_rows, int(total * frac))
    return dates_sorted[max(0, total - target)]


def train_and_score(cols, dev_rows, hol_rows, label):
    import xgboost as xgb
    dates = sorted(r["date"] for r in dev_rows)
    cut = pick_val_cut(dates)
    tr = [r for r in dev_rows if r["date"] < cut]
    va = [r for r in dev_rows if r["date"] >= cut]
    print(f"  [{label}] train={len(tr)} val={len(va)} cols={cols}")

    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": 13}
    bst = xgb.train(params, mat(tr, cols, xgb), num_boost_round=800,
                     evals=[(mat(va, cols, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    probs_hol = bst.predict(mat(hol_rows, cols, xgb), iteration_range=itr)
    labels_hol = [r["over_line"] for r in hol_rows]
    m = metrics(list(map(float, probs_hol)), labels_hol)
    imp = bst.get_score(importance_type="gain")
    return m, imp, bst, probs_hol


def main():
    print("PITCHER_WALKS_LINEUP_CONTEXT_GATE_A\n====================================")
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("loading real season data...")
    dev_pitchers = filter_starters(load_rows(DEV_SEASON, "pitcher"))
    hol_pitchers = filter_starters(load_rows(HOLDOUT_SEASON, "pitcher"))
    dev_batters = load_rows(DEV_SEASON, "batter")
    hol_batters = load_rows(HOLDOUT_SEASON, "batter")
    print(f"starters: dev={len(dev_pitchers)} holdout={len(hol_pitchers)}")
    print(f"batters:  dev={len(dev_batters)} holdout={len(hol_batters)}")

    dev_hist = BatterWalkHistory(dev_batters)
    hol_hist = BatterWalkHistory(hol_batters)
    dev_lineup = build_lineup_index(dev_batters)
    hol_lineup = build_lineup_index(hol_batters)

    print("building pitcher rows with lineup context...")
    dev_rows = build_pitcher_rows(dev_pitchers, dev_hist, dev_lineup)
    hol_rows = build_pitcher_rows(hol_pitchers, hol_hist, hol_lineup)
    print(f"eligible pitcher-starts: dev={len(dev_rows)} holdout={len(hol_rows)}")

    for r in dev_rows + hol_rows:
        r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0

    lineup_feat = [r["lineup_avg_bb_rate"] for r in dev_rows]
    print(f"\nlineup_avg_bb_rate sanity: mean={np.mean(lineup_feat):.4f} "
          f"std={np.std(lineup_feat):.4f} min={min(lineup_feat):.4f} max={max(lineup_feat):.4f}")

    print(f"\n{'='*70}\nARM A: pitcher_only (already validated)\n{'='*70}")
    m_a, imp_a, _, probs_a = train_and_score(PITCHER_ONLY_COLS, dev_rows, hol_rows, "pitcher_only")
    print(f"  HOLDOUT: AUC={m_a['auc']:.4f}  logloss={m_a['log_loss']:.5f}  "
          f"Brier={m_a['brier']:.5f}  ECE={m_a['ece']:.4f}")

    print(f"\n{'='*70}\nARM B: pitcher_only + lineup_avg_bb_rate (challenger)\n{'='*70}")
    m_b, imp_b, _, probs_b = train_and_score(CHALLENGER_COLS, dev_rows, hol_rows, "challenger")
    print(f"  HOLDOUT: AUC={m_b['auc']:.4f}  logloss={m_b['log_loss']:.5f}  "
          f"Brier={m_b['brier']:.5f}  ECE={m_b['ece']:.4f}")
    print("\n  feature importance (gain):")
    for k, v in sorted(imp_b.items(), key=lambda x: -x[1]):
        print(f"    {k:24s} {v:9.2f}")

    d_auc = m_b["auc"] - m_a["auc"]
    d_ll = m_a["log_loss"] - m_b["log_loss"]
    print(f"\n{'#'*70}\nCOMPARISON (same 2025 holdout, same rows)\n{'#'*70}")
    print(f"  AUC:      pitcher_only={m_a['auc']:.4f}  challenger={m_b['auc']:.4f}  delta={d_auc:+.4f}")
    print(f"  logloss:  pitcher_only={m_a['log_loss']:.5f}  challenger={m_b['log_loss']:.5f}  gain={d_ll:+.5f}")
    passed = d_auc >= 0.01 and d_ll >= 0.0
    verdict = ("PITCHER_WALKS_LINEUP_CONTEXT_REAL_IMPROVEMENT" if passed
               else "PITCHER_WALKS_LINEUP_CONTEXT_NO_PROVEN_IMPROVEMENT")
    print(f"\n  GATE: delta AUC >= 0.01 AND logloss doesn't get worse -> {passed}")
    print(f"  VERDICT: {verdict}")

    report = {
        "script": "PITCHER_WALKS_LINEUP_CONTEXT_GATE_A",
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "pitcher_only": m_a, "challenger": m_b,
        "importance_challenger": imp_b,
        "delta_auc": round(d_auc, 4), "delta_logloss_gain": round(d_ll, 5),
        "passed": passed, "verdict": verdict,
    }
    (WORKDIR / "pitcher_walks_lineup_context_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {WORKDIR / 'pitcher_walks_lineup_context_gate_a_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

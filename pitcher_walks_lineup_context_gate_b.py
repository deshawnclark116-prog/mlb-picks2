#!/usr/bin/env python3
"""
PITCHER_WALKS_LINEUP_CONTEXT_GATE_B

Full-fidelity retest of gate_a. gate_a's lineup_avg_bb_rate used each
batter's plain overall walk rate only -- called out live as a
simplification of what the mature pitcher_strikeouts model actually does
(lineupk.py: 40% head-to-head vs THIS pitcher + 60% general rate split by
the pitcher's throwing hand). This version replicates that exact blend,
not a stand-in for it, so a "no improvement" result can't be explained
away as "the test wasn't the real thing."

Same leak-avoidance discipline as gate_a: everything computed from LOCAL
per-game data with a real strict-D-1 cutoff by calendar date, never
lineupk.py's live endDate-based API calls (proven elsewhere in this repo
to be a silent no-op that leaks future data into historical backtests).

New pieces needed for the real blend, not present in gate_a:
  - each batter's bats side / each pitcher's throwing hand (bulk-fetched
    from the MLB people endpoint, real static per-player data, saved to
    /data/player_hands.json)
  - each batter-game's opposing STARTING PITCHER identity (not just
    team), so head-to-head can be computed per specific pitcher and each
    plate-appearance-game can be bucketed by the hand actually faced

Two arms, same untouched 2025 holdout as gate_a:
  pitcher_only   unchanged from gate_a (season_rate, recent5_avg,
                 recent15_avg, season_avg_bf, games_played)
  challenger_b   pitcher_only + lineup_avg_bb_rate_blended (the real
                 H2H + hand-split blend, not the plain-average stand-in)

Same pre-registered bar as gate_a: delta AUC >= 0.01, logloss doesn't
get worse.

Read-only on season_{year}.jsonl and player_hands.json. Writes only its
own workdir.

Run:
    python -u pitcher_walks_lineup_context_gate_b.py
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
WORKDIR = Path("/data/pitcher_walks_lineup_context_gate_b_work")
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
LEAGUE_AVG_BB_RATE = 0.085
MIN_BATTER_PA = 15   # matches lineupk.py's general_k_rate_vs_hand MIN
MIN_H2H_PA = 2        # matches lineupk.py's MIN_H2H_PA
H2H_WEIGHT = 0.40
GEN_WEIGHT = 0.60
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


def build_lineup_index(batter_rows):
    idx = {}
    for r in batter_rows:
        bo = r.get("batting_order")
        if bo is None or not (1 <= bo <= 9):
            continue
        idx.setdefault((r["game_pk"], r["team"]), []).append(r)
    return idx


class BatterWalkHistoryV2:
    """Per-player prefix-sum lookups, both overall and hand-split, plus
    per-(batter,pitcher) head-to-head -- all strict D-1 by calendar date,
    all from local data (no endDate, no leak)."""

    def __init__(self, batter_rows, starter_index, hands):
        # starter_index: (game_pk, team) -> starter row (for the pitcher
        # THAT team started -- used to find each batter's opponent pitcher)
        by_player = {}
        for r in batter_rows:
            by_player.setdefault(r["player_id"], []).append(r)

        self.overall_dates = {}
        self.overall_cum_pa = {}
        self.overall_cum_bb = {}
        self.hand_dates = {"L": {}, "R": {}}
        self.hand_cum_pa = {"L": {}, "R": {}}
        self.hand_cum_bb = {"L": {}, "R": {}}
        # h2h[(batter_id, pitcher_id)] = sorted list of (date, pa, bb)
        self.h2h_games = {}

        for pid, games in by_player.items():
            games.sort(key=lambda r: r["date"])
            o_dates, o_pa, o_bb = [], [0], [0]
            h_dates = {"L": [], "R": []}
            h_pa = {"L": [0], "R": [0]}
            h_bb = {"L": [0], "R": [0]}
            o_pa_run = o_bb_run = 0
            h_pa_run = {"L": 0, "R": 0}
            h_bb_run = {"L": 0, "R": 0}

            for g in games:
                opp_pitcher = starter_index.get((g["game_pk"], g["opponent"]))
                opp_hand = hands.get(opp_pitcher["player_id"], {}).get("throws") if opp_pitcher else None
                pa, bb = _n(g.get("pa")), _n(g.get("bb"))

                o_dates.append(g["date"])
                o_pa_run += pa; o_bb_run += bb
                o_pa.append(o_pa_run); o_bb.append(o_bb_run)

                if opp_hand in ("L", "R"):
                    h_pa_run[opp_hand] += pa
                    h_bb_run[opp_hand] += bb
                    h_dates[opp_hand].append(g["date"])
                    h_pa[opp_hand].append(h_pa_run[opp_hand])
                    h_bb[opp_hand].append(h_bb_run[opp_hand])

                if opp_pitcher:
                    key = (pid, opp_pitcher["player_id"])
                    self.h2h_games.setdefault(key, []).append((g["date"], pa, bb))

            self.overall_dates[pid] = o_dates
            self.overall_cum_pa[pid] = o_pa
            self.overall_cum_bb[pid] = o_bb
            for hand in ("L", "R"):
                self.hand_dates[hand][pid] = h_dates[hand]
                self.hand_cum_pa[hand][pid] = h_pa[hand]
                self.hand_cum_bb[hand][pid] = h_bb[hand]

        for key in self.h2h_games:
            self.h2h_games[key].sort(key=lambda t: t[0])

    def general_rate_vs_hand(self, pid, hand, date_str):
        """Mirrors lineupk.general_k_rate_vs_hand's cascade: hand-split
        rate if enough PA, else season overall rate if enough PA, else
        league average -- computed locally, strict D-1."""
        if hand in ("L", "R"):
            dates = self.hand_dates[hand].get(pid, [])
            idx = bisect_left(dates, date_str)
            pa = self.hand_cum_pa[hand][pid][idx]
            bb = self.hand_cum_bb[hand][pid][idx]
            if pa >= MIN_BATTER_PA:
                return bb / pa

        dates = self.overall_dates.get(pid, [])
        idx = bisect_left(dates, date_str)
        pa = self.overall_cum_pa[pid][idx] if pid in self.overall_cum_pa else 0
        bb = self.overall_cum_bb[pid][idx] if pid in self.overall_cum_bb else 0
        if pa >= MIN_BATTER_PA:
            return bb / pa

        return LEAGUE_AVG_BB_RATE

    def h2h_rate(self, batter_id, pitcher_id, date_str):
        games = self.h2h_games.get((batter_id, pitcher_id))
        if not games:
            return None
        pa = bb = 0
        for d, p, b in games:
            if d < date_str:
                pa += p; bb += b
        if pa >= MIN_H2H_PA:
            return bb / pa
        return None

    def blended_rate(self, batter_id, pitcher_id, pitcher_hand, date_str):
        gen = self.general_rate_vs_hand(batter_id, pitcher_hand, date_str)
        h2h = self.h2h_rate(batter_id, pitcher_id, date_str)
        if h2h is not None:
            return H2H_WEIGHT * h2h + GEN_WEIGHT * gen
        return gen


def build_pitcher_rows(pitcher_starts, batter_history, lineup_index, hands):
    by_player = {}
    for r in pitcher_starts:
        by_player.setdefault(r["player_id"], []).append(r)

    out = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_bf = cum_bb = 0
        recent_bb = []
        n = 0
        pitcher_hand = hands.get(pid, {}).get("throws")
        for g in games:
            bf, bb = _n(g.get("bf")), _n(g.get("bb_allowed"))
            if n >= 5 and cum_bf >= 60:
                r5 = recent_bb[-5:]
                r15 = recent_bb[-15:]

                opp_lineup = lineup_index.get((g["game_pk"], g["opponent"]), [])
                lineup_rates = []
                for b in opp_lineup[:9]:
                    rate = batter_history.blended_rate(b["player_id"], pid, pitcher_hand, g["date"])
                    lineup_rates.append(rate)
                lineup_avg = sum(lineup_rates) / len(lineup_rates) if lineup_rates else LEAGUE_AVG_BB_RATE

                out.append({
                    "player_id": pid, "date": g["date"],
                    "season_rate": cum_bb / cum_bf if cum_bf else 0.0,
                    "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                    "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                    "season_avg_bf": cum_bf / n if n else 0.0,
                    "games_played": n,
                    "lineup_avg_bb_rate_blended": lineup_avg,
                    "actual": bb,
                })
            cum_bf += bf
            cum_bb += bb
            recent_bb.append(bb)
            n += 1
    return out


PITCHER_ONLY_COLS = ["season_rate", "recent5_avg", "recent15_avg", "season_avg_bf", "games_played"]
CHALLENGER_COLS = PITCHER_ONLY_COLS + ["lineup_avg_bb_rate_blended"]
LINE = 1.5


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
    return m, imp


def main():
    print("PITCHER_WALKS_LINEUP_CONTEXT_GATE_B\n====================================")
    WORKDIR.mkdir(parents=True, exist_ok=True)

    hands = json.loads((DATA_DIR / "player_hands.json").read_text())
    hands = {int(k): v for k, v in hands.items()}
    print(f"loaded hands for {len(hands)} players")

    print("loading real season data...")
    dev_pitchers = filter_starters(load_rows(DEV_SEASON, "pitcher"))
    hol_pitchers = filter_starters(load_rows(HOLDOUT_SEASON, "pitcher"))
    dev_batters = load_rows(DEV_SEASON, "batter")
    hol_batters = load_rows(HOLDOUT_SEASON, "batter")
    print(f"starters: dev={len(dev_pitchers)} holdout={len(hol_pitchers)}")

    dev_starter_idx = {(r["game_pk"], r["team"]): r for r in dev_pitchers}
    hol_starter_idx = {(r["game_pk"], r["team"]): r for r in hol_pitchers}
    dev_lineup = build_lineup_index(dev_batters)
    hol_lineup = build_lineup_index(hol_batters)

    print("building batter history (overall + hand-split + H2H)...")
    dev_hist = BatterWalkHistoryV2(dev_batters, dev_starter_idx, hands)
    hol_hist = BatterWalkHistoryV2(hol_batters, hol_starter_idx, hands)

    print("building pitcher rows with real H2H+hand-split lineup context...")
    dev_rows = build_pitcher_rows(dev_pitchers, dev_hist, dev_lineup, hands)
    hol_rows = build_pitcher_rows(hol_pitchers, hol_hist, hol_lineup, hands)
    print(f"eligible pitcher-starts: dev={len(dev_rows)} holdout={len(hol_rows)}")

    for r in dev_rows + hol_rows:
        r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0

    lf = [r["lineup_avg_bb_rate_blended"] for r in dev_rows]
    print(f"\nlineup_avg_bb_rate_blended sanity: mean={np.mean(lf):.4f} std={np.std(lf):.4f} "
          f"min={min(lf):.4f} max={max(lf):.4f}")

    print(f"\n{'='*70}\nARM A: pitcher_only (already validated)\n{'='*70}")
    m_a, imp_a = train_and_score(PITCHER_ONLY_COLS, dev_rows, hol_rows, "pitcher_only")
    print(f"  HOLDOUT: AUC={m_a['auc']:.4f}  logloss={m_a['log_loss']:.5f}  "
          f"Brier={m_a['brier']:.5f}  ECE={m_a['ece']:.4f}")

    print(f"\n{'='*70}\nARM B: pitcher_only + lineup_avg_bb_rate_blended (H2H + hand-split)\n{'='*70}")
    m_b, imp_b = train_and_score(CHALLENGER_COLS, dev_rows, hol_rows, "challenger_b")
    print(f"  HOLDOUT: AUC={m_b['auc']:.4f}  logloss={m_b['log_loss']:.5f}  "
          f"Brier={m_b['brier']:.5f}  ECE={m_b['ece']:.4f}")
    print("\n  feature importance (gain):")
    for k, v in sorted(imp_b.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v:9.2f}")

    d_auc = m_b["auc"] - m_a["auc"]
    d_ll = m_a["log_loss"] - m_b["log_loss"]
    print(f"\n{'#'*70}\nCOMPARISON (same 2025 holdout, same rows)\n{'#'*70}")
    print(f"  AUC:      pitcher_only={m_a['auc']:.4f}  challenger_b={m_b['auc']:.4f}  delta={d_auc:+.4f}")
    print(f"  logloss:  pitcher_only={m_a['log_loss']:.5f}  challenger_b={m_b['log_loss']:.5f}  gain={d_ll:+.5f}")
    passed = d_auc >= 0.01 and d_ll >= 0.0
    verdict = ("PITCHER_WALKS_LINEUP_CONTEXT_B_REAL_IMPROVEMENT" if passed
               else "PITCHER_WALKS_LINEUP_CONTEXT_B_NO_PROVEN_IMPROVEMENT")
    print(f"\n  GATE: delta AUC >= 0.01 AND logloss doesn't get worse -> {passed}")
    print(f"  VERDICT: {verdict}")

    report = {
        "script": "PITCHER_WALKS_LINEUP_CONTEXT_GATE_B",
        "note": "full H2H + hand-split blend, mirrors lineupk.py's real K-side methodology",
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "pitcher_only": m_a, "challenger_b": m_b,
        "importance_challenger_b": imp_b,
        "delta_auc": round(d_auc, 4), "delta_logloss_gain": round(d_ll, 5),
        "passed": passed, "verdict": verdict,
    }
    (WORKDIR / "pitcher_walks_lineup_context_gate_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {WORKDIR / 'pitcher_walks_lineup_context_gate_b_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

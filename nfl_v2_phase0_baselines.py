#!/usr/bin/env python3
"""NFL V2 Phase 0 transparent simple baselines.

These are deliberately simple, human-auditable comparators. They never see the
target week or later, never use sportsbook data, and are evaluated on the exact
same clean meaningful-player rows produced by nfl_v2_phase0_audit.py.

This is NOT a V2 model. The point is to answer a harsh question:
can the large V1 engine beat obvious pregame rules based on a player's own
recent usage/production?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

OUTCOME_STAT = {
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "rec": "receptions",
    "pass_yds": "passing_yards",
    "pass_td": "passing_tds",
    "int": "passing_interceptions",
}
OUTCOME_OPP = {
    "rush_yds": "carries",
    "rec_yds": "targets",
    "rec": "targets",
    "pass_yds": "attempts",
    "pass_td": "attempts",
    "int": "attempts",
}
PLAYER_COLS = ("player_id", "gsis_id")
TEAM_COLS = ("recent_team", "team")


def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def first(row, names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def load_history(paths):
    """Return player rows sorted chronologically. Target-week filtering happens
    later, so a file may safely contain the realized target week."""
    hist = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("season_type") or "REG") != "REG":
                    continue
                pid = first(r, PLAYER_COLS)
                if not pid:
                    continue
                try:
                    s, w = int(r["season"]), int(float(r["week"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rr = {"season": s, "week": w, "team": first(r, TEAM_COLS)}
                for col in set(OUTCOME_STAT.values()) | set(OUTCOME_OPP.values()):
                    # nflverse has used both names for passing interceptions.
                    aliases = (col, "interceptions") if col == "passing_interceptions" else (col,)
                    val = None
                    for a in aliases:
                        val = fnum(r.get(a))
                        if val is not None:
                            break
                    rr[col] = 0.0 if val is None else val
                hist[str(pid)].append(rr)
    for pid in hist:
        hist[pid].sort(key=lambda r: (r["season"], r["week"]))
    return hist


def prior_rows(hist, pid, season, week, limit=8):
    rows = [r for r in hist.get(str(pid), ()) if (r["season"], r["week"]) < (season, week)]
    return rows[-limit:]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def recent_values(rows, col, n):
    return [r[col] for r in rows[-n:]]


def baseline_predictions(row, hist):
    """Fixed transparent comparator family. No candidate is selected using the
    target week; all are reported side-by-side."""
    oc = row["outcome"]
    stat, opp = OUTCOME_STAT[oc], OUTCOME_OPP[oc]
    h = prior_rows(hist, row["player_id"], int(row["season"]), int(row["week"]), 8)
    if not h:
        return {}

    vals3, vals5, vals8 = recent_values(h, stat, 3), recent_values(h, stat, 5), recent_values(h, stat, 8)
    opp3, opp8 = recent_values(h, opp, 3), recent_values(h, opp, 8)

    out = {}
    if len(vals3) >= 2:
        out["recent3_mean"] = mean(vals3)
        out["recent3_median"] = statistics.median(vals3)
    if len(vals5) >= 3:
        out["recent5_mean"] = mean(vals5)
    if len(vals8) >= 2:
        # Latest game has weight 1; each older game receives 70% of the next
        # newer game's weight. Fixed a priori; this is not fitted to Week 4.
        weights = [0.70 ** (len(vals8)-1-i) for i in range(len(vals8))]
        out["ewma8_decay0.70"] = sum(v*w for v,w in zip(vals8, weights)) / sum(weights)
    if len(opp3) >= 2 and sum(opp8) > 0:
        expected_opp = mean(opp3)
        efficiency = sum(vals8) / sum(opp8)
        out["workload3_x_efficiency8"] = expected_opp * efficiency
    return out


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def compare(audit, hist):
    clean = [r for r in audit["rows"] if r.get("clean_meaningful") and r["outcome"] in OUTCOME_STAT]
    detail = []
    for r in clean:
        preds = baseline_predictions(r, hist)
        detail.append({
            "forecast_id": r["forecast_id"], "player": r["player"], "player_id": r["player_id"],
            "position": r["position"], "team": r["team"], "opponent": r["opponent"],
            "outcome": r["outcome"], "actual": r["actual"], "v1_median": r["median"],
            "baselines": preds,
        })

    by_outcome = {}
    candidates = ("recent3_mean", "recent3_median", "recent5_mean", "ewma8_decay0.70", "workload3_x_efficiency8")
    for oc in OUTCOME_STAT:
        rs = [r for r in detail if r["outcome"] == oc]
        cand = {}
        for name in candidates:
            paired = [r for r in rs if name in r["baselines"]]
            if not paired:
                cand[name] = {"n": 0}
                continue
            b_err = [abs(r["baselines"][name]-r["actual"]) for r in paired]
            v_err = [abs(r["v1_median"]-r["actual"]) for r in paired]
            b_signed = [r["baselines"][name]-r["actual"] for r in paired]
            cand[name] = {
                "n": len(paired),
                "coverage_of_clean": len(paired)/len(rs) if rs else None,
                "baseline_mae": avg(b_err),
                "v1_mae_same_rows": avg(v_err),
                "baseline_minus_v1_mae": avg(b_err)-avg(v_err),
                "baseline_bias": avg(b_signed),
                "baseline_row_wins": sum(be < ve for be,ve in zip(b_err,v_err)),
                "v1_row_wins": sum(ve < be for be,ve in zip(b_err,v_err)),
                "ties": sum(abs(be-ve) < 1e-12 for be,ve in zip(b_err,v_err)),
            }
        by_outcome[oc] = {"clean_n": len(rs), "candidates": cand}
    return {
        "schema": "nfl-v2-phase0-transparent-baselines-v1",
        "research_only": True,
        "sportsbook_inputs_used": False,
        "rule": "All baseline inputs must be from weeks strictly before the target row. No target-week candidate selection.",
        "candidate_definitions": {
            "recent3_mean": "mean of player's last up to 3 prior game outcomes; requires >=2",
            "recent3_median": "median of player's last up to 3 prior game outcomes; requires >=2",
            "recent5_mean": "mean of player's last up to 5 prior game outcomes; requires >=3",
            "ewma8_decay0.70": "up to 8 prior outcomes, latest weight 1 and each older weight x0.70; requires >=2",
            "workload3_x_efficiency8": "mean prior-3 opportunities x prior-8 total outcome / total opportunities; requires >=2 recent games and positive historical opportunities",
        },
        "by_outcome": by_outcome,
        "rows": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--history-stats", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    audit = json.loads(Path(a.audit).read_text(encoding="utf-8"))
    hist = load_history(a.history_stats)
    rep = compare(audit, hist)
    Path(a.out).write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(rep["by_outcome"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

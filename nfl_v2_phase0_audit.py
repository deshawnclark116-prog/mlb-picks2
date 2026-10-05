#!/usr/bin/env python3
"""NFL V2 Phase 0 audit: meaningful-player grading and error decomposition.

Research-only. This script does NOT fit a model and does NOT use sportsbook data.
It grades immutable Phase1C forecast JSONL against official nflverse weekly stats,
with the headline population restricted to pregame-meaningful players who actually
participated normally. It also decomposes mean forecast error into opportunity and
efficiency pieces where the accounting is identifiable.

Standard-library only so it can run in GitHub Actions against checked-out
shadow-state forecasts plus freshly downloaded official CSVs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CORE_OUTCOMES = ("rush_yds", "rec_yds", "rec", "pass_yds", "pass_td", "int")

STAT_ALIASES = {
    "rush_yds": ("rushing_yards",),
    "rec_yds": ("receiving_yards",),
    "rec": ("receptions",),
    "pass_yds": ("passing_yards",),
    "pass_td": ("passing_tds",),
    "int": ("passing_interceptions", "interceptions"),
}
OPP_ALIASES = {
    "rush_yds": ("carries",),
    "rec_yds": ("targets",),
    "rec": ("targets",),
    "pass_yds": ("attempts",),
    "pass_td": ("attempts",),
    "int": ("attempts",),
}
TEAM_COLS = ("team", "recent_team")
PLAYER_COLS = ("player_id", "gsis_id")

# Pregame-only definition, frozen before any Week 5 forward V2 evaluation.
# These are population definitions, NOT promotion thresholds.
MEANINGFUL = {
    "rush_yds": {"min_p_active": 0.90, "min_expected_opp": 5.0},
    "rec_yds": {"min_p_active": 0.90, "min_expected_opp": 3.0},
    "rec": {"min_p_active": 0.90, "min_expected_opp": 3.0},
    "pass_yds": {"min_p_active": 0.90, "min_expected_opp": 20.0},
    "pass_td": {"min_p_active": 0.90, "min_expected_opp": 20.0},
    "int": {"min_p_active": 0.90, "min_expected_opp": 20.0},
}

YARD_TOLS = (5, 10, 15, 20, 25)
COUNT_TOLS = (0, 1, 2)


def fnum(v: Any) -> float | None:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def first_num(row: dict[str, Any] | None, names: Iterable[str]) -> float | None:
    if row is None:
        return None
    for name in names:
        if name in row:
            x = fnum(row.get(name))
            if x is not None:
                return x
    return None


def first_text(row: dict[str, Any], names: Iterable[str]) -> str | None:
    for name in names:
        v = row.get(name)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def load_forecasts(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for p0 in paths:
        for line in Path(p0).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_batch") is not None or r.get("_footer") is True:
                continue
            if r.get("outcome") not in CORE_OUTCOMES:
                continue
            rid = str(r.get("id") or "")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            rows.append(r)
    return rows


def load_stats(path: str | Path):
    raw = Path(path).read_bytes()
    by_player = {}
    team_totals = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(raw.decode("utf-8").splitlines()):
        if (r.get("season_type") or "REG") != "REG":
            continue
        try:
            s, w = int(r["season"]), int(float(r["week"]))
        except (KeyError, TypeError, ValueError):
            continue
        pid = first_text(r, PLAYER_COLS)
        team = first_text(r, TEAM_COLS)
        if not pid:
            continue
        by_player[(s, w, pid)] = r
        if team:
            for col in ("carries", "targets", "attempts"):
                x = fnum(r.get(col))
                if x is not None:
                    team_totals[(s, w, team)][col] += x
    return by_player, team_totals, hashlib.sha256(raw).hexdigest()


def load_participation(snap_path: str | Path | None, players_path: str | Path | None):
    if not snap_path or not players_path:
        return None
    pfr2g = {}
    with open(players_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("pfr_id") and r.get("gsis_id"):
                pfr2g[r["pfr_id"]] = r["gsis_id"]
    out = set()
    with open(snap_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("game_type") or "REG") != "REG":
                continue
            if (fnum(r.get("offense_snaps")) or 0) <= 0 and (fnum(r.get("defense_snaps")) or 0) <= 0:
                continue
            gid = pfr2g.get(r.get("pfr_player_id") or "")
            if not gid:
                continue
            try:
                out.add((int(r["season"]), int(float(r["week"])), gid))
            except (KeyError, TypeError, ValueError):
                pass
    return out


def load_final_teams(path: str | Path | None):
    if not path:
        return None
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("game_type") or r.get("season_type") or "REG") != "REG":
                continue
            result = str(r.get("result") or "").strip()
            hs, aas = str(r.get("home_score") or "").strip(), str(r.get("away_score") or "").strip()
            final = result not in ("", "NA", "None") or (hs not in ("", "NA") and aas not in ("", "NA"))
            if not final:
                continue
            try:
                s, w = int(r["season"]), int(float(r["week"]))
            except (KeyError, TypeError, ValueError):
                continue
            for c in ("home_team", "away_team"):
                if r.get(c):
                    out.add((s, w, r[c]))
    return out


def load_censors(path: str | Path | None):
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = obj.get("events", obj if isinstance(obj, list) else [])
    out = {}
    for e in entries:
        try:
            out[(int(e["season"]), int(e["week"]), str(e["player_id"]))] = e
        except (KeyError, TypeError, ValueError):
            continue
    return out


def is_meaningful(f: dict[str, Any]) -> bool:
    rule = MEANINGFUL.get(f.get("outcome"))
    if not rule:
        return False
    pa, eo = fnum(f.get("p_active")), fnum(f.get("expected_opportunities"))
    return pa is not None and eo is not None and pa >= rule["min_p_active"] and eo >= rule["min_expected_opp"]


def actual_value(outcome: str, row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    v = first_num(row, STAT_ALIASES[outcome])
    return 0.0 if v is None else v


def actual_opp(outcome: str, row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    v = first_num(row, OPP_ALIASES[outcome])
    return 0.0 if v is None else v


def actual_team_share(outcome, row, team_totals, season, week, team):
    if row is None:
        return None
    opp_col = OPP_ALIASES[outcome][0]
    den = team_totals.get((season, week, team), {}).get(opp_col, 0.0)
    return None if den <= 0 else actual_opp(outcome, row) / den


def grade_rows(forecasts, official, team_totals, participation, censors, final_teams=None):
    out = []
    participation_weeks = None if participation is None else {(s, w) for s, w, _ in participation}
    for f in forecasts:
        s, w, pid = int(f["season"]), int(f["week"]), str(f["player_id"])
        if final_teams is not None and (s, w, f["team"]) not in final_teams:
            continue
        # A final game in games.csv can appear before the provider has published
        # that team's weekly stats. Never convert provider lag into fake zeros.
        if (s, w, f["team"]) not in team_totals:
            continue
        row = official.get((s, w, pid))
        y, ao = actual_value(f["outcome"], row), actual_opp(f["outcome"], row)
        meaningful = is_meaningful(f)
        if participation is None or (s, w) not in participation_weeks:
            # snap-count provider can lag the box-score provider. When the entire
            # target week is absent, a real official stats row is a conservative
            # participation proxy; no stats row stays unknown, never forced false.
            played = None if row is None else True
            played_source = "stats_row_proxy"
        else:
            played = (s, w, pid) in participation
            played_source = "snap_counts"
        censor = censors.get((s, w, pid))
        clean = bool(meaningful and played is True and not (censor and censor.get("exclude_from_clean_point_accuracy", True)))

        mean_, med, eo = fnum(f.get("mean")), fnum(f.get("median")), fnum(f.get("expected_opportunities"))
        pred_eff = mean_ / eo if mean_ is not None and eo is not None and eo > 0 else None
        act_eff = y / ao if ao > 0 else None
        opp_component = pred_eff * (eo - ao) if pred_eff is not None and eo is not None else None
        eff_component = pred_eff * ao - y if pred_eff is not None else None
        role = fnum((f.get("role_state") or {}).get("propensity_share"))
        act_share = actual_team_share(f["outcome"], row, team_totals, s, w, f["team"])

        rec = {
            "forecast_id": f["id"], "season": s, "week": w, "game_id": f["game_id"], "horizon": f["horizon"],
            "player_id": pid, "player": f.get("player_name"), "position": f.get("position"), "team": f["team"], "opponent": f.get("opponent"),
            "outcome": f["outcome"], "p_active": fnum(f.get("p_active")), "expected_opportunities": eo,
            "predicted_role_share": role, "actual_opportunities": ao, "actual_role_share": act_share,
            "mean": mean_, "median": med, "actual": y,
            "error_median": None if med is None else med-y, "abs_error_median": None if med is None else abs(med-y),
            "error_mean": None if mean_ is None else mean_-y, "abs_error_mean": None if mean_ is None else abs(mean_-y),
            "opportunity_error": None if eo is None else eo-ao,
            "predicted_efficiency_mean_per_opp": pred_eff, "actual_efficiency_per_opp": act_eff,
            "mean_error_opportunity_component": opp_component, "mean_error_efficiency_component": eff_component,
            "meaningful_pregame": meaningful, "played": played, "played_source": played_source, "clean_meaningful": clean,
            "censored": censor is not None, "censor_reason": None if not censor else censor.get("reason"),
            "uncertainty_score": fnum((f.get("uncertainty") or {}).get("score")),
            "uncertainty_reasons": list((f.get("uncertainty") or {}).get("reasons") or []),
            "role_shift_vs_last8": fnum((f.get("role_state") or {}).get("share_shift_vs_last8")),
        }
        if opp_component is not None and eff_component is not None and mean_ is not None:
            rec["decomposition_check"] = (opp_component + eff_component) - (mean_-y)
        out.append(rec)
    return out


def avg(xs):
    vals = [float(x) for x in xs if x is not None]
    return sum(vals)/len(vals) if vals else None


def medv(xs):
    vals = [float(x) for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def accuracy_curve(rows, outcome):
    tols = YARD_TOLS if outcome.endswith("_yds") else COUNT_TOLS
    errs = [r["abs_error_median"] for r in rows if r["abs_error_median"] is not None]
    if not errs:
        return {}
    return {f"within_{t}": sum(e <= t for e in errs)/len(errs) for t in tols}


def summarize_group(rows):
    if not rows:
        return {"n": 0}
    outcome = rows[0]["outcome"]
    role_err = [abs(r["predicted_role_share"]-r["actual_role_share"]) for r in rows if r["predicted_role_share"] is not None and r["actual_role_share"] is not None]
    opp_abs = [abs(r["opportunity_error"]) for r in rows if r["opportunity_error"] is not None]
    return {
        "n": len(rows),
        "mae_median": avg(r["abs_error_median"] for r in rows),
        "median_ae": medv(r["abs_error_median"] for r in rows),
        "bias_median": avg(r["error_median"] for r in rows),
        "mae_mean": avg(r["abs_error_mean"] for r in rows),
        "bias_mean": avg(r["error_mean"] for r in rows),
        "opportunity_mae": avg(opp_abs),
        "role_share_mae": avg(role_err),
        "accuracy_curve": accuracy_curve(rows, outcome),
    }


def build_report(rows, stats_sha, forecast_paths):
    report = {
        "schema": "nfl-v2-phase0-audit-v1",
        "research_only": True,
        "sportsbook_inputs_used": False,
        "headline_population": "clean_meaningful = pregame meaningful + confirmed participation + no documented in-game censor",
        "meaningful_rules": MEANINGFUL,
        "stats_sha256": stats_sha,
        "forecast_files": forecast_paths,
        "counts": {
            "all_rows": len(rows),
            "pregame_meaningful": sum(r["meaningful_pregame"] for r in rows),
            "clean_meaningful": sum(r["clean_meaningful"] for r in rows),
            "censored": sum(r["censored"] for r in rows),
        },
        "headline_by_outcome": {},
        "pregame_meaningful_by_outcome": {},
        "by_position": {},
        "largest_clean_misses": [],
        "largest_raw_misses": [],
        "rows": rows,
    }
    for oc in CORE_OUTCOMES:
        pre = [r for r in rows if r["outcome"] == oc and r["meaningful_pregame"]]
        clean = [r for r in rows if r["outcome"] == oc and r["clean_meaningful"]]
        report["pregame_meaningful_by_outcome"][oc] = summarize_group(pre)
        report["headline_by_outcome"][oc] = summarize_group(clean)
    for pos in ("QB", "RB", "WR", "TE"):
        rs = [r for r in rows if r["clean_meaningful"] and r["position"] == pos]
        if rs:
            report["by_position"][pos] = {oc: summarize_group([r for r in rs if r["outcome"] == oc]) for oc in CORE_OUTCOMES if any(r["outcome"] == oc for r in rs)}
    ranked_raw = [r for r in rows if r["meaningful_pregame"] and r["abs_error_median"] is not None]
    ranked_clean = [r for r in rows if r["clean_meaningful"] and r["abs_error_median"] is not None]
    key = lambda r: (-r["abs_error_median"], r["game_id"], r["player"], r["outcome"])
    fields = ("game_id", "player", "position", "team", "opponent", "outcome", "median", "actual", "abs_error_median",
              "expected_opportunities", "actual_opportunities", "opportunity_error", "mean_error_opportunity_component",
              "mean_error_efficiency_component", "censored", "censor_reason")
    report["largest_raw_misses"] = [{k:r.get(k) for k in fields} for r in sorted(ranked_raw, key=key)[:30]]
    report["largest_clean_misses"] = [{k:r.get(k) for k in fields} for r in sorted(ranked_clean, key=key)[:30]]
    return report


def expand_forecast_args(values):
    out = []
    for v in values:
        hits = sorted(glob.glob(v))
        out.extend(hits if hits else [v])
    seen, ans = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); ans.append(p)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", action="append", required=True, help="Forecast JSONL path or glob; repeatable")
    ap.add_argument("--stats", required=True, help="Official nflverse stats_player_week CSV")
    ap.add_argument("--games", default=None, help="Official nflverse games.csv; when provided, only final games are graded")
    ap.add_argument("--snap-counts", default=None)
    ap.add_argument("--players", default=None)
    ap.add_argument("--censors", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fps = expand_forecast_args(a.forecast)
    forecasts = load_forecasts(fps)
    official, team_totals, stats_sha = load_stats(a.stats)
    participation = load_participation(a.snap_counts, a.players)
    censors = load_censors(a.censors)
    final_teams = load_final_teams(a.games)
    rows = grade_rows(forecasts, official, team_totals, participation, censors, final_teams)
    report = build_report(rows, stats_sha, fps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": a.out, "counts": report["counts"], "headline_by_outcome": report["headline_by_outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
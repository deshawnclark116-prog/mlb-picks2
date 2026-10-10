#!/usr/bin/env python3
"""CFB independent, *read-only* forward truth / forecast-contract audit.

Uses ONLY the earliest verifiably pregame ledger entry for each canonical key.
Rechecks the grader's real outcomes against the existing SQLite player-game
and final-game records; never uses a refreshed board's probabilities to grade.

Crucial distinction: a threshold classifier's P(OVER 69.5) is NOT a forecast
of a player's exact rushing-yard total. A model-stat MAE is reported only when
an ORIGINAL pregame record actually contains a finite 'projected' value.

Produces an explicitly observational, separate JSON report. No changes to
models, calibrators, production market status, thresholds, grading ledgers,
frozen Week 5 results, or picks.
"""
import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cfb_grade_record_a as G

SCHEMA = "CFB_FORWARD_TRUTH_AUDIT_V1"
BINS = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.000000001))
SEED = 20261009
BOOT_REPS = 1000
FREEZE_POLICY = "FORWARD_OBSERVATIONAL_NO_MODEL_TUNING_OR_PROMOTION"
POINT_FIELDS = ("projected", "sim_mean")


class AuditError(ValueError):
    pass


def _num(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise AuditError(f"INVALID_{field.upper()}")
    return float(value)


def _side_from_original(record):
    pick = str(record.get("pick") or "").upper()
    if "OVER " in pick:
        return "OVER"
    if "UNDER " in pick:
        return "UNDER"
    raise AuditError("INVALID_PROP_SIDE")


def _check_pregame_record(record, original, games_by_id, player_rows):
    """Cross-check graded snapshot against immutable FIRST entry and official DB truth."""
    r, e = record, original["entry"]
    for key in ("market", "pick", "player_id", "team", "opponent", "season", "week"):
        if r.get(key) != e.get(key):
            raise AuditError("ORIGINAL_LEDGER_FIELD_MISMATCH:" + key)
    if r.get("logged_at") != e.get("logged_at"):
        raise AuditError("ORIGINAL_TIMESTAMP_MISMATCH")
    if r.get("evaluation_probability_source") != G.EVAL_SOURCE:
        raise AuditError("NOT_FIRST_PREGAME_PROBABILITY")
    if r.get("result") not in ("hit", "miss"):
        raise AuditError("ONLY_GRADED_RESULTS_ALLOWED")
    if not original["valid"]:
        raise AuditError("LATE_OR_UNVERIFIABLE_FORECAST")
    if original["gstatus"] != "OK":
        raise AuditError("UNRESOLVED_GAME")
    g = original["game"]
    if str(g["game_id"]) != str(r.get("game_id")):
        raise AuditError("GAME_ID_MISMATCH")
    if G.parse_ts(r.get("kickoff_utc")) != original["kickoff"]:
        raise AuditError("KICKOFF_MISMATCH")
    prob = _num(e.get("model_prob"), "original_model_prob")
    if not 0 <= prob <= 1:
        raise AuditError("ORIGINAL_PROBABILITY_OUT_OF_RANGE")
    if _num(r.get("model_prob"), "graded_model_prob") != prob or _num(r.get("original_model_prob"), "graded_original_model_prob") != prob:
        raise AuditError("REFRESHED_OR_CHANGED_PROBABILITY")
    if r.get("generation") != G.generation(e):
        raise AuditError("GENERATION_MISMATCH")
    if G.is_moneyline(e["market"]):
        if g.get("home_points") is None or g.get("away_points") is None or g["home_points"] == g["away_points"]:
            raise AuditError("NO_FINAL_WINNER")
        winner = g["home_team"] if g["home_points"] > g["away_points"] else g["away_team"]
        if winner != r.get("actual") or e["team"] not in (g["home_team"], g["away_team"]):
            raise AuditError("MONEYLINE_TRUTH_MISMATCH")
        if r["result"] != ("hit" if e["team"] == winner else "miss"):
            raise AuditError("MONEYLINE_RESULT_MISMATCH")
        return None
    if _num(r.get("line"), "line") != _num(e.get("line"), "original_line"):
        raise AuditError("LINE_CHANGED_AFTER_PREGAME")
    db_row = player_rows.get((str(e["player_id"]), str(g["game_id"])))
    if db_row is None:
        raise AuditError("MISSING_PLAYER_GAME_TRUTH")
    actual = G.actual_stat(e["market"], db_row)
    if actual is None or _num(r.get("actual"), "actual") != _num(actual, "db_actual"):
        raise AuditError("PLAYER_GAME_TRUTH_MISMATCH")
    if r["result"] != G.grade(e, actual):
        raise AuditError("PROP_GRADE_MISMATCH")
    return float(actual)


def _cluster_gap_interval(items, reps=BOOT_REPS, seed=SEED):
    """Pregame game-level clustered *descriptive* interval; not promotion proof."""
    groups = defaultdict(list)
    for row in items:
        groups[(str(row["season"]), str(row["game_id"]))].append(
            (float(row["model_prob"]), 1.0 if row["result"] == "hit" else 0.0)
        )
    if len(groups) < 5:
        return {"status": "TOO_FEW_INDEPENDENT_GAMES", "n_game_clusters": len(groups)}
    clusters = [groups[k] for k in sorted(groups)]
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        xs = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        n, diff = 0, 0.0
        for cluster in xs:
            n += len(cluster)
            diff += sum(p - y for p, y in cluster)
        vals.append(diff / n)
    vals.sort()
    return {
        "status": "DESCRIPTIVE_CLUSTER_BOOTSTRAP",
        "n_game_clusters": len(clusters),
        "reps": reps,
        "seed": seed,
        "calibration_gap_95pct_interval": [round(vals[int(0.025 * reps)], 5),
                                            round(vals[min(reps - 1, int(0.975 * reps))], 5)],
        "note": "Resample by actual game; still observational and selection-conditional.",
    }


def _metrics(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    ps = [_num(r["model_prob"], "model_prob") for r in rows]
    ys = [int(r["result"] == "hit") for r in rows]
    avg = sum(ps) / n
    hit = sum(ys) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    eps = 1e-12
    log_loss = -sum(y * math.log(min(1 - eps, max(eps, p))) +
                    (1 - y) * math.log(min(1 - eps, max(eps, 1 - p)))
                    for p, y in zip(ps, ys)) / n
    buckets = []
    for lo, hi in BINS:
        selected = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi]
        if selected:
            buckets.append({"probability_bucket": f"{lo:.1f}-{min(hi,1):.1f}",
                            "n": len(selected),
                            "mean_claimed_probability": round(sum(p for p, _ in selected) / len(selected), 5),
                            "actual_success_rate": round(sum(y for _, y in selected) / len(selected), 5)})
    return {
        "n": n, "wins": sum(ys), "losses": n - sum(ys),
        "mean_claimed_probability": round(avg, 5),
        "actual_hit_rate": round(hit, 5),
        "overconfidence_gap_positive_is_overclaim": round(avg - hit, 5),
        "brier": round(brier, 5), "log_loss": round(log_loss, 5),
        "fixed_coin_flip_brier_reference": 0.25,
        "probability_buckets": buckets,
        "game_cluster_interval": _cluster_gap_interval(rows),
    }


def audit(record, ledger, games, player_rows, status=None):
    """Never infer point estimates from binary probabilities or from a betting threshold."""
    if not isinstance(record, dict) or not isinstance(record.get("results"), list):
        raise AuditError("INVALID_GRADED_RECORD")
    if record.get("forward_evaluation", {}).get("evaluation_probability_source") != G.EVAL_SOURCE:
        raise AuditError("GRADER_FORWARD_EVIDENCE_CONTRACT_MISSING")
    canonical, lineage = G.select_canonical(ledger, G.Schedule(games))
    rows = record["results"]
    if status and status.get("graded_predictions") != len(rows):
        raise AuditError("FORWARD_STATUS_GRADED_COUNT_MISMATCH")
    if len(rows) != record.get("summary", {}).get("total"):
        raise AuditError("RECORD_SUMMARY_COUNT_MISMATCH")
    checked = set()
    enriched, groups, generation_groups = [], defaultdict(list), defaultdict(list)
    proj = defaultdict(lambda: {"eligible_stat_rows": 0, "original_point_projection_rows": 0,
                                "absolute_error_sum": 0.0, "source_fields": defaultdict(int)})
    for r in rows:
        key = r.get("canonical_key")
        if not isinstance(key, str) or key in checked:
            raise AuditError("MISSING_OR_DUPLICATE_GRADED_KEY")
        checked.add(key)
        if key not in canonical:
            raise AuditError("MISSING_FIRST_PREGAME_LEDGER_ENTRY:" + key)
        original = canonical[key]
        actual = _check_pregame_record(r, original, None, player_rows)
        enriched.append(r)
        groups[r["market"]].append(r)
        generation_groups[r["generation"] + "::" + r["market"]].append(r)
        if actual is not None:
            section = proj[r["market"]]
            section["eligible_stat_rows"] += 1
            entry = original["entry"]
            val = entry.get("projected")
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                section["original_point_projection_rows"] += 1
                section["absolute_error_sum"] += abs(float(val) - actual)
                section["source_fields"]["projected"] += 1
                if isinstance(entry.get("sim_mean"), (int, float)):
                    section["source_fields"]["sim_mean"] += 1
    n = len(rows)
    measured = {}
    for market, section in sorted(proj.items()):
        v = section["original_point_projection_rows"]
        measured[market] = {
            "eligible_stat_rows": section["eligible_stat_rows"],
            "original_point_projection_rows": v,
            "coverage_pct": round(100 * v / section["eligible_stat_rows"], 3),
            "projection_mae_only_when_original_pregame_point_exists":
                round(section["absolute_error_sum"] / v, 4) if v else None,
            "source_fields": dict(section["source_fields"]),
            "missing_policy": "NO_PROJECTION_NO_MAE" if not v else "ORIGINAL_LEDGER_POINT_ONLY",
        }
    return {
        "schema": SCHEMA,
        "scientific_status": FREEZE_POLICY,
        "verified_first_pregame_rows": n,
        "ledger_rows_seen": len(ledger),
        "canonical_ledger_keys": len(canonical),
        "duplicate_canonical_ledger_keys": lineage["duplicate_key_count"],
        "original_ledger_integrity": "EACH_GRADED_ROW_CROSS_CHECKED_AGAINST_EARLIEST_VALID_LOG",
        "outcome_truth_integrity": "EACH_GRADED_OUTCOME_RECHECKED_AGAINST_SQLITE",
        "live_board_probabilities_used_for_grading": False,
        "overall": _metrics(enriched),
        "by_market": {k: _metrics(v) for k, v in sorted(groups.items())},
        "by_generation_and_market": {k: _metrics(v) for k, v in sorted(generation_groups.items())},
        "original_point_projection_coverage": measured,
        "fixed_lines_are_not_point_forecasts": True,
        "probability_confidence_is_conditional_on_selected_picks": True,
        "no_odds_or_profitability_claim": True,
        "new_predictions_generated": 0,
        "champion_models_or_calibration_changed": False,
        "disallowed_use": "DO_NOT_TUNE_ON_THIS_FORWARD_REPORT; 2026 performance is observational",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=G.DB_DEFAULT)
    ap.add_argument("--log", type=Path, default=G.LOG_DEFAULT)
    ap.add_argument("--record", type=Path, default=G.OUT_DEFAULT)
    ap.add_argument("--status", type=Path, default=G.DOCS / "cfb_forward_status.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ledger = G.load_ledger(args.log)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        games, _ = G.load_games(con)
        players = G.load_player_rows(con, ledger)
    finally:
        con.close()
    status = json.loads(args.status.read_text())
    report = audit(json.loads(args.record.read_text()), ledger, games, players, status)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "checked": report["verified_first_pregame_rows"],
        "overall_gap": report["overall"].get("overconfidence_gap_positive_is_overclaim"),
        "projection_coverage": report["original_point_projection_coverage"],
        "scientific_status": report["scientific_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

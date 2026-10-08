#!/usr/bin/env python3
"""Research-only October 7, 2026 shadow grading of immutable NHL B2 forward forecasts.

This script never alters the forecast ledger or its frozen engine lock. Outcomes
are retrieved after official finalization; grades live only in an ephemeral
working directory, and the output is labeled descriptive, not confirmatory.
"""
import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import nhl_v2_phase1a_sog_forward as F
import nhl_v2_forward_integrity_grade as G

DAY = "2026-10-07"
GAMES = {2026020053, 2026020054, 2026020055}
HORIZONS = ("T90", "T30")


class HistoricalView:
    """References immutable original rows; does not synthesize their hashes."""
    def __init__(self, parent, rows):
        self.parent = parent
        self.selected = rows

    def verify(self):
        return self.parent.verify()

    def rows(self):
        return self.selected


def mae(pairs, field):
    eligible = [(f, g) for f, g, _ in pairs if field in f and f[field] is not None]
    return (sum(abs(float(f[field]) - g["actual_sog"]) for f, g in eligible) / len(eligible)) if eligible else None


def run(output_path=None):
    original = F.Ledger()
    original_count = original.verify()
    all_forecasts = [r for r in original.rows() if r.get("record_type") == "FORECAST"]
    source = [r for r in all_forecasts if r.get("schedule_date") == DAY and r["game_id"] in GAMES and r["forecast_horizon"] in HORIZONS]
    assert len(source) == 352, "October 7 original forecast population changed"
    collisions = G.identity_collisions(source)
    assert len(collisions) == 8, "known cross-team identity audit changed"
    view = HistoricalView(original, source)

    with tempfile.TemporaryDirectory() as td:
        grades_path = Path(td) / "grades.jsonl"
        G.grade(ledger=view, grades_path=grades_path, log=lambda *args: print(*args))
        allgrades = F.Ledger(grades_path).rows()
        raw_pairs = G.join(ledger=view, grades_path=grades_path, censor_path=Path(td) / "no_censors.jsonl")
        records = {r["forecast_row_hash"]: r for r in allgrades}
        report = {
            "scope": "NHL V2 Oct 7 2026 official-final shadow grading, B2 v1.1 immutable",
            "evidence_grade": "CLEAN_FORWARD_DESCRIPTIVE_ONLY_NOT_CONFIRMATORY",
            "original_hash_chain_verified_count": original_count,
            "expected_forecasts": len(source),
            "quarantined_ambiguous_forecasts": len(collisions),
            "graded_forecasts": len(raw_pairs),
            "ungraded_by_reason": dict(Counter(r.get("reason", "UNKNOWN") for r in allgrades if r["record_type"] == "UNGRADED")),
            "missing_grade_or_quarantine_count": len(source) - len(records),
            "human_comparator_available": sum(f["comparators"]["human_frozen_mean"] is not None for f in source),
            "lineup_state_certified_forecasts": sum(f.get("availability_confidence") == "CERTIFIED" for f in source),
            "status": "COMPLETE" if len(records) == 352 and len(raw_pairs) == 344 and len(collisions) == 8 else "INCOMPLETE_DO_NOT_INFER_ZERO",
            "by_horizon": {},
        }
        for h in HORIZONS:
            pp = [(f, g, c) for f, g, c in raw_pairs if f["forecast_horizon"] == h]
            rep = {"graded_rows": len(pp), "played_rows": sum(g["played"] for _, g, _ in pp)}
            for pop in ("FULL", "MEANINGFUL"):
                sel = [(f, g, c) for f, g, c in pp if pop == "FULL" or f["meaningful_expected_participant"]]
                if not sel:
                    rep[pop] = {"n": 0}
                    continue
                met = G.evaluate(sel, json.loads(F.LOCK.read_text()), h, clean=False, population=pop)["V2_B2"]
                simple = [(f, g) for f, g, _ in sel if f["comparators"]["simple_prior10_mean"] is not None]
                sim_mae = sum(abs(f["comparators"]["simple_prior10_mean"] - g["actual_sog"]) for f, g in simple) / len(simple) if simple else None
                model_on_simple = sum(abs(f["expected_sog"] - g["actual_sog"]) for f, g in simple) / len(simple) if simple else None
                misses = sorted([
                    {"name": f["receipt"].get("player_name"), "player_id": f["player_id"],
                     "game_id": f["game_id"], "predicted": round(f["expected_sog"], 4),
                     "actual": g["actual_sog"], "played": g["played"],
                     "abs_error": round(abs(f["expected_sog"] - g["actual_sog"]), 4)}
                    for f, g, _ in sel
                ], key=lambda x: -x["abs_error"])
                rep[pop] = {
                    "n": len(sel), "mean_absolute_error": met["central"]["mae_mean"] if "mae_mean" in met["central"] else met["central"],
                    "central": met["central"],
                    "CRPS_game_macro": met["crps_macro_game"],
                    "NLL_game_macro": met["nll_macro_game"],
                    "P3": met["thresholds"].get("3", {}),
                    "simple_prior10_same_rows_n": len(simple),
                    "simple_prior10_MAE_same_rows": sim_mae,
                    "B2_MAE_on_simple_rows": model_on_simple,
                    "large_misses": misses[:8],
                }
            report["by_horizon"][h] = rep
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="")
    args = p.parse_args()
    run(args.output or None)

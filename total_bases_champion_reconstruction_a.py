#!/usr/bin/env python3
"""
TOTAL_BASES_CHAMPION_RECONSTRUCTION_A

Purpose
-------
Reconstruct and FREEZE the exact CURRENT batter-total-bases production champion
before any new research branch is opened. This is rung 1 of the total-bases
promotion pipeline and mirrors hits_champion_reconstruction_a.py.

This script is read-only. It does not modify code, database tables, predictions,
models, or production state.

It produces:
1. A frozen, human-readable specification of the current TB champion (the exact
   projection -> probability -> nudge -> Monte Carlo -> board path), with
   file evidence, so the champion cannot silently drift during research.
2. A source AST inventory of the functions/constants that drive TB.
3. A SQLite inventory plus a total-bases TARGET AVAILABILITY check: total bases
   is derivable from batter_games as hits + doubles + 2*triples + 3*home_runs,
   so this confirms the holdout target exists and reports coverage by season.
4. A scan of recent /data TB prediction/candidate artifacts.

Run (on Render, where /data and the sqlite baseline live)
--------------------------------------------------------
python -u total_bases_champion_reconstruction_a.py 2>&1 | tee /data/hr_model/total_bases_champion_reconstruction_a.log

Run (local smoke test; degrades gracefully with no /data)
---------------------------------------------------------
python -u total_bases_champion_reconstruction_a.py --root .

Output
------
/data/hr_model/total_bases_champion_reconstruction_a_report.json
/data/hr_model/total_bases_champion_reconstruction_a_report.txt
(falls back to ./<same names> when /data/hr_model is not writable)
"""

import argparse
import ast
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# =============================================================================
# Paths
# =============================================================================

OUT_DIR_PRIMARY = Path("/data/hr_model")
DATA_ROOT = Path("/data")

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".pytest_cache", "dist", "build",
}

TB_SOURCE_FILES = ("api.py", "lineupk.py", "bvp.py", "build.py")

# Patterns that identify TB-driving logic in source.
TB_PATTERNS = [
    "batter_total_bases",
    "total_bases",
    "_rec_tb",
    "tb_per_pa",
    "totalBases",
    "power_flag",
    "CANDIDATE_ONLY_HITTER_PROPS",
    "PROBATIONARY_MARKETS",
    "hitter_mc",
    "prob_over",
]


# =============================================================================
# FROZEN CHAMPION SPEC
# -----------------------------------------------------------------------------
# This is the exact current production computation for batter_total_bases,
# traced from api.py at version 8.19B. Each step carries a file::symbol anchor
# so the reconstruction can be re-verified against source. Keep this in sync
# with source ONLY by re-tracing; never edit to "make it pass".
# =============================================================================

FROZEN_CHAMPION_SPEC = {
    "market": "batter_total_bases",
    "api_version_traced": "8.19B",
    "lifecycle_status": "active_probationary / candidate_only",
    "board_promotion_ceiling": "watchlist_prediction (never official)",
    "pipeline": [
        {
            "step": "1_feature_row",
            "symbol": "api.py::batter_feature_row",
            "detail": (
                "MLB gameLog season-to-date. Cumulative + recent windows. "
                "Eligibility gate: cum_ab >= 20 and games_played >= 5. "
                "Emits _rec_tb (per-game total bases list) and tb_per_pa."
            ),
            "known_risk": (
                "TIME LEAK: season-to-date gameLog has no strict D-1 "
                "exclusion, so earlier same-day games can enter features. "
                "Same defect class as pitcher-K TIME-01. Must be resolved "
                "before the TB formal gate, exactly as it was for K."
            ),
        },
        {
            "step": "2_feature_adapt",
            "symbol": "api.py::_batter_feat_for",
            "detail": (
                "For batter_total_bases, recent5_target/recent15_target are "
                "set from the mean of _rec_tb over the last 5 / 15 games. "
                "_rec_* helper lists are popped before model input."
            ),
        },
        {
            "step": "3_projection",
            "symbol": "api.py::model_predict('batter_total_bases')",
            "detail": (
                "XGBoost Booster loaded from MODEL_DIR/batter_total_bases.json "
                "with batter_total_bases_columns.json. Regression output = "
                "projected total bases for the game."
            ),
        },
        {
            "step": "4_line",
            "symbol": "api.py::STANDARD_LINES / over_under",
            "detail": (
                "FanDuel line when present; otherwise standard fallback "
                "line = 1.5. Pick is OVER the line."
            ),
        },
        {
            "step": "5_base_probability",
            "symbol": "api.py::prob_over",
            "detail": (
                "Poisson over-probability: 1 - poisson_cdf(floor(line), proj)."
            ),
        },
        {
            "step": "6_bvp_nudge",
            "symbol": "api.py::build_batter_prop_picks._bvp_nudge",
            "detail": (
                "power_flag from bvp.power_flag: 'power' => +0.04, "
                "'weak' => -0.03. No nudge otherwise."
            ),
        },
        {
            "step": "7_monte_carlo_overlay",
            "symbol": "api.py::_hitter_mc_over_probability",
            "detail": (
                "hitter_poisson_mc_v1, 250000 sims, stable per-pick seed. "
                "raw_mc_prob is the final gate probability "
                "(final_gate_version 8.19B_raw_mc_not_bvp_adjusted). "
                "append_prob = max(raw_mc_prob, legacy_adjusted_prob)."
            ),
        },
        {
            "step": "8_board_governance",
            "symbol": "api.py::govern_hitter_board / CANDIDATE_ONLY_HITTER_PROPS",
            "detail": (
                "batter_total_bases is in CANDIDATE_ONLY_HITTER_PROPS and "
                "PROBATIONARY_MARKETS. Capped at watchlist_prediction; "
                "watchlist threshold ~0.63. Never official."
            ),
        },
        {
            "step": "9_grading",
            "symbol": "api.py::get_actual_stat -> ('hitting','totalBases')",
            "detail": (
                "Graded against MLB StatsAPI hitting.totalBases for the date."
            ),
        },
    ],
    "live_record_at_freeze": {
        "source": "docs/record.json by_prop.batter_total_bases",
        "hits": 44, "total": 100, "hit_rate_pct": 44.0,
        "note": "Underwater; below coin-flip. This is why TB is candidate-only.",
    },
    "offline_target_derivation": (
        "total_bases = hits + doubles + 2*triples + 3*home_runs, "
        "computed from hr_model.sqlite::batter_games."
    ),
    "promotion_doctrine": (
        "No production change to TB from this script. The champion is frozen "
        "here so a challenger can later be isolated, validated on a locked "
        "out-of-time holdout, and pass a pre-registered formal gate before any "
        "implementation-parity patch or promotion. Same discipline as K/hits."
    ),
}


# =============================================================================
# Helpers
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def discover_source_file(name, root):
    for cand in (
        Path(root) / name if root else None,
        Path.cwd() / name,
        Path("/opt/render/project/src") / name,
        Path("/opt/render/project/src/src") / name,
    ):
        if cand and cand.exists():
            return cand.resolve()
    return None


def line_hits(text, patterns):
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        for p in patterns:
            if p in line:
                out.append({"line": i, "pattern": p, "text": line.strip()[:200]})
                break
    return out


def ast_inventory(path):
    """Functions/assignments in a source file that reference TB patterns."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "functions": [], "assignments": []}
    lines = text.splitlines()

    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = "\n".join(lines[node.lineno - 1: getattr(node, "end_lineno", node.lineno)])
            if any(p in body for p in TB_PATTERNS):
                funcs.append({"name": node.name, "lineno": node.lineno})

    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if any(p in name for p in ("TOTAL", "TB", "CANDIDATE", "PROBATION", "STANDARD_LINES", "PROP_MODEL")):
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    val = "<non-literal>"
                assigns.append({"name": name, "lineno": node.lineno, "value": val})
    return {"functions": funcs, "assignments": assigns}


def discover_sqlite_files():
    found = []
    for base in (DATA_ROOT, Path.cwd()):
        if not base.exists():
            continue
        for p in base.rglob("*.sqlite"):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            found.append(str(p))
            if len(found) >= 20:
                break
    return sorted(set(found))


def inspect_tb_target(db_path):
    """Confirm TB target derivability from batter_games and report coverage."""
    info = {"db": db_path, "batter_games_present": False}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception as e:
        info["error"] = f"open failed: {e}"
        return info
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        info["tables"] = sorted(tables)
        if "batter_games" not in tables:
            return info
        info["batter_games_present"] = True

        cur.execute("PRAGMA table_info(batter_games)")
        cols = {r[1] for r in cur.fetchall()}
        needed = {"hits", "doubles", "triples", "home_runs", "game_date", "batter_id"}
        info["has_columns_for_tb"] = needed.issubset(cols)
        info["missing_columns"] = sorted(needed - cols)
        if not needed.issubset(cols):
            return info

        tb_expr = "(hits + doubles + 2*triples + 3*home_runs)"
        cur.execute(f"""
            SELECT substr(game_date,1,4) AS yr,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT batter_id) AS batters,
                   ROUND(AVG({tb_expr}),3) AS mean_tb,
                   SUM(CASE WHEN {tb_expr} >= 2 THEN 1 ELSE 0 END) AS over_1p5
            FROM batter_games
            WHERE at_bats IS NOT NULL
            GROUP BY yr ORDER BY yr
        """)
        info["tb_coverage_by_year"] = [
            {"year": r[0], "rows": r[1], "batters": r[2],
             "mean_tb": r[3], "over_1_5": r[4],
             "over_1_5_rate": round(r[4] / r[1], 4) if r[1] else None}
            for r in cur.fetchall()
        ]
    except Exception as e:
        info["error"] = f"query failed: {e}"
    finally:
        conn.close()
    return info


def scan_artifacts():
    arts = []
    for base in (DATA_ROOT, Path.cwd() / "docs", Path.cwd() / "candidate_logs"):
        if not base.exists():
            continue
        for p in base.rglob("*.json"):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            try:
                if "total_bases" in p.read_text(encoding="utf-8", errors="replace")[:200000]:
                    arts.append({"path": str(p), "size": p.stat().st_size})
            except Exception:
                pass
            if len(arts) >= 40:
                break
    return arts


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="source root containing api.py etc.")
    args = ap.parse_args()

    print("TOTAL_BASES_CHAMPION_RECONSTRUCTION_A", flush=True)
    print("=====================================", flush=True)
    print(f"generated_at_utc={now_utc()}", flush=True)

    # 1. Source inventory
    sources = {}
    for name in TB_SOURCE_FILES:
        path = discover_source_file(name, args.root)
        if not path:
            sources[name] = {"found": False}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sources[name] = {
            "found": True,
            "path": str(path),
            "match_lines": line_hits(text, TB_PATTERNS)[:120],
            "ast": ast_inventory(path),
        }

    # 2. SQLite / target availability
    dbs = discover_sqlite_files()
    target = None
    for db in dbs:
        probe = inspect_tb_target(db)
        if probe.get("batter_games_present"):
            target = probe
            break

    # 3. Artifacts
    artifacts = scan_artifacts()

    report = {
        "script": "TOTAL_BASES_CHAMPION_RECONSTRUCTION_A",
        "generated_at_utc": now_utc(),
        "frozen_champion_spec": FROZEN_CHAMPION_SPEC,
        "source_inventory": sources,
        "sqlite_files_found": dbs,
        "tb_target_availability": target,
        "tb_artifacts": artifacts,
        "next_rung": (
            "total_bases_clean_baseline_a.py: build strict-D-1 TB dataset from "
            "batter_games (target = hits+doubles+2*triples+3*home_runs), then "
            "reconstruct the champion projection offline for holdout scoring."
        ),
    }

    # Console summary
    print("\nFROZEN CHAMPION SPEC (summary)")
    print("------------------------------")
    for s in FROZEN_CHAMPION_SPEC["pipeline"]:
        print(f"  {s['step']:22s} {s['symbol']}")
        if s.get("known_risk"):
            print(f"      RISK: {s['known_risk'].splitlines()[0]} ...")
    rec = FROZEN_CHAMPION_SPEC["live_record_at_freeze"]
    print(f"  live record: {rec['hits']}/{rec['total']} = {rec['hit_rate_pct']}%")

    print("\nSOURCE INVENTORY")
    print("----------------")
    for name, info in sources.items():
        if info.get("found"):
            nf = len(info["ast"].get("functions", []))
            nm = len(info["match_lines"])
            print(f"  {name}: {nm} match lines, {nf} TB-driving functions")
        else:
            print(f"  {name}: NOT FOUND")

    print("\nTB TARGET AVAILABILITY")
    print("----------------------")
    if not target:
        print("  batter_games not found in any sqlite (expected on Render only).")
    else:
        print(f"  db: {target['db']}")
        print(f"  derivable: {target.get('has_columns_for_tb')}")
        for row in target.get("tb_coverage_by_year", []):
            print(f"    {row['year']}: {row['rows']} rows, "
                  f"mean_tb={row['mean_tb']}, over1.5={row['over_1_5_rate']}")

    # Write outputs (fall back to cwd if /data not writable)
    out_dir = OUT_DIR_PRIMARY if OUT_DIR_PRIMARY.parent.exists() else Path.cwd()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = Path.cwd()
    out_json = out_dir / "total_bases_champion_reconstruction_a_report.json"
    out_txt = out_dir / "total_bases_champion_reconstruction_a_report.txt"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_txt.write_text(_text_report(report), encoding="utf-8")

    print(f"\nOUTPUTS\n  {out_json}\n  {out_txt}", flush=True)
    print("\nCHAMPION FROZEN. No production state changed.", flush=True)
    return 0


def _text_report(report):
    lines = ["TOTAL_BASES_CHAMPION_RECONSTRUCTION_A", "=" * 38, ""]
    lines.append(f"generated_at_utc: {report['generated_at_utc']}")
    lines.append("")
    lines.append("FROZEN CHAMPION SPEC")
    lines.append("--------------------")
    for s in report["frozen_champion_spec"]["pipeline"]:
        lines.append(f"[{s['step']}] {s['symbol']}")
        lines.append(f"    {s['detail']}")
        if s.get("known_risk"):
            lines.append(f"    RISK: {s['known_risk']}")
    rec = report["frozen_champion_spec"]["live_record_at_freeze"]
    lines.append(f"\nlive record at freeze: {rec['hits']}/{rec['total']} = {rec['hit_rate_pct']}%")
    lines.append("")
    lines.append("NEXT RUNG")
    lines.append("---------")
    lines.append(report["next_rung"])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

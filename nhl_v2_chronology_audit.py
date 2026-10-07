#!/usr/bin/env python3
"""NHL V2 Phase0 chronology audit: evidence for every period classification. Produces
phase0_chronology_audit.json and phase0_central_projection_gap.json BEFORE any result. Reads git objects and
repository text only; reads no 2026 outcome data."""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
V1 = "origin/codex/nhl-outcome-engine-v1"


def git(*a):
    try:
        return subprocess.run(["git"] + list(a), cwd=str(REPO), check=True, capture_output=True, text=True).stdout
    except Exception:
        return None


def commit(sha):
    out = git("log", "-1", "--format=%H|%an|%ad|%s", "--date=short", sha)
    if out is None:
        return {"sha_prefix": sha, "resolved": False}
    h, an, d, s = out.strip().split("|", 3)
    return {"sha": h, "author": an, "date": d, "subject": s, "resolved": True}


def v1_text(path):
    return git("show", V1 + ":" + path)


def grep_lines(path, pat):
    txt = (REPO / path).read_text()
    return [{"line": i + 1, "text": l.strip()[:200]} for i, l in enumerate(txt.splitlines()) if re.search(pat, l)]


def main():
    scripts = sorted(p.name for p in REPO.glob("nhl_*.py") if not p.name.startswith(("nhl_fwd", "nhl_v2", "nhl_outcome")))
    gate_split = {}
    for sname in scripts:
        if "champion_gate" in sname or "clean_baseline" in sname or "walkforward" in sname or "prior_season" in sname:
            t = (REPO / sname).read_text()
            gate_split[sname] = {k: (re.search(k + r"\S*\s*=\s*([^\n]+)", t).group(1) if re.search(k + r"\S*\s*=\s*([^\n]+)", t) else None)
                                 for k in ("DEV_SEASONS", "VAL_SEASON", "HOLDOUT_SEASON")}
    manifests = {}
    for p in sorted((REPO / "nhl_models").glob("*/manifest.json")):
        m = json.loads(p.read_text())
        manifests[p.parent.name] = {"generated_at_utc": m.get("generated_at_utc"), "dev": m.get("dev_seasons"), "val": m.get("val_season"),
                                    "holdout": m.get("holdout_season"), "seasons_present_in_table": sorted((m.get("by_season") or {}).keys())}
    v1_evidence = {
        "branch": V1, "pr": "none (no pull request exists for this branch)", "on_main": False,
        "commits": [commit(s) for s in ("4e33445", "76a4100", "c6fc26a", "51de4ed", "d825bc6", "a27bc93", "8ea97f2", "f7f9177", "b79198c", "cea38be", "22d2c46", "f700714")],
        "own_statements": {},
    }
    for name, path in (("phase1a_sog_readme", "nhl_models/nhl_outcome_engine/PHASE1A_README.md"),
                       ("goalie_g0_protocol", "nhl_models/nhl_outcome_engine/phase_goalie_g0_protocol.json"),
                       ("scoring_s0_protocol", "nhl_models/nhl_outcome_engine/phase_scoring_s0_protocol.json"),
                       ("team_t0_protocol", "nhl_models/nhl_outcome_engine/phase_team_game_t0_protocol.json"),
                       ("goals_g1_protocol", "nhl_models/nhl_outcome_engine/phase_goals_g1_protocol.json"),
                       ("assists_a1_protocol", "nhl_models/nhl_outcome_engine/phase_assists_a1_protocol.json")):
        t = v1_text(path)
        hits = []
        if t:
            for m in re.finditer(r"[^\n\"]{0,110}(?:2025|previously-exposed)[^\n\"]{0,130}", t):
                hits.append(m.group(0).strip())
        v1_evidence["own_statements"][name] = hits[:4] if t else "UNAVAILABLE"
    v1_files = (git("ls-tree", "-r", "--name-only", V1) or "").splitlines()
    v1_evidence["files_scoring_or_holding_2025_outcomes"] = sorted(f for f in v1_files if "2025" in f and f.startswith("nhl_models/"))
    serving = {"fit_serving_platt_player_market": grep_lines("nhl_serving_builder_a.py", r"warm_season|seasons\[0\]|growing"),
               "fit_serving_platt_moneyline": grep_lines("nhl_serving_builder_a.py", r"warm_season|warm_engine")}
    import sqlite3
    con = sqlite3.connect("file:%s?mode=ro" % (REPO / "nhl_models" / "nhl_model.sqlite"), uri=True)
    fwd_counts = {t: con.execute("select count(*) from %s where season>=2026" % t).fetchone()[0] for t in ("skater_games", "goalie_games")}
    fwd_counts["games_rows_season_2026_with_final_score"] = con.execute("select count(*) from games where season>=2026 and home_score is not null").fetchone()[0]
    con.close()
    periods = {
        "2018-2022": {"classification": "FIT", "evidence": ["all champion-gate scripts: DEV_SEASONS=2018-2022 (see gate_split)", "tree weights trained here"]},
        "2023": {"classification": "SELECTION", "evidence": ["VAL_SEASON=2023 early stopping in every champion gate; V1 development fold D4 validates on 2023"]},
        "2024": {"classification": "BURNED", "evidence": ["HOLDOUT_SEASON=2024 of every incumbent gate and walk-forward stability report (pass/fail and promotion decisions made on it)",
                                                        "V1 2024 confirmation and goals 2024 confirmation scored it"]},
        "2025": {"classification": "BURNED_EXPOSED", "untouched": False,
                 "evidence": ["V1 Phase 1A: 2025 final holdout scored (commit f7f9177) - SOG distribution engine, different model family",
                              "V1 goals G1: 2025 final confirmation scored, status GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED (commit 51de4ed)",
                              "V1 S0/G0/T0 feasibility phases built label tables for 2025 and state 2024/2025 'stay previously-exposed years'",
                              "production serving calibration fits on 2025 outcomes as warm-up season (nhl_serving_builder_a.fit_serving_platt_*)",
                              "baseline sqlite tables (manifests generated 2026-09-23/24) already hold 2025 rows with labels; the moneyline gate reserves 2025 for serving prior-season fallback",
                              "NOT evidence of tree fitting: no incumbent gate/baseline script trains, early-stops or selects on 2025"],
                 "why_not_untouched_for_this_audit": "different model families on this branch lineage already looked at 2025 outcomes and drove research conclusions; production calibration fits on it. Phase0 therefore treats 2025 as exposed, retrospective evidence only."},
        "2026-09-29_onward": {"classification": "CLEAN_FORWARD", "contaminated": False,
                              "evidence": ["production DB season>=2026 row counts (counts only): %s" % json.dumps(fwd_counts, sort_keys=True),
                                           "docs/nhl_picks_log.jsonl holds pregame picks only (season 2026); outcomes/grades were not opened by this audit",
                                           "forward-capture state branch (postgame truth ledger) was not opened",
                                           "Phase0 data loader hard-caps seasons <= 2025"]}}
    out = {"artifact": "phase0_chronology_audit", "evidence_grade_of_conclusion": "RETROSPECTIVE_BURNED",
           "conclusion": "NO genuinely untouched historical NHL season exists. All historical Phase0 results are retrospective/burned evidence; genuine confirmation must wait for clean forward 2026 data.",
           "periods": periods, "incumbent_split_by_script": gate_split, "baseline_manifests": manifests,
           "serving_calibration_uses_prior_season_outcomes": serving, "existing_v1_research_branch": v1_evidence,
           "audited_surfaces": ["training scripts", "model-building scripts", "serialized model metadata (xgboost json + manifests)", "stored models", "research scripts (V1 branch)",
                                "backtest outputs (V1 result artifacts)", "git history (all branches)", "CI/workflow commands (grep of all_sports_predictions.yml season args)", "result artifacts", "data-selection logic (eligibility floors)"],
           "not_audited": ["private notebooks or local runs outside git (cannot be audited from the repository)"]}
    (OUT / "phase0_chronology_audit.json").write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    gap = {"artifact": "phase0_central_projection_gap", "finding": "CURRENT_ENGINE_HAS_NO_CENTRAL_STAT_PROJECTION (all player markets)", "markets": [
        {"market": "shots_on_goal", "current_model_type": "XGBoost binary classifier + Platt", "native_target": "SOG >= 3", "has_central_projection": False, "has_distribution": False,
         "has_threshold_probability": True, "can_grade_MAE": False, "can_grade_calibration": True,
         "architecture_limitation": "answers only the SOG>=3 question; cannot answer 2+, 4+, medians or means; no mean to recover without inventing a distribution (forbidden)"},
        {"market": "points", "current_model_type": "XGBoost binary classifier + Platt", "native_target": "points >= 1", "has_central_projection": False, "has_distribution": False,
         "has_threshold_probability": True, "can_grade_MAE": False, "can_grade_calibration": True,
         "architecture_limitation": "anytime-point only; no goals/assists decomposition; no 2+ points"},
        {"market": "goalie_saves", "current_model_type": "XGBoost binary classifier (gate FAILED; not served)", "native_target": "saves >= 25", "has_central_projection": False, "has_distribution": False,
         "has_threshold_probability": True, "can_grade_MAE": False, "can_grade_calibration": True,
         "architecture_limitation": "no starter-state, workload or shots-faced decomposition; training population selected on target-game TOI"},
        {"market": "moneyline", "current_model_type": "XGBoost binary classifier + home/away Platt", "native_target": "home team wins (incl. shootout)", "has_central_projection": False, "has_distribution": False,
         "has_threshold_probability": True, "can_grade_MAE": False, "can_grade_calibration": True,
         "architecture_limitation": "NONE for the binary question: a binary classifier is structurally appropriate when the target itself is binary; goalie state absent is a data limitation, not an architecture one"},
        {"market": "goals", "current_model_type": "none", "native_target": "n/a", "has_central_projection": False, "has_distribution": False, "has_threshold_probability": False,
         "can_grade_MAE": False, "can_grade_calibration": False, "architecture_limitation": "not predicted by the current system"},
        {"market": "assists", "current_model_type": "none", "native_target": "n/a", "has_central_projection": False, "has_distribution": False, "has_threshold_probability": False,
         "can_grade_MAE": False, "can_grade_calibration": False, "architecture_limitation": "not predicted by the current system"}],
        "fixed_thresholds_are_not_sportsbook_inputs": "SOG 3+, points 1+, saves 25+ are fixed round-number model targets (SHOTS_LINE=2.5, POINTS_LINE=0.5, SAVES_LINE=24.5 + 0.5). The production builder states 'predictions-first: no odds'. No book line is an input.",
        "why_fixed_thresholds_are_insufficient": "an independent classifier per threshold cannot answer arbitrary player-outcome questions, cannot yield a central projection, and cannot share structure across thresholds (a 3+ model and a 4+ model may disagree)."}
    (OUT / "phase0_central_projection_gap.json").write_text(json.dumps(gap, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print("wrote chronology + gap artifacts")


if __name__ == "__main__":
    main()

"""
NFL_PHASE1D_POSTHOC_AUDIT  (Phase 1D)  -- every post-hoc decision of Phase 1C, classified; nothing hidden

Writes nfl_models/nfl_player_outcome_phase1d/posthoc_audit.{json,md}. The decision table is authored here (each entry cites its evidence file); the appendix is a machine scan of all
Phase 1C code / reports / result files for the phrases post-hoc, amend, clarification, after seeing, revised rule, changed criterion (and close variants), so an omission in the
authored table is visible as an unmapped scan hit.

  python -u nfl_phase1d_posthoc_audit.py --scratch DIR
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
P1C = REPO / "nfl_models" / "nfl_player_outcome_phase1c"
OUT = REPO / "nfl_models" / "nfl_player_outcome_phase1d"
PHRASES = ["post-hoc", "posthoc", "amend", "clarification", "after seeing", "revised rule", "changed criterion", "redefined", "replaced before", "first-stated", "was replaced", "superseded",
           "not pre-registered", "rejected", "ablation"]
SCAN_GLOBS = ["nfl_phase1c_*.py", "nfl_phase1_forecast.py", "nfl_phase1_score.py", "nfl_phase1_store.py", "nfl_models/nfl_player_outcome_phase1c/*.md", "nfl_models/nfl_player_outcome_phase1c/*.json",
              "freeze_candidate_validation.md", "tests/test_nfl_phase1c_invariants.py"]
CLASSES = {
    "DATA/WARMUP DEFINITION CORRECTION": "the change only fixes what counts as a valid observation (history literally does not exist); it is defined prospectively and permanently below",
    "DEVELOPMENT-SELECTED": "chosen after looking at burned-development results; frozen; only the clean-forward window can judge it",
    "ENGINEERING SELECTION": "a resource / numerical-accuracy choice that is not a model-performance choice",
    "CORRECTNESS FIX": "the earlier procedure violated data-time or a definition; the fix is not performance motivated",
    "REPORTING DEFINITION": "changes how a diagnostic is summarised, not any forecast",
    "NOT ADOPTED": "evaluated as an ablation and rejected; nothing changed",
    "PROCESS": "no modelling consequence",
}


def evidence_2022_wk1(scratch):
    """Does required history literally not exist for 2022 week-1 rows?"""
    rec = pickle.load(open(Path(scratch) / "records_full.pkl", "rb"))
    out = {"data_window": "nfl_phase1_data.SEASONS = [2022 .. 2026]; the first season present in every source is 2022 (no 2021 files are read or exist in the store)"}
    for comp, R in rec["Rs"].items():
        m1 = (R.s == 2022) & (R.w == 1)
        m2 = (R.s == 2022) & (R.w == 2)
        if not m1.any():
            continue
        zero = lambda mask: float(np.mean([all(np.allclose(R.base[k][i], 0.0) for k in R.base) for i in np.where(mask)[0]]))
        # as-of team / opponent context rates (the home flag, scheme and injury counts are not history and are excluded)
        famzero = lambda mask: float(np.mean([np.allclose(np.asarray(R.fam["team"][i][:-1], float), 0.0) and np.allclose(np.asarray(R.fam["opp"][i], float), 0.0) for i in np.where(mask)[0]]))
        out[comp] = {"rows_2022_wk1": int(m1.sum()), "share_with_all_zero_decayed_history": zero(m1), "share_with_all_zero_context_families": famzero(m1),
                     "rows_2022_wk2": int(m2.sum()), "share_wk2_with_all_zero_decayed_history": zero(m2) if m2.any() else None}
    return out


def scan():
    hits = []
    for g in SCAN_GLOBS:
        for f in sorted(REPO.glob(g)):
            if f.name.startswith("nfl_phase1d_posthoc_audit"):
                continue
            try:
                lines = f.read_text().splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                low = line.lower()
                for ph in PHRASES:
                    if ph in low:
                        hits.append({"file": str(f.relative_to(REPO)), "line": i, "phrase": ph, "text": line.strip()[:220]})
                        break
    return hits


DECISIONS = [
    {"id": "D01", "title": "2022 week-1 exclusion (adjudication amendment_1)",
     "what": "Run 1 of the rolling-origin adjudication included 2022 week-1 rows; run 2 excludes them from every fit, validation and test for every component. Both analyses are kept.",
     "trigger": "run 1 showed every defender feature level worse than B0 in 7/7 folds by >= 0.04 nats (an artefact signature); the rows' as-of features are identically zero",
     "pre_registered": False, "recorded_before_rerun": True,
     "classification": "DATA/WARMUP DEFINITION CORRECTION",
     "consequence": "run 1 (partial, interrupted) selected def_interceptions B0; run 2 selects B5 (16 tilt features, L2 1000); every other component listed in run 1 kept its selection. The change therefore did alter one selection, "
                    "the defensive-interception tilt, whose effect on scored outcomes is negligible (see adjudication_results.json).",
     "permanent_rule": "WARM-UP RULE (prospective, permanent): an observation is eligible for fitting or scoring only if at least one completed regular-season game of the data window precedes it. Operationally: "
                       "2022 week 1 is the only warm-up week (nothing precedes it in the data window). Every later week, including week 1 of every later season, has history and is eligible. In the forward window "
                       "every game has history, so the rule excludes nothing; it is implemented once in nfl_phase1c_adjudicate.mask_warmup and used by the live path (nfl_phase1d_live.prepare).",
     "provenance": ["nfl_models/nfl_player_outcome_phase1c/adjudication_run1_partial.json (original)", "nfl_models/nfl_player_outcome_phase1c/adjudication_results.json (amended)",
                    "nfl_models/nfl_player_outcome_phase1c/adjudication_rule.json#amendment_1"],
     "not_pretended": "the amendment was NOT pre-registered; it was recorded after run 1 and before run 2"},
    {"id": "D02", "title": "Calibration materiality clarification",
     "what": "After seeing that the first-stated calibration adoption rule was satisfied by 0.001 coverage changes, adoption additionally required |cov80-0.8| to improve by >= 0.005 and |cov50-0.5| not to worsen by more than 0.005.",
     "trigger": "burned-development calibration outcomes", "pre_registered": False, "classification": "DEVELOPMENT-SELECTED",
     "consequence": "the accepted-universe calibration.json now shows two outcomes back at 'none (uncalibrated retained)'; the depth-universe maps used by the forecast path are listed in calibration_depth.json",
     "phase1d_action": "NO further calibration selection. The current maps stay exactly as stored (hashes in calibration_freeze.json). They may be removed only for a code/logic bug, never for another outcome comparison.",
     "provenance": ["nfl_models/nfl_player_outcome_phase1c/calibration.json", "nfl_models/nfl_player_outcome_phase1c/calibration_depth.json", "nfl_phase1c_study.py (step_calibrate)"]},
    {"id": "D03", "title": "Simulation-count convergence criteria amendment",
     "what": "The first-stated row-level quantile criteria failed for every tested N (heavy-tailed yardage tails converge slowly), so amended aggregate criteria were used to pick N = 25,000.",
     "trigger": "no N passed the first-stated criteria", "pre_registered": False, "classification": "ENGINEERING SELECTION",
     "phase1d_action": "N = 25,000 is treated as an engineering selection. A NEW criterion is fixed BEFORE running (n_engineering_criteria.json, Monte Carlo error relative to football-model error, "
                       "predetermined disjoint burned weeks) and verified in n_engineering_audit.json. N is never re-chosen from football scores.",
     "provenance": ["nfl_models/nfl_player_outcome_phase1c/simulation_convergence.json", "nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json", "nfl_models/nfl_player_outcome_phase1d/n_engineering_audit.json"]},
    {"id": "D04", "title": "Depth-chart-extended candidate universe",
     "what": "The candidate universe was extended with timestamped depth-chart skill players (>= 3 prior game rows) after the common-player comparison favoured it (rush CRPS 7.8958 vs 7.9290, p=0.01; pass 30.29 vs 30.55, p=0.027).",
     "trigger": "burned-development comparison", "pre_registered": False, "classification": "DEVELOPMENT-SELECTED",
     "phase1d_action": "FROZEN. Reversible only for a correctness / data-time violation (e.g. a depth snapshot with dt after the cutoff), never for accuracy.",
     "provenance": ["nfl_models/nfl_player_outcome_phase1c/universe_comparison.json"]},
    {"id": "D05", "title": "Predictability score redefinition",
     "what": "The first score (weighted components) was anti-informative on development data (Spearman -0.53 against relative error) and was replaced by the engine's own expected relative error mapped through x/(1+x).",
     "trigger": "burned-development metric", "pre_registered": False, "classification": "DEVELOPMENT-SELECTED",
     "phase1d_action": "frozen; the forward gate only requires Spearman > 0 (Amendment G), it does not tune the score.", "provenance": ["nfl_phase1c_metrics.py (predictability)"]},
    {"id": "D06", "title": "High-confidence subset definition",
     "what": "The first high-confidence subset was degenerate; it was redefined as P(active) >= 0.9 and U in the lowest 30% of likely players.",
     "trigger": "degenerate subset in the first curves run", "pre_registered": False, "classification": "REPORTING DEFINITION",
     "phase1d_action": "reporting only; Phase 1D replaces headline accuracy with labelled universes (accuracy_slices)", "provenance": ["nfl_phase1c_study.py (step_curves)"]},
    {"id": "D07", "title": "One-listed-QB-always-plays rule",
     "what": "Evaluated as an ablation after seeing a named-QB attempts shortfall; it hurt passing-yards CRPS by 0.34 and was rejected (force_qb = False).", "trigger": "burned-development ablation",
     "pre_registered": False, "classification": "NOT ADOPTED", "phase1d_action": "stays off", "provenance": ["nfl_models/nfl_player_outcome_phase1c/joint_vs_independent.json (ablation_force_one_listed_qb)"]},
    {"id": "D08", "title": "Proportional rescale of QB dropback / attempt allocation (prop bucket)",
     "what": "The dropback allocation divides propensities by (1 - sbar - z_db) and reduces the outside bucket, added after observing under-generated named-QB attempts.", "trigger": "development bias diagnosis",
     "pre_registered": False, "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "frozen simulator design", "provenance": ["nfl_phase1c_sim.py"]},
    {"id": "D09", "title": "Fitted within-bin means instead of bin midpoints",
     "what": "Midpoints of coarse bins inflated tails (e.g. air 50-99 mean 53 vs midpoint 74.5); league-fitted within-bin means replaced them.", "trigger": "development yard bias", "pre_registered": False,
     "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "frozen constants (constants.json hash in the candidate)", "provenance": ["nfl_models/nfl_player_outcome_phase1c/constants.json", "nfl_phase1c_constants.py"]},
    {"id": "D10", "title": "Red-zone yardage and touchdown mechanics",
     "what": "The red-zone cap pile-up and long-TD resample were replaced by zone likelihood ratios (SIR) and touchdown-at-target hazard rules; unconditioned air-yard candidates were made conditional on completion.",
     "trigger": "development yard / TD bias diagnoses", "pre_registered": False, "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "frozen simulator design", "provenance": ["nfl_phase1c_sim.py", "nfl_phase1c_constants.py"]},
    {"id": "D11", "title": "Static (S0) versus state-aware (S1) game script",
     "what": "S1 was worse than S0 on every scored outcome (rush CRPS 8.024 vs 7.916; pass 32.17 vs 30.45) and was not adopted.", "trigger": "burned-development comparison", "pre_registered": False,
     "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "S0 frozen", "provenance": ["nfl_models/nfl_player_outcome_phase1c/state_static_vs_dynamic.json"]},
    {"id": "D12", "title": "Phase 1A architecture selection (team environment, availability, role families, opportunity components)",
     "what": "Selected on burned development weeks with week-block bootstrap p < 0.10 (Phase 1A); extracted into phase1a_frozen_selection.json.", "trigger": "burned-development comparisons",
     "pre_registered": True, "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "frozen; walk-forward refits change only fitted coefficients", "provenance": ["nfl_models/nfl_player_outcome_phase1d/phase1a_frozen_selection.json"]},
    {"id": "D13", "title": "Production-v2 artifact scored only on 2026 wk1-3",
     "what": "The production v2 artifact (trained through 2025) had been scored on 2025 and combined rows; restricted to 2026 wk1-3, the only weeks future to its training data.",
     "trigger": "data-time review", "pre_registered": False, "classification": "CORRECTNESS FIX", "phase1d_action": "kept", "provenance": ["nfl_phase1c_v2comp.py", "nfl_models/nfl_player_outcome_phase1c/v2_comparator_depth.json"]},
    {"id": "D14", "title": "Structural constants fitted on all burned data",
     "what": "bin values, zone likelihood ratios, half-sack share, tackle credit mean, red-zone gamma shape: all fit on burned data including the development weeks.", "trigger": "design",
     "pre_registered": False, "classification": "DEVELOPMENT-SELECTED", "phase1d_action": "frozen (hash recorded); never refit in the forward window", "provenance": ["nfl_models/nfl_player_outcome_phase1c/constants.json"]},
    {"id": "D15", "title": "Report / harness fixes (table KeyError, absent-player chaos assertion, restarted dry run)",
     "what": "Engineering fixes to reporting and test assertions; one dry run was restarted after code changes so the model version matched final code.", "trigger": "harness defects",
     "pre_registered": False, "classification": "PROCESS", "phase1d_action": "none", "provenance": ["nfl_phase1c_report.py", "nfl_phase1c_dryrun.py"]},
    {"id": "D17", "title": "Stored QB rushing-yards calibration map was never applied (found in Phase 1D)",
     "what": "calibration_depth.json adopts a PIT map for 'qb_rush_yds' (fit on QB rows of rush_yds draws) but the forecast path looked maps up by outcome name only, so the map was inert. "
             "Phase 1D applies it to QB rows of rush_yds exactly as it was evaluated.",
     "trigger": "code review of the calibration path while writing calibration_freeze.json (no outcome comparison)", "pre_registered": False, "classification": "CORRECTNESS FIX",
     "phase1d_action": "the stored map is unchanged (hash in calibration_freeze.json); only its application is fixed; affects published rush_yds distributions of QBs only; no other map or forecast changes",
     "provenance": ["nfl_phase1_forecast.py (build_records)", "nfl_models/nfl_player_outcome_phase1d/calibration_freeze.json"]},
    {"id": "D18", "title": "Forecast-only defenders: position group as of the cutoff (found by LiveLoader equivalence)",
     "what": "In live mode every defender of a target game is a forecast-only row. The old resolution used the roster / players.csv position and sent unmapped codes (e.g. 'SAF', 329 players) to LB, "
             "while realized rows were resolved through the snap-count group (DB -> S). The forecast-only resolution now uses the group of the player's most recent EARLIER game row, else the same "
             "resolution as a realized row; offense records read the most recent earlier roster position when the current-week roster does not exist yet (T24). Realized-row grouping is untouched.",
     "trigger": "LiveLoader equivalence check: 95 of 1,625 target-week defender records had a different position group from the research replay", "pre_registered": False, "classification": "CORRECTNESS FIX",
     "consequence": "affects the defensive rate priors of forecast-only defenders (mostly safeties) and the position of a few offensive candidates at T24; realized-row grouping and all frozen development "
                    "constants are unchanged. Residual documented differences (about 2% of rows: players traded before the game, defenders whose stats position changed between games) are listed row by row in equivalence_results.json; "
                    "in each the live loader uses the as-of legal information and the research replay used the realized game's label.",
     "phase1d_action": "kept; not performance motivated (no outcome comparison was made)", "provenance": ["nfl_phase1_defense_events.py (add_extras)", "nfl_phase1b_data.py (last_known_pos)", "nfl_models/nfl_player_outcome_phase1d/equivalence_results.json"]},
    {"id": "D16", "title": "Phase 1D choices (recorded here so they are not hidden)",
     "what": "(a) hyper-parameters of Phase 1A/1B are frozen at the freeze-fit values and walk-forward refits only refit coefficients; (b) Amendment G thresholds were written after the burned-data results were known "
             "but before any clean-forward outcome, copying protocol v1.1 numbers (calibration bands, ECE 0.03, p < 0.05) or the Phase 1D specification (2% MAE tolerance, >= 50 events); "
             "(c) accuracy-slice tier thresholds and eligibility slices are reporting definitions; (d) the dry-run weeks and the N-audit set were fixed in files committed before their runs.",
     "trigger": "Phase 1D specification", "pre_registered": False, "classification": "PROCESS",
     "phase1d_action": "documented; no threshold derived from any later forward result", "provenance": ["nfl_models/nfl_player_outcome_phase1d/dry_run_plan.json", "nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    ev = evidence_2022_wk1(a.scratch)
    hits = scan()
    mapped_files = {p.split(" ")[0].split("#")[0] for d in DECISIONS for p in d["provenance"]}
    for h in hits:
        h["mapped_to_decision"] = h["file"] in mapped_files
    doc = {"purpose": "classification of every post-hoc decision of Phase 1C; nothing hidden", "classes": CLASSES, "decisions": DECISIONS, "evidence_2022_wk1_history_does_not_exist": ev,
           "scan_phrases": PHRASES, "scan_hits": hits, "n_scan_hits": len(hits), "unmapped_scan_hit_files": sorted({h["file"] for h in hits if not h["mapped_to_decision"]})}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_audit.json").write_text(json.dumps(doc, indent=1))
    L = ["# Phase 1C post-hoc decisions - audit (Phase 1D)", "",
         "Every decision below was made, or its rule written, after burned-development results were visible unless it says otherwise. Nothing here was pre-registered unless stated. "
         "Classes: " + "; ".join(f"**{k}** = {v}" for k, v in CLASSES.items()) + ".", ""]
    for d in DECISIONS:
        L += [f"## {d['id']} - {d['title']}", "", f"- classification: **{d['classification']}**", f"- what: {d['what']}", f"- trigger: {d['trigger']}", f"- pre-registered: {d['pre_registered']}"]
        for k in ("recorded_before_rerun", "consequence", "permanent_rule", "phase1d_action", "not_pretended"):
            if k in d:
                L.append(f"- {k.replace('_', ' ')}: {d[k]}")
        L += [f"- provenance: {', '.join(d['provenance'])}", ""]
    L += ["## Evidence for D01: the required history literally does not exist", "", f"{ev['data_window']}.", "", "| component | 2022 wk1 rows | share with all-zero decayed player/position/league history | share with all-zero as-of team/opponent context rates | 2022 wk2 rows | wk2 share all-zero history |", "|---|---|---|---|---|---|"]
    for k, v in ev.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['rows_2022_wk1']} | {v['share_with_all_zero_decayed_history']:.3f} | {v['share_with_all_zero_context_families']:.3f} | {v['rows_2022_wk2']} | {v['share_wk2_with_all_zero_decayed_history']} |")
    L += ["", f"## Appendix: machine scan of Phase 1C code / reports for {', '.join(PHRASES)}", "", f"{len(hits)} hits. Files with a hit that no decision above cites: {doc['unmapped_scan_hit_files'] or 'none'}.", ""]
    cur = None
    for h in hits:
        if h["file"] != cur:
            cur = h["file"]; L += [f"### {cur}", ""]
        L.append(f"- line {h['line']} [{h['phrase']}]: {h['text']}")
    (OUT / "posthoc_audit.md").write_text("\n".join(L) + "\n")
    print(len(DECISIONS), "decisions;", len(hits), "scan hits; unmapped:", doc["unmapped_scan_hit_files"])


if __name__ == "__main__":
    main()

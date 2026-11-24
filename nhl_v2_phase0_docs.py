"""NHL V2 Phase0 documents generated deterministically from the result artifacts (research only)."""
import json
from pathlib import Path


def _j(p):
    return json.loads(Path(p).read_text())


def f(x, nd=4):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def write_docs(out):
    out = Path(out)
    res = _j(out / "phase0_current_model_results.json"); base = _j(out / "phase0_simple_baselines.json")
    ver = _j(out / "phase0_verdicts.json")["verdicts"]; forens = _j(out / "phase0_opportunity_forensics.json")
    gel = _j(out / "phase0_goalie_eligibility_audit.json"); miss = _j(out / "phase0_catastrophic_miss_summary.json")["summary"]
    L = []
    A = L.append
    A("# NHL Outcome Engine V2 - Phase0 findings (research only, promotes nothing)\n")
    A("Evidence grade of EVERY number below: **RETROSPECTIVE_BURNED**. There is no untouched historical NHL season (see `phase0_chronology_audit.json`); genuine confirmation waits for clean forward 2026 data, which this phase did not read.\n")
    A("## Verdicts (two questions kept separate)\n")
    A("| market | native binary-target verdict | architecture verdict |\n|---|---|---|")
    for k, v in ver.items():
        A("| %s | %s | %s |" % (k, v["native"], v["architecture"]))
    A("\nCURRENT_ENGINE_HAS_NO_CENTRAL_STAT_PROJECTION for SOG, points and goalie saves: the engine emits only P(SOG>=3), P(points>=1), P(saves>=25) and P(home win). No mean was reverse-engineered from those probabilities.\n")
    A("## Native-target performance (as-served causal calibration), Brier / log loss / ECE / BSS vs FIT base rate\n")
    A("| market | season | n | base rate | Brier | BSS | log loss | ECE | slope | AUC (secondary) | best baseline (Brier) | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mk in ("skater_sog_ge3", "skater_points_ge1", "team_moneyline"):
        for s, v in res[mk]["seasons"].items():
            m = v["incumbent_as_served_causal_platt"]; bs = v["baselines"]; bn = min(bs, key=lambda n: bs[n]["brier"])
            A("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s %s | %s |" % (mk, s, m["n"], f(m["event_base_rate"]), f(m["brier"]), f(m["brier_skill_vs_fit_rate"]), f(m["log_loss"]),
                                                                             f(m["ece_10bin"]), f(m["calibration_slope"], 3), f(m["roc_auc_secondary"], 3), bn, f(bs[bn]["brier"]), v["verdict_detail"]["verdict"]))
    A("\nPaired block-bootstrap Brier difference (incumbent minus baseline; negative = incumbent better), SOG and points:\n")
    A("| market | season | vs A (rate) | vs B (player rate) | vs C (+opp/home) | vs D (human count map) |\n|---|---|---|---|---|---|")
    for mk in ("skater_sog_ge3", "skater_points_ge1"):
        for s, v in res[mk]["seasons"].items():
            d = v["brier_diff_incumbent_minus_baseline"]
            A("| %s | %s | " % (mk, s) + " | ".join("%s [%s, %s]" % (f(d[b]["mean"], 5), f(d[b]["ci95_lo"], 5), f(d[b]["ci95_hi"], 5)) for b in "ABCD") + " |")
    A("\nReading: the incumbent classifiers clear constant-rate and player-rate baselines by a small margin (~0.002 Brier) but their edge over a transparent frozen human opportunity chain (expected TOI x shrunk shot/point rate x damped opponent factor, mapped to a threshold probability by a locked Poisson) is about 0.0002-0.0006 Brier and its confidence interval touches zero in three of four skater cases. The gradient-boosted classifier adds almost nothing beyond the opportunity chain it implicitly learns.\n")
    A("## Central baselines (the floor Phase1 must beat)\n")
    A("| stat | season | baseline | MAE | bias | median AE | within 1 | within 2 | miss >3 |\n|---|---|---|---|---|---|---|---|---|")
    for mk, nm in (("skater_sog", "SOG"), ("skater_points", "points")):
        for s in ("2024", "2025"):
            for b in ("human_frozen", "season_rate_x_prior3_toi", "prior10_mean", "ewma_0.2"):
                c = base[mk][s]["central_baselines"][b]
                A("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (nm, s, b, f(c["mae"]), f(c["bias_pred_minus_actual"]), f(c["median_ae"]), f(c["within_1"], 3), f(c["within_2"], 3), f(c["miss_gt_3"], 4)))
    A("\nSOG count error is close to the variance a perfect-mean count model would still show: MSE / mean(mu) = %s (2024) and %s (2025) against a locked dispersion index of %s. Better means (not better distributions) are the only upside, and means depend on role information (PP unit, line, deployment) that the historical production data does not contain.\n" % (
        f(res["skater_sog_ge3"]["seasons"]["2024"]["count_error_vs_dispersion_floor"]["mse_over_mean_mu"], 3), f(res["skater_sog_ge3"]["seasons"]["2025"]["count_error_vs_dispersion_floor"]["mse_over_mean_mu"], 3),
        f(res["skater_sog_ge3"]["constants"]["count_distribution"]["dispersion_index"], 3)))
    A("## Opportunity vs efficiency (ORACLE diagnostics: postgame, descriptive only)\n")
    fo = forens["skater_sog"]["2025"]
    A("SOG 2025: TOI prediction MAE %s min. Replacing predicted TOI with the actual TOI removes only %s of MSE; the actual-rate oracle removes %s but that oracle absorbs the outcome itself (rate = shots / TOI), so it mixes rate predictability with irreducible count variance and is not recoverable skill. Deployment (TOI) uncertainty is minor next to shot-count variance given the available inputs; the missing PP/line state is the untested lever.\n" % (
        f(fo["toi_prediction_error"]["mae_minutes"], 2), f(fo["ORACLE_actual_TOI_x_predicted_rate_x_opp"]["mse_removed_vs_full"], 3), f(fo["ORACLE_predicted_TOI_x_actual_game_rate"]["mse_removed_vs_full"], 3)))
    gf = forens["goalie_saves"]["2025"]["B_APPEARANCE_CONDITIONAL_NO_TARGET_TOI_FILTER"]
    A("Goalie saves 2025 (B): shots-against prediction MAE %s (bias %s); actual shots-against oracle removes %s of MSE, actual save-percentage oracle removes %s: volume faced, not save percentage, is the dominant reducible component, and that volume depends on the unknown starter and game script.\n" % (
        f(gf["shots_against_prediction"]["mae"], 2), f(gf["shots_against_prediction"]["bias_pred_minus_actual"], 2), f(gf["ORACLE_actual_SA_x_predicted_sv"]["mse_removed_vs_full"], 3), f(gf["ORACLE_predicted_SA_x_actual_sv"]["mse_removed_vs_full"], 3)))
    A("## Goalie eligibility leakage (resolved before reporting goalie results)\n")
    cp = gel["code_path"]
    A("- Code path: `%s::%s`, `%s` on `%s` (also in `%s`). Training population: **changed**; scoring population: **changed**; labels: unchanged; features: unchanged (history is prior qualifying appearances only)." % (cp["file"], cp["function"], cp["sql"], cp["field"], cp["also_in"]))
    A("- Own re-implementation reproduces the incumbent exactly: same population = %s, max feature difference = %s (n = %d)." % (gel["integrity"]["same_population"], gel["integrity"]["max_feature_abs_diff"], gel["integrity"]["n_incumbent"]))
    A("- Rows removed by the filter per season (2018-2025): " + ", ".join("%s: %d" % (s, v["rows_added_by_removing_target_toi_filter"]) for s, v in gel["accounting_by_season"].items()) + "; their mean saves are ~7-9 and essentially none reach 25. The filter inflates the OVER base rate by ~2-4 points (2025: A %s vs B %s)." % (f(gel["accounting_by_season"]["2025"]["A_event_rate"], 3), f(gel["accounting_by_season"]["2025"]["B_event_rate"], 3)))
    A("- Diagnostic verdicts: A (as implemented) = %s; B (no target-TOI filter, still conditioned on appearing) = %s. **Final goalie status: %s** - no timestamped pregame starter exists, so no real-world deployment claim is made.\n" % (gel["diagnostic_verdicts"]["A_CURRENT_AS_IMPLEMENTED"], gel["diagnostic_verdicts"]["B_APPEARANCE_CONDITIONAL_NO_TARGET_TOI_FILTER"], gel["final_status"]))
    A("The preregistered FIT-constant baselines are handicapped on goalie saves by event-rate drift (saves>=25 rate fell from ~0.66 to ~0.47). A POST-HOC sensitivity (declared after first results, not used for any verdict) re-centred baselines on the previous season's rate; the incumbent still wins by 0.002-0.016 Brier, with one interval (population B, 2025, vs the rate-shrunk baseline) touching zero.\n")
    cov = _j(out / "phase0_stat_definitions_audit.json")["coverage_vs_v1_official_boxscore"]
    A("## Production data-quality finding\n")
    A("The production `skater_games` table holds only " + ", ".join("%s: %s" % (s, f(v["production_coverage_ratio"], 3)) for s, v in cov.items() if isinstance(v, dict)) + " of the official boxscore skater rows (toi>0) counted in the V1 acquisition. Rolling histories and games-played counts in the incumbent features are therefore understated for many players, and ~40% of regulation games show total goals not matched by credited skater goals. This is a data ceiling on every number above and on any Phase1 model built from this table. Shootout shots/goals are not in skater stats, so shootouts do not contaminate player-stat grading; 68 goalie rows break saves = shots against - goals against.\n")
    A("## Role / availability data\n")
    A("Historical line combinations, PP assignment, PP TOI, scratch state and confirmed starter for the TARGET game: **BLOCKED_TIMING**. Prior-game EV/PP TOI and games-started exist as postgame box scores (V1 acquisition on `codex/nhl-outcome-engine-v1`, not in the production DB) and are legitimate rolling-history features. Forward capture (Phase0C) exists and was not opened.\n")
    A("## Catastrophic misses (evidence-flag attribution only; flags overlap)\n")
    for k, v in miss.items():
        A("- `%s`: n=%d, flags=%s" % (k, v["n"], json.dumps(v["cause_flags"], sort_keys=True)))
    A("\n## Architecture conclusion (earned from the evidence above)\n")
    A("1. The incumbent is a threshold classifier per prop with no central projection, no distribution, no availability/role/PP/goalie structure: it extrapolates recent averages with coarse team margin.\n2. A transparent opportunity chain (TOI x rate x damped opponent) matches it to within ~0.0005 Brier on the same binary targets and also yields counts, medians and any threshold: model the hockey count first, derive probabilities from it.\n3. The upside left for a count-first engine is bounded: SOG count error is already near the count-variance floor given the available means, so Phase1 value must come from better means (role/PP/deployment/starter state), which are forward-only data. Historical results cannot validate that; they can only reject.\n")
    A("## Phase1 market ranking (evidence-based; first Phase1 = skater SOG)\n")
    A("| rank | market | current weakness | addressable error | data quality | honest validation | practical use |\n|---|---|---|---|---|---|---|")
    A("| 1 | skater SOG | classifier ~ human baseline; no count | role/PP means (forward) + count distribution | official SOG, n~37k/season, but production DB holds ~85-92% of official skater rows | historical burned; forward only | high |")
    A("| 2 | goalie saves | blocked by starter state | starter + volume | usable; production DB holds ~85-92% of official skater rows | forward only | high once starter captured |")
    A("| 3 | skater points | classifier ~ human baseline | needs goals/assists decomposition (V1: goals head not established on 2025 mean-bias guard; assists A1/A2 rejected in development) | good | burned | medium |")
    A("| 4 | team moneyline | 2025 BSS ~0.005, slope ~0.8 | derive from team goals later | good | burned | low |")
    A("\nThe V1 research branch already holds an SOG distribution engine (B2) and goals/assists work scored on 2024/2025; Phase0 did not re-score it. Whether Phase1 builds on that lineage or restarts is a decision for the user.\n")
    (out / "phase0_findings.md").write_text("\n".join(L) + "\n")

    sources = _j(out / "phase0_source_audit.json")["sources"]
    (out / "source_inventory.json").write_text(json.dumps({"artifact": "source_inventory", "classification_vocabulary": _j(Path(__file__).resolve().parent / "nhl_models" / "nhl_player_outcome_v2" / "protocol.json")["source_classification_vocabulary"],
                                                           "sources": sources}, indent=1, sort_keys=True) + "\n")
    reg = {"artifact": "research_registry", "project": "NHL Outcome Engine V2", "phase": "Phase0", "status": "PHASE0_COMPLETE_PROMOTES_NOTHING", "evidence_grade": "RETROSPECTIVE_BURNED",
           "periods": {"FIT": "2018-2022", "SELECTION": "2023", "BURNED": "2024", "BURNED_EXPOSED": "2025", "CLEAN_FORWARD": "2026-09-29 onward (unread)"},
           "no_untouched_historical_season": True, "verdicts": ver,
           "central_projection_gap": "CURRENT_ENGINE_HAS_NO_CENTRAL_STAT_PROJECTION",
           "protocol_files": ["protocol.json", "protocol_amendment_1.json", "protocol_amendment_2.json"],
           "recommended_first_phase1_market": "skater_shots_on_goal", "existing_research_lineage": "origin/codex/nhl-outcome-engine-v1 (not re-scored, not merged)",
           "production_changed": False, "nfl_pr_64_touched": False, "merge_allowed": False}
    (out / "research_registry.json").write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    (out / "feature_routing.md").write_text("""# NHL V2 feature routing (Phase0 plan, nothing built)

Causal chain: AVAILABILITY/LINEUP -> TEAM GAME ENVIRONMENT -> ICE TIME / LINE / PP ROLE -> OPPORTUNITY -> EFFICIENCY -> CENTRAL PROJECTION -> (later) UNCERTAINTY.

| layer | candidate inputs | pregame availability | status |
|---|---|---|---|
| availability / lineup | confirmed starter, backup, scratches, injuries | forward capture only (T24H..T2); historical BLOCKED_TIMING | ACCEPTED_FORWARD_ONLY |
| team game environment | opponent prior shots-for / shots-against / goals, home flag, days of rest, venue | strictly earlier games | ACCEPTED_HISTORICAL |
| ice time / role | prior-3/10 total TOI, prior EV and PP TOI, games started | prior-game box scores (V1 acquisition) | ACCEPTED_HISTORICAL for history; target-game PP unit BLOCKED_TIMING |
| opportunity | TOI x shot rate by game state (EV/PP) | derived | to build in Phase1 |
| efficiency | shrunk shooting pct, save pct (regressed) | derived | to build in Phase1 |
| central projection | count mean/median per stat | derived | Phase1 |
| derived thresholds | P(X>=k) from the count distribution | derived | Phase1 |

Forbidden routes: target-game score state, any betting-market line, price or derived probability, target-game TOI, postgame lines/PP relabelled as pregame, shootout events as skater stats.
Opponent features route mechanically through the TEAM GAME ENVIRONMENT layer only (damped, season-to-date league normalised). A feature enters only with mechanism, causal layer, pregame availability and out-of-time value.
""")

"""NHL engine contract tests: registry consistency, shared-state leakage rules, goalie conditional-vs-pregame separation, scoring identity, no sportsbook inputs, no target-game leakage, forecast contract.
Real frozen tables for the identity checks; synthetic leagues (Phase 1A helpers) for behavioural leakage proofs.  python tests/test_nhl_engine_contracts.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
import nhl_engine_feasibility_data as FD  # noqa: E402
import nhl_forecast_contract as FC  # noqa: E402
import nhl_sog_phase1a_data as D  # noqa: E402
import test_nhl_phase1a as T1A  # noqa: E402

OUT = REPO / "nhl_models" / "nhl_outcome_engine"
SPORTSBOOK = ("odds", "sportsbook", "bookmaker", "vegas", "juice", "vig", "implied", "moneyline_odds", "prop_line", "market_line", "closing_line")


def j(name):
    return json.loads((OUT / name).read_text())


def keys_of(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from keys_of(v)
    elif isinstance(o, list):
        for v in o:
            yield from keys_of(v)


# ------------------------------------------------------------------ registry
def test_registry_consistency_and_real_current_statuses():
    reg = j("NHL_OUTCOME_ENGINE_REGISTRY.json")
    vocab = set(reg["status_vocabulary"])
    assert vocab == {"UNBUILT", "LEGACY_UNAUDITED", "LEGACY_FAILED", "LEGACY_STABLE_COMPARATOR", "RESEARCH", "COMPONENT_CANDIDATE", "HISTORICAL_CHAMPION", "FORWARD_SHADOW", "PRODUCTION", "FROZEN_REJECTED", "BLOCKED"}
    required_state = {"player_identity", "team_identity", "target_game_membership", "availability_participation", "current_team_role_deployment", "toi_distribution_state", "team_pace_shot_environment", "opponent_defensive_environment", "goalie_candidate_start_state", "schedule_venue_home_away"}
    required_heads = {"skater_shots_on_goal", "skater_goals", "skater_assists", "skater_points", "goalie_saves", "team_goals", "moneyline_win_probability"}
    assert required_state <= set(reg["shared_state"]) and required_heads <= set(reg["outcome_heads"])
    everything = [v["status"] for v in reg["shared_state"].values()] + [v["status"] for v in reg["outcome_heads"].values()] + [v["status"] for v in reg["optional_future"].values()] + [v["status"] for v in reg["legacy_comparators"].values()] + \
                 [v["status"] for v in reg["derived_state_components_unbuilt"].values()] + [reg["outcome_heads"]["skater_shots_on_goal"]["extension"]["status"]]
    assert set(everything) <= vocab, set(everything) - vocab
    h = reg["outcome_heads"]
    assert h["skater_shots_on_goal"]["status"] == "HISTORICAL_CHAMPION" and h["skater_shots_on_goal"]["final_sog_status"] == "B2_HISTORICAL_CHAMPION_RETAINED" and h["skater_shots_on_goal"]["extension"]["status"] == "FROZEN_REJECTED"
    assert h["goalie_saves"]["status"] == "LEGACY_FAILED" and "0.5673" in h["goalie_saves"]["note"] and "0.001" in h["goalie_saves"]["note"]
    assert h["skater_points"]["status"] == "LEGACY_FAILED" and "0.0271" in h["skater_points"]["note"]
    assert h["moneyline_win_probability"]["status"] == "LEGACY_STABLE_COMPARATOR" and "NOT a new-standard champion" in h["moneyline_win_probability"]["note"]
    assert h["skater_assists"]["status"] == h["team_goals"]["status"] == "UNBUILT"
    assert h["skater_goals"]["status"] == "RESEARCH" and h["skater_goals"]["final_goals_status"] == "GOALS_HISTORICAL_CHAMPION_NOT_ESTABLISHED"
    assert reg["shared_state"]["goalie_candidate_start_state"]["status"] == "BLOCKED"
    assert not any(v["status"] == "PRODUCTION" for v in list(h.values()) + list(reg["shared_state"].values()))                  # nothing of the new engine is in production
    known = set(reg["shared_state"]) | set(h) | set(reg["derived_state_components_unbuilt"])
    for name, hv in h.items():
        assert set(hv["depends_on"]) <= known, (name, set(hv["depends_on"]) - known)
    order = reg["build_order"]["order"]
    assert [o["rank"] for o in order] == list(range(1, 9)) and "goals" in order[2]["item"] and "goalie" in order[5]["item"] and reg["build_order"]["why_changed"]
    assert {k: v["status"] for k, v in reg["legacy_comparators"].items()} == {"legacy_shots_on_goal": "LEGACY_FAILED", "legacy_points": "LEGACY_FAILED", "legacy_goalie_saves": "LEGACY_FAILED", "legacy_moneyline": "LEGACY_STABLE_COMPARATOR"}
    col = reg["live_collector_status_readonly"]
    assert col["read_only"] is True and col["minimum_qualification_sample_met"] is False and set(col["horizons"]) == {"T24H", "T90", "T30", "T10", "T2"}
    graph = (OUT / "NHL_OUTCOME_DEPENDENCY_GRAPH.md").read_text()
    assert "B2_HISTORICAL_CHAMPION_RETAINED" in graph and "goals / assists / points move ahead" in graph


# ------------------------------------------------------------------ shared skater state
def test_shared_state_contract_covers_every_b2_feature_and_separates_skill_from_role():
    c = j("nhl_shared_skater_state_contract.json")
    reg = c["feature_registry"]
    assert set(reg) == set(D.FEATURES) and c["feature_count"] == len(D.FEATURES) == 33
    for f, v in reg.items():
        assert v["block"] in c["state_blocks"] and v["history_domain"]
    assert all(reg[f]["history_domain"] == "all_team" for f, v in reg.items() if v["block"] == "SKILL")
    assert all(reg[f]["history_domain"] == "current_team" for f, v in reg.items() if v["block"] == "CURRENT_ROLE_DEPLOYMENT")
    assert all(f.startswith(("TOI_", "PP_", "SHIFT_")) or f in ("POS_F", "POS_D", "POS_UNKNOWN") for f, v in reg.items() if v["history_domain"] == "current_team" and v["block"] == "CURRENT_ROLE_DEPLOYMENT")
    rules = " ".join(c["leakage_rules"])
    for tag in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        assert tag in rules
    assert "target-game" in rules and "sportsbook" in rules and "mutating any target-game row" in rules
    assert "shot attempts" in " ".join(c["explicitly_excluded"]) and "goalie identity" in " ".join(c["explicitly_excluded"])
    assert c["contract_content_sha256"] == hashlib.sha256(json.dumps({k: c[k] for k in ("feature_registry", "leakage_rules", "strict_separation")}, sort_keys=True).encode()).hexdigest()


def test_shared_state_leakage_is_proven_behaviourally_on_a_synthetic_league():
    # the Phase 1A builder IS the state implementation: reuse its invariance proofs (target-row mutation, membership, cutoff rule, role-vs-skill domains)
    T1A.test_target_row_mutation_leaves_every_feature_unchanged()
    T1A.test_old_team_only_history_cannot_establish_membership_but_first_same_team_game_does()
    T1A.test_old_team_skill_history_is_usable_after_membership_but_role_rows_are_not()
    T1A.test_source_games_completed_before_cutoff_and_target_excluded()
    T1A.test_no_target_position_leakage_position_from_latest_prior_same_team_appearance()
    games, rows = T1A.world()
    tab, _ = T1A.build(games, rows)
    c = j("nhl_shared_skater_state_contract.json")
    assert set(c["feature_registry"]) <= set(tab) and not ({"sog", "played"} & set(c["feature_registry"]))                    # labels are never state features


# ------------------------------------------------------------------ goalie: conditional vs pregame
def test_goalie_conditional_component_is_separated_from_the_pregame_start_component():
    f = j("phase_goalie_g0_feasibility.json")
    cs = f["component_status"]
    assert cs["GOALIE_SAVES_CONDITIONAL_ON_START_COMPONENT"]["status"] == "RESEARCH" and cs["PREGAME_START_PROBABILITY_COMPONENT"]["status"] == "BLOCKED" and cs["UNCONDITIONAL_PREGAME_GOALIE_SAVES_HEAD"]["status"] == "BLOCKED"
    assert "LABEL" in cs["GOALIE_SAVES_CONDITIONAL_ON_START_COMPONENT"]["label_rule"].upper() and "never labelled a confirmation" in cs["PREGAME_START_PROBABILITY_COMPONENT"]["why"]
    forb = " ".join(f["leak_safe_target_feature_contract"]["forbidden"])
    assert "the starter flag as a feature" in forb and "target-game" in forb and "sportsbook" in forb
    assert f["questions"]["Q5"]["verdict"] == "UNRECONCILED" and f["questions"]["Q7"]["verdict"].startswith("BLOCKED")                  # strict verdicts, not waved through
    assert f["questions"]["Q1"]["exactly_one_starter"] == f["questions"]["Q1"]["team_games"]
    state = j("nhl_shared_skater_state_contract.json")
    assert "goalie identity / starter state" in state["explicitly_excluded"] and not any("start" in k.lower() and "appearance" not in k.lower() and "starts" not in k.lower() for k in D.FEATURES if k.lower().startswith("start"))
    assert not any("goalie" in k.lower() or "starter" in k.lower() for k in state["feature_registry"])                             # no goalie / starter feature in the skater state
    proto = j("phase_goalie_g0_protocol.json")
    assert "NEVER a pregame feature" in proto["leak_rule"] and proto["kind"].startswith("DATA / IDENTIFIABILITY")


# ------------------------------------------------------------------ scoring identity (real frozen tables)
def test_points_equal_goals_plus_assists_on_every_frozen_row_and_splits_reconcile():
    n = 0
    for s in FD.SEASONS:
        for r in FD.read_table(f"phase_scoring_s0_data/skater_scoring_{s}.jsonl.gz"):
            n += 1
            assert r["points"] == r["goals"] + r["assists"], (s, r["gameId"], r["playerId"])
            assert r["evGoals"] + r["ppGoals"] + r["shGoals"] == r["goals"] and r["evPoints"] + r["ppPoints"] + r["shPoints"] == r["points"]
    assert n == 397778
    f = j("phase_scoring_s0_feasibility.json")
    assert f["questions"]["S2"]["points_identity_exact"] is True and f["points_from_joint_process"]["verdict"].startswith("POINTS_CAN_BE_DERIVED")
    assert f["questions"]["S2"]["goals_gt_shots_mechanism"]["goals_gt_shots_rows"] == 30 and f["questions"]["S2"]["goals_gt_shots_mechanism"]["all_exactly_plus_1"] is True        # the one violated identity is reported, never hidden
    assert f["questions"]["S3"]["goals_mismatch"] == 0 and f["questions"]["S3"]["assists_mismatch"] == 0


# ------------------------------------------------------------------ sportsbook / leakage / forecast contract
def test_no_sportsbook_inputs_anywhere_in_the_engine_contracts_or_features():
    for name in ("NHL_OUTCOME_ENGINE_REGISTRY.json", "nhl_shared_skater_state_contract.json", "nhl_universal_forecast_contract.json", "NHL_HEAD_GOVERNANCE_TEMPLATE.json", "phase_goalie_g0_protocol.json", "phase_scoring_s0_protocol.json", "phase_team_game_t0_protocol.json"):
        ks = list(keys_of(j(name)))
        bad = [k for k in ks if any(w in k.lower() for w in SPORTSBOOK)]
        assert not bad, (name, bad)
    assert not [f for f in D.FEATURES if any(w in f.lower() for w in SPORTSBOOK)]
    for name in ("phase_goalie_g0_protocol.json", "phase_scoring_s0_protocol.json", "phase_team_game_t0_protocol.json"):
        assert any("sportsbook" in x for x in j(name)["never"])
    assert any("sportsbook" in x.lower() for x in j("nhl_shared_skater_state_contract.json")["leakage_rules"]) and any("sportsbook" in x for x in j("NHL_HEAD_GOVERNANCE_TEMPLATE.json")["prohibitions"])


def test_forecast_contract_validates_distributions_and_rejects_sportsbook_fields_and_leaks():
    ex = FC.example_skater_sog()
    assert FC.validate(ex) == []
    c = j("nhl_universal_forecast_contract.json")
    assert c["required_fields"] == FC.REQUIRED and FC.validate(c["example"]) == [] and "NOT SERVED" in c["serving_status"]
    for f in ("entity", "game_id", "team", "opponent", "prediction_timestamp", "information_cutoff", "availability_participation_state", "role_deployment_state", "distribution", "mean", "median", "quantiles", "model_version", "data_snapshot_hash", "coverage_uncertainty_quality"):
        assert f in c["required_fields"]
    import copy
    bad = copy.deepcopy(ex); bad["distribution"]["pmf"][0] += 0.1
    assert FC.validate(bad)                                                                                         # pmf must sum to 1
    bad = copy.deepcopy(ex); bad["mean"] += 0.01
    assert any("mean" in p for p in FC.validate(bad))
    bad = copy.deepcopy(ex); bad["odds"] = -110
    assert any("sportsbook" in p for p in FC.validate(bad))                                                          # a sportsbook line can never ride along
    bad = copy.deepcopy(ex); bad["role_deployment_state"]["prop_line"] = 2.5
    assert any("sportsbook" in p for p in FC.validate(bad))
    bad = copy.deepcopy(ex); bad["information_cutoff"] = "2026-10-11T00:00:00Z"
    assert any("information_cutoff" in p for p in FC.validate(bad))                                                  # information after the prediction is a leak
    bad = copy.deepcopy(ex); bad["median"] = 99
    assert any("median" in p for p in FC.validate(bad))
    bad = copy.deepcopy(ex); bad.pop("data_snapshot_hash")
    assert FC.validate(bad)
    # the engine yields a full distribution with NO market input at all
    assert "pmf" in ex["distribution"] and not any(any(w in k.lower() for w in SPORTSBOOK) for k in keys_of(ex))


def test_feasibility_results_match_their_preregistered_protocols_and_follow_them_in_history():
    for ph, proto in (("goalie_g0", "phase_goalie_g0_protocol.json"), ("scoring_s0", "phase_scoring_s0_protocol.json"), ("team_game_t0", "phase_team_game_t0_protocol.json")):
        res = j(f"phase_{ph}_feasibility.json"); p = j(proto)
        body = {k: v for k, v in p.items() if k != "protocol_body_sha256"}
        assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == p["protocol_body_sha256"] == res["protocol_body_sha256"]
        assert res["kind"].endswith("NO model was fit") and p["status"].startswith("PREREGISTERED_BEFORE")
        pc = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", f"nhl_models/nhl_outcome_engine/{proto}"], cwd=REPO, capture_output=True, text=True).stdout.split()[-1]
        rc = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", f"nhl_models/nhl_outcome_engine/phase_{ph}_feasibility.json"], cwd=REPO, capture_output=True, text=True).stdout.split()[-1]
        assert subprocess.run(["git", "merge-base", "--is-ancestor", pc, rc], cwd=REPO).returncode == 0 and pc != rc                # protocol committed strictly before the results
    t = j("phase_team_game_t0_feasibility.json")["questions"]
    assert t["T2"]["legacy_games_compared"] == t["T2"]["legacy_home_win_equals_final_winner"] and t["T4"]["verdict"] == "LIMITED"


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)

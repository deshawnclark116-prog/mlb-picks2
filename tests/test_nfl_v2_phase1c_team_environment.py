#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nfl_v2_phase1c_team_environment as T


def fixture():
    team_totals = {}
    team_opp = {}
    pbp = {}
    pbp_opp = {}
    schedule = {}
    for w in (1, 2, 3):
        team_totals[(2024, w, "A")] = {"attempts": 30.0 + w, "targets": 27.0 + w, "carries": 24.0}
        team_opp[(2024, w, "A")] = "B"
        pbp[(2024, w, "A")] = {
            "plays": 62.0, "dropbacks": 36.0, "rush_plays": 26.0,
            "neutral_pass_rate": 0.58, "neutral_rush_rate": 0.42,
        }
        pbp_opp[(2024, w, "A")] = "B"

        team_totals[(2024, w, "C")] = {"attempts": 34.0, "targets": 31.0, "carries": 22.0}
        team_opp[(2024, w, "C")] = "B"
        pbp[(2024, w, "C")] = {
            "plays": 64.0, "dropbacks": 39.0, "rush_plays": 25.0,
            "neutral_pass_rate": 0.61, "neutral_rush_rate": 0.39,
        }
        pbp_opp[(2024, w, "C")] = "B"

        for i, (tm, de, att, car) in enumerate((
            ("D", "E", 29.0, 25.0), ("E", "F", 33.0, 23.0),
            ("F", "G", 31.0, 24.0), ("G", "H", 35.0, 21.0),
            ("H", "I", 32.0, 22.0), ("I", "D", 30.0, 26.0),
        )):
            team_totals[(2024, w, tm)] = {"attempts": att, "targets": att - 3, "carries": car}
            team_opp[(2024, w, tm)] = de
            pbp[(2024, w, tm)] = {
                "plays": att + car + 4.0,
                "dropbacks": att + 4.0,
                "rush_plays": car,
                "neutral_pass_rate": 0.57,
                "neutral_rush_rate": 0.43,
            }
            pbp_opp[(2024, w, tm)] = de

    # Current target game intentionally absurd; it must never leak into features.
    team_totals[(2024, 4, "A")] = {"attempts": 99.0, "targets": 90.0, "carries": 1.0}
    team_opp[(2024, 4, "A")] = "B"
    pbp[(2024, 4, "A")] = {
        "plays": 100.0, "dropbacks": 99.0, "rush_plays": 1.0,
        "neutral_pass_rate": 0.99, "neutral_rush_rate": 0.01,
    }
    pbp_opp[(2024, 4, "A")] = "B"
    schedule[(2024, 4, "A")] = {
        "home": 1.0, "rest_days": 7, "opponent_rest_days": 7
    }
    return team_totals, team_opp, pbp, pbp_opp, schedule


def cfg():
    return {
        "window": 5,
        "league_window": 128,
        "decay": 0.85,
        "offense_weight": 0.70,
        "defense_weight": 0.20,
        "structure_weight": 0.75,
        "neutral_weight": 0.25,
        "conversion_offense_weight": 0.75,
        "home_adjust": 0.0,
        "rest_adjust": 0.0,
    }


def test_team_environment_is_prior_only_and_auditable():
    tt, to, pbp, po, sched = fixture()
    T.build_indexes(tt, to, pbp, po, sched)
    rec = T.feature_receipt("A", "B", 2024, 4, "attempts", cfg())
    assert rec is not None
    assert rec["offense_recent_direct_opportunity"] < 40.0
    assert rec["projected_total_plays"] < 80.0
    assert rec["team_opportunity_projection"] < 60.0
    assert rec["structured_projection"] > 0.0
    assert rec["opportunity_per_component_conversion"] > 0.0


def test_schedule_context_is_explicit_not_hidden():
    tt, to, pbp, po, sched = fixture()
    T.build_indexes(tt, to, pbp, po, sched)
    base = cfg()
    home = T.feature_receipt("A", "B", 2024, 4, "attempts", base)
    altered = dict(base)
    altered["home_adjust"] = 1.0
    home_plus = T.feature_receipt("A", "B", 2024, 4, "attempts", altered)
    assert abs(home_plus["team_opportunity_projection"] - home["team_opportunity_projection"] - 1.0) < 1e-9


def main():
    test_team_environment_is_prior_only_and_auditable()
    test_schedule_context_is_explicit_not_hidden()
    print("NFL V2 Phase 1C-T team-environment tests: PASS")


if __name__ == "__main__":
    main()

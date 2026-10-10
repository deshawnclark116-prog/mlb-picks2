"""Adversarial NHL public board policy tests with real archived slate evidence.

Do not touch forward grading ledger, make up player book lines or enforce a
50/50 OVER/UNDER quota. Hide both unpriced sides consistently.
"""
import copy
import json
from pathlib import Path

import pytest

from nhl_sog_publication_gate_a import (
    SCHEMA, SOG_CLASSIFIERS, quarantine_fixed_line_sog, fail_closed_official_sog,
)

ROOT = Path(__file__).resolve().parents[1]


def sog(pid, team="PHI", opp="BOS", side="UNDER", market="shots_on_goal_early_season"):
    return {
        "market": market, "player_id": pid, "player": "Player X",
        "team": team, "opponent": opp, "season": 2026,
        "game_date": "2026-10-10", "line": 2.5,
        "pick": f"{side} 2.5", "model_prob": .9,
        "prob_over": .1 if side == "UNDER" else .9,
        "model_source": "prior_season_informed",
    }


def test_blocks_both_directions_not_a_fake_unders_balancing_strategy():
    rows = [sog(1, side="OVER"), sog(2, side="UNDER"),
            {"market": "moneyline", "team": "NYR", "opponent": "BOS", "pick": "NYR ML"},
            {"market": "points", "player_id": 3, "pick": "OVER 0.5"}]
    original = copy.deepcopy(rows)
    kept, audit = quarantine_fixed_line_sog(rows, season=2026, game_date="2026-10-10")
    assert len(kept) == 2
    assert kept == rows[2:]
    assert rows == original
    assert audit["schema"] == SCHEMA
    assert audit["removed_total"] == 2
    assert audit["removed_by_market_and_side"]["shots_on_goal_early_season"] == {
        "OVER": 1, "UNDER": 1
    }
    assert audit["suppressed_overs_as_well_as_unders"] is True
    assert audit["official_shots_on_goal_picks"] == 0
    assert audit["book_main_lines_verified"] is False
    assert audit["dressed_lineups_verified"] is False
    assert fail_closed_official_sog(kept)


def test_both_sog_generations_for_same_player_quarantined_even_with_conflicting_directions():
    rows = [sog(1, side="OVER"), sog(1, side="UNDER", market="shots_on_goal"),
            sog(2, side="UNDER"), sog(3, team="BOS", opp="PHI", side="OVER"),
            sog(3, team="NYR", opp="NJD", side="UNDER")]
    kept, audit = quarantine_fixed_line_sog(rows, season=2026, game_date="2026-10-10")
    assert kept == []
    assert audit["removed_total"] == 5
    assert audit["distinct_candidate_player_games"] == 4
    assert audit["duplicate_candidate_groups"] == 1
    assert audit["duplicate_candidate_rows"] == 2
    assert audit["contradictory_player_game_groups"] == 1


def test_unknown_player_identity_cannot_deduplicate_into_false_verified_roster():
    a = sog(None)
    b = sog(None)
    kept, audit = quarantine_fixed_line_sog([a,b], season=2026, game_date="2026-10-10")
    assert not kept
    assert audit["distinct_candidate_player_games"] == 2
    assert audit["duplicate_candidate_groups"] == 0


def test_hard_publisher_assertion_blocks_legacy_candidate_leakage():
    with pytest.raises(RuntimeError, match="LEGACY_NHL_SOG_UNVERIFIED_MAIN_LINE"):
        fail_closed_official_sog([sog(1, side="OVER")])
    with pytest.raises(RuntimeError, match="LEGACY_NHL_SOG_UNVERIFIED_MAIN_LINE"):
        fail_closed_official_sog([sog(1, side="UNDER", market="shots_on_goal")])


def test_malformed_rows_fail_closed():
    with pytest.raises(ValueError, match="malformed"):
        quarantine_fixed_line_sog([None], season=2026, game_date="2026-10-10")
    with pytest.raises(TypeError):
        quarantine_fixed_line_sog(None, season=2026, game_date="2026-10-10")


def test_actual_oct_10_frozen_nhl_board_has_only_research_legacy_sog_not_bets():
    path = ROOT / "docs" / "nhl_predictions_2026_2026-10-10.json"
    if not path.exists():
        pytest.skip("historical archived 2026-10-10 research board not mounted")
    data = json.loads(path.read_text())
    if data.get("game_date") != "2026-10-10":
        pytest.fail("unexpected archive date")
    original = data["picks"]
    kept, audit = quarantine_fixed_line_sog(original, season=data["season"],
                                              game_date=data["game_date"])
    n_sog = sum(row.get("market") in SOG_CLASSIFIERS for row in original)
    assert n_sog >= 600  # regression: fixed 2.5 mass-issued on actual archive
    assert audit["removed_total"] == n_sog
    assert audit["removed_by_market_and_side"]["shots_on_goal_early_season"]["UNDER"] > 400
    assert audit["contradictory_player_game_groups"] >= 1
    assert len(kept) + n_sog == len(original)
    assert all(row not in kept for row in original if row["market"] in SOG_CLASSIFIERS)
    assert fail_closed_official_sog(kept)


def test_builder_calls_publication_gate_after_logging_but_before_output():
    source = (ROOT / "nhl_serving_builder_a.py").read_text()
    assert "picks, sog_publication_audit = quarantine_fixed_line_sog(" in source
    assert "fail_closed_official_sog(picks)" in source
    assert source.index("append_new_picks_to_log(PICKS_LOG_PATH,") < source.index(
        "picks, sog_publication_audit = quarantine_fixed_line_sog(") < source.index(
        'picks.sort(key=lambda p:')
    assert '"nhl_sog_publication_integrity": sog_publication_audit' in source

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
    write_daily_snapshot_preserving_legacy,
    split_record_by_publication_scope, result_scope_summary,
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




def test_original_pre_policy_snapshot_archived_exact_bytes_without_overwriting_history(tmp_path):
    snapshot = tmp_path / "nhl_predictions_2026_2026-10-10.json"
    original = (
        '{\n  "game_date": "2026-10-10", "picks": '
        '[{"market":"shots_on_goal_early_season","pick":"UNDER 2.5"}]\n}\n'
    ).encode()
    snapshot.write_bytes(original)
    payload = {
        "nhl_sog_publication_integrity":{"schema":SCHEMA},
        "picks":[{"market":"moneyline","team":"PHI","pick":"PHI ML"}],
    }
    backup = write_daily_snapshot_preserving_legacy(snapshot, payload)
    assert backup is not None
    assert backup.read_bytes() == original
    assert json.loads(snapshot.read_text()) == payload
    assert write_daily_snapshot_preserving_legacy(snapshot, payload) == backup
    assert backup.read_bytes() == original


def test_preexisting_different_original_archive_fails_closed(tmp_path):
    p = tmp_path / "nhl_predictions_2026_2026-10-10.json"
    p.write_text('{"picks":[{"market":"shots_on_goal","pick":"UNDER 2.5"}]}')
    backup = tmp_path / "nhl_predictions_2026_2026-10-10_legacy_fixed_2_5_pre_policy.json"
    backup.write_text('{"picks":[]}')
    with pytest.raises(RuntimeError,match="HISTORICAL_LEGACY_BACKUP_CONFLICT"):
        write_daily_snapshot_preserving_legacy(p, {
            "picks":[], "nhl_sog_publication_integrity":{"schema":SCHEMA},
        })
    assert "shots_on_goal" in p.read_text()
    assert backup.read_text() == '{"picks":[]}'


def test_refuse_republish_if_audit_missing_or_legacy_sog_still_present(tmp_path):
    p=tmp_path/"daily.json"
    with pytest.raises(RuntimeError,match="PUBLICATION_GATE_AUDIT_MISSING"):
        write_daily_snapshot_preserving_legacy(p,{"picks":[]})
    with pytest.raises(RuntimeError,match="LEGACY_NHL_SOG"):
        write_daily_snapshot_preserving_legacy(p,{
            "picks":[sog(999)], "nhl_sog_publication_integrity":{"schema":SCHEMA},
        })
    assert not p.exists()


def test_corrupt_historical_daily_snapshot_never_discarded(tmp_path):
    p=tmp_path/"daily.json"
    p.write_bytes(b"not json")
    with pytest.raises(RuntimeError,match="CORRUPT_HISTORICAL"):
        write_daily_snapshot_preserving_legacy(p,{
            "picks":[], "nhl_sog_publication_integrity":{"schema":SCHEMA},
        })
    assert p.read_bytes() == b"not json"




def test_frozen_historical_grading_keeps_every_old_classifier_outcome_in_research():
    frozen_record = ROOT / "docs" / "nhl_record.json"
    if not frozen_record.exists():
        pytest.skip("historical NHL record not available")
    source = json.loads(frozen_record.read_text())
    original = source["results"]
    active, legacy = split_record_by_publication_scope(original)
    assert len(active) + len(legacy) == len(original)
    assert len(legacy) > 1000
    assert all(r["market"] not in SOG_CLASSIFIERS for r in active)
    assert all(r["market"] in SOG_CLASSIFIERS for r in legacy)
    active_summary,_ = result_scope_summary(active)
    research_summary,_ = result_scope_summary(legacy)
    assert active_summary["total"] + research_summary["total"] == len(original)
    assert research_summary["hits"] <= research_summary["total"]
    assert original == source["results"]


def test_new_research_only_2_5_classifiers_cannot_be_added_to_main_grader_log():
    source=(ROOT/"nhl_serving_builder_a.py").read_text()
    assert "new_official_log_candidates = [" in source
    assert "if p.get(\"market\") not in SOG_CLASSIFIERS" in source
    assert "append_new_picks_to_log(" in source
    assert "PICKS_LOG_PATH, logged_keys, new_official_log_candidates" in source
    grader=(ROOT/"nhl_grade_record_a.py").read_text()
    assert "split_record_by_publication_scope(results)" in grader
    assert '"legacy_fixed_line_sog_research": {' in grader
    assert '"summary": active_summary' in grader


def test_public_scope_split_fails_closed_if_result_lacks_real_hit_or_miss():
    with pytest.raises(ValueError,match="binary outcome"):
        result_scope_summary([{"market":"moneyline","result":None}])
    active,legacy=split_record_by_publication_scope([
        {"market":"shots_on_goal","result":"hit"},
        {"market":"points","result":"miss"},
    ])
    assert len(active)==1 and len(legacy)==1


def test_builder_calls_publication_gate_after_logging_but_before_output():
    source = (ROOT / "nhl_serving_builder_a.py").read_text()
    assert "picks, sog_publication_audit = quarantine_fixed_line_sog(" in source
    assert "fail_closed_official_sog(picks)" in source
    assert source.index("append_new_picks_to_log(PICKS_LOG_PATH,") < source.index(
        "picks, sog_publication_audit = quarantine_fixed_line_sog(") < source.index(
        'picks.sort(key=lambda p:')
    assert '"nhl_sog_publication_integrity": sog_publication_audit' in source
    assert "write_daily_snapshot_preserving_legacy(hist, payload)" in source

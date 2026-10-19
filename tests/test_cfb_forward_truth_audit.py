"""CFB read-only forward truth audit — synthetic fixtures, no network or model changes."""
import copy

import pytest

import cfb_grade_record_a as G
import cfb_forward_truth_audit as A


GAME = {
    "game_id": "G1", "season": 2026, "week": 5,
    "kickoff_utc": "2026-10-03T16:00:00Z",
    "home_team": "A", "away_team": "B",
    "home_points": 24, "away_points": 14,
}


def rush(prob=0.75, projection=None, logged_at="2026-10-03T10:00:00Z"):
    d = {
        "market": "rushing_yards", "player_id": "P1", "player": "Pat",
        "team": "A", "opponent": "B", "season": 2026, "week": 5,
        "game_id": "G1", "line": 69.5, "pick": "OVER 69.5",
        "model_prob": prob, "logged_at": logged_at, "model_source": "context_v2",
    }
    if projection is not None:
        d["projected"] = projection
    return d


def moneyline(prob=0.7):
    return {
        "market": "moneyline", "player_id": None, "player": "A",
        "team": "A", "opponent": "B", "season": 2026, "week": 5,
        "game_id": "G1", "line": None, "pick": "A ML",
        "model_prob": prob, "logged_at": "2026-10-03T11:00:00Z", "model_source": "context_v2",
    }


def fixture(with_point=False):
    ledger = [rush(projection=85.0 if with_point else None), moneyline()]
    players = {("P1", "G1"): {"rushing_yards": 80}}
    results, ungraded, excluded, _ = G.evaluate(ledger, [GAME], players, {})
    assert len(results) == 2 and not ungraded and not excluded
    rec = {
        "results": results,
        "summary": {"total": len(results)},
        "forward_evaluation": {"evaluation_probability_source": G.EVAL_SOURCE},
    }
    return rec, ledger, [GAME], players, {"graded_predictions": len(results)}


def run(case):
    return A.audit(*case)


def test_independent_audit_does_not_invoke_production_grading_or_canonical_selection(monkeypatch):
    case = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("audit must independently compute result and original prediction")
    monkeypatch.setattr(G, "select_canonical", forbidden)
    monkeypatch.setattr(G, "grade", forbidden)
    monkeypatch.setattr(G, "actual_stat", forbidden)
    monkeypatch.setattr(G, "generation", forbidden)
    report = A.audit(*case)
    assert report["verified_first_pregame_rows"] == 2
    assert report["original_ledger_integrity"] == "EACH_GRADED_ROW_CROSS_CHECKED_AGAINST_EARLIEST_VALID_LOG"


def test_offline_original_ledger_inventory_is_not_misrepresented_as_graded_truth():
    ledger = [rush(), rush(prob=0.7, logged_at="2026-10-03T15:00:00Z"),
              rush(projection=82), moneyline()]
    report = A.ledger_projection_contract(ledger)
    assert report["original_saved_rows"] == 4
    assert report["original_point_estimate_rows"] == 1
    assert report["by_market"]["rushing_yards"]["saved_rows"] == 3
    assert report["by_market"]["rushing_yards"]["point_estimates"] == 1
    assert report["pregame_and_outcomes_verified"] is False
    assert report["new_model_predictions_generated"] == 0


def test_model_confidence_and_true_first_probability_are_not_confused_with_stat_projection():
    result = run(fixture(with_point=False))
    assert result["verified_first_pregame_rows"] == 2
    assert result["by_market"]["rushing_yards"]["n"] == 1
    assert result["original_point_projection_coverage"]["rushing_yards"]["original_point_projection_rows"] == 0
    assert result["original_point_projection_coverage"]["rushing_yards"]["projection_mae_only_when_original_pregame_point_exists"] is None
    assert result["fixed_lines_are_not_point_forecasts"]
    assert result["live_board_probabilities_used_for_grading"] is False
    assert result["scientific_status"] == A.FREEZE_POLICY
    assert result["new_predictions_generated"] == 0


def test_only_original_pregame_projection_supports_point_mae():
    result = run(fixture(with_point=True))
    p = result["original_point_projection_coverage"]["rushing_yards"]
    assert p["coverage_pct"] == 100
    assert p["projection_mae_only_when_original_pregame_point_exists"] == 5.0
    assert p["source_fields"]["projected"] == 1


def test_only_first_ledger_probability_even_after_later_refresh():
    rec, ledger, games, players, status = fixture()
    ledger.append(rush(prob=0.99, logged_at="2026-10-03T13:00:00Z"))
    result = run((rec, ledger, games, players, status))
    assert result["duplicate_canonical_ledger_keys"] == 1
    assert result["by_market"]["rushing_yards"]["mean_claimed_probability"] == 0.75


@pytest.mark.parametrize("field,changed,reason", [
    ("model_prob", 0.99, "REFRESHED_OR_CHANGED_PROBABILITY"),
    ("original_model_prob", 0.99, "REFRESHED_OR_CHANGED_PROBABILITY"),
    ("logged_at", "2026-10-03T17:00:00Z", "ORIGINAL_TIMESTAMP_MISMATCH"),
    ("pick", "UNDER 69.5", "ORIGINAL_LEDGER_FIELD_MISMATCH"),
    ("actual", 10, "PLAYER_GAME_TRUTH_MISMATCH"),
    ("result", "miss", "PROP_GRADE_MISMATCH"),
])
def test_fail_closed_if_grade_or_original_evidence_tampered(field, changed, reason):
    rec, ledger, games, players, status = fixture()
    pick = next(r for r in rec["results"] if r["market"] == "rushing_yards")
    pick[field] = changed
    with pytest.raises(A.AuditError, match=reason):
        A.audit(rec, ledger, games, players, status)


def test_duplicate_result_fails_before_group_metrics():
    rec, ledger, games, players, status = fixture()
    rec["results"].append(copy.deepcopy(rec["results"][0]))
    rec["summary"]["total"] = 3
    status["graded_predictions"] = 3
    with pytest.raises(A.AuditError, match="DUPLICATE_GRADED_KEY"):
        A.audit(rec, ledger, games, players, status)


def test_post_kickoff_entry_does_not_become_forward_truth():
    rec, ledger, games, players, status = fixture()
    ledger[0]["logged_at"] = "2026-10-03T16:00:00Z"
    with pytest.raises(A.AuditError, match="MISSING_FIRST_PREGAME_LEDGER_ENTRY"):
        A.audit(rec, ledger, games, players, status)


def test_game_result_verifies_official_winner_not_predicted_side_only():
    rec, ledger, games, players, status = fixture()
    pick = next(r for r in rec["results"] if r["market"] == "moneyline")
    pick["actual"] = "B"
    with pytest.raises(A.AuditError, match="MONEYLINE_TRUTH_MISMATCH"):
        A.audit(rec, ledger, games, players, status)


def test_reported_graded_count_must_match_snapshot():
    rec, ledger, games, players, status = fixture()
    status["graded_predictions"] += 1
    with pytest.raises(A.AuditError, match="FORWARD_STATUS_GRADED_COUNT_MISMATCH"):
        A.audit(rec, ledger, games, players, status)


def test_cluster_bootstrap_acknowledges_tiny_n_and_is_deterministic():
    rec, ledger, games, players, status = fixture()
    report = A.audit(rec, ledger, games, players, status)
    assert report["overall"]["game_cluster_interval"]["status"] == "TOO_FEW_INDEPENDENT_GAMES"
    items = []
    for i in range(8):
        items.extend([
            {"season": 2026, "game_id": f"G{i}", "model_prob": 0.65, "result": "hit" if i % 2 else "miss"},
            {"season": 2026, "game_id": f"G{i}", "model_prob": 0.75, "result": "hit"},
        ])
    a = A._metrics(items)
    b = A._metrics(items)
    assert a == b
    assert a["game_cluster_interval"]["status"] == "DESCRIPTIVE_CLUSTER_BOOTSTRAP"
    assert a["mean_claimed_probability"] == 0.7
    assert a["actual_hit_rate"] == 0.75
    assert a["overconfidence_gap_positive_is_overclaim"] == -0.05
    assert sum(x["n"] for x in a["probability_buckets"]) == 16

"""Raw event-field semantics tests: only actual source cooccurrence counts."""
import csv
import hashlib

import pytest

import cfb_pass_td_source_semantics_a as S

FIELDS = ["completion_player_id", "touchdown_player_id",
          "reception_player_id", "rush_player_id",
          "interception_thrown_player_id"]


def write_rows(tmp_path, rows):
    p = tmp_path / "player_stats.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return p


def test_positive_completion_touchdown_overlap_and_independent_rush(tmp_path):
    p = write_rows(tmp_path, [
        {"completion_player_id": "QB1", "touchdown_player_id": "WR1",
         "reception_player_id": "WR1"},
        {"rush_player_id": "RB1", "touchdown_player_id": "RB1"},
        {"completion_player_id": "QB1", "interception_thrown_player_id": "QB1"},
        {"completion_player_id": "QB1"},
    ])
    report = S.scan(p)
    c = report["counts"]
    assert c["source_rows"] == 4
    assert c["completion_rows"] == 3
    assert c["touchdown_identity_rows"] == 2
    assert c["completion_with_touchdown_identity"] == 1
    assert c["reception_with_touchdown_identity"] == 1
    assert c["rush_with_touchdown_identity"] == 1
    assert c["pass_interception_rows"] == 1
    assert report["existing_completion_plus_td_rule_has_any_positive_evidence"]
    assert report["positive_overlap_does_not_alone_certify_true_passing_touchdown_semantics"]
    assert report["source_file_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_structural_all_zero_passing_td_labels_cannot_be_claimed_valid(tmp_path):
    p = write_rows(tmp_path, [
        {"completion_player_id": "QB1"},
        {"completion_player_id": "QB1"},
        {"rush_player_id": "RB1", "touchdown_player_id": "RB1"},
    ])
    report = S.scan(p)
    assert report["counts"]["completion_rows"] == 2
    assert report["counts"]["touchdown_identity_rows"] == 1
    assert report["counts"]["completion_with_touchdown_identity"] == 0
    assert report["existing_completion_plus_td_rule_has_any_positive_evidence"] is False


def test_missing_semantic_columns_cannot_silently_pass(tmp_path):
    p = tmp_path / "broken.csv"
    p.write_text("completion_player_id,touchdown_player_id\nQB1,WR1\n")
    with pytest.raises(ValueError, match="REQUIRED_REAL_PLAYER_STAT_COLUMNS_MISSING"):
        S.scan(p)

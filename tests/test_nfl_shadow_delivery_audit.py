"""Adversarial parity tests for the read-only real-delivery gate.

Fixtures simulate timestamps and states; passing these is NOT a real-live receipt.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_nfl_new_engine_frontend import make_state, rec, G1, C24, C90
import nfl_phase1_publisher as PUB
import nfl_shadow_delivery_audit as AUD

NOW = datetime(2026, 10, 4, 12, 5, tzinfo=timezone.utc)


def build_fixture(tmp_path, t90="DONE"):
    r24 = rec(G1, "T24", C24, "alpha", "rush_yds")
    r90 = rec(G1, "T90", C90, "alpha", "rush_yds")
    st = make_state(tmp_path, [
        (G1, "T24", C24, ["PLANNED", "DONE"], [r24], {}),
        (G1, "T90", C90, ["PLANNED", t90], [r90] if t90 == "DONE" else [], {}),
    ])
    out = tmp_path / "public.json"
    PUB.publish(st, out)
    return st, out


def test_delivery_passes_only_identical_verified_ids(tmp_path):
    st, out = build_fixture(tmp_path)
    result = AUD.audit(st, out, NOW, G1)
    assert result["status"] == "PASS"
    assert result["n_due"] == result["n_delivered"] == 2
    assert all(x["expected_verified_rows"] == x["published_rows"] == 1 for x in result["receipts"])
    assert all(len(x["matching_receipt_ids"]) == 1 for x in result["receipts"])
    assert all(x["source_retrieved_at"] and x["model_generated_at"] for x in result["receipts"])


def test_missing_public_row_is_a_failure_not_no_eligible(tmp_path):
    st, out = build_fixture(tmp_path)
    doc = json.loads(out.read_text())
    doc["forecasts"]["T90"] = []
    out.write_text(json.dumps(doc))
    result = AUD.audit(st, out, NOW, G1)
    assert result["status"] == "FAIL"
    assert result["n_recent_failures"] == 1
    bad = next(r for r in result["receipts"] if r["horizon"] == "T90")
    assert bad["verdict"] == "PUBLICATION_MISMATCH"
    assert bad["missing_public_ids"]


def test_genuine_missing_cutoff_is_not_counted_delivered(tmp_path):
    st, out = build_fixture(tmp_path, "MISSED_REAL_CUTOFF")
    result = AUD.audit(st, out, NOW, G1)
    assert result["status"] == "FAIL"
    assert result["n_delivered"] == 1
    assert next(r for r in result["receipts"] if r["horizon"] == "T90")["verdict"] == "MISSED_CUTOFF"


def test_duplicate_public_receipts_are_a_failure(tmp_path):
    st, out = build_fixture(tmp_path)
    doc = json.loads(out.read_text())
    doc["forecasts"]["T90"].append(list(doc["forecasts"]["T90"][0]))
    out.write_text(json.dumps(doc))
    result = AUD.audit(st, out, NOW, G1)
    assert result["status"] == "FAIL"
    assert next(r for r in result["receipts"] if r["horizon"] == "T90")["verdict"] == "DUPLICATE_PUBLIC_IDS"


def test_published_unknown_game_never_claimed_delivered(tmp_path):
    st, out = build_fixture(tmp_path)
    doc = json.loads(out.read_text())
    ghost = list(doc["forecasts"]["T90"][0])
    ghost[doc["row_fields"].index("game_id")] = "2026_04_FAKE_GAME"
    ghost[doc["row_fields"].index("id")] = "fake-receipt"
    doc["forecasts"]["T90"].append(ghost)
    out.write_text(json.dumps(doc))
    result = AUD.audit(st, out, NOW)
    assert result["status"] == "FAIL"
    assert result["unknown_public_games"]


def test_no_cutoff_due_is_not_falsely_marked_a_success(tmp_path):
    st, out = build_fixture(tmp_path)
    earlier = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
    result = AUD.audit(st, out, earlier, G1)
    assert result["status"] == "NOT_DUE"
    assert result["n_due"] == 0

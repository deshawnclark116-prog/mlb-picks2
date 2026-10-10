"""NHL Phase1C offline source admission: adversarial authorization and timing tests."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

import nhl_v2_phase1c_ingest as I
from nhl_v2_phase1b_snapshots import SnapshotStore

NOW = datetime(2026, 10, 11, 20, 0, tzinfo=timezone.utc)
KICK = (NOW + timedelta(minutes=100)).isoformat().replace("+00:00", "Z")


def packet(kind="licensed_deployment"):
    return {
        "schema": I.PACKET_SCHEMA, "source_id": "contract-provider-A",
        "source_document_id": "immutable-provider-payload-123",
        "source_kind": kind, "game_id": 2026020401,
        "game_start_utc": KICK, "home_team_id": 1, "away_team_id": 2,
        "horizon": "T90", "freshness_seconds": 900,
        "published_at": (NOW - timedelta(minutes=3)).isoformat().replace("+00:00", "Z"),
        "observations": [
            {"player_id": 1001, "team_id": 1, "observed_state": "EXPECTED_DRESSED",
             "role": {"ev_line": 2, "pp_unit": 1}, "membership_verified": True,
             "membership_evidence": "roster-authorized-123"},
            {"player_id": 1002, "team_id": 2, "observed_state": "ROSTER_OBSERVED"},
        ]
    }


def grant(kind="licensed_deployment"):
    return {
        "schema": I.GRANT_SCHEMA, "source_id": "contract-provider-A",
        "allowed_source_kind": kind, "review_status": "OPERATOR_REVIEWED",
        "agreement_reference": "synthetic-test-contract-NOT_REAL_PERMISSION",
        "authorized_by": "test-fixture-only",
        "authorized_on_utc": (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "rights_scope_description": "synthetic test observations",
        **{k: True for k in I.REQUIRED_RIGHTS}
    }


def raw(p):
    return json.dumps(p, separators=(",", ":"), sort_keys=True).encode()


def test_rights_attested_live_packet_is_only_uncertified_research(tmp_path):
    obj = packet()
    b = raw(obj)
    record, info = I.normalize_packet(b, grant(), received_at=NOW)
    assert record["raw_source_sha256"] == info["packet_sha256"]
    assert record["timing_eligible"]
    assert info["both_teams_observed"] is True
    assert info["dressed_roster_certified"] is False
    assert info["new_model_training_enabled"] is False
    assert info["rights_status"] == I.RIGHTS_ATTESTATION
    assert record["retrieved_at"] == NOW.isoformat().replace("+00:00", "Z")
    store = SnapshotStore(tmp_path / "research")
    assert store.append(record, b) is True
    assert store.append(record, b) is False
    assert store.state_at(
        {"game_id": obj["game_id"], "game_start_utc": KICK,
         "home_team_id": 1, "away_team_id": 2},
        "T90", record["cutoff"]
    )[0]["record_sha256"] == record["record_sha256"]
    store.ledger.verify()


@pytest.mark.parametrize("right", I.REQUIRED_RIGHTS)
def test_each_permission_is_mandatory(right):
    g = grant()
    g[right] = False
    with pytest.raises(I.SourceAdmissionError, match="RIGHT_NOT_APPROVED"):
        I.normalize_packet(raw(packet()), g, received_at=NOW)


def test_no_backfill_even_if_packet_claims_old_timestamp():
    p = packet()
    p["retrieved_at"] = (NOW - timedelta(days=2)).isoformat()
    with pytest.raises(I.SourceAdmissionError, match="LATE_INGESTION"):
        I.normalize_packet(raw(p), grant(), received_at=NOW + timedelta(minutes=11))


def test_roster_api_cannot_claim_dressed_or_deployment():
    p = packet("game_roster")
    with pytest.raises(I.SourceAdmissionError, match="ROSTER_SOURCE_CANNOT_ASSERT_DEPLOYMENT"):
        I.normalize_packet(raw(p), grant("game_roster"), received_at=NOW)
    p["observations"][0].pop("role")
    p["observations"][0]["observed_state"] = "CONFIRMED_DRESSED"
    p["observations"][0]["explicit_confirmation_evidence"] = "artificial"
    with pytest.raises(I.SourceAdmissionError, match="ROSTER_SOURCE_NOT_CONFIRMED"):
        I.normalize_packet(raw(p), grant("game_roster"), received_at=NOW)


def test_missing_confirmation_proof_and_future_revision_rejected():
    p = packet()
    p["observations"][0]["observed_state"] = "CONFIRMED_DRESSED"
    with pytest.raises(I.SourceAdmissionError, match="confirmation_evidence"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)
    p["observations"][0]["explicit_confirmation_evidence"] = "explicit-licensed-feed-123"
    p["observations"][0]["effective_at"] = (NOW + timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
    with pytest.raises(I.SourceAdmissionError, match="POST_RECEIPT_EFFECTIVE"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)


def test_future_publication_and_rights_are_rejected():
    p = packet()
    p["published_at"] = (NOW + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(I.SourceAdmissionError, match="SOURCE_PUBLICATION_AFTER_RECEIPT"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)
    g = grant()
    g["authorized_on_utc"] = (NOW + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(I.SourceAdmissionError, match="FUTURE_RIGHTS"):
        I.normalize_packet(raw(packet()), g, received_at=NOW)


def test_foreign_team_duplicate_player_and_missing_membership_proof_rejected():
    p = packet()
    p["observations"][0]["team_id"] = 42
    with pytest.raises(I.SourceAdmissionError, match="INVALID_PLAYER_TEAM_IDENTITY"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)
    p = packet()
    p["observations"].append(copy.deepcopy(p["observations"][0]))
    with pytest.raises(I.SourceAdmissionError, match="DUPLICATE_PLAYER_TEAM"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)
    p = packet()
    p["observations"][0].pop("membership_evidence")
    with pytest.raises(I.SourceAdmissionError, match="membership_evidence"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)


def test_provider_wrong_source_kind_unapproved_rights_and_empty_packet_blocked():
    with pytest.raises(I.SourceAdmissionError, match="GRANT_SOURCE_KIND_MISMATCH"):
        I.normalize_packet(raw(packet()), grant("authorized_announcement"), received_at=NOW)
    g = grant()
    g["review_status"] = "PENDING"
    with pytest.raises(I.SourceAdmissionError, match="GRANT_NOT_REVIEWED"):
        I.normalize_packet(raw(packet()), g, received_at=NOW)
    p = packet()
    p["observations"] = []
    with pytest.raises(I.SourceAdmissionError, match="EMPTY_OBSERVATION"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)


def test_cross_team_duplicate_player_is_quarantined_not_chosen():
    p = packet()
    p["observations"][1]["player_id"] = p["observations"][0]["player_id"]
    with pytest.raises(I.SourceAdmissionError, match="CONFLICTING_PLAYER_TEAM_IDENTITY"):
        I.normalize_packet(raw(p), grant(), received_at=NOW)


def test_rights_document_is_content_addressed():
    b = raw(packet())
    grant_a = grant()
    _, a = I.normalize_packet(b, grant_a, received_at=NOW)
    assert len(a["grant_sha256"]) == 64
    grant_b = copy.deepcopy(grant_a)
    grant_b["rights_scope_description"] = "synthetic test scope modified"
    _, c = I.normalize_packet(b, grant_b, received_at=NOW)
    assert a["grant_sha256"] != c["grant_sha256"]


def test_real_original_publication_vintage_not_implied_when_missing():
    p = packet()
    p["published_at"] = None
    rec, info = I.normalize_packet(raw(p), grant(), received_at=NOW)
    assert info["publication_time_status"] == "UNKNOWN_ORIGINAL_PUBLICATION_VINTAGE"
    assert rec["retrieved_at"] == NOW.isoformat().replace("+00:00", "Z")

"""Phase1C authorization-gated ingestion of *supplied* pregame NHL evidence.

No network access and no automatic provider scraper. Packet rows are normalized
observations supplied with an explicit operator-attested, scoped rights grant.
Packet bytes become an immutable content-addressed research source in the
existing Phase1B SnapshotStore. Runtime reception time is the authoritative
as-observed timestamp; packet-authored timestamps cannot backdate a capture.
The importer does NOT certify a lineup, model a forecast or qualify a source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhl_v2_phase1a_sog_forward import HORIZON_MIN
from nhl_v2_phase1b_snapshots import SnapshotStore, snapshot, timestamp

PACKET_SCHEMA = "nhl-phase1c-pregame-packet-v1"
GRANT_SCHEMA = "nhl-phase1c-rights-grant-v1"
REQUIRED_RIGHTS = ("access_authorized", "automated_collection_allowed",
                   "internal_storage_allowed", "modeling_allowed",
                   "retain_raw_evidence_allowed")
RIGHTS_ATTESTATION = "OPERATOR_ATTESTED_NOT_INDEPENDENTLY_LEGAL_VERIFIED"
CONFIRMATION_KINDS = {"licensed_deployment", "authorized_announcement"}
ALLOWED_ROLES = {"ev_line", "pp_unit", "pk_unit", "defensive_pair"}
ALLOWED_SOURCE_KINDS = {"game_roster", "official_team_roster",
                        "licensed_deployment", "authorized_announcement"}
KNOWN_STATES = {"ROSTER_OBSERVED", "EXPECTED_DRESSED", "CONFIRMED_DRESSED",
                "SCRATCH_OBSERVED", "INJURED_OBSERVED", "UNKNOWN"}


class SourceAdmissionError(ValueError):
    pass


def _required_string(data, key, where):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SourceAdmissionError(f"{where}.{key}: nonempty text required")
    return value


def _runtime(now):
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise SourceAdmissionError("UTC-aware capture time required")
    return now.astimezone(timezone.utc)


def _grant_check(grant, packet, now):
    if grant.get("schema") != GRANT_SCHEMA:
        raise SourceAdmissionError("GRANT_SCHEMA_UNRECOGNIZED")
    sid = _required_string(packet, "source_id", "packet")
    if sid != _required_string(grant, "source_id", "grant"):
        raise SourceAdmissionError("GRANT_PROVIDER_MISMATCH")
    for key in ("agreement_reference", "authorized_by", "rights_scope_description",
                "authorized_on_utc"):
        _required_string(grant, key, "grant")
    if grant.get("review_status") != "OPERATOR_REVIEWED":
        raise SourceAdmissionError("GRANT_NOT_REVIEWED")
    for right in REQUIRED_RIGHTS:
        if grant.get(right) is not True:
            raise SourceAdmissionError(f"RIGHT_NOT_APPROVED: {right}")
    if timestamp(grant["authorized_on_utc"]) > now:
        raise SourceAdmissionError("FUTURE_RIGHTS_AUTHORIZATION")
    if grant.get("allowed_source_kind") != packet.get("source_kind"):
        raise SourceAdmissionError("GRANT_SOURCE_KIND_MISMATCH")
    if packet.get("source_kind") not in ALLOWED_SOURCE_KINDS:
        raise SourceAdmissionError("UNKNOWN_SOURCE_KIND")
    return sid + ":" + grant["agreement_reference"]


def normalize_packet(packet_raw: bytes, grant: dict, *, received_at=None) -> tuple[dict, dict]:
    """Admission is prospective only; the test clock is injectable, CLI clock is not."""
    now = _runtime(received_at)
    try:
        packet = json.loads(packet_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAdmissionError("PACKET_NOT_JSON") from exc
    if not isinstance(packet, dict) or packet.get("schema") != PACKET_SCHEMA:
        raise SourceAdmissionError("PACKET_SCHEMA_UNRECOGNIZED")
    rights = _grant_check(grant, packet, now)
    grant_sha256 = hashlib.sha256(json.dumps(grant, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rights += ":grant_sha256=" + grant_sha256
    source = _required_string(packet, "source_document_id", "packet")
    gid = packet.get("game_id")
    if not isinstance(gid, int) or isinstance(gid, bool) or gid <= 0:
        raise SourceAdmissionError("INVALID_GAME_ID")
    teams = (packet.get("home_team_id"), packet.get("away_team_id"))
    if any(not isinstance(t, int) or isinstance(t, bool) or t <= 0 for t in teams) or teams[0] == teams[1]:
        raise SourceAdmissionError("INVALID_OPPOSING_TEAMS")
    h = packet.get("horizon")
    if h not in HORIZON_MIN:
        raise SourceAdmissionError("INVALID_HORIZON")
    kick = timestamp(_required_string(packet, "game_start_utc", "packet"))
    cutoff = kick - timedelta(minutes=HORIZON_MIN[h])
    if now > cutoff:
        raise SourceAdmissionError("LATE_INGESTION_AFTER_CUTOFF_NO_REPLAY")
    pub = packet.get("published_at")
    if pub and timestamp(pub) > now:
        raise SourceAdmissionError("SOURCE_PUBLICATION_AFTER_RECEIPT")
    freshness = packet.get("freshness_seconds", 900)
    if not isinstance(freshness, int) or isinstance(freshness, bool) or not 0 < freshness <= 21600:
        raise SourceAdmissionError("INVALID_FRESHNESS_POLICY")
    observations = packet.get("observations")
    if not isinstance(observations, list) or not observations:
        raise SourceAdmissionError("EMPTY_OBSERVATION_NOT_COVERAGE")
    unique = set()
    player_team = {}
    validated = []
    for item in observations:
        if not isinstance(item, dict):
            raise SourceAdmissionError("INVALID_OBSERVATION")
        pid, team = item.get("player_id"), item.get("team_id")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or team not in teams:
            raise SourceAdmissionError("INVALID_PLAYER_TEAM_IDENTITY")
        ident = (pid, team)
        if ident in unique:
            raise SourceAdmissionError("DUPLICATE_PLAYER_TEAM_IDENTITY")
        unique.add(ident)
        if pid in player_team and player_team[pid] != team:
            raise SourceAdmissionError("CONFLICTING_PLAYER_TEAM_IDENTITY")
        player_team[pid] = team
        state = item.get("observed_state")
        if state not in KNOWN_STATES:
            raise SourceAdmissionError("INVALID_OBSERVATION_STATE")
        observation = {"player_id": pid, "team_id": team, "observed_state": state,
                       "source_document_id": source, "source_id": packet["source_id"]}
        if item.get("membership_verified") is True:
            membership_ref = _required_string(item, "membership_evidence", "observation")
            observation.update({"membership_verified": True, "membership_evidence": membership_ref})
        role = item.get("role") or {}
        if not isinstance(role, dict) or set(role) - ALLOWED_ROLES:
            raise SourceAdmissionError("INVALID_DEPLOYMENT_FIELDS")
        for key, number in role.items():
            if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 5:
                raise SourceAdmissionError("INVALID_ROLE_ORDINAL")
        if role:
            if packet["source_kind"] not in CONFIRMATION_KINDS:
                raise SourceAdmissionError("ROSTER_SOURCE_CANNOT_ASSERT_DEPLOYMENT")
            observation["role"] = role
        if state == "CONFIRMED_DRESSED":
            if packet["source_kind"] not in CONFIRMATION_KINDS:
                raise SourceAdmissionError("ROSTER_SOURCE_NOT_CONFIRMED_DRESSED")
            confirmation = _required_string(item, "explicit_confirmation_evidence", "observation")
            observation["explicit_confirmation_evidence"] = confirmation
        if item.get("effective_at"):
            if timestamp(item["effective_at"]) > now:
                raise SourceAdmissionError("POST_RECEIPT_EFFECTIVE_REVISION")
            observation["effective_at"] = item["effective_at"]
        if item.get("published_at"):
            if timestamp(item["published_at"]) > now:
                raise SourceAdmissionError("POST_RECEIPT_PUBLICATION_REVISION")
            observation["published_at"] = item["published_at"]
        validated.append(observation)
    game = {"game_id": gid, "game_start_utc": kick.isoformat().replace("+00:00", "Z"),
            "home_team_id": teams[0], "away_team_id": teams[1]}
    record = snapshot(packet_raw, game=game, horizon=h,
                      cutoff=cutoff.isoformat().replace("+00:00", "Z"),
                      retrieved_at=now.isoformat().replace("+00:00", "Z"),
                      source_kind=packet["source_kind"],
                      rights_basis=rights,
                      published_at=pub, freshness_seconds=freshness,
                      observations=validated)
    if not record["timing_eligible"]:
        raise SourceAdmissionError("SOURCE_FAILS_TIMESTAMP_ELIGIBILITY")
    # An operator attestation is not source authentication, vendor certification,
    # final-dressed truth, an achieved coverage gate, or a license adjudication.
    diagnostics = {
        "packet_sha256": hashlib.sha256(packet_raw).hexdigest(),
        "grant_sha256": grant_sha256,
        "source_id": packet["source_id"], "source_document_id": source,
        "status": "IMPORTED_OPERATOR_ATTESTED_UNCERTIFIED",
        "rights_status": RIGHTS_ATTESTATION,
        "actual_received_at_utc": record["retrieved_at"],
        "pregame_timestamp_eligible": True,
        "publication_time_status": ("PROVIDER_CLAIMED_NOT_VERIFIED"
                                    if pub else "UNKNOWN_ORIGINAL_PUBLICATION_VINTAGE"),
        "observed_player_count": len(validated),
        "both_teams_observed": {r["team_id"] for r in validated} == set(teams),
        "dressed_roster_certified": False,
        "new_model_training_enabled": False,
    }
    return record, diagnostics


def main():
    ap = argparse.ArgumentParser(description="Authorized offline NHL research observation import; no HTTP")
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--grant", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True, help="Independent Phase1C store only")
    opts = ap.parse_args()
    raw = opts.packet.read_bytes()
    grant = json.loads(opts.grant.read_text())
    record, diagnostics = normalize_packet(raw, grant)
    stored = SnapshotStore(opts.state).append(record, raw)
    print(json.dumps({**diagnostics, "store_write": "NEW" if stored else "DUPLICATE",
                      "record_sha256": record["record_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

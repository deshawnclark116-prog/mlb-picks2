"""NHL V2 public-facing research projections must be sourced and count-based.

No fixture below can create/publish an original pregame forecast. These
synthetic source bytes only exercise the read-only output contract.
"""
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

import nhl_v2_page_publisher as P


def fixture(tmp_path, *, horizon="T90", player_id=100, team="BOS"):
    root=tmp_path/"nhl_models/nhl_player_outcome_v2/phase1a_sog_forward"
    (root/"blobs").mkdir(parents=True)
    (root/"manifests").mkdir()
    raw=b'{"event":"source sample"}'
    source_sha=hashlib.sha256(raw).hexdigest()
    (root/"blobs"/(source_sha+".gz")).write_bytes(gzip.compress(raw))
    manifest={"run_id":"synthetic","generated_at":"2026-10-10T17:25:00Z",
              "decision_cutoffs":["2026-10-10T17:30:00Z"],
              "sources":[{"predictive":True,"kind":"schedule",
                          "retrieval_completed_utc":"2026-10-10T17:20:00Z",
                          "completeness_status":"COMPLETE","http_status":200,
                          "sha256":source_sha}],
              "engine_version":"nhl-v2-sog-b2-1.1"}
    manifest["manifest_sha256"]=hashlib.sha256(P.canonical(manifest).encode()).hexdigest()
    (root/"manifests"/(manifest["manifest_sha256"]+".json")).write_text(P.canonical(manifest)+"\n")
    row={"record_type":"FORECAST","forecast_id":f"2026020001-{horizon}-20261010T190000-{player_id}",
         "game_id":2026020001,"player_id":player_id,"team":team,
         "opponent":"PHI" if team=="BOS" else "BOS",
         "scheduled_start":"2026-10-10T19:00:00Z",
         "schedule_date":"2026-10-10",
         "forecast_horizon":horizon,"cutoff_at":"2026-10-10T17:30:00Z",
         "generated_at":"2026-10-10T17:25:00Z","engine_version":"nhl-v2-sog-b2-1.1",
         "schedule_state":"OK","source_manifest_sha256":manifest["manifest_sha256"],
         "expected_sog":2.3,"median_sog":2.0,"variance":3.2,"dispersion":0.35,
         # DELIBERATELY WRONG legacy fields -- must never be republished.
         "P1":1.0,"P2":0.65,"P3":0.35,"P4":0.2,"P5":0.08,
         "meaningful_expected_participant":True,
         "availability_confidence":"NOT_CERTIFIED","availability_state":"NO_ROSTER_DATA_PUBLISHED_YET",
         "receipt":{"player_name":"Research Skater","position":"F",
                    "toi_seconds":{"recent3":970},"recent_sog_history_last10_appearances":[1,2,3],
                    "opp_sog_allowed_mean5":29.4,"pp_toi_recent3_seconds":73},
         "comparators":{"simple_prior10_mean":2.1}}
    return root,row


def chain(*rows):
    out=[];prev=P.GENESIS
    for i,r in enumerate(rows):
        r=dict(r,seq=i,prev_hash=prev)
        r["row_hash"]=P.digest(P.canonical(r).encode())
        out.append(r);prev=r["row_hash"]
    return out


def test_nhl_count_projection_directly_visible_and_legacy_threshold_not_reused(tmp_path):
    root,row=fixture(tmp_path)
    frozen=chain(row)
    data=P.build(frozen,root,"frozen-test")
    assert data["published_player_rows"]==1
    assert data["official_betting_picks"]==0
    assert data["default_horizon"]=="T90"
    r=data["forecasts"]["T90"][0]
    assert r["mean"]==2.3 and r["median"]==2
    assert r["p10"]<=r["median"]<=r["p90"]
    assert r["probabilities_posthoc"]["ge1"]<1
    assert 0<r["probabilities_posthoc"]["ge3"]<r["probabilities_posthoc"]["ge2"]
    assert r["availability_verified"] is False
    assert r["row_hash"]==frozen[0]["row_hash"]
    assert "pick" not in r and "line" not in r
    assert "P1" not in r and "P3" not in r
    assert data["original_ledger_modified"] is False


def test_correct_negative_binomial_integer_survival_and_quantiles():
    quantiles,probs=P.nb2(2,0.5)
    r=2.0;p=0.5
    # P(X >= 1) = 1 - p**r, P(X >= 2) subtract P1.
    expected_ge1=1-p**r
    expected_ge2=1-p**r-r*(1-p)*p**r
    assert probs["ge1"]==pytest.approx(expected_ge1,abs=1e-6)
    assert probs["ge2"]==pytest.approx(expected_ge2,abs=1e-6)
    assert quantiles["p10"]<=quantiles["median"]<=quantiles["p90"]


def test_conflicting_player_team_duplicated_id_quarantines_both_rows(tmp_path):
    root,r=fixture(tmp_path)
    second=dict(r,team="PHI",opponent="BOS",expected_sog=4.2)
    original=chain(r,second)
    data=P.build(original,root,"fixture")
    assert data["published_player_rows"]==0
    assert data["quarantined_records"]==2
    assert data["exclusion_reasons"]["DUPLICATE_PLAYER_OR_FORECAST_ID"]==2
    assert len(original)==2 and original[0]["team"]=="BOS"


@pytest.mark.parametrize("mutator,reason",[
    (lambda rows: rows[0].update(expected_sog=9.0),"FROZEN_LEDGER_HASH_INVALID"),
    (lambda rows: rows[0].update(prev_hash="tampered"),"FROZEN_LEDGER_CHAIN_INVALID"),
    (lambda rows: rows[0].update(seq=11),"FROZEN_LEDGER_CHAIN_INVALID"),
])
def test_original_hash_or_chain_tampering_hard_fails(tmp_path,mutator,reason):
    root,r=fixture(tmp_path); rows=chain(r)
    mutator(rows)
    with pytest.raises(P.NHLPublicationError,match=reason):
        P.build(rows,root,"fixture")


def test_predictive_blob_mutation_fails_even_if_forecast_row_unchanged(tmp_path):
    root,r=fixture(tmp_path)
    src=next((root/"blobs").iterdir())
    src.write_bytes(gzip.compress(b"wrong bytes"))
    with pytest.raises(P.NHLPublicationError,match="UNVERIFIED_PREDICTIVE_SOURCE_BLOB"):
        P.build(chain(r),root,"fixture")


def test_predictive_source_late_retrieval_fails_even_if_model_snapshot_locked(tmp_path):
    root,r=fixture(tmp_path)
    key=r["source_manifest_sha256"]
    f=root/"manifests"/(key+".json")
    data=json.loads(f.read_text())
    data["sources"][0]["retrieval_completed_utc"]="2026-10-10T17:40:00Z"
    newhash=P.digest(P.canonical({k:v for k,v in data.items() if k!="manifest_sha256"}).encode())
    data["manifest_sha256"]=newhash
    f.unlink()
    (root/"manifests"/(newhash+".json")).write_text(P.canonical(data)+"\n")
    r["source_manifest_sha256"]=newhash
    with pytest.raises(P.NHLPublicationError,match="LATE_PREDICTIVE_SOURCE"):
        P.build(chain(r),root,"fixture")


def test_post_cutoff_legacy_forecast_cannot_be_backfilled_into_page(tmp_path):
    root,r=fixture(tmp_path)
    r["generated_at"]="2026-10-10T17:50:00Z"
    data=P.build(chain(r),root,"fixture")
    assert data["published_player_rows"]==0
    assert data["exclusion_reasons"]["LATE_OR_UNTIMED_PREGAME_ROW"]==1


def test_missing_source_and_malformed_nbinom_fail_closed(tmp_path):
    root,r=fixture(tmp_path)
    with pytest.raises(P.NHLPublicationError,match="INVALID_DISPERSION"):
        P.nb2(2,-0.01)
    (root/"manifests"/(r["source_manifest_sha256"]+".json")).unlink()
    with pytest.raises(P.NHLPublicationError,match="MISSING_SOURCE_MANIFEST"):
        P.build(chain(r),root,"fixture")


def test_second_nonforecast_record_preserves_chain_not_displayed(tmp_path):
    root,r=fixture(tmp_path)
    other={"record_type":"MISSED_CUTOFF","game_id":2026020001,
           "scheduled_start":"2026-10-10T19:00:00Z"}
    data=P.build(chain(r,other),root,"fixture")
    assert data["source_ledger_rows"]==2
    assert data["original_frozen_forecasts"]==1
    assert data["published_player_rows"]==1


def test_output_readonly_and_source_files_unchanged(tmp_path):
    root,r=fixture(tmp_path)
    rows=chain(r)
    path=root/"ledger.jsonl"
    path.write_text("\n".join(P.canonical(x) for x in rows)+"\n")
    original=path.read_bytes()
    outfile=tmp_path/"docs"/"nhl_v2_page.json"
    P.publish(tmp_path,outfile,"frozen-test")
    assert original==path.read_bytes()
    data=json.loads(outfile.read_text())
    assert data["official_betting_picks"]==0
    assert outfile.is_file()

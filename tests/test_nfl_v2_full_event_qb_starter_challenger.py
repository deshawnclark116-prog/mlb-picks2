"""Live simulator starter-role experiment must conserve team allocations."""
import copy
import numpy as np
import pytest
import nfl_v2_full_event_qb_starter_challenger as M

def fixture():
    return {"PHI":{"types":{
       "qb_att":{"ids":["HURTS","MCKEE","MILTON"],
                 "P1":np.array([.75,.22,.08]),
                 "pact24":np.array([.0057,.1852,.0796]),
                 "pact90":np.array([.005,.2,.08])},
       "carry":{"ids":["HURTS","MCKEE","PIERCE"],
                "P1":np.array([.16,.02,.72]),
                "pact24":np.array([.0057,.1852,.9237]),
                "pact90":np.array([.0,.22,.99])},
       "rz_carry":{"ids":["HURTS","MCKEE","PIERCE"],
                   "P1":np.array([.3,.04,.66]),
                   "pact24":np.array([.0057,.1852,.9237]),
                   "pact90":np.array([.0,.22,.99])},
       "target":{"ids":["SMITH","BROWN"],
                 "P1":np.array([.6,.4]),
                 "pact24":np.array([.96,.9]),"pact90":np.array([.9,.9])}
       }},
        "CAR":{"types":{"qb_att":{"ids":["YOUNG"],
                         "P1":np.array([.92]),
                         "pact24":np.array([.98]),"pact90":np.array([.98])}}}}

def test_full_game_qb_starter_role_transfer_not_talent_transfer():
    a=fixture();original=copy.deepcopy(a)
    b,proof=M.role_override_games(a,team="PHI",starter_id="MCKEE",
              qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    assert proof["original_starter_play_probability"]==.1852
    assert proof["position_prior_source"].startswith("highest")
    np.testing.assert_allclose(b["PHI"]["types"]["qb_att"]["P1"],[0,.75,0])
    np.testing.assert_allclose(b["PHI"]["types"]["qb_att"]["pact24"],[0,1,0])
    np.testing.assert_allclose(b["PHI"]["types"]["carry"]["pact24"],[0,1,.9237])
    np.testing.assert_allclose(b["PHI"]["types"]["rz_carry"]["pact24"],[0,1,.9237])
    assert b["PHI"]["types"]["carry"]["P1"][0]==.16
    np.testing.assert_array_equal(b["PHI"]["types"]["target"]["pact24"],
                                  original["PHI"]["types"]["target"]["pact24"])
    np.testing.assert_array_equal(b["CAR"]["types"]["qb_att"]["P1"],
                                  original["CAR"]["types"]["qb_att"]["P1"])
    for name in ("qb_att","carry","rz_carry","target"):
        for key in original["PHI"]["types"][name]:
            if isinstance(original["PHI"]["types"][name][key],np.ndarray):
                np.testing.assert_array_equal(original["PHI"]["types"][name][key],
                                             a["PHI"]["types"][name][key])

def test_qb_same_league_dropbacks_conserved_under_fixed_outside_bucket():
    from nfl_phase1c_sim import alloc_given
    before=fixture()
    after,_=M.role_override_games(before,team="PHI",starter_id="MCKEE",
                  qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    q=after["PHI"]["types"]["qb_att"]
    ns=2000
    counts,other,active=alloc_given(q["pact24"],q["P1"],.04,None,
             np.full(ns,30,dtype=np.int64),np.random.default_rng(42))
    assert np.all(counts.sum(1)+other==30)
    assert np.all(counts[:,0]==0) and np.all(counts[:,2]==0)
    assert np.all(active[:,1]) and not active[:,0].any()
    assert counts[:,1].mean()>25  # role now owns team's 30-pass budget

def test_wrong_id_and_invalid_propensity_rejected_without_model_mutation():
    d=fixture()
    with pytest.raises(ValueError,match="STARTER_NOT_EXACT"):
        M.role_override_games(d,team="PHI",starter_id="WRONG",
                             qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")
    d["PHI"]["types"]["qb_att"]["P1"][:]=0
    with pytest.raises(ValueError,match="INVALID_FROZEN_QB_ROLE"):
        M.role_override_games(d,team="PHI",starter_id="MCKEE",
                             qb_ids=["HURTS","MCKEE","MILTON"],horizon="T24")

def test_t90_starter_condition_and_roster_lineage():
    d=fixture()
    o,p=M.role_override_games(d,team="PHI",starter_id="MCKEE",
                         qb_ids=["HURTS","MCKEE","MILTON"],horizon="T90")
    np.testing.assert_allclose(o["PHI"]["types"]["qb_att"]["pact90"],[0,1,0])
    np.testing.assert_allclose(o["PHI"]["types"]["carry"]["pact90"],[0,1,.99])
    assert p["horizon"]=="T90"
    assert p["QB_efficiency_models_not_substituted"] is True

def test_durable_pre_cutoff_original_blobs_can_restore_without_network(tmp_path):
    import gzip,hashlib,json
    raw=b"ORIGINAL_HASH_VERIFIED_SNAPSHOT_SOURCE_FROM_SUNDAY_T24"
    sha=hashlib.sha256(raw).hexdigest()
    directory=tmp_path/"durable_blobs"/"cas"/sha
    directory.mkdir(parents=True)
    zipped=gzip.compress(raw,mtime=0)
    (directory/"00000.gz").write_bytes(zipped)
    (directory/"manifest.json").write_text(json.dumps({
      "sha256":sha,"bytes":len(raw),"chunks":[
        {"name":"00000.gz","sha256":hashlib.sha256(zipped).hexdigest()}]}))
    proof=M.restore_original_source_blobs(tmp_path,{"files":{"games.csv":sha}})
    assert proof["original_pre_cutoff_blob_count"]==1
    assert proof["rehydrated_from_durable_original_state"]==1
    assert (tmp_path/"cas"/"blobs"/sha[:2]/sha).read_bytes()==raw
    again=M.restore_original_source_blobs(tmp_path,{"files":{"games.csv":sha}})
    assert again["rehydrated_from_durable_original_state"]==0
    assert again["all_original_blob_sha256_verified"] is True


def test_durable_blob_tampering_is_never_assumed_to_be_valid(tmp_path):
    import gzip,hashlib,json
    raw=b"ORIGINAL_CAPTURE"
    sha=hashlib.sha256(raw).hexdigest()
    directory=tmp_path/"durable_blobs"/"cas"/sha
    directory.mkdir(parents=True)
    zipped=gzip.compress(raw,mtime=0)
    (directory/"00000.gz").write_bytes(zipped+b"TAMPERED")
    (directory/"manifest.json").write_text(json.dumps({
      "sha256":sha,"bytes":len(raw),"chunks":[
        {"name":"00000.gz","sha256":hashlib.sha256(zipped).hexdigest()}]}))
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_CHUNK_HASH_MISMATCH"):
        M.restore_original_source_blobs(tmp_path,{"files":{"games.csv":sha}})


def rb_fixture():
    a=fixture()
    ids=["BARKLEY","SHIPLEY","BIGSBY","PIERCE","STEELE"]
    a["PHI"]["types"]["carry"]={
      "ids":ids,"P1":np.array([.33,.33,.20,.60,.46]),
      "pact24":np.array([.0117,.0082,.034,.9237,.3493]),
      "pact90":np.array([.01,.01,.04,.95,.40])}
    a["PHI"]["types"]["rz_carry"]={
      "ids":ids,"P1":np.array([.35,.31,.19,.59,.45]),
      "pact24":np.array([.0117,.0082,.034,.9237,.3493]),
      "pact90":np.array([.01,.01,.04,.95,.40])}
    a["PHI"]["types"]["target"]={
      "ids":ids+["SMITH"],"P1":np.array([.03,.04,.02,.05,.04,.82]),
      "pact24":np.array([.0117,.0082,.034,.9237,.3493,.96]),
      "pact90":np.array([.01,.01,.04,.95,.40,.99])}
    return a


def test_rb_verified_outs_and_available_conditional_active_preserve_team_budget():
    from nfl_phase1c_sim import alloc_given
    a=rb_fixture()
    original=copy.deepcopy(a)
    b,proof=M.condition_rb_availability(
        a,team="PHI",out_ids=["BARKLEY","SHIPLEY","BIGSBY"],
        conditional_active_ids=["PIERCE","STEELE"],horizon="T24")
    expected=np.array([0,0,0,1,1.])
    for market in ("carry","rz_carry"):
        np.testing.assert_array_equal(b["PHI"]["types"][market]["pact24"],expected)
        np.testing.assert_array_equal(b["PHI"]["types"][market]["P1"],
                                      original["PHI"]["types"][market]["P1"])
    np.testing.assert_array_equal(b["PHI"]["types"]["target"]["pact24"][-1:], [.96])
    np.testing.assert_array_equal(a["PHI"]["types"]["carry"]["pact24"],
                                  original["PHI"]["types"]["carry"]["pact24"])
    assert proof["not_game_day_dressed_certification"]
    assert proof["unmodeled_new_callups_handled_only_by_original_outside_bucket"]
    t=b["PHI"]["types"]["carry"]
    n=2000
    played,other,active=alloc_given(t["pact24"],t["P1"],.15,None,
         np.full(n,22,dtype=np.int64),np.random.default_rng(109))
    assert np.all(played.sum(1)+other==22)
    assert not active[:,:3].any()
    assert active[:,3:].all()


def test_rb_role_conditional_rejects_non_snapshot_promoted_blue_and_overlap():
    a=rb_fixture()
    with pytest.raises(ValueError,match="OFFICIAL_RUNNING_BACK_NOT_UNIQUELY"):
        M.condition_rb_availability(a,team="PHI",out_ids=["BARKLEY"],
                                    conditional_active_ids=["BLUE"],horizon="T24")
    with pytest.raises(ValueError,match="OVERLAPPING_OR_EMPTY"):
        M.condition_rb_availability(a,team="PHI",out_ids=["STEELE"],
                                    conditional_active_ids=["STEELE"],horizon="T24")


def test_official_eagles_rb_evidence_cutoff_and_document_contract():
    from datetime import datetime,timezone
    now=datetime(2026,10,17,22,0,tzinfo=timezone.utc)
    kickoff=datetime(2026,10,18,17,0,tzinfo=timezone.utc)
    news={"schema":"NFL_V2_OFFICIAL_EAGLES_RUNNING_BACK_AVAILABILITY_V1",
        "game_id":"2026_06_CAR_PHI","team":"PHI",
        "ruled_out":["Saquon Barkley","Will Shipley","Tank Bigsby"],
        "conditional_healthy":["Dameon Pierce","Carson Steele"],
        "new_unmodeled_additions":["Jaydon Blue"],
        "official_sources":[{"url":"https://www.philadelphiaeagles.com/news/eagles-place-rb-will-shipley-on-injured-reserve-2026-week-6",
                             "published_at_utc":"2026-10-16T20:01:00Z"}]}
    assert M.verify_rb_news(news,now=now,game_id="2026_06_CAR_PHI",kickoff=kickoff)==news
    future=copy.deepcopy(news)
    future["official_sources"][0]["published_at_utc"]="2026-10-18T20:01:00Z"
    with pytest.raises(ValueError,match="SOURCE_NOT_KNOWN_PREGAME"):
        M.verify_rb_news(future,now=now,game_id="2026_06_CAR_PHI",kickoff=kickoff)


def test_original_nfl_page_shows_real_role_scenario_with_frozen_result_not_overwritten():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    html=(root/"docs/nfl.html").read_text()
    assert 'fetch("nfl_phase1_shadow.json"' in html
    assert 'fetch("nfl_v2_full_event_qb_challenger_2026_w06.json"' in html
    assert "ROLE.generated_at_utc" in html
    assert "Date.parse(game.kickoff_utc)" in html
    assert "verified_original_forecast_unchanged === true" in html
    for name in ("Tanner McKee","Dameon Pierce","Carson Steele"):
        assert name in html
    assert "Original forecast unchanged" in html
    assert "Not a game-day lineup certification or sportsbook recommendation" in html

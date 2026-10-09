"""Deterministic football-chain, temporal, hierarchy and no-rescue checks."""
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1i_target_depth as I
import nfl_v2_phase1i_sources as S


def event(week=1, air=5., qb="q", receiver="r", team="A", complete=1., season=2024, **kw):
    return {"key":(season,week),"date":f"{season}-09-{week:02d}","game_id":f"{season}_{week:02d}_{team}",
            "air":air,"bucket":I.bucket(air),"completion":complete,"position":"WR",
            "qb":qb,"receiver":receiver,"team":team,**kw}


def prior():
    p={"depth":[.1,.4,.3,.2],"completion":[.85,.75,.6,.4],"air":[-3.,5.,14.,27.]}
    return {"league":p,"positions":{pos:copy.deepcopy(p) for pos in ("WR","TE","RB")}}


def row(i=0, season=2024, week=9):
    stats=I.sufficient([event(air=a,complete=c) for a,c in [(-3,1),(5,1),(8,1),(14,1),(25,0)]])
    return {"season":season,"week":week,"game_id":f"{season}_{week:02d}_{i//3}","position_group":"WR",
            "qb":"q","receiver":"r","targets":5.,"receptions":4.,"receiving_yards":39.,
            "projected_targets":6.,"component_reconciled":True,"actual_depth":stats,
            "entity_history":{k:copy.deepcopy(stats) for k in ("qb","receiver","pair","opponent")},
            "history_stats":stats,"history_ypt":7.8,"phase1f_ypt":7.6,
            "phase1g_ypt":7.7,"phase1g_air_ypt":4.7,"phase1g_catch":.7,
            "incumbent_yac_per_target":3.}


def spec(family="qb_receiver_pair"):
    return I.model_spec(family,I.P["priors"]["concentrations"][0])


def store(events):
    s=object.__new__(I.DepthStore)
    s.idx={k:{} for k in ("receiver","qb","pair","opponent","team")};s.keys={};s.cache={}
    for e in events:
        for k,identity in (("receiver",e["receiver"]),("qb",(e["qb"],e["position"])),
                           ("pair",(e["qb"],e["receiver"])),("opponent",("B",e["position"])),("team",e["team"])):
            s.idx[k].setdefault(identity,[]).append(e)
    for k,idx in s.idx.items():
        for identity,es in idx.items():
            es.sort(key=lambda e:(e["key"],e["date"],e["game_id"]))
            s.keys[k,identity]=[e["key"] for e in es]
    return s


@pytest.mark.parametrize("air,b",[(-10,0),(-.01,0),(0,1),(9.99,1),(10,2),(19.99,2),(20,3),(65,3)])
def test_locked_depth_edges(air,b):
    assert I.bucket(air)==b


@pytest.mark.parametrize("air",[np.nan,np.inf,-np.inf])
def test_missing_air_is_not_zero(air):
    with pytest.raises(ValueError):I.bucket(air)


def test_pmf_completion_bounds_and_conditional_air_bucket_support():
    q=I.project(row(),spec(),prior())
    assert sum(q["depth_probabilities"])==pytest.approx(1.)
    assert all(0<=x<=1 for x in q["completion_probabilities"])
    a=q["expected_air_within_bucket"]
    assert a[0]<0 and 0<=a[1]<10 and 10<=a[2]<20 and a[3]>=20
    assert q["completed_air_per_target"]==pytest.approx(sum(p*a*c for p,a,c in zip(q["depth_probabilities"],a,q["completion_probabilities"])))


def test_yac_is_bit_identical_not_scaled_by_new_catch_rate():
    r=row();a=I.project(r,spec(),prior());b=I.project(r,spec("depth_catchability"),prior())
    assert a["catch_probability"]!=b["catch_probability"]
    assert a["incumbent_yac_per_target"]==b["incumbent_yac_per_target"]==r["incumbent_yac_per_target"]
    assert b["yards_per_target"]==b["completed_air_per_target"]+r["incumbent_yac_per_target"]
    assert b["direct_receiving_yard_projection"]==r["projected_targets"]*b["yards_per_target"]


def test_qb_and_receiver_both_change_completed_air():
    r=row();a=I.project(r,spec("qb_receiver_hierarchy"),prior())
    other=copy.deepcopy(r);other["qb"]="different";other["entity_history"]["qb"]=I.sufficient([event(air=30) for _ in range(80)])
    b=I.project(other,spec("qb_receiver_hierarchy"),prior())
    assert a["completed_air_per_target"]!=b["completed_air_per_target"]
    other=copy.deepcopy(r);other["receiver"]="different";other["entity_history"]["receiver"]=I.sufficient([event(air=2) for _ in range(50)])
    c=I.project(other,spec("qb_receiver_hierarchy"),prior())
    assert a["completed_air_per_target"]!=c["completed_air_per_target"]


def test_tiny_pair_cannot_override_parent_and_receipt_reports_it():
    r=row();r["entity_history"]["pair"]=I.sufficient([event(air=60) for _ in range(19)])
    a=I.project(r,spec(),prior());b=I.project(r,spec("qb_receiver_hierarchy"),prior())
    assert not a["pair_correction_applied"]
    assert a["completed_air_per_target"]==b["completed_air_per_target"]
    r["entity_history"]["pair"]=I.sufficient([event(air=60) for _ in range(20)])
    a=I.project(r,spec(),prior())
    assert a["pair_correction_applied"] and a["completed_air_per_target"]!=b["completed_air_per_target"]
    assert a["expected_air_within_bucket"][3]<60


def test_receiver_and_pair_skill_follow_stable_ids_across_teams():
    s=store([event(team="OLD"),event(week=2,team="NEW")])
    assert s.stats("receiver","r",(2024,3),"2024-10-01")["n"]==2
    assert s.stats("pair",("q","r"),(2024,3),"2024-10-01")["n"]==2
    assert s.stats("team","NEW",(2024,3),"2024-10-01")["n"]==1


def test_same_week_future_and_incomplete_date_are_excluded():
    es=[event(),event(week=2),event(week=3,date="2024-10-15"),event(week=4),event(season=2025)]
    s=store(es)
    assert s.prior("receiver","r",(2024,2),"2024-09-20")==es[:1]
    assert s.prior("receiver","r",(2024,4),"2024-09-20")==es[:2]


def test_prior_roster_not_target_week_status_or_identity():
    rs={(2024,1,"A"):[{"player_id":"q","position":"QB","status":"INA"}],
        (2024,2,"A"):[{"player_id":"oracle","position":"QB","status":"ACT"}]}
    ids,key=I.prior_roster_ids(rs,"A",(2024,2))
    assert ids=={"q"} and key==(2024,1)
    rs[(2024,2,"A")][0]["player_id"]="poisoned"
    assert I.prior_roster_ids(rs,"A",(2024,2))==(ids,key)
    assert I.prior_roster_ids(rs,"B",(2024,2))==(set(),None)


def test_qb_assignment_is_prior_current_team_not_actual_starter_or_departed_qb():
    s=store([event(qb="departed") for _ in range(50)]+[event(qb="backup")]+[event(week=2,qb="oracle") for _ in range(100)])
    assert s.assign_qb("A",(2024,2),"2024-09-20",{"backup","oracle"})["qb"]=="backup"
    assert s.assign_qb("A",(2024,2),"2024-09-20",{"oracle"})["qb"] is None
    result=s.assign_qb("NEW",(2024,2),"2024-09-20",{"departed"})
    assert result["qb"]=="departed" and "CAREER" in result["source"]


def test_target_week_pbp_poisoning_cannot_change_features(tmp_path):
    fields=["season_type","two_point_attempt","no_play","week","game_id","game_date","posteam","defteam",
            "receiver_player_id","receiver_player_name","passer_player_id","passer_player_name","play_type","complete_pass","air_yards","yards_after_catch","yards_gained"]
    previous=dict(zip(fields,["REG","0","0","1","2024_01_A_B","2024-09-01","A","B","r","Receiver","q","QB","pass","1","5","3","8"]))
    current={**previous,"week":"2","game_id":"2024_02_A_B","game_date":"2024-09-08"}
    def write(changed):
        for year in (2023,2024,2025,2026):
            with gzip.open(tmp_path/f"pbp_{year}.csv.gz","wt") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                if year==2024:w.writerows([previous,changed])
                if year==2026:w.writerow({**changed,"week":"5"})
    write(current);s=I.DepthStore(tmp_path,{(2024,"r"):"WR",(2026,"r"):"WR"})
    before=s.stats("receiver","r",(2024,2),"2024-09-08")
    write({**current,"passer_player_id":"future","complete_pass":"0","air_yards":"1000","yards_after_catch":"900","yards_gained":"2000"})
    after=I.DepthStore(tmp_path,{(2024,"r"):"WR",(2026,"r"):"WR"})
    assert after.stats("receiver","r",(2024,2),"2024-09-08")==before
    assert all(e["key"]!=(2026,5) for e in after.events)


def test_training_priors_ignore_later2024_and2025_2026():
    es=[event(air=a) for a in (-3,5,14,25)]
    p=I.fit_priors(es,8)
    assert I.fit_priors(es+[event(week=9,air=1000),event(season=2025,air=5000),event(season=2026,air=9999)],8)==p
    with pytest.raises(ValueError):I.fit_priors([event(season=2025)],8)


@pytest.mark.parametrize("family",I.FAMILIES)
def test_projection_ignores_all_target_labels_and_sportsbooks(family):
    r=row();before=I.project(r,spec(family),prior())
    r.update(targets=500.,receptions=400.,receiving_yards=100000.,actual_depth=None,
             sportsbook_line=500.,odds=-2000,spread_line=-15)
    assert I.project(r,spec(family),prior())==before


def test_explicit_component_routing():
    r=row();reference=I.project(r,spec("opponent_depth_allowance"),prior())
    changed=copy.deepcopy(r);changed["entity_history"]["opponent"]=I.sufficient([event(air=35,complete=0) for _ in range(200)])
    a=I.project(r,I.model_spec("opponent_depth_allowance",spec()["depth_strengths"],.3),prior())
    b=I.project(changed,I.model_spec("opponent_depth_allowance",spec()["depth_strengths"],.3),prior())
    assert a["depth_probabilities"]!=b["depth_probabilities"]
    assert a["completion_probabilities"]==b["completion_probabilities"]
    assert a["expected_air_within_bucket"]==b["expected_air_within_bucket"]
    c=I.project(r,spec("depth_catchability"),prior())
    assert c["depth_probabilities"]==reference["depth_probabilities"]
    assert c["expected_air_within_bucket"]==reference["expected_air_within_bucket"]


def test_reconciled_denominators_zero_target_and_empty_depth_buckets():
    r=row();q=I.project(r,spec(),prior())
    zero={**r,"targets":0.}
    assert I.metrics([r,zero],[q,q])["n"]==1
    bad={**r,"component_reconciled":False}
    m=I.metrics([r,bad],[q,q])
    assert m["n"]==2 and m["metric_n"]["completed_air_per_target_mae"]==1
    r["actual_depth"]=I.sufficient([event(air=5) for _ in range(5)])
    m=I.metric_values(r,q)
    assert "catch_rate_0_9_mae" in m and "catch_rate_20_plus_mae" not in m
    assert I.comparator(r,"phase1f")["completed_air_per_target"] is None


def test_bootstrap_deterministic_and_whole_game_not_rowwise():
    rs=[row(i,week=9+i//6) for i in range(48)]
    a=I.predict(rs,spec(),prior());b=[I.comparator(r,"phase1g") for r in rs]
    x=I.bootstrap(rs,a,b,"completed_air_per_target_mae")
    assert x==I.bootstrap(rs,a,b,"completed_air_per_target_mae")
    assert x==I.bootstrap(rs+rs,a+a,b+b,"completed_air_per_target_mae")
    assert I.bootstrap(rs,a,a,"completed_air_per_target_mae")["ci95"]==[0.,0.]


def test_no_final_cancellation_or_failed_component_can_pass():
    m={"completed_air_per_target_mae":3.,"oracle_target_rec_yds_mae":17.,"catch_rate_mae":.2,
       "macro_depth_catch_rate_mae":.25,"target_depth_distribution_error":.3,"deep_target_frequency_error":.1}
    boot={"ci95":[-.2,-.1]}
    assert not I.family_gate("receiver_depth_only",{**m,"completed_air_per_target_mae":3.1,"oracle_target_rec_yds_mae":16.},m,boot)["passes"]
    bad={**m,"completed_air_per_target_mae":2.9,"oracle_target_rec_yds_mae":16.9,"catch_rate_mae":.21}
    assert not I.family_gate("depth_catchability",bad,m,boot)["passes"]
    assert not I.overall_gate({**m,"oracle_target_rec_yds_mae":17.1,"full_yards_mae":1.},m,m,m,{"air_vs_g":boot,"oracle_vs_f":boot})["passes"]


def test_development_is_2024_only_no_failed_family_combination(tmp_path):
    audit=tmp_path/"audit.json";audit.write_text("{}")
    es=[event(air=a) for a in (-3,5,14,25)]
    rs=[row(i,week=9+i//6) for i in range(48)]
    lock=I.develop(rs,es,{},audit)
    poison=row(season=2025);poison["receiving_yards"]=9999999.
    assert I.develop(rs+[poison,row(season=2026)],es+[event(season=2025,air=9999)],{},audit)==lock
    for name,s in lock["frozen_specs"].items():
        if "+" in name:
            assert all(lock["candidates"][f]["status"]=="SURVIVES" for f in s["families"])


def test_confirmation_cannot_fit_select_or_score_full_after_failure(monkeypatch,tmp_path):
    audit=tmp_path/"audit.json";audit.write_text("{}")
    lock=I.develop([row(i,week=9+i//6) for i in range(48)],[event(air=a) for a in (-3,5,14,25)],{},audit)
    lock["selected"]=None
    monkeypatch.setattr(I,"fit_priors",lambda *args:pytest.fail("No validation fitting"))
    monkeypatch.setattr(I,"develop",lambda *args:pytest.fail("No validation selection"))
    monkeypatch.setattr(I,"full_metrics",lambda *args:pytest.fail("Failed efficiency cannot score predicted workload"))
    out,receipts=I.confirm([row(season=2025),row(season=2026)],lock)
    assert out["verdict"]=="REJECTED_EFFICIENCY_REPLACEMENT"
    assert out["full_projection_evaluation"]["status"]=="NOT_RUN_EFFICIENCY_GATE_FAILED"
    assert receipts and all(len(r["depth_probabilities"])==4 for r in receipts)


def test_protocol_amendment_after_selection_is_refused(monkeypatch,tmp_path):
    p=tmp_path/"protocol.json";p.write_text("changed");monkeypatch.setattr(I,"PROTOCOL",p)
    with pytest.raises(ValueError,match="Protocol changed"):I.confirm([],{"protocol_sha256":"old"})


def test_source_revisions_and_new_unaudited_provider_file_refused(tmp_path):
    source=tmp_path/"pbp_2024.csv.gz";source.write_bytes(b"frozen")
    frozen=tmp_path/"frozen.json"
    frozen.write_text(json.dumps({"seasons":{"2024":{"pbp":{"status":"PUBLISHED","sha256":hashlib.sha256(b'frozen').hexdigest()}},"2026":{"participation":{"status":"NOT_PUBLISHED"}}}}))
    S.verify(tmp_path,frozen)
    (tmp_path/"participation_2026.csv").write_text("new")
    with pytest.raises(ValueError,match="Unaudited"):S.verify(tmp_path,frozen)
    (tmp_path/"participation_2026.csv").unlink();source.write_bytes(b"revised")
    with pytest.raises(ValueError,match="Frozen source changed"):S.verify(tmp_path,frozen)


def test_frozen_phase1h_and_protected_files_untouched():
    # Audit the published Phase1I change, not later independent research phases.
    frozen_head="411d6f8ca12f24e1f192ab2d7768589cf77d1bc0"
    changed=subprocess.check_output(["git","diff","--name-only",I.P["base_head"],frozen_head],cwd=I.ROOT,text=True).splitlines()
    allowed={"nfl_v2_phase1i_sources.py","nfl_v2_phase1i_target_depth.py",
             "tests/test_nfl_v2_phase1i_target_depth.py","tests/test_nfl_v2_phase1h_routed_receiving.py",
             ".github/workflows/nfl_v2_phase1i_target_depth.yml",
             "nfl_models/nfl_player_outcome_v2/research_registry.json"}
    assert all(f in allowed or f.startswith("nfl_models/nfl_player_outcome_v2/phase1i_") for f in changed)
    protected=subprocess.check_output(["git","ls-tree","-r","--name-only",I.P["base_head"]],cwd=I.ROOT,text=True).splitlines()
    protected=[p for p in protected if p not in allowed and not p.startswith("nfl_models/nfl_player_outcome_v2/phase1i_")]
    for p in protected:
        old=subprocess.check_output(["git","show",f"{I.P['base_head']}:{p}"],cwd=I.ROOT)
        frozen=subprocess.check_output(["git","show",f"{frozen_head}:{p}"],cwd=I.ROOT)
        assert hashlib.sha256(old).digest()==hashlib.sha256(frozen).digest(),p
    assert not I.P["sportsbook_inputs"] and not I.P["monte_carlo"] and I.P["no_rescue"]


def test_frozen_results_snapshot_hashes_no_rescue_or_prior_registry_change():
    path=I.ART/"phase1i_target_depth_results.json"
    if not path.exists():pytest.skip("No real-data results before code-only commit")
    result=json.loads(path.read_text())
    lock=json.loads((I.ART/"phase1i_development_lock.json").read_text())
    snapshot=json.loads((I.ART/"phase1i_target_depth_snapshot.json").read_text())
    for name,sha in snapshot["artifact_sha256"].items():
        assert hashlib.sha256((I.ART/name).read_bytes()).hexdigest()==sha
    assert lock["protocol_sha256"]==hashlib.sha256(I.PROTOCOL.read_bytes()).hexdigest()
    assert lock["audit_sha256"]==hashlib.sha256((I.ART/"phase1i_source_audit.json").read_bytes()).hexdigest()
    assert result["selected"]==lock["selected"]==snapshot["selected_architecture"]
    for n,c in lock["candidates"].items():
        if c["status"]=="REJECTED":
            assert result["periods"]["validation_2025"]["decisions"][n]["component_status"]=="REJECTED"
    if result["selected"] is None:
        assert result["full_projection_evaluation"]["status"]=="NOT_RUN_EFFICIENCY_GATE_FAILED"
    previous=json.loads(subprocess.check_output(["git","show",f"{I.P['base_head']}:nfl_models/nfl_player_outcome_v2/research_registry.json"],cwd=I.ROOT))
    current=json.loads((I.ART/"research_registry.json").read_text())
    for k in previous:
        if k not in {"status","next_milestone"}:assert current[k]==previous[k]


def test_frozen_receipts_reconstruct_chain_fixed_yac_and_legal_history():
    path=I.ART/"phase1i_receipts.jsonl.gz"
    if not path.exists():pytest.skip("No evaluated real rows before code-only commit")
    seen={};count=0
    with gzip.open(path,"rt") as f:
        for line in f:
            r=json.loads(line);count+=1
            assert r["season"] in {2025,2026}
            assert r["season"]!=2026 or r["week"]<=4
            assert sum(r["depth_probabilities"])==pytest.approx(1.,abs=1e-12)
            air=sum(p*a*c for p,a,c in zip(r["depth_probabilities"],r["expected_air_within_bucket"],r["completion_probabilities"]))
            assert r["completed_air_per_target"]==pytest.approx(air,abs=1e-12)
            assert r["yards_per_target"]==r["completed_air_per_target"]+r["incumbent_yac_per_target"]
            k=(r["season"],r["week"],r["receiver"])
            assert seen.setdefault(k,r["incumbent_yac_per_target"])==r["incumbent_yac_per_target"]
            rk=r["qb_assignment"]["roster_history_key"]
            assert rk is None or tuple(rk)<(r["season"],r["week"])
            for s in r["entity_history"].values():
                assert s["last_key"] is None or tuple(s["last_key"])<(r["season"],r["week"])
                assert s["last_date"] is None or s["last_date"]<r["target_game_date"]
    assert count>0

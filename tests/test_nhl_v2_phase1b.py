"""Adversarial source, identity, component and prospective-integrity tests."""
import copy
import gzip
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path

import numpy as np
import pytest

from nhl_v2_phase1b_snapshots import snapshot,SnapshotStore,resolve_identity,certification,horizon_changes
from nhl_v2_phase1b_opportunity import History,fit_priors,component_forecast,mixture_pmf,distribution_receipt,attempt_forecast,role_support,reconcile_expected_attempt_budget
from nhl_v2_phase1b_sources import parse_attempt_labels
from nhl_v2_phase1b_comparators import accept_independent_forecast
from nhl_v2_phase1b_forward import append_forecast
from nhl_v2_phase1a_sog_forward import cutoff_of,sha_file,Ledger

GAME={'game_id':2023020003,'game_start_utc':'2023-10-05T23:00:00Z','home_team_id':1,'away_team_id':2}
CUTOFF=cutoff_of(GAME['game_start_utc'],'T90').isoformat()
OBS={'player_id':10,'team_id':1,'observed_state':'ROSTER_OBSERVED'}


def snap(**kwargs):
    args=dict(raw=b'{}',game=GAME,horizon='T90',cutoff=CUTOFF,retrieved_at='2023-10-05T21:20:00Z',
              source_kind='game_roster',rights_basis='TEST_ONLY_AUTHORIZED_SYNTHETIC',observations=[OBS])
    args.update(kwargs)
    return snapshot(**args)


def data():
    games={}; rows=[]
    for i in range(1,4):
        gid=2022020000+i
        games[gid]={'game_id':gid,'game_start_utc':f'2022-10-0{i}T20:00:00Z','completed_at':f'2022-10-0{i}T23:00:00Z','stats_retrieved_at':f'2022-10-0{i}T23:05:00Z','home_team_id':1,'away_team_id':2}
        for pid,team,toi in [(10,1,1000),(20,2,1500)]:
            rows.append(dict(game_id=gid,team_id=team,player_id=pid,position='C',season_start_year=2022,
                             toi_sec=toi,ev_toi_sec=toi-200,pp_toi_sec=120,sh_toi_sec=80,sog=i))
    games[GAME['game_id']]=GAME
    return games,rows


@pytest.mark.parametrize('field,value',[('retrieved_at','2023-10-05T21:31:00Z'),('published_at','2023-10-05T21:31:00Z'),('retrieved_at','2023-10-05T20:00:00Z')])
def test_late_or_stale_observation_never_eligible(field,value):
    assert not snap(**{field:value})['timing_eligible']


def test_snapshot_hash_append_only_revision_and_wrong_horizon(tmp_path):
    store=SnapshotStore(tmp_path)
    r=snap(); assert store.append(r,b'{}'); before=store.ledger.path.read_bytes()
    assert not store.append(r,b'{}'); assert store.ledger.path.read_bytes()==before
    later=snap(retrieved_at='2023-10-05T21:31:00Z',observations=[dict(OBS,observed_state='SCRATCH_OBSERVED')])
    store.append(later,b'{}')
    assert store.state_at(GAME,'T90',CUTOFF)[0]['record_sha256']==r['record_sha256']
    assert store.state_at(GAME,'T30',cutoff_of(GAME['game_start_utc'],'T30').isoformat())==[]
    corrupt=dict(r,observations=[])
    with pytest.raises(ValueError,match='hash'):store.append(corrupt,b'{}')
    with pytest.raises(ValueError,match='hash'):store.append(r,b'{"wrong":true}')


def test_roster_not_lineup_and_timezone_required():
    with pytest.raises(ValueError,match='not confirmed'):snap(observations=[dict(OBS,observed_state='CONFIRMED_DRESSED',explicit_confirmation_evidence='unproven')])
    with pytest.raises(ValueError,match='timezone'):snap(retrieved_at='2023-10-05T21:20:00')
    with pytest.raises(ValueError,match='rights'):snap(rights_basis=None)
    with pytest.raises(ValueError,match='mismatch'):snap(raw=b'{"id":44}')


def test_cross_team_conflicts_quarantined_before_outcomes():
    candidates=[{'player_id':10,'team_id':1},{'player_id':10,'team_id':2},{'player_id':20,'team_id':1}]
    ok,bad=resolve_identity(candidates,[])
    assert [c['player_id'] for c in ok]==[20] and len(bad)==2
    ok,bad=resolve_identity(candidates,[dict(OBS,membership_verified=True)])
    assert {(c['player_id'],c['team_id']) for c in ok}=={(10,1),(20,1)}
    assert len(bad)==1
    ok,bad=resolve_identity(candidates,[dict(OBS,membership_verified=True),dict(OBS,team_id=2,membership_verified=True)])
    assert [c['player_id'] for c in ok]==[20] and len(bad)==2


def test_each_horizon_requires_40_games_3_dates_and_995_recall():
    records=[]
    for i in range(40):
        records.append({'game_id':i,'horizon':'T90','date':f'2026-10-0{1+i%3}', 'capture_eligible':True,
                        'both_teams_complete':True,'final_truth_complete':True,
                        'dressed_ids':[[1,p] for p in range(18)]+[[2,p+18] for p in range(18)],
                        'observed_ids':[[1,p] for p in range(18)]+[[2,p+18] for p in range(18)]})
    c=certification(records); assert c['T90']['qualified_candidate_source']; assert not c['T90']['confirmed_lineup']
    assert not c['T24H']['qualified_candidate_source'] and not c['T30']['qualified_candidate_source']
    assert not certification(records[:39])['T90']['qualified_candidate_source']
    bad=copy.deepcopy(records)
    for r in bad:r['observed_ids']=r['observed_ids'][:-1]
    assert not certification(bad)['T90']['qualified_candidate_source']
    bad=copy.deepcopy(records)
    for r in bad:r['date']='2026-10-01'
    assert not certification(bad)['T90']['qualified_candidate_source']
    with pytest.raises(ValueError,match='duplicate'):certification(records+[records[0]])


def test_horizon_changes_do_not_mix_games():
    a=snap(); b=dict(a,horizon='T30',observations=[dict(OBS,observed_state='SCRATCH_OBSERVED')])
    assert horizon_changes([a,b])[0]['from']=='T90'
    assert len(horizon_changes([a,b])[0]['added'])==1


def test_target_outcomes_do_not_change_forecast():
    games,rows=data();hist=History(games,rows);priors=fit_priors(rows)
    c=hist.candidates(GAME,CUTOFF)[0][0]
    before=component_forecast(hist,GAME,c,CUTOFF,priors)
    rows2=rows+[dict(rows[0],game_id=GAME['game_id'],sog=99,toi_sec=5000,ev_toi_sec=4800)]
    assert component_forecast(History(games,rows2),GAME,c,CUTOFF,priors)==before
    assert component_forecast(hist,GAME,c,CUTOFF,priors)==before
    assert before['total_toi_conditional']==sum(before[k] for k in ['ev_toi','pp_toi','pk_toi'])


def test_skill_crosses_teams_role_never_does():
    games,rows=data(); rows[-2]['team_id']=2
    hist=History(games,rows)
    assert len(hist.prior_player(10,CUTOFF))==3
    assert len(hist.prior_player(10,CUTOFF,1))==2
    assert len(hist.prior_player(10,CUTOFF,2))==1


def test_unknown_completion_is_blocked_forward_and_proxy_is_explicit():
    games,rows=data()
    for g in games.values():g.pop('completed_at',None)
    assert History(games,rows).candidates(GAME,CUTOFF)==([],[])
    h=History(games,rows,allow_completion_proxy=True);assert h.missing_final_timestamp==4
    c=h.candidates(GAME,CUTOFF)[0][0]
    r=component_forecast(h,GAME,c,CUTOFF,fit_priors(rows))
    assert 'EXPOSED_HISTORICAL_COMPLETION_PROXY_NOT_CERTIFIED' in r['uncertainty']


def test_history_completed_exactly_at_cutoff_is_not_usable():
    games,rows=data();games[rows[0]['game_id']]['completed_at']=CUTOFF
    h=History(games,rows)
    assert rows[0]['game_id'] not in h.prior_team(1,CUTOFF)


def test_early_season_transparent_baseline_does_not_require_5_current_games():
    games,rows=data();h=History(games,rows);c=h.candidates(GAME,CUTOFF)[0][0]
    r=component_forecast(h,GAME,c,CUTOFF,fit_priors(rows))
    assert r['transparent_analyst_mean']>0 and r['simple_mean']>0
    assert all(str(g).startswith('2022') for g in r['skill_games'])
    assert 'NO_INDEPENDENT_PROFESSIONAL_COMPARATOR' in r['uncertainty']


def test_unsupported_attempts_not_manufactured_from_SOG():
    games,rows=data();h=History(games,rows);priors=fit_priors(rows)
    c=h.candidates(GAME,CUTOFF)[0][0];r=component_forecast(h,GAME,c,CUTOFF,priors)
    assert attempt_forecast(r,h.prior_player(10,CUTOFF),priors['position']['F'])['status']=='BLOCKED_ATTEMPT_DATA'


@pytest.mark.parametrize('p',[0.,.3,1.])
def test_full_count_PMF_thinning_survival_intervals_deterministic(p):
    pm=mixture_pmf(p,[1.,3.],[.4,.6],.2)
    assert abs(sum(pm)-1)<1e-12 and (pm>=0).all()
    assert np.array_equal(pm,mixture_pmf(p,[1.,3.],[.4,.6],.2))
    r=distribution_receipt(pm)
    assert r['p_ge']['1']==pytest.approx(1-pm[0])
    assert r['p_ge']['2']==pytest.approx(1-pm[0]-pm[1])
    assert r['expected_sog']==pytest.approx(p*2.2)
    assert all(x>=0 and isinstance(x,int) for x in r['intervals']['0.9'])


def test_attempt_labels_require_real_shooters_and_official_reconciliation():
    official={10:{'team_id':1,'sog':2}}
    def play(i,k):return {'eventId':i,'typeDescKey':k,'situationCode':'1551','details':{'shootingPlayerId':10},'periodDescriptor':{'periodType':'REG'}}
    pbp={'gameState':'OFF','homeTeam':{'id':1},'plays':[play(1,'goal'),play(2,'shot-on-goal'),play(3,'blocked-shot'),play(4,'missed-shot')]}
    r=parse_attempt_labels(pbp,official)[10];assert r['shot_attempts']==4 and r['counted_sog']==2 and r['ev_attempts']==4
    pp=copy.deepcopy(pbp);pp['plays'][0]['situationCode']='1451'
    assert parse_attempt_labels(pp,official)[10]['pp_attempts']==1
    so=play(5,'goal');so['periodDescriptor']['periodType']='SO';pbp['plays'].append(so)
    assert parse_attempt_labels(pbp,official)[10]['shot_attempts']==4
    pbp['plays'][0]['details']['shootingPlayerId']=999
    with pytest.raises(ValueError,match='unmapped'):parse_attempt_labels(pbp,official)
    with pytest.raises(ValueError,match='disagreement'):parse_attempt_labels({'gameState':'OFF','homeTeam':{'id':1},'plays':[]},official)


def test_expected_team_attempt_budget_coherent_not_claiming_joint_draws():
    x=reconcile_expected_attempt_budget([1,3,6],40)
    assert sum(x)==pytest.approx(40) and x.tolist()==[4,12,24]
    with pytest.raises(ValueError):reconcile_expected_attempt_budget([-1],20)


def test_independent_human_intake_prospective_no_retroactive_rewrite(tmp_path):
    d={'game_id':GAME['game_id'],'team_id':1,'player_id':10,'scheduled_start':GAME['game_start_utc'],'horizon':'T90','cutoff':CUTOFF}
    rec=dict(d,author='Independent researcher',method_version='1',publication_timestamp='2023-10-05T21:20:00Z',retrieval_timestamp='2023-10-05T21:20:00Z',
             evidence_hashes=['a'*64],expected_sog=1.,pmf=[0,1],rights_basis='test',independent_of_engine=True,hockey_evidence_only=True)
    accept_independent_forecast(rec,d,'2023-10-05T21:20:00Z',tmp_path/'analysts')
    with pytest.raises(ValueError,match='already frozen'):accept_independent_forecast(rec,d,'2023-10-05T21:20:00Z',tmp_path/'analysts')
    with pytest.raises(ValueError,match='late'):accept_independent_forecast(rec,d,'2023-10-05T21:31:00Z',tmp_path/'late')
    with pytest.raises(ValueError,match='hockey'):accept_independent_forecast(dict(rec,hockey_evidence_only=False),d,'2023-10-05T21:20:00Z',tmp_path/'bad')


def test_blocked_partial_engine_cannot_emit_clean_forecast(tmp_path):
    with pytest.raises(ValueError,match='blocked'):append_forecast({'files':{},'status':'PARTIAL_BLOCKED'},tmp_path,{},'2023-10-05T21:20:00Z',[],tmp_path/'forecasts')
    assert not (tmp_path/'forecasts').exists()


def test_pregame_receipt_whitelist_excludes_market_and_target_data():
    games,rows=data();h=History(games,rows);c=h.candidates(GAME,CUTOFF)[0][0]
    a=component_forecast(h,GAME,c,CUTOFF,fit_priors(rows))
    polluted=copy.deepcopy(rows)
    for r in polluted:r.update(odds=-110,spread=5,book_line=4,target_sog=88)
    b=component_forecast(History(games,polluted),GAME,c,CUTOFF,fit_priors(polluted))
    assert a==b


def test_postgame_payload_and_late_effective_revision_not_pregame():
    assert not snap(raw=b'{"gameState":"FINAL"}')['timing_eligible']
    assert not snap(observations=[dict(OBS,effective_at='2023-10-05T21:31:00Z')])['timing_eligible']


def test_observed_attempt_rate_component_uses_actual_attempts_not_SOG_shortcut():
    from nhl_v2_phase1b_opportunity import total_attempt_component
    games,rows=data()
    for r in rows:r['shot_attempts']=r['sog']+4
    h=History(games,rows);p=fit_priors(rows);c=h.candidates(GAME,CUTOFF)[0][0]
    a=total_attempt_component(h,GAME,c,CUTOFF,p)
    doubled=copy.deepcopy(rows)
    for r in doubled:r['shot_attempts']*=2
    b=total_attempt_component(History(games,doubled),GAME,c,CUTOFF,fit_priors(doubled))
    assert b['A1']==pytest.approx(a['A1']*2)
    assert b['R1']<a['R1']
    assert a['strength_attempt_rates'] is None


def test_permissioned_full_strength_chain_thinning_is_implemented_but_unvalidated():
    games,rows=data()
    for r in rows:r.update(ev_attempts=4,pp_attempts=2,pk_attempts=1)
    h=History(games,rows);p=fit_priors(rows);c=h.candidates(GAME,CUTOFF)[0][0]
    r=component_forecast(h,GAME,c,CUTOFF,p)
    a=attempt_forecast(r,h.prior_player(10,CUTOFF),p['position']['F'])
    assert a['expected_attempts']>0 and sum(a['pmf'])==pytest.approx(1)
    assert a['status']=='EXPERIMENTAL_REQUIRES_COMPONENT_VALIDATION'


def test_development_chronology_fit_priors_ignore_later_seasons():
    games,rows=data();p=fit_priors(rows)
    changed=rows+[dict(rows[0],season_start_year=2025,sog=999)]
    assert fit_priors(changed)==p


def test_authorized_capture_pipeline_does_not_confirm_roster_or_backfill(tmp_path):
    from nhl_v2_phase1b_intelligence import capture_game,compare_final_dressed
    import hashlib
    raw=json.dumps({'id':GAME['game_id'],'gameState':'FUT','rosterSpots':[{'playerId':10,'teamId':1,'positionCode':'C'}]}).encode()
    def transport(url):return raw,{'url':url,'http_status':200,'retrieval_completed_utc':'2023-10-05T21:20:00Z','sha256':hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError,match='authorized'):capture_game(GAME,'T90',transport,None,tmp_path)
    captured=capture_game(GAME,'T90',transport,'TEST_ONLY',tmp_path)
    assert len(captured)==3 and all(r['timing_eligible'] for r in captured)
    assert all(o['observed_state']=='ROSTER_OBSERVED' for r in captured for o in r['observations'])
    truth={'state':'OFF','both_teams_complete':True,'game':GAME,'retrieved_at':'2023-10-06T02:00:00Z','source_hash':'a'*64,'dressed_ids':[[1,10],[2,20]]}
    evidence=compare_final_dressed(captured,truth)
    assert next(r for r in evidence if r['horizon']=='T90')['capture_eligible']
    assert certification(evidence)['T90']['dressed_recall']==.5
    assert certification(evidence)['T24H']['dressed_recall'] is None


def test_capture_store_keeps_all_endpoint_revisions(tmp_path):
    from nhl_v2_phase1b_intelligence import capture_game
    import hashlib
    def transport(url):return b'{}',{'url':url,'http_status':200,'retrieval_completed_utc':'2023-10-05T21:20:00Z','sha256':hashlib.sha256(b'{}').hexdigest()}
    capture_game(GAME,'T90',transport,'TEST_ONLY',tmp_path)
    assert len(SnapshotStore(tmp_path).state_at(GAME,'T90',CUTOFF))==3


def test_ready_synthetic_forward_path_rejects_duplicate_team_hash_and_late(tmp_path):
    lock={'files':{},'status':'READY_FOR_FORWARD','engine_version':'TEST_ONLY','eligible_from_cutoff_utc':'2023-10-01T00:00:00Z',
          'availability_certification':{'T90':{'qualified_candidate_source':True}},'components_validated':True}
    r={'game_id':GAME['game_id'],'player_id':10,'team_id':1,'horizon':'T90','scheduled_start':GAME['game_start_utc'],'cutoff':CUTOFF}
    kwargs=dict(lock=lock,repo=tmp_path,receipt=r,now='2023-10-05T21:20:00Z',sources=[snap()],ledger_path=tmp_path/'ledger',lock_commit_pushed=True,ci_green=True)
    append_forecast(**kwargs)
    before=(tmp_path/'ledger').read_bytes()
    with pytest.raises(ValueError,match='duplicate'):append_forecast(**kwargs)
    with pytest.raises(ValueError,match='duplicate'):append_forecast(**dict(kwargs,receipt=dict(r,team_id=2)))
    assert (tmp_path/'ledger').read_bytes()==before
    with pytest.raises(ValueError,match='window'):append_forecast(**dict(kwargs,now='2023-10-05T21:31:00Z'))
    tamper=dict(snap(),observations=[])
    with pytest.raises(ValueError,match='hash'):append_forecast(**dict(kwargs,sources=[tamper]))


def test_final_qualification_uses_latest_snapshot_not_union_of_revisions(tmp_path):
    from nhl_v2_phase1b_intelligence import capture_game,compare_final_dressed
    import hashlib
    def transport(pid,at):
        def fetch(url):
            raw=json.dumps({'rosterSpots':[{'playerId':pid,'teamId':1,'positionCode':'C'}]}).encode()
            return raw,{'url':url,'http_status':200,'sha256':hashlib.sha256(raw).hexdigest(),'retrieval_completed_utc':at}
        return fetch
    old=capture_game(GAME,'T90',transport(10,'2023-10-05T21:20:00Z'),'TEST_ONLY',tmp_path)
    new=capture_game(GAME,'T90',transport(30,'2023-10-05T21:25:00Z'),'TEST_ONLY',tmp_path)
    truth={'state':'OFF','both_teams_complete':True,'game':GAME,'retrieved_at':'2023-10-06T02:00:00Z','source_hash':'a'*64,'dressed_ids':[[1,10]]}
    record=next(r for r in compare_final_dressed(old+new,truth) if r['horizon']=='T90')
    assert record['observed_ids']==[(1,30)]
    assert certification([record])['T90']['dressed_recall']==0


def test_phase1b_engine_scope_preserves_parent_frozen_files():
    import subprocess
    # Stacked PR68 owns all integrity repairs. New scientific work must remain
    # confined to Phase1B files and additive registry/source references.
    changed=subprocess.check_output(['git','diff','--name-only','f75b164','--']).decode().splitlines()
    allowed={'nhl_models/nhl_player_outcome_v2/research_registry.json','nhl_models/nhl_player_outcome_v2/source_inventory.json',
             '.github/workflows/nhl_v2_phase1b_research.yml','tests/test_nhl_v2_phase1b.py'}
    assert all(p in allowed or p.startswith('nhl_v2_phase1b_') or p.startswith('nhl_models/nhl_player_outcome_v2/phase1b_') for p in changed)


def test_game_level_calendar_block_bootstrap_deterministic_and_fail_closed():
    from nhl_v2_phase1b_validation import moving_block
    a=moving_block([.1,.2,-.1,.3],[1,1,2,3],reps=100)
    assert a==moving_block([.1,.2,-.1,.3],[1,1,2,3],reps=100)
    assert a['game_mean_delta']==pytest.approx(.125)
    assert moving_block([1,2],[1,4])['status']=='INSUFFICIENT_CONTIGUOUS_CALENDAR_WEEKS'


def test_strict_history_rejects_late_statistics_revision_and_missing_vintage():
    games,rows=data()
    late=copy.deepcopy(games)
    for g in late.values():
        if g.get('completed_at'):g['stats_retrieved_at']='2023-10-06T04:00:00Z'
    assert History(late,rows).candidates(GAME,CUTOFF)==([],[])
    for g in late.values():g.pop('stats_retrieved_at',None)
    assert History(late,rows).candidates(GAME,CUTOFF)==([],[])
    changed=copy.deepcopy(rows)
    for r in changed:r['retrieved_at']='2023-10-06T04:00:00Z'
    assert History(games,changed).candidates(GAME,CUTOFF)==([],[])


def test_latest_upstream_sync_preserves_original_prefix_and_every_source_blob():
    import subprocess
    root='nhl_models/nhl_player_outcome_v2/phase1a_sog_forward/'
    ledger=root+'ledger.jsonl'
    original=subprocess.check_output(['git','show','3ee6054:'+ledger])
    upstream=subprocess.check_output(['git','show','f75b164:'+ledger])
    assert upstream.startswith(original)
    assert Path(ledger).read_bytes()==upstream
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','f75b164',root]).decode().splitlines()
    for name in names:
        assert Path(name).read_bytes()==subprocess.check_output(['git','show','f75b164:'+name]), name


def test_cross_platform_float_comparison_never_relaxes_identity_decisions_or_counts():
    from nhl_v2_phase1b_reproduction import compare,semantic
    compare({'metric':1.0,'n':10,'status':'REJECTED'},{'metric':1.+1e-12,'n':10,'status':'REJECTED'})
    for changed in ({'metric':1.+1e-6,'n':10,'status':'REJECTED'},
                    {'metric':1.,'n':11,'status':'REJECTED'},
                    {'metric':1.,'n':10,'status':'SURVIVES'},
                    {'metric':1.,'n':10.,'status':'REJECTED'}):
        with pytest.raises(ValueError):compare({'metric':1.,'n':10,'status':'REJECTED'},changed)
    assert semantic(10)!=semantic(10.)
    with pytest.raises(ValueError):semantic(float('nan'))


def test_full_typed_receipt_fingerprint_checks_raw_hash_and_real_label_changes(tmp_path):
    import hashlib
    from nhl_v2_phase1b_reproduction import receipt_fingerprint
    from nhl_v2_phase1a_sog_forward import canon
    def write(row):
        row=dict(row,receipt_sha256=hashlib.sha256(canon(row).encode()).hexdigest())
        path=tmp_path/'receipt.gz';path.write_bytes(gzip.compress((canon(row)+'\n').encode(),mtime=0))
        return path
    base={'player_id':10,'projection':1.1234567891234,'actual_sog':2,'meaningful':True,'source_hash':'a'*64}
    a=receipt_fingerprint(write(base))
    assert receipt_fingerprint(write(dict(base,projection=1.1234567891235)))['semantic_sha256']==a['semantic_sha256']
    for changed in (dict(base,actual_sog=3),dict(base,player_id=20),dict(base,meaningful=False),dict(base,projection=1.1234568),dict(base,source_hash='b'*64)):
        assert receipt_fingerprint(write(changed))['semantic_sha256']!=a['semantic_sha256']
    path=write(base);row=json.loads(gzip.decompress(path.read_bytes()));row['actual_sog']=99
    path.write_bytes(gzip.compress((canon(row)+'\n').encode(),mtime=0))
    with pytest.raises(ValueError,match='self hash'):receipt_fingerprint(path)


def test_legacy_plot_exchange_exception_is_narrow_and_never_used_for_prediction():
    from nhl_v2_phase1b_reproduction import compare
    p='phase1b_results.json/development/FULL/SOG/simple_participation_mean/thresholds/P(SOG>=2)/reliability/7/observed'
    compare(.5,.5+2/5968,p,{p:2/5968+1e-12})
    with pytest.raises(ValueError):compare(.5,.5+3/5968,p,{p:2/5968+1e-12})
    with pytest.raises(ValueError):compare(.5,.5+2/5968,p.replace('/7/','/6/'),{p:2/5968+1e-12})


def test_portable_reproduction_compares_all_raw_receipt_fields_and_archive_provenance(tmp_path):
    import hashlib
    from nhl_v2_phase1b_reproduction import verify,receipt_fingerprint
    from nhl_v2_phase1a_sog_forward import canon
    base=tmp_path/'base';out=tmp_path/'output'
    (base/'phase1b_frozen_receipts').mkdir(parents=True);out.mkdir()
    name='receipts.gz';row={'player_id':10,'team_id':1,'projection':1.2,'actual_sog':2,'eligible':True,'source_hash':'a'*64}
    def archive(root,value,level):
        r=dict(value,receipt_sha256=hashlib.sha256(canon(value).encode()).hexdigest())
        (root/name).write_bytes(gzip.compress((canon(r)+'\n').encode(),mtime=0,compresslevel=level))
        return receipt_fingerprint(root/name)
    expected=archive(base/'phase1b_frozen_receipts',row,9)
    def write(root,record):
        fp=expected if root==base else archive(out,record,1)
        def put(name,value): (root/name).write_text(json.dumps(value))
        put('phase1b_development_lock.json',{'development_receipts':fp['archive_sha256'],'status':'BLOCKED','code_hash':'b'*64})
        put('phase1b_results.json',{'development':{'receipt_sha256':fp['content_sha256'],'receipt_file_sha256':fp['archive_sha256'],'n':1},'diagnostics':{}})
        put('phase1b_retrospective_uncertainty.json',{'status':'EXPOSED_ONLY'})
    write(base,row)
    (base/'phase1b_reproduction_contract.json').write_text(json.dumps({'receipts':{'development':dict(file=name,**expected)},'legacy_plot_absolute_limits':{}}))
    write(out,dict(row,projection=1.2+1e-12))
    assert verify(base,out)['status']=='VERIFIED_EVERY_RECEIPT_FIELD_AND_FROZEN_DECISIONS'
    for changed in (dict(row,projection=1.2+1e-6),dict(row,actual_sog=3),dict(row,eligible=False),dict(row,player_id=20),dict(row,source_hash='c'*64)):
        write(out,changed)
        with pytest.raises(ValueError):verify(base,out)
    write(out,row)
    report=json.loads((out/'phase1b_results.json').read_text());report['development']['receipt_sha256']='x'*64
    (out/'phase1b_results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='provenance'):verify(base,out)

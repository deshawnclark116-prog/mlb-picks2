"""Deterministic mechanic/chronology/cohort/scope tests; no real-season scoring."""
import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1k_sources as S
import nfl_v2_phase1k_rushing_efficiency as K


def carry(i,y,team='A'):
    return {'game_id':f'2023_01_{team}_B','season':2023,'week':1,'date':'2023-09-01','team':team,'position':'RB','player_id':'p','opponent':'B','y':y,'epa':1. if y>3 else -1.,'negative':float(y<0),'zero':float(y==0),'success':float(y>3),'ten':float(y>=10),'twenty':float(y>=20),'short':float(i%5==0),'goal':float(i%7==0),'loc':K.LOCATIONS[i%3],'context':K.CONTEXTS[i%8],'no_huddle':float(i%6==0),'play_id':str(i)}


@pytest.fixture
def toy():
    player=[carry(i,y) for i,y in enumerate([-2,0,2,3,4,8,11,22]*4)]
    league=player+[carry(i+100,y,'B') for i,y in enumerate([0,1,2,4,6,7,13,30]*6)]
    c=K.components(player,league,player,league)
    out=[]
    for i in range(24):
        comp=deepcopy(c);comp['player_profile']['ypc']+=i/100
        base=4.2
        out.append({'game_id':f'2024_{i//2+1:02d}_A_B','date':f'2024-09-{i//2+1:02d}','season':2024,'week':i//2+1,'player_id':f'p{i%2}','position':'RB','player':'Runner','team':'A','opponent':'B','actual_carries':float(i%12+1),'actual_ypc':4.+i/30,'actual_rushing_yards':(i%12+1)*(4.+i/30),'components':comp,'vectors':K.vectors(comp,base),'comparators':{'phase1f_incumbent':base},'actual_rates':{k:c['player_profile'][k] for k in K.RATES},'recent_rate_baseline':{k:c['player_profile'][k]+.02 for k in K.RATES},'label_reconciliation':{'pbp_carries':i%12+1,'official_carries':i%12+1,'pbp_yards':0.,'official_yards':0.},'oracle_metadata':{'label':'POSTGAME_ORACLE_DIAGNOSTIC_ONLY','location_mixture':comp['location_mixture'],'context_mixture':comp['context_mixture'],'tail_counts':[i%12+1,0,0]},'prior_current_team_mean_carries':12.})
    return out


def test_protocol_audit_precedes_models():
    p=json.loads((K.ART/'phase1k_rushing_protocol.json').read_text())
    assert p['chronology']['fit']=='2024 W1-8'
    assert p['chronology']['selection']=='2024 W9-18'
    assert 'NOT globally fresh' in p['chronology']['confirmation']
    assert 'freeze phase immediately' in p['stop_rule']
    assert p['freeze_receiving']=='FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION'
    a=json.loads((K.ART/'phase1k_pbp_source_audit.json').read_text())
    assert a['no_fit_or_performance'] is True
    for y in range(2023,2027):
        assert a['seasons'][str(y)]['fields']['run_location']['missing_rate']<.001
        assert a['seasons'][str(y)]['fields']['run_gap']['status']=='BLOCKED_DATA'
        assert a['seasons'][str(y)]['fields']['designed_qb_run']['status']=='BLOCKED_DATA'
    assert a['seasons']['2026']['weeks']==[1,2,3,4]


@pytest.mark.parametrize('change',[{'week':2},{'season':2025},{'date':'2024-09-07'}])
def test_cutoff_rejects_week_or_incomplete_source(change):
    source={'season':2024,'week':1,'date':'2024-09-01'}
    target={'season':2024,'week':2,'date':'2024-09-08'}
    assert K.legal(source,target)
    assert not K.legal({**source,**change},target)


def test_conservative_cutoff_equality():
    source={'season':2024,'week':1,'date':'2024-09-05'}
    target={'season':2024,'week':2,'date':'2024-09-08'}
    assert K.legal(source,target)
    assert not K.legal({**source,'date':'2024-09-06'},target)


@pytest.mark.parametrize('family',K.FAMILIES)
def test_fit_predict_deterministic_and_finite(toy,family):
    m=K.fit(toy,family,1.)
    assert m==K.fit(toy,family,1.)
    pred=[K.predict(r,m) for r in toy]
    assert all(np.isfinite(p) and 0<=p<=12 for p in pred)
    assert pred==[K.predict(r,m) for r in toy]


def test_target_labels_not_predictive_inputs(toy):
    model=K.fit(toy,'F_location',1.)
    r=deepcopy(toy[0]);original=K.predict(r,model)
    r.update(actual_carries=99.,actual_rushing_yards=999.,actual_ypc=77.,actual_rates={},oracle_metadata={'location_mixture':{'left':1.}})
    assert K.predict(r,model)==original
    assert K.vectors(r['components'],r['comparators']['phase1f_incumbent'])==r['vectors']


def test_context_labels_not_ol_proxies():
    assert K.context({'yardline_100':'2','ydstogo':'1','down':'3','shotgun':'0'})=='goal_under'
    assert K.context({'yardline_100':'50','ydstogo':'2','down':'3','shotgun':'1'})=='short_shotgun'
    assert K.context({'yardline_100':'50','ydstogo':'10','down':'1','shotgun':'1'})=='early_shotgun'


def test_probability_decompositions_sum_one_and_exact_parts(toy):
    c=toy[0]['components']
    for key in ('tail','states'):
        assert sum(c[key]['probabilities'])==pytest.approx(1.)
        assert sum(c[key]['contributions'])==pytest.approx(c[key]['ypc'])
    for key in ('location_mixture','context_mixture'):
        assert sum(c[key].values())==pytest.approx(1.)


def test_current_team_context_does_not_cross_teams():
    prior=[carry(i,4,'OLD') for i in range(10)]
    team=[carry(i,4,'NEW') for i in range(10)]
    league=prior+team
    c=K.components(prior,team,team,league)
    changed=[{**r,'context':'goal_shotgun'} for r in prior]
    # Skill can follow stable identity; role mixture for NEW never follows OLD.
    x=K.components(changed,team,team,league)
    assert c['context_mixture']==x['context_mixture']
    assert c['player_profile']['n']==len(prior)


def test_population_and_zero_carry_not_candidate_filtered(toy):
    toy.append({**deepcopy(toy[0]),'actual_carries':0.,'actual_ypc':None,'actual_rushing_yards':0.})
    m=K.fit(toy,'A_player_profile',1.);p=[K.predict(r,m) for r in toy]
    a=K.metrics(toy,p,'A_player_profile')
    assert a['candidate_rows']==25 and a['positive_carry_rows']==24 and a['zero_carry_rows']==1
    for name in K.FAMILIES:
        assert K.metrics(toy,[K.predict(r,K.fit(toy,name,1.)) for r in toy],name)['candidate_rows']==25


def test_actual_carries_only_multiply_pregame_efficiency(toy):
    r=toy[0];m=K.metrics([r],[4.])
    assert m['oracle_carry_mae']==pytest.approx(abs(4*r['actual_carries']-r['actual_rushing_yards']))
    assert m['ypc_mae']==pytest.approx(abs(4-r['actual_ypc']))


def test_bootstrap_game_level_and_deterministic(toy):
    for r in toy:r['comparators']['phase1f_incumbent']=r['actual_ypc']+1/r['actual_carries']
    p=[r['actual_ypc'] for r in toy]
    a=K.blocked_bootstrap(toy,p,40)
    assert a==K.blocked_bootstrap(toy,p,40)
    assert a['game_count']==12
    assert a['delta_mae']==pytest.approx(-1.)
    assert a['ci95']==pytest.approx([-1.,-1.])


def test_no_microscopic_or_cancellation_promotion(toy):
    b=K.metrics(toy,[4.2]*len(toy));a=deepcopy(b)
    a['oracle_carry_mae']-=.01;a['ypc_mae']+=.1
    result=K.gate(a,b,{'ci95':[-.02,-.001]},'A_player_profile')
    assert not result['passed']
    assert {'ORACLE_PRACTICAL_GATE','YPC_PRACTICAL_GATE','INSUFFICIENT_ROWS'}<=set(result['reasons'])


def test_ledger_exact_deterministic_zero_timestamp(toy,tmp_path):
    model=K.fit(toy,'A_player_profile',1.)
    a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    K.save_receipts(a,K.ledger(toy,{'A_player_profile':model},'synthetic_test_only'))
    K.save_receipts(b,K.ledger(toy,{'A_player_profile':model},'synthetic_test_only'))
    assert a.read_bytes()==b.read_bytes()
    with gzip.open(a,'rt') as f:r=json.loads(next(f))
    assert r['predicted_carries'] is None
    assert r['full_direct_projection'] is None
    assert r['oracle_carry_rushing_yards_projection']==r['actual_carries']*r['final_predicted_ypc']
    assert r['oracle_metadata']['label']=='POSTGAME_ORACLE_DIAGNOSTIC_ONLY'


def test_source_loader_refuses_future_before_outcome(tmp_path):
    p=tmp_path/'future.csv'
    p.write_text('season,week,rushing_yards\n2026,5,9999\n')
    with pytest.raises(ValueError,match='Week 5'):
        list(S.records(p,S.PBP_FIELDS))


def test_source_sha_refuses_revised_bytes(tmp_path):
    meta=S.sources()['2023']['pbp'];(tmp_path/meta['local_name']).write_text('revision')
    with pytest.raises(ValueError,match='Frozen source changed'):
        S.verify(tmp_path,[2023])


def test_failed_development_never_opens_validation(monkeypatch,tmp_path,toy):
    opened=[]
    def assemble(directory,year,minweek,maxweek):
        opened.append(year);assert year==2024;return toy
    monkeypatch.setattr(K,'assemble',assemble)
    lock={'selected':None,'frozen_specifications':{'D_explosive':K.fit(toy,'D_explosive',1.)},'families':{k:{'status':'FROZEN_REJECTED'} for k in K.FAMILIES},'selection_comparators':{},'protocol_sha256':S.sha(K.ART/'phase1k_rushing_protocol.json')}
    r=K.confirm(tmp_path,lock,tmp_path/'receipts.gz')
    assert opened==[2024]
    assert r['2025_model_performance_accessed'] is False
    assert r['2026_model_performance_accessed'] is False
    assert r['periods']['validation_2025']['status']=='NOT_RUN_DEVELOPMENT_GATE_FAILED'


def test_oracles_are_descriptive_only(toy):
    d=K.oracle_diagnostics(toy,[4.2]*len(toy))
    assert d['label']=='POSTGAME_ORACLE_DIAGNOSTIC_ONLY'
    assert d['not_used_for_fit_selection_promotion'] is True
    source=Path(K.__file__).read_text()
    fit_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    fit_source=ast.get_source_segment(source,fit_node)
    assert 'oracle_metadata' not in fit_source and 'actual_carries' in fit_source
    assert 'actual_carries' not in ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='predict'))


def test_protected_history_scope():
    allowed={'nfl_v2_phase1k_rushing_efficiency.py','nfl_v2_phase1k_sources.py','tests/test_nfl_v2_phase1k_rushing_efficiency.py','tests/test_nfl_v2_phase1j_information_audit.py','.github/workflows/nfl_v2_phase1k_rushing_efficiency.yml',
             *{f'nfl_models/nfl_player_outcome_v2/{p}' for p in ('phase1k_pbp_source_audit.json','phase1k_rushing_protocol.json','phase1k_development_lock.json','phase1k_rushing_results.json','phase1k_rushing_receipts.jsonl.gz','phase1k_rushing_findings.md','phase1k_rushing_snapshot.json','research_registry.json')}}
    root=S.ROOT
    paths=subprocess.check_output(['git','diff','--name-only',K.BASE_HEAD],cwd=root,text=True).splitlines()
    assert set(paths)<=allowed
    protected=subprocess.check_output(['git','ls-tree','-r','--name-only',K.BASE_HEAD],cwd=root,text=True).splitlines()
    for path in protected:
        if path in allowed:continue
        expected=subprocess.check_output(['git','show',f'{K.BASE_HEAD}:{path}'],cwd=root)
        assert hashlib.sha256(expected).hexdigest()==S.sha(root/path),path
    old=json.loads(subprocess.check_output(['git','show',f'{K.BASE_HEAD}:nfl_models/nfl_player_outcome_v2/research_registry.json'],cwd=root,text=True))
    current=json.loads((K.ART/'research_registry.json').read_text())
    for key,value in old.items():
        if key not in ('status','next_milestone'):assert current[key]==value


def mini_source(tmp_path):
    import csv
    manifest={str(y):{} for y in (2023,2024)}
    for year in (2023,2024):
        pbp=[];stats=[]
        weeks=(1,2) if year==2023 else (1,)
        for week in weeks:
            day=f'{year}-09-{week*7:02d}';gid=f'{year}_{week:02d}_A_B'
            for team,opp in [('A','B'),('B','A')]:
                names=['p'+team,'q'+team] if year==2023 else ['q'+team] # RB non-participant in target
                for pid in names:
                    stats.append({'game_id':gid,'season':year,'week':week,'season_type':'REG','player_id':pid,'player_display_name':pid,'position':'RB' if pid.startswith('p') else 'QB','team':team,'opponent_team':opp,'carries':5 if pid.startswith('p') else 1,'rushing_yards':20 if pid.startswith('p') else 100})
                for i in range(6):
                    pbp.append({'game_id':gid,'season':year,'week':week,'season_type':'REG','game_date':day,'away_team':'A','home_team':'B','posteam':team,'defteam':opp,'rusher_player_id':'p'+team if i<5 else 'q'+team,'rush_attempt':'1','qb_kneel':'0','qb_scramble':'0' if i<5 else '1','two_point_attempt':'0','play_type':'run','rushing_yards':'4' if i<5 else '100','epa':'.2','run_location':'left','down':'1','ydstogo':'10','yardline_100':'50','shotgun':'1','no_huddle':'0','play_id':str(i+(10 if team=='B' else 0))})
        for kind,rows in [('stats',stats),('pbp',pbp)]:
            path=tmp_path/f'{kind}_{year}.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            manifest[str(year)][kind]={'local_name':path.name,'sha256':S.sha(path),'url':'test-only'}
    return manifest


def test_pregame_candidates_survive_nonparticipation_and_qb_exclusion(tmp_path,monkeypatch):
    m=mini_source(tmp_path);monkeypatch.setattr(S,'sources',lambda:m)
    rows=K.assemble(tmp_path,2024,1,1)
    assert {r['player_id'] for r in rows}=={'pA','pB'}
    assert all(r['actual_carries']==0 for r in rows)
    assert all(r['components']['player_profile']['ypc']==pytest.approx(4.) for r in rows)
    assert all(r['components']['league_profile']['ypc']==pytest.approx(4.) for r in rows)
    assert all(r['actual_ypc'] is None for r in rows)
    assert all('2024' not in gid for r in rows for gid in r['history_receipt']['prior_player_game_ids'])


def test_assembly_target_context_and_outcome_cannot_change_features(tmp_path,monkeypatch):
    m=mini_source(tmp_path);monkeypatch.setattr(S,'sources',lambda:m)
    before=K.assemble(tmp_path,2024,1,1)
    p=tmp_path/'pbp_2024.csv';text=p.read_text().replace(',left,1,10,50,1,0,',',right,4,1,2,0,1,');p.write_text(text)
    m['2024']['pbp']['sha256']=S.sha(p)
    after=K.assemble(tmp_path,2024,1,1)
    assert [r['vectors'] for r in before]==[r['vectors'] for r in after]


def test_develop_never_assembles_2025_and_failed_families_never_combined(monkeypatch,toy):
    rows=deepcopy(toy)*20
    # Same synthetic fixtures duplicated solely for gate-population tests.
    opened=[]
    def assemble(directory,year,a,b):opened.append(year);assert year==2024;return rows
    monkeypatch.setattr(K,'assemble',assemble)
    monkeypatch.setattr(K,'blocked_bootstrap',lambda *a,**k:{'ci95':[-.1,.1]})
    audit=json.loads((K.ART/'phase1k_pbp_source_audit.json').read_text())
    lock=K.develop('unused',audit)
    assert opened==[2024]
    assert lock['selected'] is None
    assert lock['families']['C_player_defense']['status']=='NOT_RUN_FAILED_PARENT_FAMILIES'
    assert 'C_player_defense' not in lock['frozen_specifications']
    assert not lock['validation_performance_accessed']


def test_component_source_failure_blocks_not_imputes(monkeypatch,toy):
    rows=deepcopy(toy)*20
    monkeypatch.setattr(K,'assemble',lambda *a:rows)
    monkeypatch.setattr(K,'blocked_bootstrap',lambda *a,**k:{'ci95':[-.1,.1]})
    audit=json.loads((K.ART/'phase1k_pbp_source_audit.json').read_text())
    audit['seasons']['2024']['fields']['run_location']['status']='BLOCKED_DATA'
    lock=K.develop('unused',audit)
    assert lock['families']['F_location']['status']=='BLOCKED_DATA'
    assert 'F_location' not in lock['frozen_specifications']


def test_frozen_failure_and_no_confirmation_performance():
    lock=json.loads((K.ART/'phase1k_development_lock.json').read_text())
    result=json.loads((K.ART/'phase1k_rushing_results.json').read_text())
    snap=json.loads((K.ART/'phase1k_rushing_snapshot.json').read_text())
    assert lock['selected'] is None and not lock['development_survivors']
    assert all(not v['gate']['passed'] for v in lock['families'].values())
    assert result['verdict']=='REJECTED_EFFICIENCY_REPLACEMENT'
    assert result['2025_model_performance_accessed'] is False
    assert result['2026_model_performance_accessed'] is False
    assert result['week5_plus_accessed'] is False
    assert result['full_projection_evaluation']['status']=='NOT_RUN_INDEPENDENT_EFFICIENCY_GATE'
    assert result['development_lock_sha256']==K.digest(lock)
    assert lock['protocol_sha256']==S.sha(K.ART/'phase1k_rushing_protocol.json')
    assert lock['source_audit_sha256']==K.digest(json.loads((K.ART/'phase1k_pbp_source_audit.json').read_text()))
    for name,value in snap['artifacts_sha256'].items():assert S.sha(K.ART/name)==value
    clarification=snap['run_gap_semantics_correction']
    assert clarification['status']=='AVAILABLE_CONDITIONALLY_NOT_TESTED_THIS_PHASE'
    assert not clarification['used_for_selection_or_fit']
    assert all(v['conditional_missing_rate']==0 for v in clarification['conditional_coverage'].values())


def test_frozen_receipts_identical_population_and_history_cutoff():
    ledger=K.ART/'phase1k_rushing_receipts.jsonl.gz';result=json.loads((K.ART/'phase1k_rushing_results.json').read_text())
    assert S.sha(ledger)==result['receipt_ledger_sha256']
    groups={};n=0
    with gzip.open(ledger,'rt') as f:
        for line in f:
            n+=1;r=json.loads(line)
            assert r['season']==2024 and 9<=r['week']<=18
            assert r['position'] in S.POSITIONS
            assert r['predicted_carries'] is None and r['full_direct_projection'] is None
            keys=groups.setdefault(r['architecture'],set());keys.add((r['game_id'],r['player_id']))
            target=(r['season'],r['week'])
            for gid in r['history_receipt']['prior_player_game_ids']+r['history_receipt']['prior_defense_game_ids']:
                assert tuple(map(int,gid.split('_')[:2]))<target
            assert r['oracle_metadata']['label']=='POSTGAME_ORACLE_DIAGNOSTIC_ONLY'
            assert K.legal({'season':2023,'week':1,'date':r['history_receipt']['max_prior_game_date']},r)
    assert n==result['receipt_rows']==6963
    assert len(groups)==11
    assert all(v==next(iter(groups.values())) for v in groups.values())
    assert len(next(iter(groups.values())))==633

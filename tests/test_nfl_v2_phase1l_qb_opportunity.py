"""Source-block and opportunity-contract checks; synthetic math is not fitting."""
from copy import deepcopy
import gzip
import hashlib
import json
import subprocess

import pytest

import nfl_v2_phase1l_qb_opportunity as L
import nfl_v2_phase1l_qb_sources as S


def game(week, date='2024-09-22', season=2024, **kw):
    return {'season': str(season), 'week': str(week), 'game_date': date,
            'game_id': f'{season}_{week:02d}_A_B', 'home_team': 'B', 'away_team': 'A', **kw}


def tiny():
    prior = game(1, '2024-09-01')
    prior2 = game(2, '2024-09-08')
    target = game(3)
    data = {'games': {g['game_id']: g for g in [prior, prior2, target]},
            'teams': {}, 'qbs': {}, 'official': {}}
    for g in [prior, prior2, target]:
        for team in ['A', 'B']:
            data['teams'][(g['game_id'], team)] = {'plays': 60, 'dropbacks': 35}
            pid = team + 'QB'
            data['qbs'][(g['game_id'], team, pid)] = {'attempts': 30, 'dropbacks': 35}
            data['official'][(g['game_id'], team, pid)] = {
                'attempts': 30, 'passing_yards': 200, 'player_display_name': pid}
    return data


def test_chain_has_four_stages_and_no_yards_fit():
    p = L.chain(64, .625, .96, .875)
    assert p['expected_dropbacks'] == 40
    assert p['final_projected_attempts'] == pytest.approx(33.6)
    assert not any('yards' in k for k in p)


@pytest.mark.parametrize('args', [(-1,.5,1,.9),(60,1.1,1,.9),(60,.6,-.1,.9),(60,.6,1,float('nan'))])
def test_chain_rejects_invalid_component(args):
    with pytest.raises(ValueError):
        L.chain(*args)


def test_human_formula_frozen_deterministic():
    own = [{'plays': 60, 'dropbacks': 40, 'qb_dropbacks': 40}] * 3
    qb = [{'attempts': 35, 'dropbacks': 40}] * 5
    league = [{'plays': 60, 'dropbacks': 40, 'attempts': 35}] * 128
    assert L.hand_formula(own,qb,league,True) == L.chain(60,2/3,1,.875)
    assert L.hand_formula(own,qb,league,False) is None
    assert L.hand_formula(own,qb,league,None) is None


def test_stable_history_is_not_current_starter_evidence():
    data = tiny(); hist = L.prior_team_games(data, game(3), 'A')
    assert L.continuity(data,hist,'A','AQB')
    assert L.classify_state(True,True) == 'UNCERTAIN_STARTER'
    assert L.classify_state(False) == 'INSUFFICIENT_PREGAME_EVIDENCE'


def test_postgame_starter_and_final_roster_cannot_lift_gate():
    assert L.classify_state(True,True,{'official_starter':'AQB','status':'ACT','depth':1}) == 'UNCERTAIN_STARTER'
    audit = S.read_json(S.ART/'phase1l_qb_source_audit.json')
    audit['source_gate']['passed'] = True
    with pytest.raises(ValueError, match='NEW_PROTOCOL_REQUIRED'):
        L.blocked_lock(audit)


def test_temporal_rule_excludes_target_week_and_incomplete_source():
    target = game(3)
    assert L.legal_prior(game(2,'2024-09-08'),target)
    assert not L.legal_prior(game(3,'2024-09-10'),target)
    assert not L.legal_prior(game(2,'2024-09-21'),target)
    assert not L.legal_prior(game(2,'2024-09-20'),target)
    assert L.cutoff(target).isoformat() == '2024-09-21T00:00:00+00:00'


def test_candidates_do_not_use_target_participant():
    data = tiny(); original = list(L.receipts(data))
    target = game(3)
    changed = deepcopy(data)
    key = (target['game_id'],'A','NEWQB')
    changed['official'][key] = {'attempts':52,'passing_yards':400,'player_display_name':'new'}
    changed['qbs'][key] = {'attempts':52,'dropbacks':55}
    changed['official'].pop((target['game_id'],'A','AQB'))
    updated = list(L.receipts(changed))
    strip = lambda rows: [{k:v for k,v in r.items() if k!='evaluation_truth'} for r in rows]
    assert strip(updated) == strip(original)
    assert all(r['player_id']!='NEWQB' for r in updated if r['game_id']==target['game_id'])


def test_candidate_role_history_stays_on_current_team():
    data = tiny()
    g = game(2,'2024-09-08')
    data['official'][(g['game_id'],'B','AQB')] = {'attempts':40,'passing_yards':300,'player_display_name':'AQB'}
    data['qbs'][(g['game_id'],'B','AQB')] = {'dropbacks':45,'attempts':40}
    a = [r for r in L.receipts(data) if r['game_id']==game(3)['game_id'] and r['team']=='A'][0]
    assert a['prior_current_team_qb_games']==2


def test_censor_is_not_inferred_from_low_attempts():
    rows = list(L.receipts(tiny()))
    assert all(r['censor_status']=='UNKNOWN_NOT_CENSORED_NO_ACCEPTED_EXIT_LABEL' for r in rows)
    assert all(r['starter_confidence'] is None for r in rows)
    assert all(r['final_projected_attempts'] is None for r in rows)


def test_spikes_count_as_official_attempt_opportunity():
    assert S.opportunity_play({'play_type':'qb_spike','qb_spike':'1','pass_attempt':'1','qb_dropback':'0'})
    assert not S.opportunity_play({'play_type':'no_play','qb_dropback':'1'})
    assert not S.opportunity_play({'play_type':'run','qb_kneel':'1','rush_attempt':'1'})
    assert not S.opportunity_play({'play_type':'pass','qb_dropback':'1','two_point_attempt':'1'})


def test_future_rows_fail_before_label_exposure(tmp_path):
    path = tmp_path/'pbp.csv'
    path.write_text('season,week,season_type,attempts\n2026,5,REG,999\n')
    with pytest.raises(ValueError, match='Week 5'):
        list(S.records(path,S.STATS_FIELDS))


def test_later_seasons_are_not_opened_by_blocked_runner(tmp_path):
    path=tmp_path/'stats.csv';path.write_text('season,week,season_type,attempts\n2025,1,REG,999\n')
    with pytest.raises(ValueError, match='2023/2024 only'):
        list(S.records(path,S.STATS_FIELDS))
    assert set(S.manifest()) == {'2023','2024'}


def test_sportsbook_and_postgame_starter_fields_are_not_allowlisted():
    forbidden={'spread_line','total_line','vegas_wp','odds','starting_qb','official_starter','home_score','away_score'}
    assert not (S.PBP_FIELDS|S.STATS_FIELDS)&forbidden
    assert not any(s.startswith('nfl_phase1c_sim') for s in L.__dict__)


def test_source_classifications_preserve_j():
    audit=S.read_json(S.ART/'phase1l_qb_source_audit.json')
    j=S.read_json(S.ART/'phase1j_information_gap_audit.json')
    catalog={s['id']:s for s in j['sources']}
    assert all(s==catalog[s['id']] for s in audit['source_classifications_preserved'])
    assert audit['source_gate']['primary_stable_starter_rows_certifiable']==0
    assert audit['seasons']['2024']['depth']['dt_count']==0
    assert audit['seasons']['2025']['injury_original_publication_rows']==0
    assert audit['seasons']['2026']['scope'].startswith('INHERITED')
    for y in ['2023','2024']:
        assert audit['seasons'][y]['prior_game_qb_history']['qb_pbp_official_attempt_mismatches']==0


def test_locked_metrics_are_not_manufactured():
    audit=S.read_json(S.ART/'phase1l_qb_source_audit.json'); lock=L.blocked_lock(audit)
    report=L.results(audit,lock,list(L.receipts(tiny())))
    assert report['verdict']==L.BLOCKED
    assert all(f['metrics'] is None for f in report['families'].values())
    assert all(p['metrics'] is None for p in report['periods'].values())
    assert report['full_passing_yards']['metrics'] is None
    assert report['oracle_decomposition']['metrics'] is None
    assert report['raw_audit_ledger']['forecasted_rows']==0


def test_stale_lock_is_rejected():
    audit=S.read_json(S.ART/'phase1l_qb_source_audit.json');lock=L.blocked_lock(audit)
    lock['selected']='A'
    with pytest.raises(ValueError, match='lock changed'):
        L.results(audit,lock,[])


def test_receipt_gzip_is_deterministic(tmp_path):
    rows=list(L.receipts(tiny()))
    a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    L.write_receipts(a,rows);L.write_receipts(b,rows)
    assert a.read_bytes()==b.read_bytes()
    assert len(gzip.decompress(a.read_bytes()).splitlines())==len(rows)


def test_protocol_practical_gate_and_censor_are_locked():
    p=S.read_json(S.ART/'phase1l_qb_protocol.json')
    assert set(p['state_categories'])==set(L.STATES)
    assert '1.0 attempt' in p['promotion']['development_and_2025']
    assert 'No poor-performance censor'==p['censor_rule']['benching']
    assert p['source_gate']['current_accepted_archives']==[]
    assert p['no_rescue']
    assert p['architectures']['G_coherent_chain']['parents_required']


def test_frozen_artifacts_when_present():
    snapshot=S.ART/'phase1l_qb_snapshot.json'
    if not snapshot.exists():
        return  # Implementation commit precedes result generation.
    d=S.read_json(snapshot)
    for name,digest in d['artifacts_sha256'].items():
        assert S.sha(S.ART/name)==digest
    rows=[json.loads(line) for line in gzip.decompress((S.ART/'phase1l_qb_receipts.jsonl.gz').read_bytes()).splitlines()]
    assert len(rows)==d['receipt_rows']
    assert all(r['season']==2024 and r['final_projected_attempts'] is None for r in rows)


def test_all_protected_files_byte_identical():
    allowed={'nfl_models/nfl_player_outcome_v2/research_registry.json',
             'nfl_models/nfl_player_outcome_v2/source_inventory.json',  # Phase1M adds one additive metadata key; every Phase1L-era key is checked below
             'tests/test_nfl_v2_phase1k_rushing_efficiency.py'}
    old_paths=subprocess.check_output(['git','ls-tree','-r','--name-only',S.BASE_HEAD],cwd=S.ROOT,text=True).splitlines()
    for path in old_paths:
        if path in allowed:
            continue
        old=subprocess.check_output(['git','show',f'{S.BASE_HEAD}:{path}'],cwd=S.ROOT)
        assert hashlib.sha256(old).hexdigest()==S.sha(S.ROOT/path),path
    changed=subprocess.check_output(['git','diff','--name-only',S.BASE_HEAD],cwd=S.ROOT,text=True).splitlines()
    assert all(p in allowed or 'phase1l_' in p or 'phase1m_' in p or 'phase1n_' in p for p in changed)
    old=json.loads(subprocess.check_output(['git','show',f'{S.BASE_HEAD}:nfl_models/nfl_player_outcome_v2/research_registry.json'],cwd=S.ROOT,text=True))
    current=S.read_json(S.ART/'research_registry.json')
    assert all(current[k]==v for k,v in old.items() if k not in ('status','next_milestone'))
    old_inv=json.loads(subprocess.check_output(['git','show',f'{S.BASE_HEAD}:nfl_models/nfl_player_outcome_v2/source_inventory.json'],cwd=S.ROOT,text=True))
    current_inv=S.read_json(S.ART/'source_inventory.json')
    assert all(current_inv[k]==v for k,v in old_inv.items())

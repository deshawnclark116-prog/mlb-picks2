"""Data-audit integrity only: no predictive fitting or model evaluation."""
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1j_information_audit as J


@pytest.fixture
def bundle():
    return J.validate_bundle()


def test_catalog_complete_and_evidence_factual(bundle):
    assert len(bundle['sources']) == 29
    counts = {}
    for s in bundle['sources']:
        assert J.REQUIRED_SOURCE_FIELDS <= s.keys()
        counts[s['classification']] = counts.get(s['classification'],0) + 1
    assert counts == bundle['classification_counts']
    for e in bundle['evidence']:
        assert e['retrieved_at_utc'] and e['url'].startswith('https://')
        if e['sha256'] is not None:
            assert len(e['sha256']) == 64
    assert any('not parsed' in note.lower() for note in bundle['scope_notes'])


@pytest.mark.parametrize('family',['routes_run','route_participation','alignment','separation','defender_assignment'])
def test_no_proxy_claimed_as_route_tracking(bundle,family):
    assert 'BLOCKED_DATA' in bundle['route_separation_audit'][family]
    assert 'NOT routes-run' in bundle['route_separation_audit']['repo_mislabel']


def test_qb_state_is_not_postgame_or_depth_rank_verification(bundle):
    state = bundle['qb_state_audit']
    assert state['state_enum'] == ['VERIFIED_STARTER','EXPECTED_STARTER','UNCERTAIN_STARTER','MULTI_QB_PACKAGE','EMERGENCY_REPLACEMENT']
    assert 'EXPECTED' in state['classification_rules'][0]
    assert {'observed_at_utc','published_at_utc','version_id','source_sha256','restriction','missingness'} <= state['proposed_schema'].keys()
    assert 'postcutoffreplacement' in state['classification_rules'][-1]
    assert bundle['depth_snapshot_coverage']['2024']['dt_count'] == 0
    assert bundle['depth_snapshot_coverage']['2025']['dt_count'] == 221


def test_current_2026_coverage_and_blitz_not_silently_fresh(bundle):
    m = bundle['current_2026_release_metadata']
    assert m['participation'] == []
    frozen = bundle['frozen_source_coverage']['2026']
    assert m['ftn'][0]['sha256_digest'] == 'sha256:' + frozen['hashes']['ftn']
    assert frozen['ftn_games_by_week_including_postseason']['4'] == 1
    assert sum(frozen['ftn_games_by_week_including_postseason'].values()) == 49
    assert sum(frozen['pbp_reg_games_by_week'].values()) == 63
    assert not frozen['participation_published']
    assert all(not asset['payload_downloaded_this_phase'] for assets in m.values() for asset in assets)


def test_ftn_backfilled_retrieval_is_not_historical_availability(bundle):
    sources = {s['id']:s for s in bundle['sources']}
    assert sources['ftn_public_charting']['classification'] == 'BLOCKED_TIMING'
    c = bundle['frozen_source_coverage']
    assert c['2024']['ftn_date_pulled_min'].startswith('2024-11-13')
    assert c['2025']['ftn_date_pulled_min'].startswith('2026-09-22')
    assert c['2025']['ftn_date_pulled_max'].startswith('2026-09-23')
    assert 'not first public' in sources['ftn_public_charting']['publication_timestamp']
    assert c['2026']['counts']['is_catchable_ball'] == 3029
    assert c['2026']['counts']['pbp_targets'] == 3936


def test_paid_ftn_is_real_schema_not_public_subset_entitlement(bundle):
    f = next(s for s in bundle['sources'] if s['id']=='ftn_full_api')
    assert f['classification'] == 'PAID_BUT_VIABLE'
    assert '$5,000' in f['access_cost'] and '2019' in f['seasons_available']
    facts = bundle['ftn_full_api_schema_facts']
    assert {'qbp','ttp','ttpr','shell','fread'} <= set(facts['NFLChartingConditionTypeEnum']['enum'])
    assert 'rte' in facts['NFLChartingActionTypeEnum']['enum']
    assert {'sep','trg_sep','cball','drp'} <= set(facts['NFLChartingActionDetailType']['enum'])
    assert 'last_updated' in facts['ChartingEventStatusSchema']['properties']
    assert 'first_published_at' not in facts['ChartingEventStatusSchema']['properties']
    assert f['already_in_repo'] is False


def test_no_optimistic_pff_or_ngs_route_claim(bundle):
    s = {s['id']:s for s in bundle['sources']}
    assert s['pff']['classification'] == 'UNKNOWN_NEEDS_ACCESS'
    assert s['ngs_receiving']['play_ids'] is None
    assert 'NOT routes-run' in s['ngs_receiving']['classification_reason']
    assert s['espn_undocumented']['classification'] == 'BLOCKED_LICENSE'
    assert s['gamebook_starters']['classification'] == 'NOT_USEFUL'


def test_ol_still_blocked_and_rushing_scope_limited(bundle):
    assert bundle['offensive_line_audit']['quality_status'] == 'BLOCKED_NO_TIMESTAMP_SAFE_SOURCE_IN_CURRENT_REPO'
    r = bundle['rushing_readiness']
    assert r['model_built'] is False
    assert len(r['allowed_without_ol']) == 6
    assert 'BLOCKED_TIMING' in r['conditional_or_blocked']['DL_LB_absences']
    assert 'BLOCKED_DATA' in r['conditional_or_blocked']['front_tendencies']


def test_receiving_freeze_receiver_signal_and_no_new_protocol(bundle):
    priority = json.loads((J.ART/'phase1j_source_priority.json').read_text())
    assert bundle['receiving_decision'] == 'FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION'
    assert bundle['receiver_depth_signal'] == 'SURVIVED_SIGNAL_NOT_PROMOTED'
    assert priority['next_model_protocol'] is None
    assert priority['procurement_or_collection_performed'] is False
    assert priority['priority_is_not_model_selection'] is True


def test_prior_history_preserved_in_registry_and_inventory(bundle):
    for name in ['source_inventory.json','research_registry.json']:
        old = json.loads(subprocess.check_output(['git','show',f'{J.BASE_HEAD}:nfl_models/nfl_player_outcome_v2/{name}'],cwd=J.ROOT,text=True))
        current = json.loads((J.ART/name).read_text())
        for key,value in old.items():
            if name=='research_registry.json' and key in ('status','next_milestone'):
                continue
            assert current[key] == value, (name,key)
    for path,old_sha in bundle['locked_history_sha256_before'].items():
        old = subprocess.check_output(['git','show',f'{J.BASE_HEAD}:{path}'],cwd=J.ROOT)
        assert hashlib.sha256(old).hexdigest() == old_sha
        if path.endswith(('research_registry.json','source_inventory.json')):
            continue
        assert J.sha(J.ROOT/path) == old_sha


def test_entire_protected_tree_byte_identical():
    allowed = {
        'nfl_v2_phase1j_information_audit.py',
        'tests/test_nfl_v2_phase1j_information_audit.py',
        'tests/test_nfl_v2_phase1i_target_depth.py',
        '.github/workflows/nfl_v2_phase1j_information_audit.yml',
        *{f'nfl_models/nfl_player_outcome_v2/{name}' for name in (
            'phase1j_information_gap_audit.json','phase1j_information_gap_findings.md',
            'phase1j_human_research_gap_matrix.json','phase1j_source_priority.json',
            'research_registry.json','source_inventory.json')},
    }
    published_head = 'edf84ca5be15d53d5525e8b613dac7961797a35d'
    changed = subprocess.check_output(['git','diff','--name-only',J.BASE_HEAD,published_head],cwd=J.ROOT,text=True).splitlines()
    assert set(changed) <= allowed
    protected = subprocess.check_output(['git','ls-tree','-r','--name-only',J.BASE_HEAD],cwd=J.ROOT,text=True).splitlines()
    for path in protected:
        if path in allowed:
            continue
        old = subprocess.check_output(['git','show',f'{J.BASE_HEAD}:{path}'],cwd=J.ROOT)
        published = subprocess.check_output(['git','show',f'{published_head}:{path}'],cwd=J.ROOT)
        assert hashlib.sha256(old).hexdigest() == hashlib.sha256(published).hexdigest(), path


def test_audit_imports_no_models_or_network():
    tree = ast.parse((J.ROOT/'nfl_v2_phase1j_information_audit.py').read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imported.update(n.name for n in node.names)
        if isinstance(node,ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {'__future__','argparse','collections','csv','gzip','hashlib','json','pathlib'}
    assert not any(isinstance(n,ast.FunctionDef) and n.name in {'fit','predict','develop','confirm','score'} for n in ast.walk(tree))


def test_schema_refuses_source_without_evidence(tmp_path):
    for name in ['phase1j_information_gap_audit.json','phase1j_human_research_gap_matrix.json','phase1j_source_priority.json']:
        (tmp_path/name).write_bytes((J.ART/name).read_bytes())
    p = tmp_path/'phase1j_information_gap_audit.json'
    audit = json.loads(p.read_text());audit['sources'][0]['evidence_ids']=['imaginary_source']
    p.write_text(json.dumps(audit))
    with pytest.raises(AssertionError):
        J.validate_bundle(tmp_path)


def put_csv(path,rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def toy_corpus(tmp_path):
    frozen={'seasons':{}}
    for year in (2023,2024,2025,2026):
        rows={'pbp':{'week':'1','season_type':'REG','play_type':'pass','no_play':'0','two_point_attempt':'0','receiver_player_id':'r','passer_player_id':'q','game_id':f'{year}_01_A_B','play_id':'1','air_yards':'10','complete_pass':'0','game_date':f'{year}-09-01'},
              'ftn':{'week':'1','nflverse_game_id':f'{year}_01_A_B','nflverse_play_id':'1','date_pulled':f'{year}-09-03T01:00:00Z','is_catchable_ball':'FALSE','is_contested_ball':'TRUE','is_drop':'FALSE','read_thrown':'DES'},
              'injuries':{'week':'1','date_modified':''}}
        sources={}
        for kind,row in rows.items():
            name=f'{kind}_{year}.csv'+('.gz' if kind=='pbp' else '')
            put_csv(tmp_path/name,[row]);sources[kind]={'status':'PUBLISHED','local_name':name,'sha256':J.sha(tmp_path/name)}
        sources['participation']={'status':'NOT_PUBLISHED','local_name':f'participation_{year}.csv'}
        frozen['seasons'][str(year)]=sources
    return frozen


def test_coverage_deterministic_counts_not_performance(tmp_path):
    frozen=toy_corpus(tmp_path)
    a=J.frozen_coverage(tmp_path,frozen)
    assert a == J.frozen_coverage(tmp_path,frozen)
    assert a['2026']['counts']['is_catchable_ball'] == 1 # FALSE is present, not missing
    assert a['2026']['counts']['date_pulled_before_own_game_date'] == 0
    assert a['2026']['injury_original_publication_timestamp_rows'] == 0


@pytest.mark.parametrize('bad',['revision','new_source','future_week'])
def test_source_guard_refuses_revision_new_source_or_week5(tmp_path,bad):
    frozen=toy_corpus(tmp_path)
    if bad=='revision':
        (tmp_path/'ftn_2024.csv').write_text('changed')
    elif bad=='new_source':
        (tmp_path/'participation_2026.csv').write_text('new')
    else:
        path=tmp_path/'ftn_2026.csv';text=path.read_text().replace('1,2026_01','5,2026_01');path.write_text(text)
        frozen['seasons']['2026']['ftn']['sha256']=J.sha(path)
    with pytest.raises(ValueError):J.frozen_coverage(tmp_path,frozen)


def test_depth_counts_do_not_merge_weeks_or_invent_pubtime(tmp_path):
    r={'season':'2024','week':'1','game_type':'REG','club_code':'A','position':'T','depth_position':'LT','formation':'Offense','depth_team':'1','gsis_id':'p'}
    path=tmp_path/'depth.csv';put_csv(path,[r,{**r,'week':'2'}])
    a=J.depth_coverage(path,2024)
    assert a['dt_count'] == 0 and a['duplicate_rank1_position_slots'] == 0
    assert a['counts']['LT_with_gsis'] == 2
    with pytest.raises(ValueError):J.depth_coverage(path,2026)

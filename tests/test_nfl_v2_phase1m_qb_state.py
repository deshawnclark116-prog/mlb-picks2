import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

import nfl_v2_phase1m_qb_state_acceptance as A
import nfl_v2_phase1m_qb_state_sources as S

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
KICK = '2024-10-06T17:00:00Z'
T24 = S.cutoff_time(KICK, 'T24')
T90 = S.cutoff_time(KICK, 'T90')


def row(**kw):
    base = {'season': 2024, 'week': 5, 'game_id': '2024_05_AAA_BBB', 'team': 'AAA', 'opponent': 'BBB', 'kickoff_time': KICK, 'player_id': '00-1', 'player_name': 'A Qb',
            'provider_player_id': 'p1', 'normalized_player_id': '00-1', 'state_type': 'EXPECTED_STARTER', 'source_provider': 'TEST', 'source_product': 'depth', 'source_record_id': 'r1',
            'publication_timestamp': '2024-10-04T12:00:00Z', 'retrieval_timestamp': '2024-10-04T12:05:00Z', 'valid_from': '2024-10-04T12:00:00Z', 'revision_id': 'v', 'evidence_type': 'PROVIDER_DEPTH',
            'cutoff_eligibility': 'BOTH', 'as_of_certified': False, 'stale_flag': False, 'postgame_only_flag': False, 'backfilled_flag': False, 'original_vintage_flag': True,
            'raw_source_hash': 'a' * 64}
    base.update(kw)
    return base


# ---------------------------------------------------------------- schema
def test_schema_has_every_required_field_and_enum():
    spec = ['season', 'week', 'game_id', 'team', 'opponent', 'kickoff_time', 'player_id', 'player_name', 'provider_player_id', 'normalized_player_id', 'state_type', 'starter_rank', 'depth_rank',
            'expected_start_probability', 'health_state', 'practice_state', 'injury_body_part', 'injury_status', 'roster_status', 'source_provider', 'source_product', 'source_record_id',
            'source_timestamp', 'publication_timestamp', 'retrieval_timestamp', 'effective_timestamp', 'valid_from', 'valid_to', 'revision_id', 'revision_sequence', 'evidence_type',
            'cutoff_eligibility', 'as_of_certified', 'confidence', 'stale_flag', 'postgame_only_flag', 'backfilled_flag', 'original_vintage_flag', 'raw_source_hash', 'normalized_record_hash']
    assert list(S.FIELD_NAMES) == spec
    assert set(S.STATE_TYPES) == {'VERIFIED_STARTER', 'EXPECTED_STARTER', 'PROBABLE_STARTER', 'BACKUP', 'EMERGENCY_QB', 'MULTI_QB', 'UNCERTAIN_STARTER', 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'LIMITED',
                                  'FULL_PARTICIPATION', 'RETURNING_FROM_INJURY', 'RECENTLY_BENCHED', 'RECENTLY_PROMOTED', 'RECENTLY_ACQUIRED'}
    assert set(S.EVIDENCE_TYPES) == {'OFFICIAL_DEPTH', 'OFFICIAL_INJURY', 'PROVIDER_DEPTH', 'PROVIDER_EXPECTED_STARTER', 'TRANSACTION', 'PRACTICE_REPORT', 'COACH_ANNOUNCEMENT', 'OTHER'}
    assert set(S.CUTOFF_ELIGIBILITY) == {'T24', 'T90', 'BOTH', 'NEITHER'}
    assert S.schema_document() == json.loads(json.dumps(S.schema_document()))


def test_cutoffs():
    assert S.cutoff_time(KICK, 'T24') == datetime(2024, 10, 5, 17, 0, tzinfo=timezone.utc)
    assert S.cutoff_time(KICK, 'T90') == datetime(2024, 10, 6, 15, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        S.cutoff_time(KICK, 'T48')


def test_certification_rules_are_strict():
    ok = S.finalize(row(as_of_certified=True, confidence=0.9, revision_sequence=1))
    assert S.validate(ok) == []
    assert 'certified_without_publication_timestamp' in S.validate(row(as_of_certified=True, publication_timestamp=None, confidence=0.9, normalized_record_hash='x', revision_sequence=1))
    assert 'certified_without_original_vintage' in S.validate(row(as_of_certified=True, original_vintage_flag=False, confidence=0.9, normalized_record_hash='x', revision_sequence=1))
    assert 'certified_but_backfilled_or_postgame' in S.validate(row(as_of_certified=True, postgame_only_flag=True, confidence=0.9, normalized_record_hash='x', revision_sequence=1))
    assert 'confidence_without_certification' in S.validate(row(confidence=0.7, normalized_record_hash='x', revision_sequence=1))


# ---------------------------------------------------------------- append-only archive and temporal integrity
def test_append_only_revisions_never_overwrite():
    a = S.QBStateArchive()
    first = a.append(row())
    second = a.append(row(state_type='VERIFIED_STARTER', publication_timestamp='2024-10-06T14:00:00Z', retrieval_timestamp='2024-10-06T14:02:00Z', valid_from='2024-10-06T14:00:00Z'))
    assert (first['revision_sequence'], second['revision_sequence']) == (1, 2)
    assert len(a) == 2 and len(a.history('TEST', 'r1')) == 2
    with pytest.raises(S.ArchiveError):
        a.append(row(revision_sequence=1))
    rows = a.rows()
    rows[0]['state_type'] = 'OUT'
    assert a.rows()[0]['state_type'] == 'EXPECTED_STARTER'
    assert S.finalize(row())['normalized_record_hash'] == S.finalize(row())['normalized_record_hash']
    assert S.finalize(row())['normalized_record_hash'] != S.finalize(row(state_type='OUT'))['normalized_record_hash']


def test_sequence_questionable_expected_verified_has_no_backward_leak():
    a = S.QBStateArchive()
    a.append(row(state_type='QUESTIONABLE', publication_timestamp='2024-10-04T20:00:00Z', retrieval_timestamp='2024-10-04T20:03:00Z', valid_from='2024-10-04T20:00:00Z', cutoff_eligibility='BOTH'))
    a.append(row(state_type='EXPECTED_STARTER', publication_timestamp='2024-10-05T20:00:00Z', retrieval_timestamp='2024-10-05T20:03:00Z', valid_from='2024-10-05T20:00:00Z'))   # after T24
    a.append(row(state_type='VERIFIED_STARTER', publication_timestamp='2024-10-06T15:45:00Z', retrieval_timestamp='2024-10-06T15:50:00Z', valid_from='2024-10-06T15:45:00Z'))   # after T90
    t24 = a.state_at('AAA', '2024_05_AAA_BBB', T24, 'T24')
    t90 = a.state_at('AAA', '2024_05_AAA_BBB', T90, 'T90')
    late = a.state_at('AAA', '2024_05_AAA_BBB', S.parse_ts(KICK), None)
    assert [r['state_type'] for r in t24] == ['QUESTIONABLE']
    assert [r['state_type'] for r in t90] == ['EXPECTED_STARTER']
    assert [r['state_type'] for r in late] == ['VERIFIED_STARTER']
    assert a.state_at('AAA', '2024_05_AAA_BBB', S.parse_ts('2024-10-04T00:00:00Z')) == []


def test_visibility_excludes_late_retrieval_postgame_backfill_and_ineligible():
    a = S.QBStateArchive()
    a.append(row(source_record_id='late', publication_timestamp='2024-10-04T00:00:00Z', retrieval_timestamp='2024-10-08T00:00:00Z'))          # published earlier but WE retrieved it later
    a.append(row(source_record_id='post', postgame_only_flag=True))
    a.append(row(source_record_id='fill', backfilled_flag=True, original_vintage_flag=False))
    a.append(row(source_record_id='t90only', cutoff_eligibility='T90'))
    a.append(row(source_record_id='noPub', publication_timestamp=None, retrieval_timestamp='2024-10-06T16:00:00Z'))                           # retrieval after T90 and no publication time
    a.append(row(source_record_id='expired', valid_to='2024-10-05T00:00:00Z'))
    ids = lambda cutoff, label: {r['source_record_id'] for r in a.state_at('AAA', '2024_05_AAA_BBB', cutoff, label)}
    assert ids(T24, 'T24') == set()
    assert ids(T90, 'T90') == {'t90only'}
    assert ids(T90, 'T24') == set()


def test_certified_only_filter_and_summary():
    a = S.QBStateArchive()
    a.append(row(source_record_id='a', as_of_certified=True, confidence=0.9))
    a.append(row(source_record_id='b', player_id='00-2', normalized_player_id='00-2', player_name='B Qb'))
    assert len(a.state_at('AAA', '2024_05_AAA_BBB', T24)) == 2
    assert len(a.state_at('AAA', '2024_05_AAA_BBB', T24, certified_only=True)) == 1
    assert S.expected_starter_summary(a.state_at('AAA', '2024_05_AAA_BBB', T24))['state'] == 'MULTI_QB'
    assert S.expected_starter_summary([])['state'] == 'UNCERTAIN_STARTER'
    one = S.expected_starter_summary(a.state_at('AAA', '2024_05_AAA_BBB', T24, certified_only=True))
    assert one['state'] == 'EXPECTED_STARTER' and one['players'] == ['00-1']
    out = S.expected_starter_summary([S.finalize(row(state_type='OUT'))])
    assert out['state'] == 'UNCERTAIN_STARTER'


# ---------------------------------------------------------------- scorecard and decision
def test_scorecard_is_complete_honest_and_has_no_invented_prices():
    assert S.validate_scorecard() == []
    names = {r['provider'] for r in S.PROVIDERS}
    for need in ('Sportradar', 'SportsDataIO', 'Stats Perform / Opta', 'Genius Sports', 'Sports Info Solutions (SIS)', 'TruMedia', 'FTN Data', 'PFF', 'NFL official injury report (NFL.com)', 'nflverse'):
        assert need in names
    for r in S.PROVIDERS:
        assert set(S.SCORECARD_FIELDS) <= set(r)
        assert r['payload_tested'] != '' and r['final_classification'] in S.CLASSIFICATIONS
        for key in ('annual_cost', 'historical_backfill_cost'):
            text = str(r[key])
            assert '$' not in text or 'Phase1J' in text                      # the only dollar figures are cited from the frozen Phase1J audit
    assert not any(r['final_classification'].startswith('ACCEPTED') for r in S.PROVIDERS)
    assert S.reopen_decision() == 'PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS'


def test_acceptance_without_payload_is_rejected():
    bad = [dict(S.PROVIDERS[0], final_classification='ACCEPTED_FOR_PHASE1L')]
    assert S.validate_scorecard(bad) and S.validate_scorecard(bad)[0][1] == 'accepted_without_payload'


def test_reopen_decision_rule_variants():
    base = [dict(r, final_classification='NOT_USEFUL') for r in S.PROVIDERS]
    assert S.reopen_decision(base) == 'PHASE1L_REMAINS_BLOCKED_STARTER_STATE_DATA'
    hist = [dict(base[0], final_classification='ACCEPTED_FOR_PHASE1L', payload_tested='TRIAL')] + base[1:]
    assert S.reopen_decision(hist, {'historical_thresholds_met': False}) != 'PHASE1L_CAN_REOPEN_HISTORICALLY'
    assert S.reopen_decision(hist, {'historical_thresholds_met': True}) == 'PHASE1L_CAN_REOPEN_HISTORICALLY'
    fwd = [dict(base[0], final_classification='ACCEPTED_FORWARD_ONLY', payload_tested='LIVE', starter_state='PARTIAL: depth')] + base[1:]
    assert S.reopen_decision(fwd) == 'PHASE1L_CAN_REOPEN_FORWARD_ONLY'
    fwd_no_starter = [dict(base[0], final_classification='ACCEPTED_FORWARD_ONLY', payload_tested='LIVE', starter_state='NO')] + base[1:]
    assert S.reopen_decision(fwd_no_starter) == 'PHASE1L_REMAINS_BLOCKED_STARTER_STATE_DATA'
    pending = [dict(base[0], final_classification='PROMISING_NEEDS_SAMPLE')] + base[1:]
    assert S.reopen_decision(pending) == 'PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS'
    assert set(S.DECISIONS) >= {S.reopen_decision(x) for x in (base, hist, fwd, pending)}


def test_threshold_gate_matches_protocol():
    proto = json.loads((ART / 'phase1m_qb_state_protocol.json').read_text())['historical_reopen_numeric_thresholds']
    assert proto['team_games_with_certified_pre_cutoff_starter_state_min'] == 0.95 and proto['player_identity_join_to_gsis_min'] == 0.99
    good = {'certified_pre_cutoff_team_game_fraction': 0.95, 'gsis_join_fraction': 0.99, 'publication_timestamp_fraction': 1.0, 'original_vintage_required_met': True}
    assert A.historical_thresholds_met(good)
    for key, bad in (('certified_pre_cutoff_team_game_fraction', 0.94), ('gsis_join_fraction', 0.98), ('publication_timestamp_fraction', 0.99), ('original_vintage_required_met', False)):
        assert not A.historical_thresholds_met({**good, key: bad})
    assert not A.historical_thresholds_met(None)


# ---------------------------------------------------------------- acceptance harness
def write_games(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id', 'season', 'game_type', 'week', 'gameday', 'gametime', 'away_team', 'home_team', 'spread_line', 'total_line', 'home_moneyline'])
        w.writeheader()
        w.writerows(rows)


def test_games_loader_is_allowlisted_regular_season_only(tmp_path):
    p = tmp_path / 'g.csv'
    write_games(p, [{'game_id': '2024_05_AAA_BBB', 'season': 2024, 'game_type': 'REG', 'week': 5, 'gameday': '2024-10-06', 'gametime': '13:00', 'away_team': 'AAA', 'home_team': 'BBB', 'spread_line': -3, 'total_line': 44, 'home_moneyline': -150},
                    {'game_id': '2024_19_X_Y', 'season': 2024, 'game_type': 'WC', 'week': 19, 'gameday': '2025-01-11', 'gametime': '16:30', 'away_team': 'X', 'home_team': 'Y', 'spread_line': 1, 'total_line': 40, 'home_moneyline': 100}])
    games = A.load_games(p, 2024)
    assert len(games) == 1 and games[0]['kickoff_utc'] == '2024-10-06T17:00:00Z'
    assert set(games[0]) == set(A.GAMES_ALLOWLIST) | {'kickoff_utc'}
    assert not any('spread' in k or 'moneyline' in k or 'total_line' in k for k in games[0])
    g26 = tmp_path / 'g26.csv'
    write_games(g26, [{'game_id': '2026_01_A_B', 'season': 2026, 'game_type': 'REG', 'week': 1, 'gameday': '2026-09-10', 'gametime': '20:20', 'away_team': 'A', 'home_team': 'B', 'spread_line': 0, 'total_line': 0, 'home_moneyline': 0}])
    with pytest.raises(ValueError):
        A.load_games(g26, 2026)


def synthetic_qb():
    teams = ['A', 'B', 'C', 'D', 'E', 'F']
    qb = {}
    for w in range(1, 8):
        for t in teams:
            qb[(w, t)] = {f'{t}1': 30.0}
    qb[(5, 'B')] = {'B1': 12.0, 'B2': 14.0}          # multi-QB (second >= 10)
    qb[(5, 'C')] = {'C9': 33.0}                      # new qb
    qb[(5, 'D')] = {'D1': 4.0, 'D2': 30.0}            # D2 never appeared earlier unless added below
    qb[(3, 'D')] = {'D1': 20.0, 'D2': 10.0}
    qb[(4, 'D')] = {'D1': 30.0}
    qb[(5, 'E')] = {'E1': 30.0}
    qb[(1, 'F')] = {'F2': 25.0}                       # F2 had earlier attempts ...
    qb[(4, 'F')] = {'F1': 28.0}
    qb[(5, 'F')] = {'F2': 30.0}                       # ... and returns: change to prior backup / return
    return qb


def synthetic_games():
    return [{'game_id': f'2024_{w:02d}_{a}_{h}', 'week': str(w), 'away_team': a, 'home_team': h, 'kickoff_utc': f'2024-10-{w:02d}T17:00:00Z', 'season': '2024', 'game_type': 'REG', 'gameday': '', 'gametime': ''}
            for w in range(1, 8) for a, h in (('A', 'B'), ('C', 'D'), ('E', 'F'))]


def test_archetype_labels_precedence_and_selection_rule():
    labelled = A.label_team_games(synthetic_games(), synthetic_qb())
    by = {(r['week'], r['team']): r['archetype'] for r in labelled}
    assert by[(5, 'B')] == 'MULTI_QB_GAME'
    assert by[(5, 'C')] == 'NEW_OR_SURPRISE_QB'
    assert by[(5, 'D')] == 'CHANGE_TO_PRIOR_BACKUP_OR_RETURN'
    assert by[(5, 'F')] == 'CHANGE_TO_PRIOR_BACKUP_OR_RETURN'
    assert by[(5, 'E')] == 'STABLE_VETERAN'
    assert not any(r['week'] < 3 for r in labelled)
    sample = A.select_sample(labelled)
    assert len(sample) <= 4 * len(A.LABELS)
    for label in A.LABELS:
        picks = [r for r in sample if r['archetype'] == label]
        pool = sorted((r for r in labelled if r['archetype'] == label), key=lambda r: A.sample_key(r['game_id'], r['team']))
        assert [(r['game_id'], r['team']) for r in picks] == [(r['game_id'], r['team']) for r in pool[:4]]
    assert A.select_sample(labelled) == sample                                   # deterministic
    # the digest depends only on ids and the preregistered salt, never on the source looking correct
    import hashlib
    r0 = sample[0]
    assert r0['sample_digest'] == hashlib.sha256((r0['game_id'] + '|' + r0['team'] + '|phase1m-sample-v1').encode()).hexdigest()


def test_pinned_hashes_refuse_changed_inputs(tmp_path):
    p = tmp_path / 's.csv'
    p.write_text('season,season_type,week,team,position,player_id,attempts\n2024,REG,1,A,QB,x,10\n')
    with pytest.raises(ValueError):
        A.load_qb_attempts(p)
    with pytest.raises(ValueError):
        A.timing_probe(p, [])


def test_wayback_status_and_probe_handles_429_without_network():
    cut = datetime(2024, 10, 5, 17, 0, tzinfo=timezone.utc)
    assert A.wayback_status(None, cut) == 'NONE'
    assert A.wayback_status('20241005120000', cut) == 'PRE_CUTOFF_WITHIN_72H'
    assert A.wayback_status('20241001120000', cut) == 'PRE_CUTOFF_STALE'
    assert A.wayback_status('20241006000000', cut) == 'POST_CUTOFF_ONLY_UNKNOWN'
    sample = [{'game_id': '2024_05_AAA_BBB', 'team': 'AAA', 'week': 5, 'kickoff_utc': KICK, 'archetype': 'STABLE_VETERAN'}]
    calls, sleeps = [], []

    def fake(url):
        calls.append(url)
        if len(calls) == 4:
            return 429, ''
        snap = {'archived_snapshots': {'closest': {'available': True, 'timestamp': '20241005100000', 'status': '200'}}}
        return 200, json.dumps(snap)
    out = A.wayback_probe(sample, fetch=fake, sleep=lambda s: sleeps.append(s), throttle=6.0, clock=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert out['stopped']['reason'] == 'HTTP_429' and out['completed_requests'] == 3 and len(calls) == 4
    assert out['planned_requests'] == 6 and sleeps == [6.0, 6.0, 6.0]
    assert all(r['url'].startswith('https://') and 'archive.org/wayback/available' in c for r, c in zip(out['results'], calls))


def test_continuity_audit_not_run_without_certified_state_and_diagnostic_when_supplied():
    receipts = [{'game_id': 'g1', 'team': 'A', 'player_id': 'p1', 'pregame_continuity_hint_not_certified': True},
                {'game_id': 'g2', 'team': 'A', 'player_id': 'p1', 'pregame_continuity_hint_not_certified': True},
                {'game_id': 'g3', 'team': 'B', 'player_id': 'p9', 'pregame_continuity_hint_not_certified': False}]
    assert A.continuity_hint_audit(receipts, {})['status'] == 'NOT_RUN_NO_CERTIFIED_STARTER_STATE'
    out = A.continuity_hint_audit(receipts, {('g1', 'A'): {'player_id': 'p1', 'archetype': 'stable'}, ('g2', 'A'): {'player_id': 'p2', 'archetype': 'injury_replacement'}})
    assert out['status'] == 'RUN_DIAGNOSTIC_ONLY'
    assert out['metrics']['ALL']['n'] == 2 and out['metrics']['ALL']['continuity_accuracy'] == 0.5 and out['metrics']['ALL']['false_continuity_rate'] == 0.5
    assert out['metrics']['injury_replacement']['false_continuity_rate'] == 1.0


def test_acceptance_questions_validate_answers():
    with pytest.raises(ValueError):
        A.question_row('x', {'bogus': 'PASS'}, '')
    with pytest.raises(ValueError):
        A.question_row('x', {'can_query_2024_game': 'MAYBE'}, '')
    assert not A.question_row('x', {}, '')['passes_all']
    assert A.question_row('x', {q: 'PASS' for q in A.QUESTIONS}, '')['passes_all']


# ---------------------------------------------------------------- committed artifacts
def load(name):
    return json.loads((ART / name).read_text())


def test_committed_design_artifacts_match_the_code():
    for name, value in S.build_outputs().items():
        assert (ART / name).read_text() == S.dump(value), name


def test_results_never_claim_a_payload_pass():
    res = load('phase1m_qb_state_acceptance_results.json')
    assert 'NO_PROVIDER_PAYLOAD_TESTED' in res['headline'] and res['vendor_payloads_tested'] == 0 and res['historical_thresholds_met'] is False
    assert not any(r['passes_all'] for r in res['acceptance_by_provider'])
    assert res['historical_archetype_test']['status'] == 'NOT_RUN_NO_SOURCE_ACCESS'
    assert res['prior_continuity_hint_audit']['status'] == 'NOT_RUN_NO_CERTIFIED_STARTER_STATE'
    assert res['no_models'] is True and res['no_sportsbook'] is True
    sc = load('phase1m_qb_state_provider_scorecard.json')
    assert sc['decision_from_scorecard'] == S.reopen_decision()
    snap = load('phase1m_qb_state_snapshot.json')
    assert snap['decision'] == S.reopen_decision() and snap['decision'] in S.DECISIONS
    findings = (ART / 'phase1m_qb_state_findings.md').read_text()
    assert snap['decision'] in findings and 'NO_PROVIDER_PAYLOAD_TESTED' in findings


def test_sample_in_results_follows_the_preregistered_rule():
    sample = load('phase1m_qb_state_acceptance_results.json')['sample']
    rows = sample['sample']
    assert 0 < len(rows) <= 16
    import hashlib
    for r in rows:
        assert r['sample_digest'] == hashlib.sha256((r['game_id'] + '|' + r['team'] + '|phase1m-sample-v1').encode()).hexdigest()
    for label in A.LABELS:
        picks = [r['sample_digest'] for r in rows if r['archetype'] == label]
        assert picks == sorted(picks) and len(picks) <= 4
    assert sample['stats_sha256'] == A.STATS_2024_SHA256


def test_protocol_was_committed_before_results():
    log = subprocess.run(['git', 'log', '--format=%H', '--diff-filter=A', '--', 'nfl_models/nfl_player_outcome_v2/phase1m_qb_state_protocol.json',
                          'nfl_models/nfl_player_outcome_v2/phase1m_qb_state_acceptance_results.json'], cwd=ROOT, capture_output=True, text=True)
    if log.returncode != 0 or not log.stdout.strip():
        pytest.skip('git history unavailable')
    first = {}
    for name in ('phase1m_qb_state_protocol.json', 'phase1m_qb_state_acceptance_results.json'):
        out = subprocess.run(['git', 'log', '--format=%H', '--diff-filter=A', '--', 'nfl_models/nfl_player_outcome_v2/' + name], cwd=ROOT, capture_output=True, text=True).stdout.split()
        first[name] = out[-1] if out else None
    if not first['phase1m_qb_state_acceptance_results.json']:
        pytest.skip('results not committed yet')
    order = subprocess.run(['git', 'rev-list', '--reverse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert order.index(first['phase1m_qb_state_protocol.json']) < order.index(first['phase1m_qb_state_acceptance_results.json'])


def test_modules_are_data_only_no_models_no_betting_no_network_in_sources():
    src = (ROOT / 'nfl_v2_phase1m_qb_state_sources.py').read_text().lower()
    acc = (ROOT / 'nfl_v2_phase1m_qb_state_acceptance.py').read_text().lower()
    for word in ('numpy', 'scipy', 'sklearn', 'xgboost', 'torch', 'statsmodels'):
        assert 'import ' + word not in src and 'import ' + word not in acc and 'from ' + word not in src and 'from ' + word not in acc
    assert 'urllib' not in src and 'requests' not in src and 'socket' not in src
    assert not any(k in A.GAMES_ALLOWLIST for k in ('spread_line', 'total_line', 'home_moneyline', 'away_moneyline'))


def test_frozen_research_and_other_sports_are_byte_identical_to_base():
    base = S.BASE_HEAD
    r = subprocess.run(['git', 'diff', '--name-status', base, 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable (shallow clone)')
    changed = {}
    for line in r.stdout.splitlines():
        status, _, path = line.partition('\t')
        changed[path] = status
    allowed_new = {'nfl_v2_phase1m_qb_state_sources.py', 'nfl_v2_phase1m_qb_state_acceptance.py', 'tests/test_nfl_v2_phase1m_qb_state.py', '.github/workflows/nfl_v2_phase1m_qb_state_audit.yml'}
    allowed_new |= {'nfl_models/nfl_player_outcome_v2/' + n for n in (
        'phase1m_qb_state_protocol.json', 'phase1m_qb_state_provider_scorecard.json', 'phase1m_qb_state_acceptance_results.json', 'phase1m_qb_state_schema.json',
        'phase1m_qb_state_findings.md', 'phase1m_qb_state_snapshot.json', 'phase1m_forward_capture_plan.json')}
    allowed_modified = {'nfl_models/nfl_player_outcome_v2/source_inventory.json', 'nfl_models/nfl_player_outcome_v2/research_registry.json',
                        'tests/test_nfl_v2_phase1l_qb_opportunity.py', 'tests/test_nfl_v2_phase1m_qb_state.py'}   # test-only: its byte-identity guard now permits Phase1M's additive inventory key
    for path, status in changed.items():
        assert (status == 'A' and (path in allowed_new or 'phase1n' in path)) or (status == 'M' and path in allowed_modified), (status, path)   # later phases add only new phase1n files
    # the only edit to the Phase1L test widens its guard; the Phase1L scientific code/protocol/results are untouched
    diff = subprocess.run(['git', 'diff', '-U0', base, 'HEAD', '--', 'tests/test_nfl_v2_phase1l_qb_opportunity.py'], cwd=ROOT, capture_output=True, text=True).stdout
    assert 'phase1m_' in diff and 'source_inventory' in diff
    # additive-only edits to the two shared registries: every pre-existing top-level key keeps its exact value
    for name in ('source_inventory.json', 'research_registry.json'):
        old = json.loads(subprocess.run(['git', 'show', base + ':nfl_models/nfl_player_outcome_v2/' + name], cwd=ROOT, capture_output=True, text=True).stdout)
        new = load(name)
        for key, value in old.items():
            if name == 'research_registry.json' and key in ('status', 'next_milestone'):
                continue                                                         # the two headline pointers move forward; every phase entry below stays identical
            assert new[key] == value, (name, key)


def test_workflow_is_read_only_and_runs_the_checks():
    text = (ROOT / '.github/workflows/nfl_v2_phase1m_qb_state_audit.yml').read_text()
    assert 'permissions:\n  contents: read' in text and 'secrets.' not in text
    assert 'nfl_v2_phase1m_qb_state_sources.py --check' in text and 'tests/test_nfl_v2_phase1m_qb_state.py' in text
    assert 'probe-wayback' not in text                                                  # CI never calls an external archive or provider

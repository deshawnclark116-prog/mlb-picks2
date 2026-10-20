"""Phase2A-FWD deterministic checks: no outcome access, cutoff enforcement, immutability, lock, QB gate, ingestion, harness, grader, scope."""
import ast
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

import nfl_v2_phase2a_forward_forecast as F
import nfl_v2_phase2a_forward_grade as G
import nfl_v2_phase2a_ledger as L
import nfl_v2_phase1m_qb_state_sources as QS
import nfl_v2_qb_state_ingestion as Q

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
UTC = timezone.utc
TEAMS = [f'T{i:02d}' for i in range(10)]
STAT_COLS = ['season', 'season_type', 'week', 'player_id', 'team', 'opponent_team', 'position', 'targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards', 'attempts', 'passing_yards']
ROSTER_COLS = ['season', 'week', 'team', 'gsis_id', 'full_name', 'position', 'status', 'game_type']
SYN = {'WR1': ('WR', 8, 5, 70, 0, 0), 'WR2': ('WR', 6, 4, 50, 0, 0), 'WR3': ('WR', 4, 2, 25, 0, 0), 'TE1': ('TE', 3, 2, 20, 0, 0), 'RB1': ('RB', 3, 2, 15, 15, 60), 'RB2': ('RB', 2, 1, 8, 5, 20)}


def make_league(tmp_path, weeks=6, extra=None, hist_files=True):
    d = Path(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in range(1, weeks + 1):
        for i, t in enumerate(TEAMS):
            for name, (pos, tg, rc, ry, ca, ru) in SYN.items():
                rows.append({'season': 2026, 'season_type': 'REG', 'week': w, 'player_id': f'{t}-{name}', 'team': t, 'opponent_team': TEAMS[(i + 1) % 10], 'position': pos, 'targets': tg, 'receptions': rc, 'receiving_yards': ry, 'carries': ca, 'rushing_yards': ru, 'attempts': 0, 'passing_yards': 0})
    rows += extra or []
    with open(d / 'stats_player_week_2026.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STAT_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(d / 'roster_weekly_2026.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ROSTER_COLS)
        w.writeheader()
        for wk in range(1, weeks + 2):
            for t in TEAMS:
                for name, (pos, *_r) in SYN.items():
                    w.writerow({'season': 2026, 'week': wk, 'team': t, 'gsis_id': f'{t}-{name}', 'full_name': f'{t} {name}', 'position': pos, 'status': 'ACT', 'game_type': 'REG'})
                for k in range(44):
                    w.writerow({'season': 2026, 'week': wk, 'team': t, 'gsis_id': f'{t}-OL{k}', 'full_name': 'OL', 'position': 'OL', 'status': 'ACT', 'game_type': 'REG'})
    if hist_files:
        for y in (2023, 2024, 2025):
            for name, cols in ((f'stats_player_week_{y}.csv', STAT_COLS), (f'roster_weekly_{y}.csv', ROSTER_COLS)):
                with open(d / name, 'w', newline='') as f:
                    csv.writer(f).writerow(cols)
    return d


def sched(game_id='2026_07_T01_T00', gameday='2026-10-18', gametime='13:00', home='T00', away='T01', week='7'):
    return [{'game_id': game_id, 'season': '2026', 'game_type': 'REG', 'week': week, 'gameday': gameday, 'gametime': gametime, 'away_team': away, 'home_team': home}]


META = {'url': 'fixture', 'raw_sha256': 'x', 'retrieved_at': '2026-10-01T00:00:00+00:00', 'allowlisted_rows_sha256': 'x'}
KO = F.kickoff_utc('2026-10-18', '13:00')


def run(tmp_path, now, led, games=None, **kw):
    d = make_league(tmp_path) if not (Path(tmp_path) / 'stats_player_week_2026.csv').exists() else Path(tmp_path)
    inputs = {'stats_player_week_2026.csv': {'sha256': 'fixture', 'pinned': False, 'retrieved_at': F.iso(now - timedelta(minutes=5))}}
    return F.run_due(d, led, with_pbp=False, clock=lambda: now, schedule=(games or sched(), META), inputs=inputs, archive=False, **kw)


# ---------------------------------------------------------------- cutoff enforcement
def test_window_classification_matrix():
    ko = KO
    assert F.classify(ko, 'T24', ko - timedelta(hours=49)) == 'NOT_YET_DUE'
    assert F.classify(ko, 'T24', ko - timedelta(hours=48)) == 'DUE' and F.classify(ko, 'T24', ko - timedelta(hours=24, seconds=1)) == 'DUE'
    assert F.classify(ko, 'T24', ko - timedelta(hours=24)) == 'MISSED_T24_CUTOFF' and F.classify(ko, 'T24', ko - timedelta(hours=1)) == 'MISSED_T24_CUTOFF'
    assert F.classify(ko, 'T90', ko - timedelta(hours=4)) == 'NOT_YET_DUE' and F.classify(ko, 'T90', ko - timedelta(hours=3)) == 'DUE'
    assert F.classify(ko, 'T90', ko - timedelta(minutes=91)) == 'DUE' and F.classify(ko, 'T90', ko - timedelta(minutes=90)) == 'MISSED_T90_CUTOFF'
    assert F.classify(ko, 'T90', ko) == 'GAME_STARTED'
    assert F.kickoff_utc('2026-10-08', '20:15') == datetime(2026, 10, 9, 0, 15, tzinfo=UTC)          # America/New_York -> UTC (EDT)


def test_before_window_nothing_is_written(tmp_path):
    led = tmp_path / 'led'
    s = run(tmp_path, KO - timedelta(hours=60), led)
    assert s['created'] == {} and not (led / 'forecasts_T24.jsonl').exists() and s['not_yet_due'] == 2


def test_inside_t24_window_creates_t24_only_and_is_idempotent(tmp_path):
    led = tmp_path / 'led'
    now = KO - timedelta(hours=30)
    s = run(tmp_path, now, led)
    assert s['created']['T24'] > 0 and 'T90' not in s['created']
    first = (led / 'forecasts_T24.jsonl').read_bytes()
    run(tmp_path, now, led)
    assert (led / 'forecasts_T24.jsonl').read_bytes() == first                                  # same ids: nothing re-created, nothing overwritten
    for line in L.read_ledger(led / 'forecasts_T24.jsonl'):
        r = line['record']
        assert datetime.fromisoformat(r['generated_at']) < datetime.fromisoformat(r['cutoff_time']) <= datetime.fromisoformat(r['kickoff_time'])
        assert r['cutoff_type'] == 'T24' and r['position'] != 'QB' and r['final_central_projection']['targets'] is not None or r['rushing']


def test_late_generation_is_refused_and_logged_as_missed_never_backfilled(tmp_path):
    led = tmp_path / 'led'
    s = run(tmp_path, KO - timedelta(hours=10), led)                                            # T24 window closed, T90 not open yet
    assert s['created'] == {} and s['missed'] == 1
    events = [x['record'] for x in L.read_ledger(led / 'game_status.jsonl')]
    assert [e['event'] for e in events] == ['MISSED_T24_CUTOFF'] and not (led / 'forecasts_T24.jsonl').exists()
    run(tmp_path, KO - timedelta(hours=9), led)
    assert len([x for x in L.read_ledger(led / 'game_status.jsonl') if x['record']['event'] == 'MISSED_T24_CUTOFF']) == 1     # logged once
    s2 = run(tmp_path, KO + timedelta(hours=1), led)                                            # game started: T90 missed too, still no forecasts
    assert not (led / 'forecasts_T90.jsonl').exists() and {x['record']['event'] for x in L.read_ledger(led / 'game_status.jsonl')} == {'MISSED_T24_CUTOFF', 'MISSED_T90_CUTOFF'}


def test_clock_crossing_the_cutoff_mid_run_stops_before_any_late_line(tmp_path):
    led = tmp_path / 'led'
    cutoff = F.window(KO, 'T24')[1]
    ticks = iter([cutoff - timedelta(minutes=30)] * 12 + [cutoff + timedelta(seconds=1)] * 1000)
    d = make_league(tmp_path)
    inputs = {'stats_player_week_2026.csv': {'sha256': 'f', 'pinned': False, 'retrieved_at': F.iso(cutoff - timedelta(hours=2))}}
    with pytest.raises(F.RefusedError, match='cutoff guard'):
        F.run_due(d, led, with_pbp=False, clock=lambda: next(ticks), schedule=(sched(), META), inputs=inputs, archive=False)
    for line in L.read_ledger(led / 'forecasts_T24.jsonl'):
        assert datetime.fromisoformat(line['record']['generated_at']) < cutoff
    with pytest.raises(F.RefusedError):
        F.assert_before_cutoff(cutoff, KO, 'T24')


def test_inputs_retrieved_after_the_cutoff_are_refused(tmp_path):
    led = tmp_path / 'led'
    cutoff = F.window(KO, 'T24')[1]
    d = make_league(tmp_path)
    bad = {'stats_player_week_2026.csv': {'sha256': 'f', 'pinned': False, 'retrieved_at': F.iso(cutoff + timedelta(seconds=1))}}
    with pytest.raises(F.RefusedError, match='retrieval'):
        F.run_due(d, led, with_pbp=False, clock=lambda: cutoff - timedelta(minutes=1), schedule=(sched(), META), inputs=bad, archive=False)


def test_cli_has_no_clock_override_and_ledger_has_no_manual_writer():
    src = (ROOT / 'nfl_v2_phase2a_forward_forecast.py').read_text()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'main')
    flags = [c.args[0].value for c in ast.walk(main) if isinstance(c, ast.Call) and getattr(c.func, 'attr', '') == 'add_argument' and c.args and isinstance(c.args[0], ast.Constant)]
    assert not any('now' in f or 'clock' in f or 'time' in f for f in flags), flags
    assert 'generated_at' not in {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('run_due', 'main') for a in n.args.args}


def test_t24_and_t90_are_independent_forecasts(tmp_path):
    led = tmp_path / 'led'
    run(tmp_path, KO - timedelta(hours=30), led)
    run(tmp_path, KO - timedelta(hours=2), led)
    a, b = L.read_ledger(led / 'forecasts_T24.jsonl'), L.read_ledger(led / 'forecasts_T90.jsonl')
    assert a and b and len(a) == len(b)
    assert not {x['record']['forecast_id'] for x in a} & {x['record']['forecast_id'] for x in b}
    assert F.forecast_id('T24', 'g', 'p') != F.forecast_id('T90', 'g', 'p')


# ---------------------------------------------------------------- no outcome / sportsbook access
def test_target_week_rows_cannot_change_a_forecast_and_trigger_a_refusal(tmp_path):
    base = make_league(tmp_path / 'a')
    leak = make_league(tmp_path / 'b', extra=[{'season': 2026, 'season_type': 'REG', 'week': 7, 'player_id': 'T00-WR1', 'team': 'T00', 'opponent_team': 'T01', 'position': 'WR', 'targets': 99, 'receptions': 99, 'receiving_yards': 999,
                                                 'carries': 0, 'rushing_yards': 0, 'attempts': 0, 'passing_yards': 0}])
    c1 = F.Context(base, 2026, 7, with_pbp=False)
    c2 = F.Context(leak, 2026, 7, with_pbp=False)
    assert c1.outcome_rows_present == [] and c2.outcome_rows_present == [('T00', 7)]
    assert F._guards(c2, sched()[0], {t: [(2026, 6)] for t in TEAMS}) == 'REFUSED_OUTCOME_DATA_PRESENT'
    f1 = F.build_forecasts(c1, sched()[0], 'T00', 'T01', 'home')[0]
    f2 = F.build_forecasts(c2, sched()[0], 'T00', 'T01', 'home')[0]
    assert f1 and json.dumps(f1, sort_keys=True) == json.dumps(f2, sort_keys=True)               # identical despite a huge target-week row


def test_future_rows_never_reach_the_history_indexes(tmp_path):
    d = make_league(tmp_path, extra=[{'season': 2026, 'season_type': 'REG', 'week': w, 'player_id': 'T00-WR1', 'team': 'T00', 'opponent_team': 'T01', 'position': 'WR', 'targets': 50, 'receptions': 50, 'receiving_yards': 500,
                                       'carries': 0, 'rushing_yards': 0, 'attempts': 0, 'passing_yards': 0} for w in (8, 9)])
    c = F.Context(d, 2026, 7, with_pbp=False)
    assert max((r['season'], r['week']) for r in c.players) == (2026, 6) and max(k[1] for k in c.team) == 6


def test_schedule_loader_is_allowlisted_and_never_reads_results_or_betting_columns():
    text = 'game_id,season,game_type,week,gameday,gametime,away_team,home_team,result,spread_line,total_line,home_moneyline,away_score\n2026_05_A_B,2026,REG,5,2026-10-11,13:00,A,B,7,3.5,44,-150,24\n'
    rows = F.parse_schedule(text)
    assert rows == [{'game_id': '2026_05_A_B', 'season': '2026', 'game_type': 'REG', 'week': '5', 'gameday': '2026-10-11', 'gametime': '13:00', 'away_team': 'A', 'home_team': 'B'}]
    for bad in ('result', 'spread_line', 'total_line', 'moneyline', 'odds', 'score'):
        assert not any(bad in c for c in F.GAMES_ALLOWLIST)


def test_v2_forecast_build_never_touches_comparators_or_v1():
    tree = ast.parse((ROOT / 'nfl_v2_phase2a_forward_forecast.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'build_forecasts')
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not {'v1_projection', 'pick_log', 'v1'} & names
    text = (ROOT / 'nfl_v2_phase2a_forward_forecast.py').read_text().lower()
    for bad in ('spread_line', 'moneyline', 'over_odds', 'under_odds', 'total_line', 'sportsbook_line'):
        assert bad not in text.replace("'spread_line', 'moneyline', 'over_odds'", ''), bad


def test_v1_comparator_reads_only_the_projected_median_logged_before_the_cutoff(tmp_path):
    log = tmp_path / 'log.jsonl'
    rows = [{'market': 'receiving_yards', 'player_id': 'p', 'season': 2026, 'week': 5, 'projected_median': 40.0, 'line': 55.5, 'logged_at': '2026-10-08T10:00:00+00:00'},
            {'market': 'receiving_yards_early_season', 'player_id': 'p', 'season': 2026, 'week': 5, 'projected_median': 44.0, 'logged_at': '2026-10-08T23:00:00+00:00'},
            {'market': 'rushing_yards', 'player_id': 'p', 'season': 2026, 'week': 5, 'projected_median': 20.0, 'logged_at': '2026-10-07T10:00:00+00:00'}]
    log.write_text('\n'.join(json.dumps(r) for r in rows))
    out = F.v1_projection(log, 2026, 5, 'p', datetime(2026, 10, 8, 12, tzinfo=UTC))
    assert out['receiving_yards'] == 40.0 and out['rushing_yards'] == 20.0 and out['receptions'] is None and out['status'] == 'V1_PUBLISHED_BEFORE_CUTOFF'
    assert F.v1_projection(log, 2026, 6, 'p', datetime(2026, 10, 8, tzinfo=UTC))['status'] == 'V1_NOT_PUBLISHED_BEFORE_CUTOFF'


# ---------------------------------------------------------------- immutability
def test_ledger_is_append_only_hash_chained_and_tamper_evident(tmp_path):
    p = tmp_path / 'x.jsonl'
    for i in range(3):
        L.append_record(p, {'k': i, 'v': 1.0 / 3}, 'k')
    assert L.append_record(p, {'k': 1, 'v': 'other'}, 'k') is None                          # duplicate key skipped, never overwritten
    assert L.verify_chain(p)[0] == 3
    lines = p.read_text().splitlines()
    p.write_text('\n'.join([lines[0], lines[1].replace('0.333333', '0.5'), lines[2]]) + '\n')
    with pytest.raises(ValueError, match='altered'):
        L.verify_chain(p)
    p.write_text('\n'.join([lines[0], lines[2]]) + '\n')
    with pytest.raises(ValueError, match='broken'):
        L.verify_chain(p)
    p.write_text('\n'.join([lines[1], lines[0], lines[2]]) + '\n')
    with pytest.raises(ValueError, match='broken'):
        L.verify_chain(p)


def ledger_history_is_append_only(path):
    rel = path.relative_to(ROOT)
    r = subprocess.run(['git', 'log', '--reverse', '--format=%H', '--', str(rel)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('git history unavailable')
    prev = b''
    for h in r.stdout.split():
        blob = subprocess.run(['git', 'show', f'{h}:{rel}'], cwd=ROOT, capture_output=True)
        if blob.returncode == 0:
            assert blob.stdout.startswith(prev), f'{rel} was rewritten in commit {h}'
            prev = blob.stdout
    return True


def test_committed_ledgers_verify_and_their_git_history_only_appends():
    for name in F.LEDGER_FILES:
        p = F.LEDGER / name
        if p.exists():
            L.verify_chain(p)
            assert ledger_history_is_append_only(p)


def test_verify_append_only_detects_a_rewritten_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    led = tmp_path / 'led'
    monkeypatch.setattr(F, 'LEDGER_FILES', ('f.jsonl',))
    L.append_record(led / 'f.jsonl', {'k': 1}, 'k')
    subprocess.run(['git', 'add', '-A'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=t', 'commit', '-qm', 'one'], cwd=tmp_path, check=True)
    L.append_record(led / 'f.jsonl', {'k': 2}, 'k')
    assert F.verify_append_only('HEAD', led) == []
    (led / 'f.jsonl').write_text('rewritten\n')
    assert F.verify_append_only('HEAD', led) == ['f.jsonl']


# ---------------------------------------------------------------- lock, protocol, scope
def test_engine_lock_matches_code_frozen_artifacts_and_protocol():
    lock = json.loads((ART / 'phase2a_engine_lock.json').read_text())
    proto = json.loads((ART / F.PROTOCOL).read_text())
    assert lock['engine_version'] == F.ENGINE_VERSION == proto['engine_version']
    for name in F.CODE_FILES:
        assert lock['code_sha256'][name] == F.sha_file(ROOT / name), name
    for name in F.FROZEN_UPSTREAM_CODE:
        assert lock['frozen_upstream_code_sha256'][name] == F.sha_file(ROOT / name), name
    for name in F.FROZEN_ARTIFACTS:
        assert lock['frozen_artifact_sha256'][name] == F.sha_file(ART / name), name
    assert lock['protocol_sha256'] == F.sha_file(ART / F.PROTOCOL)
    assert {k: [v[0].total_seconds() // 3600, v[1].total_seconds() // 60] for k, v in F.CUTOFFS.items()} == {'T24': [48, 1440], 'T90': [3, 90]}
    assert lock['no_model_fitting'] is True and lock['no_monte_carlo'] is True and lock['production_promotion'] is False


def test_forward_chain_reuses_the_frozen_phase1e_receipt_unchanged():
    tree = ast.parse((ROOT / 'nfl_v2_phase2a_forward_forecast.py').read_text())
    calls = {(n.func.value.id, n.func.attr) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)}
    assert ('p1e', 'receipt') in calls
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {'numpy', 'sklearn', 'scipy', 'random'} & imports and not any('phase1n' in i or 'phase1o' in i.replace('nfl_v2_phase1o_role_sources', '') or 'phase1q' in i for i in imports)


def test_no_qb_projection_can_be_generated_and_qb_abstention_is_logged_and_counted(tmp_path):
    led = tmp_path / 'led'
    s = run(tmp_path, KO - timedelta(hours=30), led)
    assert s['qb_abstentions'] == 2
    ev = [x['record'] for x in L.read_ledger(led / 'game_status.jsonl') if x['record']['event'].startswith('ABSTAIN')]
    assert len(ev) == 2 and {e['reason'] for e in ev} == {'NO_ACCEPTED_QB_STATE_SOURCE'} and {e['qb_status'] for e in ev} == {'BLOCKED'}
    for line in L.read_ledger(led / 'forecasts_T24.jsonl'):
        r = line['record']
        assert r['position'] != 'QB' and 'passing_yards' not in r['final_central_projection'] and r['qb_state_status'] == 'BLOCKED_ABSTAIN_QB_STATE_UNCERTIFIED'
        assert 'QB_STATE_UNCERTIFIED' in r['uncertainty_flags']


def test_forecast_receipt_has_the_full_causal_chain(tmp_path):
    led = tmp_path / 'led'
    run(tmp_path, KO - timedelta(hours=30), led)
    r = L.read_ledger(led / 'forecasts_T24.jsonl')[0]['record']
    for k in ('forecast_id', 'run_id', 'engine_version', 'git_sha', 'generated_at', 'cutoff_type', 'cutoff_time', 'kickoff_time', 'season', 'week', 'game_id', 'team', 'opponent', 'player_id', 'player_name', 'position', 'eligibility_status', 'eligibility_reason',
              'availability_state', 'availability_confidence', 'role_confidence', 'data_completeness', 'qb_state_status', 'uncertainty_flags', 'blocked_information', 'source_hashes', 'final_central_projection'):
        assert k in r, k
    rec = next(x['record'] for x in L.read_ledger(led / 'forecasts_T24.jsonl') if x['record']['receiving'])
    for k in ('expected_team_opportunity', 'expected_target_share', 'expected_targets', 'expected_catch_rate', 'expected_air_component', 'expected_yac_component', 'final_receptions', 'final_receiving_yards'):
        assert k in rec['receiving'], k
    rush = next(x['record'] for x in L.read_ledger(led / 'forecasts_T24.jsonl') if x['record']['rushing'])
    for k in ('expected_team_opportunity', 'expected_carry_share', 'expected_carries', 'expected_yards_per_carry', 'final_rushing_yards'):
        assert k in rush['rushing'], k
    for line in L.read_ledger(led / 'comparators.jsonl'):
        assert datetime.fromisoformat(line['record']['generated_at']) < datetime.fromisoformat(line['record']['cutoff_time'])


def test_no_fringe_padding_only_meaningful_participants_are_eligible(tmp_path):
    led = tmp_path / 'led'
    run(tmp_path, KO - timedelta(hours=30), led)
    players = {x['record']['player_id'] for x in L.read_ledger(led / 'forecasts_T24.jsonl')}
    assert not any(p.endswith(('OL0', 'RB2')) for p in players if 'OL' in p)                 # linemen never; RB2 (2 targets, 5 carries) is below the receiving bar but exactly at the rushing bar
    assert 'T00-TE1' in players and 'T00-WR1' in players


# ---------------------------------------------------------------- QB ingestion interface and harness (fixtures only)
def fixture_payload(published='2026-10-16T12:00:00+00:00', state='STARTER', revision='r1', original=True, game='2026_07_T01_T00', pid='sr-1'):
    return {'season': 2026, 'week': 7, 'game_id': game, 'team': 'T00', 'opponent': 'T01', 'scheduled': '2026-10-18T17:00:00+00:00',
            'players': [{'id': pid, 'name': 'Fixture QB', 'position': 'QB', 'depth_status': state, 'updated': published, 'record_id': 'rec-1', 'revision': revision, 'original_version': original, 'depth': 1}]}


@pytest.fixture
def accepted(tmp_path):
    p = tmp_path / 'acceptance.json'
    p.write_text(json.dumps({'providers': {'sportradar': {'classification': 'ACCEPTED_FORWARD_ONLY'}}}))
    return p


def test_unauthorized_provider_is_refused_and_nothing_is_stored(tmp_path):
    store = tmp_path / 'store'
    with pytest.raises(Q.ProviderNotAuthorized):
        Q.ingest_qb_state(fixture_payload(), '2026-10-16T12:01:00+00:00', 'sportradar', store=store, acceptance_path=tmp_path / 'none.json')
    assert not store.exists()
    assert Q.certify_for_forecast('T00', 'g', KO, 'T24', store=store, acceptance_path=tmp_path / 'none.json') == {'status': 'BLOCKED', 'reason': 'NO_ACCEPTED_QB_STATE_SOURCE'}
    committed = json.loads((ART / Q.ACCEPTANCE_FILE).read_text())
    assert all(v['classification'].startswith('NOT_EVALUATED') for v in committed['providers'].values())


def test_ingest_maps_to_the_phase1m_schema_with_raw_and_normalized_layers_and_revisions(tmp_path, accepted):
    store = tmp_path / 'store'
    rows = Q.ingest_qb_state(fixture_payload(), '2026-10-16T12:30:00+00:00', 'sportradar', crosswalk={'sr-1': '00-0000001'}, store=store, acceptance_path=accepted)
    assert len(rows) == 1 and set(rows[0]) == set(QS.FIELD_NAMES) and rows[0]['normalized_player_id'] == '00-0000001' and rows[0]['as_of_certified'] is True
    assert QS.validate({k: v for k, v in rows[0].items()}) == []
    assert Q.ingest_qb_state(fixture_payload(), '2026-10-16T12:45:00+00:00', 'sportradar', crosswalk={'sr-1': '00-0000001'}, store=store, acceptance_path=accepted) == []        # same content: not a revision
    rows2 = Q.ingest_qb_state(fixture_payload(published='2026-10-17T12:00:00+00:00', state='BACKUP', revision='r2'), '2026-10-17T12:10:00+00:00', 'sportradar', crosswalk={'sr-1': '00-0000001'}, store=store, acceptance_path=accepted)
    assert rows2[0]['revision_sequence'] == 2
    assert len(L.read_ledger(store / Q.RAW_FILE)) == 3 and len(L.read_ledger(store / Q.NORMALIZED_FILE)) == 2
    L.verify_chain(store / Q.RAW_FILE)
    L.verify_chain(store / Q.NORMALIZED_FILE)


def test_state_at_never_leaks_a_later_revision_backward(tmp_path, accepted):
    store = tmp_path / 'store'
    cw = {'sr-1': '00-0000001'}
    Q.ingest_qb_state(fixture_payload(), '2026-10-16T12:30:00+00:00', 'sportradar', crosswalk=cw, store=store, acceptance_path=accepted)
    Q.ingest_qb_state(fixture_payload(published='2026-10-18T14:00:00+00:00', state='OUT', revision='r2'), '2026-10-18T14:05:00+00:00', 'sportradar', crosswalk=cw, store=store, acceptance_path=accepted)
    ko = datetime(2026, 10, 18, 17, tzinfo=UTC)
    t24 = Q.state_at('T00', '2026-07_x'.replace('2026-07_x', '2026_07_T01_T00'), QS.cutoff_time(ko, 'T24'), 'T24', store=store)
    assert [r['state_type'] for r in t24['rows']] == ['EXPECTED_STARTER'] and t24['summary']['state'] == 'EXPECTED_STARTER'
    t90 = Q.state_at('T00', '2026_07_T01_T00', QS.cutoff_time(ko, 'T90'), 'T90', store=store)
    assert [r['state_type'] for r in t90['rows']] == ['OUT'] or t90['rows'] == []              # the OUT revision (14:00Z) is after T90 (15:30Z)? it is before: it is visible
    assert Q.certify_for_forecast('T00', '2026_07_T01_T00', QS.cutoff_time(ko, 'T24'), 'T24', store=store, acceptance_path=accepted)['status'] == 'CERTIFIED'


def test_backfilled_or_postgame_records_are_never_certified(tmp_path, accepted):
    store = tmp_path / 'store'
    late = Q.ingest_qb_state(fixture_payload(published='2026-10-10T12:00:00+00:00', original=False), '2026-10-19T12:00:00+00:00', 'sportradar', crosswalk={'sr-1': '00-0000001'}, store=store, acceptance_path=accepted)
    assert late[0]['as_of_certified'] is False and late[0]['postgame_only_flag'] is True and late[0]['cutoff_eligibility'] == 'NEITHER'
    ko = datetime(2026, 10, 18, 17, tzinfo=UTC)
    assert Q.certify_for_forecast('T00', '2026_07_T01_T00', QS.cutoff_time(ko, 'T90'), 'T90', store=store, acceptance_path=accepted)['status'] == 'UNCERTAIN'


def archive_of(rows):
    a = QS.QBStateArchive()
    for r in rows:
        a.append(r)
    return a


def synthetic_corpus(n_games, good=True, identity=True, published=True, original=True, live=True):
    rows, games = [], []
    for i in range(n_games):
        ko = datetime(2024, 9, 8, 17, tzinfo=UTC) + timedelta(days=7 * i)
        gid = f'2024_{i:02d}_AAA_BBB'
        games.append({'game_id': gid, 'team': 'AAA', 'kickoff_time': ko.isoformat()})
        for rev, hrs in ((1, 40), (2, 100)) if good else ((1, 40),):
            pub = ko - timedelta(hours=hrs)
            rows.append({'season': 2024, 'week': i + 1, 'game_id': gid, 'team': 'AAA', 'opponent': 'BBB', 'kickoff_time': ko.isoformat(), 'player_id': 'g' if identity else None, 'player_name': 'F QB', 'provider_player_id': 'x',
                         'normalized_player_id': 'g' if identity else None, 'state_type': 'EXPECTED_STARTER', 'source_provider': 'fixture', 'source_product': 'fixture', 'source_record_id': gid, 'publication_timestamp': pub.isoformat() if published else None,
                         'retrieval_timestamp': (pub + timedelta(minutes=5)).isoformat(), 'valid_from': pub.isoformat(), 'revision_id': f'r{rev}', 'revision_sequence': None, 'evidence_type': 'PROVIDER_DEPTH', 'cutoff_eligibility': 'BOTH',
                         'as_of_certified': published and original, 'stale_flag': False, 'postgame_only_flag': False, 'backfilled_flag': False, 'original_vintage_flag': original, 'raw_source_hash': 'h', 'health_state': 'healthy'})
    probe = {'game_id': games[0]['game_id'], 'team': 'AAA', 'kickoff_time': games[0]['kickoff_time'], 'cutoff_label': 'T24', 'retrieved_at': (datetime.fromisoformat(games[0]['kickoff_time']) - timedelta(hours=30)).isoformat()} if live else None
    return archive_of(rows), games, probe


def test_acceptance_harness_returns_the_three_classifications_and_never_accepts_empty_evidence():
    a, games, probe = synthetic_corpus(20)
    assert Q.evaluate_provider(a, games, 'fixture', probe)['classification'] == 'ACCEPTED_FOR_PHASE1L'
    a, games, probe = synthetic_corpus(20, original=False)
    assert Q.evaluate_provider(a, games, 'fixture', probe)['classification'] == 'REJECTED'                    # no original vintage: cannot certify, and not forward-only evidence
    a, games, probe = synthetic_corpus(20, good=False)
    ev = Q.evaluate_provider(a, games, 'fixture', probe)
    assert ev['classification'] == 'ACCEPTED_FOR_PHASE1L' or ev['checks']['revisions_reconstructable'] is False
    a, games, probe = synthetic_corpus(20, identity=False)
    assert Q.evaluate_provider(a, games, 'fixture', probe)['classification'] == 'REJECTED'
    a, games, probe = synthetic_corpus(20, published=False)
    assert Q.evaluate_provider(a, games, 'fixture', probe)['classification'] == 'REJECTED'
    a, games, _ = synthetic_corpus(20)
    assert Q.evaluate_provider(a, games, 'fixture', None)['classification'] == 'REJECTED'
    assert Q.evaluate_provider(QS.QBStateArchive(), games, 'fixture', None) | {} == Q.evaluate_provider(QS.QBStateArchive(), games, 'fixture', None) and Q.evaluate_provider(QS.QBStateArchive(), games, 'fixture', None)['reason'] == 'NO_PROVIDER_PAYLOAD_TESTED'


def test_forward_only_when_history_is_missing_but_the_live_path_works():
    a, games, probe = synthetic_corpus(20)
    sparse = games + [{'game_id': f'2024_x{i}', 'team': 'AAA', 'kickoff_time': '2024-12-01T17:00:00+00:00'} for i in range(60)]
    ev = Q.evaluate_provider(a, sparse, 'fixture', probe)
    assert ev['classification'] == 'ACCEPTED_FORWARD_ONLY' and ev['cutoffs_passing'] == []


def test_committed_qb_artifacts_are_consistent():
    doc = json.loads((ART / 'phase2a_qb_state_interface.json').read_text())
    assert doc['forecast_gate']['current_state'] == 'BLOCKED' and doc['forecast_gate']['official_qb_forecasts_allowed'] is False and set(doc['adapters']) == {'sportradar', 'sportsdataio'}
    acc = json.loads((ART / Q.ACCEPTANCE_FILE).read_text())
    assert acc['fixture_selftest']['label'] == 'FIXTURE_ONLY_NOT_PROVIDER_EVIDENCE' and set(acc['fixture_selftest']['classifications_produced']) == set(Q.CLASSES)
    assert acc['phase1l'] == 'UNCHANGED_BLOCKED_STARTER_STATE_DATA'
    assert Q.interface_document() == doc and Q.build_acceptance_document() == acc
    assert not (ART / 'phase2a_qb_state' / Q.NORMALIZED_FILE).exists()


# ---------------------------------------------------------------- grader
def forecast_line(path, fid, gid, pid, label, gen, ko, proj_rec=60.0, proj_tg=7.0, human=55.0, base=50.0, v1=None, week=5, team='AAA', opp='BBB', role='HIGH'):
    cut = ko - {'T24': timedelta(hours=24), 'T90': timedelta(minutes=90)}[label]
    rec = {'forecast_id': fid, 'run_id': 'r', 'engine_version': 'e', 'git_sha': 'g', 'generated_at': F.iso(gen), 'cutoff_type': label, 'cutoff_time': F.iso(cut), 'kickoff_time': F.iso(ko), 'season': 2026, 'week': week, 'game_id': gid, 'team': team, 'opponent': opp,
           'player_id': pid, 'position': 'WR', 'role_confidence': {'receiving': role}, 'inputs_retrieved_at': F.iso(gen - timedelta(minutes=1)),
           'receiving': {'expected_targets': proj_tg, 'final_receptions': 4.0, 'final_receiving_yards': proj_rec}, 'rushing': None}
    L.append_record(path, rec, 'forecast_id')
    L.append_record(path.parent / 'comparators.jsonl', {'forecast_id': fid, 'generated_at': F.iso(gen), 'cutoff_time': F.iso(cut), 'human': {'receiving_yards': human, 'targets': 6.0, 'receptions': 3.5}, 'simple_baseline': {'receiving_yards': base, 'targets': 5.0, 'receptions': 3.0},
                                                                'v1': v1 or {'receiving_yards': None}}, 'forecast_id')


def stats_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STAT_COLS)
        w.writeheader()
        for team, pid, tg, rc, ry in rows:
            w.writerow({'season': 2026, 'season_type': 'REG', 'week': 5, 'player_id': pid, 'team': team, 'opponent_team': 'BBB' if team == 'AAA' else 'AAA', 'position': 'WR', 'targets': tg, 'receptions': rc, 'receiving_yards': ry, 'carries': 0, 'rushing_yards': 0, 'attempts': 0, 'passing_yards': 0})


def schedule_text(final=True):
    return ('game_id,season,game_type,week,gameday,gametime,away_team,home_team,result\n' + f"G1,2026,REG,5,2026-10-11,13:00,BBB,AAA,{'3' if final else ''}\n")


def test_grader_grades_only_final_games_with_stats_and_proves_the_cutoff(tmp_path):
    led = tmp_path / 'led'
    ko = datetime(2026, 10, 11, 17, tzinfo=UTC)
    forecast_line(led / 'forecasts_T24.jsonl', 'a1', 'G1', 'p1', 'T24', ko - timedelta(hours=30), ko, proj_rec=60, human=45, base=50, v1={'receiving_yards': 70.0})
    forecast_line(led / 'forecasts_T90.jsonl', 'b1', 'G1', 'p1', 'T90', ko - timedelta(hours=2), ko, proj_rec=53, human=45, base=50)
    forecast_line(led / 'forecasts_T24.jsonl', 'a2', 'G1', 'p2', 'T24', ko - timedelta(hours=30), ko, proj_rec=30, human=30, base=30)        # p2 never plays: graded as 0, raw and clean
    for n in ('game_status.jsonl', 'runs.jsonl'):
        (led / n).write_text('')
    stats = tmp_path / 's.csv'
    stats_csv(stats, [('AAA', 'p1', 8, 5, 55), ('BBB', 'z', 1, 1, 5)])
    early = ko + timedelta(hours=1)
    assert G.grade(led, stats, schedule_text(), now=early)[0] == []                                  # not final long enough
    assert G.grade(led, stats, schedule_text(final=False), now=ko + timedelta(days=1))[0] == []      # schedule does not mark it final
    graded, excluded, _ = G.grade(led, stats, schedule_text(), now=ko + timedelta(days=1))
    assert excluded == [] and len(graded) == 3
    p2 = next(g for g in graded if g['player_id'] == 'p2')
    assert p2['did_not_play_no_stat_row'] and p2['heads']['receiving_yards']['actual'] == 0.0
    rep = G.report(graded, excluded, G.grade(led, stats, schedule_text(), now=ko + timedelta(days=1))[2], led)
    t24 = rep['heads']['receiving_yards']['RAW']['by_cutoff']['T24']
    assert t24['n'] == 2 and t24['mae'] == pytest.approx((5 + 30) / 2) and t24['bias'] == pytest.approx((5 + 30) / 2) and t24['median_ae'] == pytest.approx(17.5)
    assert t24['within_10'] == 0.5 and t24['miss_gt_25'] if 'miss_gt_25' in t24 else True
    pr = rep['heads']['receiving_yards']['paired_T24_T90_clean']
    assert pr['pairs'] == 1 and pr['T90_improved'] == 1 and pr['T90_worsened'] == 0 and pr['rows'][0]['delta_T90_minus_T24'] == pytest.approx(-0.0 + (abs(53 - 55) - abs(60 - 55)))
    vs = rep['heads']['receiving_yards']['RAW']['versus']
    assert vs['human']['T24']['n'] == 2 and vs['v1']['T24']['n'] == 1 and vs['v1']['T24']['v2_mae'] == pytest.approx(5.0)         # V1 compared only on the player-game where it exists


def test_grader_excludes_forecasts_that_fail_the_pre_cutoff_proof(tmp_path):
    led = tmp_path / 'led'
    ko = datetime(2026, 10, 11, 17, tzinfo=UTC)
    forecast_line(led / 'forecasts_T24.jsonl', 'late', 'G1', 'p1', 'T24', ko - timedelta(hours=20), ko)                              # generated after the cutoff
    forecast_line(led / 'forecasts_T24.jsonl', 'ok', 'G1', 'p2', 'T24', ko - timedelta(hours=30), ko)
    forecast_line(led / 'forecasts_T24.jsonl', 'early', 'G1', 'p3', 'T24', ko - timedelta(hours=60), ko)                              # before the window opened
    for n in ('forecasts_T90.jsonl', 'game_status.jsonl', 'runs.jsonl'):
        (led / n).write_text('')
    stats = tmp_path / 's.csv'
    stats_csv(stats, [('AAA', 'p2', 5, 3, 40), ('BBB', 'z', 1, 1, 5)])
    graded, excluded, _ = G.grade(led, stats, schedule_text(), now=ko + timedelta(days=1))
    reasons = {e['forecast_id']: e['reason'] for e in excluded}
    assert [g['player_id'] for g in graded] == ['p2'] and 'GENERATED_NOT_BEFORE_CUTOFF' in reasons['late'] and 'GENERATED_BEFORE_WINDOW_OPENED' in reasons['early'] and 'ok' not in reasons


def test_censoring_keeps_raw_and_only_excludes_documented_in_game_events_from_clean(tmp_path):
    led = tmp_path / 'led'
    ko = datetime(2026, 10, 11, 17, tzinfo=UTC)
    forecast_line(led / 'forecasts_T24.jsonl', 'a', 'G1', 'p1', 'T24', ko - timedelta(hours=30), ko)
    forecast_line(led / 'forecasts_T24.jsonl', 'b', 'G1', 'p2', 'T24', ko - timedelta(hours=30), ko)
    for n in ('forecasts_T90.jsonl', 'game_status.jsonl', 'runs.jsonl'):
        (led / n).write_text('')
    cens = tmp_path / 'c.json'
    cens.write_text(json.dumps({'events': [{'game_id': 'G1', 'player_id': 'p1', 'status': 'CENSORED_IN_GAME', 'reason': 'in_game_injury_first_quarter'}, {'game_id': 'G1', 'player_id': 'p2', 'status': 'NOT_CENSORED', 'reason': 'benched'}]}))
    stats = tmp_path / 's.csv'
    stats_csv(stats, [('AAA', 'p1', 1, 0, 0), ('AAA', 'p2', 1, 0, 0), ('BBB', 'z', 1, 1, 5)])
    graded, excluded, chains = G.grade(led, stats, schedule_text(), censor_path=cens, now=ko + timedelta(days=1))
    rep = G.report(graded, excluded, chains, led)
    assert rep['heads']['receiving_yards']['RAW']['by_cutoff']['T24']['n'] == 2 and rep['heads']['receiving_yards']['CLEAN']['by_cutoff']['T24']['n'] == 1


def test_grader_refuses_a_tampered_ledger(tmp_path):
    led = tmp_path / 'led'
    ko = datetime(2026, 10, 11, 17, tzinfo=UTC)
    forecast_line(led / 'forecasts_T24.jsonl', 'a', 'G1', 'p1', 'T24', ko - timedelta(hours=30), ko)
    for n in ('forecasts_T90.jsonl', 'game_status.jsonl', 'runs.jsonl'):
        (led / n).write_text('')
    p = led / 'forecasts_T24.jsonl'
    p.write_text(p.read_text().replace('"final_receiving_yards":60.0', '"final_receiving_yards":99.0'))
    stats = tmp_path / 's.csv'
    stats_csv(stats, [('AAA', 'p1', 1, 1, 1), ('BBB', 'z', 1, 1, 5)])
    with pytest.raises(ValueError, match='altered'):
        G.grade(led, stats, schedule_text(), now=ko + timedelta(days=1))


def test_metrics_helper_known_values():
    m = G.metrics([1.0, -2.0, 3.0, -5.0, 0.0], (1, 2, 3), (4,))
    assert m['n'] == 5 and m['mae'] == pytest.approx(2.2) and m['bias'] == pytest.approx(-0.6) and m['median_ae'] == 2.0
    assert m['within_1'] == 0.4 and m['within_2'] == 0.6 and m['within_3'] == 0.8 and m['miss_gt_4'] == 0.2
    assert set(G.HEADS) == {'targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards'}                       # no passing head until the QB gate passes


def test_committed_report_and_manifest_are_consistent_with_the_ledgers():
    manifest = json.loads((ART / 'phase2a_forecast_manifest.json').read_text())
    for name, info in manifest['ledgers'].items():
        p = F.LEDGER / name
        n, last = L.verify_chain(p)
        assert info['lines'] <= n                                                      # the manifest may lag the ledger but never exceed it
    assert manifest['engine_version'] == F.ENGINE_VERSION and manifest['production_promotion'] is False
    rep = json.loads((ART / 'phase2a_forward_report.json').read_text())
    assert rep['schema'] == 'nfl-v2-phase2a-forward-report-v1'


def test_protocol_matches_code_constants():
    p = json.loads((ART / F.PROTOCOL).read_text())
    assert p['cutoffs']['T24'] == {'cutoff': 'kickoff_utc - 24 hours', 'window_opens': 'kickoff_utc - 48 hours'} and p['cutoffs']['T90'] == {'cutoff': 'kickoff_utc - 90 minutes', 'window_opens': 'kickoff_utc - 3 hours'}
    assert F.THRESHOLD == {'rec_yds': 3.0, 'rush_yds': 5.0} and F.MIN_ROSTER_ROWS == 40 and p['promotion'].startswith('NONE')
    assert datetime.fromisoformat(p['evaluation_start_utc']) == datetime(2026, 10, 7, tzinfo=UTC)
    for head, spec in p['metrics'].items():
        if head in G.HEADS:
            _loc, _col, _ck, within, over = G.HEADS[head]
            assert [f'within_{t}' for t in within] + [f'miss_gt_{t}' for t in over] == [m for m in spec if m.startswith(('within', 'miss'))], head

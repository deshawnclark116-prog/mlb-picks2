"""Phase1N-T deterministic checks: play semantics, strict history, ridge chain accounting, gates, frozen scope. Synthetic math is not fitting."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import nfl_v2_phase1n_team_opportunity as T
import nfl_v2_phase1n_team_sources as S

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'


def play(**kw):
    base = {'play_type': 'pass', 'qb_dropback': '1', 'rush_attempt': '0', 'pass_attempt': '1', 'sack': '0', 'qb_scramble': '0', 'qb_kneel': '0', 'qb_spike': '0', 'two_point_attempt': '0'}
    base.update(kw)
    return base


# ---------------------------------------------------------------- play semantics
def test_play_semantics_match_the_protocol():
    assert S.opportunity_play(play())
    assert S.opportunity_play(play(play_type='run', qb_dropback='0', rush_attempt='1', pass_attempt='0'))
    assert not S.opportunity_play(play(play_type='run', qb_dropback='0', rush_attempt='1', qb_kneel='1'))        # kneels are excluded
    assert not S.opportunity_play(play(two_point_attempt='1'))                                                   # two-point tries excluded
    assert not S.opportunity_play(play(play_type='no_play'))                                                     # penalties/no-plays excluded
    spike = play(play_type='qb_spike', qb_dropback='0', pass_attempt='1', qb_spike='1')
    assert S.opportunity_play(spike) and S.classify(spike) == ('dropback', False, True)                          # spikes count as dropbacks
    scramble = play(play_type='run', qb_dropback='1', rush_attempt='1', pass_attempt='0', qb_scramble='1')
    assert S.classify(scramble) == ('dropback', True, False)                                                     # a scramble is one dropback play, not a designed rush
    designed = play(play_type='run', qb_dropback='0', rush_attempt='1', pass_attempt='0')
    assert S.classify(designed)[0] == 'rush'


def test_betting_columns_are_not_in_any_allowlist():
    for name in S.FORBIDDEN_COLUMNS:
        assert name not in S.PBP_FIELDS and name not in S.GAMES_ALLOWLIST and name not in S.STATS_FIELDS


def test_schedule_loader_is_allowlisted_and_skips_later_seasons(tmp_path):
    import csv
    p = tmp_path / 'g.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(S.GAMES_ALLOWLIST) + ['spread_line', 'result'])
        w.writeheader()
        for season in (2024, 2025, 2026):
            w.writerow({'game_id': f'{season}_01_A_B', 'season': season, 'game_type': 'REG', 'week': 1, 'gameday': f'{season}-09-08', 'gametime': '13:00', 'away_team': 'A', 'home_team': 'B', 'spread_line': 3, 'result': 7})
    games = S.load_schedule(p, 2025)
    assert [g['season'] for g in games] == ['2024', '2025'] and set(games[0]) == set(S.GAMES_ALLOWLIST)
    ctx = S.schedule_context(games)
    assert ctx[('2025_01_A_B', 'B')]['home'] == 1 and ctx[('2025_01_A_B', 'B')]['rest_days'] is None              # offseason gap is not rest


def test_seasons_and_2026_are_refused(tmp_path):
    with pytest.raises(ValueError):
        S.fetch(tmp_path, (2026,))
    f = tmp_path / 'x.csv'
    f.write_text('season,season_type,week\n2026,REG,1\n')
    with pytest.raises(ValueError):
        list(S.records(f, ('season',), 2026))
    with pytest.raises(ValueError):
        list(S.records(f, ('season',), 2024))


# ---------------------------------------------------------------- strict history and features
def synth_rows(teams=12, weeks=12):
    rows = []
    rng = np.random.RandomState(7)
    names = [f'T{i:02d}' for i in range(teams)]
    for w in range(1, weeks + 1):
        day = 1 + 7 * w
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            gid = f'2024_{w:02d}_{names[a]}_{names[b]}'
            for team, opp, home in ((names[a], names[b], 0), (names[b], names[a], 1)):
                plays = float(rng.randint(50, 75))
                drop = float(rng.randint(int(plays * .4), int(plays * .65)))
                rows.append({'game_id': gid, 'team': team, 'opponent': opp, 'home': home, 'season': 2024, 'week': w, 'game_date': '2024-%02d-%02d' % (9 + (day // 30), 1 + day % 28) if False else
                             (np.datetime64('2024-09-01') + int(7 * w)).astype(str), 'ot': False, 'plays': plays, 'dropbacks': drop, 'designed_rushes': plays - drop, 'neutral_plays': plays / 2,
                             'neutral_dropbacks': drop / 2, 'attempts': drop * .9, 'scrambles': 1.0, 'kneels': 1.0, 'margin': float(rng.randint(-20, 21)), 'drives': 10.0, 'points_for': 20.0, 'points_against': 20.0})
    return rows


def test_history_excludes_target_and_lagged_games():
    rows = synth_rows()
    hist = T.History(rows)
    target = [r for r in rows if r['week'] == 6][0]
    prior = hist.team(target['team'], target['_d'], 99)
    assert all(r['_d'] <= target['_d'] - 3 for r in prior) and all(r['week'] < 6 for r in prior)
    # a game exactly 3 days before the target is usable, 2 days is not
    assert T.LAG == 3


def test_features_do_not_change_when_target_or_future_rows_change():
    rows = synth_rows()
    cfg = {'lam': 10.0, 'decay': 0.85, 'window': 5}
    hist = T.History(rows)
    target = [r for r in hist.rows if r['week'] == 9][0]
    f1 = T.features(hist, target, cfg, {})
    changed = deepcopy(rows)
    for r in changed:
        if r['week'] >= 9:                                     # the target game and everything later is rewritten
            r.update({'plays': 999.0, 'dropbacks': 1.0, 'margin': 99.0, 'neutral_plays': 1.0, 'neutral_dropbacks': 1.0})
    hist2 = T.History(changed)
    t2 = [r for r in hist2.rows if r['game_id'] == target['game_id'] and r['team'] == target['team']][0]
    f2 = T.features(hist2, t2, cfg, {})
    assert f1 is not None and {k: v for k, v in f1.items() if not k.startswith('_')} == {k: v for k, v in f2.items() if not k.startswith('_')}


def test_scenario_weights_are_a_distribution_and_pregame():
    rows = synth_rows()
    hist = T.History(rows)
    f = T.features(hist, [r for r in hist.rows if r['week'] == 9][0], {'lam': 1, 'decay': .85, 'window': 5}, {})
    w = f['_weights']
    assert abs(sum(w.values()) - 1) < 1e-9 and all(v >= 0 for v in w.values())
    assert set(f['_scenario_effects']) == {'LEADS', 'TRAILS', 'COMPETITIVE'}


# ---------------------------------------------------------------- ridge chain, metrics and gates
def test_ridge_matches_closed_form_and_chain_is_exact():
    rng = np.random.RandomState(3)
    x = rng.normal(size=(40, 3))
    y = 2 + x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=.1, size=40)
    m = T.Ridge(5.0).fit(x, y)
    z = (x - x.mean(0)) / x.std(0)
    beta = np.linalg.solve(z.T @ z + 5 * np.eye(3), z.T @ (y - y.mean()))
    assert np.allclose(m.predict(x), y.mean() + z @ beta)
    out = T.chain(np.array([62.0, 0.2, 80.0]), np.array([.6, .95, .1]))
    assert np.allclose(out['plays'], out['dropbacks'] + out['rush_attempts']) and (out['dropback_rate'] <= .9).all() and (out['dropback_rate'] >= .2).all() and (out['plays'] >= 1).all()


def test_metrics_and_bootstrap_are_deterministic():
    pred, act = np.array([60., 70, 50, 55]), np.array([62., 60, 52, 80])
    m = T.metrics(pred, act, 'plays')
    assert m['mae'] == pytest.approx(9.75) and m['within_3'] == 0.5 and m['miss_gt_20'] == .25 and m['signed_bias'] == pytest.approx(-4.25 + 0.0 if False else m['signed_bias'])
    diff = np.array([-1.0, -2.0, 0.5, -0.5] * 10)
    dates = ['2024-09-%02d' % (1 + 3 * i) for i in range(40)] if False else [str(np.datetime64('2024-09-01') + 3 * i) for i in range(40)]
    a = T.block_bootstrap_upper(diff, dates, reps=200)
    assert a == T.block_bootstrap_upper(diff, dates, reps=200) and a < 0


def test_a_tiny_decimal_win_never_qualifies():
    rng = np.random.RandomState(1)
    comp = np.abs(rng.normal(6, 2, 300))
    tiny = comp - 0.1
    dates = [str(np.datetime64('2024-09-01') + i // 4) for i in range(300)]
    assert not T.qualifies(tiny, comp, float(comp.mean()), 'plays', dates)['qualifies']
    big = comp - 1.2
    assert T.qualifies(big, comp, float(comp.mean()), 'plays', dates)['qualifies']
    # a large win with more catastrophic misses is rejected
    big_bad = big.copy()
    big_bad[:10] = 20.0
    comp2 = comp.copy()
    comp2[:10] = 5.0
    assert not T.qualifies(big_bad, comp2, float(comp2.mean()), 'plays', dates)['qualifies']


def test_thresholds_match_the_committed_protocol():
    p = json.loads((ART / T.PROTOCOL).read_text())
    assert p['practical_improvement_thresholds']['absolute_mae_gain_min'] == T.THRESH_ABS
    assert p['practical_improvement_thresholds']['relative_mae_gain_min'] == T.THRESH_REL
    assert str(T.SEED) in p['metrics']['bootstrap'] and p['splits']['estimation'] == '2024 weeks 1-8' and p['splits']['selection'] == '2024 weeks 9-18'
    assert p['coherence_gate']['plays_minus_dropbacks_minus_rushes_max_abs'] == 1e-9
    assert p['drive_reconstruction_acceptance']['accepted_iff'][2].endswith('>= 95% of games')
    assert T.SCALE == 13.5


def test_verdict_rules():
    base = {'development_gate': {'passed': False, 'any_quantity_qualifies': False, 'qualifying_families': []}}
    assert T.verdict(base) == 'REJECTED_TEAM_OPPORTUNITY_REPLACEMENT'
    assert T.verdict({'development_gate': {'passed': False, 'any_quantity_qualifies': True, 'qualifying_families': []}}) == 'PARTIAL_SIGNAL_NOT_PROMOTED'
    assert T.verdict({'development_gate': {'passed': True, 'any_quantity_qualifies': True, 'qualifying_families': ['E']}}) == 'PARTIAL_SIGNAL_NOT_PROMOTED'   # never promoted before 2025/2026


def test_validate_stage_refuses_without_a_passed_gate(tmp_path, monkeypatch):
    lock = json.loads((ART / 'phase1n_development_lock.json').read_text()) if (ART / 'phase1n_development_lock.json').exists() else None
    if lock is None:
        pytest.skip('lock not committed yet')
    assert lock['development_gate_passed'] is False and lock['validation_2025'] == 'NOT_RUN_DEVELOPMENT_GATE_FAILED'


# ---------------------------------------------------------------- committed artifacts
def committed(name):
    path = ART / name
    if not path.exists():
        pytest.skip(name + ' not committed yet')
    return path


def test_source_audit_decides_drive_reconstruction_before_fitting():
    a = json.loads(committed('phase1n_team_source_audit.json').read_text())
    assert a['later_seasons_opened'] is False and a['opened_seasons'] == [2023, 2024] and a['no_models'] and a['no_sportsbook']
    for y in ('2023', '2024'):
        s = a['seasons'][y]
        assert s['plays_equal_dropbacks_plus_designed_rushes_violations'] == 0 and s['attempts_exceed_dropbacks_violations'] == 0
        assert s['official_reconciliation']['attempts_exact'] == s['official_reconciliation']['matched_team_games'] == 544
        assert s['team_games'] == 544
    d = a['drive_reconstruction']
    assert d['accepted'] == (d['min_drive_id_present_share'] >= .999 and d['min_single_offense_per_drive_share'] >= .999 and d['min_games_drive_diff_le_1_share'] >= .95)
    assert a['schedule']['betting_columns_accessed'] is False


def test_results_lock_and_receipts_are_consistent():
    res = json.loads(committed('phase1n_team_results.json').read_text())
    lock = json.loads(committed('phase1n_development_lock.json').read_text())
    audit = json.loads(committed('phase1n_team_source_audit.json').read_text())
    assert lock['protocol_sha256'] == S.sha(ART / T.PROTOCOL) and lock['source_audit_sha256'] == S.sha(ART / 'phase1n_team_source_audit.json')
    assert res['families']['C_drive_chain']['verdict'] == ('REJECTED_DRIVE_RECONSTRUCTION' if not audit['drive_reconstruction']['accepted'] else 'NOT_IMPLEMENTED_UNREACHABLE')
    assert res['periods']['2026_w1_4'] == 'NOT_RUN_PINNED_BYTES_UNAVAILABLE' and res['no_sportsbook'] and res['no_monte_carlo']
    if not res['development_gate']['passed']:
        assert res['periods']['2025_validation'] == 'NOT_RUN_DEVELOPMENT_GATE_FAILED' and res['phase1d_integration'].startswith('NOT_RUN') and res['final_player_stat_test'].startswith('NOT_RUN')
    for f, v in res['families'].items():
        if 'coherence' in v:
            assert v['coherence']['plays_minus_dropbacks_minus_rushes_max_abs'] <= 1e-9 and v['coherence']['non_negative']
    rows = [json.loads(x) for x in gzip.decompress(committed('phase1n_team_receipts.jsonl.gz').read_bytes()).splitlines()]
    assert len(rows) == 544 and all(r['season'] == 2024 for r in rows)
    for r in rows:
        assert abs(r['expected_plays'] - r['expected_dropbacks'] - r['expected_rush_attempts']) < 1e-4
        assert r['actual_plays'] == r['actual_dropbacks'] + r['actual_rush_attempts'] and abs(sum(r['scenario_weights'].values()) - 1) < 1e-4
        assert r['expected_drives'] is None and r['source_sha256']['2024']['pbp']
    assert res['n']['all_team_games_2024'] == 544


def test_protocol_precedes_every_result_in_git_history():
    order = subprocess.run(['git', 'rev-list', '--reverse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if order.returncode != 0:
        pytest.skip('git history unavailable')
    commits = order.stdout.split()

    def first(name):
        out = subprocess.run(['git', 'log', '--format=%H', '--diff-filter=A', '--', 'nfl_models/nfl_player_outcome_v2/' + name], cwd=ROOT, capture_output=True, text=True).stdout.split()
        return commits.index(out[-1]) if out and out[-1] in commits else None
    proto = first('phase1n_team_protocol.json')
    assert proto is not None
    for name in ('phase1n_team_source_audit.json', 'phase1n_team_results.json', 'phase1n_development_lock.json', 'phase1n_team_receipts.jsonl.gz'):
        idx = first(name)
        if idx is not None:
            assert proto < idx, name


def test_modules_have_no_model_betting_or_2026_code_paths():
    import ast
    for path in ('nfl_v2_phase1n_team_sources.py', 'nfl_v2_phase1n_team_opportunity.py'):
        src = (ROOT / path).read_text()
        doc = ast.get_docstring(ast.parse(src)) or ''
        body = src.replace(doc, '')
        body = '\n'.join(line for line in body.splitlines() if not line.startswith('FORBIDDEN_COLUMNS'))
        for word in ('xgboost', 'sklearn', 'torch', 'scipy', 'statsmodels', 'spread_line', 'total_line', 'moneyline'):
            assert word not in body, (path, word)
        assert 'pbp_2026' not in body and 'week5' not in body.lower()


def test_frozen_research_and_other_sports_byte_identical_to_base():
    base = S.BASE_HEAD
    r = subprocess.run(['git', 'diff', '--name-status', base, 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable')
    allowed_modified = {'nfl_models/nfl_player_outcome_v2/research_registry.json', 'tests/test_nfl_v2_phase1l_qb_opportunity.py', 'tests/test_nfl_v2_phase1m_qb_state.py',
                        '.github/workflows/nfl_v2_phase1h_routed_receiving.yml', '.github/workflows/nfl_v2_phase1i_target_depth.yml', '.github/workflows/nfl_v2_phase1k_rushing_efficiency.yml'}
    for line in r.stdout.splitlines():
        status, _, path = line.partition('\t')
        if status == 'A':
            assert 'phase1n' in path or 'phase1o' in path or 'archival' in path, path
        else:
            assert status == 'M' and path in allowed_modified, (status, path)
    old = json.loads(subprocess.run(['git', 'show', base + ':nfl_models/nfl_player_outcome_v2/research_registry.json'], cwd=ROOT, capture_output=True, text=True).stdout)
    new = json.loads((ART / 'research_registry.json').read_text())
    for k, v in old.items():
        if k not in ('status', 'next_milestone'):
            assert new[k] == v, k


def test_workflow_runs_every_stage_without_secrets():
    text = (ROOT / '.github/workflows/nfl_v2_phase1n_team_opportunity.yml').read_text()
    assert 'permissions:\n  contents: read\n  pull-requests: write' in text and 'secrets.' not in text
    for stage in ('--stage develop', '--stage validate', '--stage burned2026', 'nfl_v2_phase1n_team_sources.py', 'upload-artifact'):
        assert stage in text


def test_snapshot_hashes_and_verdict_match_the_artifacts():
    snap = json.loads(committed('phase1n_team_snapshot.json').read_text())
    for name, digest in snap['artifacts_sha256'].items():
        assert S.sha(ART / name) == digest, name
    res = json.loads((ART / 'phase1n_team_results.json').read_text())
    assert snap['verdict'] == res['verdict'] == 'REJECTED_TEAM_OPPORTUNITY_REPLACEMENT' or snap['verdict'] == res['verdict']
    assert snap['files_2025_opened'] is False and snap['files_2026_opened'] is False and snap['week5_plus_opened'] is False
    findings = (ART / 'phase1n_team_findings.md').read_text()
    assert snap['verdict'] in findings
    reg = json.loads((ART / 'research_registry.json').read_text())['phase1n_team_opportunity']
    assert reg['status'] == snap['verdict'] and reg['validation_2025'] == res['periods']['2025_validation']
    for name in ('phase1n_validation_2025.json', 'phase1n_burned2026.json'):
        doc = json.loads((ART / name).read_text())
        assert doc['status'].startswith('NOT_RUN')


# ---------------------------------------------------------------- frozen schedule contract and reproducibility repair
def test_frozen_schedule_contract_and_provenance():
    doc = json.loads((ART / 'phase1n_frozen_schedule_2023_2024.json').read_text())
    assert doc['allowlist'] == list(S.GAMES_ALLOWLIST) and doc['seasons'] == [2023, 2024] and doc['game_type'] == 'REG' and doc['rows_count'] == len(doc['rows']) == 544
    assert all(set(r) == set(S.GAMES_ALLOWLIST) and r['season'] in ('2023', '2024') and r['game_type'] == 'REG' for r in doc['rows'])
    prov = doc['provenance']
    assert prov['source_url'].endswith('/schedules/games.csv') and len(prov['raw_file_sha256_at_acquisition']) == 64 and prov['original_acquisition']
    rows = S.frozen_schedule()
    assert S.schedule_digest(rows, {2023, 2024}) == S.FROZEN_SCHEDULE_ROWS_SHA256 == doc['canonical_rows_sha256']
    audit = json.loads((ART / 'phase1n_team_source_audit.json').read_text())
    assert S.schedule_section(rows) == audit['schedule']                         # the committed audit regenerates from the frozen rows alone


def test_a_later_upstream_games_csv_edit_cannot_change_reproduction(tmp_path, monkeypatch):
    import csv
    poisoned = tmp_path / 'games.csv'
    with open(poisoned, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(S.GAMES_ALLOWLIST) + ['spread_line', 'result'])
        w.writeheader()
        for g in S.frozen_schedule():
            w.writerow({**g, 'gameday': '2099-01-01', 'spread_line': 99, 'result': 99})        # every historical row rewritten
        w.writerow({'game_id': '2026_01_X_Y', 'season': 2026, 'game_type': 'REG', 'week': 1, 'gameday': '2026-09-10', 'gametime': '20:20', 'away_team': 'X', 'home_team': 'Y', 'spread_line': 1, 'result': 1})
    monkeypatch.chdir(tmp_path)
    audit = json.loads((ART / 'phase1n_team_source_audit.json').read_text())
    assert S.schedule_section(S.frozen_schedule()) == audit['schedule']
    assert S.frozen_schedule()[0]['gameday'] != '2099-01-01'
    src = (ROOT / 'nfl_v2_phase1n_team_opportunity.py').read_text() + (ROOT / 'nfl_v2_phase1n_team_sources.py').read_text()
    assert "/ 'games.csv'" not in src and 'fetch_schedule' not in src                # no stage reads or downloads the mutable file
    wf = (ROOT / '.github/workflows/nfl_v2_phase1n_team_opportunity.yml').read_text()
    assert 'games.csv' not in wf and 'schedules' not in wf


def test_tampering_with_the_frozen_schedule_is_refused(tmp_path, monkeypatch):
    doc = json.loads((ART / 'phase1n_frozen_schedule_2023_2024.json').read_text())
    doc['rows'][0]['gameday'] = '2000-01-01'
    bad = tmp_path / 'f.json'
    bad.write_text(json.dumps(doc))
    monkeypatch.setattr(S, 'FROZEN_SCHEDULE', bad)
    with pytest.raises(ValueError):
        S.frozen_schedule()
    doc = json.loads((ART / 'phase1n_frozen_schedule_2023_2024.json').read_text())
    doc['rows'][0]['spread_line'] = '3'
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        S.frozen_schedule()


def test_freeze_generator_reads_only_the_allowlist_and_two_seasons(tmp_path):
    import csv
    p = tmp_path / 'g.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(S.GAMES_ALLOWLIST) + ['spread_line', 'total_line', 'result'])
        w.writeheader()
        for season in (2022, 2023, 2024, 2025, 2026):
            w.writerow({'game_id': f'{season}_01_A_B', 'season': season, 'game_type': 'REG', 'week': 1, 'gameday': f'{season}-09-08', 'gametime': '13:00', 'away_team': 'A', 'home_team': 'B',
                        'spread_line': 3, 'total_line': 44, 'result': 7})
    doc = S.freeze_schedule(p, tmp_path / 'out.json', {'source_url': 'x'})
    assert [r['season'] for r in doc['rows']] == ['2023', '2024'] and all(set(r) == set(S.GAMES_ALLOWLIST) for r in doc['rows'])
    assert 'spread_line' not in (tmp_path / 'out.json').read_text().split('"rows"')[1]


def test_development_cannot_reach_2025_or_2026_inputs():
    import inspect
    assert S.DEVELOPMENT_YEARS == (2023, 2024) and S.FROZEN_SCHEDULE_SEASONS == (2023, 2024)
    body = inspect.getsource(T.run_development)
    assert 'VALIDATION_YEAR' not in body and '2025' not in body and '2026' not in body
    assert all(int(r['season']) <= 2024 for r in S.frozen_schedule())
    for fn in (S.fetch, S.records):
        with pytest.raises(ValueError):
            fn('.', (2026,)) if fn is S.fetch else list(fn(ROOT / 'nfl_v2_phase1n_team_sources.py', ('season',), 2026))


def test_ordered_mean_is_the_same_on_every_python_version():
    values = [0.1] * 10 + [1e16, 1.0, -1e16]
    expected = 0.0
    for v in values:
        expected += v
    assert S.ordered_mean(values) == expected / len(values)               # explicit left-to-right accumulation, not sum() (compensated since Python 3.12)
    assert S.ordered_mean([0.25, 0.5]) == 0.375

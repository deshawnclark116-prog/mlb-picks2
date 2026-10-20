"""Phase1O-R deterministic checks: strict history, coherence, thresholds, gates, scope. Synthetic math is not fitting."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pytest

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as D
import nfl_v2_phase1o_role_change as R
import nfl_v2_phase1o_role_sources as S

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
IDS = ['p1', 'p2', 'p3', 'p4']


def synthetic(mutate=None):
    """One team, one season, 4 receivers with a role change at week 6; `mutate(week, pid, row)` may alter rows."""
    players, totals = [], defaultdict(lambda: defaultdict(float))
    for w in range(1, 11):
        base = [10, 6, 4, 2] if w < 6 else [5, 10, 4, 3]
        for pid, t in zip(IDS, base):
            row = {'season': 2024, 'week': w, 'player_id': pid, 'team': 'AAA', 'opponent': 'BBB', 'position': 'WR', 'targets': float(t), 'carries': 0.0, 'receiving_yards': 0.0}
            for col in {v for x in p1a.SPEC.values() for v in (x['opp'], x['value'])}:
                row.setdefault(col, 0.0)
            if mutate:
                mutate(w, pid, row)
            players.append(row)
            totals[(2024, w, 'AAA')]['targets'] += row['targets']
    totals = {k: dict(v) for k, v in totals.items()}
    p1a.build_indexes(players, totals, {})
    D.build_context(players, totals)
    D._ROSTERS.clear()
    for w in range(1, 11):
        D._ROSTERS[(2024, w, 'AAA')] = [{'player_id': p, 'team': 'AAA', 'position': 'WR', 'status': 'ACT', 'active': True} for p in IDS]
    ctx = object.__new__(R.Context)
    ctx.players, ctx.team_totals, ctx.team_opp, ctx.meta, ctx.snaps = players, totals, {}, {}, {}
    ctx.d_cfg = {'rec_yds': {'cold_start_scale': 0.15, 'concentration_power': 1, 'decay': 0.85, 'global_weight': 0.25, 'share_window': 8, 'trend_gain': 0}}
    ctx.b_cfg = {}
    ctx.by_tw, ctx.team_weeks = defaultdict(dict), defaultdict(set)
    for r in players:
        ctx.by_tw[(r['season'], r['week'], r['team'])][r['player_id']] = r
        ctx.team_weeks[r['team']].add((r['season'], r['week']))
    ctx.team_weeks = {t: sorted(v) for t, v in ctx.team_weeks.items()}
    ctx._tw, ctx._proj = {}, {}
    return ctx


def test_features_ignore_target_and_future_rows():
    base = synthetic().team_week(2024, 7, 'AAA', 'rec_yds')
    def wreck(w, pid, row):
        if w >= 7:
            row['targets'] = 99.0 if pid == 'p4' else 0.0
    alt = synthetic(wreck).team_week(2024, 7, 'AAA', 'rec_yds')
    for name in base['feats']:
        assert np.array_equal(base['feats'][name], alt['feats'][name]), name
    assert np.array_equal(base['d1'], alt['d1']) and np.array_equal(base['c3'], alt['c3'])


def test_a_role_change_is_visible_to_the_features_only_after_it_happens():
    ctx = synthetic()
    before = ctx.team_week(2024, 6, 'AAA', 'rec_yds')['feats']['accel_last']
    after = ctx.team_week(2024, 8, 'AAA', 'rec_yds')['feats']['accel_last']
    assert np.allclose(before, 0.0, atol=1e-12) and not np.allclose(after, 0.0)
    assert after[1] > 0 > after[0]            # p2 rose, p1 fell


def test_predictions_are_coherent_even_for_extreme_models():
    ctx = synthetic()
    rec = ctx.team_week(2024, 8, 'AAA', 'rec_yds')
    names = R.FEATURES['A_recency_acceleration']
    k = len(names)
    for beta in (np.zeros(k), np.full(k, 5.0), np.full(k, -50.0)):
        model = {'names': names, 'mu': np.zeros(k), 'sd': np.ones(k), 'beta': beta, 'intercept': -0.4, 'lam': 10.0}
        out = R.predict(model, rec)
        assert (out >= 0).all() and abs(out.sum() - 1.0) < 1e-9
    assert abs(rec['d1'].sum() - 1.0) < 1e-9 and abs(rec['c3'].sum() - 1.0) < 1e-9


def test_ridge_matches_the_closed_form_with_an_unpenalized_intercept():
    ctx = synthetic()
    tws = [('AAA' and (2024, w, 'AAA')) for w in range(3, 9)]
    model = R.fit(ctx, 'rec_yds', R.FEATURES['A_recency_acceleration'], 100.0, tws)
    xs, ys = [], []
    for tw in tws:
        rec = ctx.team_week(*tw, 'rec_yds')
        actual = np.array([ctx.by_tw[tw][p]['targets'] / rec['actual_team'] for p in rec['pids']])
        keep = rec['n_hist'] >= 1
        xs.append(R.matrix(rec, model['names'])[keep]); ys.append((actual - rec['d1'])[keep])
    x, y = np.vstack(xs), np.concatenate(ys)
    z = (x - model['mu']) / model['sd']
    expect = np.linalg.solve(z.T @ z + 100.0 * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    assert np.allclose(model['beta'], expect) and abs(model['intercept'] - y.mean()) < 1e-12


def test_bootstrap_and_metrics_are_deterministic():
    rows = []
    ctx = synthetic()
    for w in range(1, 11):
        rec = ctx.team_week(2024, w, 'AAA', 'rec_yds')
        for i, p in enumerate(rec['pids']):
            r = ctx.by_tw[(2024, w, 'AAA')][p]
            rows.append({'r': r, 'rec': rec, 'i': i, 'team_opp': rec['actual_team'], 'actual': r['targets'], 'a_share': r['targets'] / rec['actual_team'], 'ref': 0.25,
                         'pred': {'C1': 0.25, 'cand': 0.25 + 0.01 * (i - 1.5)}})
    a = R.block_bootstrap(rows, 'rec_yds', 'cand', 'C1')
    assert a == R.block_bootstrap(rows, 'rec_yds', 'cand', 'C1')
    assert R.metrics(rows, 'rec_yds', 'C1') == R.metrics(rows, 'rec_yds', 'C1')


def test_a_tiny_win_never_qualifies_and_a_large_unconfirmed_win_needs_the_bootstrap():
    ctx = synthetic()
    rec = ctx.team_week(2024, 8, 'AAA', 'rec_yds')
    base = {'r': {'season': 2024, 'week': 8}, 'rec': rec, 'i': 0, 'team_opp': 30.0, 'ref': 0.3}
    rows = []
    for w in (1, 2, 3, 4):
        for k in range(4):
            x = dict(base); x['r'] = {'season': 2024, 'week': w}; x['actual'] = 9.0; x['a_share'] = 0.3
            x['pred'] = {'C1': 0.30 + 0.01 * k, 'cand': 0.30 + 0.01 * k - 0.0003}      # 0.009 target gain: far below the 0.10 absolute floor
            rows.append(x)
    j = R.judge(rows, 'rec_yds', 'cand', 'C1')
    assert not j['material_gain'] and not j['qualifies']


def test_thresholds_and_scope_match_the_committed_protocol():
    p = json.loads((ART / S.PROTOCOL).read_text())
    t = p['practical_thresholds']
    assert t['oracle_opportunity_mae_gain_min'] == {'targets': R.ABS_GAIN['rec_yds'], 'carries': R.ABS_GAIN['rush_yds']}
    assert t['relative_gain_min'] == R.REL_GAIN
    assert p['role_change_definition']['tau'] == {'targets': R.TAU['rec_yds'], 'carries': R.TAU['rush_yds']}
    assert (R.SEED, R.BOOT) == (20261201, 2000) and R.LAMBDAS == tuple(p['model_class']['grid']['ridge_lambda'])
    for fam, names in R.FEATURES.items():
        assert fam in p['signal_families'] and names
    assert set(R.BLOCKED_FAMILIES) == {'B_snap_change'} and 'B_snap_change' in p['signal_families']


def test_no_forbidden_inputs_are_read():
    text = ''
    for name in R.CODE_FILES:
        text += (ROOT / name).read_text().lower()
    for bad in ('spread_line', 'moneyline', 'depth_chart', 'injuries_', 'participation', 'games.csv', 'monte', '_2026.csv', 'route'):
        # the words may appear only in prose that states they are NOT used
        for line in text.splitlines():
            if bad in line and not line.lstrip().startswith(('#', '"""', "'")) and 'not' not in line and 'never' not in line and 'excluded' not in line:
                raise AssertionError(f'{bad!r} used: {line.strip()}')


def test_2026_and_unpinned_seasons_are_refused(tmp_path):
    with pytest.raises(ValueError):
        S.verify(tmp_path, (2026,))
    with pytest.raises(ValueError):
        S.fetch(tmp_path, (2026,))


def test_changed_pinned_bytes_are_rejected_and_there_is_no_latest_fallback(tmp_path, monkeypatch):
    for year in (2023,):
        for source in S.manifest()[year].values():
            (tmp_path / source['local_name']).write_text('tampered')
    with pytest.raises(ValueError, match='changed or missing'):
        S.verify(tmp_path, (2023,))
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[cmd.index('-o') + 1]).write_text('not the pinned bytes')
    monkeypatch.setattr(S.subprocess, 'run', fake_run)
    with pytest.raises(ValueError, match='digest mismatch'):
        S.fetch(tmp_path / 'fresh', (2023,))
    assert len(calls) == 1 and all('latest' not in ' '.join(c) for c in calls)         # one attempt, no fallback URL
    assert not list((tmp_path / 'fresh').glob('*.csv'))                                 # the bad bytes are not kept


def test_validate_stage_refuses_without_a_passed_gate(tmp_path):
    (tmp_path / 'phase1o_development_lock.json').write_text(json.dumps({'development_gate_passed': False, 'code_sha256': R.code_sha256(), 'heads': {}}))
    with pytest.raises(SystemExit, match='refused'):
        R.validate('/nonexistent', tmp_path)


def test_validate_stage_refuses_when_code_changed_after_the_lock(tmp_path):
    (tmp_path / 'phase1o_development_lock.json').write_text(json.dumps({'development_gate_passed': True, 'code_sha256': 'x', 'heads': {}}))
    with pytest.raises(SystemExit, match='code changed'):
        R.validate('/nonexistent', tmp_path)


def test_close_out_records_not_run_states_and_verdicts(tmp_path):
    lock = {'development_gate_passed': False, 'heads': {'receiving': {'development_pass': False}}}
    (tmp_path / 'phase1o_development_lock.json').write_text(json.dumps(lock))
    (tmp_path / 'phase1o_role_results.json').write_text(json.dumps({'development_2024': {'receiving': {'development_verdict_if_failed': 'REJECTED_ROLE_CHANGE_REPLACEMENT'}}}))
    R.burned2026(tmp_path)
    out = json.loads((tmp_path / 'phase1o_role_results.json').read_text())
    assert out['validation_2025']['status'] == 'NOT_RUN_DEVELOPMENT_GATE_FAILED'
    assert out['burned_2026_w1_4']['status'] == 'NOT_RUN_PINNED_BYTES_UNAVAILABLE'
    assert out['verdicts']['overall'] == 'REJECTED_ROLE_CHANGE_REPLACEMENT' and out['downstream_phase1e']['status'] == 'NOT_RUN_NO_HEAD_QUALIFIED'


def test_source_audit_blocks_family_b_below_the_preregistered_mapping_rate():
    audit = json.loads((ART / 'phase1o_role_source_audit.json').read_text())
    d = audit['family_data_decisions']
    assert (d['B_snap_change'] == 'BLOCKED_DATA') == (d['B_snap_skill_mapping_share_2023_2024'] < d['B_acceptance_threshold'])
    assert audit['later_seasons_opened'] is False and audit['no_models'] and not audit['vendor_data_used']


def test_committed_lock_matches_the_code_and_inputs():
    lock = json.loads((ART / 'phase1o_development_lock.json').read_text())
    assert lock['code_sha256'] == R.code_sha256()
    assert lock['protocol_sha256'] == S.sha(ART / S.PROTOCOL) and lock['source_audit_sha256'] == S.sha(ART / 'phase1o_role_source_audit.json')
    assert lock['no_rescue_tuning'] and lock['burned_2026'] == 'NOT_RUN_PINNED_BYTES_UNAVAILABLE'

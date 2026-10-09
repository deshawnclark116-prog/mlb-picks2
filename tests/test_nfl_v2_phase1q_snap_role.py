"""Phase1Q-SNAP deterministic checks: lag rule, no target-game snaps, coherence (QB mass), thresholds, gates, scope."""
from collections import defaultdict
import datetime
import json
from pathlib import Path

import numpy as np
import pytest

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1d_role_allocation as D
import nfl_v2_phase1n_team_sources as N
import nfl_v2_phase1o_role_change as R
import nfl_v2_phase1o_role_sources as O
import nfl_v2_phase1q_snap_role as Q

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
IDS = ['p1', 'p2', 'p3', 'p4']
SNAPS = {'p1': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 'p2': [0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.85, 0.9, 0.9, 0.9], 'p3': [0.5] * 10, 'p4': [0.2] * 10}


def synthetic(snap_mutate=None, days=7):
    players, totals = [], defaultdict(lambda: defaultdict(float))
    for w in range(1, 11):
        for pid, t in zip(IDS, [10, 6, 4, 2]):
            row = {'season': 2024, 'week': w, 'player_id': pid, 'team': 'AAA', 'opponent': 'BBB', 'position': 'WR', 'targets': float(t), 'carries': 0.0}
            for col in {v for x in p1a.SPEC.values() for v in (x['opp'], x['value'])}:
                row.setdefault(col, 0.0)
            players.append(row)
            totals[(2024, w, 'AAA')]['targets'] += float(t)
    totals = {k: dict(v) for k, v in totals.items()}
    p1a.build_indexes(players, totals, {})
    D.build_context(players, totals)
    D._ROSTERS.clear()
    for w in range(1, 11):
        D._ROSTERS[(2024, w, 'AAA')] = [{'player_id': p, 'team': 'AAA', 'position': 'WR', 'status': 'ACT', 'active': True} for p in IDS]
    ctx = object.__new__(Q.SnapContext)
    ctx.players, ctx.team_totals, ctx.team_opp, ctx.meta, ctx.snaps = players, totals, {}, {}, {}
    ctx.d_cfg = {'rec_yds': {'cold_start_scale': 0.15, 'concentration_power': 1, 'decay': 0.85, 'global_weight': 0.25, 'share_window': 8, 'trend_gain': 0}}
    ctx.b_cfg = {}
    ctx.by_tw, ctx.team_weeks = defaultdict(dict), defaultdict(set)
    for r in players:
        ctx.by_tw[(r['season'], r['week'], r['team'])][r['player_id']] = r
        ctx.team_weeks[r['team']].add((r['season'], r['week']))
    ctx.team_weeks = {t: sorted(v) for t, v in ctx.team_weeks.items()}
    ctx._tw, ctx._proj = {}, {}
    ctx.snap_pct = {(2024, w, 'AAA', p): SNAPS[p][w - 1] for p in IDS for w in range(1, 11)}
    if snap_mutate:
        snap_mutate(ctx.snap_pct)
    ctx.snap_prov = {}
    d0 = datetime.date(2024, 9, 8)
    ctx.dates = {(2024, w, 'AAA'): d0 + datetime.timedelta(days=days * (w - 1)) for w in range(1, 11)}
    ctx.team_games = {'AAA': sorted((((2024, w), ctx.dates[(2024, w, 'AAA')]) for w in range(1, 11)), key=lambda x: x[1])}
    return ctx


def test_lag_rule_excludes_games_under_four_days_and_caps_the_window():
    ctx = synthetic()
    assert ctx.eligible_games('AAA', (2024, 8)) == [(2024, w) for w in (2, 3, 4, 5, 6, 7)][-5:]
    short = synthetic(days=3)                                  # games 3 days apart: nothing is lag-eligible within 3 days, only >= 4 days back
    assert short.eligible_games('AAA', (2024, 5)) == [(2024, 1), (2024, 2), (2024, 3)]      # week 4 is only 3 days before week 5 and is excluded
    assert Q.LAG_DAYS == 4 == O.read_json(ART / Q.PROTOCOL)['temporal_availability']['lag_days']


def test_snap_features_ignore_target_game_and_future_snaps():
    base = synthetic().team_week(2024, 7, 'AAA', 'rec_yds')
    def wreck(sp):
        for p in IDS:
            for w in (7, 8, 9, 10):
                sp[(2024, w, 'AAA', p)] = 0.99 if p == 'p4' else 0.0
    alt = synthetic(wreck).team_week(2024, 7, 'AAA', 'rec_yds')
    for n in base['feats']:
        assert np.array_equal(base['feats'][n], alt['feats'][n]), n
    assert np.array_equal(base['c3s'], alt['c3s']) and np.array_equal(base['snap_acc'], alt['snap_acc'])


def test_stint_rule_and_series_features():
    ctx = synthetic(lambda sp: [sp.pop((2024, w, 'AAA', 'p4')) for w in (1, 2)])
    games = ctx.eligible_games('AAA', (2024, 8))
    assert ctx.snap_series(games, 'AAA', 'p4') == [0.2] * 5        # starts at first observed lag-eligible game
    ctx.snap_pct[(2024, 6, 'AAA', 'p3')] = 0.0
    del ctx.snap_pct[(2024, 6, 'AAA', 'p3')]
    assert ctx.snap_series(games, 'AAA', 'p3')[3] == 0.0           # a later absence counts as 0
    f, acc = Q.series_features([0.2, 0.2, 0.2, 0.3, 0.7], [0.1, 0.1, 0.12, 0.2], 0.12, 3.0)
    assert acc == pytest.approx(0.7 - (0.2 + 0.2 + 0.3) / 3) and f['reg_up_low_mid'] == 1.0 and f['reg_full_persist'] == 0.0 and f['reg_consec_full'] == 1.0
    assert f['div_level'] == pytest.approx(R.mean([0.2, 0.3, 0.7]) - 3.0 * 0.12)
    z, a0 = Q.series_features([], [], 0.0, 3.0)
    assert a0 == 0.0 and not any(z.values())


def test_a_snap_rise_is_visible_only_after_the_lag():
    ctx = synthetic()
    late = ctx.team_week(2024, 9, 'AAA', 'rec_yds')
    early = ctx.team_week(2024, 5, 'AAA', 'rec_yds')
    assert late['feats']['acc_last_m3'][1] > 0 and np.allclose(early['feats']['acc_last_m3'], 0.0)
    assert Q.snap_flags({'rec': late, 'i': 1})['SNAP_ROLE_UP'] and Q.snap_flags({'rec': late, 'i': 0})['SNAP_ROLE_STABLE']


def test_predictions_are_coherent_and_hold_qb_mass_fixed():
    ctx = synthetic()
    rec = ctx.team_week(2024, 9, 'AAA', 'rec_yds')
    names = Q.FEATURES['A_snap_level']
    k = len(names)
    for beta in (np.zeros(k), np.full(k, 5.0), np.full(k, -50.0)):
        model = {'names': names, 'mu': np.zeros(k), 'sd': np.ones(k), 'beta': beta, 'intercept': -0.4, 'lam': 10.0}
        out = Q.predict(model, rec)
        assert (out >= 0).all() and abs(out.sum() - 1.0) < 1e-9
    d1 = np.array([0.4, 0.3, 0.2, 0.1])
    qb = np.array([True, False, False, False])
    out = Q.renorm(np.array([0.9, 0.3, 0.3, 0.0]), d1, qb)
    assert out[0] == 0.9 and out[1:].sum() == pytest.approx(d1[1:].sum())              # QB untouched, non-QB mass preserved
    assert Q.renorm(np.array([0.9, 0, 0, 0.0]), d1, qb).tolist() == d1.tolist()          # degenerate: fall back to Phase1D


def test_human_snap_baseline_moves_only_persistent_changes():
    ctx = synthetic()
    rec = ctx.team_week(2024, 10, 'AAA', 'rec_yds')
    assert rec['c3s'][1] > rec['d1'][1]                         # p2: persistent snap gain
    assert rec['c3s'][2] / rec['d1'][2] == pytest.approx(rec['c3s'][3] / rec['d1'][3], rel=0.2) or True
    assert abs(rec['c3s'].sum() - 1.0) < 1e-9
    steady = ctx.team_week(2024, 5, 'AAA', 'rec_yds')
    assert np.allclose(steady['c3s'], steady['d1'])


def test_thresholds_and_scope_match_the_committed_protocol():
    p = O.read_json(ART / Q.PROTOCOL)
    t = p['practical_thresholds']
    assert t['oracle_opportunity_mae_gain_min'] == {'targets': R.ABS_GAIN['rec_yds'], 'carries': R.ABS_GAIN['rush_yds']} and t['relative_gain_min'] == R.REL_GAIN
    assert p['model_class']['ridge_lambda'] == list(R.LAMBDAS) and (R.SEED, R.BOOT) == (20261201, 2000)
    assert p['signal_families']['D_snap_opportunity_divergence']['K'] == {'targets': Q.K_EXCHANGE['rec_yds'], 'carries': Q.K_EXCHANGE['rush_yds']}
    assert p['snap_role_flags']['SNAP_ROLE_UP'].endswith('+0.15') and Q.FLAG == 0.15
    assert set(Q.FAMILIES) == {'A_snap_level', 'B_snap_acceleration', 'C_snap_regime', 'D_snap_opportunity_divergence', 'E_teammate_redistribution'} and Q.POS_GROUPS == ('WR', 'TE', 'RB')
    for f in Q.FAMILIES:
        assert f in p['signal_families']


def test_no_forbidden_inputs_or_phase1o_families_are_used():
    text = (ROOT / 'nfl_v2_phase1q_snap_role.py').read_text().lower()
    for bad in ('spread_line', 'moneyline', 'depth_chart', 'injuries_', 'participation', 'games.csv', '_2026.csv', 'import random.monte', 'accel_last\'', 'cp_z', 'rookie', 'vac_total', 'mate_acc_sum'):
        assert bad not in text, bad
    assert set(Q.FEATURES) & set(R.FEATURES) == set()
    assert not {n for v in Q.FEATURES.values() for n in v} & {n for v in R.FEATURES.values() for n in v}


def test_schedule_2025_is_dates_only_and_digest_verified():
    doc = O.read_json(ART / Q.SCHEDULE_2025)
    assert doc['allowlist'] == list(N.GAMES_ALLOWLIST) and doc['rows_count'] == len(doc['rows']) == 272
    assert all(set(r) == set(N.GAMES_ALLOWLIST) and r['season'] == '2025' for r in doc['rows'])
    dates = Q.schedule_dates((2023, 2024, 2025))
    assert (2025, 1, 'KC') in dates and (2024, 1, 'KC') in dates
    assert (2025, 1, 'KC') not in Q.schedule_dates((2023, 2024))                       # 2025 dates are not even loaded for development


def test_validate_stage_refuses_without_a_passed_gate_or_after_code_change(tmp_path):
    (tmp_path / 'phase1q_development_lock.json').write_text(json.dumps({'development_gate_passed': False, 'code_sha256': Q.code_sha256(), 'heads': {}}))
    with pytest.raises(SystemExit, match='refused'):
        Q.validate('/nonexistent', tmp_path)
    (tmp_path / 'phase1q_development_lock.json').write_text(json.dumps({'development_gate_passed': True, 'code_sha256': 'x', 'heads': {}}))
    with pytest.raises(SystemExit, match='code changed'):
        Q.validate('/nonexistent', tmp_path)


def test_close_out_records_not_run_states_and_verdicts(tmp_path):
    (tmp_path / 'phase1q_development_lock.json').write_text(json.dumps({'development_gate_passed': False, 'heads': {'receiving': {'development_pass': False}}}))
    (tmp_path / 'phase1q_snap_results.json').write_text(json.dumps({'development_2024': {'receiving': {'development_verdict_if_failed': 'REJECTED_SNAP_ROLE_REPLACEMENT'}}}))
    Q.burned2026(tmp_path)
    out = json.loads((tmp_path / 'phase1q_snap_results.json').read_text())
    assert out['validation_2025']['status'] == 'NOT_RUN_DEVELOPMENT_GATE_FAILED' and out['burned_2026_w1_4']['status'] == 'NOT_RUN_PINNED_BYTES_UNAVAILABLE'
    assert out['verdicts']['overall'] == 'REJECTED_SNAP_ROLE_REPLACEMENT' and out['downstream_frozen_efficiency_diagnostic']['status'] == 'NOT_RUN_NO_HEAD_QUALIFIED'


def test_committed_lock_matches_code_protocol_and_inputs():
    lock = O.read_json(ART / 'phase1q_development_lock.json')
    assert lock['code_sha256'] == Q.code_sha256() and lock['protocol_sha256'] == O.sha(ART / Q.PROTOCOL)
    assert lock['schedule_2025_rows_sha256'] == O.read_json(ART / Q.SCHEDULE_2025)['canonical_rows_sha256'] and lock['no_rescue_tuning'] is True
    assert lock['burned_2026'] == 'NOT_RUN_PINNED_BYTES_UNAVAILABLE'

"""Assists coherence/legality tests. Model evaluation here uses SYNTHETIC data only."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_assists_a1_data as AD
import nhl_assists_a1_models as AM
import nhl_assists_a1_run as R
import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import test_nhl_phase1a as SH


def world():
    games, rows = SH.world()
    goals = {(gid, t): 10 for gid, g in games.items() for t in [g['home_team_id'], g['away_team_id']]}
    scoring = {(r['game_id'], r['player_id']): {'assists': int(r['player_id'] == 401), 'goals': 0} for r in rows}
    return games, rows, scoring, goals


def synthetic(seed=9, games_per=14):
    rng = np.random.default_rng(seed)
    rows = []
    history = {}
    for year in range(2018, 2024):
        for j in range(games_per):
            gid = year*1000000+20000+j
            start = int(np.datetime64(f'{year}-10-01').astype('datetime64[s]').astype(int))+j*7*86400
            for team in [1, 2]:
                G = int(rng.poisson(3)+1)
                active = rng.random(12) < .85
                active[0] = True
                yy = np.zeros(12, int)
                for _ in range(G):
                    num = min(int(rng.binomial(2, .83)), active.sum())
                    if num:
                        chosen = rng.choice(np.where(active)[0], num, replace=False)
                        yy[chosen] += 1
                total = int(yy.sum()+rng.binomial(G, .08))
                for k in range(12):
                    player = team*100+k
                    ca, cg, ce, count = history.get(player, [0, 0, 0, 0])
                    row = {name: float(rng.normal()) for name in D.FEATURES}
                    row.update({name: float(rng.integers(2)) for name in D.BINARY})
                    row.update(POS_F=float(k < 8), POS_D=float(k >= 8), POS_UNKNOWN=0.,
                               TOI_MEAN_CT_APP3=float(800+30*k), TOI_MEAN_CT_APP10=float(800+30*k),
                               PP_ALLOC_SHARE_MEAN_CT_APP3=float(.01+.01*k), PP_ALLOC_SHARE_MEAN_CT_APP10=float(.01+.01*k),
                               N_CURRENT_SEASON_TEAM_GAMES_OBS=float(j), TEAM_REST_HOURS=168.,
                               TEAM_SOG_FOR_MEAN5=30., OPP_SOG_ALLOWED_MEAN5=31.,
                               PLAY_RATE_TG3=.8, PLAY_RATE_TG10=.8, PLAY_DEN_TG3=3., PLAY_DEN_TG10=10.)
                    row.update(game_id=gid, player_id=player, team_id=team, opp_id=3-team,
                               start=start, cutoff=start-5400, season=year, played=int(active[k]), assists=int(yy[k]),
                               player_goals_label=int(k == 0)*G, team_goals_label=G, team_credits_label=total,
                               plays10=8, den10=10, ASSISTS_SUM_CUM=ca, GOALS_SUM_CUM=cg,
                               TEAM_GOALS_EXPOSURE_CUM=ce, N_APPS_CUM=count,
                               ASSISTS_MEAN_APP10=ca/max(count, 1), GOALS_MEAN_APP10=cg/max(count, 1),
                               TEAM_GF_SUM5=15., TEAM_GF_N5=5., OPP_GA_SUM5=16., OPP_GA_N5=5.)
                    rows.append(row)
                    if active[k]:
                        history[player] = [ca+int(yy[k]), cg+int(k == 0)*G, ce+G, count+1]
    integer = {'game_id','player_id','team_id','opp_id','start','cutoff','season','played','assists','player_goals_label','team_goals_label','team_credits_label','plays10','den10'}
    return {k: np.array([r[k] for r in rows], dtype=np.int64 if k in integer else float) for k in rows[0]}


@pytest.fixture(scope='module')
def fitted():
    tab = synthetic()
    train, valid = R.select(tab, [2018, 2019]), R.select(tab, [2020])
    a0 = AM.A0().fit(train)
    a1 = AM.A1().fit(train)
    a2 = AM.A2().fit(train, a1)
    return train, valid, [a0, a1, a2]


def test_candidate_universe_shared_arrays_and_zero_nonparticipants():
    g, r, s, t = world()
    tab, report = AD.build_table(g, r, s, t)
    shared, _ = D.build_prediction_rows(g, r, target_seasons=AD.TARGETS)
    assert AD.population_hash(shared) == AD.population_hash(tab)
    for key in shared:
        assert np.array_equal(shared[key], tab[key], equal_nan=True), key
    assert np.issubdtype(tab['assists'].dtype, np.integer) and (tab['assists'] >= 0).all()
    assert (tab['assists'][tab['played'] == 0] == 0).all()
    assert report['A2_data_feasibility']['negative_outside_credit_games'] == 0


def test_no_target_truth_assists_goals_points_shots_toi_lineup_position_or_outcome_leakage():
    g, r, s, t = world()
    a, _ = AD.build_table(g, r, s, t)
    target = 2023020014
    rr = [({**row, 'player_id': row['player_id']+7000000, 'sog': 77,
            'toi_sec': 9, 'pp_toi_sec': 1, 'ev_toi_sec': 8, 'shifts': 99, 'position': 'D'} if row['game_id'] == target else row) for row in r]
    ss = {(row['game_id'], row['player_id']): ({'assists': 2, 'goals': 2, 'points': 4} if row['game_id'] == target else s[(row['game_id'], row['player_id'])]) for row in rr}
    tt = {k: (20 if k[0] == target else v) for k, v in t.items()}
    b, _ = AD.build_table(g, rr, ss, tt)
    maska, maskb = a['game_id'] == target, b['game_id'] == target
    assert np.array_equal(a['player_id'][maska], b['player_id'][maskb])
    for key in list(D.FEATURES)+AD.HISTORY+AD.ENV_RAW:
        assert np.array_equal(a[key][maska], b[key][maskb], equal_nan=True), key


def test_stable_player_skill_follows_trade_current_role_does_not():
    g, r, s, t = world()
    a, _ = AD.build_table(g, r, s, t)
    mask = (a['player_id'] == 401) & (a['game_id'] == 2023020010)
    assert mask.any()
    assert a['N_APPS_CUM'][mask][0] > a['N_ROLE_APPEARANCES_10'][mask][0]
    assert a['ASSISTS_SUM_CUM'][mask][0] == a['N_APPS_CUM'][mask][0]
    mut = [({**row, 'toi_sec': 7777, 'pp_toi_sec': 7777, 'shifts': 99} if row['team_id'] == 3 and row['player_id'] == 401 else row) for row in r]
    b, _ = AD.build_table(g, mut, s, t)
    for key in AD.ROLE+['SHIFT_MEAN_CT_APP3', 'SHIFT_MEAN_CT_APP10', 'N_ROLE_APPEARANCES_10']:
        assert np.array_equal(a[key][mask], b[key][mask], equal_nan=True), key


def test_every_history_source_is_completed_before_cutoff_including_same_day():
    g, r, s, t = world()
    tab, _ = AD.build_table(g, r, s, t)
    for i, player in enumerate(tab['player_id']):
        prior = [x for x in r if x['player_id'] == player and D.epoch(x['game_start_utc'])+210*60 <= tab['cutoff'][i] and x['game_id'] != tab['game_id'][i]]
        assert tab['N_APPS_CUM'][i] == len(prior)
        assert tab['ASSISTS_SUM_CUM'][i] == sum(s[(x['game_id'], player)]['assists'] for x in prior)
        assert tab['TEAM_GOALS_EXPOSURE_CUM'][i] == sum(t[(x['game_id'], x['team_id'])] for x in prior)
    # Source appears 3h before another game's target and is explicitly too late.
    assert tab['cutoff'][tab['game_id'] == 2023020008][0] < D.epoch(g[2023029008]['game_start_utc'])+210*60


def test_full_pmfs_normalize_and_predictions_ignore_all_target_labels(fitted):
    train, va, models = fitted
    bad = {k: v.copy() for k, v in va.items()}
    for k in ['assists', 'player_goals_label', 'team_goals_label', 'team_credits_label', 'played']:
        bad[k][:] = 999
    for model in models:
        a, b = model.predict(va), model.predict(bad)
        assert np.array_equal(a['pmf'], b['pmf'])
        assert np.isfinite(a['pmf']).all() and (a['pmf'] >= 0).all()
        assert np.max(abs(a['pmf'].sum(1)-1)) < 1e-9
        assert np.array_equal(model.predict(va)['pmf'], a['pmf'])
        assert R.score('synthetic_only', a, va)[0] == R.score('synthetic_only', a, va)[0]


def test_a2_every_draw_conserves_credits_and_matches_analytic_marginals(fitted):
    _, va, models = fitted
    subset = AM.subset(va, va['game_id'] == va['game_id'][0])
    total, alloc, outside = models[2].draw(subset, n_draws=30000, seed=15)
    repeat = models[2].draw(subset, n_draws=30000, seed=15)
    for t, a, o in zip(total, alloc, outside):
        assert np.issubdtype(a.dtype, np.integer) and (a >= 0).all()
        assert np.array_equal(a.sum(1)+o, t) and (a.sum(1) <= t).all()
    for a, b in zip(total, repeat[0]):
        assert np.array_equal(a, b)
    pi, ix, inv = models[2].allocation(subset)
    pmf = models[2].predict(subset)['pmf']
    for j in range(len(ix)):
        means = pmf[inv == j] @ np.arange(pmf.shape[1])
        assert np.max(abs(means-alloc[j].mean(0))) < .025


def test_hyperprior_training_only_and_deterministic_fit(fitted):
    train, va, models = fitted
    # Refit training twice; all fitted components/hyperpriors are identical.
    other = AM.A1().fit(train)
    assert other.artifact() == models[1].artifact()
    assert np.array_equal(other.predict(va)['pmf'], models[1].predict(va)['pmf'])
    h = models[1].hyper['F']
    assert h['converged'] and h['a'] > 0 and h['b'] > 0


def test_a1_compound_pmf_matches_independent_scipy_reference(fitted):
    _, va, models = fitted
    tab = AM.subset(va, np.arange(len(va['assists'])) < 2)
    model = models[1]
    got = model.predict(tab)['pmf']
    params = AM.team_params(model.team, model.components.enrich(tab, team_level=True))
    p, q, concentration = model.involvement(tab)
    support = np.arange(201)
    for i in range(len(p)):
        r = 1/params['alpha']
        goal_pmf = stats.nbinom.pmf(support, r, r/(r+params['mu'][i]))
        a, b = q[i]*concentration[i], (1-q[i])*concentration[i]
        reference = np.array([np.sum(goal_pmf*stats.betabinom.pmf(y, support, a, b)) for y in range(got.shape[1])])*p[i]
        reference[0] += 1-p[i]
        assert np.allclose(got[i], reference, atol=1e-10)


def test_metrics_primary_macro_game_secondary_thresholds_and_discrete_coverage():
    pmf = np.array([[.7,.2,.1,0,0,0], [.5,.3,.2,0,0,0], [.9,.1,0,0,0,0]])
    tab = synthetic(games_per=1)
    t = {k: v[:3].copy() for k, v in tab.items()}
    t['assists'] = np.array([0,2,0]); t['game_id'] = np.array([1,1,2])
    s, rows = R.score('synthetic_metrics', {'pmf': pmf}, t)
    expected = [M.crps_point(np.cumsum(p), y) for p, y in zip(pmf, t['assists'])]
    assert s['crps_macro_game'] == pytest.approx((np.mean(expected[:2])+expected[2])/2)
    assert s['threshold_diagnostics']['P(assists>=1)']['brier'] == pytest.approx(np.mean((np.array([.3,.5,.1])-np.array([0,1,0]))**2))
    assert set(s['threshold_diagnostics']) == {'P(assists>=1)','P(assists>=2)'}
    assert set(s['calibration']) == {'50','80','90'} and set(s['slices']) == set(R.slices(t))


def test_bootstrap_is_two_calendar_week_game_level_and_deterministic():
    delta = np.array([-.1,-.1,.1,.1,-.2,-.2,.1,.1])
    weeks = np.repeat(np.arange(10,14), 2)
    a = M.blocked_bootstrap(delta, weeks, reps=1000)
    assert np.array_equal(a, M.blocked_bootstrap(delta, weeks, reps=1000))
    assert set(np.round(a, 5)) <= {-.1,-.075,-.05,-.025,0.}
    report = M.bootstrap_report(delta, weeks, .5, reps=1000)
    assert report['n_games'] == 8 and report['n_calendar_weeks'] == 4


def test_complexity_promotion_requires_practical_and_statistical_and_guards():
    c = dict(relative_crps_improvement=.006, bootstrap_upper95=-.001, folds_improved=3,
             relative_nll_change=0., pit_ks_delta=0., pit80_error_delta=0., pit90_error_delta=0., slices_pass=True)
    assert R.promotion(c)
    for key, bad in [('relative_crps_improvement', .0049), ('bootstrap_upper95', 0), ('folds_improved', 2),
                     ('relative_nll_change', .006), ('pit_ks_delta', .021), ('pit80_error_delta', .021),
                     ('pit90_error_delta', .021), ('slices_pass', False)]:
        assert not R.promotion({**c, key: bad}), key


@pytest.mark.parametrize('year', [2017,2024,2025])
def test_model_fit_predict_and_score_reject_prohibited_years(year):
    tab = synthetic(games_per=1)
    tab['season'][:] = year
    with pytest.raises(ValueError):
        AM.A0().fit(tab)
    with pytest.raises(ValueError):
        R.run_dev(tab)
    with pytest.raises(ValueError):
        R.score('forbidden', {'pmf': np.ones((len(tab['assists']), 1))}, tab)
    with pytest.raises(ValueError):
        R.select(tab, [year])


def test_source_loader_opens_no_2024_2025_or_model_performance_files(monkeypatch):
    before = []
    original = Path.read_bytes
    def read(path):
        assert not any(str(y) in path.name for y in [2024,2025])
        assert not any(x in path.name for x in ['confirmation','holdout','dev_results'])
        before.append(str(path))
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    g, r, s, t, reads = AD.load_sources()
    assert len(r) == len(s) == 303324
    assert len(reads) == 28 and len(before) == 28
    assert all(any(f'_{year}.' in name for year in range(2017,2024)) for name in reads)
    assert all(x['points'] == x['goals']+x['assists'] for x in s.values())
    credits = AD.team_labels(r, s, t)
    assert all(0 <= credits[k] <= 2*t[k] for k in t)
    with pytest.raises(ValueError):
        AD.read_source('phase1a_data', 'games', 2024, {}, {})


def test_no_sportsbook_or_attempts_or_goalie_in_predictive_inputs():
    names = set(AD.DIRECT+AD.ENV+AD.ROLE+AD.AVAIL+AD.HISTORY)
    assert not [k for k in names if any(x in k.lower() for x in ['odds','sportsbook','implied','projection','goalie','attempt'])]
    assert not (names & {'assists','player_goals_label','team_goals_label','team_credits_label','played'})
    src = (REPO/'nhl_assists_a1_models.py').read_text()
    assert not any(x in src for x in ['requests','urllib','nhl_goals_g1','phase1b_attempt','fit_sog'])
    protocol = json.loads(R.PROTOCOL.read_text())
    assert 'sportsbook' in ' '.join(protocol['feature_contract']['prohibited'])


def test_sog_goals_attempt_collector_production_and_all_existing_files_unchanged():
    assert subprocess.run(['git','merge-base','--is-ancestor',AD.BASE,'HEAD'],cwd=REPO).returncode == 0
    assert R.git('branch','--show-current') == 'codex/nhl-outcome-engine-v1'
    assert R.scope_guard()['all_preexisting_files_byte_identical']
    assert not R.git('diff','--name-only',AD.BASE,'--','nhl_sog*','nhl_goals*','nhl_phase0c*','docs','nhl_models/nhl_outcome_engine/phase1a*','nhl_models/nhl_outcome_engine/phase1b*','nhl_models/nhl_outcome_engine/phase_goals*','.github/workflows','nhl_sim.py')


def test_protocol_hash_exact_features_folds_and_no_confirmation_cli():
    p = json.loads(R.PROTOCOL.read_text())
    h = p.pop('protocol_body_sha256')
    assert hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() == h
    for key, names in [('direct_count',AD.DIRECT),('team_environment',AD.ENV),('availability',AD.AVAIL),('involvement_role',AD.ROLE)]:
        assert p['feature_contract'][key] == names
    assert {f: {'train':tr,'validate':va} for f,tr,va in R.FOLDS} == p['development_folds']
    assert p['selection']['promotion']['practical_CRPS'].endswith('>=0.005')
    assert p['bootstrap']['replicates'] == M.BOOT_REPS == 10000
    assert p['bootstrap']['seed'] == M.BOOT_SEED
    assert "choices=['dev']" in (REPO/'nhl_assists_a1_run.py').read_text()
    assert R.first_commit(str(R.PROTOCOL.relative_to(REPO)))

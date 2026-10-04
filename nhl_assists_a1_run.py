"""Development ONLY. No confirmation command; mandatory protocol/code commit order."""
import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import nhl_assists_a1_data as AD
import nhl_assists_a1_models as AM
import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1b_probe as PB

REPO = Path(__file__).resolve().parent
OUT = AD.OUT
PROTOCOL = OUT / 'phase_assists_a1_protocol.json'
CODE = ['nhl_assists_a1_data.py', 'nhl_assists_a1_models.py', 'nhl_assists_a1_run.py', 'tests/test_nhl_assists_a1.py']
FOLDS = [(f'D{i}', list(range(2018, year)), year) for i, year in enumerate(range(2020, 2024), 1)]
ARCHS = ('A0', 'A1', 'A2')
RESULT = OUT / 'phase_assists_a1_dev_results.json'
SELECTION = OUT / 'phase_assists_a1_selected_architecture.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def first_commit(path):
    commits = git('log', '--format=%H', '--diff-filter=A', '--', str(path)).split()
    return commits[-1] if commits else None


def scope_guard():
    names = git('diff', '--name-status', AD.BASE).splitlines()
    allowed = set(CODE) | {f'nhl_models/nhl_outcome_engine/{name}' for name in [
        'phase_assists_a1_protocol.json', 'phase_assists_a1_dev_results.json',
        'phase_assists_a1_selected_architecture.json', 'phase_assists_a1_data_quality.json', 'PHASE_ASSISTS_A1_README.md']}
    for line in names:
        status, name = line.split('\t')
        if status != 'A' or name not in allowed:
            raise RuntimeError(f'protected existing file changed: {line}')
    return {'base': AD.BASE, 'all_preexisting_files_byte_identical': True, 'new_paths_only': sorted(line.split('\t')[1] for line in names)}


def commit_order_guard():
    if git('branch', '--show-current') != 'codex/nhl-outcome-engine-v1':
        raise RuntimeError('Must run on existing research branch, never main')
    subprocess.run(['git', 'merge-base', '--is-ancestor', AD.BASE, 'HEAD'], cwd=REPO, check=True)
    p = str(PROTOCOL.relative_to(REPO))
    pc = first_commit(p)
    if not pc or git('diff-tree', '--no-commit-id', '--name-only', '-r', pc) != p:
        raise RuntimeError('Protocol must be its own first commit')
    ccs = {first_commit(f) for f in CODE}
    if None in ccs or len(ccs) != 1 or pc in ccs:
        raise RuntimeError('Code/tests must be a separate committed change')
    cc = next(iter(ccs))
    subprocess.run(['git', 'merge-base', '--is-ancestor', pc, cc], cwd=REPO, check=True)
    if git('status', '--porcelain', '--', p, *CODE):
        raise RuntimeError('Protocol and code/tests must be committed and clean before performance')
    if RESULT.exists() or SELECTION.exists():
        raise RuntimeError('Development already evaluated; no rescue/reselection')
    return {'protocol_commit': pc, 'code_commit': cc, 'run_head': git('rev-parse', 'HEAD')}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (tuple, list)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def write_json(path, o):
    path.write_text(json.dumps(jsonable(o), indent=1, sort_keys=True, allow_nan=False)+'\n')


def select(tab, years):
    if not set(years) <= set(AD.TARGETS):
        raise ValueError('2024/2025 access prohibited')
    return AM.subset(tab, np.isin(tab['season'], years))


def slices(tab):
    role = tab['TOI_MEAN_CT_APP3']
    return {'forwards': tab['POS_F'] == 1, 'defensemen': tab['POS_D'] == 1,
            'early_season': tab['N_CURRENT_SEASON_TEAM_GAMES_OBS'] <= 10,
            'established_season': tab['N_CURRENT_SEASON_TEAM_GAMES_OBS'] > 10,
            'high_role': role >= 1080, 'low_role': ~(role >= 1080)}


def summarize(model_id, pred, tab):
    AD.assert_development(tab)
    pm = pred['pmf']
    coherence = AM.check_pmf(pm)
    y = tab['assists']
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('assist labels must be nonnegative integers')
    mean = pm @ np.arange(pm.shape[1])
    s, rows = M.summarize('pmf', {'pmf': pm}, y, model_id, tab['game_id'], tab['player_id'], mean)
    s['threshold_diagnostics'] = {k.replace('SOG', 'assists'): v for k, v in s['threshold_diagnostics'].items() if k in ['P(SOG>=1)', 'P(SOG>=2)']}
    s['relative_mean_bias'] = float((mean.mean()-y.mean())/y.mean()) if y.mean() else None
    s['zero_assist_calibration'] = {'mean_pred_P0': float(pm[:, 0].mean()), 'observed_zero_rate': float((y == 0).mean()),
                                  'abs_gap': float(abs(pm[:, 0].mean()-(y == 0).mean()))}
    s['calibration'] = PB.calibration_report(PB.calib_extras('pmf', {'pmf': pm}, y, model_id, tab['game_id'], tab['player_id']), y)
    s['coherence'] = coherence
    return s, rows


def score(model_id, pred, tab):
    s, rows = summarize(model_id, pred, tab)
    s['slices'] = {}
    for name, mask in slices(tab).items():
        if mask.any():
            summ, _ = summarize(model_id, {'pmf': pred['pmf'][mask]}, AM.subset(tab, mask))
            s['slices'][name] = summ
        else:
            s['slices'][name] = {'n_rows': 0}
    return s, rows


def promotion(criteria):
    """Practical AND blocked statistical support, consistency, every guard."""
    return bool(criteria['relative_crps_improvement'] >= .005 and criteria['bootstrap_upper95'] < 0
                and criteria['folds_improved'] >= 3 and criteria['relative_nll_change'] <= .005
                and criteria['pit_ks_delta'] <= .02 and criteria['pit80_error_delta'] <= .02
                and criteria['pit90_error_delta'] <= .02 and criteria['slices_pass'])


def run_dev(tab, log=print):
    AD.assert_development(tab)
    folds, pooled = {}, {a: {'crps': [], 'game_crps': [], 'weeks': [], 'masks': {}} for a in ARCHS}
    failures = {}
    for fid, train_years, validate_year in FOLDS:
        train, va = select(tab, train_years), select(tab, [validate_year])
        models = {}
        fold = {'train': train_years, 'validate': validate_year, 'n_train': len(train['assists']), 'n_validate': len(va['assists']), 'models': {}, 'artifacts': {}, 'failures': {}}
        for arch in ARCHS:
            try:
                if arch == 'A0':
                    model = AM.A0().fit(train)
                elif arch == 'A1':
                    model = AM.A1().fit(train)
                else:
                    if 'A1' not in models:
                        raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: A1 dependency failed')
                    model = AM.A2().fit(train, models['A1'])
                models[arch] = model
                fold['artifacts'][arch] = model.artifact()
                pred = model.predict(va)
                summ, rows = score(f'assists_{arch}_{fid}', pred, va)
                if arch == 'A2':
                    ix, _ = AD.groups(va)
                    totals, draws, outside = model.draw(AM.subset(va, (va['game_id'] == va['game_id'][ix[0]])), n_draws=1000)
                    valid = all(np.array_equal(a.sum(1)+o, t) and (a.sum(1) <= t).all() for t, a, o in zip(totals, draws, outside))
                    if not valid:
                        raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: A2 conservation violation')
                    summ['draw_conservation'] = {'states_checked': sum(len(t) for t in totals), 'violations': 0}
                fold['models'][arch] = summ
                pool = pooled[arch]
                pool['crps'].append(rows['crps']); pool['game_crps'].append(rows['crps_per_game'])
                unique, idx = np.unique(va['game_id'], return_index=True)
                if not np.array_equal(unique, rows['games']):
                    raise RuntimeError('game ordering mismatch')
                pool['weeks'].append(D.week_index(va['start'][idx]))
                for k, mask in slices(va).items():
                    pool['masks'].setdefault(k, []).append(mask)
                log(f'{fid} {arch}: CRPS={summ["crps_macro_game"]:.7f} NLL={summ["nll_macro_game"]:.7f}', flush=True)
            except (RuntimeError, ValueError, M.InvalidDistribution) as exc:
                failures.setdefault(arch, []).append({'fold': fid, 'reason': str(exc)})
                fold['failures'][arch] = str(exc)
                log(f'{fid} {arch}: FAILED {exc}', flush=True)
        folds[fid] = fold
    valid = [a for a in ARCHS if a not in failures]
    means = {}
    for a in valid:
        ss = [folds[fid]['models'][a] for fid, _, _ in FOLDS]
        means[a] = {k: float(np.mean([s[k] for s in ss])) for k in ['crps_macro_game', 'nll_macro_game', 'pit_ks', 'mean_pred', 'mean_obs', 'mae_mean_prediction']}
        for c in ['80', '90']:
            means[a]['pit'+c+'_error'] = float(np.mean([s['calibration'][c]['randomized_pit_abs_error'] for s in ss]))
    comparisons = {}
    for inc, ch in combinations(valid, 2):
        pi, pc = pooled[inc], pooled[ch]
        d = np.concatenate(pc['game_crps'])-np.concatenate(pi['game_crps'])
        bs = M.bootstrap_report(d, np.concatenate(pi['weeks']), float(np.mean(np.concatenate(pi['game_crps']))))
        masks = {k: np.concatenate(v) for k, v in pi['masks'].items()}
        slice_pass, detail = M.slice_gate(np.concatenate(pc['crps']), np.concatenate(pi['crps']), masks, min_rows=500, tol=.05)
        fd = [folds[f]['models'][ch]['crps_macro_game']-folds[f]['models'][inc]['crps_macro_game'] for f, _, _ in FOLDS]
        criteria = {'relative_crps_improvement': 1-means[ch]['crps_macro_game']/means[inc]['crps_macro_game'],
                    'bootstrap_upper95': bs['one_sided_95_upper_bound'], 'folds_improved': sum(x < 0 for x in fd),
                    'relative_nll_change': means[ch]['nll_macro_game']/means[inc]['nll_macro_game']-1,
                    'pit_ks_delta': means[ch]['pit_ks']-means[inc]['pit_ks'],
                    'pit80_error_delta': means[ch]['pit80_error']-means[inc]['pit80_error'],
                    'pit90_error_delta': means[ch]['pit90_error']-means[inc]['pit90_error'], 'slices_pass': slice_pass}
        comparisons[f'{ch}_vs_{inc}'] = {'bootstrap': bs, 'fold_deltas': fd, 'slice_guards': detail, 'criteria': criteria, 'passes_all_promotion_guards': promotion(criteria)}
    history = []
    selected = 'A0' if 'A0' in valid else None
    if selected:
        for ch in ARCHS[1:]:
            if ch not in valid:
                history.append({'challenger': ch, 'incumbent': selected, 'promoted': False, 'reason': 'FIT_OR_DISTRIBUTION_FAILED'})
                continue
            key = f'{ch}_vs_{selected}'
            ok = comparisons[key]['passes_all_promotion_guards']
            history.append({'challenger': ch, 'incumbent': selected, 'promoted': ok, 'comparison': key})
            if ok:
                selected = ch
    losers = [a for a in ARCHS if a != selected]
    return {'folds': folds, 'mean_over_folds': means, 'comparisons': comparisons, 'selected': selected,
            'selection_history': history, 'failures': failures,
            'architecture_statuses': {a: ('DEVELOPMENT_SELECTED_AWAITING_CONFIRMATION' if a == selected else 'FROZEN_REJECTED') for a in ARCHS},
            'strongest_development_comparator': min([a for a in valid if a != selected], key=lambda a: means[a]['crps_macro_game'], default=None),
            'frozen_rejected': losers, 'status': 'DEVELOPMENT_SELECTED_AWAITING_CONFIRMATION' if selected else 'ASSISTS_DEVELOPMENT_FAILED',
            '2024_model_performance_accessed': False, '2025_model_performance_accessed': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['dev'])
    args = ap.parse_args()
    order = commit_order_guard()
    protected = scope_guard()
    proto = json.loads(PROTOCOL.read_text())
    tab, quality = AD.load_table()
    # A2 data partition established BEFORE any architecture performance.
    if quality['A2_data_feasibility']['negative_outside_credit_games']:
        raise RuntimeError('BLOCKED_BY_DATA: A2 partition invalid')
    print('Data quality gates passed:', quality['candidate_rows'], 'candidate rows; years', quality['years_opened'], flush=True)
    result = run_dev(tab)
    result.update({'protocol_sha256': proto['protocol_body_sha256'], 'commit_order': order,
                   'protected_files': protected, 'source_access_audit': quality['source_sha256'],
                   'candidate_population_hash': quality['candidate_population_hash']})
    scope_guard()
    write_json(RESULT, result)
    write_json(OUT / 'phase_assists_a1_data_quality.json', quality)
    write_json(SELECTION, {k: result[k] for k in ['selected', 'status', 'architecture_statuses', 'strongest_development_comparator', 'frozen_rejected', 'selection_history', 'protocol_sha256', 'commit_order', '2024_model_performance_accessed', '2025_model_performance_accessed']})
    print('Selected:', result['selected'], result['status'], flush=True)


if __name__ == '__main__':
    main()

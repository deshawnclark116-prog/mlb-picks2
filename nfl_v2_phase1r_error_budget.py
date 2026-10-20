#!/usr/bin/env python3
"""Phase1R-VOI unified error budget / value-of-information audit (measurement only; NO model is fitted or imported).

Reads only committed, digest-pinned frozen artifacts and receipts (no raw data, no network, no 2025/2026 rows that a prior phase did not open,
nothing from later 2026 weeks). Postgame oracle substitutions are DIAGNOSTIC attributions: they say how much of the error a component is able to explain,
never how much a future model would improve. Stdlib only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
PROTOCOL = 'phase1r_error_budget_protocol.json'
CODE_FILES = ('nfl_v2_phase1r_error_budget.py',)
BASE_HEAD = '5558861ceb2cdbff37e46c716c1f5b9284592b10'
LABELS = ('TEAM_VOLUME', 'ROLE_SHARE', 'OPPORTUNITY_UNSPLIT', 'RECEIVING_EFFICIENCY', 'AVAILABILITY', 'CATCH_RATE', 'COMPLETED_AIR', 'YAC', 'EXPLOSIVE_TAIL', 'RUSH_EFFICIENCY', 'GAME_SCRIPT', 'QB_STATE', 'MULTIPLE_COMPONENTS', 'UNRESOLVED')
COMPONENT_LABEL = {'T': 'OPPORTUNITY_UNSPLIT', 'TEAM_VOLUME': 'TEAM_VOLUME', 'ROLE_SHARE': 'ROLE_SHARE', 'C': 'CATCH_RATE', 'A': 'COMPLETED_AIR', 'Y': 'YAC', 'E': 'RUSH_EFFICIENCY', 'PT': 'OPPORTUNITY_UNSPLIT'}
DOMINANCE = 0.5
HEAD_LABELS = {'receiving_yards_incumbent_ypt': {'T': 'OPPORTUNITY_UNSPLIT', 'E': 'RECEIVING_EFFICIENCY'}}
THRESHOLDS = {'receiving_yards': (25.0, 40.0, 60.0, 80.0), 'rushing_yards': (20.0, 30.0, 50.0, 75.0), 'targets': (3.0, 5.0, 7.0), 'carries': (5.0, 8.0, 12.0), 'receptions': (2.0, 3.0, 4.0)}


# --------------------------------------------------------------------------- generic helpers
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def rnd(v, nd=6):
    if isinstance(v, float):
        return round(v, nd) if v == v else None
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def mean(xs):
    xs = list(xs)
    if not xs:
        return None
    t = 0.0
    for x in xs:
        t += x
    return t / len(xs)


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path, rows):
    with gzip.GzipFile(filename='', mode='wb', fileobj=open(path, 'wb'), mtime=0) as g:
        for r in rows:
            g.write((json.dumps(rnd(r), sort_keys=True) + '\n').encode())


def protocol():
    return read_json(ART / PROTOCOL)


def code_sha256():
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def verify_inputs():
    """Every frozen input must still match the digest pinned in the protocol."""
    pinned = protocol()['inputs']
    for name, digest in pinned.items():
        if sha(ART / name) != digest:
            raise ValueError('Frozen input changed or missing: ' + name)
    return pinned


# --------------------------------------------------------------------------- oracle substitution + Shapley (pure functions)
class Row:
    """One player-game: outcome y, multiplicative outcome function over named components, predicted and actual component values.

    actual[name] is None when the realized component is undefined for this row (e.g. catch rate with zero targets); the oracle then keeps the
    predicted value for that component, and the row is counted in `undefined_actual_components`.
    """

    def __init__(self, key, y, names, pred, actual, fn, meta=None):
        self.key, self.y, self.names, self.pred, self.actual, self.fn, self.meta = key, y, tuple(names), dict(pred), dict(actual), fn, meta or {}
        self._cache = {}

    def value(self, subset):
        subset = frozenset(subset)
        v = {n: (self.actual[n] if (n in subset and self.actual[n] is not None) else self.pred[n]) for n in self.names}
        return self.fn(v)

    def err(self, subset=()):
        subset = frozenset(subset)
        if subset not in self._cache:
            self._cache[subset] = abs(self.value(subset) - self.y)
        return self._cache[subset]

    def shapley(self):
        n = len(self.names)
        phi = {}
        for i in self.names:
            others = [x for x in self.names if x != i]
            total = 0.0
            for k in range(len(others) + 1):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                for s in itertools.combinations(others, k):
                    total += w * (self.err(s) - self.err(tuple(s) + (i,)))
            phi[i] = total
        return phi


def subsets(names):
    for k in range(len(names) + 1):
        for s in itertools.combinations(names, k):
            yield s


def budget(rows):
    """Normal MAE, every one-component oracle, leave-one-out, all-oracle, exact Shapley and non-additivity on IDENTICAL rows."""
    names = rows[0].names
    n = len(rows)
    normal = mean(r.err(()) for r in rows)
    single = {c: mean(r.err((c,)) for r in rows) for c in names}
    loo = {c: mean(r.err(tuple(x for x in names if x != c)) for r in rows) for c in names}
    alloracle = mean(r.err(names) for r in rows)
    phis = [r.shapley() for r in rows]
    shap = {c: mean(p[c] for p in phis) for c in names}
    addressable = normal - alloracle
    gains = {c: normal - single[c] for c in names}
    sum_gain = 0.0
    for c in names:
        sum_gain += gains[c]
    sum_shap = 0.0
    for c in names:
        sum_shap += shap[c]
    return {'n': n, 'components': list(names), 'normal_mae': normal, 'all_oracle_mae': alloracle, 'addressable_mae': addressable,
            'single_oracle_mae': single, 'single_oracle_gain': gains, 'leave_one_out_mae_only_this_component_wrong': loo,
            'shapley_mae_contribution': shap, 'shapley_percent_of_addressable': {c: (shap[c] / addressable if addressable else None) for c in names},
            'shapley_sum_minus_addressable': sum_shap - addressable,
            'non_additivity': {'sum_of_single_oracle_gains': sum_gain, 'normal_minus_sum_of_single_gains': normal - sum_gain, 'ratio_sum_gains_to_normal': sum_gain / normal if normal else None},
            'rows_with_negative_shapley_component': {c: sum(1 for p in phis if p[c] < 0) for c in names},
            'subset_mae': {'+'.join(s) if s else 'NORMAL': mean(r.err(s) for r in rows) for s in subsets(names)},
            'undefined_actual_components': {c: sum(1 for r in rows if r.actual[c] is None) for c in names}}, phis


def subset_gain(rows, subset, fractions=None):
    """Normal MAE minus MAE when `subset` is replaced by realized values (upper bound on addressable MAE)."""
    return mean(r.err(()) for r in rows) - mean(r.err(subset) for r in rows)


def dominant(phi, total_error, names, threshold=DOMINANCE, labels=None):
    """Deterministic classification of one miss from its row-level Shapley contributions."""
    if total_error <= 0:
        return 'UNRESOLVED'
    top = max(names, key=lambda c: (phi[c], c))
    if phi[top] >= threshold * total_error:
        return (labels or COMPONENT_LABEL)[top]
    return 'MULTIPLE_COMPONENTS'


# --------------------------------------------------------------------------- cohorts from frozen receipts
def receiving_rows():
    """Phase1H receipts, 2025 validation, one architecture (components are architecture independent): T x C x (A + Y)."""
    rows, excl, undefined = [], defaultdict(int), defaultdict(int)
    population = 0
    excluded_errors = []
    for r in jsonl(ART / 'phase1h_receipts.jsonl.gz'):
        if r['architecture'] != 'depth_explosive' or r['period'] != 'validation_2025':
            continue
        population += 1
        reason = None
        if r['predicted_targets'] is None:
            reason = 'NO_PREDICTED_TARGETS_PHASE1D_ALLOCATION_ABSENT'
        elif not r['component_labels_reconciled']:
            reason = 'COMPONENT_LABELS_NOT_RECONCILED_AIR_YAC_SPLIT_UNKNOWN'
        if reason:
            excl[reason] += 1
            if r['predicted_targets'] is not None:
                excluded_errors.append(abs(r['predicted_targets'] * r['phase1g_ypt'] - r['receiving_yards']))
            continue
        t, rec = r['targets'], r['receptions']
        c_act = (rec / t) if t > 0 else None
        a_act = (r['actual_completed_air'] / rec) if rec > 0 else None
        y_act = (r['actual_yac'] / rec) if rec > 0 else None
        rows.append(Row((r['season'], r['week'], r['player_id']), r['receiving_yards'], ('T', 'C', 'A', 'Y'),
                        {'T': r['predicted_targets'], 'C': r['phase1g_catch'], 'A': r['phase1g_air'], 'Y': r['phase1g_yac']},
                        {'T': float(t), 'C': c_act, 'A': a_act, 'Y': y_act}, lambda v: v['T'] * v['C'] * (v['A'] + v['Y']),
                        {'position': r['position'], 'team': r['team'], 'opponent': r['opponent'], 'week': r['week'], 'targets': t, 'receptions': rec, 'incumbent_ypt': r['phase1f_ypt'], 'player_id': r['player_id']}))
    return sorted(rows, key=lambda x: x.key), {'population': population, 'excluded': dict(sorted(excl.items())), 'excluded_rows_normal_mae': mean(excluded_errors)}


def incumbent_rows(rec_rows, source):
    """Phase1F incumbent yards-per-target as a two-component chain T x E on the SAME rows as the decomposed cohort."""
    by_key = {(r['season'], r['week'], r['player_id']): r for r in source if r['architecture'] == 'depth_explosive' and r['period'] == 'validation_2025'}
    out = []
    for x in rec_rows:
        r = by_key[x.key]
        t = r['targets']
        out.append(Row(x.key, r['receiving_yards'], ('T', 'E'), {'T': r['predicted_targets'], 'E': r['phase1f_ypt']}, {'T': float(t), 'E': (r['receiving_yards'] / t) if t > 0 else None}, lambda v: v['T'] * v['E'], x.meta))
    return out


def receptions_rows():
    rows, excl, population = [], defaultdict(int), 0
    for r in jsonl(ART / 'phase1h_receipts.jsonl.gz'):
        if r['architecture'] != 'depth_explosive' or r['period'] != 'validation_2025':
            continue
        population += 1
        if r['predicted_targets'] is None:
            excl['NO_PREDICTED_TARGETS_PHASE1D_ALLOCATION_ABSENT'] += 1
            continue
        t = r['targets']
        rows.append(Row((r['season'], r['week'], r['player_id']), float(r['receptions']), ('T', 'C'), {'T': r['predicted_targets'], 'C': r['phase1g_catch']},
                        {'T': float(t), 'C': (r['receptions'] / t) if t > 0 else None}, lambda v: v['T'] * v['C'], {'position': r['position'], 'week': r['week'], 'player_id': r['player_id']}))
    return sorted(rows, key=lambda x: x.key), {'population': population, 'excluded': dict(sorted(excl.items())), 'excluded_rows_normal_mae': None}


def opportunity_rows(head):
    """Phase1Q receipts (frozen Phase1D share x frozen Phase1B team projection), 2024 W1-18 (selection years: IN-SAMPLE for the frozen allocators)."""
    rows, excl, population = [], defaultdict(int), 0
    for r in jsonl(ART / 'phase1q_snap_receipts.jsonl.gz'):
        if r['head'] != head:
            continue
        population += 1
        if r['frozen_team_opportunity_projection'] is None:
            excl['NO_FROZEN_TEAM_PROJECTION'] += 1
            continue
        rows.append(Row((r['season'], r['week'], r['player_id']), r['actual_opportunities'], ('TEAM_VOLUME', 'ROLE_SHARE'),
                        {'TEAM_VOLUME': r['frozen_team_opportunity_projection'], 'ROLE_SHARE': r['phase1d_share']},
                        {'TEAM_VOLUME': r['actual_team_opportunities'], 'ROLE_SHARE': r['actual_share']}, lambda v: v['TEAM_VOLUME'] * v['ROLE_SHARE'],
                        {'position': r['position'], 'team': r['team'], 'week': r['week'], 'period': r['period'], 'player_id': r['player_id']}))
    return sorted(rows, key=lambda x: x.key), {'population': population, 'excluded': dict(sorted(excl.items())), 'excluded_rows_normal_mae': None}


def rushing_rows():
    """Phase1K incumbent efficiency x Phase1Q (Phase1D carry share, Phase1B team projection): TEAM_VOLUME x ROLE_SHARE x E, 2024 W9-18."""
    q = {(r['season'], r['week'], r['player_id']): r for r in jsonl(ART / 'phase1q_snap_receipts.jsonl.gz') if r['head'] == 'rushing'}
    rows, excl, population, excluded_err = [], defaultdict(int), 0, []
    for r in jsonl(ART / 'phase1k_rushing_receipts.jsonl.gz'):
        if r['architecture'] != 'phase1f_incumbent':
            continue
        population += 1
        k = (r['season'], r['week'], r['player_id'])
        x = q.get(k)
        if x is None:
            excl['NOT_IN_PHASE1D_FIXED_MEANINGFUL_POPULATION_NO_FROZEN_SHARE_OR_TEAM_PROJECTION'] += 1
            excluded_err.append(abs(r['oracle_carry_rushing_yards_projection'] - r['actual_rushing_yards']))
            continue
        if x['frozen_team_opportunity_projection'] is None:
            excl['NO_FROZEN_TEAM_PROJECTION'] += 1
            continue
        carries = r['actual_carries']
        rows.append(Row(k, r['actual_rushing_yards'], ('TEAM_VOLUME', 'ROLE_SHARE', 'E'),
                        {'TEAM_VOLUME': x['frozen_team_opportunity_projection'], 'ROLE_SHARE': x['phase1d_share'], 'E': r['final_predicted_ypc']},
                        {'TEAM_VOLUME': x['actual_team_opportunities'], 'ROLE_SHARE': x['actual_share'], 'E': (r['actual_rushing_yards'] / carries) if carries > 0 else None},
                        lambda v: v['TEAM_VOLUME'] * v['ROLE_SHARE'] * v['E'],
                        {'position': r['position'], 'team': r['team'], 'week': r['week'], 'player': r['player'], 'actual_carries': carries, 'player_id': r['player_id']}))
    return sorted(rows, key=lambda x: x.key), {'population': population, 'excluded': dict(sorted(excl.items())), 'excluded_rows_normal_mae': mean(excluded_err) if excluded_err else None,
                                              'excluded_rows_metric_note': 'oracle-carry yards MAE of the incumbent on the excluded rows (carries known), shown so the exclusion is not hiding difficult rows'}


def tail_diagnostic(rushing):
    """Descriptive partition of the efficiency-yards error using the Phase1K explosive-family tier mixture and realized tier counts."""
    d = {}
    for r in jsonl(ART / 'phase1k_rushing_receipts.jsonl.gz'):
        if r['architecture'] == 'D_explosive':
            d[(r['season'], r['week'], r['player_id'])] = r
    inc = {}
    for r in jsonl(ART / 'phase1k_rushing_receipts.jsonl.gz'):
        if r['architecture'] == 'phase1f_incumbent':
            inc[(r['season'], r['week'], r['player_id'])] = r
    out, supported = {}, 0
    for x in rushing:
        r, di = inc[x.key], d.get(x.key)
        counts = (di or {}).get('oracle_metadata', {}).get('tail_counts') if di else None
        if not counts or di is None:
            continue
        p, mu = di['components']['tail']['probabilities'], di['components']['tail']['conditional_yields'] if 'conditional_yields' in di['components']['tail'] else di['components']['tail']['conditional_yards']
        n = float(sum(counts))
        if n <= 0:
            continue
        mix = (counts[1] - n * p[1]) * (mu[1] - mu[0]) + (counts[2] - n * p[2]) * (mu[2] - mu[0])
        within = r['actual_rushing_yards'] - sum(c * m for c, m in zip(counts, mu))
        tail_model_ypc = sum(pp * m for pp, m in zip(p, mu))
        diff = n * (tail_model_ypc - r['final_predicted_ypc'])
        eff_err = r['actual_rushing_yards'] - n * r['final_predicted_ypc']
        out[x.key] = {'efficiency_error_yards': eff_err, 'tail_mix_surprise_yards': mix, 'within_tier_residual_yards': within, 'tail_model_minus_incumbent_yards': diff}
        supported += 1
    return out, supported


# --------------------------------------------------------------------------- Week 4 burned forensic
def parse_week4_table(text):
    rows = []
    for line in text.splitlines():
        m = re.match(r'^\| ([A-Za-z.\' ]+) \| (rec yds|rush yds) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| (\d+) \|$', line.strip())
        if m:
            rows.append({'player': m.group(1), 'outcome': m.group(2), 'projection': float(m.group(3)), 'actual': float(m.group(4)), 'error': float(m.group(5)), 'expected_opportunities': float(m.group(6)), 'actual_opportunities': float(m.group(7))})
    return rows


def week4_receipts():
    proto = protocol()
    text = (ART / 'week4_system_pick_projection_findings.md').read_text()
    ids = proto['week4_player_ids']
    h_rows = {}
    for r in jsonl(ART / 'phase1h_receipts.jsonl.gz'):
        if r['architecture'] == 'depth_explosive' and r['period'] == 'diagnostic_2026_wk1_4' and r['week'] == 4:
            h_rows[r['player_id']] = r
    out = []
    for c in parse_week4_table(text):
        eff_p = c['projection'] / c['expected_opportunities']
        eff_a = c['actual'] / c['actual_opportunities']
        row = Row((c['player'],), c['actual'], ('O', 'E'), {'O': c['expected_opportunities'], 'E': eff_p}, {'O': c['actual_opportunities'], 'E': eff_a}, lambda v: v['O'] * v['E'])
        phi = row.shapley()
        e0 = row.err(())
        rec = {'player': c['player'], 'outcome': c['outcome'], 'projection': c['projection'], 'actual': c['actual'], 'signed_error_actual_minus_projection': c['actual'] - c['projection'], 'absolute_error': e0,
               'expected_opportunities': c['expected_opportunities'], 'actual_opportunities': c['actual_opportunities'], 'implied_projected_efficiency_per_opportunity': eff_p, 'realized_efficiency_per_opportunity': eff_a,
               'oracle_opportunity_error_yards': row.err(('O',)), 'oracle_efficiency_error_yards': row.err(('E',)), 'shapley_opportunity': phi['O'], 'shapley_efficiency': phi['E'],
               'best_single_oracle': 'ACTUAL_OPPORTUNITY' if row.err(('O',)) < row.err(('E',)) else 'ACTUAL_EFFICIENCY',
               'q1_team_opportunity_wrong': 'UNKNOWN: no frozen 2026 Week 4 team-volume receipt; player opportunity error is not split into team volume and role share',
               'q2_player_share_wrong': 'UNKNOWN_SPLIT: opportunity count differs by %.2f (expected %.2f, actual %.2f) without a team/role split' % (c['actual_opportunities'] - c['expected_opportunities'], c['expected_opportunities'], c['actual_opportunities']),
               'q3_efficiency_wrong': 'YES: projected %.2f vs realized %.2f per opportunity' % (eff_p, eff_a) if abs(eff_a - eff_p) > 0.25 * max(eff_p, 1e-9) else 'MODEST',
               'q4_explosive_tail_dominant': 'UNKNOWN: no frozen run-level or reception-level Week 4 labels',
               'q5_availability_deployment_different': 'UNKNOWN: no frozen Week 4 snap, route or participation receipt' + ('; documented in-game injury' if c['player'] in proto['week4_censored_players'] else ''),
               'component_receipt_from_phase1h_diagnostic': None}
        pid = ids.get(c['player'])
        h = h_rows.get(pid) if (pid and c['outcome'] == 'rec yds') else None
        if h and h['predicted_targets'] is not None and h['component_labels_reconciled']:
            t, rcv = h['targets'], h['receptions']
            hr = Row((c['player'],), h['receiving_yards'], ('T', 'C', 'A', 'Y'), {'T': h['predicted_targets'], 'C': h['phase1g_catch'], 'A': h['phase1g_air'], 'Y': h['phase1g_yac']},
                     {'T': float(t), 'C': (rcv / t) if t > 0 else None, 'A': (h['actual_completed_air'] / rcv) if rcv > 0 else None, 'Y': (h['actual_yac'] / rcv) if rcv > 0 else None}, lambda v: v['T'] * v['C'] * (v['A'] + v['Y']))
            hp = hr.shapley()
            rec['component_receipt_from_phase1h_diagnostic'] = {'note': 'Phase1H diagnostic pipeline (Phase1G components), not the T90 system-pick median; same player and game', 'predicted_targets': h['predicted_targets'], 'actual_targets': t, 'predicted_catch_rate': h['phase1g_catch'],
                                                               'actual_receptions': rcv, 'predicted_air_per_catch': h['phase1g_air'], 'actual_air_per_catch': hr.actual['A'], 'predicted_yac_per_catch': h['phase1g_yac'], 'actual_yac_per_catch': hr.actual['Y'],
                                                               'pipeline_projection': hr.value(()), 'pipeline_actual': h['receiving_yards'], 'shapley': hp, 'dominant_cause': dominant(hp, hr.err(()), hr.names)}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- human comparison (frozen receipts / snapshots)
def human_comparison():
    ch, e = read_json(ART / 'competent_human_snapshot.json')['outcomes'], read_json(ART / 'phase1e_integrated_snapshot.json')['outcomes']
    agg = {}
    for o in ('rec_yds', 'rec', 'rush_yds', 'pass_yds'):
        agg[o] = {p: {'system_mae': e[o][p]['mae'], 'human_mae': ch[o][p]['mae'], 'system_n': e[o][p]['n'], 'human_n': ch[o][p]['n'], 'system_bias': e[o][p]['bias'], 'human_bias': ch[o][p]['bias'],
                      'system_opportunity_mae': e[o][p]['player_opportunity_mae'], 'human_opportunity_mae': ch[o][p]['player_opportunity_mae'],
                      'system_better': e[o][p]['mae'] < ch[o][p]['mae']} for p in ('validation_2025', 'diagnostic_2026_wk1_4')}
    out = {'aggregate_frozen_phase1e_vs_competent_human': agg}
    # rushing efficiency, identical Phase1K rows
    inc, hum, tails = [], [], []
    for r in jsonl(ART / 'phase1k_rushing_receipts.jsonl.gz'):
        if r['architecture'] != 'phase1f_incumbent':
            continue
        n, y = r['actual_carries'], r['actual_rushing_yards']
        inc.append(abs(r['comparators']['phase1f_incumbent'] * n - y))
        hum.append(abs(r['comparators']['competent_human'] * n - y))
        tw = r['actual_rates']['twenty']
        tails.append(None if tw is None else tw * n >= 0.5)
    sl = {}
    for label, flag in (('has_20_plus_run', True), ('no_20_plus_run', False), ('no_carries_in_game', None)):
        idx = [i for i, t in enumerate(tails) if t == flag]
        sl[label] = {'n': len(idx), 'incumbent_oracle_carry_mae': mean(inc[i] for i in idx), 'human_oracle_carry_mae': mean(hum[i] for i in idx)}
    out['rushing_efficiency_phase1k_identical_rows'] = {'n': len(inc), 'incumbent_oracle_carry_mae': mean(inc), 'human_oracle_carry_mae': mean(hum), 'period': 'selection_2024_w9_18', 'slices_by_realized_tail': sl,
                                                       'note': 'oracle-carry yards = actual carries x forecast yards per carry; isolates efficiency'}
    # role share, Phase1O receipts (human role baseline vs frozen Phase1D) by pregame slice
    for head, tau, vac in (('receiving', 0.04, 0.10), ('rushing', 0.08, 0.15)):
        rows = [r for r in jsonl(ART / 'phase1o_role_receipts.jsonl.gz') if r['head'] == head]
        slices = defaultdict(lambda: {'d': [], 'h': []})
        for r in rows:
            e1 = r['phase1d_error_oracle']
            eh = abs(r['human_share'] * r['actual_team_opportunities'] - r['actual_opportunities'])
            tags = ['ALL']
            if r['accel'] >= tau:
                tags.append('INCREASING_ROLE')
            if r['accel'] <= -tau:
                tags.append('DECREASING_ROLE')
            if r['vacated_prior3_share'] >= vac:
                tags.append('TEAMMATE_VACANCY')
            if r['rookie']:
                tags.append('ROOKIE')
            if len(tags) == 1:
                tags.append('STABLE_ROLE')
            for t in tags:
                slices[t]['d'].append(e1)
                slices[t]['h'].append(eh)
        out[f'role_share_{head}_phase1o_receipts_2024'] = {k: {'n': len(v['d']), 'phase1d_oracle_mae': mean(v['d']), 'human_oracle_mae': mean(v['h']), 'human_better': mean(v['h']) < mean(v['d'])} for k, v in sorted(slices.items())}
    f = read_json(ART / 'phase1f_efficiency_snapshot.json')['diagnostics_2025']
    out['efficiency_swap_phase1f_2025'] = {o: {'phase1e_mae': f[o]['phase1e_mae'], 'human_eff_swap_mae': f[o]['human_eff_swap_mae']} for o in f}
    n = read_json(ART / 'phase1n_team_results.json')
    out['team_environment_phase1n_selection_2024'] = {'strongest_comparator': n['strongest_comparator'], 'human_recent3_vs_incumbent_note': 'C3_recent3_mean is the transparent human team baseline; C2 is the Phase1C-T offset-corrected incumbent'}
    return out


# --------------------------------------------------------------------------- stage assembly
def key_digest(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r.key).encode())
    return h.hexdigest()


def run():
    verify_inputs()
    proto = protocol()
    heads = {}
    cohorts = {}
    recy, c_recy = receiving_rows()
    heads['receiving_yards_decomposed'] = (recy, 'receiving_yards', c_recy, 'Phase1H receipts 2025 validation x Phase1G components (T x C x (A+Y))')
    h_source = list(jsonl(ART / 'phase1h_receipts.jsonl.gz'))
    heads['receiving_yards_incumbent_ypt'] = (incumbent_rows(recy, h_source), 'receiving_yards', dict(c_recy), 'same rows; Phase1F incumbent yards per target (T x E)')
    rcp, c_rcp = receptions_rows()
    heads['receptions'] = (rcp, 'receptions', c_rcp, 'Phase1H receipts 2025 validation (T x C)')
    tgt, c_tgt = opportunity_rows('receiving')
    heads['targets_2024_in_sample'] = (tgt, 'targets', c_tgt, 'Phase1Q receipts 2024 W1-18 (frozen Phase1B team projection x frozen Phase1D share)')
    car, c_car = opportunity_rows('rushing')
    heads['carries_2024_in_sample'] = (car, 'carries', c_car, 'Phase1Q receipts 2024 W1-18, RB/FB/HB (team projection x carry share)')
    rush, c_rush = rushing_rows()
    heads['rushing_yards_2024_in_sample'] = (rush, 'rushing_yards', c_rush, 'Phase1K incumbent efficiency x Phase1Q carry chain, 2024 W9-18')
    results, shap, catas = {}, {}, []
    for name, (rows, thr_key, c, desc) in heads.items():
        b, phis = budget(rows)
        b['description'] = desc
        results[name] = b
        shap[name] = {'method': 'exact Shapley over all 2^n replacement orders of the absolute-error reduction, averaged over identical rows', 'components': b['components'], 'n': b['n'],
                      'normal_mae': b['normal_mae'], 'all_oracle_mae': b['all_oracle_mae'], 'addressable_mae': b['addressable_mae'], 'shapley_mae_contribution': b['shapley_mae_contribution'],
                      'percent_of_addressable': b['shapley_percent_of_addressable'], 'single_oracle_gain': b['single_oracle_gain'], 'non_additivity': b['non_additivity'], 'shapley_sum_minus_addressable': b['shapley_sum_minus_addressable'],
                      'rows_with_negative_component': b['rows_with_negative_shapley_component'],
                      'caveat': 'DIAGNOSTIC attribution of the existing frozen projection error; not predictive lift and not additive MAE gains (MAE is nonlinear)'}
        cohorts[name] = {'description': desc, 'source_population_n': c['population'], 'common_cohort_n': len(rows), 'cohort_key_sha256': key_digest(rows), 'exclusions': c['excluded'],
                         'excluded_rows_normal_mae': c.get('excluded_rows_normal_mae'), 'excluded_rows_metric_note': c.get('excluded_rows_metric_note'),
                         'rows_with_undefined_actual_component': b['undefined_actual_components'], 'cohort_period': 'validation_2025' if '2025' in desc else '2024 (frozen-allocator selection years: in-sample)'}
        for thr in THRESHOLDS[thr_key]:
            pass
        lowest = THRESHOLDS[thr_key][0]
        for r, phi in zip(rows, phis):
            e0 = r.err(())
            if e0 > lowest:
                catas.append({'head': name, 'key': list(r.key), 'absolute_error': e0, 'signed_error_actual_minus_projection': r.y - r.value(()), 'exceeds': [t for t in THRESHOLDS[thr_key] if e0 > t],
                              'dominant_cause': dominant(phi, e0, r.names, labels=HEAD_LABELS.get(name)), 'shapley': phi, 'position': r.meta.get('position'), 'player_id': r.meta.get('player_id'), 'week': r.meta.get('week')})
    # excluded-but-resolvable rows are reported, never hidden: catastrophic misses without a component split are UNRESOLVED
    unresolved = []
    for r in jsonl(ART / 'phase1h_receipts.jsonl.gz'):
        if r['architecture'] == 'depth_explosive' and r['period'] == 'validation_2025' and (r['predicted_targets'] is None or not r['component_labels_reconciled']) and r['predicted_targets'] is not None:
            e0 = abs(r['predicted_targets'] * r['phase1g_ypt'] - r['receiving_yards'])
            if e0 > THRESHOLDS['receiving_yards'][0]:
                unresolved.append({'head': 'receiving_yards_decomposed', 'key': [r['season'], r['week'], r['player_id']], 'absolute_error': e0, 'signed_error_actual_minus_projection': r['receiving_yards'] - r['predicted_targets'] * r['phase1g_ypt'],
                                   'exceeds': [t for t in THRESHOLDS['receiving_yards'] if e0 > t], 'dominant_cause': 'UNRESOLVED', 'shapley': None, 'position': r['position'], 'player_id': r['player_id'], 'week': r['week'], 'reason': 'AIR_YAC_LABELS_NOT_RECONCILED'})
    catas += unresolved
    catas.sort(key=lambda x: (x['head'], tuple(x['key'])))
    # catastrophic summary per head and threshold
    summary = {}
    for name, (rows, thr_key, _c, _d) in heads.items():
        s = {}
        for t in THRESHOLDS[thr_key]:
            sel = [x for x in catas if x['head'] == name and x['absolute_error'] > t]
            cnt = defaultdict(int)
            for x in sel:
                cnt[x['dominant_cause']] += 1
            s[str(t)] = {'n_misses': len(sel), 'share_of_cohort': len(sel) / len(rows), 'dominant_cause_counts': dict(sorted(cnt.items())), 'mean_abs_error': mean(x['absolute_error'] for x in sel)}
        summary[name] = s
    tail, supported = tail_diagnostic(rush)
    cat_tail = defaultdict(int)
    tail_summary = {'rows_with_tier_labels': supported, 'rows_in_cohort': len(rush), 'note': 'descriptive partition of the efficiency-yards error (actual yards minus actual carries x incumbent YPC) using the Phase1K explosive-family tier probabilities and realized tier counts; not a promoted model'}
    if tail:
        big = [v for v in tail.values() if abs(v['efficiency_error_yards']) >= 20]
        tail_summary['rows_with_efficiency_error_ge_20_yards'] = len(big)
        tail_summary['mean_abs_efficiency_error_yards_all'] = mean(abs(v['efficiency_error_yards']) for v in tail.values())
        tail_summary['mean_abs_tail_mix_surprise_yards_all'] = mean(abs(v['tail_mix_surprise_yards']) for v in tail.values())
        tail_summary['mean_abs_within_tier_residual_yards_all'] = mean(abs(v['within_tier_residual_yards']) for v in tail.values())
        tail_summary['big_efficiency_misses_where_tail_mix_explains_ge_50_percent_same_sign'] = sum(1 for v in big if v['tail_mix_surprise_yards'] * v['efficiency_error_yards'] > 0 and abs(v['tail_mix_surprise_yards']) >= 0.5 * abs(v['efficiency_error_yards']))
    for x in catas:
        if x['head'] == 'rushing_yards_2024_in_sample' and x['dominant_cause'] == 'RUSH_EFFICIENCY':
            v = tail.get(tuple(x['key']))
            if v and v['efficiency_error_yards'] * v['tail_mix_surprise_yards'] > 0 and abs(v['tail_mix_surprise_yards']) >= DOMINANCE * abs(v['efficiency_error_yards']):
                x['dominant_cause'] = 'EXPLOSIVE_TAIL'
                x['tail_partition'] = v
            elif v:
                x['tail_partition'] = v
    for name in summary:                                      # recompute counts after the tail relabel
        thr_key = heads[name][1]
        for t in THRESHOLDS[thr_key]:
            sel = [x for x in catas if x['head'] == name and x['absolute_error'] > t]
            cnt = defaultdict(int)
            for x in sel:
                cnt[x['dominant_cause']] += 1
            summary[name][str(t)]['dominant_cause_counts'] = dict(sorted(cnt.items()))
    w4 = week4_receipts()
    human = human_comparison()
    return heads, results, shap, cohorts, catas, summary, tail_summary, w4, human


# --------------------------------------------------------------------------- value of information, stack, ceiling
CAT_THRESHOLD_INDEX = 1                                        # second preregistered threshold per head


def mapped_labels(head, comps):
    labels = {COMPONENT_LABEL[c] for c in comps}
    if head.startswith('rushing_yards') and 'E' in comps:
        labels.add('EXPLOSIVE_TAIL')
    return labels


def team_role_fraction(results):
    sh = results['targets_2024_in_sample']['shapley_mae_contribution']
    total = sh['TEAM_VOLUME'] + sh['ROLE_SHARE']
    return {'team': sh['TEAM_VOLUME'] / total, 'role': sh['ROLE_SHARE'] / total}


def information_value(heads, results, catas, summary):
    proto = protocol()
    frac = team_role_fraction(results)
    tcfg = proto['tiering']
    out, qb_note = [], read_json(ART / 'phase1f_efficiency_snapshot.json')['diagnostics_2025']['pass_yds']
    for fam in proto['information_families']:
        per_head = {}
        for head, cfg in fam['heads'].items():
            rows = heads[head][0]
            comps = tuple(cfg['components'])
            normal = mean(r.err(()) for r in rows)
            unsplit = normal - mean(r.err(comps) for r in rows)
            est = unsplit
            if 'T' in comps and cfg.get('t_part') in ('team', 'role'):
                rest = tuple(c for c in comps if c != 'T')
                base = normal - mean(r.err(rest) for r in rows) if rest else 0.0
                est = base + frac[cfg['t_part']] * (unsplit - base)
            sel = [x for x in catas if x['head'] == head and x['absolute_error'] > THRESHOLDS[heads[head][1]][CAT_THRESHOLD_INDEX]]
            labs = mapped_labels(head, comps)
            hit = sum(1 for x in sel if x['dominant_cause'] in labs)
            per_head[head] = {'components_replaced_by_realized': list(comps), 'normal_mae': normal, 'max_addressable_mae_upper_bound_unsplit': unsplit, 'estimated_addressable_mae': est, 'estimated_addressable_share_of_normal_mae': est / normal,
                              'unsplit_share_of_normal_mae': unsplit / normal, 'catastrophic_misses_at_threshold': len(sel), 'catastrophic_misses_with_mapped_dominant_cause': hit, 'catastrophic_threshold': THRESHOLDS[heads[head][1]][CAT_THRESHOLD_INDEX]}
        if fam.get('qb_aggregate_evidence'):
            gain = qb_note['phase1e_mae'] - qb_note['perfect_workload_current_eff_mae']
            per_head['qb_passing_yards_aggregate_phase1f'] = {'components_replaced_by_realized': ['attempt_workload (upper bound for starter state)'], 'normal_mae': qb_note['phase1e_mae'], 'max_addressable_mae_upper_bound_unsplit': gain, 'estimated_addressable_mae': gain,
                                                              'estimated_addressable_share_of_normal_mae': gain / qb_note['phase1e_mae'], 'unsplit_share_of_normal_mae': gain / qb_note['phase1e_mae'], 'evidence_level': 'POSTGAME_DIAGNOSTIC_ONLY_AGGREGATE',
                                                              'catastrophic_misses_at_threshold': None, 'catastrophic_misses_with_mapped_dominant_cause': None, 'catastrophic_threshold': None}
        best = max(per_head, key=lambda h: (per_head[h]['estimated_addressable_share_of_normal_mae'], h))
        share = per_head[best]['estimated_addressable_share_of_normal_mae']
        tier = tier_for(fam, share, tcfg)
        out.append({'id': fam['id'], 'name': fam['name'], 'tier': tier, 'best_evidenced_head': best, 'estimated_addressable_share_on_best_head': share,
                    'component_addressed': sorted({c for h in fam['heads'].values() for c in h['components']}), 'per_head': per_head, 'existing_source_status': fam['represented_note'], 'represented_by_failed_features': fam['represented_by_failed'],
                    'directly_measures': fam['directly_measures'], 'directly_measures_note': fam['direct_note'], 'timing_validity': fam['timing'], 'historical_availability': fam['historical_availability'], 'live_availability': fam['live_availability'],
                    'acquisition_blocker': fam['acquisition_blocker'], 'confidence_source_measures_required_thing': fam['measurement_confidence'], 'phase1j_rank': fam['phase1j_rank'], 'quantification': fam['quantification'],
                    'warning': 'UPPER BOUND on addressable error; not expected model improvement'})
    order = {'TIER_1_ACQUIRE_FIRST': 1, 'TIER_2_HIGH_VALUE_IF_ACCESSIBLE': 2, 'TIER_3_SECONDARY': 3, 'TIER_4_LOW_EXPECTED_VALUE': 4, 'DO_NOT_PURSUE': 5}
    out.sort(key=lambda x: (order[x['tier']], -x['estimated_addressable_share_on_best_head'], x['id']))
    for i, x in enumerate(out):
        x['rank'] = i + 1
    return {'schema': 'nfl-v2-phase1r-information-value-ranking-v1', 'status': 'UPPER_BOUNDS_NOT_EXPECTED_IMPROVEMENT', 'team_role_split_from_2024_targets_head': frac, 'tier_thresholds': tcfg['thresholds'], 'families': out,
            'qb_head_aggregate': {'status': 'POSTGAME_DIAGNOSTIC_ONLY_AGGREGATE_BLOCKED_STARTER_STATE_DATA', 'source': 'phase1f_efficiency_snapshot.json diagnostics_2025.pass_yds (frozen 2025 validation; row-level QB receipts do not exist)',
                                  'normal_mae': qb_note['phase1e_mae'], 'perfect_workload_current_efficiency_mae': qb_note['perfect_workload_current_eff_mae'], 'perfect_efficiency_current_workload_mae': qb_note['perfect_efficiency_current_workload_mae'],
                                  'workload_oracle_gain': qb_note['phase1e_mae'] - qb_note['perfect_workload_current_eff_mae'], 'efficiency_oracle_gain': qb_note['phase1e_mae'] - qb_note['perfect_efficiency_current_workload_mae'],
                                  'note': 'attempt workload is the larger share of the QB head error; whether starter identity explains it cannot be tested because expected-starter state is not certifiable pregame'}}


CONF_RANK = {'LOW': 0, 'MEDIUM': 1, 'MEDIUM_HIGH': 2, 'HIGH': 3}


def tier_for(fam, share, tcfg):
    th = tcfg['thresholds']
    if fam['timing'] == 'no' or not fam['directly_measures']:
        return 'DO_NOT_PURSUE'
    rep, path = fam['represented_by_failed'], fam['validation_path']
    if rep == 'FAILED_SAME_SIGNAL':
        return 'TIER_4_LOW_EXPECTED_VALUE'
    if share >= th['tier1_min_share'] and rep == 'NONE' and path == 'HISTORICAL_AND_LIVE_PLAUSIBLE' and CONF_RANK[fam['measurement_confidence']] >= CONF_RANK['MEDIUM_HIGH']:
        return 'TIER_1_ACQUIRE_FIRST'
    if path == 'NONE_IDENTIFIED':
        return 'TIER_3_SECONDARY' if share >= th['tier3_min_share'] else 'TIER_4_LOW_EXPECTED_VALUE'
    if share >= th['tier2_min_share'] and rep in ('NONE', 'PARTIAL_PROXY_FAILED') and path in ('HISTORICAL_AND_LIVE_PLAUSIBLE', 'FORWARD_ONLY'):
        return 'TIER_2_HIGH_VALUE_IF_ACCESSIBLE'
    if share >= th['tier3_min_share']:
        return 'TIER_3_SECONDARY'
    return 'TIER_4_LOW_EXPECTED_VALUE'


def surviving_stack():
    b = read_json(ART / 'phase1b_opportunity_snapshot.json')['outcomes']
    d = read_json(ART / 'phase1d_role_allocation_snapshot.json')['outcomes']
    e = read_json(ART / 'phase1e_integrated_snapshot.json')['outcomes']
    f = read_json(ART / 'phase1f_efficiency_snapshot.json')['diagnostics_2025']
    n = read_json(ART / 'phase1n_team_results.json')
    rows = [
        {'layer': 'TEAM_OPPORTUNITY', 'incumbent': 'Phase1B team-volume formula (Phase1C-T generic PBP environment frozen, not promoted)', 'status': n['verdict'] + ' (Phase1N); incumbent unchanged', 'validation_level': 'Phase1B 2025 validation (research, shadow)',
         'known_mae': {'receiving_head_team_targets_2025': b['rec_yds']['burned_validation_2025']['team_opportunity_mae'], 'rushing_head_team_carries_2025': b['rush_yds']['burned_validation_2025']['team_opportunity_mae'], 'phase1n_selection_2024_incumbent_plays_mae': n['strongest_comparator']['plays']['mae']},
         'main_blocker': 'genuinely new pregame information (QB state, availability); the accepted free pregame state added nothing', 'production_ready': False},
        {'layer': 'PLAYER_TARGET_SHARE', 'incumbent': 'Phase1D target-share allocator', 'status': 'SURVIVED (Phase1D); Phase1O and Phase1Q replacements rejected', 'validation_level': '2025 untouched validation + 2026 W1-4 burned diagnostic',
         'known_mae': {'role_share_mae_2025': d['rec_yds']['validation_2025']['role_share_mae'], 'oracle_target_mae_2025': d['rec_yds']['validation_2025']['oracle_player_opportunity_mae']}, 'main_blocker': 'route/availability information not accessible', 'production_ready': False},
        {'layer': 'PLAYER_CARRY_SHARE', 'incumbent': 'Phase1D carry-share allocator', 'status': 'SURVIVED (Phase1D); Phase1O and Phase1Q replacements rejected', 'validation_level': '2025 untouched validation + 2026 W1-4 burned diagnostic',
         'known_mae': {'role_share_mae_2025': d['rush_yds']['validation_2025']['role_share_mae'], 'oracle_carry_mae_2025': d['rush_yds']['validation_2025']['oracle_player_opportunity_mae']}, 'main_blocker': 'availability / committee information not accessible', 'production_ready': False},
        {'layer': 'RECEIVING_EFFICIENCY', 'incumbent': 'Phase1F/Phase1B yards-per-target (Phase1G decomposition kept for diagnostics)', 'status': 'FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION (Phase1G/H/I replacements rejected)', 'validation_level': '2025 validation (research)',
         'known_mae': {'yards_per_target_mae_2025': f['rec_yds']['current_eff_per_opp_mae'], 'final_receiving_yards_mae_2025': e['rec_yds']['validation_2025']['mae']}, 'main_blocker': 'tracking/charting information (separation, routes, defenders) with valid history and live timing', 'production_ready': False},
        {'layer': 'RUSHING_EFFICIENCY', 'incumbent': 'Phase1F/Phase1B yards-per-carry', 'status': 'REJECTED_EFFICIENCY_REPLACEMENT (Phase1K); incumbent only, not certified', 'validation_level': '2025 validation (research); Phase1K development 2024 only',
         'known_mae': {'yards_per_carry_mae_2025': f['rush_yds']['current_eff_per_opp_mae'], 'final_rushing_yards_mae_2025': e['rush_yds']['validation_2025']['mae']}, 'main_blocker': 'OL / box / blocking context; explosive tail is mostly realized variance', 'production_ready': False},
        {'layer': 'QB_OPPORTUNITY', 'incumbent': 'Phase1B QB attempt path (unchanged)', 'status': 'BLOCKED_STARTER_STATE_DATA (Phase1L); no source accepted (Phase1M)', 'validation_level': '2025 validation (research)',
         'known_mae': {'passing_yards_mae_2025': e['pass_yds']['validation_2025']['mae'], 'attempt_oracle_player_opportunity_mae_2025': e['pass_yds']['validation_2025']['player_opportunity_mae']}, 'main_blocker': 'certifiable pregame QB starter / health state', 'production_ready': False},
        {'layer': 'QB_EFFICIENCY', 'incumbent': 'Phase1B QB efficiency (unchanged)', 'status': 'NO_REPLACEMENT_TESTED (blocked behind QB state)', 'validation_level': '2025 validation (research)', 'known_mae': {'yards_per_attempt_mae_2025': f['pass_yds']['current_eff_per_opp_mae']},
         'main_blocker': 'QB assignment must be known before efficiency can be separated', 'production_ready': False},
        {'layer': 'MONTE_CARLO', 'incumbent': 'none for V2 (the V1 / Phase1C simulator is a benchmark only)', 'status': 'NOT_BUILT_FOR_V2: Monte Carlo must sit downstream of a promoted direct football projection', 'validation_level': 'none', 'known_mae': None,
         'main_blocker': 'no promoted direct projection chain', 'production_ready': False}]
    return {'schema': 'nfl-v2-phase1r-surviving-stack-v1', 'rule': 'Nothing is promoted because alternatives failed. Production remains the V1 engine; every V2 layer below is a research incumbent.', 'layers': rows}


def ceiling_verdict(results):
    verdict = {}
    for name in ('phase1h_routed_receiving_results.json', 'phase1i_target_depth_results.json', 'phase1k_rushing_results.json', 'phase1n_team_results.json'):
        verdict[name] = read_json(ART / name)['verdict']
    verdict['phase1o_role_results.json'] = read_json(ART / 'phase1o_role_results.json')['verdicts']['overall']
    verdict['phase1q_snap_results.json'] = read_json(ART / 'phase1q_snap_results.json')['verdicts']['overall']
    q = read_json(ART / 'phase1q_snap_results.json')['development_2024']
    best_gain = {}
    for h, v in q.items():
        best_gain[h] = max(x['vs']['C1']['gain'] for x in v['families'].values())
    o = read_json(ART / 'phase1o_role_results.json')['development_2024']
    best_gain_o = {h: max(x['vs']['C1']['gain'] for x in v['families'].values()) for h, v in o.items()}
    promoted = [k for k, v in verdict.items() if v.startswith('SURVIVES') or v.startswith('PARTIAL')]
    thr = {'receiving': 0.10, 'rushing': 0.20}
    below = all(best_gain[h] < thr[h] and best_gain_o[h] < thr[h] for h in thr)
    v = 'CURRENT_FREE_INFORMATION_EXHAUSTED_FOR_TESTED_HEADS' if (not promoted and below) else ('CURRENT_FREE_INFORMATION_NEAR_CEILING' if below else 'CURRENT_FREE_INFORMATION_NOT_EXHAUSTED')
    return {'verdict': v, 'phase_verdicts': verdict, 'phases_with_survival_or_partial': promoted, 'best_gain_vs_phase1d_phase1q': best_gain, 'best_gain_vs_phase1d_phase1o': best_gain_o, 'practical_thresholds': thr,
            'oracle_context': 'the error that remains is concentrated in components (target opportunity, catch/air, rushing efficiency tail) that the accepted free information did not move',
            'not_covered': ['QB head (BLOCKED_STARTER_STATE_DATA: untested, not exhausted)', 'information that is not historically testable (forward injury capture, depth charts 2025+, licensed charting)'],
            'one_signal_not_promoted': read_json(ART / 'phase1j_source_priority.json')['receiver_depth_signal']}


def assemble(out_dir):
    heads, results, shap, cohorts, catas, summary, tail, w4, human = run()
    proto = protocol()
    voi = information_value(heads, results, catas, summary)
    stack = surviving_stack()
    ceiling = ceiling_verdict(results)
    qb = voi['qb_head_aggregate']
    res = {'schema': 'nfl-v2-phase1r-error-budget-results-v1', 'base_head': proto['base_head'], 'verdict': 'PHASE1R_ERROR_BUDGET_COMPLETE_PARTIAL_DECOMPOSITION_QB_BLOCKED', 'free_information_ceiling': ceiling['verdict'], 'free_information_ceiling_detail': ceiling,
           'heads': results, 'catastrophic_summary': summary, 'rushing_tail_partition': tail, 'qb': {'decomposition': 'BLOCKED_STARTER_STATE_DATA', 'aggregate_diagnostic': qb}, 'human_comparison': human, 'week4_summary': {'cases': len(w4)},
           'caveats': ['oracle substitutions are diagnostic attributions of existing frozen projection error, not predictive lift', 'MAE gains from single oracles are not additive; exact Shapley contributions are reported with the non-additivity gap',
                       'targets, carries and rushing-yards cohorts are 2024 (frozen-allocator selection years) and optimistic for the frozen chain', 'receiving yards and receptions cohorts are 2025 validation receipts',
                       'availability, game script and QB state are not measurable from the receipts and were not inferred']}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'phase1r_component_cohorts.json').write_text(dump(rnd({'schema': 'nfl-v2-phase1r-component-cohorts-v1', 'rule': proto['fixed_cohort_rule'], 'cohorts': cohorts, 'qb': 'BLOCKED_STARTER_STATE_DATA'})))
    (out / 'phase1r_error_budget_results.json').write_text(dump(rnd(res)))
    (out / 'phase1r_shapley_attribution.json').write_text(dump(rnd({'schema': 'nfl-v2-phase1r-shapley-attribution-v1', 'heads': shap, 'status': 'EXACT_SHAPLEY_ON_AVAILABLE_COMPONENTS; QB PARTIAL_DECOMPOSITION_NOT_POSSIBLE'})))
    write_jsonl(out / 'phase1r_catastrophic_miss_attribution.jsonl.gz', catas)
    write_jsonl(out / 'phase1r_week4_forensic_receipts.jsonl.gz', w4)
    (out / 'phase1r_information_value_ranking.json').write_text(dump(rnd(voi)))
    (out / 'phase1r_surviving_stack.json').write_text(dump(rnd(stack)))
    return res, voi, stack, w4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    res, voi, stack, w4 = assemble(a.out_dir)
    print(res['verdict'], res['free_information_ceiling'], [(x['rank'], x['id'], x['tier']) for x in voi['families']])


if __name__ == '__main__':
    main()

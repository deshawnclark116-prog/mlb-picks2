#!/usr/bin/env python3
"""Phase1O-R role change / usage inflection engine (research only).

Workload only (receiver targets, RB/QB/skill carries). The candidate is a ridge delta on top of the frozen Phase1D allocation, renormalised over
the active eligible roster so shares always sum to 1. Estimation 2024 W1-8, selection 2024 W9-18, then (only if a family survives and the lock
passes) refit on 2024 W1-18 and an untouched 2025 validation. 2026 is never opened. No yards, no efficiency, no Monte Carlo, no sportsbook.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as D
import nfl_v2_phase1o_role_sources as S

ART = S.ART
HEADS = {'receiving': 'rec_yds', 'rushing': 'rush_yds'}
OPP = {'rec_yds': 'targets', 'rush_yds': 'carries'}
ABS_GAIN = {'rec_yds': 0.10, 'rush_yds': 0.20}
REL_GAIN = 0.05
CATASTROPHIC = {'rec_yds': 6.0, 'rush_yds': 10.0}
CAT_TOL = 0.005
TAU = {'rec_yds': 0.05, 'rush_yds': 0.10}
SIGMA0 = {'rec_yds': 0.04, 'rush_yds': 0.08}
INC_DEC = {'rec_yds': 0.04, 'rush_yds': 0.08}
VACANCY_FLAG = {'rec_yds': 0.10, 'rush_yds': 0.15}
COMMITTEE_TOP1 = {'rec_yds': 0.25, 'rush_yds': 0.55}
LAMBDAS = (10.0, 100.0, 1000.0)
SEED = 20261201
BOOT = 2000
FAMILIES = ('A_recency_acceleration', 'C_usage_concentration', 'D_teammate_vacancy', 'E_rookie_development', 'F_change_point', 'G_role_competition')
BLOCKED_FAMILIES = {'B_snap_change': 'BLOCKED_DATA: pfr_id to gsis mapping of skill snap rows 93.8% < 95% preregistered minimum'}
FEATURES = {
    'A_recency_acceleration': ['accel_last', 'accel_short_long', 'slope4', 'persist2'],
    'C_usage_concentration': ['top1', 'top2', 'hhi', 'rank1', 'rank2', 'share_x_hhi'],
    'D_teammate_vacancy': ['vac_total', 'vac_same', 'vac_other', 'vac_absorb', 'dropped_n'],
    'E_rookie_development': ['rookie', 'draft_r1', 'draft_r2', 'draft_r3', 'draft_later', 'career_games', 'rookie_x_accel'],
    'F_change_point': ['cp_z', 'cp_cusum', 'cp_flag', 'cp_newregime'],
    'G_role_competition': ['mate_acc_sum', 'mate_rise_n', 'mate_acc_rank'],
}
CODE_FILES = ('nfl_v2_phase1o_role_change.py', 'nfl_v2_phase1o_role_sources.py')
STAGES = ('develop', 'validate', 'burned2026')


# --------------------------------------------------------------------------- arithmetic helpers (explicit left-to-right)
def mean(xs):
    xs = list(xs)
    if not xs:
        return None
    return S.ordered_mean(xs)


def ewma(xs, decay):
    if not xs:
        return None
    n = len(xs)
    num = den = 0.0
    for i, x in enumerate(xs):
        w = decay ** (n - 1 - i)
        num += x * w
        den += w
    return num / den


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    t = 0.0
    for x in xs:
        t += (x - m) ** 2
    return t / len(xs)


def slope(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = mean(xs)
    num = den = 0.0
    for i, y in enumerate(xs):
        num += (i - xm) * (y - ym)
        den += (i - xm) ** 2
    return num / den


def rnd(value, nd=6):
    if isinstance(value, float):
        return round(value, nd) if value == value else None
    if isinstance(value, dict):
        return {k: rnd(v, nd) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rnd(v, nd) for v in value]
    return value


def code_sha256():
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((S.ROOT / name).read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- data context
class Context:
    """Pinned inputs for a set of seasons plus the frozen Phase1D / Phase1B configs."""

    def __init__(self, data_dir, years):
        S.verify(data_dir, years)
        self.years = years
        self.players, self.team_totals, self.team_opp = p1a.load_stats(S.paths(data_dir, years, 'stats'))
        p1a.build_indexes(self.players, self.team_totals, self.team_opp)
        p1b.build_extra_indexes(self.players, self.team_totals, self.team_opp)
        D.build_context(self.players, self.team_totals)
        D.load_rosters(S.paths(data_dir, years, 'roster'))
        self.meta = S.player_meta(data_dir, years)
        mapping, _ = S.pfr_to_gsis(data_dir, years)
        self.snaps, _ = S.load_snaps(data_dir, years, mapping)
        d1 = S.read_json(ART / 'phase1d_role_allocation_snapshot.json')['outcomes']
        b1 = S.read_json(ART / 'phase1b_opportunity_snapshot.json')['outcomes']
        self.d_cfg = {o: d1[o]['selected_config'] for o in HEADS.values()}
        self.b_cfg = {o: b1[o]['selected_opportunity_config'] for o in HEADS.values()}
        self.by_tw = defaultdict(dict)
        self.team_weeks = defaultdict(set)
        for r in self.players:
            self.by_tw[(r['season'], r['week'], r['team'])][r['player_id']] = r
            self.team_weeks[r['team']].add((r['season'], r['week']))
        self.team_weeks = {t: sorted(v) for t, v in self.team_weeks.items()}
        self._tw = {}
        self._proj = {}

    def prior_team_weeks(self, team, target, n):
        out = [k for k in self.team_weeks.get(team, ()) if k < target]
        return out[-n:]

    def shares(self, target, pid, team, outcome, window):
        return D.prior_share_series(self.players, self.team_totals, target, pid, outcome, window, team)

    def career_games(self, target, pid):
        return len(p1a.prior_player_rows(self.players, target, pid, None, 10 ** 6))

    def snap_trend(self, target, pid, team):
        vals = []
        for tw in self.prior_team_weeks(team, target, 3):
            v = self.snaps.get((tw[0], tw[1], team, pid))
            if v is not None:
                vals.append(v)
        if len(vals) < 2:
            return 0.0
        return vals[-1] - mean(vals[-3:-1])

    # ----- per team-week feature build (pure pregame)
    def team_week(self, season, week, team, outcome):
        key = (season, week, team, outcome)
        if key in self._tw:
            return self._tw[key]
        rec = self._build_tw(season, week, team, outcome)
        self._tw[key] = rec
        return rec

    def _build_tw(self, season, week, team, outcome):
        target = (season, week)
        alloc = D.allocation(self.players, self.team_totals, {'season': season, 'week': week, 'team': team}, outcome, self.d_cfg[outcome])
        if alloc is None:
            return None
        roster = D.eligible_roster(season, week, team, outcome)
        pids = [x['player_id'] for x in roster]
        pos = {x['player_id']: (x.get('position') or '').upper() for x in roster}
        sig, tau_inc = SIGMA0[outcome], INC_DEC[outcome]
        hist = {p: self.shares(target, p, team, outcome, 8) for p in pids}
        m3 = {p: (mean(h[-3:]) if h else 0.0) for p, h in hist.items()}
        have = {p: len(h) for p, h in hist.items()}
        d1 = np.array([alloc['predicted_shares'][p] for p in pids])

        # team concentration over the eligible group
        ordered = sorted(m3.values(), reverse=True)
        top1 = ordered[0] if ordered else 0.0
        top2 = top1 + (ordered[1] if len(ordered) > 1 else 0.0)
        hhi = 0.0
        for v in ordered:
            hhi += v * v
        rank = {}
        for i, p in enumerate(sorted(pids, key=lambda p: (-m3[p], p))):
            rank[p] = i + 1

        # teammates seen in the last three team-weeks (vacancy / drops)
        last3 = self.prior_team_weeks(team, target, 3)
        seen = {}
        for tw in last3:
            for pid2 in self.by_tw.get((tw[0], tw[1], team), {}):
                seen.setdefault(pid2, None)
        absent = [p for p in seen if p not in set(pids)]
        absent_m3 = {}
        for p in absent:
            h3 = self.shares(target, p, team, outcome, 3)
            absent_m3[p] = mean(h3) if h3 else 0.0
        absent_pos = {}
        for p in absent:
            for tw in last3:
                row = self.by_tw.get((tw[0], tw[1], team), {}).get(p)
                if row is not None:
                    absent_pos[p] = (row.get('position') or '').upper()
        vac_total = 0.0
        for p in absent:
            vac_total += absent_m3[p]
        dropped = 0
        if len(last3) >= 2:
            lw, pw = last3[-1], last3[-2]
            den_pw = self.team_totals.get((pw[0], pw[1], team), {}).get(OPP[outcome], 0.0)
            for p, r in self.by_tw.get((pw[0], pw[1], team), {}).items():
                if den_pw > 0 and r.get(OPP[outcome], 0.0) / den_pw >= 0.10 and p not in self.by_tw.get((lw[0], lw[1], team), {}):
                    dropped += 1

        accel = {}
        for p in pids:
            h = hist[p]
            accel[p] = (h[-1] - ewma(h[:-1], 0.85)) if len(h) >= 2 else 0.0
        feats = {n: [] for names in FEATURES.values() for n in names}
        labels = {'accel': [], 'vac_total': vac_total, 'rookie': [], 'top1_group': []}
        for p in pids:
            h = hist[p]
            ps = pos[p]
            a1 = accel[p]
            f = {}
            f['accel_last'] = a1
            f['accel_short_long'] = (ewma(h, 0.5) - ewma(h, 0.85)) if len(h) >= 2 else 0.0
            f['slope4'] = slope(h[-4:])
            earlier6 = h[:-2][-6:]
            f['persist2'] = 1.0 if (len(h) >= 3 and h[-1] >= mean(earlier6) + 0.02 and h[-2] >= mean(earlier6) + 0.02) else 0.0
            f['top1'], f['top2'], f['hhi'] = top1, top2, hhi
            f['rank1'], f['rank2'] = float(rank[p] == 1), float(rank[p] == 2)
            f['share_x_hhi'] = m3[p] * hhi
            f['vac_total'] = vac_total
            same = 0.0
            for q in absent:
                if absent_pos[q] == ps:
                    same += absent_m3[q]
            f['vac_same'], f['vac_other'] = same, vac_total - same
            rest = 0.0
            for q in pids:
                rest += m3[q]
            f['vac_absorb'] = vac_total * m3[p] / max(rest, 1e-6)
            f['dropped_n'] = float(dropped)
            meta = self.meta.get((season, p), {})
            rookie = meta.get('years_exp') == 0
            dn = meta.get('draft_number')
            f['rookie'] = float(rookie)
            f['draft_r1'] = float(dn is not None and dn <= 32)
            f['draft_r2'] = float(dn is not None and 32 < dn <= 64)
            f['draft_r3'] = float(dn is not None and 64 < dn <= 100)
            f['draft_later'] = float(dn is not None and dn > 100)
            f['career_games'] = float(min(self.career_games(target, p), 40))
            f['rookie_x_accel'] = f['rookie'] * a1
            if len(earlier6) >= 2 and len(h) >= 3:
                mu = mean(earlier6)
                last2 = mean(h[-2:])
                z = (last2 - mu) / math.sqrt(sig * sig + variance(earlier6))
                s = 0.0
                for x in h[-4:]:
                    s = max(0.0, s + (x - mu - 0.01))
                flag = 1.0 if (z >= 1.5 and h[-1] >= mu + 0.02 and h[-2] >= mu + 0.02) else 0.0
                f['cp_z'], f['cp_cusum'], f['cp_flag'], f['cp_newregime'] = z, s, flag, flag * (last2 - ewma(h, 0.85))
            else:
                f['cp_z'] = f['cp_cusum'] = f['cp_flag'] = f['cp_newregime'] = 0.0
            mates = [q for q in pids if q != p and pos[q] == ps]
            f['mate_acc_sum'] = sum_ordered(accel[q] for q in mates)
            f['mate_rise_n'] = float(sum(1 for q in mates if accel[q] >= tau_inc))
            f['mate_acc_rank'] = sum_ordered(accel[q] / rank[q] for q in mates)
            for n in feats:
                feats[n].append(f[n])
            labels['accel'].append(a1)
            labels['rookie'].append(bool(rookie))
            group = sorted((m3[q] for q in pids if pos[q] == ps), reverse=True)
            labels['top1_group'].append(group[0] if group else 0.0)
        rec = {'season': season, 'week': week, 'team': team, 'pids': pids, 'pos': pos, 'd1': d1, 'n_hist': np.array([have[p] for p in pids]),
               'feats': {n: np.array(v) for n, v in feats.items()}, 'm3': {p: m3[p] for p in pids}, 'hist': hist,
               'labels': {'accel': np.array(labels['accel']), 'vac_total': vac_total, 'rookie': np.array(labels['rookie']), 'top1_group': np.array(labels['top1_group']), 'hhi': hhi},
               'index': {p: i for i, p in enumerate(pids)}}
        rec['c3'] = self._human(rec, target, team, outcome)
        rec['actual_team'] = self.team_totals.get((season, week, team), {}).get(OPP[outcome], 0.0)
        return rec

    def _human(self, rec, target, team, outcome):
        raw = []
        for p in rec['pids']:
            h = rec['hist'][p]
            if not h:
                pp = D.position_share_prior(target, rec['pos'][p], outcome)
                raw.append(0.15 * (pp or 0.0))
                continue
            m3 = rec['m3'][p]
            earlier = h[:-2][-6:]
            persist = 0.0
            if len(h) >= 3 and earlier and min(h[-2:]) >= mean(earlier) + 0.03:
                persist = mean(h[-2:]) - mean(earlier)
            trend = max(-0.3, min(0.3, self.snap_trend(target, p, team)))
            raw.append(m3 * (1.0 + 0.5 * trend) + 0.5 * max(0.0, persist))
        total = 0.0
        for v in raw:
            total += v
        raw = np.array(raw)
        return raw / total if total > 0 else rec['d1'].copy()


def sum_ordered(xs):
    t = 0.0
    for x in xs:
        t += x
    return t


# --------------------------------------------------------------------------- ridge model
def matrix(rec, names):
    return np.column_stack([rec['feats'][n] for n in names])


def fit(ctx, outcome, names, lam, team_weeks):
    xs, ys = [], []
    for tw in team_weeks:
        rec = ctx.team_week(*tw, outcome)
        if rec is None or rec['actual_team'] <= 0:
            continue
        keep = rec['n_hist'] >= 1
        if not keep.any():
            continue
        actual = np.array([ctx.by_tw[(tw[0], tw[1], tw[2])].get(p, {}).get(OPP[outcome], 0.0) / rec['actual_team'] for p in rec['pids']])
        xs.append(matrix(rec, names)[keep])
        ys.append((actual - rec['d1'])[keep])
    x = np.vstack(xs)
    y = np.concatenate(ys)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return {'names': names, 'lam': lam, 'mu': mu, 'sd': sd, 'beta': beta, 'intercept': float(ym), 'n_fit': int(len(y))}


def predict(model, rec):
    z = (matrix(rec, model['names']) - model['mu']) / model['sd']
    delta = model['intercept'] + z @ model['beta']
    delta = np.where(rec['n_hist'] >= 1, delta, 0.0)
    raw = np.maximum(0.0, rec['d1'] + delta)
    total = raw.sum()
    return raw / total if total > 0 else rec['d1'].copy()


# --------------------------------------------------------------------------- evaluation
def primary_rows(ctx, outcome, weeks):
    rows = D.fixed_rows(ctx.players, p1a.target_rows(ctx.players, weeks), outcome)
    out = []
    for r in sorted(rows, key=lambda r: (r['season'], r['week'], r['team'], r['player_id'])):
        rec = ctx.team_week(r['season'], r['week'], r['team'], outcome)
        if rec is None or r['player_id'] not in rec['index'] or rec['actual_team'] <= 0:
            continue
        out.append(r)
    return out


def comparators(rec, i, model=None):
    out = {'C1': float(rec['d1'][i]), 'C3': float(rec['c3'][i])}
    h = rec['hist'][rec['pids'][i]]
    out['C2'] = mean(h[-3:]) if len(h) >= 2 else float(rec['d1'][i])
    return out


def role_rows(ctx, outcome, rows, models):
    """Per primary player-game: actual, comparators, each model's predicted share, slice labels."""
    out = []
    cache = {}
    for r in rows:
        rec = ctx.team_week(r['season'], r['week'], r['team'], outcome)
        i = rec['index'][r['player_id']]
        k = (r['season'], r['week'], r['team'])
        if k not in cache:
            cache[k] = {name: predict(m, rec) for name, m in models.items()}
        t = rec['actual_team']
        a = r.get(OPP[outcome], 0.0)
        h = rec['hist'][r['player_id']]
        ref = mean(h[-3:])
        rec_out = {'r': r, 'rec': rec, 'i': i, 'team_opp': t, 'actual': a, 'a_share': a / t, 'ref': ref,
                   'pred': {'C1': float(rec['d1'][i]), 'C3': float(rec['c3'][i]), 'C2': comparators(rec, i)['C2']}}
        for name in models:
            rec_out['pred'][name] = float(cache[k][name][i])
        out.append(rec_out)
    return out


def terciles(ctx, outcome, team_weeks):
    vals = []
    for tw in team_weeks:
        rec = ctx.team_week(*tw, outcome)
        if rec is not None:
            vals.append(rec['labels']['hhi'])
    vals.sort()
    return vals[len(vals) // 3], vals[(2 * len(vals)) // 3]


def slices_for(row, outcome, tercs):
    rec, i = row['rec'], row['i']
    lab = rec['labels']
    t = INC_DEC[outcome]
    flags = {
        'INCREASING_ROLE': bool(lab['accel'][i] >= t),
        'DECREASING_ROLE': bool(lab['accel'][i] <= -t),
        'TEAMMATE_VACANCY': bool(lab['vac_total'] >= VACANCY_FLAG[outcome]),
        'ROOKIE_DEVELOPMENT': bool(lab['rookie'][i]),
        'COMMITTEE': bool(lab['top1_group'][i] < COMMITTEE_TOP1[outcome]),
        'HIGH_CONCENTRATION': bool(lab['hhi'] >= tercs[1]),
        'LOW_CONCENTRATION': bool(lab['hhi'] <= tercs[0]),
    }
    union = flags['INCREASING_ROLE'] or flags['DECREASING_ROLE'] or flags['TEAMMATE_VACANCY'] or flags['ROOKIE_DEVELOPMENT']
    flags['CHANGE_FLAGGED_UNION'] = union
    flags['STABLE_ROLE'] = not union
    return flags


def metrics(rows, outcome, name):
    s_err, o_err, bias = [], [], []
    for x in rows:
        p = x['pred'][name]
        s_err.append(abs(p - x['a_share']))
        bias.append(p - x['a_share'])
        o_err.append(abs(p * x['team_opp'] - x['actual']))
    if not rows:
        return {'n': 0}
    cat = CATASTROPHIC[outcome]
    srt = sorted(o_err)
    return {'n': len(rows), 'share_mae': mean(s_err), 'share_bias': mean(bias), 'oracle_mae': mean(o_err), 'median_ae': srt[len(srt) // 2],
            'catastrophic_rate': mean([1.0 if e > cat else 0.0 for e in o_err])}


def block_bootstrap(rows, outcome, cand, comp):
    """Upper-95 of mean(candidate_err - comparator_err); resamples moving 2-week blocks of complete team-weeks."""
    weeks = sorted({(x['r']['season'], x['r']['week']) for x in rows})
    blocks = [weeks[i:i + 2] for i in range(len(weeks) - 1)] or [weeks]
    stats = []
    for b in blocks:
        s, n = 0.0, 0
        for x in rows:
            if (x['r']['season'], x['r']['week']) in b:
                s += abs(x['pred'][cand] * x['team_opp'] - x['actual']) - abs(x['pred'][comp] * x['team_opp'] - x['actual'])
                n += 1
        stats.append((s, n))
    rng = random.Random(SEED)
    per = max(1, math.ceil(len(weeks) / 2))
    reps = []
    for _ in range(BOOT):
        s = n = 0
        for _ in range(per):
            a, b = stats[rng.randrange(len(stats))]
            s += a
            n += b
        reps.append(s / n if n else 0.0)
    reps.sort()
    return reps[int(0.95 * BOOT) - 1]


def judge(rows, outcome, cand, comp):
    c, b = metrics(rows, outcome, cand), metrics(rows, outcome, comp)
    gain = b['oracle_mae'] - c['oracle_mae']
    need = max(ABS_GAIN[outcome], REL_GAIN * b['oracle_mae'])
    upper = block_bootstrap(rows, outcome, cand, comp)
    cat_ok = c['catastrophic_rate'] <= b['catastrophic_rate'] + CAT_TOL
    return {'comparator': comp, 'candidate_oracle_mae': c['oracle_mae'], 'comparator_oracle_mae': b['oracle_mae'], 'gain': gain, 'required_gain': need,
            'relative_gain': gain / b['oracle_mae'], 'share_mae_relative_gain': (b['share_mae'] - c['share_mae']) / b['share_mae'],
            'bootstrap_upper95_diff': upper, 'catastrophic_candidate': c['catastrophic_rate'], 'catastrophic_comparator': b['catastrophic_rate'],
            'material_gain': gain >= need, 'qualifies': bool(gain >= need and upper < 0 and cat_ok)}


def role_change(rows, outcome, name):
    tau = TAU[outcome]
    tp = pred_n = act_n = tp_up = pred_up = act_up = 0
    for x in rows:
        if len(x['rec']['hist'][x['r']['player_id']]) < 2:
            continue
        pd, ad = x['pred'][name] - x['ref'], x['a_share'] - x['ref']
        pu, pdn = pd >= tau / 2, pd <= -tau / 2
        au, adn = ad >= tau, ad <= -tau
        pred_n += pu or pdn
        act_n += au or adn
        pred_up += pu
        act_up += au
        tp += (pu and au) or (pdn and adn)
        tp_up += pu and au
    prec = tp / pred_n if pred_n else None
    return {'tau': tau, 'predicted_flags': int(pred_n), 'actual_changes': int(act_n), 'true_flags': int(tp), 'precision': prec,
            'recall': tp / act_n if act_n else None, 'false_promotion_rate': (1 - prec) if prec is not None else None,
            'up': {'predicted': int(pred_up), 'actual': int(act_up), 'true': int(tp_up), 'precision': tp_up / pred_up if pred_up else None,
                   'recall': tp_up / act_up if act_up else None, 'false_promotion_rate': (1 - tp_up / pred_up) if pred_up else None}}


def slice_report(rows, outcome, tercs, cand):
    out = {}
    tagged = [(x, slices_for(x, outcome, tercs)) for x in rows]
    for label in ('INCREASING_ROLE', 'DECREASING_ROLE', 'TEAMMATE_VACANCY', 'ROOKIE_DEVELOPMENT', 'COMMITTEE', 'HIGH_CONCENTRATION', 'LOW_CONCENTRATION', 'CHANGE_FLAGGED_UNION', 'STABLE_ROLE'):
        sub = [x for x, f in tagged if f[label]]
        if len(sub) < 5:
            out[label] = {'n': len(sub)}
            continue
        m = {k: metrics(sub, outcome, k) for k in ('C1', 'C2', 'C3', cand)}
        strongest = min(('C1', 'C2', 'C3'), key=lambda k: m[k]['oracle_mae'])
        out[label] = {'n': len(sub), 'candidate_oracle_mae': m[cand]['oracle_mae'], 'strongest_comparator': strongest, 'strongest_oracle_mae': m[strongest]['oracle_mae'],
                      'relative_gain_vs_strongest': (m[strongest]['oracle_mae'] - m[cand]['oracle_mae']) / m[strongest]['oracle_mae'], 'C1_oracle_mae': m['C1']['oracle_mae']}
    return out


def slice_rule(sl):
    u, s = sl.get('CHANGE_FLAGGED_UNION', {}), sl.get('STABLE_ROLE', {})
    ok_u = u.get('n', 0) >= 5 and u['relative_gain_vs_strongest'] >= 0.05
    ok_s = s.get('n', 0) < 5 or s['relative_gain_vs_strongest'] >= -0.02
    return bool(ok_u and ok_s)


def coherence(ctx, outcome, models, team_weeks):
    worst = 0.0
    for tw in team_weeks:
        rec = ctx.team_week(*tw, outcome)
        if rec is None:
            continue
        for m in models.values():
            worst = max(worst, abs(float(predict(m, rec).sum()) - 1.0), float((predict(m, rec) < 0).any()))
        worst = max(worst, abs(float(rec['d1'].sum()) - 1.0), abs(float(rec['c3'].sum()) - 1.0))
    return worst


def weeks_of(year, lo, hi):
    return [(year, w) for w in range(lo, hi + 1)]


def team_weeks_of(ctx, weeks):
    wanted = set(weeks)
    return sorted(k for k in ctx.by_tw if (k[0], k[1]) in wanted)


# --------------------------------------------------------------------------- forensics (postgame descriptive only)
def forensics(ctx, outcome, rows, cand):
    cat = {'role_error_mae': [], 'team_volume_error_mae': [], 'by_actual_snap_bucket': defaultdict(list), 'late_absent': defaultdict(list)}
    comp = {'C1': []}
    for x in rows:
        r, rec = x['r'], x['rec']
        p = x['pred'][cand]
        cat['role_error_mae'].append(abs(p * x['team_opp'] - x['actual']))
        proj = ctx._proj.get((r['season'], r['week'], r['team'], r['player_id'], outcome))
        if proj is None:
            proj = D.phase1b_team_projection(ctx.players, ctx.team_totals, ctx.team_opp, r, outcome, ctx.b_cfg[outcome])
            ctx._proj[(r['season'], r['week'], r['team'], r['player_id'], outcome)] = proj
        if proj is not None:
            cat['team_volume_error_mae'].append(abs(proj * x['a_share'] - x['actual']))
        snap = ctx.snaps.get((r['season'], r['week'], r['team'], r['player_id']))
        bucket = 'missing' if snap is None else ('lt25' if snap < 0.25 else ('25_60' if snap < 0.60 else 'ge60'))
        cat['by_actual_snap_bucket'][bucket].append(abs(p * x['team_opp'] - x['actual']))
        late = False
        for q in rec['pids']:
            if q != r['player_id'] and rec['m3'][q] >= 0.10 and q not in ctx.by_tw[(r['season'], r['week'], r['team'])]:
                late = True
        cat['late_absent'][str(late)].append(abs(p * x['team_opp'] - x['actual']))
    return {'status': 'POSTGAME DESCRIPTIVE ONLY; never used for fitting, selection or promotion',
            'role_error_oracle_mae': mean(cat['role_error_mae']), 'team_volume_error_mae': mean(cat['team_volume_error_mae']),
            'by_actual_snap_bucket': {k: {'n': len(v), 'oracle_mae': mean(v)} for k, v in sorted(cat['by_actual_snap_bucket'].items())},
            'expected_teammate_absent_from_target_game': {k: {'n': len(v), 'oracle_mae': mean(v)} for k, v in sorted(cat['late_absent'].items())}}


# --------------------------------------------------------------------------- stages
def develop(data_dir, out_dir):
    ctx = Context(data_dir, (2023, 2024))
    est_weeks, sel_weeks = weeks_of(2024, 1, 8), weeks_of(2024, 9, 18)
    results, lock_heads, receipts = {}, {}, []
    for head, outcome in HEADS.items():
        est_tw = team_weeks_of(ctx, est_weeks)
        tercs = terciles(ctx, outcome, est_tw)
        rows = primary_rows(ctx, outcome, sel_weeks)
        fams, models_by_fam = {}, {}
        for fam in FAMILIES:
            best = None
            for lam in LAMBDAS:
                model = fit(ctx, outcome, FEATURES[fam], lam, est_tw)
                rr = role_rows(ctx, outcome, rows, {'cand': model})
                m = metrics(rr, outcome, 'cand')
                key = (m['oracle_mae'], lam)
                if best is None or key < best[0]:
                    best = (key, lam, model, rr, m)
            _, lam, model, rr, m = best
            judged = {c: judge(rr, outcome, 'cand', c) for c in ('C1', 'C3')}
            fams[fam] = {'selected_lambda': lam, 'oracle_mae': m['oracle_mae'], 'share_mae': m['share_mae'], 'n': m['n'], 'vs': judged,
                         'survives': bool(all(j['qualifies'] for j in judged.values())), 'material_gain_vs_phase1d': bool(judged['C1']['material_gain'])}
            models_by_fam[fam] = model
        survivors = [f for f in FAMILIES if fams[f]['survives']]
        combo = None
        if len(survivors) >= 2:
            names = [n for f in survivors for n in FEATURES[f]]
            best = None
            for lam in LAMBDAS:
                model = fit(ctx, outcome, names, lam, est_tw)
                rr = role_rows(ctx, outcome, rows, {'cand': model})
                m = metrics(rr, outcome, 'cand')
                if best is None or (m['oracle_mae'], lam) < best[0]:
                    best = ((m['oracle_mae'], lam), lam, model, rr, m)
            _, lam, model, rr, m = best
            judged = {c: judge(rr, outcome, 'cand', c) for c in ('C1', 'C3')}
            combo = {'families': survivors, 'selected_lambda': lam, 'oracle_mae': m['oracle_mae'], 'vs': judged, 'survives': bool(all(j['qualifies'] for j in judged.values()))}
            models_by_fam['H_survivor_combination'] = model
        # comparator context on the selection period (every family candidate reported against all comparators)
        all_models = {f: models_by_fam[f] for f in models_by_fam}
        rr_all = role_rows(ctx, outcome, rows, all_models)
        comp = {k: metrics(rr_all, outcome, k) for k in ('C1', 'C2', 'C3')}
        contenders = [(f, fams[f]['oracle_mae']) for f in survivors] + ([('H_survivor_combination', combo['oracle_mae'])] if combo and combo['survives'] else [])
        best_name = min(contenders, key=lambda t: (t[1], t[0]))[0] if contenders else min(((f, fams[f]['oracle_mae']) for f in FAMILIES), key=lambda t: (t[1], t[0]))[0]
        passed = bool(contenders)
        material = any(fams[f]['material_gain_vs_phase1d'] for f in FAMILIES)
        verdict = None if passed else ('PARTIAL_SIGNAL_NOT_PROMOTED' if material else 'REJECTED_ROLE_CHANGE_REPLACEMENT')
        best_model = models_by_fam[best_name]
        best_rr = role_rows(ctx, outcome, rows, {'cand': best_model})
        results[head] = {'outcome': outcome, 'population': 'Phase1D fixed meaningful rows, 2024 W9-18', 'n': len(rows), 'comparators': comp, 'families': fams,
                         'combination': combo, 'blocked_families': BLOCKED_FAMILIES, 'survivors': survivors, 'development_pass': passed,
                         'development_verdict_if_failed': verdict, 'best_candidate': best_name, 'terciles': list(tercs),
                         'role_change_best': {k: role_change(best_rr, outcome, k) for k in ('cand', 'C1', 'C3')},
                         'slices_best': slice_report(best_rr, outcome, tercs, 'cand'),
                         'coherence_max_abs_error': coherence(ctx, outcome, {'best': best_model}, team_weeks_of(ctx, sel_weeks)),
                         'forensics_best': forensics(ctx, outcome, best_rr, 'cand')}
        lock_heads[head] = {'outcome': outcome, 'best_candidate': best_name, 'candidate_features': list(best_model['names']), 'selected_lambda': best_model['lam'],
                            'development_pass': passed, 'survivors': survivors, 'terciles': list(tercs)}
        # receipts: all 2024 primary rows of the best candidate (model frozen from W1-8; period flagged)
        all24 = primary_rows(ctx, outcome, weeks_of(2024, 1, 18))
        rr24 = role_rows(ctx, outcome, all24, {'cand': best_model})
        for x in rr24:
            r, rec, i = x['r'], x['rec'], x['i']
            p = x['pred']
            receipts.append(rnd({'head': head, 'season': r['season'], 'week': r['week'], 'period': 'estimation_W1_8' if r['week'] <= 8 else 'selection_W9_18', 'player_id': r['player_id'], 'team': r['team'],
                                 'opponent': r['opponent'], 'position': r.get('position'), 'eligible_teammates': len(rec['pids']), 'ref_share_prior3': x['ref'], 'last_share': (rec['hist'][r['player_id']] or [None])[-1],
                                 'accel': float(rec['labels']['accel'][i]), 'vacated_prior3_share': float(rec['labels']['vac_total']), 'rookie': bool(rec['labels']['rookie'][i]), 'hhi': float(rec['labels']['hhi']),
                                 'actual_team_opportunities': x['team_opp'], 'actual_opportunities': x['actual'], 'actual_share': x['a_share'], 'phase1d_share': p['C1'], 'human_share': p['C3'],
                                 'candidate_share': p['cand'], 'candidate_error_oracle': abs(p['cand'] * x['team_opp'] - x['actual']), 'phase1d_error_oracle': abs(p['C1'] * x['team_opp'] - x['actual']),
                                 'family': best_name, 'source_sha256': {k: v for k, v in S.read_json(ART / 'phase1o_role_source_audit.json')['source_sha256']['2024'].items()}}))
    gate = any(h['development_pass'] for h in lock_heads.values())
    lock = {'schema': 'nfl-v2-phase1o-development-lock-v1', 'base_head': S.BASE_HEAD, 'code_sha256': code_sha256(), 'protocol_sha256': S.sha(ART / S.PROTOCOL),
            'source_audit_sha256': S.sha(ART / 'phase1o_role_source_audit.json'), 'heads': lock_heads, 'development_gate_passed': gate,
            'validation_2025_allowed': gate, 'burned_2026': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'seed': SEED, 'bootstrap_reps': BOOT,
            'no_rescue_tuning': True}
    out_dir = Path(out_dir)
    (out_dir / 'phase1o_development_lock.json').write_text(S.dump(rnd(lock)))
    (out_dir / 'phase1o_role_results.json').write_text(S.dump(rnd({'schema': 'nfl-v2-phase1o-role-results-v1', 'development_2024': results})))
    with gzip.GzipFile(filename='', mode='wb', fileobj=open(out_dir / 'phase1o_role_receipts.jsonl.gz', 'wb'), mtime=0) as g:
        for line in receipts:
            g.write((json.dumps(line, sort_keys=True) + '\n').encode())
    human = {'schema': 'nfl-v2-phase1o-competent-human-role-v1', 'definition': S.read_json(ART / S.PROTOCOL)['comparators']['C3_competent_human'],
             'frozen_before_candidate': True, 'snap_note': 'uses offense snap trend where the exact id join exists (missing -> 0); a conservative (stronger) comparator than the blocked family B',
             'development_2024_W9_18': {h: results[h]['comparators']['C3'] for h in results}, 'phase1d_2024_W9_18': {h: results[h]['comparators']['C1'] for h in results}}
    (out_dir / 'phase1o_competent_human_role.json').write_text(S.dump(rnd(human)))
    return results, lock


def validate(data_dir, out_dir):
    out_dir = Path(out_dir)
    lock = S.read_json(out_dir / 'phase1o_development_lock.json')
    if not lock.get('development_gate_passed'):
        raise SystemExit('development gate did not pass: 2025 validation is refused')
    if lock['code_sha256'] != code_sha256():
        raise SystemExit('code changed since the development lock')
    ctx = Context(data_dir, (2023, 2024, 2025))
    fit_weeks, val_weeks = weeks_of(2024, 1, 18), weeks_of(2025, 1, 18)
    results = {}
    for head, h in lock['heads'].items():
        outcome = h['outcome']
        if not h['development_pass']:
            results[head] = {'verdict': 'NOT_VALIDATED_DEVELOPMENT_FAILED', 'development_verdict': None}
            continue
        model = fit(ctx, outcome, h['candidate_features'], h['selected_lambda'], team_weeks_of(ctx, fit_weeks))
        rows = primary_rows(ctx, outcome, val_weeks)
        rr = role_rows(ctx, outcome, rows, {'cand': model})
        tercs = tuple(h['terciles'])
        judged = {c: judge(rr, outcome, 'cand', c) for c in ('C1', 'C3')}
        judged['C2'] = judge(rr, outcome, 'cand', 'C2')
        sl = slice_report(rr, outcome, tercs, 'cand')
        passed = bool(judged['C1']['qualifies'] and judged['C3']['qualifies'] and slice_rule(sl))
        material = bool(judged['C1']['material_gain'])
        results[head] = {'outcome': outcome, 'candidate': h['best_candidate'], 'n': len(rr), 'comparators': {k: metrics(rr, outcome, k) for k in ('C1', 'C2', 'C3', 'cand')},
                         'vs': judged, 'slices': sl, 'slice_rule_passed': slice_rule(sl), 'role_change': {k: role_change(rr, outcome, k) for k in ('cand', 'C1', 'C3')},
                         'coherence_max_abs_error': coherence(ctx, outcome, {'cand': model}, team_weeks_of(ctx, val_weeks)), 'forensics': forensics(ctx, outcome, rr, 'cand'),
                         'validation_pass': passed, 'verdict': 'SURVIVES_ROLE_CHANGE_REPLACEMENT' if passed else ('PARTIAL_SIGNAL_NOT_PROMOTED' if material else 'REJECTED_ROLE_CHANGE_REPLACEMENT')}
    path = out_dir / 'phase1o_role_results.json'
    data = S.read_json(path)
    data['validation_2025'] = rnd(results)
    path.write_text(S.dump(data))
    return results


def burned2026(out_dir):
    """Close-out: records the 2026 / 2025 / downstream not-run states and the per-head and overall verdicts."""
    out_dir = Path(out_dir)
    lock = S.read_json(out_dir / 'phase1o_development_lock.json')
    path = out_dir / 'phase1o_role_results.json'
    data = S.read_json(path)
    data['burned_2026_w1_4'] = {'status': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'reason': 'exact pinned 2026 upstream bytes are no longer served (see phase1h/1i/1k archival status); no replacement data are used'}
    if 'validation_2025' not in data:
        data['validation_2025'] = {'status': 'NOT_RUN_DEVELOPMENT_GATE_FAILED' if not lock['development_gate_passed'] else 'NOT_RUN'}
    validation = data['validation_2025']
    verdicts = {}
    for head, v in data['development_2024'].items():
        if lock['heads'][head]['development_pass'] and isinstance(validation.get(head), dict):
            verdicts[head] = validation[head]['verdict']
        else:
            verdicts[head] = v['development_verdict_if_failed'] or 'BLOCKED_DATA'
    order = ['SURVIVES_ROLE_CHANGE_REPLACEMENT', 'PARTIAL_SIGNAL_NOT_PROMOTED', 'REJECTED_ROLE_CHANGE_REPLACEMENT', 'BLOCKED_DATA']
    data['verdicts'] = {'per_head': verdicts, 'overall': min(verdicts.values(), key=order.index), 'note': 'heads are independent; overall is the best head verdict',
                        'blocked_families': BLOCKED_FAMILIES}
    qualifies = [h for h, x in verdicts.items() if x == 'SURVIVES_ROLE_CHANGE_REPLACEMENT']
    data['downstream_phase1e'] = {'status': 'NOT_RUN_NO_HEAD_QUALIFIED' if not qualifies else 'REQUIRED_FOR_' + '_'.join(qualifies)}
    path.write_text(S.dump(rnd(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=STAGES, required=True)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out-dir', default=str(ART))
    a = ap.parse_args()
    if a.stage == 'develop':
        develop(a.data_dir, a.out_dir)
    elif a.stage == 'validate':
        validate(a.data_dir, a.out_dir)
    else:
        burned2026(a.out_dir)


if __name__ == '__main__':
    main()

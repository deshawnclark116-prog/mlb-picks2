#!/usr/bin/env python3
"""Phase1Q-SNAP snap-share change / deployment inflection engine (research only).

Workload only. The candidate is a ridge delta on the frozen Phase1D share built from PRIOR lag-eligible offensive snap deployment (never the
target-game snaps), renormalised so the non-QB mass is preserved. Estimation 2024 W1-8, selection 2024 W9-18, then (only if a family survives and
the lock passes) refit on 2024 W1-18 and an untouched 2025 validation. 2026 is never opened. No yards, efficiency, Monte Carlo or sportsbook.
It reuses (does not modify) the frozen Phase1D allocator, Phase1B team projection, and the Phase1O evaluation harness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import nfl_v2_phase1d_role_allocation as D
import nfl_v2_phase1n_team_sources as N
import nfl_v2_phase1o_role_change as R
import nfl_v2_phase1o_role_sources as O
import nfl_v2_phase1p_identity_sources as I
import nfl_v2_phase1p_snap_mapping as M

ART = O.ART
PROTOCOL = 'phase1q_snap_protocol.json'
SCHEDULE_2025 = 'phase1q_frozen_schedule_2025.json'
HEADS = R.HEADS
LAG_DAYS = 4
WINDOW = 5
K_EXCHANGE = {'rec_yds': 3.0, 'rush_yds': 1.5}
FLAG = 0.15
HEAD_POSITIONS = {'rec_yds': None, 'rush_yds': {'RB', 'FB', 'HB'}}
GROUP = {'FB': 'RB', 'HB': 'RB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE'}
FAMILIES = ('A_snap_level', 'B_snap_acceleration', 'C_snap_regime', 'D_snap_opportunity_divergence', 'E_teammate_redistribution')
FEATURES = {
    'A_snap_level': ['snap_last', 'snap_m3', 'snap_m5', 'snap_ewma_short', 'snap_ewma_long'],
    'B_snap_acceleration': ['acc_last_m3', 'acc_m2_m5', 'acc_ewma', 'snap_slope'],
    'C_snap_regime': ['reg_up_low_mid', 'reg_up_mid_full', 'reg_full_persist', 'reg_rapid_decline', 'reg_consec_full'],
    'D_snap_opportunity_divergence': ['div_level', 'div_gain'],
    'E_teammate_redistribution': ['mate_gain', 'mate_loss', 'group_hhi_m3', 'committee_compression', 'own_minus_mates'],
}
CODE_FILES = ('nfl_v2_phase1q_snap_role.py',)
STAGES = ('develop', 'validate', 'burned2026')
POS_GROUPS = ('WR', 'TE', 'RB')


# --------------------------------------------------------------------------- helpers
def clip(x, lo, hi):
    return max(lo, min(hi, x))


def code_sha256():
    h = hashlib.sha256()
    for name in CODE_FILES + R.CODE_FILES + M.CODE_FILES:
        h.update(name.encode())
        h.update((O.ROOT / name).read_bytes())
    return h.hexdigest()


def protocol():
    return O.read_json(ART / PROTOCOL)


def schedule_dates(years):
    """{(season, week, team): date} from the frozen, digest-verified schedule artifacts (dates only; no outcomes)."""
    rows = list(N.frozen_schedule())
    if 2025 in years:
        doc = O.read_json(ART / SCHEDULE_2025)
        if N.schedule_digest(doc['rows'], {2025}) != doc['canonical_rows_sha256'] or any(set(r) != set(N.GAMES_ALLOWLIST) for r in doc['rows']):
            raise ValueError('frozen 2025 schedule changed')
        rows += doc['rows']
    out = {}
    for g in rows:
        d = datetime.date.fromisoformat(g['gameday'])
        for team in (g['home_team'], g['away_team']):
            out[(int(g['season']), int(g['week']), team)] = d
    return out


def renorm(raw, d1, fixed):
    """Preserve the Phase1D mass of the fixed (QB) players, rescale the rest to the Phase1D non-fixed mass."""
    free = ~fixed
    mass = d1[free].sum()
    cur = raw[free].sum()
    out = raw.copy()
    if cur > 0:
        out[free] = raw[free] * (mass / cur)
    else:
        out = d1.copy()
    return out


def series_features(ss, share_hist, share_m3, k):
    """Snap features from a lag-eligible snap series (oldest -> newest) and the player's stat-share history."""
    f = dict.fromkeys([n for v in FEATURES.values() for n in v], 0.0)
    n = len(ss)
    if n == 0:
        return f, 0.0
    s1, m3, m5 = ss[-1], R.mean(ss[-3:]), R.mean(ss[-5:])
    f['snap_last'], f['snap_m3'], f['snap_m5'] = s1, m3, m5
    f['snap_ewma_short'], f['snap_ewma_long'] = R.ewma(ss, 0.5), R.ewma(ss, 0.85)
    acc = 0.0
    if n >= 2:
        prev3 = R.mean(ss[:-1][-3:])
        acc = s1 - prev3
        f['acc_last_m3'] = s1 - m3
        f['acc_m2_m5'] = R.mean(ss[-2:]) - m5
        f['acc_ewma'] = f['snap_ewma_short'] - f['snap_ewma_long']
        f['snap_slope'] = R.slope(ss[-5:])
        f['reg_up_low_mid'] = float(prev3 < 0.25 and s1 >= 0.25)
        f['reg_up_mid_full'] = float(0.25 <= prev3 < 0.60 and s1 >= 0.60)
        f['reg_full_persist'] = float(ss[-1] >= 0.60 and ss[-2] >= 0.60)
        f['reg_rapid_decline'] = float(s1 - prev3 <= -0.20)
    c = 0
    for v in reversed(ss):
        if v >= 0.60:
            c += 1
        else:
            break
    f['reg_consec_full'] = float(min(c, 5))
    f['div_level'] = m3 - k * share_m3
    if n >= 2 and len(share_hist) >= 2:
        f['div_gain'] = acc - k * (share_hist[-1] - R.mean(share_hist[:-1][-3:]))
    return f, acc


# --------------------------------------------------------------------------- context
class SnapContext(R.Context):
    """Phase1O context plus lag-eligible snap series with Phase1P exact identity mapping."""

    def __init__(self, data_dir, years):
        super().__init__(data_dir, years)
        canon, _, _ = I.canonical_map()
        roster, blank = I.roster_crosswalk(data_dir, years)
        rows = list(I.snap_rows(data_dir, years))
        items = M.classify([r for r in rows if r['position'] in M.POSITIONS], canon, roster, blank)
        self.snap_pct, self.snap_prov = {}, {}
        for x in items:
            if x['gsis']:
                r = x['row']
                key = (int(r['season']), int(r['week']), r['team'], x['gsis'])
                self.snap_pct[key] = float(r['offense_pct'])
                self.snap_prov[key] = (r['pfr_player_id'], x['rule'])
        self.mapping = {str(y): M.summarize([x for x in items if x['row']['position'] in I.PRIMARY_POSITIONS and int(x['row']['season']) == y]) for y in years}
        self.dates = schedule_dates(years)
        self.team_games = defaultdict(list)
        for (s, w, team), d in sorted(self.dates.items(), key=lambda kv: (kv[1], kv[0])):
            self.team_games[team].append(((s, w), d))

    def eligible_games(self, team, target):
        d_t = self.dates[(target[0], target[1], team)]
        out = [k for k, d in self.team_games[team] if k < target and (d_t - d).days >= LAG_DAYS]
        return out[-WINDOW:]

    def snap_series(self, games, team, pid):
        vals = [self.snap_pct.get((g[0], g[1], team, pid)) for g in games]
        start = next((i for i, v in enumerate(vals) if v is not None), None)
        return [] if start is None else [v if v is not None else 0.0 for v in vals[start:]]

    def _build_tw(self, season, week, team, outcome):
        rec = super()._build_tw(season, week, team, outcome)
        if rec is None:
            return None
        target = (season, week)
        games = self.eligible_games(team, target)
        pids, pos = rec['pids'], rec['pos']
        k = K_EXCHANGE[outcome]
        series = {p: self.snap_series(games, team, p) for p in pids}
        feats = {n: [] for v in FEATURES.values() for n in v}
        acc = {}
        base = {}
        for p in pids:
            base[p], acc[p] = series_features(series[p], rec['hist'][p], rec['m3'][p], k)
        groups = defaultdict(list)
        for p in pids:
            if series[p] and pos[p] != 'QB':
                groups[GROUP.get(pos[p], pos[p])].append(p)

        def hhi(vals):
            tot = 0.0
            for v in vals:
                tot += v
            if tot <= 0:
                return 0.0
            h = 0.0
            for v in vals:
                h += (v / tot) ** 2
            return h

        for p in pids:
            f = dict(base[p])
            g = GROUP.get(pos[p], pos[p])
            mates = [q for q in groups.get(g, ()) if q != p] if series[p] else []
            f['mate_gain'] = R.sum_ordered(max(0.0, acc[q]) for q in mates)
            f['mate_loss'] = R.sum_ordered(min(0.0, acc[q]) for q in mates)
            members = groups.get(g, [])
            if series[p] and members:
                f['group_hhi_m3'] = hhi([base[q]['snap_m3'] for q in members])
                f['committee_compression'] = hhi([base[q]['snap_last'] for q in members]) - hhi([base[q]['snap_m5'] for q in members])
                f['own_minus_mates'] = acc[p] - (R.mean([acc[q] for q in mates]) if mates else 0.0)
            for n in feats:
                feats[n].append(f[n])
        for n, v in feats.items():
            rec['feats'][n] = np.array(v)
        for n in [n for v in FEATURES.values() for n in v]:           # F: position interactions (built for every feature; only survivors are used)
            for grp in POS_GROUPS:
                rec['feats'][f'{n}__{grp}'] = rec['feats'][n] * np.array([float(GROUP.get(pos[p]) == grp) for p in pids])
        rec['snap_n'] = np.array([len(series[p]) for p in pids])
        rec['snap_acc'] = np.array([acc[p] for p in pids])
        rec['is_qb'] = np.array([pos[p] == 'QB' for p in pids])
        rec['snap_series'] = series
        rec['c3s'] = self._human_snap(rec)
        return rec

    @staticmethod
    def _human_snap(rec):
        raw = rec['d1'].copy()
        for i, p in enumerate(rec['pids']):
            ss = rec['snap_series'][p]
            if rec['is_qb'][i] or len(ss) < 3:
                continue
            last2, earlier = R.mean(ss[-2:]), R.mean(ss[:-2][-3:])
            persist = 0.0
            if all(v - earlier >= 0.10 for v in ss[-2:]) or all(earlier - v >= 0.10 for v in ss[-2:]):
                persist = last2 - earlier
            raw[i] = rec['d1'][i] * (1.0 + 1.0 * clip(persist, -0.3, 0.3))
        return renorm(raw, rec['d1'], rec['is_qb'])


# --------------------------------------------------------------------------- model
def fit(ctx, outcome, names, lam, team_weeks):
    xs, ys = [], []
    for tw in team_weeks:
        rec = ctx.team_week(*tw, outcome)
        if rec is None or rec['actual_team'] <= 0:
            continue
        keep = (rec['n_hist'] >= 1) & (rec['snap_n'] >= 1) & (~rec['is_qb'])
        if not keep.any():
            continue
        actual = np.array([ctx.by_tw[tw].get(p, {}).get(R.OPP[outcome], 0.0) / rec['actual_team'] for p in rec['pids']])
        xs.append(R.matrix(rec, names)[keep])
        ys.append((actual - rec['d1'])[keep])
    x, y = np.vstack(xs), np.concatenate(ys)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return {'names': names, 'lam': lam, 'mu': mu, 'sd': sd, 'beta': beta, 'intercept': float(ym), 'n_fit': int(len(y))}


def predict(model, rec):
    z = (R.matrix(rec, model['names']) - model['mu']) / model['sd']
    delta = model['intercept'] + z @ model['beta']
    use = (rec['n_hist'] >= 1) & (rec['snap_n'] >= 1) & (~rec['is_qb'])
    raw = np.where(use, np.maximum(0.0, rec['d1'] + delta), rec['d1'])
    return renorm(raw, rec['d1'], rec['is_qb'])


def head_rows(ctx, outcome, weeks):
    rows = R.primary_rows(ctx, outcome, weeks)
    allowed = HEAD_POSITIONS[outcome]
    return [r for r in rows if allowed is None or (r.get('position') or '').upper() in allowed]


def role_rows(ctx, outcome, rows, models):
    out, cache = [], {}
    for r in rows:
        rec = ctx.team_week(r['season'], r['week'], r['team'], outcome)
        i = rec['index'][r['player_id']]
        k = (r['season'], r['week'], r['team'])
        if k not in cache:
            cache[k] = {name: predict(m, rec) for name, m in models.items()}
        t = rec['actual_team']
        a = r.get(R.OPP[outcome], 0.0)
        h = rec['hist'][r['player_id']]
        x = {'r': r, 'rec': rec, 'i': i, 'team_opp': t, 'actual': a, 'a_share': a / t, 'ref': R.mean(h[-3:]),
             'pred': {'C1': float(rec['d1'][i]), 'C3': float(rec['c3s'][i]), 'C2': R.comparators(rec, i)['C2']}}
        for name in models:
            x['pred'][name] = float(cache[k][name][i])
        out.append(x)
    return out


def snap_flags(x):
    rec, i = x['rec'], x['i']
    a = rec['snap_acc'][i] if rec['snap_n'][i] >= 2 else 0.0
    return {'SNAP_ROLE_UP': bool(a >= FLAG), 'SNAP_ROLE_DOWN': bool(a <= -FLAG), 'SNAP_ROLE_STABLE': bool(abs(a) < FLAG),
            'SNAP_CHANGED': bool(abs(a) >= FLAG), 'HAS_SNAP_SERIES': bool(rec['snap_n'][i] >= 1)}


def slice_report(rows, outcome, cand):
    out = {}
    tagged = [(x, snap_flags(x)) for x in rows]
    for label in ('SNAP_ROLE_UP', 'SNAP_ROLE_DOWN', 'SNAP_ROLE_STABLE', 'SNAP_CHANGED', 'HAS_SNAP_SERIES'):
        sub = [x for x, f in tagged if f[label]]
        if len(sub) < 5:
            out[label] = {'n': len(sub)}
            continue
        m = {k: R.metrics(sub, outcome, k) for k in ('C1', 'C2', 'C3', cand)}
        strongest = min(('C1', 'C2', 'C3'), key=lambda k: m[k]['oracle_mae'])
        out[label] = {'n': len(sub), 'candidate_oracle_mae': m[cand]['oracle_mae'], 'phase1d_oracle_mae': m['C1']['oracle_mae'], 'strongest_comparator': strongest,
                      'strongest_oracle_mae': m[strongest]['oracle_mae'], 'relative_gain_vs_strongest': (m[strongest]['oracle_mae'] - m[cand]['oracle_mae']) / m[strongest]['oracle_mae'],
                      'gain_vs_phase1d': m['C1']['oracle_mae'] - m[cand]['oracle_mae']}
    return out


def slice_rule(sl):
    u, s = sl.get('SNAP_CHANGED', {}), sl.get('SNAP_ROLE_STABLE', {})
    ok_u = u.get('n', 0) >= 5 and u['relative_gain_vs_strongest'] >= 0.05
    ok_s = s.get('n', 0) < 5 or s['relative_gain_vs_strongest'] >= -0.02
    return bool(ok_u and ok_s)


def snap_flag_report(rows, outcome):
    """Precision / recall of the PREGAME snap flags against actual share change vs the prior-3 share reference."""
    tau = R.TAU[outcome]
    out = {'tau': tau, 'rows': len(rows)}
    actual_up = sum(1 for x in rows if x['a_share'] - x['ref'] >= tau)
    actual_dn = sum(1 for x in rows if x['a_share'] - x['ref'] <= -tau)
    for label, sign, total in (('SNAP_ROLE_UP', 1, actual_up), ('SNAP_ROLE_DOWN', -1, actual_dn)):
        flagged = [x for x in rows if snap_flags(x)[label]]
        tp = sum(1 for x in flagged if sign * (x['a_share'] - x['ref']) >= tau)
        out[label] = {'flagged': len(flagged), 'actual_changes_in_population': total, 'true_positives': tp, 'precision': tp / len(flagged) if flagged else None,
                      'recall': tp / total if total else None, 'false_promotion_rate': (1 - tp / len(flagged)) if flagged else None}
        if flagged:
            out[label]['phase1d_oracle_mae'] = R.metrics(flagged, outcome, 'C1')['oracle_mae']
            out[label]['human_oracle_mae'] = R.metrics(flagged, outcome, 'C3')['oracle_mae']
    return out


def coherence(ctx, outcome, models, team_weeks):
    worst = 0.0
    for tw in team_weeks:
        rec = ctx.team_week(*tw, outcome)
        if rec is None:
            continue
        mass = rec['d1'][~rec['is_qb']].sum()
        for m in list(models.values()) + [None]:
            out = predict(m, rec) if m is not None else rec['c3s']
            worst = max(worst, abs(float(out[~rec['is_qb']].sum() - mass)), float((out < 0).any()), float(np.abs(out[rec['is_qb']] - rec['d1'][rec['is_qb']]).max(initial=0.0)))
    return worst


def forensics(ctx, outcome, rows, cand):
    snap_bucket, snap_cat = defaultdict(list), defaultdict(list)
    role, teamvol = [], []
    up_flat = up_n = 0
    for x in rows:
        r, rec, i = x['r'], x['rec'], x['i']
        p = x['pred'][cand]
        err = abs(p * x['team_opp'] - x['actual'])
        role.append(err)
        key = (r['season'], r['week'], r['team'], r['player_id'], outcome)
        proj = ctx._proj.get(key)
        if proj is None:
            proj = D.phase1b_team_projection(ctx.players, ctx.team_totals, ctx.team_opp, r, outcome, ctx.b_cfg[outcome])
            ctx._proj[key] = proj
        if proj is not None:
            teamvol.append(abs(proj * x['a_share'] - x['actual']))
        tgt = ctx.snap_pct.get((r['season'], r['week'], r['team'], r['player_id']))          # POSTGAME ONLY
        snap_bucket['missing' if tgt is None else ('lt25' if tgt < 0.25 else ('25_60' if tgt < 0.60 else 'ge60'))].append(err)
        ss = rec['snap_series'][r['player_id']]
        if tgt is not None and ss:
            ch = tgt - R.mean(ss[-3:])
            cat = 'TARGET_GAME_SNAP_UP' if ch >= FLAG else ('TARGET_GAME_SNAP_DOWN' if ch <= -FLAG else 'TARGET_GAME_SNAP_STABLE')
            snap_cat[cat].append(err)
            if ch >= FLAG:
                up_n += 1
                up_flat += (x['a_share'] - x['ref']) < R.TAU[outcome]
    summ = lambda d: {k: {'n': len(v), 'oracle_mae': R.mean(v)} for k, v in sorted(d.items())}
    return {'status': 'POSTGAME DESCRIPTIVE ONLY; never used for fitting, selection or promotion', 'role_error_oracle_mae': R.mean(role), 'team_volume_error_mae': R.mean(teamvol),
            'by_actual_target_game_snap_bucket': summ(snap_bucket), 'by_actual_target_game_snap_change': summ(snap_cat),
            'target_game_snap_up_but_share_did_not_rise': {'snap_up_rows': up_n, 'share_did_not_rise_by_tau': int(up_flat)}}


# --------------------------------------------------------------------------- stages
def pick_lambda(ctx, outcome, names, est_tw, rows):
    best = None
    for lam in R.LAMBDAS:
        model = fit(ctx, outcome, names, lam, est_tw)
        rr = role_rows(ctx, outcome, rows, {'cand': model})
        m = R.metrics(rr, outcome, 'cand')
        if best is None or (m['oracle_mae'], lam) < best[0]:
            best = ((m['oracle_mae'], lam), lam, model, rr, m)
    return best[1:]


def judged(rr, outcome):
    return {c: R.judge(rr, outcome, 'cand', c) for c in ('C1', 'C3')}


def develop(data_dir, out_dir):
    ctx = SnapContext(data_dir, (2023, 2024))
    est_weeks, sel_weeks = R.weeks_of(2024, 1, 8), R.weeks_of(2024, 9, 18)
    results, lock_heads, receipts = {}, {}, []
    for head, outcome in HEADS.items():
        est_tw = R.team_weeks_of(ctx, est_weeks)
        rows = head_rows(ctx, outcome, sel_weeks)
        fams, models = {}, {}
        for fam in FAMILIES:
            lam, model, rr, m = pick_lambda(ctx, outcome, FEATURES[fam], est_tw, rows)
            j = judged(rr, outcome)
            fams[fam] = {'selected_lambda': lam, 'oracle_mae': m['oracle_mae'], 'share_mae': m['share_mae'], 'n': m['n'], 'vs': j, 'survives': bool(all(x['qualifies'] for x in j.values())),
                         'material_gain_vs_phase1d': bool(j['C1']['material_gain']), 'n_fit': model['n_fit']}
            models[fam] = model
        survivors = [f for f in FAMILIES if fams[f]['survives']]
        f_family = {'status': 'NOT_TESTED_NO_BASIC_SNAP_FAMILY_SURVIVED'}
        if survivors:
            fit_rows = {g: 0 for g in POS_GROUPS}
            for tw in est_tw:
                rec = ctx.team_week(*tw, outcome)
                if rec is None:
                    continue
                use = (rec['n_hist'] >= 1) & (rec['snap_n'] >= 1) & (~rec['is_qb'])
                for g in POS_GROUPS:
                    fit_rows[g] += int(sum(1 for i, p in enumerate(rec['pids']) if use[i] and GROUP.get(rec['pos'][p]) == g))
            groups = [g for g in POS_GROUPS if fit_rows[g] >= 30]
            if len(groups) < 2:
                f_family = {'status': 'NOT_TESTED_NO_POSITION_VARIATION', 'fit_rows_by_position_group': fit_rows}
            else:
                names = [f'{n}__{g}' for fam in survivors for n in FEATURES[fam] for g in groups]
                lam, model, rr, m = pick_lambda(ctx, outcome, names, est_tw, rows)
                j = judged(rr, outcome)
                f_family = {'status': 'TESTED', 'groups': groups, 'selected_lambda': lam, 'oracle_mae': m['oracle_mae'], 'vs': j, 'survives': bool(all(x['qualifies'] for x in j.values())), 'fit_rows_by_position_group': fit_rows}
                fams['F_position_routing'] = {k: f_family[k] for k in ('selected_lambda', 'oracle_mae', 'vs', 'survives')} | {'material_gain_vs_phase1d': bool(j['C1']['material_gain'])}
                models['F_position_routing'] = model
                if f_family['survives']:
                    survivors = survivors + ['F_position_routing']
        combo = None
        basic = [f for f in survivors if f != 'F_position_routing']
        if len(survivors) >= 2:
            names = [n for f in basic for n in FEATURES[f]]
            if 'F_position_routing' in survivors:
                names = names + [n for n in models['F_position_routing']['names'] if n not in names]
            lam, model, rr, m = pick_lambda(ctx, outcome, names, est_tw, rows)
            j = judged(rr, outcome)
            combo = {'families': survivors, 'selected_lambda': lam, 'oracle_mae': m['oracle_mae'], 'vs': j, 'survives': bool(all(x['qualifies'] for x in j.values()))}
            models['H_survivor_combination'] = model
        comp_rows = role_rows(ctx, outcome, rows, models)
        comp = {k: R.metrics(comp_rows, outcome, k) for k in ('C1', 'C2', 'C3')}
        allmae = {f: fams[f]['oracle_mae'] for f in fams}
        contenders = [(f, fams[f]['oracle_mae']) for f in survivors] + ([('H_survivor_combination', combo['oracle_mae'])] if combo and combo['survives'] else [])
        best_name = min(contenders, key=lambda t: (t[1], t[0]))[0] if contenders else min(allmae.items(), key=lambda t: (t[1], t[0]))[0]
        passed = bool(contenders)
        material = any(fams[f]['material_gain_vs_phase1d'] for f in fams)
        verdict = None if passed else ('PARTIAL_SNAP_SIGNAL_NOT_PROMOTED' if material else 'REJECTED_SNAP_ROLE_REPLACEMENT')
        best_model = models[best_name]
        best_rr = role_rows(ctx, outcome, rows, {'cand': best_model})
        cov = {y: m_['mapping_share'] for y, m_ in ctx.mapping.items()}
        results[head] = {'outcome': outcome, 'population': 'Phase1D fixed meaningful rows, 2024 W9-18' + (' restricted to RB/FB/HB' if HEAD_POSITIONS[outcome] else ''), 'n': len(rows), 'comparators': comp,
                         'families': fams, 'F_position_routing_status': f_family['status'], 'combination': combo, 'survivors': survivors, 'development_pass': passed, 'development_verdict_if_failed': verdict,
                         'best_candidate': best_name, 'snap_flag_report_best': snap_flag_report(best_rr, outcome), 'role_change_best': {k: R.role_change(best_rr, outcome, k) for k in ('cand', 'C1', 'C3')},
                         'slices_best': slice_report(best_rr, outcome, 'cand'), 'slice_rule_best': slice_rule(slice_report(best_rr, outcome, 'cand')),
                         'coherence_max_abs_error': coherence(ctx, outcome, {'best': best_model}, R.team_weeks_of(ctx, sel_weeks)), 'forensics_best': forensics(ctx, outcome, best_rr, 'cand'),
                         'snap_mapping_share_primary': cov, 'snap_row_coverage_selection': {'rows_with_snap_series': int(sum(1 for x in best_rr if x['rec']['snap_n'][x['i']] >= 1)), 'rows': len(best_rr)}}
        lock_heads[head] = {'outcome': outcome, 'best_candidate': best_name, 'candidate_features': list(best_model['names']), 'selected_lambda': best_model['lam'], 'development_pass': passed, 'survivors': survivors}
        all24 = head_rows(ctx, outcome, R.weeks_of(2024, 1, 18))
        rr24 = role_rows(ctx, outcome, all24, {'cand': best_model})
        src = O.read_json(ART / 'phase1o_role_source_audit.json')['source_sha256']['2024']
        for x in rr24:
            r, rec, i = x['r'], x['rec'], x['i']
            ss = rec['snap_series'][r['player_id']]
            prov = ctx.snap_prov.get((r['season'], r['week'], r['team'], r['player_id']))
            prior_prov = None
            for g in reversed(ctx.eligible_games(r['team'], (r['season'], r['week']))):
                prior_prov = ctx.snap_prov.get((g[0], g[1], r['team'], r['player_id']))
                if prior_prov:
                    break
            f = {n: float(rec['feats'][n][i]) for n in ('snap_last', 'snap_m3', 'snap_m5', 'acc_last_m3', 'reg_up_low_mid', 'reg_up_mid_full', 'reg_full_persist', 'reg_rapid_decline', 'div_level', 'div_gain', 'mate_gain', 'mate_loss', 'committee_compression')}
            proj = ctx._proj.get((r['season'], r['week'], r['team'], r['player_id'], outcome))
            if proj is None:
                proj = D.phase1b_team_projection(ctx.players, ctx.team_totals, ctx.team_opp, r, outcome, ctx.b_cfg[outcome])
            p = x['pred']
            receipts.append(R.rnd({'head': head, 'season': r['season'], 'week': r['week'], 'period': 'estimation_W1_8' if r['week'] <= 8 else 'selection_W9_18', 'player_id': r['player_id'], 'pfr_id': prior_prov[0] if prior_prov else None,
                                   'mapping_rule': prior_prov[1] if prior_prov else None, 'team': r['team'], 'opponent': r['opponent'], 'position': r.get('position'), 'phase1d_share': p['C1'], 'human_snap_share': p['C3'],
                                   'snap_series_lag_eligible': ss, 'snap_features': f, 'snap_flag': [k for k, v in snap_flags(x).items() if v and k.startswith('SNAP_ROLE')], 'candidate_adjustment': p['cand'] - p['C1'],
                                   'candidate_share': p['cand'], 'frozen_team_opportunity_projection': proj, 'predicted_opportunities_coupled': None if proj is None else p['cand'] * proj, 'actual_team_opportunities': x['team_opp'],
                                   'actual_share': x['a_share'], 'actual_opportunities': x['actual'], 'candidate_error_oracle': abs(p['cand'] * x['team_opp'] - x['actual']), 'phase1d_error_oracle': abs(p['C1'] * x['team_opp'] - x['actual']),
                                   'team_volume_error_coupled': None if proj is None else abs(p['cand'] * proj - x['actual']), 'family': best_name, 'source_sha256': src}))
    gate = any(h['development_pass'] for h in lock_heads.values())
    proto = O.sha(ART / PROTOCOL)
    lock = {'schema': 'nfl-v2-phase1q-development-lock-v1', 'base_head': protocol()['base_head'], 'code_sha256': code_sha256(), 'protocol_sha256': proto, 'schedule_2023_2024_rows_sha256': N.FROZEN_SCHEDULE_ROWS_SHA256,
            'schedule_2025_rows_sha256': O.read_json(ART / SCHEDULE_2025)['canonical_rows_sha256'], 'identity_extract_sha256': I.sha(ART / I.EXTRACT), 'heads': lock_heads, 'development_gate_passed': gate, 'validation_2025_allowed': gate,
            'burned_2026': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'seed': R.SEED, 'bootstrap_reps': R.BOOT, 'lag_days': LAG_DAYS, 'no_rescue_tuning': True}
    out = Path(out_dir)
    (out / 'phase1q_development_lock.json').write_text(O.dump(R.rnd(lock)))
    (out / 'phase1q_snap_results.json').write_text(O.dump(R.rnd({'schema': 'nfl-v2-phase1q-snap-results-v1', 'development_2024': results})))
    with gzip.GzipFile(filename='', mode='wb', fileobj=open(out / 'phase1q_snap_receipts.jsonl.gz', 'wb'), mtime=0) as g:
        for line in receipts:
            g.write((json.dumps(line, sort_keys=True) + '\n').encode())
    human = {'schema': 'nfl-v2-phase1q-competent-human-snap-v1', 'definition': protocol()['competent_human_snap_baseline'], 'frozen_before_candidate': True,
             'development_2024_W9_18': {h: results[h]['comparators']['C3'] for h in results}, 'phase1d_2024_W9_18': {h: results[h]['comparators']['C1'] for h in results}}
    (out / 'phase1q_competent_human_snap.json').write_text(O.dump(R.rnd(human)))
    return results, lock


def validate(data_dir, out_dir):
    out = Path(out_dir)
    lock = O.read_json(out / 'phase1q_development_lock.json')
    if not lock.get('development_gate_passed'):
        raise SystemExit('development gate did not pass: 2025 validation is refused')
    if lock['code_sha256'] != code_sha256():
        raise SystemExit('code changed since the development lock')
    ctx = SnapContext(data_dir, (2023, 2024, 2025))
    results = {}
    for head, h in lock['heads'].items():
        outcome = h['outcome']
        if not h['development_pass']:
            results[head] = {'verdict': 'NOT_VALIDATED_DEVELOPMENT_FAILED'}
            continue
        share = ctx.mapping['2025']['mapping_share']
        if share < M.THRESHOLD:
            results[head] = {'verdict': 'BLOCKED_DATA', 'reason': '2025 snap mapping below 95%', 'mapping_share_2025': share}
            continue
        model = fit(ctx, outcome, h['candidate_features'], h['selected_lambda'], R.team_weeks_of(ctx, R.weeks_of(2024, 1, 18)))
        val_weeks = R.weeks_of(2025, 1, 18)
        rows = head_rows(ctx, outcome, val_weeks)
        rr = role_rows(ctx, outcome, rows, {'cand': model})
        j = judged(rr, outcome)
        j['C2'] = R.judge(rr, outcome, 'cand', 'C2')
        sl = slice_report(rr, outcome, 'cand')
        passed = bool(j['C1']['qualifies'] and j['C3']['qualifies'] and slice_rule(sl))
        results[head] = {'outcome': outcome, 'candidate': h['best_candidate'], 'n': len(rr), 'mapping_share_2025': share, 'comparators': {k: R.metrics(rr, outcome, k) for k in ('C1', 'C2', 'C3', 'cand')},
                         'vs': j, 'slices': sl, 'slice_rule_passed': slice_rule(sl), 'snap_flag_report': snap_flag_report(rr, outcome), 'role_change': {k: R.role_change(rr, outcome, k) for k in ('cand', 'C1', 'C3')},
                         'coherence_max_abs_error': coherence(ctx, outcome, {'cand': model}, R.team_weeks_of(ctx, val_weeks)), 'forensics': forensics(ctx, outcome, rr, 'cand'), 'validation_pass': passed,
                         'verdict': 'SURVIVES_SNAP_ROLE_REPLACEMENT' if passed else ('PARTIAL_SNAP_SIGNAL_NOT_PROMOTED' if j['C1']['material_gain'] else 'REJECTED_SNAP_ROLE_REPLACEMENT')}
    path = out / 'phase1q_snap_results.json'
    data = O.read_json(path)
    data['validation_2025'] = R.rnd(results)
    path.write_text(O.dump(data))
    return results


def burned2026(out_dir):
    """Close-out: records the 2026 / 2025 / downstream not-run states and the per-head and overall verdicts."""
    out = Path(out_dir)
    lock = O.read_json(out / 'phase1q_development_lock.json')
    path = out / 'phase1q_snap_results.json'
    data = O.read_json(path)
    data['burned_2026_w1_4'] = {'status': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'reason': 'exact pinned 2026 upstream bytes are no longer served (see phase1h/1i/1k archival status); no replacement data are used'}
    if 'validation_2025' not in data:
        data['validation_2025'] = {'status': 'NOT_RUN_DEVELOPMENT_GATE_FAILED' if not lock['development_gate_passed'] else 'NOT_RUN'}
    val = data['validation_2025']
    verdicts = {}
    for head, v in data['development_2024'].items():
        if lock['heads'][head]['development_pass'] and isinstance(val.get(head), dict):
            verdicts[head] = val[head]['verdict']
        else:
            verdicts[head] = v['development_verdict_if_failed'] or 'BLOCKED_DATA'
    order = ['SURVIVES_SNAP_ROLE_REPLACEMENT', 'PARTIAL_SNAP_SIGNAL_NOT_PROMOTED', 'REJECTED_SNAP_ROLE_REPLACEMENT', 'BLOCKED_DATA']
    data['verdicts'] = {'per_head': verdicts, 'overall': min(verdicts.values(), key=order.index), 'note': 'heads are independent; overall is the best head verdict'}
    qual = [h for h, x in verdicts.items() if x == 'SURVIVES_SNAP_ROLE_REPLACEMENT']
    data['downstream_frozen_efficiency_diagnostic'] = {'status': 'NOT_RUN_NO_HEAD_QUALIFIED' if not qual else 'REQUIRED_FOR_' + '_'.join(qual)}
    path.write_text(O.dump(R.rnd(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=STAGES, required=True)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out-dir', default=str(ART))
    a = ap.parse_args()
    {'develop': lambda: develop(a.data_dir, a.out_dir), 'validate': lambda: validate(a.data_dir, a.out_dir), 'burned2026': lambda: burned2026(a.out_dir)}[a.stage]()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Phase1N-T team opportunity / game-state engine V2 (research only).

Interpretable ridge chain: expected plays x expected dropback rate = dropbacks; plays - dropbacks = designed rushes (exact
accounting). Estimation 2024 W1-8, selection 2024 W9-18, then (only if the development gate passes) refit and untouched 2025.
No Monte Carlo, no sportsbook input, no 2026 data, no player allocation or efficiency change.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import date
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

import nfl_v2_phase1n_team_sources as S

ART = S.ART
PROTOCOL = 'phase1n_team_protocol.json'
LAG = S.HISTORY_LAG_DAYS
QUANTITIES = ('plays', 'dropbacks', 'rush_attempts')
TARGET_KEY = {'plays': 'plays', 'dropbacks': 'dropbacks', 'rush_attempts': 'designed_rushes'}
THRESH_ABS = {'plays': 0.35, 'dropbacks': 0.35, 'rush_attempts': 0.30}
THRESH_REL = 0.05
SEED = 20261111
BOOT = 2000
SCALE = 13.5
FAMILIES = ('A_own_recent_volume', 'B_own_plus_opponent_pace', 'C_drive_chain', 'D_neutral_tendency', 'E_opponent_invitation', 'F_pregame_scenarios')
GRID = [{'lam': lam, 'decay': decay, 'window': window} for lam, decay, window in itertools.product((1.0, 10.0, 100.0), (0.65, 0.85), (5, 8))]

B_P = ['own_pl', 'opp_allowed_pl', 'opp_own_pl', 'lg_pl', 'home', 'rest_diff']
R0 = ['own_dbr', 'lg_dbr']
D_Q = ['own_dbr', 'own_neutral_dbr', 'lg_dbr', 'lg_neutral_dbr']
E_Q = D_Q + ['opp_allowed_dbr', 'opp_allowed_neutral_dbr']
FEATURES = {
    'A_own_recent_volume': {'P': ['own_pl', 'lg_pl'], 'q': R0},
    'B_own_plus_opponent_pace': {'P': B_P, 'q': R0},
    'D_neutral_tendency': {'P': B_P, 'q': D_Q},
    'E_opponent_invitation': {'P': B_P, 'q': E_Q},
    'F_pregame_scenarios': {'P': B_P + ['mix_pl', 'lowfrac', 'highfrac', 'e_margin'], 'q': E_Q + ['mix_dbr', 'e_margin']},
}
ALL_FEATURES = sorted({f for v in FEATURES.values() for part in v.values() for f in part})


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def ewma(xs, decay):
    if not xs:
        return None
    n = len(xs)
    ws = [decay ** (n - 1 - i) for i in range(n)]
    return sum(x * w for x, w in zip(xs, ws)) / sum(ws)


def ordinal(day):
    return date.fromisoformat(day[:10]).toordinal()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ratio(num, den):
    return num / den if den > 0 else None


class History:
    """Strict pregame history with the preregistered lag: prior game_date <= target_date - 3 days."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (r['game_date'], r['game_id'], r['team']))
        self.own = defaultdict(list)
        self.allowed = defaultdict(list)
        self.league = []
        for r in self.rows:
            r['_d'] = ordinal(r['game_date'])
            self.own[r['team']].append(r)
            self.allowed[r['opponent']].append(r)
            self.league.append(r)
        self._dates = {}
        self._cache = {}

    def _slice(self, bucket_name, key, day, n):
        bucket = {'own': self.own, 'allowed': self.allowed}[bucket_name][key] if bucket_name != 'league' else self.league
        ck = (bucket_name, key, 'dates')
        if ck not in self._dates:
            self._dates[ck] = [r['_d'] for r in bucket]
        end = bisect_right(self._dates[ck], day - LAG)
        return bucket[max(0, end - n):end]

    def team(self, team, day, n):
        return self._slice('own', team, day, n)

    def allowed_by(self, team, day, n):
        return self._slice('allowed', team, day, n)

    def league_rows(self, day, n):
        return self._slice('league', None, day, n)


def series(rows, field):
    return [r[field] for r in rows]


def rate_series(rows, num, den):
    out = []
    for r in rows:
        v = ratio(r[num], r[den])
        if v is not None:
            out.append(v)
    return out


def scenario_class(margin):
    return 'LEADS' if margin >= 7 else 'TRAILS' if margin <= -7 else 'COMPETITIVE'


def features(hist, row, cfg, context):
    """Pregame feature dict for one team-game or None when history is too short (>=3 own, >=3 opponent, >=64 league)."""
    day, team, opp, n, decay = row['_d'], row['team'], row['opponent'], cfg['window'], cfg['decay']
    own, opp_own = hist.team(team, day, n), hist.team(opp, day, n)
    opp_allowed = hist.allowed_by(opp, day, n)
    league = hist.league_rows(day, 128)
    if len(own) < 3 or len(opp_own) < 3 or len(opp_allowed) < 3 or len(league) < 64:
        return None
    lg_plays = mean(series(league, 'plays'))
    lg_dbr = ratio(sum(series(league, 'dropbacks')), sum(series(league, 'plays')))
    lg_neutral = ratio(sum(series(league, 'neutral_dropbacks')), sum(series(league, 'neutral_plays')))
    f = {
        'own_pl': ewma(series(own, 'plays'), decay), 'opp_own_pl': ewma(series(opp_own, 'plays'), decay), 'opp_allowed_pl': ewma(series(opp_allowed, 'plays'), decay), 'lg_pl': lg_plays,
        'own_dbr': ewma(rate_series(own, 'dropbacks', 'plays'), decay), 'lg_dbr': lg_dbr,
        'own_neutral_dbr': ewma(rate_series(own, 'neutral_dropbacks', 'neutral_plays') or [lg_neutral], decay), 'lg_neutral_dbr': lg_neutral,
        'opp_allowed_dbr': ewma(rate_series(opp_allowed, 'dropbacks', 'plays'), decay),
        'opp_allowed_neutral_dbr': ewma(rate_series(opp_allowed, 'neutral_dropbacks', 'neutral_plays') or [lg_neutral], decay),
    }
    ctx = context.get((row['game_id'], team), {})
    f['home'] = 1.0 if ctx.get('home', row['home']) else -1.0
    rd, ord_ = ctx.get('rest_days'), ctx.get('opp_rest_days')
    f['rest_diff'] = float(max(-4, min(4, rd - ord_))) if rd is not None and ord_ is not None else 0.0
    # pregame scenario engine (family F): internally derived expected margin from prior completed games only
    margin_own = mean(series(hist.team(team, day, 8), 'margin'))
    margin_opp = mean(series(hist.team(opp, day, 8), 'margin'))
    home_adv = mean([r['margin'] for r in hist.league_rows(day, 256) if r['home']]) or 0.0
    e = (margin_own - margin_opp) + (home_adv if row['home'] else -home_adv)
    wl, wt = 1.0 - norm_cdf((7.0 - e) / SCALE), norm_cdf((-7.0 - e) / SCALE)
    weights = {'LEADS': wl, 'TRAILS': wt, 'COMPETITIVE': max(0.0, 1.0 - wl - wt)}
    prior_league = hist.league_rows(day, 512)
    cls_pl, cls_q = {}, {}
    for c in weights:
        lr = [r for r in prior_league if scenario_class(r['margin']) == c]
        tr = [r for r in hist.team(team, day, 16) if scenario_class(r['margin']) == c]
        lp, lq = mean(series(lr, 'plays')) or lg_plays, ratio(sum(series(lr, 'dropbacks')), sum(series(lr, 'plays'))) or lg_dbr
        k = len(tr)
        tp = mean(series(tr, 'plays')) if tr else lp
        tq = ratio(sum(series(tr, 'dropbacks')), sum(series(tr, 'plays'))) if tr else lq
        cls_pl[c] = (k * tp + 5 * lp) / (k + 5)
        cls_q[c] = (k * (tq if tq is not None else lq) + 5 * lq) / (k + 5)
    f['e_margin'] = e
    f['mix_pl'] = sum(weights[c] * cls_pl[c] for c in weights)
    f['mix_dbr'] = sum(weights[c] * cls_q[c] for c in weights)
    both = series(own[-8:], 'plays') + series(opp_own[-8:], 'plays')
    f['lowfrac'] = sum(1 for p in both if p < 55) / len(both)
    f['highfrac'] = sum(1 for p in both if p > 75) / len(both)
    f['_weights'] = weights
    f['_scenario_effects'] = {c: {'plays': cls_pl[c], 'dropback_rate': cls_q[c]} for c in weights}
    return f


class Ridge:
    """Closed-form ridge on standardized features; the intercept is unpenalized."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, x, y):
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        z = (x - self.mu) / self.sd
        self.ybar = float(y.mean())
        a = z.T @ z + self.lam * np.eye(z.shape[1])
        self.beta = np.linalg.solve(a, z.T @ (y - self.ybar))
        return self

    def predict(self, x):
        return self.ybar + ((x - self.mu) / self.sd) @ self.beta


def matrix(feats, names):
    return np.array([[f[n] for n in names] for f in feats], dtype=float)


def chain(plays_hat, q_hat):
    """Exact accounting: plays = dropbacks + designed rushes. Rates are clipped to [0.2, 0.9] (documented football bound)."""
    q = np.clip(q_hat, 0.2, 0.9)
    p = np.maximum(plays_hat, 1.0)
    d = p * q
    return {'plays': p, 'dropbacks': d, 'rush_attempts': p - d, 'dropback_rate': q}


# --------------------------------------------------------------------------- metrics
def thresholds_counts(err, edges):
    return {'within_%d' % e: float(np.mean(np.abs(err) <= e)) for e in edges}


def metrics(pred, actual, kind):
    err = pred - actual
    out = {'n': int(len(err)), 'mae': float(np.mean(np.abs(err))), 'signed_bias': float(np.mean(err)), 'median_ae': float(np.median(np.abs(err)))}
    if kind == 'plays':
        out.update(thresholds_counts(err, (3, 5, 8, 10)))
        out.update({'miss_gt_12': float(np.mean(np.abs(err) > 12)), 'miss_gt_15': float(np.mean(np.abs(err) > 15)), 'miss_gt_20': float(np.mean(np.abs(err) > 20))})
    else:
        out.update(thresholds_counts(err, (3, 5, 8)))
        out.update({'miss_gt_10': float(np.mean(np.abs(err) > 10)), 'miss_gt_15': float(np.mean(np.abs(err) > 15))})
    return out


def block_bootstrap_upper(diff, game_dates, seed=SEED, reps=BOOT):
    """Paired 95% upper bound of mean(candidate abs error - comparator abs error) over 14-day moving blocks of complete games."""
    diff = np.asarray(diff, float)
    days = np.array([ordinal(d) for d in game_dates])
    starts = np.unique(days)
    rng = np.random.RandomState(seed)
    block_idx = [np.where((days >= s) & (days < s + 14))[0] for s in starts]
    means = []
    n = len(diff)
    for _ in range(reps):
        taken, count = [], 0
        while count < n:
            b = block_idx[rng.randint(len(block_idx))]
            taken.append(b)
            count += len(b)
        idx = np.concatenate(taken)
        means.append(diff[idx].mean())
    return float(np.percentile(means, 95))


def qualifies(cand_err, comp_err, comp_mae, quantity, game_dates):
    gain = float(np.mean(comp_err) - np.mean(cand_err))
    need = max(THRESH_ABS[quantity], THRESH_REL * comp_mae)
    upper = block_bootstrap_upper(cand_err - comp_err, game_dates)
    miss15_c, miss15_p = float(np.mean(cand_err > 15)), float(np.mean(comp_err > 15))
    ok = gain >= need and upper < 0 and miss15_c <= miss15_p + 1e-12
    return {'mae_gain': gain, 'required_gain': need, 'paired_upper95': upper, 'miss_gt_15_candidate': miss15_c, 'miss_gt_15_comparator': miss15_p, 'qualifies': bool(ok)}


# --------------------------------------------------------------------------- comparators
def frozen_comparators(rows, hist, windows=(3, 5)):
    """C3/C4 recent means and C5 competent-human (frozen formula, no optimization). Returns {name: {row_key: dict(plays, dropbacks, rush_attempts, dropback_rate)}}."""
    out = {'C3_recent3_mean': {}, 'C4_recent5_mean': {}, 'C5_competent_human': {}}
    for r in rows:
        key = (r['game_id'], r['team'])
        day = r['_d']
        for name, n in (('C3_recent3_mean', 3), ('C4_recent5_mean', 5)):
            own = hist.team(r['team'], day, n)
            if len(own) < n:
                continue
            p, d, ru = mean(series(own, 'plays')), mean(series(own, 'dropbacks')), mean(series(own, 'designed_rushes'))
            out[name][key] = {'plays': p, 'dropbacks': d, 'rush_attempts': ru, 'dropback_rate': ratio(d, p)}
        own5, opp_allowed5 = hist.team(r['team'], day, 5), hist.allowed_by(r['opponent'], day, 5)
        league = hist.league_rows(day, 128)
        if len(own5) < 5 or len(opp_allowed5) < 5 or len(league) < 64:
            continue
        lg_p = mean(series(league, 'plays'))
        lg_q = ratio(sum(series(league, 'dropbacks')), sum(series(league, 'plays')))
        p = 0.8 * (0.5 * mean(series(own5, 'plays')) + 0.5 * mean(series(opp_allowed5, 'plays'))) + 0.2 * lg_p
        q = 0.8 * (0.5 * ratio(sum(series(own5, 'dropbacks')), sum(series(own5, 'plays'))) + 0.5 * ratio(sum(series(opp_allowed5, 'dropbacks')), sum(series(opp_allowed5, 'plays')))) + 0.2 * lg_q
        out['C5_competent_human'][key] = {'plays': p, 'dropbacks': p * q, 'rush_attempts': p - p * q, 'dropback_rate': q}
    return out


def phase1c_t_comparator(directory, rows, years):
    """Run the FROZEN Phase1C-T code with its selected configs; returns {row_key: dict(plays, dropbacks, rush_attempts)} (raw, before offset correction)."""
    import nfl_v2_phase1a_direct as p1a
    import nfl_v2_phase1c_team_environment as C
    stats_paths = [str(Path(directory) / S.manifest()[y]['stats']['local_name']) for y in years]
    pbp_paths = [str(Path(directory) / S.manifest()[y]['pbp']['local_name']) for y in years]
    games = S.frozen_schedule()
    sched_path = Path(directory) / 'phase1n_schedule_allowlisted.csv'
    with open(sched_path, 'w', newline='') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=S.GAMES_ALLOWLIST)
        w.writeheader()
        w.writerows(games)
    _players, team_totals, team_opp = p1a.load_stats(stats_paths)
    pbp, pbp_opp = C.load_pbp(pbp_paths)
    C.build_indexes(team_totals, team_opp, pbp, pbp_opp, C.load_schedule(str(sched_path)))
    cfgs = S.read_json(ART / 'phase1c_team_environment_snapshot.json')['opportunities']
    out = {}
    for r in rows:
        ra = C.feature_receipt(r['team'], r['opponent'], r['season'], r['week'], 'attempts', cfgs['attempts']['selected_config'])
        rc = C.feature_receipt(r['team'], r['opponent'], r['season'], r['week'], 'carries', cfgs['carries']['selected_config'])
        if ra is None or rc is None:
            continue
        out[(r['game_id'], r['team'])] = {'plays': ra['projected_total_plays'], 'dropbacks': ra['projected_pbp_component'], 'rush_attempts': rc['projected_pbp_component']}
    return out


def phase1b_comparator(directory, rows, years):
    """Frozen Phase1B team-volume formula on official totals: attempts, targets, official carries."""
    import nfl_v2_phase1a_direct as p1a
    import nfl_v2_phase1b_opportunity as p1b
    stats_paths = [str(Path(directory) / S.manifest()[y]['stats']['local_name']) for y in years]
    players, team_totals, team_opp = p1a.load_stats(stats_paths)
    p1a.build_indexes(players, team_totals, team_opp)
    p1b.build_extra_indexes(players, team_totals, team_opp)
    snap = S.read_json(ART / 'phase1b_opportunity_snapshot.json')['outcomes']
    mapping = {'attempts': 'pass_yds', 'targets': 'rec_yds', 'carries': 'rush_yds'}
    out = {}
    for r in rows:
        target = (r['season'], r['week'])
        rec = {}
        for col, outcome in mapping.items():
            cfg = snap[outcome]['selected_opportunity_config']
            own = p1a.prior_team_values(team_totals, team_opp, target, r['team'], col, cfg['team_window'], False)
            allowed = p1a.prior_team_values(team_totals, team_opp, target, r['opponent'], col, cfg['team_window'], True)
            league = p1b.prior_league_team_values(target, col, cfg['league_window'])
            if len(own) < 2 or len(allowed) < 2 or len(league) < 16:
                rec = None
                break
            lw = 1.0 - cfg['offense_weight'] - cfg['defense_weight']
            rec[col] = cfg['offense_weight'] * p1b.ewma(own, cfg['team_decay']) + cfg['defense_weight'] * p1b.ewma(allowed, cfg['team_decay']) + lw * p1b.mean(league)
        if rec:
            out[(r['game_id'], r['team'])] = rec
    return out, team_totals


# --------------------------------------------------------------------------- development
def split_rows(rows, season, weeks):
    return [r for r in rows if r['season'] == season and weeks[0] <= r['week'] <= weeks[1]]


def build_matrix(hist, rows, cfg, context):
    kept, feats = [], []
    for r in rows:
        f = features(hist, r, cfg, context)
        if f is not None:
            kept.append(r)
            feats.append(f)
    return kept, feats


def fit_family(family, hist, train_rows, cfg, context):
    kept, feats = build_matrix(hist, train_rows, cfg, context)
    spec = FEATURES[family]
    y_p = np.array([r['plays'] for r in kept])
    y_q = np.array([r['dropbacks'] / r['plays'] for r in kept])
    return {'P': Ridge(cfg['lam']).fit(matrix(feats, spec['P']), y_p), 'q': Ridge(cfg['lam']).fit(matrix(feats, spec['q']), y_q), 'cfg': cfg, 'n_train': len(kept)}


def predict_family(model, family, hist, rows, context):
    kept, feats = build_matrix(hist, rows, model['cfg'], context)
    spec = FEATURES[family]
    out = chain(model['P'].predict(matrix(feats, spec['P'])), model['q'].predict(matrix(feats, spec['q'])))
    return kept, feats, out


def comparator_arrays(comp, kept, quantity):
    key = lambda r: (r['game_id'], r['team'])
    return np.array([comp[key(r)][quantity] for r in kept])


def run_development(directory, log=print):
    S.verify(directory, S.DEVELOPMENT_YEARS)
    tgs, meta, raw = S.build_team_games(directory, S.DEVELOPMENT_YEARS)
    rows = S.team_game_rows(tgs, meta)
    games = S.frozen_schedule()
    context = S.schedule_context(games)
    hist = History(rows)
    audit = S.read_json(ART / 'phase1n_team_source_audit.json')
    if S.schedule_digest(S.frozen_schedule(), {2023, 2024}) != audit['schedule']['sha256_2023_2024_allowlisted_rows']:
        raise ValueError('frozen schedule differs from the audited schedule')
    drives_ok = audit['drive_reconstruction']['accepted']
    est = split_rows(rows, 2024, (1, 8))
    sel = split_rows(rows, 2024, (9, 18))
    all24 = split_rows(rows, 2024, (1, 18))
    c3c4c5 = frozen_comparators(all24, hist)
    c2_raw = phase1c_t_comparator(directory, all24, S.DEVELOPMENT_YEARS)
    p1b_pred, _ = phase1b_comparator(directory, all24, S.DEVELOPMENT_YEARS)
    # offset correction of C2 on estimation weeks only
    key = lambda r: (r['game_id'], r['team'])
    offsets = {}
    for qn in QUANTITIES:
        errs = [c2_raw[key(r)][qn] - r[TARGET_KEY[qn]] for r in est if key(r) in c2_raw]
        offsets[qn] = float(np.mean(errs))
    c2 = {k: {qn: v[qn] - offsets[qn] for qn in QUANTITIES} for k, v in c2_raw.items()}
    for k, v in c2.items():
        v['dropback_rate'] = v['dropbacks'] / v['plays'] if v['plays'] else None
    comparators = {'C2_phase1c_t_offset_corrected': c2, **c3c4c5}
    common = [r for r in sel if all(key(r) in c for c in comparators.values())]
    log('selection rows with all comparators: %d of %d' % (len(common), len(sel)))
    comp_metrics, strongest = {}, {}
    for qn in QUANTITIES:
        truth = np.array([r[TARGET_KEY[qn]] for r in common])
        best = None
        for name, comp in comparators.items():
            arr = comparator_arrays(comp, common, qn)
            m = metrics(arr, truth, 'plays' if qn == 'plays' else 'other')
            comp_metrics.setdefault(name, {})[qn] = m
            if best is None or m['mae'] < best[1]:
                best = (name, m['mae'])
        strongest[qn] = {'name': best[0], 'mae': best[1]}
    results = {'families': {}, 'comparators_selection': comp_metrics, 'strongest_comparator': strongest, 'c2_offsets_estimation_weeks': offsets, 'drive_reconstruction_accepted': drives_ok}
    truth = {qn: np.array([r[TARGET_KEY[qn]] for r in common]) for qn in QUANTITIES}
    game_dates = [r['game_date'] for r in common]
    per_family_best = {}
    for family in FAMILIES:
        if family == 'C_drive_chain':
            results['families'][family] = {'verdict': 'REJECTED_DRIVE_RECONSTRUCTION' if not drives_ok else 'NOT_IMPLEMENTED_UNREACHABLE',
                                           'reason': 'The preregistered drive-reconstruction acceptance failed in the source audit (|offense drives - opponent drives| <= 1 in only %.1f%% of 2023-2024 games vs the 95%% requirement); the family was not fitted.' % (100 * audit['drive_reconstruction']['min_games_drive_diff_le_1_share'])}
            continue
        best = None
        grid_log = []
        for cfg in GRID:
            model = fit_family(family, hist, est, cfg, context)
            kept, feats, pred = predict_family(model, family, hist, common, context)
            if len(kept) != len(common):
                continue
            score = sum(float(np.mean(np.abs(pred[qn] - truth[qn]))) / strongest[qn]['mae'] for qn in QUANTITIES)
            grid_log.append({'cfg': cfg, 'normalized_sum': score})
            if best is None or (score, json.dumps(cfg, sort_keys=True)) < (best[0], json.dumps(best[1], sort_keys=True)):
                best = (score, cfg, model, kept, feats, pred)
        score, cfg, model, kept, feats, pred = best
        fam = {'selected_config': cfg, 'normalized_sum_vs_strongest': score, 'grid': grid_log, 'n_train': model['n_train'], 'quantities': {}, 'coherence': coherence(pred, kept, hist), 'slices': slices(pred, common, truth, strongest, comparators)}
        qualifies_all = True
        for qn in QUANTITIES:
            comp_arr = comparator_arrays(comparators[strongest[qn]['name']], common, qn)
            cand_err, comp_err = np.abs(pred[qn] - truth[qn]), np.abs(comp_arr - truth[qn])
            gate = qualifies(cand_err, comp_err, strongest[qn]['mae'], qn, game_dates)
            fam['quantities'][qn] = {'metrics': metrics(pred[qn], truth[qn], 'plays' if qn == 'plays' else 'other'), 'strongest_comparator': strongest[qn]['name'], 'gate': gate}
            qualifies_all &= gate['qualifies']
        rate_truth = np.array([r['dropbacks'] / r['plays'] for r in common])
        fam['dropback_rate'] = {'abs_pp_error': float(np.mean(np.abs(pred['dropback_rate'] - rate_truth)) * 100), 'signed_bias_pp': float(np.mean(pred['dropback_rate'] - rate_truth) * 100)}
        fam['qualifies_all_three'] = bool(qualifies_all)
        fam['quantities_qualifying'] = [qn for qn in QUANTITIES if fam['quantities'][qn]['gate']['qualifies']]
        fam['coherence_ok'] = fam['coherence']['plays_minus_dropbacks_minus_rushes_max_abs'] <= 1e-9 and fam['coherence']['non_negative'] and fam['coherence']['attempts_le_dropbacks']
        fam['verdict'] = ('QUALIFIES_ON_DEVELOPMENT' if qualifies_all and fam['coherence_ok'] else 'PARTIAL_SIGNAL_ONE_OR_TWO_QUANTITIES' if fam['quantities_qualifying'] else 'REJECTED_NO_PRACTICAL_GAIN')
        results['families'][family] = fam
        per_family_best[family] = (model, cfg, score)
    results['G_full_chain'] = evaluate_g(results)
    results['direct_head_ablation'] = direct_head_ablation(hist, est, common, truth, context, results)
    best_family = min((f for f in per_family_best), key=lambda f: per_family_best[f][2])
    results['best_development_family'] = best_family
    results['downstream_opportunity'] = downstream(directory, hist, rows, common, per_family_best[best_family], best_family, context, p1b_pred, truth)
    results['oracle_forensics'] = oracle(hist, per_family_best[best_family], best_family, common, context, drives_ok)
    qualifying = [f for f, v in results['families'].items() if v.get('qualifies_all_three') and v.get('coherence_ok')]
    any_partial = any(v.get('quantities_qualifying') for v in results['families'].values())
    results['development_gate'] = {'passed': bool(qualifying), 'qualifying_families': qualifying, 'any_quantity_qualifies': bool(any_partial)}
    results['n'] = {'estimation_team_games': len(est), 'selection_team_games': len(sel), 'selection_with_all_comparators': len(common), 'all_team_games_2024': len(all24)}
    return results, {'rows': rows, 'hist': hist, 'context': context, 'common': common, 'per_family_best': per_family_best, 'comparators': comparators, 'p1b_pred': p1b_pred, 'est': est, 'all24': all24, 'sel': sel}


def coherence(pred, kept, hist):
    d, ru, p = pred['dropbacks'], pred['rush_attempts'], pred['plays']
    att_rate = []
    for r in kept:
        league = hist.league_rows(r['_d'], 128)
        att_rate.append(sum(series(league, 'attempts')) / max(sum(series(league, 'dropbacks')), 1))
    att = d * np.array(att_rate)
    targets = att * 0.955
    return {'plays_minus_dropbacks_minus_rushes_max_abs': float(np.max(np.abs(p - d - ru))), 'non_negative': bool((d >= 0).all() and (ru >= 0).all() and (p >= 0).all()),
            'attempts_le_dropbacks': bool((att <= d + 1e-9).all()), 'targets_le_attempts': bool((targets <= att + 1e-9).all()), 'rates_in_bounds': bool(((pred['dropback_rate'] >= 0.2) & (pred['dropback_rate'] <= 0.9)).all()),
            'dropback_rate_share_clipped': float(np.mean((pred['dropback_rate'] <= 0.2 + 1e-12) | (pred['dropback_rate'] >= 0.9 - 1e-12)))}


def tercile_edges(values):
    return float(np.percentile(values, 33.3)), float(np.percentile(values, 66.7))


def slices(pred, common, truth, strongest, comparators):
    out = {}
    own_pace = np.array([r.get('_own_recent_plays', np.nan) for r in common])
    groups = {'overtime': np.array([r['ot'] for r in common]), 'regulation': np.array([not r['ot'] for r in common]), 'home': np.array([r['home'] == 1 for r in common]), 'away': np.array([r['home'] == 0 for r in common]),
              'early_weeks_9_12': np.array([r['week'] <= 12 for r in common]), 'later_weeks_13_18': np.array([r['week'] >= 13 for r in common])}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        out[name] = {'n': int(mask.sum()), **{qn: {'candidate_mae': float(np.mean(np.abs(pred[qn][mask] - truth[qn][mask]))),
                                                 'strongest_comparator_mae': float(np.mean(np.abs(comparator_arrays(comparators[strongest[qn]['name']], common, qn)[mask] - truth[qn][mask])))} for qn in QUANTITIES}}
    plays = truth['plays']
    lo, hi = tercile_edges(plays)
    for name, mask in (('low_possession_actual_terciles', plays <= lo), ('high_possession_actual_terciles', plays >= hi)):
        out[name] = {'n': int(mask.sum()), 'plays': {'candidate_mae': float(np.mean(np.abs(pred['plays'][mask] - plays[mask]))),
                                                      'strongest_comparator_mae': float(np.mean(np.abs(comparator_arrays(comparators[strongest['plays']['name']], common, 'plays')[mask] - plays[mask])))}}
    out['catastrophic_miss_rate'] = {qn: float(np.mean(np.abs(pred[qn] - truth[qn]) > 15)) for qn in QUANTITIES}
    out['note'] = 'possession slices use realized play counts and are descriptive stratifications only; they never enter fitting or selection'
    return out


def evaluate_g(results):
    fams = results['families']
    plays_q = [f for f, v in fams.items() if 'quantities' in v and v['quantities']['plays']['gate']['qualifies']]
    both_q = [f for f, v in fams.items() if 'quantities' in v and v['quantities']['dropbacks']['gate']['qualifies'] and v['quantities']['rush_attempts']['gate']['qualifies']]
    if not plays_q or not both_q:
        return {'verdict': 'NOT_EVALUATED_PARENTS_DID_NOT_SURVIVE', 'plays_parents': plays_q, 'dropback_and_rush_parents': both_q,
                'rule': 'G combines only separately qualifying parents; failed families are never combined.'}
    return {'verdict': 'PARENTS_EXIST_NOT_COMBINED_IN_THIS_RUN', 'plays_parents': plays_q, 'dropback_and_rush_parents': both_q}


def direct_head_ablation(hist, est, common, truth, context, results):
    """Non-promotable ablation: independent ridge heads for dropbacks and rushes (family E features) to show incoherence."""
    fam = results['families'].get('E_opponent_invitation')
    if not fam:
        return None
    cfg = fam['selected_config']
    kept_e, feats_e = build_matrix(hist, est, cfg, context)
    names = sorted(set(FEATURES['E_opponent_invitation']['P'] + FEATURES['E_opponent_invitation']['q']))
    heads = {qn: Ridge(cfg['lam']).fit(matrix(feats_e, names), np.array([r[TARGET_KEY[qn]] for r in kept_e])) for qn in QUANTITIES}
    kept, feats = build_matrix(hist, common, cfg, context)
    pred = {qn: heads[qn].predict(matrix(feats, names)) for qn in QUANTITIES}
    incoherence = pred['plays'] - pred['dropbacks'] - pred['rush_attempts']
    return {'family_features': 'E', 'maes': {qn: float(np.mean(np.abs(pred[qn] - truth[qn]))) for qn in QUANTITIES},
            'incoherence_mean_abs': float(np.mean(np.abs(incoherence))), 'incoherence_max_abs': float(np.max(np.abs(incoherence))),
            'status': 'ABLATION_ONLY_NOT_PROMOTABLE: independent heads violate plays = dropbacks + rushes'}


def downstream(directory, hist, rows, common, best, family, context, p1b_pred, truth):
    """Attempts, targets and official carries from the chain vs frozen Phase1B on the same rows (secondary; not a promotion criterion)."""
    model, cfg, _ = best
    kept, feats, pred = predict_family(model, family, hist, common, context)
    import nfl_v2_phase1a_direct as p1a
    stats_paths = [str(Path(directory) / S.manifest()[y]['stats']['local_name']) for y in S.DEVELOPMENT_YEARS]
    _p, team_totals, _o = p1a.load_stats(stats_paths)
    att_pred, tgt_pred, car_pred, tru = [], [], [], {'attempts': [], 'targets': [], 'carries': []}
    p1b_vals = {'attempts': [], 'targets': [], 'carries': []}
    used = 0
    for i, r in enumerate(kept):
        key = (r['game_id'], r['team'])
        league = hist.league_rows(r['_d'], 128)
        att_rate = sum(series(league, 'attempts')) / max(sum(series(league, 'dropbacks')), 1)
        scr = sum(series(league, 'scrambles')) / max(sum(series(league, 'dropbacks')), 1)
        kneel = mean(series(league, 'kneels'))
        tgt_ratio = None
        official = team_totals.get((r['season'], r['week'], r['team']))
        if official is None or key not in p1b_pred:
            continue
        att = pred['dropbacks'][i] * att_rate
        att_pred.append(att)
        tgt_pred.append(att * 0.955)
        car_pred.append(pred['rush_attempts'][i] + pred['dropbacks'][i] * scr + kneel)
        for col in tru:
            tru[col].append(official[col])
            p1b_vals[col].append(p1b_pred[key][col])
        used += 1
    out = {'n': used, 'note': 'targets use the fixed league targets/attempts ratio 0.955 (documented in the source audit); carries add expected scrambles and kneels to designed rushes. Secondary; not a promotion criterion.'}
    for col, cand in (('attempts', att_pred), ('targets', tgt_pred), ('carries', car_pred)):
        t = np.array(tru[col])
        out[col] = {'candidate_mae': float(np.mean(np.abs(np.array(cand) - t))), 'phase1b_mae': float(np.mean(np.abs(np.array(p1b_vals[col]) - t))), 'candidate_bias': float(np.mean(np.array(cand) - t)),
                    'phase1b_bias': float(np.mean(np.array(p1b_vals[col]) - t))}
    return out


def oracle(hist, best, family, common, context, drives_ok):
    """Postgame-only descriptive decomposition on the selection weeks; nothing here enters fitting or promotion."""
    model, cfg, _ = best
    kept, feats, pred = predict_family(model, family, hist, common, context)
    act_p = np.array([r['plays'] for r in kept])
    act_d = np.array([r['dropbacks'] for r in kept])
    act_q = act_d / act_p
    pq = pred['dropback_rate']
    out = {'status': 'POSTGAME_DESCRIPTIVE_ONLY', 'family': family, 'n': len(kept)}
    out['actual_plays_x_predicted_rate'] = {'dropback_mae': float(np.mean(np.abs(act_p * pq - act_d))), 'signed_bias': float(np.mean(act_p * pq - act_d))}
    out['predicted_plays_x_actual_rate'] = {'dropback_mae': float(np.mean(np.abs(pred['plays'] * act_q - act_d))), 'signed_bias': float(np.mean(pred['plays'] * act_q - act_d))}
    out['actual_both_perfect'] = {'dropback_mae': 0.0}
    out['chain'] = {'dropback_mae': float(np.mean(np.abs(pred['dropbacks'] - act_d)))}
    out['plays_error_mae'] = float(np.mean(np.abs(pred['plays'] - act_p)))
    out['rate_error_pp_mae'] = float(np.mean(np.abs(pq - act_q)) * 100)
    out['dropback_error_attribution'] = {'plays_error_share': out['predicted_plays_x_actual_rate']['dropback_mae'] / max(out['chain']['dropback_mae'], 1e-12),
                                         'rate_error_share': out['actual_plays_x_predicted_rate']['dropback_mae'] / max(out['chain']['dropback_mae'], 1e-12),
                                         'note': 'shares need not sum to 1 because the two errors interact'}
    # actual realized final-margin class x predicted scenario-conditioned rates (family F only carries scenario effects)
    scen = {}
    for c in ('LEADS', 'TRAILS', 'COMPETITIVE'):
        mask = np.array([scenario_class(r['margin']) == c for r in kept])
        if mask.sum():
            scen[c] = {'n': int(mask.sum()), 'plays_bias': float(np.mean(pred['plays'][mask] - act_p[mask])), 'dropback_rate_bias_pp': float(np.mean(pq[mask] - act_q[mask]) * 100),
                       'mean_actual_plays': float(act_p[mask].mean()), 'mean_actual_rate': float(act_q[mask].mean())}
    out['by_realized_final_margin_class'] = scen
    class_rate_err, class_play_err = [], []
    for r, f in zip(kept, feats):
        eff = f['_scenario_effects'][scenario_class(r['margin'])]
        class_rate_err.append(abs(eff['dropback_rate'] - r['dropbacks'] / r['plays']))
        class_play_err.append(abs(eff['plays'] - r['plays']))
    out['actual_game_state_class_x_predicted_state_rates'] = {'rate_mae_pp': float(np.mean(class_rate_err) * 100), 'plays_mae': float(np.mean(class_play_err)),
                                                              'note': 'oracle: the realized final-margin class selects the scenario-conditioned league/team rate; compare with rate_error_pp_mae of the pregame mixture'}
    if drives_ok:
        out['drive_decompositions'] = 'NOT_IMPLEMENTED'
    else:
        out['drive_decompositions'] = 'NOT_RUN_DRIVE_RECONSTRUCTION_REJECTED (actual drives x predicted plays/drive and predicted drives x actual plays/drive need an accepted drive reconstruction)'
    # which single component explains the most remaining play error: rank by variance explained by realized margin class and by overtime
    resid = pred['plays'] - act_p
    out['plays_residual_by_overtime'] = {'overtime_mean': float(resid[[r['ot'] for r in kept]].mean()) if any(r['ot'] for r in kept) else None, 'regulation_mean': float(resid[[not r['ot'] for r in kept]].mean())}
    margin = np.array([abs(r['margin']) for r in kept])
    out['plays_abs_error_corr_with_abs_realized_margin'] = float(np.corrcoef(np.abs(resid), margin)[0, 1])
    out['plays_abs_error_corr_with_overtime'] = float(np.corrcoef(np.abs(resid), np.array([r['ot'] for r in kept], float))[0, 1]) if any(r['ot'] for r in kept) else None
    return out


# --------------------------------------------------------------------------- baseline artifact, lock, receipts
def human_baseline_artifact(state):
    comp = state['comparators']['C5_competent_human']
    out = {'schema': 'nfl-v2-phase1n-competent-human-team-baseline-v1', 'status': 'FROZEN_DEFINITION; metrics are 2024 only', 'definition': S.read_json(ART / PROTOCOL)['comparators']['C5_competent_human'],
           'no_optimization': True, 'no_sportsbook': True, 'no_monte_carlo': True, 'periods': {}}
    for label, rows in (('2024_weeks_1_8', state['est']), ('2024_weeks_9_18', state['sel']), ('2024_weeks_1_18', state['all24'])):
        kept = [r for r in rows if (r['game_id'], r['team']) in comp]
        out['periods'][label] = {'n': len(kept)}
        for qn in QUANTITIES:
            t = np.array([r[TARGET_KEY[qn]] for r in kept])
            p = np.array([comp[(r['game_id'], r['team'])][qn] for r in kept])
            out['periods'][label][qn] = metrics(p, t, 'plays' if qn == 'plays' else 'other')
        q = np.array([comp[(r['game_id'], r['team'])]['dropback_rate'] for r in kept])
        out['periods'][label]['dropback_rate'] = {'abs_pp_error': float(np.mean(np.abs(q - np.array([r['dropbacks'] / r['plays'] for r in kept]))) * 100)}
    return out


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def r6(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    return x


def clean(results):
    """Drop big grids' float noise and numpy types for deterministic JSON."""
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return r6(conv(results))


def verdict(results):
    gate = results['development_gate']
    if gate['passed']:
        return 'PARTIAL_SIGNAL_NOT_PROMOTED'     # development pass alone cannot promote; 2025 and 2026 gates still apply
    return 'PARTIAL_SIGNAL_NOT_PROMOTED' if gate['any_quantity_qualifies'] else 'REJECTED_TEAM_OPPORTUNITY_REPLACEMENT'


def lock_document(results, audit_sha, protocol_sha, code_sha):
    return {'schema': 'nfl-v2-phase1n-development-lock-v1', 'status': 'FROZEN_AFTER_2024_DEVELOPMENT', 'protocol_sha256': protocol_sha, 'source_audit_sha256': audit_sha, 'code_sha256': code_sha,
            'development_gate_passed': results['development_gate']['passed'], 'qualifying_families': results['development_gate']['qualifying_families'],
            'selected_configs': {f: v.get('selected_config') for f, v in results['families'].items() if 'selected_config' in v},
            'family_verdicts': {f: v['verdict'] for f, v in results['families'].items()}, 'G_full_chain': results['G_full_chain']['verdict'],
            'strongest_comparators_2024_selection': results['strongest_comparator'],
            'validation_2025': 'OPEN_IF_GATE_PASSED' if results['development_gate']['passed'] else 'NOT_RUN_DEVELOPMENT_GATE_FAILED',
            'burned_2026_w1_4': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'no_rescue_tuning': True}


def receipts(state, results):
    best_family = results['best_development_family']
    model, cfg, _ = state['per_family_best'][best_family]
    hist, context = state['hist'], state['context']
    sd = {}
    kept_sel, feats_sel, pred_sel = predict_family(model, best_family, hist, state['common'], context)
    truth = {qn: np.array([r[TARGET_KEY[qn]] for r in kept_sel]) for qn in QUANTITIES}
    for qn in QUANTITIES:
        sd[qn] = float(np.std(pred_sel[qn] - truth[qn]))
    src = {str(y): {k: S.manifest()[y][k]['sha256'] for k in ('pbp', 'stats')} for y in S.DEVELOPMENT_YEARS}
    lines = []
    period_of = lambda r: 'ESTIMATION_W1_8_IN_SAMPLE' if r['week'] <= 8 else 'SELECTION_W9_18'
    kept, feats, pred = predict_family(model, best_family, hist, state['all24'], context)
    comps = state['comparators']
    for i, (r, f) in enumerate(zip(kept, feats)):
        key = (r['game_id'], r['team'])
        rec = {
            'season': r['season'], 'week': r['week'], 'game': r['game_id'], 'team': r['team'], 'opponent': r['opponent'], 'home_away': 'home' if r['home'] else 'away', 'period': period_of(r), 'family': best_family,
            'expected_drives': None, 'expected_plays_per_drive': None, 'drive_status': 'NOT_MODELED_DRIVE_RECONSTRUCTION_REJECTED',
            'expected_plays': pred['plays'][i], 'expected_dropback_rate': pred['dropback_rate'][i], 'expected_dropbacks': pred['dropbacks'][i], 'expected_rush_rate': 1.0 - pred['dropback_rate'][i], 'expected_rush_attempts': pred['rush_attempts'][i],
            'expected_attempts': None, 'expected_targets': None,
            'scenario_weights': f['_weights'], 'scenario_effects': f['_scenario_effects'], 'pregame_expected_margin': f['e_margin'],
            'actual_drives': r['drives'], 'actual_plays': r['plays'], 'actual_dropbacks': r['dropbacks'], 'actual_rush_attempts': r['designed_rushes'], 'actual_dropback_rate': r['dropbacks'] / r['plays'], 'overtime': r['ot'],
            'component_errors': {'dropback_rate_pp': (pred['dropback_rate'][i] - r['dropbacks'] / r['plays']) * 100},
            'total_errors': {'plays': pred['plays'][i] - r['plays'], 'dropbacks': pred['dropbacks'][i] - r['dropbacks'], 'rush_attempts': pred['rush_attempts'][i] - r['designed_rushes']},
            'uncertainty_selection_residual_sd': sd, 'comparators': {name: ({qn: c[key][qn] for qn in QUANTITIES} if key in c else None) for name, c in comps.items()},
            'source_sha256': src, 'features': {k: v for k, v in f.items() if not k.startswith('_')}}
        lines.append(json.dumps(clean(rec), sort_keys=True, separators=(',', ':')))
    return ('\n'.join(lines) + '\n').encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--stage', choices=('develop', 'validate', 'burned2026'), required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'burned2026':
        doc = {'status': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE', 'reason': 'The current 2026 release assets no longer match the Phase1H pins and post-date Week 4; no 2026 file is opened.'}
        (out_dir / 'phase1n_burned2026.json').write_text(S.dump(doc))
        return
    if args.stage == 'validate':
        lock = S.read_json(ART / 'phase1n_development_lock.json')
        if not lock['development_gate_passed']:
            doc = {'status': 'NOT_RUN_DEVELOPMENT_GATE_FAILED', '2025_files_opened': False}
            (out_dir / 'phase1n_validation_2025.json').write_text(S.dump(doc))
            return
        raise SystemExit('2025 validation requires a new reviewed implementation step; not reachable in this run')
    S.fetch(args.data_dir)
    audit = S.read_json(ART / 'phase1n_team_source_audit.json')
    results, state = run_development(args.data_dir)
    results['verdict'] = verdict(results)
    results.update({'schema': 'nfl-v2-phase1n-team-results-v1', 'phase': 'PHASE1N-T', 'periods': {'2024_development': 'RUN', '2025_validation': 'NOT_RUN_DEVELOPMENT_GATE_FAILED' if not results['development_gate']['passed'] else 'OPEN',
                                                                                                    '2026_w1_4': 'NOT_RUN_PINNED_BYTES_UNAVAILABLE'},
                    'phase1d_integration': 'NOT_RUN_TEAM_LAYER_NOT_PROMOTED', 'final_player_stat_test': 'NOT_RUN_OPPORTUNITY_LAYERS_NOT_PROMOTED', 'no_sportsbook': True, 'no_monte_carlo': True})
    cleaned = clean(results)
    (out_dir / 'phase1n_team_results.json').write_text(S.dump(cleaned))
    (out_dir / 'phase1n_competent_human_team_environment.json').write_text(S.dump(clean(human_baseline_artifact(state))))
    blob = receipts(state, results)
    with open(out_dir / 'phase1n_team_receipts.jsonl.gz', 'wb') as fh:
        with gzip.GzipFile(filename='', mode='wb', fileobj=fh, mtime=0) as g:
            g.write(blob)
    protocol_sha = S.sha(ART / PROTOCOL)
    audit_sha = S.sha(ART / 'phase1n_team_source_audit.json')
    code_sha = hashlib.sha256(Path(__file__).read_bytes() + Path(S.__file__).read_bytes()).hexdigest()
    (out_dir / 'phase1n_development_lock.json').write_text(S.dump(clean(lock_document(results, audit_sha, protocol_sha, code_sha))))


if __name__ == '__main__':
    main()

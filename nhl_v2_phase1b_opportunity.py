"""Separate NHL Phase1B mechanistic components. Never modifies the frozen B2.

Historical inputs support participation, joint TOI and transparent SOG/TOI
comparators. Attempts/conversion REQUIRE independent observed attempt labels;
no attempt count is invented from SOG. No random simulation is used.
"""
import bisect
import math
from collections import defaultdict
from datetime import timedelta

import numpy as np
from scipy.stats import nbinom

from nhl_v2_phase1a_sog_forward import parse_iso
from nhl_v2_phase1b_snapshots import resolve_identity

STRENGTHS = ('ev', 'pp', 'pk')
TOI_KEYS = ('ev_toi_sec', 'pp_toi_sec', 'sh_toi_sec')


def position(value):
    return 'D' if value == 'D' else 'F' if value in ('C', 'L', 'R', 'W', 'F') else 'U'


class History:
    def __init__(self, games, rows, allow_completion_proxy=False):
        self.games = games
        self.allow_proxy = allow_completion_proxy
        self.apps = defaultdict(list)
        self.team_games = defaultdict(list)
        self.by_game_team = defaultdict(dict)
        self.ends = {}
        self.missing_final_timestamp = 0
        for gid, g in games.items():
            if g.get('completed_at') and (allow_completion_proxy or g.get('stats_retrieved_at')):
                # A completed source game is not enough: the statistics version
                # itself must have been observed by the historical decision.
                self.ends[gid] = max(parse_iso(g['completed_at']).timestamp(),
                                     parse_iso(g.get('stats_retrieved_at', g['completed_at'])).timestamp())
            elif allow_completion_proxy:
                self.ends[gid] = (parse_iso(g['game_start_utc']) + timedelta(hours=24)).timestamp()
                self.missing_final_timestamp += 1
        if not allow_completion_proxy:
            for r in rows:
                if r['game_id'] in self.ends and r.get('retrieved_at'):
                    self.ends[r['game_id']] = max(self.ends[r['game_id']], parse_iso(r['retrieved_at']).timestamp())
        for r in rows:
            gid = r['game_id']
            if gid not in self.ends:
                continue
            key = (gid, r['team_id'])
            if r['player_id'] in self.by_game_team[key]:
                raise ValueError('duplicate historical game/team/player')
            self.by_game_team[key][r['player_id']] = r
            self.apps[r['player_id']].append((self.ends[gid], gid, r))
        for gid, g in games.items():
            if gid not in self.ends:
                continue
            for team in (g['away_team_id'], g['home_team_id']):
                if (gid, team) in self.by_game_team:
                    self.team_games[team].append((self.ends[gid], gid))
        for seq in self.apps.values():
            seq.sort(key=lambda x: (x[0], x[1]))
        for seq in self.team_games.values():
            seq.sort()
        self.app_times = {p: [x[0] for x in seq] for p, seq in self.apps.items()}
        self.team_times = {p: [x[0] for x in seq] for p, seq in self.team_games.items()}

    def prior_team(self, team, cutoff, n=10):
        seq = self.team_games.get(team, [])
        # Strictly completed before cutoff; equality is excluded.
        index = bisect.bisect_left(self.team_times.get(team, []), parse_iso(cutoff).timestamp())
        return [g for _, g in seq[max(0, index - n):index]]

    def prior_player(self, player, cutoff, team=None, n=10):
        seq = self.apps.get(player, [])
        index = bisect.bisect_left(self.app_times.get(player, []), parse_iso(cutoff).timestamp())
        if team is None:
            return [x[2] for x in seq[max(0, index - n):index]]
        # Role history cannot cross teams; take latest current-team appearances.
        out = []
        for _, _, r in reversed(seq[:index]):
            if r['team_id'] == team:
                out.append(r)
                if len(out) == n:
                    break
        return list(reversed(out))

    def candidates(self, game, cutoff, observations=()):
        cs = []
        for team, opponent in ((game['away_team_id'], game['home_team_id']), (game['home_team_id'], game['away_team_id'])):
            prior = self.prior_team(team, cutoff)
            ids = {pid for gid in prior for pid in self.by_game_team[(gid, team)]}
            ids |= {o['player_id'] for o in observations if o['team_id'] == team and o['observed_state'] in ('ROSTER_OBSERVED', 'EXPECTED_DRESSED', 'CONFIRMED_DRESSED')}
            for pid in sorted(ids):
                cs.append({'game_id': game['game_id'], 'player_id': pid, 'team_id': team, 'opponent_id': opponent})
        return resolve_identity(cs, observations)


def fit_priors(rows):
    """2018-22 exposed training only. No fitted final SOG projection model."""
    filtered = [r for r in rows if 2018 <= r['season_start_year'] <= 2022 and r.get('toi_sec', 0) > 0]
    priors = {}
    for pos in ('F', 'D', 'U'):
        rs = [r for r in filtered if position(r['position']) == pos]
        if not rs:
            rs = filtered
        if not rs:
            raise ValueError('no training exposure')
        vec = np.asarray([[r[k] for k in TOI_KEYS] for r in rs], float)
        if not np.isfinite(vec).all() or (vec < 0).any():
            raise ValueError('invalid prior TOI')
        priors[pos] = {'toi_seconds': vec.mean(axis=0).tolist(),
                       'sog_per_second': sum(r['sog'] for r in rs) / sum(r['toi_sec'] for r in rs), 'n': len(rs)}
        actual_attempts = [r for r in rs if r.get('shot_attempts') is not None]
        if actual_attempts:
            priors[pos]['total_attempts_per_second'] = sum(r['shot_attempts'] for r in actual_attempts) / sum(r['toi_sec'] for r in actual_attempts)
            priors[pos]['total_conversion'] = sum(r['sog'] for r in actual_attempts) / sum(r['shot_attempts'] for r in actual_attempts)
        # Optional authorized attempt corpus; all segments must be complete.
        attempt_keys = [s + '_attempts' for s in STRENGTHS]
        if all(all(r.get(k) is not None for k in attempt_keys) for r in rs):
            for r in rs:
                attempts = sum(r[k] for k in attempt_keys)
                if attempts < r['sog'] or any(not isinstance(r[k], int) or r[k] < 0 for k in attempt_keys):
                    raise ValueError('invalid attempt counts')
            priors[pos]['attempts_per_second'] = [sum(r[k] for r in rs) / max(1, sum(r[t] for r in rs)) for k, t in zip(attempt_keys, TOI_KEYS)]
            means = np.array([sum(r[k] for k in attempt_keys) for r in rs], float)
            m = float(means.mean())
            priors[pos]['conversion'] = sum(r['sog'] for r in rs) / max(1, means.sum())
            priors[pos]['attempt_alpha'] = max(.01, min(2., (float(means.var()) - m) / max(m * m, 1e-9)))
    team_sums = defaultdict(float)
    for r in filtered:
        team_sums[(r['game_id'], r['team_id'])] += r['sog']
    attempt_sums = defaultdict(float)
    for r in filtered:
        if r.get('shot_attempts') is not None:
            attempt_sums[(r['game_id'], r['team_id'])] += r['shot_attempts']
    return {'team_attempts': float(np.mean(list(attempt_sums.values()))) if attempt_sums else None, 'position': priors, 'team_sog': float(np.mean(list(team_sums.values()))), 'fit_seasons': [2018, 2022]}


def role_support(role, prior, variant):
    """Joint EV/PP/PK empirical distribution, not independent segment draws."""
    valid = [r for r in role if all(r.get(k) is not None for k in TOI_KEYS) and r['toi_sec'] > 0]
    if any(any(r[k] < 0 for k in TOI_KEYS) or abs(sum(r[k] for k in TOI_KEYS) - r['toi_sec']) > 1 for r in valid):
        raise ValueError('TOI components do not reconcile')
    n = len(valid)
    vecs, weights = [], []
    if n:
        weights_by_row = np.ones(n) / n
        if variant == 'C1':
            weights_by_row[:] = 0
            weights_by_row[-3:] = 1 / min(3, n)
        elif variant == 'C2':
            weights_by_row *= .3
            weights_by_row[-3:] += .7 / min(3, n)
        elif variant != 'C0':
            raise ValueError('unknown TOI variant')
        scale = min(1., n / 3)
        vecs = [[float(r[k]) for k in TOI_KEYS] for r in valid]
        weights = (weights_by_row * scale).tolist()
    if n < 3:
        vecs.append(prior['toi_seconds'])
        weights.append(1 - n / 3)
    return np.asarray(vecs), np.asarray(weights)


def component_forecast(history, game, candidate, cutoff, priors, variant='C2', environment=True):
    pid, team, opp = candidate['player_id'], candidate['team_id'], candidate['opponent_id']
    role = history.prior_player(pid, cutoff, team)
    skill = history.prior_player(pid, cutoff)
    pos = position(role[-1]['position'] if role else skill[-1]['position'] if skill else None)
    prior = priors['position'][pos]
    tg = history.prior_team(team, cutoff)
    appearances = sum(pid in history.by_game_team[(gid, team)] for gid in tg)
    p = (appearances + 8) / (len(tg) + 10)
    raw_p = appearances / len(tg) if tg else .8
    vecs, weights = role_support(role, prior, variant)
    expected = weights @ vecs
    exposure = sum(r['toi_sec'] for r in skill if r.get('toi_sec'))
    rate = (sum(r['sog'] for r in skill if r.get('toi_sec')) + 3600 * prior['sog_per_second']) / (exposure + 3600)
    recent_sog = float(np.mean([r['sog'] for r in skill])) if skill else float(sum(prior['toi_seconds']) * prior['sog_per_second'])
    own_games, opp_games = history.prior_team(team, cutoff, 5), history.prior_team(opp, cutoff, 5)
    own_sog = [sum(r['sog'] for r in history.by_game_team[(gid, team)].values()) for gid in own_games]
    allowed = []
    for gid in opp_games:
        g = history.games[gid]
        other = g['away_team_id'] if g['home_team_id'] == opp else g['home_team_id']
        if (gid, other) in history.by_game_team:
            allowed.append(sum(r['sog'] for r in history.by_game_team[(gid, other)].values()))
    own = (sum(own_sog) + 2 * priors['team_sog']) / (len(own_sog) + 2)
    defense = (sum(allowed) + 2 * priors['team_sog']) / (len(allowed) + 2)
    factor = ((own + defense) / 2) / own if environment else 1.
    mu = p * float(expected.sum()) * rate * factor
    receipt = dict(candidate, position=pos, cutoff=cutoff, p_active=p,
                   total_toi_conditional=float(expected.sum()), ev_toi=float(expected[0]), pp_toi=float(expected[1]), pk_toi=float(expected[2]),
                   total_toi_unconditional=p * float(expected.sum()), toi_support=vecs.tolist(), toi_weights=weights.tolist(),
                   skill_games=[r['game_id'] for r in skill], current_team_role_games=[r['game_id'] for r in role],
                   skill_sog_per60=3600 * rate, team_sog_prior=own, opponent_sog_allowed=defense,
                   history_participation_unsmoothed=raw_p, environment_factor=factor, transparent_analyst_mean=mu,
                   simple_mean=recent_sog, simple_participation_mean=p * recent_sog,
                   meaningful=p >= .75 and float(expected.sum()) >= 480,
                   uncertainty=['NO_CERTIFIED_DRESSED_LINEUP', 'NO_TARGET_GAME_LINE_OR_PP_UNIT', 'NO_INDEPENDENT_PROFESSIONAL_COMPARATOR'])
    if len(role) < 3:
        receipt['uncertainty'].append('THIN_CURRENT_TEAM_ROLE_HISTORY')
    if history.allow_proxy:
        receipt['uncertainty'].append('EXPOSED_HISTORICAL_COMPLETION_PROXY_NOT_CERTIFIED')
    return receipt


def attempt_forecast(receipt, skill, prior):
    """Eligible only with actual prior attempts; never uses target-game outcomes."""
    if not prior.get('attempts_per_second') or any(any(r.get(s + '_attempts') is None for s in STRENGTHS) for r in skill):
        return {'status': 'BLOCKED_ATTEMPT_DATA', 'expected_attempts': None, 'conversion': None, 'pmf': None}
    rates = []
    for s, t, rate in zip(STRENGTHS, TOI_KEYS, prior['attempts_per_second']):
        rates.append((sum(r[s + '_attempts'] for r in skill) + 10000 * rate) / (sum(r[t] for r in skill) + 10000))
    attempts = sum(sum(r[s + '_attempts'] for s in STRENGTHS) for r in skill)
    shots = sum(r['sog'] for r in skill)
    if shots > attempts:
        raise ValueError('SOG exceeds attempts')
    conversion = (shots + 20 * prior['conversion']) / (attempts + 20)
    means = np.asarray(receipt['toi_support']) @ np.asarray(rates)
    weights = np.asarray(receipt['toi_weights'])
    pm = mixture_pmf(receipt['p_active'], means * conversion, weights, prior['attempt_alpha'])
    return {'status': 'EXPERIMENTAL_REQUIRES_COMPONENT_VALIDATION', 'expected_attempts': receipt['p_active'] * float(means @ weights),
            'rates_per60': [r * 3600 for r in rates], 'conversion': conversion, 'pmf': pm.tolist(),
            'distribution': distribution_receipt(pm)}


def mixture_pmf(participation, conditional_means, weights, alpha):
    means, w = np.asarray(conditional_means, float), np.asarray(weights, float)
    if not 0 <= participation <= 1 or not np.isfinite(means).all() or (means < 0).any() or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1) > 1e-10 or alpha <= 0:
        raise ValueError('invalid distribution inputs')
    shape = 1 / alpha
    probability = shape / (shape + means)
    k = 40
    while np.max(nbinom.sf(k, shape, probability)) > 1e-12:
        k *= 2
        if k > 2560:
            raise ValueError('unresolved count tail')
    pm = participation * (w @ nbinom.pmf(np.arange(k + 1)[None, :], shape, probability[:, None]))
    pm[0] += 1 - participation
    # Only rounding/tiny tail is renormalized; no probability shortcut.
    return pm / pm.sum()


def distribution_receipt(pm):
    pm = np.asarray(pm, float)
    cdf = np.cumsum(pm)
    return {'expected_sog': float(pm @ np.arange(len(pm))), 'median_sog': int(np.searchsorted(cdf, .5)),
            'intervals': {str(c): [int(np.searchsorted(cdf, (1-c)/2)), int(np.searchsorted(cdf, (1+c)/2))] for c in (.5, .8, .9)},
            'p_ge': {str(k): float(pm[k:].sum()) for k in range(1, 6)}}


def reconcile_expected_attempt_budget(player_means, team_mean):
    """Expected opportunities only, NOT certification of joint draw allocation."""
    values = np.asarray(player_means, float)
    if not np.isfinite(values).all() or (values < 0).any() or not math.isfinite(team_mean) or team_mean < 0:
        raise ValueError('invalid team opportunity')
    if values.sum() == 0:
        return np.zeros_like(values)
    return values / values.sum() * team_mean


def total_attempt_component(history, game, candidate, cutoff, priors):
    """Separate total-attempt hypothesis; no falsely labeled EV/PP/PK attempt rates."""
    base = component_forecast(history,game,candidate,cutoff,priors,'C0',environment=False)
    skill = [r for r in history.prior_player(candidate['player_id'],cutoff) if r.get('shot_attempts') is not None]
    prior = priors['position'][base['position']]
    if 'total_attempts_per_second' not in prior:
        return {'status':'BLOCKED_ATTEMPT_DATA'}
    for r in skill:
        if r['shot_attempts'] < r['sog']:
            raise ValueError('official SOG exceeds adjudicated total attempts')
    exposure = sum(r['toi_sec'] for r in skill)
    attempts = sum(r['shot_attempts'] for r in skill)
    rate = (attempts+10000*prior['total_attempts_per_second'])/(exposure+10000)
    conv = (sum(r['sog'] for r in skill)+20*prior['total_conversion'])/(attempts+20)
    simple = np.mean([r['shot_attempts'] for r in skill]) if skill else sum(prior['toi_seconds'])*prior['total_attempts_per_second']
    own,allowed=[],[]
    for team,sequence,target in [(candidate['team_id'],history.prior_team(candidate['team_id'],cutoff,5),own),
                                  (candidate['opponent_id'],history.prior_team(candidate['opponent_id'],cutoff,5),allowed)]:
        for gid in sequence:
            if target is allowed:
                g=history.games[gid]; owner=g['home_team_id'] if g['away_team_id']==team else g['away_team_id']
            else:
                owner=team
            rs=list(history.by_game_team[(gid,owner)].values())
            if rs and all(r.get('shot_attempts') is not None for r in rs):
                target.append(sum(r['shot_attempts'] for r in rs))
    league=priors['team_attempts']
    own_mean=(sum(own)+2*league)/(len(own)+2)
    allowed_mean=(sum(allowed)+2*league)/(len(allowed)+2)
    a1=base['p_active']*base['total_toi_conditional']*rate
    return {'status':'EXPOSED_TOTAL_ATTEMPT_COMPONENT_ONLY', 'A0':float(base['p_active']*simple),'A1':a1,
            'A2':a1*((own_mean+allowed_mean)/2)/own_mean,'R0':prior['total_conversion'],'R1':conv,
            'expected_attempts_per60':rate*3600,'prior_attempt_games':[r['game_id'] for r in skill],
            'strength_attempt_rates':None,'uncertainty':'No strength-specific attempts; no current/forward attempt source'}

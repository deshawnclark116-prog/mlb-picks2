"""Development-only assists labels and legal prior state; frozen shared builder unchanged."""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import nhl_sog_phase1a_data as D

OUT = D.OUT
YEARS = tuple(range(2017, 2024))
TARGETS = tuple(range(2018, 2024))
HISTORY = ['ASSISTS_SUM_CUM', 'GOALS_SUM_CUM', 'TEAM_GOALS_EXPOSURE_CUM',
           'N_APPS_CUM', 'ASSISTS_MEAN_APP10', 'GOALS_MEAN_APP10']
ENV_RAW = ['TEAM_GF_SUM5', 'TEAM_GF_N5', 'OPP_GA_SUM5', 'OPP_GA_N5']
ENV = ['IS_HOME', 'TEAM_REST_HOURS', 'BACK_TO_BACK', 'TEAM_SOG_FOR_MEAN5',
       'OPP_SOG_ALLOWED_MEAN5', 'TEAM_GF_SHRUNK5', 'OPP_GA_SHRUNK5', 'TEAMMATE_GOAL_STRENGTH']
DIRECT = list(D.FEATURES) + HISTORY + ENV[-3:]
AVAIL = ['POS_F', 'POS_D', 'POS_UNKNOWN', 'PLAYED_LAST1', 'PLAY_RATE_TG3',
         'PLAY_RATE_TG10', 'PLAY_DEN_TG3', 'PLAY_DEN_TG10', 'TEAM_GAMES_SINCE_APPEARANCE',
         'DAYS_SINCE_LAST_APPEARANCE', 'TOI_MEAN_CT_APP3', 'PP_ALLOC_SHARE_MEAN_CT_APP3']
ROLE = ['TOI_MEAN_CT_APP3', 'TOI_MEAN_CT_APP10', 'TOI_DELTA_CT_3_10',
        'PP_TOI_MEAN_CT_APP3', 'PP_ALLOC_SHARE_MEAN_CT_APP3',
        'PP_ALLOC_SHARE_MEAN_CT_APP10', 'PP_ALLOC_SHARE_DELTA_CT_3_10']
BASE = 'd825bc629c2490aa118ff43d8c510764facafe15'


def assert_development(tab):
    seasons = set(np.asarray(tab['season']).tolist())
    if not seasons or not seasons <= set(TARGETS):
        raise ValueError('2024/2025 and warmup are forbidden model fit/score targets')


def read_source(folder, stem, year, manifest, reads):
    if year not in YEARS:
        raise ValueError('2024/2025 source access prohibited')
    rel = f'{folder}/{stem}_{year}.jsonl.gz'
    raw = (OUT / rel).read_bytes()
    entry = manifest['files'][Path(rel).name if folder == 'phase1a_data' else rel]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'frozen source hash mismatch: {rel}')
    reads[rel] = digest
    return [json.loads(line) for line in gzip.decompress(raw).decode().splitlines() if line]


def load_sources():
    """No generic all-season loader; only the 28 explicit 2017-2023 tables."""
    m1 = json.loads((OUT / 'phase1a_data_manifest.json').read_text())
    mf = json.loads((OUT / 'nhl_engine_feasibility_data_manifest.json').read_text())
    games, rows, scoring, teamgoals, reads = {}, [], {}, {}, {}
    for year in YEARS:
        gs = read_source('phase1a_data', 'games', year, m1, reads)
        games.update({g['game_id']: g for g in gs})
        rs = read_source('phase1a_data', 'skater_games', year, m1, reads)
        rows.extend(rs)
        scores = read_source('phase_scoring_s0_data', 'skater_scoring', year, mf, reads)
        for r in scores:
            key = (int(r['gameId']), int(r['playerId']))
            if key in scoring:
                raise ValueError('duplicate scoring identity')
            if not all(isinstance(r[k], int) and r[k] >= 0 for k in ['assists', 'goals', 'points']):
                raise ValueError('invalid assist/goal/point label')
            if r['points'] != r['goals'] + r['assists']:
                raise ValueError('S0 point identity failed')
            scoring[key] = r
        tg = read_source('phase_team_game_t0_data', 'team_games', year, mf, reads)
        for r in tg:
            gid = int(r['gameId'])
            g = games[gid]
            if (r['homeTeamId'], r['awayTeamId'], r['startTimeUTC']) != (g['home_team_id'], g['away_team_id'], g['game_start_utc']):
                raise ValueError('T0 schedule identity mismatch')
            for side in ('home', 'away'):
                score = r[side + 'Score']
                other = r['awayScore' if side == 'home' else 'homeScore']
                # The shootout winner bonus is not a scoring opportunity.
                teamgoals[(gid, r[side + 'TeamId'])] = int(score - (r['lastPeriodType'] == 'SO' and score > other))
    if set(scoring) != {(r['game_id'], r['player_id']) for r in rows}:
        raise ValueError('Phase1A/S0 appearance universe mismatch')
    for r in rows:
        s = scoring[(r['game_id'], r['player_id'])]
        if s['shots'] != r['sog'] or s['teamAbbrev'] != r['team_abbrev']:
            raise ValueError('frozen scoring join mismatch')
    return games, rows, scoring, teamgoals, reads


def team_labels(rows, scoring, teamgoals):
    credits = {k: 0 for k in teamgoals}
    for r in rows:
        s = scoring[(r['game_id'], r['player_id'])]
        key = (r['game_id'], r['team_id'])
        if s['assists'] > teamgoals[key]:
            raise ValueError('player assists exceed real team scoring opportunities')
        credits[key] += s['assists']
    if any(not 0 <= credits[k] <= 2 * g for k, g in teamgoals.items()):
        raise ValueError('aggregate S0 assist-credit constraint failed')
    return credits


def history_features(tab, games, rows, scoring, teamgoals):
    """History indices can contain later rows; search bounds exclude them BEFORE reads."""
    apps, teamgames = defaultdict(list), defaultdict(list)
    for r in rows:
        s = scoring[(r['game_id'], r['player_id'])]
        apps[r['player_id']].append((D.epoch(r['game_start_utc']), r['game_id'],
                                    s['assists'], s['goals'], teamgoals[(r['game_id'], r['team_id'])]))
    prior = {}
    for player, a in apps.items():
        a.sort(key=lambda x: (x[0], x[1]))
        v = np.array(a, dtype=np.int64)
        cumulative = [np.r_[0, np.cumsum(v[:, j])] for j in (2, 3, 4)]
        prior[player] = (v, cumulative)
    for gid, g in games.items():
        start = D.epoch(g['game_start_utc'])
        h, a = g['home_team_id'], g['away_team_id']
        for team, opp in [(h, a), (a, h)]:
            teamgames[team].append((start, gid, teamgoals[(gid, team)], teamgoals[(gid, opp)]))
    for team, a in teamgames.items():
        a.sort(key=lambda x: (x[0], x[1]))
        teamgames[team] = np.array(a, dtype=np.int64)
    n = len(tab['game_id'])
    out = {k: np.full(n, np.nan) for k in HISTORY + ENV_RAW}
    cache = {}
    for i, (player, start, team, opp) in enumerate(zip(tab['player_id'], tab['start'], tab['team_id'], tab['opp_id'])):
        limit = int(start) - D.CUTOFF_BACK_S
        v, (ca, cg, ce) = prior[int(player)]
        k = int(np.searchsorted(v[:, 0], limit, side='right'))
        out['ASSISTS_SUM_CUM'][i], out['GOALS_SUM_CUM'][i], out['TEAM_GOALS_EXPOSURE_CUM'][i] = ca[k], cg[k], ce[k]
        out['N_APPS_CUM'][i] = k
        out['ASSISTS_MEAN_APP10'][i] = v[max(0, k-10):k, 2].mean() if k else np.nan
        out['GOALS_MEAN_APP10'][i] = v[max(0, k-10):k, 3].mean() if k else np.nan
        key = (int(start), int(team), int(opp))
        if key not in cache:
            own, opposition = teamgames[int(team)], teamgames[int(opp)]
            t = int(np.searchsorted(own[:, 0], limit, side='right'))
            o = int(np.searchsorted(opposition[:, 0], limit, side='right'))
            cache[key] = (own[max(0, t-5):t, 2].sum(), min(t, 5), opposition[max(0, o-5):o, 3].sum(), min(o, 5))
        for name, value in zip(ENV_RAW, cache[key]):
            out[name][i] = value
    return out


def build_table(games, rows, scoring, teamgoals):
    shared, coverage = D.build_prediction_rows(games, rows, target_seasons=TARGETS)
    tab = dict(shared)
    tab.update(history_features(shared, games, rows, scoring, teamgoals))
    # Target labels are attached only now; these columns are NEVER features.
    credits = team_labels(rows, scoring, teamgoals)
    tab['assists'] = np.array([scoring[(int(g), int(p))]['assists'] if played else 0
                             for g, p, played in zip(tab['game_id'], tab['player_id'], tab['played'])], dtype=np.int64)
    tab['player_goals_label'] = np.array([scoring[(int(g), int(p))]['goals'] if played else 0
                                        for g, p, played in zip(tab['game_id'], tab['player_id'], tab['played'])], dtype=np.int64)
    tab['team_goals_label'] = np.array([teamgoals[(int(g), int(t))] for g, t in zip(tab['game_id'], tab['team_id'])], dtype=np.int64)
    tab['team_credits_label'] = np.array([credits[(int(g), int(t))] for g, t in zip(tab['game_id'], tab['team_id'])], dtype=np.int64)
    report = {'source_appearance_rows': len(rows), 'candidate_rows': len(tab['assists']),
              'candidate_population_hash': population_hash(tab), 'shared_feature_hash': D.table_hash({k: shared[k] for k in D.FEATURES}),
              'exact_shared_arrays_unchanged': all(np.array_equal(tab[k], shared[k], equal_nan=True) for k in shared),
              'nonparticipant_rows': int((tab['played'] == 0).sum()),
              'nonparticipants_with_assists': int(((tab['played'] == 0) & (tab['assists'] > 0)).sum()),
              'max_assists': int(tab['assists'].max()), 'label_join_mismatches': 0,
              'point_identity_failures': 0, 'skater_credit_constraint_failures': 0,
              'candidate_coverage_by_season': coverage, 'years_opened': list(YEARS), 'performance_years': list(TARGETS)}
    ix, inv = groups(tab)
    sums = np.bincount(inv, weights=tab['assists'], minlength=len(ix))
    outside = tab['team_credits_label'][ix] - sums
    if (outside < 0).any():
        raise ValueError('negative outside-candidate credit label')
    report['A2_data_feasibility'] = {'status': 'SUPPORTED_AGGREGATE_CREDIT_ALLOCATION', 'candidate_team_games': len(ix),
                                   'total_credits': int(tab['team_credits_label'][ix].sum()),
                                   'candidate_credits': int(sums.sum()), 'outside_candidate_credits': int(outside.sum()),
                                   'negative_outside_credit_games': 0, 'note': 'aggregate skater credits; not per-goal event multiplicity'}
    return tab, report


def groups(tab):
    _, ix, inv = np.unique(np.column_stack([tab['game_id'], tab['team_id']]), axis=0, return_index=True, return_inverse=True)
    return ix, inv


def population_hash(tab):
    h = hashlib.sha256()
    for key in ('game_id', 'team_id', 'player_id'):
        h.update(np.ascontiguousarray(tab[key]).tobytes())
    return h.hexdigest()


def load_table():
    g, r, s, t, reads = load_sources()
    tab, report = build_table(g, r, s, t)
    report['source_sha256'] = reads
    return tab, report

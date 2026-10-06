#!/usr/bin/env python3
"""Phase1N-T source layer: pinned PBP/stat access, team-game state table and the PBP team-state audit.

No model is fitted here. Only pinned 2023/2024 files are opened by the development stage; 2025 is opened
only through open_validation_season() after the development lock says the gate passed. 2026 is never opened.
Betting columns that exist in nflverse files (spread_line, total_line, moneylines, result) are never read.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
BASE_HEAD = 'e4da623d62644c8cecb51aab305ebc9af053ce6e'
DEVELOPMENT_YEARS = (2023, 2024)
VALIDATION_YEAR = 2025
PBP_FIELDS = (
    'season', 'season_type', 'week', 'game_id', 'game_date', 'home_team', 'away_team', 'posteam', 'defteam', 'play_id', 'play_type',
    'qb_dropback', 'rush_attempt', 'pass_attempt', 'sack', 'qb_scramble', 'qb_kneel', 'qb_spike', 'two_point_attempt', 'score_differential',
    'qtr', 'down', 'ydstogo', 'game_seconds_remaining', 'no_huddle', 'shotgun', 'first_down', 'third_down_converted', 'third_down_failed',
    'penalty', 'interception', 'fumble_lost', 'timeout', 'drive', 'fixed_drive', 'home_score', 'away_score')
STATS_FIELDS = ('season', 'season_type', 'week', 'team', 'opponent_team', 'attempts', 'targets', 'carries')
FORBIDDEN_COLUMNS = ('spread_line', 'total_line', 'home_moneyline', 'away_moneyline', 'over_odds', 'under_odds', 'result', 'total')
GAMES_ALLOWLIST = ('game_id', 'season', 'game_type', 'week', 'gameday', 'gametime', 'away_team', 'home_team')
FROZEN_SCHEDULE = ART / 'phase1n_frozen_schedule_2023_2024.json'
FROZEN_SCHEDULE_ROWS_SHA256 = '3d53248e4edd94e085497d5ae0fd58e286c8c6f58161045d76f83e0b949c1919'   # canonical allowlisted 2023-2024 REG rows used by the committed Phase1N audit
FROZEN_SCHEDULE_SEASONS = (2023, 2024)
HISTORY_LAG_DAYS = 3            # game_date + 2 days <= target_date - 1 day


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


def number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if out != out else out


def present(value):
    return value not in (None, '', 'NA', 'NaN', 'nan')


def manifest():
    h = read_json(ART / 'phase1h_source_coverage.json')
    return {year: {kind: {k: h['seasons'][str(year)][kind][k] for k in ('local_name', 'url', 'sha256')} for kind in ('pbp', 'stats')}
            for year in (2023, 2024, 2025)}


def verify(directory, years):
    for year in years:
        for source in manifest()[year].values():
            if sha(Path(directory) / source['local_name']) != source['sha256']:
                raise ValueError('Frozen source changed or missing: ' + source['local_name'])


def fetch(directory, years=DEVELOPMENT_YEARS):
    """Download pinned bytes only (curl); a digest mismatch is a hard stop, never a latest-data fallback."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for year in years:
        if year >= 2026:
            raise ValueError('2026 data is not opened by Phase1N')
        for source in manifest()[year].values():
            dest = directory / source['local_name']
            if dest.exists() and sha(dest) == source['sha256']:
                continue
            temp = dest.with_suffix(dest.suffix + '.partial')
            subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '280', source['url'], '-o', str(temp)], check=True)
            if sha(temp) != source['sha256']:
                temp.unlink()
                raise ValueError('Pinned source digest mismatch: ' + source['local_name'])
            temp.replace(dest)


def open_text(path):
    return gzip.open(path, 'rt', newline='', encoding='utf-8-sig') if str(path).endswith('.gz') else open(path, newline='', encoding='utf-8-sig')


def records(path, fields, year):
    """Allowlisted rows of one pinned season; refuses any other season and any Week 5+ 2026 content."""
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            season = int(number(row.get('season')))
            if season != year:
                raise ValueError('unexpected season %s in %s' % (season, path))
            if season >= 2026:
                raise ValueError('2026 rows are never opened by Phase1N')
            if row.get('season_type', 'REG') == 'REG':
                yield {k: row.get(k) for k in fields}


# --------------------------------------------------------------------------- schedule
def load_schedule(path, max_season=2025):
    """Allowlisted regular-season rows with rest/home context. Seasons after max_season are skipped unread."""
    games = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            season = int(number(row['season']))
            if season > max_season or row['game_type'] != 'REG':
                continue
            games.append({k: row[k] for k in GAMES_ALLOWLIST})
    games.sort(key=lambda g: (g['gameday'], g['game_id']))
    return games


def freeze_schedule(games_csv, out_path, provenance):
    """ONE-TIME generator of the frozen schedule artifact from a retrieved games.csv (allowlisted columns, 2023-2024 REG only).

    Never called by the audit, development, validation or CI: those consume only the committed artifact.
    """
    rows = [g for g in load_schedule(games_csv, max(FROZEN_SCHEDULE_SEASONS)) if int(g['season']) in FROZEN_SCHEDULE_SEASONS]
    rows.sort(key=lambda g: g['game_id'])
    doc = {'schema': 'nfl-v2-phase1n-frozen-schedule-v1',
           'contract': 'Phase1N consumes ONLY these rows. The mutable upstream games.csv is never read by the audit, development, validation stages or CI.',
           'allowlist': list(GAMES_ALLOWLIST), 'game_type': 'REG', 'seasons': list(FROZEN_SCHEDULE_SEASONS), 'rows_count': len(rows),
           'canonical_rows_sha256': schedule_digest(rows, set(FROZEN_SCHEDULE_SEASONS)), 'provenance': provenance, 'rows': rows}
    Path(out_path).write_text(dump(doc))
    return doc


def frozen_schedule():
    """The committed, digest-verified allowlisted 2023-2024 REG schedule rows (sorted by gameday, game_id)."""
    doc = read_json(FROZEN_SCHEDULE)
    rows = doc['rows']
    if doc['allowlist'] != list(GAMES_ALLOWLIST) or tuple(doc['seasons']) != FROZEN_SCHEDULE_SEASONS or doc['game_type'] != 'REG':
        raise ValueError('frozen schedule contract changed')
    if any(set(r) != set(GAMES_ALLOWLIST) or int(r['season']) not in FROZEN_SCHEDULE_SEASONS or r['game_type'] != 'REG' for r in rows):
        raise ValueError('frozen schedule holds a non-allowlisted column or an out-of-scope row')
    if schedule_digest(rows, set(FROZEN_SCHEDULE_SEASONS)) != FROZEN_SCHEDULE_ROWS_SHA256 or doc['canonical_rows_sha256'] != FROZEN_SCHEDULE_ROWS_SHA256:
        raise ValueError('frozen schedule digest mismatch')
    return sorted(rows, key=lambda g: (g['gameday'], g['game_id']))


def ordered_mean(values):
    """Left-to-right float mean, identical on every Python version (3.12 made sum() compensated, which changed last digits)."""
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def schedule_section(games):
    return {'allowlist': list(GAMES_ALLOWLIST), 'regular_season_rows_2023_2024': sum(1 for g in games if int(g['season']) in (2023, 2024)),
            'sha256_2023_2024_allowlisted_rows': schedule_digest(games, {2023, 2024}), 'betting_columns_accessed': False}


def schedule_digest(games, seasons):
    body = [g for g in games if int(g['season']) in seasons]
    body.sort(key=lambda g: g['game_id'])
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def schedule_context(games):
    """-> {(game_id, team): {'home': 0/1, 'rest_days': n or None, 'opp_rest_days': n or None}} (offseason gaps are not rest)."""
    last = {}
    out = {}
    for g in games:
        d = date.fromisoformat(g['gameday'][:10])
        rest = {}
        for team in (g['home_team'], g['away_team']):
            r = (d - last[team]).days if team in last else None
            rest[team] = None if r is not None and r > 21 else r
        out[(g['game_id'], g['home_team'])] = {'home': 1, 'rest_days': rest[g['home_team']], 'opp_rest_days': rest[g['away_team']]}
        out[(g['game_id'], g['away_team'])] = {'home': 0, 'rest_days': rest[g['away_team']], 'opp_rest_days': rest[g['home_team']]}
        last[g['home_team']] = last[g['away_team']] = d
    return out


# --------------------------------------------------------------------------- play semantics
def opportunity_play(row):
    """Preregistered semantics (protocol play_semantics.opportunity_play)."""
    if row.get('play_type') not in ('run', 'pass', 'qb_spike') or number(row.get('two_point_attempt')) or number(row.get('qb_kneel')):
        return False
    return bool(number(row.get('qb_dropback')) or number(row.get('rush_attempt')) or number(row.get('qb_spike')))


def classify(row):
    """-> ('dropback'|'rush', is_scramble, is_spike) for an opportunity play."""
    spike = bool(number(row.get('qb_spike')))
    drop = bool(number(row.get('qb_dropback'))) or spike
    return ('dropback' if drop else 'rush'), bool(number(row.get('qb_scramble'))), spike


def new_team_game():
    return defaultdict(float)


def build_team_games(directory, years):
    """-> (team_games, game_meta, drive_audit_raw). team_games keyed (game_id, team)."""
    team_games = {}
    meta = {}
    drive_posteams = defaultdict(set)         # (game, drive) -> posteams among ALL plays with a posteam
    drive_off = defaultdict(set)              # (game, team) -> drive ids among opportunity plays
    fixed_off = defaultdict(set)
    counts = Counter()
    for year in years:
        path = Path(directory) / manifest()[year]['pbp']['local_name']
        for row in records(path, PBP_FIELDS, year):
            gid = row['game_id']
            meta.setdefault(gid, {'game_id': gid, 'season': int(number(row['season'])), 'week': int(number(row['week'])), 'game_date': row['game_date'][:10],
                                  'home_team': row['home_team'], 'away_team': row['away_team'], 'home_score': number(row['home_score']), 'away_score': number(row['away_score']), 'ot': False})
            if number(row.get('qtr')) >= 5:
                meta[gid]['ot'] = True
            team = row.get('posteam')
            if present(team) and present(row.get('drive')):
                drive_posteams[(gid, row['drive'])].add(team)
            counts['rows'] += 1
            if not present(team):
                continue
            tg = team_games.setdefault((gid, team), new_team_game())
            if number(row.get('qb_kneel')):
                tg['kneels'] += 1
            if number(row.get('two_point_attempt')):
                tg['two_point'] += 1
            if row.get('play_type') == 'no_play':
                tg['no_play_rows'] += 1
            if not opportunity_play(row):
                continue
            kind, scramble, spike = classify(row)
            tg['plays'] += 1
            counts['opportunity_plays'] += 1
            if not present(row.get('drive')):
                counts['opportunity_plays_without_drive'] += 1
            else:
                drive_off[(gid, team)].add(row['drive'])
            if present(row.get('fixed_drive')):
                fixed_off[(gid, team)].add(row['fixed_drive'])
            reg = number(row.get('qtr')) <= 4
            if reg:
                tg['plays_reg'] += 1
            if kind == 'dropback':
                tg['dropbacks'] += 1
                tg['dropbacks_reg'] += reg
                tg['scrambles'] += scramble
                tg['spikes'] += spike
                tg['sacks'] += bool(number(row.get('sack')))
                tg['attempts'] += bool(number(row.get('pass_attempt'))) and not number(row.get('sack'))
            else:
                tg['designed_rushes'] += 1
                tg['rush_reg'] += reg
            qtr, sd, down = number(row.get('qtr')), row.get('score_differential'), number(row.get('down'))
            if 1 <= qtr <= 3 and present(sd) and abs(number(sd)) <= 7:
                tg['neutral_plays'] += 1
                tg['neutral_dropbacks'] += kind == 'dropback'
            if down in (1.0, 2.0):
                tg['early_down_plays'] += 1
                tg['early_down_dropbacks'] += kind == 'dropback'
            if down in (3.0, 4.0) and number(row.get('ydstogo')) <= 2:
                tg['short_yardage_plays'] += 1
                tg['short_yardage_rushes'] += kind == 'rush'
            tg['first_downs'] += bool(number(row.get('first_down')))
            if down == 3.0:
                tg['third_down_plays'] += 1
                tg['third_down_converted'] += bool(number(row.get('third_down_converted')))
            tg['no_huddle_plays'] += bool(number(row.get('no_huddle')))
            tg['shotgun_plays'] += bool(number(row.get('shotgun')))
            tg['turnovers'] += bool(number(row.get('interception'))) or bool(number(row.get('fumble_lost')))
    for (gid, team), tg in team_games.items():
        m = meta[gid]
        tg['drives'] = len(drive_off.get((gid, team), ()))
        tg['fixed_drives'] = len(fixed_off.get((gid, team), ()))
        tg['season'], tg['week'] = m['season'], m['week']
    raw = {'drive_posteams': drive_posteams, 'counts': counts}
    return team_games, meta, raw


def team_game_rows(team_games, meta):
    """Flat, deterministic rows (list of dicts) with opponent/home/final margin attached."""
    rows = []
    for (gid, team), tg in sorted(team_games.items()):
        m = meta[gid]
        home = team == m['home_team']
        opp = m['away_team'] if home else m['home_team']
        pf, pa = (m['home_score'], m['away_score']) if home else (m['away_score'], m['home_score'])
        row = {k: float(v) for k, v in tg.items()}
        row.update({'game_id': gid, 'team': team, 'opponent': opp, 'home': int(home), 'season': m['season'], 'week': m['week'], 'game_date': m['game_date'],
                    'ot': bool(m['ot']), 'points_for': pf, 'points_against': pa, 'margin': pf - pa})
        for key in ('plays', 'dropbacks', 'designed_rushes', 'scrambles', 'attempts', 'sacks', 'kneels', 'drives', 'spikes', 'neutral_plays', 'neutral_dropbacks', 'early_down_plays',
                    'early_down_dropbacks', 'first_downs', 'third_down_plays', 'third_down_converted', 'plays_reg', 'dropbacks_reg', 'rush_reg', 'fixed_drives', 'two_point', 'turnovers',
                    'short_yardage_plays', 'short_yardage_rushes', 'no_huddle_plays', 'shotgun_plays'):
            row.setdefault(key, 0.0)
        rows.append(row)
    return rows


def official_team_totals(directory, years):
    """Official stats_player_week team totals {(season, week, team): {...}} (audit and the downstream targets/carries)."""
    totals = defaultdict(lambda: defaultdict(float))
    opp = {}
    for year in years:
        path = Path(directory) / manifest()[year]['stats']['local_name']
        for row in records(path, STATS_FIELDS, year):
            key = (int(number(row['season'])), int(number(row['week'])), row['team'])
            opp[key] = row['opponent_team']
            for col in ('attempts', 'targets', 'carries'):
                totals[key][col] += number(row.get(col))
    return {k: dict(v) for k, v in totals.items()}, opp


# --------------------------------------------------------------------------- audit
def audit(directory):
    verify(directory, DEVELOPMENT_YEARS)
    out = {'schema': 'nfl-v2-phase1n-team-source-audit-v1', 'base_head': BASE_HEAD, 'seasons': {},
           'opened_seasons': list(DEVELOPMENT_YEARS), 'later_seasons_opened': False, 'forbidden_columns_read': [],
           'semantics': read_json(ART / 'phase1n_team_protocol.json')['play_semantics'],
           'source_sha256': {str(y): {k: manifest()[y][k]['sha256'] for k in ('pbp', 'stats')} for y in DEVELOPMENT_YEARS}}
    thresholds = read_json(ART / 'phase1n_team_protocol.json')['drive_reconstruction_acceptance']['accepted_iff']
    all_rows = []
    for year in DEVELOPMENT_YEARS:
        tgs, meta, raw = build_team_games(directory, (year,))
        rows = team_game_rows(tgs, meta)
        all_rows.extend(rows)
        official, _ = official_team_totals(directory, (year,))
        c = raw['counts']
        by_game = defaultdict(dict)
        for r in rows:
            by_game[r['game_id']][r['team']] = r
        diffs = [abs(list(g.values())[0]['drives'] - list(g.values())[1]['drives']) for g in by_game.values() if len(g) == 2]
        multi = sum(1 for teams in raw['drive_posteams'].values() if len(teams) > 1)
        recon = {'attempts_exact': 0, 'targets_total_vs_attempts_mean_ratio': None, 'carries_exact': 0, 'carries_mean_abs_diff': 0.0, 'matched_team_games': 0}
        ratios = []
        for r in rows:
            o = official.get((r['season'], r['week'], r['team']))
            if not o:
                continue
            recon['matched_team_games'] += 1
            recon['attempts_exact'] += int(o['attempts'] == r['attempts'])
            carries_pbp = r['designed_rushes'] + r['scrambles'] + r['kneels']
            recon['carries_exact'] += int(o['carries'] == carries_pbp)
            recon['carries_mean_abs_diff'] += abs(o['carries'] - carries_pbp)
            if o['attempts']:
                ratios.append(o['targets'] / o['attempts'])
        n = max(recon['matched_team_games'], 1)
        recon['carries_mean_abs_diff'] /= n
        recon['targets_total_vs_attempts_mean_ratio'] = ordered_mean(ratios) if ratios else None
        plays_identity = sum(1 for r in rows if r['plays'] != r['dropbacks'] + r['designed_rushes'])
        att_gt_drop = sum(1 for r in rows if r['attempts'] > r['dropbacks'])
        out['seasons'][str(year)] = {
            'pbp_rows': int(c['rows']), 'team_games': len(rows), 'games': len(by_game), 'overtime_games': sum(1 for g in by_game.values() if list(g.values())[0]['ot']),
            'opportunity_plays': int(c['opportunity_plays']), 'dropbacks': int(sum(r['dropbacks'] for r in rows)), 'designed_rushes': int(sum(r['designed_rushes'] for r in rows)),
            'scrambles': int(sum(r['scrambles'] for r in rows)), 'spikes': int(sum(r['spikes'] for r in rows)), 'kneels_excluded': int(sum(r['kneels'] for r in rows)),
            'two_point_excluded': int(sum(r['two_point'] for r in rows)), 'no_play_rows_excluded': int(sum(r.get('no_play_rows', 0) for r in rows)),
            'plays_equal_dropbacks_plus_designed_rushes_violations': plays_identity, 'attempts_exceed_dropbacks_violations': att_gt_drop,
            'team_game_min_max_plays': [min(r['plays'] for r in rows), max(r['plays'] for r in rows)],
            'official_reconciliation': recon,
            'drive_audit': {'opportunity_plays_without_drive_id': int(c['opportunity_plays_without_drive']), 'drive_id_present_share': 1 - c['opportunity_plays_without_drive'] / max(c['opportunity_plays'], 1),
                            'game_drives_with_multiple_offenses': multi, 'game_drives_total': len(raw['drive_posteams']),
                            'single_offense_per_drive_share': 1 - multi / max(len(raw['drive_posteams']), 1),
                            'games_with_drive_count_diff_le_1_share': sum(1 for d in diffs if d <= 1) / max(len(diffs), 1), 'drive_count_diff_distribution': dict(sorted(Counter(int(d) for d in diffs).items())),
                            'mean_drives_per_team_game': sum(r['drives'] for r in rows) / len(rows), 'mean_fixed_drives_per_team_game': sum(r['fixed_drives'] for r in rows) / len(rows),
                            'mean_plays_per_drive': sum(r['plays'] for r in rows) / max(sum(r['drives'] for r in rows), 1)},
            'field_null_shares': {f: None for f in ()},
        }
    d = out['seasons']
    present_share = min(d[str(y)]['drive_audit']['drive_id_present_share'] for y in DEVELOPMENT_YEARS)
    single = min(d[str(y)]['drive_audit']['single_offense_per_drive_share'] for y in DEVELOPMENT_YEARS)
    diff_ok = min(d[str(y)]['drive_audit']['games_with_drive_count_diff_le_1_share'] for y in DEVELOPMENT_YEARS)
    out['drive_reconstruction'] = {'criteria': thresholds, 'min_drive_id_present_share': present_share, 'min_single_offense_per_drive_share': single, 'min_games_drive_diff_le_1_share': diff_ok,
                                   'accepted': bool(present_share >= 0.999 and single >= 0.999 and diff_ok >= 0.95)}
    out['schedule'] = schedule_section(frozen_schedule())
    out['field_availability'] = field_availability(directory)
    out['team_games_total'] = len(all_rows)
    out['no_models'] = True
    out['no_sportsbook'] = True
    return out


def field_availability(directory):
    """Null shares of the semantic fields on opportunity plays in 2024 (documentation of what is reliable)."""
    fields = ('down', 'ydstogo', 'qtr', 'score_differential', 'no_huddle', 'shotgun', 'first_down', 'third_down_converted', 'game_seconds_remaining', 'timeout', 'drive', 'fixed_drive')
    total, null = 0, Counter()
    path = Path(directory) / manifest()[2024]['pbp']['local_name']
    for row in records(path, PBP_FIELDS, 2024):
        if opportunity_play(row):
            total += 1
            for f in fields:
                null[f] += not present(row.get(f))
    return {'season': 2024, 'opportunity_plays': total, 'null_share': {f: null[f] / max(total, 1) for f in fields},
            'note': 'timeouts are a play-level flag, not a pregame feature; not used by any candidate. third_down_converted is null on non-third-down plays by design.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--data-dir', required=False)
    ap.add_argument('--out', default=None)
    ap.add_argument('--freeze-schedule-from', default=None, help='one-time freeze of a retrieved games.csv into the committed artifact (not used by CI)')
    ap.add_argument('--provenance', default=None)
    args = ap.parse_args()
    if args.freeze_schedule_from:
        freeze_schedule(args.freeze_schedule_from, FROZEN_SCHEDULE, json.loads(args.provenance))
        return
    if args.fetch:
        fetch(args.data_dir)
    if args.out:
        Path(args.out).write_text(dump(audit(args.data_dir)))


if __name__ == '__main__':
    main()

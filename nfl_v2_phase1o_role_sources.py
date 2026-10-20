#!/usr/bin/env python3
"""Phase1O-R source layer: pinned stats/roster/snap-count access, exact-id joins and the role-source audit.

No model is fitted here. The development stage opens only 2023/2024 pinned files; 2025 is opened only through the validate stage after
the development lock passes; 2026 is never opened. Depth charts, injury reports, routes and any vendor data are not used.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
BASE_HEAD = 'ea4732ea3db2490b530d4d6c3d1793f2b8a151a3'
PROTOCOL = 'phase1o_role_protocol.json'
DEVELOPMENT_YEARS = (2023, 2024)
ALLOWED_YEARS = (2023, 2024, 2025)
SKILL = {'RB', 'FB', 'HB', 'WR', 'TE'}


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


def ordered_mean(values):
    """Left-to-right float mean (identical on every Python version)."""
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def manifest():
    h = read_json(ART / 'phase1h_source_coverage.json')['seasons']
    snap = read_json(ART / PROTOCOL)['sources']['snap_counts']['sha256']
    out = {}
    for year in ALLOWED_YEARS:
        y = str(year)
        out[year] = {
            'stats': {k: h[y]['stats'][k] for k in ('local_name', 'url', 'sha256')},
            'roster': {k: h[y]['roster'][k] for k in ('local_name', 'url', 'sha256')},
            'snaps': {'local_name': f'snap_counts_{year}.csv', 'sha256': snap[y],
                      'url': f'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.csv'}}
    return out


def verify(directory, years):
    for year in years:
        if year not in ALLOWED_YEARS:
            raise ValueError('only pinned 2023-2025 inputs exist; 2026 is never opened')
        for source in manifest()[year].values():
            if sha(Path(directory) / source['local_name']) != source['sha256']:
                raise ValueError('Frozen source changed or missing: ' + source['local_name'])


def fetch(directory, years=DEVELOPMENT_YEARS):
    """Pinned bytes only; a digest mismatch is a hard stop (no latest-data fallback). Transient transport failures are retried."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for year in years:
        if year not in ALLOWED_YEARS:
            raise ValueError('2026 data is not opened by Phase1O')
        for source in manifest()[year].values():
            dest = directory / source['local_name']
            if dest.exists() and sha(dest) == source['sha256']:
                continue
            temp = dest.with_suffix(dest.suffix + '.partial')
            subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '5', '--max-time', '280', source['url'], '-o', str(temp)], check=True)
            if sha(temp) != source['sha256']:
                temp.unlink()
                raise ValueError('Pinned source digest mismatch: ' + source['local_name'])
            temp.replace(dest)


def paths(directory, years, kind):
    return [str(Path(directory) / manifest()[y][kind]['local_name']) for y in years]


# --------------------------------------------------------------------------- roster metadata and snap counts
def roster_rows(directory, years):
    """Regular-season weekly roster rows (allowlisted columns)."""
    cols = ('season', 'week', 'team', 'position', 'status', 'gsis_id', 'pfr_id', 'years_exp', 'draft_number', 'game_type')
    for year in years:
        with open(Path(directory) / manifest()[year]['roster']['local_name'], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if int(number(row.get('season'))) != year:
                    raise ValueError('unexpected season in roster file')
                if row.get('game_type', 'REG') in ('REG', ''):
                    yield {c: row.get(c) for c in cols}


def pfr_to_gsis(directory, years):
    """Exact id map from the pinned rosters; a pfr_id with two GSIS ids is ambiguous and dropped (counted)."""
    seen = defaultdict(set)
    for r in roster_rows(directory, years):
        if r['pfr_id'] and r['gsis_id']:
            seen[r['pfr_id']].add(r['gsis_id'])
    mapping = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    return mapping, sum(1 for v in seen.values() if len(v) > 1)


def player_meta(directory, years):
    """{(season, gsis_id): {'years_exp', 'draft_number'}} from the first regular-season roster row of the season (weak priors only)."""
    meta = {}
    for r in sorted(roster_rows(directory, years), key=lambda r: (int(number(r['season'])), int(number(r['week'])))):
        if not r['gsis_id']:
            continue
        key = (int(number(r['season'])), r['gsis_id'])
        if key not in meta:
            meta[key] = {'years_exp': int(number(r['years_exp'])) if r['years_exp'] not in (None, '') else None,
                         'draft_number': int(number(r['draft_number'])) if r['draft_number'] not in (None, '') else None}
    return meta


def load_snaps(directory, years, mapping=None):
    """{(season, week, team, gsis_id): offense_pct} for regular-season rows with an exact id join; also returns coverage counters."""
    mapping = mapping if mapping is not None else pfr_to_gsis(directory, years)[0]
    out, cov = {}, Counter()
    for year in years:
        with open(Path(directory) / manifest()[year]['snaps']['local_name'], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if int(number(row['season'])) != year or row['game_type'] != 'REG':
                    continue
                pct = number(row['offense_pct'])
                skill = row['position'] in SKILL or row['position'] == 'QB'
                if skill and pct > 0:
                    cov['skill_rows_with_offense_snaps'] += 1
                gsis = mapping.get(row['pfr_player_id'])
                if gsis is None:
                    if skill and pct > 0:
                        cov['skill_rows_unmapped'] += 1
                    continue
                out[(year, int(number(row['week'])), row['team'], gsis)] = pct
                if skill and pct > 0:
                    cov['skill_rows_mapped'] += 1
    return out, cov


# --------------------------------------------------------------------------- audit
def audit(directory):
    verify(directory, DEVELOPMENT_YEARS)
    protocol = read_json(ART / PROTOCOL)
    mapping, ambiguous = pfr_to_gsis(directory, DEVELOPMENT_YEARS)
    out = {'schema': 'nfl-v2-phase1o-role-source-audit-v1', 'base_head': BASE_HEAD, 'opened_seasons': list(DEVELOPMENT_YEARS), 'later_seasons_opened': False,
           'source_sha256': {str(y): {k: manifest()[y][k]['sha256'] for k in ('stats', 'roster', 'snaps')} for y in DEVELOPMENT_YEARS},
           'pfr_gsis_map': {'mapped_ids': len(mapping), 'ambiguous_pfr_ids_dropped': ambiguous, 'method': 'exact id join through pinned roster pfr_id; no name matching'},
           'seasons': {}, 'excluded_sources': protocol['sources']['excluded_sources'], 'no_models': True, 'no_sportsbook': True, 'vendor_data_used': False}
    total_mapped = total_skill = 0
    for year in DEVELOPMENT_YEARS:
        snaps, cov = load_snaps(directory, (year,), mapping)
        stats_rows = Counter()
        positions = Counter()
        players_targets = set()
        players_carries = set()
        with open(Path(directory) / manifest()[year]['stats']['local_name'], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['season_type'] != 'REG':
                    continue
                stats_rows['rows'] += 1
                positions[row['position']] += 1
                if number(row['targets']) > 0:
                    players_targets.add(row['player_id'])
                if number(row['carries']) > 0:
                    players_carries.add(row['player_id'])
        status, rookies, drafted, rows_n = Counter(), 0, 0, 0
        seen = set()
        team_weeks = defaultdict(int)
        for r in roster_rows(directory, (year,)):
            status[r['status']] += 1
            rows_n += 1
            if r['gsis_id'] and (year, r['gsis_id']) not in seen:
                seen.add((year, r['gsis_id']))
                rookies += number(r['years_exp']) == 0 and r['years_exp'] not in (None, '')
                drafted += r['draft_number'] not in (None, '')
            team_weeks[(r['team'], r['week'])] += 1
        mapped, skill = cov['skill_rows_mapped'], cov['skill_rows_with_offense_snaps']
        total_mapped += mapped
        total_skill += skill
        out['seasons'][str(year)] = {
            'stats_regular_season_rows': stats_rows['rows'], 'players_with_targets': len(players_targets), 'players_with_carries': len(players_carries), 'stats_positions': dict(positions.most_common(12)),
            'roster_rows': rows_n, 'roster_team_weeks': len(team_weeks), 'roster_status_counts': dict(status.most_common()),
            'roster_unique_players': len(seen), 'roster_rookies_years_exp_0': int(rookies), 'roster_players_with_draft_number': int(drafted),
            'snap_skill_rows_with_offense_snaps': skill, 'snap_skill_rows_mapped_to_gsis': mapped, 'snap_skill_mapping_share': mapped / skill if skill else None, 'snap_keys_mapped': len(snaps)}
    share = total_mapped / total_skill if total_skill else 0.0
    out['family_data_decisions'] = {
        'B_snap_change': 'ACCEPTED' if share >= 0.95 else 'BLOCKED_DATA', 'B_snap_skill_mapping_share_2023_2024': share, 'B_acceptance_threshold': 0.95,
        'A_recency_acceleration': 'ACCEPTED (weekly stats history)', 'C_usage_concentration': 'ACCEPTED (weekly stats history)',
        'D_teammate_vacancy': 'ACCEPTED under the inherited Phase1D target-week roster-status (T90) contract; historical rosters are final-week snapshots',
        'E_rookie_development': 'ACCEPTED as weak priors (roster years_exp / draft_number); draft metadata is missing for undrafted players',
        'F_change_point': 'ACCEPTED (weekly stats history)', 'G_role_competition': 'ACCEPTED (weekly stats history + roster)'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--years', default='2023,2024', help='pinned seasons to fetch; 2025 only for the validate stage')
    args = ap.parse_args()
    if args.fetch:
        fetch(args.data_dir, tuple(int(y) for y in args.years.split(',')))
    if args.out:
        Path(args.out).write_text(dump(audit(args.data_dir)))


if __name__ == '__main__':
    main()

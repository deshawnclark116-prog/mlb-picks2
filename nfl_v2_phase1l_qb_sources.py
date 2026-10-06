#!/usr/bin/env python3
"""Phase1L source/starter audit, never a starter oracle or a model fit.

Only the pinned 2023/2024 PBP/stat corpus is opened. Later source coverage is
explicitly inherited from the frozen J/H audits, not refreshed current state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
BASE_HEAD = '28c96ef98cb0835f732d822dc8d1a5adaaa942f3'
YEARS = (2023, 2024)
PBP_FIELDS = {'season', 'season_type', 'week', 'game_id', 'game_date',
              'home_team', 'away_team', 'posteam', 'defteam', 'play_id',
              'play_type', 'qb_dropback', 'rush_attempt', 'pass_attempt',
              'sack', 'qb_scramble', 'qb_kneel', 'qb_spike', 'two_point_attempt',
              'passer_player_id', 'passer_player_name', 'rusher_player_id',
              'score_differential', 'qtr', 'down', 'no_huddle', 'shotgun'}
STATS_FIELDS = {'season', 'season_type', 'week', 'game_id', 'team',
                'opponent_team', 'position', 'player_id', 'player_display_name',
                'attempts', 'passing_yards'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def number(value):
    return float(value) if value not in (None, '', 'NA', 'NaN', 'nan') else 0.0


def records(path, fields):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            season = int(number(row.get('season')))
            week = int(number(row.get('week')))
            if season == 2026 and week > 4:
                raise ValueError('Week 5+ forbidden, before labels are exposed')
            if season not in YEARS:
                raise ValueError('Phase1L source-gated runner opens 2023/2024 only')
            if row.get('season_type', 'REG') == 'REG':
                yield {key: row[key] for key in fields if key in row}


def manifest():
    h = read_json(ART / 'phase1h_source_coverage.json')
    return {str(y): {kind: {key: h['seasons'][str(y)][kind][key]
                           for key in ('local_name', 'url', 'sha256')}
                    for kind in ('pbp', 'stats')} for y in YEARS}


def verify(directory):
    for sources in manifest().values():
        for source in sources.values():
            if sha(Path(directory) / source['local_name']) != source['sha256']:
                raise ValueError('Frozen source changed: ' + source['local_name'])


def fetch(directory):
    """No current-season download and no revised-byte fallback."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for kind, tag in (('pbp', 'pbp'), ('stats', 'stats_player')):
        metadata = directory / (kind + '-release.json')
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '60',
                        'https://api.github.com/repos/nflverse/nflverse-data/releases/tags/' + tag,
                        '-o', str(metadata)], check=True)
        assets = {a['name']: a for a in read_json(metadata)['assets']}
        for year in YEARS:
            source = manifest()[str(year)][kind]
            dest = directory / source['local_name']
            if dest.exists() and sha(dest) == source['sha256']:
                continue
            asset = assets.get(source['url'].split('/')[-1])
            if not asset or asset.get('digest') != 'sha256:' + source['sha256']:
                raise ValueError('Archived source unavailable: supply pinned bytes, never latest data')
            temporary = dest.with_suffix(dest.suffix + '.partial')
            subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '120',
                            source['url'], '-o', str(temporary)], check=True)
            if sha(temporary) != source['sha256']:
                temporary.unlink()
                raise ValueError('Digest changed during download')
            temporary.replace(dest)


def opportunity_play(row):
    if row.get('play_type') not in {'run', 'pass', 'qb_spike'} or number(row.get('two_point_attempt')):
        return False
    # Kneels excluded; spikes retained because they are official QB attempts.
    return not number(row.get('qb_kneel')) and bool(
        number(row.get('qb_dropback')) or number(row.get('rush_attempt')) or number(row.get('qb_spike')))


def aggregate(directory):
    """Labels/history in a separate ledger; no features or starter labels here."""
    verify(directory)
    teams = defaultdict(lambda: defaultdict(float))
    qbs = defaultdict(lambda: defaultdict(float))
    games = {}
    official = {}
    coverage = {}
    for year in YEARS:
        missing = Counter()
        count = Counter()
        headers = set()
        for row in records(Path(directory) / manifest()[str(year)]['pbp']['local_name'], PBP_FIELDS):
            headers.update(row)
            gid = row['game_id']
            games[gid] = {k: row[k] for k in ('game_id', 'game_date', 'home_team', 'away_team', 'week', 'season')}
            if not opportunity_play(row) or not row.get('posteam'):
                continue
            tm = teams[(gid, row['posteam'])]
            # nflfastR marks spikes qb_dropback=0 although they are attempts.
            # Add the disjoint spike category to this documented denominator.
            drop = float(bool(number(row.get('qb_dropback')) or number(row.get('qb_spike'))))
            att = int(bool(number(row.get('pass_attempt'))) and not number(row.get('sack')))
            tm['plays'] += 1
            tm['dropbacks'] += drop
            tm['attempts'] += att
            for k in PBP_FIELDS:
                if row.get(k) in (None, '', 'NA', 'NaN'):
                    missing[k] += 1
            count['opportunity_plays'] += 1
            count['dropbacks'] += int(drop)
            count['attempts'] += att
            pid = row.get('passer_player_id') or (row.get('rusher_player_id') if number(row.get('qb_scramble')) else '')
            if drop and not pid:
                count['dropbacks_missing_owner'] += 1
            if att and not row.get('passer_player_id'):
                count['attempts_missing_passer'] += 1
            if drop and pid:
                qb = qbs[(gid, row['posteam'], pid)]
                qb['dropbacks'] += drop
                qb['attempts'] += att
                qb['sacks'] += number(row.get('sack'))
                qb['scrambles'] += number(row.get('qb_scramble'))
        for row in records(Path(directory) / manifest()[str(year)]['stats']['local_name'], STATS_FIELDS):
            if row.get('position') != 'QB':
                continue
            key = (row['game_id'], row['team'], row['player_id'])
            if key in official:
                raise ValueError('Duplicate official QB label')
            official[key] = {**row, 'attempts': number(row['attempts']),
                             'passing_yards': number(row['passing_yards'])}
        diffs = [(key, q['attempts'] - official.get(key, {}).get('attempts', 0))
                 for key, q in qbs.items() if games[key[0]]['season'] == str(year)]
        mismatches = [(key, delta) for key, delta in diffs if delta]
        coverage[str(year)] = {
            'regular_games': sum(g['season'] == str(year) for g in games.values()),
            'counts': dict(sorted(count.items())),
            'opportunity_field_missing_counts': {k: missing[k] for k in sorted(PBP_FIELDS)},
            'header_fields': sorted(headers),
            'qb_pbp_official_attempt_mismatches': len(mismatches),
            'mismatches': [{'key': list(key), 'pbp_minus_official': delta} for key, delta in sorted(mismatches)],
            'official_positive_qbs_without_pbp_owner': sum(
                r['season'] == str(year) and r['attempts'] > 0 and key not in qbs for key, r in official.items()),
            'original_pregame_starter_receipts': 0,
        }
    return {'games': games, 'teams': dict(teams), 'qbs': dict(qbs), 'official': official, 'coverage': coverage}


def audit(directory):
    data = aggregate(directory)
    j = read_json(ART / 'phase1j_information_gap_audit.json')
    h = read_json(ART / 'phase1h_source_coverage.json')
    source_ids = {'weekly_rosters', 'nflverse_injuries', 'depth_2024', 'depth_2025_plus',
                  'transactions', 'gamebook_starters', 'coach_announcements',
                  'official_practice_reports', 'espn_undocumented', 'sportradar_depth',
                  'sportradar_injuries', 'sportsdataio_state'}
    seasons = {}
    for year in range(2023, 2027):
        ys = str(year)
        injury = h['seasons'][ys]['injuries']
        seasons[ys] = {
            'scope': 'RECOMPUTED_PBP_STATS_SOURCE_ONLY' if year in YEARS else 'INHERITED_FROZEN_H_J_SOURCE_ONLY_NOT_REFRESHED',
            'weekly_roster_rows': h['seasons'][ys]['roster']['rows'],
            'weekly_roster_as_of_status_receipts': 0,
            'injury_rows': injury['rows'],
            'injury_original_publication_rows': injury['publication_timestamp_rows'],
            'injury_modification_rows': injury['modification_timestamp_rows'],
            'depth': j['depth_snapshot_coverage'].get(ys, {'status': 'NO_ACCEPTED_ARCHIVE; 2026 metadata only, no payload'}),
            'accepted_licensed_starter_health_versions': 0,
            'verified_pregame_starter_coverage': 0,
            'expected_starter_as_of_coverage': 0,
            'transactions_original_version_archive': 'ABSENT',
            'official_game_starters': 'POSTGAME_EVALUATION_ONLY; not loaded in Phase1L',
            'competition': 'Prior same-team passer usage observable, current roster/health/role unresolved',
            'prior_game_qb_history': data['coverage'].get(ys, 'Previously audited prior PBP exists; not opened this phase'),
        }
    inspected = ['nfl_phase1_team_environment.py', 'nfl_phase1_opportunity.py',
                 'nfl_phase1c_sim.py', 'nfl_phase1c_constants.py',
                 'nfl_v2_phase1b_opportunity.py', 'nfl_v2_phase1c_team_environment.py',
                 'nfl_v2_competent_human_baseline.py', 'nfl_v2_phase1e_integrated.py',
                 'nfl_v2_phase1f_efficiency_forensic.py']
    return {
        'schema': 'nfl-v2-phase1l-qb-starter-source-audit-v1', 'phase': 'PHASE1L-QB',
        'base_head': BASE_HEAD, 'sources': manifest(), 'seasons': seasons,
        'source_classifications_preserved': [s for s in j['sources'] if s['id'] in source_ids],
        'prior_audits_sha256': {p: sha(ART / p) for p in ['phase1j_information_gap_audit.json', 'phase1h_source_coverage.json']},
        'existing_code_sha256': {p: sha(ROOT / p) for p in inspected},
        'existing_logic_findings': {
            'v1_qb_out': 'Most-used recent passer plus Out/Doubtful report; report-asof assumed, not archived. Not imported/executed.',
            'v1_phase1c': 'Allocates dropbacks from rescaled attempts/dropback propensity, subtracts sacks/scrambles; simulation not executed.',
            'phase1b_e': 'Prior QB attempt share multiplied by blended team attempts; no verified expected-starter state.',
            'human_old': 'Recent QB attempts + opponent allowance + league; useful baseline, not a starter-state source.',
            'phase1c_team': 'Generic team model is frozen, not a certified V1/QB replacement.',
        },
        'opportunity_semantics': {
            'plays': 'Dropback OR non-kneel rush OR spike; scrambles count once; two-point/no-play excluded. Opportunity dropbacks add spikes (raw qb_dropback=0) to keep official attempts within the denominator.',
            'attempts': 'pass_attempt AND NOT sack on eligible football plays; reconciliation to official QB counts audited, not assumed.',
            'ownership': 'passer GSIS, or rusher GSIS on scramble only; no name-based or participant starter inference.',
            'prior_cutoff': 'Earlier season/week AND source game-date +48h <= target game-date -24h at UTC00:00; corrected retrospective releases, not original provider vintages.',
        },
        'source_gate': {
            'passed': False, 'verdict': 'BLOCKED_STARTER_STATE_DATA',
            'reason': 'No accepted 2024 pregame depth/health/announcement version archive. Prior usage cannot certify current participation; 2025 loaded snapshots do not repair development.',
            'primary_stable_starter_rows_certifiable': 0,
            'continuity_policy': 'Retain prior-continuity hints, label UNCERTAIN/INSUFFICIENT; never equate past passer dominance with current expected starter.',
            'unblock_requires_new_protocol': ['Authorized expected-starter/depth and health version archive covering 2024 development plus 2025 confirmation',
                                            'Published/observed timestamps, revisions, stable GSIS/team/game joins and cutoff replay',
                                            'Documented in-game abnormal-exit labels for clean censor slices; no outcome-count proxy'],
        },
        'no_fit_or_performance': True, 'week5_plus_accessed': False,
        'new_2025_2026_payloads_opened': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.fetch:
        fetch(args.data_dir)
    out = audit(args.data_dir)
    write_json(args.out, out)
    print(json.dumps({'source_gate': out['source_gate'], 'coverage': {y: d['regular_games'] for y, d in aggregate(args.data_dir)['coverage'].items()}}))


if __name__ == '__main__':
    main()

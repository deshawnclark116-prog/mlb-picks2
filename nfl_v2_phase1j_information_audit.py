#!/usr/bin/env python3
"""Phase1J DATA ONLY: validate catalog and reproduce frozen-source coverage.

No network, model imports, fitting, forecast evaluation, or source revisions.
Provider documentation/metadata is a dated evidence snapshot, not auto-refreshed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "nfl_models/nfl_player_outcome_v2"
BASE_HEAD = "411d6f8ca12f24e1f192ab2d7768589cf77d1bc0"
CLASSIFICATIONS = {
    "USABLE_NOW", "USABLE_WITH_API_OR_CONNECTOR", "PAID_BUT_VIABLE",
    "BLOCKED_TIMING", "BLOCKED_LICENSE", "NOT_HISTORICAL", "NOT_LIVE",
    "STALE_ONLY", "NOT_USEFUL", "UNKNOWN_NEEDS_ACCESS",
}
REQUIRED_SOURCE_FIELDS = {
    "id", "provider", "dataset_product", "evidence_ids", "urls", "access_cost",
    "seasons_available", "granularity", "player_ids", "team_ids", "play_ids",
    "nflverse_join", "historical_availability", "live_current_availability",
    "publication_timestamp", "true_as_of_reconstruction", "T24_usability",
    "T90_usability", "latency", "update_cadence", "licensing", "acquisition_method",
    "reliability", "already_in_repo", "football_layers", "leakage_risk",
    "implementation_difficulty", "classification", "classification_reason",
    "unverified_requirements",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f)


def present(value):
    return value not in (None, '', 'NA', 'NaN', 'nan')


def safe_week(row, season):
    week = int(float(row['week']))
    if season == 2026 and week > 4:
        raise ValueError('Week 5+ input forbidden; stop before processing labels')
    return week


def frozen_coverage(directory, frozen=None):
    """Coverage and timestamp counts only. Never compute a prediction/error."""
    frozen = frozen or json.loads((ART / 'phase1h_source_coverage.json').read_text())
    directory = Path(directory)
    # Check every frozen byte first, including unpublished-source absence.
    for season, sources in frozen['seasons'].items():
        for kind, meta in sources.items():
            path = directory / (meta.get('local_name') or f'{kind}_{season}.csv')
            if meta['status'] == 'NOT_PUBLISHED':
                if path.exists():
                    raise ValueError(f'Unaudited source: {path.name}')
            elif not path.exists() or sha(path) != meta['sha256']:
                raise ValueError(f'Frozen source changed: {path.name}')
    out = {}
    for year in (2023, 2024, 2025, 2026):
        ftn = {}
        dates = Counter()
        ftn_games = {}
        for r in records(directory / f'ftn_{year}.csv'):
            week = safe_week(r, year)
            dates[r.get('date_pulled', '')] += 1
            ftn[(r['nflverse_game_id'], str(int(float(r['nflverse_play_id']))))] = r
            ftn_games.setdefault(week, set()).add(r['nflverse_game_id'])
        counts = Counter()
        pbp_games = {}
        for r in records(directory / f'pbp_{year}.csv.gz'):
            week = safe_week(r, year)
            if r.get('season_type') != 'REG':
                continue
            pbp_games.setdefault(week, set()).add(r['game_id'])
            if (r.get('play_type') != 'pass' or r.get('no_play') == '1'
                    or r.get('two_point_attempt') == '1' or not present(r.get('receiver_player_id'))):
                continue
            counts['pbp_targets'] += 1
            for field in ('passer_player_id', 'receiver_player_id', 'air_yards', 'complete_pass'):
                counts[field] += present(r.get(field))
            q = ftn.get((r['game_id'], str(int(float(r['play_id'])))))
            if q is None:
                continue
            counts['ftn_joined_targets'] += 1
            for field in ('is_catchable_ball', 'is_contested_ball', 'is_drop', 'read_thrown', 'date_pulled'):
                counts[field] += present(q.get(field))
            # Not a feature test. Count whether the retained retrieval date even
            # predates its own source game. Backfilled files normally do not.
            counts['date_pulled_before_own_game_date'] += q.get('date_pulled', '9999')[:10] < r['game_date']
        injury = list(records(directory / f'injuries_{year}.csv'))
        for r in injury:
            safe_week(r, year)
        p = directory / f'participation_{year}.csv'
        part = Counter()
        if p.exists():
            for r in records(p):
                for field in ('was_pressure', 'defense_man_zone_type', 'route'):
                    part[field] += present(r.get(field))
        out[str(year)] = {
            'counts': dict(sorted(counts.items())),
            'pbp_reg_games_by_week': {str(w): len(g) for w,g in sorted(pbp_games.items())},
            'ftn_games_by_week_including_postseason': {str(w): len(g) for w,g in sorted(ftn_games.items())},
            'ftn_rows': sum(dates.values()),
            'ftn_date_pulled_min': min(dates), 'ftn_date_pulled_max': max(dates),
            'ftn_date_pulled_distinct': len(dates), 'ftn_date_pulled_missing': dates.get('', 0),
            'participation_published': p.exists(), 'participation_field_counts_all_plays': dict(sorted(part.items())),
            'injury_rows': len(injury),
            'injury_modification_timestamp_rows': sum(present(r.get('date_modified')) for r in injury),
            'injury_original_publication_timestamp_rows': 0,
            'hashes': {k:m.get('sha256') for k,m in sorted(frozen['seasons'][str(year)].items())},
        }
    return out


def depth_coverage(path, season):
    """Only 2024/2025 historical files; no unbounded current-season download."""
    if season not in (2024, 2025):
        raise ValueError('Depth record audit restricted to 2024/2025')
    counts = Counter(); timestamps = set(); teams = set(); rank1 = Counter()
    fields = None
    for r in records(path):
        fields = sorted(r)
        counts['rows'] += 1
        dt = r.get('dt'); position = r.get('pos_abb', r.get('depth_position', r.get('position')))
        if dt:
            timestamps.add(dt)
        teams.add(r.get('team',r.get('club_code')))
        if position in ('QB','LT','LG','C','RG','RT'):
            counts[f'{position}_rows'] += 1
            counts[f'{position}_with_gsis'] += present(r.get('gsis_id'))
            if r.get('pos_rank',r.get('depth_team')) == '1':
                rank1[(dt or (r.get('season'),r.get('week'),r.get('game_type')),
                       r.get('team',r.get('club_code')),position,
                       r.get('pos_slot',r.get('formation')))] += 1
    return {'season':season,'sha256':sha(path),'fields':fields,'counts':dict(sorted(counts.items())),
            'teams':len(teams),'dt_count':len(timestamps),'dt_min':min(timestamps) if timestamps else None,
            'dt_max':max(timestamps) if timestamps else None,
            'duplicate_rank1_position_slots':sum(v>1 for v in rank1.values()),
            'certification':'LOADED_TIMESTAMP_NOT_VERIFIED_STARTER; upstream reuse rights separate; 2024 timing BLOCKED'}


def validate_bundle(directory=ART):
    directory = Path(directory)
    audit = json.loads((directory/'phase1j_information_gap_audit.json').read_text())
    matrix = json.loads((directory/'phase1j_human_research_gap_matrix.json').read_text())
    priority = json.loads((directory/'phase1j_source_priority.json').read_text())
    assert audit['base_head'] == BASE_HEAD
    assert audit['predictive_fitting'] is False and audit['week5_plus_records_used'] is False
    ids = set(); evidence = {e['id']:e for e in audit['evidence']}
    for source in audit['sources']:
        assert REQUIRED_SOURCE_FIELDS <= source.keys(), source['id']
        assert source['id'] not in ids; ids.add(source['id'])
        assert source['classification'] in CLASSIFICATIONS
        assert source['urls'] and source['classification_reason']
        assert set(source['evidence_ids']) <= evidence.keys()
        for key in ('T24_usability','T90_usability'):
            assert isinstance(source[key],dict) and {'historical','live'} <= source[key].keys()
        if source['classification'] == 'PAID_BUT_VIABLE':
            assert source['access_cost'].startswith('PAID')
            assert source['unverified_requirements']
    gaps = matrix['items']
    assert len(gaps) >= 15 and len({g['item'] for g in gaps}) == len(gaps)
    assert len(priority['top_five']) == 5
    assert [p['rank'] for p in priority['top_five']] == list(range(1,6))
    for item in priority['top_five']:
        assert set(item['source_ids']) <= ids
        assert all(item[k] for k in ('football_mechanism','phase1i_failure_addressed','component',
                                    'historical_validation','live_deployment','cost_access','recommendation'))
    assert audit['receiver_depth_signal'] == 'SURVIVED_SIGNAL_NOT_PROMOTED'
    assert audit['receiving_decision'] == priority['receiving_decision']
    assert audit['rushing_decision'] == priority['rushing_decision']
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--data-dir')
    parser.add_argument('--depth-2024')
    parser.add_argument('--depth-2025')
    parser.add_argument('--out')
    args = parser.parse_args()
    audit = validate_bundle()
    if args.data_dir:
        coverage = frozen_coverage(args.data_dir)
        if args.depth_2024 and args.depth_2025:
            depth = {str(y):depth_coverage(getattr(args,f'depth_{y}'),y) for y in (2024,2025)}
        else:
            depth = None
        if args.check:
            assert coverage == audit['frozen_source_coverage'], 'Frozen coverage drift'
            if depth is not None:
                assert depth == audit['depth_snapshot_coverage'], 'Depth coverage drift'
        if args.out:
            Path(args.out).write_text(json.dumps({'coverage':coverage,'depth':depth},indent=2,sort_keys=True)+'\n')
    print(f"Phase1J audit valid: {len(audit['sources'])} sources; no model execution")


if __name__ == '__main__':
    main()

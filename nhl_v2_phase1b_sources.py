"""Research source audit only. Reuses frozen bytes; never fetches or fits a model."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import nhl_v2_phase1a_sog_forward as F

OUT = F.OUT


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def nested_values(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for v in value.values():
            yield from nested_values(v, key)
    elif isinstance(value, list):
        for v in value:
            yield from nested_values(v, key)


def audit():
    F.verify_lock()
    ledger = F.Ledger()
    count = ledger.verify()
    forecasts = [r for r in ledger.rows() if r['record_type'] == 'FORECAST']
    decisions = {}
    mismatch = Counter()
    for r in forecasts:
        decisions.setdefault((r['game_id'], r['forecast_horizon'], r['scheduled_start']), r)
        # Frozen summary helper shifted survival indices. Preserve original fields,
        # disclose separately; repaired metrics compute directly from NB2, not P1..P5.
        from scipy.stats import nbinom
        shape = 1 / r['dispersion']
        for k in range(1, 6):
            correct = nbinom.sf(k - 1, shape, shape / (shape + r['expected_sog']))
            if abs(r['P' + str(k)] - correct) > 1e-8:
                mismatch['P' + str(k)] += 1
    horizons = {}
    blobs_checked = set()
    for h in F.HORIZONS:
        rs = [r for (_, hh, _), r in decisions.items() if hh == h]
        complete = timed = roster = scratches = 0
        for r in rs:
            sources = r['raw_source_hashes']['game_payloads']
            if len(sources) != 3:
                continue
            complete += 1
            eligible = all(s.get('retrieval_completed_utc') and F.parse_iso(s['retrieval_completed_utc']) <= F.parse_iso(r['cutoff_at']) for s in sources)
            timed += eligible
            found_roster = found_scratch = False
            for s in sources:
                raw = gzip.decompress((F.BLOBS / (s['sha256'] + '.gz')).read_bytes())
                if hashlib.sha256(raw).hexdigest() != s['sha256']:
                    raise ValueError('source blob hash mismatch')
                blobs_checked.add(s['sha256'])
                data = json.loads(raw)
                found_roster |= any(bool(x) for x in nested_values(data, 'rosterSpots'))
                found_scratch |= any(bool(x) for x in nested_values(data, 'scratches'))
            roster += eligible and found_roster
            scratches += eligible and found_scratch
        horizons[h] = {'decisions': len(rs), 'complete_endpoint_captures': complete,
                       'complete_pre_cutoff_captures': timed, 'nonempty_roster_observations': roster,
                       'nonempty_scratch_observations': scratches, 'confirmed_dressed_captures': 0,
                       'complete_truth_comparisons': 0, 'dressed_recall': None,
                       'availability_certified': False, 'status': 'UNQUALIFIED_NO_DRESSED_EVIDENCE'}
    coverage = {}
    for season in range(2017, 2026):
        path = OUT / 'phase1a_data' / f'skater_games_{season}.jsonl.gz'
        rows = [json.loads(l) for l in gzip.open(path, 'rt')]
        fields = {}
        for key in ('player_id', 'team_id', 'toi_sec', 'ev_toi_sec', 'pp_toi_sec', 'sh_toi_sec', 'sog', 'shot_attempts', 'ev_attempts', 'pp_attempts', 'pk_attempts', 'completed_at', 'line', 'pp_unit', 'zone_starts'):
            n = sum(r.get(key) is not None for r in rows)
            fields[key] = {'present': n, 'missing': len(rows) - n, 'usable': n == len(rows)}
        coverage[str(season)] = {'rows': len(rows), 'games': len({r['game_id'] for r in rows}),
                                'file_sha256': F.sha_file(path), 'fields': fields,
                                'TOI_reconciles': all(r['toi_sec'] == r['ev_toi_sec'] + r['pp_toi_sec'] + r['sh_toi_sec'] for r in rows)}
    return {'evidence': 'EXISTING_FROZEN_CORPUS_NOT_NEW_COLLECTION', 'ledger_rows': count,
            'ledger_sha256': F.sha_file(F.LEDGER), 'game_source_blobs_verified': len(blobs_checked),
            'horizons': horizons, 'historical_coverage': coverage,
            'B2_frozen_survival_field_mismatches': dict(mismatch),
            'B2_policy': 'Do not change frozen forecasts or engine. Stored P1..P5 are shifted (P1=1); use NB2(mu,alpha) for repaired evaluations and disclose. New engine tests correct P>=k semantics.',
            'publication_times_in_existing_game_payloads': 'NOT_PROVEN; retrieval proves observation only',
            'new_collection': 'DISABLED_PENDING_AUTHORIZED_ACCESS',
            'attempt_component': 'ADJUDICATED_TOTAL_COUNTS_2017_2023_ONLY_STRENGTH_AND_LIVE_BLOCKED',
            'attempt_migration': json.loads((OUT/'phase1b_attempt_history/migration_manifest.json').read_text()),
            'attempt_join_coverage': join_adjudicated_attempts([r for season in range(2017,2026) for r in [json.loads(l) for l in gzip.open(OUT/'phase1a_data'/f'skater_games_{season}.jsonl.gz','rt')]])[1],
            'professional_comparator': 'NO_INDEPENDENT_PROSPECTIVE_FORECASTS_AVAILABLE'}




def parse_attempt_labels(pbp, official_players):
    """PRIOR completed-game labels only, with official shooter ownership and SOG check.

    Requires an authorized payload supplied by the caller. Blocked-shot event owner
    may be the defending team, so ownership comes from shooter ID crosswalk.
    Empty-net/unknown manpower stays unsegmented; never mislabeled PP.
    """
    if pbp.get('gameState') not in ('OFF', 'FINAL'):
        raise ValueError('attempt labels require official final')
    counts = {pid: {'shot_attempts': 0, 'counted_sog': 0, 'ev_attempts': 0, 'pp_attempts': 0, 'pk_attempts': 0, 'unsegmented_attempts': 0} for pid in official_players}
    seen = set()
    for play in pbp.get('plays', []):
        if play.get('periodDescriptor', {}).get('periodType') == 'SO':
            continue
        kind = play.get('typeDescKey')
        if kind not in ('goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'):
            continue
        event = play.get('eventId')
        if event is None or event in seen:
            raise ValueError('missing/duplicate PBP attempt event')
        seen.add(event)
        details = play.get('details', {})
        pid = details.get('shootingPlayerId', details.get('scoringPlayerId'))
        if pid not in counts:
            raise ValueError('unmapped shooter; incomplete attempt labels')
        c = counts[pid]
        c['shot_attempts'] += 1
        c['counted_sog'] += kind in ('goal', 'shot-on-goal')
        code = str(play.get('situationCode', ''))
        if len(code) == 4 and code.isdigit() and code[0] == code[3] == '1':
            away, home = int(code[1]), int(code[2])
            is_home = official_players[pid]['team_id'] == pbp['homeTeam']['id']
            ours, theirs = (home, away) if is_home else (away, home)
            state = 'ev' if ours == theirs else 'pp' if ours > theirs else 'pk'
            c[state + '_attempts'] += 1
        else:
            c['unsegmented_attempts'] += 1
    for pid, c in counts.items():
        if c['counted_sog'] != official_players[pid]['sog']:
            raise ValueError('PBP SOG / official SOG disagreement; labels quarantined')
        if c['unsegmented_attempts']:
            c['strength_rates_usable'] = False
        else:
            c['strength_rates_usable'] = True
    return counts



def join_adjudicated_attempts(rows):
    """Pinned data-only migration; exact identity/SOG match, never silently impute."""
    root=OUT/'phase1b_attempt_history'
    manifest=json.loads((root/'migration_manifest.json').read_text())
    lookup={}
    for name,meta in manifest['files'].items():
        path=root/name
        if F.sha_file(path)!=meta['sha256']:
            raise ValueError('migrated attempt source hash mismatch')
        for r in [json.loads(l) for l in gzip.open(path,'rt')]:
            key=(r['game_id'],r['player_id'])
            if key in lookup:raise ValueError('duplicate adjudicated attempt identity')
            if r['shot_attempts']!=r['sog_official']+r['missed_attempts']+r['blocked_attempts']:raise ValueError('attempt arithmetic failed')
            lookup[key]=r
    result=[]; coverage=Counter()
    for r in rows:
        a=lookup.get((r['game_id'],r['player_id']))
        copy=dict(r)
        if a is not None:
            if a['sog_official']!=r['sog']:
                coverage['SOG_disagreement_quarantined']+=1
            else:
                copy['shot_attempts']=a['shot_attempts']
                coverage['joined']+=1
        else:
            coverage['missing_never_imputed']+=1
        result.append(copy)
    return result,dict(coverage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUT / 'phase1b_source_audit.json'))
    args = parser.parse_args()
    write_json(args.output, audit())

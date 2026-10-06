#!/usr/bin/env python3
"""Phase1M-DATA acceptance harness. Source qualification only: no model, no fit, no scoring.

Stages (each is deterministic given its pinned inputs, except the dated web-archive probe):
  --select-sample   preregistered 2024 archetype sample (labels from realized attempts stratify the sample ONLY)
  --timing-probe    nflverse depth_charts_2025 load-timestamp coverage (counts only; no passer/outcome labels)
  --probe-wayback   Internet Archive Availability API metadata probe (existence/capture time only; throttled)
  --assemble        write phase1m_qb_state_acceptance_results.json from the stage outputs
No Week 5+ (2026) player-state payload and no sportsbook column is ever opened.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import nfl_v2_phase1m_qb_state_sources as S

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
UTC = timezone.utc
ET = ZoneInfo('America/New_York')
GAMES_ALLOWLIST = ('game_id', 'season', 'game_type', 'week', 'gameday', 'gametime', 'away_team', 'home_team')
STATS_FIELDS = ('season', 'season_type', 'week', 'team', 'position', 'player_id', 'attempts')
STATS_2024_SHA256 = '3ddc45a84f759aa348ce465ae001752c530575455717657cdfe1f8abfcdb4759'   # Phase1H manifest pin
DEPTH_2025_SHA256 = 'f5a4aa3fa70150e810b2255200c8735a6c1cc8ff77361308ce39149345b39b4a'   # asset observed 2026-10-06; last updated 2026-03-14
SAMPLE_SALT = '|phase1m-sample-v1'
LABELS = ('MULTI_QB_GAME', 'NEW_OR_SURPRISE_QB', 'CHANGE_TO_PRIOR_BACKUP_OR_RETURN', 'STABLE_VETERAN')
PER_LABEL = 4
FRESH_HOURS = 72
WAYBACK_URL = 'https://archive.org/wayback/available?url={url}&timestamp={ts}'
FAMILIES = (
    ('nfl_injuries_week', 'https://www.nfl.com/injuries/league/{season}/reg{week}'),
    ('espn_team_depth', 'https://www.espn.com/nfl/team/depth/_/name/{team_lower}'),
    ('ourlads_team_depth', 'https://www.ourlads.com/nfldepthcharts/depthchart/{team}'),
)
THROTTLE_SECONDS = 6.0


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def kickoff_utc(gameday, gametime):
    local = datetime.strptime(gameday + ' ' + gametime, '%Y-%m-%d %H:%M').replace(tzinfo=ET)
    return local.astimezone(UTC)


def load_games(path, season):
    """Regular-season rows of ONE season through an allowlist (betting columns can never be read)."""
    out = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['season'] != str(season) or row['game_type'] != 'REG':
                continue
            if season >= 2026:
                raise ValueError('2026 rows are not opened by Phase1M')
            rec = {k: row[k] for k in GAMES_ALLOWLIST}
            rec['kickoff_utc'] = kickoff_utc(rec['gameday'], rec['gametime']).strftime('%Y-%m-%dT%H:%M:%SZ')
            out.append(rec)
    out.sort(key=lambda r: r['game_id'])
    return out


def games_digest(games):
    return hashlib.sha256(json.dumps(games, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_qb_attempts(path):
    """-> {(week, team): {player_id: attempts}} for QB-position rows of 2024 REG. File sha is pinned."""
    if sha_file(path) != STATS_2024_SHA256:
        raise ValueError('stats_player_week_2024.csv does not match the pinned Phase1H hash')
    table = defaultdict(lambda: defaultdict(float))
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['season'] != '2024' or row['season_type'] != 'REG' or row['position'] != 'QB':
                continue
            att = float(row['attempts'] or 0)
            if att > 0:
                table[(int(row['week']), row['team'])][row['player_id']] += att
    return {k: dict(v) for k, v in table.items()}


def top_qb(att):
    if not att:
        return None
    return sorted(att.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def label_team_games(games, qb):
    """Archetype labels (weeks 3-18) by the preregistered precedence. Realized attempts stratify the sample only."""
    rows = []
    for g in games:
        week = int(g['week'])
        if week < 3:
            continue
        for team in (g['away_team'], g['home_team']):
            target = qb.get((week, team))
            if not target:
                continue
            prior_weeks = [w for w in range(week - 1, 0, -1) if qb.get((w, team))]
            if len(prior_weeks) < 2:
                continue
            prior1, prior2 = qb[(prior_weeks[0], team)], qb[(prior_weeks[1], team)]
            prior_qb = top_qb(prior1)
            tqb = top_qb(target)
            earlier = Counter()
            for w in prior_weeks:
                for pid, a in qb[(w, team)].items():
                    earlier[pid] += a
            total = sum(target.values())
            ranked = sorted(target.items(), key=lambda kv: (-kv[1], kv[0]))
            multi = len(ranked) > 1 and ranked[1][1] >= 10
            if multi:
                label = 'MULTI_QB_GAME'
            elif earlier.get(tqb, 0) == 0:
                label = 'NEW_OR_SURPRISE_QB'
            elif tqb != prior_qb:
                label = 'CHANGE_TO_PRIOR_BACKUP_OR_RETURN'
            elif tqb == top_qb(prior2) and target[tqb] / total >= 0.8 and earlier[tqb] >= 20:
                label = 'STABLE_VETERAN'
            else:
                continue
            rows.append({'game_id': g['game_id'], 'team': team, 'opponent': g['home_team'] if team == g['away_team'] else g['away_team'],
                         'week': week, 'kickoff_utc': g['kickoff_utc'], 'archetype': label})
    return rows


def sample_key(game_id, team):
    return hashlib.sha256((game_id + '|' + team + SAMPLE_SALT).encode()).hexdigest()


def select_sample(labelled):
    out = []
    for label in LABELS:
        pool = sorted((r for r in labelled if r['archetype'] == label), key=lambda r: sample_key(r['game_id'], r['team']))
        for r in pool[:PER_LABEL]:
            out.append({**r, 'sample_digest': sample_key(r['game_id'], r['team'])})
    return out


def eligible_counts(labelled):
    return {label: sum(1 for r in labelled if r['archetype'] == label) for label in LABELS}


# --------------------------------------------------------------------------- web archive probe
def probe_urls(row, season=2024):
    out = []
    for family, template in FAMILIES:
        url = template.format(season=season, week=row['week'], team=row['team'], team_lower=row['team'].lower())
        out.append((family, url))
    return out


def wayback_status(returned_ts, cutoff):
    if returned_ts is None:
        return 'NONE'
    ts = datetime.strptime(returned_ts, '%Y%m%d%H%M%S').replace(tzinfo=UTC)
    if ts > cutoff:
        return 'POST_CUTOFF_ONLY_UNKNOWN'
    if ts >= cutoff - timedelta(hours=FRESH_HOURS):
        return 'PRE_CUTOFF_WITHIN_72H'
    return 'PRE_CUTOFF_STALE'


def http_get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': 'nfl-v2-phase1m-acceptance/1.0 (research; contact repo owner)'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        return err.code, ''
    except Exception as err:                                                # noqa: BLE001
        return -1, repr(err)


def wayback_probe(sample, fetch=http_get, sleep=time.sleep, throttle=THROTTLE_SECONDS, clock=lambda: datetime.now(UTC)):
    results, stopped = [], None
    for row in sample:
        kick = S.parse_ts(row['kickoff_utc'])
        for label in S.CUTOFFS:
            cutoff = S.cutoff_time(kick, label)
            for family, url in probe_urls(row):
                if stopped:
                    break
                query = WAYBACK_URL.format(url=url.replace('https://', ''), ts=cutoff.strftime('%Y%m%d%H%M%S'))
                status, body = fetch(query)
                observed = clock().strftime('%Y-%m-%dT%H:%M:%SZ')
                entry = {'game_id': row['game_id'], 'team': row['team'], 'archetype': row['archetype'], 'cutoff_label': label,
                         'cutoff_utc': cutoff.strftime('%Y-%m-%dT%H:%M:%SZ'), 'family': family, 'url': url, 'http_status': status, 'observed_at_utc': observed}
                if status == 429:
                    stopped = {'reason': 'HTTP_429', 'after_requests': len(results)}
                    entry['capture_status'] = 'NOT_COMPLETED'
                    results.append(entry)
                    break
                snap = None
                if status == 200:
                    try:
                        closest = (json.loads(body).get('archived_snapshots') or {}).get('closest')
                        snap = closest['timestamp'] if closest and closest.get('available') else None
                    except Exception:                                      # noqa: BLE001
                        entry['parse_error'] = True
                entry['returned_snapshot_timestamp'] = snap
                entry['capture_status'] = wayback_status(snap, cutoff) if status == 200 else 'NOT_COMPLETED'
                results.append(entry)
                sleep(throttle)
            if stopped:
                break
        if stopped:
            break
    done = [r for r in results if r['capture_status'] != 'NOT_COMPLETED']
    counts = Counter(r['capture_status'] for r in results)
    by_family = {f: dict(Counter(r['capture_status'] for r in done if r['family'] == f)) for f, _ in FAMILIES}
    planned = len(sample) * len(S.CUTOFFS) * len(FAMILIES)
    return {'planned_requests': planned, 'completed_requests': len(done), 'stopped': stopped, 'capture_status_counts': dict(counts), 'by_family': by_family,
            'pre_cutoff_within_72h_fraction_of_completed': (counts.get('PRE_CUTOFF_WITHIN_72H', 0) / len(done)) if done else None, 'results': results}


# --------------------------------------------------------------------------- repository depth timing probe
def timing_probe(depth_path, games_2025):
    """Counts only: QB-bearing depth snapshot timing relative to T24/T90 cutoffs for 2025 REG team-games."""
    if sha_file(depth_path) != DEPTH_2025_SHA256:
        raise ValueError('depth_charts_2025.csv does not match the pinned observation hash')
    snaps = defaultdict(set)                       # team -> set of dt strings with a QB row
    rank1 = defaultdict(set)                       # (team, dt) -> QB rank-1 identities
    with open(depth_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['pos_abb'] != 'QB':
                continue
            snaps[row['team']].add(row['dt'])
            if row['pos_rank'] in ('1', '1.0'):
                rank1[(row['team'], row['dt'])].add(row['gsis_id'] or row['espn_id'])
    ordered = {t: sorted(S.parse_ts(d) for d in v) for t, v in snaps.items()}
    raw = {t: sorted(v) for t, v in snaps.items()}
    result = {}
    for label in S.CUTOFFS:
        ages, have, fresh, single_r1, multi_r1 = [], 0, 0, 0, 0
        total = 0
        for g in games_2025:
            for team in (g['away_team'], g['home_team']):
                total += 1
                cutoff = S.cutoff_time(g['kickoff_utc'], label)
                before = [d for d in ordered.get(team, []) if d <= cutoff]
                if not before:
                    continue
                have += 1
                latest = before[-1]
                age = (cutoff - latest).total_seconds() / 3600
                ages.append(age)
                if age <= FRESH_HOURS:
                    fresh += 1
                key = (team, [r for r in raw[team] if S.parse_ts(r) == latest][0])
                n1 = len(rank1.get(key, ()))
                single_r1 += n1 == 1
                multi_r1 += n1 > 1
        ages.sort()
        result[label] = {'team_games': total, 'with_any_qb_snapshot_before_cutoff': have, 'with_snapshot_within_72h': fresh, 'single_rank1_qb_in_latest_snapshot': single_r1,
                         'multiple_rank1_qbs_in_latest_snapshot': multi_r1,
                         'median_age_hours': ages[len(ages) // 2] if ages else None, 'p90_age_hours': ages[int(len(ages) * 0.9)] if ages else None}
    # post-registration descriptive addition (amendment 1 in the results file): do consecutive snapshots ever differ? Counts only; no outcome is read.
    transitions, teams_with_change = 0, 0
    for team, stamps in raw.items():
        keys = sorted(stamps, key=S.parse_ts)
        sets = [frozenset(rank1.get((team, k), ())) for k in keys]
        changes = sum(1 for a, b in zip(sets, sets[1:]) if a != b)
        transitions += changes
        teams_with_change += changes > 0
    moved = 0
    for g in games_2025:
        for team in (g['away_team'], g['home_team']):
            sets = []
            for label in S.CUTOFFS:
                cutoff = S.cutoff_time(g['kickoff_utc'], label)
                before = [k for k in raw.get(team, []) if S.parse_ts(k) <= cutoff]
                sets.append(frozenset(rank1.get((team, max(before, key=S.parse_ts)), ())) if before else None)
            moved += sets[0] is not None and sets[1] is not None and sets[0] != sets[1]
    revision_visibility = {'consecutive_snapshot_rank1_qb_changes': transitions, 'teams_with_any_rank1_qb_change': teams_with_change, 'team_games_rank1_qb_differs_between_T24_and_T90_snapshots': moved}
    dts = sorted({d for v in snaps.values() for d in v})
    return {'revision_visibility_post_registration_addition': revision_visibility, 'file_sha256': DEPTH_2025_SHA256, 'qb_snapshot_datetimes': len(dts), 'first_dt': dts[0], 'last_dt': dts[-1], 'teams_with_qb_rows': len(snaps),
            'distinct_snapshot_days': len({d[:10] for d in dts}), 'cutoffs': result,
            'interpretation': 'dt is when the record was LOADED. These counts measure timing availability only; they do not certify announced starters, health or original vintage, and no outcome was read.'}


# --------------------------------------------------------------------------- acceptance questions
QUESTIONS = ('can_query_2024_game', 'expected_qb_state_before_kickoff', 'timestamp_proving_it', 'revisions_available', 'health_injury_available',
             'joins_to_nflverse_ids', 't24_reconstructable', 't90_reconstructable', 'original_vintage_not_backfill', 'same_mechanism_live')
STATUSES = ('PASS', 'FAIL', 'NOT_TESTED')


def question_row(provider, answers, basis):
    bad = [k for k, v in answers.items() if k not in QUESTIONS or v not in STATUSES]
    if bad:
        raise ValueError('bad answers: ' + ','.join(bad))
    full = {q: answers.get(q, 'NOT_TESTED') for q in QUESTIONS}
    return {'provider': provider, 'answers': full, 'basis': basis, 'passes_all': all(v == 'PASS' for v in full.values())}


def historical_thresholds_met(coverage):
    """Preregistered numeric gate. `coverage` is measured on a TESTED payload; None means never measured."""
    if not coverage:
        return False
    return (coverage.get('certified_pre_cutoff_team_game_fraction', 0) >= 0.95 and coverage.get('gsis_join_fraction', 0) >= 0.99
            and coverage.get('publication_timestamp_fraction', 0) >= 1.0 and coverage.get('original_vintage_required_met') is True)


def continuity_hint_audit(receipts, certified_state):
    """Diagnostic only. certified_state: {(game_id, team): {'player_id', 'archetype'}} from CERTIFIED pregame rows.

    Returns NOT_RUN when no certified rows exist. Hints are never promoted.
    """
    if not certified_state:
        return {'status': 'NOT_RUN_NO_CERTIFIED_STARTER_STATE', 'hint_rows': sum(1 for r in receipts if r.get('pregame_continuity_hint_not_certified')),
                'metrics': None}
    slices = defaultdict(lambda: {'n': 0, 'continuity_correct': 0, 'false_continuity': 0, 'false_change': 0})

    def add(name, correct, fc, fch):
        s = slices[name]
        s['n'] += 1
        s['continuity_correct'] += correct
        s['false_continuity'] += fc
        s['false_change'] += fch
    for r in receipts:
        if not r.get('pregame_continuity_hint_not_certified'):
            continue
        cert = certified_state.get((r['game_id'], r['team']))
        if not cert:
            continue
        hint_matches = cert['player_id'] == r['player_id']
        # A hint asserts continuity (the prior-dominant QB starts). It is a false continuity when the certified starter differs.
        for name in ('ALL', cert.get('archetype', 'UNLABELLED')):
            add(name, int(hint_matches), int(not hint_matches), 0)
    out = {}
    for name, s in sorted(slices.items()):
        n = s['n']
        out[name] = {**s, 'continuity_accuracy': s['continuity_correct'] / n, 'false_continuity_rate': s['false_continuity'] / n}
    return {'status': 'RUN_DIAGNOSTIC_ONLY', 'metrics': out,
            'note': 'false_change_rate needs certified rows for teams WITHOUT a hint; it is reported only when such rows are supplied'}


def assemble(sample_doc, timing, wayback):
    """Results document. Explicit about what was and was not tested."""
    q = [
        question_row('Sportradar', {}, 'Documentation only. NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('SportsDataIO', {}, 'Documentation only; no NFL trial. NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('Stats Perform / Opta', {}, 'No documentation of the product; NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('Genius Sports', {}, 'NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('Sports Info Solutions', {}, 'NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('TruMedia', {}, 'API has no pregame-state endpoint; NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('FTN full API', {}, 'Postgame charting; NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('PFF', {}, 'NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('NFL official injury report', {}, 'Public mutable page; not scraped; NO_PROVIDER_PAYLOAD_TESTED.'),
        question_row('nflverse depth_charts 2024', {'can_query_2024_game': 'PASS', 'expected_qb_state_before_kickoff': 'FAIL', 'timestamp_proving_it': 'FAIL', 'revisions_available': 'FAIL',
                                                    'health_injury_available': 'FAIL', 'joins_to_nflverse_ids': 'PASS', 't24_reconstructable': 'FAIL', 't90_reconstructable': 'FAIL',
                                                    'original_vintage_not_backfill': 'FAIL', 'same_mechanism_live': 'NOT_TESTED'},
                     'Repository audit (Phase1L): 2024 QB rows exist but zero dt timestamps.'),
        question_row('nflverse injuries 2024', {'can_query_2024_game': 'PASS', 'health_injury_available': 'PASS', 'joins_to_nflverse_ids': 'PASS', 'expected_qb_state_before_kickoff': 'FAIL',
                                               'timestamp_proving_it': 'FAIL', 'revisions_available': 'FAIL', 't24_reconstructable': 'FAIL', 't90_reconstructable': 'FAIL',
                                               'original_vintage_not_backfill': 'FAIL', 'same_mechanism_live': 'NOT_TESTED'},
                     'Repository audit (Phase1L): modification timestamps only (6,215 rows), zero original publication rows; no starter identity.'),
        question_row('nflverse depth_charts 2025+ (forward)', {'can_query_2024_game': 'FAIL', 'timestamp_proving_it': 'FAIL', 'joins_to_nflverse_ids': 'PASS', 'same_mechanism_live': 'NOT_TESTED'},
                     'dt is a load time (not publication); 2024 absent; timing probe is counts only; live payload not opened (Week 5+ policy).'),
        question_row('Internet Archive Availability API', {'can_query_2024_game': 'PASS' if wayback and wayback['completed_requests'] else 'NOT_TESTED', 'joins_to_nflverse_ids': 'FAIL',
                                                          'revisions_available': 'NOT_TESTED', 'original_vintage_not_backfill': 'NOT_TESTED'},
                     'Existence/capture-time metadata only; archived page bodies were not opened; HTML pages carry names, not ids.'),
    ]
    return {
        'schema': 'nfl-v2-phase1m-qb-state-acceptance-results-v1',
        'base_head': S.BASE_HEAD, 'evidence_date_utc': S.EVIDENCE_DATE_UTC,
        'headline': 'NO_PROVIDER_PAYLOAD_TESTED for every vendor. No source passed historical or forward acceptance.',
        'vendor_payloads_tested': 0,
        'historical_thresholds_met': False,
        'acceptance_by_provider': q,
        'sample': sample_doc,
        'repository_depth_timing_probe_2025': timing,
        'web_archive_probe': wayback,
        'historical_archetype_test': {'status': 'NOT_RUN_NO_SOURCE_ACCESS', 'reason': 'The sample is preregistered and fixed, but no accepted payload exists to evaluate. Archetypes not separable from realized data (injury replacement, returning starter, rookie promotion, benching, recent acquisition) require independent pregame evidence and remain unlabelled.'},
        'prior_continuity_hint_audit': {'status': 'NOT_RUN_NO_CERTIFIED_STARTER_STATE', 'receipts_file': 'phase1l_qb_receipts.jsonl.gz', 'hint_rows': None,
                                         'note': 'Hints are not promoted. The metrics function exists and is tested on synthetic certified rows only.'},
        'live_payload_policy': 'No Week 5+ (2026) player-state payload was opened. Live capability is assessed from documentation, release metadata and the forward plan only.',
        'no_models': True, 'no_sportsbook': True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=False)
    ap.add_argument('--select-sample', action='store_true')
    ap.add_argument('--timing-probe', action='store_true')
    ap.add_argument('--probe-wayback', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--timing', default=None)
    ap.add_argument('--wayback', action='append', default=[], help='one or more probe attempt files, in attempt order')
    ap.add_argument('--sample', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    data = Path(args.data_dir) if args.data_dir else None
    if args.select_sample:
        games = load_games(data / 'games.csv', 2024)
        labelled = label_team_games(games, load_qb_attempts(data / 'stats_player_week_2024.csv'))
        doc = {'games_2024_reg_digest': games_digest(games), 'games': len(games), 'stats_sha256': STATS_2024_SHA256, 'labelled_team_games': len(labelled), 'eligible_per_archetype': eligible_counts(labelled),
               'per_archetype_selected': PER_LABEL, 'sample': select_sample(labelled)}
        Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    elif args.timing_probe:
        games = load_games(data / 'games.csv', 2025)
        Path(args.out).write_text(json.dumps(timing_probe(data / 'depth_charts_2025.csv', games), indent=2, sort_keys=True) + '\n')
    elif args.assemble:
        sample_doc = json.loads(Path(args.sample).read_text())
        timing = json.loads(Path(args.timing).read_text())
        attempts = [json.loads(Path(w).read_text()) for w in args.wayback]
        # the last attempt is the headline; every attempt is retained for audit
        headline = dict(attempts[-1]) if attempts else None
        if headline is not None:
            headline['attempt_summaries'] = [{'attempt': i + 1, 'completed_requests': a['completed_requests'], 'planned_requests': a['planned_requests'], 'stopped': a['stopped'],
                                              'first_observed_at_utc': a['results'][0]['observed_at_utc'] if a['results'] else None} for i, a in enumerate(attempts)]
        Path(args.out).write_text(S.dump(assemble(sample_doc, timing, headline)))
    elif args.probe_wayback:
        sample = json.loads(Path(args.sample).read_text())['sample']
        Path(args.out).write_text(json.dumps(wayback_probe(sample), indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

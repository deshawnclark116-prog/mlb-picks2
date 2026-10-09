#!/usr/bin/env python3
"""Phase2B-DATA forward availability capture (research only; data capture / measurement, NOT a model).

Provider-neutral, append-only archive of offensive-player availability observations at the Phase2A T24 / T90 windows, with an as-of lookup
availability_at(player, game, cutoff_time) that can never leak a later update backward. It READS Phase2A (windows, ledger) and never writes to it:
Phase2A engine v1.0, its ledgers and its lock are untouched. Nothing connects to an unaccepted source; QB state is routed through the existing
Phase1M / Phase2A QB-state ingestion layer, not duplicated. No data are invented: fixtures live only in tests.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile

import nfl_v2_phase2a_forward_forecast as F
import nfl_v2_phase2a_forward_grade as G
import nfl_v2_qb_state_ingestion as Q
from nfl_v2_phase2a_ledger import append_record, canon, read_ledger, rnd, sha_bytes, verify_chain

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
RAW_DIR = ART / 'phase2b_availability_raw'
NORM_DIR = ART / 'phase2b_availability_normalized'
RAW_FILE, NORM_FILE, EVENT_FILE = 'raw_payloads.jsonl', 'availability_normalized.jsonl', 'capture_events.jsonl'
ACCEPTANCE = 'phase2b_source_acceptance.json'
PROTOCOL = 'phase2b_availability_protocol.json'
UTC = timezone.utc
STATES = ('ACTIVE', 'EXPECTED_ACTIVE', 'QUESTIONABLE', 'DOUBTFUL', 'OUT', 'INACTIVE', 'IR', 'PUP', 'LIMITED', 'FULL', 'DNP', 'RETURNING', 'UNKNOWN')
SEVERITY = ('OUT', 'INACTIVE', 'IR', 'PUP', 'DOUBTFUL', 'DNP', 'QUESTIONABLE', 'LIMITED', 'RETURNING', 'UNKNOWN', 'EXPECTED_ACTIVE', 'FULL', 'ACTIVE')
OFFENSIVE_POSITIONS = ('QB', 'RB', 'FB', 'HB', 'WR', 'TE')
ROSTER_STATE = {'ACT': 'EXPECTED_ACTIVE', 'INA': 'INACTIVE', 'RES': 'IR', 'PUP': 'PUP'}
REPORT_STATE = {'OUT': 'OUT', 'DOUBTFUL': 'DOUBTFUL', 'QUESTIONABLE': 'QUESTIONABLE'}
PRACTICE_STATE = {'FULL PARTICIPATION IN PRACTICE': 'FULL', 'LIMITED PARTICIPATION IN PRACTICE': 'LIMITED', 'DID NOT PARTICIPATE IN PRACTICE': 'DNP'}
CLASSIFICATIONS = ('ACCEPTED_FORWARD_CAPTURE', 'PROMISING_NEEDS_TERMS', 'PROMISING_NEEDS_ACCESS', 'BLOCKED_LICENSE', 'BLOCKED_TIMING', 'NOT_USEFUL')
FIXTURE_LABEL = 'FIXTURE_ONLY_NOT_REAL_AVAILABILITY_EVIDENCE'
# (name, type, required, description)
SCHEMA_FIELDS = (
    ('season', 'integer', True, 'NFL season'), ('week', 'integer', True, 'regular-season week'), ('game_id', 'string', True, 'nflverse game id'), ('team', 'string', True, 'team abbreviation'),
    ('opponent', 'string', True, 'opponent abbreviation'), ('kickoff_time', 'string', True, 'UTC ISO-8601 kickoff'), ('player_id', 'string', True, 'GSIS id (exact; never name-matched)'),
    ('player_name', 'string', True, 'as published'), ('position', 'string', True, 'position as published'), ('state', 'enum', True, 'one of STATES'),
    ('practice_status', 'string|null', False, 'practice participation wording'), ('game_status', 'string|null', False, 'game designation wording'), ('injury_body_part', 'string|null', False, 'body part if supplied'),
    ('injury_detail', 'string|null', False, 'injury detail if supplied'), ('roster_status', 'string|null', False, 'roster status code'), ('source_provider', 'string', True, 'provider'),
    ('source_product', 'string', True, 'product / endpoint'), ('source_record_id', 'string', True, 'stable id of the source record across revisions'), ('published_at', 'string|null', False, 'FIRST public publication of this revision; null when the source gives none'),
    ('retrieved_at', 'string', True, 'when WE retrieved the bytes'), ('effective_at', 'string|null', False, 'when the state became effective, if supplied'), ('cutoff_eligibility', 'enum', True, 'T24 | T90 | BOTH | NEITHER (visibility at the cutoffs)'),
    ('valid_from', 'string', True, 'start of validity'), ('valid_to', 'string|null', False, 'end of validity; null = not superseded within the row'), ('revision_sequence', 'integer', True, 'monotone per (source_provider, source_record_id)'),
    ('original_vintage', 'boolean', True, 'bytes captured live, inside the capture window'), ('backfilled', 'boolean', True, 'added after the fact'), ('stale', 'boolean', True, 'source known to be stale'),
    ('raw_payload_sha256', 'string', True, 'sha256 of the raw payload'), ('normalized_record_sha256', 'string', True, 'sha256 of the canonical record without this field'), ('confidence', 'enum', False, 'LOW | MEDIUM | HIGH'),
    ('capture_window_status', 'enum', True, 'IN_T24_WINDOW | IN_T90_WINDOW | OUTSIDE_WINDOWS (Phase2A windows)'), ('qb_route', 'string|null', False, 'ROSTER_STATUS_ONLY_NOT_A_QB_STATE for QB rows'))
FIELD_NAMES = tuple(f[0] for f in SCHEMA_FIELDS)
REQUIRED = tuple(f[0] for f in SCHEMA_FIELDS if f[2])


class SourceNotAccepted(RuntimeError):
    """Raised when a provider is not ACCEPTED_FORWARD_CAPTURE for the requested scope."""


# --------------------------------------------------------------------------- source acceptance
def acceptance_doc(path=None):
    path = Path(path or ART / ACCEPTANCE)
    return json.loads(path.read_text()) if path.exists() else {'sources': []}


def source_entry(provider, path=None):
    return next((s for s in acceptance_doc(path)['sources'] if s['provider'] == provider), None)


def accepted(provider, path=None):
    s = source_entry(provider, path)
    return bool(s and s['classification'] == 'ACCEPTED_FORWARD_CAPTURE' and s.get('authorized') is True)


# --------------------------------------------------------------------------- adapters (pure transforms of data handed in; never network)
def parse_ts(v):
    return F.datetime.fromisoformat(str(v).replace('Z', '+00:00')) if v else None


def window_status(kickoff, retrieved):
    for label, name in (('T24', 'IN_T24_WINDOW'), ('T90', 'IN_T90_WINDOW')):
        opens, cut = F.window(kickoff, label)
        if opens <= retrieved < cut:
            return name
    return 'OUTSIDE_WINDOWS'


def eligibility(known_at, retrieved, kickoff):
    labels = [lab for lab in ('T24', 'T90') if known_at <= F.window(kickoff, lab)[1] and retrieved <= F.window(kickoff, lab)[1]]
    return 'BOTH' if len(labels) == 2 else (labels[0] if labels else 'NEITHER')


def build_record(game, item, provider, product, retrieved_at, raw_sha, **kw):
    kickoff = F.kickoff_utc(game['gameday'], game['gametime'])
    retrieved = parse_ts(retrieved_at)
    published = parse_ts(kw.get('published_at'))
    known = published or retrieved
    wstat = window_status(kickoff, retrieved)
    postgame = retrieved >= kickoff
    rec = {'season': int(game['season']), 'week': int(game['week']), 'game_id': game['game_id'], 'team': item['team'], 'opponent': game['away_team'] if item['team'] == game['home_team'] else game['home_team'], 'kickoff_time': F.iso(kickoff),
           'player_id': item['player_id'], 'player_name': item.get('player_name') or '', 'position': item.get('position') or '', 'state': kw['state'], 'practice_status': kw.get('practice_status'), 'game_status': kw.get('game_status'),
           'injury_body_part': kw.get('injury_body_part'), 'injury_detail': kw.get('injury_detail'), 'roster_status': kw.get('roster_status'), 'source_provider': provider, 'source_product': product,
           'source_record_id': f"{provider}|{game['season']}|{game['week']}|{item['team']}|{item['player_id']}", 'published_at': F.iso(published) if published else None, 'retrieved_at': F.iso(retrieved), 'effective_at': kw.get('effective_at'),
           'cutoff_eligibility': eligibility(known, retrieved, kickoff), 'valid_from': F.iso(known), 'valid_to': None, 'revision_sequence': None, 'original_vintage': bool(kw.get('original_vintage', wstat != 'OUTSIDE_WINDOWS') and not postgame),
           'backfilled': bool(kw.get('backfilled', postgame)), 'stale': bool(kw.get('stale', False)), 'raw_payload_sha256': raw_sha, 'normalized_record_sha256': None, 'confidence': kw.get('confidence'), 'capture_window_status': wstat,
           'qb_route': 'ROSTER_STATUS_ONLY_NOT_A_QB_STATE' if (item.get('position') == 'QB' and provider == 'nflverse_roster_weekly') else None}
    return rec


class RosterAdapter:
    provider, product, scope = 'nflverse_roster_weekly', 'weekly_rosters', 'roster_status_only'

    def records(self, payload, games, retrieved_at, raw_sha):
        out = []
        for r in payload['rows']:
            game = games.get((int(r['season']), int(float(r['week'])), r['team']))
            if game is None or not r.get('gsis_id') or (r.get('position') or '') not in OFFENSIVE_POSITIONS or r.get('game_type', 'REG') not in ('REG', ''):
                continue
            status = (r.get('status') or '').upper()
            item = {'team': r['team'], 'player_id': r['gsis_id'], 'player_name': r.get('full_name'), 'position': r.get('position')}
            out.append(build_record(game, item, self.provider, self.product, retrieved_at, raw_sha, state=ROSTER_STATE.get(status, 'UNKNOWN'), roster_status=r.get('status') or None, confidence='LOW'))
        return out


class InjuriesAdapter:
    """nflverse injuries release shape. NOT accepted (PROMISING_NEEDS_TERMS): only usable after the acceptance artifact says so."""
    provider, product, scope = 'nflverse_injuries', 'injuries', 'injury_designations'

    def records(self, payload, games, retrieved_at, raw_sha):
        out = []
        for r in payload['rows']:
            game = games.get((int(r['season']), int(float(r['week'])), r['team']))
            if game is None or not r.get('gsis_id') or r.get('position') not in OFFENSIVE_POSITIONS:
                continue
            report = (r.get('report_status') or '').upper()
            practice = (r.get('practice_status') or '').upper()
            state = REPORT_STATE.get(report) or PRACTICE_STATE.get(practice) or 'UNKNOWN'
            item = {'team': r['team'], 'player_id': r['gsis_id'], 'player_name': r.get('full_name'), 'position': r.get('position')}
            out.append(build_record(game, item, self.provider, self.product, retrieved_at, raw_sha, state=state, practice_status=r.get('practice_status') or None, game_status=r.get('report_status') or None, injury_body_part=r.get('report_primary_injury') or r.get('practice_primary_injury') or None,
                                    injury_detail=r.get('report_secondary_injury') or r.get('practice_secondary_injury') or None, confidence='MEDIUM'))
        return out


class ManualAdapter:
    """Operator-supplied records; each must carry an authorization_ref and an original publication time or it is rejected."""
    provider, product, scope = 'manually_supplied_authorized_records', 'manual', 'records_with_authorization_ref_and_original_publication_time'

    def records(self, payload, games, retrieved_at, raw_sha):
        out = []
        for r in payload['rows']:
            if not r.get('authorization_ref') or not r.get('published_at'):
                raise SourceNotAccepted('manual record without authorization_ref and original publication time')
            game = games.get((int(r['season']), int(float(r['week'])), r['team']))
            if game is None or not r.get('player_id'):
                continue
            item = {'team': r['team'], 'player_id': r['player_id'], 'player_name': r.get('player_name'), 'position': r.get('position')}
            out.append(build_record(game, item, self.provider, self.product, retrieved_at, raw_sha, state=r['state'], published_at=r['published_at'], practice_status=r.get('practice_status'), game_status=r.get('game_status'),
                                    injury_body_part=r.get('injury_body_part'), injury_detail=r.get('injury_detail'), roster_status=r.get('roster_status'), effective_at=r.get('effective_at'), confidence=r.get('confidence') or 'MEDIUM'))
        return out


ADAPTERS = {a.provider: a for a in (RosterAdapter, InjuriesAdapter, ManualAdapter)}


def validate(rec):
    problems = [f'missing:{n}' for n in REQUIRED if rec.get(n) is None and n not in ('normalized_record_sha256', 'revision_sequence')]
    if set(rec) - set(FIELD_NAMES):
        problems.append('unknown_fields:' + ','.join(sorted(set(rec) - set(FIELD_NAMES))))
    if rec.get('state') not in STATES:
        problems.append('bad_state')
    if rec.get('cutoff_eligibility') not in ('T24', 'T90', 'BOTH', 'NEITHER'):
        problems.append('bad_cutoff_eligibility')
    if not str(rec.get('player_id') or '').startswith('00-'):
        problems.append('player_id_not_gsis')
    return problems


def record_hash(rec):
    return sha_bytes(canon({k: v for k, v in rec.items() if k != 'normalized_record_sha256'}).encode())


def _content(rec):
    skip = {'retrieved_at', 'raw_payload_sha256', 'revision_sequence', 'normalized_record_sha256', 'cutoff_eligibility', 'capture_window_status', 'valid_from', 'original_vintage', 'backfilled', 'published_at'}
    return {k: v for k, v in rec.items() if k not in skip}


# --------------------------------------------------------------------------- ingestion
def ingest_availability(payload, retrieved_at, provider, games, raw_dir=None, norm_dir=None, acceptance_path=None):
    """Provider-neutral entry point. Refuses any provider that is not ACCEPTED_FORWARD_CAPTURE. games: {(season, week, team): schedule game row}."""
    if provider not in ADAPTERS:
        raise SourceNotAccepted(f'unknown provider adapter: {provider}')
    if not accepted(provider, acceptance_path):
        raise SourceNotAccepted(f'{provider} is not ACCEPTED_FORWARD_CAPTURE ({(source_entry(provider, acceptance_path) or {}).get("classification", "NOT_AUDITED")})')
    raw_dir, norm_dir = Path(raw_dir or RAW_DIR), Path(norm_dir or NORM_DIR)
    raw_sha = payload.get('source_file_sha256') or sha_bytes(canon(payload['rows']).encode())
    append_record(raw_dir / RAW_FILE, {'raw_id': f'{provider}|{raw_sha}|{retrieved_at}', 'provider': provider, 'retrieved_at': retrieved_at, 'source_url': payload.get('source_url'), 'source_file_sha256': raw_sha, 'rows': len(payload['rows']), 'payload': payload['rows']}, 'raw_id')
    records = ADAPTERS[provider]().records(payload, games, retrieved_at, raw_sha)
    existing = defaultdict(list)
    for line in read_ledger(norm_dir / NORM_FILE):
        existing[line['record']['source_record_id']].append(line['record'])
    stats = {'rows': len(records), 'new': 0, 'revisions': 0, 'unchanged': 0}
    for rec in records:
        problems = validate(rec)
        if problems:
            raise ValueError('invalid availability record: ' + ';'.join(problems))
        hist = existing.get(rec['source_record_id'], [])
        if hist and _content(rec) == _content(hist[-1]):
            stats['unchanged'] += 1
            continue
        rec['revision_sequence'] = (hist[-1]['revision_sequence'] + 1) if hist else 1
        rec['normalized_record_sha256'] = record_hash(rec)
        append_record(norm_dir / NORM_FILE, rec, 'normalized_record_sha256')
        existing[rec['source_record_id']].append(rec)
        stats['revisions' if hist else 'new'] += 1
    return stats


def route_qb_payload(payload, retrieved_at, provider, crosswalk=None, **kw):
    """QB starter / health provider payloads are NOT stored here: they go through the existing Phase1M / Phase2A QB-state ingestion layer."""
    return Q.ingest_qb_state(payload, retrieved_at, provider, crosswalk=crosswalk, **kw)


def qb_state_link(team, game_id, cutoff, label, **kw):
    """The QB gate for a team-game exactly as the Phase2A engine asks it (BLOCKED while no QB provider is accepted)."""
    return Q.certify_for_forecast(team, game_id, cutoff, label, **kw)


# --------------------------------------------------------------------------- as-of lookup
def visible(rec, cutoff, label=None):
    cutoff = parse_ts(cutoff) if not isinstance(cutoff, datetime) else cutoff
    retrieved = parse_ts(rec['retrieved_at'])
    published = parse_ts(rec['published_at'])
    known = published or retrieved
    if (published is not None and published > cutoff) or retrieved > cutoff or known > cutoff:
        return False
    if parse_ts(rec['valid_from']) > cutoff or (rec['valid_to'] and cutoff >= parse_ts(rec['valid_to'])):
        return False
    if retrieved >= parse_ts(rec['kickoff_time']):
        return False                                                       # postgame capture
    if rec['backfilled'] and not rec['original_vintage']:
        return False
    if label is not None and rec['cutoff_eligibility'] not in (label, 'BOTH'):
        return False
    return True


def availability_at(player, game, cutoff_time, label=None, norm_dir=None, raw_dir=None):
    """Availability of `player` (GSIS id) for `game` (game_id or dict) as of cutoff_time using ONLY records visible by then."""
    norm_dir = Path(norm_dir or NORM_DIR)
    game_id = game['game_id'] if isinstance(game, dict) else game
    cutoff = parse_ts(cutoff_time) if not isinstance(cutoff_time, datetime) else cutoff_time
    best = {}
    for line in read_ledger(norm_dir / NORM_FILE):
        r = line['record']
        if r['game_id'] != game_id or r['player_id'] != player or not visible(r, cutoff, label):
            continue
        if r['source_record_id'] not in best or r['revision_sequence'] > best[r['source_record_id']]['revision_sequence']:
            best[r['source_record_id']] = r
    rows = [best[k] for k in sorted(best)]
    confirmations = [e['record'] for e in read_ledger(norm_dir / EVENT_FILE) if e['record']['event'] == 'CAPTURED' and e['record']['game_id'] == game_id and parse_ts(e['record']['retrieved_at']) <= cutoff]
    if not rows:
        return {'state': 'UNKNOWN', 'reason': 'NO_VISIBLE_RECORD', 'records': [], 'conflict': False, 'captures_confirming': len(confirmations)}
    states = sorted({r['state'] for r in rows}, key=SEVERITY.index)
    return {'state': states[0], 'reason': 'MOST_SEVERE_VISIBLE_STATE' if len(states) > 1 else 'VISIBLE_RECORD', 'records': rows, 'conflict': len(states) > 1, 'captures_confirming': len(confirmations)}


# --------------------------------------------------------------------------- capture run (inside the Phase2A windows only)
def capture_roster(roster_rows, source_file_sha256, retrieved_at, schedule, clock=F.now_utc, raw_dir=None, norm_dir=None, acceptance_path=None, game_filter=None, source_url=None, evaluation_start=None):
    """Capture roster-status observations for every in-scope game whose window is open; log MISSED_CAPTURE for closed windows. Never creates a forecast."""
    start = evaluation_start or datetime.fromisoformat(F.read_json(ART / F.PROTOCOL)['evaluation_start_utc'])
    raw_dir, norm_dir = Path(raw_dir or RAW_DIR), Path(norm_dir or NORM_DIR)
    games = {}
    out = {'captured': [], 'missed': [], 'not_yet_due': 0}
    now = clock()
    retrieved = parse_ts(retrieved_at)
    scope = [g for g in schedule if F.kickoff_utc(g['gameday'], g['gametime']) >= start and (game_filter is None or g['game_id'] in game_filter)]
    for g in scope:
        ko = F.kickoff_utc(g['gameday'], g['gametime'])
        for label in ('T24', 'T90'):
            status = F.classify(ko, label, now)
            if status == 'NOT_YET_DUE':
                out['not_yet_due'] += 1
                continue
            cutoff = F.window(ko, label)[1]
            done = any(e['record']['event'] == 'CAPTURED' and e['record']['game_id'] == g['game_id'] and e['record']['cutoff_type'] == label for e in read_ledger(norm_dir / EVENT_FILE))
            if status != 'DUE' or retrieved >= cutoff:
                if not done:
                    line = append_record(norm_dir / EVENT_FILE, {'event_id': f"{g['game_id']}|{label}|MISSED_CAPTURE", 'event': 'MISSED_CAPTURE', 'game_id': g['game_id'], 'cutoff_type': label, 'provider': 'nflverse_roster_weekly', 'cutoff_time': F.iso(cutoff), 'observed_at': F.iso(now),
                                                                  'note': 'no capture existed before the cutoff; never backfilled'}, 'event_id')
                    if line:
                        out['missed'].append((g['game_id'], label))
                continue
            wk_games = {(int(g['season']), int(g['week']), g['home_team']): g, (int(g['season']), int(g['week']), g['away_team']): g}
            rows = [r for r in roster_rows if (int(r['season']), int(float(r['week'])), r['team']) in wk_games]
            payload = {'rows': rows, 'source_file_sha256': source_file_sha256, 'source_url': source_url}
            stats = ingest_availability(payload, retrieved_at, 'nflverse_roster_weekly', wk_games, raw_dir, norm_dir, acceptance_path)
            append_record(norm_dir / EVENT_FILE, {'event_id': f"{g['game_id']}|{label}|CAPTURED|{retrieved_at}", 'event': 'CAPTURED', 'game_id': g['game_id'], 'cutoff_type': label, 'provider': 'nflverse_roster_weekly', 'cutoff_time': F.iso(cutoff),
                                                  'retrieved_at': retrieved_at, 'observed_at': F.iso(clock()), **stats}, 'event_id')
            out['captured'].append((g['game_id'], label, stats))
    return out


# --------------------------------------------------------------------------- observational link to Phase2A (read-only)
def link_phase2a(ledger_dir=None, norm_dir=None):
    """For every Phase2A forecast: the availability state visible at ITS cutoff. Reads the Phase2A ledger; never writes to it."""
    ledger_dir = Path(ledger_dir or F.LEDGER)
    links = []
    for name in ('forecasts_T24.jsonl', 'forecasts_T90.jsonl'):
        for line in read_ledger(ledger_dir / name):
            r = line['record']
            a = availability_at(r['player_id'], r['game_id'], r['cutoff_time'], r['cutoff_type'], norm_dir)
            links.append({'forecast_id': r['forecast_id'], 'game_id': r['game_id'], 'player_id': r['player_id'], 'cutoff_type': r['cutoff_type'], 'availability_state': a['state'], 'reason': a['reason'], 'conflict': a['conflict'],
                          'record_sha256': [x['normalized_record_sha256'] for x in a['records']], 'phase2a_availability_state': r['availability_state'], 'phase2a_record_sha256': line['record_sha256']})
    return links


DOWN = {'OUT', 'INACTIVE', 'IR', 'PUP'}
QUESTIONABLE = {'QUESTIONABLE', 'DOUBTFUL', 'DNP', 'LIMITED'}


def state_group(state):
    if state in DOWN:
        return 'OUT_OR_INACTIVE'
    if state == 'DOUBTFUL':
        return 'DOUBTFUL'
    if state in QUESTIONABLE:
        return 'QUESTIONABLE'
    if state == 'UNKNOWN':
        return 'UNKNOWN'
    return 'ACTIVE'


def availability_report(graded, links):
    """Descriptive value report over graded Phase2A rows and their availability links. No causality is claimed."""
    by_id = {x['forecast_id']: x for x in links}
    thr = {'receiving_yards': 40, 'rushing_yards': 30, 'targets': 4, 'carries': 8, 'receptions': 3}
    out = {'schema': 'nfl-v2-phase2b-availability-report-v1', 'status': 'NO_GRADED_GAMES_YET' if not graded else 'DESCRIPTIVE_ONLY_NO_CAUSALITY', 'graded_player_games': len(graded), 'links': len(links), 'unlinked_graded': 0,
           'counts_by_state': {}, 'mae_by_state': {}, 'status_changes_T24_to_T90': {}, 'catastrophic_association': {}}
    cnt = defaultdict(int)
    errs = defaultdict(lambda: defaultdict(list))
    cat = defaultdict(lambda: defaultdict(int))
    by_pg = defaultdict(dict)
    for r in graded:
        lk = by_id.get(r['forecast_id'])
        if lk is None:
            out['unlinked_graded'] += 1
            continue
        grp = state_group(lk['availability_state'])
        cnt[(r['cutoff_type'], grp)] += 1
        by_pg[(r['game_id'], r['player_id'])][r['cutoff_type']] = grp
        for head, v in r['heads'].items():
            e = abs(v['projection'] - v['actual'])
            errs[(r['cutoff_type'], grp)][head].append(e)
            if e > thr.get(head, 1e9):
                cat[(r['cutoff_type'], head)][grp] += 1
    out['counts_by_state'] = {f'{k[0]}|{k[1]}': v for k, v in sorted(cnt.items())}
    out['mae_by_state'] = {f'{k[0]}|{k[1]}': {h: {'n': len(v), 'mae': sum(v) / len(v)} for h, v in sorted(hv.items())} for k, hv in sorted(errs.items())}
    changes = defaultdict(int)
    order = ['ACTIVE', 'UNKNOWN', 'QUESTIONABLE', 'DOUBTFUL', 'OUT_OR_INACTIVE']
    for (_g, _p), d in by_pg.items():
        if 'T24' in d and 'T90' in d:
            a, b = d['T24'], d['T90']
            kind = 'UNCHANGED' if a == b else ('DOWNGRADE' if order.index(b) > order.index(a) else 'UPGRADE')
            changes[f'{kind}:{a}->{b}'] += 1
    out['status_changes_T24_to_T90'] = dict(sorted(changes.items()))
    out['catastrophic_association'] = {f'{k[0]}|{k[1]}': dict(sorted(v.items())) for k, v in sorted(cat.items())}
    out['categories'] = {'status_downgrade': 'T24 group less severe than T90 group', 'late_inactive': 'OUT_OR_INACTIVE at T90', 'incomplete_availability_state': 'UNKNOWN or no visible record'}
    return rnd(out)


# --------------------------------------------------------------------------- operator aid (read-only)
def next_due(schedule, ledger_dir=None, now=None, start=None):
    """The next (or currently open) Phase2A forecast window; creates nothing."""
    now = now or F.now_utc()
    start = start or datetime.fromisoformat(F.read_json(ART / F.PROTOCOL)['evaluation_start_utc'])
    ledger_dir = Path(ledger_dir or F.LEDGER)
    have = {lab: {x['record']['game_id'] for x in read_ledger(ledger_dir / f'forecasts_{lab}.jsonl')} for lab in ('T24', 'T90')}
    windows = defaultdict(list)
    for g in schedule:
        ko = F.kickoff_utc(g['gameday'], g['gametime'])
        if ko < start or ko <= now:
            continue
        for lab in ('T24', 'T90'):
            opens, cut = F.window(ko, lab)
            if cut > now and g['game_id'] not in have[lab]:
                windows[(lab, opens, cut)].append(g['game_id'])
    if not windows:
        return {'NEXT_DUE_FORECAST_WINDOW': None, 'opens_at': None, 'cutoff_at': None, 'games_due': [], 'state': 'NO_FUTURE_WINDOW'}
    open_now = sorted(k for k in windows if k[1] <= now < k[2])
    key = open_now[0] if open_now else min(windows, key=lambda k: (k[1], k[0]))
    lab, opens, cut = key
    return {'NEXT_DUE_FORECAST_WINDOW': lab, 'opens_at': F.iso(opens), 'cutoff_at': F.iso(cut), 'games_due': sorted(windows[key]), 'state': 'OPEN_NOW' if key in open_now else 'NOT_YET_OPEN',
            'minutes_until_open': None if key in open_now else round((opens - now).total_seconds() / 60, 1), 'minutes_until_cutoff': round((cut - now).total_seconds() / 60, 1), 'other_windows_open_now': [{'cutoff_type': k[0], 'opens_at': F.iso(k[1]), 'cutoff_at': F.iso(k[2]), 'games': sorted(windows[k])} for k in open_now[1:]]}


# --------------------------------------------------------------------------- documents
def schema_document():
    return {'schema': 'nfl-v2-phase2b-availability-schema-v1', 'status': 'DESIGN_COMPLETE_POPULATED_ONLY_BY_REAL_ACCEPTED_CAPTURES', 'append_only': True, 'fields': [{'name': n, 'type': t, 'required': r, 'description': d} for n, t, r, d in SCHEMA_FIELDS], 'states': list(STATES),
            'severity_order': list(SEVERITY), 'uniqueness': '(source_provider, source_record_id, revision_sequence)', 'canonical_hash': 'normalized_record_sha256 = sha256(canonical JSON of every other field)', 'visibility_rule': 'see phase2b_availability_protocol.json as_of_semantics; implemented by availability_at',
            'qb': 'QB starter / health state is NOT part of this schema: it uses the Phase1M schema through nfl_v2_qb_state_ingestion', 'fixtures': FIXTURE_LABEL + ' (tests only; never stored)'}


def build_manifest(raw_dir=None, norm_dir=None):
    raw_dir, norm_dir = Path(raw_dir or RAW_DIR), Path(norm_dir or NORM_DIR)
    led = {}
    for d, n in ((raw_dir, RAW_FILE), (norm_dir, NORM_FILE), (norm_dir, EVENT_FILE)):
        cnt, last = verify_chain(d / n)
        led[f'{d.name}/{n}'] = {'lines': cnt, 'last_line_sha256': last}
    events = [e['record'] for e in read_ledger(norm_dir / EVENT_FILE)]
    rows = [e['record'] for e in read_ledger(norm_dir / NORM_FILE)]
    return {'schema': 'nfl-v2-phase2b-capture-manifest-v1', 'ledgers': led, 'counts': {'normalized_rows': len(rows), 'captures': sum(1 for e in events if e['event'] == 'CAPTURED'), 'missed_captures': sum(1 for e in events if e['event'] == 'MISSED_CAPTURE'),
                                                                                       'games_captured': len({(e['game_id'], e['cutoff_type']) for e in events if e['event'] == 'CAPTURED'}),
                                                                                       'providers': sorted({r['source_provider'] for r in rows}), 'by_state': {s: sum(1 for r in rows if r['state'] == s) for s in sorted({r['state'] for r in rows})}},
            'capture_pairs': sorted([e['game_id'], e['cutoff_type']] for e in events if e['event'] == 'CAPTURED'), 'missed': sorted([e['game_id'], e['cutoff_type']] for e in events if e['event'] == 'MISSED_CAPTURE'),
            'accepted_sources': [s['provider'] for s in acceptance_doc()['sources'] if s['classification'] == 'ACCEPTED_FORWARD_CAPTURE'], 'phase2a_engine_modified': False, 'production_promotion': False}


# --------------------------------------------------------------------------- CLI
def _fetch_roster(dest, url):
    subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '5', '--max-time', '600', url, '-o', str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('next-due', 'capture-roster', 'link', 'build-docs'), required=True)
    ap.add_argument('--schedule-file', default=None, help='games csv (allowlisted columns are read); default: fetch with the Phase2A fallback logic')
    a = ap.parse_args()
    if a.mode == 'build-docs':
        (ART / 'phase2b_availability_schema.json').write_text(json.dumps(schema_document(), indent=2, sort_keys=True) + '\n')
        (ART / 'phase2b_capture_manifest.json').write_text(json.dumps(rnd(build_manifest()), indent=2, sort_keys=True) + '\n')
        return
    if a.schedule_file:
        sched = F.parse_schedule(Path(a.schedule_file).read_text())
    else:
        with tempfile.TemporaryDirectory() as d:
            sched, _meta = F.fetch_schedule(d)
    if a.mode == 'next-due':
        r = next_due(sched)
        print('NEXT_DUE_FORECAST_WINDOW', r['NEXT_DUE_FORECAST_WINDOW'])
        print('state', r['state'])
        print('opens_at', r['opens_at'])
        print('cutoff_at', r['cutoff_at'])
        print('games_due', ','.join(r['games_due']))
        return
    if a.mode == 'link':
        links = link_phase2a()
        print(json.dumps(availability_report([], links), indent=2, sort_keys=True))
        return
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / 'roster.csv'
        url = 'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.csv'
        _fetch_roster(dest, url)
        retrieved = F.iso(F.now_utc())
        data = dest.read_bytes()
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
    res = capture_roster(rows, sha_bytes(data), retrieved, sched, source_url=url)
    (ART / 'phase2b_capture_manifest.json').write_text(json.dumps(rnd(build_manifest()), indent=2, sort_keys=True) + '\n')
    print(json.dumps({'captured': [(g, l, s) for g, l, s in res['captured']], 'missed': res['missed'], 'not_yet_due': res['not_yet_due']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

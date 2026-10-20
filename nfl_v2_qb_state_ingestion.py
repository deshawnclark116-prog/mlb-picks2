#!/usr/bin/env python3
"""Provider-neutral QB starter / health state ingestion interface, as-of lookup and provider acceptance harness (research only).

Maps provider payloads into the Phase1M canonical QB-state schema, stores them append-only (raw layer + normalized layer) and answers
state_at(team, game, cutoff_time) with only what legitimately existed by the cutoff. NOTHING here connects to a vendor: adapters only transform
payloads handed to them, no credentials are read, no network is used, and fixtures used in tests are labelled FIXTURE_ONLY (never accepted data).
Phase1L stays BLOCKED until real payloads pass the acceptance harness.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import nfl_v2_phase1m_qb_state_sources as QS
from nfl_v2_phase2a_ledger import append_record, read_ledger, sha_bytes, canon

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
STORE = ART / 'phase2a_qb_state'
RAW_FILE = 'raw_payloads.jsonl'
NORMALIZED_FILE = 'qb_state_normalized.jsonl'
ACCEPTANCE_FILE = 'phase2a_qb_provider_acceptance.json'
UTC = timezone.utc
MAX_CAPTURE_LAG = timedelta(hours=6)
CLASSES = ('ACCEPTED_FOR_PHASE1L', 'ACCEPTED_FORWARD_ONLY', 'REJECTED')
THRESHOLDS = {'certified_starter_state_coverage_min': 0.95, 'identity_join_min': 0.99, 'publication_timestamp_min': 1.0}
ADAPTER_STATUS = 'CONCEPTUAL_FIELD_MAP_NOT_VALIDATED_AGAINST_ANY_REAL_PAYLOAD'


class ProviderNotAuthorized(RuntimeError):
    """Raised for any attempt to act as a provider whose acceptance classification does not authorize it."""


# --------------------------------------------------------------------------- adapters (field maps only; no network, no credentials)
class Adapter:
    provider = None
    product = None
    evidence_type = 'OTHER'
    field_map = {}
    state_map = {}

    def to_records(self, payload, retrieved_at, crosswalk=None):
        """Provider payload -> list of Phase1M canonical records (unfinalized). Pure function of its inputs."""
        crosswalk = crosswalk or {}
        retrieved = QS.parse_ts(retrieved_at)
        fm = self.field_map
        out = []
        for item in payload.get(fm['items'], []):
            if (item.get(fm['position']) or '').upper() != 'QB':
                continue
            published = item.get(fm['published'])
            kickoff = QS.parse_ts(payload[fm['kickoff']])
            raw_state = (item.get(fm['state']) or '').upper()
            state = self.state_map.get(raw_state, 'UNCERTAIN_STARTER')
            provider_pid = str(item.get(fm['player_id'])) if item.get(fm['player_id']) is not None else None
            gsis = crosswalk.get(provider_pid)
            known_at = QS.parse_ts(published) if published else None
            lag_ok = known_at is not None and (retrieved - known_at) <= MAX_CAPTURE_LAG and retrieved >= known_at
            original = bool(item.get(fm.get('version_history_original', '__none__'))) or lag_ok
            postgame = retrieved >= kickoff
            backfilled = known_at is not None and (retrieved - known_at) > MAX_CAPTURE_LAG and not item.get(fm.get('version_history_original', '__none__'))
            certified = bool(published) and original and not postgame and not backfilled
            elig = []
            for label in ('T24', 'T90'):
                cut = QS.cutoff_time(kickoff, label)
                if (known_at or retrieved) <= cut and retrieved <= cut:
                    elig.append(label)
            eligibility = 'BOTH' if len(elig) == 2 else (elig[0] if elig else 'NEITHER')
            rec = {'season': int(payload[fm['season']]), 'week': int(payload[fm['week']]), 'game_id': payload[fm['game_id']], 'team': payload[fm['team']], 'opponent': payload[fm['opponent']], 'kickoff_time': QS.parse_ts(kickoff).isoformat(),
                   'player_id': gsis, 'player_name': item.get(fm['player_name']) or '', 'provider_player_id': provider_pid, 'normalized_player_id': gsis, 'state_type': state, 'starter_rank': item.get(fm.get('starter_rank', '__none__')),
                   'depth_rank': item.get(fm.get('depth_rank', '__none__')), 'expected_start_probability': item.get(fm.get('start_probability', '__none__')), 'health_state': item.get(fm.get('health', '__none__')),
                   'practice_state': item.get(fm.get('practice', '__none__')), 'injury_body_part': item.get(fm.get('body_part', '__none__')), 'injury_status': item.get(fm.get('injury_status', '__none__')), 'roster_status': item.get(fm.get('roster_status', '__none__')),
                   'source_provider': self.provider, 'source_product': self.product, 'source_record_id': str(item.get(fm['record_id'])), 'source_timestamp': published, 'publication_timestamp': published, 'retrieval_timestamp': QS.parse_ts(retrieved).isoformat(),
                   'effective_timestamp': item.get(fm.get('effective', '__none__')), 'valid_from': (QS.parse_ts(published) if published else retrieved).isoformat(), 'valid_to': None, 'revision_id': str(item.get(fm.get('revision_id', '__none__')) or sha_bytes(canon(item).encode())[:16]),
                   'revision_sequence': None, 'evidence_type': self.evidence_type, 'cutoff_eligibility': eligibility, 'as_of_certified': certified, 'confidence': None, 'stale_flag': False, 'postgame_only_flag': postgame, 'backfilled_flag': backfilled,
                   'original_vintage_flag': original, 'raw_source_hash': sha_bytes(canon(payload).encode())}
            out.append(rec)
        return out


class SportradarAdapter(Adapter):
    provider, product, evidence_type = 'sportradar', 'depth_and_injuries', 'PROVIDER_DEPTH'
    field_map = {'items': 'players', 'position': 'position', 'published': 'updated', 'kickoff': 'scheduled', 'state': 'depth_status', 'player_id': 'id', 'player_name': 'name', 'record_id': 'record_id', 'season': 'season', 'week': 'week', 'game_id': 'game_id',
                 'team': 'team', 'opponent': 'opponent', 'health': 'injury_status', 'injury_status': 'injury_status', 'depth_rank': 'depth', 'revision_id': 'revision', 'version_history_original': 'original_version'}
    state_map = {'STARTER': 'EXPECTED_STARTER', 'CONFIRMED_STARTER': 'VERIFIED_STARTER', 'BACKUP': 'BACKUP', 'OUT': 'OUT', 'DOUBTFUL': 'DOUBTFUL', 'QUESTIONABLE': 'QUESTIONABLE'}


class SportsDataIOAdapter(Adapter):
    provider, product, evidence_type = 'sportsdataio', 'depth_and_injuries', 'PROVIDER_EXPECTED_STARTER'
    field_map = {'items': 'Players', 'position': 'Position', 'published': 'Updated', 'kickoff': 'DateTimeUTC', 'state': 'DepthStatus', 'player_id': 'PlayerID', 'player_name': 'Name', 'record_id': 'RecordID', 'season': 'Season', 'week': 'Week', 'game_id': 'GameKey',
                 'team': 'Team', 'opponent': 'Opponent', 'health': 'InjuryStatus', 'injury_status': 'InjuryStatus', 'depth_rank': 'DepthOrder', 'revision_id': 'Revision', 'version_history_original': 'OriginalVersion'}
    state_map = {'STARTER': 'EXPECTED_STARTER', 'BACKUP': 'BACKUP', 'OUT': 'OUT', 'DOUBTFUL': 'DOUBTFUL', 'QUESTIONABLE': 'QUESTIONABLE'}


ADAPTERS = {'sportradar': SportradarAdapter, 'sportsdataio': SportsDataIOAdapter}


# --------------------------------------------------------------------------- acceptance state
def acceptance_document(path=None):
    path = Path(path or ART / ACCEPTANCE_FILE)
    return json.loads(path.read_text()) if path.exists() else {'providers': {}}


def provider_classification(provider, path=None):
    return acceptance_document(path).get('providers', {}).get(provider, {}).get('classification', 'NOT_EVALUATED_NO_PROVIDER_PAYLOAD_TESTED')


def authorized(provider, path=None):
    return provider_classification(provider, path) in ('ACCEPTED_FOR_PHASE1L', 'ACCEPTED_FORWARD_ONLY')


# --------------------------------------------------------------------------- store (append-only raw + normalized layers)
def load_archive(store=None):
    """Rebuild the in-memory QBStateArchive from the append-only normalized layer."""
    arch = QS.QBStateArchive()
    for line in read_ledger(Path(store or STORE) / NORMALIZED_FILE):
        rec = {k: v for k, v in line['record'].items() if k != 'normalized_record_hash'}
        arch.append(rec)
    return arch


def ingest_qb_state(provider_payload, retrieved_at, provider, crosswalk=None, store=None, require_authorization=True, acceptance_path=None):
    """Canonical entry point: raw layer + normalized layer, append-only. Refuses providers that are not authorized by an acceptance result.

    The raw layer keeps (provider, retrieved_at, payload_hash, payload); the normalized layer keeps Phase1M canonical rows with revision handling.
    """
    if provider not in ADAPTERS:
        raise ValueError('unknown provider adapter: ' + provider)
    if require_authorization and not authorized(provider, acceptance_path):
        raise ProviderNotAuthorized(f'{provider}: no accepted payload; classification {provider_classification(provider, acceptance_path)}')
    store = Path(store or STORE)
    raw_hash = sha_bytes(canon(provider_payload).encode())
    append_record(store / RAW_FILE, {'raw_id': f'{provider}|{raw_hash}|{retrieved_at}', 'provider': provider, 'retrieved_at': retrieved_at, 'payload_sha256': raw_hash, 'payload': provider_payload}, 'raw_id')
    records = ADAPTERS[provider]().to_records(provider_payload, retrieved_at, crosswalk)
    arch = load_archive(store)
    out = []
    for rec in records:
        key = (rec['source_provider'], rec['source_record_id'])
        hist = arch.history(*key)
        if hist and _content(rec) == _content(hist[-1]):
            continue                                                       # same content re-seen (only the retrieval time differs): not a revision
        row = arch.append(rec)
        append_record(store / NORMALIZED_FILE, row, 'normalized_record_hash')
        out.append(row)
    return out


def _content(rec):
    skip = {'retrieval_timestamp', 'raw_source_hash', 'revision_sequence', 'normalized_record_hash', 'cutoff_eligibility', 'as_of_certified', 'postgame_only_flag', 'backfilled_flag', 'original_vintage_flag', 'valid_from'}
    return {k: v for k, v in rec.items() if k not in skip}


def state_at(team, game, cutoff_time, label=None, certified_only=True, archive=None, store=None):
    """What the QB state legitimately was at cutoff_time (visibility rule of Phase1M): rows, plus the conservative reading."""
    archive = archive if archive is not None else load_archive(store)
    game_id = game['game_id'] if isinstance(game, dict) else game
    rows = archive.state_at(team, game_id, cutoff_time, label, certified_only)
    return {'rows': rows, 'summary': QS.expected_starter_summary(rows)}


def certify_for_forecast(team, game_id, cutoff, label, archive=None, store=None, acceptance_path=None):
    """The forecast engine's QB gate. CERTIFIED only with a visible, as-of-certified single starter from an accepted provider."""
    providers = [p for p in ADAPTERS if authorized(p, acceptance_path)]
    if not providers:
        return {'status': 'BLOCKED', 'reason': 'NO_ACCEPTED_QB_STATE_SOURCE'}
    st = state_at(team, game_id, cutoff, label, True, archive, store)
    rows = [r for r in st['rows'] if r['source_provider'] in providers]
    if not rows:
        return {'status': 'UNCERTAIN', 'reason': 'NO_CERTIFIED_VISIBLE_STATE_AT_CUTOFF'}
    summary = QS.expected_starter_summary(rows)
    if summary['state'] == 'EXPECTED_STARTER':
        return {'status': 'CERTIFIED', 'reason': 'SINGLE_CERTIFIED_VISIBLE_STARTER', 'players': summary['players']}
    return {'status': 'UNCERTAIN', 'reason': summary['reason']}


# --------------------------------------------------------------------------- provider acceptance harness
def evaluate_provider(archive, team_games, provider, live_probe=None, identity_sample=None):
    """Acceptance checks for one provider's normalized corpus; returns the evidence and one classification of CLASSES.

    team_games: [{'game_id','team','kickoff_time'}] for the historical 2024 population. Both cutoffs are judged separately.
    live_probe: {'game_id','team','kickoff_time','cutoff_label','retrieved_at'} for the current live game (may be None).
    """
    rows = [r for r in archive.rows() if r['source_provider'] == provider]
    ev = {'provider': provider, 'rows': len(rows), 'thresholds': THRESHOLDS}
    if not rows:
        ev.update(classification='REJECTED', reason='NO_PROVIDER_PAYLOAD_TESTED', checks={}, cutoffs_passing=[])
        return ev
    checks = {}
    pub = sum(1 for r in rows if r['publication_timestamp']) / len(rows)
    checks['original_publication_timestamp_share'] = pub
    checks['original_vintage_share'] = sum(1 for r in rows if r['original_vintage_flag']) / len(rows)
    starters = [r for r in rows if r['state_type'] in ('VERIFIED_STARTER', 'EXPECTED_STARTER', 'PROBABLE_STARTER')]
    checks['identity_join_share'] = (sum(1 for r in starters if r['normalized_player_id']) / len(starters)) if starters else 0.0
    checks['health_status_share'] = sum(1 for r in rows if r['health_state'] or r['injury_status']) / len(rows)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r['source_record_id'], []).append(r['revision_sequence'])
    checks['revisions_reconstructable'] = any(len(v) >= 2 for v in by_rec.values())
    passing, hist = [], {}
    for label in ('T24', 'T90'):
        covered = 0
        for tg in team_games:
            cut = QS.cutoff_time(tg['kickoff_time'], label)
            st = archive.state_at(tg['team'], tg['game_id'], cut, label, True)
            st = [r for r in st if r['source_provider'] == provider]
            if st and QS.expected_starter_summary(st)['state'] == 'EXPECTED_STARTER':
                covered += 1
        share = covered / len(team_games) if team_games else 0.0
        hist[label] = share
        checks[f'certified_starter_state_coverage_{label}'] = share
        if share >= THRESHOLDS['certified_starter_state_coverage_min']:
            passing.append(label)
    live_ok = False
    if live_probe:
        cut = QS.cutoff_time(live_probe['kickoff_time'], live_probe['cutoff_label'])
        st = [r for r in archive.state_at(live_probe['team'], live_probe['game_id'], cut, live_probe['cutoff_label'], True) if r['source_provider'] == provider]
        live_ok = bool(st) and QS.parse_ts(live_probe['retrieved_at']) <= cut
    checks['live_game_captured_before_cutoff'] = live_ok
    historical_ok = (len(passing) == 2 and pub >= THRESHOLDS['publication_timestamp_min'] and checks['original_vintage_share'] == 1.0 and checks['identity_join_share'] >= THRESHOLDS['identity_join_min'] and checks['revisions_reconstructable'])
    if historical_ok and live_ok:
        cls, reason = 'ACCEPTED_FOR_PHASE1L', 'ALL_HISTORICAL_THRESHOLDS_MET_BOTH_CUTOFFS_AND_LIVE_PATH_WORKS'
    elif live_ok and pub >= THRESHOLDS['publication_timestamp_min'] and checks['identity_join_share'] >= THRESHOLDS['identity_join_min']:
        cls, reason = 'ACCEPTED_FORWARD_ONLY', 'LIVE_PATH_WORKS_HISTORICAL_ORIGINAL_VINTAGE_NOT_PROVEN'
    else:
        cls, reason = 'REJECTED', 'FAILED_ACCEPTANCE_CHECKS'
    ev.update(classification=cls, reason=reason, checks=checks, cutoffs_passing=passing, historical_coverage=hist)
    return ev


def evaluate_from_store(provider, team_games, live_probe=None, store=None):
    return evaluate_provider(load_archive(store), team_games, provider, live_probe)


def interface_document():
    return {'schema': 'nfl-v2-phase2a-qb-state-interface-v1', 'status': 'INTERFACE_BUILT_NO_PROVIDER_CONNECTED', 'canonical_schema': 'phase1m_qb_state_schema.json (nfl-v2-phase1m-qb-pregame-state-schema-v1)',
            'functions': {'ingest_qb_state': 'ingest_qb_state(provider_payload, retrieved_at, provider, crosswalk=None) -> appends the raw payload and the normalized Phase1M rows; refuses unauthorized providers',
                          'state_at': 'state_at(team, game, cutoff_time, label=None, certified_only=True) -> visible rows and the conservative starter reading (Phase1M visibility rule; later revisions never leak backward)',
                          'certify_for_forecast': 'certify_for_forecast(team, game_id, cutoff, label) -> CERTIFIED | UNCERTAIN | BLOCKED used by the forward engine; only CERTIFIED may produce an official QB projection',
                          'evaluate_provider': 'acceptance harness: historical 2024 record, original timestamp, revisions, expected starter, health status, T24 and T90 reconstruction, player identity join, live game -> ACCEPTED_FOR_PHASE1L | ACCEPTED_FORWARD_ONLY | REJECTED'},
            'adapters': {k: {'provider': v.provider, 'product': v.product, 'status': ADAPTER_STATUS, 'field_map': v.field_map, 'note': 'documented-shape mapping for payloads handed in by an authorized operator; never connects, never reads credentials'} for k, v in sorted(ADAPTERS.items())},
            'storage': {'raw_layer': 'phase2a_qb_state/raw_payloads.jsonl (provider, retrieved_at, payload_sha256, payload) hash-chained append-only', 'normalized_layer': 'phase2a_qb_state/qb_state_normalized.jsonl (Phase1M canonical rows) hash-chained append-only', 'population': 'EMPTY: no provider is authorized'},
            'capture_design': {'cutoffs': ['T24', 'T90'], 'rule': 'once a provider is authorized (acceptance classification ACCEPTED_*), a capture job calls ingest_qb_state inside each cutoff window with no architecture change; revisions become new rows', 'nothing_runs_against_an_unauthorized_vendor': True},
            'forecast_gate': {'states': ['CERTIFIED', 'UNCERTAIN', 'BLOCKED'], 'current_state': 'BLOCKED', 'reason': 'NO_ACCEPTED_QB_STATE_SOURCE', 'official_qb_forecasts_allowed': False, 'abstention': 'ABSTAIN_QB_STATE_UNCERTIFIED is logged and counted per team-game and cutoff'},
            'phase1l': 'unchanged: BLOCKED_STARTER_STATE_DATA until real payloads pass the harness'}


# --------------------------------------------------------------------------- fixture self-test of the harness (FIXTURE ONLY; not provider evidence)
def _fixture_corpus(n_games, original=True, identity=True, published=True, live=True, revisions=True):
    arch = QS.QBStateArchive()
    games = []
    for i in range(n_games):
        ko = datetime(2024, 9, 8, 17, tzinfo=UTC) + timedelta(days=7 * i)
        gid = f'FIXTURE_{i:02d}_AAA_BBB'
        games.append({'game_id': gid, 'team': 'AAA', 'kickoff_time': ko.isoformat()})
        for hrs in ((40, 100) if revisions else (40,)):
            pub = ko - timedelta(hours=hrs)
            arch.append({'season': 2024, 'week': i + 1, 'game_id': gid, 'team': 'AAA', 'opponent': 'BBB', 'kickoff_time': ko.isoformat(), 'player_id': 'g' if identity else None, 'player_name': 'Fixture QB', 'provider_player_id': 'x',
                         'normalized_player_id': 'g' if identity else None, 'state_type': 'EXPECTED_STARTER', 'source_provider': 'fixture', 'source_product': 'fixture', 'source_record_id': gid, 'publication_timestamp': pub.isoformat() if published else None,
                         'retrieval_timestamp': (pub + timedelta(minutes=5)).isoformat(), 'valid_from': pub.isoformat(), 'revision_id': f'r{hrs}', 'revision_sequence': None, 'evidence_type': 'PROVIDER_DEPTH', 'cutoff_eligibility': 'BOTH',
                         'as_of_certified': published and original, 'stale_flag': False, 'postgame_only_flag': False, 'backfilled_flag': False, 'original_vintage_flag': original, 'raw_source_hash': 'fixture', 'health_state': 'healthy'})
    probe = {'game_id': games[0]['game_id'], 'team': 'AAA', 'kickoff_time': games[0]['kickoff_time'], 'cutoff_label': 'T24', 'retrieved_at': (QS.parse_ts(games[0]['kickoff_time']) - timedelta(hours=30)).isoformat()} if live else None
    return arch, games, probe


def fixture_selftest():
    cases = {'complete_history_and_live': dict(), 'no_original_vintage': dict(original=False), 'missing_identity': dict(identity=False), 'live_only_sparse_history': None}
    out = {}
    for name, kw in cases.items():
        if kw is None:
            arch, games, probe = _fixture_corpus(20)
            games = games + [{'game_id': f'FIXTURE_X{i}', 'team': 'AAA', 'kickoff_time': '2024-12-01T17:00:00+00:00'} for i in range(60)]
        else:
            arch, games, probe = _fixture_corpus(20, **kw)
        ev = evaluate_provider(arch, games, 'fixture', probe)
        out[name] = {'classification': ev['classification'], 'reason': ev['reason']}
    out['empty_corpus'] = {k: evaluate_provider(QS.QBStateArchive(), [], 'fixture', None)[k] for k in ('classification', 'reason')}
    return out


def build_acceptance_document():
    st = fixture_selftest()
    return {'schema': 'nfl-v2-phase2a-qb-provider-acceptance-v1', 'status': 'HARNESS_BUILT_NO_PROVIDER_EVALUATED', 'phase1l': 'UNCHANGED_BLOCKED_STARTER_STATE_DATA',
            'rule': 'A provider is evaluated only on a real payload from an authorized trial / sample / contract. Fixtures prove the harness logic only and are never provider evidence. Phase1L does not change until a provider returns ACCEPTED_FOR_PHASE1L on real payloads.',
            'classes': list(CLASSES), 'thresholds': THRESHOLDS,
            'checks': ['historical 2024 record (all 544 regular-season team-games)', 'original publication timestamp', 'revisions reconstructable', 'expected starter state', 'health / injury status', 'T24 reconstruction', 'T90 reconstruction', 'player identity join to GSIS', 'current live game captured before its cutoff'],
            'class_rules': {'ACCEPTED_FOR_PHASE1L': 'publication timestamp share 1.0, every row original vintage, identity join >= 0.99, revisions reconstructable, certified starter-state coverage >= 0.95 at BOTH T24 and T90, and the live game captured before its cutoff',
                            'ACCEPTED_FORWARD_ONLY': 'the live path works with timestamps and identity but the historical original-vintage requirements are not met; forward capture only, never validates Phase1L on 2024', 'REJECTED': 'otherwise (or no payload tested)'},
            'providers': {p: {'classification': 'NOT_EVALUATED_NO_PROVIDER_PAYLOAD_TESTED', 'evidence': 'NO_PROVIDER_PAYLOAD_TESTED', 'authorized': False, 'adapter_status': ADAPTER_STATUS} for p in sorted(ADAPTERS)},
            'fixture_selftest': {'label': 'FIXTURE_ONLY_NOT_PROVIDER_EVIDENCE', 'cases': st, 'classifications_produced': sorted({v['classification'] for v in st.values()})}}

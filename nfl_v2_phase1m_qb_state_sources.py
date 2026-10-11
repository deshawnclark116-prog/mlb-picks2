#!/usr/bin/env python3
"""Phase1M-DATA: canonical QB pregame-state schema, as-of archive, provider scorecard and forward-capture plan.

Data engineering / source qualification ONLY. No model is fitted, no outcome is scored, no
network is used here. The provider scorecard is a dated evidence snapshot (2026-10-06), not an
auto-refreshed feed. Nothing in this module may certify a source without a tested payload.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
BASE_HEAD = 'c3c6b025ccc8bb7b07b73bde4cbf84f6c1bb1ecf'
EVIDENCE_DATE_UTC = '2026-10-06'
UTC = timezone.utc

STATE_TYPES = (
    'VERIFIED_STARTER', 'EXPECTED_STARTER', 'PROBABLE_STARTER', 'BACKUP', 'EMERGENCY_QB',
    'MULTI_QB', 'UNCERTAIN_STARTER', 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'LIMITED',
    'FULL_PARTICIPATION', 'RETURNING_FROM_INJURY', 'RECENTLY_BENCHED', 'RECENTLY_PROMOTED',
    'RECENTLY_ACQUIRED')
EVIDENCE_TYPES = (
    'OFFICIAL_DEPTH', 'OFFICIAL_INJURY', 'PROVIDER_DEPTH', 'PROVIDER_EXPECTED_STARTER',
    'TRANSACTION', 'PRACTICE_REPORT', 'COACH_ANNOUNCEMENT', 'OTHER')
CUTOFF_ELIGIBILITY = ('T24', 'T90', 'BOTH', 'NEITHER')
CUTOFFS = ('T24', 'T90')
CLASSIFICATIONS = (
    'ACCEPTED_FOR_PHASE1L', 'ACCEPTED_FORWARD_ONLY', 'ACCEPTED_HISTORICAL_ONLY',
    'PROMISING_NEEDS_SAMPLE', 'PROMISING_NEEDS_CONTRACT', 'BLOCKED_TIMING', 'BLOCKED_LICENSE',
    'BLOCKED_ACCESS', 'BLOCKED_HISTORY', 'BLOCKED_IDENTITY', 'POSTGAME_ONLY', 'NOT_USEFUL')
DECISIONS = (
    'PHASE1L_CAN_REOPEN_HISTORICALLY', 'PHASE1L_CAN_REOPEN_FORWARD_ONLY',
    'PHASE1L_REMAINS_BLOCKED_STARTER_STATE_DATA', 'PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS')

# (name, json type, required, description)
SCHEMA_FIELDS = (
    ('season', 'integer', True, 'NFL season'),
    ('week', 'integer', True, 'regular-season week'),
    ('game_id', 'string', True, 'nflverse game id'),
    ('team', 'string', True, 'nflverse team abbreviation'),
    ('opponent', 'string', True, 'opponent abbreviation'),
    ('kickoff_time', 'string', True, 'UTC ISO-8601 kickoff'),
    ('player_id', 'string|null', False, 'GSIS id when known'),
    ('player_name', 'string', True, 'as published by the source'),
    ('provider_player_id', 'string|null', False, 'provider native id'),
    ('normalized_player_id', 'string|null', False, 'crosswalked nflverse GSIS id; never name-only'),
    ('state_type', 'enum', True, 'one of STATE_TYPES'),
    ('starter_rank', 'integer|null', False, 'provider starter rank if supplied'),
    ('depth_rank', 'integer|null', False, 'provider depth rank if supplied'),
    ('expected_start_probability', 'number|null', False, 'only if the provider supplies it; never derived from rank'),
    ('health_state', 'string|null', False, 'provider health wording'),
    ('practice_state', 'string|null', False, 'practice participation wording'),
    ('injury_body_part', 'string|null', False, 'body part'),
    ('injury_status', 'string|null', False, 'game designation'),
    ('roster_status', 'string|null', False, 'roster status'),
    ('source_provider', 'string', True, 'provider'),
    ('source_product', 'string', True, 'product/endpoint'),
    ('source_record_id', 'string', True, 'provider record id'),
    ('source_timestamp', 'string|null', False, 'timestamp inside the provider record'),
    ('publication_timestamp', 'string|null', False, 'FIRST public publication of this revision; null when unknown'),
    ('retrieval_timestamp', 'string', True, 'when WE retrieved the bytes'),
    ('effective_timestamp', 'string|null', False, 'when the state became effective'),
    ('valid_from', 'string', True, 'start of validity'),
    ('valid_to', 'string|null', False, 'end of validity (superseded or expired)'),
    ('revision_id', 'string', True, 'provider/our revision id'),
    ('revision_sequence', 'integer', True, 'monotone per (provider, source_record_id)'),
    ('evidence_type', 'enum', True, 'one of EVIDENCE_TYPES'),
    ('cutoff_eligibility', 'enum', True, 'one of CUTOFF_ELIGIBILITY (certification of timing)'),
    ('as_of_certified', 'boolean', True, 'true only when timing is proven by original-vintage evidence'),
    ('confidence', 'number|null', False, 'null unless certified'),
    ('stale_flag', 'boolean', True, 'older than the freshness bound at the cutoff'),
    ('postgame_only_flag', 'boolean', True, 'derived from target-game participation or later'),
    ('backfilled_flag', 'boolean', True, 'loaded after the game/period'),
    ('original_vintage_flag', 'boolean', True, 'bytes captured when first public'),
    ('raw_source_hash', 'string', True, 'sha256 of the raw payload'),
    ('normalized_record_hash', 'string', True, 'sha256 of the canonical record excluding this field'),
)
FIELD_NAMES = tuple(f[0] for f in SCHEMA_FIELDS)
REQUIRED = tuple(f[0] for f in SCHEMA_FIELDS if f[2])


def parse_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace('Z', '+00:00')
    out = datetime.fromisoformat(text)
    return out if out.tzinfo else out.replace(tzinfo=UTC)


def cutoff_time(kickoff, label):
    kickoff = parse_ts(kickoff)
    if label == 'T24':
        return kickoff - timedelta(hours=24)
    if label == 'T90':
        return kickoff - timedelta(minutes=90)
    raise ValueError('cutoff must be T24 or T90')


def canon(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def record_hash(record):
    body = {k: v for k, v in record.items() if k != 'normalized_record_hash'}
    return hashlib.sha256(canon(body).encode()).hexdigest()


def validate(record):
    """Problems (empty = valid). Structure/timing hygiene only; never evidence of truth."""
    problems = []
    for name in REQUIRED:
        if record.get(name) is None:
            problems.append('missing:' + name)
    extra = set(record) - set(FIELD_NAMES)
    if extra:
        problems.append('unknown_fields:' + ','.join(sorted(extra)))
    if record.get('state_type') not in STATE_TYPES:
        problems.append('bad_state_type')
    if record.get('evidence_type') not in EVIDENCE_TYPES:
        problems.append('bad_evidence_type')
    if record.get('cutoff_eligibility') not in CUTOFF_ELIGIBILITY:
        problems.append('bad_cutoff_eligibility')
    if record.get('as_of_certified') and not record.get('publication_timestamp'):
        problems.append('certified_without_publication_timestamp')
    if record.get('as_of_certified') and not record.get('original_vintage_flag'):
        problems.append('certified_without_original_vintage')
    if record.get('as_of_certified') and (record.get('backfilled_flag') or record.get('postgame_only_flag')):
        problems.append('certified_but_backfilled_or_postgame')
    if not record.get('as_of_certified') and record.get('confidence') is not None:
        problems.append('confidence_without_certification')
    if record.get('expected_start_probability') is not None and not (0 <= record['expected_start_probability'] <= 1):
        problems.append('bad_probability')
    for name in ('kickoff_time', 'retrieval_timestamp', 'valid_from'):
        try:
            parse_ts(record.get(name))
        except Exception:                                  # noqa: BLE001
            problems.append('bad_timestamp:' + name)
    return problems


def finalize(record):
    """Return a copy with normalized_record_hash set (hash covers every other field, revision fields included)."""
    out = deepcopy(record)
    for name in FIELD_NAMES:
        out.setdefault(name, None)
    out['normalized_record_hash'] = record_hash(out)
    return out


class ArchiveError(RuntimeError):
    pass


class QBStateArchive:
    """Append-only QB pregame-state archive. Nothing is overwritten or deleted.

    A revision is a new row with a higher revision_sequence for the same (source_provider,
    source_record_id). state_at() replays only what legitimately existed by the cutoff.
    """

    def __init__(self):
        self._rows = []
        self._seq = {}

    def __len__(self):
        return len(self._rows)

    def rows(self):
        return tuple(deepcopy(self._rows))

    def append(self, record):
        record = deepcopy(record)
        key = (record.get('source_provider'), record.get('source_record_id'))
        if record.get('revision_sequence') is None:
            record['revision_sequence'] = self._seq.get(key, 0) + 1
        problems = validate({**{n: None for n in FIELD_NAMES}, **record, 'normalized_record_hash': 'x'})
        problems = [p for p in problems if p != 'missing:normalized_record_hash']
        if problems:
            raise ArchiveError('invalid record: ' + ';'.join(problems))
        if record['revision_sequence'] <= self._seq.get(key, 0):
            raise ArchiveError('revision_sequence must increase; prior revisions are never overwritten')
        out = finalize(record)
        self._seq[key] = out['revision_sequence']
        self._rows.append(out)
        return deepcopy(out)

    def visible(self, row, cutoff, label=None):
        """Single predicate used by state_at; mirrors the preregistered visibility rule."""
        cutoff = parse_ts(cutoff)
        pub = parse_ts(row['publication_timestamp']) if row.get('publication_timestamp') else None
        ret = parse_ts(row['retrieval_timestamp'])
        known_at = pub if pub is not None else ret
        if known_at > cutoff or ret > cutoff:
            return False
        if parse_ts(row['valid_from']) > cutoff:
            return False
        if row.get('valid_to') and cutoff >= parse_ts(row['valid_to']):
            return False
        if row.get('postgame_only_flag'):
            return False
        if row.get('backfilled_flag') and not row.get('original_vintage_flag'):
            return False
        if label is not None and row['cutoff_eligibility'] not in (label, 'BOTH'):
            return False
        return True

    def state_at(self, team, game_id, cutoff, label=None, certified_only=False):
        """Latest visible revision of every source record for (team, game) as of `cutoff`.

        Later revisions published after the cutoff cannot leak backward: they fail visible()
        and the previous visible revision is returned instead.
        """
        best = {}
        for row in self._rows:
            if row['team'] != team or row['game_id'] != game_id:
                continue
            if certified_only and not row['as_of_certified']:
                continue
            if not self.visible(row, cutoff, label):
                continue
            key = (row['source_provider'], row['source_record_id'])
            if key not in best or row['revision_sequence'] > best[key]['revision_sequence']:
                best[key] = row
        return [deepcopy(best[k]) for k in sorted(best)]

    def history(self, provider, source_record_id):
        return [deepcopy(r) for r in self._rows
                if r['source_provider'] == provider and r['source_record_id'] == source_record_id]


def expected_starter_summary(rows):
    """Deterministic, conservative reading of visible rows. Never fabricates a probability."""
    starters = {r['normalized_player_id'] or r['player_name'] for r in rows
                if r['state_type'] in ('VERIFIED_STARTER', 'EXPECTED_STARTER', 'PROBABLE_STARTER')}
    out_players = {r['normalized_player_id'] or r['player_name'] for r in rows
                   if r['state_type'] in ('OUT', 'DOUBTFUL')}
    starters -= out_players
    if not rows:
        return {'state': 'UNCERTAIN_STARTER', 'players': [], 'reason': 'NO_VISIBLE_STATE'}
    if len(starters) == 1:
        return {'state': 'EXPECTED_STARTER', 'players': sorted(starters), 'reason': 'SINGLE_VISIBLE_STARTER'}
    if len(starters) > 1:
        return {'state': 'MULTI_QB', 'players': sorted(starters), 'reason': 'CONFLICTING_VISIBLE_STARTERS'}
    return {'state': 'UNCERTAIN_STARTER', 'players': [], 'reason': 'NO_VISIBLE_STARTER_ROW'}


# --------------------------------------------------------------------------- provider scorecard
SCORECARD_FIELDS = (
    'provider', 'product', 'historical_start_season', '2024_coverage', '2025_coverage', '2026_live',
    'starter_state', 'depth_chart', 'injury_status', 'practice_status', 'active_inactive',
    'revision_history', 'publication_timestamp', 'historical_original_vintage', 'T24_supported',
    'T90_supported', 'player_id_quality', 'nflverse_crosswalk_quality', 'API_available',
    'bulk_download', 'webhook', 'archive_endpoint', 'versioned_records', 'historical_backfill_cost',
    'annual_cost', 'trial_available', 'sample_available', 'sales_contact_required',
    'internal_use_allowed', 'redistribution_allowed', 'storage_allowed', 'derived_model_use_allowed',
    'implementation_effort', 'data_quality_confidence', 'final_classification', 'notes',
    'evidence_links', 'payload_tested')
UNK = 'UNKNOWN'
SALES = 'CONTACT SALES / UNKNOWN'
NPT = 'NO_PROVIDER_PAYLOAD_TESTED'


def _p(**kw):
    row = {f: UNK for f in SCORECARD_FIELDS}
    row['payload_tested'] = NPT
    row.update(kw)
    return row


_SR = 'https://developer.sportradar.com/football/reference/'
PROVIDERS = [
    _p(provider='Sportradar', product='NFL v7 Weekly Depth Charts + Weekly Injuries + Game Roster + Daily Change Log (documented API; not tested)',
       historical_start_season='Depth chart endpoint season parameter documented 2000-2026; weekly injuries 2009-2026 (OpenAPI parameter range, not a coverage claim)',
       **{'2024_coverage': 'ENDPOINT_PARAMETER_DOCUMENTED; NO_PAYLOAD_TESTED', '2025_coverage': 'ENDPOINT_PARAMETER_DOCUMENTED; NO_PAYLOAD_TESTED', '2026_live': 'DOCUMENTED_LIVE_API; NO_PAYLOAD_TESTED'},
       starter_state='PARTIAL: depth rank 1 = starter on a depth chart; no verified-starter announcement field documented',
       depth_chart='YES (weekly report feed, pull on game-week cadence; may update during/after the game)',
       injury_status='YES (Questionable/Doubtful/Out)', practice_status='YES (practice status text)',
       active_inactive='YES (inactives entered about 90 minutes before kickoff; Game Roster is game-day truth)',
       revision_history='PARTIAL: Daily Change Log lists modification timestamps/ids of changed entities; retention of ORIGINAL prior versions not documented',
       publication_timestamp='status_date = last status update (not proven first publication); depth chart carries no per-row publication time in the documented schema',
       historical_original_vintage='NOT_DOCUMENTED: a past-week query returns the current stored state; the docs say depth charts may update during/post game',
       T24_supported='FORWARD: YES by polling at least hourly; HISTORICAL: NOT_PROVEN', T90_supported='FORWARD: inactives at about T-90 with polling lag; HISTORICAL: NOT_PROVEN',
       player_id_quality='Sportradar UUIDs; GSIS mapping needs a crosswalk (not tested)', nflverse_crosswalk_quality='UNKNOWN (needs sample)',
       API_available=True, bulk_download=False, webhook='Push feeds exist for live game events only; none documented for depth/injury changes', archive_endpoint=False, versioned_records='NOT_DOCUMENTED',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available='YES (trial key via Sportradar marketplace; trial terms/limits not read)', sample_available='Trial key (terms unread)',
       sales_contact_required='Likely for production/historical',
       internal_use_allowed=UNK, redistribution_allowed=UNK, storage_allowed=UNK, derived_model_use_allowed=UNK,
       implementation_effort='MEDIUM', data_quality_confidence='MEDIUM', final_classification='PROMISING_NEEDS_SAMPLE',
       notes='Best documented structure for expected depth + injury + practice + inactives and a change log. A documented endpoint is not original-vintage proof: a trial test must show (a) 2024 queries return pre-kickoff states with timestamps, (b) the change log reconstructs revisions, (c) GSIS joins. Until then it cannot certify Phase1L. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=[_SR + 'nfl-weekly-depth-charts', _SR + 'nfl-weekly-injuries', 'https://developer.sportradar.com/football/reference/nfl-overview',
                       'https://developer.sportradar.com/football/docs/nfl-ig-rosters', 'https://developer.sportradar.com/getting-started/docs/get-started']),
    _p(provider='SportsDataIO', product='NFL DepthCharts / Injuries / Player.DepthOrder, InjuryStatus (documented; not tested)',
       historical_start_season='Historical database advertised ("decades"); per-season detail requires contacting SportsDataIO',
       **{'2024_coverage': 'UNKNOWN: requires contacting SportsDataIO', '2025_coverage': 'UNKNOWN', '2026_live': 'DOCUMENTED_LIVE_API; NO_PAYLOAD_TESTED'},
       starter_state='PARTIAL: DepthOrder 1 = Starter; Player.InjuryStatus; no announced-starter field seen',
       depth_chart='YES: "Each individual change is timestamped" (2022 announcement); field Updated is US Eastern',
       injury_status='YES (Status Probable/Questionable/Doubtful/Out; BodyPart; DeclaredInactive; Updated)', practice_status='DEPRECATED in the data dictionary (Practice and PracticeDescription deprecated)',
       active_inactive='PARTIAL: DeclaredInactive flag', revision_history='UNKNOWN: per-change timestamps documented for live depth charts; retained historical versions not documented',
       publication_timestamp='Updated = date/time last updated (not first publication)', historical_original_vintage='NOT_DOCUMENTED',
       T24_supported='FORWARD: plausible by polling; HISTORICAL: NOT_PROVEN', T90_supported='FORWARD: plausible; HISTORICAL: NOT_PROVEN',
       player_id_quality='SportsDataIO PlayerID; GSIS crosswalk not documented', nflverse_crosswalk_quality='UNKNOWN',
       API_available=True, bulk_download='UNKNOWN', webhook='UNKNOWN', archive_endpoint=False, versioned_records='NOT_DOCUMENTED',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available='NO NFL TRIAL SEEN: the API documentation states the free trial only provides access to UEFA Champions League', sample_available='NO (not for NFL)',
       sales_contact_required='YES (historical content: "Get in touch")', internal_use_allowed=UNK, redistribution_allowed=UNK, storage_allowed=UNK, derived_model_use_allowed=UNK,
       implementation_effort='MEDIUM', data_quality_confidence='MEDIUM', final_classification='PROMISING_NEEDS_CONTRACT',
       notes='Per-change timestamps on depth charts are the right shape for forward capture, but no NFL sample exists and historical original-version retention is unanswered. Ask in writing for (a) archived per-change depth/injury rows for 2024 with first-publication times, (b) license for derived model use. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://sportsdata.io/developers/data-dictionary/nfl', 'https://sportsdata.io/sportsdataio-nfl-depth-chart-api-update', 'https://sportsdata.io/nfl-api',
                       'https://sportsdata.io/developers/api-documentation/nfl']),
    _p(provider='Stats Perform / Opta', product='NFL player data feeds (betting/live player stats)',
       historical_start_season='Archive advertised (35 years across 15 sports); NFL pregame state fields not documented in found material',
       starter_state='NOT_DOCUMENTED', depth_chart='NOT_DOCUMENTED', injury_status='NOT_DOCUMENTED', practice_status='NOT_DOCUMENTED', active_inactive='NOT_DOCUMENTED',
       revision_history='UNKNOWN', publication_timestamp='UNKNOWN', historical_original_vintage='UNKNOWN', T24_supported='UNKNOWN', T90_supported='UNKNOWN',
       API_available='PRIVATE_SALES_ONLY', bulk_download='UNKNOWN', webhook='UNKNOWN', archive_endpoint='UNKNOWN', versioned_records='UNKNOWN',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available=UNK, sample_available=UNK, sales_contact_required=True,
       implementation_effort='HIGH', data_quality_confidence='LOW', final_classification='BLOCKED_ACCESS',
       notes='Found documentation covers live player stats and betting products, not an NFL expected-starter / depth / injury version feed. Not rejected as a company: no documentation exists in public evidence to test. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://www.statsperform.com/products/live-player-stats/', 'https://www.statsperform.com/products/opta-for-betting/']),
    _p(provider='Genius Sports', product='GeniusPremium (NFL official data distribution)',
       historical_start_season='UNKNOWN', starter_state='NOT_DOCUMENTED in found material (play-by-play, NGS and betting feeds are documented)', depth_chart='NOT_DOCUMENTED', injury_status='NOT_DOCUMENTED', practice_status='NOT_DOCUMENTED',
       active_inactive='NOT_DOCUMENTED', revision_history='UNKNOWN', publication_timestamp='UNKNOWN', historical_original_vintage='UNKNOWN', T24_supported='UNKNOWN', T90_supported='UNKNOWN',
       API_available='PRIVATE (media/sportsbook operator contracts)', bulk_download='UNKNOWN', webhook='LOW_LATENCY_FEED (play-by-play)', archive_endpoint='UNKNOWN', versioned_records='UNKNOWN',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available=UNK, sample_available=UNK, sales_contact_required=True,
       implementation_effort='HIGH', data_quality_confidence='LOW', final_classification='BLOCKED_ACCESS',
       notes='Genius is the NFL official-data distributor to media and betting operators; found documentation is play/tracking oriented. Betting-operator terms may conflict with this project\'s no-sportsbook rule; no pregame QB-state product was evidenced. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://nfl.com/_amp/nfl-extends-strategic-partnership-with-genius-sports', 'https://www.businesswire.com/news/home/20210930005582/en/Genius-Sports-Announces-Expansion-of-Its-Agreement-With-Entain-and-BetMGM-With-Official-NFL-Data-and-Fan-Engagement-Solutions']),
    _p(provider='Sports Info Solutions (SIS)', product='SIS Data Hub NFL injury database + football data products',
       historical_start_season='NFL/CFB injury data from 2016 per SIS description', **{'2024_coverage': 'ADVERTISED_NOT_TESTED', '2025_coverage': 'ADVERTISED_NOT_TESTED', '2026_live': 'UNKNOWN for NFL pregame state'},
       starter_state='NO (not documented)', depth_chart='NOT_DOCUMENTED', injury_status='YES in-game and off-field injuries with diagnoses, prognoses, return dates (video-reviewed)', practice_status='NOT_DOCUMENTED',
       active_inactive='NOT_DOCUMENTED', revision_history='UNKNOWN', publication_timestamp='UNKNOWN', historical_original_vintage='UNKNOWN', T24_supported='UNKNOWN', T90_supported='UNKNOWN',
       player_id_quality='UNKNOWN', nflverse_crosswalk_quality='UNKNOWN', API_available='CUSTOM FEEDS (sales)', bulk_download='UNKNOWN', webhook='UNKNOWN', archive_endpoint='UNKNOWN', versioned_records='UNKNOWN',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available=UNK, sample_available='Request needed', sales_contact_required=True,
       implementation_effort='HIGH', data_quality_confidence='MEDIUM', final_classification='PROMISING_NEEDS_SAMPLE',
       notes='Injury severity/return-date history may later help a health layer, but it does not identify the expected starter and publication times are unknown. Sample needed: injury rows with first-record timestamps for 2024 QBs. Does not by itself reopen Phase1L. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://www.sportsinfosolutions.com/football/', 'https://www.sportsinfosolutions.com/solution/football']),
    _p(provider='TruMedia', product='TruMedia football API (TeamTotals/TeamGames/PlayerGames/PlayerPlays tables)',
       historical_start_season='Queries documented back to 2004', starter_state='NO', depth_chart='NO (not in API documentation)', injury_status='NO (not in API documentation)', practice_status='NO',
       active_inactive='NO', revision_history='NOT_DOCUMENTED', publication_timestamp='NOT_DOCUMENTED', historical_original_vintage='NOT_APPLICABLE', T24_supported='NO', T90_supported='NO',
       API_available=True, bulk_download='CSV/JSON query output', webhook=False, archive_endpoint=False, versioned_records='NOT_DOCUMENTED',
       historical_backfill_cost=SALES, annual_cost=SALES, trial_available=UNK, sample_available=UNK, sales_contact_required=True,
       implementation_effort='MEDIUM', data_quality_confidence='MEDIUM', final_classification='NOT_USEFUL',
       notes='Documented API is completed-game statistics. It contains no pregame depth/injury/roster-state endpoint. Useful only as postgame history, which is what Phase1L already has. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://football.help.trumedianetworks.com/football/api-documentation']),
    _p(provider='FTN Data', product='FTN full API (charting/participation) and public FTN charting via nflverse',
       historical_start_season='Charting/all-22 participation 2019+, expanded participation 2021+ (FTN FAQ per Phase1J audit)', starter_state='NO', depth_chart='NO', injury_status='NO', practice_status='NO', active_inactive='NO',
       revision_history='Status flags is_charted / needs_reimport / last_updated only (Phase1J OpenAPI audit)', publication_timestamp='NO first-publication history', historical_original_vintage='NO', T24_supported='NO', T90_supported='NO',
       API_available=True, bulk_download='UNKNOWN', webhook=False, archive_endpoint=False, versioned_records=False,
       historical_backfill_cost='Phase1J FAQ note: commercial from $5,000/year, private $3,000/year (FAQ page now returns HTTP 403; not re-verified)', annual_cost='Phase1J FAQ note: from $3,000-$5,000/year; package/agreement dependent',
       trial_available=UNK, sample_available=UNK, sales_contact_required='Agreement', implementation_effort='MEDIUM', data_quality_confidence='MEDIUM', final_classification='POSTGAME_ONLY',
       notes='Postgame charting; routes/target quality, not a pregame QB starter/health state. FTN pages returned HTTP 403 this phase, so the Phase1J audit evidence is cited rather than refreshed. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://ftnfantasy.com/data', 'https://charting.ftntools.com/api/openapi.json', 'nfl_models/nfl_player_outcome_v2/phase1j_information_gap_audit.json']),
    _p(provider='PFF', product='PFF Pro API/CLI (grades, advanced stats, charted details)',
       historical_start_season='NOT_STATED', starter_state='NOT_DOCUMENTED', depth_chart='NOT_DOCUMENTED', injury_status='NOT_DOCUMENTED', practice_status='NOT_DOCUMENTED', active_inactive='NOT_DOCUMENTED',
       revision_history='UNKNOWN', publication_timestamp='UNKNOWN', historical_original_vintage='UNKNOWN', T24_supported='UNKNOWN', T90_supported='UNKNOWN',
       API_available='YES per PFF Pro page', bulk_download='Spreadsheet export with scheduled refresh', webhook=UNK, archive_endpoint=UNK, versioned_records=UNK,
       historical_backfill_cost='NOT_STATED', annual_cost='NOT_STATED (PFF+ consumer price shown; API price not stated)', trial_available=UNK, sample_available=UNK, sales_contact_required=UNK,
       implementation_effort='MEDIUM', data_quality_confidence='LOW', final_classification='BLOCKED_ACCESS',
       notes='PFF Pro advertises programmatic access to grades/charted data. Nothing documents depth chart, expected starter or injury state versions. Not a pregame QB-state source on available evidence. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://www.pff.com/news/introducing-the-next-generation-of-pff-products']),
    _p(provider='NFL official injury report (NFL.com)', product='Public weekly injury report pages (practice participation by day and game status)',
       historical_start_season='Pages exist by week (for example /injuries/league/2024/reg5)', **{'2024_coverage': 'PAGES_EXIST; NO_VERSIONED_ARCHIVE', '2025_coverage': 'PAGES_EXIST', '2026_live': 'PUBLIC_PAGE_CURRENT_STATE_ONLY'},
       starter_state='NO (injury designations only)', depth_chart='NO', injury_status='YES', practice_status='YES (Full/Limited/Did Not Participate)', active_inactive='NO (inactives are a separate game-day list)',
       revision_history='NO (mutable page)', publication_timestamp='NO visible last-updated timestamp on the page', historical_original_vintage='NO', T24_supported='FORWARD only if an authorized capture is permitted', T90_supported='NO documented game-day API',
       player_id_quality='Names only on the page; no GSIS', nflverse_crosswalk_quality='NAME_ONLY (unacceptable without an id)', API_available=False, bulk_download=False, webhook=False, archive_endpoint=False, versioned_records=False,
       historical_backfill_cost='NOT_AVAILABLE', annual_cost='NOT_AVAILABLE', trial_available=False, sample_available='Public page (current state only)', sales_contact_required='Rights holder terms unresolved',
       internal_use_allowed='UNRESOLVED (no systematic-collection license found; Phase1J: BLOCKED_LICENSE)', redistribution_allowed=UNK, storage_allowed=UNK, derived_model_use_allowed=UNK,
       implementation_effort='HIGH', data_quality_confidence='HIGH for content, LOW for versioning', final_classification='BLOCKED_LICENSE',
       notes='Authoritative source but a mutable page with no visible timestamp, no API and no authorization for systematic collection. Not scraped in this phase. NO_PROVIDER_PAYLOAD_TESTED.',
       evidence_links=['https://www.nfl.com/injuries/']),
    _p(provider='nflverse', product='depth_charts_{2001..2024} (history)', historical_start_season='2001 per nfldata.org coverage; no per-record load time before 2025',
       **{'2024_coverage': 'YES rows (1,539 QB rows) but ZERO dt timestamps (Phase1L audit)', '2025_coverage': 'see depth_charts_2025+ record', '2026_live': 'NOT_APPLICABLE'},
       starter_state='PARTIAL (depth rank only)', depth_chart='YES weekly listing', injury_status='NO', practice_status='NO', active_inactive='NO', revision_history='NO', publication_timestamp='NO (no dt for 2024)',
       historical_original_vintage='NO', T24_supported='NO', T90_supported='NO', player_id_quality='GSIS where present', nflverse_crosswalk_quality='HIGH', API_available='GitHub release assets', bulk_download=True, webhook=False,
       archive_endpoint=False, versioned_records=False, historical_backfill_cost='FREE', annual_cost='FREE', trial_available='N/A', sample_available=True, sales_contact_required=False,
       internal_use_allowed='CC-BY-4.0 root license; upstream rights unresolved', redistribution_allowed=UNK, storage_allowed=True, derived_model_use_allowed=UNK,
       implementation_effort='LOW', data_quality_confidence='LOW for timing', final_classification='BLOCKED_TIMING',
       notes='Cannot reconstruct what a depth chart said before a 2024 kickoff. Phase1L audited payload counts (not re-read here).', payload_tested='REPOSITORY_AUDIT_ONLY (Phase1H/J/L frozen audits; no new payload opened)',
       evidence_links=['https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html', 'https://nfldata.org/data_coverage.php', 'nfl_models/nfl_player_outcome_v2/phase1l_qb_source_audit.json']),
    _p(provider='nflverse', product='depth_charts_2025+ (ESPN-derived daily loads with dt)', historical_start_season='2025 (221 retained load timestamps Aug 3 2025 - Mar 14 2026 per Phase1J)',
       **{'2024_coverage': 'NO', '2025_coverage': 'YES with dt = when the record was LOADED; timing coverage measured in this phase', '2026_live': 'RELEASE_METADATA_ONLY; no payload opened (Week 5+ policy)'},
       starter_state='PARTIAL (rank-1 depth slot is not an announced starter)', depth_chart='YES', injury_status='NO', practice_status='NO', active_inactive='NO', revision_history='Daily loads create successive snapshots; retention of every load not proven',
       publication_timestamp='dt = load time, not publication', historical_original_vintage='NO for 2024; 2025 loads may be original-vintage but unproven', T24_supported='FORWARD: yes if daily loads continue and are hash-captured', T90_supported='FORWARD: partial (daily cadence only)',
       player_id_quality='ESPN id plus GSIS (21,342 of 21,429 QB rows mapped per Phase1J)', nflverse_crosswalk_quality='HIGH for QBs', API_available='GitHub release assets', bulk_download=True, webhook=False, archive_endpoint=False, versioned_records='NO (asset overwritten)',
       historical_backfill_cost='FREE', annual_cost='FREE', trial_available='N/A', sample_available=True, sales_contact_required=False, internal_use_allowed='UNRESOLVED upstream (ESPN-derived)', redistribution_allowed=UNK, storage_allowed=True, derived_model_use_allowed=UNK,
       implementation_effort='LOW', data_quality_confidence='MEDIUM', final_classification='PROMISING_NEEDS_CONTRACT',
       notes='The repository V1 shadow pipeline already hash-captures depth_charts_2026.csv at cutoffs. The only open item is upstream reuse rights for ESPN-derived rows. Forward-only; cannot validate 2024. See acceptance_results for the 2025 timing probe.',
       payload_tested='REPOSITORY_TIMING_PROBE_2025 (timing counts only; see acceptance results)', evidence_links=['https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html', 'nfl_models/nfl_player_outcome_phase1e/live_source_registry.json']),
    _p(provider='nflverse', product='injuries_{season} (official NFL injury/practice data)', historical_start_season='2009 per nfldata.org coverage',
       **{'2024_coverage': 'YES 6,215 rows; modification timestamps on all rows; ZERO original publication rows (Phase1L)', '2025_coverage': 'YES 6,068 rows; ZERO modification timestamps (Phase1L)', '2026_live': 'FORWARD hashed capture exists in V1 shadow; no payload opened here'},
       starter_state='NO', depth_chart='NO', injury_status='YES report_status/report_primary_injury', practice_status='YES practice_status/practice_primary_injury', active_inactive='NO',
       revision_history='NO: date_modified only', publication_timestamp='date_modified = date and time injury information was updated (not first publication)', historical_original_vintage='NO', T24_supported='NO historically; FORWARD yes with hashed captures', T90_supported='NO historically',
       player_id_quality='GSIS', nflverse_crosswalk_quality='HIGH', API_available='GitHub release assets', bulk_download=True, webhook=False, archive_endpoint=False, versioned_records=False,
       historical_backfill_cost='FREE', annual_cost='FREE', trial_available='N/A', sample_available=True, sales_contact_required=False, internal_use_allowed='CC-BY-4.0 root; official-report upstream rights unresolved (Phase1J: official reports BLOCKED_LICENSE)', redistribution_allowed=UNK, storage_allowed=True, derived_model_use_allowed=UNK,
       implementation_effort='LOW', data_quality_confidence='MEDIUM', final_classification='BLOCKED_TIMING',
       notes='Health/practice component only (no starter identity); overwritten release assets cannot reconstruct 2024 original states.', payload_tested='REPOSITORY_AUDIT_ONLY (frozen Phase1H/J/L audits)',
       evidence_links=['https://nflreadr.nflverse.com/articles/dictionary_injuries.html', 'nfl_models/nfl_player_outcome_v2/phase1l_qb_source_audit.json']),
    _p(provider='Internet Archive Wayback Machine (public web archive of mutable provider pages)', product='Availability API (documented) for archived snapshots of nfl.com injuries, ESPN team depth and Ourlads depth pages',
       historical_start_season='Per URL; sparse crawl', **{'2024_coverage': 'EXPLORATORY: snapshots exist for some pages around Oct 2024; sample probe in acceptance results', '2025_coverage': 'NOT_TESTED', '2026_live': 'NOT_APPLICABLE'},
       starter_state='PAGE_CONTENT_NOT_OPENED', depth_chart='PAGE_EXISTS_IN_ARCHIVE (content not opened)', injury_status='PAGE_EXISTS_IN_ARCHIVE (content not opened)', practice_status='UNKNOWN', active_inactive='NO',
       revision_history='Snapshots carry capture timestamps; enumeration (CDX) blocked by this environment\'s egress policy and not used', publication_timestamp='Capture time proves the page showed that content at that time; not announcement time',
       historical_original_vintage='PARTIAL: genuine original-vintage captures, but crawl-scheduled, not game-scheduled', T24_supported='NOT_GUARANTEED (sparse)', T90_supported='NOT_GUARANTEED (sparse)', player_id_quality='NAMES ONLY (HTML)', nflverse_crosswalk_quality='NAME_ONLY (unacceptable without ids)',
       API_available='Availability API (rate limited; HTTP 429 observed)', bulk_download=False, webhook=False, archive_endpoint='YES (Availability API; CDX blocked here)', versioned_records='YES (timestamped captures)',
       historical_backfill_cost='FREE', annual_cost='FREE', trial_available='N/A', sample_available=True, sales_contact_required=False,
       internal_use_allowed='UNRESOLVED: third-party page content rights (nfl.com/ESPN/Ourlads terms) and Internet Archive terms for bulk extraction not cleared', redistribution_allowed=False, storage_allowed=UNK, derived_model_use_allowed=UNK,
       implementation_effort='HIGH', data_quality_confidence='LOW', final_classification='BLOCKED_LICENSE',
       notes='The only route to genuinely original-vintage 2024 pages, but coverage is crawl-scheduled, identities are names, content rights are unresolved and extraction would be scraping of archived third-party pages. Existence of a capture is not evidence of a certified starter state. No page body opened.',
       payload_tested='AVAILABILITY_API_METADATA_ONLY (see acceptance results)', evidence_links=['https://archive.org/wayback/available']),
    _p(provider='Repository sources (Phase1H/J/L frozen audits)', product='weekly_rosters, transactions, gamebook starters, V1 qb_out flag',
       historical_start_season='2023-2026 frozen', starter_state='NO historical certified rows (0 certified, 394 continuity hints are not state)', depth_chart='see nflverse', injury_status='see nflverse', practice_status='NO', active_inactive='Final weekly roster status only',
       revision_history='NO', publication_timestamp='NO', historical_original_vintage='NO', T24_supported='NO', T90_supported='NO', player_id_quality='GSIS', nflverse_crosswalk_quality='HIGH', API_available='N/A', bulk_download=True, webhook=False, archive_endpoint=False, versioned_records=False,
       historical_backfill_cost='FREE', annual_cost='FREE', trial_available='N/A', sample_available=True, sales_contact_required=False, internal_use_allowed=True, redistribution_allowed=UNK, storage_allowed=True, derived_model_use_allowed=True,
       implementation_effort='LOW', data_quality_confidence='LOW for starter state', final_classification='BLOCKED_TIMING',
       notes='Official game starters are POSTGAME_EVALUATION_ONLY. V1 qb_out flag assumes an unarchived report as-of; not a state archive.', payload_tested='REPOSITORY_AUDIT_ONLY',
       evidence_links=['nfl_models/nfl_player_outcome_v2/phase1l_qb_source_audit.json', 'nfl_models/nfl_player_outcome_v2/phase1j_information_gap_audit.json']),
]


def validate_scorecard(rows=None):
    rows = PROVIDERS if rows is None else rows
    problems = []
    for i, row in enumerate(rows):
        missing = [f for f in SCORECARD_FIELDS if f not in row]
        if missing:
            problems.append((i, 'missing', missing))
        if row.get('final_classification') not in CLASSIFICATIONS:
            problems.append((i, 'bad_classification', row.get('final_classification')))
        if row.get('final_classification', '').startswith('ACCEPTED') and row.get('payload_tested') == NPT:
            problems.append((i, 'accepted_without_payload', row['provider']))
        if row.get('implementation_effort') not in ('LOW', 'MEDIUM', 'HIGH'):
            problems.append((i, 'bad_effort', row.get('implementation_effort')))
        if row.get('data_quality_confidence') not in ('HIGH', 'MEDIUM', 'LOW') and not str(row.get('data_quality_confidence', '')).startswith(('HIGH', 'MEDIUM', 'LOW')):
            problems.append((i, 'bad_confidence', row.get('data_quality_confidence')))
        if not row.get('evidence_links'):
            problems.append((i, 'no_evidence_links', row['provider']))
    return problems


def reopen_decision(rows=None, acceptance=None):
    """Exactly one decision, per the preregistered rule. ACCEPTED_* requires a tested payload."""
    rows = PROVIDERS if rows is None else rows
    cls = [r['final_classification'] for r in rows]
    thresholds = (acceptance or {}).get('historical_thresholds_met', False)
    if any(c in ('ACCEPTED_FOR_PHASE1L', 'ACCEPTED_HISTORICAL_ONLY') for c in cls) and thresholds:
        return 'PHASE1L_CAN_REOPEN_HISTORICALLY'
    if any(c == 'ACCEPTED_FORWARD_ONLY' for r, c in zip(rows, cls) if str(r.get('starter_state', '')).startswith(('YES', 'PARTIAL'))):
        return 'PHASE1L_CAN_REOPEN_FORWARD_ONLY'
    if any(c in ('PROMISING_NEEDS_SAMPLE', 'PROMISING_NEEDS_CONTRACT') for c in cls):
        return 'PHASE1L_BLOCKED_PENDING_VENDOR_ACCESS'
    return 'PHASE1L_REMAINS_BLOCKED_STARTER_STATE_DATA'


def best_picks(rows=None):
    """Purchase/access shortlist derived from the scorecard text, never from price claims."""
    return {
        'BEST_TECHNICAL_FIT': 'Sportradar (documented depth rank + injury/practice + inactives at about T-90 + Daily Change Log); unproven until a trial shows retained 2024 versions',
        'BEST_PRACTICAL_FIT': 'Sportradar trial key as an acceptance test BEFORE any spend; SportsDataIO only after written answers on archived per-change history',
        'BEST_LOW_COST_FIT': 'nflverse depth_charts_2025+ and injuries via the existing hash-captured forward snapshots (free), forward-only; upstream reuse rights for ESPN-derived rows still to be cleared',
        'BEST_ENTERPRISE_FIT': 'Sportradar (production NFL package) or SportsDataIO (per-change timestamped depth charts); contract and license questions open',
        'BEST_FORWARD_ONLY_OPTION': 'Repository forward capture of nflverse depth/injury/roster at T24/T90 with raw hashes (already operating in the V1 shadow store; read-only reuse) plus a vendor trial in parallel',
        'NOT_WORTH_PURSUING': ['TruMedia (no depth/injury endpoint)', 'FTN full API for this gate (postgame charting)', 'PFF API for this gate (no documented pregame state)', 'Stats Perform/Opta and Genius for this gate (no documented NFL pregame state product)',
                               'Wayback scraping of NFL.com/ESPN/Ourlads (rights unresolved, sparse, names only)']}


# --------------------------------------------------------------------------- forward capture plan
def forward_capture_plan():
    raw = ('provider', 'retrieved_at', 'published_at', 'payload_hash', 'game', 'team', 'player', 'raw_payload_reference')
    return {
        'schema': 'nfl-v2-phase1m-forward-capture-plan-v1',
        'status': 'DESIGN_ONLY_NO_COLLECTION_STARTED',
        'scope': 'A clean-forward collection plan for future games. No future outcome is read or scored, no scheduler/production file is changed, no unauthorized recurring scraping is started.',
        'raw_layer': {
            'fields': list(raw),
            'rules': ['append-only, content-addressed by sha256(payload bytes)', 'never overwrite a prior capture; a changed payload is a NEW capture with a new hash',
                      'published_at is stored only when the provider supplies it; otherwise null (never inferred)', 'retrieved_at is our clock at the last received byte (UTC)',
                      'store request URL (without secrets), HTTP status and response headers needed for audit']},
        'normalized_layer': {'schema': 'phase1m_qb_state_schema.json', 'rules': ['one normalized row per (source record, revision) with revision_sequence', 'raw_source_hash links to the raw layer',
                                                                                  'as_of_certified only when a registered certification rule passes', 'postgame_only/backfilled rows are stored but never visible to state_at']},
        'sources': {
            'primary_forward_candidate': {'provider': 'nflverse depth_charts_{season} (2025+ format with dt) and injuries_{season}', 'status': 'PROMISING_NEEDS_CONTRACT (ESPN-derived reuse rights) ',
                                          'mechanism': 'existing repository forward snapshots (HTTP GET of public GitHub release assets, sha256 recorded); Phase 1M adds no new collector',
                                          'note': 'dt is a load time; our retrieved_at is the only timestamp that proves what was knowable'},
            'vendor_acceptance_track': {'provider': 'Sportradar trial key (documented API)', 'status': 'PROMISING_NEEDS_SAMPLE', 'mechanism': 'poll Weekly Depth Charts, Weekly Injuries, Game Roster and Daily Change Log at the schedule below; store every payload; only after a written acceptance run'},
            'forbidden': ['systematic scraping of NFL.com/ESPN/Ourlads pages', 'any sportsbook feed', 'any outcome or postgame starter labels as features']},
        'capture_schedule': {
            'T24': {'target': 'kickoff - 24h', 'window': 'capture starts at T24 - 10 minutes and must complete before T24; a capture completing after T24 is stored as LATE and is invisible to state_at(T24)',
                    'content': 'depth chart (QB slots), injury report with practice status, roster status, recent transactions'},
            'T90': {'target': 'kickoff - 90 minutes', 'window': 'capture starts at T90 - 10 minutes and must complete before T90; inactives become public about T-90 and are expected to arrive after a provider lag, so T90 inactives need explicit lag receipts',
                    'content': 'all of T24 plus active/inactive list'},
            'between_cutoffs': 'optional hourly captures (Sportradar guidance: pull every hour or less) create the revision sequence; none may be backdated'},
        'confidence_state': {
            'rule': 'confidence stays null unless a registered certification rule passes; no probability is derived from depth rank',
            'states': 'canonical STATE_TYPES; MULTI_QB or UNCERTAIN_STARTER whenever visible rows disagree or none exist'},
        'timestamps': {'publication_timestamp': 'provider supplied only', 'retrieval_timestamp': 'our UTC clock', 'effective_timestamp': 'provider supplied or null', 'visibility': 'record visible at a cutoff only if retrieval_timestamp <= cutoff (and publication <= cutoff when present)'},
        'hashes': {'raw_source_hash': 'sha256(raw bytes)', 'normalized_record_hash': 'sha256(canonical record excluding itself)', 'batch_manifest': 'sha256 over the ordered hashes of a capture group'},
        'immutability': {'storage': 'append-only files or an immutable object store; files are written once and then marked read-only; a store verifier recomputes hashes', 'no_deletes': True, 'corrections': 'a correction is a new revision row'},
        'revision_handling': {'same_source_record': 'revision_sequence increases; older revisions remain', 'state_at': 'returns the highest revision visible at the cutoff; later revisions never leak backward',
                              'provider_retracts': 'a retraction is a new row (valid_to set on the superseded interval), not a deletion'},
        'leakage_controls': ['no target-game participation field may populate a pregame row', 'postgame_only_flag rows are excluded by construction', 'game-day starters, snap counts and box scores are evaluation labels only'],
        'authorization': 'The plan activates only after (1) the repository owner authorizes the specific source and terms, and (2) a vendor trial passes the acceptance harness. This phase starts no collection.',
        'activation_checklist': ['written terms for storage and derived-model use', 'GSIS crosswalk check on a QB sample', 'timestamp semantics confirmed in writing', 'dry run on a past week with synthetic clock', 'owner approves a protocol before any score is computed'],
    }


def schema_document():
    return {
        'schema': 'nfl-v2-phase1m-qb-pregame-state-schema-v1',
        'status': 'DESIGN_COMPLETE_NOT_POPULATED',
        'append_only': True,
        'description': 'Canonical QB pregame-state row. A provider revision is a NEW row with a higher revision_sequence; nothing is overwritten.',
        'enums': {'state_type': list(STATE_TYPES), 'evidence_type': list(EVIDENCE_TYPES), 'cutoff_eligibility': list(CUTOFF_ELIGIBILITY)},
        'fields': [{'name': n, 'type': t, 'required': r, 'description': d} for n, t, r, d in SCHEMA_FIELDS],
        'visibility_rule': 'See phase1m_qb_state_protocol.json temporal_integrity; implemented by QBStateArchive.visible/state_at.',
        'canonical_hash': 'normalized_record_hash = sha256(canonical JSON of every other field)',
        'uniqueness': '(source_provider, source_record_id, revision_sequence)',
        'certification': 'as_of_certified requires publication_timestamp, original_vintage_flag, no backfill and no postgame flag.',
        'sequence_example': ['QUESTIONABLE', 'EXPECTED_STARTER', 'VERIFIED_STARTER'],
        'population_status': 'NO ROWS: no accepted source exists; the archive is empty.',
    }


def build_outputs():
    return {
        'phase1m_qb_state_schema.json': schema_document(),
        'phase1m_forward_capture_plan.json': forward_capture_plan(),
        'phase1m_qb_state_provider_scorecard.json': {
            'schema': 'nfl-v2-phase1m-provider-scorecard-v1', 'base_head': BASE_HEAD, 'evidence_date_utc': EVIDENCE_DATE_UTC,
            'rule': 'Dated documentation evidence. Prices are never invented. A documented endpoint is not original-vintage proof. NO_PROVIDER_PAYLOAD_TESTED means no vendor bytes were opened.',
            'classification_counts': {c: sum(1 for r in PROVIDERS if r['final_classification'] == c) for c in CLASSIFICATIONS},
            'providers': PROVIDERS, 'picks': best_picks(), 'decision_from_scorecard': reopen_decision()},
    }


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='write deterministic outputs')
    ap.add_argument('--check', action='store_true', help='fail if committed outputs differ')
    ap.add_argument('--out-dir', default=str(ART))
    args = ap.parse_args()
    problems = validate_scorecard()
    if problems:
        raise SystemExit('scorecard invalid: ' + json.dumps(problems))
    outputs = build_outputs()
    out_dir = Path(args.out_dir)
    for name, value in outputs.items():
        path = out_dir / name
        if args.write:
            path.write_text(dump(value))
        if args.check and path.read_text() != dump(value):
            raise SystemExit('differs: ' + name)
    print(reopen_decision())


if __name__ == '__main__':
    main()

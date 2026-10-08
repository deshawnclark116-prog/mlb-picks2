"""Immutable pregame observations, identity quarantine and horizon qualification.

No live source is enabled by default. A caller must supply documented access rights.
Observed roster membership never becomes confirmed dressing implicitly.
"""
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhl_v2_phase1a_sog_forward import canon, parse_iso, sha_text, HORIZON_MIN, Ledger

STATES = {'ROSTER_OBSERVED', 'EXPECTED_DRESSED', 'CONFIRMED_DRESSED', 'SCRATCH_OBSERVED', 'INJURED_OBSERVED', 'UNKNOWN'}
KINDS = {'game_roster', 'official_team_roster', 'licensed_deployment', 'authorized_announcement'}


def timestamp(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('timezone required')
    return d.astimezone(timezone.utc)


def eligible(record, game, horizon, cutoff):
    """Each decision binds game/start/horizon; later revisions never leak backward."""
    c = timestamp(cutoff)
    expected = timestamp(game['game_start_utc']) - timedelta(minutes=HORIZON_MIN[horizon])
    if c != expected or timestamp(record['cutoff']) != c or record['game_id'] != game['game_id'] or record['scheduled_start'] != game['game_start_utc'] or record['horizon'] != horizon:
        return False
    if not record.get('rights_basis') or record.get('postgame_only', False):
        return False
    for key in ('retrieved_at', 'published_at', 'effective_at'):
        if record.get(key) and timestamp(record[key]) > c:
            return False
    for observation in record.get('observations', []):
        for key in ('retrieved_at', 'published_at', 'effective_at'):
            if observation.get(key) and timestamp(observation[key]) > c:
                return False
    age = (c - timestamp(record['retrieved_at'])).total_seconds()
    if not 0 <= age <= record['freshness_seconds']:
        return False
    return True


def snapshot(raw, *, game, horizon, cutoff, retrieved_at, source_kind, rights_basis,
             published_at=None, freshness_seconds=900, observations=()):
    """Import a documented observation with raw bytes/provenance. No inferred confirmations."""
    if not rights_basis or source_kind not in KINDS:
        raise ValueError('authorized source and rights basis required')
    if horizon not in HORIZON_MIN or freshness_seconds <= 0:
        raise ValueError('invalid source freshness or horizon')
    # Game-scoped JSON APIs must agree with decision identity if they expose it.
    payload = json.loads(raw)
    if isinstance(payload, dict):
        if payload.get('id', game['game_id']) != game['game_id'] or payload.get('startTimeUTC', game['game_start_utc']) != game['game_start_utc']:
            raise ValueError('source game/start mismatch')
    base = {'game_id': game['game_id'], 'scheduled_start': game['game_start_utc'],
            'horizon': horizon, 'cutoff': cutoff, 'retrieved_at': retrieved_at,
            'published_at': published_at, 'source_kind': source_kind,
            'raw_source_sha256': hashlib.sha256(raw).hexdigest(), 'rights_basis': rights_basis,
            'freshness_seconds': freshness_seconds, 'observations': list(observations),
            'postgame_only': isinstance(payload, dict) and payload.get('gameState') not in (None, 'FUT', 'PRE')}
    teams = {game['home_team_id'], game['away_team_id']}
    identities = set()
    for r in base['observations']:
        if r['observed_state'] not in STATES or r['team_id'] not in teams or not isinstance(r['player_id'], int) or r['player_id'] <= 0:
            raise ValueError('invalid observation identity/state')
        if (r['team_id'], r['player_id']) in identities:
            raise ValueError('duplicate observation')
        identities.add((r['team_id'], r['player_id']))
        if r['observed_state'] == 'CONFIRMED_DRESSED' and (not r.get('explicit_confirmation_evidence') or source_kind in {'game_roster', 'official_team_roster'}):
            raise ValueError('roster observation is not confirmed dressed evidence')
    base['timing_eligible'] = eligible(base, game, horizon, cutoff)
    base['record_sha256'] = sha_text(canon(base))
    return base


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)
        self.ledger = Ledger(self.root / 'snapshots.jsonl')

    def append(self, record, raw):
        self.ledger.verify()
        unhashed = {k: v for k, v in record.items() if k != 'record_sha256'}
        if sha_text(canon(unhashed)) != record['record_sha256'] or hashlib.sha256(raw).hexdigest() != record['raw_source_sha256']:
            raise ValueError('snapshot hash mismatch')
        if record['record_sha256'] in {r['record_sha256'] for r in self.ledger.rows()}:
            return False
        blob = self.root / 'blobs' / (record['raw_source_sha256'] + '.gz')
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists() and gzip.decompress(blob.read_bytes()) != raw:
            raise ValueError('existing content address corrupted')
        if not blob.exists():
            blob.write_bytes(gzip.compress(raw, mtime=0))
        self.ledger.append(record)
        return True

    def state_at(self, game, horizon, cutoff):
        self.ledger.verify()
        by_source = {}
        for r in self.ledger.rows():
            if eligible(r, game, horizon, cutoff):
                key = (r['source_kind'], r.get('endpoint', ''))
                if key not in by_source or timestamp(r['retrieved_at']) > timestamp(by_source[key]['retrieved_at']):
                    by_source[key] = r
        return [by_source[k] for k in sorted(by_source)]


def resolve_identity(candidates, observations):
    """Historical candidate membership is a hypothesis; opposing-team conflicts abstain.

    Eligible explicit current ownership may resolve an ambiguity, but no outcome may.
    All conflicts remain in the receipt; no winner inferred from actual participation.
    """
    groups = defaultdict(list)
    for c in candidates:
        groups[c['player_id']].append(c)
    evidence = defaultdict(set)
    for o in observations:
        if o.get('membership_verified'):
            evidence[o['player_id']].add(o['team_id'])
    accepted, abstentions = [], []
    for pid in sorted(groups):
        group = groups[pid]
        teams = {c['team_id'] for c in group}
        proof = evidence[pid]
        if len(proof) > 1:
            abstentions.extend(dict(c, reason='CONFLICTING_PREGAME_OWNERSHIP') for c in group)
        elif proof:
            chosen = next(iter(proof))
            accepted.extend(c for c in group if c['team_id'] == chosen)
            abstentions.extend(dict(c, reason='PREGAME_OWNERSHIP_CHANGED') for c in group if c['team_id'] != chosen)
        elif len(teams) > 1:
            abstentions.extend(dict(c, reason='AMBIGUOUS_PLAYER_TEAM_IDENTITY') for c in group)
        elif len(group) > 1:
            raise ValueError('duplicate same-team candidate')
        else:
            accepted.extend(group)
    return accepted, abstentions


def certification(records):
    """Each item is ONE complete, unique game/horizon plus final dressed truth.

    Completeness of both teams, original pre-cutoff capture and final truth is
    mandatory. Aggregate recall never pools horizons; fail one game/date gate.
    """
    seen = set()
    out = {}
    for h in HORIZON_MIN:
        games = []
        for r in records:
            if r['horizon'] != h:
                continue
            key = (r['game_id'], h)
            if key in seen:
                raise ValueError('duplicate certification game/horizon')
            seen.add(key)
            if not r.get('capture_eligible') or not r.get('both_teams_complete') or not r.get('final_truth_complete'):
                continue
            truth = {tuple(x) for x in r['dressed_ids']}
            pool = {tuple(x) for x in r['observed_ids']}
            if not truth:
                continue
            games.append((r, len(truth & pool), len(truth), len(pool - truth)))
        n = len(games)
        den = sum(g[2] for g in games)
        hits = sum(g[1] for g in games)
        dates = len({g[0]['date'] for g in games})
        recall = hits / den if den else None
        extra = sum(g[3] for g in games)
        out[h] = {'complete_games': n, 'distinct_dates': dates, 'dressed_recall': recall,
                  'extra_roster_members': extra, 'precision': hits / (hits + extra) if hits + extra else None,
                  'qualified_candidate_source': bool(n >= 40 and dates >= 3 and recall >= .995),
                  'confirmed_lineup': False,
                  'meaning': 'Recall qualifies candidate coverage, never turns a roster into confirmed dressing'}
    return out


def horizon_changes(records):
    pools = defaultdict(dict)
    for r in records:
        key = (r['game_id'], r['scheduled_start'])
        pools[key][r['horizon']] = {(o['team_id'], o['player_id'], o['observed_state']) for o in r['observations']}
    result = []
    for key, hs in sorted(pools.items()):
        for a, b in [('T24H', 'T90'), ('T90', 'T30')]:
            if a in hs and b in hs:
                result.append({'game_id': key[0], 'from': a, 'to': b, 'added': sorted(hs[b] - hs[a]), 'removed': sorted(hs[a] - hs[b])})
    return result

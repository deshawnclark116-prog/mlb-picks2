#!/usr/bin/env python3
"""Phase2A-FWD forward V2 shadow forecasts (research only; NOT production).

Builds immutable T24 / T90 central projections for games that have not started, with the SURVIVING V2 chain only (frozen Phase1B team volume x
frozen Phase1D shares x frozen Phase1F efficiency). No model is fitted here. The CLI has no clock override: a forecast can only be written inside
its preregistered window, strictly before its cutoff, and is appended to a hash-chained append-only ledger. Target-game outcomes, sportsbook data
and later snapshots are never read.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as p1d
import nfl_v2_phase1e_integrated as p1e
import nfl_v2_phase1g_receiving_mechanics as p1g
import nfl_v2_competent_human_baseline as human
import nfl_v2_phase1o_role_sources as O
import nfl_v2_qb_state_ingestion as QB
from nfl_v2_phase2a_ledger import append_record, canon, read_ledger, rnd, sha_bytes, verify_chain

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
LEDGER = ART / 'phase2a_ledger'
INPUTS = ART / 'phase2a_inputs'
PROTOCOL = 'phase2a_forward_protocol.json'
ENGINE_VERSION = 'nfl-v2-phase2a-forward-1.0'
UTC = timezone.utc
ET = ZoneInfo('America/New_York')
CUTOFFS = {'T24': (timedelta(hours=48), timedelta(hours=24)), 'T90': (timedelta(hours=3), timedelta(minutes=90))}
GAMES_ALLOWLIST = ('game_id', 'season', 'game_type', 'week', 'gameday', 'gametime', 'away_team', 'home_team')
SCHEDULE_URLS = ('https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv', 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv')
LIVE_SEASON_FILES = {'stats': 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv',
                     'roster': 'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{y}.csv',
                     'pbp': 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz'}
HISTORY_SEASONS = (2023, 2024, 2025)
THRESHOLD = {'rec_yds': 3.0, 'rush_yds': 5.0}
RECEIVING_POS = {'RB', 'FB', 'HB', 'WR', 'TE'}
RUSHING_POS = {'RB', 'FB', 'HB'}
MIN_ROSTER_ROWS = 40
LEDGER_FILES = ('forecasts_T24.jsonl', 'forecasts_T90.jsonl', 'comparators.jsonl', 'game_status.jsonl', 'runs.jsonl')
FROZEN_ARTIFACTS = ('phase1b_opportunity_snapshot.json', 'phase1d_role_allocation_snapshot.json', 'phase1e_integrated_snapshot.json', 'phase1f_efficiency_snapshot.json', 'phase1g_receiving_mechanics_snapshot.json', 'competent_human_snapshot.json', 'phase1h_source_coverage.json')
CODE_FILES = ('nfl_v2_phase2a_forward_forecast.py', 'nfl_v2_phase2a_forward_grade.py', 'nfl_v2_phase2a_ledger.py', 'nfl_v2_qb_state_ingestion.py')
FROZEN_UPSTREAM_CODE = ('nfl_v2_phase1a_direct.py', 'nfl_v2_phase1b_opportunity.py', 'nfl_v2_phase1d_role_allocation.py', 'nfl_v2_phase1e_integrated.py', 'nfl_v2_phase1g_receiving_mechanics.py', 'nfl_v2_competent_human_baseline.py', 'nfl_v2_phase1m_qb_state_sources.py')


class RefusedError(RuntimeError):
    """A guard (cutoff, outcome-data, history) refused to produce a forecast."""


# --------------------------------------------------------------------------- generic helpers
def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def iso(dt):
    return dt.astimezone(UTC).isoformat()


def now_utc():
    return datetime.now(UTC)


def code_sha256():
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def git_sha():
    env = os.environ.get('GITHUB_SHA')
    if env:
        return env
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        return 'UNKNOWN'


# --------------------------------------------------------------------------- schedule and windows
def kickoff_utc(gameday, gametime):
    local = datetime.strptime(gameday + ' ' + gametime, '%Y-%m-%d %H:%M').replace(tzinfo=ET)
    return local.astimezone(UTC)


def parse_schedule(text, seasons=None):
    """Allowlisted columns only; result, scores and every odds / line column are never read."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r['game_type'] != 'REG' or (seasons and int(r['season']) not in seasons) or not r['gameday'] or not r['gametime']:
            continue
        rows.append({k: r[k] for k in GAMES_ALLOWLIST})
    rows.sort(key=lambda g: (g['gameday'], g['gametime'], g['game_id']))
    return rows


def window(kickoff, label):
    opens, cutoff = CUTOFFS[label]
    return kickoff - opens, kickoff - cutoff


def classify(kickoff, label, now):
    """DUE | NOT_YET_DUE | MISSED_<label>_CUTOFF | GAME_STARTED for one game and cutoff at time `now`."""
    opens, cutoff = window(kickoff, label)
    if now >= kickoff:
        return 'GAME_STARTED'
    if now >= cutoff:
        return f'MISSED_{label}_CUTOFF'
    if now < opens:
        return 'NOT_YET_DUE'
    return 'DUE'


def assert_before_cutoff(generated_at, kickoff, label):
    _opens, cutoff = window(kickoff, label)
    if not (generated_at < cutoff <= kickoff):
        raise RefusedError(f'cutoff guard: generated_at {iso(generated_at)} is not before the {label} cutoff {iso(cutoff)}')


# --------------------------------------------------------------------------- append-only hash-chained ledger
def ledger_path(name, ledger_dir=None):
    return Path(ledger_dir or LEDGER) / name


def forecast_id(cutoff, game_id, player_id):
    return sha_bytes(f'{ENGINE_VERSION}|{cutoff}|{game_id}|{player_id}'.encode())[:24]


# --------------------------------------------------------------------------- inputs
def curl(url, dest):
    subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '5', '--max-time', '600', url, '-o', str(dest)], check=True)


def pinned_manifest():
    seasons = read_json(ART / 'phase1h_source_coverage.json')['seasons']
    return {int(y): {k: {kk: v[kk] for kk in ('local_name', 'url', 'sha256')} for k, v in seasons[y].items() if k in ('stats', 'roster', 'pbp')} for y in map(str, HISTORY_SEASONS)}


def fetch_inputs(data_dir, with_pbp=True, fetch=curl, clock=now_utc):
    """Pinned 2023-2025 bytes (digest verified, mismatch is fatal) plus the live 2026 files (digest and retrieval time recorded)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest, inputs = pinned_manifest(), {}
    for year, files in manifest.items():
        for kind, src in files.items():
            if kind == 'pbp' and not with_pbp:
                continue
            dest = data_dir / src['local_name']
            if not (dest.exists() and sha_file(dest) == src['sha256']):
                fetch(src['url'], dest)
                if sha_file(dest) != src['sha256']:
                    raise ValueError('Pinned source digest mismatch: ' + src['local_name'])
            inputs[src['local_name']] = {'sha256': src['sha256'], 'pinned': True, 'url': src['url'], 'retrieved_at': None}
    for kind, url in LIVE_SEASON_FILES.items():
        if kind == 'pbp' and not with_pbp:
            continue
        year = max(HISTORY_SEASONS) + 1
        name = {'stats': f'stats_player_week_{year}.csv', 'roster': f'roster_weekly_{year}.csv', 'pbp': f'pbp_{year}.csv.gz'}[kind]
        dest = data_dir / name
        u = url.format(y=year)
        fetch(u, dest)
        inputs[name] = {'sha256': sha_file(dest), 'pinned': False, 'url': u, 'retrieved_at': iso(clock())}
    return inputs


def fetch_schedule(data_dir, fetch=curl, clock=now_utc):
    last = None
    for url in SCHEDULE_URLS:
        dest = Path(data_dir) / 'games_raw.csv'
        try:
            fetch(url, dest)
            raw = dest.read_bytes()
            rows = parse_schedule(raw.decode('utf-8'))
            if rows:
                dest.unlink()                                          # the raw file carries sportsbook columns: it is not kept
                return rows, {'url': url, 'raw_sha256': sha_bytes(raw), 'retrieved_at': iso(clock()), 'allowlisted_rows_sha256': sha_bytes(canon(rows).encode())}
        except Exception as e:                                         # noqa: BLE001
            last = e
    raise RuntimeError(f'schedule unavailable: {last}')


def archive_input(path):
    """Content-addressed gz copy of a live input (stats / roster) so the exact bytes the engine saw are preserved."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    h = sha_file(path)
    out = INPUTS / f'{Path(path).stem}_{h[:16]}.csv.gz'
    if not out.exists():
        with gzip.GzipFile(filename='', mode='wb', fileobj=open(out, 'wb'), mtime=0) as g:
            g.write(Path(path).read_bytes())
    return out.name


# --------------------------------------------------------------------------- engine context (pure function of pre-cutoff inputs)
class Context:
    """Frozen-chain context for ONE target (season, week): history is filtered to strictly earlier weeks BEFORE any index is built."""

    def __init__(self, data_dir, season, week, with_pbp=True, stats_paths=None, roster_paths=None, pbp_paths=None):
        data_dir = Path(data_dir)
        self.season, self.week = season, week
        stats_paths = stats_paths or [data_dir / f'stats_player_week_{y}.csv' for y in (*HISTORY_SEASONS, season)]
        roster_paths = roster_paths or [data_dir / f'roster_weekly_{season}.csv']
        players, team, opp = p1a.load_stats(stats_paths)
        target = (season, week)
        self.outcome_rows_present = sorted({(r['team'], r['week']) for r in players if (r['season'], r['week']) == target})
        keep = lambda s, w: (s, w) < target
        self.players = [r for r in players if keep(r['season'], r['week'])]
        self.team = {k: v for k, v in team.items() if keep(k[0], k[1])}
        self.opp = {k: v for k, v in opp.items() if keep(k[0], k[1])}
        p1a.build_indexes(self.players, self.team, self.opp)
        p1b.build_extra_indexes(self.players, self.team, self.opp)
        p1d.build_context(self.players, self.team)
        self.roster_meta = p1d.load_rosters(roster_paths)
        human.load_roster_membership(roster_paths)
        human.build_indexes(self.players, self.team, self.opp)
        self.b = read_json(ART / 'phase1b_opportunity_snapshot.json')
        self.d = read_json(ART / 'phase1d_role_allocation_snapshot.json')
        self.names, self.pos, self.status_rows = {}, {}, {}
        for path in roster_paths:
            with open(path, newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('game_type', 'REG') in ('REG', '') and r.get('gsis_id'):
                        self.names[r['gsis_id']] = r.get('full_name') or ''
        self.latest_pos = {}
        for r in self.players:
            self.latest_pos[r['player_id']] = r.get('position')
        self.pbp_ok = False
        self.g_cfg = read_json(ART / 'phase1g_receiving_mechanics_snapshot.json')['selected_config']
        if with_pbp:
            try:
                pbp_paths = pbp_paths or [data_dir / n for n in ('pbp_2023.csv.gz', 'pbp_2024.csv.gz', 'pbp_2025.csv.gz', f'pbp_{season}.csv.gz')]
                posmap = {(r['season'], r['player_id']): r['position'] for r in self.players}
                p1g.load_pbp([p for p in pbp_paths if Path(p).exists()], posmap)
                self.pbp_ok = True
            except Exception as e:                                       # noqa: BLE001
                self.pbp_error = str(e)

    # ----- team-level readiness guards
    def history_complete(self, team, previous_week_key):
        return previous_week_key is None or (previous_week_key[0], previous_week_key[1], team) in self.team

    def roster_rows(self, team):
        return len(p1d._ROSTERS.get((self.season, self.week, team), ()))

    # ----- eligibility and forecast for one (team, opponent)
    def candidates(self, team, opponent):
        out = []
        for outcome, positions in (('rec_yds', RECEIVING_POS), ('rush_yds', RUSHING_POS)):
            rows = []
            for x in p1d.eligible_roster(self.season, self.week, team, outcome):
                pos = (x.get('position') or '').upper()
                if pos in positions:
                    rows.append({'season': self.season, 'week': self.week, 'player_id': x['player_id'], 'team': team, 'opponent': opponent, 'position': pos or self.latest_pos.get(x['player_id'])})
            out.append((outcome, p1b.fixed_meaningful_rows(self.players, rows, outcome)))
        return out

    def role_info(self, r, outcome):
        key = p1a.SPEC[outcome]['opp']
        hist = p1a.prior_player_rows(self.players, (self.season, self.week), r['player_id'], r['team'], 8)
        vals = [x.get(key, 0.0) for x in hist][-3:]
        shares = p1a.player_share_history(self.players, self.team, (self.season, self.week), r['player_id'], r['team'], key, 3)
        rng = (max(shares) - min(shares)) if len(shares) >= 2 else None
        thr = THRESHOLD[outcome]
        avg = sum(vals) / len(vals) if vals else 0.0
        limit = 0.08 if outcome == 'rec_yds' else 0.15
        if len(hist) >= 4 and avg >= 2 * thr and rng is not None and rng <= limit:
            conf = 'HIGH'
        elif len(hist) >= 3 and avg >= thr:
            conf = 'MEDIUM'
        else:
            conf = 'LOW'
        return conf, {'prior_team_games_last8': len(hist), 'prior3_average_opportunities': avg, 'last3_share_range': rng, 'share_history_n': len(shares)}

    def chain(self, r, outcome):
        return p1e.receipt(self.players, self.team, self.opp, r, outcome, self.b, self.d)

    def human_projection(self, r, outcome):
        x = human.receipt(self.players, self.team, self.opp, r, outcome)
        return None if x is None else {'projection': x['point_projection'], 'opportunity': x['player_opportunity_projection']}

    def simple_baseline(self, r):
        hist = p1a.prior_player_rows(self.players, (self.season, self.week), r['player_id'], None, 3)
        if not hist:
            return None
        m = lambda k: sum(x.get(k, 0.0) for x in hist) / len(hist)
        return {'targets': m('targets'), 'receptions': m('receptions'), 'receiving_yards': m('receiving_yards'), 'carries': m('carries'), 'rushing_yards': m('rushing_yards'), 'games': len(hist)}

    def phase1g(self, r):
        if not self.pbp_ok:
            return None
        x = p1g.efficiency_receipt({**r}, self.g_cfg)
        return None if x is None else {'projected_catch_rate': x['projected_catch_rate'], 'projected_air_per_catch': x['projected_air_per_catch'], 'projected_yac_per_catch': x['projected_yac_per_catch'], 'projected_yards_per_target': x['projected_yards_per_target']}

    def availability(self, team, pid):
        for x in p1d._ROSTERS.get((self.season, self.week, team), ()):
            if x['player_id'] == pid:
                st = (x.get('status') or '').upper()
                return (x.get('status') or 'UNKNOWN'), ('MEDIUM' if st == 'ACT' else 'LOW')
        return 'NO_TARGET_WEEK_ROSTER_ROW', 'LOW'


def build_forecasts(ctx, game, team, opponent, home_away):
    """V2 forecast objects (no comparator, no V1 field, no outcome) and their comparator rows for one team in one game."""
    forecasts, comparators, summary = {}, {}, {'considered': 0}
    by_player = defaultdict(dict)
    for outcome, rows in ctx.candidates(team, opponent):
        summary[outcome + '_eligible'] = len(rows)
        for r in rows:
            by_player[r['player_id']][outcome] = r
    for pid in sorted(by_player):
        per = by_player[pid]
        any_r = next(iter(per.values()))
        state, conf = ctx.availability(team, pid)
        flags = ['QB_STATE_UNCERTIFIED', 'INJURY_DESIGNATIONS_NOT_AN_INPUT', 'ROSTER_STATUS_ONLY_AVAILABILITY']
        rec = {'receiving': None, 'rushing': None}
        reasons, role_confs, completeness = {}, {}, {}
        final = {'targets': None, 'receptions': None, 'receiving_yards': None, 'carries': None, 'rushing_yards': None}
        cmp_ = {'human': {}, 'simple_baseline': ctx.simple_baseline(any_r)}
        if 'rec_yds' in per:
            r = per['rec_yds']
            ry, rc = ctx.chain(r, 'rec_yds'), ctx.chain(r, 'rec')
            conf_r, info = ctx.role_info(r, 'rec_yds')
            role_confs['receiving'], completeness['receiving'] = conf_r, info
            reasons['receiving'] = f"frozen Phase1D meaningful receiving participant: prior-3 average targets {info['prior3_average_opportunities']:.2f} >= {THRESHOLD['rec_yds']} with {info['prior_team_games_last8']} prior team games; active target-week roster status {state}"
            if ry is not None:
                g = ctx.phase1g(r)
                catch = (rc['final_efficiency_projection'] if rc else None)
                rec['receiving'] = {'expected_team_opportunity': ry['team_opportunity_projection'], 'expected_target_share': ry['role_share_projection'], 'expected_targets': ry['player_opportunity_projection'],
                                    'expected_catch_rate': catch, 'expected_yards_per_target': ry['final_efficiency_projection'], 'final_receptions': None if rc is None else rc['point_projection'], 'final_receiving_yards': ry['point_projection'],
                                    'expected_air_component': None if g is None else g['projected_air_per_catch'], 'expected_yac_component': None if g is None else g['projected_yac_per_catch'],
                                    'phase1g_diagnostic': g, 'phase1g_status': 'DIAGNOSTIC_ONLY_NEVER_USED_IN_FINAL' if g else 'UNAVAILABLE_NO_PBP_OR_NO_HISTORY', 'rec_head_team_targets': None if rc is None else rc['team_opportunity_projection']}
                final.update(targets=ry['player_opportunity_projection'], receptions=None if rc is None else rc['point_projection'], receiving_yards=ry['point_projection'])
                for o, k in (('rec_yds', 'receiving_yards'), ('rec', 'receptions')):
                    h = ctx.human_projection(r, o)
                    cmp_['human'][k] = None if h is None else h['projection']
                    if o == 'rec_yds':
                        cmp_['human']['targets'] = None if h is None else h['opportunity']
            else:
                flags.append('RECEIVING_CHAIN_UNAVAILABLE_INSUFFICIENT_HISTORY')
        if 'rush_yds' in per:
            r = per['rush_yds']
            rr = ctx.chain(r, 'rush_yds')
            conf_r, info = ctx.role_info(r, 'rush_yds')
            role_confs['rushing'], completeness['rushing'] = conf_r, info
            reasons['rushing'] = f"frozen Phase1D meaningful rushing participant: prior-3 average carries {info['prior3_average_opportunities']:.2f} >= {THRESHOLD['rush_yds']} with {info['prior_team_games_last8']} prior team games; active target-week roster status {state}"
            if rr is not None:
                rec['rushing'] = {'expected_team_opportunity': rr['team_opportunity_projection'], 'expected_carry_share': rr['role_share_projection'], 'expected_carries': rr['player_opportunity_projection'],
                                  'expected_yards_per_carry': rr['final_efficiency_projection'], 'final_rushing_yards': rr['point_projection']}
                final.update(carries=rr['player_opportunity_projection'], rushing_yards=rr['point_projection'])
                h = ctx.human_projection(r, 'rush_yds')
                cmp_['human']['rushing_yards'] = None if h is None else h['projection']
                cmp_['human']['carries'] = None if h is None else h['opportunity']
            else:
                flags.append('RUSHING_CHAIN_UNAVAILABLE_INSUFFICIENT_HISTORY')
        if rec['receiving'] is None and rec['rushing'] is None:
            continue
        if any(c != 'HIGH' for c in role_confs.values()):
            flags.append('ROLE_NOT_HIGH_CONFIDENCE')
        if conf == 'LOW':
            flags.append('AVAILABILITY_LOW_CONFIDENCE')
        summary['considered'] += 1
        forecasts[pid] = {'game_id': game['game_id'], 'season': int(game['season']), 'week': int(game['week']), 'team': team, 'opponent': opponent, 'home_away': home_away, 'player_id': pid, 'player_name': ctx.names.get(pid, ''),
                          'position': any_r['position'], 'eligibility_status': 'ELIGIBLE', 'eligibility_reason': reasons, 'role_confidence': role_confs, 'availability_state': state, 'availability_confidence': conf, 'data_completeness': completeness,
                          'qb_state_status': 'BLOCKED_ABSTAIN_QB_STATE_UNCERTIFIED', 'receiving': rec['receiving'], 'rushing': rec['rushing'], 'final_central_projection': final, 'uncertainty_flags': flags,
                          'blocked_information': ['qb_starter_health_state', 'injury_designations', 'routes_alignment', 'snaps', 'defensive_personnel']}
        comparators[pid] = cmp_
    return forecasts, comparators, summary


# --------------------------------------------------------------------------- V1 comparator (read-only, separate from the V2 build)
def v1_projection(pick_log_path, season, week, player_id, before):
    """Latest V1 central projection logged before `before` (projected_median only). Nothing else in the log is read; missing is recorded, never imputed."""
    markets = {'receiving_yards': 'receiving_yards', 'rushing_yards': 'rushing_yards', 'receptions': 'receptions'}
    best = {}
    try:
        lines = Path(pick_log_path).read_text().splitlines()
    except OSError:
        return {k: None for k in markets.values()} | {'status': 'V1_LOG_UNAVAILABLE'}
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('player_id') != player_id or r.get('season') != season or r.get('week') != week:
            continue
        base = str(r.get('market', '')).replace('_early_season', '')
        if base not in markets or r.get('projected_median') is None or not r.get('logged_at'):
            continue
        try:
            ts = datetime.fromisoformat(str(r['logged_at']).replace('Z', '+00:00'))
        except ValueError:
            continue
        if ts >= before:
            continue
        if base not in best or ts > best[base][0]:
            best[base] = (ts, float(r['projected_median']))
    out = {markets[k]: (best[k][1] if k in best else None) for k in markets}
    out['status'] = 'V1_PUBLISHED_BEFORE_CUTOFF' if best else 'V1_NOT_PUBLISHED_BEFORE_CUTOFF'
    return out


# --------------------------------------------------------------------------- the due run
def run_due(data_dir, ledger_dir=None, with_pbp=True, run_id=None, clock=now_utc, fetch=curl, schedule=None, pick_log=None, inputs=None, archive=True, game_filter=None):
    """Check every in-scope game, create due forecasts, log missed ones, refuse anything late. Returns a run summary."""
    proto = read_json(ART / PROTOCOL)
    start = datetime.fromisoformat(proto['evaluation_start_utc'])
    ledger_dir = Path(ledger_dir or LEDGER)
    started = clock()
    run_id = run_id or started.strftime('%Y%m%dT%H%M%SZ')
    sched, sched_meta = schedule if schedule else fetch_schedule(data_dir, fetch, clock)
    inputs = inputs if inputs is not None else fetch_inputs(data_dir, with_pbp, fetch, clock)
    live_retrieved = max([v['retrieved_at'] for v in inputs.values() if v.get('retrieved_at')] + [sched_meta['retrieved_at']])
    summary = {'run_id': run_id, 'engine_version': ENGINE_VERSION, 'git_sha': git_sha(), 'started_at': iso(started), 'games': [], 'created': defaultdict(int), 'missed': 0, 'not_yet_due': 0, 'blocked': 0, 'qb_abstentions': 0}
    games = [g for g in sched if kickoff_utc(g['gameday'], g['gametime']) >= start and (game_filter is None or g['game_id'] in game_filter)]
    ctxs = {}
    prev_game = defaultdict(dict)
    by_team = defaultdict(list)
    for g in sched:
        for t in (g['home_team'], g['away_team']):
            by_team[t].append((int(g['season']), int(g['week'])))
    for g in games:
        ko = kickoff_utc(g['gameday'], g['gametime'])
        wk = (int(g['season']), int(g['week']))
        for label in ('T24', 'T90'):
            status_now = classify(ko, label, clock())
            ev = {'game_id': g['game_id'], 'cutoff_type': label, 'kickoff_time': iso(ko), 'cutoff_time': iso(window(ko, label)[1]), 'window_opens': iso(window(ko, label)[0]), 'status': status_now}
            summary['games'].append(ev)
            if status_now == 'NOT_YET_DUE':
                summary['not_yet_due'] += 1
                continue
            if status_now in (f'MISSED_{label}_CUTOFF', 'GAME_STARTED'):
                if not _has_forecast(ledger_dir, label, g['game_id']):
                    line = append_record(ledger_path('game_status.jsonl', ledger_dir), {'event_id': f"{g['game_id']}|{label}|MISSED", 'event': f'MISSED_{label}_CUTOFF', 'game_id': g['game_id'], 'cutoff_type': label, 'kickoff_time': iso(ko),
                                                                                       'cutoff_time': iso(window(ko, label)[1]), 'observed_at': iso(clock()), 'engine_version': ENGINE_VERSION, 'note': 'no forecast was created before the cutoff; never backfilled'}, 'event_id')
                    summary['missed'] += line is not None
                continue
            # DUE: build and freeze
            key = wk
            if key not in ctxs:
                ctxs[key] = Context(data_dir, wk[0], wk[1], with_pbp)
            ctx = ctxs[key]
            blocked = _guards(ctx, g, by_team)
            if blocked:
                append_record(ledger_path('game_status.jsonl', ledger_dir), {'event_id': f"{g['game_id']}|{label}|{run_id}|{blocked}", 'event': blocked, 'game_id': g['game_id'], 'cutoff_type': label, 'observed_at': iso(clock()), 'engine_version': ENGINE_VERSION, 'run_id': run_id}, 'event_id')
                summary['blocked'] += 1
                continue
            _write_game(ctx, g, ko, label, ledger_dir, run_id, inputs, sched_meta, live_retrieved, clock, pick_log, summary)
    if archive:
        summary['archived_inputs'] = {n: archive_input(Path(data_dir) / n) for n in sorted(inputs) if not inputs[n]['pinned'] and n.endswith('.csv')}
    summary['finished_at'] = iso(clock())
    summary['created'] = dict(summary['created'])
    append_record(ledger_path('runs.jsonl', ledger_dir), {k: v for k, v in summary.items() if k != 'games'} | {'game_count': len(summary['games']), 'inputs': inputs, 'schedule': sched_meta}, 'run_id')
    return summary


def _has_forecast(ledger_dir, label, game_id):
    return any(x['record']['game_id'] == game_id for x in read_ledger(ledger_path('forecasts_%s.jsonl' % label, ledger_dir)))


def _guards(ctx, g, by_team):
    """Refusals that precede any forecast: outcome rows present, stale history, unpublished roster."""
    teams = (g['home_team'], g['away_team'])
    if any(t in {x[0] for x in ctx.outcome_rows_present} for t in teams):
        return 'REFUSED_OUTCOME_DATA_PRESENT'
    wk = (int(g['season']), int(g['week']))
    for t in teams:
        prior = sorted(x for x in by_team[t] if x < wk)
        if prior and not ctx.history_complete(t, prior[-1]):
            return 'BLOCKED_HISTORY_INCOMPLETE'
        if ctx.roster_rows(t) < MIN_ROSTER_ROWS:
            return 'BLOCKED_ROSTER_NOT_PUBLISHED'
    return None


def _write_game(ctx, g, ko, label, ledger_dir, run_id, inputs, sched_meta, live_retrieved, clock, pick_log, summary):
    cutoff = window(ko, label)[1]
    if datetime.fromisoformat(live_retrieved) >= cutoff:
        raise RefusedError('input retrieval finished after the cutoff')
    for team, opp, ha in ((g['home_team'], g['away_team'], 'home'), (g['away_team'], g['home_team'], 'away')):
        qb = QB.certify_for_forecast(team, g['game_id'], cutoff, label)
        append_record(ledger_path('game_status.jsonl', ledger_dir), {'event_id': f"{g['game_id']}|{team}|{label}|QB", 'event': 'ABSTAIN_QB_STATE_UNCERTIFIED' if qb['status'] != 'CERTIFIED' else 'QB_STATE_CERTIFIED', 'game_id': g['game_id'], 'team': team,
                                                                           'cutoff_type': label, 'qb_status': qb['status'], 'reason': qb['reason'], 'observed_at': iso(clock()), 'engine_version': ENGINE_VERSION}, 'event_id')
        if qb['status'] != 'CERTIFIED':
            summary['qb_abstentions'] += 1
        forecasts, comparators, s = build_forecasts(ctx, g, team, opp, ha)
        for pid, f in forecasts.items():
            generated = clock()
            assert_before_cutoff(generated, ko, label)
            fid = forecast_id(label, g['game_id'], pid)
            rec = {'forecast_id': fid, 'run_id': run_id, 'engine_version': ENGINE_VERSION, 'git_sha': git_sha(), 'generated_at': iso(generated), 'cutoff_type': label, 'cutoff_time': iso(cutoff), 'kickoff_time': iso(ko),
                   'window_opens': iso(window(ko, label)[0]), **f, 'source_hashes': {k: v['sha256'] for k, v in sorted(inputs.items())}, 'inputs_retrieved_at': live_retrieved, 'schedule_source': {k: sched_meta[k] for k in ('url', 'allowlisted_rows_sha256', 'retrieved_at')}}
            line = append_record(ledger_path('forecasts_%s.jsonl' % label, ledger_dir), rec, 'forecast_id')
            if line is None:
                continue
            summary['created'][label] += 1
            h = comparators[pid]
            v1 = v1_projection(pick_log, int(g['season']), int(g['week']), pid, generated) if pick_log else {'status': 'V1_LOG_NOT_PROVIDED'}
            append_record(ledger_path('comparators.jsonl', ledger_dir), {'forecast_id': fid, 'forecast_record_sha256': line['record_sha256'], 'generated_at': iso(generated), 'cutoff_time': iso(cutoff), 'human': h['human'], 'simple_baseline': h['simple_baseline'], 'v1': v1,
                                                                         'note': 'comparators are frozen before the cutoff and never feed the V2 forecast'}, 'forecast_id')


# --------------------------------------------------------------------------- append-only verification against a git base
def verify_append_only(base_ref, ledger_dir=None):
    """Every committed ledger line at base_ref must be a byte-identical prefix of the current file."""
    problems = []
    rel = Path(ledger_dir or LEDGER).relative_to(ROOT)
    for name in LEDGER_FILES:
        r = subprocess.run(['git', 'show', f'{base_ref}:{rel}/{name}'], cwd=ROOT, capture_output=True)
        if r.returncode != 0:
            continue
        cur = ledger_path(name, ledger_dir)
        data = cur.read_bytes() if cur.exists() else b''
        if not data.startswith(r.stdout):
            problems.append(name)
    return problems


def build_manifest(ledger_dir=None):
    """Deterministic manifest of the ledgers (line counts, last chain hash, counts by event) from the ledgers themselves."""
    ledger_dir = Path(ledger_dir or LEDGER)
    out = {'schema': 'nfl-v2-phase2a-forecast-manifest-v1', 'engine_version': ENGINE_VERSION, 'production_promotion': False, 'ledgers': {}, 'counts': {}}
    for name in LEDGER_FILES:
        p = ledger_path(name, ledger_dir)
        n, last = verify_chain(p)
        out['ledgers'][name] = {'lines': n, 'last_line_sha256': last, 'file_sha256': sha_file(p) if p.exists() else None}
    status = [x['record'] for x in read_ledger(ledger_path('game_status.jsonl', ledger_dir))]
    forecasts = {lab: [x['record'] for x in read_ledger(ledger_path('forecasts_%s.jsonl' % lab, ledger_dir))] for lab in ('T24', 'T90')}
    out['counts'] = {'forecasts_T24': len(forecasts['T24']), 'forecasts_T90': len(forecasts['T90']), 'games_with_T24': len({r['game_id'] for r in forecasts['T24']}), 'games_with_T90': len({r['game_id'] for r in forecasts['T90']}),
                     'MISSED_T24_CUTOFF': sum(1 for e in status if e['event'] == 'MISSED_T24_CUTOFF'), 'MISSED_T90_CUTOFF': sum(1 for e in status if e['event'] == 'MISSED_T90_CUTOFF'),
                     'QB_ABSTENTIONS': sum(1 for e in status if e['event'].startswith('ABSTAIN_QB')), 'BLOCKED_OR_REFUSED_GAMES': sum(1 for e in status if e['event'].startswith(('BLOCKED', 'REFUSED'))),
                     'runs': len(read_ledger(ledger_path('runs.jsonl', ledger_dir)))}
    out['run_ids'] = [x['record']['run_id'] for x in read_ledger(ledger_path('runs.jsonl', ledger_dir))]
    out['forecast_hashes'] = {lab: [{'forecast_id': x['record']['forecast_id'], 'record_sha256': x['record_sha256']} for x in read_ledger(ledger_path('forecasts_%s.jsonl' % lab, ledger_dir))] for lab in ('T24', 'T90')}
    out['code_sha256'] = code_sha256()
    return out


def build_lock():
    proto = ART / PROTOCOL
    return {'schema': 'nfl-v2-phase2a-engine-lock-v1', 'engine_version': ENGINE_VERSION, 'protocol_sha256': sha_file(proto), 'code_sha256': {n: sha_file(ROOT / n) for n in CODE_FILES}, 'frozen_upstream_code_sha256': {n: sha_file(ROOT / n) for n in FROZEN_UPSTREAM_CODE},
            'frozen_artifact_sha256': {n: sha_file(ART / n) for n in FROZEN_ARTIFACTS}, 'frozen_configs': {'phase1b_selected': 'see phase1b_opportunity_snapshot.json outcomes.<head>.selected_*_config', 'phase1d_selected': 'see phase1d_role_allocation_snapshot.json outcomes.<head>.selected_config', 'phase1g_selected': read_json(ART / 'phase1g_receiving_mechanics_snapshot.json')['selected_config']},
            'thresholds': THRESHOLD, 'windows_hours': {'T24': [48, 24], 'T90': [3, 1.5]}, 'min_roster_rows': MIN_ROSTER_ROWS, 'no_model_fitting': True, 'no_monte_carlo': True, 'no_sportsbook_input': True, 'production_promotion': False,
            'rule': 'No engine change during the evaluation window. Any change creates a NEW engine_version and a new ledger family; existing forecasts are never rewritten.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('due',), default='due')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--ledger-dir', default=None)
    ap.add_argument('--no-pbp', action='store_true')
    ap.add_argument('--pick-log', default=str(ROOT / 'docs/nfl_picks_log.jsonl'))
    ap.add_argument('--game', action='append', default=None, help='restrict to game ids')
    ap.add_argument('--write-manifest', action='store_true', help='rewrite phase2a_forecast_manifest.json from the ledgers (no forecasting)')
    a = ap.parse_args()
    if a.write_manifest:
        (ART / 'phase2a_forecast_manifest.json').write_text(json.dumps(rnd(build_manifest()), indent=2, sort_keys=True) + '\n')
        return
    s = run_due(a.data_dir, a.ledger_dir, with_pbp=not a.no_pbp, pick_log=a.pick_log, game_filter=set(a.game) if a.game else None)
    if a.ledger_dir is None:
        (ART / 'phase2a_forecast_manifest.json').write_text(json.dumps(rnd(build_manifest()), indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in s.items() if k != 'games'}, indent=2, sort_keys=True))
    for g in s['games']:
        print(g['game_id'], g['cutoff_type'], g['status'])


if __name__ == '__main__':
    main()

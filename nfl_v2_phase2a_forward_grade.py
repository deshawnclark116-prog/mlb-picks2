#!/usr/bin/env python3
"""Phase2A-FWD forward grader (research only). Grades ONLY games that are final with official stats present, from the append-only ledgers.

Verifies the hash chain and the pre-cutoff proof of every forecast before grading it, keeps RAW and CLEAN ledgers, grades T24 and T90 separately,
pairs them, and compares V2 against the frozen competent human, the simple baseline and V1 on IDENTICAL player-games. It never changes a forecast.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import statistics
import subprocess

from nfl_v2_phase2a_ledger import read_ledger, rnd, verify_chain

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
LEDGER = ART / 'phase2a_ledger'
UTC = timezone.utc
CENSORS = 'phase2a_censors.json'
GAMES_RESULT_FIELDS = ('game_id', 'season', 'week', 'gameday', 'gametime', 'away_team', 'home_team', 'result')
HEADS = {  # head -> (value in the forecast record, stat column, comparator key, tolerances within, misses over)
    'targets': (('receiving', 'expected_targets'), 'targets', 'targets', (1, 2, 3), (4, 6)),
    'receptions': (('receiving', 'final_receptions'), 'receptions', 'receptions', (1, 2), (3,)),
    'receiving_yards': (('receiving', 'final_receiving_yards'), 'receiving_yards', 'receiving_yards', (10, 20, 25), (40, 60, 80)),
    'carries': (('rushing', 'expected_carries'), 'carries', 'carries', (2, 4, 6), (8, 12)),
    'rushing_yards': (('rushing', 'final_rushing_yards'), 'rushing_yards', 'rushing_yards', (10, 20, 25), (30, 50, 75)),
}
FINAL_AFTER = timedelta(hours=4)


def parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


# --------------------------------------------------------------------------- metrics
def metrics(signed, within, over):
    """signed = projection - actual."""
    if not signed:
        return {'n': 0}
    ae = [abs(x) for x in signed]
    n = len(ae)
    out = {'n': n, 'mae': sum(ae) / n, 'bias': sum(signed) / n, 'median_ae': statistics.median(ae)}
    for t in within:
        out[f'within_{t}'] = sum(1 for e in ae if e <= t) / n
    for t in over:
        out[f'miss_gt_{t}'] = sum(1 for e in ae if e > t) / n
    return out


def head_value(rec, head):
    (blk, key) = HEADS[head][0]
    b = rec.get(blk)
    return None if not b else b.get(key)


# --------------------------------------------------------------------------- proof of existence before cutoff
def forecast_proof(rec):
    """Every graded forecast must prove generated_at < cutoff_time <= kickoff_time and a consistent cutoff definition."""
    gen, cut, ko = parse_ts(rec['generated_at']), parse_ts(rec['cutoff_time']), parse_ts(rec['kickoff_time'])
    lead = {'T24': timedelta(hours=24), 'T90': timedelta(minutes=90)}[rec['cutoff_type']]
    opens = {'T24': timedelta(hours=48), 'T90': timedelta(hours=3)}[rec['cutoff_type']]
    problems = []
    if not (gen < cut <= ko):
        problems.append('GENERATED_NOT_BEFORE_CUTOFF')
    if cut != ko - lead:
        problems.append('CUTOFF_DEFINITION_MISMATCH')
    if gen < ko - opens:
        problems.append('GENERATED_BEFORE_WINDOW_OPENED')
    if parse_ts(rec['inputs_retrieved_at']) >= cut:
        problems.append('INPUTS_RETRIEVED_AFTER_CUTOFF')
    return problems


def git_first_commit_times(path, root=ROOT):
    """{line_count_threshold: commit_time_iso} so each ledger line can be tied to the first commit that contained it."""
    rel = Path(path).resolve().relative_to(root)
    r = subprocess.run(['git', 'log', '--reverse', '--format=%H %cI', '--', str(rel)], cwd=root, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        h, t = line.split(' ', 1)
        blob = subprocess.run(['git', 'show', f'{h}:{rel}'], cwd=root, capture_output=True)
        if blob.returncode == 0:
            out.append((len(blob.stdout.splitlines()), t))
    return out


def commit_time_for(seq, commits):
    for n, t in commits:
        if n >= seq:
            return t
    return None


# --------------------------------------------------------------------------- data
def load_final_games(schedule_text):
    """Games the schedule marks final. `result` is read here only to know that a game is complete; it is never a feature."""
    out = {}
    for r in csv.DictReader(io.StringIO(schedule_text)):
        if r['game_type'] != 'REG' or not r.get('result') or not r['gametime']:
            continue
        out[r['game_id']] = {k: r[k] for k in GAMES_RESULT_FIELDS}
    return out


def load_actuals(stats_path):
    stats, teams = {}, set()
    with open(stats_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get('season_type') or 'REG') != 'REG':
                continue
            key = (int(r['season']), int(float(r['week'])), r['team'])
            teams.add(key)

            def num(c):
                try:
                    return float(r.get(c) or 0.0)
                except ValueError:
                    return 0.0
            stats[key + (r['player_id'],)] = {c: num(c) for c in ('targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards')}
    return stats, teams


def load_censors(path=None):
    path = Path(path or ART / CENSORS)
    if not path.exists():
        return {}
    return {(e['game_id'], e['player_id']): e for e in json.loads(path.read_text()).get('events', []) if e.get('status') == 'CENSORED_IN_GAME'}


# --------------------------------------------------------------------------- grading
def grade(ledger_dir, stats_path, schedule_text, censor_path=None, now=None, git_provenance=False):
    ledger_dir = Path(ledger_dir)
    now = now or datetime.now(UTC)
    chains = {}
    for name in ('forecasts_T24.jsonl', 'forecasts_T90.jsonl', 'comparators.jsonl', 'game_status.jsonl', 'runs.jsonl'):
        chains[name] = verify_chain(ledger_dir / name)[0]
    finals = load_final_games(schedule_text)
    stats, stat_teams = load_actuals(stats_path)
    censors = load_censors(censor_path)
    cmp_by_id = {x['record']['forecast_id']: x['record'] for x in read_ledger(ledger_dir / 'comparators.jsonl')}
    commits = {n: git_first_commit_times(ledger_dir / n) for n in ('forecasts_T24.jsonl', 'forecasts_T90.jsonl')} if git_provenance else {}
    graded, excluded = [], []
    for fname in ('forecasts_T24.jsonl', 'forecasts_T90.jsonl'):
        for line in read_ledger(ledger_dir / fname):
            rec = line['record']
            gid = rec['game_id']
            problems = forecast_proof(rec)
            cm = cmp_by_id.get(rec['forecast_id'])
            if cm is None or parse_ts(cm['generated_at']) >= parse_ts(rec['cutoff_time']):
                problems.append('COMPARATOR_ROW_MISSING_OR_LATE')
            ct = commit_time_for(line['seq'], commits.get(fname, [])) if git_provenance else None
            if git_provenance and (ct is None or parse_ts(ct) >= parse_ts(rec['kickoff_time'])):
                problems.append('PROVENANCE_NOT_COMMITTED_BEFORE_KICKOFF')
            if problems:
                excluded.append({'forecast_id': rec['forecast_id'], 'game_id': gid, 'reason': problems})
                continue
            g = finals.get(gid)
            if g is None or now < parse_ts(rec['kickoff_time']) + FINAL_AFTER:
                continue                                                     # not final yet: never graded
            if (rec['season'], rec['week'], rec['team']) not in stat_teams or (rec['season'], rec['week'], rec['opponent']) not in stat_teams:
                continue                                                     # official stats for both teams not present yet
            key = (rec['season'], rec['week'], rec['team'], rec['player_id'])
            act = stats.get(key)
            did_not_play = act is None
            act = act or dict.fromkeys(('targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards'), 0.0)
            cens = censors.get((gid, rec['player_id']))
            row = {'forecast_id': rec['forecast_id'], 'cutoff_type': rec['cutoff_type'], 'game_id': gid, 'season': rec['season'], 'week': rec['week'], 'team': rec['team'], 'position': rec['position'], 'player_id': rec['player_id'],
                   'role_stable': all(c == 'HIGH' for c in rec['role_confidence'].values()), 'did_not_play_no_stat_row': did_not_play, 'ledger': 'CLEAN' if not cens else 'CENSORED_IN_GAME', 'censor_reason': cens['reason'] if cens else None, 'heads': {}}
            for head, (_loc, col, ckey, *_t) in HEADS.items():
                v = head_value(rec, head)
                if v is None:
                    continue
                human = (cm.get('human') or {}).get(ckey)
                base = (cm.get('simple_baseline') or {}).get(ckey)
                v1 = (cm.get('v1') or {}).get(ckey)
                row['heads'][head] = {'projection': v, 'actual': act[col], 'human': human, 'simple_baseline': base, 'v1': v1}
            graded.append(row)
    return graded, excluded, chains


def cells(rows, head, keyfn=lambda r: 'ALL'):
    groups = defaultdict(list)
    for r in rows:
        if head in r['heads']:
            groups[keyfn(r)].append(r)
    _loc, _col, _ck, within, over = HEADS[head]
    return {k: metrics([x['heads'][head]['projection'] - x['heads'][head]['actual'] for x in v], within, over) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def comparator_table(rows, head, comparator):
    """V2 vs a comparator on the player-games where that comparator exists (identical rows)."""
    both = [r for r in rows if head in r['heads'] and r['heads'][head].get(comparator) is not None]
    if not both:
        return {'n': 0}
    v2 = sum(abs(r['heads'][head]['projection'] - r['heads'][head]['actual']) for r in both) / len(both)
    c = sum(abs(r['heads'][head][comparator] - r['heads'][head]['actual']) for r in both) / len(both)
    return {'n': len(both), 'v2_mae': v2, 'comparator_mae': c, 'difference_v2_minus_comparator': v2 - c}


def paired(rows, head):
    by = defaultdict(dict)
    for r in rows:
        if head in r['heads']:
            by[(r['game_id'], r['player_id'])][r['cutoff_type']] = r['heads'][head]
    deltas = []
    for k, v in by.items():
        if 'T24' in v and 'T90' in v:
            e24 = abs(v['T24']['projection'] - v['T24']['actual'])
            e90 = abs(v['T90']['projection'] - v['T90']['actual'])
            deltas.append({'game_id': k[0], 'player_id': k[1], 'T24_abs_error': e24, 'T90_abs_error': e90, 'delta_T90_minus_T24': e90 - e24})
    n = len(deltas)
    return {'pairs': n, 'T90_improved': sum(1 for d in deltas if d['delta_T90_minus_T24'] < -1e-9), 'T90_worsened': sum(1 for d in deltas if d['delta_T90_minus_T24'] > 1e-9),
            'unchanged': sum(1 for d in deltas if abs(d['delta_T90_minus_T24']) <= 1e-9), 'mean_delta': (sum(d['delta_T90_minus_T24'] for d in deltas) / n) if n else None, 'rows': deltas}


def report(graded, excluded, chains, ledger_dir=None):
    ledger_dir = Path(ledger_dir or LEDGER)
    status = read_ledger(ledger_dir / 'game_status.jsonl')
    missed = [x['record'] for x in status if x['record']['event'].startswith('MISSED_')]
    abst = [x['record'] for x in status if x['record']['event'].startswith('ABSTAIN_QB')]
    out = {'schema': 'nfl-v2-phase2a-forward-report-v1', 'status': 'NO_GRADED_GAMES_YET' if not graded else 'GRADED_PARTIAL_OR_COMPLETE', 'ledger_lines': chains, 'graded_player_games': len(graded), 'excluded_failed_proof': excluded,
           'missed_cutoffs': {'MISSED_T24_CUTOFF': sum(1 for m in missed if m['event'] == 'MISSED_T24_CUTOFF'), 'MISSED_T90_CUTOFF': sum(1 for m in missed if m['event'] == 'MISSED_T90_CUTOFF')},
           'qb_abstentions': {'count': len(abst), 'official_qb_forecasts': 0, 'reason_counts': {r: sum(1 for a in abst if a['reason'] == r) for r in sorted({a['reason'] for a in abst})}}, 'heads': {}}
    for head in HEADS:
        h = {}
        for ledger_name, sel in (('RAW', lambda r: True), ('CLEAN', lambda r: r['ledger'] == 'CLEAN')):
            sub = [r for r in graded if sel(r)]
            h[ledger_name] = {'by_cutoff': cells(sub, head, lambda r: r['cutoff_type']), 'by_position': cells(sub, head, lambda r: (r['cutoff_type'], r['position'])), 'by_team': cells(sub, head, lambda r: (r['cutoff_type'], r['team'])),
                              'by_week': cells(sub, head, lambda r: (r['cutoff_type'], r['week'])), 'by_role_stability': cells(sub, head, lambda r: (r['cutoff_type'], 'STABLE_HIGH_ROLE' if r['role_stable'] else 'UNCERTAIN_ROLE')),
                              'versus': {c: {ct: comparator_table([r for r in sub if r['cutoff_type'] == ct], head, c) for ct in ('T24', 'T90')} for c in ('human', 'simple_baseline', 'v1')}}
        h['paired_T24_T90_clean'] = paired([r for r in graded if r['ledger'] == 'CLEAN'], head)
        out['heads'][head] = h
    return rnd(_stringify(out))


def _stringify(v):
    if isinstance(v, dict):
        return {('|'.join(str(i) for i in k) if isinstance(k, tuple) else str(k)): _stringify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_stringify(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('grade',), default='grade')
    ap.add_argument('--stats', required=True)
    ap.add_argument('--schedule', required=True, help='games csv text file (result column read only to detect finality)')
    ap.add_argument('--ledger-dir', default=str(LEDGER))
    ap.add_argument('--out', default=str(ART / 'phase2a_forward_report.json'))
    ap.add_argument('--git-provenance', action='store_true')
    a = ap.parse_args()
    graded, excluded, chains = grade(a.ledger_dir, a.stats, Path(a.schedule).read_text(), git_provenance=a.git_provenance)
    rep = report(graded, excluded, chains, a.ledger_dir)
    Path(a.out).write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')
    print(rep['status'], rep['graded_player_games'], 'excluded', len(excluded))


if __name__ == '__main__':
    main()

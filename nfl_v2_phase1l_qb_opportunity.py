#!/usr/bin/env python3
"""Phase1L QB opportunity: frozen source-gated research, not serving code.

The historical starter source gate failed. This module freezes an unscored
hand-calculable benchmark and a raw uncertain-state ledger, NOT a replacement.
No fitting/scoring path may run by editing a source-gate boolean. Data acceptance
and any later experiment require a new preregistration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path

import nfl_v2_phase1l_qb_sources as S

ART = S.ART
STATES = ('STABLE_EXPECTED_STARTER', 'EXPECTED_STARTER_WITH_RECENT_CHANGE',
          'QB_CHANGE_OR_REPLACEMENT', 'MULTI_QB_OR_PACKAGE',
          'UNCERTAIN_STARTER', 'INSUFFICIENT_PREGAME_EVIDENCE')
FAMILIES = ('A_recent_qb_workload', 'B_team_volume', 'C_opponent_environment',
            'D_qb_conversion', 'E_starter_role', 'F_script_scenarios',
            'G_coherent_chain')
BLOCKED = 'BLOCKED_STARTER_STATE_DATA'
NOT_RUN = 'NOT_RUN_SOURCE_GATE_FAILED'


def cutoff(game):
    return datetime.fromisoformat(game['game_date']).replace(tzinfo=timezone.utc) - timedelta(hours=24)


def legal_prior(source, target):
    earlier = (int(source['season']), int(source['week'])) < (int(target['season']), int(target['week']))
    completed_bound = datetime.fromisoformat(source['game_date']).replace(tzinfo=timezone.utc) + timedelta(hours=48)
    return earlier and completed_bound <= cutoff(target)


def prior_team_games(data, target, team, window=8):
    return sorted([g for g in data['games'].values()
                   if team in (g['home_team'], g['away_team']) and legal_prior(g, target)],
                  key=lambda g: (g['game_date'], g['game_id']))[-window:]


def continuity(data, histories, team, pid):
    last = histories[-2:]
    shares = [data['qbs'].get((g['game_id'], team, pid), {}).get('dropbacks', 0) /
              max(data['teams'].get((g['game_id'], team), {}).get('dropbacks', 0), 1) for g in last]
    attempts = [data['official'].get((g['game_id'], team, pid), {}).get('attempts', 0) for g in last]
    return len(last) == 2 and all(s >= .80 for s in shares) and all(a >= 20 for a in attempts)


def classify_state(has_history, continuity_hint=False, evidence=None):
    """No accepted archived state receipts in this phase; hints never certify.

    Real data acceptance is intentionally absent. Synthetic tests can establish
    that postgame/depth-rank inputs never lift this frozen source gate.
    """
    del continuity_hint, evidence
    return 'UNCERTAIN_STARTER' if has_history else 'INSUFFICIENT_PREGAME_EVIDENCE'


def chain(plays, dropback_rate, qb_share, conversion):
    """Pure arithmetic contract for a future source-certified opportunity row."""
    if not math.isfinite(plays) or plays < 0:
        raise ValueError('Invalid plays')
    for value in (dropback_rate, qb_share, conversion):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid opportunity probability/share')
    dropbacks = plays * dropback_rate
    return {'expected_team_plays': plays, 'expected_dropbacks': dropbacks,
            'expected_pass_dropback_rate': dropback_rate,
            'expected_qb_participation_share': qb_share,
            'expected_attempts_per_dropback': conversion,
            'final_projected_attempts': dropbacks * qb_share * conversion}


def hand_formula(own3, qb5, league128, expected_single_starter):
    """Frozen benchmark arithmetic, no optimization and no identity guessing.

    Inputs are already legal prior summaries. Runner never executes this on
    real seasons while starter gate is blocked. Synthetic fixtures test math.
    """
    if expected_single_starter is not True:
        return None
    if not own3 or not league128:
        return None
    lg_plays = sum(r['plays'] for r in league128) / len(league128)
    lg_play_sum = sum(r['plays'] for r in league128)
    lg_drop = sum(r['dropbacks'] for r in league128)
    if lg_play_sum <= 0 or lg_drop <= 0:
        return None
    lg_rate = lg_drop / lg_play_sum
    lg_conversion = sum(r['attempts'] for r in league128) / lg_drop
    own3 = own3[-3:]
    qb5 = qb5[-5:]
    plays = (sum(r['plays'] for r in own3) + 2 * lg_plays) / (len(own3) + 2)
    rate = (sum(r['dropbacks'] for r in own3) + 100 * lg_rate) / (sum(r['plays'] for r in own3) + 100)
    share = (sum(r['qb_dropbacks'] for r in own3) + 20) / (sum(r['dropbacks'] for r in own3) + 20)
    conversion = (sum(r['attempts'] for r in qb5) + 50 * lg_conversion) / (sum(r['dropbacks'] for r in qb5) + 50)
    return chain(plays, rate, share, conversion)


def benchmark():
    protocol = S.read_json(ART / 'phase1l_qb_protocol.json')
    return {'schema': 'nfl-v2-phase1l-competent-human-volume-v1',
            'status': 'FROZEN_DEFINITION_UNSCORED_SOURCE_BLOCKED',
            'formula': 'plays × dropback rate × expected QB share × attempts/dropback',
            'rules': protocol['competent_human'], 'required_comparators': protocol['comparators'],
            'starter_missing_behavior': 'ABSTAIN; no prior-passer substitution',
            'parameter_search': False, 'monte_carlo': False, 'sportsbook_inputs': False,
            'performance': None, 'protocol_sha256': S.sha(ART / 'phase1l_qb_protocol.json')}


def blocked_lock(audit):
    if audit['source_gate']['passed'] or audit['source_gate']['verdict'] != BLOCKED:
        raise ValueError('NEW_PROTOCOL_REQUIRED; this frozen phase cannot reopen by changing audit flags')
    return {'schema': 'nfl-v2-phase1l-development-lock-v1', 'phase': 'PHASE1L-QB',
            'status': 'FROZEN_BEFORE_FITTING_SOURCE_GATE_FAILED', 'verdict': BLOCKED,
            'protocol_sha256': S.sha(ART / 'phase1l_qb_protocol.json'),
            'audit_sha256': S.sha(ART / 'phase1l_qb_source_audit.json'),
            'selected': None, 'fitting_performed': False, 'performance_accessed': [],
            'families': {name: {'status': BLOCKED, 'metrics': None,
                                'reason': 'Primary expected-starter population unavailable; not a test of this family\'s predictive value'} for name in FAMILIES},
            'development_2024': {'status': NOT_RUN, 'attempt_metrics': None},
            'validation_2025': NOT_RUN, 'diagnostic_2026_w1_4': NOT_RUN,
            'refit': NOT_RUN, 'passing_yards': 'NOT_RUN_INDEPENDENT_ATTEMPT_GATE',
            'no_rescue': True}


def receipts(data):
    """2024 prior candidate/state audit rows, not scored model forecasts.

    Target official counts are kept under evaluation_truth only. No target
    participant lookup enters candidate construction or continuity hints.
    """
    for game in sorted(data['games'].values(), key=lambda g: (g['game_date'], g['game_id'])):
        if int(game['season']) != 2024:
            continue
        for team, opponent in [(game['away_team'], game['home_team']), (game['home_team'], game['away_team'])]:
            histories = prior_team_games(data, game, team)
            ids = sorted({pid for g in histories for gid, tm, pid in data['qbs']
                          if gid == g['game_id'] and tm == team and (gid, tm, pid) in data['official']})
            # Previous games' position tags are legal, target position is not.
            for pid in ids or [None]:
                historical = [data['official'][(g['game_id'], team, pid)] for g in histories
                              if (g['game_id'], team, pid) in data['official']]
                state = classify_state(bool(historical))
                hint = bool(pid and continuity(data, histories, team, pid))
                actual = data['official'].get((game['game_id'], team, pid))
                tm = data['teams'].get((game['game_id'], team), {})
                qb = data['qbs'].get((game['game_id'], team, pid), {})
                # Team stats coverage validated by aggregate; absent player row
                # is zero only for evaluation, not proof of pregame unavailability.
                team_official = [r for (gid, t, _), r in data['official'].items() if gid == game['game_id'] and t == team]
                if not team_official:
                    raise ValueError('No official QB coverage for target team')
                yield {
                    'schema': 'nfl-v2-phase1l-source-blocked-raw-receipt-v1',
                    'receipt_kind': 'SOURCE_STATE_AUDIT_NOT_A_PREDICTIVE_FORECAST',
                    'game_id': game['game_id'], 'season': 2024, 'week': int(game['week']),
                    'player_id': pid, 'player': historical[-1]['player_display_name'] if historical else None,
                    'team': team, 'opponent': opponent, 'home': team == game['home_team'],
                    'information_cutoff_utc': cutoff(game).isoformat(),
                    'starter_state': state, 'starter_confidence': None,
                    'population': 'UNCERTAIN' if pid else 'EMERGENCY_OR_CENSORED',
                    'pregame_continuity_hint_not_certified': hint,
                    'prior_game_ids': [g['game_id'] for g in histories],
                    'current_team_membership': 'UNKNOWN_AT_CUTOFF',
                    'prior_current_team_qb_games': len(historical),
                    'expected_team_plays': None, 'expected_dropbacks': None,
                    'expected_pass_dropback_rate': None, 'expected_qb_participation_share': None,
                    'expected_attempts_per_dropback': None, 'final_projected_attempts': None,
                    'incumbent_yards_per_attempt': None, 'final_passing_yard_projection': None,
                    'attempt_error': None, 'error_decomposition': NOT_RUN,
                    'uncertainty_reasons': ['NO_ACCEPTED_ASOF_STARTER_HEALTH_ARCHIVE',
                                           'CURRENT_MEMBERSHIP_OR_REPLACEMENT_UNRESOLVED',
                                           'PRIOR_USAGE_IS_NOT_CURRENT_STARTER_STATE'],
                    'censor_status': 'UNKNOWN_NOT_CENSORED_NO_ACCEPTED_EXIT_LABEL',
                    'evaluation_truth': {
                        'actual_attempts': None if pid is None else (actual['attempts'] if actual else 0),
                        'actual_passing_yards': None if pid is None else (actual['passing_yards'] if actual else 0),
                        'actual_team_plays': tm.get('plays'), 'actual_team_dropbacks': tm.get('dropbacks'),
                        'actual_qb_dropbacks': None if pid is None else qb.get('dropbacks', 0),
                        'actual_share': None if pid is None else qb.get('dropbacks', 0) / max(tm.get('dropbacks', 0), 1),
                        'official_starter': 'NOT_LOADED', 'use': 'RAW_LABEL_ONLY_NOT_FOR_SELECTION_OR_SCORING',
                    },
                    'source_versions': manifest_receipt(),
                }


def manifest_receipt():
    return {y: {kind: source['sha256'] for kind, source in sources.items()} for y, sources in S.manifest().items()}


def write_receipts(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable gzip bytes independent of wall clock and output filename.
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())


def results(audit, lock, rows):
    if lock != blocked_lock(audit):
        raise ValueError('Frozen development lock changed')
    counts = Counter(r['starter_state'] for r in rows)
    hints = sum(r['pregame_continuity_hint_not_certified'] for r in rows)
    return {'schema': 'nfl-v2-phase1l-qb-results-v1', 'phase': 'PHASE1L-QB',
            'verdict': BLOCKED, 'selected': None,
            'families': lock['families'],
            'periods': {label: {'status': NOT_RUN, 'metrics': None} for label in
                        ('development_2024_w9_18', 'validation_2025', 'diagnostic_2026_w1_4')},
            'primary_population': {'certifiable_rows': 0, 'headline_metrics': None},
            'raw_audit_ledger': {'season': 2024, 'rows': len(rows), 'states': dict(sorted(counts.items())),
                                 'prior_continuity_hints': hints,
                                 'forecasted_rows': 0, 'scored_rows': 0,
                                 'candidate_rule': 'All prior-known QB identities last8 legal team-games; absent identity represented by team-level insufficient row'},
            'stable_vs_change_vs_uncertain': {'status': NOT_RUN, 'metrics': None},
            'full_passing_yards': {'status': 'NOT_RUN_INDEPENDENT_ATTEMPT_GATE', 'metrics': None},
            'oracle_decomposition': {'status': 'NOT_RUN_NO_NORMAL_WORKLOAD_EVALUATION', 'metrics': None},
            'largest_failure_source': 'Historical pregame expected QB assignment/health is unobservable in the accepted corpus. New component-error ranking cannot be established without evaluation.',
            'newly_audited_semantic_limitations': 'Spikes have qb_dropback=0; opportunity denominator adds spikes. Official QB attempts reconcile exactly in 2023/2024 after separating owners without official QB position labels. Non-QB/unclassified owners remain visible in source counts, never silently treated as QBs.',
            'benchmark': benchmark(), 'no_predictive_fitting': True,
            'new_2025_2026_performance_accessed': False, 'week5_plus_accessed': False,
            'receiving': 'FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION',
            'receiver_depth': 'SURVIVED_SIGNAL_NOT_PROMOTED',
            'rushing': 'REJECTED_EFFICIENCY_REPLACEMENT', 'no_rescue': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--stage', choices=['develop', 'confirm'], required=True)
    ap.add_argument('--audit', default=str(ART / 'phase1l_qb_source_audit.json'))
    ap.add_argument('--lock', default=str(ART / 'phase1l_development_lock.json'))
    ap.add_argument('--benchmark')
    ap.add_argument('--out')
    ap.add_argument('--receipts')
    args = ap.parse_args()
    audit = S.read_json(args.audit)
    # Reproduction/changed sources checked before a gate/lock can be trusted.
    if audit != S.audit(args.data_dir):
        raise ValueError('Source audit drift; no model or changed archive fallback')
    lock = blocked_lock(audit)
    if args.stage == 'develop':
        S.write_json(args.lock, lock)
        if args.benchmark:
            S.write_json(args.benchmark, benchmark())
    else:
        if not args.out or not args.receipts:
            raise ValueError('confirm requires --out and --receipts')
        rows = list(receipts(S.aggregate(args.data_dir)))
        report = results(audit, S.read_json(args.lock), rows)
        S.write_json(args.out, report)
        write_receipts(args.receipts, rows)
    print(json.dumps({'verdict': BLOCKED, 'stage': args.stage, 'no_fitting_or_scoring': True}))


if __name__ == '__main__':
    main()

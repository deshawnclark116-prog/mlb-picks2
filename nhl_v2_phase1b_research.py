"""Deterministic exposed-corpus component diagnostics; no clean forward claims.

Run develop first. Its lock is required by diagnose and cannot be rewritten.
Outputs never overwrite Phase0/Phase1A ledgers. No automatic network collection.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np

import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_model as B
from nhl_v2_phase1a_sog_forward import canon, sha_file, cutoff_of
from nhl_v2_phase1b_opportunity import History, fit_priors, component_forecast, total_attempt_component
from nhl_v2_phase1b_sources import write_json, join_adjudicated_attempts

ROOT = Path(__file__).resolve().parent
PROTOCOL = D.OUT / 'phase1b_opportunity_protocol.json'
CODE = ('nhl_v2_phase1b_opportunity.py', 'nhl_v2_phase1b_research.py', 'nhl_v2_phase1b_sources.py', 'nhl_v2_phase1b_snapshots.py', 'nhl_v2_phase1b_comparators.py', 'nhl_v2_phase1b_forward.py')


def inputs():
    games, rows = D.load_frozen()
    rows, _ = join_adjudicated_attempts(rows)
    history = History(games, rows, allow_completion_proxy=True)
    priors = fit_priors(rows)
    return games, rows, history, priors


def errors(pred, actual):
    p, y = np.asarray(pred, float), np.asarray(actual, float)
    if not len(y):
        return None
    e = p-y
    return {'n': len(y), 'mae': float(np.abs(e).mean()), 'bias': float(e.mean()), 'median_AE': float(np.median(np.abs(e)))}


def score_rows(records, baseline):
    out = {}
    for pop in ('FULL', 'PRIMARY_EXPECTED_PARTICIPANT', 'ACTUALLY_PLAYED'):
        rs = [r for r in records if pop == 'FULL' or (r['meaningful'] if pop == 'PRIMARY_EXPECTED_PARTICIPANT' else r['played'])]
        if not rs:
            out[pop] = {'n': 0}
            continue
        y = np.array([r['actual_sog'] for r in rs], int)
        gids = np.array([r['game_id'] for r in rs]); pids = np.array([r['player_id'] for r in rs])
        out[pop] = {'n': len(rs), 'participation_Brier': float(np.mean([(r['p_active']-r['played'])**2 for r in rs])),
                    'raw_participation_Brier': float(np.mean([(r['history_participation_unsmoothed']-r['played'])**2 for r in rs])),
                    'predicted_participation': float(np.mean([r['p_active'] for r in rs])), 'observed_participation': float(np.mean([r['played'] for r in rs])),
                    'SOG': {}}
        for name in ('B2', 'simple_mean', 'simple_participation_mean', 'transparent_analyst_mean', 'analyst_no_environment'):
            m = np.array([r[name] for r in rs], float)
            if name == 'B2':
                kind, params = 'nb2', {'mu': m, 'alpha': baseline['nb2']['alpha']}
            else:
                kind, params = 'poisson', {'mu': np.maximum(m, .000001)}
            sc, _ = B.summarize(kind, params, y, name, gids, pids, m, np.full((len(y),5), .5))
            # No Brier-skill claim against invented positional rates.
            for report in sc['thresholds'].values():
                report.pop('BSS', None)
                report.pop('brier_base_rate_comparator', None)
                report.pop('brier_skill', None)
                report.pop('brier_skill_score', None)
            sc.pop('brier_skill', None)
            for key in list(sc):
                if 'skill' in key:
                    sc.pop(key)
            out[pop]['SOG'][name] = {'central': sc['central'], 'CRPS_game_macro': sc['crps_macro_game'], 'NLL_game_macro': sc['nll_macro_game'],
                                    'interval_coverage': sc['interval_coverage'], 'thresholds': sc['thresholds']}
        played = [r for r in rs if r['played']]
        out[pop]['TOI_conditional_played'] = {v: {s: errors([r['variants'][v][s] for r in played], [r['actual_' + s] for r in played]) for s in ('total_toi','ev_toi','pp_toi','pk_toi')} for v in ('C0','C1','C2')}
        out[pop]['TOI_unconditional'] = errors([r['total_toi_unconditional'] for r in rs], [r['actual_total_toi'] for r in rs])
    return out


def replay(seasons, games, rows, history, priors, receipts_path):
    targets = {gid for gid in games if int(str(gid)[:4]) in seasons}
    # Incumbent remains untouched; these seasons occurred within its fit corpus.
    table, _ = D.build_rows(games, rows, target_gids=targets, labels=False, horizon_min=90)
    baseline = json.loads((D.OUT / 'phase1a_sog_models/engine_T90.json').read_text())
    mu = B.predict_mu(baseline, table)
    b2 = {(int(g),int(t),int(p)): float(m) for g,t,p,m in zip(table['game_id'],table['team_id'],table['player_id'],mu)}
    records, cov = [], Counter()
    history_hash=sha_file(D.OUT/'phase1a_sog_data_manifest.json')
    attempt_hash=sha_file(D.OUT/'phase1b_attempt_history/migration_manifest.json')
    for gid in sorted(targets, key=lambda x:(games[x]['game_start_utc'],x)):
        game = games[gid]
        cutoff = cutoff_of(game['game_start_utc'],'T90').isoformat()
        candidates, quarantine = history.candidates(game,cutoff)
        cov['quarantined_identity_rows'] += len(quarantine)
        cov['candidate_rows_before_pairing'] += len(candidates)
        by_team = {}
        for c in candidates:
            by_team.setdefault(c['team_id'],set()).add(c['player_id'])
        for team in (game['away_team_id'],game['home_team_id']):
            actual = history.by_game_team[(gid,team)]
            cov['actual_dressed'] += len(actual)
            cov['missing_dressed_candidate'] += len(set(actual)-by_team.get(team,set()))
        for c in candidates:
            key = gid,c['team_id'],c['player_id']
            if key not in b2:
                cov['unpaired_candidate_rows'] += 1
                continue
            rec = component_forecast(history,game,c,cutoff,priors,'C2')
            variants = {}
            for v in ('C0','C1','C2'):
                r = rec if v == 'C2' else component_forecast(history,game,c,cutoff,priors,v)
                variants[v] = {'total_toi':r['total_toi_conditional'],'ev_toi':r['ev_toi'],'pp_toi':r['pp_toi'],'pk_toi':r['pk_toi']}
            actual = history.by_game_team[(gid,c['team_id'])].get(c['player_id'])
            # Outcomes enter only after the complete immutable pregame receipt above.
            attempt = total_attempt_component(history,game,c,cutoff,priors) if int(str(gid)[:4]) == 2023 else {'status':'BLOCKED_CURRENT_ATTEMPT_LABELS'}
            rec.update(attempt_component=attempt,actual_attempts=(actual.get('shot_attempts') if actual else 0) if int(str(gid)[:4])==2023 else None,variants=variants, actual_sog=actual['sog'] if actual else 0,played=actual is not None,
                       actual_total_toi=actual['toi_sec'] if actual else 0,actual_ev_toi=actual['ev_toi_sec'] if actual else 0,
                       actual_pp_toi=actual['pp_toi_sec'] if actual else 0,actual_pk_toi=actual['sh_toi_sec'] if actual else 0,
                       B2=b2[key],attempt_prediction=None,conversion_prediction=None,attempt_status='BLOCKED_ATTEMPT_DATA',
                       independent_professional_forecast=None,horizon='T90',scheduled_start=game['game_start_utc'],
                       evidence='EXPOSED_RETROSPECTIVE_NOT_CONFIRMATION',
                       analyst_no_environment=component_forecast(history,game,c,cutoff,priors,'C2',environment=False)['transparent_analyst_mean'])
            conditional=rec['total_toi_conditional']*rec['skill_sog_per60']/3600
            rec['source_versions']={'official_history_manifest':history_hash,'adjudicated_attempt_migration_manifest':attempt_hash}
            rec['error_decomposition_diagnostic_only']={
                'availability':(rec['p_active']-int(rec['played']))*conditional,
                'TOI':int(rec['played'])*(rec['total_toi_conditional']-rec['actual_total_toi'])*rec['skill_sog_per60']/3600,
                'shot_rate_conversion_and_variance':rec['actual_total_toi']*rec['skill_sog_per60']/3600-rec['actual_sog'],
                'team_environment':rec['transparent_analyst_mean']-rec['p_active']*conditional}
            rec['oracle_diagnostic_only'] = {'actual_TOI_times_pregame_rate':rec['actual_total_toi']*rec['skill_sog_per60']/3600,
                                           'scope':'POSTGAME_ORACLE_DIAGNOSTIC_ONLY_NOT_SELECTION'}
            rec['receipt_sha256'] = hashlib.sha256(canon(rec).encode()).hexdigest()
            records.append(rec)
    content = ''.join(canon(r)+'\n' for r in records).encode()
    receipts_path.parent.mkdir(parents=True,exist_ok=True)
    receipts_path.write_bytes(gzip.compress(content,mtime=0))
    summary = score_rows(records,baseline)
    # Descriptive oracle receipt, never input to fit/selection.
    summary['oracle_diagnostic_only'] = {
        'label':'POSTGAME_ORACLE_DIAGNOSTIC_ONLY_NOT_SELECTION',
        'played_SOG_MAE_actual_TOI_times_pregame_rate':errors([r['oracle_diagnostic_only']['actual_TOI_times_pregame_rate'] for r in records if r['played']],[r['actual_sog'] for r in records if r['played']]),
        'mean_absolute_receipt_contribution':{k:float(np.mean([abs(r['error_decomposition_diagnostic_only'][k]) for r in records if r['meaningful']])) for k in ('availability','TOI','shot_rate_conversion_and_variance','team_environment')},
        'interpretation':'Algebraic attribution, not causal proof; contributions may cancel; no oracle was used in selection.',
        'independent_attempt_vs_conversion_decomposition':'BLOCKED_NO_ATTEMPT_LABELS',
    }
    summary['coverage'] = dict(cov)
    summary['receipt_sha256'] = hashlib.sha256(content).hexdigest()
    summary['receipt_file_sha256'] = sha_file(receipts_path)
    summary['B2_note'] = 'Locked B2 was fitted on 2018-2025: exposed/in-sample comparator, not untouched temporal validation; no incumbent claim from this comparison.'
    actual_attempt = [r for r in records if r['actual_attempts'] is not None and r['attempt_component']['status']=='EXPOSED_TOTAL_ATTEMPT_COMPONENT_ONLY' and r['meaningful']]
    summary['total_attempts_component'] = {'paired_primary_rows':len(actual_attempt), 'missing_attempt_labels':sum(r['actual_attempts'] is None for r in records)}
    if actual_attempt:
        summary['total_attempts_component']['metrics']={v:dict(errors([r['attempt_component'][v] for r in actual_attempt],[r['actual_attempts'] for r in actual_attempt]),miss_gt5=float(np.mean([abs(r['attempt_component'][v]-r['actual_attempts'])>5 for r in actual_attempt]))) for v in ('A0','A1','A2')}
        conv=[r for r in actual_attempt if r['actual_attempts']>0]
        summary['total_attempts_component']['conversion_rows']=len(conv)
        summary['total_attempts_component']['conversion']={v:errors([r['attempt_component'][v] for r in conv],[r['actual_sog']/r['actual_attempts'] for r in conv]) for v in ('R0','R1')}
    summary['independent_professional_forecasts'] = 0
    return summary


def develop(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output / 'phase1b_development_lock.json').exists():
        raise ValueError('development lock already exists; never overwrite')
    games, rows, hist, priors = inputs()
    summary = replay({2023}, games, rows, hist, priors, output / 'phase1b_development_receipts.jsonl.gz')
    met = summary['PRIMARY_EXPECTED_PARTICIPANT']['TOI_conditional_played']
    decisions = {'C0':{'status':'REFERENCE_NOT_PROMOTED'}}
    for v in ('C1','C2'):
        improvement = 1-met[v]['total_toi']['mae']/met['C0']['total_toi']['mae']
        guard = all(met[v][s]['mae'] <= met['C0'][s]['mae']*1.05 for s in ('ev_toi','pp_toi'))
        decisions[v] = {'status':'EXPOSED_SIGNAL_NOT_FORWARD_VALIDATED' if improvement >= .02 and guard else 'FROZEN_COMPONENT_REJECTED',
                        'relative_total_TOI_MAE_improvement':improvement,'EV_PP_guard_pass':guard}
    attempt=summary['total_attempts_component']
    if attempt['paired_primary_rows']:
        for v in ('A1','A2'):
            gain=1-attempt['metrics'][v]['mae']/attempt['metrics']['A0']['mae']
            guard=attempt['metrics'][v]['miss_gt5']<=attempt['metrics']['A0']['miss_gt5']+.01
            decisions[v]={'status':'EXPOSED_SIGNAL_NOT_FORWARD_VALIDATED' if gain>=.02 and guard else 'FROZEN_COMPONENT_REJECTED','relative_attempt_MAE_improvement':gain,'catastrophic_guard_pass':guard}
        gain=1-attempt['conversion']['R1']['mae']/attempt['conversion']['R0']['mae']
        decisions['R1']={'status':'EXPOSED_SIGNAL_NOT_FORWARD_VALIDATED' if gain>=.02 else 'FROZEN_COMPONENT_REJECTED','relative_conversion_MAE_improvement':gain}
    eligible = [v for v in ('C1','C2') if decisions[v]['status'] == 'EXPOSED_SIGNAL_NOT_FORWARD_VALIDATED']
    selected = min(eligible,key=lambda v:(met[v]['total_toi']['mae'],v)) if eligible else None
    lock = {'evidence':'EXPOSED_DEVELOPMENT_ONLY','attempt_addendum_sha256':sha_file(D.OUT/'phase1b_attempt_audit_addendum.json'),'protocol_sha256':sha_file(PROTOCOL),
            'code_hashes':{p:sha_file(ROOT/p) for p in CODE},'priors':priors,'component_decisions':decisions,
            'selected_TOI_component':selected,'full_engine':'BLOCKED_STRENGTH_ATTEMPT_DATA_LIVE_ACCESS_AND_AVAILABILITY_CERTIFICATION',
            'selected_final_SOG_model':None,'no_rescue':True,'development_receipts':summary['receipt_file_sha256']}
    write_json(output/'phase1b_development_results.json',summary)
    write_json(output/'phase1b_development_lock.json',lock)
    return lock


def diagnose(output):
    lock = json.loads((output/'phase1b_development_lock.json').read_text())
    if lock['attempt_addendum_sha256']!=sha_file(D.OUT/'phase1b_attempt_audit_addendum.json') or lock['protocol_sha256'] != sha_file(PROTOCOL) or any(sha_file(ROOT/p)!=h for p,h in lock['code_hashes'].items()):
        raise ValueError('frozen development specification changed')
    games,rows,hist,priors=inputs()
    if priors!=lock['priors']:
        raise ValueError('fitted priors changed')
    results = {'verdict':'BLOCKED_SOURCE_CERTIFICATION_AND_ATTEMPT_DATA','forward_forecasts':0,
               'phase1a_confirmation_preserved':'28 days /15000 meaningful T90 player-games',
               'development':json.loads((output/'phase1b_development_results.json').read_text()),'diagnostics':{}}
    for season in (2024,2025):
        results['diagnostics'][str(season)] = replay({season},games,rows,hist,priors,output/f'phase1b_{season}_diagnostic_receipts.jsonl.gz')
    write_json(output/'phase1b_results.json',results)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['develop','diagnose'])
    parser.add_argument('--output',type=Path,default=Path('research_out/nhl_phase1b'))
    args=parser.parse_args()
    result=develop(args.output) if args.mode=='develop' else diagnose(args.output)
    print(result.get('verdict',result.get('full_engine')))

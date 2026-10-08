#!/usr/bin/env python3
"""Phase1H permitted retrospective research. Never opens seasons beyond 2024.

Commands: select -> commit development lock -> diagnostics. Reproduce never
changes the selected specification. All new files go to an explicit output
folder; frozen Phase0 and production records are never written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import tennis_v2_data as D
import tennis_v2_baselines as B
import tennis_v2_incumbent as INC
import tennis_v2_metrics as M
from tennis_v2_phase1_guards import EvidenceBlocked
from tennis_v2_phase1_scoring import match_win_probability, hold_probability
from tennis_v2_phase1h_distributions import match_distribution
from tennis_v2_phase1h_inputs import P, Target, State, SURFACES, load_tour, rule_for, stats_channels, observed_hold
from tennis_v2_phase1h_components import point_projection, fit_stack, stack_predict, logit

BASE_FAMILIES = ('U','S','I','C','HUMAN')
REFERENCES = ('overall_elo','surface_elo','rank','phase0_blend','U','S','HUMAN')
CODE_FILES = ('tennis_v2_phase1h_source_audit.py','tennis_v2_phase1h_inputs.py','tennis_v2_phase1h_components.py','tennis_v2_phase1h_distributions.py','tennis_v2_phase1h_research.py','tennis_v2_phase1_guards.py','tennis_v2_phase1_scoring.py','tennis_v2_data.py','tennis_v2_baselines.py','tennis_v2_incumbent.py','tennis_v2_metrics.py')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(out,name,obj):
    out.mkdir(parents=True,exist_ok=True)
    (out/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def cluster_boot(a,b,groups):
    """Paired, match-weighted resampling of whole season/event clusters."""
    sums = defaultdict(lambda:[0.,0])
    for x,y,g in zip(a,b,groups):
        sums[g][0] += x-y; sums[g][1] += 1
    v = np.array([sums[g] for g in sorted(sums)],float)
    if len(v)==0: return None
    rng = np.random.default_rng(20261008)
    draws = rng.integers(0,len(v),size=(2000,len(v)))
    sampled = v[draws].sum(axis=1)
    means = sampled[:,0]/sampled[:,1]
    return dict(mean_diff=float(v[:,0].sum()/v[:,1].sum()),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975)),clusters=len(v),n=int(v[:,1].sum()),resamples=2000,seed=20261008)


def replay(tour):
    rows,stats,audit = load_tour(tour)
    usable = []; rejected = Counter()
    for r in rows:
        try:
            fmt = rule_for(r)
            if r['surface'] not in SURFACES: raise EvidenceBlocked('BLOCKED_SURFACE')
            if not r['winner_id'] or not r['loser_id'] or r['winner_id']==r['loser_id']: raise EvidenceBlocked('BLOCKED_IDENTITY')
            if not r['score'] or 'W/O' in r['score'].upper(): raise EvidenceBlocked('UNPLAYED')
            usable.append((r,fmt))
        except EvidenceBlocked as e: rejected[str(e)] += 1
    state = State(tour); release_index = 0; remembered = {}; outputs = []; populations = defaultdict(Counter)
    for r,fmt in usable:
        cutoff = (date.fromisoformat(r['match_date'])-timedelta(days=28)).isoformat()
        while release_index<len(usable) and usable[release_index][0]['match_date']<=cutoff:
            old,_ = usable[release_index]
            state.release(old,{p:stats.get((old['match_id'],p),{}) for p in (old['winner_id'],old['loser_id'])},remembered.get(old['match_id']))
            release_index += 1
        p1,p2 = sorted((r['winner_id'],r['loser_id']))
        target = Target(tour,r['tourney_id'],r['match_date'],r['surface'],p1,p2,fmt)
        f = state.features(target)
        all_point = {fam:[point_projection(x,fam) for x in f['players']] for fam in BASE_FAMILIES+('D','R')}
        remembered[r['match_id']] = {p:all_point['C'][i]['serve_point_win'] for i,p in enumerate((p1,p2))}
        season = r['match_date'][:4]; populations[season]['format_surface_identity_decided'] += 1
        if min(x['prior_matches'] for x in f['players'])<5:
            populations[season]['blocked_prior_history'] += 1; continue
        populations[season]['eligible'] += 1
        # Ranking metadata is a source pre-event field, not a certified
        # provider vintage; missing ranks stay missing, fallback declared .5.
        pts = {r['winner_id']:r['winner_rank_points'],r['loser_id']:r['loser_rank_points']}
        rank_x = float(np.log(pts[p1]/pts[p2])) if all(pts[p] and pts[p]>0 for p in (p1,p2)) else None
        pstats = [stats.get((r['match_id'],p),{}) for p in (p1,p2)]
        actual = []
        if not r['is_incomplete'] and all(stats_channels(x) for x in pstats):
            for x in pstats:
                w,n = stats_channels(x)
                actual.append(dict(point=w[0]/n[0],n=n[0],first=w[2]/n[2] if n[2] else None,second=w[3]/n[3] if n[3] else None,hold=observed_hold(x)))
            populations[season]['point_eligible_matches'] += 1
        outputs.append(dict(tour=tour,match_id=r['match_id'],event=r['tourney_id'],day=r['match_date'],surface=r['surface'],fmt=fmt,p1=p1,p2=p2,names={r['winner_id']:r['winner_name'],r['loser_id']:r['loser_name']},y=int(p1==r['winner_id']),clean=not bool(r['is_incomplete']),features=f,points=all_point,actual=actual,rank_x=rank_x))
    return outputs,dict(season_source=audit,source_exclusions=dict(rejected),population={k:dict(v) for k,v in sorted(populations.items())})


def components(rows):
    rr = [r for r in rows if r['actual']]
    groups = [r['event'] for r in rr for _ in (1,2)]
    y = np.array([a['point'] for r in rr for a in r['actual']])
    holds = [a['hold'] for r in rr for a in r['actual']]
    ih = [i for i,h in enumerate(holds) if h is not None]
    out = {}
    for fam in BASE_FAMILIES+('D','R'):
        pts = [p for r in rr for p in r['points'][fam]]
        p = np.array([p['serve_point_win'] for p in pts]); h = np.array([p['hold'] for p in pts])
        ph = h[ih]; ah = np.array([holds[i] for i in ih])
        out[fam] = dict(n=len(y),point_mae=float(np.abs(p-y).mean()),point_bias=float((p-y).mean()),return_mae=float(np.abs((1-p)-(1-y)).mean()),hold_n=len(ih),hold_mae=float(np.abs(ph-ah).mean()),hold_bias=float((ph-ah).mean()),point_rmse=float(np.sqrt(((p-y)**2).mean())))
        for channel in ('first','second'):
            idx = [i for i,a in enumerate([a for r in rr for a in r['actual']]) if a[channel] is not None]
            pred = [pts[i][channel+'_serve_win'] for i in idx]
            act = [a[channel] for r in rr for a in r['actual'] if a[channel] is not None]
            out[fam][channel+'_channel_mae'] = float(np.abs(np.array(pred)-act).mean())
    gates = {}
    for fam,parents in {'S':['U'],'I':['U'],'C':['S','I'],'D':['C'],'R':['C']}.items():
        comparisons = []
        for parent in parents:
            a = np.array([p['serve_point_win'] for r in rr for p in r['points'][fam]])
            b = np.array([p['serve_point_win'] for r in rr for p in r['points'][parent]])
            ci = cluster_boot(abs(a-y),abs(b-y),groups)
            rel = 1-out[fam]['point_mae']/out[parent]['point_mae']
            passed = rel>=.01 and ci['hi']<0 and out[fam]['hold_mae']<=out[parent]['hold_mae']+.005
            comparisons.append(dict(parent=parent,relative_point_mae_gain=rel,paired=ci,passes=bool(passed)))
        parent_pass = (all(gates[k]['verdict']=='SURVIVES' for k in ('S','I')) if fam=='C' else gates['C']['verdict']=='SURVIVES' if fam in ('D','R') else True)
        # C's existing human benchmark is always evaluated, but it cannot
        # authorize a new combination unless its own routed tests succeed.
        gates[fam] = dict(verdict='SURVIVES' if parent_pass and all(x['passes'] for x in comparisons) else 'WEAK' if parent_pass and any(x['relative_point_mae_gain']>0 for x in comparisons) else 'REJECTED',comparisons=comparisons,parent_gate=parent_pass)
    # Oracle is a descriptive iid-points upper bound, not calibrated causal truth.
    oracle_hold = [a['point'] if a['point'] in (0.,1.) else hold_probability(a['point']) for r in rr for a in r['actual']]
    out['POSTGAME_ORACLE_DIAGNOSTIC_ONLY'] = dict(hold_mae=float(np.abs(np.array(oracle_hold)[ih]-np.array(holds)[ih]).mean()),explanation='Actual server point rate through stationary hold solver; same-game inputs, never forecast/selection')
    return dict(metrics=out,family_gates=gates)


def add_predictions(rows,slope,allowed,stack=None):
    for r in rows:
        rate = r['features']['rating']
        p1,p2,s = r['p1'],r['p2'],r['surface']
        # Surface-shrunk Elo is the canonical surface evidence comparator;
        # the frozen incumbent blend is retained separately.
        # Elo's raw surface prediction is also preserved in receipt.
        pe = rate['overall']; ps = rate['surface_shrunk']
        pr = float(1/(1+np.exp(-slope*r['rank_x']))) if r['rank_x'] is not None else .5
        pred = dict(overall_elo=pe,surface_elo=ps,rank=pr,phase0_blend=.5*rate['blend']+.5*pr)
        for fam in allowed:
            a,b = [x['serve_point_win'] for x in r['points'][fam]]
            pred[fam] = match_win_probability(a,b,fmt=r['fmt'])
        if stack is not None:
            pred['STACK'] = stack_predict([pred['C'],ps,pr],stack)
        r['predictions'] = pred


def summarize(rows,names):
    if not rows: return None
    y = np.array([r['y'] for r in rows]); groups = [r['event'] for r in rows]
    mets = {}
    for name in names:
        p = np.array([r['predictions'][name] for r in rows])
        mets[name] = M.prob_metrics(p,y)
        fav = np.maximum(p,1-p); wrong = ((p>=.5)!=(y==1))
        mets[name]['high_confidence_miss_ge_85'] = dict(n=int((fav>=.85).sum()),misses=int(((fav>=.85)&wrong).sum()),rate=float(wrong[fav>=.85].mean()) if (fav>=.85).any() else None)
        mets[name]['brier_cluster_ci'] = cluster_boot((p-y)**2,np.zeros(len(y)),groups)
    best = min(REFERENCES,key=lambda k:mets[k]['brier'])
    comparisons = {}
    for name in names:
        p = np.array([r['predictions'][name] for r in rows]); b = np.array([r['predictions'][best] for r in rows])
        ci = cluster_boot((p-y)**2,(b-y)**2,groups)
        rel = 1-mets[name]['brier']/mets[best]['brier']
        passes = rel>=.02 and ci['hi']<0 and mets[name]['ece_10bin']<=.03 and mets[name]['log_loss']<=mets[best]['log_loss']
        comparisons[name] = dict(reference=best,relative_brier_gain=rel,paired=ci,match_gate_passes=bool(passes))
    return dict(n=len(rows),metrics=mets,comparisons=comparisons,rank_metadata_coverage=sum(r['rank_x'] is not None for r in rows)/len(rows))


def stage_metrics(rows,names):
    out = {'RAW':summarize(rows,names),'CLEAN':summarize([r for r in rows if r['clean']],names),'components':components(rows)}
    out['surfaces'] = {s:summarize([r for r in rows if r['surface']==s],names) for s in sorted({r['surface'] for r in rows})}
    out['matchup_component_slices'] = {}
    for label,fn in [('close',lambda r:max(r['predictions']['phase0_blend'],1-r['predictions']['phase0_blend'])<=.6),('strong_favorite',lambda r:max(r['predictions']['phase0_blend'],1-r['predictions']['phase0_blend'])>=.75)]:
        rr = [r for r in rows if fn(r) and r['actual']]
        if rr: out['matchup_component_slices'][label] = components(rr)['metrics']
    return out


def lock_hashes():
    return {f:digest(D.REPO/f) for f in CODE_FILES} | {'protocol':digest(D.OUT/'phase1h_protocol_addendum.json'),'source_manifest':digest(D.OUT/'source_manifest.json'),'population_amendment':digest(D.OUT/'phase1h_population_amendment.json')}


def receipts(out,tour,rows,focus):
    out.mkdir(parents=True,exist_ok=True)
    path = out/f'phase1h_{tour}_receipts.jsonl.gz'
    hashes = lock_hashes()
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mtime=0,mode='wb') as gz:
            for r in rows:
                a,b = [x['serve_point_win'] for x in r['points'][focus]]
                d = match_distribution(a,b,r['fmt'])
                if abs(d['p1_win']-r['predictions'][focus])>1e-10: raise AssertionError('scoring engines disagree')
                receipt = dict(tour=tour,match_id=r['match_id'],event=r['event'],tournament_start=r['day'],surface=r['surface'],best_of=r['fmt'].best_of,final_set_rule=r['fmt'].final_set_rule,p1=r['p1'],p2=r['p2'],names=r['names'],history=r['features'],point_components=r['points'],match_probabilities=r['predictions'],scoring_distribution=d,distribution_family=focus,uncertainty_reasons=['historical timing proxy not original-vintage certified','current health/rest/travel and within-event state unavailable','stationary point assumption','thin surface history' if min(x['own']['surface_points'][0] for x in r['features']['players'])<300 else 'point dependence/day variation not modeled'],evaluation_only={'p1_won':r['y'],'CLEAN':r['clean'],'point_labels':r['actual']},evidence_grade='RETROSPECTIVE_BURNED',code_and_source_hashes=hashes)
                gz.write((json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    return dict(path=path.name,rows=len(rows),sha256=digest(path))


def run(stage,out):
    if stage not in ('select','diagnostics'): raise ValueError('unsupported stage')
    lock_path = D.OUT/'phase1h_development_lock.json'
    lock = json.loads(lock_path.read_text()) if stage=='diagnostics' else {'protocol':P['protocol_id'],'hashes':lock_hashes(),'tours':{},'sealed_confirmation':'NOT_OPENED'}
    if stage=='diagnostics' and lock['hashes']!=lock_hashes(): raise EvidenceBlocked('BLOCKED_DATA: frozen code/source mismatch')
    result = {'stage':stage,'evidence_grade':'RETROSPECTIVE_BURNED','sealed_outcomes_read':False,'tours':{}}
    for tour in ('atp','wta'):
        print('replay',tour,stage,flush=True)
        all_rows,audit = replay(tour)
        dev = [r for r in all_rows if '2016'<=r['day'][:4]<='2019']
        select = [r for r in all_rows if '2020'<=r['day'][:4]<='2022']
        diagnostic = [r for r in all_rows if '2023'<=r['day'][:4]<='2024']
        if stage=='select':
            slope = B.rank_logit_fit([(r['rank_x'],r['y']) for r in dev if r['rank_x'] is not None])
            comp = components(select)
            approved = [k for k in ('S','I','C','D','R') if comp['family_gates'][k]['verdict']=='SURVIVES']
            allowed = list(BASE_FAMILIES)+[k for k in ('D','R') if k in approved]
            add_predictions(dev,slope,allowed)
            stack = None
            if 'C' in approved:
                stack = fit_stack([[float(logit(r['predictions'][k])) for k in ('C','surface_elo','rank')] for r in dev],[r['y'] for r in dev])
            add_predictions(select,slope,allowed,stack)
            names = list(select[0]['predictions'])
            met = stage_metrics(select,names)
            candidates = approved+(['STACK'] if stack is not None else [])
            chosen = min(candidates,key=lambda k:met['RAW']['metrics'][k]['brier']) if candidates else None
            passed = chosen is not None and met['RAW']['comparisons'][chosen]['match_gate_passes']
            lock['tours'][tour] = dict(rank_slope=slope,stack=stack,allowed_families=allowed,candidate=chosen,selection_gate_passed=bool(passed),verdict='SURVIVES_RETROSPECTIVE_GATE_NOT_PROMOTED' if passed else 'PARTIAL_SIGNAL_NOT_PROMOTED' if approved else 'REJECTED',component_gates=comp['family_gates'],next_action='SEALED_CONFIRMATION_NOT_READY: no confirmation source/driver certified; no season opened')
            result['tours'][tour] = dict(source_and_population=audit,selection=met)
        else:
            spec = lock['tours'][tour]
            add_predictions(diagnostic,spec['rank_slope'],spec['allowed_families'],spec['stack'])
            names = list(diagnostic[0]['predictions']); met = stage_metrics(diagnostic,names)
            # Literal replica kept secondary: uses unsafe within-event ordering.
            replica = {r['match_id']:r for r in INC.moneyline(D.all_matches(D.build_db(tour)),'as_implemented')}
            paired = [r for r in diagnostic if r['match_id'] in replica]
            for r in paired: r['predictions']['incumbent_unsafe'] = replica[r['match_id']]['p1_prob'] if replica[r['match_id']]['p1']==r['p1'] else 1-replica[r['match_id']]['p1_prob']
            literal = summarize(paired,names+['incumbent_unsafe'])
            # Same postgame point rate supplied to exact solver: diagnostic only.
            orows = [r for r in diagnostic if r['actual'] and all(.000001<=a['point']<=.999999 for a in r['actual'])]
            op = [match_win_probability(r['actual'][0]['point'],r['actual'][1]['point'],fmt=r['fmt']) for r in orows]
            oracle = dict(label='POSTGAME_ORACLE_DIAGNOSTIC_ONLY',n=len(orows),boundary_point_matches_excluded_from_oracle_only=sum(bool(r['actual']) for r in diagnostic)-len(orows),actual_point_rate_match_metrics=M.prob_metrics(op,[r['y'] for r in orows]))
            focus = spec['candidate'] if spec['candidate'] in spec['allowed_families'] else 'C'
            result['tours'][tour] = dict(diagnostics=met,incumbent_secondary_chronology_unsafe_paired=literal,oracle=oracle,receipt=receipts(out,tour,diagnostic,focus))
    write(out,'phase1h_'+('selection_results' if stage=='select' else 'diagnostic_results')+'.json',result)
    if stage=='select': write(out,'phase1h_development_lock.json',lock)
    print('complete',stage,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('select','diagnostics'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); run(args.stage,args.out)

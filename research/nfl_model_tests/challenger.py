#!/usr/bin/env python3
"""Leak-safe, no-bookmaker NFL player yardage challenger.

Research-only. It never changes serving predictions. The v2 reference may use
its original sportsbook context; ALL challenger columns exclude it.

Example:
 python research/nfl_model_tests/challenger.py \
   --db nfl_models/nfl_model.sqlite --pbp-dir /tmp/nflcsv \
   --out research/nfl_model_tests/challenger_result.json

Inputs: nflverse pbp/play_by_play_2023.csv.gz .. play_by_play_2026.csv.gz
with only completed historical games used before each target week.
"""
import argparse, csv, gzip, json, math, sqlite3, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nfl_yardage_v2 as v2

SEED = 20261122
BASE = [f for f in v2.FEATURES if f not in ('team_spread', 'implied_team_total')]
ROLE = ['role_ewm_vol', 'role_ewm_yds', 'role_last1_vol', 'role_last1_yds',
        'role_slope_vol', 'role_slope_yds', 'role_std_vol', 'role_std_yds',
        'role_stable_share', 'role_eff_recent', 'role_eff_prior', 'role_boom_rate',
        'role_team_changed']
QUALITY = ['air_per_target_5', 'air_per_target_12', 'deep_target_pct_12',
           'catch_pct_12', 'yac_per_catch_12', 'target_depth_change',
           'qb_cpoe_5', 'qb_pass_ypa_5', 'rush_explosive_8', 'rush_stuffed_8', 'rush_ypc_8']
VARS = {'no_line_core': BASE, 'no_line_role': BASE+ROLE,
        'no_line_quality': BASE+QUALITY,
        'no_line_role_quality': BASE+ROLE+QUALITY}
assert not any('spread' in c or 'total_line' in c or 'implied_team' in c or
               'odds' in c or 'price' in c for cs in VARS.values() for c in cs)
PARAMS = dict(v2.PARAMS, nthread=4, seed=SEED)


def mean(seq):
    return sum(seq)/len(seq) if seq else float('nan')


def weighted(seq, rate=.68):
    if not seq: return float('nan')
    a = np.asarray(seq[-8:], dtype=float)
    w = rate ** np.arange(len(a)-1, -1, -1)
    return float(np.sum(a*w)/sum(w))


def build_role(con, market):
    """Only features known strictly before the evaluated (season,week)."""
    vol, yds = v2.MARKETS[market]['vol'], v2.MARKETS[market]['yds']
    rows = con.execute(f'''SELECT player_id,team,season,week,
        COALESCE({vol},0), COALESCE({yds},0)
        FROM player_games WHERE season_type='REG'
        ORDER BY season,week''').fetchall()
    by_week = defaultdict(list)
    for p,t,s,w,vo,y in rows: by_week[(s,w)].append((p,t,float(vo),float(y)))
    hist = defaultdict(list)
    result = {}
    for key in sorted(by_week):
        for p,t,vo,y in by_week[key]:
            h = hist[p]
            if len(h)<3: continue
            last = h[-8:]
            vs = [q[1] for q in last]; ys = [q[2] for q in last]
            prev_vs = [q[1] for q in h[-4:-1]]
            prev_ys = [q[2] for q in h[-4:-1]]
            share = [q[1]/q[3] for q in last if q[3]>0]
            recent_volume = sum(vs[-5:]); prior_volume = sum(q[1] for q in h[-16:-5])
            f = {
                'role_ewm_vol':weighted(vs),'role_ewm_yds':weighted(ys),
                'role_last1_vol':vs[-1],'role_last1_yds':ys[-1],
                'role_slope_vol':vs[-1]-mean(prev_vs),
                'role_slope_yds':ys[-1]-mean(prev_ys),
                'role_std_vol':float(np.std(vs)),
                'role_std_yds':float(np.std(ys)),
                'role_stable_share':float(np.std(share)) if share else float('nan'),
                'role_eff_recent':sum(ys[-5:])/recent_volume if recent_volume else float('nan'),
                'role_eff_prior':sum(q[2] for q in h[-16:-5])/prior_volume if prior_volume else float('nan'),
                'role_boom_rate':mean([z>=1.75*max(mean(ys),5.0) for z in ys]),
                'role_team_changed':float(h[-1][0] != t),
            }
            result[(p,key)] = f
        # Team denominator is computed from the completed current week, but
        # is not exposed to current-week rows; it enters only future features.
        denom=defaultdict(float)
        for p,t,vo,y in by_week[key]:denom[t]+=vo
        for p,t,vo,y in by_week[key]: hist[p].append((t,vo,y,denom[t]))
    return result


def number(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else float('nan')
    except (ValueError,TypeError): return float('nan')


def get_quality(pbp_dir):
    """Past-game PBP evidence only. No current-game outcomes in the features."""
    weekly=defaultdict(lambda:defaultdict(list))
    qb_weekly=defaultdict(lambda:defaultdict(list))
    rush_weekly=defaultdict(lambda:defaultdict(list))
    for season in (2023,2024,2025,2026):
        f=Path(pbp_dir)/f'play_by_play_{season}.csv.gz'
        if not f.exists():raise FileNotFoundError(f'Missing required NFLverse file: {f}')
        with gzip.open(f,'rt',newline='') as stream:
            for r in csv.DictReader(stream):
                if r.get('season_type')!='REG':continue
                wk = r.get('week')
                if not wk:continue
                key=(season,int(wk))
                rid=r.get('receiver_player_id','')
                if rid and rid!='NA' and number(r.get('pass_attempt'))==1:
                    weekly[key][rid].append((number(r.get('air_yards')),
                        number(r.get('complete_pass')),number(r.get('yards_after_catch'))))
                run_id=r.get('rusher_player_id','')
                if run_id and run_id!='NA' and number(r.get('rush_attempt'))==1:
                    rush_weekly[key][run_id].append(number(r.get('rushing_yards')))
                qb=r.get('posteam','')
                if qb and qb!='NA' and number(r.get('pass_attempt'))==1:
                    qb_weekly[key][qb].append((number(r.get('cpoe')),
                        number(r.get('passing_yards'))))
    qhist=defaultdict(list);thist=defaultdict(list);rhist=defaultdict(list);output={}; qteam={}
    all_keys=sorted(set(weekly)|set(qb_weekly)|set(rush_weekly))
    for key in all_keys:
        for pid in set(qhist)|set(weekly[key]):
            entries=qhist[pid]
            if not entries:continue
            short=[v for game in entries[-5:] for v in game]
            long=[v for game in entries[-12:] for v in game]
            a5=[v[0] for v in short if np.isfinite(v[0])]
            a12=[v[0] for v in long if np.isfinite(v[0])]
            cp=[v[1] for v in long if np.isfinite(v[1])]
            yc=[v[2] for v in long if np.isfinite(v[2])]
            output[(pid,key)]={
                'air_per_target_5':mean(a5),'air_per_target_12':mean(a12),
                'deep_target_pct_12':mean([a>=20 for a in a12]),
                'catch_pct_12':mean(cp),'yac_per_catch_12':mean(yc),
                'target_depth_change':mean(a5)-mean(a12),
            }
        for pid in set(rhist)|set(rush_weekly[key]):
            a=[v for game in rhist[pid][-8:] for v in game if np.isfinite(v)]
            if a:
                output.setdefault((pid,key),{}).update({
                  'rush_explosive_8':mean([v>=15 for v in a]),
                  'rush_stuffed_8':mean([v<=0 for v in a]),
                  'rush_ypc_8':mean(a)})
        for team in set(thist)|set(qb_weekly[key]):
            vals=[v for game in thist[team][-5:] for v in game]
            cp=[a for a,b in vals if np.isfinite(a)]
            py=[b for a,b in vals if np.isfinite(b)]
            qteam[(team,key)]={'qb_cpoe_5':mean(cp), 'qb_pass_ypa_5':mean(py)}
        for pid,pv in weekly[key].items():qhist[pid].append(pv)
        for pid,pv in rush_weekly[key].items():rhist[pid].append(pv)
        for t,pv in qb_weekly[key].items():thist[t].append(pv)
    return output,qteam


def matrix(rows,cols):
    return np.asarray([[r[1].get(c, np.nan) if r[1].get(c) is not None
        else np.nan for c in cols] for r in rows],dtype=np.float32)


def bootstrap(d, groups, seed=SEED, reps=4000):
    rng=np.random.default_rng(seed)
    n=len(d)
    row=np.mean(d[rng.integers(n,size=(reps,n))],axis=1)
    uniq=sorted(set(groups))
    blocks=[np.asarray([i for i,g in enumerate(groups) if g==k],dtype=int) for k in uniq]
    # Week resampling: preserve real clustering by week.
    draws=[]
    for _ in range(reps):
        z=rng.integers(len(blocks),size=len(blocks))
        indices=np.concatenate([blocks[i] for i in z])
        draws.append(float(np.mean(d[indices])))
    return {'p_row_gain':round(float(np.mean(row>0)),4),
            'p_week_block_gain':round(float(np.mean(np.asarray(draws)>0)),4),
            'ci95_row':[round(float(v),4) for v in np.quantile(row,[.025,.975])],
            'ci95_week_block':[round(float(v),4) for v in np.quantile(draws,[.025,.975])]}


def proper_scores(pred, y, past_oof, fixed_threshold=None):
    """CRPS for empirical predictive residuals from strictly earlier weeks."""
    r=np.asarray(past_oof[-3000:],dtype=float)
    if len(r)<30:return None
    r.sort()
    n=len(r)
    pair=(2.0/(n*n))*np.dot((2*np.arange(1,n+1)-n-1),r)
    crps=float(np.mean(np.abs(pred+r-y))-.5*pair)
    threshold=.7*pred
    prob=float(np.mean(pred+r>=threshold))
    if fixed_threshold is None: fixed_threshold=threshold
    fixed_prob=float(np.mean(pred+r>=fixed_threshold))
    return crps, prob, float(y>=threshold), fixed_prob, float(y>=fixed_threshold)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db', default='nfl_models/nfl_model.sqlite')
    ap.add_argument('--pbp-dir',required=True)
    ap.add_argument('--out',default='research/nfl_model_tests/challenger_result.json')
    ap.add_argument('--allow-count-mismatch', action='store_true')
    args=ap.parse_args()
    con=sqlite3.connect(f'file:{Path(args.db).resolve()}?mode=ro',uri=True)
    pbp,qb_quality=get_quality(args.pbp_dir)
    results={'protocol':'weekly walk-forward / train strictly earlier weeks',
             'baseline':'historical v2 refit with its original spread inputs (comparison only)',
             'candidate':'no bookmaker inputs, historical role and PBP target-quality features',
             'train':'same XGBoost settings and 140 rounds as README tests',
             'evaluation':'2025 W6-W18 and 2026 W1-W11, identical v2 eligible players',
             'variants':list(VARS), 'markets':{},'source_files':[str(Path(args.pbp_dir)/f'play_by_play_{y}.csv.gz') for y in (2023,2024,2025,2026)]}
    for market,expected in [('rushing_yards',1119),('receiving_yards',2956)]:
        role=build_role(con,market)
        replay=v2.History(con,market)
        rows,_,_=replay.replay()
        for meta,f,y in rows:
            key=(meta['season'],meta['week']); pid=meta['player_id']
            f.update(role.get((pid,key),{}))
            if market=='receiving_yards':f.update(pbp.get((pid,key),{}))
            f.update(qb_quality.get((meta['team'],key),{}))
        byweek=defaultdict(list)
        for row in rows:byweek[(row[0]['season'],row[0]['week'])].append(row)
        variants={'live_v2':v2.FEATURES}|VARS
        train=[]; residuals={k:[] for k in variants}
        p_store=defaultdict(list); actual=[]; groups=[]; scores=defaultdict(lambda:defaultdict(list))
        coverage={'player_rows':0,'receiver_prior_air_data':0,'team_prior_qb_data':0}
        for key in sorted(byweek):
            sample=byweek[key]
            cal=(key >= (2024,10))
            test=(key >= (2025,6) and key <= (2025,18)) or (key[0]==2026 and key[1]<=11)
            if cal and len(train)>=300:
                ytr=np.asarray([r[2] for r in train],dtype=np.float32)
                obs=np.asarray([r[2] for r in sample],dtype=np.float32)
                week_predictions={}
                for name,cols in variants.items():
                    model=xgb.train(PARAMS,xgb.DMatrix(matrix(train,cols),label=ytr,
                       feature_names=cols),num_boost_round=140)
                    week_predictions[name]=model.predict(xgb.DMatrix(matrix(sample,cols),feature_names=cols))
                live_week=week_predictions['live_v2']
                for name,pred in week_predictions.items():
                    if test:p_store[name].extend(pred.tolist())
                    for i,(p,y) in enumerate(zip(pred,obs)):
                        if test:
                            ps=proper_scores(float(p),float(y),residuals[name],
                                             fixed_threshold=.7*float(live_week[i]))
                            if ps is not None:
                                scores[name]['crps'].append(ps[0])
                                scores[name]['brier'].append((ps[1]-ps[2])**2)
                                scores[name]['brier_fixed'].append((ps[3]-ps[4])**2)
                    # residual from a pregame forecast: add only AFTER scoring
                    residuals[name].extend((obs-pred).tolist())
                if test:
                    actual.extend(obs.tolist())
                    groups.extend([f'{key[0]}-{key[1]}' for _ in sample])
                    for meta,f,y in sample:
                        coverage['player_rows']+=1
                        coverage['receiver_prior_air_data']+=int(np.isfinite(f.get('air_per_target_5',float('nan'))))
                        coverage['team_prior_qb_data']+=int(np.isfinite(f.get('qb_cpoe_5',float('nan'))))
            train.extend(sample) # update only after the current week was forecast
        if len(actual)!=expected and not args.allow_count_mismatch:
            raise AssertionError(f'{market}: {len(actual)} rows, README expects {expected}; refusing incomparable result')
        y=np.asarray(actual,dtype=float)
        baseline=np.asarray(p_store['live_v2'])
        summary={'n':len(y),'expected_n':expected,'comparable_count':len(y)==expected,
                 'pbp_coverage':coverage,'results':{}}
        for name in variants:
            p=np.asarray(p_store[name]); e=np.abs(p-y)
            rep={'mae':round(float(e.mean()),4),
                 'within10':round(float((e<=10).mean()),4),
                 'bias':round(float((p-y).mean()),4),
                 'crps':round(float(np.mean(scores[name]['crps'])),4) if scores[name]['crps'] else None,
                 'brier_70pct_own_projection':round(float(np.mean(scores[name]['brier'])),5) if scores[name]['brier'] else None,
                 'brier_70pct_live_projection_fixed_event':round(float(np.mean(scores[name]['brier_fixed'])),5) if scores[name]['brier_fixed'] else None,
                 'n_calibrated_scores':len(scores[name]['crps'])}
            if name!='live_v2':
                gain=np.abs(baseline-y)-e
                rep['gain_vs_live_refit_mae']=round(float(gain.mean()),4)
                rep['paired_bootstrap']=bootstrap(gain,groups)
                rep['beat_24_86_or_22_44']=bool(rep['mae'] < (24.86 if market=='rushing_yards' else 22.44))
                rep['meaningful_0_3_gain']=bool(rep['gain_vs_live_refit_mae']>=.3)
            summary['results'][name]=rep
        results['markets'][market]=summary
        print(f'{market}: n={len(y)}, live={summary["results"]["live_v2"]["mae"]}',flush=True)
        for name in VARS:print(f'  {name}: {summary["results"][name]["mae"]}',flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(results,indent=2,allow_nan=False))
    print('REPORT',args.out)

if __name__=='__main__':main()
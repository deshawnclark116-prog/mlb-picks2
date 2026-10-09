#!/usr/bin/env python3
"""Research-only Phase1K-R. Frozen pregame PBP mechanics, oracle-carry first.

No network/model serving, simulation, sportsbook, FTN or personnel inputs.
Development mode opens 2023/2024 only. Confirmation refuses an empty lock
before opening any validation payload. Every target's history uses a conservative
completed-game bound, strictly earlier weeks and stable identities.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

import nfl_v2_phase1k_sources as S
import nfl_v2_phase1a_direct as A
import nfl_v2_phase1b_opportunity as B
import nfl_v2_competent_human_baseline as H

ART=S.ART
BASE_HEAD='edf84ca5be15d53d5525e8b613dac7961797a35d'
FAMILIES=('A_player_profile','B_defense_allowance','C_player_defense','D_explosive','E_states','F_location','G_context')
RATES=('negative','zero','success','ten','twenty')
LOCATIONS=('left','middle','right','unknown')
CONTEXTS=tuple(f'{situ}_{formation}' for situ in ('goal','short','early','other') for formation in ('shotgun','under'))


def canonical(value):
    """Twelve-decimal artifact precision prevents irrelevant BLAS last-bit drift."""
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,float):return round(value,12)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(canonical(value),indent=2,sort_keys=True,allow_nan=False)+'\n')


def number(value,default=0.):
    try:
        x=float(value)
        return x if np.isfinite(x) else default
    except (TypeError,ValueError):return default


def legal(source,target):
    return (source['season'],source['week']) < (target['season'],target['week']) and date.fromisoformat(source['date'])+timedelta(days=2) <= date.fromisoformat(target['date'])-timedelta(days=1)


def window(rows,n):
    games=sorted({(r['season'],r['week'],r['game_id']) for r in rows})[-n:]
    keys=set(games)
    return [r for r in rows if (r['season'],r['week'],r['game_id']) in keys]


def context(row):
    situation='goal' if number(row.get('yardline_100'),100)<=5 else 'short' if number(row.get('ydstogo'),10)<=2 else 'early' if number(row.get('down'),4)<=2 else 'other'
    return situation+('_shotgun' if row.get('shotgun')=='1' else '_under')


def load(directory,years):
    """Only explicitly authorized seasons; source hashes checked before parsing."""
    S.verify(directory,years)
    manifest=S.sources(); stats=[]; carries=[];games={}
    for year in years:
        raw=list(S.records(Path(directory)/manifest[str(year)]['stats']['local_name'],S.STATS_FIELDS))
        identities={(r['game_id'],r['player_id']):r['position'] for r in raw}
        for r in S.records(Path(directory)/manifest[str(year)]['pbp']['local_name'],S.PBP_FIELDS):
            gid=r['game_id'];g={'game_id':gid,'season':year,'week':int(float(r['week'])),'date':r['game_date'],'away':r['away_team'],'home':r['home_team']}
            if gid in games and games[gid]!=g:raise ValueError('Inconsistent game metadata')
            games[gid]=g
            if r.get('rush_attempt')!='1' or r.get('qb_kneel')=='1' or r.get('qb_scramble')=='1' or r.get('two_point_attempt')=='1' or r.get('play_type')=='no_play':continue
            pid=r.get('rusher_player_id')
            if identities.get((gid,pid)) not in S.POSITIONS:continue
            if not S.present(r.get('rushing_yards')):raise ValueError('Missing carry yards')
            y=number(r['rushing_yards']);epa=number(r.get('epa'))
            carries.append({**g,'player_id':pid,'position':identities[(gid,pid)],'team':r['posteam'],'opponent':r['defteam'],'y':y,'epa':epa,'negative':float(y<0),'zero':float(y==0),'success':float(epa>0),'ten':float(y>=10),'twenty':float(y>=20),'loc':r['run_location'] if r.get('run_location') in LOCATIONS else 'unknown','context':context(r),'no_huddle':number(r.get('no_huddle')),'short':float(number(r.get('ydstogo'),10)<=2),'goal':float(number(r.get('yardline_100'),100)<=5),'play_id':r['play_id']})
        for r in raw:
            g=games.get(r['game_id'])
            if g is None:raise ValueError('Official stats game missing in frozen PBP')
            stats.append({**g,'player_id':r['player_id'],'player':r['player_display_name'],'position':r['position'],'team':r['team'],'opponent':r['opponent_team'],'carries':number(r.get('carries')),'rushing_yards':number(r.get('rushing_yards'))})
    stats.sort(key=lambda r:(r['season'],r['week'],r['game_id'],r['player_id']))
    carries.sort(key=lambda r:(r['season'],r['week'],r['game_id'],number(r['play_id'])))
    # Whole-game team validation makes absent player stat rows interpretable as zero.
    covered={(r['game_id'],r['team']) for r in stats}
    for g in games.values():
        if any((g['game_id'],tm) not in covered for tm in (g['away'],g['home'])):raise ValueError('Incomplete official game/team labels')
    return stats,carries,sorted(games.values(),key=lambda g:(g['date'],g['game_id']))


def profile(rows,prior,k):
    n=len(rows);den=n+k
    if den==0:raise ValueError('Empty unshrunk profile')
    out={'ypc':(sum(r['y'] for r in rows)+k*prior['ypc'])/den,
         'epa':(sum(r['epa'] for r in rows)+k*prior.get('epa',0.))/den,'n':n}
    for key in RATES:out[key]=(sum(r[key] for r in rows)+k*prior[key])/den
    out['median']=statistics.median([r['y'] for r in rows]) if rows else prior['ypc']
    for key in ('short','goal'):
        subset=[r for r in rows if r[key]]
        out[key+'_ypc']=(sum(r['y'] for r in subset)+20*out['ypc'])/(len(subset)+20)
    return out


def decompose(rows,league,states):
    total=0.;parts=[];probs=[];means=[]
    for test in states:
        own=[r for r in rows if test(r['y'])];pri=[r for r in league if test(r['y'])]
        probability=(len(own)+50*len(pri)/len(league))/(len(rows)+50)
        prior_mean=sum(r['y'] for r in pri)/len(pri) if pri else 0.
        conditional=(sum(r['y'] for r in own)+20*prior_mean)/(len(own)+20)
        contribution=probability*conditional;total+=contribution
        parts.append(contribution);probs.append(probability);means.append(conditional)
    return {'ypc':total,'contributions':parts,'probabilities':probs,'conditional_yards':means}


def mixture(rows,keys,key,prior,k=50):
    return {v:(sum(r[key]==v for r in rows)+k*prior[v])/(len(rows)+k) for v in keys}


def mixture_yields(player,defense,league,mix,key):
    py=sum(r['y'] for r in player)/len(player) if player else sum(r['y'] for r in league)/len(league)
    league_mean=sum(r['y'] for r in league)/len(league)
    yields={}
    for v in mix:
        ls=[r for r in league if r[key]==v];ps=[r for r in player if r[key]==v];ds=[r for r in defense if r[key]==v]
        lp=sum(r['y'] for r in ls)/len(ls) if ls else league_mean
        pp=(sum(r['y'] for r in ps)+20*py)/(len(ps)+20)
        dp=(sum(r['y'] for r in ds)+160*lp)/(len(ds)+160)
        yields[v]=pp+.30*(dp-lp)
    return sum(mix[v]*yields[v] for v in mix),yields


def components(player,defense,team,league):
    empty={'ypc':0.,'epa':0.,**{k:0. for k in RATES}}
    lp=profile(league,empty,0);pp=profile(player,lp,50);dp=profile(defense,lp,160)
    tail=decompose(player,league,[lambda y:y<10,lambda y:10<=y<20,lambda y:y>=20])
    state=decompose(player,league,[lambda y:y<=0,lambda y:0<y<10,lambda y:y>=10])
    prior_loc=mixture(league,LOCATIONS,'loc',{v:0. for v in LOCATIONS},0)
    prior_ctx=mixture(league,CONTEXTS,'context',{v:0. for v in CONTEXTS},0)
    loc=mixture(team,LOCATIONS,'loc',prior_loc)
    team_ctx=mixture(team,CONTEXTS,'context',prior_ctx)
    # Current-team deployment only, while skill profile is stable-player across teams.
    current_player=[r for r in player if team and r['team']==team[-1]['team']]
    ctx=mixture(current_player,CONTEXTS,'context',team_ctx)
    loc_y,loc_yields=mixture_yields(player,defense,league,loc,'loc')
    ctx_y,ctx_yields=mixture_yields(player,defense,league,ctx,'context')
    return {'player_profile':pp,'defense_profile':dp,'league_profile':lp,'tail':tail,'states':state,
            'location_mixture':loc,'context_mixture':ctx,'league_location_mixture':prior_loc,'league_context_mixture':prior_ctx,
            'location_ypc':loc_y,'context_ypc':ctx_y,'location_yields':loc_yields,'context_yields':ctx_yields,
            'no_huddle_rate':sum(r['no_huddle'] for r in team)/len(team) if team else 0.}


def vectors(c,base):
    p,d,l=c['player_profile'],c['defense_profile'],c['league_profile']
    a=[p['ypc']-base,p['median']-p['ypc'],*(p[k] for k in RATES),p['short_ypc']-p['ypc'],p['goal_ypc']-p['ypc']]
    b=[d['ypc']-l['ypc'],d['epa']-l['epa'],*(d[k]-l[k] for k in ('success','negative','ten','twenty'))]
    t=c['tail'];s=c['states'];ctx=c['context_mixture']
    return {'A_player_profile':a,'B_defense_allowance':b,'C_player_defense':a+b,
            'D_explosive':[t['ypc']-base,*t['contributions'][1:],sum(t['probabilities'][1:]),t['probabilities'][2]],
            'E_states':[s['ypc']-base,*s['contributions'],p['success']],
            'F_location':[c['location_ypc']-base,*(c['location_mixture'][v] for v in LOCATIONS[:3])],
            'G_context':[c['context_ypc']-base,*(sum(v for k,v in ctx.items() if k.startswith(sit+'_')) for sit in ('goal','short','early')),sum(v for k,v in ctx.items() if k.endswith('_shotgun')),c['no_huddle_rate']]}


def assemble(directory,season,min_week,max_week):
    stats,carries,games=load(directory,range(2023,season+1));labels={(r['game_id'],r['player_id']):r for r in stats}
    pi=defaultdict(list);di=defaultdict(list);ti=defaultdict(list)
    for r in carries:pi[r['player_id']].append(r);di[r['opponent']].append(r);ti[r['team']].append(r)
    frozen=json.loads((ART/'phase1b_opportunity_snapshot.json').read_text())['outcomes']['rush_yds']['selected_efficiency_config']
    out=[]
    for game in games:
        if game['season']!=season or not min_week<=game['week']<=max_week:continue
        history=[r for r in stats if legal(r,game)]
        totals=defaultdict(lambda:defaultdict(float));opponents={}
        for r in history:
            tk=(r['season'],r['week'],r['team']);totals[tk]['carries']+=r['carries'];totals[tk]['rushing_yards']+=r['rushing_yards'];opponents[tk]=r['opponent']
        A.build_indexes(history,totals,opponents);B.build_extra_indexes(history,totals,opponents);H.build_indexes(history,totals,opponents)
        league=window([r for r in carries if legal(r,game)],256)
        # window counts offensive team-games, not global weeks; explicit cap below.
        keys=sorted({(r['season'],r['week'],r['game_id'],r['team']) for r in league})[-256:]
        league=[r for r in league if (r['season'],r['week'],r['game_id'],r['team']) in set(keys)]
        prior_ids={r['player_id'] for r in history if r['position'] in S.POSITIONS}
        for team,opponent in ((game['away'],game['home']),(game['home'],game['away'])):
            team_hist=window([r for r in ti[team] if legal(r,game)],8)
            defense=window([r for r in di[opponent] if legal(r,game)],8)
            for pid in sorted(prior_ids):
                all_hist=[r for r in history if r['player_id']==pid]
                if not all_hist or all_hist[-1]['team']!=team or all_hist[-1]['position'] not in S.POSITIONS:continue
                role=[r for r in all_hist if r['team']==team][-3:]
                if len(role)<2 or sum(r['carries'] for r in role)/len(role)<5:continue
                target={**game,'player_id':pid,'player':all_hist[-1]['player'],'team':team,'opponent':opponent,'position':all_hist[-1]['position']}
                incumbent=B.efficiency_receipt(history,totals,opponents,target,'rush_yds',frozen)
                human=H.efficiency_receipt(history,totals,opponents,target,'rush_yds')
                if incumbent is None or human is None:raise ValueError('Fixed candidate population lacks comparator; no silent row drop')
                player=window([r for r in pi[pid] if legal(r,game)],8)
                if not player or not league:raise ValueError('Fixed candidate lacks PBP history')
                c=components(player,defense,team_hist,league)
                position_rows=[q for q in league if q['position']==target['position']]
                position_ypc=sum(q['y'] for q in position_rows)/len(position_rows) if len(position_rows)>=30 else c['league_profile']['ypc']
                recent=[r for r in all_hist[-5:] if r['carries']>0];recentypc=sum(r['rushing_yards'] for r in recent)/sum(r['carries'] for r in recent)
                label=labels.get((game['game_id'],pid),{});actualc=label.get('carries',0.);actualy=label.get('rushing_yards',0.)
                plays=[r for r in pi[pid] if r['game_id']==game['game_id']]
                rates={k:sum(r[k] for r in plays)/len(plays) if plays else None for k in RATES}
                recentplays=window([r for r in pi[pid] if legal(r,game)],5)
                recentrates={k:sum(r[k] for r in recentplays)/len(recentplays) for k in RATES}
                actual_loc=mixture(plays,LOCATIONS,'loc',{k:0. for k in LOCATIONS},0) if plays else None
                actual_ctx=mixture(plays,CONTEXTS,'context',{k:0. for k in CONTEXTS},0) if plays else None
                out.append({**target,'actual_carries':actualc,'actual_rushing_yards':actualy,'actual_ypc':actualy/actualc if actualc>0 else None,
                            'prior_current_team_mean_carries':sum(r['carries'] for r in role)/len(role),'components':c,'vectors':vectors(c,incumbent['final_efficiency_projection']),
                            'comparators':{'phase1f_incumbent':incumbent['final_efficiency_projection'],'phase1e_efficiency_same_as_f':incumbent['final_efficiency_projection'],'recent_player_ypc':recentypc,'competent_human':human['projection'],'position_league':position_ypc},
                            'incumbent_receipt':incumbent,'actual_rates':rates,'recent_rate_baseline':recentrates,
                            'label_reconciliation':{'pbp_carries':len(plays),'official_carries':actualc,'pbp_yards':sum(r['y'] for r in plays),'official_yards':actualy},
                            'oracle_metadata':{'label':'POSTGAME_ORACLE_DIAGNOSTIC_ONLY','location_mixture':actual_loc,'context_mixture':actual_ctx,'tail_counts':[sum(r['y']<10 for r in plays),sum(10<=r['y']<20 for r in plays),sum(r['y']>=20 for r in plays)]},
                            'history_receipt':{'cutoff_bound':str(date.fromisoformat(game['date'])-timedelta(days=1))+'T00:00:00Z','prior_player_game_ids':sorted({r['game_id'] for r in player}),'prior_defense_game_ids':sorted({r['game_id'] for r in defense}),'max_prior_game_date':max(r['date'] for r in league)}})
    return out


def fit(rows,family,alpha):
    rs=[r for r in rows if r['actual_carries']>0]
    x=np.asarray([r['vectors'][family] for r in rs]); y=np.asarray([r['actual_ypc']-r['comparators']['phase1f_incumbent'] for r in rs])
    mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd<1e-12]=1.;z=(x-mu)/sd
    intercept=float(y.mean());beta=np.linalg.solve(z.T@z+len(rs)*alpha*np.eye(z.shape[1]),z.T@(y-intercept))
    return canonical({'family':family,'alpha':alpha,'mean':mu.tolist(),'std':sd.tolist(),'beta':beta.tolist(),'intercept':intercept,'fit_positive_rows':len(rs)})


def predict(row,model):
    x=np.asarray(row['vectors'][model['family']]);z=(x-np.asarray(model['mean']))/np.asarray(model['std'])
    raw=row['comparators']['phase1f_incumbent']+model['intercept']+float(z@np.asarray(model['beta']))
    return round(float(np.clip(raw,0.,12.)),12)


def component_rates(row,family):
    p=row['components']['player_profile'];d=row['components']['defense_profile']
    if family=='B_defense_allowance':return {k:d[k] for k in RATES}
    if family=='C_player_defense':return {k:.7*p[k]+.3*d[k] for k in RATES}
    if family=='D_explosive':
        t=row['components']['tail'];return {**{k:p[k] for k in RATES},'ten':sum(t['probabilities'][1:]),'twenty':t['probabilities'][2]}
    if family=='E_states':return {**{k:p[k] for k in RATES},'ten':row['components']['states']['probabilities'][2]}
    return {k:p[k] for k in RATES}


def mean(xs):return float(statistics.mean(xs)) if xs else None


def metrics(rows,predictions,family=None):
    positive=[(r,p) for r,p in zip(rows,predictions) if r['actual_carries']>0]
    errors=[p*r['actual_carries']-r['actual_rushing_yards'] for r,p in positive]
    yerrors=[p-r['actual_ypc'] for r,p in positive]
    out={'candidate_rows':len(rows),'positive_carry_rows':len(positive),'zero_carry_rows':len(rows)-len(positive),
         'ypc_mae':mean([abs(e) for e in yerrors]),'ypc_bias':mean(yerrors),'oracle_carry_mae':mean([abs(e) for e in errors]),'oracle_carry_bias':mean(errors),'oracle_carry_median_absolute_error':float(statistics.median([abs(e) for e in errors])) if errors else None,
         'within':{str(t):mean([float(abs(e)<=t) for e in errors]) for t in (5,10,15,20)},'miss_rate':{str(t):mean([float(abs(e)>t) for e in errors]) for t in (30,40,50)}}
    # Incumbent has no carry-state probabilities; comparator below is explicitly
    # raw recent-player rate, never fabricated from an incumbent mean.
    rate_metrics={}
    for k in ('negative','success','ten','twenty'):
        valid=[r for r in rows if r['actual_rates'][k] is not None and r['label_reconciliation']['pbp_carries']==r['actual_carries']]
        predicted=[component_rates(r,family)[k] if family else r['recent_rate_baseline'][k] for r in valid]
        actual=[r['actual_rates'][k] for r in valid]
        rate_metrics[k]={'n':len(valid),'mae':mean([abs(a-p) for a,p in zip(actual,predicted)]),'predicted_mean':mean(predicted),'observed_mean':mean(actual),'baseline_recent5_mae':mean([abs(r['actual_rates'][k]-r['recent_rate_baseline'][k]) for r in valid])}
    out['component_rates']=rate_metrics
    for key in ('location','context'):
        keys=LOCATIONS if key=='location' else CONTEXTS
        valid=[r for r in rows if r['oracle_metadata'][key+'_mixture'] is not None]
        out[key+'_mixture_mae']=mean([sum(abs(r['components'][key+'_mixture'][k]-r['oracle_metadata'][key+'_mixture'][k]) for k in keys)/len(keys) for r in valid])
        out['league_'+key+'_mixture_mae']=mean([sum(abs(r['components']['league_'+key+'_mixture'][k]-r['oracle_metadata'][key+'_mixture'][k]) for k in keys)/len(keys) for r in valid])
    return out


def blocked_bootstrap(rows,predictions,replicates=2000,seed=6412024):
    bygame=defaultdict(list);dates={}
    for r,p in zip(rows,predictions):
        if r['actual_carries']<=0:continue
        base=r['comparators']['phase1f_incumbent'];n=r['actual_carries'];y=r['actual_rushing_yards']
        bygame[r['game_id']].append(abs(p*n-y)-abs(base*n-y));dates[r['game_id']]=date.fromisoformat(r['date'])
    gids=sorted(bygame,key=lambda g:(dates[g],g))
    # Each block begins on an observed game date and includes 14 calendar days.
    blocks=[[g for g in gids if dates[start]<=dates[g]<dates[start]+timedelta(days=14)] for start in gids]
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        selected=[]
        while len(selected)<len(gids):selected.extend(blocks[int(rng.integers(len(blocks)))])
        selected=selected[:len(gids)] # truncate by whole games, never by rows
        values.append(mean([v for g in selected for v in bygame[g]]))
    return {'method':'14-calendar-day moving blocks, complete games, paired oracle loss','replicates':replicates,'seed':seed,'game_count':len(gids),'delta_mae':mean([v for g in gids for v in bygame[g]]),'ci95':np.quantile(values,[.025,.975]).tolist()}


def gate(candidate,incumbent,bootstrap,family):
    reasons=[]
    gain=incumbent['oracle_carry_mae']-candidate['oracle_carry_mae'];ygain=incumbent['ypc_mae']-candidate['ypc_mae']
    if candidate['positive_carry_rows']<150:reasons.append('INSUFFICIENT_ROWS')
    if gain<max(.5,.02*incumbent['oracle_carry_mae']):reasons.append('ORACLE_PRACTICAL_GATE')
    if ygain<max(.03,.01*incumbent['ypc_mae']):reasons.append('YPC_PRACTICAL_GATE')
    if bootstrap['ci95'][1]>=0:reasons.append('BLOCK_BOOTSTRAP_GATE')
    if any(candidate['miss_rate'][str(t)]-incumbent['miss_rate'][str(t)]>.01+1e-12 for t in (40,50)):reasons.append('CATASTROPHIC_GUARD')
    if family in ('D_explosive','E_states'):
        keys=('ten','twenty') if family=='D_explosive' else ('negative','success','ten')
        claimed=mean([candidate['component_rates'][k]['mae'] for k in keys]);baseline=mean([candidate['component_rates'][k]['baseline_recent5_mae'] for k in keys])
        if claimed>.99*baseline:reasons.append('CLAIMED_COMPONENT_GATE')
    if family in ('F_location','G_context'):
        k='location' if family=='F_location' else 'context'
        if candidate[k+'_mixture_mae']>.99*candidate['league_'+k+'_mixture_mae']:reasons.append('CLAIMED_COMPONENT_GATE')
    return {'passed':not reasons,'reasons':reasons,'oracle_gain_yards':gain,'ypc_gain':ygain}


def slices(rows,predictions,family=None):
    rules={'RB':lambda r:r['position']=='RB','FB_HB':lambda r:r['position'] in {'FB','HB'},'early_w1_4':lambda r:r['week']<=4,'established':lambda r:r['week']>4,'high_prior_role':lambda r:r['prior_current_team_mean_carries']>=15,'low_prior_role':lambda r:r['prior_current_team_mean_carries']<15}
    return {name:metrics([r for r in rows if rule(r)],[p for r,p in zip(rows,predictions) if rule(r)],family) for name,rule in rules.items()}


def develop(directory,audit):
    protocol=json.loads((ART/'phase1k_rushing_protocol.json').read_text())
    rows=assemble(directory,2024,1,18);early=[r for r in rows if r['week']<=8];late=[r for r in rows if r['week']>=9]
    if sum(r['actual_carries']>0 for r in early)<150:raise ValueError('Not enough fixed training rows')
    base=metrics(late,[r['comparators']['phase1f_incumbent'] for r in late]);families={};models={}
    for family in FAMILIES:
        required={'F_location':['run_location'],'G_context':['down','ydstogo','yardline_100','shotgun','no_huddle']}.get(family,[])
        if any(audit['seasons'][str(y)]['fields'][field]['status']!='USABLE_PRIOR_ONLY' for y in (2023,2024) for field in required):
            families[family]={'status':'BLOCKED_DATA','gate':{'passed':False,'reasons':['SOURCE_COVERAGE_GATE']}};continue
        if family=='C_player_defense' and not all(families[k]['gate']['passed'] for k in ('A_player_profile','B_defense_allowance')):
            families[family]={'status':'NOT_RUN_FAILED_PARENT_FAMILIES','gate':{'passed':False,'reasons':['FAILED_FAMILY_COMBINATION_FORBIDDEN']}};continue
        options=[]
        for alpha in protocol['fitting']['ridge_alpha_grid']:
            model=fit(early,family,alpha);pred=[predict(r,model) for r in late];ev=metrics(late,pred,family)
            options.append((ev['oracle_carry_mae'],ev['ypc_mae'],alpha,model,ev,pred))
        chosen=min(options,key=lambda x:x[:3]);_,_,alpha,model,ev,pred=chosen
        boot=blocked_bootstrap(late,pred);decision=gate(ev,base,boot,family)
        families[family]={'status':'SURVIVES_DEVELOPMENT' if decision['passed'] else 'FROZEN_REJECTED','alpha':alpha,'metrics':ev,'bootstrap':boot,'gate':decision,'slice_metrics':slices(late,pred,family),'alpha_selection_metrics':[{ 'alpha':v[2],'metrics':v[4]} for v in options]}
        models[family]=model
    survivors=[k for k,v in families.items() if v['gate']['passed']]
    selected=min(survivors,key=lambda k:(families[k]['metrics']['oracle_carry_mae'],families[k]['metrics']['ypc_mae'],k)) if survivors else None
    if selected:models['refit_selected']=fit(rows,selected,families[selected]['alpha'])
    return {'schema':'nfl-v2-phase1k-development-lock-v1','protocol_sha256':S.sha(ART/'phase1k_rushing_protocol.json'),'source_audit_sha256':digest(audit),'base_head':BASE_HEAD,'fit_period':'2024 W1-8','selection_period':'2024 W9-18','validation_performance_accessed':False,'diagnostic_performance_accessed':False,'frozen_specifications':models,'families':families,'selected':selected,'development_survivors':survivors,'development_cohort_digest':digest([(r['game_id'],r['player_id']) for r in rows]),'fit_positive_rows':sum(r['actual_carries']>0 for r in early),'selection_positive_rows':base['positive_carry_rows'],'selection_comparators':{k:metrics(late,[r['comparators'][k] for r in late]) for k in late[0]['comparators']},'confirmation_status':'READY_FROZEN_SPECIFICATION' if selected else 'NOT_RUN_DEVELOPMENT_GATE_FAILED'}


def oracle_diagnostics(rows,predictions):
    """All postgame covariates are isolated here, after frozen evaluation."""
    positive=[(r,p) for r,p in zip(rows,predictions) if r['actual_carries']>0 and r['label_reconciliation']['pbp_carries']==r['actual_carries']]
    out={'label':'POSTGAME_ORACLE_DIAGNOSTIC_ONLY','not_used_for_fit_selection_promotion':True,'n':len(positive)}
    for k in ('location','context'):
        vals=[]
        for r,p in positive:
            c=r['components'];mix=r['oracle_metadata'][k+'_mixture'];oracle=sum(v*c[k+'_yields'][name] for name,v in mix.items())
            vals.append(abs(oracle*r['actual_carries']-r['actual_rushing_yards']))
        out['actual_'+k+'_mixture_prior_yield_mae']=mean(vals)
    tail=[]
    for r,p in positive:
        q=r['components']['tail'];point=sum(n*m for n,m in zip(r['oracle_metadata']['tail_counts'],q['conditional_yards']));tail.append(abs(point-r['actual_rushing_yards']))
    out['actual_tail_occurrence_prior_conditional_yields_mae']=mean(tail)
    out['pregame_incumbent_same_diagnostic_rows_mae']=mean([abs(r['comparators']['phase1f_incumbent']*r['actual_carries']-r['actual_rushing_yards']) for r,p in positive])
    return out


def ledger(rows,models,period):
    for r in rows:
        predictions={k:predict(r,m) for k,m in models.items() if k!='refit_selected'}
        for name,ypc in {**r['comparators'],**predictions}.items():
            c=r['components'];p=c['player_profile'];d=c['defense_profile'];ao=r['actual_carries'];y=r['actual_rushing_yards']
            yield {k:v for k,v in {**r,'period':period,'architecture':name,'baseline_player_ypc':p['ypc'],'player_negative_rate':p['negative'],'player_success_rate':p['success'],'player_explosive10_rate':p['ten'],'player_explosive20_rate':p['twenty'],'defense_ypc_allowed':d['ypc'],'defense_success_rate_allowed':d['success'],'defense_explosive10_rate_allowed':d['ten'],'defense_explosive20_rate_allowed':d['twenty'],'predicted_run_location_mixture':c['location_mixture'] if name=='F_location' else None,'predicted_context_mixture':c['context_mixture'] if name=='G_context' else None,'predicted_carries':None,'final_predicted_ypc':ypc,'oracle_carry_rushing_yards_projection':ao*ypc,'full_direct_projection':None,'full_projection_status':'NOT_RUN_INDEPENDENT_EFFICIENCY_GATE','error_decomposition':{'efficiency_error_ypc':ypc-r['actual_ypc'] if ao>0 else None,'oracle_signed_yards_error':ao*ypc-y,'workload_error':None}}.items() if k not in {'vectors'}}


def save_receipts(path,items):
    # Zero timestamp and no embedded path make repeated ledger bytes identical.
    with open(path,'wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as z:
            for item in items:z.write((json.dumps(canonical(item),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())


def conclude_development(directory,lock,receipts):
    if lock['selected'] is not None:raise ValueError('A development survivor requires frozen confirmation, not development-only conclusion')
    rows=assemble(directory,2024,9,18);models=lock['frozen_specifications'];preds={k:[predict(r,m) for r in rows] for k,m in models.items()}
    save_receipts(receipts,ledger(rows,models,'selection_2024_w9_18'))
    return {'schema':'nfl-v2-phase1k-results-v1','verdict':'REJECTED_EFFICIENCY_REPLACEMENT','selected':None,'development_lock_sha256':digest(lock),'family_decisions':{k:'BLOCKED_DATA' if lock['families'][k]['status']=='BLOCKED_DATA' else 'REJECTED' for k in FAMILIES},'family_status_detail':{k:v['status'] for k,v in lock['families'].items()},'periods':{'development_2024_w9_18':{'comparators':lock['selection_comparators'],'families':{k:v.get('metrics') for k,v in lock['families'].items()}},'validation_2025':{'status':'NOT_RUN_DEVELOPMENT_GATE_FAILED'},'diagnostic_2026_wk1_4':{'status':'NOT_RUN_DEVELOPMENT_GATE_FAILED'}},'full_projection_evaluation':{'status':'NOT_RUN_INDEPENDENT_EFFICIENCY_GATE'},'oracle_diagnostics':oracle_diagnostics(rows,preds['D_explosive']),'receipt_ledger_sha256':S.sha(receipts),'receipt_rows':len(rows)*(len(models)+len(rows[0]['comparators'])),'population_reconciliation':{'fixed_candidates':len(rows),'positive_carry_rows':sum(r['actual_carries']>0 for r in rows),'pbp_official_mismatch_rows':sum(r['label_reconciliation']['pbp_carries']!=r['actual_carries'] or r['label_reconciliation']['pbp_yards']!=r['actual_rushing_yards'] for r in rows)},'2025_model_performance_accessed':False,'2026_model_performance_accessed':False,'week5_plus_accessed':False,'sportsbook_inputs_used':False,'monte_carlo_used':False,'receiving_status':'FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION','receiver_depth_status':'SURVIVED_SIGNAL_NOT_PROMOTED'}


def confirm(directory,lock,receipts):
    if lock['protocol_sha256']!=S.sha(ART/'phase1k_rushing_protocol.json'):raise ValueError('Protocol drift')
    if lock['selected'] is None:return conclude_development(directory,lock,receipts)
    # This phase intentionally stops on development failure. A survivor path is
    # implemented below without allowing any validation-based architecture choice.
    family=lock['selected'];model=lock['frozen_specifications']['refit_selected'];periods={};accepted=True;allreceipts=[]
    for season,weekmax,label in ((2025,18,'validation_2025'),(2026,4,'diagnostic_2026_wk1_4')):
        rows=assemble(directory,season,1,weekmax);pred=[predict(r,model) for r in rows];ev=metrics(rows,pred,family);base=metrics(rows,[r['comparators']['phase1f_incumbent'] for r in rows]);boot=blocked_bootstrap(rows,pred)
        decision=gate(ev,base,boot,family) if season==2025 else {'passed':ev['oracle_carry_mae']<=1.05*base['oracle_carry_mae'] and ev['ypc_mae']<=1.05*base['ypc_mae'] and all(ev['miss_rate'][str(t)]<=base['miss_rate'][str(t)]+.03 for t in (40,50)),'reasons':[]}
        comp={k:metrics(rows,[r['comparators'][k] for r in rows]) for k in rows[0]['comparators']}
        sliced=slices(rows,pred,family);base_sliced=slices(rows,[r['comparators']['phase1f_incumbent'] for r in rows])
        if season==2025 and (any(ev['oracle_carry_mae']>=v['oracle_carry_mae'] for v in comp.values()) or any(v['positive_carry_rows']>=40 and v['oracle_carry_mae']>1.1*base_sliced[k]['oracle_carry_mae'] for k,v in sliced.items())):
            decision['passed']=False;decision['reasons'].append('STRONGEST_COMPARATOR_OR_SLICE_GUARD')
        accepted &= decision['passed'];periods[label]={'metrics':ev,'comparators':comp,'bootstrap':boot,'decision':decision,'slices':sliced,'oracle_diagnostics':oracle_diagnostics(rows,pred)};allreceipts.extend(ledger(rows,{family:model},label))
    if accepted:
        # No speculative workload/roster source is permitted: exact frozen E
        # receipts must exist before full reintroduction. No fake carry allocator.
        full={'status':'BLOCKED_FROZEN_PHASE1E_FULL_RECEIPTS_REQUIRED','reason':'No licensed/as-of target roster input may be invented; independent efficiency pass preserved, full projection not evaluated.'}
    else:full={'status':'NOT_RUN_INDEPENDENT_EFFICIENCY_GATE'}
    save_receipts(receipts,allreceipts)
    return {'schema':'nfl-v2-phase1k-results-v1','verdict':'SURVIVES_RUSHING_EFFICIENCY_REPLACEMENT' if accepted else 'REJECTED_EFFICIENCY_REPLACEMENT','selected':family,'periods':periods,'family_decisions':{k:('SURVIVES' if k==family and accepted else 'REJECTED') for k in FAMILIES},'full_projection_evaluation':full,'receipt_ledger_sha256':S.sha(receipts),'receipt_rows':len(allreceipts),'2025_model_performance_accessed':True,'2026_model_performance_accessed':True,'week5_plus_accessed':False,'sportsbook_inputs_used':False,'monte_carlo_used':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--stage',choices=['develop','confirm'],required=True);ap.add_argument('--audit',default=str(ART/'phase1k_pbp_source_audit.json'));ap.add_argument('--lock',default=str(ART/'phase1k_development_lock.json'));ap.add_argument('--out');ap.add_argument('--receipts');a=ap.parse_args()
    audit=json.loads(Path(a.audit).read_text())
    if audit!=json.loads((ART/'phase1k_pbp_source_audit.json').read_text()):raise ValueError('Source audit drift')
    if a.stage=='develop':
        result=develop(a.data_dir,audit);write_json(a.lock,result);print(json.dumps({'selected':result['selected'],'families':{k:v['status'] for k,v in result['families'].items()}},indent=2))
    else:
        if not a.out or not a.receipts:ap.error('confirm requires --out and --receipts')
        lock=json.loads(Path(a.lock).read_text());result=confirm(a.data_dir,lock,a.receipts);write_json(a.out,result);print(result['verdict'])


if __name__=='__main__':main()

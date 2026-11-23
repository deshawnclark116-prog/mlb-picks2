#!/usr/bin/env python3
"""Independent, research-only football yardage challenger using pregame history.

No sportsbook variables in any challenger. The only sportsbook-based comparator
is the historical live v2 model, kept unmodified for fair comparison.
2025 W6–2026 W11 is REUSED development data, not a pristine holdout.
"""
import argparse, csv, gzip, json, math, sqlite3, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nfl_yardage_v2 as v2
import challenger as ch

SEED = 20261122
BASE = ch.BASE
ROLE = ch.ROLE
OLDQUALITY = ch.QUALITY
PROPS = [
    'pbp_ypt_8', 'pbp_air_3', 'pbp_air_8', 'pbp_air_trend',
    'pbp_redzone_target_8','pbp_third_target_8','pbp_target_epa_8','pbp_target_comp_8',
    'pbp_rush_epa_8','pbp_rush_redzone_8','pbp_rush_third_8','pbp_rush_explosive_8',
    'pbp_rush_stuffed_8',
]
FTN = [
    'ftn_catchable_8','ftn_primary_read_8','ftn_designed_read_8',
    'ftn_screen_8','ftn_play_action_8','ftn_contested_8','ftn_drop_8',
    'ftn_blitz_recv_8','ftn_catchable_trend','ftn_screen_trend',
    'ftn_rush_box_8','ftn_rush_box_trend','ftn_rush_motion_8',
    'ftn_rush_shotgun_8','ftn_rush_rpo_8',
]
TEAM = [
    'tm_plays_5','tm_pass_rate_5','tm_run_ypc_5','tm_pass_ypt_5',
    'tm_epa_per_play_5','tm_redzone_plays_5','tm_pass_rate_trend',
    'opp_pass_ypt_5','opp_rush_ypc_5','opp_pass_epa_5','opp_rush_epa_5',
    'opp_blitz_rate_5', 'opp_box_5', 'team_rz_trend',
]
DEPTH = ['other_last3_team_vol','other_top1_team_vol','other_top2_team_vol',
         'other_active_reg_count','other_rank','team_own_last3_vol_share',
         'team_last3_total_vol','team_last3_total_trend',
         'other_competition_last1']
INJ = ['injury_status','injury_practice','snap_last3','snap_recent_trend',
       'teammate_out_opportunity','teammate_questionable_opportunity']

# Each entry must contain NO spread, total, odds, or price input.
FEATURES = {
    'role_team': BASE+ROLE+TEAM+DEPTH,
    'role_team_inj': BASE+ROLE+TEAM+DEPTH+INJ,
    'role_team_quality': BASE+ROLE+TEAM+DEPTH+PROPS,
    'role_team_ftn': BASE+ROLE+TEAM+DEPTH+PROPS+FTN,
    'role_team_ftn_inj': BASE+ROLE+TEAM+DEPTH+PROPS+FTN+INJ,
    'market_free_team': BASE+TEAM+DEPTH,
    'ftn_only': BASE+TEAM+PROPS+FTN,
    'old_quality_context': BASE+ROLE+OLDQUALITY+TEAM+DEPTH,
}
assert all(not any(x in feature for x in ('spread','total_line','implied_team','odds','price'))
           for fs in FEATURES.values() for feature in fs)

CFG = {}
for name,cols in FEATURES.items():
    CFG[name] = {'cols':cols,'params':dict(v2.PARAMS,nthread=3), 'rounds':140}
CFG['role_team_depth3'] = dict(cols=FEATURES['role_team'],
     params=dict(v2.PARAMS,nthread=3,max_depth=3,min_child_weight=15), rounds=185)
CFG['role_team_depth5'] = dict(cols=FEATURES['role_team'],
     params=dict(v2.PARAMS,nthread=3,max_depth=5,min_child_weight=15,reg_lambda=10),rounds=140)
CFG['role_team_huber'] = dict(cols=FEATURES['role_team'],
     params=dict(v2.PARAMS,nthread=3,objective='reg:pseudohubererror',huber_slope=10.0),rounds=170)
CFG['role_team_recent'] = dict(cols=FEATURES['role_team'],
     params=dict(v2.PARAMS,nthread=3),rounds=140,half_life=38)
CFG['full_depth3'] = dict(cols=FEATURES['role_team_ftn_inj'],
     params=dict(v2.PARAMS,nthread=3,max_depth=3,min_child_weight=15),rounds=185)
CFG['full_recent'] = dict(cols=FEATURES['role_team_ftn_inj'],
     params=dict(v2.PARAMS,nthread=3),rounds=140,half_life=38)


def num(v):
    try:
        k=float(v)
        return k if math.isfinite(k) else 0.
    except (ValueError,TypeError): return 0.


def boolean(v):
    return int(str(v).lower() in ('true','t','1','yes'))


def mdiv(x,y):
    return float(x/y) if y>0 else float('nan')


def rates(entries, names, count):
    """Sum numerator and denominator over last games, never average tiny-game ratios."""
    recent=entries[-count:]
    a=defaultdict(float)
    for item in recent:
        for k,v in item.items():a[k]+=v
    return a


def read_ftn(directory):
    ftn={};coverage={}
    for season in (2023,2024,2025,2026):
        p=Path(directory)/f'ftn_charting_{season}.csv'
        if not p.exists():raise FileNotFoundError(f'FTN charting source absent: {p}')
        n=0
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                k=(row.get('nflverse_game_id',''),round(num(row.get('nflverse_play_id'))))
                if k[0]:ftn[k]=row;n+=1
        coverage[season]=n
    return ftn,coverage


def load_pbp(pbpdir, ftndir):
    """Preaggregate per-player/per-team/per-defense true historical week games."""
    ftn,cover=read_ftn(ftndir)
    players=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    offense=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    defense=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    match=0
    for season in (2023,2024,2025,2026):
        p=Path(pbpdir)/f'play_by_play_{season}.csv.gz'
        with gzip.open(p,'rt',newline='') as f:
            for row in csv.DictReader(f):
                if row.get('season_type')!='REG':continue
                week=round(num(row.get('week')))
                if week<1 or week>18:continue
                key=(season,week)
                ispass=num(row.get('pass_attempt'))==1
                isrush=num(row.get('rush_attempt'))==1
                if not ispass and not isrush:continue
                team=row.get('posteam','')
                opp=row.get('defteam','')
                air=num(row.get('air_yards'))
                yac=num(row.get('yards_after_catch'))
                epa=num(row.get('epa'))
                pass_y=num(row.get('passing_yards'))
                rush_y=num(row.get('rushing_yards'))
                zone=num(row.get('yardline_100'))
                ftn_row=ftn.get((row.get('game_id',''),round(num(row.get('play_id')))))
                if ftn_row:match+=1
                ppid=row.get('receiver_player_id','') if ispass else row.get('rusher_player_id','')
                if ppid and ppid!='NA':
                    a=players[key][ppid]
                    if ispass:
                        a['target']+=1
                        a['target_yards']+=pass_y
                        a['target_air']+=air
                        a['target_comp']+=num(row.get('complete_pass'))
                        a['target_epa']+=epa
                        a['target_rz']+=int(0<zone<=20)
                        a['target_third']+=int(round(num(row.get('down')))==3)
                        if ftn_row:
                            a['ftn_target']+=1
                            a['ftn_catchable']+=boolean(ftn_row.get('is_catchable_ball'))
                            a['ftn_contested']+=boolean(ftn_row.get('is_contested_ball'))
                            a['ftn_drop']+=boolean(ftn_row.get('is_drop'))
                            a['ftn_screen']+=boolean(ftn_row.get('is_screen_pass'))
                            a['ftn_pa']+=boolean(ftn_row.get('is_play_action'))
                            a['ftn_primary']+=int(str(ftn_row.get('read_thrown','')).strip()=='0')
                            a['ftn_designed']+=int(str(ftn_row.get('read_thrown','')).strip()=='DES')
                            a['ftn_blitz_target']+=int(num(ftn_row.get('n_blitzers'))>0)
                    if isrush:
                        a['rush']+=1
                        a['rush_yards']+=rush_y
                        a['rush_epa']+=epa
                        a['rush_stuffed']+=int(rush_y<=0)
                        a['rush_explosive']+=int(rush_y>=15)
                        a['rush_rz']+=int(0<zone<=20)
                        a['rush_third']+=int(round(num(row.get('down')))==3)
                        if ftn_row:
                            a['ftn_rush']+=1
                            a['ftn_rush_box']+=num(ftn_row.get('n_defense_box'))
                            a['ftn_motion']+=boolean(ftn_row.get('is_motion'))
                            a['ftn_shotgun']+=int(str(ftn_row.get('qb_location',''))=='S')
                            a['ftn_rpo']+=boolean(ftn_row.get('is_rpo'))
                if team and team!='NA':
                    a=offense[key][team]
                    a['plays']+=1;a['passes']+=ispass;a['rushes']+=isrush
                    a['pass_yards']+=pass_y if ispass else 0.
                    a['rush_yards']+=rush_y if isrush else 0.
                    a['epa']+=epa
                    a['redzone']+=int(0<zone<=20)
                if opp and opp!='NA':
                    a=defense[key][opp]
                    a['passes']+=ispass;a['rushes']+=isrush
                    a['pass_yards']+=pass_y if ispass else 0.
                    a['rush_yards']+=rush_y if isrush else 0.
                    a['pass_epa']+=epa if ispass else 0.
                    a['rush_epa']+=epa if isrush else 0.
                    if ftn_row and ispass:
                        a['ftn_pass']+=1;a['ftn_blitz']+=int(num(ftn_row.get('n_blitzers'))>0)
                    if ftn_row and isrush:
                        a['ftn_rush']+=1;a['ftn_box']+=num(ftn_row.get('n_defense_box'))
    print('FTN downloaded rows',cover,'joined plays',match,flush=True)
    return players,offense,defense,{'ftn_rows':cover,'ftn_joined_plays':match}


def hist_features(p_hist,tm_hist,opp_hist):
    """Previous games only: historical aggregates and sample sizes."""
    t5=rates(p_hist, ['unused'],5);t8=rates(p_hist,[],8)
    t3=rates(p_hist,[],3)
    off5=rates(tm_hist,[],5);off3=rates(tm_hist,[],3);off8=rates(tm_hist,[],8)
    opp5=rates(opp_hist,[],5)
    f={
      'pbp_ypt_8':mdiv(t8['target_yards'],t8['target']),
      'pbp_air_3':mdiv(t3['target_air'],t3['target']),
      'pbp_air_8':mdiv(t8['target_air'],t8['target']),
      'pbp_air_trend':mdiv(t3['target_air'],t3['target'])-mdiv(t8['target_air'],t8['target']),
      'pbp_redzone_target_8':mdiv(t8['target_rz'],t8['target']),
      'pbp_third_target_8':mdiv(t8['target_third'],t8['target']),
      'pbp_target_epa_8':mdiv(t8['target_epa'],t8['target']),
      'pbp_target_comp_8':mdiv(t8['target_comp'],t8['target']),
      'pbp_rush_epa_8':mdiv(t8['rush_epa'],t8['rush']),
      'pbp_rush_redzone_8':mdiv(t8['rush_rz'],t8['rush']),
      'pbp_rush_third_8':mdiv(t8['rush_third'],t8['rush']),
      'pbp_rush_explosive_8':mdiv(t8['rush_explosive'],t8['rush']),
      'pbp_rush_stuffed_8':mdiv(t8['rush_stuffed'],t8['rush']),
      'ftn_catchable_8':mdiv(t8['ftn_catchable'],t8['ftn_target']),
      'ftn_primary_read_8':mdiv(t8['ftn_primary'],t8['ftn_target']),
      'ftn_designed_read_8':mdiv(t8['ftn_designed'],t8['ftn_target']),
      'ftn_screen_8':mdiv(t8['ftn_screen'],t8['ftn_target']),
      'ftn_play_action_8':mdiv(t8['ftn_pa'],t8['ftn_target']),
      'ftn_contested_8':mdiv(t8['ftn_contested'],t8['ftn_target']),
      'ftn_drop_8':mdiv(t8['ftn_drop'],t8['ftn_target']),
      'ftn_blitz_recv_8':mdiv(t8['ftn_blitz_target'],t8['ftn_target']),
      'ftn_catchable_trend':mdiv(t3['ftn_catchable'],t3['ftn_target'])-mdiv(t8['ftn_catchable'],t8['ftn_target']),
      'ftn_screen_trend':mdiv(t3['ftn_screen'],t3['ftn_target'])-mdiv(t8['ftn_screen'],t8['ftn_target']),
      'ftn_rush_box_8':mdiv(t8['ftn_rush_box'],t8['ftn_rush']),
      'ftn_rush_box_trend':mdiv(t3['ftn_rush_box'],t3['ftn_rush'])-mdiv(t8['ftn_rush_box'],t8['ftn_rush']),
      'ftn_rush_motion_8':mdiv(t8['ftn_motion'],t8['ftn_rush']),
      'ftn_rush_shotgun_8':mdiv(t8['ftn_shotgun'],t8['ftn_rush']),
      'ftn_rush_rpo_8':mdiv(t8['ftn_rpo'],t8['ftn_rush']),
      'tm_plays_5':mdiv(off5['plays'],len(tm_hist[-5:])),
      'tm_pass_rate_5':mdiv(off5['passes'],off5['plays']),
      'tm_run_ypc_5':mdiv(off5['rush_yards'],off5['rushes']),
      'tm_pass_ypt_5':mdiv(off5['pass_yards'],off5['passes']),
      'tm_epa_per_play_5':mdiv(off5['epa'],off5['plays']),
      'tm_redzone_plays_5':mdiv(off5['redzone'],len(tm_hist[-5:])),
      'tm_pass_rate_trend':mdiv(off3['passes'],off3['plays'])-mdiv(off8['passes'],off8['plays']),
      'opp_pass_ypt_5':mdiv(opp5['pass_yards'],opp5['passes']),
      'opp_rush_ypc_5':mdiv(opp5['rush_yards'],opp5['rushes']),
      'opp_pass_epa_5':mdiv(opp5['pass_epa'],opp5['passes']),
      'opp_rush_epa_5':mdiv(opp5['rush_epa'],opp5['rushes']),
      'opp_blitz_rate_5':mdiv(opp5['ftn_blitz'],opp5['ftn_pass']),
      'opp_box_5':mdiv(opp5['ftn_box'],opp5['ftn_rush']),
      'team_rz_trend':mdiv(off3['redzone'],len(tm_hist[-3:]))-mdiv(off8['redzone'],len(tm_hist[-8:])),
    }
    return f


def build_pbp_asof(rows, pbpdir, ftndir):
    player,offense,defense,quality=load_pbp(pbpdir,ftndir)
    byweek=defaultdict(list)
    for i,r in enumerate(rows):byweek[(r[0]['season'],r[0]['week'])].append(i)
    keys=sorted(set(byweek)|set(player)|set(offense)|set(defense))
    p_hist=defaultdict(list);off_hist=defaultdict(list);def_hist=defaultdict(list)
    for key in keys:
        for idx in byweek.get(key,[]):
            meta,fe,y=rows[idx]
            fe.update(hist_features(p_hist[meta['player_id']],off_hist[meta['team']],def_hist[meta['opp']]))
        for pid,stats in player.get(key,{}).items():p_hist[pid].append(stats)
        for team,stats in offense.get(key,{}).items():off_hist[team].append(stats)
        for team,stats in defense.get(key,{}).items():def_hist[team].append(stats)
    return quality


def build_competition(con,market,rows):
    vol=v2.MARKETS[market]['vol'];positions=v2.MARKETS[market]['positions']
    allp=con.execute(f'''SELECT player_id,team,position,season,week,COALESCE({vol},0)
                            FROM player_games WHERE season_type='REG' ORDER BY season,week''').fetchall()
    eligible=defaultdict(list)
    for meta,f,y in rows:eligible[(meta['season'],meta['week'])].append((meta,f))
    byweek=defaultdict(list)
    for p,t,pos,s,w,v in allp:byweek[(s,w)].append((p,t,pos,float(v)))
    keys=sorted(byweek)
    h=defaultdict(list); latest_team={}; lastseen={}; week_num={k:i for i,k in enumerate(keys)}
    for key in keys:
        now=week_num[key]
        for meta,f in eligible[key]:
            me=meta['player_id'];team=meta['team'];mine=h[me][-3:]
            v_me=ch.mean([x[1] for x in mine]) if mine else 0.
            last1_me=mine[-1][1] if mine else 0.
            c=[];old_total=0.
            for pid,tm in latest_team.items():
                if tm!=team or not h[pid] or now-lastseen.get(pid, -99)>4:continue
                pos=h[pid][-1][2]
                if pos not in positions:continue
                recent=[v for t,v,po in h[pid][-3:] if t==team]
                if not recent:continue
                av=sum(recent)/len(recent)
                c.append((pid,av,h[pid][-1][1]))
                old_total+=av
            other=sorted([(pid,a,l) for pid,a,l in c if pid!=me],key=lambda x:x[1],reverse=True)
            n_active=sum(1 for pid,a,l in other if a>0)
            f.update({
                'other_last3_team_vol':sum(a for pid,a,l in other),
                'other_top1_team_vol':other[0][1] if other else 0.,
                'other_top2_team_vol':other[1][1] if len(other)>1 else 0.,
                'other_active_reg_count':n_active,
                'other_rank':1+sum(1 for pid,a,l in other if a>v_me),
                'team_own_last3_vol_share':mdiv(v_me,old_total),
                'team_last3_total_vol':old_total,
                'team_last3_total_trend':mdiv(sum(l for pid,a,l in c),old_total)-1 if old_total else float('nan'),
                'other_competition_last1':(other[0][2]-last1_me) if other else 0.,
            })
        # Only now does realized current-week activity enter the historical player state.
        for pid,team,pos,v in byweek[key]:
            latest_team[pid]=team;lastseen[pid]=now;h[pid].append((team,v,pos))


def load_injuries(injdir):
    statuses=defaultdict(dict)
    status_map={'Out':3,'Doubtful':2,'Questionable':1}
    practice_map={'Did Not Participate In Practice':2,'Limited Participation In Practice':1,'Full Participation In Practice':0}
    counts={}
    for season in (2023,2024,2025,2026):
        p=Path(injdir)/f'injuries_{season}.csv'
        if not p.exists():raise FileNotFoundError(p)
        n=0
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                if row.get('game_type')!='REG':continue
                pid=row.get('gsis_id') or row.get('player_id')
                w=round(num(row.get('week')))
                if not pid or w<=0:continue
                k=(pid,season,w)
                status=status_map.get(row.get('report_status'),0)
                pc=practice_map.get(row.get('practice_status'),-1)
                old=statuses.get(k,{'injury_status':0,'injury_practice':-1})
                statuses[k]={'injury_status':max(status,old['injury_status']),
                           'injury_practice':max(pc,old['injury_practice'])};n+=1
        counts[season]=n
    return statuses,counts


def build_injuries(con,market,rows,injdir):
    statuses,cnt=load_injuries(injdir)
    # Legacy snap history from PREVIOUS WEEKS only (never current-game snaps).
    snhist=defaultdict(list);snap_week=defaultdict(list)
    for season in (2023,2024,2025,2026):
        p=Path(injdir)/f'snap_counts_{season}.csv'
        if not p.exists():raise FileNotFoundError(p)
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                if row.get('game_type')!='REG':continue
                nm=v2.norm_name(row.get('player') or row.get('player_name') or '')
                w=round(num(row.get('week')))
                if nm and w>0:
                    snap_week[(season,w)].append((nm,num(row.get('offense_pct'))))
    byweek=defaultdict(list)
    for i,r in enumerate(rows):byweek[(r[0]['season'],r[0]['week'])].append(i)
    # Injury opponent teammate risk: only the team and player history available before kickoff.
    vol=v2.MARKETS[market]['vol']
    played=con.execute(f'''SELECT player_id,team,position,season,week,coalesce({vol},0)
                          FROM player_games WHERE season_type='REG' ORDER BY season,week''').fetchall()
    p_hist=defaultdict(list);team_mem=defaultdict(dict);allweeks=defaultdict(list)
    for pid,team,pos,s,w,v in played:allweeks[(s,w)].append((pid,team,pos,float(v)))
    keys=sorted(set(byweek)|set(allweeks))
    for key in keys:
        for i in byweek.get(key,[]):
            m,f,y=rows[i]
            pid=m['player_id'];nm=v2.norm_name(m['name']);team=m['team']
            f.update(statuses.get((pid,*key),{'injury_status':0,'injury_practice':-1}))
            sh=snhist[nm]; f['snap_last3']=ch.mean(sh[-3:]);
            f['snap_recent_trend']=sh[-1]-ch.mean(sh[-3:]) if sh else float('nan')
            outs=quest=0.
            for mate,(tm,position,hist) in team_mem.get(team,{}).items():
                if mate==pid or position not in v2.MARKETS[market]['positions']:continue
                av=ch.mean(hist[-3:]) if hist else 0.
                if not math.isfinite(av):continue
                stat=statuses.get((mate,*key),{}).get('injury_status',0)
                if stat>=2:outs+=av
                elif stat==1:quest+=av
            f['teammate_out_opportunity']=outs
            f['teammate_questionable_opportunity']=quest
        for nm,share in snap_week.get(key,[]):snhist[nm].append(share)
        fresh=defaultdict(dict)
        for pid,team,pos,v in allweeks.get(key,[]):
            hist=p_hist[pid];hist.append(v)
            if v>0:fresh[team][pid]=(team,pos,hist)
        for team,mem in fresh.items():team_mem[team]=mem
    return cnt


def matrix(rows,cols):
    return np.asarray([[r[1].get(c,np.nan) if r[1].get(c) is not None else np.nan
                       for c in cols] for r in rows],dtype=np.float32)


def bootstrap(gains,groups,seed=SEED,reps=1500):
    rng=np.random.default_rng(seed);n=len(gains)
    # Use 1D batches, not huge reps x n memory.
    row=[]
    for _ in range(reps):row.append(float(np.mean(gains[rng.integers(n,size=n)])))
    blocks=[np.where(groups==g)[0] for g in sorted(set(groups))]
    week=[]
    for _ in range(reps):
        sample=rng.integers(len(blocks),size=len(blocks))
        q=np.concatenate([blocks[i] for i in sample])
        week.append(float(np.mean(gains[q])))
    return {'p_better_row':round(float(np.mean(np.asarray(row)>0)),4),
            'p_better_week':round(float(np.mean(np.asarray(week)>0)),4),
            'ci95_row':np.quantile(row,[.025,.975]).round(4).tolist(),
            'ci95_week':np.quantile(week,[.025,.975]).round(4).tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='nfl_models/nfl_model.sqlite')
    p.add_argument('--pbp-dir',required=True)
    p.add_argument('--feature-dir',required=True)
    p.add_argument('--out',default='research/nfl_model_tests/ftn_v2_result.json')
    p.add_argument('--max-candidates',type=int,default=0)
    p.add_argument('--market', choices=('both','rushing_yards','receiving_yards'), default='both')
    args=p.parse_args()
    con=sqlite3.connect(f'file:{Path(args.db).resolve()}?mode=ro',uri=True)
    results={'protocol':'strict weekly walk-forward 2025 W6–2026 W11; burned DEVELOPMENT set',
             'no_sportsbook_challenger':True,'versions':{},'markers':{},'markets':{}}
    for market,expected in (('rushing_yards',1119),('receiving_yards',2956)):
        if args.market != 'both' and market != args.market:
            continue
        print(market,'building feature rows',flush=True)
        replay=v2.History(con,market)
        rows,_,_=replay.replay()
        role=ch.build_role(con,market)
        for meta,fe,y in rows:
            key=(meta['season'],meta['week'])
            fe.update(role.get((meta['player_id'],key),{}))
        ftn_coverage=build_pbp_asof(rows,args.pbp_dir,args.feature_dir)
        build_competition(con,market,rows)
        injury_coverage=build_injuries(con,market,rows,args.feature_dir)
        results['markers'][market]={'ftn':ftn_coverage,'injuries':injury_coverage}
        byweek=defaultdict(list)
        for r in rows:byweek[(r[0]['season'],r[0]['week'])].append(r)
        variants={'live_v2':dict(cols=v2.FEATURES,params=dict(v2.PARAMS,nthread=3),rounds=140)}
        variants.update(CFG if args.max_candidates<=0 else dict(list(CFG.items())[:args.max_candidates]))
        pred=defaultdict(list);valid=[];groups=[]
        residuals=defaultdict(list);scores=defaultdict(lambda:defaultdict(list)); train=[]
        for i,key in enumerate(sorted(byweek)):
            sample=byweek[key]
            test=(2025,6)<=key<=(2025,18) or (key[0]==2026 and key[1]<=11)
            if key>=(2024,10) and len(train)>=300:
                ytr=np.asarray([r[2] for r in train],np.float32)
                obs=np.asarray([r[2] for r in sample],np.float32)
                tkeys=np.asarray([r[0]['season']*20+r[0]['week'] for r in train],float)
                weekpred={}
                for name,spec in variants.items():
                    d=xgb.DMatrix(matrix(train,spec['cols']),label=ytr,feature_names=spec['cols'])
                    if 'half_life' in spec:
                        d.set_weight(np.exp(-np.log(2)*(key[0]*20+key[1]-tkeys)/spec['half_life']).clip(.15,1))
                    fit=xgb.train(spec['params'],d,num_boost_round=spec['rounds'])
                    preds=fit.predict(xgb.DMatrix(matrix(sample,spec['cols']),feature_names=spec['cols']))
                    weekpred[name]=preds
                # Ensemble weights derive ONLY from residuals on completed
                # earlier weeks. No evaluated-week results enter this step.
                for mixname,spec in globals().get('ENSEMBLES',{}).items():
                    member=[n for n in spec['members'] if n in weekpred]
                    if len(member)<2:continue
                    arr=np.stack([weekpred[n] for n in member])
                    style=spec.get('style','mean')
                    if style=='median':
                        combo=np.median(arr,axis=0)
                    elif style in ('adaptive_mae','adaptive_cov'):
                        z=min((len(residuals[n]) for n in member),default=0)
                        if z<150:
                            weights=np.ones(len(member))/len(member)
                        else:
                            z=min(z,1400)
                            rr=np.asarray([residuals[n][-z:] for n in member],dtype=float)
                            if style=='adaptive_mae':
                                means=np.mean(np.abs(rr),axis=1)
                                e=np.exp(-(means-np.min(means))/.50)
                                weights=e/e.sum()
                            else:
                                cov=rr@rr.T/rr.shape[1]+12*np.eye(len(member))
                                try:weights=np.linalg.solve(cov,np.ones(len(member)))
                                except np.linalg.LinAlgError:weights=np.ones(len(member))
                                weights=np.clip(weights,.02,None)
                                weights/=weights.sum()
                        combo=weights@arr
                    else:
                        combo=np.mean(arr,axis=0)
                    weekpred[mixname]=np.asarray(combo,dtype=np.float32)
                live=weekpred['live_v2']
                for name,p in weekpred.items():
                    if test:pred[name].extend(p.tolist())
                    if test:
                        for j,(guess,y) in enumerate(zip(p,obs)):
                            z=ch.proper_scores(float(guess),float(y),residuals[name],
                                              fixed_threshold=.7*float(live[j]))
                            if z is not None:
                                scores[name]['crps'].append(z[0]);scores[name]['brier'].append((z[1]-z[2])**2)
                                scores[name]['fixed_brier'].append((z[3]-z[4])**2)
                    residuals[name].extend((obs-p).tolist())
                if test:
                    valid.extend(obs.tolist());groups.extend([str(key) for _ in sample])
            train.extend(sample)
            if i%15==0:print('  ',key,'weeks',i,flush=True)
        if len(valid)!=expected:raise AssertionError(f'{market}: actual eligible {len(valid)} != target {expected}')
        actual=np.asarray(valid,dtype=float);groups=np.asarray(groups)
        rep={'count':len(valid),'baseline':{},'candidates':{}}
        for name,p in pred.items():
            p=np.asarray(p,dtype=float);loss=np.abs(p-actual)
            rep2={'mae':round(float(np.mean(loss)),4),'bias':round(float(np.mean(p-actual)),4),
                  'within10':round(float(np.mean(loss<=10)),4),
                  'crps':round(float(np.mean(scores[name]['crps'])),4),
                  'brier_own_event':round(float(np.mean(scores[name]['brier'])),5),
                  'brier_fixed_v2_event':round(float(np.mean(scores[name]['fixed_brier'])),5)}
            if name=='live_v2':rep['baseline']=rep2
            else:
                gains=np.abs(np.asarray(pred['live_v2'])-actual)-loss
                rep2['improvement_vs_live']=round(float(gains.mean()),4)
                rep2['bootstrap']=bootstrap(gains,groups)
                rep2['beats_prior_best']=bool(rep2['mae'] < (24.86 if market=='rushing_yards' else 22.44))
                rep2['meaningful_ge_0_3']=bool(rep2['improvement_vs_live']>=.3)
                rep['candidates'][name]=rep2
        results['markets'][market]=rep
        print(market,'live',rep['baseline']['mae'],'rank',sorted([(n,r['mae']) for n,r in rep['candidates'].items()],key=lambda x:x[1]),flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(results,indent=2,allow_nan=False))
    print('report',args.out,flush=True)

if __name__=='__main__':main()
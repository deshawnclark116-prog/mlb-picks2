import sys, sqlite3, json
from collections import defaultdict
import numpy as np, xgboost as xgb
sys.path.insert(0,'/home/user/mlb-picks2')
import nfl_yardage_v2 as v2

con=sqlite3.connect('file:/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/nfl_ext.sqlite?mode=ro',uri=True)
import csv
D='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/'
STAT={'Out':3,'Doubtful':2,'Questionable':1}; PRAC={'Did Not Participate In Practice':2,'Limited Participation in Practice':1,'Full Participation in Practice':0}
INJ={}; SNAPW=defaultdict(list)
for y in (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026):
    for r in csv.DictReader(open(D+'injuries_%d.csv'%y)):
        if r['game_type']!='REG': continue
        k=(r['gsis_id'],int(r['season']),int(r['week'])); st=STAT.get(r['report_status'],0); pr=PRAC.get(r['practice_status'],-1)
        o=INJ.get(k,(0,-1)); INJ[k]=(max(o[0],st),max(o[1],pr))
    for r in csv.DictReader(open(D+'snap_counts_%d.csv'%y)):
        if r['game_type']!='REG': continue
        SNAPW[(int(r['season']),int(r['week']))].append((v2.norm_name(r['player']),float(r['offense_pct'] or 0)))
import pickle, datetime
_X=pickle.load(open(D+'../feat_extra.pkl','rb'))
WEATHER=_X['weather']
_pg=_X['pass']; _rg=_X['run']
PASS_BY=defaultdict(list)
for r in _pg.itertuples(): PASS_BY[(r.defteam,r.pos)].append((r.key,r.n,r.y))
RUN_BY=defaultdict(list)
for r in _rg.itertuples(): RUN_BY[r.defteam].append((r.key,r.n,r.y))
for v in list(PASS_BY.values())+list(RUN_BY.values()): v.sort()
def _allowed(lst,key,n_games=8):
    prev=[x for x in lst if x[0]<key][-n_games:]
    n=sum(x[1] for x in prev); y=sum(x[2] for x in prev)
    return (y,n,len(prev))
GDATE={}
import sqlite3 as _s
_c=_s.connect('file:/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/nfl_ext.sqlite?mode=ro',uri=True)
_gm=sorted(_c.execute("select season,week,home_team,away_team,game_date from games where season_type='REG'"))
_last={}
REST={}
for season,week,h,a,gd in _gm:
    try: dt=datetime.date.fromisoformat(str(gd)[:10])
    except Exception: continue
    for t in (h,a):
        REST[(season,week,t)]=(dt-_last[t]).days if t in _last else 14
        _last[t]=dt
import csv as _csv
DEPTH={}
for _y in range(2016,2025):
    for _r in _csv.DictReader(open(D+'depth_%d.csv'%_y)):
        if _r['game_type']!='REG' or _r['formation']!='Offense' or not _r['week'] or not _r['gsis_id']: continue
        if _r['position'] not in ('QB','RB','WR','TE','FB'): continue
        _k=(_r['gsis_id'],int(_r['season']),int(_r['week'])); _d=int(_r['depth_team'] or 9)
        DEPTH[_k]=min(DEPTH.get(_k,9),_d)
NEW=["wind","temp","dome","rest_days","opp_pos_ypt","opp_pos_ypt_vs_lg","opp_pos_tshare","opp_ypc","own_status","own_practice","tm_out_vol","tm_quest_vol","snap3","snap1"]
OLDNEW=["team_vol5","team_other_vol5","vacated_vol","vacated_share","n_regs_absent","qb_changed","team_pass_att5","opp_def_pass_yds5"]

def build(mkt):
    h=v2.History(con,mkt); cfg=h.cfg
    # team passing attempts per game (QB) for context
    pa=defaultdict(float); qb={}; tcar=defaultdict(float)
    for gid_,team_,c_ in con.execute("select game_id,team,coalesce(carries,0) from player_games where season_type='REG'"): tcar[(gid_,team_)]+=c_
    for pid,name,pos,team,opp,season,week,gid,ih,att,py in con.execute("select player_id,player_name,position,team,opponent,season,week,game_id,is_home,coalesce(attempts,0),coalesce(passing_yards,0) from player_games where season_type='REG'"):
        pa[(gid,team)]+=att
        if att>qb.get((gid,team),(0,None))[0]: qb[(gid,team)]=(att,pid)
    hist=defaultdict(list); opp_state={}; snap_hist=defaultdict(list); team_hist_opp=defaultdict(list)
    team_regs=defaultdict(dict)       # team -> {pid: (avg last3 vol, pos)}
    team_hist=defaultdict(list)       # team -> [(vol_for_market, pass_att, qb)]
    opp_pass=defaultdict(list)        # team -> opponent passing yards allowed per game
    by_week=defaultdict(list)
    for r in h.rows: by_week[(r[5],r[6])].append(r)
    out=[]
    for key in sorted(by_week):
        wk=by_week[key]
        present=defaultdict(set)
        for (pid,name,pos,team,opp,season,week,gid,is_home,vol,yds) in wk:
            if vol>0: present[(gid,team)].add(pid)
        for (pid,name,pos,team,opp,season,week,gid,is_home,vol,yds) in wk:
            ph=hist[pid]
            if pos in cfg['positions'] and len(ph)>=3 and v2._mean([g['vol'] for g in ph[-3:]])>=cfg['min_last3_vol']:
                f=h.features(ph,season,week,team,opp,is_home,opp_state)
                th=team_hist[team][-5:]
                f['team_vol5']=v2._mean([t[0] for t in th])
                f['team_other_vol5']=v2._mean([t[1] for t in th])      # pass attempts (rec) / other facet
                regs=team_regs[team]
                absent=[(p,(a,ps)) for p,(a,ps) in regs.items() if p!=pid and ps in cfg['positions'] and p not in present[(gid,team)]]
                vac=sum(a for p,(a,ps) in absent)
                f['vacated_vol']=vac
                f['vacated_share']=vac/f['team_vol5'] if f['team_vol5'] else None
                f['n_regs_absent']=len(absent)
                lastqb=th[-1][2] if th else None
                f['qb_changed']=1.0 if (th and qb.get((gid,team),(0,None))[1]!=lastqb) else 0.0
                f['team_pass_att5']=v2._mean([t[1] for t in th])
                f['opp_def_pass_yds5']=v2._mean(opp_pass[opp][-5:])
                wd=WEATHER.get(gid,(None,None,None)); f['wind'],f['temp'],f['dome']=wd
                f['rest_days']=min(REST.get((season,week,team),7),14)
                pg='RB' if pos=='FB' else pos
                y_,n_,g_=_allowed(PASS_BY.get((opp,pg),[]),season*100+week)
                lg_y=sum(x[2] for v in PASS_BY.values() for x in v if x[0]<season*100+week and x[0]>=season*100+week-300)
                f['opp_pos_ypt']=(y_/n_) if n_>=20 else None
                tot_n=sum(_allowed(PASS_BY.get((opp,p),[]),season*100+week)[1] for p in ('WR','TE','RB'))
                f['opp_pos_tshare']=(n_/tot_n) if tot_n>0 else None
                f['opp_pos_ypt_vs_lg']=f['opp_pos_ypt']
                yr,nr,gr_=_allowed(RUN_BY.get(opp,[]),season*100+week)
                f['opp_ypc']=(yr/nr) if nr>=40 else None
                o=INJ.get((pid,season,week),(0,-1)); f['own_status']=o[0]; f['own_practice']=o[1]
                tmo=tmq=0.0
                for p_,(a_,ps_) in regs.items():
                    if p_==pid or ps_ not in cfg['positions']: continue
                    st_=INJ.get((p_,season,week),(0,-1))[0]
                    if st_>=2: tmo+=a_
                    elif st_==1: tmq+=a_
                f['tm_out_vol']=tmo; f['tm_quest_vol']=tmq
                sh=snap_hist[v2.norm_name(name)]; f['snap3']=v2._mean(sh[-3:]); f['snap1']=sh[-1] if sh else None
                f['depth_rank']=DEPTH.get((pid,season,week))
                v8=[g['vol'] for g in ph[-8:]]
                sh5=[g['vol']/g['team_vol'] for g in ph[-5:] if g['team_vol']>0]; sh8=[g['vol']/g['team_vol'] for g in ph[-8:] if g['team_vol']>0]
                f['share5']=v2._mean(sh5); f['share8']=v2._mean(sh8)
                f['share_trend']=(f['last3_share']-f['share8']) if (f['last3_share'] is not None and f['share8'] is not None) else None
                f['vol_std8']=float(np.std(v8)); f['vol_max3']=max(g['vol'] for g in ph[-3:]); f['vol_min3']=min(g['vol'] for g in ph[-3:])
                th5=team_hist[team][-5:]
                f['team_plays5']=v2._mean([t[1]+t[3] for t in th5]); f['team_pass_rate5']=v2._mean([t[1]/max(t[1]+t[3],1) for t in th5])
                f['team_car5']=v2._mean([t[3] for t in th5])
                lk=h.lines.get((season,week,team),(None,None,None)); f['game_total']=lk[1]
                f['vacated_share']=(f['tm_out_vol']/f['team_vol5']) if f.get('team_vol5') else None
                tk={t[4] for t in th5}; f['games_missed5']=sum(1 for k_ in tk if not any(g['season']*100+g['week']==k_ for g in ph[-6:]))
                f['opp_faced_vol5']=v2._mean([t[0] for t in team_hist_opp.get(opp,[])[-5:]])
                out.append(({'pid':pid,'name':name,'team':team,'season':season,'week':week},f,float(yds)))
        # update state after the week
        for (pid,name,pos,team,opp,season,week,gid,is_home,vol,yds) in wk:
            hist[pid].append({"season":season,"week":week,"vol":vol,"yds":yds,"team_vol":h.team_vol[(gid,team)],"team":team,"name":name,"pos":pos})
            st=opp_state.setdefault((season,opp),[0.0,0]); st[0]+=yds
        seen={(r[5],r[4],r[7]) for r in wk}
        for (season,opp,gid) in seen: opp_state[(season,opp)][1]+=1
        teams={(r[7],r[3],r[4]) for r in wk}
        for (gid,team,opp) in teams:
            team_hist[team].append((h.team_vol[(gid,team)],pa[(gid,team)],qb.get((gid,team),(0,None))[1],tcar[(gid,team)],wk[0][5]*100+wk[0][6]))
            team_hist_opp[opp].append((h.team_vol[(gid,team)],))
            opp_pass[opp].append(sum(r[10] for r in wk if r[7]==gid and r[3]==team and False) or pa[(gid,team)])  # placeholder (attempts faced)
        for nm_,pc_ in SNAPW.get(key,[]): snap_hist[nm_].append(pc_)
        # regulars = players who played in the team's most recent game (long-term absences are already in the history)
        week_teams={r[3] for r in wk}
        fresh={t:{} for t in week_teams}
        for (pid,name,pos,team,opp,season,week,gid,is_home,vol,yds) in wk:
            if vol<=0: continue
            a=v2._mean([g['vol'] for g in hist[pid][-3:]])
            if a is not None and a>=cfg['min_last3_vol']: fresh[team][pid]=(a,pos)
        for t,d in fresh.items(): team_regs[t]=d
    return out

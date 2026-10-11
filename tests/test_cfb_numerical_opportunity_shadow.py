import sqlite3
import math
import cfb_numerical_opportunity_shadow as m

def database():
 c=sqlite3.connect(':memory:')
 c.executescript("""
 CREATE TABLE games (game_id TEXT PRIMARY KEY,home_points INT,away_points INT);
 CREATE TABLE player_games(player_id TEXT,player_name TEXT,position TEXT,team TEXT,
    opponent TEXT,season INT,week INT,game_id TEXT,
    carries REAL,rushing_yards REAL,receptions REAL,receiving_yards REAL,
    pass_attempts REAL,passing_touchdowns REAL);
 """)
 for season in (2024,2025,2026):
  for w in range(1,9):
   gid=f'{season}_{w:02}_Hawks_Bears'
   c.execute('INSERT INTO games VALUES(?,?,?)',(gid,42,21))
   rows=[]
   for team,opp in (('Hawks','Bears'),('Bears','Hawks')):
    bias=1.2 if team=='Hawks' else .8
    rows.extend([
     (team+'_rb','Running Back','RB',team,opp,season,w,gid,
      (13+w*.5)*bias,(13+w*.5)*bias*5,0,0,0,0),
     (team+'_reserve','Reserve RB','RB',team,opp,season,w,gid,
      2,6,0,0,0,0),
     (team+'_qb','Quarterback','QB',team,opp,season,w,gid,
      3,9,0,0,28,2 if w%3!=0 else 1),
     (team+'_wr','Receiver','WR',team,opp,season,w,gid,
      0,0,5+w*.2,(5+w*.2)*13*bias,0,0),
     (team+'_te','Tight End','TE',team,opp,season,w,gid,
      0,0,3,25,0,0),
    ])
   c.executemany('INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
 c.commit();return c

def test_candidate_generates_actual_expected_counts_and_asof_role_share():
 c=database()
 rows,eligible,state=m.walkforward(c,target=(2025,6))
 vals=[r for r in rows if r['season']==2025 and r['week']==5]
 assert vals
 rush=next(r for r in vals if r['market']=='rushing_yards' and r['player_id']=='Hawks_rb')
 assert 0<rush['expected_stat']<200
 assert rush['expected_opportunities']>=12
 assert 0<rush['observed_team_share_recent3']<1
 assert rush['statistic']=='estimated_conditional_mean_NOT_median_or_bet'
 pass_td=next(r for r in vals if r['market']=='passing_touchdowns')
 assert 0<=pass_td['expected_stat']<5
 recv=next(r for r in vals if r['market']=='receiving_yards')
 assert recv['expected_stat']>0
 assert all('prob' not in r for r in vals)
 assert not any(r['week']>=6 and r['season']==2025 for r in rows)

def test_target_week_outcome_never_enters_forecast():
 c=database()
 before,_,_=m.walkforward(c,target=(2025,6))
 c.execute("UPDATE player_games SET rushing_yards=5000,receiving_yards=8000,passing_touchdowns=9 WHERE season=2025 AND week=5")
 after,_,_=m.walkforward(c,target=(2025,6))
 start=[(r['player_id'],r['week'],r['market'],r['expected_stat']) for r in before]
 end=[(r['player_id'],r['week'],r['market'],r['expected_stat']) for r in after]
 assert start==end
 assert any(r['season']==2025 and r['week']==5 and r['actual_stat']!=q['actual_stat']
            for r,q in zip(before,after) if (r['season'],r['week'])==(2025,5))

def test_no_unsafe_one_or_two_game_rookie_certainty():
 assert m.project([],[],[],(100,500),'rushing_yards') is None
 hist=[(2026,w,'Hawks',15,75,45) for w in (1,2)]
 assert m.project(hist,[(2026,w,'game',45) for w in (1,2)],[],(100,500),'rushing_yards') is None
 hist.append((2026,3,'Other Team',15,75,45))
 assert m.project(hist,[(2026,w,'game',45) for w in (1,2,3)],[],(100,500),'rushing_yards') is None

def test_opponent_history_shrunk_and_cannot_create_extreme_efficiency_illusion():
 h=[(2026,w,'Hawks',15,70,40) for w in (1,2,3,4)]
 team=[(2026,w,'game',40) for w in (1,2,3,4)]
 low=m.project(h,team,[(300,600)],(2000,9000),'rushing_yards')
 high=m.project(h,team,[(300,2700)],(2000,9000),'rushing_yards')
 assert low['expected_stat']<high['expected_stat']
 assert low['opponent_prior_efficiency_adjustment']>=-.12
 assert high['opponent_prior_efficiency_adjustment']<=.12

def test_report_is_research_only_and_never_claims_unseen_validation():
 c=database()
 rows,eligible,_=m.walkforward(c)
 doc=m.report(rows,eligible)
 assert doc['status']=='FROZEN_SHADOW_NO_PROMOTION'
 assert doc['cohorts']['2025_rushing_yards']['n']>0
 assert doc['cohorts']['2026_receiving_yards']['n']>0
 assert doc['cohorts']['2025_rushing_yards']['paired_game_bootstrap_ci95_lift'] is not None
 assert any('not untouched holdouts' in x for x in doc['limitations'])
 assert 'bookmaker' not in doc.get('status','').lower()

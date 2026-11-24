import pandas as pd, numpy as np, sys, time
D='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/'; OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
t0=time.time()
cols=['game_id','play_id','season','week','season_type','posteam','defteam','posteam_type','home_team','away_team','play_type','yards_gained','epa','success','pass_attempt','sack','qb_hit','qb_scramble','complete_pass','touchdown','pass_touchdown','rush_touchdown','qb_kneel','qb_spike','two_point_attempt',
 'rusher_player_id','receiver_player_id','passer_player_id','air_yards','yards_after_catch','pass_location','pass_length','run_location','run_gap','shotgun','no_huddle','xpass','pass_oe','down','ydstogo','yardline_100','score_differential','game_seconds_remaining','qtr','spread_line','total_line','wp','temp','wind','roof','fumble_lost','interception','penalty']
fr=[]
for y in range(2016,2027):
    d=pd.read_csv(D+('pbp_%d.csv.gz'%y),usecols=lambda c:c in cols,low_memory=False)
    fr.append(d)
p=pd.concat(fr,ignore_index=True); p=p[(p.season_type=='REG')&p.posteam.notna()]
p=p[p.play_type.isin(['run','pass'])&(p.two_point_attempt.fillna(0)==0)&(p.qb_kneel.fillna(0)==0)&(p.qb_spike.fillna(0)==0)].copy()
print('plays',len(p),'seasons',sorted(p.season.unique()),round(time.time()-t0),'s',flush=True)
# participation
pc=['nflverse_game_id','play_id','offense_formation','offense_personnel','defenders_in_box','defense_personnel','number_of_pass_rushers','offense_players','defense_players','ngs_air_yards','time_to_throw','was_pressure','route','defense_man_zone_type','defense_coverage_type']
pf=[]
for y in range(2016,2026):
    q=pd.read_csv(D+'part_%d.csv'%y,usecols=lambda c:c in pc,low_memory=False); q['season_p']=y; pf.append(q)
q=pd.concat(pf,ignore_index=True).rename(columns={'nflverse_game_id':'game_id'})
p=p.merge(q.drop(columns='season_p'),on=['game_id','play_id'],how='left')
print('merged participation',round(time.time()-t0),'s',flush=True)
# FTN
ff=[]
for y in range(2022,2027):
    try: f=pd.read_csv(D+'ftn_%d.csv'%y if y<2026 else D+'ftn_charting_2026.csv',low_memory=False)
    except Exception as e: print('ftn',y,e); continue
    ff.append(f[['nflverse_game_id','nflverse_play_id','starting_hash','qb_location','n_offense_backfield','n_defense_box','is_no_huddle','is_motion','is_play_action','is_screen_pass','is_rpo','is_qb_out_of_pocket','is_throw_away','is_catchable_ball','is_contested_ball','is_drop','n_blitzers','n_pass_rushers','is_qb_fault_sack','read_thrown']].rename(columns={'nflverse_game_id':'game_id','nflverse_play_id':'play_id'}))
f=pd.concat(ff,ignore_index=True).drop_duplicates(['game_id','play_id'])
p=p.merge(f,on=['game_id','play_id'],how='left',suffixes=('','_ftn'))
print('merged ftn',round(time.time()-t0),'s',flush=True)
p.to_pickle(OUT+'plays.pkl')
print('\nCOVERAGE BY SEASON (share of run/pass plays with the item):')
chk=['offense_personnel','defenders_in_box','number_of_pass_rushers','offense_players','was_pressure','time_to_throw','route','defense_man_zone_type','defense_coverage_type','n_defense_box','n_blitzers','is_play_action','is_motion','run_gap','air_yards','xpass']
g=p.groupby('season')
print(pd.DataFrame({c:g[c].apply(lambda s:round(float(s.notna().mean()),2)) for c in chk}).to_string())
print('\nrows',len(p),'cols',p.shape[1])

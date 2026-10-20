"""No hindsight, no forced zero for players with absent official rows."""
from collections import defaultdict
from datetime import datetime,timezone
import nfl_v2_firstseen_true_stats_backtest as A

NOW=datetime(2026,10,17,17,0,tzinfo=timezone.utc)
G='2026_06_SEA_DEN'
K='2026-10-16T00:15:00Z'
C='2026-10-15T22:45:00Z'


def forecast(outcome='rec_yds',team='SEA'):
    return {'game_id':G,'season':2026,'week':6,'horizon':'T90','team':team,
            'opponent':'DEN','player_name':'test player','player_id':'00-0011111',
            'position':'WR','outcome':outcome,'cutoff':C,'kickoff':K,
            'median':70,'mean':73,'p_active':0.9,
            'quantiles':{'p10':20,'p90':130}}


def game(final=True):
    return {G:{'game_id':G,'season':2026,'week':6,'season_type':'REG',
               'kickoff_utc':K,'home_team':'DEN','away_team':'SEA',
               'home_score':17 if final else None,'away_score':14 if final else None}}


def player(yds=134):
    return {'player_id':'00-0011111','player_name':'test player',
            'season':2026,'week':6,'team':'SEA','season_type':'REG',
            'receiving_yards':yds,'rushing_tds':0,'receiving_tds':0}


def evaluate(rows,players=None,games=None):
    pp=defaultdict(list)
    for p in ([player()] if players is None else players):
        pp[(2026,6,'SEA',p['player_id'])].append(p)
    return A.evaluate({'T90':{G:rows},'T24':{}},games or game(),pp,{},NOW)


def test_actual_median_not_alternate_betting_line():
    out=evaluate([forecast()])
    assert out['graded_player_outcomes']==1
    assert out['by_horizon']['T90']['median_mae']==64
    assert out['by_horizon']['T90']['mean_mae']==61
    assert out['by_horizon']['T90']['signed_bias_median_minus_actual']==-64
    assert out['automatic_model_promotion'] is False


def test_not_yet_final_game_never_score():
    d=evaluate([forecast()],games=game(False))
    assert d['graded_player_outcomes']==0
    assert d['exclusions']['NOT_FINAL_OR_NOT_INGESTED']==1


def test_missing_actual_never_counted_as_zero():
    d=evaluate([forecast()],players=[])
    assert d['graded_player_outcomes']==0
    assert d['exclusions']['NO_UNIQUE_PLAYER_ACTUAL_NOT_ZERO']==1


def test_t24_t90_are_not_pooled_into_one_prediction():
    a=forecast()
    b={**a,'horizon':'T24','cutoff':'2026-10-15T00:15:00Z'}
    pp=defaultdict(list,{(2026,6,'SEA','00-0011111'):[player()]})
    d=A.evaluate({'T24':{G:[b]},'T90':{G:[a]}},game(),pp,{},NOW)
    assert d['completed_games_with_verified_forecasts']==1
    assert d['completed_game_horizon_pairs']==2
    assert d['by_horizon']['T24']['n']==d['by_horizon']['T90']['n']==1


def test_previous_three_are_strictly_prior_2026_games():
    h={'a':[{'season':2026,'week':1,'season_type':'REG','rushing_yards':10},
            {'season':2026,'week':2,'season_type':'REG','rushing_yards':20},
            {'season':2026,'week':3,'season_type':'REG','rushing_yards':30},
            {'season':2026,'week':6,'season_type':'REG','rushing_yards':1000}]}
    assert A.last3(h,'a',2026,6,'rush_yds')==20


def test_wrong_team_and_mismatched_kickoff_fail_closed():
    d=evaluate([forecast(team='NYJ')])
    assert d['exclusions']['TEAM_OR_STAGE_MISMATCH']==1
    v=forecast();v['cutoff']='2026-10-17T22:00:00Z'
    d=evaluate([v])
    assert d['exclusions']['INVALID_TIMESTAMP_OR_KICKOFF']==1


def test_atd_uses_rushing_or_receiving_touchdowns_not_pass_touchdowns():
    p=player()
    p['passing_tds']=5
    assert A.actual(p,'atd')==0
    p['receiving_tds']=1
    assert A.actual(p,'atd')==1
    p['receiving_tds']=None
    assert A.actual(p,'atd') is None


def test_assisted_tackle_field_is_not_total_tackles_and_cannot_be_graded():
    p=player();p['def_tackles_with_assist']=0
    d=evaluate([{**forecast(),'outcome':'tackles'}],players=[p])
    assert d['graded_player_outcomes']==0
    assert d['exclusions']['TACKLES_TOTAL_ACTUAL_NOT_AVAILABLE_IN_THIS_SOURCE']==1
    assert 'tackles' not in d['by_market_horizon']['T90']
    assert 'quarantined' in d['source_contract_warning']


def test_paired_horizon_reports_exact_same_player_and_same_game_improvement():
    a=forecast()
    b={**a,'horizon':'T24','cutoff':'2026-10-15T00:15:00Z','median':100}
    pp=defaultdict(list,{(2026,6,'SEA','00-0011111'):[player()]})
    d=A.evaluate({'T24':{G:[b]},'T90':{G:[a]}},game(),pp,{},NOW)
    paired=d['paired_t24_vs_t90_same_player']
    assert paired['all_markets']['n']==1
    assert paired['all_markets']['t24_mae']==34
    assert paired['all_markets']['t90_mae_on_same_players']==64
    assert paired['all_markets']['t24_improvement_positive_is_better']==30
    assert paired['all_markets']['unchanged_both_horizons_within_001']==0


def test_equal_horizon_counts_not_mistaken_for_fresh_model_intelligence():
    a=forecast()
    b={**a,'horizon':'T24','cutoff':'2026-10-15T00:15:00Z'}
    pp=defaultdict(list,{(2026,6,'SEA','00-0011111'):[player()]})
    d=A.evaluate({'T24':{G:[b]},'T90':{G:[a]}},game(),pp,{},NOW)
    paired=d['paired_t24_vs_t90_same_player']['all_markets']
    assert paired['unchanged_both_horizons_within_001']==1
    assert paired['t24_improvement_positive_is_better']==0


def test_espn_verified_total_tackles_scored_instead_of_conflating_assists():
    from nfl_week6_first_seen_grade import namekey
    p=player()
    p['def_tackles_with_assist']=0 # distinct stat: NOT total
    stat={**forecast(outcome='tackles'),'median':7,'mean':7}
    stat['player_name']='test player'
    ep={"schema":"NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1","games":{
        G:{"game_id":G,"source":"ESPN_FINAL_GAME_BOXSCORE",
           "season":2026,"week":6,"home":"DEN","away":"SEA",
           "kickoff_utc":K,"home_score":17,"away_score":14,
           "players":[{"team":"SEA","name_key":"test player",
                       "player":"test player","espn_athlete_id":"test123",
                       "offense_appeared":False,"defense_appeared":True,
                       "stats":{"tackles":9}}]}}}
    pp=defaultdict(list,{(2026,6,'SEA','00-0011111'):[p]})
    out=A.evaluate({'T90':{G:[stat]},'T24':{}},game(),pp,{},NOW,espn=ep)
    assert out['every_genuine_original_forecast_accounted']
    assert out['graded_player_outcomes']==1
    assert out['all_graded_rows'][0]['actual']==9
    assert out['all_graded_rows'][0]['truth_source']=='ESPN_FINAL_BOX'
    assert out['by_game_horizon'][0]['markets']['tackles']['graded']==1

def test_v2_all_genuine_forecasts_are_accounted_even_without_outcome():
    good=forecast()
    missing={**forecast(outcome='rush_yds'),"id":"different","player_id":"00-0000002"}
    pp=defaultdict(list,{(2026,6,'SEA','00-0011111'):[player()]})
    out=A.evaluate({'T90':{G:[good,missing]},'T24':{}},game(),pp,{},NOW)
    assert out['total_original_pregame_forecasts']==2
    assert out['graded_player_outcomes']==1
    assert out['unverified_original_forecasts']==1
    assert out['all_unverified_rows'][0]['player_id']=='00-0000002'
    assert out['by_game_horizon'][0]['unverified']==1
    assert out['by_game_horizon'][0]['graded']==1

def test_week_discovery_not_hardcoded_to_2026_week6(tmp_path):
    folder=tmp_path/'forecasts'/'batches'
    folder.mkdir(parents=True)
    (folder/'2026_wk07_T24_2026_07_CHI_NE.jsonl').write_text('original')
    (folder/'2026_wk09_T90_2026_09_NO_ATL.jsonl').write_text('original')
    (folder/'2026_wk07_AD_HOC_2026_07_CHI_NE.jsonl').write_text('invalid')
    assert A.available_weeks(tmp_path)==[7,9]



def test_late_but_pregame_2026_week6_batch_cannot_masquerade_as_true_T24():
    from copy import deepcopy
    r=forecast(outcome='pass_yds')
    r["kickoff"]="2026-10-18T17:00:00.000000Z"
    r["cutoff"]="2026-10-17T17:00:00.000000Z"
    r["input_snapshots"]={"retrieval_ts":"2026-10-17T16:56:13.144921Z",
                          "cutoff":r["cutoff"],"time_travel":False}
    head={"game_id":G,"horizon":"T90","generated_at":"2026-10-17T17:00:29.663027+00:00"}
    # Correct physical T24 values, but 29.7 seconds late for its horizon.
    r["horizon"]="T24";head["horizon"]="T24"
    assert A.pregame_timing_tier(head,r)=="LATE_PREGAME"
    good={**head,"generated_at":"2026-10-17T16:59:59Z"}
    assert A.pregame_timing_tier(good,r)=="ON_CUTOFF"

    lost={**head,"generated_at":"2026-10-18T17:00:01Z"}
    assert A.pregame_timing_tier(lost,r) is None
    stale=deepcopy(r)
    stale["input_snapshots"]["retrieval_ts"]="2026-10-17T17:00:02Z"
    assert A.pregame_timing_tier(head,stale) is None
    travel=deepcopy(r)
    travel["input_snapshots"]["time_travel"]=True
    assert A.pregame_timing_tier(head,travel) is None
    wrong=deepcopy(r)
    wrong["cutoff"]="2026-10-17T18:00:00Z"
    wrong["input_snapshots"]["cutoff"]=wrong["cutoff"]
    assert A.pregame_timing_tier(head,wrong) is None


def test_late_pregame_is_a_separate_scored_horizon_never_true_T24():
    orig=forecast()
    orig["horizon"]="T24"
    orig["original_horizon"]="T24"
    orig["grading_provenance"]="HASH_VERIFIED_PRE_KICKOFF_BUT_GENERATED_AFTER_NAMED_HORIZON_CUTOFF"
    p=defaultdict(list,{(2026,6,'SEA','00-0011111'):[player()]})
    d=A.evaluate({'LATE_T24':{G:[orig]},'T24':{},'T90':{},'LATE_T90':{}},
                 game(),p,{},NOW)
    assert d['by_horizon']['T24']['n']==0
    assert d['by_horizon']['LATE_T24']['n']==1
    assert d['late_horizon_but_before_kickoff_forecasts']==1
    assert d['late_forecasts_are_NOT_valid_original_T24_T90_cutoff']
    assert d['by_game_horizon'][0]['original_horizon']=='T24'
    assert d['by_game_horizon'][0]['horizon']=='LATE_T24'

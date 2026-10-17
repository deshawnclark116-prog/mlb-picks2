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

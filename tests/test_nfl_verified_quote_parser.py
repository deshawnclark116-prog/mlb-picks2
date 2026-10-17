from datetime import datetime,timezone,timedelta
from nfl_verified_quote_parser import parse_market_quotes

NOW=datetime(2026,10,17,14,0,tzinfo=timezone.utc)
UPDATED="2026-10-17T13:59:00Z"

def market(lines,ts=UPDATED):
    return {"key":"player_rush_yds","last_update":ts,"outcomes":lines}

def point(side,n,price=-110):
    return {"description":"Bijan Robinson","name":side,"point":n,"price":price}

def p(m):
    return parse_market_quotes(m,book="fanduel",event_id="realgame",
                               normalize=lambda x:x.casefold(),now=NOW)

def test_one_exact_main_pair_admitted_with_provenance():
    accepted,reject=p(market([point("Over",83.5,-110),point("Under",83.5,-120)]))
    assert reject=={}
    assert accepted["bijan robinson"]=={
        "line":83.5,"over_price":-110,"under_price":-120,
        "book":"fanduel","event_id":"realgame","market_updated_at_utc":UPDATED}

def test_old_mixed_point_false_main_line_is_never_admitted():
    accepted,reject=p(market([point("Over",147.5,-115),point("Under",83.5,-115)]))
    assert accepted=={}
    assert reject["NOT_ONE_UNIQUE_PAIRED_MAIN_LINE"]==1

def test_multiple_alt_point_pairs_all_refused_instead_of_cherry_picking():
    accepted,reject=p(market([
        point("Over",83.5),point("Under",83.5),
        point("Over",147.5),point("Under",147.5)]))
    assert accepted=={}
    assert reject["NOT_ONE_UNIQUE_PAIRED_MAIN_LINE"]==1

def test_missing_side_or_impossible_odds_or_old_timestamp_rejected():
    assert not p(market([point("Over",75.5)]))[0]
    assert not p(market([point("Over",75.5,-90),point("Under",75.5,-90)]))[0]
    old=(NOW-timedelta(hours=4)).isoformat()
    assert not p(market([point("Over",75.5),point("Under",75.5)],ts=old))[0]

def test_duplicate_same_side_refused_even_at_same_line():
    got,_=p(market([point("Over",50.5),point("Over",50.5)]))
    assert got=={}

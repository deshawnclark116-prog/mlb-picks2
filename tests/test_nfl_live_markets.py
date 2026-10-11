"""Real bookmaker availability and request-budget invariants for NFL player props."""
from datetime import datetime,timezone,timedelta
import nfl_live_markets as m

CLOCK=datetime(2026,10,11,1,40,tzinfo=timezone.utc)
KICKOFF="2026-10-11T13:30:00Z"
def event():
 return {"id":"real-event","home_team":"Jacksonville Jaguars",
         "away_team":"Philadelphia Eagles","commence_time":KICKOFF}
def quotes(age=0,home="Jacksonville Jaguars"):
 t=(CLOCK-timedelta(seconds=age)).isoformat()
 return {"home_team":home,"away_team":"Philadelphia Eagles","commence_time":KICKOFF,
         "bookmakers":[{"key":"fanduel","last_update":t,
         "markets":[{"key":"player_reception_yds","last_update":t,"outcomes":[
           {"name":"Over","description":"Jakobi Meyers","point":42.5,"price":-110},
           {"name":"Under","description":"Jakobi Meyers","point":42.5,"price":-110},
           {"name":"Over","description":"Dameon Pierce","point":3.5,"price":-125},
           {"name":"Under","description":"Dameon Pierce","point":4.5,"price":110}
         ]},{"key":"player_rush_yds","last_update":t,"outcomes":[
           {"name":"Over","description":"Bhayshul Tuten","point":60.5,"price":-120},
           {"name":"Under","description":"Bhayshul Tuten","point":60.5,"price":105}
         ]}]}]}
class R:
 def __init__(self,data,status=200):self.data=data;self.status_code=status
 def json(self):return self.data

def test_exact_fanduel_event_two_way_main_market_only():
 lines=m.parse_event(quotes(),home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)
 assert len(lines)==2
 assert lines[0]["player"]=="Jakobi Meyers" or lines[1]["player"]=="Jakobi Meyers"
 assert not any(x["player"]=="Dameon Pierce" for x in lines)
 assert {x["market"] for x in lines}=={"receiving_yards","rushing_yards"}
 assert {x["book"] for x in lines}=={"fanduel"}

def test_stale_wrong_event_and_game_already_started_are_rejected():
 assert m.parse_event(quotes(age=1810),home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)==[]
 try:
  m.parse_event(quotes(home="Dallas Cowboys"),home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)
  assert False
 except ValueError as e: assert "event_not_confirmed" in str(e)

def test_no_player_line_if_under_or_bookmaker_is_missing():
 q=quotes()
 q["bookmakers"][0]["markets"][0]["outcomes"]=q["bookmakers"][0]["markets"][0]["outcomes"][:1]
 q["bookmakers"][0]["markets"][1]["outcomes"]=[]
 assert m.parse_event(q,home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)==[]

def test_one_selected_game_only_and_cache_prevents_duplicate_quota():
 m.CACHE.clear(); urls=[]
 def get(url,params,timeout):
  urls.append((url,params))
  return R([event()] if url.endswith("/events") else quotes())
 result=m.collect("JAX","PHI",KICKOFF,"private",now=CLOCK,http_get=get)
 assert result["status"]=="VERIFIED_QUOTES" and len(result["lines"])==2
 assert len(urls)==2 and urls[1][1]["bookmakers"]=="fanduel"
 assert urls[1][1]["markets"]=="player_rush_yds,player_reception_yds"
 again=m.collect("JAX","PHI",KICKOFF,"private",now=CLOCK+timedelta(seconds=90),http_get=get)
 assert len(urls)==2 and again["lines"]==result["lines"]
 assert "private" not in str(result), "The response cannot leak provider credentials"

def test_no_token_no_request_and_no_cached_prices_after_kickoff():
 m.CACHE.clear()
 def denied(*args,**kwargs):raise AssertionError("no network request allowed")
 assert m.collect("JAX","PHI",KICKOFF,"",now=CLOCK,http_get=denied)["status"]=="ODDS_KEY_NOT_CONFIGURED"
 assert m.collect("JAX","PHI",KICKOFF,"secret",now=m.utc(KICKOFF),http_get=denied)["lines"]==[]

def test_provider_quota_exhaustion_fails_closed():
 m.CACHE.clear()
 def quota(url,params,timeout):
  return R([],429)
 result=m.collect("JAX","PHI",KICKOFF,"secret",now=CLOCK,http_get=quota)
 assert result["status"]=="ODDS_PROVIDER_UNAVAILABLE" and result["lines"]==[]

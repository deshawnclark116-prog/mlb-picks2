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

def test_oddsapiio_single_event_fanduel_player_props_work_without_theoddsapi_key():
    m.CACHE.clear()
    observed=[]
    def get(url,params,timeout):
        observed.append((url,params))
        if url.endswith("/events"):
            return R([{"id":1234567,"home":"Jacksonville Jaguars",
                       "away":"Philadelphia Eagles","date":KICKOFF}])
        assert url.endswith("/odds") and params["bookmakers"]=="FanDuel"
        t=(CLOCK-timedelta(minutes=3)).isoformat()
        return R({"home":"Jacksonville Jaguars","away":"Philadelphia Eagles","date":KICKOFF,
           "bookmakers":{"FanDuel":[{"name":"Player Props","updatedAt":t,
             "odds":[{"label":"Parker Washington (Receiving Yards)","hdp":43.5,
                      "over":"1.91","under":"1.91"},
                     {"label":"Bhayshul Tuten (Rushing Yards)","hdp":59.5,
                      "over":"1.90","under":"1.92"},
                     {"label":"Parker Washington (Anytime TD)","hdp":0.5,"over":"2.90"}]}]}})
    r=m.collect("JAX","PHI",KICKOFF,"",now=CLOCK,http_get=get,oddsapiio_key="private-key")
    assert r["status"]=="VERIFIED_QUOTES" and r["provider"]=="oddsapiio"
    assert len(r["lines"])==2
    assert {q["market"] for q in r["lines"]}=={"rushing_yards","receiving_yards"}
    assert {q["book"] for q in r["lines"]}=={"fanduel"}
    assert len(observed)==2
    assert observed[1][1]["markets"]=="Player Props"
    assert "private-key" not in str(r)

def test_oddsapiio_malformed_or_wrong_event_fails_closed_without_fake_books():
    t=(CLOCK-timedelta(minutes=2)).isoformat()
    doc={"home":"Philadelphia Eagles","away":"Jacksonville Jaguars",
         "date":KICKOFF,"bookmakers":{"FanDuel":[{"name":"Player Props","updatedAt":t,
          "odds":[{"label":"Fake (Receiving Yards)","hdp":40.5,"over":"1.90","under":"1.90"}]}]}}
    try:
        m.parse_io_event(doc,home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)
        assert False
    except ValueError: pass
    doc["home"]="Jacksonville Jaguars";doc["away"]="Philadelphia Eagles"
    doc["bookmakers"]["FanDuel"][0]["updatedAt"]=(CLOCK-timedelta(hours=4)).isoformat()
    assert m.parse_io_event(doc,home="JAX",away="PHI",kickoff=m.utc(KICKOFF),clock=CLOCK)==[]

def test_oddsapiio_unavailable_falls_back_to_existing_trusted_theoddsapi():
    m.CACHE.clear()
    calls=[]
    def get(url,params,timeout):
        calls.append(url)
        if url.startswith(m.IO_BASE):
            return R({},429)
        return R([event()] if url.endswith("/events") else quotes())
    r=m.collect("JAX","PHI",KICKOFF,"v4-secret",now=CLOCK,http_get=get,oddsapiio_key="io-secret")
    assert r["provider"]=="theoddsapi"
    assert len(r["lines"])==2
    assert len(calls)==4

def test_decimal_conversion_never_invents_a_price():
    assert m.decimal_to_american("1.91")<=-100
    assert m.decimal_to_american("2.10")==110
    assert m.decimal_to_american("1.0") is None
    assert m.decimal_to_american("garbage") is None

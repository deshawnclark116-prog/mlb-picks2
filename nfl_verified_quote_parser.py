"""Strict The Odds API NFL player prop pairing: never join prices from different lines.

Original serving parser took point from the FIRST outcome and over/under odds
from ANY later outcome. On alternate-heavy feeds this could combine an OVER
price for e.g. 147.5 and an UNDER price for 74.5 into a fake wager.
Only accept exactly one pair, same player, same point, opposite side,
valid book prices, and provider-issued last_update. If multiple quoted
lines exist, quarantine until a provider-certified main line is available.
"""
import math
from datetime import datetime,timezone


def parse_market_quotes(mkt_data, *,book,event_id,normalize,now=None):
    market_time=mkt_data.get("last_update")
    try:
        t=datetime.fromisoformat(str(market_time).replace("Z","+00:00"))
        if t.tzinfo is None:raise ValueError()
        current=now or datetime.now(timezone.utc)
        age=(current-t).total_seconds()
        if age<0 or age>3600: return {},{"MISSING_OR_STALE_MARKET_TIMESTAMP":1}
    except (ValueError,TypeError):
        return {},{"MISSING_OR_STALE_MARKET_TIMESTAMP":1}
    groups={}
    rejects={}
    for obj in mkt_data.get("outcomes") or []:
        player=obj.get("description")
        side=str(obj.get("name") or "").strip().lower()
        point=obj.get("point");price=obj.get("price")
        if not isinstance(player,str) or not player.strip() or side not in ("over","under"):
            continue
        if (not isinstance(point,(int,float)) or isinstance(point,bool)
                or not math.isfinite(point) or not 0<=point<5000
                or not isinstance(price,(int,float)) or isinstance(price,bool)
                or not math.isfinite(price) or not (100<=abs(price)<=20000)):
            rejects["INVALID_LINE_OR_PRICE"]=rejects.get("INVALID_LINE_OR_PRICE",0)+1
            continue
        key=normalize(player)
        groups.setdefault(key,[]).append((side,float(point),int(price)))
    result={}
    for name,items in groups.items():
        if len(items)!=2 or {x[0] for x in items}!={"over","under"} or items[0][1]!=items[1][1]:
            rejects["NOT_ONE_UNIQUE_PAIRED_MAIN_LINE"]=rejects.get("NOT_ONE_UNIQUE_PAIRED_MAIN_LINE",0)+1
            continue
        prices={side:price for side,_,price in items}
        result[name]={"line":items[0][1],"over_price":prices["over"],
                      "under_price":prices["under"],"book":book,
                      "event_id":event_id,"market_updated_at_utc":market_time}
    return result,rejects

#!/usr/bin/env python3
"""Diagnostic ONLY: discover ESPN Week 6 2026 game and boxscore source shape."""
import json,urllib.request
from datetime import datetime,timezone
BASE='https://site.api.espn.com/apis/site/v2/sports/football/nfl/'
def get(path):
  req=urllib.request.Request(BASE+path,headers={'User-Agent':'PropEdgeIndependentNFLGrade/1.0','Accept':'application/json'})
  with urllib.request.urlopen(req,timeout=25) as r:return json.load(r)
def main():
 for date in ('20261018','20261019'):
  j=get('scoreboard?dates='+date+'&limit=50')
  evs=j.get('events') or []
  print('SOURCE_SCHEDULE',date,'n',len(evs),flush=True)
  for e in evs:
   g=e.get('competitions',[{}])[0]
   print('NFL_EVENT',e.get('id'),e.get('date'),str(e.get('status',{}).get('type',{}).get('completed')),[(x.get('team',{}).get('abbreviation'),x.get('homeAway'),x.get('score')) for x in g.get('competitors',[])],flush=True)
  if not evs:continue
  eid=evs[0]['id']
  b=get('summary?event='+str(eid))
  print('SUMMARY',eid,'keys',list(b),'boxscore',list(b.get('boxscore') or {}),flush=True)
  for team in (b.get('boxscore') or {}).get('players',[]):
   print('TEAM',team.get('team',{}).get('abbreviation'), 'GROUPS',[(z.get('name'),z.get('type'),z.get('labels'),z.get('keys'),len(z.get('athletes') or []), (z.get('athletes') or [{}])[0].get('stats')) for z in team.get('statistics',[])],flush=True)
if __name__=='__main__':main()

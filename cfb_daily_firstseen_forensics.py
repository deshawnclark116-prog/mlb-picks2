#!/usr/bin/env python3
"""Read-only CFB forensic audit: first pregame ledger predictions, ET kickoff dates.

DO NOT use the report to recalibrate this already-seen slate, change
betting eligibility, or claim profit on unverified bookmaker lines.
"""
import argparse,json
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
from pathlib import Path

ET=ZoneInfo('America/New_York')
SOURCE='FIRST_PREGAME_LEDGER_ENTRY'
FREEZE='OBSERVATIONAL_ONLY_NO_POST_HOLDOUT_TUNING'

def ts(x):
    if not isinstance(x,str): return None
    try:
        v=datetime.fromisoformat(x.replace('Z','+00:00'))
        return v if v.tzinfo is not None else None
    except ValueError: return None

def stats(rows):
    n=len(rows)
    if not n:return {'n':0}
    wins=sum(r['result']=='hit' for r in rows)
    ps=[(r['model_prob'],int(r['result']=='hit')) for r in rows
        if isinstance(r.get('model_prob'),(int,float)) and not isinstance(r.get('model_prob'),bool)
        and 0<=r['model_prob']<=1]
    return {'n':n,'wins':wins,'losses':n-wins,'hit_rate':round(wins/n,4),
       'mean_stated_probability':round(sum(p for p,_ in ps)/len(ps),4) if ps else None,
       'mean_brier':round(sum((p-y)**2 for p,y in ps)/len(ps),5) if ps else None,
       'n_probability_scored':len(ps)}

def evaluate(record,status,date_et):
    datetime.strptime(date_et,'%Y-%m-%d')
    rows=[]
    for r in record.get('results',[]):
        ko,logged=ts(r.get('kickoff_utc')),ts(r.get('logged_at'))
        if not ko or ko.astimezone(ET).date().isoformat()!=date_et: continue
        if not logged or logged>=ko or r.get('evaluation_probability_source')!=SOURCE:
            raise ValueError('unverifiable pregame record: '+str(r.get('canonical_key')))
        if r.get('result') not in ('hit','miss'):raise ValueError('invalid outcome')
        rows.append(r)
    markets={m:stats([r for r in rows if r['market']==m])
             for m in sorted({r['market'] for r in rows})}
    gens={m:stats([r for r in rows if r.get('generation')==m])
          for m in sorted({r.get('generation') for r in rows},key=str)}
    moneyline=[r for r in rows if r['market'].startswith('moneyline')]
    high=[r for r in moneyline if isinstance(r.get('model_prob'),(float,int)) and r['model_prob']>=.8]
    reliability=[]
    for lo,hi in [(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00001)]:
        selected=[r for r in moneyline if isinstance(r.get('model_prob'),(float,int))
                  and lo<=r['model_prob']<hi]
        reliability.append({'bin':f'{lo:.1f}-{min(hi,1):.1f}',**stats(selected)})
    sides=defaultdict(Counter)
    for r in rows:
        side=str(r.get('pick') or '').split(' ')[0].upper()
        if side in ('OVER','UNDER'):sides[r['market']][side]+=1
    worst=sorted((r for r in moneyline if r['result']=='miss'
        and isinstance(r.get('model_prob'),(float,int))),
        key=lambda r:-r['model_prob'])
    ml=stats(moneyline)
    return {'date_et':date_et,'first_pregame_only':True,'decision':FREEZE,
      'all':stats(rows),'games':len({r['game_id'] for r in rows}),
      'markets':markets,'model_generations':gens,
      'moneyline':{'overall':ml,'high80':stats(high),
         'confidence_gap_pp':round(100*(ml['mean_stated_probability']-ml['hit_rate']),1)
             if ml.get('mean_stated_probability') is not None else None,
         'reliability_bins':reliability,
         'highest_confidence_losses':[{'predicted_team':r['team'],
             'actual_winner':r.get('actual'),'stated_p':r['model_prob'],
             'opponent':r['opponent']} for r in worst[:15]]},
      'model_threshold_side_counts':{m:dict(c) for m,c in sorted(sides.items())},
      'global_ungraded_NOT_date_scoped':status.get('ungraded_reasons',{}),
      'limits':['Stats describe first published pregame predictions only.',
        'Player prop thresholds are NOT confirmed sportsbook lines; ROI is unknown.',
        'Final-game player missing from stat feed is excluded by current grader, not scored as a loss.',
        'Ungraded counts shown are ALL dates, not the audited date.',
        'No injury, roster, starter or game-script validation from this audit.',
        'No predictive model was altered, tuned or promoted after outcomes.']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--record',default='docs/cfb_record.json')
    p.add_argument('--status',default='docs/cfb_forward_status.json')
    p.add_argument('--date-et',required=True)
    p.add_argument('--out',default='docs/cfb_daily_forensics.json')
    a=p.parse_args()
    doc=evaluate(json.loads(Path(a.record).read_text()),
       json.loads(Path(a.status).read_text()),a.date_et)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'date_et':doc['date_et'],'graded':doc['all'],
       'moneyline':doc['moneyline']['overall'],
       'confidence_gap_pp':doc['moneyline']['confidence_gap_pp'],
       'decision':doc['decision']}))
if __name__=='__main__':main()

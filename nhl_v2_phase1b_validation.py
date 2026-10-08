"""Supplemental retrospective paired uncertainty; never fitting or selection.

Reads frozen receipts; future clean-forward claims still require their own sample.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

import nhl_v2_phase1a_sog_model as B
from nhl_v2_phase1a_sog_forward import parse_iso,OUT
from nhl_v2_phase1b_sources import write_json


def moving_block(delta,weeks,reps=10000,seed=1701):
    delta=np.asarray(delta,float);weeks=np.asarray(weeks)
    unique=np.unique(weeks);count=len(unique)
    starts=np.array([i for i in range(count-1) if unique[i+1]==unique[i]+1])
    if not len(starts):return {'status':'INSUFFICIENT_CONTIGUOUS_CALENDAR_WEEKS'}
    sums=np.array([delta[weeks==w].sum() for w in unique]);ns=np.array([(weeks==w).sum() for w in unique])
    rng=np.random.default_rng(seed)
    sampled=starts[rng.integers(0,len(starts),size=(reps,math.ceil(count/2)))]
    indices=np.stack([sampled,sampled+1],axis=2).reshape(reps,-1)[:,:count]
    boot=sums[indices].sum(axis=1)/ns[indices].sum(axis=1)
    return {'status':'EXPOSED_DESCRIPTIVE_ONLY','game_mean_delta':float(delta.mean()),'CI95':[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
            'one_sided95_upper':float(np.percentile(boot,95)), 'replicates':reps,'seed':seed,'calendar_weeks':count}


def validate(receipt_dir):
    result={'evidence':'EXPOSED_RETROSPECTIVE_PAIRED_UNCERTAINTY_NOT_PROMOTION','fits_performed':0,'periods':{}}
    alpha=json.loads((OUT/'phase1a_sog_models/engine_T90.json').read_text())['nb2']['alpha']
    for year,path in [('2023',Path(receipt_dir)/'phase1b_development_receipts.jsonl.gz')]+[(str(y),Path(receipt_dir)/f'phase1b_{y}_diagnostic_receipts.jsonl.gz') for y in (2024,2025)]:
        rows=[json.loads(l) for l in gzip.open(path,'rt')];rows=[r for r in rows if r['meaningful']]
        y=np.array([r['actual_sog'] for r in rows],int);g=np.array([r['game_id'] for r in rows]);p=np.array([r['player_id'] for r in rows]);starts={r['game_id']:parse_iso(r['scheduled_start']).timestamp() for r in rows}
        b2=B.row_metrics('nb2',{'mu':np.array([r['B2'] for r in rows]),'alpha':alpha},y,'B2',g,p)
        analyst=B.row_metrics('poisson',{'mu':np.maximum([r['transparent_analyst_mean'] for r in rows],1e-6)},y,'transparent_analyst_mean',g,p)
        _,gids,delta=B.game_macro(analyst['crps']-b2['crps'],g)
        weeks=np.array([(starts[int(gid)]-345600)//604800 for gid in gids])
        result['periods'][year]={'analyst_minus_B2_CRPS':moving_block(delta,weeks),'paired_primary_rows':len(rows),
                                'probability_source':'NB2/Poisson CDF; never the frozen shifted P1..P5 fields'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    write_json(args.output,validate(args.receipts))

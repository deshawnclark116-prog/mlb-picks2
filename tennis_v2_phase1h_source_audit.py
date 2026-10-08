"""Raw field feasibility audit, independent of labels or model performance."""
import csv
from collections import Counter
import json
from pathlib import Path
import tennis_v2_data as D

FIELDS=('tourney_date','indoor','surface','minutes','winner_hand','loser_hand','winner_ht','loser_ht','w_svpt','l_svpt','w_1stIn','l_1stIn','w_1stWon','l_1stWon','w_2ndWon','l_2ndWon','w_SvGms','l_SvGms')


def audit():
    out={'manifest':'source_manifest.json','sealed_sources_opened':False,'tours':{},'limitations':{
        'tourney_date':'Tournament start only. Fixed 28-day result embargo retained.',
        'indoor':'Raw column exists although Phase0 SQLite schema discards it. I/O is venue metadata, not a certified pre-match roof state; not added after seeing performance.',
        'hand_height':'Raw columns exist although Phase0 SQLite schema discards them. No style effect authorized in current frozen specification.',
        'minutes':'Postgame duration exists but without actual match/end timestamp cannot safely reconstruct recent fatigue.',
        'wta_I':'Phase0 ITF interpretation is unreliable: observed names include WTA International Hobart/Auckland. Preserve locked I exclusion; correct classification requires separate season/event evidence gate.'}}
    for t in ('atp','wta'):
        out['tours'][t]={}
        for year in D.SEASONS:
            p=D.fetch_verified(t,year)
            with p.open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f);headers=reader.fieldnames;rows=list(reader)
            out['tours'][t][str(year)]={'n':len(rows),'sha256':D.MANIFEST['files'][f'{t}_{year}.csv']['sha256'],'headers':headers,'levels':dict(sorted(Counter(r['tourney_level'] for r in rows).items())),
               'fields':{k:{'exists':k in headers,'nonmissing':sum(r.get(k) not in (None,'','NA') for r in rows),'missing_rate':sum(r.get(k) in (None,'','NA') for r in rows)/len(rows)} for k in FIELDS},
               'indoor_values':dict(sorted(Counter(r.get('indoor','ABSENT') for r in rows).items()))}
    return out


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(audit(),sort_keys=True,indent=2)+'\n')

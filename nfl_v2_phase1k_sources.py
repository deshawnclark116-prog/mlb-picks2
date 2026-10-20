#!/usr/bin/env python3
"""Phase1K source-only audit. Reads only frozen PBP and official weekly stats."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
FIELDS = ('rusher_player_id','rushing_yards','rush_attempt','qb_scramble','qb_kneel',
          'qb_dropback','down','ydstogo','yardline_100','goal_to_go','shotgun',
          'no_huddle','run_location','run_gap','score_differential','epa','success',
          'first_down','touchdown','tackled_for_loss','game_id','posteam','defteam',
          'play_id','game_date','start_time','time_of_day')
PBP_FIELDS = set(FIELDS) | {'season','week','season_type','play_type','no_play',
                          'two_point_attempt','home_team','away_team'}
STATS_FIELDS = {'player_id','player_display_name','position','season','week',
                'season_type','game_id','team','opponent_team','carries','rushing_yards'}
POSITIONS = {'RB','FB','HB'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def present(value):
    return str(value or '').upper() not in {'','NA','NAN','NONE'}


def records(path, columns):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            # Reject future rows using metadata before exposing any outcome values.
            if int(float(row.get('season') or str(path)[-8:-4] or 0)) == 2026 and int(float(row.get('week') or 0)) > 4:
                raise ValueError('Week 5+ forbidden')
            if row.get('season_type','REG') != 'REG':
                continue
            yield {k:row[k] for k in columns if k in row}


def sources():
    h=json.loads((ART/'phase1h_source_coverage.json').read_text())
    return {str(y):{k:{n:h['seasons'][str(y)][k][n] for n in ('local_name','sha256','url')}
                   for k in ('pbp','stats')} for y in range(2023,2027)}


def verify(directory, years):
    manifest=sources()
    for y in years:
        for meta in manifest[str(y)].values():
            if sha(Path(directory)/meta['local_name']) != meta['sha256']:
                raise ValueError(f'Frozen source changed: {meta["local_name"]}')


def fetch(directory):
    """Fail closed before downloading a revised/current 2026 payload."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest=sources()
    for kind,tag in [('pbp','pbp'),('stats','stats_player')]:
        metadata=directory/f'{kind}-release.json'
        subprocess.run(['curl','-fsSL','--retry','3','--max-time','60',
                        f'https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}',
                        '-o',str(metadata)],check=True)
        assets={a['name']:a for a in json.loads(metadata.read_text())['assets']}
        for year in range(2023,2027):
            m=manifest[str(year)][kind]; dest=directory/m['local_name']
            if dest.exists() and sha(dest)==m['sha256']:
                continue
            a=assets.get(m['url'].split('/')[-1])
            if not a or a.get('digest')!='sha256:'+m['sha256']:
                raise ValueError('Frozen source no longer published; supply the exact archived bytes. No latest-data fallback.')
            temporary=dest.with_suffix(dest.suffix+'.partial')
            subprocess.run(['curl','-fsSL','--retry','3','--max-time','120',m['url'],'-o',str(temporary)],check=True)
            if sha(temporary)!=m['sha256']:
                temporary.unlink();raise ValueError('Source digest changed during retrieval')
            temporary.replace(dest)


def audit(directory):
    verify(directory,range(2023,2027))
    out={'schema':'nfl-v2-phase1k-pbp-audit-v1','sources':sources(),'seasons':{},
         'timing_limit':'Retrospective corrected PBP has no original publication vintage. Prior completed games only; conservative game-date +48h completion bound must precede target game-date -24h cutoff AND source week must be earlier. This is an approved historical doctrine, not certification of archived T24/T90 files.',
         'no_fit_or_performance':True,'blocked':['OL quality/starters','box counts','defensive absences','paid vendor data','verified QB designed-run intent'],
         'success_semantics':'nflverse success = EPA >0, not an independently charted blocking/runner success label.',
         'negative_semantics':'rushing_yards <0; zero separate; tackle-for-loss flag is charted differently and is descriptive only.',
         'qb_semantics':'qb_scramble identifies scramble; QB non-scramble is not proven designed intent. Kneels excluded; QB excluded from promotion population.'}
    for year in range(2023,2027):
        stats=list(records(Path(directory)/sources()[str(year)]['stats']['local_name'],STATS_FIELDS))
        pos={r['player_id']:r['position'] for r in stats}
        missing=Counter(); n=0;types=Counter();games=set();weeks=set();values={k:Counter() for k in ['run_location','run_gap']}
        header=set();all_rush=0
        for r in records(Path(directory)/sources()[str(year)]['pbp']['local_name'],PBP_FIELDS):
            header.update(r);games.add(r['game_id']);weeks.add(int(float(r['week'])))
            if r.get('rush_attempt')!='1' or r.get('qb_kneel')=='1' or r.get('two_point_attempt')=='1' or r.get('play_type')=='no_play':continue
            all_rush+=1; p=pos.get(r.get('rusher_player_id'),'UNKNOWN')
            typ='RB_DESIGNED_CARRY_PROXY' if p in POSITIONS else ('QB_SCRAMBLE' if p=='QB' and r.get('qb_scramble')=='1' else 'QB_NON_SCRAMBLE_INTENT_UNKNOWN' if p=='QB' else 'OTHER_OR_UNKNOWN')
            types[typ]+=1
            if p not in POSITIONS or r.get('qb_scramble')=='1':continue
            n+=1
            for k in FIELDS:missing[k]+=not present(r.get(k))
            for k in values:values[k][r.get(k) or 'MISSING']+=1
        fields={k:{'exists':k in header,'missing_count':missing[k], 'missing_rate':missing[k]/n if n else None,'usable_population':n-missing[k],
                   'status':'USABLE_PRIOR_ONLY' if k in header and missing[k]/n<=.05 else 'BLOCKED_DATA',
                   'semantic_limit':'History only; target-game value forbidden as predictor' if k not in ['game_id','posteam','defteam','play_id','rusher_player_id','game_date'] else 'Identity/join/date only'} for k in FIELDS}
        for k,dependency,meaning in [('negative_rush','rushing_yards','<0, not TFL responsibility'),('explosive_10','rushing_yards','>=10 yards'),('explosive_20','rushing_yards','>=20 yards')]:
            fields[k]={**fields[dependency],'derived_from':dependency,'semantic_limit':meaning}
        fields['designed_qb_run']={'exists':False,'status':'BLOCKED_DATA','semantic_limit':'Non-scramble does not prove designed run; exclude QB from primary.'}
        out['seasons'][str(year)]={'regular_games':len(games),'weeks':sorted(weeks),'all_non_kneel_rush_attempts':all_rush,'rb_carries':n,'rush_types':dict(types),'fields':fields,'direction_values':{k:dict(v) for k,v in values.items()}}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--fetch',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args()
    if a.fetch:fetch(a.data_dir)
    result=audit(a.data_dir);Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({y:{'rb_carries':s['rb_carries'],'blocked':[k for k,v in s['fields'].items() if v['status']=='BLOCKED_DATA']} for y,s in result['seasons'].items()},indent=2))


if __name__=='__main__':main()

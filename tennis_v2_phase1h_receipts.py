"""Receipt interpretation only. Reconstruct routed contributions, never refit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tennis_v2_phase1_guards import EvidenceBlocked


def explanation(feature,family,projected):
    own,opp=feature['own'],feature['opponent_conceded']
    if family=='U':
        base=own['overall'][0]; adjustment=0.; form=0.; unbounded=base
    elif family=='S':
        base=own['rate'][0]; adjustment=0.; form=0.; unbounded=base
    elif family=='I':
        base=own['overall'][0]; adjustment=opp['overall'][0]-own['league'][0]; form=0.; unbounded=base+adjustment
    elif family in ('C','R','HUMAN'):
        base=own['rate'][0]; adjustment=opp['rate'][0]-own['league'][0]
        form=feature['old_adjusted_residual'] if family in ('R','HUMAN') else 0.
        unbounded=base+adjustment+form
    elif family=='D':
        first=min(.90,max(.30,own['rate'][2]+opp['rate'][2]-own['league'][2]))
        second=min(.90,max(.30,own['rate'][3]+opp['rate'][3]-own['league'][3]))
        unbounded=own['rate'][1]*first+(1-own['rate'][1])*second
        base=None;adjustment=None;form=0.
    else: raise EvidenceBlocked('BLOCKED_DATA: unsupported explanation')
    result=min(.90,max(.30,unbounded))
    if abs(result-projected)>1e-12: raise EvidenceBlocked('BLOCKED_DATA: receipt arithmetic mismatch')
    return dict(player=feature['player'],opponent=feature['opponent'],family=family,
                used_own_serve=base,used_opponent_serve_conceded_adjustment=adjustment,
                used_old_trajectory_adjustment=form,unbounded_point_expectation=unbounded,
                boundary_effect=result-unbounded,final_server_point_win=result,
                first_second_channels_used=family=='D',surface_channel_used=family not in ('U','I'),
                unsupported_current_health_rest_travel='BLOCKED_DATA; no invented contribution',
                other_point_component_fields='diagnostic hypotheses; not contributions unless used here')


def enrich(source,destination):
    n=0
    destination.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(source,'rt') as inp,destination.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as out:
            for line in inp:
                r=json.loads(line)
                if not '2015-01-01'<=r['tournament_start']<='2024-12-31':
                    raise EvidenceBlocked('BLOCKED_TIMING: receipt outside allowed research')
                fam=r['distribution_family']
                r['forecast_explanation']=[explanation(f,fam,p['serve_point_win']) for f,p in zip(r['history']['players'],r['point_components'][fam])]
                r['baseline_differences']={k:r['match_probabilities'][fam]-r['match_probabilities'][k] for k in ('phase0_blend','HUMAN','U')}
                out.write((json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());n+=1
    return {'path':destination.name,'rows':n,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();print(json.dumps(enrich(a.source,a.destination),sort_keys=True))

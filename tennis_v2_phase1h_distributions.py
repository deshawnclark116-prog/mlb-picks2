"""Exact set/match race distributions under the existing stationary scoring model."""
from collections import defaultdict
from functools import lru_cache

from tennis_v2_phase1_guards import EvidenceBlocked
from tennis_v2_phase1_scoring import hold_probability, tiebreak_win_probability, _advantage_two_game_win


@lru_cache(maxsize=512)
def set_kernel(p1,p2,rule,first_server):
    """(winner, next server, tiebreak played) probability mass; no score sampling."""
    if rule not in ('TB7_AT_6_ALL_SETS','TB10_AT_6_FINAL','ADVANTAGE_FINAL_SET','TB7_AT_12_FINAL') or first_server not in (1,2):
        raise EvidenceBlocked('BLOCKED_FORMAT: set kernel')
    a,b = hold_probability(p1),1-hold_probability(p2)
    @lru_cache(None)
    def solve(g1,g2,server):
        if g1>=6 and g1-g2>=2: return {(1,server,0):1.}
        if g2>=6 and g2-g1>=2: return {(2,server,0):1.}
        if g1==g2 and g1 in (6,12):
            if g1==6 and rule=='ADVANTAGE_FINAL_SET':
                q = _advantage_two_game_win(a,b)
                return {(1,server,0):q,(2,server,0):1-q}
            if not (g1==6 and rule=='TB7_AT_12_FINAL'):
                q = tiebreak_win_probability(p1,p2,first_server=server,points_to_win=10 if rule=='TB10_AT_6_FINAL' else 7)
                return {(1,3-server,1):q,(2,3-server,1):1-q}
        q = a if server==1 else b
        out = defaultdict(float)
        for k,v in solve(g1+1,g2,3-server).items(): out[k] += q*v
        for k,v in solve(g1,g2+1,3-server).items(): out[k] += (1-q)*v
        return dict(out)
    return tuple(sorted(solve(0,0,first_server).items()))


def match_distribution(p1,p2,fmt):
    if fmt.best_of not in (3,5): raise EvidenceBlocked('BLOCKED_FORMAT: match distribution')
    need = fmt.best_of//2+1
    live = {(0,0,1,0):.5,(0,0,2,0):.5}
    finals = defaultdict(float); tbmass = defaultdict(float)
    while live:
        nxt = defaultdict(float)
        for (s1,s2,server,tbs),mass in live.items():
            deciding = s1==s2==need-1
            rule = fmt.final_set_rule if deciding else 'TB7_AT_6_ALL_SETS'
            for (winner,ns,tb),v in set_kernel(p1,p2,rule,server):
                a,b = s1+(winner==1),s2+(winner==2)
                if a==need or b==need:
                    finals[f'{a}-{b}'] += mass*v
                    tbmass[str(tbs+tb)] += mass*v
                else: nxt[(a,b,ns,tbs+tb)] += mass*v
        live = nxt
    return dict(set_score=dict(sorted(finals.items())),tiebreak_count=dict(sorted(tbmass.items())),p1_win=sum(v for k,v in finals.items() if int(k.split('-')[0])==need))

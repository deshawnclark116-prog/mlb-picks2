"""Phase1H research inputs: source-locked, tour-isolated, embargoed evidence."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math

import tennis_v2_data as D
from tennis_v2_incumbent import Elo
from tennis_v2_phase1_guards import EvidenceBlocked, FormatEvidence

P = json.loads((D.OUT / 'phase1h_protocol_addendum.json').read_text())
SURFACES = {'Hard', 'Clay', 'Grass', 'Carpet'}


def rule_for(row):
    """Only declared historical formats. Never infer a rule from a score."""
    season = int(row['match_date'][:4])
    if not 2015 <= season <= 2024:
        raise EvidenceBlocked('BLOCKED_TIMING: research season outside allowed range')
    if row['best_of'] not in (3, 5):
        raise EvidenceBlocked('BLOCKED_FORMAT: best of')
    if row['tourney_level'] in ('A', 'M', 'F', 'P', 'PM') and row['best_of'] == 3:
        # Nonstandard invitational/short-set events do not inherit ATP rules.
        if any(x in (row['tourney_name'] or '').lower() for x in ('next gen', 'nextgen', 'laver', 'hopman', 'olympic', 'cup')):
            raise EvidenceBlocked('BLOCKED_FORMAT: special event')
        return FormatEvidence(3, 'TB7_AT_6_ALL_SETS')
    if row['tourney_level'] != 'G':
        raise EvidenceBlocked('BLOCKED_FORMAT: unqualified competition')
    name = row['tourney_name']
    if name not in ('Australian Open', 'Roland Garros', 'Wimbledon', 'US Open'):
        raise EvidenceBlocked('BLOCKED_FORMAT: unknown major identity')
    if row['best_of'] != (5 if row['tour'] == 'atp' else 3):
        raise EvidenceBlocked('BLOCKED_FORMAT: major format mismatch')
    if season >= 2022 or (name == 'Australian Open' and season >= 2019):
        rule = 'TB10_AT_6_FINAL'
    elif name == 'Wimbledon' and season >= 2019:
        rule = 'TB7_AT_12_FINAL'
    elif name == 'US Open':
        rule = 'TB7_AT_6_ALL_SETS'
    else:
        rule = 'ADVANTAGE_FINAL_SET'
    return FormatEvidence(row['best_of'], rule)


def stats_channels(x):
    """Observed count numerators/denominators; invalid/missing stays None."""
    keys = ('serve_points', 'first_serve_in', 'first_serve_won', 'second_serve_won')
    v = [x.get(k) for k in keys]
    if any(type(a) is not int for a in v):
        return None
    n, i, f, s = v
    if not (n > 0 and 0 <= i <= n and 0 <= f <= i and 0 <= s <= n-i):
        return None
    for k in ('aces', 'double_faults'):
        if x.get(k) is not None and (type(x[k]) is not int or not 0 <= x[k] <= n):
            return None
    if x.get('aces') is not None and x.get('double_faults') is not None and x['aces'] + x['double_faults'] > n:
        return None
    b, z, g = x.get('break_points_faced'), x.get('break_points_saved'), x.get('serve_games')
    if b is not None and z is not None and not (0 <= z <= b):
        return None
    return ([f+s, i, f, s], [n, n, i, n-i])


def observed_hold(x):
    g, f, s = x.get('serve_games'), x.get('break_points_faced'), x.get('break_points_saved')
    if any(type(a) is not int for a in (g, f, s)) or g <= 0 or not 0 <= f-s <= g or not 0 <= s <= f:
        return None
    return 1-(f-s)/g


@dataclass(frozen=True)
class Target:
    tour: str
    event: str
    day: str
    surface: str
    p1: str
    p2: str
    fmt: FormatEvidence


class State:
    """Builder only sees explicit pre-event Target and previously released records."""
    def __init__(self, tour):
        if tour not in ('atp', 'wta'):
            raise EvidenceBlocked('BLOCKED_DATA: tour')
        self.tour = tour
        self.elo = Elo()
        self.serve = defaultdict(lambda: deque(maxlen=50))
        self.conceded = defaultdict(lambda: deque(maxlen=50))
        self.residual = defaultdict(lambda: deque(maxlen=10))
        self.league = [0.]*4, [0.]*4
        self.released = []
        self.max_day = None
        self.history_hash = hashlib.sha256(b'').hexdigest()

    def _rates(self, hist, surface, use_surface):
        prior = P['hyperparameters']['initial_tour_rates'][self.tour]
        league = [(w+300*p)/(n+300) for w,n,p in zip(*self.league,prior)]
        def pool(h, base):
            num = [sum(r[1][j] for r in h) for j in range(4)]
            den = [sum(r[2][j] for r in h) for j in range(4)]
            return [(w+300*p)/(n+300) for w,n,p in zip(num,den,base)], den
        overall, den = pool(hist, league)
        if use_surface:
            rate, surface_den = pool([r for r in hist if r[0] == surface], overall)
        else:
            rate, surface_den = overall, [0]*4
        return dict(rate=rate, overall=overall, points=den, surface_points=surface_den, league=league)

    def features(self, target):
        if target.tour != self.tour or target.surface not in SURFACES or target.p1 == target.p2:
            raise EvidenceBlocked('BLOCKED_DATA: target identity or surface')
        cutoff = (date.fromisoformat(target.day)-timedelta(days=28)).isoformat()
        if self.max_day and self.max_day > cutoff:
            raise EvidenceBlocked('BLOCKED_TIMING: unreleased history')
        if any(e == target.event for _,e,_ in self.released):
            raise EvidenceBlocked('BLOCKED_TIMING: same event')
        out = []
        for pid, opp in ((target.p1,target.p2),(target.p2,target.p1)):
            own = self._rates(self.serve[pid], target.surface, True)
            ret = self._rates(self.conceded[opp], target.surface, True)
            rs = self.residual[pid]
            residual = sum(v*n for v,n in rs)/(1000+sum(n for _,n in rs))
            # Prior beta-binomial SE is only a sampling diagnostic, not a
            # certified match-day performance interval (points are dependent).
            n = own['points'][0]
            se = math.sqrt(own['rate'][0]*(1-own['rate'][0])/(n+301))
            out.append(dict(player=pid, opponent=opp, own=own, opponent_conceded=ret,
                            old_adjusted_residual=residual, residual_matches=len(rs),
                            sampling_se_only=se, prior_matches=self.elo.on.get(pid,0)))
        ratings = self.elo.predict(target.p1,target.p2,target.surface)
        def shrunk(pid):
            n = self.elo.sn.get((pid,target.surface),0)
            return (n*self.elo.se.get((pid,target.surface),1500.)+15*self.elo.oe.get(pid,1500.))/(n+15)
        from tennis_v2_incumbent import elo_expected
        ratings['surface_shrunk'] = elo_expected(shrunk(target.p1),shrunk(target.p2))
        return dict(players=out, cutoff=cutoff, latest_source_start=self.max_day,
                    history_hash=self.history_hash,
                    rating=ratings)

    def release(self, row, player_stats, prior_c):
        """Only replay controls availability; features() independently verifies cutoff."""
        if row['tour'] != self.tour or self.max_day and row['match_date'] < self.max_day:
            raise EvidenceBlocked('BLOCKED_TIMING: unordered history')
        if 'W/O' in row['score'].upper() or not row['score']:
            return
        p1,p2 = sorted((row['winner_id'],row['loser_id']))
        if p1 == p2 or not p1 or not p2:
            raise EvidenceBlocked('BLOCKED_DATA: history identity')
        self.elo.update(p1,p2,row['surface'],int(p1==row['winner_id']))
        self.released.append((row['match_date'],row['tourney_id'],row['match_id']))
        self.history_hash = hashlib.sha256((self.history_hash+json.dumps(self.released[-1],separators=(',',':'))).encode()).hexdigest()
        self.max_day = row['match_date']
        if row['is_incomplete'] or any(stats_channels(player_stats.get(p,{})) is None for p in (p1,p2)):
            return
        for pid,opp in ((p1,p2),(p2,p1)):
            wins, ns = stats_channels(player_stats[pid])
            self.serve[pid].append((row['surface'], wins, ns))
            # Opponent's conceded channel is serve WIN fraction allowed,
            # equivalent to one minus their return WIN fraction.
            self.conceded[opp].append((row['surface'], wins, ns))
            for j in range(4):
                self.league[0][j] += wins[j]
                self.league[1][j] += ns[j]
            if prior_c and pid in prior_c:
                self.residual[pid].append((wins[0]/ns[0]-prior_c[pid],ns[0]))


def load_tour(tour):
    """Read ONLY manifest-permitted files; fail closed before building state."""
    rows, stats, audit = [], {}, {}
    ids = set()
    for year in D.SEASONS:
        path = D.fetch_verified(tour,year)
        ms, ps = D.F.parse_matches_csv(path,tour)
        for r in ms:
            if r['match_id'] in ids:
                raise EvidenceBlocked('BLOCKED_DATA: duplicate source identity')
            ids.add(r['match_id'])
            date.fromisoformat(r['match_date'])
        for x in ps:
            key = (x['match_id'],x['player_id'])
            if key in stats:
                raise EvidenceBlocked('BLOCKED_DATA: duplicate player row')
            stats[key] = x
        reasons = Counter()
        for r in ms:
            try: rule_for(r)
            except EvidenceBlocked as e: reasons[str(e)] += 1
        audit[str(year)] = dict(source_rows=len(ms), player_rows=len(ps), valid_point_rows=sum(stats_channels(x) is not None for x in ps), missing_or_invalid_point_rows=sum(stats_channels(x) is None for x in ps), format_blocked=dict(reasons), sha256=D.MANIFEST['files'][f'{tour}_{year}.csv']['sha256'])
        rows.extend(ms)
    return sorted(rows,key=D.canonical_key),stats,audit

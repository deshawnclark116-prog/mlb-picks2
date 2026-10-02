"""
NFL_PHASE1B_DATA  (Phase 1B, shadow research)

Per-opportunity outcome tallies + AS-OF features for every Phase 1A candidate player-game.

Unit of analysis: (team-game, candidate). All pregame features are as-of the GAME (not the play), so every carry / target /
attempt of a player in a game shares one feature record. That is exactly the interface the Phase 1A opportunity samples
need in the development assembly, and it keeps training and serving identical.

Reuse (no duplicated parser):
  * nfl_context_v4.PlayData  -> opponent defensive scheme profile (box / stacked-box rate, blitz, pressure, man/zone,
    Cover 0/1/2/3/4/6, explosive-run / explosive-pass rates allowed, sack rate, red-zone TD rates allowed) and the
    player-vs-scheme weekly splits (RB vs stacked/light boxes, receiver vs man/zone). Its as-of rule is week-level
    ('games strictly before this week'), which is conservative relative to Phase 1A's kickoff-level cutoff.
  * nfl_phase1_data.Data / units -> candidates, injury report (assumption A2), defenders and their T-24h statuses.
New here: per-PLAY outcome tallies (yards, air yards, catch, YAC, sack, INT, TD; RZ flag) per (season, week, player), decayed
as-of accumulators (player, position, league, team offense, team defense allowed), OL availability (injury-report based).

Never used: realized same-game participation, same-game box / personnel / coverage, sportsbook columns.
2026: participation (coverage, pressure, man/zone) is unpublished, so those rates come from prior-season accumulators
that simply stop growing (stale but as-of legal); FTN box / blitz continue through 2026.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1

GAMMAS = np.array([1.0, 0.985, 0.95])          # weekly recency decay candidates (tuned on validation per component)
NG = len(GAMMAS)
GRID = np.arange(-10, 100)                       # integer support for yards / air yards / YAC (110 points)
G = len(GRID)
RZ = 20


def week_index(s, w):
    return (s - 2022) * 22 + w


def gclip(y):
    return int(min(max(y, -10), 99)) + 10


# ------------------------------------------------------------------ bin systems (indices into GRID)
def make_bins(lumps_lo, singles, coarse, hi=99):
    """Bins as (lo, hi) inclusive integer intervals: one lower lump, singles, coarse upper bins."""
    bins = [lumps_lo] + [(v, v) for v in singles] + list(coarse)
    return bins


RUSH_BINS = make_bins((-10, -4), range(-3, 30), [(30, 34), (35, 39), (40, 49), (50, 59), (60, 79), (80, 99)])      # 40 bins
AIR_BINS = make_bins((-10, -6), range(-5, 26), [(26, 29), (30, 34), (35, 39), (40, 49), (50, 99)])                  # 37 bins
YAC_BINS = make_bins((-10, -1), range(0, 31), [(31, 35), (36, 40), (41, 50), (51, 60), (61, 99)])                    # 37 bins


def bin_matrix(bins):
    """M [G, K] (indicator of grid point in bin) and E [K, G] (uniform expansion of a bin over its grid points)."""
    K = len(bins)
    M = np.zeros((G, K)); E = np.zeros((K, G))
    for k, (lo, hi) in enumerate(bins):
        idx = np.arange(lo, hi + 1) + 10
        M[idx, k] = 1.0
        E[k, idx] = 1.0 / len(idx)
    assert M.sum() == G and abs(E.sum() - K) < 1e-9
    return M, E


M_R, E_R = bin_matrix(RUSH_BINS)
M_A, E_A = bin_matrix(AIR_BINS)
M_Y, E_Y = bin_matrix(YAC_BINS)
K_R, K_A, K_Y = len(RUSH_BINS), len(AIR_BINS), len(YAC_BINS)


def air_bucket(a):
    return 0 if a < 0 else 1 if a <= 4 else 2 if a <= 14 else 3


# ------------------------------------------------------------------ decayed accumulators
class Store:
    """key -> [NG, dim] decayed sums, as of a week index (lazy decay)."""

    def __init__(self, dim):
        self.dim = dim
        self.d = {}

    def get(self, key, wi):
        e = self.d.get(key)
        if e is None:
            return np.zeros((NG, self.dim))
        vec, last = e
        return vec * (GAMMAS ** (wi - last))[:, None] if wi > last else vec

    def add(self, key, delta, wi):
        cur = self.get(key, wi)
        self.d[key] = [cur + delta[None, :], wi]


def fnum(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ per-play tallies
class PlayTallies:
    """Per (season, week, gid) and per (season, week, team) tallies from play-by-play (same filters as Phase 1A)."""

    def __init__(self, data_dir, seasons):
        d = Path(data_dir)
        self.rush = defaultdict(lambda: np.zeros(G))                 # (s,w,gid) exact yard histogram
        self.rush_ev = defaultdict(lambda: np.zeros(6))              # rz_n, rz_td, out_n, out_td, outside_dir_n, inside_dir_n
        self.rush_team = defaultdict(lambda: np.zeros(8))            # (s,w,team) offense; and ('D', ...) defense allowed
        self.tgt_air = defaultdict(lambda: np.zeros(G))              # exact air-yard histogram (all targets, air defined)
        self.tgt_cb = defaultdict(lambda: np.zeros((4, 2)))          # by air bucket: [targets, catches]
        self.tgt_yac = defaultdict(lambda: np.zeros((4, G)))         # exact YAC histogram of catches by air bucket
        self.tgt_ev = defaultdict(lambda: np.zeros(4))               # rz_n, rz_td, out_n, out_td (targets)
        self.tgt_n_total = defaultdict(float)                        # all targets incl. air undefined (reconciliation)
        self.pass_ = defaultdict(lambda: np.zeros(6))                # dropbacks, sacks, attempts, comps, ints, tds
        self.pass_air = defaultdict(lambda: np.zeros(G))
        self.pass_cb = defaultdict(lambda: np.zeros((4, 2)))
        self.pass_yac = defaultdict(lambda: np.zeros((4, G)))
        self.recv_team = defaultdict(lambda: np.zeros(5))            # (s,w,team): tgt, catches, air_sum, yac_sum, catches (air defined)
        self.pass_team = defaultdict(lambda: np.zeros(8))            # (s,w,team) passing side (offense) and 'D' allowed
        self.tgt_grp = defaultdict(lambda: np.zeros(4))              # per-target outcome group: incomplete, <10, 10-19, 20+ (receiving yards)
        self.pass_grp = defaultdict(lambda: np.zeros(4))
        self.zone_rush = {"rz": np.zeros(G), "out": np.zeros(G)}         # league integer-yard histograms by field zone (Phase 1C zone likelihood ratios)
        self.zone_comp = {"rz": np.zeros(G), "out": np.zeros(G)}         # completion receiving yards by zone
        self.zone_targets = {"rz": np.zeros(2), "out": np.zeros(2)}     # [targets, completions] by zone
        self.tm = {}                                                 # (s,w,gid) -> (team, opp)
        self.play_rows = 0
        for s in seasons:
            with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                        continue
                    if r.get("two_point_attempt") == "1" or r.get("qb_kneel") == "1" or r.get("qb_spike") == "1":
                        continue
                    self.play_rows += 1
                    w = int(r["week"]); off, de = r["posteam"], r["defteam"]
                    yl = fnum(r["yardline_100"]); rz = yl is not None and yl <= RZ
                    rid, qid, pid = r["rusher_player_id"], r["receiver_player_id"], r["passer_player_id"]
                    for g_ in (rid, qid, pid):
                        if g_:
                            self.tm[(s, w, g_)] = (off, de)
                    if r["rush_attempt"] == "1" and rid:
                        y = fnum(r["rushing_yards"]) or 0.0
                        self.rush[(s, w, rid)][gclip(y)] += 1
                        self.zone_rush["rz" if rz else "out"][gclip(y)] += 1
                        ev = self.rush_ev[(s, w, rid)]
                        td = r["rush_touchdown"] == "1"
                        ev[0 if rz else 2] += 1; ev[1 if rz else 3] += td
                        loc = r.get("run_location") or ""
                        if loc in ("left", "right"):
                            ev[4] += 1                              # outside-tackle-ish (left/right); 'middle' = inside
                        elif loc == "middle":
                            ev[5] += 1
                        for key in ((s, w, off), ("D", s, w, de)):
                            t = self.rush_team[key]
                            t[0] += 1; t[1] += y; t[2] += y < 0; t[3] += y >= 15; t[4] += y >= 20; t[5] += y >= 25
                            t[6] += y if y < 20 else 0.0; t[7] += y < 20
                    is_pass = r["pass_attempt"] == "1"
                    if r["qb_dropback"] == "1" and pid:
                        pt = self.pass_[(s, w, pid)]
                        pt[0] += 1
                        sk = r["sack"] == "1"
                        pt[1] += sk
                        for key in ((s, w, off), ("D", s, w, de)):
                            tt = self.pass_team[key]; tt[0] += 1; tt[1] += sk
                        if is_pass and not sk:
                            pt[2] += 1; pt[3] += r["complete_pass"] == "1"; pt[4] += r["interception"] == "1"; pt[5] += r["pass_touchdown"] == "1"
                            for key in ((s, w, off), ("D", s, w, de)):
                                tt = self.pass_team[key]; tt[2] += 1; tt[3] += r["complete_pass"] == "1"; tt[4] += r["interception"] == "1"; tt[5] += r["pass_touchdown"] == "1"
                    elif is_pass and pid and r["sack"] != "1":
                        # attempts without a recorded dropback flag (rare): still count the attempt
                        pt = self.pass_[(s, w, pid)]
                        pt[0] += 1; pt[2] += 1; pt[3] += r["complete_pass"] == "1"; pt[4] += r["interception"] == "1"; pt[5] += r["pass_touchdown"] == "1"
                    if is_pass and r["sack"] != "1":
                        ay = fnum(r["air_yards"])
                        comp = r["complete_pass"] == "1"
                        ry = (fnum(r["receiving_yards"]) or 0.0) if comp else 0.0
                        grp = 0 if not comp else 1 if ry < 10 else 2 if ry < 20 else 3
                        if qid:
                            self.zone_targets["rz" if rz else "out"] += (1, comp)
                        if comp:
                            self.zone_comp["rz" if rz else "out"][gclip(ry)] += 1
                        if pid:
                            self.pass_grp[(s, w, pid)][grp] += 1
                        if qid:
                            self.tgt_grp[(s, w, qid)][grp] += 1
                            self.tgt_n_total[(s, w, qid)] += 1
                            tev = self.tgt_ev[(s, w, qid)]
                            tdp = r["pass_touchdown"] == "1"
                            tev[0 if rz else 2] += 1; tev[1 if rz else 3] += tdp
                        if ay is not None:
                            b = air_bucket(ay)
                            if pid:
                                self.pass_air[(s, w, pid)][gclip(ay)] += 1
                                self.pass_cb[(s, w, pid)][b, 0] += 1; self.pass_cb[(s, w, pid)][b, 1] += comp
                                if comp:
                                    self.pass_yac[(s, w, pid)][b, gclip(ry - ay)] += 1
                            if qid:
                                self.tgt_air[(s, w, qid)][gclip(ay)] += 1
                                self.tgt_cb[(s, w, qid)][b, 0] += 1; self.tgt_cb[(s, w, qid)][b, 1] += comp
                                if comp:
                                    self.tgt_yac[(s, w, qid)][b, gclip(ry - ay)] += 1
                            for key in ((s, w, off), ("D", s, w, de)):
                                rt = self.recv_team[key]
                                rt[0] += 1; rt[1] += comp
                                if comp:
                                    rt[2] += ay; rt[3] += ry - ay; rt[4] += 1
        self.players_by_week = defaultdict(set)
        for store in (self.rush, self.tgt_air, self.pass_):
            for (s, w, gid) in store:
                self.players_by_week[(s, w)].add(gid)


# ================================================================== as-of record builder
OL_POS = {"T", "G", "C", "OL", "OT", "OG"}
DL_POS = {"DE", "DT", "NT", "DL"}
LB_POS = {"LB", "ILB", "OLB", "MLB"}
DB_POS = {"CB", "S", "SS", "FS", "DB"}
OUT_ST = ("Out", "Doubtful")
GI = 1                                    # decay index used for team / opponent context rates (gamma 0.985)
K_SHRINK = {"rush": 150.0, "pass": 300.0, "sack": 300.0}

SCHEME_RUSH = ["def_stacked_rate", "def_avg_box", "def_blitz_rate", "def_rush_epa"]
SCHEME_PASS = ["def_man_rate", "def_c0", "def_c1", "def_c2", "def_c4", "def_c6", "def_blitz_rate", "def_pressure_rate"]


def shrunk_rate(num, den, lg, k):
    return (num + k * lg) / (den + k)


def pos_group(pos, kind):
    if kind == "rush":
        return "QB" if pos == "QB" else "RB" if pos in ("RB", "FB") else "OTH"
    if kind == "rec":
        return "RB" if pos in ("RB", "FB") else "TE" if pos == "TE" else "WR"
    return "QB"


def build_inj_index(D):
    """(s,w,team) -> Out/Doubtful and Questionable counts by unit (historical final-weekly-report proxy, assumption A2)."""
    idx = defaultdict(lambda: {"ol_out": 0, "ol_q": 0, "dl_out": 0, "lb_out": 0, "db_out": 0, "dl_q": 0, "lb_q": 0, "db_q": 0})
    for (s, w, gid), (st, pr, tm) in D.inj.items():
        pos = (D.players.get(gid) or {}).get("pos")
        out, q = st in OUT_ST, st == "Questionable"
        if not pos or not (out or q):
            continue
        u = "ol" if pos in OL_POS else "dl" if pos in DL_POS else "lb" if pos in LB_POS else "db" if pos in DB_POS else None
        if u is not None:
            idx[(s, w, tm)][f"{u}_out" if out else f"{u}_q"] += 1
    return idx


def last_known_pos(D, s, w, gid):
    """Position as of the forecast: this week's roster entry when it exists (game-day roster, usable at T-90m and in research replays), else the most recent earlier roster entry of the
    season (Phase 1D: at T-24h the current-week roster does not exist yet), else players.csv."""
    r = D.roster.get((s, w, gid))
    if r and r.get("pos"):
        return r["pos"]
    for back in range(1, 5):
        if w - back < 1:
            break
        r = D.roster.get((s, w - back, gid))
        if r and r.get("pos"):
            return r["pos"]
    return (D.players.get(gid) or {}).get("pos")


class Records:
    """Container: per component, aligned arrays of per-player-game records."""

    def __init__(self, rows):
        self.rows = rows
        self.n = len(rows)
        self.active = np.array([not r.get("extra", False) for r in rows], bool)
        self.s = np.array([r["s"] for r in rows]); self.w = np.array([r["w"] for r in rows])
        self.key = [r["key"] for r in rows]
        self.pos = np.array([r["pg"] for r in rows])
        self.n_opp = np.array([r["n"] for r in rows], float)
        self.cnt = {k: np.array([r["cnt"][k] for r in rows], float) for k in rows[0]["cnt"]} if rows else {}
        self.base = {k: np.array([r["base"][k] for r in rows], float) for k in rows[0]["base"]} if rows else {}
        self.fam = {f: np.array([r["fam"][f] for r in rows], float) for f in rows[0]["fam"]} if rows else {}

    def sel(self, mask):
        return Records([r for r, m in zip(self.rows, mask) if m])


def build_records(D, T, PD, inj, progress=None, extras=None):
    """extras: {(s, w): [(gid, team), ...]} forecast-only candidates (players Phase 1A may allocate opportunities to whether or not they
    realized any). They get zero-count records (flag extra=True), are excluded from every metric and fit (Records.active), and never
    enter the as-of stores."""
    extras = extras or {}
    S = {}

    def st(name, dim):
        S[name] = Store(dim)
        return S[name]
    for nm, dim in (("rush", K_R), ("air", K_A), ("cb", 8), ("yac", 4 * K_Y), ("pass", 6), ("tev", 4), ("rev", 4)):
        for lvl in ("p", "pos", "lg"):
            st(f"{nm}_{lvl}", dim)
    st("t_rush", 8); st("t_recv", 5); st("t_pass", 8)
    by_week = defaultdict(list)
    for k in T.rush_team:
        by_week[(k[-3], k[-2])].append(("rush", k))
    for k in T.recv_team:
        by_week[(k[-3], k[-2])].append(("recv", k))
    for k in T.pass_team:
        by_week[(k[-3], k[-2])].append(("pass", k))
    weeks = sorted(set(T.players_by_week) | set(extras))            # Phase 1D live mode: a target week may have no realized play yet, only forecast-only candidates
    rec = {"rush": [], "rec": [], "pass": []}
    def_cache = {}
    lgvals = {}

    def defp(s, w, team):
        k = (s, w, team)
        if k not in def_cache:
            def_cache[k] = PD.def_profile(s, w, team)
        return def_cache[k]

    def tkey(side, team):
        return ("O", team) if side == "O" else ("D", team)

    def tstats(name, side, team, wi):
        return S[name].get(tkey(side, team), wi)[GI]

    def rush_ctx(side, team, wi):
        x = tstats("t_rush", side, team, wi); lg = S["t_rush"].get("lg", wi)[GI]
        n = x[0]; k = K_SHRINK["rush"]
        out = []
        for j in (3, 2):                                                    # >=15 rate, negative rate
            out.append(shrunk_rate(x[j], n, lg[j] / max(lg[0], 1.0), k))
        out.append(shrunk_rate(x[1], n, lg[1] / max(lg[0], 1.0), k))       # mean yards
        return out

    def pass_ctx(side, team, wi):
        r = tstats("t_recv", side, team, wi); lr = S["t_recv"].get("lg", wi)[GI]
        p = tstats("t_pass", side, team, wi); lp = S["t_pass"].get("lg", wi)[GI]
        k = K_SHRINK["pass"]
        return [shrunk_rate(r[1], r[0], lr[1] / max(lr[0], 1.0), k),                                        # completion rate
                shrunk_rate(r[2], r[4], lr[2] / max(lr[4], 1.0), 60.0),                                    # air yards per completion
                shrunk_rate(r[3], r[4], lr[3] / max(lr[4], 1.0), 60.0),                                    # YAC per completion
                shrunk_rate(p[1], p[0], lp[1] / max(lp[0], 1.0), K_SHRINK["sack"]),                        # sack rate per dropback
                shrunk_rate(p[4], p[2], lp[4] / max(lp[2], 1.0), K_SHRINK["pass"]),                        # INT per attempt
                shrunk_rate(p[5], p[2], lp[5] / max(lp[2], 1.0), K_SHRINK["pass"])]                        # TD per attempt

    def scheme(s, w, opp, names):
        d = defp(s, w, opp)
        return [d.get(nm) if d.get(nm) is not None else np.nan for nm in names]

    def personnel(s, w, team, opp):
        a, b = inj.get((s, w, team)), inj.get((s, w, opp))
        a = a or {"ol_out": 0, "ol_q": 0}; b = b or {"dl_out": 0, "lb_out": 0, "db_out": 0, "dl_q": 0, "lb_q": 0, "db_q": 0}
        return [a["ol_out"], a["ol_q"], b["dl_out"] + 0.5 * b["dl_q"], b["lb_out"] + 0.5 * b["lb_q"], b["db_out"] + 0.5 * b["db_q"]]

    def fam(kind, s, w, team, opp, wi):
        home = 1.0 if D.game.get((s, w, team), {}).get("home") else 0.0
        if kind == "rush":
            return {"team": rush_ctx("O", team, wi) + [home], "opp": rush_ctx("D", opp, wi), "scheme": scheme(s, w, opp, SCHEME_RUSH),
                    "personnel": personnel(s, w, team, opp)}
        pc_t, pc_o = pass_ctx("O", team, wi), pass_ctx("D", opp, wi)
        return {"team": pc_t + [home], "opp": pc_o, "scheme": scheme(s, w, opp, SCHEME_PASS), "personnel": personnel(s, w, team, opp)}

    for (s, w) in weeks:
        wi = week_index(s, w)
        emitted = []
        realized = T.players_by_week[(s, w)]
        exmap = {g: t for g, t in extras.get((s, w), [])}
        zg = np.zeros(G); z4 = np.zeros(4); z6 = np.zeros(6); z42 = np.zeros((4, 2)); z4g = np.zeros((4, G))
        for gid in sorted(set(realized) | set(exmap)):
            real = gid in realized
            if real:
                team, opp = T.tm[(s, w, gid)]
            else:
                team = exmap[gid]; opp = D.game[(s, w, team)]["opp"]
            pos = last_known_pos(D, s, w, gid) or "UNK"
            k3 = (s, w, gid)
            ex = k3 not in T.rush
            if not ex or (gid in exmap and pos in ("RB", "FB", "QB", "WR", "TE")):
                pg = pos_group(pos, "rush"); h = T.rush.get(k3, zg)
                ev = T.rush_ev.get(k3, np.zeros(6))[:4]
                rec["rush"].append({"s": s, "w": w, "key": k3, "pg": pg, "n": float(h.sum()), "team": team, "opp": opp, "extra": ex,
                                    "cnt": {"y": h @ M_R, "ev": ev.copy()},
                                    "base": {"pl": S["rush_p"].get(gid, wi), "pp": S["rush_pos"].get(pg, wi), "lg": S["rush_lg"].get("lg", wi),
                                             "evp": S["rev_p"].get(gid, wi), "evpos": S["rev_pos"].get(pg, wi), "evlg": S["rev_lg"].get("lg", wi)},
                                    "fam": fam("rush", s, w, team, opp, wi)})
            ex = k3 not in T.tgt_n_total
            if not ex or (gid in exmap and pos in ("RB", "FB", "WR", "TE")):
                pg = pos_group(pos, "rec")
                ah = T.tgt_air.get(k3, zg); cb = T.tgt_cb.get(k3, z42); yh = T.tgt_yac.get(k3, z4g); tv = T.tgt_ev.get(k3, z4)
                rec["rec"].append({"s": s, "w": w, "key": k3, "pg": pg, "n": float(ah.sum()), "team": team, "opp": opp, "extra": ex,
                                   "cnt": {"air": ah @ M_A, "cb": cb.reshape(-1), "yac": np.concatenate([yh[b] @ M_Y for b in range(4)]), "tev": tv.copy(),
                                           "grp": T.tgt_grp.get(k3, z4).copy()},
                                   "base": {"air_pl": S["air_p"].get(gid, wi), "air_pp": S["air_pos"].get(pg, wi), "air_lg": S["air_lg"].get("lg", wi),
                                            "cb_pl": S["cb_p"].get(gid, wi), "cb_pp": S["cb_pos"].get(pg, wi), "cb_lg": S["cb_lg"].get("lg", wi),
                                            "yac_pl": S["yac_p"].get(gid, wi), "yac_pp": S["yac_pos"].get(pg, wi), "yac_lg": S["yac_lg"].get("lg", wi),
                                            "tv_pl": S["tev_p"].get(gid, wi), "tv_pp": S["tev_pos"].get(pg, wi), "tv_lg": S["tev_lg"].get("lg", wi)},
                                   "fam": fam("rec", s, w, team, opp, wi)})
            ex = not (k3 in T.pass_ and T.pass_[k3][0] >= 1)
            if not ex or (gid in exmap and pos == "QB"):
                pt = T.pass_.get(k3, z6); ah = T.pass_air.get(k3, zg); cb = T.pass_cb.get(k3, z42); yh = T.pass_yac.get(k3, z4g)
                rec["pass"].append({"s": s, "w": w, "key": k3, "pg": "QB", "n": float(pt[0]), "team": team, "opp": opp, "extra": ex,
                                    "cnt": {"pt": pt.copy(), "air": ah @ M_A, "cb": cb.reshape(-1), "yac": np.concatenate([yh[b] @ M_Y for b in range(4)]),
                                            "grp": T.pass_grp.get(k3, z4).copy()},
                                    "base": {"pt_pl": S["pass_p"].get(gid, wi), "pt_pp": S["pass_pos"].get("QB", wi), "pt_lg": S["pass_lg"].get("lg", wi),
                                             "air_pl": S["air_p"].get(("P", gid), wi), "air_pp": S["air_pos"].get("PQB", wi), "air_lg": S["air_lg"].get("lg", wi),
                                             "cb_pl": S["cb_p"].get(("P", gid), wi), "cb_pp": S["cb_pos"].get("PQB", wi), "cb_lg": S["cb_lg"].get("lg", wi),
                                             "yac_pl": S["yac_p"].get(("P", gid), wi), "yac_pp": S["yac_pos"].get("PQB", wi), "yac_lg": S["yac_lg"].get("lg", wi)},
                                    "fam": fam("pass", s, w, team, opp, wi)})
            if real:
                emitted.append((gid, pos))
        # absorb week (after every record of the week is emitted)
        for gid, pos in emitted:
            k3 = (s, w, gid)
            if k3 in T.rush:
                pg = pos_group(pos, "rush"); y = T.rush[k3] @ M_R
                S["rush_p"].add(gid, y, wi); S["rush_pos"].add(pg, y, wi); S["rush_lg"].add("lg", y, wi)
                ev = T.rush_ev[k3][:4]
                S["rev_p"].add(gid, ev, wi); S["rev_pos"].add(pg, ev, wi); S["rev_lg"].add("lg", ev, wi)
            if k3 in T.tgt_n_total:
                pg = pos_group(pos, "rec")
                ah = T.tgt_air[k3] @ M_A; cb = T.tgt_cb[k3].reshape(-1); yh = np.concatenate([T.tgt_yac[k3][b] @ M_Y for b in range(4)])
                S["air_p"].add(gid, ah, wi); S["air_pos"].add(pg, ah, wi); S["air_lg"].add("lg", ah, wi)
                S["cb_p"].add(gid, cb, wi); S["cb_pos"].add(pg, cb, wi); S["cb_lg"].add("lg", cb, wi)
                S["yac_p"].add(gid, yh, wi); S["yac_pos"].add(pg, yh, wi); S["yac_lg"].add("lg", yh, wi)
                tv = T.tgt_ev[k3]
                S["tev_p"].add(gid, tv, wi); S["tev_pos"].add(pg, tv, wi); S["tev_lg"].add("lg", tv, wi)
            if k3 in T.pass_ and T.pass_[k3][0] >= 1:
                pt = T.pass_[k3]
                S["pass_p"].add(gid, pt, wi); S["pass_pos"].add("QB", pt, wi); S["pass_lg"].add("lg", pt, wi)
                ah = T.pass_air[k3] @ M_A; cb = T.pass_cb[k3].reshape(-1); yh = np.concatenate([T.pass_yac[k3][b] @ M_Y for b in range(4)])
                S["air_p"].add(("P", gid), ah, wi); S["air_pos"].add("PQB", ah, wi)
                S["cb_p"].add(("P", gid), cb, wi); S["cb_pos"].add("PQB", cb, wi)
                S["yac_p"].add(("P", gid), yh, wi); S["yac_pos"].add("PQB", yh, wi)
        for typ, k in by_week.get((s, w), []):
            if typ == "rush":
                v = T.rush_team[k]; side = "O" if len(k) == 3 else "D"; team = k[-1]
                S["t_rush"].add(tkey(side, team), v, wi)
                if side == "O":
                    S["t_rush"].add("lg", v, wi)
            elif typ == "recv":
                v = T.recv_team[k]; side = "O" if len(k) == 3 else "D"; team = k[-1]
                S["t_recv"].add(tkey(side, team), v, wi)
                if side == "O":
                    S["t_recv"].add("lg", v, wi)
            else:
                v = T.pass_team[k]; side = "O" if len(k) == 3 else "D"; team = k[-1]
                S["t_pass"].add(tkey(side, team), v, wi)
                if side == "O":
                    S["t_pass"].add("lg", v, wi)
        if progress:
            progress(s, w)
    # league store for air/cb/yac is fed by receiving records only; passers share the same league pmf on purpose
    return {k: Records(v) for k, v in rec.items() if v}

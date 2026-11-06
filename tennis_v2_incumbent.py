"""Faithful replicas of the INCUMBENT tennis champion-gate model logic (research only; the production scripts are not modified).

Each replica is a literal port of the corresponding tennis_*_champion_gate_a.py feature/prediction code that additionally records per-match fields so the incumbent can be
graded with V2 metrics. `order` selects AS_IMPLEMENTED (match_date, text match_id) or CANONICAL (tourney, round, numeric match_num) processing order."""
from collections import Counter, defaultdict
from math import comb

import numpy as np

import tennis_v2_data as TD

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES_ELO = 5
MIN_PRIOR_MATCHES_GAMES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
INITIAL_ELO = 1500.0
DEV_END = "2019-12-31"
VAL = ("2020-01-01", "2022-12-31")
HOLD = ("2023-01-01", "2024-12-31")


def order_rows(rows, order):
    key = TD.as_implemented_key if order == "as_implemented" else TD.canonical_key
    return sorted(rows, key=key)


def k_factor(n):
    return 250.0 / ((n + 5) ** 0.4)


def elo_expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def recency_weighted_mean(values, decay=RECENCY_DECAY):
    if not values:
        return None, 0
    num = den = 0.0
    w = 1.0
    for v in reversed(values):
        num += w * v
        den += w
        w *= decay
    return (num / den if den > 0 else None), len(values)


def match_win_prob_from_set_prob(q, races_to):
    return sum(comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** (k - races_to)) for k in range(races_to, 2 * races_to))


def invert_to_set_prob(p, races_to):
    if p == 0.5:
        return 0.5
    if p < 0.5:
        return 1 - invert_to_set_prob(1 - p, races_to)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if match_win_prob_from_set_prob(mid, races_to) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def outcome_distribution(q, races_to):
    dist = {}
    for k in range(races_to, 2 * races_to):
        ls = k - races_to
        dist[("P1", races_to, ls)] = comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** ls)
        dist[("P2", races_to, ls)] = comb(k - 1, races_to - 1) * ((1 - q) ** races_to) * (q ** ls)
    return dist


def expected_sets(q, races_to):
    d = outcome_distribution(q, races_to)
    return sum((races_to + k[2]) * v for k, v in d.items())


class Elo:
    """Gate Elo state: overall + surface Elo, gate blend by min surface count (moneyline / set-betting gate form)."""

    def __init__(self):
        self.oe, self.on, self.se, self.sn = {}, {}, {}, {}

    def predict(self, p1, p2, surface):
        e1o, e2o = self.oe.get(p1, INITIAL_ELO), self.oe.get(p2, INITIAL_ELO)
        n1o, n2o = self.on.get(p1, 0), self.on.get(p2, 0)
        e1s, e2s = self.se.get((p1, surface), INITIAL_ELO), self.se.get((p2, surface), INITIAL_ELO)
        n1s, n2s = self.sn.get((p1, surface), 0), self.sn.get((p2, surface), 0)
        mins = min(n1s, n2s)
        if mins >= MIN_SURFACE_MATCHES_FOR_BLEND:
            bw = min(mins / (mins + 15), 0.6)
            c1, c2 = bw * e1s + (1 - bw) * e1o, bw * e2s + (1 - bw) * e2o
        else:
            c1, c2 = e1o, e2o
        return {"blend": elo_expected(c1, c2), "overall": elo_expected(e1o, e2o), "surface": elo_expected(e1s, e2s), "n1o": n1o, "n2o": n2o, "n1s": n1s, "n2s": n2s}

    def update(self, p1, p2, surface, p1_is_winner):
        e1o, e2o = self.oe.get(p1, INITIAL_ELO), self.oe.get(p2, INITIAL_ELO)
        n1o, n2o = self.on.get(p1, 0), self.on.get(p2, 0)
        e1s, e2s = self.se.get((p1, surface), INITIAL_ELO), self.se.get((p2, surface), INITIAL_ELO)
        n1s, n2s = self.sn.get((p1, surface), 0), self.sn.get((p2, surface), 0)
        k1, k2 = k_factor(n1o), k_factor(n2o)
        self.oe[p1] = e1o + k1 * (p1_is_winner - elo_expected(e1o, e2o))
        self.oe[p2] = e2o + k2 * ((1 - p1_is_winner) - elo_expected(e2o, e1o))
        self.on[p1], self.on[p2] = n1o + 1, n2o + 1
        ks1, ks2 = k_factor(n1s), k_factor(n2s)
        self.se[(p1, surface)] = e1s + ks1 * (p1_is_winner - elo_expected(e1s, e2s))
        self.se[(p2, surface)] = e2s + ks2 * ((1 - p1_is_winner) - elo_expected(e2s, e1s))
        self.sn[(p1, surface)], self.sn[(p2, surface)] = n1s + 1, n2s + 1


def moneyline(rows, order):
    """rows: all matches (incl. incomplete) with known result+surface. Returns observations (eligible: both >=5 prior)."""
    rows = [r for r in rows if r["winner_id"] and r["loser_id"] and r["surface"]]
    rows = order_rows(rows, order)
    leaks = TD.leak_flags(rows)
    elo = Elo(); obs = []
    for r in rows:
        p1, p2 = sorted([r["winner_id"], r["loser_id"]])
        p1w = 1 if p1 == r["winner_id"] else 0
        pr = elo.predict(p1, p2, r["surface"])
        if pr["n1o"] >= MIN_PRIOR_MATCHES_ELO and pr["n2o"] >= MIN_PRIOR_MATCHES_ELO:
            wr, lr = r["winner_rank"], r["loser_rank"]
            p1_rank = p2_rank = None
            if wr is not None and lr is not None:
                p1_rank = wr if p1 == r["winner_id"] else lr
                p2_rank = lr if p1 == r["winner_id"] else wr
            obs.append({"match_id": r["match_id"], "match_date": r["match_date"], "p1": p1, "p2": p2, "p1_prob": pr["blend"], "p1_overall": pr["overall"], "p1_surface": pr["surface"],
                        "p1_wins": p1w, "p1_rank": p1_rank, "p2_rank": p2_rank, "is_incomplete": r["is_incomplete"], "surface": r["surface"], "round": r["round"], "best_of": r["best_of"],
                        "order_leak": leaks[r["match_id"]], "p1_pts": None, "tourney_id": r["tourney_id"]})
        elo.update(p1, p2, r["surface"], p1w)
    return obs


def total_games(rows, order):
    """Literal port of tennis_total_games_champion_gate_a: complete matches with known surface; predicted_mean and the gate's 'naive' mean."""
    rows = [r for r in rows if not r["is_incomplete"] and r["surface"]]
    rows = order_rows(rows, order)
    leaks = TD.leak_flags(rows)
    hist = defaultdict(list); obs = []
    for r in rows:
        sets = TD.parse_sets(r["score"])
        if not sets:
            continue
        total = sum(a + b for a, b in sets); n_sets = len(sets); gps = total / n_sets
        surface, best_of = r["surface"], r["best_of"]
        feats = {}; eligible = True
        for pid in (r["winner_id"], r["loser_id"]):
            h = hist[pid]
            if len(h) < MIN_PRIOR_MATCHES_GAMES:
                eligible = False; continue
            og = [g for g, _, _, _ in h]; os_ = [s for _, s, _, _ in h]
            sg = [g for g, _, sf, _ in h if sf == surface]; bo = [s for _, s, _, b in h if b == best_of]
            own_gps, _ = recency_weighted_mean(og); own_sets, _ = recency_weighted_mean(os_)
            sgm, sn = recency_weighted_mean(sg); bom, bn = recency_weighted_mean(bo)
            if sn >= MIN_SURFACE_MATCHES_FOR_BLEND and sgm is not None:
                bw = min(sn / (sn + 10), 0.7); comb_g = bw * sgm + (1 - bw) * own_gps
            else:
                comb_g = own_gps
            exp_sets = bom if bn >= MIN_SURFACE_MATCHES_FOR_BLEND else own_sets
            feats[pid] = (own_gps, own_sets, comb_g, exp_sets)
        if eligible:
            fw, fl = feats[r["winner_id"]], feats[r["loser_id"]]
            pm = ((fw[2] * fw[3]) + (fl[2] * fl[3])) / 2.0
            nm = ((fw[0] * fw[1]) + (fl[0] * fl[1])) / 2.0
            obs.append({"match_id": r["match_id"], "match_date": r["match_date"], "predicted_mean": pm, "naive_mean": nm, "actual": total, "n_sets": n_sets, "best_of": best_of, "surface": surface,
                        "round": r["round"], "tb_sets": TD.tiebreak_sets(r["score"]), "order_leak": leaks[r["match_id"]], "winner_id": r["winner_id"], "loser_id": r["loser_id"], "tourney_id": r["tourney_id"]})
        for pid in (r["winner_id"], r["loser_id"]):
            hist[pid].append((gps, n_sets, surface, best_of))
    return obs


def games_spread(rows, order):
    rows = [r for r in rows if not r["is_incomplete"] and r["surface"]]
    rows = order_rows(rows, order)
    hist = defaultdict(list); obs = []
    for r in rows:
        sets = TD.parse_sets(r["score"])
        if not sets:
            continue
        n_sets = len(sets); wg = sum(w for w, _ in sets); lg = sum(l for _, l in sets)
        p1, p2 = sorted([r["winner_id"], r["loser_id"]]); p1w = p1 == r["winner_id"]
        p1_games = wg if p1w else lg; p2_games = lg if p1w else wg
        actual = p1_games - p2_games
        surface, best_of = r["surface"], r["best_of"]
        feats = {}; eligible = True
        for pid in (p1, p2):
            h = hist[pid]
            if len(h) < MIN_PRIOR_MATCHES_GAMES:
                eligible = False; continue
            om = [m for m, _, _, _ in h]; os_ = [s for _, s, _, _ in h]
            sm = [m for m, _, sf, _ in h if sf == surface]; bo = [s for _, s, _, b in h if b == best_of]
            own_m, _ = recency_weighted_mean(om); own_s, _ = recency_weighted_mean(os_)
            smm, sn = recency_weighted_mean(sm); bom, bn = recency_weighted_mean(bo)
            if sn >= MIN_SURFACE_MATCHES_FOR_BLEND and smm is not None:
                bw = min(sn / (sn + 10), 0.7); cm = bw * smm + (1 - bw) * own_m
            else:
                cm = own_m
            feats[pid] = (own_m, own_s, cm, bom if bn >= MIN_SURFACE_MATCHES_FOR_BLEND else own_s)
        if eligible:
            f1, f2 = feats[p1], feats[p2]
            pred = (f1[2] - f2[2]) * ((f1[3] + f2[3]) / 2.0)
            naive = (f1[0] - f2[0]) * ((f1[1] + f2[1]) / 2.0)
            obs.append({"match_id": r["match_id"], "match_date": r["match_date"], "predicted": pred, "naive": naive, "actual": actual, "best_of": best_of, "surface": surface, "p1": p1, "p2": p2})
        for pid, win_side in ((p1, p1w), (p2, not p1w)):
            sg = wg if win_side else lg; og = lg if win_side else wg
            hist[pid].append(((sg - og) / n_sets, n_sets, surface, best_of))
    return obs


def set_score(rows, order, dev_out=None):
    """Literal port of the set-betting gate: returns evaluation observations (after DEV) with the full predicted distribution and the DEV category counts."""
    rows = [r for r in rows if not r["is_incomplete"] and r["surface"] and r["best_of"] in (3, 5)]
    rows = order_rows(rows, order)
    elo = Elo(); obs = []; dev_counts = Counter()
    for r in rows:
        w_sets = l_sets = 0
        for a, b in TD.parse_sets(r["score"]):
            if a > b:
                w_sets += 1
            elif b > a:
                l_sets += 1
        if w_sets == 0 and l_sets == 0:
            continue
        best_of = r["best_of"]; races_to = best_of // 2 + 1
        p1, p2 = sorted([r["winner_id"], r["loser_id"]]); p1w = 1 if p1 == r["winner_id"] else 0
        actual = ("P1" if p1w else "P2", w_sets, l_sets)
        pr = elo.predict(p1, p2, r["surface"])
        if r["match_date"] <= DEV_END:
            dev_counts[(best_of, actual)] += 1
            if dev_out is not None and pr["n1o"] >= MIN_PRIOR_MATCHES_ELO and pr["n2o"] >= MIN_PRIOR_MATCHES_ELO:
                dev_out.append({"match_id": r["match_id"], "best_of": best_of, "p1_prob": pr["blend"], "actual": actual})
        elif pr["n1o"] >= MIN_PRIOR_MATCHES_ELO and pr["n2o"] >= MIN_PRIOR_MATCHES_ELO:
            q = invert_to_set_prob(pr["blend"], races_to)
            dist = outcome_distribution(q, races_to)
            obs.append({"match_id": r["match_id"], "match_date": r["match_date"], "best_of": best_of, "p1_prob": pr["blend"], "q": q, "dist": dist, "predicted": max(dist, key=dist.get), "actual": actual,
                        "p1": p1, "p2": p2, "p1_rank": r["winner_rank"] if p1 == r["winner_id"] else r["loser_rank"], "p2_rank": r["loser_rank"] if p1 == r["winner_id"] else r["winner_rank"], "surface": r["surface"]})
        elo.update(p1, p2, r["surface"], p1w)
    return obs, dev_counts


def player_serve_counts(con, order_rows_list):
    """player-match rows with serve stats in canonical order for aces / double faults (aces gate replica form)."""
    return None

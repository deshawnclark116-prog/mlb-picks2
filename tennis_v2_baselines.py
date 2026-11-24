"""Tennis V2 Phase0 baselines (research only): simple + competent-human baselines in CANONICAL order, constants estimated on DEV 2015-2019 only. No betting-market inputs."""
from collections import Counter, defaultdict

import numpy as np

import tennis_v2_data as TD
import tennis_v2_incumbent as INC

DEV_END = "2019-12-31"
K_SHRINK = 5
ELO_BUCKETS = [0.10, 0.20, 0.30]
SURF_SHRINK_K = 15
PBINS = [0.55, 0.65, 0.75]
DECILE_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def mismatch_bucket(p):
    d = abs(p - 0.5)
    return 0 if d < ELO_BUCKETS[0] else 1 if d < ELO_BUCKETS[1] else 2 if d < ELO_BUCKETS[2] else 3


def decile(p):
    return int(np.searchsorted(DECILE_EDGES, p, side="right"))


def pbucket(p):
    return int(np.searchsorted(PBINS, max(p, 1 - p), side="right"))


def e_sets(p, best_of):
    q = INC.invert_to_set_prob(max(p, 1 - p), best_of // 2 + 1)
    return INC.expected_sets(q, best_of // 2 + 1)


def rank_logit_fit(pairs):
    """MLE slope b of P(p1 wins)=sigmoid(b*log(pts1/pts2)) on DEV pairs (x, y). Newton, no intercept (lexicographic player order carries no information)."""
    x = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs], float)
    b = 0.5
    for _ in range(100):
        q = 1 / (1 + np.exp(-b * x)); g = np.sum((y - q) * x); h = np.sum(q * (1 - q) * x * x) + 1e-12
        b2 = b + g / h
        if abs(b2 - b) < 1e-10:
            b = b2; break
        b = b2
    return float(b)


# ---------------------------------------------------------------- moneyline
def moneyline_canonical(rows):
    """canonical-order observations extended with rank-points logit and shrunk-surface Elo; every match with a decided winner and known surface (RAW), eligibility = both >=5 prior."""
    rows = [r for r in rows if r["winner_id"] and r["loser_id"] and r["surface"]]
    rows = INC.order_rows(rows, "canonical")
    leaks = TD.leak_flags(rows)
    elo = INC.Elo(); out = []
    for r in rows:
        p1, p2 = sorted([r["winner_id"], r["loser_id"]]); p1w = 1 if p1 == r["winner_id"] else 0
        pr = elo.predict(p1, p2, r["surface"])
        if pr["n1o"] >= 5 and pr["n2o"] >= 5:
            def sh(pid, n):
                es = elo.se.get((pid, r["surface"]), INC.INITIAL_ELO); eo = elo.oe.get(pid, INC.INITIAL_ELO)
                return (n * es + SURF_SHRINK_K * eo) / (n + SURF_SHRINK_K)
            ps = INC.elo_expected(sh(p1, pr["n1s"]), sh(p2, pr["n2s"]))
            wr, lr = r["winner_rank"], r["loser_rank"]
            wp, lp = r["winner_rank_points"], r["loser_rank_points"]
            x = None
            if wp and lp and wp > 0 and lp > 0:
                a, b = (wp, lp) if p1 == r["winner_id"] else (lp, wp)
                x = float(np.log(a / b))
            out.append({"match_id": r["match_id"], "match_date": r["match_date"], "p1": p1, "p2": p2, "p1_wins": p1w, "p_blend": pr["blend"], "p_overall": pr["overall"], "p_surface_shrunk": ps,
                        "rank_x": x, "is_incomplete": r["is_incomplete"], "surface": r["surface"], "round": r["round"], "best_of": r["best_of"], "order_leak": leaks[r["match_id"]],
                        "n1o": pr["n1o"], "n2o": pr["n2o"], "tourney_id": r["tourney_id"], "score": r["score"], "winner_id": r["winner_id"],
                        "p1_rank": (wr if p1 == r["winner_id"] else lr), "p2_rank": (lr if p1 == r["winner_id"] else wr)})
        elo.update(p1, p2, r["surface"], p1w)
    return out


def moneyline_models(obs):
    """returns (slope, dict name -> list p1_prob or None per obs)."""
    dev = [(o["rank_x"], o["p1_wins"]) for o in obs if o["match_date"] <= DEV_END and o["rank_x"] is not None]
    b = rank_logit_fit(dev)
    models = defaultdict(list)
    for o in obs:
        pr = None if o["rank_x"] is None else float(1 / (1 + np.exp(-b * o["rank_x"])))
        o["p_rank"] = pr
        o["p_human"] = None if pr is None else 0.5 * o["p_blend"] + 0.5 * pr
    return b


# -------------------------------------------------------------- total games
def total_games_baselines(rows):
    """canonical order, incumbent eligibility, identical rows. Returns observations with every baseline prediction."""
    inc = INC.total_games(rows, "canonical")
    mon = {o["match_id"]: o for o in moneyline_canonical(rows)}
    # per-player completed-match histories in canonical order (total games, games per set)
    rows_c = [r for r in INC.order_rows([r for r in rows if not r["is_incomplete"] and r["surface"]], "canonical")]
    hist = defaultdict(list)
    per = {}
    for r in rows_c:
        sets = TD.parse_sets(r["score"])
        if not sets:
            continue
        tot = sum(a + b for a, b in sets); n = len(sets)
        per[r["match_id"]] = ([h[0] for h in hist[r["winner_id"]]], [h[1] for h in hist[r["winner_id"]]], [h[0] for h in hist[r["loser_id"]]], [h[1] for h in hist[r["loser_id"]]])
        for pid in (r["winner_id"], r["loser_id"]):
            hist[pid].append((tot, tot / n))
    byid = {o["match_id"]: o for o in inc}
    # DEV constants
    dev = [o for o in inc if o["match_date"] <= DEV_END and o["match_id"] in mon]
    surf_bo = defaultdict(list); bo_only = defaultdict(list); elo_b = defaultdict(list); gps_bucket = defaultdict(list)
    for o in dev:
        m = mon[o["match_id"]]
        surf_bo[(o["surface"], o["best_of"])].append(o["actual"]); bo_only[o["best_of"]].append(o["actual"])
        elo_b[(o["best_of"], mismatch_bucket(m["p_blend"]))].append(o["actual"])
        gps_bucket[(o["surface"], mismatch_bucket(m["p_blend"]))].append(o["actual"] / o["n_sets"])
    mean = lambda v: float(np.mean(v)) if v else None
    surf_fit = {k: mean(v) for k, v in surf_bo.items()}; bo_fit = {k: mean(v) for k, v in bo_only.items()}
    elo_fit = {k: mean(v) for k, v in elo_b.items()}; gps_fit = {k: mean(v) for k, v in gps_bucket.items()}
    out = []
    for o in inc:
        m = mon.get(o["match_id"])
        if m is None:
            continue
        wt, wg, lt, lg = per[o["match_id"]]
        # p1 / p2 lexicographic by id
        p1_is_w = o["winner_id"] < o["loser_id"]
        t1, t2 = (wt, lt) if p1_is_w else (lt, wt)
        g1, g2 = (wg, lg) if p1_is_w else (lg, wg)
        mk = lambda h, k: float(np.mean(h[-k:]))
        ew = lambda h: INC.recency_weighted_mean(h)[0]
        pred = {}
        for k in (3, 5, 10):
            pred["prior%d" % k] = (mk(t1, k) + mk(t2, k)) / 2
        pred["ewma_0.6"] = (ew(t1) + ew(t2)) / 2
        pred["p1_prior10"] = mk(t1, 10); pred["p2_prior10"] = mk(t2, 10)
        pred["surface_fit"] = surf_fit.get((o["surface"], o["best_of"]), bo_fit.get(o["best_of"]))
        pred["bestof_fit"] = bo_fit.get(o["best_of"])
        pred["elo_mismatch_fit"] = elo_fit.get((o["best_of"], mismatch_bucket(m["p_blend"])), bo_fit.get(o["best_of"]))
        bucket_gps = gps_fit.get((o["surface"], mismatch_bucket(m["p_blend"])))
        es = e_sets(m["p_blend"], o["best_of"])
        if bucket_gps is not None:
            sh = lambda g: (min(len(g), 10) * float(np.mean(g[-10:])) + K_SHRINK * bucket_gps) / (min(len(g), 10) + K_SHRINK)
            gps_h = 0.6 * bucket_gps + 0.4 * (sh(g1) + sh(g2)) / 2
            pred["human"] = es * gps_h
        else:
            pred["human"] = None
        oo = dict(o); oo["pred"] = pred; oo["p_blend"] = m["p_blend"]; oo["e_sets"] = es
        oo["p1_is_winner"] = p1_is_w
        out.append(oo)
    return out, {"surface_fit": {"%s|%s" % k: v for k, v in surf_fit.items()}, "bestof_fit": {str(k): v for k, v in bo_fit.items()}, "elo_fit": {"%s|%s" % k: v for k, v in elo_fit.items()},
                 "gps_bucket": {"%s|%s" % k: v for k, v in gps_fit.items()}, "n_dev": len(dev)}


# ---------------------------------------------------------------- set score
def set_score_baselines(rows, order="canonical"):
    dev_out = []
    obs, dev_counts = INC.set_score(rows, order, dev_out)
    # winner side shares from DEV: conditional on best_of the (ws, ls) split frequency
    split = defaultdict(Counter); psplit = defaultdict(Counter); cat = defaultdict(Counter)
    for (bo, (side, ws, ls)), n in dev_counts.items():
        split[bo][(ws, ls)] += n; cat[bo][(side, ws, ls)] += n
    mon_dev = {}
    # human: bucket of the winner prob (blend) by best_of from DEV
    for d in dev_out:
        a = d["actual"]; psplit[(d["best_of"], pbucket(d["p1_prob"]))][(a[1], a[2])] += 1
    out = []
    for o in obs:
        bo = o["best_of"]; a = o["actual"]; races = bo // 2 + 1
        tot = sum(cat[bo].values())
        d_freq = {c: cat[bo].get(c, 0) / tot for c in o["dist"]}
        p1 = o["p1_prob"]
        sp = split[bo]; st = sum(sp.values())
        d_wp = {}
        for (side, w, l) in o["dist"]:
            pside = p1 if side == "P1" else 1 - p1
            d_wp[(side, w, l)] = pside * sp.get((w, l), 0) / st
        ps = psplit[(bo, pbucket(p1))]; pt = sum(ps.values())
        d_h = {}
        for (side, w, l) in o["dist"]:
            pside = p1 if side == "P1" else 1 - p1
            d_h[(side, w, l)] = pside * (ps.get((w, l), 0) / pt if pt else sp.get((w, l), 0) / st)
        oo = dict(o); oo["d_freq"] = d_freq; oo["d_wp"] = d_wp; oo["d_human"] = d_h
        out.append(oo)
    return out


# ------------------------------------------------------------------- spread
def spread_baselines(rows):
    inc = INC.games_spread(rows, "canonical")
    mon = {o["match_id"]: o for o in moneyline_canonical(rows)}
    dev = defaultdict(list)
    for o in inc:
        m = mon.get(o["match_id"])
        if m is not None and o["match_date"] <= DEV_END:
            dev[(o["best_of"], decile(m["p_blend"]))].append(o["actual"])
    fit = {k: float(np.mean(v)) for k, v in dev.items()}
    out = []
    for o in inc:
        m = mon.get(o["match_id"])
        if m is None:
            continue
        oo = dict(o); oo["zero"] = 0.0
        oo["elo_bucket"] = fit.get((o["best_of"], decile(m["p_blend"])), 0.0)
        oo["p_blend"] = m["p_blend"]
        out.append(oo)
    return out


# --------------------------------------------------------------- aces / df
def serve_count_obs(rows_pm, stat):
    """rows_pm: player_match dict rows with serve data in canonical MATCH order. Literal gate feature logic for aces/DF; both rows of a match are predicted BEFORE either updates the state."""
    hist = defaultdict(list); conceded = defaultdict(list)
    st = {"stat": 0, "sv": 0}
    obs = []
    groups = []
    for r in rows_pm:
        if groups and groups[-1][0]["match_id"] == r["match_id"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    cap = 0.25 if stat == "double_faults" else 0.5
    for g in groups:
        for r in g:
            pid, opp = r["player_id"], r["opponent_id"]
            h = hist[pid]
            if len(h) < 10:
                continue
            own, _ = rw_rate([(a, s) for a, s, _, _ in h])
            sfr, sfn = rw_rate([(a, s) for a, s, sf, _ in h if sf == r["surface"]])
            if sfn >= 4 and sfr is not None:
                bw = min(sfn / (sfn + 10), 0.7); comb = bw * sfr + (1 - bw) * own
            else:
                comb = own
            orate, on = rw_rate(list(conceded.get(opp, [])))
            if on >= 4 and orate is not None and st["sv"] > 0 and st["stat"] > 0:
                adj = min(max(orate / (st["stat"] / st["sv"]), 0.6), 1.6)
            else:
                adj = 1.0
            bo_pairs = [(a, s) for a, s, _, bo in h if bo == r["best_of"]]
            use = bo_pairs if len(bo_pairs) >= 4 else [(a, s) for a, s, _, _ in h]
            exp_sv = sum(s for _, s in use[-15:]) / len(use[-15:])
            rate = min(max(comb * adj, 0.005), cap)
            obs.append({"match_id": r["match_id"], "match_date": r["match_date"], "player_id": pid, "surface": r["surface"], "best_of": r["best_of"],
                        "pred": rate * exp_sv, "pred_rate": rate, "exp_sv": exp_sv, "naive": min(max(own, 0.005), cap) * exp_sv,
                        "prior10_mean": float(np.mean([a for a, _, _, _ in h[-10:]])), "actual": r[stat], "actual_sv": r["serve_points"], "round": r["round"]})
        for r in g:
            v, sv = r[stat], r["serve_points"]
            hist[r["player_id"]].append((v, sv, r["surface"], r["best_of"])); conceded[r["opponent_id"]].append((v, sv))
            st["stat"] += v; st["sv"] += sv
    return obs


def rw_rate(pairs, decay=0.6):
    if not pairs:
        return None, 0
    num = den = 0.0; w = 1.0
    for a, s in reversed(pairs):
        num += w * a; den += w * s; w *= decay
    return (num / den if den > 0 else None), len(pairs)

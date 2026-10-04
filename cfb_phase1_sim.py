"""
CFB_PHASE1_SIM -- ONE coherent simulated offensive team-game (CFB Outcome Engine v1, RESEARCH / SHADOW ONLY). Monte Carlo is the FINAL propagation layer for validated components; it rescues nothing.
Causal order per draw (vectorised over M draws):
  availability (activity flags) -> team plays N ~ NB2 -> dropbacks D | N ~ BetaBinomial -> sacks S | D, attempts A = D - S, rush plays R = N - D
  -> gadget carries G | R ~ BetaBinomial, primary pool R - G -> carry allocation (Dirichlet-multinomial: primary rushers + OTHER; gadget candidates + OTHER)
  -> QB attempt allocation of A (Dirichlet-multinomial: QBs + OTHER) -> completions per QB ~ BetaBinomial(attempts) -> reception allocation of the team completions (receivers + OTHER)
  -> yards (exact iid per-event draws from hierarchical pmfs) -> TD / INT hazards -> team points = 7 * offensive TDs + residual
Accounting identities hold draw by draw: R + A + S = N ; sum(carries) + OTHER = R ; sum(QB attempts) + OTHER = A ; sum(receptions) + OTHER = team completions ; completions <= attempts ; receptions <= completions ;
passing yards (all QBs) = receiving yards (all receivers + OTHER) ; passing TDs = receiving TDs ; points = 7 * (rush TD + receiving TD) + residual.
"""
import numpy as np

import cfb_phase1_efficiency as EF

DEFAULT_N = 4000


def nb_draw(rng, mu, alpha, M):
    r = 1.0 / alpha
    return rng.negative_binomial(r, r / (r + max(mu, 1e-6)), size=M)


def bb_draw(rng, n, s, kappa):
    s = float(np.clip(s, 1e-5, 1 - 1e-5))
    p = rng.beta(s * kappa, (1 - s) * kappa, size=len(n))
    return rng.binomial(n, p)


def dm_alloc(rng, total, shapes, active=None):
    """Dirichlet-multinomial allocation of `total` [M] among J buckets with Dirichlet shapes [J] (last bucket = OTHER, always active). active [M, J-1] bool zeroes a bucket's weight. Returns counts [M, J] summing to total."""
    M, J = len(total), len(shapes)
    w = rng.gamma(np.maximum(np.asarray(shapes, float), 1e-9)[None, :], size=(M, J))
    if active is not None:
        w[:, :J - 1] *= active
    rem = total.astype(np.int64).copy(); wrem = w.sum(axis=1); out = np.zeros((M, J), np.int64)
    for j in range(J - 1):
        p = np.where(wrem > 0, w[:, j] / np.maximum(wrem, 1e-300), 0.0)
        x = rng.binomial(rem, np.clip(p, 0, 1)); out[:, j] = x; rem -= x; wrem -= w[:, j]
    out[:, J - 1] = rem
    return out


def sum_draws(rng, counts, pmf):
    """Per-draw sum of `counts[m]` iid yard draws from pmf (indexed y - YMIN). Returns yards [M]."""
    M = len(counts); T = int(counts.sum())
    if T == 0:
        return np.zeros(M, np.int64)
    cdf = np.cumsum(pmf); cdf[-1] = 1.0
    ys = np.searchsorted(cdf, rng.random(T)) + EF.YMIN
    return np.bincount(np.repeat(np.arange(M), counts), weights=ys, minlength=M).astype(np.int64)


def simulate_team(spec, M=DEFAULT_N, seed=0):
    """spec keys (all produced by cfb_phase1_forecast.CoreEngine.team_spec):
    vol{mu,alpha,pi,kappa,sack}; carry{rho,kg,primary[{kappa,s,a}],primary_other{kappa,s},gadget[{alpha}],gadget_other}; qb[{s,kappa,p,kc}], qb_other{s,kappa,p}; recv[{s,kappa,a}], recv_other{s,kappa};
    pmf{rush[cand], rec[cand], rush_other, rec_other}; td{rush[cand]{p}, rec[cand]{p}, kappa_rush, kappa_rec, rush_other_p, rec_other_p, int[qb]{p}, kappa_int}; resid{mu,alpha,shift}. Candidates are referenced by index."""
    rng = np.random.default_rng(seed)
    v = spec["vol"]
    N = nb_draw(rng, v["mu"], v["alpha"], M)
    pi = rng.beta(v["pi"] * v["kappa"], (1 - v["pi"]) * v["kappa"], size=M)
    D = rng.binomial(N, pi); S = rng.binomial(D, v["sack"]); A = D - S; R = N - D
    out = {"N": N, "R": R, "D": D, "S": S, "A": A, "players": {}}
    # ---- carries
    cs = spec["carry"]
    G = bb_draw(rng, R, cs["rho"], cs["kg"]); P = R - G
    prim = cs["primary"]
    shapes = [x["kappa"] * x["s"] for x in prim] + [cs["primary_other"]["kappa"] * cs["primary_other"]["s"]]
    act = np.stack([(rng.random(M) < x["a"]) if x.get("a") is not None else np.ones(M, bool) for x in prim], axis=1) if prim else None
    car_p = dm_alloc(rng, P, shapes, act) if prim else np.zeros((M, 1), np.int64)
    gad = cs["gadget"]
    car_g = dm_alloc(rng, G, [x["alpha"] for x in gad] + [cs["gadget_other"]])
    carries = {}
    for i, x in enumerate(prim):
        carries[x["idx"]] = car_p[:, i]
    for i, x in enumerate(gad):
        carries[x["idx"]] = car_g[:, i]
    other_car = car_p[:, -1] + car_g[:, -1]
    # ---- QB attempts / completions / interceptions
    qbs = spec["qb"]
    qshapes = [x["kappa"] * x["s"] for x in qbs] + [spec["qb_other"]["kappa"] * spec["qb_other"]["s"]]
    att = dm_alloc(rng, A, qshapes)
    comp_q = []
    for i, x in enumerate(qbs):
        comp_q.append(bb_draw(rng, att[:, i], x["p"], x["kc"]))
    comp_other = bb_draw(rng, att[:, -1], spec["qb_other"]["p"], spec["qb_other"]["kc"])
    Ct = sum(comp_q) + comp_other if qbs else comp_other
    # ---- receptions of the team completions
    rv = spec["recv"]
    rshapes = [x["kappa"] * x["s"] for x in rv] + [spec["recv_other"]["kappa"] * spec["recv_other"]["s"]]
    ract = np.stack([(rng.random(M) < x["a"]) if x.get("a") is not None else np.ones(M, bool) for x in rv], axis=1) if rv else None
    rec = dm_alloc(rng, Ct, rshapes, ract) if rv else np.concatenate([np.zeros((M, 0), np.int64), Ct[:, None]], axis=1)
    receptions = {x["idx"]: rec[:, i] for i, x in enumerate(rv)}
    other_rec = rec[:, -1]
    # ---- yards
    pm = spec["pmf"]; td = spec["td"]
    rush_yds, rec_yds, rush_td, rec_td = {}, {}, {}, {}
    for idx, c in carries.items():
        rush_yds[idx] = sum_draws(rng, c, pm["rush"][idx]); rush_td[idx] = bb_draw(rng, c, td["rush"][idx], td["kappa_rush"])
    other_rush_yds = sum_draws(rng, other_car, pm["rush_other"]); other_rush_td = rng.binomial(other_car, td["rush_other_p"])
    for idx, c in receptions.items():
        rec_yds[idx] = sum_draws(rng, c, pm["rec"][idx]); rec_td[idx] = bb_draw(rng, c, td["rec"][idx], td["kappa_rec"])
    other_rec_yds = sum_draws(rng, other_rec, pm["rec_other"]); other_rec_td = rng.binomial(other_rec, td["rec_other_p"])
    team_rec_yds = sum(rec_yds.values()) + other_rec_yds if rec_yds else other_rec_yds
    team_rec_td = sum(rec_td.values()) + other_rec_td if rec_td else other_rec_td
    # ---- QB passing yards / TDs by completions share (exact conservation); interceptions per attempt
    pass_yds, pass_td, ints = {}, {}, {}
    if qbs:
        tot_c = np.maximum(Ct, 1)
        ys_alloc = np.zeros((M, len(qbs) + 1), np.int64); td_alloc = dm_alloc_det(rng, team_rec_td, comp_q + [comp_other])
        share = np.stack(comp_q + [comp_other], axis=1) / tot_c[:, None]
        yq = np.floor(team_rec_yds[:, None] * share).astype(np.int64)
        yq[np.arange(M), np.argmax(share, axis=1)] += team_rec_yds - yq.sum(axis=1)
        out["other_pass_yards"] = yq[:, -1]; out["other_pass_td"] = td_alloc[:, -1]
        for i, x in enumerate(qbs):
            pass_yds[x["idx"]] = yq[:, i]; pass_td[x["idx"]] = td_alloc[:, i]; ints[x["idx"]] = bb_draw(rng, att[:, i], x["int_p"], td["kappa_int"])
    qb_att = {x["idx"]: att[:, i] for i, x in enumerate(qbs)}; qb_comp = {x["idx"]: comp_q[i] for i, x in enumerate(qbs)}
    # ---- team points
    off_td = sum(rush_td.values()) + other_rush_td + team_rec_td if rush_td else other_rush_td + team_rec_td
    r = spec["resid"]
    resid = nb_draw(rng, r["mu"], r["alpha"], M) - r["shift"]
    out.update({"team_rec_td": team_rec_td, "points": 7 * off_td + resid, "off_td": off_td, "team_rec_yards": team_rec_yds, "team_completions": Ct, "G": G, "other_carries": other_car, "other_receptions": other_rec, "other_att": att[:, -1]})
    out["players"] = {"carries": carries, "rush_yards": rush_yds, "rush_td": rush_td, "receptions": receptions, "rec_yards": rec_yds, "rec_td": rec_td, "pass_att": qb_att, "completions": qb_comp, "pass_yards": pass_yds, "pass_td": pass_td, "int": ints}
    if spec.get("points_q"):
        out = reweight_to_points(out, spec["points_q"], rng)
    return out


def _take(o, idx):
    if isinstance(o, dict):
        return {k: _take(v, idx) for k, v in o.items()}
    if isinstance(o, np.ndarray) and o.ndim >= 1 and len(o) == len(idx):
        return o[idx]
    return o


def reweight_to_points(o, q, rng):
    """Calibration method (registered): importance RESAMPLING of the joint draws so the team-points marginal equals the validated direct points component (NB2, C1 state GLM): w_m = q(points_m) / p_hat(points_m) with p_hat the
    smoothed emergent histogram; every player / team array is resampled with the same indices, so all accounting identities (which hold draw by draw) keep holding and player outcomes stay conditioned on the team score."""
    from scipy import stats
    pts = o["points"]; M = len(pts)
    lo = int(pts.min()); hi = int(pts.max())
    hist = np.bincount(pts - lo, minlength=hi - lo + 1).astype(float) / M
    ker = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    phat = np.convolve(hist, ker, mode="same") + 1e-4
    r = 1.0 / q["alpha"]
    qpm = stats.nbinom.pmf(np.clip(pts, 0, None), r, r / (r + q["mu"]))
    w = qpm / phat[pts - lo]
    w = np.where(pts < 0, 0.0, w)
    if w.sum() <= 0:
        return o
    p = w / w.sum()
    ess = float(1.0 / np.sum(p ** 2))
    idx = rng.choice(M, size=M, p=p)
    o2 = _take(o, idx)
    o2["ess_points_reweight"] = ess
    return o2


def dm_alloc_det(rng, total, weights_list):
    """Multinomial allocation of `total` [M] with per-draw probabilities proportional to the integer weights (completions) of each bucket."""
    W = np.stack(weights_list, axis=1).astype(float); M, J = W.shape
    rem = total.astype(np.int64).copy(); wrem = W.sum(axis=1); out = np.zeros((M, J), np.int64)
    for j in range(J - 1):
        p = np.where(wrem > 0, W[:, j] / np.maximum(wrem, 1e-300), 0.0)
        x = rng.binomial(rem, np.clip(p, 0, 1)); out[:, j] = x; rem -= x; wrem -= W[:, j]
    out[:, J - 1] = rem
    return out


def check_identities(o):
    """Draw-level accounting identities; returns {name: bool} (True = holds in every draw)."""
    P = o["players"]
    ok = {}
    ok["R+A+S=N"] = bool(np.all(o["R"] + o["A"] + o["S"] == o["N"]))
    ok["carries+other=R"] = bool(np.all(sum(P["carries"].values()) + o["other_carries"] == o["R"])) if P["carries"] else True
    ok["qb_att+other=A"] = bool(np.all(sum(P["pass_att"].values()) + o["other_att"] == o["A"])) if P["pass_att"] else True
    ok["receptions+other=team completions"] = bool(np.all(sum(P["receptions"].values()) + o["other_receptions"] == o["team_completions"])) if P["receptions"] else True
    ok["completions<=attempts"] = bool(np.all(o["team_completions"] <= o["A"]))
    ok["receptions<=completions"] = bool(np.all(sum(P["receptions"].values()) <= o["team_completions"])) if P["receptions"] else True
    if P["pass_yards"]:
        ok["passing yards (QBs + OTHER) = receiving yards"] = bool(np.all(sum(P["pass_yards"].values()) + o["other_pass_yards"] == o["team_rec_yards"]))
        ok["passing TD (QBs + OTHER) = receiving TD"] = bool(np.all(sum(P["pass_td"].values()) + o["other_pass_td"] == o["team_rec_td"]))
    ok["nonnegative_opportunities"] = bool(all((v >= 0).all() for k in ("carries", "receptions", "pass_att", "completions") for v in P[k].values()))
    ok["td<=opportunity"] = bool(all((P["rush_td"][i] <= P["carries"][i]).all() for i in P["rush_td"]) and all((P["rec_td"][i] <= P["receptions"][i]).all() for i in P["rec_td"]))
    return ok

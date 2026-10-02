"""
NFL_PHASE1C_SIM  (Phase 1C, shadow research)

ONE coherent simulated NFL game.  Every player statistic is an aggregation of simulated PLAY events; no stat category is drawn independently.

Per offensive team-game, per simulation draw d (vectorised over draws):
  availability   each candidate is active with the Phase 1A probability (T24 or T90); inactive => zero opportunities and zero stats
  volume         team rush attempts R and dropbacks D from the Phase 1A team-environment negative binomials (S0), or from the game-script layer (S1)
  rush plays     R carries allocated among active rushers + explicit outside bucket (Phase 1A Dirichlet-multinomial); each carry sampled from
                 the rusher's Phase 1B pmf; red-zone carries = binomial thinning of the rusher's carries by a shared team red-zone latent
  dropbacks      D allocated among active QBs + outside bucket; per QB: sacks ~ Bin(dropbacks, hazard); scrambles ~ Bin(QB carries, share);
                 attempts = dropbacks - sacks - scrambles (never negative; adjustment is logged)
  attempts       no-target attempts ~ Bin(attempts, rate); targets = attempts - no-target, allocated among active receivers + outside bucket;
                 red-zone targets = thinning of receiver targets; attempt slots are paired with target slots by a within-draw random permutation
  pass events    per slot: interception (QB hazard) -> else air yards (receiver pmf) -> completion (receiver catch hazard by air bucket,
                 QB log-odds adjustment, marginal completion probability preserved) -> YAC (receiver pmf); yards = air + YAC; TD attached to the
                 completed red-zone / long play; the SAME event increments QB and receiver
  defence        opposing defenders' snaps = the opponent's simulated plays; sacks / interceptions are allocated FROM the offensive events
                 (half-sacks as 0.5 + 0.5); tackle credits = tackle-ending plays x (1 + assists), allocated by per-snap rates
Official-stat conventions documented in README: sacks are not attempts; scrambles are QB carries; TD play yards are bounded by field position.
"""
import numpy as np

import nfl_phase1b_data as B
import nfl_phase1_defense_events as DE
import nfl_phase1_receiving_efficiency as RC
import nfl_phase1_rushing_efficiency as RU

K_R, K_A, K_Y = B.K_R, B.K_A, B.K_Y
RUSH_V, AIR_V, YAC_V = RU.BIN_VALS, RC.AIR_V, RC.YAC_V           # Phase 1B midpoints; the simulator uses fitted within-bin means (Const)
AIR_BK = RC.AIR_BUCKET


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _cdf(P):
    c = np.cumsum(P, -1)
    c[..., -1] = 1.0
    return c


def sample_bins(cdf_rows, owner, u):
    """cdf_rows [m,K]; owner [E] row index per event; u [E] uniforms -> bin index [E]."""
    out = np.zeros(len(owner), int)
    for j in np.unique(owner):
        m = owner == j
        out[m] = np.minimum(np.searchsorted(cdf_rows[j], u[m]), cdf_rows.shape[1] - 1)
    return out


def restricted_cdf(P, mask):
    Q = P * mask[None, :]
    s = Q.sum(1, keepdims=True)
    Q = np.where(s > 0, Q / np.maximum(s, 1e-300), mask[None, :] / max(mask.sum(), 1))
    return _cdf(Q)


def alloc_given(pact, prop, other, alpha, Tvec, rng, other_mode="fixed", force_one=False):
    """Phase 1A Dirichlet-multinomial allocation of a GIVEN team total among active candidates + one outside bucket.
    other_mode "fixed": bucket weight is an absolute constant (Phase 1A). "prop": bucket weight = other x (sum of active named weights), i.e. the bucket
    keeps a constant SHARE when a named player is out (a backup absorbs the vacated volume); if nobody named is active the bucket takes everything."""
    N, n = len(Tvec), len(prop)
    active = rng.random((N, n)) < pact[None, :]
    if force_one:
        # a team always has a quarterback on the field: when no listed QB is active, one listed QB (chosen in proportion to his propensity) plays unless the
        # snaps go to an unlisted QB (probability = the outside bucket's share of the propensity mass)
        none = ~active.any(1)
        if none.any() and prop.sum() > 0:
            pick = rng.choice(n, size=int(none.sum()), p=prop / prop.sum())
            listed = rng.random(int(none.sum())) >= other / (prop.sum() + other)
            rows = np.where(none)[0]
            active[rows[listed], pick[listed]] = True
    w = active * prop[None, :]
    wsum = w.sum(1)
    wo = other * wsum if other_mode == "prop" else np.full(N, float(other))
    tot = wsum + wo
    zero = tot <= 0
    tot = np.where(zero, 1.0, tot)
    full = np.column_stack([w, wo]) / tot[:, None]
    full[zero, :] = 0.0; full[zero, -1] = 1.0
    if alpha is not None and np.isfinite(alpha):
        g = rng.gamma(np.maximum(alpha * full, 1e-9)); g[full == 0] = 0.0
        s = g.sum(1, keepdims=True)
        full = np.where(s > 0, g / np.maximum(s, 1e-300), full)
    full = full / full.sum(1, keepdims=True)
    cnt = rng.multinomial(Tvec.astype(np.int64), full)
    return cnt[:, :n], cnt[:, n], active


def flat_groups(counts):
    """counts [N, m] -> (draw index, group index, position within group) for every unit event, draw-major."""
    N, m = counts.shape
    c = counts.ravel().astype(np.int64)
    tot = int(c.sum())
    gid = np.repeat(np.arange(N * m), c)
    starts = np.cumsum(c) - c
    pos = np.arange(tot) - np.repeat(starts, c)
    return gid // m, gid % m, pos


class Const:
    """League structural constants (constants.json, estimated on TRAIN + VALID only)."""

    def __init__(self, d):
        self.nt = d["no_target_rate_of_nonsack_attempts"]; self.f_scr = d["scramble_share_of_qb_carries"]
        self.sbar = d["league_sack_rate_per_dropback"]; self.p_half = d["half_sack_probability"]
        self.mu_c = d["tackle_credits_per_eligible_play"]; self.rz_shape = d["rz_latent_gamma_shape"]
        self.gl_r = d["gl_share_of_rz_rushes"]; self.gl_t = d["gl_share_of_rz_targets"]
        self.d_rz_r = d["rz_share_of_rushes"]; self.d_rz_t = d["rz_share_of_targets"]
        self.z_db = d["scramble_per_dropback"]
        self.qb_bucket_mode = d.get("_qb_bucket_mode", "fixed"); self.force_qb = d.get("_force_qb", False)      # evidence (dev ablation): forcing a listed QB to play worsens pass-yards CRPS; the outside bucket absorbs the snaps instead
        bv = d.get("bin_values")
        self.rush_v = np.array(bv["rush"]) if bv else RUSH_V.copy()
        self.air_v = np.array(bv["air"]) if bv else AIR_V.copy()
        self.yac_v = np.array(bv["yac_by_air_bucket"]) if bv else np.tile(YAC_V, (4, 1))
        zr = (bv or {}).get("zone_ratios")
        self.zr = {k: {z: np.array(v[z]) for z in ("rz", "out")} for k, v in zr.items()} if zr else None
        zc = (bv or {}).get("zone_completion_rate") or {"rz": 0.6, "out": 0.69, "all": 0.68}
        self.comp_rate_rz, self.comp_rate_out, self.comp_rate_all = zc["rz"], zc["out"], zc["all"]

    def zone_w(self, yards, kind, rz_flag):
        """League zone likelihood ratio at each candidate's (rounded) yardage; rz_flag is a boolean per event (rows of `yards`); ones without ratios."""
        if self.zr is None:
            return np.ones_like(yards, dtype=float)
        idx = np.clip(np.round(yards).astype(int) + 10, 0, 109)
        return np.where(np.asarray(rz_flag)[:, None], self.zr[kind]["rz"][idx], self.zr[kind]["out"][idx])


class EffView:
    """Per-player efficiency lookups for one team-game, built from fit_all() arrays and the Records index maps."""

    def __init__(self, eff, idx, s, w, ids_carry, ids_target, ids_qb, defaults):
        self.d = defaults
        self.rush_i = [idx["rush"].get((s, w, g)) for g in ids_carry]
        self.rec_i = [idx["rec"].get((s, w, g)) for g in ids_target]
        self.qb_i = [idx["pass"].get((s, w, g)) for g in ids_qb]
        self.eff = eff
        self.missing = {"rush": sum(i is None for i in self.rush_i), "rec": sum(i is None for i in self.rec_i), "qb": sum(i is None for i in self.qb_i)}


def defaults_from(eff, Rs):
    """Outside-bucket / missing-record efficiency = league-average rows over the most recent active records (as-of by construction)."""
    R = Rs["rush"]; E = Rs["rec"]; Q = Rs["pass"]
    tr = R.active & (R.s * 100 + R.w <= 202418); te = E.active & (E.s * 100 + E.w <= 202418); tq = Q.active & (Q.s * 100 + Q.w <= 202418)
    w_r = R.n_opp[tr]; w_e = E.n_opp[te]
    avg = lambda A, m, w: (A[m] * w.reshape((-1,) + (1,) * (A.ndim - 1))).sum(0) / w.sum()
    d = {"rush": avg(eff["rush"], tr, w_r), "rush_td": (avg(eff["rush_td"][0][:, None], tr, w_r)[0], avg(eff["rush_td"][1][:, None], tr, w_r)[0]),
         "air": avg(eff["air"], te, w_e), "catch": avg(eff["catch"], te, w_e), "yac": avg(eff["yac"], te, w_e),
         "rec_td": (avg(eff["rec_td"][0][:, None], te, w_e)[0], avg(eff["rec_td"][1][:, None], te, w_e)[0]),
         "qb_sack": float(eff["qb_sack"][tq].mean()), "qb_int": float(eff["qb_int"][tq].mean())}
    return d


def simulate_offense(g, eff_v, C, N, rng, horizon, script=None, qb_adjust=True, trace=False):
    """g: Phase 1A game inputs (dict types). eff_v: EffView. Returns event-derived statistics for the offense + the events the defence needs."""
    T = g["types"]; ct, tt, qt = T["carry"], T["target"], T["qb_att"]
    pk = "pact24" if horizon == "T24" else "pact90"
    E, D_ = eff_v.eff, eff_v.d
    n_c, n_t, n_q = len(ct["ids"]), len(tt["ids"]), len(qt["ids"])
    # ------------- team volume
    def nb(mu, k):
        return rng.negative_binomial(k, k / (k + max(mu, 1e-6)), size=N)
    if script is not None:
        Rn, Dn = script["R"], script["D"]
    else:
        Rn, Dn = nb(ct["mu"], ct["k"]), nb(qt["mu"], qt["k"])
    # red-zone latent shared by rushes and targets
    psi = rng.gamma(C.rz_shape, 1.0 / C.rz_shape, size=N)
    q_r = min(max(T["rz_carry"]["mu"] / max(ct["mu"], 1e-6), 1e-3), 0.6) if "rz_carry" in T else C.d_rz_r
    q_t = min(max(T["rz_target"]["mu"] / max(tt["mu"], 1e-6), 1e-3), 0.6) if "rz_target" in T else C.d_rz_t
    # ------------- rush allocation
    oth = g["_meta"]
    cnt_c, out_c, act_c = alloc_given(ct[pk], ct["P1"], oth["carry"]["other"], oth["carry"]["alpha"], Rn, rng)
    wr = _rz_weights(g, "rz_carry", ct)
    rz_c = rng.binomial(cnt_c, np.clip(q_r * psi[:, None] * wr[None, :], 0, 0.98))
    rz_c_out = rng.binomial(out_c, np.clip(q_r * psi, 0, 0.98))
    # ------------- quarterbacks / dropbacks
    qb_ids = list(qt["ids"]); carry_pos = {gid: j for j, gid in enumerate(ct["ids"])}
    # Phase 1A's QB share is attempts / dropbacks; dropbacks additionally contain sacks and scrambles, so the dropback propensity is rescaled
    # by 1/(1 - sack rate - scramble rate) and the outside bucket loses the sack + scramble mass it used to carry (never below 3%)
    prop_db = np.clip(qt["P1"] / (1 - C.sbar - C.z_db), 0, 1)
    other_db = max(oth["qb_att"]["other"] - C.sbar - C.z_db, 0.03)
    d_q, d_out, act_q = alloc_given(qt[pk], prop_db, other_db, oth["qb_att"]["alpha"], Dn, rng, C.qb_bucket_mode, force_one=C.force_qb)
    h_q = np.array([E["qb_sack"][i] if i is not None else D_["qb_sack"] for i in eff_v.qb_i])
    p_int_q = np.array([E["qb_int"][i] if i is not None else D_["qb_int"] for i in eff_v.qb_i])
    sk_q = rng.binomial(d_q, np.clip(h_q, 0, 1)[None, :])
    sk_out = rng.binomial(d_out, C.sbar)
    scr_q = np.zeros((N, n_q), np.int64)
    for j, gid in enumerate(qb_ids):
        if gid in carry_pos:
            scr_q[:, j] = rng.binomial(cnt_c[:, carry_pos[gid]], C.f_scr)
    need = sk_q + scr_q
    adj_q = np.maximum(need - d_q, 0)
    scr_q = np.minimum(scr_q, np.maximum(d_q - sk_q, 0))
    a_q = np.maximum(d_q - sk_q - scr_q, 0)
    a_out = d_out - sk_out
    A = a_q.sum(1) + a_out
    NT = rng.binomial(A, C.nt)
    Tt = A - NT
    # ------------- target allocation
    cnt_t, out_t, act_t = alloc_given(tt[pk], tt["P1"], oth["target"]["other"], oth["target"]["alpha"], Tt, rng)
    wt = _rz_weights(g, "rz_target", tt)
    rz_t = rng.binomial(cnt_t, np.clip(q_t * psi[:, None] * wt[None, :], 0, 0.98))
    rz_t_out = rng.binomial(out_t, np.clip(q_t * psi, 0, 0.98))
    rz_nt = rng.binomial(NT, np.clip(q_t * psi, 0, 0.98))
    # goal-line opportunities are a binomial thinning of red-zone opportunities (separate deterministic generator: the main random stream is untouched)
    rng_gl = np.random.default_rng(int(rz_c.sum() * 7919 + rz_t.sum() * 104729 + N))
    gl_c = rng_gl.binomial(rz_c, C.gl_r); gl_t = rng_gl.binomial(rz_t, C.gl_t)
    # =============== rush events
    res = {"n_c": n_c, "n_t": n_t, "n_q": n_q, "gl_rush": gl_c, "gl_tgt": gl_t}
    rush_att = cnt_c; rush_yds = np.zeros((N, n_c + 1)); rush_td = np.zeros((N, n_c + 1), np.int64)
    m_full = np.column_stack([cnt_c, out_c]); m_rz = np.column_stack([rz_c, rz_c_out])
    dr, pr, pos = flat_groups(m_full)
    if len(dr):
        rz_flag = pos < m_rz[dr, pr]
        P_r = np.vstack([E["rush"][i] if i is not None else D_["rush"] for i in eff_v.rush_i] + [D_["rush"]])
        cdf_r = _cdf(P_r)
        td_rz = np.array([E["rush_td"][0][i] if i is not None else D_["rush_td"][0] for i in eff_v.rush_i] + [D_["rush_td"][0]])
        td_out = np.array([E["rush_td"][1][i] if i is not None else D_["rush_td"][1] for i in eff_v.rush_i] + [D_["rush_td"][1]])
        p_long = np.maximum((P_r * (C.rush_v >= 21)[None, :]).sum(1), 1e-4)
        y, td = draw_rush(cdf_r, pr, rz_flag, td_rz, td_out, p_long, C, rng)
        np.add.at(rush_yds, (dr, pr), y); np.add.at(rush_td, (dr, pr), td.astype(np.int64))
        if trace:
            res["_rush_events"] = {"draw": dr, "rusher": pr, "rz": rz_flag, "td": td, "yards": y}
    res.update(rush_att=rush_att, rush_att_out=out_c, rz_rush=rz_c, rush_yds=rush_yds[:, :n_c], rush_yds_out=rush_yds[:, n_c], rush_td=rush_td[:, :n_c], rush_td_out=rush_td[:, n_c])
    # =============== pass events
    att_lab = np.column_stack([a_q, a_out])                       # QB label groups (n_q outside)
    d1, q1, _ = flat_groups(att_lab)
    rec_lab = np.column_stack([cnt_t, out_t, NT])                  # receiver labels: named, outside, no-target
    d2, r2, pos2 = flat_groups(rec_lab)
    rz_lab = np.column_stack([rz_t, rz_t_out, rz_nt])
    rz2 = pos2 < rz_lab[d2, r2]
    qb_stat = {k: np.zeros((N, n_q + 1)) for k in ("att", "cmp", "yds", "td", "int")}
    rc_stat = {k: np.zeros((N, n_t + 2)) for k in ("tgt", "rec", "yds", "td", "rz_tgt", "air")}
    tot_int = np.zeros(N, np.int64); tot_cmp = np.zeros(N, np.int64); tot_ptd = np.zeros(N, np.int64)
    if len(d1):
        assert len(d1) == len(d2)
        order = np.lexsort((rng.random(len(d2)), d2))
        r_s, rz_s = r2[order], rz2[order]                          # receiver label / rz flag re-paired within draw
        L = len(d1)
        rows_air = np.vstack([E["air"][i] if i is not None else D_["air"] for i in eff_v.rec_i] + [D_["air"], D_["air"]])
        rows_catch = np.vstack([E["catch"][i] if i is not None else D_["catch"] for i in eff_v.rec_i] + [D_["catch"], D_["catch"]])
        rows_yac = np.stack([E["yac"][i] if i is not None else D_["yac"] for i in eff_v.rec_i] + [D_["yac"], D_["yac"]])
        td_rz_t = np.array([E["rec_td"][0][i] if i is not None else D_["rec_td"][0] for i in eff_v.rec_i] + [D_["rec_td"][0]] * 2)
        td_out_t = np.array([E["rec_td"][1][i] if i is not None else D_["rec_td"][1] for i in eff_v.rec_i] + [D_["rec_td"][1]] * 2)
        cbar = np.array([(rows_air[j] * rows_catch[j][AIR_BK]).sum() for j in range(len(rows_air))])
        p_long_c = plong_completion(rows_air, rows_catch, rows_yac, C)
        dq = np.array([(logit(E["qb_comp"][i]) - logit(E["qb_comp_league"][i])) if (i is not None and qb_adjust) else np.zeros(4)
                       for i in eff_v.qb_i] + [np.zeros(4)]).reshape(n_q + 1, 4)
        pint = np.append(p_int_q, D_["qb_int"])
        is_int = rng.random(L) < pint[q1]
        targeted = r_s < n_t + 1                                   # named or outside receiver (not the no-target sentinel)
        zr = np.where(rz_s, C.comp_rate_rz / C.comp_rate_all, C.comp_rate_out / C.comp_rate_all)      # completion rate of the play's field zone relative to league
        # completion probability by air-yard bin (receiver catch hazard by air bucket, QB log-odds adjustment, zone scaling); P(complete) marginalises over the air pmf
        c_b = np.clip(sigmoid(logit(rows_catch[r_s]) + dq[q1]) * zr[:, None], 0, 0.995)               # [L,4]
        Wc = rows_air[r_s] * c_b[:, AIR_BK]                                                            # [L,K_A] P(air bin, complete)
        P_c = Wc.sum(1)
        c_all = np.minimum(P_c / np.maximum(1 - pint[q1], 1e-6), 0.995)
        # red-zone touchdown decided at the TARGET (hazard per target, among non-interception attempts); the play is then a completion of 1-20 yards
        p_td_rz = np.minimum(td_rz_t[r_s] / np.maximum(1 - pint[q1], 1e-6), 0.9)
        td_rz_flag = targeted & ~is_int & rz_s & (rng.random(L) < p_td_rz)
        c_nontd = np.clip((c_all - p_td_rz) / np.maximum(1 - p_td_rz, 1e-6), 0, 0.995)
        c_eff = np.where(rz_s, c_nontd, c_all)
        comp = targeted & ~is_int & (td_rz_flag | (rng.random(L) < c_eff))
        yards = np.zeros(L); air_val = np.zeros(L); td = np.zeros(L, bool)
        # completions: K candidate (air | complete, YAC) chains; candidates are conditional on completion (air ~ P(a) c(bucket(a))), weighted by the league zone yardage ratio
        if comp.any():
            ci = np.where(comp)[0]
            cand_y, cand_a = completion_candidates(Wc[ci], rows_yac, r_s[ci], K_SIR, C, rng)
            rzc = rz_s[ci]; tdc = td_rz_flag[ci]
            lo = np.where(tdc, 1.0, -10.0); hi = np.where(rzc, np.where(tdc, 20.0, 19.0), 99.0)
            w = C.zone_w(cand_y, "comp", rzc) * ((cand_y >= lo[:, None]) & (cand_y <= hi[:, None]))
            pick = sir_pick(w, rng)
            none = w.sum(1) <= 0
            ysel = np.take_along_axis(cand_y, pick[:, None], 1)[:, 0]; asel = np.take_along_axis(cand_a, pick[:, None], 1)[:, 0]
            ysel = np.where(none, np.clip(cand_y[:, 0], lo, hi), ysel)
            yards[ci] = ysel; air_val[ci] = asel
            # touchdown outside the red zone: only on a completion covering >= 21 yards; P = P(TD | target) / (P(complete) x P(>= 21 yds | complete))
            q_out = np.minimum(td_out_t[r_s[ci]] / np.maximum(P_c[ci] * p_long_c[r_s[ci]], 1e-4), 0.9)
            td[ci] = np.where(rzc, tdc, (ysel >= 21) & (rng.random(len(ci)) < q_out))
        np.add.at(qb_stat["att"], (d1, q1), 1); np.add.at(qb_stat["cmp"], (d1, q1), comp); np.add.at(qb_stat["yds"], (d1, q1), yards)
        np.add.at(qb_stat["td"], (d1, q1), td); np.add.at(qb_stat["int"], (d1, q1), is_int)
        real = targeted
        np.add.at(rc_stat["tgt"], (d1[real], r_s[real]), 1); np.add.at(rc_stat["rec"], (d1, r_s), comp); np.add.at(rc_stat["yds"], (d1, r_s), yards)
        np.add.at(rc_stat["td"], (d1, r_s), td); np.add.at(rc_stat["rz_tgt"], (d1[real], r_s[real]), rz_s[real])
        np.add.at(rc_stat["air"], (d1, r_s), air_val)
        np.add.at(tot_int, d1, is_int); np.add.at(tot_cmp, d1, comp); np.add.at(tot_ptd, d1, td)
        if trace:
            res["_pass_events"] = {"draw": d1, "qb": q1, "rec": r_s, "rz": rz_s, "int": is_int, "comp": comp, "td": td, "yards": yards, "targeted": targeted}
    res.update(qb=qb_stat, rc=rc_stat, sacks_q=sk_q, sacks_out=sk_out, scr_q=scr_q, adj_q=adj_q, A=A, NT=NT, Tt=Tt, Rn=Rn, Dn=Dn,
               tot_int=tot_int, tot_cmp=tot_cmp, tot_ptd=tot_ptd, tgt_named=cnt_t, tgt_out=out_t, rz_tgt=rz_t, rz_tgt_out=rz_t_out, rz_nt=rz_nt,
               act_c=act_c, act_t=act_t, act_q=act_q, dropbacks_q=d_q, dropbacks_out=d_out)
    res["sacks_total"] = sk_q.sum(1) + sk_out
    res["scr_total"] = scr_q.sum(1)
    res["plays"] = Rn + Dn - res["scr_total"]
    res["rush_td_total"] = rush_td.sum(1)
    return res



K_SIR = 6


def sir_pick(w, rng):
    """Sampling-importance-resampling: pick one candidate per row with probability proportional to its weight (rows with no weight return column 0)."""
    tot = w.sum(1, keepdims=True)
    c = np.cumsum(w / np.maximum(tot, 1e-300), 1)
    u = rng.random((len(w), 1))
    return np.minimum((u > c).sum(1), w.shape[1] - 1)


def completion_candidates(Wc, rows_yac, owner, K, C, rng):
    """K candidate (air | complete, YAC) completions per event. Wc [E,K_A] = P(air bin, complete) for the event's receiver / QB / zone; YAC ~ receiver pmf by air bucket.
    Returns (yards [E,K], air value [E,K])."""
    E = len(owner)
    cum = np.cumsum(Wc / np.maximum(Wc.sum(1, keepdims=True), 1e-300), 1)
    u = rng.random((E, K))
    air_bin = np.minimum((u[:, :, None] > cum[:, None, :]).sum(2), Wc.shape[1] - 1)
    bkt = AIR_BK[air_bin]
    yac_bin = np.zeros((E, K), int)
    for b in range(4):
        m = bkt == b
        if m.any():
            ee, kk = np.where(m)
            yac_bin[ee, kk] = sample_bins(_cdf(rows_yac[:, b, :]), owner[ee], rng.random(len(ee)))
    a_val = C.air_v[air_bin]
    return a_val + C.yac_v[bkt, yac_bin], a_val


def plong_completion(rows_air, rows_catch, rows_yac, C):
    """P(receiving yards >= 21 | completion) per receiver row, from the chain pmfs."""
    out = np.zeros(len(rows_air))
    for j in range(len(rows_air)):
        num = den = 0.0
        for b in range(4):
            m = AIR_BK == b
            pa = rows_air[j] * m * rows_catch[j][b]
            tot_yac = C.air_v[:, None] + C.yac_v[b][None, :]
            num += float((pa[:, None] * rows_yac[j, b][None, :] * (tot_yac >= 21)).sum()); den += float(pa.sum())
        out[j] = max(num / max(den, 1e-9), 1e-4)
    return out


def draw_rush(cdf_r, owner, rz_flag, td_rz, td_out, p_long, C, rng):
    """Rush yards and touchdown for every carry. Red zone: TD ~ hazard; TD carries gain 1-20, other red-zone carries 0-19, all re-weighted by the league red-zone
    yardage likelihood ratio (SIR over K candidates). Outside the red zone a TD needs a gain of >= 21 (P = hazard / P(>= 21 yds))."""
    E = len(owner)
    cand = np.zeros((E, K_SIR))
    for k in range(K_SIR):
        cand[:, k] = C.rush_v[sample_bins(cdf_r, owner, rng.random(E))]
    td_rz_flag = rz_flag & (rng.random(E) < td_rz[owner])
    lo = np.where(td_rz_flag, 1.0, -10.0); hi = np.where(rz_flag, np.where(td_rz_flag, 20.0, 19.0), 99.0)
    w = C.zone_w(cand, "rush", rz_flag) * ((cand >= lo[:, None]) & (cand <= hi[:, None]))
    pick = sir_pick(w, rng)
    y = np.take_along_axis(cand, pick[:, None], 1)[:, 0]
    y = np.where(w.sum(1) <= 0, np.clip(cand[:, 0], lo, hi), y)
    q_out = np.minimum(td_out[owner] / p_long[owner], 0.9)
    td = np.where(rz_flag, td_rz_flag, (y >= 21) & (rng.random(E) < q_out))
    return y, td


def cdf_to_p(cdf):
    return np.diff(np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], 1), axis=1)


def _rz_weights(g, rz_name, base_t):
    """Player red-zone role factor = (red-zone share propensity / overall share propensity), carry-weighted mean normalised to 1."""
    T = g["types"]
    if rz_name not in T:
        return np.ones(len(base_t["ids"]))
    rz = dict(zip(T[rz_name]["ids"], T[rz_name]["P1"]))
    ratio = np.array([rz.get(i, base_t["P1"][j]) / max(base_t["P1"][j], 1e-6) if base_t["P1"][j] > 1e-6 else 1.0 for j, i in enumerate(base_t["ids"])])
    ratio = np.clip(ratio, 0.05, 6.0)
    w = base_t["P1"]
    m = (w * ratio).sum() / max(w.sum(), 1e-9)
    return ratio / max(m, 1e-9)


def simulate_defense(gd, off, def_eff, C, N, rng, horizon):
    """Defenders of team `gd` facing simulated offense `off`. def_eff = {target: per-snap rate array aligned with gd defenders}."""
    ds = gd["types"]["def_snap"]
    pk = "pact24" if horizon == "T24" else "pact90"
    n = len(ds["ids"])
    active = rng.random((N, n)) < ds[pk][None, :]
    sd = ds.get("share_sd") or 0.12
    sh = np.clip(ds["P1"][None, :] + sd * rng.standard_normal((N, n)), 0, 1)
    snaps = np.where(active, np.round(off["plays"][:, None] * sh), 0.0)
    out = {"snaps": snaps, "active": active}

    def weights(rate):
        w = rate[None, :] * snaps
        s = w.sum(1, keepdims=True)
        uni = active / np.maximum(active.sum(1, keepdims=True), 1)
        return np.where(s > 0, w / np.maximum(s, 1e-300), uni)
    # sacks come from the offensive sack events; half-sacks are 0.5 + 0.5 among two distinct defenders
    S = off["sacks_total"].astype(np.int64)
    H = rng.binomial(S, C.p_half); Fu = S - H
    w_s = weights(def_eff["sacks"])
    ok = w_s.sum(1) > 0
    cnt = np.zeros((N, n))
    cnt[ok] = rng.multinomial(Fu[ok], w_s[ok])
    nH = int(H.sum())
    if nH:
        dh = np.repeat(np.arange(N), H)
        g_ = -np.log(-np.log(rng.random((nH, n)))) + np.log(np.maximum(w_s[dh], 1e-300))
        g_[w_s[dh] <= 0] = -np.inf
        top2 = np.argsort(-g_, axis=1)[:, :2]
        for c in (0, 1):
            np.add.at(cnt, (dh, top2[:, c]), 0.5)
    out["sacks"] = cnt
    # interceptions come from the offensive interception events
    w_i = weights(def_eff["interceptions"])
    ci = np.zeros((N, n))
    ok = w_i.sum(1) > 0
    ci[ok] = rng.multinomial(off["tot_int"][ok].astype(np.int64), w_i[ok])
    out["interceptions"] = ci
    # tackle credits: tackle-ending plays x (1 + assists) allocated by per-snap tackle rates
    elig = np.maximum((off["Rn"] - off["rush_td_total"]) + (off["tot_cmp"] - off["tot_ptd"]) + off["sacks_total"], 0)
    credits = elig + rng.poisson(elig * max(C.mu_c - 1.0, 0.0))
    w_t = weights(def_eff["tackles"])
    ct = np.zeros((N, n))
    ok = w_t.sum(1) > 0
    ct[ok] = rng.multinomial(credits[ok].astype(np.int64), w_t[ok])
    out["tackles"] = ct
    out["credits"] = credits
    return out


def summarize_accounting(off, N):
    """Opportunity accounting (means per draw): named vs outside vs no-target vs scrambles vs adjustments."""
    return {"rush_named": float(off["rush_att"].sum(1).mean()), "rush_outside": float(off["rush_att_out"].mean()),
            "targets_named": float(off["tgt_named"].sum(1).mean()), "targets_outside": float(off["tgt_out"].mean()), "no_target_attempts": float(off["NT"].mean()),
            "scrambles": float(off["scr_total"].mean()), "dropbacks": float(off["Dn"].mean()), "sacks": float(off["sacks_total"].mean()),
            "attempts": float(off["A"].mean()), "dropback_adjustment_units": float(off["adj_q"].sum(1).mean())}


# ------------------------------------------------------------------ game driver
import zlib  # noqa: E402


def rng_key(*parts):
    return np.random.default_rng(zlib.crc32(repr(parts).encode()))


def defender_rates(g, eff, idx_def, defaults_rate):
    ds = g["types"]["def_snap"]; s, w, _ = g["key"]
    out = {}
    for t in DE.TARGETS:
        arr = eff["def_rate"][t]
        out[t] = np.array([arr[idx_def[(s, w, gid)]] if (s, w, gid) in idx_def else defaults_rate[t] for gid in ds["ids"]])
    return out


def run_game(pack, teamA, teamB, s, w, eff, idx, defaults, C, N, seed, horizon, script=None, qb_adjust=True, want_def=True):
    """Simulate both teams' offences and defences for one game. Returns per-team results (event-derived) keyed by team."""
    res = {}
    for tm in (teamA, teamB):
        g = pack["games"].get((s, w, tm))
        if g is not None:
            g["_meta"] = pack["meta"]
    gs = {tm: pack["games"].get((s, w, tm)) for tm in (teamA, teamB)}
    offs = {}
    for tm, opp in ((teamA, teamB), (teamB, teamA)):
        g = gs[tm]
        if g is None or not all(k in g["types"] for k in ("carry", "target", "qb_att")):
            continue
        ev = EffView(eff, idx, s, w, g["types"]["carry"]["ids"], g["types"]["target"]["ids"], g["types"]["qb_att"]["ids"], defaults)
        sc = None if script is None else script.get(tm)
        offs[tm] = simulate_offense(g, ev, C, N, rng_key(seed, s, w, tm, "off"), horizon, sc, qb_adjust)
        offs[tm]["_ev_missing"] = ev.missing
    for tm, opp in ((teamA, teamB), (teamB, teamA)):
        r = {"off": offs.get(tm)}
        if want_def and gs[tm] is not None and "def_snap" in gs[tm]["types"] and opp in offs:
            r["def"] = simulate_defense(gs[tm], offs[opp], defender_rates(gs[tm], eff, idx["def"], defaults["def_rate"]), C, N, rng_key(seed, s, w, tm, "def"), horizon)
        res[tm] = r
    return res, gs


def collect(res, gs, s, w):
    """Named-player statistic samples: {stat: (keys, S [n, N])}, following the Phase 1A scored universes."""
    out = {}
    def add(name, keys, arr):
        if name in out:
            out[name][0].extend(keys); out[name][1].append(arr)
        else:
            out[name] = (list(keys), [arr])
    for tm, r in res.items():
        o, g = r.get("off"), gs.get(tm)
        if o is None or g is None:
            continue
        ct, tt, qt = g["types"]["carry"], g["types"]["target"], g["types"]["qb_att"]
        kc = [(s, w, tm, i) for i in ct["ids"]]; kt = [(s, w, tm, i) for i in tt["ids"]]; kq = [(s, w, tm, i) for i in qt["ids"]]
        add("rush_att", kc, o["rush_att"].T); add("rush_yds", kc, o["rush_yds"].T); add("rush_td", kc, o["rush_td"].T)
        add("active_carry", kc, o["act_c"].T.astype(np.float32)); add("active_target", kt, o["act_t"].T.astype(np.float32))
        add("targets", kt, o["rc"]["tgt"][:, :len(kt)].T); add("rec", kt, o["rc"]["rec"][:, :len(kt)].T)
        add("rec_yds", kt, o["rc"]["yds"][:, :len(kt)].T); add("rec_td", kt, o["rc"]["td"][:, :len(kt)].T)
        add("pass_att", kq, o["qb"]["att"][:, :len(kq)].T); add("pass_cmp", kq, o["qb"]["cmp"][:, :len(kq)].T); add("pass_yds", kq, o["qb"]["yds"][:, :len(kq)].T)
        add("pass_td", kq, o["qb"]["td"][:, :len(kq)].T); add("int", kq, o["qb"]["int"][:, :len(kq)].T)
        add("sacks_taken", kq, o["sacks_q"].T)
        # rush TD players whose ids are also receivers -> anytime TD on the union
        td_by = {}
        for j, k in enumerate(kc):
            td_by[k] = o["rush_td"][:, j].astype(float)
        for j, k in enumerate(kt):
            td_by[k] = td_by.get(k, 0.0) + o["rc"]["td"][:, j]
        add("atd", list(td_by), np.array(list(td_by.values())))
        d = r.get("def")
        if d is not None:
            kd = [(s, w, tm, i) for i in g["types"]["def_snap"]["ids"]]
            add("active_def", kd, d["active"].T.astype(np.float32)); add("tackles", kd, d["tackles"].T); add("sacks", kd, d["sacks"].T); add("def_int", kd, d["interceptions"].T); add("def_snaps", kd, d["snaps"].T)
    return {k: (v[0], np.vstack(v[1])) for k, v in out.items()}

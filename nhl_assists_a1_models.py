"""Locked A0/A1/A2 analytic assist-count models. No network or confirmation input."""
import math
import numpy as np
from scipy import optimize, special, stats

import nhl_assists_a1_data as AD
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1a_models as MD

CLASSES = ('F', 'D', 'U')
ALPHA = .001
KAPPA = 5.
BOUNDS = (math.log(.05), math.log(5000.))


def classes(tab):
    return np.where(tab['POS_F'] == 1, 'F', np.where(tab['POS_D'] == 1, 'D', 'U'))


def subset(tab, mask):
    return {k: v[mask] for k, v in tab.items()}


class Components:
    """Training-only shrinkage; current membership untouched; stable skill carries."""
    def fit(self, tab):
        AD.assert_development(tab)
        cls, played = classes(tab), tab['played'] == 1
        self.play, self.goal = {}, {}
        poolplay = float(played.mean())
        poolgoal = float(tab['player_goals_label'][played].mean())
        for c in CLASSES:
            m = cls == c
            self.play[c] = float(played[m].mean()) if m.any() else poolplay
            self.goal[c] = float(tab['player_goals_label'][m & played].mean()) if (m & played).any() else poolgoal
        ix, _ = AD.groups(tab)
        self.goal_mean = float(tab['team_goals_label'][ix].mean())
        return self

    def enrich(self, tab, team_level=False):
        # No realized target fields read here.
        AD.assert_development(tab)
        c = classes(tab)
        p = (tab['plays10'] + KAPPA*np.array([self.play[x] for x in c])) / (tab['den10'] + KAPPA)
        strength = p * (tab['GOALS_SUM_CUM'] + KAPPA*np.array([self.goal[x] for x in c])) / (tab['N_APPS_CUM'] + KAPPA)
        ix, inv = AD.groups(tab)
        sums = np.bincount(inv, weights=strength, minlength=len(ix))
        out = dict(tab)
        out['TEAMMATE_GOAL_STRENGTH'] = sums[inv] if team_level else sums[inv] - strength
        out['TEAM_GF_SHRUNK5'] = (tab['TEAM_GF_SUM5'] + KAPPA*self.goal_mean) / (tab['TEAM_GF_N5'] + KAPPA)
        out['OPP_GA_SHRUNK5'] = (tab['OPP_GA_SUM5'] + KAPPA*self.goal_mean) / (tab['OPP_GA_N5'] + KAPPA)
        return out

    def artifact(self):
        return {'position_play_rate': self.play, 'position_goals_per_appearance': self.goal,
                'training_team_goal_mean': self.goal_mean, 'pseudogames': KAPPA}


class CountModel:
    def __init__(self, features):
        self.features = features

    def fit(self, tab, y):
        AD.assert_development(tab)
        self.prep = MD.Preprocessor(self.features).fit(tab)
        X = self.prep.transform(tab)
        self.reg, self.info = MD.fit_poisson(X, y, ALPHA)
        if not self.info['converged']:
            raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: Poisson nonconvergence')
        mu = self.reg.predict(X)
        # Same registered dispersion MLE, with its convergence recorded explicitly.
        lo, hi = math.log(1e-6), math.log(20)
        res = optimize.minimize_scalar(lambda t: -MD.nb2_loglik(math.exp(t), y, np.maximum(mu, 1e-12)),
                                      bounds=(lo, hi), method='bounded', options={'xatol': 1e-9})
        if not res.success:
            raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: dispersion nonconvergence')
        self.disp = {'alpha': math.exp(res.x), 'converged': bool(res.success),
                     'at_lower_bound': bool(res.x-lo < 1e-3), 'at_upper_bound': bool(hi-res.x < 1e-3)}
        return self

    def params(self, tab):
        AD.assert_development(tab)
        return {'mu': self.reg.predict(self.prep.transform(tab)), 'alpha': self.disp['alpha']}

    def artifact(self):
        return MD.poisson_artifact(self.reg, self.prep, ALPHA, {'dispersion': self.disp, 'fit': self.info})


def nb_pmf(params):
    n = len(params['mu'])
    blocks = list(M.pmf_matrix('nb2', params, np.zeros(n, dtype=np.int64)))
    K = max(pm.shape[1] for _, pm, _ in blocks)
    out = np.zeros((n, K))
    for sl, pm, sf in blocks:
        out[sl, :pm.shape[1]] = pm
    check_pmf(out)
    return out


def team_params(model, tab):
    """A single scoring environment per team-game, broadcast to candidates."""
    ix, inv = AD.groups(tab)
    params = model.params(subset(tab, ix))
    return {'mu': params['mu'][inv], 'alpha': params['alpha']}


def check_pmf(pmf):
    if not np.isfinite(pmf).all() or (pmf < 0).any() or np.max(abs(pmf.sum(1)-1)) > 1e-9:
        raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: invalid full PMF')
    return {'max_row_sum_error': float(np.max(abs(pmf.sum(1)-1))),
            'min_probability': float(pmf.min()), 'support_max': pmf.shape[1]-1}


class A0:
    def fit(self, train):
        AD.assert_development(train)
        self.components = Components().fit(train)
        self.count = CountModel(AD.DIRECT).fit(self.components.enrich(train), train['assists'])
        return self

    def predict(self, tab):
        params = self.count.params(self.components.enrich(tab))
        return {'pmf': nb_pmf(params)}

    def artifact(self):
        return {'model': 'A0', 'components': self.components.artifact(), 'count': self.count.artifact()}


def hyper_nll(theta, y, n):
    a, b = np.exp(theta)
    return -float(np.sum(special.gammaln(n+1)-special.gammaln(y+1)-special.gammaln(n-y+1)
                         +special.betaln(y+a, n-y+b)-special.betaln(a, b)))


def fit_hyper(y, n):
    y, n = np.asarray(y, float), np.asarray(n, float)
    ok = n > 0
    y, n = y[ok], n[ok]
    if not len(y) or (y > n).any():
        raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: no valid involvement exposure')
    mean = np.clip(y.sum()/n.sum(), .0001, .9999)
    res = optimize.minimize(hyper_nll, np.log([mean*50, (1-mean)*50]), args=(y, n),
                            method='L-BFGS-B', bounds=[BOUNDS, BOUNDS],
                            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
    if not res.success:
        raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: hyperprior nonconvergence')
    a, b = np.exp(res.x)
    return {'a': float(a), 'b': float(b), 'converged': bool(res.success), 'n_iter': int(res.nit),
            'n_players': len(y), 'total_assists': float(y.sum()), 'team_goal_exposure': float(n.sum()),
            'boundary_hits': [bool(min(t-BOUNDS[0], BOUNDS[1]-t) < 1e-3) for t in res.x]}


class A1:
    def fit(self, train):
        AD.assert_development(train)
        self.components = Components().fit(train)
        enriched = self.components.enrich(train, team_level=True)
        ix, _ = AD.groups(train)
        self.team = CountModel(AD.ENV).fit(subset(enriched, ix), train['team_goals_label'][ix])
        self.avprep = MD.Preprocessor(AD.AVAIL).fit(train)
        played = train['played'].astype(int)
        self.avreg, self.avconstant = None, None
        if len(np.unique(played)) == 1:
            self.avconstant = float(played[0]); self.avinfo = {'converged': True, 'constant_class': True}
        else:
            self.avreg, self.avinfo = MD.fit_logistic(self.avprep.transform(train), played, C=1.)
            if not self.avinfo['converged']:
                raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: availability nonconvergence')
        cls = classes(train)
        agg = {}
        for p, c, y, g in zip(train['player_id'][played == 1], cls[played == 1], train['assists'][played == 1], train['team_goals_label'][played == 1]):
            v = agg.setdefault((int(p), c), [0, 0]); v[0] += int(y); v[1] += int(g)
        vs = list(agg.values())
        pooled = fit_hyper([v[0] for v in vs], [v[1] for v in vs])
        self.hyper, self.hyper_source = {}, {}
        for c in CLASSES:
            vals = [v for (_, cc), v in agg.items() if cc == c and v[1] > 0]
            self.hyper[c] = fit_hyper([v[0] for v in vals], [v[1] for v in vals]) if len(vals) >= 30 else pooled
            self.hyper_source[c] = 'class' if len(vals) >= 30 else 'pooled_fallback(<30 players)'
        self.roleprep = MD.Preprocessor(AD.ROLE).fit(subset(train, played == 1))
        a, b = self.posterior(train)
        offset = special.logit(np.clip(a/(a+b), 1e-9, 1-1e-9))
        mask = (played == 1) & (train['team_goals_label'] > 0)
        X = np.column_stack([np.ones(mask.sum()), self.roleprep.transform(subset(train, mask))])
        y, g, off = train['assists'][mask], train['team_goals_label'][mask], offset[mask]
        def obj(beta):
            z = off + X @ beta
            loss = np.mean(g*np.logaddexp(0, z)-y*z) + .5*ALPHA*np.dot(beta[1:], beta[1:])
            grad = X.T @ (g*special.expit(z)-y)/len(y) + ALPHA*np.r_[0, beta[1:]]
            return loss, grad
        res = optimize.minimize(obj, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
        if not res.success:
            raise RuntimeError('FIT_OR_DISTRIBUTION_FAILED: involvement-role nonconvergence')
        self.beta = res.x; self.roleinfo = {'converged': bool(res.success), 'n_iter': int(res.nit)}
        return self

    def posterior(self, tab):
        a0 = np.array([self.hyper[c]['a'] for c in classes(tab)])
        b0 = np.array([self.hyper[c]['b'] for c in classes(tab)])
        return a0+tab['ASSISTS_SUM_CUM'], b0+tab['TEAM_GOALS_EXPOSURE_CUM']-tab['ASSISTS_SUM_CUM']

    def involvement(self, tab):
        AD.assert_development(tab)
        a, b = self.posterior(tab)
        X = np.column_stack([np.ones(len(a)), self.roleprep.transform(tab)])
        q = np.clip(special.expit(special.logit(np.clip(a/(a+b), 1e-9, 1-1e-9))+X @ self.beta), 1e-9, 1-1e-9)
        p = np.full(len(q), self.avconstant) if self.avreg is None else self.avreg.predict_proba(self.avprep.transform(tab))[:, 1]
        return p, q, a+b

    def predict(self, tab):
        enriched = self.components.enrich(tab, team_level=True)
        params = team_params(self.team, enriched)
        goals = nb_pmf(params)
        p, q, concentration = self.involvement(tab)
        n, width = goals.shape
        out = np.zeros_like(goals)
        g = np.arange(width)[None, :, None]
        y = np.arange(width)[None, None, :]
        # Small deterministic chunks avoid a giant rows x goals x assists tensor.
        for j in range(0, n, 200):
            sl = slice(j, min(n, j+200))
            a = (q[sl]*concentration[sl])[:, None, None]
            b = ((1-q[sl])*concentration[sl])[:, None, None]
            rem = np.maximum(g-y, 0)
            lp = special.gammaln(g+1)-special.gammaln(y+1)-special.gammaln(rem+1)+special.betaln(y+a, rem+b)-special.betaln(a, b)
            bb = np.where(y <= g, np.exp(lp), 0.)
            out[sl] = np.einsum('ng,ngy->ny', goals[sl], bb)*p[sl, None]
            out[sl, 0] += 1-p[sl]
        check_pmf(out)
        return {'pmf': out, 'p_active': p, 'assist_involvement': q,
                'team_goal_mean': params['mu'],
                'prior_dominated': np.array([self.hyper[c]['a']+self.hyper[c]['b'] for c in classes(tab)]) > tab['TEAM_GOALS_EXPOSURE_CUM']}

    def artifact(self):
        return {'model': 'A1', 'components': self.components.artifact(), 'team_goals_component': self.team.artifact(),
                'availability': MD.logistic_artifact(self.avreg, self.avprep, 1.) if self.avreg is not None else {'constant': self.avconstant},
                'availability_fit': self.avinfo, 'hyperprior': self.hyper, 'hyperprior_source': self.hyper_source,
                'role': {'coefficients': self.beta.tolist(), 'preprocessing': self.roleprep.schema(), 'fit': self.roleinfo}}


class A2:
    def fit(self, train, involvement):
        AD.assert_development(train)
        self.involvement_model = involvement
        self.components = involvement.components
        ix, _ = AD.groups(train)
        self.credits = CountModel(AD.ENV).fit(subset(self.components.enrich(train, team_level=True), ix), train['team_credits_label'][ix])
        total = float(train['team_credits_label'][ix].sum())
        outside = total-float(train['assists'].sum())
        if not 0 <= outside <= total or total <= 0:
            raise ValueError('BLOCKED_BY_DATA: outside-candidate credit partition')
        self.rho = outside/total
        self.total_train_credits, self.outside_train_credits = total, outside
        return self

    def allocation(self, tab):
        p, q, _ = self.involvement_model.involvement(tab)
        ix, inv = AD.groups(tab)
        weights = p*q
        sums = np.bincount(inv, weights=weights, minlength=len(ix))
        pi = np.divide((1-self.rho)*weights, sums[inv], out=np.zeros_like(weights), where=sums[inv] > 0)
        return pi, ix, inv

    def predict(self, tab):
        pi, ix, inv = self.allocation(tab)
        params = team_params(self.credits, self.components.enrich(tab, team_level=True))
        return {'pmf': nb_pmf({'mu': params['mu']*pi, 'alpha': params['alpha']}),
                'allocation_probability': pi, 'team_credit_mean': params['mu'], 'outside_share': self.rho}

    def draw(self, tab, n_draws=100, seed=20261002):
        """Joint credit states, with explicit outside bucket; no target participant oracle."""
        pi, ix, inv = self.allocation(tab)
        params = team_params(self.credits, self.components.enrich(tab, team_level=True))
        rng = np.random.default_rng(seed)
        totals, allocations, outside = [], [], []
        for j, i in enumerate(ix):
            m = inv == j
            probs = pi[m]
            probs = np.r_[probs, max(0., 1-probs.sum())]
            probs /= probs.sum()  # floating point roundoff only
            r, mu = 1/params['alpha'], params['mu'][i]
            total = rng.negative_binomial(r, r/(r+mu), size=n_draws)
            alloc = np.stack([rng.multinomial(int(t), probs) for t in total])
            totals.append(total); allocations.append(alloc[:, :-1]); outside.append(alloc[:, -1])
        return totals, allocations, outside

    def artifact(self):
        return {'model': 'A2', 'team_credit_component': self.credits.artifact(), 'outside_share': self.rho,
                'training_total_credits': self.total_train_credits, 'training_outside_credits': self.outside_train_credits,
                'involvement_source': 'frozen A1 within fold; no A2-specific involvement fit'}

"""Routed tennis point mechanics and transparent evidence-limited human baseline."""
import math
import numpy as np
from tennis_v2_phase1_guards import EvidenceBlocked
from tennis_v2_phase1_scoring import hold_probability


def clip(p):
    return min(.90,max(.30,float(p)))


def point_projection(feature, family):
    o, r = feature['own'], feature['opponent_conceded']
    overall = o['overall'][0]
    surface = o['rate'][0]
    adj = r['rate'][0]-o['league'][0]
    first = clip(o['rate'][2]+r['rate'][2]-o['league'][2])
    second = clip(o['rate'][3]+r['rate'][3]-o['league'][3])
    if family == 'U': p = overall
    elif family == 'S': p = surface
    elif family == 'I': p = overall+r['overall'][0]-o['league'][0]
    elif family == 'C': p = surface+adj
    elif family == 'D': p = o['rate'][1]*first+(1-o['rate'][1])*second
    elif family in ('R','HUMAN'): p = surface+adj+feature['old_adjusted_residual']
    else: raise EvidenceBlocked('BLOCKED_DATA: unknown point family')
    p = clip(p)
    return dict(serve_point_win=p, receiver_return_point_win=1-p, hold=hold_probability(p),
                break_probability=1-hold_probability(p), baseline_overall=overall,
                surface_adjustment=surface-overall, opponent_adjustment=adj,
                old_form_adjustment=feature['old_adjusted_residual'] if family in ('R','HUMAN') else 0.,
                first_in=o['rate'][1], first_serve_win=first, second_serve_win=second,
                channel_mixture=o['rate'][1]*first+(1-o['rate'][1])*second)


def logit(p):
    p = np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def fit_stack(x,y):
    """One symmetric DEV-only fit, L2=50 around mechanics-only; no intercept."""
    x,y = np.asarray(x,float),np.asarray(y,float)
    beta = np.array([1.,0.,0.]); center = beta.copy()
    for _ in range(100):
        z = np.clip(x@beta,-30,30); p = 1/(1+np.exp(-z))
        grad = x.T@(y-p)-50*(beta-center)
        hess = (x.T*(p*(1-p)))@x+50*np.eye(3)
        step = np.linalg.solve(hess,grad); beta += step
        if max(abs(step)) < 1e-9: break
    if not np.isfinite(beta).all(): raise EvidenceBlocked('BLOCKED_DATA: stack fit')
    return beta.tolist()


def stack_predict(probabilities,beta):
    z = sum(float(logit(p))*b for p,b in zip(probabilities,beta))
    return 1/(1+math.exp(-max(-30,min(30,z))))

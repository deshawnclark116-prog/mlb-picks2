#!/usr/bin/env python3
"""Never silently claim that paired CFB moneyline probabilities are Platt calibrated.

The existing production moneyline context branch computes one calibrated team
win probability PER side and *then* divides each by their sum. That changes
the numerical probability after calibration. This module only describes the
transform. It does NOT tune, scale, rank or replace a production probability.
"""
import math

SCHEMA="CFB_MONEYLINE_PREPAIR_POSTPAIR_PROVENANCE_V1"


def describe_pair(p_team, p_opponent, p_published):
    for name,x in (("team",p_team),("opponent",p_opponent),("published",p_published)):
        if not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) or not 0<=x<=1:
            raise ValueError(f"invalid {name} probability")
    tot=p_team+p_opponent
    if not 0<tot<=2:
        raise ValueError("no normalizable two-sided pair")
    implied=p_team/tot
    if abs(implied-p_published)>1e-6:
        raise ValueError("published probability does not match original paired normalization")
    return {
        "schema":SCHEMA,"pre_pair_calibrated_team_probability":round(float(p_team),7),
        "pre_pair_calibrated_opponent_probability":round(float(p_opponent),7),
        "pre_pair_sum":round(float(tot),7),
        "pair_normalization_shift":round(float(p_published-p_team),7),
        "published_probability":round(float(p_published),7),
        "note":"Platt calibration preceding a nonlinear two-team rescale does not by itself certify calibration of the final served probability",
        "model_changed":False,
    }

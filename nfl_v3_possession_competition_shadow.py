"""NFL V3 alternate shadow hypothesis: one game, competing possession volume.

The original +tempo correlation proposal is FROZEN, not promoted. Real 2024
and 2025 matched team-game residuals were NEGATIVE (~-0.47 and ~-0.39)
using a carries+pass-attempts proxy. Rather than merely flipping a random
correlation sign, derive both teams' volumes from a *shared finite pool* of
game opportunities. Each team must compete for a slice.

This remains an UNVALIDATED scenario. The current actual-frozen Phase1C
model still receives script=None; a later runner would have to supply
script={team: {'R': draws, 'D': draws}} explicitly to compare them.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from nfl_v3_joint_game_shadow import _team_baseline

VERSION="NFL_V3_POSSESSION_CONSERVATION_SHADOW_V1"


@dataclass(frozen=True)
class Scenario:
    # Chosen as an explanatory research scenario, not a fitted validated
    # production parameter. 2025 evidence has ALREADY been inspected;
    # no claim remains that year is an untouched holdout.
    shared_total_sigma:float=.07
    possession_concentration:float=45.
    win_script_sigma:float=.60
    run_share_shift:float=.065

    def check(self):
        vals=(self.shared_total_sigma,self.possession_concentration,
              self.win_script_sigma,self.run_share_shift)
        if any(not isfinite(x) for x in vals):
            raise ValueError("NFL_V3_POSSESSION_NONFINITE_PARAMETER")
        if not(0<=self.shared_total_sigma<=.4 and
               2<=self.possession_concentration<=500 and
               0<=self.win_script_sigma<=3 and
               0<=self.run_share_shift<=.15):
            raise ValueError("NFL_V3_POSSESSION_PARAMETER_OUT_OF_RESEARCH_RANGE")


def possession_competition_scripts(
    frozen_team_inputs:Mapping[str,Mapping], team_a:str,team_b:str,
    *, draws:int,seed:int,parameters:Scenario=Scenario()
):
    parameters.check()
    if team_a==team_b or not isinstance(team_a,str) or not isinstance(team_b,str) or not team_a or not team_b:
        raise ValueError("NFL_V3_POSSESSION_DUPLICATE_OR_INVALID_TEAMS")
    if not isinstance(draws,int) or isinstance(draws,bool) or not 10<=draws<=1_000_000:
        raise ValueError("NFL_V3_POSSESSION_INVALID_N")
    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("NFL_V3_POSSESSION_NONREPRODUCIBLE_SEED")
    ar,ad=_team_baseline(frozen_team_inputs.get(team_a),team_a)
    br,bd=_team_baseline(frozen_team_inputs.get(team_b),team_b)
    am=ar+ad;bm=br+bd
    rng=np.random.default_rng(seed)
    # One finite TOTAL of offensive opportunity proxies per game. This
    # is not a complete game clock with drives, turnovers, sack/penalty
    # accounting, or exact on-field plays; it is a conserving scaffold.
    sigma=parameters.shared_total_sigma
    common=np.exp(sigma*rng.standard_normal(draws)-.5*sigma**2)
    full_game=rng.poisson((am+bm)*common)
    split_alpha=am/(am+bm)*parameters.possession_concentration
    split_beta=bm/(am+bm)*parameters.possession_concentration
    share=rng.beta(split_alpha,split_beta,size=draws)
    team_total_a=rng.binomial(full_game,share)
    team_total_b=full_game-team_total_a
    # Neutral hypothetical margin script: one team in a world leans toward
    # rushing while the other leans toward passing. No betting spread or
    # leaked winner is used to infer the sign.
    script_state=np.tanh(parameters.win_script_sigma*rng.standard_normal(draws))
    scripts={}
    for team,total,run0,pass0,sign in (
        (team_a,team_total_a,ar,ad,1),
        (team_b,team_total_b,br,bd,-1)
    ):
        p=np.clip(run0/(run0+pass0)+sign*parameters.run_share_shift*script_state,.02,.95)
        r=rng.binomial(total,p)
        d=total-r
        assert np.array_equal(r+d,total)
        scripts[team]={"R":r,"D":d}
    # Conserves both within-team run/pass proxies AND across-team pool.
    if not np.array_equal(
        scripts[team_a]["R"]+scripts[team_a]["D"]+
        scripts[team_b]["R"]+scripts[team_b]["D"],full_game
    ):
        raise AssertionError("NFL_V3_POSSESSION_CONSERVATION_FAILURE")
    corr=float(np.corrcoef(team_total_a,team_total_b)[0,1])
    report={
        "schema":VERSION,
        "status":"SHADOW_RESEARCH_NOT_DEPLOYED",
        "method":"finite_game_opportunity_pool_then_complementary_team_split",
        "old_positive_common_pace_branch":"FROZEN_FAILED_EMPIRICAL_SUPPORT",
        "parameters_are_exploratory_not_calibrated":True,
        "raw_empirical_evidence_source":"nfl_v3_joint_pace_falsification.py",
        "approximation":"carries + pass-attempts proxy, NOT exact NFL snaps/dropbacks",
        "team_strength_or_winner_inferred":False,
        "bookmaker_data_used":False,
        "new_live_prediction_effect":False,
        "draws":draws,"seed":seed,
        "teams":[team_a,team_b],
        "mean_volumes":{team_a:{"rush":ar,"dropback":ad},
                        team_b:{"rush":br,"dropback":bd}},
        "mean_simulated_team_volumes":{
            t:{"rush":round(float(scripts[t]["R"].mean()),3),
               "dropback":round(float(scripts[t]["D"].mean()),3)}
            for t in (team_a,team_b)
        },
        "mean_simulated_game_pool":round(float(full_game.mean()),3),
        "simulated_team_total_proxy_correlation":round(corr,4),
        "comparison_rule":"Compare only against the original pretip independent runner and real later outcomes with equal seeds, lookback and CRPS; no promotion based on observed sign alone.",
        "production_promotion_allowed":False,
    }
    return scripts,report

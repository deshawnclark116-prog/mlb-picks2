"""NFL V3 *unpromoted* joint-game opportunity/script challenger.

The incumbent Phase1C runner calls the play-event simulator with script=None:
each team's rushes and dropbacks are sampled independently. This challenger
returns the exact existing run_game(script={team: {"R","D"}}) interface.
Common game pace and complementary score-script factors induce between-team
dependence. Player opportunities are STILL allocated inside the unchanged
Phase1C event simulator, conditional on those opportunities.

No sportsbook spreads, implied totals, future injuries, outcome statistics, or
postkickoff information enter. The script assumptions are explicitly NOT fit or
calibrated and MUST NOT be silently used by the production model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

VERSION = "nfl-v3-joint-game-script-research-v1"


@dataclass(frozen=True)
class ShadowParameters:
    # Fixed demonstration coefficients, NOT optimized on historical outcomes.
    pace_sigma: float = 0.16
    team_noise_sigma: float = 0.10
    game_script_sigma: float = 0.60
    run_share_shift: float = 0.065
    run_share_concentration: float = 80.0

    def validate(self):
        values=(self.pace_sigma,self.team_noise_sigma,
                self.game_script_sigma,self.run_share_shift,
                self.run_share_concentration)
        if not all(isfinite(v) for v in values) or any(v<0 for v in values):
            raise ValueError("NFL_V3_GAME_SCRIPT_INVALID_PARAMETER")
        if self.run_share_shift>.15 or self.run_share_concentration<1:
            raise ValueError("NFL_V3_GAME_SCRIPT_UNSAFE_EXPLORATORY_PARAMETER")


def _team_baseline(game: Mapping, team: str):
    if not isinstance(game, Mapping) or not isinstance(team,str) or not team:
        raise ValueError("NFL_V3_GAME_SCRIPT_INVALID_TEAM")
    types=game.get("types")
    if not isinstance(types,Mapping) or not {"carry","qb_att","target"}.issubset(types):
        raise ValueError("NFL_V3_GAME_SCRIPT_MISSING_PHASE1_TEAM_INPUTS")
    rush=types["carry"]
    drop=types["qb_att"]
    if not isinstance(rush,Mapping) or not isinstance(drop,Mapping):
        raise ValueError("NFL_V3_GAME_SCRIPT_INVALID_OPPORTUNITY_OBJECT")
    r,d=rush.get("mu"),drop.get("mu")
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool)
               and isfinite(x) and x>0 for x in (r,d)):
        raise ValueError("NFL_V3_GAME_SCRIPT_MISSING_FINITE_VOLUME")
    if r+d<10 or r+d>150:
        raise ValueError("NFL_V3_GAME_SCRIPT_UNREALISTIC_TEAM_VOLUME")
    return float(r),float(d)


def joint_game_scripts(
    pretip_team_inputs:Mapping[str,Mapping],
    team_a:str,team_b:str,
    *,
    draws:int,
    seed:int,
    parameters:ShadowParameters=ShadowParameters(),
):
    """Return candidate coupled R/D vectors + auditable unpromoted diagnostics.

    Both teams' pace shares the SAME draw-level common latent. Run share
    moves oppositely within each hypothetical game script: a team leading
    tends to run more, the opponent tends to pass more. Score margin is an
    intentionally symmetric hypothetical latent, NOT a score forecast.
    The baseline joint means can differ by team. We never infer who wins.

    Existing frozen Phase1C player simulator can consume these vectors
    unchanged. Caller MUST keep this on a separately named shadow branch.
    """
    parameters.validate()
    if team_a==team_b or not all(isinstance(t,str) and t for t in (team_a,team_b)):
        raise ValueError("NFL_V3_GAME_SCRIPT_DUPLICATE_OR_INVALID_TEAMS")
    if not isinstance(draws,int) or isinstance(draws,bool) or not 10<=draws<=1_000_000:
        raise ValueError("NFL_V3_GAME_SCRIPT_INVALID_DRAW_COUNT")
    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("NFL_V3_GAME_SCRIPT_SEED_NOT_REPRODUCIBLE")
    baselines={t:_team_baseline(pretip_team_inputs.get(t),t) for t in (team_a,team_b)}
    rng=np.random.default_rng(seed)
    # Shared game-wide tempo. Normalize lognormal mean to 1.
    common=np.exp(parameters.pace_sigma*rng.standard_normal(draws)
                  -.5*parameters.pace_sigma**2)
    # Exactly one opposing game-script factor for both clubs.
    score_state=np.tanh(parameters.game_script_sigma*rng.standard_normal(draws))
    out={}
    for sign,team in ((1.,team_a),(-1.,team_b)):
        rmu,dmu=baselines[team]
        total= rmu+dmu
        team_mult=np.exp(parameters.team_noise_sigma*rng.standard_normal(draws)
                         -.5*parameters.team_noise_sigma**2)
        mean_total=total*common*team_mult
        # Total plays are a conditional Poisson draw; the common latent
        # introduces overdispersion and correlation but needs validation.
        total_plays=rng.poisson(mean_total)
        base_share=rmu/total
        shifted_share=np.clip(base_share+sign*parameters.run_share_shift*score_state,.02,.95)
        alpha=shifted_share*parameters.run_share_concentration
        beta=(1-shifted_share)*parameters.run_share_concentration
        share=rng.beta(alpha,beta)
        rush=rng.binomial(total_plays,share)
        drop=total_plays-rush
        if not (len(rush)==draws and len(drop)==draws and
                np.all(rush>=0) and np.all(drop>=0) and
                np.array_equal(rush+drop,total_plays)):
            raise AssertionError("NFL_V3_GAME_SCRIPT_OPPORTUNITY_CONSERVATION")
        out[team]={"R":rush,"D":drop}
    corr=float(np.corrcoef(out[team_a]["R"]+out[team_a]["D"],
                           out[team_b]["R"]+out[team_b]["D"])[0,1])
    report={
        "schema":VERSION,
        "model_impact":"SHADOW_ONLY_NOT_PRODUCTION",
        "unvalidated_hyperparameters":True,
        "pregame_only_inputs":True,
        "has_real_team_strength_or_final_score_data":False,
        "sportsbook_inputs_used":False,
        "draws":draws,
        "seed":seed,
        "teams":[team_a,team_b],
        "estimated_joint_total_plays_correlation":round(corr,5),
        "parameters":{k:getattr(parameters,k) for k in parameters.__dataclass_fields__},
        "baseline_means":{
            t:{"rushes":r,"dropbacks":d}
            for t,(r,d) in baselines.items()},
        "shadow_result_means":{
            t:{"rushes":round(float(out[t]["R"].mean()),3),
               "dropbacks":round(float(out[t]["D"].mean()),3)}
            for t in (team_a,team_b)},
        "warning":"Research scenario, not empirically calibrated. The shared script is NOT the game's actual score trajectory, and must not be promoted on an inspected week."
    }
    return out, report

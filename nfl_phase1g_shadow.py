"""Phase1G role/matchup/game-state research shadow (not production).

Runs only after the original Phase1E/V2 captures, using the exact same
pre-cutoff CAS blobs and verified prefit artifact. Never modifies Phase1E
inputs, scorer, champion source files, model version, or production board.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import zlib
from datetime import datetime, timezone
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1d_cas as CAS
import nfl_phase1d_live as LV
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH
import nfl_phase1_forecast as FC
import nfl_phase1c_sim as SM
import nfl_phase1_store as ST
import nfl_phase1e_ops as OPS
import nfl_phase1f_intelligence as INT
import nfl_phase1g_player_intelligence as G

UTC=timezone.utc
STATUS_SAVED="PHASE1G_REAL_PREGAME_RESEARCH_SAVED"


def asof_context(asof, targets):
    """Only replay pre-cutoff data; coach annotation in the research copy."""
    data=P1.Data(str(asof))
    units=P1.build(data,depth_universe=True,targets=targets)
    out={}
    want=set(targets)
    for unit in units:
        if unit["key"] not in want:
            continue
        team=unit["key"][2]
        # This copies the already-replayed completed game histories. It
        # does not modify nfl_phase1_data.py or Phase1E's frozen pack.
        x=copy.deepcopy(unit)
        for row in x["team_hist"]:
            g=data.game.get((row["s"],row["w"],team))
            row["coach"]=(g or {}).get("coach")
        out[x["key"]]=x
    if len(out)!=len(want):
        raise ValueError("PHASE1F_MISSING_PREGAME_GAME_UNIT")
    return out


def run_group(runner,horizon,entries,season,week,run_id,only_games=None):
    """Generate separate current-game originals, fail closed on any missing input.

    entries: (kickoff, source_set_record, game_ids) grouped by same immutable
    content ID, exactly as the incumbent's run_context receives them.
    """
    results={}
    for kickoff, setrec, ids in entries:
        ids=[gid for gid in ids if only_games is None or gid in only_games]
        if not ids:
            continue
        for gid in ids:
            results[gid]={"game_id":gid,"horizon":horizon,
                           "status":"PHASE1G_RESEARCH_UNAVAILABLE"}
        try:
            if setrec.get("time_travel") is not False:
                raise ValueError("NOT_AN_ORIGINAL_REAL_PREGAME_CAPTURE")
            cutoff=SCH.forecast_cutoff(kickoff,horizon)
            if SCH.parse_iso(setrec["retrieval_ts"])>cutoff:
                raise ValueError("LATE_SOURCE_NOT_ORIGINAL_CUTOFF")
            if datetime.now(UTC)>=kickoff:
                raise ValueError("AFTER_KICKOFF_NO_FORECAST")
            asof=CAS.materialize(runner.store,setrec,runner.root/"tmp"/
                                ("p1g_"+setrec["content_id"][:16]))
            games=SCH.parse_schedule((Path(asof)/"games.csv").read_bytes(),
                                     seasons={season})
            chosen=[]
            targets=[]
            for gid in ids:
                g=games.get(gid)
                if not g or g["kick"]!=kickoff:
                    raise ValueError("GAME_ABSENT_OR_REVISED_KICKOFF")
                chosen.append((gid,g))
                targets.extend([(season,week,g["home"]),(season,week,g["away"])])
            targets=sorted(set(targets))
            # Use the original immutable prefit; never regenerate future-
            # informed fitted coefficients from postgame stats or lines.
            prefit=OPS.prefit_gate(runner.root,season,week,cutoff,datetime.now(UTC))
            art=OPS.load_verified_prefit(runner.root,prefit)
            prep=LV.prepare(str(asof),targets,lambda D,U,sw:art,
                            runner.cfg,runner.consts,runner.cal,
                            runner.n_draws,runner.p1b_hyper,(season,week),
                            log=runner.log)
            contexts,asof_data=G.asof_context_with_data(asof,targets)
            defense=G.defensive_rates(asof,asof_data,cutoff,season)
            inj=FC._parse_csv(runner.store.get(
                setrec["files"][f"injuries_{season}.csv"]))
            roster_key=f"roster_weekly_{season}.csv"
            ros=FC._parse_csv(runner.store.get(setrec["files"][roster_key])
                            ) if roster_key in setrec["files"] else []
            cn=int(os.environ.get("NFL_PHASE1G_SIM_N","25000"))
            if cn<10000 or cn>int(prep.bundle.n_draws):
                raise ValueError("INVALID_PHASE1F_RESEARCH_DRAW_BUDGET")
            # Distinct identity if the research source changes, even if
            # the frozen Phase1 bundle and the CAS snapshot stay the same.
            own_digest=hashlib.sha256(
                Path(INT.__file__).read_bytes()+
                Path(G.__file__).read_bytes()+
                Path(__file__).read_bytes()).hexdigest()
            for gid,g in chosen:
                try:
                    if datetime.now(UTC)>=kickoff:
                        raise ValueError("AFTER_KICKOFF_NO_FORECAST")
                    gs={}
                    log=[]
                    for team in (g["home"],g["away"]):
                        base=prep.pack["games"].get((season,week,team))
                        if base is None:raise ValueError("NO_TEAM_PHASE1A_INPUT")
                        gs[team]=FC.apply_snapshot_rules(
                            base,horizon,inj,ros,season,week,team,log)
                    RN.resolve_duplicate_candidates(gs,
                        (g["home"],g["away"]),horizon,log)
                    view={"games":{(season,week,t):gs[t] for t in gs},
                          "meta":prep.pack["meta"]}
                    cpack,report=INT.make_challenger_pack(view,contexts,
                        season,week,(g["home"],g["away"]),horizon,cutoff)
                    # Optional pregame source receipts. Missing means no override.
                    personnel_path=None
                    pdir=os.environ.get("NFL_PHASE1G_PERSONNEL_DIR")
                    if pdir:
                        candidate=Path(pdir)/(gid+".json")
                        if candidate.is_file():personnel_path=candidate
                    personnel=G.apply_documented_personnel(cpack,
                        personnel_path,{**g,"horizon":horizon},
                        cutoff,setrec["retrieval_ts"])
                    role_evidence=G.apply_player_roles(
                        cpack,contexts,season,week,
                        (g["home"],g["away"]),horizon,cutoff)
                    wdir=os.environ.get("NFL_PHASE1G_WEATHER_DIR")
                    weather_path=None
                    if wdir:
                        candidate=Path(wdir)/(gid+".json")
                        if candidate.is_file():weather_path=candidate
                    weather,weather_receipt=G.load_weather_evidence(
                        weather_path,gid,g["kick"],cutoff,g["home"])
                    tds=G.build_context_team_tds(
                        contexts,(g["home"],g["away"]),season,week)
                    for team in (g["home"],g["away"]):
                        report[team]["expected_offensive_tds"]=tds[team]
                    eff,eff_receipts=G.matchup_event_efficiencies(
                        prep.bundle.eff,prep.bundle.idx,prep.bundle.C,
                        contexts,(g["home"],g["away"]),season,week,defense)
                    cv=G.model_identity(prep.bundle.model_version,
                        setrec["content_id"],own_digest,cn,
                        weather_receipt.get("sha256"),
                        personnel.get("source_sha256"))
                    seed=zlib.crc32(repr((cv,gid,horizon)).encode())
                    script=G.correlated_script(report,
                        (g["home"],g["away"]),cn,seed,cpack,weather=weather)
                    samples,cgs=SM.run_game(cpack,g["home"],g["away"],
                        season,week,eff,prep.bundle.idx,
                        prep.bundle.defaults,prep.bundle.C,cn,seed,
                        horizon,script,prep.bundle.qb_adjust)
                    if datetime.now(UTC)>=kickoff:
                        raise ValueError("AFTER_KICKOFF_NO_FORECAST")
                    cb=copy.copy(prep.bundle)
                    cb.model_version=cv
                    cb.n_draws=cn
                    prov={"snapshot_set_id":setrec["set_id"],
                        "content_id":setrec["content_id"],
                        "retrieval_ts":setrec["retrieval_ts"],
                        "cutoff":SCH.iso(cutoff),
                        "artifact_bundle_sha256":prefit["artifact_bundle_sha256"],
                        "schedule_revision":"SAME_SNAPSHOT_AS_PHASE1E",
                        "time_travel":False}
                    recs=FC.build_records(cb,prep.D,prep.names,
                        season,week,cgs,samples,horizon,kickoff,cutoff,
                        gid,prov,log)
                    for r in recs:
                        r["phase1g_evidence"]={
                            "version":G.VERSION,
                            "team_context":report[r["team"]],
                            "player_role_receipt":role_evidence[r["team"]],
                            "matchup_efficiency_receipt":eff_receipts[r["team"]],
                            "personnel_source_receipt":personnel,
                            "weather_source_receipt":weather_receipt,
                            "source_code_sha256":own_digest,
                            "data_limitations":{
                                "weather":"NWS_WHEN_SOURCED_ELSE_NO_CHANGE",
                                "individual_corner_coverage":"UNSOURCED",
                                "starting_qb":"DEPTH_PRIOR_UNCONFIRMED_UNLESS_SOURCE"},
                            "forecast_promoted":False}
                    saved=ST.Store(runner.root,"player_challenger_forecasts").append_batch(
                        f"{season}_wk{week:02d}_{horizon}_{gid}",
                        {"generated_at":datetime.now(UTC).isoformat(),
                         "run_id":run_id,"code_sha":FC.git_sha(),
                         "game_id":gid,"horizon":horizon,
                         "baseline_model_version":prep.bundle.model_version,
                         "research_model_version":cv},recs)
                    results[gid]={"game_id":gid,"horizon":horizon,
                        "status":STATUS_SAVED,"original_content_id":setrec["content_id"],
                        "model_version":cv,"draws":cn,
                        "n_player_markets":len(recs),"saved":saved}
                except Exception as e:
                    results[gid]={"game_id":gid,"horizon":horizon,
                        "status":"PHASE1G_RESEARCH_SAFE_EXPLICIT_FAILURE",
                        "reason":f"{type(e).__name__}: {e}"[:300]}
        except Exception as e:
            for gid in ids:
                results[gid]={"game_id":gid,"horizon":horizon,
                    "status":"PHASE1G_RESEARCH_SAFE_EXPLICIT_FAILURE",
                    "reason":f"{type(e).__name__}: {e}"[:300]}
    if results:
        CAS.Ledger(runner.root,name="phase1g_events.jsonl").append_many([
            {"at":datetime.now(UTC).isoformat(),**r} for r in results.values()])
    return results

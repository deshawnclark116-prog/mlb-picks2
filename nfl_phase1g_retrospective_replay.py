#!/usr/bin/env python3
"""Retrospective *counterfactual* original-snapshot NFL Phase1F numerical replay.

Important: Phase1F did not exist at these historical kickoffs. This script is
NOT an original pregame Phase1G forecast and CANNOT satisfy clean-forward
promotion gates. It re-simulates using the ORIGINAL immutable pregame CAS
snapshot and pregame prefit, against the same original Phase1E+V2 forecasts
and independent final player-game box scores. The historical result is used
only *after* simulation, for grading. Every missing source fails closed.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import sqlite3
import zlib
from collections import Counter,defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import nfl_phase1d_cas as CAS
import nfl_phase1d_schedule as SCH
import nfl_phase1e_ops as OPS
import nfl_phase1d_live as LV
import nfl_phase1d_runner as RN
import nfl_phase1_forecast as FC
import nfl_phase1_store as ST
import nfl_phase1c_sim as SM
import nfl_phase1f_shadow as F
import nfl_phase1f_intelligence as INT
import nfl_phase1g_player_intelligence as G

BASE_MARKETS={"rec_yds":"receiving_yards","rush_yds":"rushing_yards",
              "pass_yds":"passing_yards"}
UTC=timezone.utc

def restore_blob(root, sha):
    """Restore ONLY original scenario's required CAS bytes (no public refetch)."""
    dst=Path(root)/"cas"/"blobs"/sha[:2]/sha
    if dst.exists():
        if hashlib.sha256(dst.read_bytes()).hexdigest()!=sha:
            raise ValueError("CORRUPT_CAS_EXISTING")
        return
    src=Path(root)/"durable_blobs"/"cas"/sha
    manifest=json.loads((src/"manifest.json").read_text())
    if manifest["sha256"]!=sha:raise ValueError("BAD_ARCHIVED_BLOB_ID")
    pieces=[]
    for row in manifest["chunks"]:
        b=(src/row["name"]).read_bytes()
        if hashlib.sha256(b).hexdigest()!=row["sha256"]:
            raise ValueError("CORRUPT_ARCHIVED_CHUNK")
        pieces.append(gzip.decompress(b))
    blob=b"".join(pieces)
    if len(blob)!=manifest["bytes"] or hashlib.sha256(blob).hexdigest()!=sha:
        raise ValueError("CORRUPT_ARCHIVED_SOURCE")
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_bytes(blob)
    dst.chmod(0o444)


def get_originals(root, game_id, horizon):
    week=int(game_id.split("_")[1])
    file=Path(root)/"forecasts"/"batches"/f"2026_wk{week:02d}_{horizon}_{game_id}.jsonl"
    v2=Path(root)/"v2forecasts"/"batches"/file.name
    if not v2.exists():raise ValueError("MISSING_ORIGINAL_V2_COMPARATOR")
    meta, rows=ST.Store(root,"forecasts").read_batch(file)
    vmeta,vrows=ST.Store(root,"v2forecasts").read_batch(v2)
    if meta["game_id"]!=game_id or vmeta["game_id"]!=game_id:
        raise ValueError("ORIGINAL_WRONG_GAME")
    content_ids={r["input_snapshots"]["content_id"] for r in rows if r.get("outcome") in BASE_MARKETS}
    set_ids={r["input_snapshots"]["snapshot_set_id"] for r in rows if r.get("outcome") in BASE_MARKETS}
    if len(content_ids)!=1 or len(set_ids)!=1:raise ValueError("MULTIPLE_ORIGINAL_INPUT_SNAPSHOTS")
    setrec=CAS.get_set(CAS.Ledger(Path(root)/"cas"),set_ids.pop())
    if setrec["time_travel"] is not False:
        raise ValueError("ORIGINAL_WAS_NOT_REAL_CAPTURE")
    if setrec["content_id"]!=content_ids.pop():
        raise ValueError("INCONSISTENT_ORIGINAL_SOURCE_ID")
    if CAS.parse_iso(setrec["retrieval_ts"])>CAS.parse_iso(setrec["cutoff"]):
        raise ValueError("ORIGINAL_SOURCE_LATE")
    if datetime.fromisoformat(meta["generated_at"].replace("Z","+00:00"))>=CAS.parse_iso(setrec["kickoff"]):
        raise ValueError("ORIGINAL_FORECAST_AFTER_KICKOFF")
    return setrec,meta,rows,vrows


def replay(root, db_path, game_id, horizon, n=10000):
    if n<10000:raise ValueError("MINIMUM_RESEARCH_DRAWS_10000")
    root=Path(root)
    original,base_header,baseline,v2=get_originals(root,game_id,horizon)
    for i,sha in enumerate(sorted(set(original["files"].values())),1):
        restore_blob(root,sha)
        if i%10==0:print("RESTORED_CAS_FILES",i,flush=True)
    store=CAS.BlobStore(root/"cas")
    bad=CAS.verify_set(store,CAS.Ledger(root/"cas"),original)
    if bad:raise ValueError(f"ORIGINAL_CAS_INVALID {bad[:3]}")
    source_path=CAS.materialize(store,original,root/"tmp"/
                                ("offline_replay_"+original["content_id"][:16]))
    schedule=SCH.parse_schedule((Path(source_path)/"games.csv").read_bytes(),seasons={2026})
    game=schedule[game_id];season,week=game["season"],game["week"]
    cutoff=SCH.forecast_cutoff(game["kick"],horizon)
    if CAS.iso(cutoff)!=original["cutoff"]:
        raise ValueError("CHANGED_ORIGINAL_KICKOFF_OR_CUTOFF")
    teams=(game["home"],game["away"])
    targets=[(season,week,t) for t in teams]
    prefit=OPS.prefit_gate(root,season,week,cutoff,datetime.now(UTC))
    art=OPS.load_verified_prefit(root,prefit)
    cfg,const,cal=RN.load_config()
    print("PREPARING_ARCHIVED_ASOF",game_id,horizon,"n",n,
          "prefit",prefit["artifact_bundle_sha256"][:14],flush=True)
    prep=LV.prepare(str(source_path),targets,lambda D,U,sw:art,
                    cfg,const,cal,n,LV.load_p1b_hyper(),
                    (season,week),log=lambda *x:print("REPLAY_FIT",*x,flush=True))
    contexts,asof_data=G.asof_context_with_data(source_path,targets)
    defense=G.defensive_rates(source_path,asof_data,cutoff,season)
    inj=FC._parse_csv(store.get(original["files"][f"injuries_{season}.csv"]))
    roster_name=f"roster_weekly_{season}.csv"
    ros=FC._parse_csv(store.get(original["files"][roster_name])
                    ) if roster_name in original["files"] else []
    gs={};rule_log=[]
    for team in teams:
        base=prep.pack["games"].get((season,week,team))
        if base is None:raise ValueError("NO_ARCHIVED_PHASE1A_GAME_INPUT")
        gs[team]=FC.apply_snapshot_rules(base,horizon,inj,ros,
                                         season,week,team,rule_log)
    RN.resolve_duplicate_candidates(gs,teams,horizon,rule_log)
    pack={"games":{(season,week,t):gs[t] for t in teams},"meta":prep.pack["meta"]}
    print("BASELINE_REPLAY_START",flush=True)
    seed=zlib.crc32(repr(("CONTROL",game_id,horizon,original["content_id"])).encode())
    output_base,bg=SM.run_game(pack,*teams,season,week,
                     prep.bundle.eff,prep.bundle.idx,prep.bundle.defaults,
                     prep.bundle.C,n,seed,horizon,None,prep.bundle.qb_adjust)
    control=FC.build_records(prep.bundle,prep.D,prep.names,season,week,
        bg,output_base,horizon,game["kick"],cutoff,game_id,
        {"content_id":original["content_id"],"time_travel":False,
         "retrospective_replay":True},rule_log)
    cpack,ctx=INT.make_challenger_pack(pack,contexts,season,week,
                                      teams,horizon,cutoff)
    script=INT.joint_scripts(ctx,teams,n,seed,cpack)
    print("CHALLENGER_REPLAY_START",flush=True)
    output_f,fg=SM.run_game(cpack,*teams,season,week,
        prep.bundle.eff,prep.bundle.idx,prep.bundle.defaults,
        prep.bundle.C,n,seed,horizon,script,prep.bundle.qb_adjust)
    cb=copy.copy(prep.bundle);cb.model_version=INT.challenger_identity(
        prep.bundle.model_version,original["content_id"],
        {"retrospective":True,"n":n})
    challenger=FC.build_records(cb,prep.D,prep.names,season,week,
        fg,output_f,horizon,game["kick"],cutoff,game_id,
        {"content_id":original["content_id"],"time_travel":False,
         "retrospective_replay":True},rule_log)
    # Independently re-run Phase1G using the ORIGINAL 2026 pregame source.
    # This is retrospective: not a genuine Phase1G original pregame record.
    gpack,gctx=INT.make_challenger_pack(pack,contexts,season,week,
                                      teams,horizon,cutoff)
    role=G.apply_player_roles(gpack,contexts,season,week,teams,horizon,cutoff)
    geff,eff_receipts=G.matchup_event_efficiencies(
        prep.bundle.eff,prep.bundle.idx,prep.bundle.C,
        contexts,teams,season,week,defense)
    td=G.build_context_team_tds(contexts,teams,season,week)
    for tm in teams:gctx[tm]["expected_offensive_tds"]=td[tm]
    gscript=G.correlated_script(gctx,teams,n,seed,gpack,weather=None)
    print("PHASE1G_PERSONNEL_ROLE_MATCHUP_REPLAY_START",flush=True)
    output_g,gg=SM.run_game(gpack,*teams,season,week,
        geff,prep.bundle.idx,prep.bundle.defaults,
        prep.bundle.C,n,seed,horizon,gscript,prep.bundle.qb_adjust)
    gb=copy.copy(prep.bundle)
    gb.model_version=G.model_identity(
        prep.bundle.model_version,original["content_id"],
        hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest(),
        n)
    candidate_g=FC.build_records(gb,prep.D,prep.names,season,week,
        gg,output_g,horizon,game["kick"],cutoff,game_id,
        {"content_id":original["content_id"],"time_travel":False,
         "retrospective_replay":True},rule_log)
    by={model:{(r["team"],str(r["player_id"]),r["outcome"]):r for r in recs
       if r.get("outcome") in BASE_MARKETS}
       for model,recs in (("phase1g_retrospective",candidate_g),
                          ("phase1f_retrospective",challenger),
                          ("phase1e_same_draw_control",control),
                          ("phase1e_original_pregame",baseline),
                          ("v2_original_pregame",v2))}
    scored=[];ex=Counter()
    with sqlite3.connect(f"file:{Path(db_path).absolute()}?mode=ro",uri=True) as con:
        for key in sorted(set().union(*[set(x) for x in by.values()])):
            pair={model:d.get(key) for model,d in by.items()}
            if any(pair.get(x) is None for x in (
                "phase1g_retrospective","phase1f_retrospective",
                "phase1e_same_draw_control","phase1e_original_pregame")):
                ex["not_in_all_three_event_models"]+=1
                continue
            if pair["v2_original_pregame"] is not None and pair["v2_original_pregame"].get("eligibility")!="eligible":
                pair["v2_original_pregame"]=None
            team,pid,market=key
            truth=con.execute(f"SELECT {BASE_MARKETS[market]} FROM player_games "
                "WHERE game_id=? AND team=? AND player_id=? AND season_type='REG'",
                (game_id,team,pid)).fetchall()
            if len(truth)!=1 or truth[0][0] is None:
                ex["missing_box_score_unresolved_not_zero"]+=1
                continue
            actual=float(truth[0][0])
            row={"game_id":game_id,"market":market,"player_id":pid,
                 "team":team,"name":pair["phase1e_original_pregame"].get("player_name"),
                 "actual":actual,"v2_available":pair["v2_original_pregame"] is not None}
            for model,r in pair.items():
                if model=="v2_original_pregame":
                    point=(r["v2_frozen_comparator"]["point"]
                           if r is not None else None)
                else:point=r["median"]
                if point is not None and not math.isfinite(float(point)):
                    raise ValueError("NONFINITE_FORECAST")
                row[model]=None if point is None else float(point)
            scored.append(row)
    def metrics(items,allv2=False):
        if allv2:items=[x for x in items if x["v2_available"]]
        if not items:return {"n":0}
        def metric(model):
            e=[abs(x[model]-x["actual"]) for x in items]
            return {"mae_yards":round(sum(e)/len(e),3),
                    "within_10pct":round(sum(x<=10 for x in e)/len(e),4),
                    "within_20pct":round(sum(x<=20 for x in e)/len(e),4),
                    "signed_bias":round(sum(x[model]-x["actual"] for x in items)/len(items),3)}
        models=["phase1g_retrospective","phase1f_retrospective","phase1e_same_draw_control",
                "phase1e_original_pregame"]
        if allv2:models.append("v2_original_pregame")
        return {"n":len(items),
                "models":{m:metric(m) for m in models},
                "phase1g_median_mae_improvement_vs_same_draw":round(
                  metric("phase1e_same_draw_control")["mae_yards"]-
                  metric("phase1g_retrospective")["mae_yards"],3),
                "phase1g_closer_vs_control":sum(
                    abs(x["phase1g_retrospective"]-x["actual"])<
                    abs(x["phase1e_same_draw_control"]-x["actual"]) for x in items),
                "phase1f_median_mae_improvement_vs_same_draw":round(
                  metric("phase1e_same_draw_control")["mae_yards"]-
                  metric("phase1f_retrospective")["mae_yards"],3),
                "phase1f_closer_vs_control":sum(
                    abs(x["phase1f_retrospective"]-x["actual"])<
                    abs(x["phase1e_same_draw_control"]-x["actual"]) for x in items)}
    return {"status":"RETROSPECTIVE_COUNTERFACTUAL_NOT_ORIGINAL_PREGAME_PHASE1G",
        "not_prospective":True,"no_model_promotion_allowed":True,
        "game_id":game_id,"horizon":horizon,"draws_each_replay":n,
        "original_snapshot_set_id":original["set_id"],
        "original_pregame_cas_content_id":original["content_id"],
        "original_retrieval_utc":original["retrieval_ts"],
        "original_champion_generated_at":base_header["generated_at"],
        "original_prefit_sha256":prefit["artifact_bundle_sha256"],
        "comparability":"Same as-of content, frozen prefit, players and actuals; replayed both engines at identical N. Original Phase1E used its original N.",
        "full_event_model_cohort":metrics(scored),
        "all_five_model_matched":metrics(scored,True),
        "pregame_50plus_cohort":metrics([
            x for x in scored if max(x["phase1e_original_pregame"],
                                     x["phase1f_retrospective"],x["phase1g_retrospective"])>=50]),
        "by_market":{k:metrics([x for x in scored if x["market"]==k])
                     for k in BASE_MARKETS},
        "exclusions":dict(ex),
        "team_context_phase1f":ctx,
        "team_context_phase1g":gctx,
        "role_changes":role,
        "matchup_efficiency_changes":eff_receipts,
        "all_player_comparisons":scored}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    ap.add_argument("--db",required=True)
    ap.add_argument("--game",default="2026_09_BUF_MIN")
    ap.add_argument("--horizon",choices=("T24","T90"),default="T90")
    ap.add_argument("--n",type=int,default=10000)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    r=replay(args.state,args.db,args.game,args.horizon,args.n)
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(r,indent=2,default=str)+"\n")
    for key in ("full_event_model_cohort","all_five_model_matched",
                "pregame_50plus_cohort","by_market"):
        print("REPLAY_GRADED",key,json.dumps(r[key]),flush=True)
    for row in r["all_player_comparisons"]:
        print("REPLAY_PLAYER",json.dumps({k:row[k] for k in (
            "name","team","market","actual","phase1g_retrospective",
            "phase1f_retrospective",
            "phase1e_same_draw_control","phase1e_original_pregame",
            "v2_original_pregame")}),flush=True)

if __name__=="__main__":main()

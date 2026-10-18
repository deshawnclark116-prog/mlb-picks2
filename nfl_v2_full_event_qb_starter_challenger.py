#!/usr/bin/env python3
"""Unpromoted, pregame, source-conditioned FULL-EVENT QB role challenger.

Unlike dividing E[y] by P(active), this preserves the frozen simulator's
team-volume NB, opponent context, QB efficiency, event-by-event coupling
with receivers/defenders and red-zone logic. Change ONLY the QB's verified
starting-role opportunity inputs for ONE source-proven matchup, in a deep
copy of the verified original game-state. The frozen source model, its
immutable T24/T90 decisions, fit and input blobs are NEVER overwritten.

Scenario is generated when this script actually runs, with its real
generation timestamp, NEVER retroactively labeled a T24/T90 prediction.
Starter plays the full game in this scenario, not a claim of guaranteed
playing time. No sportsbook recommendations or calibrated win probabilities.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import math
import tempfile
import zlib
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

import nfl_v2_verified_qb_conditional_report as Q
import nfl_v2_role_evidence_registry as REG
import nfl_v2_depthchart_role_ingest as DEPTH

SCHEMA="NFL_V2_SOURCE_VERIFIED_STARTING_QB_FULL_EVENT_CHALLENGER_V1"
METRICS=("pass_yds","pass_td","int","rush_yds","rec_yds","rec")
RESEARCH_DRAW_COUNT=25000


def restore_original_source_blobs(state, setrec):
    """Rehydrate exactly this pre-cutoff snapshot's immutable CAS bytes.

    GitHub cache is OPTIONAL. Durable blobs/chunks are append-only on the
    original shadow-state branch and individually hash-verified before use.
    Never query today's provider or take a new snapshot to reconstruct an
    old forecast.
    """
    from nfl_shadow_durability import atomic_new,digest
    root=Path(state)
    required=set(setrec["files"].values())
    restored=0
    for sha in sorted(required):
        if not isinstance(sha,str) or len(sha)!=64:
            raise ValueError("MALFORMED_ORIGINAL_SOURCE_HASH")
        dst=root/"cas"/"blobs"/sha[:2]/sha
        if dst.exists():
            if digest(dst.read_bytes())!=sha:
                raise ValueError("ORIGINAL_BLOB_CACHE_HASH_MISMATCH")
            continue
        manifest=root/"durable_blobs"/"cas"/sha/"manifest.json"
        if not manifest.is_file():
            raise ValueError("ORIGINAL_PRE_CUTOFF_BLOB_NOT_DURABLY_STORED_"+sha[:8])
        meta=json.loads(manifest.read_text())
        if meta.get("sha256")!=sha or not isinstance(meta.get("chunks"),list):
            raise ValueError("DURABLE_ORIGINAL_SOURCE_MANIFEST_INVALID")
        parts=[]
        for chunk in meta["chunks"]:
            name=chunk.get("name")
            if not isinstance(name,str) or Path(name).name!=name:
                raise ValueError("DURABLE_SOURCE_CHUNK_PATH_INVALID")
            zipped=(manifest.parent/name).read_bytes()
            if digest(zipped)!=chunk.get("sha256"):
                raise ValueError("ORIGINAL_SOURCE_CHUNK_HASH_MISMATCH")
            parts.append(gzip.decompress(zipped))
        content=b"".join(parts)
        if len(content)!=meta.get("bytes") or digest(content)!=sha:
            raise ValueError("RECONSTRUCTED_ORIGINAL_SOURCE_SHA256_MISMATCH")
        atomic_new(dst,content)
        dst.chmod(0o444)
        restored+=1
    return {"original_pre_cutoff_blob_count":len(required),
            "rehydrated_from_durable_original_state":restored,
            "all_original_blob_sha256_verified":True}


def role_override_games(gs, *,team,starter_id,qb_ids,horizon):
    """Return a copy, condition QB full-game starter using PRE-GAME roster news.

    Allocation is not a new fit: starter receives the highest incumbent
    pregame QB-att propensity. This is the original role prior, not the
    former starter's talent. Opponent/teammate volume and efficiency
    distributions are still the frozen system's fitted ones.
    """
    if horizon not in ("T24","T90"):
        raise ValueError("UNSUPPORTED_SCENARIO_HORIZON")
    if team not in gs:
        raise ValueError("TARGET_TEAM_ABSENT")
    out=copy.deepcopy(gs)
    qt=out[team]["types"]["qb_att"]
    ids=list(qt["ids"])
    if set(ids)!=set(qb_ids) or ids.count(starter_id)!=1 or len(set(ids))!=len(ids):
        raise ValueError("STARTER_NOT_EXACT_QB_ID_OR_AMBIGUOUS")
    if len(ids)<2:
        raise ValueError("ROLE_TRANSFER_HAS_NO_COMPETING_QUARTERBACK")
    original=np.asarray(qt["P1"],dtype=float).copy()
    if not np.isfinite(original).all() or (original<0).any() or original.sum()<=0:
        raise ValueError("INVALID_FROZEN_QB_ROLE_PROPENSITIES")
    incumbent=float(original.max())
    if incumbent<=0 or incumbent>1:
        raise ValueError("INVALID_INCUMBENT_QB_ROLE_PRIOR")
    j=ids.index(starter_id)
    before=np.asarray(qt["pact24" if horizon=="T24" else "pact90"],float).copy()
    if not np.isfinite(before).all() or (before<0).any() or (before>1).any():
        raise ValueError("INVALID_ORIGINAL_QB_PLAY_PROBS")
    # Explicit scenario: named verified starter plays the game; all other
    # named QBs don't. A player could be injured midgame; this is NOT an
    # unconditional real-world expected-value claim.
    qb_role=np.zeros_like(original);qb_role[j]=incumbent
    qt["P1"]=qb_role
    overrides=0
    for typename,t in out[team]["types"].items():
        if typename in ("qb_att","carry","rz_carry"):
            cids=list(t["ids"])
            for key in ("pact24","pact90"):
                if key not in t: continue
                p=np.asarray(t[key],dtype=float).copy()
                for k,pid in enumerate(cids):
                    if pid in qb_ids:
                        p[k]=1.0 if pid==starter_id else 0.0
                        overrides+=1
                t[key]=p
    return out,{
        "team":team,"horizon":horizon,
        "starter_player_id":starter_id,
        "original_starter_play_probability":round(float(before[j]),6),
        "original_qb_role_propensities":{str(i):round(float(x),6) for i,x in zip(ids,original)},
        "scenario_qb_role_propensities":{str(i):round(float(x),6) for i,x in zip(ids,qb_role)},
        "position_prior_source":"highest pre-cutoff incumbent QB propensity from frozen Phase 1A",
        "role_probability_scenario":"named starter active every draw, other named QBs inactive; frozen outside-QB bucket retained",
        "changed_role_input_cells":overrides,
        "team_expected_qb_dropbacks_not_manually_modified":True,
        "QB_efficiency_models_not_substituted":True,
    }


def condition_rb_availability(gs, *, team, out_ids, conditional_active_ids, horizon):
    """Pure research intervention, NOT assertion of confirmed game-day active status.

    Officially ruled-out RBs cannot take carries/targets. For the named
    HEALTHY but not yet dressed-confirmed backs, evaluate a *conditional*
    scenario where they are active. Keep the frozen team total and original
    carry/target propensities; the simulator's outside bucket is preserved
    for call-ups not represented in the original source snapshot.
    """
    if horizon not in ("T24", "T90") or team not in gs:
        raise ValueError("INVALID_RUNNING_BACK_INTERVENTION_CONTEXT")
    off=set(out_ids);on=set(conditional_active_ids)
    if off & on or not off or not on:
        raise ValueError("OVERLAPPING_OR_EMPTY_RUNNING_BACK_STATUS")
    out=copy.deepcopy(gs)
    carry_ids=set(out[team]["types"]["carry"]["ids"])
    if not off <= carry_ids or not on <= carry_ids:
        raise ValueError("OFFICIAL_RUNNING_BACK_NOT_UNIQUELY_IN_SNAPSHOT")
    original={};after={};changed=0
    for typ,t in out[team]["types"].items():
        if typ not in ("carry","rz_carry","target","rz_target"):
            continue
        ids=list(t["ids"])
        for k in ("pact24","pact90"):
            if k not in t:continue
            x=np.asarray(t[k],dtype=float).copy()
            if not np.isfinite(x).all() or (x<0).any() or (x>1).any():
                raise ValueError("INVALID_FROZEN_RB_ACTIVE_PROBABILITIES")
            for j,pid in enumerate(ids):
                if pid in off | on:
                    original[(typ,k,pid)]=round(float(x[j]),6)
                    x[j]=0.0 if pid in off else 1.0
                    after[(typ,k,pid)]=round(float(x[j]),6)
                    changed+=1
            t[k]=x
    return out,{
        "team":team,"horizon":horizon,
        "definitely_out_rb_ids":sorted(off),
        "conditional_active_rb_ids":sorted(on),
        "modified_input_cells":changed,
        "team_total_plays_not_overwritten":True,
        "original_rushing_target_role_shares_unchanged":True,
        "unmodeled_new_callups_handled_only_by_original_outside_bucket":True,
        "not_game_day_dressed_certification":True,
        "before":{"/".join(k):v for k,v in original.items()},
        "after":{"/".join(k):v for k,v in after.items()}
    }


def exact_named_rbs(prep,gs,season,week,team,names):
    ids=list(gs[team]["types"]["carry"]["ids"])
    matched={}
    for name in names:
        found=[gid for gid in ids if Q.normalized(
            prep.names.get((season,week,gid)))==Q.normalized(name)]
        if len(found)!=1:
            raise ValueError("RUNNING_BACK_ID_MISSING_OR_AMBIGUOUS_"+Q.normalized(name))
        matched[name]=found[0]
    if len(set(matched.values()))!=len(matched):
        raise ValueError("MULTIPLE_RB_NAMES_REFER_TO_SAME_PLAYER")
    return matched


def verify_rb_news(evidence, *,now,game_id,kickoff):
    if not isinstance(evidence,dict) or evidence.get("schema")!=(
        "NFL_V2_OFFICIAL_EAGLES_RUNNING_BACK_AVAILABILITY_V1"):
        raise ValueError("RUNNING_BACK_NEWS_NOT_A_VERIFIED_EVIDENCE_DOCUMENT")
    if evidence.get("game_id")!=game_id or evidence.get("team")!="PHI":
        raise ValueError("RUNNING_BACK_EVIDENCE_WRONG_GAME")
    if not all(evidence.get(k) for k in ("ruled_out","conditional_healthy","new_unmodeled_additions","official_sources")):
        raise ValueError("RUNNING_BACK_EVIDENCE_INCOMPLETE")
    for s in evidence["official_sources"]:
        if not str(s.get("url","")).startswith("https://www.philadelphiaeagles.com/news/"):
            raise ValueError("RUNNING_BACK_SOURCE_NOT_OFFICIAL_EAGLES")
        try:
            pub=datetime.fromisoformat(str(s.get("published_at_utc","")).replace("Z","+00:00"))
            if pub.tzinfo is None:
                raise ValueError("SOURCE_TIMESTAMP_MISSING_TIMEZONE")
            pub=pub.astimezone(timezone.utc)
        except (TypeError,ValueError) as exc:
            raise ValueError("RUNNING_BACK_SOURCE_TIMESTAMP_NOT_VALID") from exc
        if not pub<=now<kickoff:
            raise ValueError("RUNNING_BACK_SOURCE_NOT_KNOWN_PREGAME")
    return evidence


def unique_player_ids(prep,gs,season,week,team,starter_name):
    qbs=list(gs[team]["types"]["qb_att"]["ids"])
    named=[gid for gid in qbs if Q.normalized(prep.names.get((season,week,gid)))==
           Q.normalized(starter_name)]
    if len(named)!=1:
        raise ValueError("VERIFIED_STARTER_NOT_UNIQUE_QB_IN_SNAPSHOTTED_GAME")
    return named[0],qbs


def brief_recs(records,team):
    out={}
    for r in records:
        if r["team"]!=team or r["outcome"] not in METRICS:
            continue
        k=(r["player_name"],r["outcome"])
        if k in out:
            raise ValueError("DUPLICATE_FORECAST_PLAYER_OUTCOME_IN_SIMULATION")
        out[k]={"mean":float(r["mean"]),"median":float(r["median"]),
                "p10":float(r["quantiles"]["p10"]),
                "p90":float(r["quantiles"]["p90"])}
    return out


def compare_records(baseline,conditioned,starter,*,team="PHI",top=25):
    original=brief_recs(baseline,team)
    scenario=brief_recs(conditioned,team)
    rows=[]
    for key,a in original.items():
        if key not in scenario:continue
        b=scenario[key]
        rows.append({"player":key[0],"outcome":key[1],
            "original_model_resimulation":{k:round(float(v),2) for k,v in a.items()},
            "verified_starter_role_scenario":{k:round(float(v),2) for k,v in b.items()},
            "mean_change":round(b["mean"]-a["mean"],2)})
    rows.sort(key=lambda r:(r["player"]!=starter.lower(),
                            r["outcome"]!="pass_yds",
                            -abs(r["mean_change"])))
    return rows[:top]


def run(state,board,evidence,now,n_draws,horizons=("T24",),rb_evidence=None,depth_snapshot=None,expected_starter_news=None):
    if now.tzinfo is None:
        raise ValueError("NAIVE_RUN_CLOCK")
    if n_draws<1000 or n_draws>100000:
        raise ValueError("UNSAFE_SIMULATION_DRAW_COUNT")
    official=REG.targets(board,evidence,now=now,horizons=horizons)
    provisional=(DEPTH.provisional_targets(board,depth_snapshot,now=now,
                 horizons=horizons) if depth_snapshot is not None else [])
    already={(x["game_id"],x["team"],x["horizon"]) for x in official}
    expected=(DEPTH.expected_report_targets(
        board,depth_snapshot,expected_starter_news,now=now,horizons=horizons)
        if expected_starter_news is not None else [])
    for item in expected:
        if (item["game_id"],item["team"],item["horizon"]) in already:
            raise ValueError("EXPECTED_REPORT_OVERLAPS_CONFIRMED_STARTER")
    already.update((x["game_id"],x["team"],x["horizon"]) for x in expected)
    targets=official+expected+[x for x in provisional
                      if (x["game_id"],x["team"],x["horizon"]) not in already]
    if not targets:
        raise ValueError("NO_EVIDENCE_SUPPORTED_QB_ROLE_SCENARIO_IN_PUBLISHED_GAMES")
    import nfl_phase1d_runner as RN
    import nfl_phase1d_live as LV
    import nfl_phase1e_live as LIVE
    import nfl_phase1d_cas as CAS
    import nfl_phase1_forecast as FC
    import nfl_phase1c_sim as SM
    import nfl_phase1d_schedule as SCH

    runner=LIVE.LiveRunner(str(state),n_draws=n_draws,log=lambda msg:print("LIVE_SOURCE_ONLY",msg,flush=True))
    results=[]
    for entry in targets:
        hz=entry["horizon"];season=entry["season"];week=entry["week"]
        team=entry["team"];opponent=entry["opponent"]
        kickoff=SCH.parse_iso(entry["kickoff_utc"]);cutoff=SCH.parse_iso(entry["original_forecast_cutoff_utc"])
        if now>=kickoff or SCH.forecast_cutoff(kickoff,hz)!=cutoff:
            raise ValueError("KICKOFF_OR_CUTOFF_CHANGED")
        group=runner.group_name(season,week,hz,kickoff)
        sets=[x for x in runner.ledger.read() if x.get("type")=="set" and x.get("group")==group]
        if len(sets)!=1:
            raise ValueError("NO_ONE_UNAMBIGUOUS_ORIGINAL_LIVE_SNAPSHOT_SET")
        setrec=sets[0]
        if SCH.parse_iso(setrec["retrieval_ts"])>cutoff or setrec.get("time_travel"):
            raise ValueError("SNAPSHOT_LATE_OR_HISTORICALLY_RECONSTRUCTED")
        restored=restore_original_source_blobs(state,setrec)
        problems=CAS.verify_set(runner.store,runner.ledger,setrec)
        if problems:
            raise ValueError("SNAPSHOT_SHA256_VALIDATION_FAILED "+repr(problems[:2]))
        runner.current_cutoff=cutoff
        with tempfile.TemporaryDirectory(prefix="nfl_qb_role_challenger_") as td:
            root=Path(td)
            asof=CAS.materialize(runner.store,setrec,root/"asof")
            def get_art(D,U,sw):
                return runner.weekly_artifacts(D,U,sw,setrec["content_id"])
            prep=LV.prepare(str(asof),[(season,week,team),(season,week,opponent)],
                            get_art,runner.cfg,runner.consts,runner.cal,n_draws,
                            runner.p1b_hyper,(season,week),log=lambda m:print("CHALLENGER_PREPARE",m,flush=True))
            inj=FC._parse_csv(runner.store.get(setrec["files"]["injuries_2026.csv"]))
            rosfile="roster_weekly_2026.csv"
            ros=FC._parse_csv(runner.store.get(setrec["files"][rosfile])) if rosfile in setrec["files"] else []
            gs={};log=[]
            for tm in (team,opponent):
                gi=prep.pack["games"].get((season,week,tm))
                if gi is None:
                    raise ValueError("NO_TEAM_OPPORTUNITY_INPUTS_"+tm)
                gs[tm]=FC.apply_snapshot_rules(gi,hz,inj,ros,season,week,tm,log)
            RN.resolve_duplicate_candidates(gs,[team,opponent],hz,log)
            starter_id,qb_ids=unique_player_ids(prep,gs,season,week,team,
                                                  entry["official_verified_starter"])
            scenario_gs,role=role_override_games(gs,team=team,starter_id=starter_id,
                                                    qb_ids=qb_ids,horizon=hz)
            rb_intervention=None
            if rb_evidence is not None and rb_evidence.get("game_id")==entry["game_id"] and rb_evidence.get("team")==team and entry.get("source_tier") is None:
                news=verify_rb_news(rb_evidence,now=now,game_id=entry["game_id"],
                                    kickoff=kickoff)
                found=exact_named_rbs(prep,scenario_gs,season,week,team,
                    news["ruled_out"]+news["conditional_healthy"])
                scenario_gs,rb_intervention=condition_rb_availability(
                    scenario_gs,team=team,
                    out_ids=[found[n] for n in news["ruled_out"]],
                    conditional_active_ids=[found[n] for n in news["conditional_healthy"]],
                    horizon=hz)
                rb_intervention["source_urls"]=[v["url"] for v in news["official_sources"]]
                rb_intervention["new_roster_additions_not_present_in_old_T24_snapshot"]=news["new_unmodeled_additions"]
            source_sha=hashlib.sha256(json.dumps(
                {"qb":evidence,"rb":rb_evidence,"depth":depth_snapshot,"expected":expected_starter_news},sort_keys=True,
                separators=(",",":")).encode()).hexdigest()
            # Identical engine draws for control and challenger; only QB
            # availability/propensities change, not fitted model or data.
            seed=zlib.crc32(repr((prep.bundle.model_version,entry["game_id"],hz)).encode())
            def simulate_once(players):
                view={"games":{(season,week,tm):players[tm] for tm in players},
                      "meta":prep.pack["meta"]}
                sims,gm=SM.run_game(view,team,opponent,season,week,prep.bundle.eff,
                   prep.bundle.idx,prep.bundle.defaults,prep.bundle.C,n_draws,seed,
                   hz,None,prep.bundle.qb_adjust)
                prov={"snapshot_set_id":setrec["set_id"],
                      "content_id":setrec["content_id"],
                      "retrieval_ts":setrec["retrieval_ts"],
                      "cutoff":entry["original_forecast_cutoff_utc"],
                      "artifact_bundle_sha256":prep.art_manifest["bundle_sha256"],
                      "time_travel":False}
                return FC.build_records(prep.bundle,prep.D,prep.names,season,week,gm,
                   sims,hz,kickoff,cutoff,entry["game_id"],prov,log)
            baseline=simulate_once(gs)
            challenger=simulate_once(scenario_gs)
            comparison=compare_records(baseline,challenger,
                                       entry["official_verified_starter"],team=team)
            m=[x for x in comparison if
               Q.normalized(x["player"])==Q.normalized(entry["official_verified_starter"]) and
               x["outcome"]=="pass_yds"]
            if len(m)!=1:
                raise ValueError("STARTER_PASSING_YARDS_NOT_IN_RESEARCH_GAME_RESULT")
            if m[0]["verified_starter_role_scenario"]["mean"]<=0:
                raise ValueError("NO_STARTING_QUARTERBACK_PASSING_PRODUCTION")
            results.append({
                "game_id":entry["game_id"],"team":team,"horizon":hz,
                "opponent":opponent,
                "starter_is_officially_source_verified":entry.get("source_tier") is None,
                "scenario_evidence_tier":entry.get("source_tier","OFFICIAL_TEAM_STARTER_SOURCE"),
                "source_depthchart_url":entry.get("provisional_depthchart_source"),
                "source_reported_expected_start_url":entry.get("reported_expectation_source"),
                "source_was_known_at_original_cutoff":entry["source_was_known_at_original_cutoff"],
                "latest_official_source_utc":entry["latest_official_source_utc"],
                "never_a_replacement_for_original_frozen_forecast":True,
                "verified_starter":entry["official_verified_starter"],
                "start_status_sources":entry["official_sources"],
                "original_retrieval_utc":setrec["retrieval_ts"],
                "original_snapshot_set_id":setrec["set_id"],
                "original_input_content_id":setrec["content_id"],
                "original_bundle_version":prep.bundle.model_version,
                "original_source_restore_proof":restored,
                "scenario_model_version":"SOURCE_STARTER_FULL_EVENT_V1_"+source_sha[:16],
                "scenario_fitted_with_post_cutoff_stats":False,
                "scenario_generated_after_original_cutoff":now>cutoff,
                "n_draws":n_draws,"seed_same_control_and_scenario":seed,
                "conditional_if_named_qb_starts":True,
                "do_not_treat_as_unconditional_or_as_certified_bookmaker_pick":True,
                "role_intervention":role,
                "running_back_role_intervention":rb_intervention,
                "player_impact":comparison,
                "verified_original_forecast_unchanged":True
            })
    return {"schema":SCHEMA,"generated_at_utc":now.isoformat(),
       "status":"PREGAME_FULL_GAME_RESEARCH_CHALLENGER_NOT_A_VALIDATED_BETTING_MODEL",
       "t24_or_t90_label_not_reused":True,
       "not_official_original_forecast":True,
       "no_money_line_or_player_prop_bets":True,
       "source_evidence_sha256":hashlib.sha256(json.dumps({"official":evidence,"depth":depth_snapshot,"expected":expected_starter_news},
                         sort_keys=True).encode()).hexdigest(),
       "conditional_unverified_depthchart_role_scenarios":sum(
           not x["starter_is_officially_source_verified"] for x in results),
       "source_verified_official_team_role_scenarios":sum(
           x["starter_is_officially_source_verified"] for x in results),
       "scientific_note":"Conditioned full-game-start scenario, using same frozen event simulator and fit. "+
         "Role propensity from highest team QB prior. No claim of earned forward calibration "+
         "or probability of starter playing full game.",
       "scenarios":results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--board",type=Path,required=True)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--rb-evidence",type=Path)
    p.add_argument("--depthchart",type=Path)
    p.add_argument("--expected-starter-news",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--draws",type=int,default=RESEARCH_DRAW_COUNT)
    p.add_argument("--horizon",choices=("T24","T90","both"),default="T24")
    a=p.parse_args()
    now=datetime.now(timezone.utc)
    b=json.loads(a.board.read_text());e=json.loads(a.evidence.read_text())
    rb=json.loads(a.rb_evidence.read_text()) if a.rb_evidence else None
    dep=json.loads(a.depthchart.read_text()) if a.depthchart else None
    exp=json.loads(a.expected_starter_news.read_text()) if a.expected_starter_news else None
    result=run(a.state,b,e,now,a.draws,
               horizons=("T24","T90") if a.horizon=="both" else (a.horizon,),
               rb_evidence=rb,depth_snapshot=dep,expected_starter_news=exp)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    for r in result["scenarios"]:
        rows=[x for x in r["player_impact"] if x["outcome"]=="pass_yds" and
              Q.normalized(x["player"])==Q.normalized(r["verified_starter"])]
        print("NFL_QB_STARTER_FULL_EVENT_SCENARIO",json.dumps({
          "game":r["game_id"],"horizon":r["horizon"],"starter":r["verified_starter"],
          "starter_pass_yards":rows,"n_draws":r["n_draws"],
          "source_code_version":r["scenario_model_version"]}))

if __name__=="__main__":
    main()

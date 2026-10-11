#!/usr/bin/env python3
"""Phase1I-A: analytic target-depth / QB-receiver completion / air mechanics.

Research only. Separate develop/confirm commands enforce a committed 2024 lock.
YAC per target is copied from frozen Phase1G, never corrected or rescaled.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import nfl_v2_phase1a_direct as A
import nfl_v2_phase1b_opportunity as B
import nfl_v2_phase1d_role_allocation as D
import nfl_v2_phase1e_integrated as E
import nfl_v2_phase1g_receiving_mechanics as G
from nfl_v2_phase1h_sources import records, local_name
from nfl_v2_phase1i_sources import verify

ROOT = Path(__file__).resolve().parent
ART = ROOT / "nfl_models/nfl_player_outcome_v2"
PROTOCOL = ART / "phase1i_target_depth_protocol.json"
P = json.loads(PROTOCOL.read_text())
BUCKETS = ("behind_LOS", "0_9", "10_19", "20_plus")
FAMILIES = tuple(P["ablations"])


def bucket(air):
    if not np.isfinite(air):
        raise ValueError("Missing/nonfinite depth is not a zero-yard target")
    return 0 if air < 0 else 1 if air < 10 else 2 if air < 20 else 3


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def sufficient(events):
    n, c, air = np.zeros(4), np.zeros(4), np.zeros(4)
    for e in events:
        b = e["bucket"]
        n[b] += 1; c[b] += e["completion"]
        air[b] += e["air"]*e["completion"]
    return {"targets": n.tolist(), "catches": c.tolist(), "completed_air": air.tolist(),
            "n": int(n.sum()), "last_key": max((e["key"] for e in events), default=None),
            "last_date": max((e["date"] for e in events), default=None)}


def posterior(stats, parent, k):
    n,c,a = [np.asarray(stats[key],float) for key in ("targets","catches","completed_air")]
    p,comp,air = [np.asarray(parent[key],float) for key in ("depth","completion","air")]
    depth = (n+k*p)/(n.sum()+k)
    kt = np.maximum(2., k*p)
    kc = np.maximum(1., k*p*comp)
    return {"depth": depth.tolist(), "completion": ((c+kt*comp)/(n+kt)).tolist(),
            "air": ((a+kc*air)/(c+kc)).tolist()}


def fit_priors(events, max_week):
    # Fitted parameters, not historical target-row features. Validation is later.
    selected = [e for e in events if e["key"][0] == 2024 and e["key"][1] <= max_week]
    if not selected:
        raise ValueError("2024 fit-period target plays required")
    s = sufficient(selected)
    n,c,a = [np.asarray(s[k],float) for k in ("targets","catches","completed_air")]
    if np.any(n == 0) or np.any(c == 0):
        raise ValueError("Every depth bucket needs training targets and catches")
    league = {"depth": (n/n.sum()).tolist(), "completion": (c/n).tolist(), "air": (a/c).tolist()}
    positions = {pos: posterior(sufficient([e for e in selected if e["position"] == pos]),league,50.) for pos in ("WR","TE","RB")}
    return {"league": league, "positions": positions, "fit_n": len(selected), "fit_period": [2024,1,max_week]}


def prior_roster_ids(rosters, team, target):
    """Strict prior-week identities, never target-week participation/status."""
    keys = [k for k in rosters if k[2] == team and k[:2] < target]
    if not keys:
        return set(), None
    key = max(keys)
    return {q["player_id"] for q in rosters[key] if q["position"] == "QB"}, key[:2]


class DepthStore:
    def __init__(self, directory, positions):
        self.events = []
        self.idx = {k: defaultdict(list) for k in ("receiver","qb","pair","opponent","team")}
        self.labels = defaultdict(list)
        self.dates = {}
        self.keys, self.cache = {}, {}
        for year in (2023,2024,2025,2026):
            for r in records(Path(directory)/f"pbp_{year}.csv.gz"):
                if r.get("season_type") != "REG" or r.get("no_play") == "1" or r.get("two_point_attempt") == "1" or r.get("play_type") != "pass":
                    continue
                week = int(r["week"])
                if year == 2026 and week > 4:
                    continue
                receiver, qb = r.get("receiver_player_id"), r.get("passer_player_id")
                pos = G.GROUPS.get(positions.get((year,receiver)))
                if not receiver or not pos:
                    continue
                air = G.fnum(r.get("air_yards"))
                if air is None:
                    continue
                date = r.get("game_date")
                if not date:
                    raise ValueError("Historical game date required")
                comp = float(r.get("complete_pass") == "1")
                yac = G.fnum(r.get("yards_after_catch")) if comp else 0.
                yards = (G.fnum(r.get("yards_gained")) or 0.) if comp else 0.
                e = {"key": (year,week), "date": date, "game_id": r["game_id"],
                     "team": r["posteam"], "opponent": r["defteam"], "receiver": receiver,
                     "qb": qb or "UNKNOWN", "receiver_name": r.get("receiver_player_name"), "qb_name": r.get("passer_player_name"),
                     "position": pos, "air": air, "completion": comp, "bucket": bucket(air),
                     "yac": yac, "yards": yards,
                     "component_valid": not comp or (yac is not None and abs(air+yac-yards)<.01)}
                self.events.append(e)
                self.labels[(year,week,receiver)].append(e)
                self.dates[(year,week,r["posteam"])] = date
                self.idx["receiver"][receiver].append(e)
                self.idx["qb"][(e["qb"],pos)].append(e)
                self.idx["pair"][(e["qb"],receiver)].append(e)
                self.idx["opponent"][(r["defteam"],pos)].append(e)
                self.idx["team"][r["posteam"]].append(e)
        for kind, idx in self.idx.items():
            for identity, events in idx.items():
                events.sort(key=lambda e:(e["key"],e["date"],e["game_id"]))
                self.keys[(kind,identity)] = [e["key"] for e in events]

    def prior(self, kind, identity, target, date, games=8):
        vals = self.idx[kind].get(identity,())
        end = bisect_left(self.keys.get((kind,identity),()),target)
        seen, out = set(), []
        for e in reversed(vals[:end]):
            if e["date"] >= date:
                continue
            if e["game_id"] not in seen and len(seen) >= games:
                break
            seen.add(e["game_id"]);out.append(e)
        return list(reversed(out))

    def stats(self, kind, identity, target, date):
        key = (kind,identity,target,date)
        if key not in self.cache:
            self.cache[key] = sufficient(self.prior(kind,identity,target,date))
        return self.cache[key]

    def assign_qb(self, team, target, date, roster_ids):
        # Prior roster membership only. NO target-week status/participant/QB oracle.
        scores = defaultdict(float)
        prior = self.prior("team",team,target,date,games=3)
        games = sorted({e["key"] for e in prior})
        for e in prior:
            if e["qb"] in roster_ids:
                scores[e["qb"]] += .5**(len(games)-1-games.index(e["key"]))
        source = "PRIOR_CURRENT_TEAM_PASSING"
        if not scores:
            source = "PRIOR_CAREER_PASSING_COLD_TEAM_FALLBACK"
            for qb in sorted(roster_ids):
                vals = []
                for pos in ("WR","TE","RB"):
                    vals += self.prior("qb",(qb,pos),target,date,games=3)
                keys = sorted({e["key"] for e in vals})[-3:]
                score = sum(.5**(len(keys)-1-keys.index(e["key"])) for e in vals if e["key"] in keys)
                if score:
                    scores[qb] = score
        if not scores:
            return {"qb": None, "source": "UNKNOWN_PRIOR_QB", "weights": {}}
        total = sum(scores.values())
        return {"qb": min(scores,key=lambda q:(-scores[q],q)), "source": source,
                "weights": {q:scores[q]/total for q in sorted(scores)}}


def assemble(directory):
    players,team,opp = A.load_stats([Path(directory)/local_name("stats",s) for s in (2023,2024,2025,2026)])
    players = [r for r in players if r["season"] != 2026 or r["week"] <= 4]
    team = {k:v for k,v in team.items() if k[0] != 2026 or k[1] <= 4}
    opp = {k:v for k,v in opp.items() if k[0] != 2026 or k[1] <= 4}
    A.build_indexes(players,team,opp);B.build_extra_indexes(players,team,opp);D.build_context(players,team)
    D.load_rosters([Path(directory)/local_name("roster",s) for s in (2024,2025,2026)])
    positions = {(r["season"],r["player_id"]):r["position"] for r in players}
    store = DepthStore(directory,positions)
    G.load_pbp([Path(directory)/local_name("pbp",s) for s in (2023,2024,2025,2026)],positions)
    b = json.loads((ART/"phase1b_opportunity_snapshot.json").read_text())
    d = json.loads((ART/"phase1d_role_allocation_snapshot.json").read_text())
    gcfg = json.loads((ART/"phase1g_receiving_mechanics_snapshot.json").read_text())["selected_config"]
    rows, quality = [], defaultdict(int)
    for r in B.fixed_meaningful_rows(players,[r for r in players if r["season"]>=2024],"rec_yds"):
        target = (r["season"],r["week"]);pos = G.GROUPS.get(r["position"])
        date = store.dates.get((*target,r["team"]))
        f = B.efficiency_receipt(players,team,opp,r,"rec_yds",b["outcomes"]["rec_yds"]["selected_efficiency_config"])
        g = G.efficiency_receipt(r,gcfg)
        hist = A.prior_player_rows(players,target,r["player_id"],window=8)
        ht = sum(h["targets"] for h in hist)
        if not pos or not date or f is None or g is None or ht <= 0:
            quality["missing_common_comparator_rows"] += 1
            continue
        roster_ids, roster_key = prior_roster_ids(D._ROSTERS,r["team"],target)
        assignment = store.assign_qb(r["team"],target,date,roster_ids)
        assignment["roster_history_key"] = roster_key
        qb = assignment["qb"]
        quality[f"{r['season']}_qb_{assignment['source']}"] += 1
        labels = store.labels.get((*target,r["player_id"]),())
        reconciled = len(labels)==r["targets"] and sum(e["completion"] for e in labels)==r["receptions"] and abs(sum(e["yards"] for e in labels)-r["receiving_yards"])<.01 and all(e["component_valid"] for e in labels)
        quality[f"{r['season']}_component_{'reconciled' if reconciled else 'unreconciled'}"] += 1
        base = E.receipt(players,team,opp,r,"rec_yds",b,d)
        raw_history = store.prior("receiver",r["player_id"],target,date,games=1000)
        wanted = {(h["season"],h["week"]) for h in hist}
        history_stats = sufficient([e for e in raw_history if e["key"] in wanted])
        team_history = store.prior("team",r["team"],target,date)
        last_qb = next((e for e in reversed(team_history) if e["qb"]==qb),{})
        last_receiver = store.prior("receiver",r["player_id"],target,date)
        rows.append({**r,"position_group":pos,"target_game_date":date,
            "receiver":r["player_id"],"receiver_name":last_receiver[-1]["receiver_name"] if last_receiver else None,
            "qb":qb,"qb_name":last_qb.get("qb_name"),"qb_assignment":assignment,
            "game_id":f"{r['season']}_{r['week']:02d}_"+"_".join(sorted((r["team"],r["opponent"]))),
            "projected_targets":None if base is None else base["player_opportunity_projection"],
            "component_reconciled":reconciled,"actual_depth":sufficient(labels) if reconciled else None,
            "history_stats":history_stats,"history_ypt":sum(h["receiving_yards"] for h in hist)/ht,
            "phase1f_ypt":f["final_efficiency_projection"],"phase1g_ypt":g["projected_yards_per_target"],
            "phase1g_catch":g["projected_catch_rate"],"phase1g_air_ypt":g["projected_catch_rate"]*g["projected_air_per_catch"],
            "incumbent_yac_per_target":g["projected_catch_rate"]*g["projected_yac_per_catch"],
            "entity_history":{kind:store.stats(kind,identity,target,date) for kind,identity in (
                ("receiver",r["player_id"]),("qb",(qb,pos)),("pair",(qb,r["player_id"])),("opponent",(r["opponent"],pos)))}})
    return rows,store.events,dict(quality)


def model_spec(family, strengths, opponent_weight=0.):
    modes = {"receiver_depth_only":"receiver","qb_depth_only":"qb","qb_receiver_hierarchy":"hierarchy",
             "qb_receiver_pair":"pair","opponent_depth_allowance":"position","depth_catchability":"position"}
    return {"families":[family],"depth_mode":modes[family],"depth_strengths":strengths,
            "catch_mode":"pair" if family == "depth_catchability" else "position",
            "catch_strengths":strengths,"opponent_depth_weight":opponent_weight,
            "opponent_strength":strengths["opponent"]}


def hierarchy(r,priors,strengths):
    pos = priors["positions"][r["position_group"]]
    hs = r["entity_history"]
    qb = posterior(hs["qb"],pos,strengths["qb"])
    receiver = posterior(hs["receiver"],pos,strengths["receiver"])
    both = posterior(hs["receiver"],qb,strengths["receiver"])
    pair = posterior(hs["pair"],both,strengths["pair"]) if hs["pair"]["n"]>=20 else both
    return {"position":pos,"qb":qb,"receiver":receiver,"hierarchy":both,"pair":pair}


def project(r,spec,priors):
    dep = hierarchy(r,priors,spec["depth_strengths"])[spec["depth_mode"]]
    depth = np.array(dep["depth"],float);air = np.array(dep["air"],float)
    catch = hierarchy(r,priors,spec["catch_strengths"])[spec["catch_mode"]]
    c = np.array(catch["completion"],float)
    parent = priors["positions"][r["position_group"]]
    defense = posterior(r["entity_history"]["opponent"],parent,spec["opponent_strength"])
    w = spec["opponent_depth_weight"]
    if w:
        depth = (1-w)*depth+w*np.array(defense["depth"])
    if spec["catch_mode"] != "position":
        logit = lambda p: np.log(np.clip(p,.01,.99)/(1-np.clip(p,.01,.99)))
        z = logit(c)+.15*(logit(np.array(defense["completion"]))-logit(np.array(parent["completion"])))
        c = 1/(1+np.exp(-z))
    contributions = depth*air*c
    airypt = float(contributions.sum())
    ypt = airypt+r["incumbent_yac_per_target"]
    return {"depth_probabilities":depth.tolist(),"expected_air_within_bucket":air.tolist(),
            "completion_probabilities":c.tolist(),"completed_air_contributions":contributions.tolist(),
            "completed_air_per_target":airypt,"catch_probability":float(np.dot(depth,c)),
            "incumbent_yac_per_target":r["incumbent_yac_per_target"],"yards_per_target":ypt,
            "direct_receiving_yard_projection":None if r["projected_targets"] is None else r["projected_targets"]*ypt,
            "pair_correction_applied":r["entity_history"]["pair"]["n"]>=20 and (spec["depth_mode"]=="pair" or spec["catch_mode"]=="pair"),
            "pair_prior_targets":r["entity_history"]["pair"]["n"]}


def comparator(r,name):
    if name == "phase1f":
        return {"yards_per_target":r["phase1f_ypt"],"completed_air_per_target":None,"catch_probability":None,
                "depth_probabilities":None,"completion_probabilities":None,"completed_air_contributions":None}
    if name == "phase1g":
        return {"yards_per_target":r["phase1g_ypt"],"completed_air_per_target":r["phase1g_air_ypt"],"catch_probability":r["phase1g_catch"],
                "depth_probabilities":None,"completion_probabilities":[r["phase1g_catch"]]*4,"completed_air_contributions":None,
                "incumbent_yac_per_target":r["incumbent_yac_per_target"]}
    h = r["history_stats"]
    n,c,a = [np.array(h[k],float) for k in ("targets","catches","completed_air")]
    t = n.sum()
    return {"yards_per_target":r["history_ypt"],"completed_air_per_target":float(a.sum()/t) if t else None,
            "catch_probability":float(c.sum()/t) if t else None,
            "depth_probabilities":(n/t).tolist() if t else None,
            "completion_probabilities":[float(c[b]/n[b]) if n[b] else None for b in range(4)],
            "completed_air_contributions":(a/t).tolist() if t else None}


def predict(rows,spec,priors):
    return [project(r,spec,priors) for r in rows]


def metric_values(r,q):
    if r["targets"] <= 0:
        return {}
    t,y = r["targets"],r["receiving_yards"]
    error = t*q["yards_per_target"]-y
    v = {"oracle_target_rec_yds_mae":abs(error),"yards_per_target_mae":abs(error/t),
         "signed_bias":error, **{f"misses_over_{k}":float(abs(error)>k) for k in (20,30,40)}}
    if q["catch_probability"] is not None:
        v["catch_rate_mae"] = abs(q["catch_probability"]-r["receptions"]/t)
    if not r["component_reconciled"]:
        return v
    n,c,a = [np.array(r["actual_depth"][k],float) for k in ("targets","catches","completed_air")]
    if q["completed_air_per_target"] is not None:
        v["completed_air_per_target_mae"] = abs(q["completed_air_per_target"]-float(a.sum()/t))
        v["completed_air_signed_bias"] = q["completed_air_per_target"]-float(a.sum()/t)
    if "incumbent_yac_per_target" in q:
        v["fixed_yac_per_target_mae"] = abs(q["incumbent_yac_per_target"]-(y-a.sum())/t)
    if q["depth_probabilities"] is not None:
        p = np.array(q["depth_probabilities"])
        v["target_depth_distribution_error"] = float(.5*np.abs(p-n/t).sum())
        v["deep_target_frequency_error"] = abs(p[3]-n[3]/t)
        if q["completion_probabilities"] is not None and q["completion_probabilities"][3] is not None:
            v["explosive_completed_target_probability_error"] = abs(p[3]*q["completion_probabilities"][3]-c[3]/t)
    if q["completed_air_contributions"] is not None:
        v["explosive_completed_air_error"] = abs(q["completed_air_contributions"][3]-a[3]/t)
    if q["completion_probabilities"] is not None:
        for b in range(4):
            if n[b] and q["completion_probabilities"][b] is not None:
                v[f"catch_rate_{BUCKETS[b]}_mae"] = abs(q["completion_probabilities"][b]-c[b]/n[b])
    return v


def metrics(rows,preds):
    values = defaultdict(list)
    for r,q in zip(rows,preds):
        for k,v in metric_values(r,q).items():
            values[k].append(v)
    out = {"n":len(values["oracle_target_rec_yds_mae"]), **{k:mean(v) for k,v in values.items()},
           "metric_n":{k:len(v) for k,v in values.items()}}
    cs = [out[f"catch_rate_{b}_mae"] for b in BUCKETS if f"catch_rate_{b}_mae" in out]
    out["macro_depth_catch_rate_mae"] = mean(cs)
    return out


def bootstrap(rows,a,b,metric):
    sums,counts = defaultdict(float),defaultdict(int)
    for r,qa,qb in zip(rows,a,b):
        va,vb = metric_values(r,qa),metric_values(r,qb)
        if metric not in va or metric not in vb:
            continue
        k = (r["season"],r["week"],r["game_id"])
        sums[k] += va[metric]-vb[metric];counts[k] += 1
    keys = sorted(sums)
    if not keys:
        return {"delta":None,"ci95":[None,None]}
    blocks = {}
    for s in sorted({k[0] for k in keys}):
        weeks = list(range(min(k[1] for k in keys if k[0]==s),max(k[1] for k in keys if k[0]==s)+1))
        blocks[s] = [[k for k in keys if k[0]==s and k[1] in {w,weeks[(i+1)%len(weeks)]}] for i,w in enumerate(weeks)]
    rng,draws = np.random.default_rng(165),[]
    for _ in range(2000):
        ks = []
        for bs in blocks.values():
            for ix in rng.integers(0,len(bs),size=(len(bs)+1)//2):
                ks.extend(bs[ix])
        n = sum(counts[k] for k in ks)
        if n:
            draws.append(sum(sums[k] for k in ks)/n)
    return {"delta":sum(sums.values())/sum(counts.values()),"ci95":np.quantile(draws,[.025,.975]).tolist(),
            "resamples":len(draws),"seed":165,"unit":"paired2-week blocks / whole games"}


def improve(a,b,key,fraction=.005):
    return a.get(key) is not None and b.get(key) is not None and a[key] <= (1-fraction)*b[key]


def family_gate(name,m,base,boot,parents=()):
    checks = {"air_component_gain":improve(m,base,"completed_air_per_target_mae"),
              "oracle_not_worse":m["oracle_target_rec_yds_mae"]<=base["oracle_target_rec_yds_mae"],
              "air_block_support":boot["ci95"][1] is not None and boot["ci95"][1]<0}
    if name == "depth_catchability":
        checks.update(catch_overall_gain=improve(m,base,"catch_rate_mae"),catch_depth_gain=improve(m,base,"macro_depth_catch_rate_mae"))
    else:
        checks.update(depth_distribution_gain=improve(m,base,"target_depth_distribution_error"),
                      deep_frequency_not_worse=m["deep_target_frequency_error"]<=base["deep_target_frequency_error"])
    for i,parent in enumerate(parents):
        checks[f"parent_{i}_air_gain"] = improve(m,parent,"completed_air_per_target_mae")
        checks[f"parent_{i}_depth_not_worse"] = m["target_depth_distribution_error"]<=parent["target_depth_distribution_error"]
    return {"passes":all(checks.values()),"checks":checks}


def overall_gate(m,f,g,h,boots):
    checks = {"air_gain_vs_g":improve(m,g,"completed_air_per_target_mae",.01),
              "oracle_gain_vs_f":improve(m,f,"oracle_target_rec_yds_mae"),
              "oracle_gain_vs_history":improve(m,h,"oracle_target_rec_yds_mae"),
              "catch_not_worse_vs_g":m["catch_rate_mae"]<=g["catch_rate_mae"],
              "air_block_support":boots["air_vs_g"]["ci95"][1] is not None and boots["air_vs_g"]["ci95"][1]<0,
              "oracle_block_support":boots["oracle_vs_f"]["ci95"][1] is not None and boots["oracle_vs_f"]["ci95"][1]<0}
    return {"passes":all(checks.values()),"checks":checks}


def contains_both_entities(spec):
    return spec["depth_mode"] in {"hierarchy","pair"} or spec["catch_mode"] == "pair"


def develop(rows,events,quality,audit):
    priors = fit_priors(events,8)
    dev = [r for r in rows if r["season"]==2024 and r["week"]>=9]
    default = P["priors"]["concentrations"][0]
    reference_spec = {**model_spec("opponent_depth_allowance",default),"families":[],"depth_mode":"position"}
    refp = predict(dev,reference_spec,priors);refm = metrics(dev,refp)
    cp = {name:[comparator(r,name) for r in dev] for name in ("phase1f","phase1g","history")}
    cm = {name:metrics(dev,p) for name,p in cp.items()}
    specs,results,trial_list = {},{},{}
    for family in FAMILIES:
        trials = []
        for strengths in P["priors"]["concentrations"]:
            for w in (P["priors"]["opponent_depth_weights"] if family=="opponent_depth_allowance" else [0.]):
                spec = model_spec(family,strengths,w)
                m = metrics(dev,predict(dev,spec,priors))
                key = (m["completed_air_per_target_mae"],m["target_depth_distribution_error"],m["oracle_target_rec_yds_mae"],-strengths["receiver"],json.dumps(spec,sort_keys=True))
                trials.append((key,spec,m))
        _,spec,m = min(trials,key=lambda t:t[0])
        specs[family] = spec
        b = bootstrap(dev,predict(dev,spec,priors),refp,"completed_air_per_target_mae")
        trial_list[family] = [{"spec":s,"metrics":z} for _,s,z in trials]
        results[family] = {"metrics":m,"air_bootstrap_vs_position":b,"trials":trial_list[family]}
    for family in FAMILIES:
        parents = []
        if family=="qb_receiver_hierarchy":
            parents = [results[n]["metrics"] for n in ("receiver_depth_only","qb_depth_only")]
        if family=="qb_receiver_pair":
            parents = [results["qb_receiver_hierarchy"]["metrics"]]
        v = results[family]
        v["gate"] = family_gate(family,v["metrics"],refm,v["air_bootstrap_vs_position"],parents)
        v["status"] = "SURVIVES" if v["gate"]["passes"] else "REJECTED"
    # Earned component combinations only; required structural ablations above
    # are separately controlled, never an unearned all-feature candidate.
    depth_families = [f for f in FAMILIES if f not in {"depth_catchability","opponent_depth_allowance"} and results[f]["status"]=="SURVIVES"]
    additions = [f for f in ("depth_catchability","opponent_depth_allowance") if results[f]["status"]=="SURVIVES"]
    for df in depth_families:
        for size in range(1,len(additions)+1):
            for added in itertools.combinations(additions,size):
                name = "+".join((df,*added));spec = dict(specs[df]);spec["families"]=[df,*added]
                if "depth_catchability" in added:
                    spec["catch_mode"]="pair";spec["catch_strengths"]=specs["depth_catchability"]["catch_strengths"]
                if "opponent_depth_allowance" in added:
                    spec["opponent_depth_weight"]=specs["opponent_depth_allowance"]["opponent_depth_weight"]
                    spec["opponent_strength"]=specs["opponent_depth_allowance"]["opponent_strength"]
                m=metrics(dev,predict(dev,spec,priors));b=bootstrap(dev,predict(dev,spec,priors),refp,"completed_air_per_target_mae")
                g=family_gate(df,m,refm,b)
                if "depth_catchability" in added:
                    cg=family_gate("depth_catchability",m,refm,b);g={"passes":g["passes"] and cg["passes"],"checks":{**g["checks"],**cg["checks"]}}
                specs[name]=spec;results[name]={"metrics":m,"gate":g,"status":"SURVIVES" if g["passes"] else "REJECTED","air_bootstrap_vs_position":b}
    eligible=[]
    for name,v in results.items():
        pred=predict(dev,specs[name],priors)
        boots={"air_vs_g":bootstrap(dev,pred,cp["phase1g"],"completed_air_per_target_mae"),"oracle_vs_f":bootstrap(dev,pred,cp["phase1f"],"oracle_target_rec_yds_mae")}
        v["overall_gate"]=overall_gate(v["metrics"],cm["phase1f"],cm["phase1g"],cm["history"],boots)
        v["bootstrap"]=boots;v["qb_and_receiver_both_present"]=contains_both_entities(specs[name])
        if v["status"]=="SURVIVES" and v["overall_gate"]["passes"] and v["qb_and_receiver_both_present"]:
            eligible.append(name)
    selected=min(eligible,key=lambda n:(results[n]["metrics"]["oracle_target_rec_yds_mae"],len(specs[n]["families"]),n)) if eligible else None
    return {"schema":"nfl-v2-phase1i-development-lock-v1","selected":selected,
            "fit_priors":priors,"refit_priors":fit_priors(events,18),"frozen_specs":specs,
            "reference_spec":reference_spec,"reference_metrics":refm,"comparators":cm,"candidates":results,
            "selection_n":len(dev),"data_quality":quality,
            "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "audit_sha256":hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
            "selection_rule":"2024 W1-8 fit / W9-18 select; refit2024. No2025/2026 performance used."}


def slice_metrics(rows,preds):
    ids={pos:[i for i,r in enumerate(rows) if r["position_group"]==pos] for pos in ("WR","TE","RB")}
    ids.update(early=[i for i,r in enumerate(rows) if r["week"]<=4],established=[i for i,r in enumerate(rows) if r["week"]>4])
    return {n:metrics([rows[i] for i in ix],[preds[i] for i in ix]) for n,ix in ids.items()}


def full_metrics(rows,preds):
    es=[r["projected_targets"]*q["yards_per_target"]-r["receiving_yards"] for r,q in zip(rows,preds) if r["projected_targets"] is not None]
    return {"n":len(es),"mae":mean([abs(e) for e in es]),"signed_bias":mean(es),
            "within_10":mean([abs(e)<=10 for e in es]),"within_20":mean([abs(e)<=20 for e in es]),"misses_over_40":mean([abs(e)>40 for e in es])}


def confirm(rows,lock):
    if lock["protocol_sha256"]!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise ValueError("Protocol changed after selection")
    priors,specs,selected=lock["refit_priors"],lock["frozen_specs"],lock["selected"]
    out={"schema":"nfl-v2-phase1i-results-v1","selected":selected,"periods":{},
         "sportsbook_inputs_used":False,"monte_carlo_used":False,"fixed_yac":"Phase1G_YAC_per_target_unchanged",
         "development":{k:v for k,v in lock.items() if k not in {"frozen_specs","fit_priors","refit_priors"}}}
    receipts=[]
    for label,year in (("validation_2025",2025),("diagnostic_2026_wk1_4",2026)):
        rs=[r for r in rows if r["season"]==year]
        ps={name:predict(rs,spec,priors) for name,spec in specs.items()}
        ps["position_depth"]=predict(rs,lock["reference_spec"],priors)
        ps.update({n:[comparator(r,n) for r in rs] for n in ("phase1f","phase1g","history")})
        ms={name:metrics(rs,pr) for name,pr in ps.items()}
        bs,decisions={},{}
        for name,spec in specs.items():
            boots={"air_vs_g":bootstrap(rs,ps[name],ps["phase1g"],"completed_air_per_target_mae"),"oracle_vs_f":bootstrap(rs,ps[name],ps["phase1f"],"oracle_target_rec_yds_mae")}
            bs[name]=boots
            og=overall_gate(ms[name],ms["phase1f"],ms["phase1g"],ms["history"],boots)
            component_boot=bootstrap(rs,ps[name],ps["position_depth"],"completed_air_per_target_mae")
            bs[name]["air_vs_position"]=component_boot
            checks={}
            for family in spec["families"]:
                parents=[]
                if family=="qb_receiver_hierarchy":
                    parents=[ms[n] for n in ("receiver_depth_only","qb_depth_only")]
                elif family=="qb_receiver_pair":
                    parents=[ms["qb_receiver_hierarchy"]]
                checks[family]=family_gate(family,ms[name],ms["position_depth"],component_boot,parents)
            component_pass=all(g["passes"] for g in checks.values()) and lock["candidates"][name]["status"]=="SURVIVES"
            passed=og["passes"] and component_pass and name==selected
            decisions[name]={"status":"SURVIVES" if passed else "REJECTED","component_status":"SURVIVES" if component_pass else "REJECTED","overall_gate":og,"claimed_component_checks":checks,"frozen_development_status":lock["candidates"][name]["status"]}
            for r,q in zip(rs,ps[name]):
                # Every evaluated row retains all model factors and provenance.
                receipts.append({"architecture":name,"period":label,**r,**q})
        out["periods"][label]={"metrics":ms,"bootstrap":bs,"decisions":decisions,"slices":{n:slice_metrics(rs,p) for n,p in ps.items()}}
    passed=selected is not None and out["periods"]["validation_2025"]["decisions"][selected]["status"]=="SURVIVES"
    out["verdict"]="SURVIVES_RESEARCH_ONLY_PENDING_CLEAN_FORWARD" if passed else "REJECTED_EFFICIENCY_REPLACEMENT"
    out["family_decisions"]={f:out["periods"]["validation_2025"]["decisions"][f]["component_status"] for f in FAMILIES}
    out["family_decisions"].update(route_level_separation="BLOCKED_DATA",routes_run_alignment_assignment="BLOCKED_DATA",pressure_blitz="NOT_TESTED_ISOLATE_DEPTH_CHAIN")
    out["full_projection_evaluation"]={"status":"RUN_AFTER_EFFICIENCY_PASS" if passed else "NOT_RUN_EFFICIENCY_GATE_FAILED"}
    if passed:
        for label,year in (("validation_2025",2025),("diagnostic_2026_wk1_4",2026)):
            rs=[r for r in rows if r["season"]==year]
            out["full_projection_evaluation"][label]=full_metrics(rs,predict(rs,specs[selected],priors))
    return out,receipts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True);ap.add_argument("--stage",choices=("develop","confirm"),required=True)
    ap.add_argument("--audit",default=str(ART/"phase1i_source_audit.json"))
    ap.add_argument("--lock",default=str(ART/"phase1i_development_lock.json"))
    ap.add_argument("--out",default=str(ART/"phase1i_target_depth_results.json"))
    ap.add_argument("--receipts",default=str(ART/"phase1i_receipts.jsonl.gz"))
    a=ap.parse_args()
    verify(a.data_dir)
    if not Path(a.audit).exists():
        raise ValueError("Source feasibility audit required before fitting")
    rows,events,quality=assemble(a.data_dir)
    if a.stage=="develop":
        result=develop(rows,events,quality,a.audit)
        Path(a.lock).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"selected":result["selected"],"fit_n":result["fit_priors"]["fit_n"],"selection_n":result["selection_n"],"reference":result["reference_metrics"],"candidates":{n:{"status":v["status"],"metrics":v["metrics"],"gate":v["gate"]} for n,v in result["candidates"].items()}},indent=2))
    else:
        import subprocess
        rel=Path(a.lock).resolve().relative_to(ROOT)
        committed=subprocess.run(["git","show",f"HEAD:{rel}"],cwd=ROOT,capture_output=True,check=True).stdout
        if committed!=Path(a.lock).read_bytes():
            raise ValueError("2024 development lock must be committed unchanged before validation")
        lock=json.loads(Path(a.lock).read_text())
        if lock["audit_sha256"]!=hashlib.sha256(Path(a.audit).read_bytes()).hexdigest():
            raise ValueError("Source audit changed after selection")
        result,receipts=confirm(rows,lock)
        Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        with open(a.receipts,"wb") as raw:
            with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as f:
                for r in receipts:
                    f.write((json.dumps(r,sort_keys=True,separators=(",",":"))+"\n").encode())
        print(json.dumps({"verdict":result["verdict"],"selected":result["selected"],"periods":{n:v["metrics"] for n,v in result["periods"].items()}},indent=2))


if __name__=="__main__":
    main()

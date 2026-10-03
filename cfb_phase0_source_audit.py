"""
CFB_PHASE0_SOURCE_AUDIT -- reproducible source-coverage + player-identity audit for the CFB Outcome Engine v1 (RESEARCH / SHADOW ONLY; no model, no sportsbook data, no Week 5 outcome).
Reads the cfbfastR-data season files (schedules / rosters / player_stats) from a raw directory and writes:
  phase0_source_feasibility.json (quantified coverage)   cfb_identity_audit.json (ID stability under transfer)
Seasons audited: 2017-2025 only. 2026 files are NOT read (research cutoff: 2026 Week 4 completed games only; the audit needs no 2026 data).
  python cfb_phase0_source_audit.py --raw /tmp/cfbraw
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "cfb_models" / "cfb_outcome_engine"
SEASONS = list(range(2017, 2026))
csv.field_size_limit(10 ** 8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q(x, qs=(0, 5, 25, 50, 75, 95, 100)):
    x = np.asarray(x, float)
    return {f"p{int(p)}": float(np.percentile(x, p)) for p in qs} if len(x) else {}


def audit_schedule(raw, s):
    rows = list(csv.DictReader(open(raw / f"cfb_schedules_{s}.csv", newline="", encoding="utf-8")))
    reg = [r for r in rows if r["season_type"] == "regular"]
    div = lambda r: (r["home_division"], r["away_division"])
    fbs = [r for r in reg if div(r) == ("fbs", "fbs")]
    fbs_fcs = [r for r in reg if "fbs" in div(r) and "fcs" in div(r)]
    nz = lambda v: v not in ("", "NA", None)
    out = {"rows": len(rows), "regular_season_rows": len(reg), "unique_game_ids": len({r["game_id"] for r in rows}), "completed_regular": sum(r["completed"] == "TRUE" for r in reg), "fbs_vs_fbs": len(fbs), "fbs_vs_fcs": len(fbs_fcs),
           "weeks": [min(int(r["week"]) for r in reg), max(int(r["week"]) for r in reg)] if reg else None, "neutral_site_fbs": sum(r["neutral_site"] == "TRUE" for r in fbs), "venue_id_present_fbs": sum(nz(r["venue_id"]) for r in fbs) / max(1, len(fbs)),
           "start_time_tbd_fbs": sum(r["start_time_tbd"] == "TRUE" for r in fbs), "kickoff_timestamp_present_fbs": sum(nz(r["start_date"]) for r in fbs) / max(1, len(fbs)), "final_points_present_fbs_completed": sum(nz(r["home_points"]) and nz(r["away_points"]) for r in fbs if r["completed"] == "TRUE") / max(1, sum(r["completed"] == "TRUE" for r in fbs)),
           "provider_pregame_elo_present_fbs": sum(nz(r["home_pregame_elo"]) and nz(r["away_pregame_elo"]) for r in fbs) / max(1, len(fbs)), "conference_present_fbs": sum(nz(r["home_conference"]) and nz(r["away_conference"]) for r in fbs) / max(1, len(fbs)),
           "forbidden_postgame_columns_present": [c for c in ("home_post_win_prob", "home_postgame_elo", "excitement_index") if c in rows[0]]}
    return out, {r["game_id"]: r for r in reg}


def audit_player_stats(raw, s, sched, prod=None):
    """Stream the play-level file; per (game, team): distinct rush plays, pass-attempt plays, sacks, red-zone plays; missingness of situation columns."""
    path = raw / f"player_stats_{s}.csv"
    rush, passes, sacks, rz, plays = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    miss = Counter(); n = 0; targets = 0; pass_rows = 0; tds = Counter(); score_present = 0
    nz = lambda v: v not in ("", "NA", None)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            k = (r["game_id"], r["team"]); p = r["play_id"]
            plays[k].add(p)
            if nz(r["rush_player_id"]):
                rush[k].add(p)
            is_pass = nz(r["completion_player_id"]) or nz(r["incompletion_player_id"]) or nz(r["interception_thrown_player_id"])
            if is_pass:
                passes[k].add(p); pass_rows += 1
                targets += nz(r["target_player_id"])
            if nz(r["sack_taken_player_id"]):
                sacks[k].add(p)
            if prod is not None:
                for idc, nmc in (("rush_player_id", "rush_player"), ("reception_player_id", "reception_player"), ("completion_player_id", "completion_player")):
                    if nz(r[idc]):
                        e = prod.setdefault(r[idc], {"names": set(), "teams": defaultdict(set)})
                        e["names"].add(r[nmc].strip().lower()); e["teams"][s].add(r["team"])
            if nz(r["yards_to_goal"]) and float(r["yards_to_goal"]) <= 20 and (nz(r["rush_player_id"]) or is_pass):
                rz[k].add(p)
            for c in ("yards_to_goal", "down", "distance", "period", "team_score", "opponent_score", "clock_minutes"):
                miss[c] += not nz(r[c])
            score_present += nz(r["team_score"]) and nz(r["opponent_score"])
            if nz(r["touchdown_player_id"]):
                tds[r["touchdown_stat"]] += 1
    keys = set(plays)
    fbs_games = {g for g, r in sched.items() if (r["home_division"], r["away_division"]) == ("fbs", "fbs") and r["completed"] == "TRUE"}
    games_with_rows = {g for g, _ in keys}
    both_sides = sum(1 for g in fbs_games if sum(1 for (gg, t) in keys if gg == g) >= 2) if len(fbs_games) < 1500 else None
    by_game = Counter(g for g, _ in keys)
    both = sum(1 for g in fbs_games if by_game.get(g, 0) >= 2)
    fcs = {g: r for g, r in sched.items() if "fbs" in (r["home_division"], r["away_division"]) and "fcs" in (r["home_division"], r["away_division"]) and r["completed"] == "TRUE"}
    side_rows = {"fbs_side": 0, "fcs_side": 0}
    for g, r in fcs.items():
        fb, fc = (r["home_team"], r["away_team"]) if r["home_division"] == "fbs" else (r["away_team"], r["home_team"])
        side_rows["fbs_side"] += (g, fb) in keys; side_rows["fcs_side"] += (g, fc) in keys
    tp = np.array([len(rush[k]) + len(passes[k]) + len(sacks[k]) for k in keys if k[0] in fbs_games])
    tr = np.array([len(rush[k]) for k in keys if k[0] in fbs_games]); ta = np.array([len(passes[k]) for k in keys if k[0] in fbs_games]); tz = np.array([len(rz[k]) for k in keys if k[0] in fbs_games])
    return {"rows": n, "distinct_team_games": len(keys), "fbs_vs_fbs_completed_games_in_schedule": len(fbs_games), "fbs_games_with_rows_for_both_teams": both, "fbs_games_with_any_rows": len(fbs_games & games_with_rows), "fbs_vs_fcs_completed_games": len(fcs), "fbs_vs_fcs_games_with_play_rows_for_fbs_team": side_rows["fbs_side"], "fbs_vs_fcs_games_with_play_rows_for_fcs_team": side_rows["fcs_side"],
            "team_game_plays_proxy_rush+pass+sack_quantiles": q(tp), "team_game_rush_plays_quantiles": q(tr), "team_game_pass_attempt_plays_quantiles": q(ta), "team_game_red_zone_plays_quantiles": q(tz),
            "implausible_team_games_lt30_or_gt140_plays": int(((tp < 30) | (tp > 140)).sum()), "situation_column_missing_share": {c: v / max(1, n) for c, v in miss.items()}, "score_state_present_share": score_present / max(1, n),
            "pass_attempt_rows": pass_rows, "target_player_id_present_share_of_pass_attempts": targets / max(1, pass_rows), "touchdown_stat_counts": dict(tds.most_common(8))}


def audit_rosters(raw, s):
    rows = list(csv.DictReader(open(raw / f"cfb_rosters_{s}.csv", newline="", encoding="utf-8")))
    nz = lambda v: v not in ("", "NA", None)
    teams = {r["team"] for r in rows}
    key = Counter((r["athlete_id"], r["team"]) for r in rows)
    return {"rows": len(rows), "unique_athlete_ids": len({r["athlete_id"] for r in rows}), "teams": len(teams), "duplicate_athlete_team_rows": sum(v - 1 for v in key.values() if v > 1),
            "class_year_present": sum(nz(r["year"]) for r in rows) / max(1, len(rows)), "position_present": sum(nz(r["position"]) for r in rows) / max(1, len(rows)), "height_present": sum(nz(r["height"]) for r in rows) / max(1, len(rows)),
            "weight_present": sum(nz(r["weight"]) for r in rows) / max(1, len(rows)), "recruit_ids_present": sum(nz(r["recruit_ids"]) for r in rows) / max(1, len(rows)), "jersey_present": sum(nz(r["jersey"]) for r in rows) / max(1, len(rows)),
            "class_year_values": dict(Counter(r["year"] for r in rows).most_common(8)), "position_values_top": dict(Counter(r["position"] for r in rows).most_common(10))}, rows


def audit_identity(rosters):
    """rosters: {season: rows}. Provider athlete_id stability under transfer + collision / unresolved-identity counts. No name joins are ever used to stitch."""
    hist = defaultdict(dict)                    # athlete_id -> {season: set(teams)}
    names = defaultdict(lambda: defaultdict(set))
    recruit = defaultdict(set)
    for s, rows in rosters.items():
        for r in rows:
            a = r["athlete_id"]
            hist[a].setdefault(s, set()).add(r["team"])
            names[a][s].add((r["first_name"].strip().lower(), r["last_name"].strip().lower()))
            for rid in (r["recruit_ids"] or "").replace("[", "").replace("]", "").replace('"', "").split(","):
                rid = rid.strip()
                if rid and rid != "NA":
                    recruit[rid].add(a)
    seasons = sorted(rosters)
    transfers = []                               # same athlete_id, consecutive seasons, different team set
    multi_team_same_season = 0
    for a, d in hist.items():
        multi_team_same_season += sum(1 for s, ts in d.items() if len(ts) > 1)
        for s0, s1 in zip(seasons, seasons[1:]):
            if s0 in d and s1 in d and d[s0].isdisjoint(d[s1]):
                transfers.append((a, s0, s1))
    name_change = sum(1 for a, d in names.items() if len({n for ns in d.values() for n in ns}) > 1)
    ids_with_gap = sum(1 for a, d in hist.items() if len(d) >= 2 and any(s not in d for s in range(min(d), max(d) + 1) if s in seasons))
    # recruit-id deterministic cross-ID mapping: one recruit linked to >1 athlete_id
    multi_recruit = {rid: a for rid, a in recruit.items() if len(a) > 1}
    stitchable = []
    for rid, ids in multi_recruit.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                da, db = hist[a], hist[b]
                if max(da) < min(db) or max(db) < min(da):
                    stitchable.append((rid, a, b))
    # name+position successor candidates with DIFFERENT ids across a school change: unresolved, never stitched
    by_name = defaultdict(list)
    for s, rows in rosters.items():
        for r in rows:
            by_name[(r["first_name"].strip().lower(), r["last_name"].strip().lower(), r["position"])].append((s, r["team"], r["athlete_id"]))
    unresolved = 0; collisions = 0
    for nm, lst in by_name.items():
        ids = {x[2] for x in lst}
        if len(ids) > 1:
            lst2 = sorted(lst)
            for (s0, t0, a0), (s1, t1, a1) in zip(lst2, lst2[1:]):
                if a0 != a1 and s1 == s0 + 1 and t0 != t1:
                    unresolved += 1
                if a0 != a1 and s0 == s1 and t0 == t1:
                    collisions += 1
    ps_cov = None
    return {"seasons": seasons, "unique_athlete_ids": len(hist), "athlete_ids_in_2_plus_seasons": sum(1 for d in hist.values() if len(d) >= 2),
            "transfers_same_id_consecutive_seasons_different_team": len(transfers), "transfers_by_season_pair": dict(Counter(f"{s0}->{s1}" for _, s0, s1 in transfers)), "ids_with_a_roster_gap_year": ids_with_gap,
            "athlete_ids_on_2_teams_same_season": multi_team_same_season, "ids_with_name_change_across_seasons": name_change,
            "recruit_ids_linking_more_than_one_athlete_id": len(multi_recruit), "recruit_link_pairs_with_non_overlapping_seasons (deterministic cross-ID stitch candidates)": len(stitchable),
            "same_name_position_different_id_school_change_next_season (UNRESOLVED, never stitched)": unresolved, "same_name_position_two_ids_same_team_same_season (collision)": collisions,
            "sample_transfers": transfers[:5], "sample_stitch_candidates": stitchable[:5]}, hist, transfers


def audit_producers(prod, rosters):
    """Identity of box-score producers (rushers / receivers / passers): does the provider ID persist when a producer changes school, and are placeholder IDs usable?"""
    seasons = sorted(rosters)
    roster_team = defaultdict(dict)
    for s, rows in rosters.items():
        for r in rows:
            roster_team[r["athlete_id"]].setdefault(s, set()).add(r["team"])
    neg = [a for a in prod if a.startswith("-")]
    transfers = []; name_mismatch_transfers = 0; roster_confirm = 0
    for a, e in prod.items():
        for s0, s1 in zip(seasons, seasons[1:]):
            if s0 in e["teams"] and s1 in e["teams"] and e["teams"][s0].isdisjoint(e["teams"][s1]):
                transfers.append((a, s0, s1))
                name_mismatch_transfers += len(e["names"]) > 1
                rt = roster_team.get(a, {})
                roster_confirm += (s0 in rt and s1 in rt and rt[s0].isdisjoint(rt[s1]) and e["teams"][s0] & rt[s0] and e["teams"][s1] & rt[s1] and True) or 0
    pos_tr = [t for t in transfers if not t[0].startswith("-")]
    return {"producer_ids": len(prod), "placeholder_negative_ids": len(neg), "producer_ids_in_2_plus_seasons": sum(1 for e in prod.values() if len(e["teams"]) >= 2), "producer_transfers_same_id_different_team": len(transfers),
            "producer_transfers_with_positive_id": len(pos_tr), "producer_transfers_with_placeholder_id": len(transfers) - len(pos_tr), "producer_transfers_with_name_inconsistency": name_mismatch_transfers,
            "producer_transfers_confirmed_by_both_season_rosters": int(roster_confirm), "producer_transfers_by_season_pair": dict(Counter(f"{s0}->{s1}" for _, s0, s1 in transfers)),
            "placeholder_ids_by_first_season": dict(Counter(min(prod[a]["teams"]) for a in neg)), "producer_ids_with_multiple_names": sum(1 for e in prod.values() if len(e["names"]) > 1), "sample_positive_transfers": pos_tr[:5]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); a = ap.parse_args()
    raw = Path(a.raw)
    feas = {"kind": "CFB source feasibility (quantified); seasons 2017-2025 only; no 2026 file read", "files": {}, "schedules": {}, "player_stats_play_level": {}, "rosters": {}}
    rosters = {}; prod = {}
    for s in SEASONS:
        for nm in (f"cfb_schedules_{s}.csv", f"cfb_rosters_{s}.csv", f"player_stats_{s}.csv"):
            feas["files"][nm] = {"sha256": sha256_file(raw / nm), "bytes": (raw / nm).stat().st_size}
        sc, sched = audit_schedule(raw, s)
        feas["schedules"][s] = sc
        print(s, "schedule", sc["fbs_vs_fbs"], flush=True)
        feas["player_stats_play_level"][s] = audit_player_stats(raw, s, sched, prod)
        print(s, "plays", feas["player_stats_play_level"][s]["distinct_team_games"], flush=True)
        ro, rows = audit_rosters(raw, s)
        feas["rosters"][s] = ro; rosters[s] = rows
    ident, hist, transfers = audit_identity(rosters)
    ident["box_score_producers"] = audit_producers(prod, rosters)
    # player_stats ids vs roster ids (current-team membership of box-score players) for 2024
    (OUT / "phase0_source_feasibility_raw.json").write_text(json.dumps(feas, indent=1, sort_keys=True, default=float))
    (OUT / "cfb_identity_audit_raw.json").write_text(json.dumps(ident, indent=1, sort_keys=True, default=float))
    print(json.dumps(ident, indent=1)[:3000])


if __name__ == "__main__":
    main()

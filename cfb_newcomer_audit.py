"""
CFB_NEWCOMER_AUDIT -- quantifies the producer population the point-in-time candidate universe cannot see (HISTORICALLY_UNOBSERVABLE_NEWCOMERS) on burned development seasons 2019-2024 (2025 is NOT read).
Classes of an unseen producer (decided only from appearances in strictly earlier weeks, provider athlete_id history):
  FIRST_COLLEGIATE_PRODUCER   no earlier appearance anywhere in the frozen history (2018+; early seasons overstate it because history starts in 2018)
  TRANSFER_KNOWN_FIRST_FOR_TEAM   earlier appearances for ANOTHER team, none for this team (stable provider id)
  LAPSED_OR_ELSEWHERE_RETURNING   has appeared for this team before but is outside the candidate rule (lapsed recency or last seen elsewhere)
  python cfb_newcomer_audit.py
"""
import json
from collections import defaultdict
from pathlib import Path

import cfb_phase1_data as D

OUT = Path(__file__).resolve().parent / "cfb_models" / "cfb_outcome_engine"
DEV = list(range(2019, 2025))


def klass(m):
    if not m["appeared_before_anywhere"]:
        return "FIRST_COLLEGIATE_PRODUCER"
    if not m["appeared_for_team_before"]:
        return "TRANSFER_KNOWN_FIRST_FOR_TEAM"
    return "LAPSED_OR_ELSEWHERE_RETURNING"


def main():
    tg, pg = D.load_frozen(seasons=list(range(2018, 2025)))                      # 2025 never loaded
    miss = []
    cand, cov = D.build_candidates(tg, pg, DEV, missing_out=miss)
    valid_games = {(r["game_id"], r["team"]) for r in tg if r["team_div"] == "fbs" and r["opp_div"] == "fbs" and r["season"] in DEV}
    tot = defaultdict(lambda: defaultdict(float))                                   # season -> stat -> total skill production in target team-games
    stats = ("carries", "receptions", "pass_att", "rush_yds", "rec_yds", "pass_yds", "rush_td", "rec_td", "pass_td")
    mapping = {"carries": "carries", "receptions": "receptions", "pass_att": "pass_attempts", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards", "pass_yds": "passing_yards", "rush_td": "rushing_touchdowns", "rec_td": "receiving_touchdowns", "pass_td": "passing_touchdowns"}
    for r in pg:
        if (r["game_id"], r["team"]) in valid_games and r["position"] in D.SKILL:
            for k in stats:
                tot[r["season"]][k] += r[mapping[k]] or 0
                tot[0][k] += r[mapping[k]] or 0
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))              # key -> class -> stat
    cnt = defaultdict(lambda: defaultdict(int))
    for m in miss:
        c = klass(m)
        for key in (("season", m["season"]), ("position", m["position"]), ("all", "all"), ("season_position", f'{m["season"]}|{m["position"]}')):
            cnt[key][c] += 1; cnt[key]["_all"] += 1
            for k in stats:
                by[key][c][k] += m[k]; by[key]["_all"][k] += m[k]
    shares = {}
    for key, cl in by.items():
        den = tot[key[1]] if key[0] == "season" else tot[0]
        if key[0] in ("position", "season_position"):
            den = None
        shares[f"{key[0]}={key[1]}"] = {"producer_rows_missing": dict(cnt[key]), "production_share_of_all_skill_production" if den else "production_totals_missing": ({c: {k: (v[k] / den[k] if den[k] else None) for k in stats} for c, v in cl.items()} if den else {c: dict(v) for c, v in cl.items()})}
    out = {"kind": "newcomer / unseen-producer audit (burned development 2019-2024; 2025 unread)", "target_seasons": DEV,
           "observable_vs_unobservable": {s: {"actual_producers": cov[s]["actual_producers"], "producers_in_candidates": cov[s]["producers_in_candidates"], "HISTORICALLY_OBSERVABLE_share": cov[s]["producers_in_candidates"] / cov[s]["actual_producers"], "HISTORICALLY_UNOBSERVABLE_NEWCOMERS": cov[s]["newcomer_producers"]} for s in DEV},
           "classes": {"FIRST_COLLEGIATE_PRODUCER": "no earlier appearance in the frozen history (history starts 2018: overstated for 2019-2020)", "TRANSFER_KNOWN_FIRST_FOR_TEAM": "stable provider id with earlier appearances for another team only", "LAPSED_OR_ELSEWHERE_RETURNING": "appeared for the team before but outside the candidate rule"},
           "totals_skill_production_in_target_games": {str(k): dict(v) for k, v in tot.items()}, "breakdown": shares, "note": "100% historical universe coverage is never claimed: production of unobservable newcomers is reported as an explicit residual ('other') mass"}
    (OUT / "cfb_newcomer_audit.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    allk = shares["all=all"]
    print(json.dumps(out["observable_vs_unobservable"], indent=0)[:900]); print(json.dumps(allk, indent=0)[:2500])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import nfl_v2_phase0_audit as A


def forecast(**kw):
    base = {
        "id":"x","season":2026,"week":4,"game_id":"2026_04_A_B","horizon":"T90",
        "player_id":"p1","player_name":"Player One","position":"RB","team":"A","opponent":"B",
        "outcome":"rush_yds","p_active":0.99,"expected_opportunities":10.0,"mean":50.0,"median":48.0,
        "role_state":{"propensity_share":0.5,"share_shift_vs_last8":0.1},
        "uncertainty":{"score":0.2,"reasons":["opportunity_volatility"]},
    }
    base.update(kw)
    return base


def main():
    stats = {
        (2026,4,"p1"):{"season":"2026","week":"4","player_id":"p1","team":"A","carries":"8",
            "rushing_yards":"32","targets":"2","receptions":"1","receiving_yards":"5","attempts":"0",
            "passing_yards":"0","passing_tds":"0","passing_interceptions":"0"},
        (2026,4,"p2"):{"season":"2026","week":"4","player_id":"p2","team":"A","carries":"8",
            "rushing_yards":"40","targets":"0","receptions":"0","receiving_yards":"0","attempts":"0",
            "passing_yards":"0","passing_tds":"0","passing_interceptions":"0"},
    }
    totals = {(2026,4,"A"):{"carries":16,"targets":2,"attempts":0}}

    r = A.grade_rows([forecast()], stats, totals, {(2026,4,"p1")}, {})[0]
    assert r["meaningful_pregame"] is True
    assert r["clean_meaningful"] is True
    assert r["actual"] == 32
    assert r["actual_opportunities"] == 8
    assert abs(r["predicted_efficiency_mean_per_opp"] - 5) < 1e-9
    assert abs(r["mean_error_opportunity_component"] - 10) < 1e-9
    assert abs(r["mean_error_efficiency_component"] - 8) < 1e-9
    assert abs(r["decomposition_check"]) < 1e-9
    assert abs(r["actual_role_share"] - 0.5) < 1e-9

    fringe = forecast(id="f", player_id="p2", expected_opportunities=2.0, p_active=0.99)
    r2 = A.grade_rows([fringe], stats, totals, {(2026,4,"p2")}, {})[0]
    assert not r2["meaningful_pregame"] and not r2["clean_meaningful"]

    r3 = A.grade_rows([forecast()], stats, totals, set(), {})[0]
    assert r3["meaningful_pregame"] and not r3["clean_meaningful"]

    c = {(2026,4,"p1"):{"reason":"in_game_injury","exclude_from_clean_point_accuracy":True}}
    r4 = A.grade_rows([forecast()], stats, totals, {(2026,4,"p1")}, c)[0]
    assert r4["censored"] and not r4["clean_meaningful"] and r4["actual"] == 32

    assert A.is_meaningful(forecast(outcome="rec",expected_opportunities=3,p_active=0.9,position="WR"))
    assert not A.is_meaningful(forecast(outcome="rec",expected_opportunities=2.99,p_active=0.99,position="WR"))
    assert A.is_meaningful(forecast(outcome="pass_yds",expected_opportunities=20,p_active=0.9,position="QB"))
    assert not A.is_meaningful(forecast(outcome="pass_yds",expected_opportunities=19.9,p_active=0.99,position="QB"))

    rep = A.build_report([r], "abc", ["f.jsonl"])
    assert rep["counts"]["clean_meaningful"] == 1
    assert rep["headline_by_outcome"]["rush_yds"]["mae_median"] == 16
    assert rep["headline_by_outcome"]["rush_yds"]["opportunity_mae"] == 2

    # Final-game gate prevents partially-updated provider data from grading unfinished games as zeros.
    none = A.grade_rows([forecast()], stats, totals, {(2026,4,"p1")}, {}, final_teams=set())
    assert none == []

    print("NFL V2 Phase 0 audit tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

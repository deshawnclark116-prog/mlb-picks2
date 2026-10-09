#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def part(players=(), coverage=((2026,4,"A"),)):
    return {"players": set(players), "coverage": set(coverage), "raw_rows": 1, "positive_rows": 1, "mapped_positive_rows": 1, "map_rate": 1.0}


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

    r = A.grade_rows([forecast()], stats, totals, part({(2026,4,"p1")}), {})[0]
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
    r2 = A.grade_rows([fringe], stats, totals, part({(2026,4,"p2")}), {})[0]
    assert not r2["meaningful_pregame"] and not r2["clean_meaningful"]

    # A real official stat row independently proves participation even if the
    # snap id mapping misses that player.
    r3 = A.grade_rows([forecast()], stats, totals, part({(2026,4,"someone_else")}), {})[0]
    assert r3["meaningful_pregame"] and r3["clean_meaningful"]
    assert r3["played_source"] == "official_stats_row"

    # If no stats row and the exact team-week has snap coverage, a missing
    # player is genuine negative participation evidence.
    no_stat = forecast(id="nostat", player_id="p3")
    rneg = A.grade_rows([no_stat], stats, totals, part(), {})[0]
    assert rneg["played"] is False and not rneg["clean_meaningful"]
    assert rneg["played_source"] == "snap_counts_negative"

    # If that team-week is absent from snap coverage, fail open-to-unknown for
    # grading rather than pretending provider lag means the player did not play.
    runk = A.grade_rows([no_stat], stats, totals, part(coverage=((2026,3,"A"),)), {})[0]
    assert runk["played"] is None and not runk["clean_meaningful"]
    assert runk["played_source"] == "participation_source_not_ready"

    c = {(2026,4,"p1"):{"reason":"in_game_injury","exclude_from_clean_point_accuracy":True}}
    r4 = A.grade_rows([forecast()], stats, totals, part({(2026,4,"p1")}), c)[0]
    assert r4["censored"] and not r4["clean_meaningful"] and r4["actual"] == 32

    assert A.is_meaningful(forecast(outcome="rec",expected_opportunities=3,p_active=0.9,position="WR"))
    assert not A.is_meaningful(forecast(outcome="rec",expected_opportunities=2.99,p_active=0.99,position="WR"))
    assert A.is_meaningful(forecast(outcome="pass_yds",expected_opportunities=20,p_active=0.9,position="QB"))
    assert not A.is_meaningful(forecast(outcome="pass_yds",expected_opportunities=19.9,p_active=0.99,position="QB"))

    rep = A.build_report([r], "abc", ["f.jsonl"])
    assert rep["counts"]["clean_meaningful"] == 1
    assert rep["headline_by_outcome"]["rush_yds"]["mae_median"] == 16
    assert rep["headline_by_outcome"]["rush_yds"]["opportunity_mae"] == 2
    assert rep["headline_by_outcome"]["rush_yds"]["opportunity_component_abs_mean"] == 10
    assert rep["headline_by_outcome"]["rush_yds"]["efficiency_component_abs_mean"] == 8
    assert rep["headline_by_outcome"]["rush_yds"]["dominant_mean_error_component"]["opportunity"] == 1

    # Final-game gate prevents partially-updated provider data from grading unfinished games as zeros.
    none = A.grade_rows([forecast()], stats, totals, part({(2026,4,"p1")}), {}, final_teams=set())
    assert none == []

    print("NFL V2 Phase 0 audit tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
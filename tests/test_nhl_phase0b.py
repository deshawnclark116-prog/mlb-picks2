"""Focused NHL Phase 0B tests. python tests/test_nhl_phase0b.py"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_outcome_contract as CT  # noqa: E402
import nhl_outcome_phase0b_probes as PR  # noqa: E402

E = REPO / "nhl_models" / "nhl_outcome_engine"
UTC = timezone.utc


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def synthetic_table(start0=datetime(2023, 11, 1, 0, 0, tzinfo=UTC)):
    """Team AAA plays games 1..8 every second day; players p1..p3 play all, p9 only game 8 (cold start), p5 joins AAA at game 6 from team BBB."""
    tab, tot = {}, {}
    for g in range(1, 9):
        st = start0 + timedelta(days=2 * (g - 1), hours=19)
        for p, pp in (("p1", 120), ("p2", 60), ("p3", 0), ("p9", 30 if g == 8 else None), ("p5", 20 if g >= 6 else None)):
            if pp is None:
                continue
            team = "AAA" if (p != "p5" or g >= 6) else "BBB"
            tab[(g, p)] = {"gameId": g, "playerId": p, "team": team, "startTimeUTC": iso(st), "pp": pp, "ev": 900, "sh": 0, "toi": 900 + pp, "shifts": 20, "sog": 2, "name": p}
        if p == "p5" and g < 6:
            pass
    return tab


def test_no_target_game_leakage():
    tab = synthetic_table()
    base, leaks = PR.deployment_probe(tab, {8}, "/tmp", {"requests": 0, "bytes": 0, "seconds": 0.0})
    assert leaks == 0
    def feats(rows):
        return {(r["team"], x["playerId"]): (round(x["pp_share5"], 9), round(x["ev_share5"], 9), x["pp_rank"]) for r in rows for x in r["warm"]}
    f0 = feats(base)
    for k in list(tab):                                                        # destroy every target-game value: features must not move
        if k[0] == 8:
            tab[k] = {**tab[k], "pp": 999, "ev": 1, "shifts": 99, "toi": 5}
    f1 = feats(PR.deployment_probe(tab, {8}, "/tmp", {"requests": 0, "bytes": 0, "seconds": 0.0})[0])
    assert f0 == f1 and f0
    # the target game never appears in the allowed prior rows
    T = CT.cutoff_time(CT.parse_utc(tab[(8, "p1")]["startTimeUTC"]), "T90")
    allowed = CT.prior_rows(list(tab.values()), 8, T)
    assert all(r["gameId"] != 8 for r in allowed) and {r["gameId"] for r in allowed} == set(range(1, 8))


def test_cold_start_and_acquired_policy():
    tab = synthetic_table()
    rows, _ = PR.deployment_probe(tab, {8}, "/tmp", {"requests": 0, "bytes": 0, "seconds": 0.0})
    cats = {x["playerId"]: x["cat"] for r in rows for x in r["warm"] + r["cold"]}
    assert cats["p9"] == "COLD" and cats["p1"] == "KNOWN" and cats["p5"] == "KNOWN"             # p5 played games 6-7 for AAA before game 8
    cold = [x for r in rows for x in r["cold"]]
    assert [x["playerId"] for x in cold] == ["p9"] and "pp_share5" not in cold[0]              # no player-level features, never backfilled from the target game
    assert CT.classify_player("p5", "AAA", [{"team": "BBB"}] * 3) == "ACQUIRED"
    assert CT.classify_player("pX", "AAA", []) == "COLD"
    c = json.loads((E / "phase0b_historical_contract.json").read_text())
    assert "unobservable_at_T" in c["cold_start_policy"]["universe_rule"] and "NEVER" in c["cold_start_policy"]["universe_rule"]


def test_no_shootout_sog_contamination():
    ev = [{"typeDescKey": "shot-on-goal", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "goal", "periodDescriptor": {"periodType": "OT"}, "details": {"scoringPlayerId": 1}},
          {"typeDescKey": "missed-shot", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "blocked-shot", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1, "blockingPlayerId": 2}},
          {"typeDescKey": "shot-on-goal", "periodDescriptor": {"periodType": "SO"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "goal", "periodDescriptor": {"periodType": "SO"}, "details": {"scoringPlayerId": 1}}]
    c = CT.shot_counts_from_events(ev)[1]
    assert CT.sog(c) == 2 and CT.attempts(c) == 4
    p = json.loads((E / "phase0b_acquisition_benchmark.json").read_text())["C_shot_attempts"]["summary"]
    assert p["sog_mismatch_players"] == 0 and p["shootout_events_excluded"] > 0 and p["shootout_games"]


def test_pp_ev_sh_consistency_in_sample_table():
    tab = json.loads((E / "evidence" / "m2_sample_table.json").read_text())
    assert len(tab) > 5000
    bad = [r for r in tab if not CT.strength_consistent(r, tol=1)]
    assert len(bad) <= 0.002 * len(tab), len(bad)                      # EV+PP+SH(+OT) equals total TOI
    assert all(r["teamId"] and r["startTimeUTC"] and r["opponent"] for r in tab)


def test_traded_player_completeness():
    b = json.loads((E / "phase0b_acquisition_benchmark.json").read_text())
    tp = b["traded_players"]
    assert tp["traded_skater_games_in_sample"] > 500 and tp["traded_skater_games_found_by_M2"] == tp["traded_skater_games_in_sample"]
    for w in b["weeks"]:
        m = w["m2_vs_boxscore"]
        assert m["missing_vs_boxscore"] == 0 and m["extra_vs_boxscore"] == 0 and m["sog_mismatch"] == 0 and m["toi_mismatch_gt_1s"] == 0
    d = b["missing_player_defect_diagnosis"]
    assert d["date_window_query_for_same_games"]["missing_sample_rows"] == 0 and d["team_filter_with_limit_minus_1"]["missing_sample_rows"] > 0
    assert b["cap_test"]["silently_truncated_at_10000"] is True


def test_deterministic_cutoff_handling():
    st = datetime(2026, 10, 3, 23, 0, tzinfo=UTC)
    assert CT.cutoff_time(st, "T90") == datetime(2026, 10, 3, 21, 30, tzinfo=UTC) and CT.cutoff_time(st, "T24H") == datetime(2026, 10, 2, 23, 0, tzinfo=UTC)
    assert CT.cutoff_time(st, "T90") == CT.cutoff_time(st.astimezone(timezone(timedelta(hours=-4))), "T90")        # timezone independent
    try:
        CT.cutoff_time(datetime(2026, 10, 3, 23, 0), "T90"); assert False
    except ValueError:
        pass
    T = CT.cutoff_time(st, "T90")
    same_day_early = datetime(2026, 10, 3, 18, 0, tzinfo=UTC)          # starts 3.5h before T -> ends by 21:30 -> boundary included
    assert CT.completed_before(same_day_early, T) is True
    assert CT.completed_before(same_day_early + timedelta(minutes=1), T) is False            # strict: one minute later is excluded
    assert CT.completed_before(st, T) is False and CT.completed_before(st - timedelta(days=1), T) is True
    rows = [{"gameId": i, "startTimeUTC": iso(st - timedelta(hours=h))} for i, h in ((1, 30), (2, 5), (3, 3), (9, 0))]
    assert [r["gameId"] for r in CT.prior_rows(rows, 9, T)] == [1, 2]       # idempotent / deterministic
    assert CT.prior_rows(rows, 9, T) == CT.prior_rows(list(reversed(rows)), 9, T)[::-1]


def test_forward_contract_records_required_fields():
    f = json.loads((E / "phase0b_forward_snapshot_contract.json").read_text())
    need = {"retrieval_ts", "intended_horizon", "game_start", "source_url", "sha256", "availability"}
    assert need <= set(f["snapshot_record_fields"])
    assert f["recommended_horizons"] and f["observed_field_timing"]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
